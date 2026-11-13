"""Run with embedded Python: native_local_edit_smoke.py <ComfyUI root>.

Uses the installed official H3 guide/latent APIs on CPU, with mock VAE and
sampling. Does not load weights or evaluate visual quality.
"""
import importlib.util
from pathlib import Path
import sys
import os
from unittest.mock import patch

import torch

comfy_root = sys.argv[1]
sys.argv = [sys.argv[0]]
sys.path.insert(0, comfy_root)
from comfy.cli_args import args
args.cpu = True
from comfy_extras import nodes_minimax_h3 as h3
from comfy.nested_tensor import NestedTensor

root = Path(os.environ.get('SOLARL_TEST_ROOT', Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('local_native', root/'__init__.py', submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules['local_native'] = package
spec.loader.exec_module(package)
from local_native import local_edit as e
from local_native import sampler as s
from comfy_extras import nodes_audio

assert 'segment_plan' in s.SolarLH3LongVideoSampler.INPUT_TYPES()['optional']
from local_native.selflift_sampler import SolarLH3SelfLiftLongVideoSampler
assert 'segment_plan' in SolarLH3SelfLiftLongVideoSampler.INPUT_TYPES()['optional']
assert s.SolarLH3LongVideoSampler.INPUT_TYPES()['hidden']['cache_prompt'] == 'PROMPT'


class VAE:
    def encode(self, frames):
        return torch.zeros(1, 24, h3.video_latent_t(len(frames)), frames.shape[1]//16, frames.shape[2]//16)

    def decode(self, z):
        count = sum(h3.FRAME_PER_TOKEN[i%5] for i in range(z.shape[2]))
        return torch.full((1, count, z.shape[-2]*16, z.shape[-1]*16, 3), .8)


calls = []


class AudioVAE:
    audio_sample_rate = 24000

    def encode(self, waveform):
        assert waveform.ndim == 3 and waveform.shape[-1] == 2
        return torch.zeros(1, 32, 2, round(waveform.shape[1]*40/self.audio_sample_rate))


def sample(model, cond, negative, latent, sampler, sigmas, seed, cfg, mask=None):
    guides = cond[0][1]['minimax_keyframes']
    calls.append([g['resolved_frame_index'] for g in guides])
    assert all(g['latent'].shape[2] == h3.video_latent_t(22) for g in guides)
    assert all(g['audio_latent'].shape[:3] == (1, 32, 2) and g['audio_latent'].shape[-1] > 0 for g in guides)
    assert len(latent['samples'].tensors) == 2
    if mask is not None:
        vm, am = mask.unbind()
        assert vm.shape[1] == am.shape[1] == 1
        assert not vm[:, :, :h3.video_latent_t(22)].count_nonzero()
        assert not am[..., :36].count_nonzero()
        assert vm[:, :, h3.video_latent_t(22):].min() == 1
        assert am[..., 36:].min() == 1
        calls.append('locked')
    return {'samples': latent['samples'], '_denoised_samples': latent['samples']}


def audio_decode(vae, latent):
    return {'waveform': torch.zeros(1, 2, 200000), 'sample_rate': 24000}


source = torch.full((100, 32, 32, 3), .2)
with patch.object(s, 'sample_once', sample), patch.object(nodes_audio, 'vae_decode_audio', audio_decode):
    for mode, total, lock in [('replace_range', 100, 'off'), ('extend', 148, 'off'), ('extend', 148, 'head_experimental')]:
        source_audio = {'waveform': torch.zeros(1, 2, 100000), 'sample_rate': 24000}
        plan = e.SolarLH3LocalEditPlan().build(source, mode, 24, 72, 48, 22, 'preserve_source', source_audio=source_audio, context_lock=lock)[0]
        out = e.SolarLH3LocalEditSampler().run(
            plan, object(), [[torch.ones(1, 2, 3), {}]], h3._empty_av_latent(32, 32, 124)[0],
            VAE(), AudioVAE(), object(), torch.tensor([1., .5, 0.]), 42, 1.)
        assert out[0].shape == (48, 32, 32, 3)
        assert out[1].shape == (total, 32, 32, 3)
        assert out[3]['waveform'].shape[-1] == total*1000
        assert torch.equal(out[1][:24], source[:24])
assert calls == [[0, 70], [0], [0], 'locked'], calls
assert 'MiniMaxH3LocalEditSampler-solarL' in package.NODE_CLASS_MAPPINGS
print('PASS: real H3 video+audio AddGuide, AV latent, crop, composition, audio duration, node registration; CPU mock sampling only.')
