"""Run with ComfyUI Python: smoke_native.py <ComfyUI root>. No model weights loaded."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import torch

sys.path.insert(0,sys.argv[1])
from comfy_extras import nodes_minimax_h3 as h3
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('solar_native',root/'__init__.py',submodule_search_locations=[str(root)])
package=importlib.util.module_from_spec(spec)
sys.modules['solar_native']=package
spec.loader.exec_module(package)
from solar_native import sampler as s

class Clip:
    def tokenize(self,prompt,**kwargs): return prompt
    def encode_from_tokens_scheduled(self,tokens): return [[torch.zeros(1,2,3),{}]]

class VAE:
    audio_sample_rate_output=24000
    def encode(self,frames):
        return torch.zeros(1,24,h3.video_latent_t(len(frames)),frames.shape[1]//16,frames.shape[2]//16)
    def decode(self,z):
        if z.ndim==5:
            n=sum(h3.FRAME_PER_TOKEN[i%5] for i in range(z.shape[2]))
            return torch.zeros(1,n,z.shape[-2]*16,z.shape[-1]*16,3)
        return torch.zeros(1,round(z.shape[-1]*24000/40),2)

calls=[]
def fake_sample(model,cond,neg,latent,sampler,sigmas,seed,cfg,mask=None):
    refs=cond[0][1]['minimax_refs']
    assert refs[0]['kind']=='video'
    frames = 209 if mask is not None else (124 if not calls else 107)
    assert refs[0]['latent_t']==h3.video_latent_t(frames)
    assert latent['samples'].tensors[0].shape[2] == refs[0]['latent_t']
    if mask is not None:
        assert calls == [1,2]
        assert not cond[0][1].get('minimax_keyframes')
        assert mask.tensors[1].count_nonzero()==0
    elif len(calls):
        assert len(cond[0][1]['minimax_keyframes'])==1
    calls.append(seed)
    return latent

class Upscale:
    @classmethod
    def execute(cls,latent,**kwargs):
        return (latent,)

for second in (False, True):
  calls.clear()
  with patch.object(s,'sample_once',fake_sample), patch.object(s,'registered',lambda name:Upscale):
    result=s.SolarLH3LongVideoSampler().run(
        model=object(),positive=[[torch.zeros(1,2,3),{}]],latent=h3._empty_av_latent(32,32,124)[0],
        vae=VAE(),audio_vae=VAE(),sampler_1=object(),sigmas_1=torch.tensor([1.,.5,0.]),
        seed=1,cfg_1=1,target_frames=0,segment_frames=124,overlap_frames=22,
        continuation_mode='official_image_guide',enable_second_pass=second,
        upscaler_model='unused',upscale_factor=1.5,cfg_2=1,highres_tiling=False,
        source_video=torch.zeros(194,32,32,3),clip=Clip(),prompt='motion transfer',
        sampler_2=object(),sigmas_2=torch.tensor([.2,0.]))
  assert result[0].shape[0]==194
  assert result[1]['waveform'].shape[-1]==194000
  assert calls==([1,2,1000001] if second else [1,2])
print('Native official conditioning + AddGuide + AV shape/decode orchestration: PASS. Sampling mocked; no GPU render.')

