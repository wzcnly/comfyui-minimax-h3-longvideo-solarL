"""SelfLift loop, global suffix and cache resume without model weights."""
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
import test_sampler as base
from longvideo import selflift_sampler as sl


class AV(base.Nested):
    def to(self, **kwargs):
        return AV(tuple(t.to(**kwargs) for t in self.tensors))


class SelfLiftPipelineTests(unittest.TestCase):
    def test_segments_global_suffix_and_resume(self):
        names = ['comfy', 'comfy.nested_tensor', 'comfy.model_management', 'comfy.sample',
                 'comfy.samplers', 'comfy.utils', 'comfy_extras',
                 'comfy_extras.nodes_minimax_h3', 'comfy_extras.nodes_audio', 'latent_preview',
                 'longvideo.vendor.selflift_runtime', 'longvideo.vendor.selflift_runtime.nodes',
                 'longvideo.vendor.selflift_runtime.selflift', 'longvideo.vendor.selflift_runtime.h3_tiling',
                 'longvideo.vendor.selflift_runtime.h3_upscaler']
        mods = {n:types.ModuleType(n) for n in names}
        for n in ('nested_tensor','model_management','sample','samplers','utils'):
            setattr(mods['comfy'], n, mods['comfy.'+n])
        mods['comfy.nested_tensor'].NestedTensor = AV
        mm = mods['comfy.model_management']
        mm.throw_exception_if_processing_interrupted = lambda:None
        mm.intermediate_device = lambda:'cpu'
        mm.intermediate_dtype = lambda:torch.float32
        h3 = mods['comfy_extras.nodes_minimax_h3']
        h3.FRAME_PER_TOKEN = (1,4,4,4,4)
        h3.align_frame_count = lambda n:n+(5-n)%17
        h3.video_latent_t = lambda n:2 if n<=5 else (n-5)//17*5+2
        h3._empty_av_latent = lambda w,h,n:({'samples':AV(base.empty(w,h,n)[0]['samples'].tensors)},n)
        h3.MiniMaxH3AddGuide = types.SimpleNamespace(execute=lambda cond,*args,**kw:(cond,))
        prompts = []
        def reference(**kw):
            prompts.append(kw['prompt'])
            return ([[torch.ones(1,2,3)*len(kw['prompt']),{}]], h3._empty_av_latent(kw['width'],kw['height'],kw['length'])[0])
        h3.MiniMaxH3ReferenceToVideo = types.SimpleNamespace(execute=reference)
        mods['comfy_extras.nodes_audio'].vae_decode_audio = lambda *a:{'waveform':torch.zeros(1,2,303000),'sample_rate':24000}
        runtime = mods['longvideo.vendor.selflift_runtime']
        for name in ('nodes','selflift','h3_tiling','h3_upscaler'):
            setattr(runtime,name,mods['longvideo.vendor.selflift_runtime.'+name])
        runtime.nodes._validate_schedule = lambda *a:None
        runtime.nodes._validate_sampling = lambda *a:None
        runtime.nodes._resize_keyframes = lambda cond,*a:cond
        runtime.nodes._euler_step = lambda state,x0,*a:x0
        runtime.selflift.artifact_aware_consistency_lift = lambda z,p,*a:z if z is not None else p
        mods['comfy.sample'].prepare_noise = lambda z,*a:torch.zeros_like(z)
        mods['comfy.utils'].PROGRESS_BAR_ENABLED = False
        mods['latent_preview'].prepare_callback = lambda *a:lambda *a:None
        low_calls, high_calls = [], []
        def low(*args):
            low_calls.append(True)
            av = args[3]['samples']
            return {'samples':av,'_denoised_samples':AV(tuple(t+1 for t in av.tensors))}
        def high(*args, **kw):
            high_calls.append(True)
            z = kw['latent_image']
            self.assertEqual(z.tensors[0].shape[2], h3.video_latent_t(311))
            kw['callback'](0,z,z,1)
            return z
        mods['comfy.samplers'].sample = high
        identity = types.SimpleNamespace(process_in=lambda z:z)
        sampling = types.SimpleNamespace(noise_scaling=lambda sigma,noise,z:z,
                                         inverse_noise_scaling=lambda sigma,z:z)
        model = types.SimpleNamespace(model=types.SimpleNamespace(process_latent_out=lambda z:z),
            get_model_object=lambda key:identity if key=='latent_format' else sampling,
            load_device='cpu',model_options={})
        plan = {'version':1,'overlap_frames':22,'default_prompt':'common',
                'segments':[{'frames':124,'prompt':'one'},{'frames':102,'prompt':'two'},{'frames':77,'prompt':'three'}]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,mods), patch.object(sl,'sample_once',low):
            kw = dict(model=model,positive=[[torch.ones(1,2,3),{}]],latent=h3._empty_av_latent(32,32,124)[0],
                vae=base.VAE(),audio_vae=object(),sampler=object(),sigmas=torch.tensor([1.,.5,0.]),cfg=1.,seed=42,
                target_frames=0,segment_frames=124,overlap_frames=22,continuation_mode='official_image_guide',
                transition_step=1,lowres_scale=1.,upscaler_model='none',rho=1.,w_min=1.,w_max=1.,
                clip=object(),segment_plan=plan,cache_directory=directory)
            runtime.selflift._pixel_anchor_video = lambda video,*a:video
            out = sl.SolarLH3SelfLiftLongVideoSampler().run(**kw)
            self.assertEqual(out[0].shape[0],303)
            again = sl.SolarLH3SelfLiftLongVideoSampler().run(**kw)
            self.assertEqual(len(low_calls),3)
            self.assertEqual(len(high_calls),1)
            self.assertIn('global_highres_cache_hit=1',again[5])
            self.assertIn('one',prompts)
            self.assertIn('three',prompts)


if __name__ == '__main__': unittest.main()
