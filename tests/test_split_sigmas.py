import sys
import types
import unittest
from unittest.mock import patch
import torch
import test_sampler as base
from longvideo import sampler as s


class SplitTests(unittest.TestCase):
    def test_full_window_split_uses_x0_same_seed_and_final_audio(self):
        names=['comfy','comfy.nested_tensor','comfy.model_management','comfy_extras',
               'comfy_extras.nodes_minimax_h3','comfy_extras.nodes_audio','node_helpers']
        mods={n:types.ModuleType(n) for n in names}
        mods['comfy'].nested_tensor=mods['comfy.nested_tensor']
        mods['comfy.nested_tensor'].NestedTensor=base.Nested
        mods['comfy.model_management'].throw_exception_if_processing_interrupted=lambda:None
        h3=mods['comfy_extras.nodes_minimax_h3']
        h3.FRAME_PER_TOKEN=(1,4,4,4,4)
        h3._empty_av_latent=base.empty
        h3.video_latent_t=lambda n:(n-5)//17*5+2
        h3.align_frame_count=lambda n:n+(5-n)%17
        mods['comfy_extras.nodes_audio'].vae_decode_audio=lambda vae,z:{
            'waveform':torch.ones(1,2,22000)*float(z['samples'].tensors[1].mean()),'sample_rate':24000}
        calls=[]
        def sample(model,cond,neg,latent,sampler,sigmas,seed,cfg,mask=None):
            v,a=latent['samples'].tensors
            calls.append((seed,mask))
            if len(calls)==1:
                return {'samples':base.Nested((v+10,a+10)),
                        '_denoised_samples':base.Nested((v+1,a+2))}
            self.assertTrue(torch.all(v==1))
            self.assertTrue(torch.all(a==2))
            self.assertIsNone(mask)
            return {'samples':base.Nested((v+3,a+4)),
                    '_denoised_samples':base.Nested((v+5,a+6))}
        class Upscale:
            @classmethod
            def execute(cls,latent,**kw):return (latent,)
        with patch.dict(sys.modules,mods),patch.object(s,'sample_once',sample),patch.object(s,'upscaler_class',lambda:Upscale):
            out=s.SolarLH3LongVideoSampler().run(model=object(),positive=[[torch.ones(1,2,3),{}]],
                latent=base.empty(32,32,22)[0],vae=base.VAE(),audio_vae=object(),sampler_1=object(),
                sigmas_1=torch.tensor([1.,.5]),seed=42,cfg_1=1.,target_frames=22,segment_frames=0,
                overlap_frames=22,continuation_mode='official_image_guide',enable_second_pass=True,
                upscaler_model='test',upscale_factor=1.,cfg_2=1.,highres_tiling=False,
                sampler_2=object(),sigmas_2=torch.tensor([.5,0.]),second_pass_mode='split_sigmas')
        self.assertEqual([x[0] for x in calls],[42,42])
        self.assertTrue(torch.all(out[1]['waveform']==6))
        self.assertTrue(torch.all(out[6]['samples'].tensors[1]==6))
        self.assertTrue(torch.all(out[7]['samples'].tensors[1]==8))
        self.assertEqual(len(out[0]),22)


if __name__=='__main__':unittest.main()
