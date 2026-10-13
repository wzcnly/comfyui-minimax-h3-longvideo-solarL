import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('longvideo',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules['longvideo'] = package
spec.loader.exec_module(package)
from longvideo.planning import plan_segments, segment_length, validate_sigmas
from longvideo import sampler as s
from longvideo import selflift_sampler as sl


class Nested:
    def __init__(self, tensors): self.tensors = tensors
    is_nested = True
    def unbind(self): return self.tensors
    def cpu(self): return Nested(tuple(t.cpu() for t in self.tensors))


def empty(w,h,n):
    return {'samples':Nested((torch.zeros(1,24,((n-5)//17)*5+2,h//16,w//16),
                              torch.zeros(1,32,2,round(n*40/24))))}, n


class VAE:
    def decode(self,z):
        t = z.shape[2]
        count = sum((1,4,4,4,4)[i%5] for i in range(t))
        return torch.zeros(1,count,z.shape[-2]*16,z.shape[-1]*16,3)


class Upscale:
    @classmethod
    def execute(cls,latent,**kwargs):
        z = latent['samples']
        scale = kwargs['mode']['scale']
        return ({'samples':z.repeat_interleave(int(scale),dim=-2).repeat_interleave(int(scale),dim=-1)},)


class Motion:
    def apply(self,cond,current,trim_frames,context_latent):
        assert context_latent['samples'].tensors[0].shape[1]==24
        return cond,trim_frames


class Tests(unittest.TestCase):
    def test_selflift_adapter_uses_native_engine(self):
        calls=[]
        class Engine:
            def sample(self,**kwargs):
                calls.append(kwargs)
                return (kwargs['latent_image'],)
        def loop(**kwargs):
            self.assertTrue(kwargs['enable_second_pass'])
            self.assertNotIn('_first_pass_sampler',kwargs)
            result=kwargs['_whole_video_refiner']('model',[],[],
                {'samples':Nested((torch.zeros(1,24,62,4,4),torch.ones(1,32,2,348)))},
                'euler',kwargs['sigmas_2'],123,1.0,{'mode':'scale by multiplier','scale':1.5})
            self.assertEqual(result['samples'].tensors[0].shape,(1,24,62,6,6))
            return (1,2,3,4,5,'report',7,8)
        params=dict(model='model',positive=[],latent={},vae='vae',audio_vae='audio',
            sampler_1='euler',sigmas_1=torch.tensor([1.,.8,.6,.4,.3,.2,.1,.05,0.]),
            sampler_2='euler',sigmas_2=torch.tensor([.2,.1,0.]),
            seed=0,cfg_1=1.,target_frames=194,segment_frames=124,overlap_frames=22,
            continuation_mode='official_image_guide',upscaler_model='upscaler',
            highres_tiling=True,model_hires='hires')
        nested_module=types.ModuleType('comfy.nested_tensor');nested_module.NestedTensor=Nested
        with patch.object(sl,'selflift_class',return_value=Engine),patch.object(s.SolarLH3LongVideoSampler,'run',side_effect=loop),patch.dict(sys.modules,{'comfy.nested_tensor':nested_module}):
            out=sl.SolarLH3SelfLiftLongVideoSampler().run(**params)
            self.assertEqual(len(out),7)
            self.assertIn('whole-video',out[5])
            self.assertEqual(calls[0]['transition_step'],1)
            self.assertAlmostEqual(calls[0]['lowres_scale'],2/3)
            self.assertEqual(calls[0]['model_hires'],'hires')
            self.assertTrue(calls[0]['highres_tiling'])
            with self.assertRaisesRegex(ValueError,'低分辨率步数'):
                sl.SolarLH3SelfLiftLongVideoSampler().run(**params,transition_step=2)

    def test_upscale_modes(self):
        self.assertEqual(s.upscale_mode_options('scale by multiplier',1.5,1280,704,1),
                         {'mode':'scale by multiplier','scale':1.5})
        self.assertEqual(s.upscale_mode_options('target dimensions',1.5,1280,704,1),
                         {'mode':'target dimensions','width':1280,'height':704})
        self.assertEqual(s.upscale_mode_options('megapixels',1.5,1280,704,1.2),
                         {'mode':'megapixels','megapixels':1.2})
        with self.assertRaises(ValueError):s.upscale_mode_options('bad',1.5,1280,704,1)

    def test_reference_groups_and_legacy_aliases(self):
        images=[torch.ones(1,2,2,3)*i for i in range(9)]
        videos=[torch.ones(227,2,2,3)*i for i in range(3)]
        audio=[{'waveform':torch.ones(1,2,227000)*i,'sample_rate':24000} for i in range(3)]
        extra={f'ref_image_{i}':images[i] for i in range(4,9)}
        extra.update({f'ref_video_{i}':videos[i] for i in range(3)})
        extra.update({f'ref_video_audio_{i}':audio[i] for i in range(3)})
        extra.update({f'ref_audio_{i}':audio[i] for i in range(3)})
        pictures,vids,paired,standalone=s.reference_inputs(images[:4],None,None,None,extra)
        self.assertEqual((len(pictures),len(vids),len(paired),len(standalone)),(9,3,3,3))
        legacy=s.reference_inputs(images[:4],videos[0],audio[0],videos[1],{})
        self.assertIs(legacy[1]['ref_video_0'],videos[0])
        self.assertIs(legacy[1]['ref_video_1'],videos[1])
        with self.assertRaises(ValueError):s.reference_inputs([None]*4,None,None,None,{'ref_image_4':images[4]})
        with self.assertRaises(ValueError):s.reference_inputs([None]*4,None,None,None,{'ref_video_audio_0':audio[0]})
        with self.assertRaises(ValueError):s.reference_inputs([None]*4,videos[0],None,None,{'ref_video_0':videos[1]})

    def test_motion_discovery_ignores_module_class_proxies(self):
        class LibraryProxy:
            @property
            def __class__(self): return types.ModuleType
        proxy = LibraryProxy()
        self.assertTrue(isinstance(proxy, types.ModuleType))
        with self.assertRaises(TypeError):
            types.ModuleType.__getattribute__(proxy,'__dict__')
        nodes = types.ModuleType('nodes')
        nodes.NODE_CLASS_MAPPINGS = {}
        cls = type('AptMiniMaxH3MotionContext', (), {})
        provider = types.ModuleType('longvideo.vendor.apt_motion')
        provider.AptMiniMaxH3MotionContext = cls
        with patch.dict(sys.modules, {'nodes':nodes,'cffi_library_test':proxy,
                                     'longvideo.vendor.apt_motion':provider}):
            self.assertIs(s.motion_class(),cls)

    def test_motion_discovery_ignores_lazy_module_hooks(self):
        lazy = types.ModuleType('seedvr_flashattention_test')
        def fail(name):
            raise ImportError('_C_flashattention unavailable')
        lazy.__getattr__ = fail
        nodes = types.ModuleType('nodes')
        cls = type('AptMiniMaxH3MotionContext', (), {})
        provider = types.ModuleType('longvideo.vendor.apt_motion')
        provider.AptMiniMaxH3MotionContext = cls
        with patch.dict(sys.modules, {'nodes':nodes,'seedvr_flashattention_test':lazy,
                                     'longvideo.vendor.apt_motion':provider}):
            nodes.NODE_CLASS_MAPPINGS = {'motion':type('AptMiniMaxH3MotionContext',(),{})}
            self.assertIs(s.motion_class(),cls)
            nodes.NODE_CLASS_MAPPINGS = {}
            self.assertIs(s.motion_class(),cls)

    def test_sampling_progress_callback(self):
        updates=[]
        modules={name:types.ModuleType(name) for name in ('comfy','comfy.sample','comfy.model_management','comfy.utils')}
        modules['comfy'].sample=modules['comfy.sample']
        modules['comfy'].model_management=modules['comfy.model_management']
        modules['comfy'].utils=modules['comfy.utils']
        modules['comfy.model_management'].throw_exception_if_processing_interrupted=lambda:None
        class Progress:
            def __init__(self,total): self.total=total
            def update_absolute(self,value,total):updates.append((value,total))
        modules['comfy.utils'].ProgressBar=Progress
        modules['comfy.sample'].prepare_noise=lambda z,seed:torch.zeros_like(z)
        mask=object()
        def custom(model,noise,cfg,sampler,sigmas,positive,negative,z,**kwargs):
            self.assertEqual(kwargs['seed'],42)
            self.assertIs(kwargs['noise_mask'],mask)
            for i in range(2):kwargs['callback'](i,z*(i+1),z,2)
            return z
        modules['comfy.sample'].sample_custom=custom
        latent={'samples':torch.ones(1,24,7,2,2)}
        model=types.SimpleNamespace(model=types.SimpleNamespace(process_latent_out=lambda z:z+10))
        with patch.dict(sys.modules,modules):
            result=s.sample_once(model,[],[],latent,None,torch.tensor([1.,.5,0.]),42,1,mask)
        self.assertIs(result['samples'],latent['samples'])
        self.assertTrue(torch.equal(result['_denoised_samples'],latent['samples']*12))
        self.assertEqual(updates,[(0,2),(1,2),(2,2)])
        # H3 callbacks contain a packed tensor, whereas sampling returns AV streams.
        modules['comfy.nested_tensor']=types.ModuleType('comfy.nested_tensor')
        modules['comfy'].nested_tensor=modules['comfy.nested_tensor']
        modules['comfy.nested_tensor'].NestedTensor=Nested
        def unpack(flat,shapes):
            pieces=[]
            offset=0
            for shape in shapes:
                count=shape.numel()
                pieces.append(flat.flatten()[offset:offset+count].reshape(shape))
                offset+=count
            return pieces
        modules['comfy.utils'].unpack_latents=unpack
        modules['comfy.sample'].prepare_noise=lambda z,seed:z
        av=Nested((torch.ones(1,24,7,2,2),torch.ones(1,32,2,37)*2))
        def packed_custom(*args,**kwargs):
            flat=torch.cat([z.flatten() for z in av.tensors]).reshape(1,1,-1)*3
            kwargs['callback'](0,flat,flat,1)
            return av
        modules['comfy.sample'].sample_custom=packed_custom
        model.model.process_latent_out=lambda z:Nested(tuple(t+10 for t in z.tensors))
        with patch.dict(sys.modules,modules):
            packed=s.sample_once(model,[],[],{'samples':av},None,torch.tensor([1.,0.]),42,1)
        for original,den in zip(av.tensors,packed['_denoised_samples'].tensors):
            self.assertTrue(torch.equal(den,original*3+10))
        modules['comfy.sample'].sample_custom=lambda *args,**kwargs:av
        with patch.dict(sys.modules,modules):
            fallback=s.sample_once(model,[],[],{'samples':av},None,torch.tensor([1.,0.]),42,1)
        self.assertNotIn('_denoised_samples',fallback)

    def test_short_final_segment(self):
        self.assertEqual(segment_length(194,102,124),107)
        for overlap in (5,22,39,56):
            for total in (1,22,123,124,125,194,226,227,1000):
                plan = plan_segments(total,124,overlap)
                lengths = [segment_length(total,start,124) for start,_ in plan]
                self.assertTrue(all(n % 17 == 5 and 22 <= n <= 124 for n in lengths))
                self.assertTrue(all(n > trim for n,(_,trim) in zip(lengths,plan)))
                self.assertEqual(sum(min(n-trim,total-start-trim) for n,(start,trim) in zip(lengths,plan)),total)
                self.assertTrue(all(n==124 for n in lengths[:-1]))
                pieces = []
                token_count = lambda n: ((n-5)//17)*5+2
                for n,(start,trim) in zip(lengths,plan):
                    z = torch.arange(token_count(n)) + start//17*5
                    pieces.append(z[token_count(trim) if trim else 0:])
                merged = torch.cat(pieces)
                aligned = max(22,total)+(5-max(22,total))%17
                self.assertTrue(torch.equal(merged,torch.arange(token_count(aligned))))

    def test_decode_image_and_video_shapes(self):
        frames = torch.arange(124.).reshape(124,1,1,1).expand(124,2,2,3)
        for output in (frames, frames.unsqueeze(0)):
            decoder = types.SimpleNamespace(decode=lambda z: output)
            decoded = s.decode_video(decoder, None)
            self.assertEqual(decoded.shape, (124,2,2,3))
            self.assertTrue(torch.equal(decoded[-22:], frames[-22:]))

    def test_reference_cache_exact_images_only(self):
        calls = []
        vae = types.SimpleNamespace(encode=lambda frames: calls.append(frames.clone()) or frames.clone())
        cache = s.ReferenceVAECache(vae)
        image = torch.ones(1,2,2,3)
        cache.encode(image)
        cached = cache.encode(image.clone())
        cached.zero_()
        self.assertTrue(torch.equal(cache.encode(image), image))
        cache.encode(image * 2)
        cache.encode(image.repeat(22,1,1,1))
        cache.encode(image.repeat(22,1,1,1))
        self.assertEqual(len(calls), 4)
        self.assertEqual(cache.hits, 2)

    def test_plans(self):
        for n in [1,100,124,125,194,226,227,1000]:
            plan = plan_segments(n,124,22)
            emitted = sum(min(124-trim,n-start-trim) for start,trim in plan)
            self.assertEqual(emitted,n)
        with self.assertRaises(ValueError): plan_segments(200,123,22)
        with self.assertRaises(ValueError): plan_segments(200,124,0)

    def test_sigmas(self):
        validate_sigmas(torch.tensor([.4,.2,0]),True)
        for v in [[1,.5,0],[.4,.5,0],[.4,.2,.1],[float('nan'),0]]:
            with self.assertRaises(ValueError): validate_sigmas(torch.tensor(v),True)

    def test_lazy_audio_and_padding(self):
        from collections import UserDict
        audio = UserDict(waveform=torch.ones(1,2,1000),sample_rate=24000)
        out = s.slice_audio(audio,0,5)
        self.assertEqual(out['waveform'].shape[-1],5000)
        self.assertEqual(out['waveform'][...,-1].sum(),0)

    def run_fake(self, second, motion=False, source=False, device='cuda',precision='fp16',many_refs=False,standalone_only=False,tile_count=0,tile_overlap=128,encoded=None,whole_selflift=False):
        events = []
        modules = {}
        for name in ['comfy','comfy.nested_tensor','comfy.model_management','comfy_extras','comfy_extras.nodes_minimax_h3','comfy_extras.nodes_audio','node_helpers']:
            modules[name] = types.ModuleType(name)
        modules['comfy'].nested_tensor = modules['comfy.nested_tensor']
        modules['comfy.nested_tensor'].NestedTensor = Nested
        modules['comfy.model_management'].throw_exception_if_processing_interrupted = lambda:None
        h3 = modules['comfy_extras.nodes_minimax_h3']
        h3.FRAME_PER_TOKEN = (1,4,4,4,4)
        h3._empty_av_latent = empty
        h3.video_latent_t = lambda n: 2 if n<=5 else ((n-5)//17)*5+2
        h3.align_frame_count = lambda n: n+(5-n)%17
        h3.MiniMaxH3AddGuide = types.SimpleNamespace(execute=lambda cond,*args,**kwargs:(cond,))
        def reference(**kw):
            events.append(('reference',kw['length']))
            if many_refs:
                self.assertEqual((len(kw['ref_images']),len(kw['ref_videos']),len(kw['ref_video_audios']),len(kw['ref_audios'])),(9,3,3,3))
                for video in kw['ref_videos'].values():self.assertEqual(len(video),kw['length'])
                for audio in kw['ref_video_audios'].values():self.assertEqual(audio['waveform'].shape[-1],kw['length']*1000)
                self.assertIs(kw['ref_audios']['ref_audio_0'],standalone)
            if standalone_only:
                self.assertEqual(kw['ref_videos'],{})
                self.assertIs(kw['ref_audios']['ref_audio_0'],standalone)
            return [[torch.zeros(1,2,3),{}]],empty(kw['width'],kw['height'],kw['length'])[0]
        h3.MiniMaxH3ReferenceToVideo = types.SimpleNamespace(execute=reference)
        modules['comfy_extras.nodes_audio'].vae_decode_audio = lambda vae,lat:{
            'waveform':torch.ones(1,2,124000),'sample_rate':24000}
        calls = []
        def fake_sample(model,cond,neg,latent,sampler,sigmas,seed,cfg,mask=None):
            calls.append((seed,mask))
            if encoded in ('reuse_upstream','remove_reference_latents'):
                self.assertEqual(bool(cond[0][1].get('minimax_refs')),encoded=='reuse_upstream')
            if mask is not None:
                self.assertEqual(mask.tensors[1].count_nonzero(),0)
                self.assertEqual([c[0] for c in calls], [42,43,44,1000042])
                z = latent['samples'].tensors[0]
                self.assertEqual(z.shape[2],72)  # 243 aligned frames, no duplicated overlap
                self.assertTrue(torch.equal(z[0,0,:,0,0],torch.arange(72.)))
                self.assertFalse(any(d.get('minimax_keyframes') for _,d in cond))
                audio=latent['samples'].tensors[1]
                return {'samples':Nested((z,audio+999)),
                        '_denoised_samples':Nested((z+100,audio+555))}
            else:
                z = latent['samples'].tensors[0]
                z.copy_(torch.arange(z.shape[2]).reshape(1,1,-1,1,1) + (seed-42)*30)
            return {'samples':latent['samples'],
                    '_denoised_samples':Nested((z+100,latent['samples'].tensors[1]+3))}
        decoder = VAE()
        decode_calls = []
        decode_original = decoder.decode
        decoder.decode = lambda z: decode_calls.append(z.shape) or decode_original(z)
        params = dict(model=object(),positive=[[torch.zeros(1,2,3),{}]],latent=empty(32,32,124)[0],
            vae=decoder,audio_vae=object(),sampler_1=object(),sigmas_1=torch.tensor([1.,.5,0]),seed=42,
            cfg_1=1,target_frames=227,segment_frames=124,overlap_frames=22,
            continuation_mode='motion_context' if motion else 'official_image_guide',
            enable_second_pass=second,upscaler_model='test',upscale_factor=2,cfg_2=1,highres_tiling=second,
            sampler_2=object(),sigmas_2=torch.tensor([.3,0]))
        params.update(upscale_device=device,upscale_precision=precision)
        params.update(tile_count=tile_count,tile_overlap=tile_overlap)
        if source:
            params.update(source_video=torch.zeros(227,32,32,3),clip=object(),prompt='motion')
        standalone={'waveform':torch.ones(1,2,1000),'sample_rate':24000}
        if many_refs:
            params.update(clip=object(),prompt='motion')
            params.update({f'ref_image_{i}':torch.zeros(1,32,32,3) for i in range(9)})
            params.update({f'ref_video_{i}':torch.zeros(227,32,32,3) for i in range(3)})
            params.update({f'ref_video_audio_{i}':{'waveform':torch.ones(1,2,227000),'sample_rate':24000} for i in range(3)})
            params.update({f'ref_audio_{i}':standalone for i in range(3)})
        if standalone_only:
            params.update(clip=object(),prompt='motion',ref_audio_0=standalone)
        if encoded:
            params['positive'][0][1]['minimax_refs']=[{'kind':'video','latent':torch.ones(1,24,72,2,2)}]
        if whole_selflift:
            def whole_refiner(model,cond,neg,current,sampler,sigmas,seed,cfg,mode):
                self.assertEqual(len(calls),3)  # all first segments must finish first
                self.assertEqual(current['samples'].tensors[0].shape[2],72)
                events.append(('whole_selflift',72))
                video,audio=current['samples'].tensors
                high={'samples':Nested((video.repeat_interleave(2,-2).repeat_interleave(2,-1),audio))}
                return fake_sample(model,cond,neg,high,sampler,sigmas,seed,cfg,
                    Nested((torch.ones_like(high['samples'].tensors[0]),torch.zeros_like(audio))))
            params['_whole_video_refiner']=whole_refiner
        original_upscale = Upscale.execute
        def upscale(latent,**kwargs):
            self.assertEqual(kwargs['device'],device)
            self.assertEqual(kwargs['precision'],precision)
            events.append(('upscale',latent['samples'].shape[2]))
            return original_upscale(latent,**kwargs)
        def tiled(model,shapes,**kwargs):
            self.assertEqual(kwargs,dict(tile_count=tile_count,tile_overlap=tile_overlap))
            return model
        with patch.dict(sys.modules,modules), patch.object(s,'sample_once',fake_sample), \
             patch.object(Upscale,'execute',side_effect=upscale), \
             patch.object(s,'upscaler_class',lambda:Upscale),patch.object(s,'motion_class',lambda:Motion), \
             patch.object(s,'tiling_function',lambda:tiled):
            if encoded == 'normal_guard':
                with self.assertRaisesRegex(ValueError,'长视频参考不能'):
                    s.SolarLH3LongVideoSampler().run(**params)
                return
            out = s.SolarLH3LongVideoSampler().run(**params)
        self.assertEqual(out[0].shape,(227,64 if second else 32,64 if second else 32,3))
        self.assertEqual(out[1]['waveform'].shape[-1],227000)
        self.assertEqual(out[3],227)
        self.assertEqual(out[4]['samples'].tensors[0].shape[-1],2)
        self.assertEqual(len(calls),4 if second else 3)
        self.assertEqual(out[4]['samples'].tensors[0].shape[2],12)  # 39-frame final run
        self.assertEqual(len(decode_calls), 1 if second and motion else 3)
        self.assertEqual(len(out),8)
        self.assertEqual(s.SolarLH3LongVideoSampler.RETURN_NAMES[-2:],('output','denoised_output'))
        self.assertEqual(out[6]['frame_count'],227)
        self.assertEqual(out[7]['fps'],24.0)
        self.assertTrue(torch.equal(out[6]['samples'].tensors[0][0,0,:,0,0],torch.arange(72.)))
        self.assertTrue(torch.equal(out[7]['samples'].tensors[0][0,0,:,0,0],torch.arange(72.)+100))
        self.assertEqual(out[6]['samples'].tensors[1].shape[-1],405)
        self.assertEqual(out[6]['samples'].tensors[1].count_nonzero(),0)
        expected_audio=0 if second else 3
        self.assertTrue(torch.all(out[7]['samples'].tensors[1]==expected_audio))
        if source and second:
            self.assertEqual(events,[('reference',124),('reference',124),('reference',39),
                                     *([('reference',243),('whole_selflift',72)] if whole_selflift else [('upscale',72),('reference',243)])])

    def test_whole_selflift_after_join(self): self.run_fake(True,source=True,whole_selflift=True)

    def test_official_one_pass(self): self.run_fake(False)
    def test_official_two_pass_tiled(self): self.run_fake(True)
    def test_motion_one_pass(self): self.run_fake(False,True)
    def test_motion_two_pass_tiled(self): self.run_fake(True,True)
    def test_upscale_before_whole_reference_encoding(self): self.run_fake(True,source=True)
    def test_upscale_device_precision_forwarded(self): self.run_fake(True,device='cpu',precision='fp32')
    def test_all_reference_groups_reach_each_pass(self): self.run_fake(True,many_refs=True)
    def test_standalone_audio_without_video(self): self.run_fake(False,standalone_only=True)
    def test_manual_tile_settings_forwarded(self): self.run_fake(True,tile_count=4,tile_overlap=96)
    def test_normal_sampler_keeps_encoded_reference_guard(self): self.run_fake(False,encoded='normal_guard')



if __name__=='__main__': unittest.main()
