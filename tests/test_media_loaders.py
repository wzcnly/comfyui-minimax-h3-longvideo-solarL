import unittest
import types
from unittest.mock import patch
import torch
from test_sampler import package
from longvideo import media_loaders as m

class MediaTests(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(m.long_edge_size(1920,1080,720),(704,384))
        self.assertEqual(m.long_edge_size(1080,1920,720),(384,704))
        self.assertEqual(m.long_edge_size(640,352,1280),(640,352))
        self.assertEqual(m.long_edge_size(640,352,1280,True),(1280,704))
        self.assertEqual(m.image_size(2000,1000,1),(1414,707))
        self.assertEqual(m.image_size(320,160,1,True),(320,160))
        self.assertEqual(m.crop_box(1920,1080,'1:1'),(420,0,1500,1080))
    def test_alignment_stream(self):
        for n in range(5,240):
            result=list(m.aligned_stream(iter(range(n))))
            self.assertEqual(len(result),m.aligned_count(n))
            self.assertEqual(result,list(range(len(result))))
        self.assertEqual(len(list(m.aligned_stream(iter(range(194)),(4,1)))),141)
        with self.assertRaises(ValueError):list(m.aligned_stream(iter(range(4))))
    def test_custom_mode_is_exact_vhs_pass_through(self):
        called=[]
        expected=(torch.ones(10,2,2,3),10,None,{'loaded_duration':1})
        class VHS:
            def load_video(self,**kwargs):called.append(kwargs);return expected
        settings=dict(video='x.mp4',force_rate=0,custom_width=320,custom_height=640,
                      frame_load_cap=12,skip_first_frames=3,select_every_nth=2,
                      meta_batch='batch',vae='vae',format='Wan',unique_id='7')
        with patch.object(m,'vhs_provider',return_value=(VHS,types.SimpleNamespace())):
            out=m.SolarLLoadVideoUpload().load_video(**settings)
        self.assertEqual(called,[settings]);self.assertIs(out[0],expected[0])
    def test_alignment_wraps_generator_and_audio_lazily(self):
        lazy=types.SimpleNamespace(file='x',start_time=1,duration=9)
        audio_calls=[]
        def frames(**kwargs):
            yield (16,16,24,194/24,194,1/24,194,16,16,False)
            yield from range(194)
        class VHS:
            def load_video(self,**kwargs):
                gen=kwargs['generator'](meta_batch=None,unique_id='1',skip_first_frames=0,
                    select_every_nth=1,frame_load_cap=0)
                info=next(gen);self.assertion=info[6]
                result=list(gen)
                return result,len(result),lazy,{'loaded_duration':len(result)/24}
        module=types.SimpleNamespace(get_format=lambda f:{},resized_cv_frame_gen=frames,
            lazy_get_audio=lambda *a:audio_calls.append(a) or 'lazy-clipped')
        with patch.object(m,'vhs_provider',return_value=(VHS,module)):
            out=m.SolarLLoadVideoUpload().load_video(align_17n5=True,video='x')
        self.assertEqual(out[1],192)
        self.assertEqual(out[2],'lazy-clipped')
        self.assertEqual(audio_calls,[('x',1,8)])

if __name__=='__main__':unittest.main()
