"""Run with the ComfyUI embedded Python and --comfy-root PATH; no checkpoint/GPU generation."""
import argparse
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

parser=argparse.ArgumentParser()
parser.add_argument('--comfy-root',required=True)
args=parser.parse_args()
sys.path.insert(0,args.comfy_root)

class RejectOtherPlugins(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(name in fullname.lower() for name in ('apt_preset','comfyui-selflift','comfyui_minimax_h3_latent_upscaler')):
            raise AssertionError('External plugin import attempted: '+fullname)
        return None
sys.meta_path.insert(0,RejectOtherPlugins())

import torch
import comfy.nested_tensor
import comfy.model_base
from comfy_extras import nodes_minimax_h3 as h3
import nodes
nodes.NODE_CLASS_MAPPINGS={}

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builtin_test',root/'__init__.py',submodule_search_locations=[str(root)])
package=importlib.util.module_from_spec(spec)
sys.modules['builtin_test']=package
spec.loader.exec_module(package)
from builtin_test import sampler as s
from builtin_test import tiling
from builtin_test import selflift_sampler as sl
from builtin_test.vendor import upscale_3d

assert s.motion_class().__module__=='builtin_test.vendor.apt_motion'
assert s.upscaler_class().__module__=='builtin_test.vendor.upscale_3d'
assert s.tiling_function() is tiling.tiled_model
native_skeleton=object.__new__(comfy.model_base.MiniMaxH3)
wrappers={}
clone=types.SimpleNamespace(add_wrapper_with_key=lambda kind,key,fn:wrappers.setdefault(kind,fn))
patcher=types.SimpleNamespace(model=native_skeleton,clone=lambda:clone)
assert tiling.tiled_model(patcher,[(1,24,7,29,13),(1,32,2,37)]) is clone
assert len(wrappers)==2
assert len({id(fn.keywords['plan']) for fn in wrappers.values()})==1
schema=s.SolarLH3LongVideoSampler.INPUT_TYPES()
assert schema['optional']['upscale_device'][1]['default']=='cuda'
assert len(s.upscaler_choices())>0
assert sl.selflift_class().__module__=='builtin_test.vendor.selflift_runtime.nodes'
selflift_schema=sl.SolarLH3SelfLiftLongVideoSampler.INPUT_TYPES()
assert selflift_schema['optional']['transition_step'][1]['default']==1
assert 'sigmas_2' in selflift_schema['optional']
assert 'enable_second_pass' in selflift_schema['required']
print('PASS: native SelfLift runtime and Chinese schema load without an external plugin')
print('PASS: all 3 providers and schema load with an empty node registry and blocked external plugins')

current,_=h3._empty_av_latent(32,32,107)
previous,_=h3._empty_av_latent(32,32,124)
for tensor in previous['samples'].unbind():tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape))
condition=[[torch.zeros(1,2,6),{}]]
guided,trim=s.motion_class()().apply(condition,current,trim_frames=22,context_latent=previous)
assert trim==22
assert len(guided[0][1]['minimax_keyframes'])==7
assert guided[0][1]['minimax_refs'][-1]['ref_audio_t']==40
assert guided[0][1]['minimax_frame_count']==107
print('PASS: bundled Motion Context creates 7 video anchors + 40 audio steps on native H3')

# Exercise the real convolution implementation with a tiny randomly initialized model.
small=upscale_3d.LatentResizer3D(channels=32,in_blocks=1,out_blocks=1,dropout=0,attn=False,temporal_every=1).eval()
source=torch.randn(1,24,7,4,4)
with patch.object(upscale_3d,'load_model',return_value=small):
    for mode in ({'mode':'scale by multiplier','scale':1.5},
                 {'mode':'target dimensions','width':96,'height':96},
                 {'mode':'megapixels','megapixels':0.009216}):
        output=s.upscaler_class().execute({'samples':source},model_name='tiny-test',mode=mode,
            align=32,enable_temporal_chunking=True,force_unload=True,device='cpu',precision='fp32')[0]['samples']
        assert output.shape==(1,24,7,6,6),output.shape
        assert torch.isfinite(output).all()
print('PASS: bundled 3D convolution CPU inference works with all 3 sizing modes (tiny test model)')

video=torch.arange(24*7*29*13,dtype=torch.float32).reshape(1,24,7,29,13)/1000
audio=torch.ones(1,32,2,37)
context=torch.zeros(1,3,6)
payload={'keyframes':[{'latent':video[:,:,:1].clone(),'resolved_frame_index':0}],
         'refs':[{'kind':'image','latent':torch.ones(1,24,1,4,4),'latent_h':4,'latent_w':4}],
         'frame_count':22}
full=tiling.make_layout((3,7,30,14,37),payload)
calls=[]
def prediction(streams,timestep,context,options,minimax_payload=None,**kwargs):
    piece,a=streams
    layout=minimax_payload['layout']
    lo,hi=tiling.spatial_regions(29,3)[len(calls)]
    for (x,y,kind),(tx,ty,_) in zip(full.segments,layout.segments):
        expected=full.position_ids[x:y]
        if kind in ('cond','video'):
            expected=expected.reshape(-1,15,7,3)[:,lo//2:(hi+1)//2].reshape(-1,3)
        assert torch.equal(expected,layout.position_ids[tx:ty])
    assert kwargs['denoise_mask'].shape==piece.shape
    calls.append(piece.shape)
    return piece*2+3,a+len(calls)
result=tiling.model_tiles(prediction,[video,audio],torch.tensor([.5]),context,{},
    minimax_payload=payload,plan={'axis':3,'count':3},denoise_mask=torch.ones_like(video))
assert torch.allclose(result[0],video*2+3,atol=1e-5)
assert torch.equal(result[1],audio+1)
assert len(calls)==3
print('PASS: native PackedLayout positions, reference/guide handling, odd dimensions, masks and overlap blending')

wide=video.transpose(-1,-2).contiguous()
wide_payload={**payload,'keyframes':[{'latent':wide[:,:,:1].clone(),'resolved_frame_index':0}]}
wide_full=tiling.make_layout((3,7,14,30,37),wide_payload)
wide_calls=[]
def width_prediction(streams,timestep,context,options,minimax_payload=None,**kwargs):
    piece,a=streams
    lo,hi=tiling.spatial_regions(29,3)[len(wide_calls)]
    layout=minimax_payload['layout']
    for (x,y,kind),(tx,ty,_) in zip(wide_full.segments,layout.segments):
        expected=wide_full.position_ids[x:y]
        if kind in ('cond','video'):
            expected=expected.reshape(-1,7,15,3)[:,:,lo//2:(hi+1)//2].reshape(-1,3)
        assert torch.equal(expected,layout.position_ids[tx:ty])
    assert kwargs['denoise_mask'].unbind()[0].shape==piece.shape
    wide_calls.append(piece.shape)
    return piece*2+3,a
mask=comfy.nested_tensor.NestedTensor((torch.ones_like(wide),torch.zeros_like(audio)))
wide_result=tiling.model_tiles(width_prediction,[wide,audio],torch.tensor([.5]),context,{},
    minimax_payload=wide_payload,plan={'axis':4,'count':3},denoise_mask=mask)
assert torch.allclose(wide_result[0],wide*2+3,atol=1e-5)
assert torch.equal(wide_result[1],audio)
print('PASS: W-axis positions and nested AV masks remain aligned')

for length in range(1,65):
    for count in range(1,9):
        regions=tiling.spatial_regions(length,count)
        coverage=torch.zeros(length)
        for i,(lo,hi) in enumerate(regions):coverage[lo:hi]+=tiling.tile_weights(regions,i)
        assert torch.all(coverage>0),(length,count)
print('PASS: spatial regions cover every pixel for small/odd dimensions and all 1..8 tile counts')

class Planner:
    load_device=torch.device('cpu')
    model=types.SimpleNamespace(memory_required=lambda shape:100)
planner=Planner()
plan={'axis':3,'count':2,'overlap_pixels':32}
shapes=[tuple(video.shape),tuple(audio.shape)]
total=video.numel()+audio.numel()
received=[]
with patch.object(tiling,'workspace_bytes',return_value=5800000), \
     patch('comfy.sampler_helpers.estimate_memory',side_effect=lambda model,shape,conds:(shape[-1]*200,shape[-1]*100)):
    tiling.prepare_tiles(lambda model,shape,conds,**kw:received.append(shape),planner,(1,1,total),{},shapes=shapes,plan=plan)
assert 1<plan['count']<=8
assert received[0][-1]<total
print('PASS: spatial memory planning chooses a bounded tile count and reduced model load budget')

# Manual planning must retain the user's requested count regardless of the estimate.
large_shapes=[(1,24,62,118,66),(1,32,2,348)]
large_total=sum(__import__('math').prod(shape[1:]) for shape in large_shapes)
manual={'axis':3,'count':4,'requested_count':4,'overlap_pixels':128}
with patch.object(tiling,'workspace_bytes',return_value=1), \
     patch('comfy.sampler_helpers.estimate_memory',return_value=(10000,5000)):
    tiling.prepare_tiles(lambda *a,**kw:None,planner,(1,1,large_total),{},shapes=large_shapes,plan=manual)
assert manual['count']==4
assert manual['max_tile_pixels']==(1056,736),manual
assert abs(manual['max_tile_mp']-0.777216)<1e-9
auto={'axis':3,'count':2,'requested_count':0,'overlap_pixels':128}
with patch.object(tiling,'workspace_bytes',return_value=100000), \
     patch('comfy.sampler_helpers.estimate_memory',return_value=(10000,5000)):
    tiling.prepare_tiles(lambda *a,**kw:None,planner,(1,1,large_total),{},shapes=large_shapes,plan=auto)
assert auto['count']==3,auto
assert auto['max_tile_mp']<=1.0
assert tiling.spatial_regions(118,4,2)!=tiling.spatial_regions(118,4,4)
assert tiling.spatial_regions(4,64,1)==tiling.spatial_regions(4,1,1)
for count,overlap in ((65,128),(0,100),(0,0)):
    try:tiling.tiled_model(patcher,large_shapes,tile_count=count,tile_overlap=overlap)
    except ValueError:pass
    else:raise AssertionError('Invalid tile settings accepted')
print('PASS: manual count honored, exact overlap areas, auto 1MP target, grid limits and invalid-setting checks')
print('All native built-in checks passed. No full H3 checkpoint sampling was performed.')
