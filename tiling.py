"""solarL spatial model-evaluation tiling using ComfyUI's native H3 interfaces.

Independent implementation; no import of any other custom-node package.
Time remains global. Tile predictions overlap and are merged before sampler updates.
"""
from functools import partial
import inspect
import logging
import math

import torch
import comfy.ldm.common_dit
import comfy.model_base
import comfy.model_management as mm
import comfy.patcher_extension as pe
import comfy.sampler_helpers
from comfy.ldm.minimax.model import PackedLayout

LOG = logging.getLogger(__name__)


def spatial_regions(length, count, halo=4):
    """Split in H3 two-latent-pixel patches, keeping an overlap on both sides."""
    patches = (length + 1) // 2
    count = min(max(1,int(count)),max(1,patches // 2))
    if count == 1: return [(0,length)]
    halo = max(0,int(halo))
    boundaries = [i * patches // count for i in range(count+1)]
    return [(max(0,boundaries[i]-halo)*2,
             min(length,(boundaries[i+1]+halo)*2)) for i in range(count)]


def make_layout(signature, payload):
    args = {'keyframes':payload.get('keyframes'),'refs':payload.get('refs')}
    if 'frame_count' in inspect.signature(PackedLayout).parameters:
        args['frame_count'] = payload.get('frame_count')
    return PackedLayout(*signature,**args)


def spatial_payload(payload, context, video, audio, axis, lo, hi):
    """Crop target/keyframes and retain full references and absolute RoPE positions."""
    payload = payload or {}
    height,width = video.shape[-2:]
    hp,wp = (height+1)//2*2,(width+1)//2*2
    signature = (context.shape[1],video.shape[2],hp,wp,audio.shape[-1])
    original = payload.get('layout')
    if original is None or original.signature != signature:
        original = make_layout(signature,payload)
    result = dict(payload)
    if payload.get('keyframes'):
        guides = []
        for guide in payload['keyframes']:
            z = guide['latent']
            if z.shape[-2:] != (height,width):
                raise ValueError('solarL H3分块：Guide尺寸必须与目标潜变量一致')
            crop = z.narrow(axis,lo,hi-lo)
            guides.append({**guide,'latent':comfy.ldm.common_dit.pad_to_patch_size(crop,(1,2,2)).contiguous()})
        result['keyframes'] = guides
        if not payload.get('refs'):
            result['cond_video_latents'] = [guide['latent'] for guide in guides]
    h,w = (hi-lo,width) if axis == 3 else (height,hi-lo)
    layout = make_layout((context.shape[1],video.shape[2],(h+1)//2*2,(w+1)//2*2,audio.shape[-1]),result)
    if len(original.segments) != len(layout.segments):
        raise ValueError('solarL H3分块：完整与分块PackedLayout结构不一致')
    for (a,b,kind),(c,d,tile_kind) in zip(original.segments,layout.segments):
        if kind != tile_kind:
            raise ValueError('solarL H3分块：PackedLayout段类型不一致')
        positions = original.position_ids[a:b]
        if kind in ('cond','video'):
            grid = positions.reshape(-1,hp//2,wp//2,3)
            positions = grid.narrow(axis-2,lo//2,(hi-lo+1)//2).reshape(-1,3)
        layout.position_ids[c:d].copy_(positions)
    result['layout'] = layout
    return result


def tile_weights(regions, index):
    lo,hi = regions[index]
    weights = torch.ones(hi-lo,dtype=torch.float32,device='cpu')
    if index:
        overlap = min(hi,regions[index-1][1])-lo
        if overlap > 0: weights[:overlap] *= (torch.arange(overlap)+0.5)/overlap
    if index+1 < len(regions):
        overlap = hi-regions[index+1][0]
        if overlap > 0: weights[-overlap:] *= 1-(torch.arange(overlap)+0.5)/overlap
    return weights


def model_tiles(executor, streams, timestep, context, transformer_options,
                minimax_payload=None, *, plan, **kwargs):
    video,audio = streams
    axis = plan['axis']
    regions = spatial_regions(video.shape[axis],plan['count'],plan.get('overlap_pixels',128)//32)
    if len(regions) == 1:
        return executor(streams,timestep,context,transformer_options,minimax_payload=minimax_payload,**kwargs)
    if kwargs.get('control') is not None:
        raise ValueError('solarL H3空间分块暂不支持ControlNet')
    accumulated = torch.zeros(video.shape,dtype=torch.float32,device='cpu')
    coverage = torch.zeros(video.shape[axis],dtype=torch.float32,device='cpu')
    audio_prediction = None
    for i,(lo,hi) in enumerate(regions):
        mm.throw_exception_if_processing_interrupted()
        LOG.info('[solarL H3 tiling] tile %d/%d %s=%d:%d',i+1,len(regions),'H' if axis==3 else 'W',lo,hi)
        piece = video.narrow(axis,lo,hi-lo).contiguous()
        conditions = spatial_payload(minimax_payload,context,video,audio,axis,lo,hi)
        call_kwargs = dict(kwargs)
        # A spatial mask must follow the target crop as well as its AV stream.
        mask = call_kwargs.get('denoise_mask')
        if getattr(mask,'is_nested',False):
            vmask,amask = mask.unbind()
            if vmask.shape == video.shape:
                import comfy.nested_tensor
                call_kwargs['denoise_mask'] = comfy.nested_tensor.NestedTensor((vmask.narrow(axis,lo,hi-lo).contiguous(),amask))
        elif isinstance(mask,torch.Tensor) and mask.shape == video.shape:
            call_kwargs['denoise_mask'] = mask.narrow(axis,lo,hi-lo).contiguous()
        predicted_v,predicted_a = executor([piece,audio],timestep,context,
            transformer_options.copy(),minimax_payload=conditions,**call_kwargs)
        if predicted_v.shape != piece.shape:
            raise ValueError('solarL H3空间分块返回的视频尺寸不一致')
        blend = tile_weights(regions,i)
        shape = [1]*video.ndim
        shape[axis] = hi-lo
        accumulated.narrow(axis,lo,hi-lo).add_(predicted_v.detach().float().cpu()*blend.reshape(shape))
        coverage[lo:hi] += blend
        if audio_prediction is None: audio_prediction = predicted_a.detach().cpu().clone()
        del piece,conditions,predicted_v,predicted_a,call_kwargs
    if not torch.all(coverage>0): raise ValueError('solarL H3空间分块未覆盖全部像素')
    shape = [1]*video.ndim
    shape[axis] = video.shape[axis]
    accumulated /= coverage.reshape(shape)
    return [accumulated.to(video),audio_prediction.to(audio)]


def workspace_bytes(model):
    free = mm.get_free_memory(model.load_device)
    resident = 0
    seen = set()
    for entry in mm.loaded_models():
        patcher = entry if callable(getattr(entry,'loaded_size',None)) else entry.model
        if patcher.load_device != model.load_device or id(patcher.model) in seen: continue
        seen.add(id(patcher.model))
        resident += patcher.loaded_size()
    capacity = min(mm.get_total_memory(model.load_device),free+resident)
    weight_allowance = min(model.model_size(),capacity*mm.MIN_WEIGHT_MEMORY_RATIO)
    return max(0,capacity-weight_allowance-mm.minimum_inference_memory())


def prepare_tiles(executor, model, noise_shape, conds, model_options=None,
                  force_full_load=False, force_offload=False, *, shapes, plan):
    video,audio = shapes
    full_elements = math.prod(video[1:])+math.prod(audio[1:])
    if tuple(noise_shape) != (video[0],1,full_elements):
        raise ValueError('solarL H3分块规划收到不同尺寸的AV潜变量')
    if force_offload:
        plan['count'] = 1
        return executor(model,noise_shape,conds,model_options=model_options,
            force_full_load=force_full_load,force_offload=force_offload)
    available = workspace_bytes(model)
    # The full noise, latent, predictions and sampler state remain on the global timeline.
    state_reserve = full_elements*video[0]*4*8
    memory_per_element = max(1,model.model.memory_required((1,1,1)))
    requested = plan.get('requested_count',0)
    overlap = plan.get('overlap_pixels',128)
    counts = range(1,9) if requested == 0 else [requested]
    for count in counts:
        regions = spatial_regions(video[plan['axis']],count,overlap//32)
        tile = list(video)
        tile[plan['axis']] = max(b-a for a,b in regions)
        pixel_h,pixel_w = tile[3]*16,tile[4]*16
        max_mp = pixel_h*pixel_w/1_000_000
        tile[3],tile[4] = (tile[3]+1)//2*2,(tile[4]+1)//2*2
        elements = math.prod(tile[1:])+math.prod(audio[1:])+math.ceil(state_reserve/memory_per_element)
        budget_shape = (video[0],1,elements)
        # Native H3's estimator includes text/reference condition shapes itself.
        preferred,minimum = comfy.sampler_helpers.estimate_memory(model,budget_shape,conds)
        if minimum <= available and max_mp <= 1.0: break
    plan['count'] = len(regions)
    plan.update(max_tile_pixels=(pixel_w,pixel_h),max_tile_mp=max_mp)
    LOG.info('[solarL H3 tiling] plan mode=%s requested=%s axis=%s actual_tiles=%d overlap=%dpx/side shared~%dpx max_tile=%dx%d (%.3fMP) minimum=%.2f MiB workspace=%.2f MiB estimate_fits=%s',
        'auto' if requested==0 else 'manual',requested,'H' if plan['axis']==3 else 'W',plan['count'],
        overlap,2*overlap,pixel_w,pixel_h,max_mp,minimum/2**20,available/2**20,minimum<=available)
    for i,(lo,hi) in enumerate(regions):
        h,w = (hi-lo,video[4]) if plan['axis']==3 else (video[3],hi-lo)
        LOG.info('[solarL H3 tiling] tile %d/%d planned_size=%dx%d area=%.3fMP',
                 i+1,len(regions),w*16,h*16,w*h*256/1_000_000)
    if max_mp>1.0:
        LOG.warning('[solarL H3 tiling] largest tile exceeds the 1MP reference target; consider more tiles or less overlap')
    return executor(model,budget_shape if plan['count']>1 else noise_shape,conds,model_options=model_options,
        force_full_load=force_full_load,force_offload=force_offload)


def tiled_model(model, latent_shapes, tile_count=0, tile_overlap=128):
    if not isinstance(model.model,comfy.model_base.MiniMaxH3):
        raise ValueError('solarL空间分块仅支持MiniMax H3模型')
    if len(latent_shapes)!=2 or len(latent_shapes[0])!=5 or len(latent_shapes[1])!=4:
        raise ValueError('solarL空间分块需要视频和音频两个H3潜变量流')
    if not 0<=int(tile_count)<=64 or not 32<=int(tile_overlap)<=512 or int(tile_overlap)%32:
        raise ValueError('空间块数必须0至64；每侧重叠必须32至512像素且为32的倍数')
    video = latent_shapes[0]
    plan = {'axis':3 if video[3]>=video[4] else 4,'count':int(tile_count) or 2,
            'requested_count':int(tile_count),'overlap_pixels':int(tile_overlap)}
    result = model.clone()
    result._solarL_tiling_plan = plan
    result.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL,'solarL_h3_spatial_tiling',partial(model_tiles,plan=plan))
    result.add_wrapper_with_key(pe.WrappersMP.PREPARE_SAMPLING,'solarL_h3_spatial_tiling',
        partial(prepare_tiles,shapes=latent_shapes,plan=plan))
    return result
