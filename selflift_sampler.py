"""Long-video orchestration around the private, unmodified SelfLift engine."""
from .sampler import SolarLH3LongVideoSampler


def selflift_class():
    from .vendor.selflift_runtime.nodes import SelfLiftH3Sampler
    return SelfLiftH3Sampler


class SolarLH3SelfLiftLongVideoSampler(SolarLH3LongVideoSampler):
    RETURN_TYPES = SolarLH3LongVideoSampler.RETURN_TYPES[:7]
    RETURN_NAMES = SolarLH3LongVideoSampler.RETURN_NAMES[:7]

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        omit = {'upscale_device','upscale_precision','tile_count','tile_overlap'}
        for group in schema.values():
            for name in omit: group.pop(name,None)
        upstream = selflift_class().INPUT_TYPES()
        explanations = {
            'transition_step':'整片SelfLift增强的低分辨率步数，必须小于sigmas_2总步数；2步增强先用1，即低分辨率1步、目标分辨率1步。',
            'lowres_scale':'整片增强低分辨率阶段相对目标宽高的比例。0自动尽量匹配拼接一采的尺寸，避免再次降到更小尺寸；非0使用指定比例。',
            'rho':'SelfLift像素VAE锚点修正比例。0配合放大模型跳过像素VAE往返；模型选none时必须大于0，可先试0.6。',
            'w_min':'像素锚点修正强度下限，不能超过w_max。仅rho大于0时参与修正。',
            'w_max':'像素锚点修正强度上限；范围0到1，必须不小于w_min。',
            'upscaler_model':'SelfLift升分辨率时使用的原生潜变量放大模型。none使用SelfLift-zero路线，需rho大于0。',
            'highres_tiling':'使用SelfLift原生高分辨率空间分块，整片自动选择1至8块；没有手动块数和重叠参数。',
            'upscaler_unload':'分辨率提升后卸载放大器，给H3目标分辨率采样释放显存；2080Ti建议开启。',
        }
        for name in explanations:
            definition = upstream['required'].get(name,upstream['optional'].get(name))
            options = dict(definition[1]) if len(definition)>1 else {}
            options['tooltip'] = explanations[name]
            if name == 'transition_step': options['default'] = 1
            if name == 'lowres_scale': options.update(default=0.0,min=0.0)
            if name in schema['required']: schema['required'][name] = (definition[0],options)
            else: schema['optional'][name] = (definition[0],options)
        schema['optional']['model_hires'] = ('MODEL',{'tooltip':'可选目标分辨率阶段模型；未接则使用model。低分辨率阶段始终使用model，两个模型必须兼容H3结构。'})
        schema['optional']['sampler_2'][1]['tooltip'] = '整片SelfLift增强只支持标准Euler；一采sampler_1仍使用标准版的采样器。'
        schema['optional']['sigmas_2'][1]['tooltip'] = '独立整片增强Sigma，0<起始值<1并以0结尾，至少2个采样步。低噪声尾段用于保留一采动作，transition_step必须小于其总步数。'
        return schema

    def run(self, model, positive, latent, vae, audio_vae, sampler_1, sigmas_1,
            seed, cfg_1, target_frames, segment_frames, overlap_frames, continuation_mode,
            enable_second_pass=True, upscaler_model='none', upscale_factor=1.5,
            cfg_2=1.0, highres_tiling=False, transition_step=1, lowres_scale=0.0,
            rho=0.0, w_min=0.5, w_max=1.0, upscaler_unload=True,
            model_hires=None, model_2=None, sampler_2=None, sigmas_2=None, **settings):
        engine = selflift_class() if enable_second_pass else None
        if enable_second_pass:
            if sigmas_2 is None or sampler_2 is None:
                raise ValueError('开启整片SelfLift增强需连接独立sampler_2和sigmas_2')
            if not 1 <= transition_step < len(sigmas_2)-1:
                raise ValueError(f'SelfLift低分辨率步数需小于二采总步数：transition_step={transition_step}，总步数={len(sigmas_2)-1}')
            if rho == 0 and upscaler_model == 'none':
                raise ValueError('SelfLift需选择放大模型；使用none时请将rho设为大于0，例如0.6')
        def progressive(second_model,cond,neg,current,sampler,sigmas,whole_seed,cfg,mode):
            import torch
            import math
            from comfy.nested_tensor import NestedTensor
            from .sampler import resize_guides
            video,audio = current['samples'].tensors
            h,w = video.shape[-2:]
            if mode['mode']=='scale by multiplier':
                ph,pw=h*16*mode['scale'],w*16*mode['scale']
            elif mode['mode']=='target dimensions':
                ph,pw=mode['height'],mode['width']
            else:
                ph=math.sqrt(mode['megapixels']*1024*1024*h/w)
                pw=ph*w/h
            th,tw=max(2,round(ph/32)*2),max(2,round(pw/32)*2)
            if th<h or tw<w:
                raise ValueError('SelfLift目标尺寸不能小于拼接一采尺寸')
            # Clean target-grid template, not another learned-upscaler pass.
            template=torch.nn.functional.interpolate(video.float(),size=(video.shape[2],th,tw),mode='nearest').to(video.dtype)
            scale=lowres_scale or min(h/th,w/tw)
            if not 0.25 <= scale <= 1.0:
                raise ValueError('SelfLift低分辨率比例必须在0.25至1；自动比例过小请减小放大倍数或手动设置')
            result=engine().sample(model=second_model,positive=resize_guides(cond,th,tw),
                negative=resize_guides(neg,th,tw),vae=vae,
                latent_image={'samples':NestedTensor((template,audio))},sampler=sampler,
                sigmas=sigmas,seed=whole_seed,cfg=cfg,transition_step=transition_step,
                lowres_scale=scale,rho=rho,w_min=w_min,w_max=w_max,
                upscaler_model=upscaler_model,model_hires=model_hires,
                highres_tiling=highres_tiling,upscaler_unload=upscaler_unload)[0]
            # The core restores concatenated first-pass audio after refinement.
            return result
        result=list(super().run(model=model,positive=positive,latent=latent,vae=vae,
            audio_vae=audio_vae,sampler_1=sampler_1,sigmas_1=sigmas_1,seed=seed,cfg_1=cfg_1,
            target_frames=target_frames,segment_frames=segment_frames,overlap_frames=overlap_frames,
            continuation_mode=continuation_mode,enable_second_pass=enable_second_pass,
            upscaler_model=upscaler_model,upscale_factor=upscale_factor,cfg_2=cfg_2,
            highres_tiling=False,model_2=model_2,sampler_2=sampler_2,sigmas_2=sigmas_2,
            _whole_video_refiner=progressive,**settings))
        result[5]=f'SelfLift whole-video refinement enabled={enable_second_pass}; standard first-pass segments joined first; first-pass audio preserved\n'+result[5]
        return tuple(result[:7])
