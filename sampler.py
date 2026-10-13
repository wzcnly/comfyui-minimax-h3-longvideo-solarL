"""Serial H3 AV sampling. Optional integrations are resolved only when selected."""
import logging
import time

import torch

from .planning import plan_segments, segment_length, validate_sigmas

LOG = logging.getLogger(__name__)
REFERENCE_LIMITS = {'ref_image':9,'ref_video':3,'ref_video_audio':3,'ref_audio':3}

INPUT_TOOLTIPS = {
    "model": "一采使用的MiniMax H3模型。可在上游加载LoRA、设置Sigma Shift及CK分块加速。",
    "positive": "连接官方H3 Reference to Video的正向条件。请勿预接AddGuide；接续条件由本节点内部添加。",
    "latent": "连接官方H3 Reference to Video的AV LATENT，用于读取一采分辨率及默认帧数；不继续其中已有的生成内容。",
    "vae": "MiniMax H3视频VAE，用于逐段参考编码、官方图像接续和最终视频解码。",
    "audio_vae": "MiniMax H3音频VAE，用于参考音轨编码及一采音频解码；开启视频二采也保留一采音频。",
    "sampler_1": "一采采样算法，连接KSamplerSelect输出。使用普通采样，不使用SelfLift低分辨率渐进采样。",
    "sigmas_1": "一采实际使用的Sigma序列，连接上游调度器；必须严格递减并以0结尾。每段复用同一序列。",
    "seed": "第一段一采种子，后续每段递增1；整片二采使用起始seed加1000000。",
    "cfg_1": "一采条件引导强度。应匹配所用模型和Turbo LoRA；可先沿用已经验证有效的设置。",
    "target_frames": "最终成片帧数。0优先采用Video 1（ref_video_0）实际帧数；未接视频则采用输入LATENT的帧数。不能超过参考视频长度。",
    "segment_frames": "每段最大生成帧数，含重叠区。必须为17k+5且至少22帧；默认124。末段自动按剩余帧数向上对齐并缩短；每轮最多新增段长－重叠帧数。",
    "overlap_frames": "相邻段共享的接续帧数，支持5、22、39、56，且必须小于段长。默认22；后续段输出会去掉这部分。",
    "continuation_mode": "official_image_guide：上一段一采解码尾帧，再用官方AddGuide接续。motion_context：调用本插件内置Motion Context，直接使用上一段一采潜变量；不需要另装Apt_Preset。",
    "enable_second_pass": "关闭：只做普通分段一采。开启：先拼接全部一采视频潜变量，再整片3D放大并进行一次二采；音频保留一采。需sampler_2和sigmas_2。",
    "upscaler_model": "选择内置3D放大器使用的模型，支持子目录。权重放入models/latent_upscale_models；仅二采开启时加载，无需另装放大插件。",
    "upscale_factor": "二采前的视频空间放大倍率，例如1.5表示宽高各放大约1.5倍；目标尺寸按32像素对齐，帧数不变。",
    "upscale_device": "仅控制二采前3D潜变量放大器的设备：cuda用于NVIDIA，rocm用于AMD ROCm环境，cpu用于CPU。不会切换H3采样模型设备；2080Ti保持cuda。",
    "upscale_precision": "仅控制3D放大器的计算精度：fp16默认，fp32精度更高但通常占用更多内存，bf16需要设备及插件支持。不会切换H3采样精度；2080Ti保持fp16。",
    "upscale_mode": "3D放大目标计算方式：scale by multiplier按倍数（默认）；target dimensions指定宽高；megapixels按目标百万像素及原始宽高比计算。仅二采开启时使用，结果仍按32像素对齐。",
    "upscale_width": "指定宽高模式的目标像素宽度，放大器按32像素对齐。其他模式不使用。",
    "upscale_height": "指定宽高模式的目标像素高度，放大器按32像素对齐。其他模式不使用。",
    "upscale_megapixels": "百万像素模式的目标像素量，例如1.0约为100万像素；保持原始宽高比。其他模式不使用。",
    "cfg_2": "二采条件引导强度，独立于一采。用于放大后的视频细节增强；应匹配二采模型及LoRA。",
    "highres_tiling": "调用内置空间分块，仅在二采开启时生效。tile_count=0自动规划，其他值手动指定；每块保留完整时间、参考及音频。不能保证任意长度或与不分块完全等价。",
    "tile_count": "二采空间块数：0按显存估算和最大块约1MP的参考目标自动规划1至8块；1不切块；2至64手动指定。沿长边切条，实际数量受H3网格限制。块数越多重复计算通常越多，不保证更快；需同时开启二采和高分辨率分块。",
    "tile_overlap": "每块每侧扩展的重叠像素，默认128；相邻块共同处理区域约256像素。支持32至512，步进32。重叠计入每块实际面积；增大改善边界上下文但增加计算。只控制空间分块，与时间接续overlap_frames无关。",
    "negative": "可选负向条件。未连接时按当前段正向条件生成零文本条件；连接后各段复用此输入，请避免其中含有固定时间窗口的视频参考。",
    "model_2": "可选二采模型，未连接时使用一采model。应与一采共享H3架构和潜变量格式，可使用独立LoRA设置。",
    "sampler_2": "二采采样算法，独立于一采；开启二采时必须连接，关闭二采时不用。",
    "sigmas_2": "二采实际低噪声细化序列：首值必须大于0且小于1，严格递减并以0结尾。不能直接接从1开始的完整一采序列。",
    "source_video": "完整动作参考视频帧，连接VHS等加载节点的IMAGE。内部逐段裁剪和补尾帧；含视频参考的多段生成必须提供原始视频，不能仅靠已编码条件还原。",
    "source_audio": "可选源视频音轨，需与source_video同一起点、同一时间轴。内部同步裁剪，不足补静音；不接时仍由H3生成音频。最终音轨使用生成的一采音频。",
    "depth_video": "可选完整深度参考帧，作为第二个视频参考。必须与source_video帧数一致且同步；未连接就不启用。",
    "clip": "原始参考模式下必须连接H3使用的CLIP文本编码器，用于每段重新编码官方参考条件。",
    "prompt": "source_video模式下必须提供非空提示词，建议连接与官方节点相同的文本。逐段参考图片和视频的编号保持输入顺序。",
    "ref_image_0": "第一张原始参考图片，对应Picture 1。source_video模式需要重新提供官方节点使用的参考图片；从此接口开始连续连接。",
    "ref_image_1": "第二张原始参考图片，对应Picture 2。应与官方节点第二张图片一致；不要跳过ref_image_0。",
    "ref_image_2": "第三张原始参考图片，对应Picture 3。未使用第三张图片时可不接；图片接口不能留空档。",
    "ref_image_3": "第四张原始参考图片，对应Picture 4。接入后自动显示下一张，最多9张。",
}
for _prefix,_limit in REFERENCE_LIMITS.items():
    for _i in range(_limit):
        _name = f'{_prefix}_{_i}'
        if _prefix == 'ref_image':
            _tip = f'原始参考图片Picture {_i+1}，按顺序连续连接；接入后显示下一口，最多9张。'
        elif _prefix == 'ref_video':
            _tip = f'完整参考视频Video {_i+1}，24FPS。按源时间轴逐段裁剪；最多3路，各路必须覆盖目标帧数。Video 1提供target_frames=0的实际帧数。'
        elif _prefix == 'ref_video_audio':
            _tip = f'Video {_i+1}的配对参考音轨，与同编号视频同步裁剪，缺少部分补静音。最多3路；独立音频请接ref_audio接口。'
        else:
            _tip = f'独立参考音频Audio（第{_i+1}路），完整素材作为各段参考，不按动作视频时间轴裁剪；最多3路。不是成片输出音轨。'
        INPUT_TOOLTIPS[_name] = _tip


def reference_inputs(images, source_video, source_audio, depth_video, extra):
    pictures = {f'ref_image_{i}':img for i,img in enumerate(images) if img is not None}
    for i in range(4,9):
        value = extra.get(f'ref_image_{i}')
        if value is not None: pictures[f'ref_image_{i}'] = value
    videos = {'ref_video_0':source_video,'ref_video_1':depth_video}
    video_audio = {'ref_video_audio_0':source_audio}
    for prefix,group in (('ref_video',videos),('ref_video_audio',video_audio)):
        for i in range(3):
            value = extra.get(f'{prefix}_{i}')
            if value is not None:
                old = group.get(f'{prefix}_{i}')
                if old is not None and old is not value:
                    raise ValueError(f'{prefix}_{i}与旧接口重复连接了不同素材，请只使用一种接口')
                group[f'{prefix}_{i}'] = value
    videos = {k:v for k,v in videos.items() if v is not None}
    video_audio = {k:v for k,v in video_audio.items() if v is not None}
    audios = {f'ref_audio_{i}':extra[f'ref_audio_{i}'] for i in range(3) if extra.get(f'ref_audio_{i}') is not None}
    for prefix,group in (('ref_image',pictures),('ref_video',videos),('ref_audio',audios)):
        if list(group) != [f'{prefix}_{i}' for i in range(len(group))]:
            raise ValueError(f'{prefix}参考需从0开始连续连接，不能留空档')
    for name in video_audio:
        if name.replace('ref_video_audio_','ref_video_') not in videos:
            raise ValueError(f'{name}必须连接同编号参考视频；独立音频请使用ref_audio接口')
    return pictures,videos,video_audio,audios


def upscale_mode_options(mode, factor, width, height, megapixels):
    if mode == 'scale by multiplier': return {'mode':mode,'scale':factor}
    if mode == 'target dimensions': return {'mode':mode,'width':width,'height':height}
    if mode == 'megapixels': return {'mode':mode,'megapixels':megapixels}
    raise ValueError('未知3D放大模式')


def upscaler_choices():
    """Bundled 3D scanner, retaining the standard model directory and subfolders."""
    from .vendor.upscale_3d import scan_models
    return scan_models()


def upscaler_class():
    from .vendor.upscale_3d import MinimaxH3LatentUpscaler3D
    return MinimaxH3LatentUpscaler3D


def motion_class():
    from .vendor.apt_motion import AptMiniMaxH3MotionContext
    return AptMiniMaxH3MotionContext


def tiling_function():
    from .tiling import tiled_model
    return tiled_model


def slice_images(images, start, length):
    part = images[start:start + length]
    if not len(part):
        raise ValueError("参考视频比目标时间轴短，当前起点没有参考帧")
    if len(part) < length:
        part = torch.cat((part, part[-1:].repeat(length - len(part), 1, 1, 1)))
    return part


def decode_video(vae, latent):
    images = vae.decode(latent)
    # Match stock VAEDecode: video VAEs return [B,T,H,W,C].
    if images.ndim == 5:
        if images.shape[0] != 1:
            raise ValueError("视频解码仅支持batch_size=1")
        images = images.reshape(-1, *images.shape[-3:])
    if images.ndim != 4:
        raise ValueError(f"视频VAE解码形状异常：{tuple(images.shape)}")
    return images.cpu()


class ReferenceVAECache:
    """Run-local cache for identical single-image references, never video windows."""
    def __init__(self, vae):
        self.vae = vae
        self.images = []
        self.hits = 0

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, frames):
        if frames.shape[0] != 1:
            return self.vae.encode(frames)
        key = frames.detach().cpu()
        for image, latent, device in self.images:
            if image.shape == key.shape and image.dtype == key.dtype and torch.equal(image, key):
                self.hits += 1
                return latent.to(device=device).clone()
        latent = self.vae.encode(frames)
        self.images.append((key.clone(), latent.detach().cpu().clone(), latent.device))
        return latent


def slice_audio(audio, start, length):
    if audio is None:
        return None
    sr = audio['sample_rate']
    wave = audio['waveform']
    a, b = round(start * sr / 24), round((start + length) * sr / 24)
    clip = wave[..., a:b]
    if clip.shape[-1] == b-a:
        return {"waveform": clip, "sample_rate": sr}
    out = wave.new_zeros((*wave.shape[:-1], b-a))
    out[..., :clip.shape[-1]] = clip
    return {"waveform":out, "sample_rate":sr}


def sample_once(model, cond, negative, latent, sampler, sigmas, seed, cfg, mask=None):
    import comfy.sample
    import comfy.model_management
    import comfy.utils
    comfy.model_management.throw_exception_if_processing_interrupted()
    steps = len(sigmas)-1
    progress = comfy.utils.ProgressBar(steps)
    progress.update_absolute(0,steps)
    x0_output = {}
    def callback(step, x0, x, total_steps):
        comfy.model_management.throw_exception_if_processing_interrupted()
        x0_output['x0'] = x0
        progress.update_absolute(step+1,total_steps)
    noise = comfy.sample.prepare_noise(latent['samples'], seed)
    result = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas,
                                       cond, negative, latent['samples'], noise_mask=mask, callback=callback, seed=seed)
    output = latent.copy()
    output['samples'] = result
    x0 = x0_output.get('x0')
    if x0 is not None:
        # Match ComfyUI SamplerCustomAdvanced's packed AV x0 conversion.
        if result.is_nested and not x0.is_nested:
            import comfy.nested_tensor
            x0 = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(x0, [z.shape for z in result.unbind()]))
        output['_denoised_samples'] = model.model.process_latent_out(x0.cpu())
    return output


def join_av_latents(video_parts, audio_parts, template_audio, total, h3):
    """Join on H3's video-token and global 40-Hz audio grids, retaining tail padding."""
    import comfy.nested_tensor
    video = torch.cat(video_parts,dim=2)
    full_length = h3.align_frame_count(max(22,total))
    if video.shape[2] != h3.video_latent_t(full_length):
        raise ValueError("一采潜变量拼接后的时间长度不一致")
    audio = template_audio.new_zeros((*template_audio.shape[:-1],round(full_length*40/24)),device='cpu')
    for a,b,part in audio_parts:
        count = min(b-a,part.shape[-1])
        audio[...,a:a+count] = part[...,:count]
        if count < b-a:
            audio[...,a+count:b] = part[...,-1:]
    used = round(total*40/24)
    if used < audio.shape[-1]:
        audio[...,used:] = audio[...,used-1:used].clone()
    return {'samples':comfy.nested_tensor.NestedTensor((video,audio)), 'frame_count':total, 'fps':24.0}


def resize_guides(cond, height, width):
    result = []
    for embedding, metadata in cond:
        metadata = metadata.copy()
        guides = []
        for kf in metadata.get('minimax_keyframes', []):
            kf = kf.copy()
            z = kf.get('latent')
            if z is not None:
                b,c,t,h,w = z.shape
                z = torch.nn.functional.interpolate(z.permute(0,2,1,3,4).reshape(b*t,c,h,w).float(),
                                                    size=(height,width),mode='bilinear',align_corners=False).to(z)
                kf['latent'] = z.reshape(b,t,c,height,width).permute(0,2,1,3,4)
            guides.append(kf)
        metadata['minimax_keyframes'] = guides
        result.append([embedding, metadata])
    return result


class SolarLH3LongVideoSampler:
    CATEGORY = "solarL/MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "LATENT", "STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "last_first_pass_latent", "report", "output", "denoised_output")

    @classmethod
    def INPUT_TYPES(cls):
        choices = upscaler_choices()
        preferred = 'minimax_h3_latent_upscaler_3d_fp16.safetensors'
        schema = {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "latent": ("LATENT",),
            "vae": ("VAE",), "audio_vae": ("VAE",),
            "sampler_1": ("SAMPLER",), "sigmas_1": ("SIGMAS",),
            "seed": ("INT", {"default":0,"min":0,"max":0xffffffffffffffff,"control_after_generate":True}),
            "cfg_1": ("FLOAT", {"default":1.0,"min":0.0,"max":100.0}),
            "target_frames": ("INT", {"default":0,"min":0,"max":100000,
                                      "tooltip":"0优先使用原始source_video实际帧数，否则使用输入LATENT长度"}),
            "segment_frames": ("INT", {"default":124,"min":22,"max":362,"step":17}),
            "overlap_frames": ([22,5,39,56],),
            "continuation_mode": (["official_image_guide","motion_context"],),
            "enable_second_pass": ("BOOLEAN", {"default":False,
                "label_on":"二采放大：开启", "label_off":"二采放大：关闭",
                "tooltip":"关闭仅普通一采；开启3D潜变量放大及二采，保留一采音频，需连接sampler_2和sigmas_2"}),
            "upscaler_model": (choices, {"default":preferred if preferred in choices else choices[0],
                "tooltip":"选择已安装3D放大器扫描到的模型；仅二采放大开启时加载"}),
            "upscale_factor": ("FLOAT", {"default":1.5,"min":1.0,"max":4.0,"step":0.05}),
            "cfg_2": ("FLOAT", {"default":1.0,"min":0.0,"max":100.0}),
            "highres_tiling": ("BOOLEAN", {"default":False,
                "label_on":"高分辨率分块：开启", "label_off":"高分辨率分块：关闭"}),
        }, "optional": {
            "negative": ("CONDITIONING",), "model_2": ("MODEL",),
            "sampler_2": ("SAMPLER",), "sigmas_2": ("SIGMAS",),
            "source_video": ("IMAGE",), "source_audio": ("AUDIO",), "depth_video": ("IMAGE",),
            "clip": ("CLIP",), "prompt": ("STRING", {"default":"","multiline":True}),
            "ref_image_0": ("IMAGE",), "ref_image_1": ("IMAGE",), "ref_image_2": ("IMAGE",),
            "ref_image_3": ("IMAGE",),
            # Append widgets so existing positional widget values retain their indices.
            "upscale_device": (["cuda","rocm","cpu"], {"default":"cuda"}),
            "upscale_precision": (["fp32","fp16","bf16"], {"default":"fp16"}),
            "upscale_mode": (["scale by multiplier","target dimensions","megapixels"], {"default":"scale by multiplier"}),
            "upscale_width": ("INT", {"default":1280,"min":64,"max":8192,"step":8}),
            "upscale_height": ("INT", {"default":704,"min":64,"max":8192,"step":8}),
            "upscale_megapixels": ("FLOAT", {"default":1.0,"min":0.1,"max":16.0,"step":0.1}),
            "tile_count": ("INT", {"default":0,"min":0,"max":64,"step":1}),
            "tile_overlap": ("INT", {"default":128,"min":32,"max":512,"step":32}),
        }}
        for prefix,limit in REFERENCE_LIMITS.items():
            for i in range(limit):
                name = f'{prefix}_{i}'
                schema['optional'].setdefault(name, ('IMAGE' if prefix in ('ref_image','ref_video') else 'AUDIO',))
        for group in schema.values():
            for name, definition in group.items():
                options = dict(definition[1]) if len(definition) > 1 else {}
                options["tooltip"] = INPUT_TOOLTIPS[name]
                group[name] = (definition[0], options)
        return schema

    @torch.inference_mode()
    def run(self, model, positive, latent, vae, audio_vae, sampler_1, sigmas_1,
            seed, cfg_1, target_frames, segment_frames, overlap_frames,
            continuation_mode, enable_second_pass, upscaler_model, upscale_factor,
            cfg_2, highres_tiling, negative=None, model_2=None, sampler_2=None, sigmas_2=None,
            source_video=None, source_audio=None, depth_video=None, clip=None, prompt="",
            ref_image_0=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            upscale_device="cuda", upscale_precision="fp16", upscale_mode="scale by multiplier",
            upscale_width=1280, upscale_height=704, upscale_megapixels=1.0,
            tile_count=0, tile_overlap=128, _whole_video_refiner=None, **extra_references):
        from comfy_extras import nodes_minimax_h3 as h3
        from comfy_extras.nodes_audio import vae_decode_audio
        import comfy.nested_tensor
        import comfy.model_management as mm
        import node_helpers
        if continuation_mode not in ('official_image_guide','motion_context'):
            raise ValueError("未知接续模式")
        validate_sigmas(sigmas_1)
        streams = getattr(latent.get('samples'), 'tensors', None)
        if streams is None or len(streams)!=2 or streams[0].ndim!=5 or streams[0].shape[1]!=24:
            raise ValueError("请连接官方H3 Reference to Video的AV LATENT")
        if streams[0].shape[0] != 1:
            raise ValueError("当前版本仅支持batch_size=1")
        pictures,reference_videos,reference_video_audio,reference_audio = reference_inputs(
            [ref_image_0,ref_image_1,ref_image_2,ref_image_3],source_video,source_audio,depth_video,extra_references)
        source_video = reference_videos.get('ref_video_0')
        rebuild = source_video is not None or bool(reference_audio) or (bool(pictures) and clip is not None and bool(prompt.strip()))
        total = target_frames or (len(source_video) if source_video is not None else
                                 sum(h3.FRAME_PER_TOKEN[k%5] for k in range(streams[0].shape[2])))
        plan = plan_segments(total, segment_frames, int(overlap_frames))
        refs = positive[0][1].get('minimax_refs', [])
        if any(d.get('minimax_keyframes') for _,d in positive):
            raise ValueError("请使用干净的ReferenceToVideo条件；外部首尾Guide没有全片时间映射，不能直接复用到每段")
        if len(plan)>1 and any(r.get('kind') in ('video','video_audio','audio') for r in refs) and not rebuild:
            raise ValueError("长视频参考不能从已编码CONDITIONING准确还原；请额外接ref_video_0、clip、prompt和原始参考图片")
        if rebuild:
            if clip is None or not prompt.strip():
                raise ValueError("原始参考模式需连接CLIP和非空prompt，用于逐段重建官方参考条件")
            if len(pictures) < sum(r.get('kind')=='image' for r in refs):
                raise ValueError("请把官方节点使用的原始参考图片也接入ref_image接口")
            if len(pictures) and list(pictures)!=[f'ref_image_{i}' for i in range(len(pictures))]:
                raise ValueError("参考图片接口需从ref_image_0开始连续连接，不能留空档")
            for name,frames in reference_videos.items():
                if frames.ndim != 4 or len(frames)<1:
                    raise ValueError(f'{name}必须为非空IMAGE帧批次')
                if total>len(frames):
                    raise ValueError(f'{name}帧数不足以覆盖目标时间轴；0自动采用Video 1帧数')
        elif depth_video is not None or source_audio is not None:
            raise ValueError("参考视频音轨需要同时连接对应参考视频")
        motion = motion_class() if continuation_mode=='motion_context' and len(plan)>1 else None
        upscale = None
        tiled = None
        if enable_second_pass:
            upscale_options = upscale_mode_options(upscale_mode,upscale_factor,upscale_width,upscale_height,upscale_megapixels)
            if upscale_device not in ('cuda','rocm','cpu') or upscale_precision not in ('fp32','fp16','bf16'):
                raise ValueError("3D放大器的device或precision选项无效")
            if sigmas_2 is None or sampler_2 is None:
                raise ValueError("开启二采时必须连接独立sigmas_2和sampler_2")
            validate_sigmas(sigmas_2, refine=True)
            upscale = upscaler_class() if _whole_video_refiner is None else None
            if _whole_video_refiner is None and upscaler_model.startswith('('):
                raise ValueError("请在models/latent_upscale_models目录放入3D放大模型后选择；放大器实现已内置")
            tiled = tiling_function() if highres_tiling and _whole_video_refiner is None else None
        negative_base = negative
        previous = None
        previous_tail = None
        reference_vae = ReferenceVAECache(vae)
        image_parts, audio_parts, reports = [], [], []
        video_latent_parts, audio_latent_parts = [], []
        denoised_video_parts, denoised_audio_parts = [], []
        sample_rate = None
        width,height = streams[0].shape[-1]*16, streams[0].shape[-2]*16
        for index,(start,trim) in enumerate(plan):
            mm.throw_exception_if_processing_interrupted()
            tick = time.perf_counter()
            length = segment_length(total, start, segment_frames)
            LOG.info("[solarL H3] first pass %d/%d start=%d generated=%d trim=%d",index+1,len(plan),start,length,trim)
            cond = positive
            if rebuild:
                videos = {name:slice_images(frames,start,length) for name,frames in reference_videos.items()}
                audios = {name:slice_audio(sound,start,length) for name,sound in reference_video_audio.items()}
                cond,current = h3.MiniMaxH3ReferenceToVideo.execute(
                    clip=clip,prompt=prompt,width=width,height=height,length=length,
                    ref_image_size='max',vae=reference_vae,audio_vae=audio_vae,ref_images=pictures,
                    ref_videos=videos,ref_video_audios=audios,ref_audios=reference_audio)[:2]
                del videos, audios
            else:
                current,_ = h3._empty_av_latent(width,height,length)
            if previous is not None:
                if motion is not None:
                    cond,actual_trim = motion().apply(cond,current,trim_frames=int(overlap_frames),context_latent=previous)
                    if actual_trim!=trim:
                        raise ValueError("Motion Context返回重叠帧数与分段计划不一致")
                else:
                    cond = h3.MiniMaxH3AddGuide.execute(cond,current,0,vae=vae,image=previous_tail)[0]
            prepared = time.perf_counter()
            neg = negative_base if negative_base is not None else [[e,d.copy()] for e,d in cond]
            if negative_base is None:
                neg = [[torch.zeros_like(e),d] for e,d in neg]
                for _,d in neg:
                    if isinstance(d.get('pooled_output'),torch.Tensor):
                        d['pooled_output'] = torch.zeros_like(d['pooled_output'])
            first = sample_once(model,cond,neg,current,sampler_1,sigmas_1,(seed+index)%2**64,cfg_1)
            sampled = time.perf_counter()
            previous = {k:v for k,v in first.items() if k != '_denoised_samples'}
            video,audio = first['samples'].tensors
            denoised_video,denoised_audio = first.get('_denoised_samples',first['samples']).tensors
            first_decoded = None
            if motion is None and index + 1 < len(plan):
                first_decoded = decode_video(vae, video)
                previous_tail = first_decoded[-int(overlap_frames):].clone()
                if enable_second_pass:
                    # Only the cloned tail is needed; do not keep low-res RGB during refinement.
                    first_decoded = None
            refined_at = time.perf_counter()
            take = min(length-trim,total-(start+trim))
            trim_tokens = h3.video_latent_t(trim) if trim else 0
            video_latent_parts.append(video[:,:,trim_tokens:].cpu().clone())
            audio_start = round((start+trim)*40/24)
            audio_end = round((start+trim+take)*40/24)
            audio_latent_parts.append((audio_start,audio_end,
                audio[...,round(trim*40/24):].cpu().clone()))
            if not enable_second_pass:
                denoised_video_parts.append(denoised_video[:,:,trim_tokens:].cpu().clone())
                denoised_audio_parts.append((audio_start,audio_end,
                    denoised_audio[...,round(trim*40/24):].cpu().clone()))
                decoded = first_decoded if first_decoded is not None else decode_video(vae, video)
                if decoded.shape[0]<length:
                    raise ValueError(f"VAE解码帧数不足：实际{decoded.shape[0]}，需要{length}")
                image_parts.append(decoded[trim:trim+take].clone())
                del decoded
            sound = vae_decode_audio(audio_vae,first)
            clipped = slice_audio(sound,trim,take)
            if sample_rate is not None and clipped['sample_rate']!=sample_rate:
                raise ValueError("各段音频采样率不一致")
            sample_rate = clipped['sample_rate']
            audio_parts.append(clipped['waveform'].cpu().clone())
            reports.append(f"segment {index+1}/{len(plan)} start={start} generated={length} trim={trim} output={take} "
                           f"prepare={prepared-tick:.2f}s sample1={sampled-prepared:.2f}s "
                           f"post_sample1={refined_at-sampled:.2f}s decode_audio={time.perf_counter()-refined_at:.2f}s "
                           f"image_cache_hits={reference_vae.hits} elapsed={time.perf_counter()-tick:.2f}s")
            LOG.info(reports[-1])
            del current,cond,neg,video,audio,first,first_decoded,sound,clipped,denoised_video,denoised_audio
        output = join_av_latents(video_latent_parts,audio_latent_parts,streams[1],total,h3)
        video_latent_parts.clear()
        audio_latent_parts.clear()
        if enable_second_pass:
            mm.throw_exception_if_processing_interrupted()
            tick = time.perf_counter()
            LOG.info("[solarL H3] whole-video second pass: joining latents, target=%d upscale=%s tiling=%s sigmas=%s",
                     total,upscale_options,highres_tiling,sigmas_2.tolist())
            video,audio = output['samples'].tensors
            full_length = h3.align_frame_count(max(22,total))
            # Run the external upscaler before Qwen reference encoding to reduce VRAM contention.
            output_video = video if _whole_video_refiner else upscale.execute({'samples':video},model_name=upscaler_model,
                mode=upscale_options,align=32,
                enable_temporal_chunking=True,force_unload=True,device=upscale_device,precision=upscale_precision)[0]['samples']
            # Rebuild conditioning on the complete time axis, without segment-local guides.
            full_cond = positive
            if rebuild:
                videos = {name:slice_images(frames,0,full_length) for name,frames in reference_videos.items()}
                audios = {name:slice_audio(sound,0,full_length) for name,sound in reference_video_audio.items()}
                full_cond = h3.MiniMaxH3ReferenceToVideo.execute(
                    clip=clip,prompt=prompt,width=width,height=height,length=full_length,
                    ref_image_size='max',vae=reference_vae,audio_vae=audio_vae,
                    ref_images=pictures,ref_videos=videos,ref_video_audios=audios,ref_audios=reference_audio)[0]
                del videos,audios
            full_neg = negative_base
            if full_neg is None:
                full_neg = [[torch.zeros_like(e),d.copy()] for e,d in full_cond]
                for _,d in full_neg:
                    if isinstance(d.get('pooled_output'),torch.Tensor):
                        d['pooled_output'] = torch.zeros_like(d['pooled_output'])
            high = {'samples':comfy.nested_tensor.NestedTensor((output_video,audio))}
            mask = comfy.nested_tensor.NestedTensor((torch.ones_like(output_video),torch.zeros_like(audio)))
            second_model = model_2 or model
            if tiled:
                second_model = tiled(second_model,[tuple(output_video.shape),tuple(audio.shape)],
                                     tile_count=tile_count,tile_overlap=tile_overlap)
            high_cond = resize_guides(full_cond,*output_video.shape[-2:])
            high_neg = resize_guides(full_neg,*output_video.shape[-2:])
            if _whole_video_refiner:
                refined = _whole_video_refiner(second_model,high_cond,high_neg,high,sampler_2,sigmas_2,
                                              (seed+1000000)%2**64,cfg_2,upscale_options)
            else:
                refined = sample_once(second_model,high_cond,high_neg,high,sampler_2,sigmas_2,
                                      (seed+1000000)%2**64,cfg_2,mask)
            tile_plan = getattr(second_model,'_solarL_tiling_plan',None) if tiled else None
            if tile_plan and 'max_tile_pixels' in tile_plan:
                tw,th = tile_plan['max_tile_pixels']
                reports.append(f"spatial_tiling requested={tile_count} actual={tile_plan['count']} "
                               f"overlap={tile_overlap}px/side max_tile={tw}x{th} "
                               f"max_tile_mp={tile_plan['max_tile_mp']:.3f}")
            final_video = refined['samples'].tensors[0].cpu()
            predicted_video = refined.get('_denoised_samples',refined['samples']).tensors[0].cpu()
            # Both final latent outputs retain the merged first-pass audio.
            output = {'samples':comfy.nested_tensor.NestedTensor((final_video,audio)), 'frame_count':total, 'fps':24.0}
            denoised_output = {'samples':comfy.nested_tensor.NestedTensor((predicted_video,audio)), 'frame_count':total, 'fps':24.0}
            images = decode_video(vae,refined['samples'].tensors[0])[:total].clone()
            if len(images)!=total:
                raise ValueError("完整二采解码帧数不足")
            reports.append(f"whole_video_second_pass generated={full_length} output={total} elapsed={time.perf_counter()-tick:.2f}s")
            LOG.info(reports[-1])
        else:
            images = torch.cat(image_parts,dim=0)
            denoised_output = join_av_latents(denoised_video_parts,denoised_audio_parts,streams[1],total,h3)
            denoised_video_parts.clear()
            denoised_audio_parts.clear()
        waveform = torch.cat(audio_parts,dim=-1)
        expected = round(total*sample_rate/24)
        waveform = waveform[..., :expected]
        if waveform.shape[-1]<expected:
            waveform = torch.nn.functional.pad(waveform,(0,expected-waveform.shape[-1]))
        return images,{'waveform':waveform,'sample_rate':sample_rate},24.0,total,previous,'\n'.join(reports),output,denoised_output


