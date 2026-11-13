"""Experimental pixel-context continuation and two-sided local video editing."""
from collections.abc import Mapping

import torch

from .sampler import SolarLH3LongVideoSampler, slice_audio

FPS = 24
MODES = ["replace_range", "insert_at", "extend", "bridge"]


def optional_audio(audio, name):
    """Normalize an omitted/null/zero-sample AUDIO input to no source sound."""
    if audio is None:
        return None
    # VHS uses LazyAudioMap: a valid AUDIO Mapping, not a built-in dict.
    if not isinstance(audio, Mapping):
        raise ValueError(f"{name}需为AUDIO或None")
    wave = audio.get('waveform')
    if (not isinstance(wave, torch.Tensor) or wave.ndim != 3
            or wave.shape[0] != 1 or wave.shape[1] not in (1, 2)):
        raise ValueError(f"{name}需为batch=1的单声道或双声道AUDIO")
    try:
        rate = int(audio['sample_rate'])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError(f"{name}需提供有效音频采样率") from None
    if rate <= 0:
        raise ValueError("音频采样率必须大于0")
    return audio if wave.shape[-1] else None


def audio_guide(audio, start, frames):
    """Only anchor available source sound, never invent a silent guide."""
    if audio is None:
        return None
    rate = int(audio['sample_rate'])
    begin, end = round(start * rate / FPS), round((start + frames) * rate / FPS)
    if rate <= 0:
        raise ValueError("音频采样率必须大于0")
    wave = audio['waveform']
    if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] not in (1, 2):
        raise ValueError("局部编辑音频仅支持batch=1、单声道或双声道")
    # Missing/truncated source audio should not pin generated sound to silence.
    if wave.shape[-1] < end or begin >= end:
        return None
    clip = wave[..., begin:end].detach().cpu().clone()
    if clip.shape[1] == 1:
        clip = clip.repeat(1, 2, 1)
    return {'waveform': clip, 'sample_rate': rate}


def blend_audio_edges(a, b, c, generated, before, frames, rate, fade_ms,
                      blend_left=True, blend_right=True):
    """Blend inside source-side guide overlaps without shifting audio time."""
    a, c = a.clone(), c.clone()
    requested = round(rate * fade_ms / 1000)
    left_edge = round(before * rate / FPS)
    right_edge = round((before + frames) * rate / FPS)
    left_n = min(requested, a.shape[-1], left_edge) if blend_left else 0
    right_n = min(requested, c.shape[-1], max(0, generated.shape[-1]-right_edge)) if blend_right else 0
    if left_n:
        weight = torch.linspace(0., 1., left_n, dtype=a.dtype).reshape(1, 1, -1)
        a[..., -left_n:] = a[..., -left_n:]*(1-weight) + generated[..., left_edge-left_n:left_edge]*weight
    if right_n:
        weight = torch.linspace(1., 0., right_n, dtype=c.dtype).reshape(1, 1, -1)
        c[..., :right_n] = c[..., :right_n]*(1-weight) + generated[..., right_edge:right_edge+right_n]*weight
    return torch.cat((a, b, c), dim=-1), left_n, right_n


def check_video(frames, name, allow_empty=False):
    if (not isinstance(frames, torch.Tensor) or frames.ndim != 4
            or frames.shape[-1] != 3 or min(frames.shape[1:3]) < 1
            or (not allow_empty and len(frames) == 0)):
        raise ValueError(f"{name}需为RGB IMAGE帧批次，按24FPS提供")


def context_length(available, requested):
    n = min(available, requested)
    return 1 if n < 5 else n - (n - 5) % 17


def resize_frames(frames, height, width):
    if frames.shape[1:3] == (height, width):
        return frames
    return torch.nn.functional.interpolate(
        frames.permute(0, 3, 1, 2), size=(height, width),
        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)


def fit_frames(frames, height, width):
    """Fit a following clip to the source canvas without cropping or stretching."""
    if frames.shape[1:3] == (height, width):
        return frames
    h, w = frames.shape[1:3]
    scale = min(height / h, width / w)
    fit_h = max(1, min(height, round(h * scale)))
    fit_w = max(1, min(width, round(w * scale)))
    fitted = resize_frames(frames, fit_h, fit_w)
    top, left = (height-fit_h)//2, (width-fit_w)//2
    result = frames.new_zeros((len(frames), height, width, 3))
    result[:, top:top+fit_h, left:left+fit_w] = fitted
    return result


def audio_piece(audio, start, frames, rate):
    count = round((start + frames) * rate / FPS) - round(start * rate / FPS)
    if count == 0 or audio is None:
        return torch.zeros(1, 2, count)
    wave = audio['waveform'].detach().cpu()
    if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] not in (1, 2):
        raise ValueError("局部编辑音频仅支持batch=1、单声道或双声道")
    source_rate = int(audio['sample_rate'])
    if source_rate <= 0:
        raise ValueError("音频采样率必须大于0")
    if source_rate != rate:
        from torchaudio.functional import resample
        wave = resample(wave.float(), source_rate, rate)
    if wave.shape[1] == 1:
        wave = wave.repeat(1, 2, 1)
    begin = round(start * rate / FPS)
    piece = wave[..., begin:begin + count]
    return torch.nn.functional.pad(piece, (0, count - piece.shape[-1]))


class SolarLH3LocalEditPlan:
    CATEGORY = "solarL/MiniMax H3"
    FUNCTION = "build"
    RETURN_TYPES = ("SOLARL_H3_EDIT_PLAN", "STRING")
    RETURN_NAMES = ("edit_plan", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_video": ("IMAGE", {"tooltip": "原视频帧，必须按24FPS提供；extend/bridge时作为前段A。"}),
            "mode": (MODES, {"default": "replace_range", "tooltip": "replace_range等长替换；insert_at插入；extend向后续写；bridge有后段时双端补段，后段为空时自动单端续写。"}),
            "start_frame": ("INT", {"default": 24, "min": 0, "max": 1000000, "tooltip": "从0计数；替换起点或插入位置。续写和双端补段忽略。"}),
            "end_frame": ("INT", {"default": 72, "min": 0, "max": 1000000, "tooltip": "替换终点，不包含该帧。必须保留起点前的上下文。"}),
            "new_frames": ("INT", {"default": 48, "min": 1, "max": 10000, "tooltip": "插入、续写或双端补段新增帧数。替换模式自动使用end-start，保持总时长。"}),
            "context_frames": ([22, 5, 39, 56], {"tooltip": "每侧参考帧数。素材不足时缩短到H3有效长度，少于5帧使用1帧。"}),
            "audio_mode": (["preserve_source", "generated_patch"], {"tooltip": "替换模式默认保留原声；其他模式保留两侧原声，中间使用新生成音频。缺少的原声补静音。"}),
        }, "optional": {
            "following_video": ("IMAGE", {"tooltip": "可选后段C。bridge自动等比例适配到前段尺寸，比例不同居中补黑边；未接、None或有效零帧批次时自动单端续写；其他模式不使用。"}),
            "source_audio": ("AUDIO", {"tooltip": "可选前段原声，与原视频起点同步。未接、None或零采样音频均跳过声音参考；原片保留区无声。IMAGE无法自动获取原声。"}),
            "following_audio": ("AUDIO", {"tooltip": "可选后段原声，与following_video同步。未接、None或零采样音频均跳过后侧声音参考，不影响画面衔接；无后段视频时忽略。"}),
            # Appended widgets preserve the first version's serialized values.
            "audio_context": (["auto", "off"], {"default": "auto", "tooltip": "auto：将两侧同期原声与画面一起送入Guide，指导新声音接续。需要连接原声；off仅使用画面Guide。"}),
            "audio_fade_ms": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "默认0关闭淡化，原声直接接入生成片段。大于0才在上下文重叠区交叉淡化，可能改变接缝听感；不改变时长，不能修复对白或音乐语义。等长替换保留原声时不处理。"}),
            "context_lock": (["off", "head_experimental"], {"default": "off", "tooltip": "实验：锁定前侧VAE编码上下文，后侧仍使用Guide。"}),
        }}

    def build(self, source_video, mode, start_frame, end_frame, new_frames,
              context_frames, audio_mode, following_video=None, source_audio=None,
              following_audio=None, audio_context="auto", audio_fade_ms=0, context_lock="off"):
        check_video(source_video, "source_video")
        if mode not in MODES or audio_mode not in ("preserve_source", "generated_patch"):
            raise ValueError("未知局部编辑模式或音频模式")
        if audio_context not in ('auto', 'off') or not 0 <= audio_fade_ms <= 200:
            raise ValueError("音频上下文需为auto/off，淡化长度需为0至200毫秒")
        if context_lock not in ('off', 'head_experimental'):
            raise ValueError("未知上下文锁定模式")
        requested_mode = mode
        following_resize = ''
        if mode == 'bridge':
            if following_video is not None:
                check_video(following_video, 'following_video', allow_empty=True)
            if following_video is None or len(following_video) == 0:
                mode = 'extend'
        n = len(source_video)
        if int(context_frames) not in (5, 22, 39, 56):
            raise ValueError("上下文帧数需为5、22、39或56")
        if mode == "replace_range":
            if not 0 < start_frame < end_frame <= n:
                raise ValueError("替换范围需满足0<start_frame<end_frame<=原视频帧数；需保留前段上下文")
            left, right, count = source_video[:start_frame], source_video[end_frame:], end_frame-start_frame
            right_start = end_frame
        elif mode == "insert_at":
            if not 0 < start_frame <= n:
                raise ValueError("插入位置需满足0<start_frame<=原视频帧数")
            left, right, count = source_video[:start_frame], source_video[start_frame:], new_frames
            right_start = start_frame
        elif mode == "extend":
            left, right, count = source_video, source_video[:0], new_frames
            right_start = n
        else:
            if following_video.shape[1:] != source_video.shape[1:]:
                h, w = following_video.shape[1:3]
                target_h, target_w = source_video.shape[1:3]
                following_resize = (f"后段从{w}x{h}等比例适配到{target_w}x{target_h}，"
                                    "比例不同时居中补黑边。")
                following_video = fit_frames(following_video, target_h, target_w)
            left, right, count = source_video, following_video, new_frames
            right_start = 0
        if count < 1:
            raise ValueError("新增帧数必须大于0")
        source_audio = optional_audio(source_audio, 'source_audio')
        following_audio = optional_audio(following_audio, 'following_audio') if mode == 'bridge' else None
        plan = dict(version=1, mode=mode, requested_mode=requested_mode,
                    left=left, right=right, frames=int(count),
                    context_frames=int(context_frames), source_audio=source_audio,
                    right_audio=(following_audio if mode == 'bridge' else
                                 None if mode == 'extend' else source_audio),
                    right_start=right_start, audio_mode=audio_mode, fps=FPS,
                    audio_context=audio_context, audio_fade_ms=int(audio_fade_ms),
                    following_resize=following_resize, context_lock=context_lock)
        report = (f"{mode}: before={len(left)} patch={count} after={len(right)} "
                  f"output={len(left)+count+len(right)} fps={FPS}; " +
                  ("后段视频为空，bridge自动按extend单端续写。" if requested_mode != mode else '') +
                  following_resize + "仅补段参与新生成，前段原帧保留；后段按需适配尺寸。")
        return plan, report


class SolarLH3LocalEditSampler:
    CATEGORY = "solarL/MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "AUDIO", "FLOAT", "INT", "LATENT", "STRING")
    RETURN_NAMES = ("patch_images", "composed_images", "patch_audio", "composed_audio",
                    "fps", "frame_count", "window_latent", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "edit_plan": ("SOLARL_H3_EDIT_PLAN",),
            "model": ("MODEL",), "positive": ("CONDITIONING",), "latent": ("LATENT", {"tooltip": "官方H3 AV LATENT，仅用于读取生成尺寸。"}),
            "vae": ("VAE",), "audio_vae": ("VAE",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0., "max": 100.}),
        }, "optional": {"negative": ("CONDITIONING",)}}

    @torch.inference_mode()
    def run(self, edit_plan, model, positive, latent, vae, audio_vae, sampler,
            sigmas, seed, cfg, negative=None):
        from comfy_extras import nodes_minimax_h3 as h3
        plan = edit_plan
        if plan.get('version') != 1 or plan.get('fps') != FPS:
            raise ValueError("不兼容的局部编辑计划")
        streams = getattr(latent.get('samples'), 'tensors', None)
        if streams is None or len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[:2] != (1, 24):
            raise ValueError("请连接batch=1的官方H3 AV LATENT")
        if any(d.get('minimax_keyframes') for _, d in positive):
            raise ValueError("请连接未添加Guide的正向条件，双端Guide由本节点管理")
        if negative is not None and any(d.get('minimax_keyframes') for _, d in negative):
            raise ValueError("初版负向条件不能包含外部Guide")
        if any(r.get('kind') in ('video', 'video_audio') for _, d in (positive + (negative or [])) for r in d.get('minimax_refs', [])):
            raise ValueError("初版请使用文本／图片参考条件；已编码视频参考未提供局部时间映射")
        left, right, count = plan['left'], plan['right'], plan['frames']
        before = context_length(len(left), plan['context_frames'])
        after = context_length(len(right), plan['context_frames']) if len(right) else 0
        length = h3.align_frame_count(max(22, before + count + after))
        width, height = streams[0].shape[-1]*16, streams[0].shape[-2]*16
        current = h3._empty_av_latent(width, height, length)[0]
        use_audio = plan.get('audio_context', 'auto') == 'auto'
        head_audio = audio_guide(plan['source_audio'], len(left)-before, before) if use_audio else None
        tail_audio = audio_guide(plan['right_audio'], plan['right_start'], after) if use_audio and after else None
        cond = h3.MiniMaxH3AddGuide.execute(
            positive, current, 0, vae=vae, image=left[-before:],
            audio_vae=audio_vae, audio=head_audio)[0]
        locked = plan.get('context_lock', 'off') == 'head_experimental'
        if locked:
            import comfy.nested_tensor
            head = cond[0][1]['minimax_keyframes'][-1]
            video, audio = current['samples'].tensors
            video_mask = torch.ones((video.shape[0], 1, *video.shape[2:]), device=video.device)
            audio_mask = torch.ones((audio.shape[0], 1, *audio.shape[2:]), device=audio.device)
            encoded_video = head['latent'].to(video)
            vt = encoded_video.shape[2]
            video[:, :, :vt] = encoded_video
            video_mask[:, :, :vt] = 0
            if head.get('audio_latent') is not None:
                encoded_audio = head['audio_latent'].to(audio)
                at = min(encoded_audio.shape[-1], int(before * 40 / 24), audio.shape[-1])
                audio[..., :at] = encoded_audio[..., :at]
                audio_mask[..., :at] = 0
            current['noise_mask'] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        if after:
            cond = h3.MiniMaxH3AddGuide.execute(
                cond, current, before+count, vae=vae, image=right[:after],
                audio_vae=audio_vae, audio=tail_audio)[0]
        result = SolarLH3LongVideoSampler().run(
            model=model, positive=cond, latent=current, vae=vae, audio_vae=audio_vae,
            sampler_1=sampler, sigmas_1=sigmas, seed=seed, cfg_1=cfg,
            target_frames=length, segment_frames=0, overlap_frames=22,
            continuation_mode='official_image_guide', enable_second_pass=False,
            upscaler_model='', upscale_factor=1., cfg_2=1., highres_tiling=False,
            negative=negative, _guide_window_frames=length, _use_input_window=locked)
        patch = result[0][before:before+count].clone()
        if len(patch) != count:
            raise ValueError("局部生成输出帧数不足")
        # The plan already fits a mismatched following clip to the source canvas.
        composed = torch.cat((left.detach().cpu(), resize_frames(patch, *left.shape[1:3]), right.detach().cpu()), dim=0)
        sound = slice_audio(result[1], before, count)
        rate = int(sound['sample_rate'])
        a = audio_piece(plan['source_audio'], 0, len(left), rate)
        if plan['mode'] == 'replace_range' and plan['audio_mode'] == 'preserve_source':
            b = audio_piece(plan['source_audio'], len(left), count, rate)
        else:
            b = audio_piece(sound, 0, count, rate)
        c = audio_piece(plan['right_audio'], plan['right_start'], len(right), rate)
        preserve = plan['mode'] == 'replace_range' and plan['audio_mode'] == 'preserve_source'
        fade_ms = 0 if preserve else plan.get('audio_fade_ms', 0)
        full_sound = audio_piece(result[1], 0, before+count+after, rate)
        waveform, fade_left, fade_right = blend_audio_edges(
            a, b, c, full_sound, before, count, rate, fade_ms,
            blend_left=head_audio is not None, blend_right=tail_audio is not None)
        expected = round(len(composed)*rate/FPS)
        waveform = torch.nn.functional.pad(waveform[..., :expected], (0, max(0, expected-waveform.shape[-1])))
        window = dict(result[6], patch_start_frame=before, patch_frame_count=count,
                      note='完整生成窗口，含两侧Guide及补帧；不是裁剪后的patch latent')
        report = (f"experimental local edit: mode={plan['mode']} context_lock={plan.get('context_lock', 'off')} context={before}/{after} "
                  f"window={length} patch={count} composed={len(composed)} fps={FPS}; "
                  f"audio_guides={int(head_audio is not None)}/{int(tail_audio is not None)} "
                  f"audio_blend_samples={fade_left}/{fade_right}; "
                  "两侧不重新生成；patch_images为生成尺寸，composed_images中补段缩放到原片尺寸。" +
                  plan.get('following_resize', '') +
                  (" 未提供原声的保留区间使用静音。" if plan['source_audio'] is None or (len(right) and plan['right_audio'] is None) else '') +
                  (" 未获得前侧同期音频Guide，新声音没有前侧原声约束；请检查原声接线和覆盖长度。" if use_audio and head_audio is None else '') +
                  (" 未获得后侧同期音频Guide。" if use_audio and after and tail_audio is None else '') +
                  '\n' + result[5])
        return patch, composed, sound, {'waveform': waveform, 'sample_rate': rate}, float(FPS), len(composed), window, report
