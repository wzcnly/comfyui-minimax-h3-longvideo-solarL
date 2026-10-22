"""Long-video SelfLift: segmented low-resolution prefix, one full-video lift and suffix."""
import logging
import time

import torch

from .planning import plan_segments, segment_length
from .sampler import (
    SolarLH3LongVideoSampler, ReferenceVAECache, decode_video, join_av_latents,
    motion_class, reference_inputs, resize_guides, sample_once, slice_audio,
    slice_images,
)

LOG = logging.getLogger(__name__)


def selflift_class():
    from .vendor.selflift_runtime.nodes import SelfLiftH3Sampler
    return SelfLiftH3Sampler


def _join_boundary(video_parts, audio_parts, template_audio, total, h3):
    """Join clean low-resolution video x0 and noisy audio state at sigma_next."""
    return join_av_latents(video_parts, audio_parts, template_audio, total, h3)


def _zero_negative(cond):
    neg = [[torch.zeros_like(e), d.copy()] for e, d in cond]
    for _, metadata in neg:
        if isinstance(metadata.get("pooled_output"), torch.Tensor):
            metadata["pooled_output"] = torch.zeros_like(metadata["pooled_output"])
    return neg


class SolarLH3SelfLiftLongVideoSampler(SolarLH3LongVideoSampler):
    """One official SelfLift sigma schedule split across a long-video pipeline."""
    CATEGORY = "solarL/MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "LATENT", "STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "last_first_pass_latent", "report", "output", "denoised_output")

    @classmethod
    def INPUT_TYPES(cls):
        schema = super(SolarLH3SelfLiftLongVideoSampler, cls).INPUT_TYPES()
        required, optional = schema["required"], schema["optional"]
        for old in ("sampler_1", "sigmas_1", "enable_second_pass", "cfg_1", "cfg_2"):
            required.pop(old, None)
        required["sampler"] = ("SAMPLER", {"tooltip": "SelfLift 全阶段使用同一个采样器；必须选标准 Euler。"})
        required["sigmas"] = ("SIGMAS", {"tooltip": "官方 SelfLift 的完整Sigma序列。transition_step之前按段低分辨率采样，拼接与升分辨率后使用剩余Sigma整段采样。"})
        required["cfg"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "tooltip": "低分辨率分段和高分辨率整段共用同一CFG，符合官方SelfLift采样器。"})
        cfg_definition = required.pop("cfg")
        ordered_required = {}
        for name, definition in required.items():
            ordered_required[name] = definition
            if name == "seed":
                ordered_required["cfg"] = cfg_definition
        schema["required"] = ordered_required
        required = schema["required"]
        for old in ("sampler_2", "sigmas_2", "model_2"):
            optional.pop(old, None)
        # Official SelfLift targets the input latent's dimensions.  The standard
        # sampler's separate output-upscale controls do not belong on this node.
        for group in schema.values():
            for name in ("upscale_mode", "upscale_factor", "upscale_width",
                         "upscale_height", "upscale_megapixels"):
                group.pop(name, None)
        upstream = selflift_class().INPUT_TYPES()
        # Keep the actual SelfLift controls in ComfyUI's required-widget group,
        # matching the upstream node.  Optional widget serialization is unstable
        # when optional STRING inputs such as prompt are linked in saved graphs.
        for name in ("transition_step", "lowres_scale", "rho", "w_min", "w_max",
                     "upscaler_model", "highres_tiling", "upscaler_unload"):
            required.pop(name, None)
            optional.pop(name, None)
        optional["model_hires"] = ("MODEL", {"tooltip": "可选高分辨率阶段模型；未连接时沿用model。必须与低分辨率模型兼容。"})
        explanations = {
            "transition_step": "低分辨率阶段的采样步数。节点先对每段运行Sigma前段，拼接并完成SelfLift升分辨率后，再用剩余Sigma对完整视频采样。",
            "lowres_scale": "低分辨率阶段相对输入LATENT宽高的比例；官方默认0.5。SelfLift后段恢复到输入LATENT尺寸。",
            "rho": "SelfLift像素VAE锚点修正比例。0配合放大模型跳过像素VAE往返；模型选none时必须大于0。",
            "w_min": "像素锚点修正强度下限，不能超过w_max。仅rho大于0时参与修正。",
            "w_max": "像素锚点修正强度上限；范围0到1，必须不小于w_min。",
            "upscaler_model": "SelfLift升分辨率时使用的原生潜变量放大模型。none使用SelfLift-zero路线，需rho大于0。",
            "highres_tiling": "高分辨率后段使用SelfLift空间分块；整条时间轴仍共同采样，分块数由SelfLift自动规划。",
            "upscaler_unload": "升分辨率后卸载放大器，为整段高分辨率采样释放显存。",
        }
        for name, tooltip in explanations.items():
            definition = upstream["required"].get(name, upstream["optional"].get(name))
            options = dict(definition[1]) if len(definition) > 1 else {}
            options["tooltip"] = tooltip
            if name == "transition_step":
                options.update(default=6, min=1)
            elif name == "lowres_scale":
                options.update(default=0.5, min=0.25, max=1.0)
            if name in ("transition_step", "lowres_scale", "rho", "w_min", "w_max", "upscaler_model"):
                required[name] = (definition[0], options)
            else:
                optional[name] = (definition[0], options)
        # The official node places the H3 SelfLift widgets after the long-video
        # controls and before optional reference/prompt widgets.
        order = ["model", "positive", "latent", "vae", "audio_vae", "sampler", "sigmas",
                 "seed", "cfg", "target_frames", "segment_frames", "overlap_frames",
                 "continuation_mode", "transition_step", "lowres_scale", "rho", "w_min",
                 "w_max", "upscaler_model"]
        required = schema["required"] = {name: required[name] for name in order if name in required}
        required["sampler"][1]["tooltip"] = "低分辨率分段和高分辨率整段共用；SelfLift要求标准Euler。"
        required["sigmas"][1]["tooltip"] = "唯一Sigma输入。transition_step切分低分辨率前段与高分辨率剩余段；Sigma必须严格递减、以0结尾，且切换Sigma小于1。"
        if "sampler_1" in optional:
            optional.pop("sampler_1", None)
        if "sigmas_1" in optional:
            optional.pop("sigmas_1", None)
        if "upscale_device" in optional:
            optional.pop("upscale_device", None)
        if "upscale_precision" in optional:
            optional.pop("upscale_precision", None)
        if "tile_count" in optional:
            optional.pop("tile_count", None)
        if "tile_overlap" in optional:
            optional.pop("tile_overlap", None)
        for group in schema.values():
            for name, definition in list(group.items()):
                if name in ("sampler", "sigmas"):
                    continue
                base = dict(definition[1]) if len(definition) > 1 else {}
                base["tooltip"] = base.get("tooltip") or name
                group[name] = (definition[0], base)
        return schema

    @torch.inference_mode()
    def run(self, model, positive, latent, vae, audio_vae, sampler, sigmas, cfg,
            seed, target_frames, segment_frames, overlap_frames,
            continuation_mode, upscaler_model="none",
            highres_tiling=False, transition_step=6, lowres_scale=0.5,
            rho=0.0, w_min=0.5, w_max=1.0, upscaler_unload=True,
            model_hires=None, negative=None,
            source_video=None, source_audio=None, depth_video=None, clip=None, prompt="",
            ref_image_0=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            **extra_references):
        from comfy_extras import nodes_minimax_h3 as h3
        from comfy_extras.nodes_audio import vae_decode_audio
        import comfy.model_management as mm
        import comfy.nested_tensor
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import latent_preview
        from .vendor.selflift_runtime import nodes as sl
        from .vendor.selflift_runtime import selflift as sl_math
        from .vendor.selflift_runtime import h3_tiling, h3_upscaler

        if continuation_mode not in ("official_image_guide", "motion_context"):
            raise ValueError("未知接续模式")
        sl._validate_schedule(sigmas, int(transition_step))
        sl._validate_sampling(model.get_model_object("model_sampling"), sampler)
        if rho == 0 and upscaler_model == "none":
            raise ValueError("SelfLift需选择放大模型；使用none时请将rho设为大于0，例如0.6")

        streams = getattr(latent.get("samples"), "tensors", None)
        if streams is None or len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24:
            raise ValueError("请连接官方H3 Reference to Video的AV LATENT")
        if streams[0].shape[0] != 1:
            raise ValueError("当前版本仅支持batch_size=1")

        pictures, videos, video_audios, ref_audios = reference_inputs(
            [ref_image_0, ref_image_1, ref_image_2, ref_image_3], source_video, source_audio,
            depth_video, extra_references)
        source_video = videos.get("ref_video_0")
        base_h, base_w = streams[0].shape[-2:]
        width, height = base_w * 16, base_h * 16
        total = int(target_frames or (len(source_video) if source_video is not None else
                    sum(h3.FRAME_PER_TOKEN[k % 5] for k in range(streams[0].shape[2]))))
        if source_video is not None and total > len(source_video):
            raise ValueError("target_frames超过Video 1的实际帧数")
        for name, frames in videos.items():
            if total > len(frames):
                raise ValueError(f"{name}帧数不足以覆盖目标时间轴")
        plan = plan_segments(total, int(segment_frames), int(overlap_frames))
        refs = positive[0][1].get("minimax_refs", [])
        if any(data.get("minimax_keyframes") for _, data in positive):
            raise ValueError("请连接干净的Reference to Video条件；外部首尾Guide不能用于分段低分辨率采样")
        rebuild = source_video is not None or bool(ref_audios) or (bool(pictures) and clip is not None and bool(prompt.strip()))
        if len(plan) > 1 and any(ref.get("kind") in ("video", "video_audio", "audio") for ref in refs) and not rebuild:
            raise ValueError("多段参考视频需额外连接原始参考视频、CLIP、prompt和参考图片，以便按段重建条件")
        if rebuild:
            if clip is None or not prompt.strip():
                raise ValueError("原始参考模式需连接CLIP和非空prompt")
            if len(pictures) < sum(ref.get("kind") == "image" for ref in refs):
                raise ValueError("请把官方节点使用的原始参考图片也接入ref_image接口")

        target_h, target_w = base_h, base_w
        scale = float(lowres_scale)
        if not 0.25 <= scale <= 1.0:
            raise ValueError("低分辨率比例必须在0.25至1")
        low_h = max(2, round(target_h * scale / 2) * 2)
        low_w = max(2, round(target_w * scale / 2) * 2)

        motion = motion_class() if continuation_mode == "motion_context" and len(plan) > 1 else None
        reference_vae = ReferenceVAECache(vae)
        video_parts, audio_parts, reports = [], [], []
        previous_clean, previous_tail = None, None
        started = time.perf_counter()
        LOG.info("[solarL SelfLift] schedule=%s transition_step=%d low=%dx%d target=%dx%d segments=%d",
                 sigmas.tolist(), transition_step, low_w, low_h, target_w, target_h, len(plan))

        for index, (start, trim) in enumerate(plan):
            mm.throw_exception_if_processing_interrupted()
            tick = time.perf_counter()
            length = segment_length(total, start, int(segment_frames))
            current, _ = h3._empty_av_latent(low_w * 16, low_h * 16, length)
            cond = positive
            if rebuild:
                seg_videos = {name: slice_images(frames, start, length) for name, frames in videos.items()}
                seg_audio = {name: slice_audio(sound, start, length) for name, sound in video_audios.items()}
                cond, current = h3.MiniMaxH3ReferenceToVideo.execute(
                    clip=clip, prompt=prompt, width=low_w * 16, height=low_h * 16, length=length,
                    ref_image_size="max", vae=reference_vae, audio_vae=audio_vae,
                    ref_images=pictures, ref_videos=seg_videos,
                    ref_video_audios=seg_audio, ref_audios=ref_audios)[:2]
            else:
                cond = sl._resize_keyframes(positive, low_h, low_w)
            if previous_clean is not None:
                if motion is not None:
                    cond, actual_trim = motion().apply(cond, current, trim_frames=int(overlap_frames),
                                                       context_latent=previous_clean)
                    if actual_trim != trim:
                        raise ValueError("Motion Context重叠帧数与分段计划不一致")
                else:
                    cond = h3.MiniMaxH3AddGuide.execute(cond, current, 0, vae=vae,
                                                       image=previous_tail)[0]
            neg = negative if negative is not None else _zero_negative(cond)
            low = sample_once(model, cond, neg, current, sampler,
                              sigmas[:int(transition_step) + 1], (int(seed) + index) % 2**64, cfg)
            clean = low.get("_denoised_samples")
            if clean is None:
                raise RuntimeError("SelfLift低分辨率阶段未返回边界x0")
            clean_streams = getattr(clean, "tensors", None)
            if clean_streams is None or len(clean_streams) != 2:
                raise RuntimeError("SelfLift低分辨率阶段的去噪结果不是H3视频/音频NestedTensor")
            low_video = clean_streams[0]
            boundary_audio = low["samples"].tensors[1]
            trim_tokens = h3.video_latent_t(trim) if trim else 0
            video_parts.append(low_video[:, :, trim_tokens:].cpu().clone())
            take = min(length - trim, total - (start + trim))
            audio_start = round((start + trim) * 40 / 24)
            audio_end = round((start + trim + take) * 40 / 24)
            audio_parts.append((audio_start, audio_end,
                                boundary_audio[..., round(trim * 40 / 24):].cpu().clone()))
            # Motion Context consumes a ComfyUI LATENT dictionary, while
            # sample_once exposes its denoised AV value as a bare NestedTensor.
            previous_clean = low.copy()
            previous_clean["samples"] = clean
            previous_clean.pop("_denoised_samples", None)
            if motion is None and index + 1 < len(plan):
                decoded = decode_video(vae, low_video)
                previous_tail = decoded[-int(overlap_frames):].clone()
            reports.append(f"low_segment {index + 1}/{len(plan)} start={start} generated={length} trim={trim} "
                           f"output={take} elapsed={time.perf_counter() - tick:.2f}s")
            LOG.info("[solarL SelfLift] %s", reports[-1])

        boundary = _join_boundary(video_parts, audio_parts, streams[1], total, h3)
        low_x0, audio_state = boundary["samples"].tensors
        full_length = h3.align_frame_count(max(22, total))
        if rebuild:
            full_cond = h3.MiniMaxH3ReferenceToVideo.execute(
                clip=clip, prompt=prompt, width=width, height=height, length=full_length,
                ref_image_size="max", vae=reference_vae, audio_vae=audio_vae,
                ref_images=pictures,
                ref_videos={name: slice_images(frames, 0, full_length) for name, frames in videos.items()},
                ref_video_audios={name: slice_audio(sound, 0, full_length) for name, sound in video_audios.items()},
                ref_audios=ref_audios)[0]
        else:
            full_cond = positive
        full_neg = negative if negative is not None else _zero_negative(full_cond)
        high_cond = resize_guides(full_cond, target_h, target_w)
        high_neg = resize_guides(full_neg, target_h, target_w)

        transition_started = time.perf_counter()
        need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
        z_lat_vae = None
        if upscaler_model != "none" and need_lat:
            z_lat_vae = h3_upscaler.learned_latent_lift(
                low_x0, (target_h, target_w), upscaler_model, force_unload=upscaler_unload)
        need_pix = rho > 0.0 and w_max > 0.0
        z_pix_vae = sl_math._pixel_anchor_video(low_x0, vae, (target_h, target_w)) if need_pix else None
        if z_lat_vae is None and need_lat:
            z_lat_vae = torch.nn.functional.interpolate(
                low_x0.float(), size=(low_x0.shape[2], target_h, target_w),
                mode="trilinear", align_corners=False).to(low_x0.dtype)
        if z_lat_vae is not None and z_pix_vae is not None:
            z_lat_vae = z_lat_vae.to(z_pix_vae.device)
        latent_format = model.get_model_object("latent_format")
        z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
        z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
        z0_high = sl_math.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max)
        sigma_k = sigmas[int(transition_step) - 1]
        sigma_next = sigmas[int(transition_step)]
        video_noise = comfy.sample.prepare_noise(z0_high, (int(seed) + 1) % (1 << 64)).to(z0_high)
        model_sampling = model.get_model_object("model_sampling")
        video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
        video_next = sl._euler_step(video_state, z0_high, sigma_k, sigma_next)
        audio_next = model_sampling.inverse_noise_scaling(sigma_next, audio_state)
        video_next = model_sampling.inverse_noise_scaling(sigma_next, video_next)
        resume_streams = [video_next, audio_next]
        resume_latent_samples = model.model.process_latent_out(comfy.nested_tensor.NestedTensor(resume_streams))
        resume_noise = comfy.nested_tensor.NestedTensor([torch.zeros_like(stream) for stream in resume_streams])
        del low_x0, audio_state, z_lat_vae, z_pix_vae, z_lat, z_pix, z0_high, video_noise, video_state, video_next, audio_next

        hires_base = model_hires if model_hires is not None else model
        high_model = h3_tiling.tiled_model(hires_base, [tuple(resume_latent_samples.tensors[0].shape),
                                                       tuple(resume_latent_samples.tensors[1].shape)]) if highres_tiling else hires_base
        total_steps = len(sigmas) - 1
        high_steps = total_steps - int(transition_step)
        if high_steps < 1:
            raise ValueError("transition_step之后至少需要一个高分辨率采样步")
        x0_capture = {}
        preview_callback = latent_preview.prepare_callback(model, total_steps)
        high_evaluations = 0
        def callback(step, x0, x, count):
            nonlocal high_evaluations
            mm.throw_exception_if_processing_interrupted()
            high_evaluations += 1
            x0_capture["value"] = x0
            return preview_callback(step + int(transition_step), x0, x, total_steps)
        LOG.info("[solarL SelfLift] global high-resolution suffix starts sigma=%.6g steps=%d",
                 float(sigma_next), high_steps)
        high_started = time.perf_counter()
        output_samples = comfy.samplers.sample(
            high_model, resume_noise, high_cond, high_neg, cfg, high_model.load_device,
            sampler, sigmas[int(transition_step):], high_model.model_options,
            latent_image=resume_latent_samples, callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed))
        if high_evaluations != high_steps:
            raise RuntimeError(f"SelfLift高分辨率阶段预计{high_steps}步，实际回调{high_evaluations}步；请检查采样器包装")
        x0 = x0_capture.get("value")
        if x0 is None:
            denoised_samples = output_samples
        else:
            if output_samples.is_nested and not x0.is_nested:
                x0 = comfy.utils.unpack_latents(x0, [z.shape for z in output_samples.unbind()])
                x0 = comfy.nested_tensor.NestedTensor(x0)
            denoised_samples = high_model.model.process_latent_out(x0.cpu())
        output = {"samples": output_samples.to(device=mm.intermediate_device(), dtype=mm.intermediate_dtype()),
                  "frame_count": total, "fps": 24.0}
        denoised_output = {"samples": denoised_samples, "frame_count": total, "fps": 24.0}
        final_video = output_samples.tensors[0]
        images = decode_video(vae, final_video)[:total].clone()
        audio = vae_decode_audio(audio_vae, {"samples": output_samples, "frame_count": total, "fps": 24.0})
        reports.append(f"global_transition elapsed={time.perf_counter() - transition_started:.2f}s; "
                       f"global_highres_steps={high_steps} elapsed={time.perf_counter() - high_started:.2f}s")
        reports.append(f"SelfLift staged schedule: lowres_steps={transition_step} per segment; "
                       f"joined once; highres_steps={high_steps} for full timeline; "
                       f"total_elapsed={time.perf_counter() - started:.2f}s")
        return (images, audio, 24.0, total, previous_clean, "\n".join(reports), output, denoised_output)
