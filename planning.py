import math
import json


def parse_segment_schedule(schedule, overlap, default_prompt=''):
    """Validate explicit net frame counts on the H3 latent grid."""
    try:
        rows = json.loads(schedule) if isinstance(schedule, str) else schedule
    except (TypeError, ValueError) as exc:
        raise ValueError('分段计划必须是JSON数组') from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError('分段计划必须是非空数组')
    if overlap not in (5, 22, 39, 56):
        raise ValueError('重叠帧数需为5、22、39或56')
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or isinstance(row.get('frames'), bool) or not isinstance(row.get('frames'), int):
            raise ValueError(f'第{i+1}段的frames必须是整数')
        frames = row['frames']
        if frames < 1:
            raise ValueError(f'第{i+1}段帧数必须大于0')
        if i == 0 and len(rows) > 1 and frames < overlap:
            raise ValueError('第一段净帧数不能小于接续重叠帧数')
        # Each non-final boundary must land on the latent grid; arbitrary
        # final length can be padded and cropped without shifting later clips.
        generated = frames + (overlap if i else 0)
        if i < len(rows)-1 and (generated < 22 or generated % 17 != 5):
            raise ValueError(f'第{i+1}段边界不在H3潜变量网格：首段净帧数需为17k+5，后续段净帧数需为17k；末段可任意')
        text = row.get('prompt', '')
        if not isinstance(text, str):
            raise ValueError(f'第{i+1}段prompt必须为文本')
        result.append({'frames': frames, 'prompt': text.strip() or default_prompt})
    return result


class SolarLH3SegmentPlan:
    CATEGORY = 'solarL/MiniMax H3'
    FUNCTION = 'build'
    RETURN_TYPES = ('SOLARL_H3_SEGMENT_PLAN', 'STRING')
    RETURN_NAMES = ('segment_plan', 'report')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'segments_json': ('STRING', {'default': '[{"frames":124,"prompt":""},{"frames":102,"prompt":""}]', 'multiline': True}),
            'overlap_frames': ([22, 5, 39, 56],),
            'default_prompt': ('STRING', {'default': '', 'multiline': True}),
        }}

    def build(self, segments_json, overlap_frames, default_prompt):
        segments = parse_segment_schedule(segments_json, int(overlap_frames), default_prompt)
        plan = {'version': 1, 'segments': segments, 'overlap_frames': int(overlap_frames),
                'total_frames': sum(row['frames'] for row in segments), 'default_prompt': default_prompt}
        return plan, f"{len(segments)}段，净输出{plan['total_frames']}帧。"


def segment_length(total, start, segment):
    """Round the remaining timeline up to a valid H3 run; cap at configured length."""
    needed = max(22, total - start)
    aligned = needed + (5 - needed) % 17
    return aligned if segment == 0 else min(segment, aligned)


def plan_segments(total, segment, overlap):
    if total < 1:
        raise ValueError("目标帧数必须大于0")
    # Zero selects one full-timeline run; continuation settings are unused.
    if segment == 0:
        return [(0, 0)]
    if segment < 22 or segment % 17 != 5:
        raise ValueError("每段帧数需为17k+5且至少22，例如124、158、192")
    if overlap not in (5, 22, 39, 56) or overlap >= segment:
        raise ValueError("重叠需为5、22、39或56，且小于每段帧数")
    stride = segment - overlap
    count = 1 + max(0, math.ceil((total - segment) / stride))
    return [(i * stride, 0 if i == 0 else overlap) for i in range(count)]


def validate_sigmas(sigmas, refine=False):
    label = "sigmas_2（二采）" if refine else "sigmas_1（一采）"
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("Sigma必须是一维序列，包含至少一个采样步及结尾0")
    values = sigmas.tolist()
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Sigma不能包含负数或非有限值")
    if values[-1] != 0 or any(a <= b for a,b in zip(values,values[1:])):
        raise ValueError(
            f"{label}必须严格递减并以0结尾，实际序列={values}。"
            + ("二采请连接SplitSigmas的low_sigmas尾段。" if refine else
               "一采请连接调度器的完整序列；SplitSigmas的high_sigmas通常不以0结尾，不能作为本节点完整一采。"))
    if refine and not 0 < values[0] < 1:
        raise ValueError(
            f"sigmas_2起始值为{values[0]:.6g}，序列={values}。"
            "本节点二采用放大后潜变量做低噪声细化，要求0<起始Sigma<1。"
            "请将二采调度器的denoise调低，或用SplitSigmas的low_sigmas尾段接sigmas_2；"
            "不要接从1开始的完整一采序列。仅验证一采时请关闭二采放大。")
