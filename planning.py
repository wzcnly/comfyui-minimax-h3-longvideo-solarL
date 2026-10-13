import math


def segment_length(total, start, segment):
    """Round the remaining timeline up to a valid H3 run; cap at configured length."""
    needed = max(22, total - start)
    aligned = needed + (5 - needed) % 17
    return min(segment, aligned)


def plan_segments(total, segment, overlap):
    if total < 1:
        raise ValueError("目标帧数必须大于0")
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
