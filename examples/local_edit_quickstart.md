# 局部接续与双端补段：实验初版

初版使用官方H3多帧Guide，在一个局部窗口中生成，再提取补段并保留两侧原视频。复用普通Sampler的一采实现；不运行SelfLift或二采。

提供[2080Ti接线示例](2080ti_局部接续与补段_实验初版.json)，沿用包内原有工作流的模型加载及加速节点，默认extend续写48帧。打开后先在视频加载节点选择原片，核对模型、CLIP、VAE及原有第三方节点是否安装，再调整提示词。示例已检查连线一致性，尚未完成浏览器加载和GPU生成验证。

## 节点与接线

新增两个节点，位于solarL/MiniMax H3：

1. **MiniMax H3 局部编辑计划 · solarL**：选择编辑区间、两侧素材及音频策略。
2. **MiniMax H3 接续与补段 Sampler · 实验版 · solarL**：接收计划和模型条件，生成补段并合成。

视频加载器按24FPS输出IMAGE，接计划节点的source_video。原音轨接source_audio；IMAGE本身不含音频。

计划节点的edit_plan接新Sampler的edit_plan。官方H3 Reference to Video的positive和AV LATENT接新Sampler的positive、latent；再连接model、vae、audio_vae、KSamplerSelect输出的sampler和完整sigmas。

positive可使用文本／图片参考，不要提前添加AddGuide。初版不接受已编码的视频参考，以免原视频参考与局部窗口时间错位。latent只确定生成分辨率，新窗口长度自动计算。

最终把composed_images、composed_audio、fps接视频合成／保存节点。无需再手动去除引导重叠。

## 四种模式

| mode | 操作 | 需要填写 |
| --- | --- | --- |
| replace_range | 等长替换原片[start_frame,end_frame)，保留总长度 | 起点、终点；new_frames忽略 |
| insert_at | 在start_frame前插入新片段，后续顺延 | 插入位置、new_frames |
| extend | 从原片末尾继续生成 | new_frames；起点和终点忽略 |
| bridge | 有following_video时双端过渡，未提供时自动单端续写 | new_frames；following_video可选；起点和终点忽略 |

帧号从0计数，终点不包含在替换范围内。replace_range和insert_at必须保留至少1帧前段；不能从0开始替换。可替换到视频末尾，此时没有后端引导。

bridge以source_video尺寸作为输出画布。后段尺寸不同时自动等比例缩放，宽高比不同时居中补黑边，不裁剪、不拉伸；报告会记录适配尺寸。following_video未接、省略参数、传None或有效零帧IMAGE批次均自动按extend续写，忽略following_audio。所有素材按24FPS提供；节点不会检测原文件帧率或自动变速。

应用可固定使用bridge模式，根据用户选择提供或省略后段。未选素材时，应省略相应输入及无用的上游加载分支，或由支持空值的上游节点输出None；将已连接的普通加载器文件名留空可能导致加载器先报错，本节点无法绕过上游执行错误。

例如100帧原片、start_frame=24、end_frame=72：替换48帧，保留前24帧和后28帧，输出仍为100帧。默认每侧取22帧作为运动上下文，生成窗口为107帧，最终只取中间48帧。

context_frames可选5、22、39、56。素材不足时缩短到H3有效长度，不满5帧只用1帧。极短补段在结构上允许，但不保证模型能在这么短的时间完成合理过渡。

## 输出

- patch_images：仅新生成的片段，尺寸来自上游LATENT。
- composed_images：前段原片＋新片段＋后段原片。新片段缩放到原片尺寸，前段原帧保留；bridge后段仅在尺寸不同时等比例缩放并按需补黑边，两侧不重新生成。生成时建议选择与原片相同宽高比，避免补段被拉伸。
- patch_audio：生成窗口裁出的新片段音频。
- composed_audio：根据计划合成的完整音轨。
- fps：24；frame_count：完整合成视频帧数。
- window_latent：包含Guide区和模型补帧的完整生成窗口，附patch_start_frame和patch_frame_count。**它不是裁剪后的补段latent，不要直接拿它当完整成片或最终续写状态。**
- report：窗口、上下文、裁剪长度与采样报告。

## 音频

replace_range默认preserve_source：保留整条原声。generated_patch则替换区间的音频也换为生成结果。

插入、续写和bridge模式保留两侧音频，中间使用生成音频。bridge后段原声接following_audio。没有连接原声的保留区间补静音，会在报告里提示；节点不能从IMAGE还原原视频音轨。

source_audio和following_audio各自独立可选：未接、省略参数、None或合法零采样AUDIO均视为没有该侧原声，跳过该侧音频Guide及淡化，保留对应画面Guide。可以只传前侧、只传后侧或两侧都不传；有后段视频而没有后段音频仍做双端画面过渡。新片段按当前音频策略处理；缺少原声的保留区间补静音。空音频需保持合法AUDIO结构（waveform为[1,1或2,0]，sample_rate为正数），错误结构仍会报错。

输出双声道，单声道自动复制。采样率不同时使用torchaudio重采样。

2026-11-02修复：默认audio_context=auto，把前段尾部同期原声和后段头部同期原声与画面一起送入官方Guide。旧版本只在输出时拼接原声，模型没有听到这些接续上下文。请确保source_audio和following_audio接线、起点及覆盖长度正确；音轨不足以覆盖整个上下文时跳过该侧音频Guide，并在报告提示，避免补静音误导生成。

audio_fade_ms默认0，关闭淡化，原声直接接入生成片段；音频Guide仍然生效。旧工作流可能保存了20，需要手动改为0。大于0才在生成窗口的Guide重叠区交叉淡化，不删样本或改变成片时长，但会调整接缝附近原声，可能造成不自然听感；replace_range+preserve_source始终保留整条原声，不做淡化。audio_context=off关闭音频引导，该模式也不执行依赖音频Guide的交叉淡化。

report中的audio_guides=1/1表示两侧都获得音频Guide；extend通常为1/0。缺少原声或音频长度不足时会提示。生成声音会参考上下文，但仍不保证歌声、节拍、同一说话人的音色或对白语义完全连续。固定背景音乐建议使用独立完整音乐轨覆盖成片；这不应交给逐段生成来恢复。

## 范围与验证

这是模型效果验证用初版：无自动切镜、项目队列、磁盘接续状态、原latent精确复用、SelfLift续写或二采增强。像素两侧参考会经VAE编码；Guide不能保证运动和身份一定自然衔接。

输入及合成输出为整批IMAGE，长视频会占较多内存，先用短片段验证。原视频文件不会被覆盖，输出另存；重新编码成MP4可能使保留区间产生编码差异。

先验证同人物、同机位、动作较接近的两端，再测试更大动作差异。重点检查入点、出点、运动方向、身份、亮度、帧数与音画同步。

代码回归与真实H3接口检查使用CPU假采样，不代表已经完成GPU画质验收。
