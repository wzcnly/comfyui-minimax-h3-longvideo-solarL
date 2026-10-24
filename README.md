# comfyui-minimax-h3-longvideo-solarL

MiniMax H3 长视频内部循环采样、整片增强及参考素材载入节点包。

## 更新记录

- **2026-10-24：修复新版前端下工作流重开后参考接线错位。** 在 ComfyUI 0.42.5／前端 1.55.18 中，参考端口重排后再读取 `input.link` 会按新位置取得其他端口的连线。普通版和 SelfLift 版共用的前端脚本现已取消直接重排 `inputs` 和手动改写 `_links`，改用 `isInputConnected()`、`addInput()`、`removeInput()` 维护连接；保留参考口自动增长，新增端口追加到末尾。使用实际工作流的 6600 节点完成两种端口布局、共 100 轮模拟保存加载及参考口增删、上限、旧接口迁移检查，均通过；尚未完成浏览器界面验证。更新后按 Ctrl+F5 刷新页面并重新载入工作流。

## 安装

将整个文件夹复制到 `ComfyUI/custom_nodes`，重启 ComfyUI 并刷新页面。

需要支持 MiniMax H3 的 ComfyUI、匹配的模型、视频 VAE 和音频 VAE。视频上传载入节点需要已有 VideoHelperSuite（VHS）。采样、接续和放大运行代码已内置，不需要额外安装其他采样插件。

放大模型放在 `ComfyUI/models/latent_upscale_models`，支持子目录和已注册的额外模型目录。模型权重不包含在安装包中；关闭增强时无需加载放大模型。

## 长视频循环 Sampler

节点：**MiniMax H3 长视频循环 Sampler · solarL**。

处理顺序：**全部分段一采 → 去除重叠并拼接完整 latent → 可选整片放大和二采 → 输出**。二采保留一采音频。

### 接线

- 前置 H3 Reference to Video 的 `positive`、`LATENT` 分别接 `positive`、`latent`。
- 接入 `model`、`vae`、`audio_vae`、`sampler_1` 和完整的 `sigmas_1`。
- 动作视频接 `ref_video_0`，配对音轨接 `ref_video_audio_0`，同时接同一个 `clip`、`prompt` 及原始参考图片。
- 其他视频参考从 `ref_video_1` 继续接入，深度视频也作为普通视频参考使用。
- `target_frames=0` 使用 Video 1 的实际帧数；没有视频时使用输入 latent 的对齐帧数。生成尺寸来自输入 latent。

参考接口自动增长，最多支持 9 张图片、3 路视频、3 路配对音轨和3路独立音频。各路视频应按同一24 FPS时间轴提供，覆盖目标长度。图片、视频和独立音频接口从0开始连续连接。

输入 latent 用于确定生成尺寸，不能代替原始动作参考。节点根据当前分段重建参考条件；仅裁剪已有编码条件不能保证动作时间对齐。输入条件不应提前包含局部首尾 Guide。

### 分段与接续

默认每段124帧、重叠22帧，每轮新增102帧。段长必须为17n＋5；重叠支持5、22、39、56帧且须小于段长。

末段按剩余时间向上对齐，参考视频重复尾帧、配套音频补静音，最终输出裁回实际长度。例如194帧目标使用124＋107帧采样，去重后输出194帧。

- `official_image_guide`：使用上一段一采结果的尾部画面接续。
- `motion_context`：使用上一段一采 latent 的视频和音频上下文接续。

二采结果不参与后续一采接续。每段种子为seed＋段序号，整片增强种子为seed＋1000000。当前支持24 FPS、batch_size=1。

### 整片二采

开启 `enable_second_pass`，接入独立 `sampler_2`、`sigmas_2`，可选 `model_2`。二采 Sigma 必须严格递减、起始值大于0且小于1，并以0结束。

增强参数集中在提示词下方。放大模式支持倍数、指定宽高和百万像素，界面只显示对应数值，切换时保留已填值。`upscale_device` 和 `upscale_precision` 仅控制放大器，默认cuda＋fp16，不改变H3采样后端。

高分辨率分块用于降低整片增强的临时显存需求：`tile_count=0`自动选择1至8块，手动支持1至64块；`tile_overlap`是每块每侧的扩展像素，默认128，相邻块共享约256像素。分块数量受空间网格限制，更多块通常增加计算量，不能保证任意长度不爆显存。

## SelfLift分段低分辨率→整段高分辨率 Sampler

节点：**MiniMax H3 长视频循环 Sampler · SelfLift · solarL**。

处理顺序为：**同一条Sigma序列的低分辨率前段按片段循环 → 拼接各段边界状态 → 整段执行一次SelfLift升分辨率 → 用Sigma剩余段对完整视频采样 → 输出**。低分辨率阶段按片段逐段运行；高分辨率阶段对拼接后的完整视频运行一次。音频沿同一条Sigma序列完成对应阶段采样。

节点只接一个 `sampler` 和一条完整 `sigmas`，采样器必须为标准Euler。不要在节点外拆分Sigma或再串联一个普通Sampler。总采样步数等于输入Sigma序列的步数：`transition_step` 指定低分辨率阶段使用的前几步，剩余步骤用于完整视频的高分辨率阶段，不会额外增加一轮采样。

- `transition_step`：每段低分辨率阶段的步数；必须小于Sigma总步数，并至少留一步给高分辨率阶段。示例工作流使用8步Sigma，前6步分段，剩余2步整段采样。

节点默认值：`transition_step=6`、`lowres_scale=0.5`、`rho=0`、`w_min=0.5`、`w_max=1.0`、`highres_tiling=关闭`、`upscaler_unload=开启`。示例工作流开启了高分辨率分块。

- `lowres_scale=0.5`：低分辨率阶段相对输入LATENT宽高的比例；升分辨率后恢复到输入LATENT尺寸。
- `upscaler_model`：SelfLift潜变量放大模型；默认使用 `minimax_h3_latent_upscaler_3d_fp16.safetensors`。模型需放在 `models/latent_upscale_models` 或ComfyUI已注册的额外模型目录。选择 `none` 时必须设置 `rho>0`。
- `rho=0`：使用放大模型时跳过像素VAE锚点修正；此时 `w_min` 和 `w_max` 不参与计算。设置 `rho>0` 时才启用锚点修正，并由 `w_min`、`w_max` 控制修正权重范围。
- `model_hires`：可选的整段高分辨率阶段模型；未连接时沿用 `model`。
- `highres_tiling`：开启后在高分辨率阶段使用SelfLift自动空间分块以降低显存需求；它不切分时间轴。
- `upscaler_unload`：放大完成后卸载放大器，默认开启以释放显存。

SelfLift版没有独立的输出放大模式或倍率。最终宽高由输入LATENT决定；若需要更大输出尺寸，请在上游生成对应尺寸的H3 LATENT。

## 参考图像载入

节点：**参考图像载入 - solarL**。

支持上传／选择图片、EXIF方向、透明遮罩、最长边限制、小图放大开关和目标比例最大居中裁剪，输出IMAGE和MASK。

`最长边=0`保留尺寸；大于0且不超过10表示百万像素上限，只缩小；大于10表示最长边像素，按32对齐。开启调整比例时先裁剪，再限制尺寸。

## 视频上传载入

节点：**Load Video (Upload) - solarL**。

保留上传／选择、帧率、自定义宽高、加载帧数上限、跳帧、抽帧、格式、Meta Batch、VAE，以及IMAGE／LATENT、frame_count、audio、video_info四个输出。

- `size_mode=custom`：使用自定义宽高。
- `size_mode=longest_edge`：按最长边等比例缩小、按32对齐，不主动放大；最长边0保留源尺寸。
- `align_17n5`：默认关闭；开启只裁尾帧到17n＋5，不补帧，少于5帧报错，音频同步限制到输出时长。所选格式有其他帧长规则时同时满足两者。

裁尾发生在VAE编码前；Meta Batch按完整时间轴裁尾，各批次仍按设置输出。H3参考视频建议将force_rate设为24。

预览默认暂停，使用视频自带的播放／暂停、进度条和时间控件。手动播放后循环，直到暂停；更新预览或更换源视频后恢复暂停。

## 输出和测试

普通Sampler和SelfLift版均输出images、audio、fps、frame_count、last_first_pass_latent、report、output、denoised_output。普通版的`last_first_pass_latent`是最后一段一采；SelfLift版是最后一段低分辨率阶段的边界x0。`output`为整片latent，`denoised_output`为最后一步x0。

整片latent可能包含H3网格补齐尾帧，附带frame_count和fps；外部解码后需按frame_count裁剪。本节点images和audio已裁到实际长度。

支持长视频分段生成，实际可处理长度取决于内存、显存、分辨率、参考素材和增强设置。当前完整输出使用内存拼接，参考帧、输出帧及拼接过程均会占用CPU内存；开启整片增强时还需处理完整latent时间轴，空间分块仅降低空间方向的临时显存需求。当前未提供断点恢复或流式文件输出。空间分块不保证与不分块完全等价。

样例位于 `examples/`，仅保留两个流程：

- `2080ti_内部循环_终版1.json`：标准版终版流程。
- `2080ti_内部循环_终版1_SelfLift.json`：单Euler、单8步Sigma；默认前6步按段低分辨率采样、拼接后用剩余2步对完整视频做高分辨率采样，并开启SelfLift空间分块。

新版已修正SelfLift工作流的控件顺序恢复，并兼容旧版提示词位于参数前的工作流。升级插件后如仍看到放大器空白、参数错位或NaN，请在ComfyUI页面按Ctrl+F5刷新前端，再重新载入工作流；不需要手动重排参数。

先验证一采接续，再打开增强和分块。逻辑、界面模拟和小视频载入检查不代替完整GPU画质与显存测试。

## 许可与MIT来源告示

本项目由solarL编写的代码和文档采用[MIT许可](LICENSE)，版权归属为Copyright (c) 2026 solarL。第三方代码遵循其各自许可，不因本项目采用MIT而变更；具体范围见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

内置部分代码依据MIT许可使用，原版权声明及完整许可文本随包保留：

- [ComfyUI-Apt_Preset](https://github.com/cardenluo/ComfyUI-Apt_Preset)：[MIT许可](licenses/apt_motion.py.LICENSE)。
- [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：[MIT许可](licenses/upscale_3d.py.LICENSE)。

其他来源记录见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。上述MIT告示仅适用于对应代码，不涵盖模型权重或其他来源。
