from .sampler import SolarLH3LongVideoSampler
from .selflift_sampler import SolarLH3SelfLiftLongVideoSampler
from .media_loaders import SolarLReferenceImageLoad, SolarLLoadVideoUpload
from .local_edit import SolarLH3LocalEditPlan, SolarLH3LocalEditSampler
from .planning import SolarLH3SegmentPlan

NODE_CLASS_MAPPINGS = {"MiniMaxH3LongVideoSampler-solarL": SolarLH3LongVideoSampler,
    "MiniMaxH3SelfLiftLongVideoSampler-solarL": SolarLH3SelfLiftLongVideoSampler,
    "SolarLReferenceImageLoad": SolarLReferenceImageLoad,
    "VHS_LoadVideo-solarL": SolarLLoadVideoUpload,
    "MiniMaxH3LocalEditPlan-solarL": SolarLH3LocalEditPlan,
    "MiniMaxH3LocalEditSampler-solarL": SolarLH3LocalEditSampler,
    "MiniMaxH3SegmentPlan-solarL": SolarLH3SegmentPlan}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3LongVideoSampler-solarL": "MiniMax H3 长视频循环 Sampler · solarL",
    "MiniMaxH3SelfLiftLongVideoSampler-solarL": "MiniMax H3 长视频循环 Sampler · SelfLift · solarL",
    "SolarLReferenceImageLoad": "参考图像载入 - solarL",
    "VHS_LoadVideo-solarL": "Load Video (Upload) - solarL",
    "MiniMaxH3LocalEditPlan-solarL": "MiniMax H3 局部编辑计划 · solarL",
    "MiniMaxH3LocalEditSampler-solarL": "MiniMax H3 接续与补段 Sampler · 实验版 · solarL",
    "MiniMaxH3SegmentPlan-solarL": "MiniMax H3 分段计划 · solarL"}
WEB_DIRECTORY = "./web"

from .cache_directory_ui import register_routes
register_routes()
