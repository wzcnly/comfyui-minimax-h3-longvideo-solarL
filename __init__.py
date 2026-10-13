from .sampler import SolarLH3LongVideoSampler
from .selflift_sampler import SolarLH3SelfLiftLongVideoSampler
from .media_loaders import SolarLReferenceImageLoad, SolarLLoadVideoUpload

NODE_CLASS_MAPPINGS = {"MiniMaxH3LongVideoSampler-solarL": SolarLH3LongVideoSampler,
    "MiniMaxH3SelfLiftLongVideoSampler-solarL": SolarLH3SelfLiftLongVideoSampler,
    "SolarLReferenceImageLoad": SolarLReferenceImageLoad,
    "VHS_LoadVideo-solarL": SolarLLoadVideoUpload}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3LongVideoSampler-solarL": "MiniMax H3 长视频循环 Sampler · solarL",
    "MiniMaxH3SelfLiftLongVideoSampler-solarL": "MiniMax H3 长视频循环 Sampler · SelfLift · solarL",
    "SolarLReferenceImageLoad": "参考图像载入 - solarL",
    "VHS_LoadVideo-solarL": "Load Video (Upload) - solarL"}
WEB_DIRECTORY = "./web"
