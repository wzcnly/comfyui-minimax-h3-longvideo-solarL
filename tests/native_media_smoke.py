"""Real VHS/OpenCV/PIL integration smoke; small synthetic files, no generation."""
import sys,types,importlib,importlib.util,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--comfy-root',required=True);args=parser.parse_args()
sys.path.insert(0,args.comfy_root)
import torch,nodes,cv2,numpy as np,server,folder_paths
server.PromptServer.instance=types.SimpleNamespace(prompt_queue=None)
from PIL import Image
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('media_smoke',root/'__init__.py',submodule_search_locations=[str(root)])
pkg=importlib.util.module_from_spec(spec);sys.modules['media_smoke']=pkg;spec.loader.exec_module(pkg)
from media_smoke import media_loaders as m
vhs_root=Path(args.comfy_root)/'custom_nodes'/'ComfyUI-VideoHelperSuite'
p=types.ModuleType('vhs_runtime_smoke');p.__path__=[str(vhs_root)];sys.modules[p.__name__]=p
module=importlib.import_module('vhs_runtime_smoke.videohelpersuite.load_video_nodes')
nodes.NODE_CLASS_MAPPINGS['VHS_LoadVideo']=module.LoadVideoUpload
schema=m.SolarLLoadVideoUpload.INPUT_TYPES()
assert set(module.LoadVideoUpload.INPUT_TYPES()['required']).issubset(schema['required'])
assert set(module.LoadVideoUpload.INPUT_TYPES()['optional']).issubset(schema['optional'])
assert m.SolarLLoadVideoUpload.RETURN_TYPES==module.LoadVideoUpload.RETURN_TYPES
outdir=root/'tests'/'media_smoke_output';outdir.mkdir(exist_ok=True)
folder_paths.set_input_directory(str(outdir))
video=outdir/'synthetic40.mp4'
writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),24,(160,96))
assert writer.isOpened()
for i in range(40):writer.write(np.full((96,160,3),i*5,dtype=np.uint8))
writer.release()
settings=dict(video=video.name,force_rate=24,custom_width=0,custom_height=0,frame_load_cap=0,
              skip_first_frames=0,select_every_nth=1,format='None',unique_id='1')
loader=m.SolarLLoadVideoUpload()
normal=loader.load_video(**settings);assert normal[1]==40 and normal[0].shape==(40,96,160,3)
aligned=loader.load_video(size_mode='longest_edge',max_long_edge=96,align_17n5=True,**settings)
assert aligned[1]==39 and aligned[0].shape==(39,64,96,3)
assert aligned[3]['loaded_frame_count']==39 and aligned[3]['loaded_duration']==39/24
assert abs(aligned[2].duration-39/24)<1e-6
selected=loader.load_video(align_17n5=True,**{**settings,'skip_first_frames':5,'select_every_nth':2})
assert selected[1]==5
class Batch:
    inputs={};total_frames=99999;frames_per_batch=10;has_closed_inputs=False
batch=Batch();parts=[]
for _ in range(6):
    result=loader.load_video(align_17n5=True,**{**settings,'meta_batch':batch})
    parts.append(result[1])
    if batch.has_closed_inputs:break
assert parts==[10,10,10,9],parts
class VAE:
    downscale_ratio=8
    latent_channels=4
    def __init__(self):self.encoded=[]
    def encode(self,frames):
        self.encoded.append(len(frames))
        return torch.zeros(len(frames),4,frames.shape[1]//8,frames.shape[2]//8)
vae=VAE()
encoded=loader.load_video(align_17n5=True,**{**settings,'vae':vae})
assert encoded[1]==39 and encoded[0]['samples'].shape[0]==39
assert sum(vae.encoded)==39  # trimming happens before VAE encoding
rgba=np.zeros((60,100,4),dtype=np.uint8);rgba[:,:,:3]=127;rgba[:,:,3]=128
image=outdir/'alpha.png';Image.fromarray(rgba).save(image)
images,mask=m.SolarLReferenceImageLoad().load_and_scale(image.name,**{'最长边':0,'调整比例':True,'目标比例':'1:1'})
assert images.shape==(1,60,60,3) and mask.shape==(1,60,60)
assert abs(mask[0,0,0].item()-(1-128/255))<1e-6
print('PASS: real VHS passthrough, long-edge resize, 17n+5, skip/select, lazy audio duration, meta-batch EOF, image alpha/crop')
