"""Reference loaders; VHS integration uses its registered runtime without copying it."""
import copy
import math
import os
from collections import deque
from fractions import Fraction
import importlib
import torch

RATIOS=['1:1','4:3','3:4','3:2','2:3','16:9','9:16','21:9','9:21']

def long_edge_size(w,h,limit,upscale=False):
    if not limit: return w,h
    limit=max(32,int(limit)//32*32)
    scale=limit/max(w,h) if upscale or max(w,h)>limit else 1
    tw,th=max(32,round(w*scale/32)*32),max(32,round(h*scale/32)*32)
    if not upscale: tw,th=min(tw,max(32,w//32*32)),min(th,max(32,h//32*32))
    return min(tw,limit),min(th,limit)

def image_size(w,h,limit,upscale=False):
    if not limit:return w,h
    if 0<limit<=10:
        scale=min(1,math.sqrt(limit*1_000_000/(w*h)))
        return max(1,int(w*scale)),max(1,int(h*scale))
    return long_edge_size(w,h,limit,upscale)

def crop_box(w,h,aspect):
    if aspect not in RATIOS:raise ValueError('未知目标比例')
    r=Fraction(aspect.replace(':','/'));rw,rh=r.numerator,r.denominator
    n=min(w//rw,h//rh)
    cw,ch=(rw*n,rh*n) if n else (max(1,min(w,h*rw//rh)),max(1,min(h,w*rh//rw)))
    x,y=(w-cw)//2,(h-ch)//2
    return x,y,x+cw,y+ch

class SolarLReferenceImageLoad:
    CATEGORY='solarL/MiniMax H3'
    FUNCTION='load_and_scale'
    RETURN_TYPES=('IMAGE','MASK')
    RETURN_NAMES=('image','mask')
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        files=folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()),['image'])
        return {'required':{
            'image':(sorted(files),{'image_upload':True}),
            '最长边':('FLOAT',{'default':1280,'min':0,'max':8192,'step':32,'tooltip':'0保持原尺寸；0到10表示百万像素上限且不放大；大于10表示最长边像素，按32对齐。'}),
            '允许小图放大':('BOOLEAN',{'default':False,'tooltip':'开启允许按最长边放大小图；百万像素模式始终只缩小。'}),
            '调整比例':('BOOLEAN',{'default':False,'tooltip':'开启先按目标比例取最大居中裁剪，再限制尺寸。'}),
            '目标比例':(RATIOS,{'default':'16:9','tooltip':'居中裁剪比例，保留最大的原图面积。'})}}
    def load_and_scale(self,image,**kwargs):
        import folder_paths
        import numpy as np
        from PIL import Image,ImageOps
        import comfy.utils
        with Image.open(folder_paths.get_annotated_filepath(image)) as original:
            source=ImageOps.exif_transpose(original)
            if kwargs.get('调整比例',False):source=source.crop(crop_box(*source.size,kwargs.get('目标比例','16:9')))
            frames=torch.from_numpy(np.array(source.convert('RGB'),dtype=np.float32)/255).unsqueeze(0)
            mask=1-torch.from_numpy(np.array(source.getchannel('A'),dtype=np.float32)/255).unsqueeze(0) if 'A' in source.getbands() else torch.zeros(frames.shape[:3])
        h,w=frames.shape[1:3];tw,th=image_size(w,h,float(kwargs.get('最长边',1280)),kwargs.get('允许小图放大',False))
        if (tw,th)!=(w,h):
            frames=comfy.utils.common_upscale(frames.movedim(-1,1),tw,th,'area','disabled').movedim(1,-1)
            mask=torch.nn.functional.interpolate(mask.unsqueeze(1),size=(th,tw),mode='nearest-exact').squeeze(1)
        return frames,mask
    @classmethod
    def VALIDATE_INPUTS(cls,image,**kwargs):
        import folder_paths
        return True if folder_paths.exists_annotated_filepath(image) else '图片文件不存在'
    @classmethod
    def IS_CHANGED(cls,image,**kwargs):
        import folder_paths,hashlib
        with open(folder_paths.get_annotated_filepath(image),'rb') as stream:
            return hashlib.sha256(stream.read()).hexdigest()

def vhs_provider():
    import nodes
    cls=nodes.NODE_CLASS_MAPPINGS.get('VHS_LoadVideo')
    if cls is None:raise RuntimeError('视频载入节点需要已有VideoHelperSuite（VHS）；请安装或启用VHS后重启ComfyUI')
    return cls,importlib.import_module(cls.__module__)

def aligned_count(count,frames_rule=None):
    n=count-(count-5)%17
    if frames_rule:
        div,mod=frames_rule[:2]
        while n>=5 and n%div!=mod:n-=17
    if n<5:raise ValueError('裁尾后不足5帧，或所选VHS格式与17n+5没有可用帧长')
    return n

def aligned_stream(stream,frames_rule=None):
    # Retain only the possible trimmed tail, never buffer the complete video.
    hold=math.lcm(17,frames_rule[0] if frames_rule else 1)-1
    buffer=deque();count=emitted=0
    for frame in stream:
        count+=1;buffer.append(frame)
        if len(buffer)>hold:
            emitted+=1;yield buffer.popleft()
    keep=aligned_count(count,frames_rule)-emitted
    for _ in range(keep):yield buffer.popleft()

class SolarLLoadVideoUpload:
    CATEGORY='Video Helper Suite 🎥🅥🅗🅢'
    FUNCTION='load_video'
    RETURN_TYPES=('*','INT','AUDIO','VHS_VIDEOINFO')
    RETURN_NAMES=('IMAGE','frame_count','audio','video_info')
    @classmethod
    def INPUT_TYPES(cls):
        base,_=vhs_provider();cls.RETURN_TYPES=base.RETURN_TYPES
        schema=copy.deepcopy(base.INPUT_TYPES())
        schema['required'].update({
            'size_mode':(['custom','longest_edge'],{'default':'custom','tooltip':'custom使用VHS原有自定义宽高；longest_edge按最长边等比例缩小，尺寸按32对齐，不主动放大。'}),
            'max_long_edge':('INT',{'default':720,'min':0,'max':8192,'step':32,'tooltip':'最长边模式的像素上限；0保留原尺寸。自定义模式不使用此值。'}),
            'align_17n5':('BOOLEAN',{'default':False,'label_on':'17n＋5裁尾：开启','label_off':'17n＋5裁尾：关闭','tooltip':'开启只裁末尾，将最终帧长适配17n+5；不补帧。音频同步限制到裁尾后长度。Meta Batch按完整时间轴裁尾，各批次仍按VHS设置输出。H3参考应将force_rate设为24。'})})
        return schema
    @classmethod
    def IS_CHANGED(cls,video,**kwargs):return vhs_provider()[0].IS_CHANGED(video,**kwargs)
    @classmethod
    def VALIDATE_INPUTS(cls,video,**kwargs):return vhs_provider()[0].VALIDATE_INPUTS(video)
    def load_video(self,size_mode='custom',max_long_edge=720,align_17n5=False,**kwargs):
        base,module=vhs_provider()
        if size_mode not in ('custom','longest_edge'):raise ValueError('未知视频尺寸模式')
        if size_mode=='longest_edge' and max_long_edge:
            import cv2,folder_paths
            path=folder_paths.get_annotated_filepath(module.strip_path(kwargs['video']))
            cap=cv2.VideoCapture(path)
            try:
                if not cap.isOpened():raise ValueError('无法打开参考视频')
                w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not w or not h:
                    ok,frame=cap.read()
                    if not ok:raise ValueError('视频没有可读取的画面')
                    h,w=frame.shape[:2]
            finally:cap.release()
            kwargs['custom_width'],kwargs['custom_height']=long_edge_size(w,h,max_long_edge)
        elif size_mode=='longest_edge':kwargs.update(custom_width=0,custom_height=0)
        if align_17n5:
            rule=module.get_format(kwargs.get('format','None')).get('frames')
            def generator(**options):
                meta=options.get('meta_batch');uid=options.get('unique_id')
                # The wrapper owns EOF: VHS must not close the batch before buffered tail emits.
                options['meta_batch']=None
                stream=module.resized_cv_frame_gen(**options)
                info=list(next(stream))
                if info[6]:
                    available=max(0,math.ceil((int(info[4]/info[2]/info[5])-options.get('skip_first_frames',0))/options.get('select_every_nth',1)))
                    cap=options.get('frame_load_cap',0)
                    if cap:available=min(cap,available)
                    info[6]=aligned_count(available,rule)
                yield tuple(info)
                try:yield from aligned_stream(stream,rule)
                finally:
                    stream.close()
                    if meta is not None:
                        meta.inputs.pop(uid,None);meta.has_closed_inputs=True
            kwargs['generator']=generator
        result=list(base().load_video(**kwargs))
        if align_17n5:
            info=result[3];audio=result[2]
            if hasattr(audio,'file'):
                result[2]=module.lazy_get_audio(audio.file,audio.start_time,info['loaded_duration'])
            elif audio is not None:
                audio=dict(audio);audio['waveform']=audio['waveform'][...,:round(info['loaded_duration']*audio['sample_rate'])];result[2]=audio
        return tuple(result)
