import { app } from '/scripts/app.js';

function visible(widget, show) {
    if (!widget) return;
    widget._solarLMediaOriginal ||= {type:widget.type,computeSize:widget.computeSize,draw:widget.draw};
    const saved=widget._solarLMediaOriginal;
    widget.hidden=!show;
    widget.options ||= {};
    widget.options.hidden=!show;
    widget.options.canvasOnly=!show;
    widget.type=show?saved.type:'converted-widget';
    widget.computeSize=show?saved.computeSize:()=>[0,-4];
    widget.draw=show?saved.draw:()=>{};
    if (widget._state) {widget._state.hidden=!show;widget._state.type=widget.type;}
}

export function updateLoaderWidgets(node) {
    const widget=name=>node.widgets?.find(w=>w.name===name);
    if (node.comfyClass==='VHS_LoadVideo-solarL' || node.type==='VHS_LoadVideo-solarL') {
        const longest=widget('size_mode')?.value==='longest_edge';
        for (const name of ['custom_width','custom_height']) visible(widget(name),!longest);
        visible(widget('max_long_edge'),longest);
    } else {
        visible(widget('目标比例'),!!widget('调整比例')?.value);
    }
    node.setDirtyCanvas?.(true,true);
}

export function installManualPlayback(node) {
    const preview=node.widgets?.find(w=>w.name==='videopreview');
    const video=preview?.videoEl;
    if (!video || preview._solarLManualPlayback) return;
    preview._solarLManualPlayback=true;
    preview.value.paused=true;
    video.autoplay=false;
    video.removeAttribute?.('autoplay');
    video.pause();
    video.controls=true;
    video.playsInline=true;
    video.loop=true;
    const source=preview.updateSource;
    preview.updateSource=function(...args) {
        this.value.paused=true;
        video.autoplay=false;
        video.pause();
        const result=source?.apply(this,args);
        video.autoplay=false;
        video.controls=true;
        return result;
    };
    preview.callback=preview.updateSource;
    video.addEventListener('play',()=>{preview.value.paused=false;});
    video.addEventListener('pause',()=>{preview.value.paused=true;});
    // VHS's canvas drag handlers preventDefault on preview pointer events.
    // Intercept only this video before they run, preserving native control actions.
    const pointer=event=>{
        if (event.composedPath().includes(video)) event.stopPropagation();
    };
    for (const name of ['pointerdown','pointerup','pointermove','mousedown','mouseup','click']) {
        document.addEventListener(name,pointer,true);
    }
    const removed=node.onRemoved;
    node.onRemoved=function(...args) {
        for (const name of ['pointerdown','pointerup','pointermove','mousedown','mouseup','click']) {
            document.removeEventListener(name,pointer,true);
        }
        video.pause();
        return removed?.apply(this,args);
    };

}

app.registerExtension({
    name:'solarL.H3.MediaLoaders',
    async beforeRegisterNodeDef(nodeType,nodeData) {
        const isVideo=nodeData.name==='VHS_LoadVideo-solarL';
        if (!isVideo && nodeData.name!=='SolarLReferenceImageLoad') return;
        if (isVideo) {
            // Reuse VHS's upload, preview, load-format and VAE output UI intact.
            await import('/extensions/ComfyUI-VideoHelperSuite/js/VHS.core.js');
            const vhs=app.extensions.find(e=>e.name==='VideoHelperSuite.Core');
            if (!vhs) throw new Error('solarL video loader: VHS frontend extension unavailable');
            await vhs.beforeRegisterNodeDef?.(nodeType,{...nodeData,name:'VHS_LoadVideo'});
        }
        for (const hook of ['onNodeCreated','onConfigure','onWidgetChanged']) {
            const previous=nodeType.prototype[hook];
            nodeType.prototype[hook]=function(...args) {
                const result=previous?.apply(this,args);
                queueMicrotask(()=>{
                    if (isVideo) installManualPlayback(this);
                    updateLoaderWidgets(this);
                    for (const name of ['size_mode','调整比例']) {
                        const w=this.widgets?.find(w=>w.name===name);
                        if (!w || w._solarLMediaCallback) continue;
                        w._solarLMediaCallback=true;
                        const callback=w.callback;
                        w.callback=(...values)=>{callback?.apply(w,values);updateLoaderWidgets(this);};
                    }
                });
                return result;
            };
        }
        const draw=nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground=function(...args) {
            const signature=this.widgets?.filter(w=>['size_mode','调整比例'].includes(w.name)).map(w=>w.value).join(':');
            if (this._solarLLoaderMode!==signature) {
                this._solarLLoaderMode=signature;
                updateLoaderWidgets(this);
            }
            return draw?.apply(this,args);
        };
    },
});
