import { app } from '/scripts/app.js';

export function installVHSLoaderAlias(vhs) {
    if (vhs._solarLLoaderAlias) return;
    const register=vhs.beforeRegisterNodeDef;
    // Let ComfyUI invoke VHS once, including its common and video-specific hooks.
    vhs.beforeRegisterNodeDef=function(nodeType,nodeData,...args) {
        if (nodeData.name==='VHS_LoadVideo-solarL') nodeData={...nodeData,name:'VHS_LoadVideo'};
        return register?.call(this,nodeType,nodeData,...args);
    };
    vhs._solarLLoaderAlias=true;
}

function visible(widget, show) {
    if (!widget) return;
    if (widget._solarLMediaVisible===show) return;
    widget._solarLMediaVisible=show;
    widget._solarLMediaOriginal ||= {type:widget.type,computeSize:widget.computeSize,draw:widget.draw};
    const saved=widget._solarLMediaOriginal;
    // Hidden widgets are omitted from layout, but their input slots still use y.
    if (!Number.isFinite(widget.y)) widget.y=0;
    widget.hidden=!show;
    widget.options ||= {};
    widget.options.hidden=!show;
    widget.options.canvasOnly=!show;
    widget.type=show?saved.type:'converted-widget';
    widget.computeSize=show?saved.computeSize:()=>[0,-4];
    widget.draw=show?saved.draw:()=>{};
    if (widget._state) {widget._state.hidden=!show;widget._state.type=widget.type;}
}

export function findVisibleInputByType(node,type) {
    const types=value=>String(value??'*').toLowerCase().split(',');
    const wanted=types(type);
    let occupied,wildcardFree,wildcardOccupied;
    for (const [index,slot] of (node.inputs??[]).entries()) {
        const widget=node.widgets?.find(w=>w.name===slot.widget?.name);
        if (widget?.hidden) continue;
        const available=slot.link==null || !!node.graph?.getLink(slot.link)?._dragging;
        const offered=types(slot.type);
        const result={index,slot};
        if (wanted.some(t=>offered.includes(t))) {
            if (available) return result;
            occupied??=result;
        } else if (wanted.includes('*') || offered.includes('*')) {
            if (available) wildcardFree??=result;
            else wildcardOccupied??=result;
        }
    }
    return wildcardFree??occupied??wildcardOccupied;
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
        // A link dragged from another node can end over the video overlay.
        // Let the canvas finish that gesture before handling native video controls.
        const canvas=app.canvas;
        if (canvas?.linkConnector?.isConnecting || canvas?.connecting_links?.length ||
            canvas?.pointer?.isDown || canvas?.pointer?.eDown) return;
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
    async init() {
        await import('/extensions/ComfyUI-VideoHelperSuite/js/VHS.core.js');
        const vhs=app.extensions.find(e=>e.name==='VideoHelperSuite.Core');
        if (!vhs) throw new Error('solarL video loader: VHS frontend extension unavailable');
        installVHSLoaderAlias(vhs);
    },
    async beforeRegisterNodeDef(nodeType,nodeData) {
        const isVideo=nodeData.name==='VHS_LoadVideo-solarL';
        if (!isVideo && nodeData.name!=='SolarLReferenceImageLoad') return;
        if (isVideo) {
            // Canvas hover/drop must not snap to a widget excluded from layout.
            nodeType.prototype.findInputByType=function(type) {
                return findVisibleInputByType(this,type);
            };
        }
        for (const hook of ['onNodeCreated','onConfigure','onWidgetChanged']) {
            const previous=nodeType.prototype[hook];
            nodeType.prototype[hook]=function(...args) {
                const result=previous?.apply(this,args);
                if (hook==='onWidgetChanged' && !['size_mode','调整比例'].includes(args[0])) return result;
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
