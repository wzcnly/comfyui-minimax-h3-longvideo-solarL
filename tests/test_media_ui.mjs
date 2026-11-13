import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=(await readFile(new URL('../web/media_loaders.js',import.meta.url),'utf8'))
    .replace("import { app } from '/scripts/app.js';",'const app=globalThis.solarLMediaApp={registerExtension(e){globalThis.solarLMediaExtension=e;}};const document={addEventListener(name,fn){(globalThis.solarLMediaListeners??={})[name]=fn;},removeEventListener(){}};');
const {installManualPlayback,updateLoaderWidgets,installVHSLoaderAlias,findVisibleInputByType}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const events={};
const video={paused:false,autoplay:true,playCount:0,pauseCount:0,
    pause(){this.paused=true;this.pauseCount++;events.pause?.();},removeAttribute(){},
    addEventListener(name,fn){events[name]=fn;},
    play(){this.playCount++;this.paused=false;events.play?.();return Promise.resolve();}};
const preview={name:'videopreview',videoEl:video,value:{paused:false},updateSource(){video.autoplay=!this.value.paused;}};
const node={type:'VHS_LoadVideo-solarL',widgets:[
    ...['custom_width','custom_height','max_long_edge','size_mode'].map(name=>({name,type:'number',value:name==='size_mode'?'custom':1})),preview],
    addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;},
    setDirtyCanvas(){}};
installManualPlayback(node);
for (const canvas of [
    {linkConnector:{isConnecting:true}},
    {connecting_links:[{}]},
    {pointer:{isDown:true}},
    {pointer:{eDown:{}}},
]) {
    globalThis.solarLMediaApp.canvas=canvas;
    for (const name of ['pointermove','pointerup','mouseup','click']) {
        let stopped=false;
        globalThis.solarLMediaListeners[name]({composedPath:()=>[video],stopPropagation(){stopped=true;}});
        assert.equal(stopped,false,`${name} must reach canvas during a link or drag gesture`);
    }
}
globalThis.solarLMediaApp.canvas={};
let nativeStopped=false;
globalThis.solarLMediaListeners.pointerdown({composedPath:()=>[video],stopPropagation(){nativeStopped=true;}});
assert.equal(nativeStopped,true,'direct native video controls still bypass VHS canvas handlers');
assert.equal(video.autoplay,false);assert.equal(video.paused,true);
assert.equal(video.controls,true);assert.equal(video.loop,true);
assert.equal(node.widgets.filter(w=>w.type==='button').length,0);
video.play();assert.equal(video.paused,false);assert.equal(preview.value.paused,false);
video.pause();assert.equal(preview.value.paused,true);
video.play();preview.updateSource();assert.equal(video.paused,true);assert.equal(video.autoplay,false);
installManualPlayback(node);assert.equal(node.widgets.filter(w=>w.type==='button').length,0);
updateLoaderWidgets(node);assert.equal(node.widgets.find(w=>w.name==='max_long_edge').hidden,true);
node.widgets.find(w=>w.name==='size_mode').value='longest_edge';updateLoaderWidgets(node);
assert.equal(node.widgets.find(w=>w.name==='custom_width').hidden,true);
assert.equal(node.widgets.find(w=>w.name==='max_long_edge').hidden,false);
assert.ok(node.widgets.filter(w=>w.hidden).every(w=>Number.isFinite(w.y)),
    'hidden widget slots must never inherit NaN from an undefined y');
const matchNode={widgets:node.widgets,inputs:[
    {name:'custom_width',type:'INT',link:null,widget:{name:'custom_width'}},
    {name:'force_rate',type:'FLOAT,INT',link:null,widget:{name:'force_rate'}},
    {name:'max_long_edge',type:'INT',link:null,widget:{name:'max_long_edge'}},
]};
assert.equal(findVisibleInputByType(matchNode,'INT').slot.name,'force_rate');
matchNode.inputs[1].link=1;
assert.equal(findVisibleInputByType(matchNode,'INT').slot.name,'max_long_edge');
matchNode.inputs[2].link=2;
assert.equal(findVisibleInputByType(matchNode,'INT').slot.name,'force_rate');
matchNode.inputs=matchNode.inputs.slice(0,1);
assert.equal(findVisibleInputByType(matchNode,'INT'),undefined);
let writes=0;
for (const widget of node.widgets.filter(w=>['custom_width','custom_height','max_long_edge'].includes(w.name))) {
    widget._state=new Proxy({}, {set(target,key,value){writes++;target[key]=value;return true;}});
}
updateLoaderWidgets(node);
assert.equal(writes,0,'unchanged visibility must not write reactive state');

let common=0,videoHooks=0;
const appArgument={};
const vhs={beforeRegisterNodeDef(type,data,app){
    assert.equal(app,appArgument);
    if (data.name.startsWith('VHS_')) common++;
    if (data.name==='VHS_LoadVideo') videoHooks++;
}};
installVHSLoaderAlias(vhs);
const installed=vhs.beforeRegisterNodeDef;
installVHSLoaderAlias(vhs);
assert.equal(vhs.beforeRegisterNodeDef,installed);
function Loader() {}
const data={name:'VHS_LoadVideo-solarL'};
await vhs.beforeRegisterNodeDef(Loader,data,appArgument);
await globalThis.solarLMediaExtension.beforeRegisterNodeDef(Loader,data,appArgument);
assert.equal(common,1,'VHS common hooks must be installed once');
assert.equal(videoHooks,1,'VHS video hooks must be installed once');
assert.equal(data.name,'VHS_LoadVideo-solarL');
let dirty=0;
const instance=new Loader();
instance.widgets=node.widgets;
instance.type=data.name;
instance.setDirtyCanvas=()=>dirty++;
instance.onWidgetChanged('force_rate',24);
await Promise.resolve();
assert.equal(dirty,0,'INT frame rate changes must not schedule visibility updates');
console.log('PASS: manual preview playback, source-change pause, native progress controls, size mode visibility');
console.log('PASS: single VHS registration, integer rate callback isolation, idempotent reactive visibility');
console.log('PASS: link/drag events over video reach canvas; native controls retain event isolation');
console.log('PASS: hidden input positions are finite and body snap skips hidden integer inputs');
