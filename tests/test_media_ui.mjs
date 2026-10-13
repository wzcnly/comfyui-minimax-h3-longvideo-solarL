import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=(await readFile(new URL('../web/media_loaders.js',import.meta.url),'utf8'))
    .replace("import { app } from '/scripts/app.js';",'const app={registerExtension(){}};const document={addEventListener(){},removeEventListener(){}};');
const {installManualPlayback,updateLoaderWidgets}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
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
console.log('PASS: manual preview playback, source-change pause, native progress controls, size mode visibility');
