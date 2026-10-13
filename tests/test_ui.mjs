import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = (await readFile(new URL('../web/longvideo_ui_v2.js',import.meta.url),'utf8'))
    .replace("import { app } from '/scripts/app.js';", 'export const app = { registerExtension(e) { this.extension = e; } };');
const {reconcileReferences,updateModeWidgets,arrangeSecondPassWidgets,app} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const groups=[['ref_image',9],['ref_video',3],['ref_video_audio',3],['ref_audio',3]];
class Node {
    constructor() {
        this.inputs=['model','clip','source_video','source_audio','depth_video',
            ...groups.flatMap(([p,n])=>Array.from({length:n},(_,i)=>`${p}_${i}`))].map(name=>({name,link:null}));
        this.graph={_links:new Map()};
        this.widgets=['upscale_mode','upscale_factor','upscale_width','upscale_height','upscale_megapixels']
            .map(name=>({name,type:'number',value:name==='upscale_mode'?'scale by multiplier':1,computeSize:()=>[100,20]}));
    }
    addInput(name,type) { this.inputs.push({name,type,link:null}); this.onConnectionsChange?.(); }
    removeInput(index) { assert.equal(this.inputs[index].link,null); this.inputs.splice(index,1); this.onConnectionsChange?.(); }
    setDirtyCanvas() {}
    link(name,id) { const input=this.inputs.find(i=>i.name===name); assert.ok(input,name); input.link=id; this.graph._links.set(id,{target_slot:this.inputs.indexOf(input)}); }
}
let node=new Node();
reconcileReferences(node);
assert.deepEqual(node.inputs.map(i=>i.name),['model','clip',...groups.map(([p])=>`${p}_0`)]);
for (let i=0;i<9;i++) {
    node.link(`ref_image_${i}`,i+1); reconcileReferences(node);
    assert.equal(node.inputs.filter(p=>p.name.startsWith('ref_image_')).length,Math.min(9,i+2));
}
for (const input of node.inputs.filter(i=>i.link!=null)) assert.equal(node.graph._links.get(input.link).target_slot,node.inputs.indexOf(input));
node.inputs.find(i=>i.name==='ref_image_8').link=null; reconcileReferences(node);
assert.equal(node.inputs.filter(i=>i.name.startsWith('ref_image_')).length,9);
node.inputs.find(i=>i.name==='ref_image_7').link=null; reconcileReferences(node);
assert.equal(node.inputs.filter(i=>i.name.startsWith('ref_image_')).length,8);
node.inputs.find(i=>i.name==='ref_image_0').link=null; reconcileReferences(node);
assert.ok(node.inputs.find(i=>i.name==='ref_image_0')); // gaps preserve later connected IDs
const restored=new Node(); restored.inputs=JSON.parse(JSON.stringify(node.inputs)); restored.graph=node.graph;
reconcileReferences(restored); assert.deepEqual(restored.inputs,node.inputs);
node=new Node(); node.link('source_video',101); node.link('depth_video',102); node.link('source_audio',103);
reconcileReferences(node);
assert.ok(node.inputs.find(i=>i.name==='ref_video_0'&&i.link===101));
assert.ok(node.inputs.find(i=>i.name==='ref_video_1'&&i.link===102));
assert.ok(node.inputs.find(i=>i.name==='ref_video_audio_0'&&i.link===103));
assert.ok(!node.inputs.some(i=>['source_video','source_audio','depth_video'].includes(i.name)));
assert.ok(node.inputs.find(i=>i.name==='ref_video_2'));
for (const input of node.inputs.filter(i=>i.link!=null)) assert.equal(node.graph._links.get(input.link).target_slot,node.inputs.indexOf(input));
for (const [mode,visible] of [['scale by multiplier',['upscale_factor']],['target dimensions',['upscale_width','upscale_height']],['megapixels',['upscale_megapixels']]]) {
    node.widgets[0].value=mode; updateModeWidgets(node);
    for (const widget of node.widgets.slice(1)) assert.equal(widget.type==='converted-widget',!visible.includes(widget.name));
    assert.equal(node.widgets[1].value,1); // mode toggles retain serialized values
}
app.extension.beforeRegisterNodeDef(Node,{name:'MiniMaxH3LongVideoSampler-solarL'});
node=new Node(); node.onNodeCreated(); await Promise.resolve();
assert.equal(node.inputs.length,6);
node.widgets[0].value='target dimensions'; node.widgets[0].callback(); await Promise.resolve();
assert.equal(node.widgets[1].hidden,true);
assert.equal(node.widgets[1].computedHeight,0);
assert.equal(node.widgets[2].type,'number');
node.widgets[0].value='megapixels'; node.onConfigure(); await Promise.resolve();
assert.equal(node.widgets[2].hidden,true);
assert.equal(node.widgets[4].type,'number');
node.widgets[0].value='scale by multiplier'; node.onWidgetChanged('upscale_mode'); await Promise.resolve();
assert.equal(node.widgets[1].type,'number');
assert.equal(node.widgets[4].hidden,true);
node.link('ref_video_0',200); node.onConnectionsChange(); await Promise.resolve();
assert.ok(node.inputs.find(i=>i.name==='ref_video_1'));
console.log('UI checks passed: autogrow limits, disconnects, reload, legacy links, target slots, mode widgets, lifecycle hooks.');
const stubNode = new Node();
const widthWidget = stubNode.widgets.find(w=>w.name==='upscale_width');
widthWidget._state = {type:'number',hidden:false};
widthWidget.options = {};
stubNode.inputs.push({name:'upscale_width',widget:{name:'upscale_width'},link:null});
updateModeWidgets(stubNode);
assert.equal(widthWidget.hidden,true);
assert.equal(widthWidget._state.hidden,true);
assert.equal(widthWidget.options.hidden,true);
stubNode.widgets[0].value='target dimensions'; updateModeWidgets(stubNode);
assert.equal(widthWidget.hidden,false);
assert.equal(widthWidget.type,'number');
assert.equal(widthWidget._state.hidden,false);
assert.equal(widthWidget.options.hidden,false);
class LayoutNode {
    constructor() {
        this.inputs=[];
        this.widgets=['seed','cfg_1','enable_second_pass','upscaler_model','upscale_factor',
            'cfg_2','highres_tiling','prompt','upscale_device','upscale_precision','upscale_mode',
            'upscale_width','upscale_height','upscale_megapixels','tile_count','tile_overlap']
            .map((name,index)=>({name,value:index,type:'number'}));
    }
    configure(info) { this.widgets.forEach((w,i)=>w.value=info.widgets_values[i]); }
    setDirtyCanvas() {}
}
app.extension.beforeRegisterNodeDef(LayoutNode,{name:'MiniMaxH3LongVideoSampler-solarL'});
const layoutNode=new LayoutNode();
const canonical=layoutNode.widgets.map(w=>w.name);
arrangeSecondPassWidgets(layoutNode);
assert.deepEqual(layoutNode.widgets.slice(0,4).map(w=>w.name),['seed','cfg_1','prompt','enable_second_pass']);
assert.equal(layoutNode.widgets.findIndex(w=>w.name==='upscale_factor'),
    layoutNode.widgets.findIndex(w=>w.name==='upscale_mode')+1);
const info={widgets_values:layoutNode.widgets.map(w=>w.value)};
layoutNode.onSerialize(info);
assert.deepEqual(info.widgets_values,canonical.map((_,i)=>i));
layoutNode.configure({widgets_values:canonical.map((_,i)=>i+100)});
for (const w of layoutNode.widgets) assert.equal(w.value,canonical.indexOf(w.name)+100);
