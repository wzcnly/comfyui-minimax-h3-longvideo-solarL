import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=(await readFile(process.argv[2] ?? new URL('../web/longvideo_ui_v2.js',import.meta.url),'utf8')).replace("import { app } from '/scripts/app.js';",'const app={registerExtension(){}};');
const {reconcileReferences}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
// Socket-only fixture from node 6600; no media paths or prompt values.
const saved={inputs:[
  {
    "name": "model",
    "type": "MODEL",
    "link": 283
  },
  {
    "name": "positive",
    "type": "CONDITIONING",
    "link": 254
  },
  {
    "name": "latent",
    "type": "LATENT",
    "link": 255
  },
  {
    "name": "vae",
    "type": "VAE",
    "link": 300
  },
  {
    "name": "audio_vae",
    "type": "VAE",
    "link": 302
  },
  {
    "name": "sampler_1",
    "type": "SAMPLER",
    "link": 258
  },
  {
    "name": "sigmas_1",
    "type": "SIGMAS",
    "link": 319
  },
  {
    "name": "negative",
    "type": "CONDITIONING",
    "link": null
  },
  {
    "name": "model_2",
    "type": "MODEL",
    "link": null
  },
  {
    "name": "sampler_2",
    "type": "SAMPLER",
    "link": 321
  },
  {
    "name": "sigmas_2",
    "type": "SIGMAS",
    "link": 320
  },
  {
    "name": "clip",
    "type": "CLIP",
    "link": 298
  },
  {
    "name": "ref_image_0",
    "type": "IMAGE",
    "link": 348
  },
  {
    "name": "ref_image_1",
    "type": "IMAGE",
    "link": 350
  },
  {
    "name": "ref_image_2",
    "type": "IMAGE",
    "link": null
  },
  {
    "name": "ref_video_0",
    "type": "IMAGE",
    "link": 341
  },
  {
    "name": "ref_video_1",
    "type": "IMAGE",
    "link": null
  },
  {
    "name": "ref_video_audio_0",
    "type": "AUDIO",
    "link": 343
  },
  {
    "name": "ref_video_audio_1",
    "type": "AUDIO",
    "link": null
  },
  {
    "name": "ref_audio_0",
    "type": "AUDIO",
    "link": null
  },
  {
    "name": "seed",
    "type": "INT",
    "widget": {
      "name": "seed"
    },
    "link": null
  },
  {
    "name": "cfg_1",
    "type": "FLOAT",
    "widget": {
      "name": "cfg_1"
    },
    "link": null
  },
  {
    "name": "target_frames",
    "type": "INT",
    "widget": {
      "name": "target_frames"
    },
    "link": null
  },
  {
    "name": "segment_frames",
    "type": "INT",
    "widget": {
      "name": "segment_frames"
    },
    "link": null
  },
  {
    "name": "overlap_frames",
    "type": "COMBO",
    "widget": {
      "name": "overlap_frames"
    },
    "link": null
  },
  {
    "name": "continuation_mode",
    "type": "COMBO",
    "widget": {
      "name": "continuation_mode"
    },
    "link": null
  },
  {
    "name": "enable_second_pass",
    "type": "BOOLEAN",
    "widget": {
      "name": "enable_second_pass"
    },
    "link": null
  },
  {
    "name": "upscaler_model",
    "type": "COMBO",
    "widget": {
      "name": "upscaler_model"
    },
    "link": null
  },
  {
    "name": "upscale_factor",
    "type": "FLOAT",
    "widget": {
      "name": "upscale_factor"
    },
    "link": null
  },
  {
    "name": "cfg_2",
    "type": "FLOAT",
    "widget": {
      "name": "cfg_2"
    },
    "link": null
  },
  {
    "name": "highres_tiling",
    "type": "BOOLEAN",
    "widget": {
      "name": "highres_tiling"
    },
    "link": null
  },
  {
    "name": "prompt",
    "type": "STRING",
    "widget": {
      "name": "prompt"
    },
    "link": 263
  },
  {
    "name": "upscale_device",
    "type": "COMBO",
    "widget": {
      "name": "upscale_device"
    },
    "link": null
  },
  {
    "name": "upscale_precision",
    "type": "COMBO",
    "widget": {
      "name": "upscale_precision"
    },
    "link": null
  },
  {
    "name": "upscale_mode",
    "type": "COMBO",
    "widget": {
      "name": "upscale_mode"
    },
    "link": null
  },
  {
    "name": "upscale_width",
    "type": "INT",
    "widget": {
      "name": "upscale_width"
    },
    "link": null
  },
  {
    "name": "upscale_height",
    "type": "INT",
    "widget": {
      "name": "upscale_height"
    },
    "link": null
  },
  {
    "name": "upscale_megapixels",
    "type": "FLOAT",
    "widget": {
      "name": "upscale_megapixels"
    },
    "link": null
  },
  {
    "name": "tile_count",
    "type": "INT",
    "widget": {
      "name": "tile_count"
    },
    "link": null
  },
  {
    "name": "tile_overlap",
    "type": "INT",
    "widget": {
      "name": "tile_overlap"
    },
    "link": null
  }
]};
// Model the installed frontend: link is derived from topology at the current index.
class Node {
 constructor(data){
  this.links=new Map();this.inputs=data.map(d=>this.slot(d));
  data.forEach((d,i)=>{if(d.link!=null)this.links.set(d.link,{id:d.link,target_slot:i});});
 }
 slot(data){const {link,...view}=data;Object.defineProperty(view,'link',{enumerable:true,get:()=>this.getInputLink(this.inputs.indexOf(view))?.id??null,set(){throw Error('input.link must not be assigned');}});return view;}
 isInputConnected(index){return this.getInputLink(index)!=null;}
 getInputLink(index){return [...this.links.values()].find(l=>l.target_slot===index)??null;}
 addInput(name,type){const slot=this.slot({name,type});this.inputs.push(slot);return slot;}
 removeInput(index){assert.equal(this.isInputConnected(index),false);this.inputs.splice(index,1);for(const l of this.links.values())if(l.target_slot>index)l.target_slot--;}
 connect(name,id){const index=this.inputs.findIndex(i=>i.name===name);assert.ok(index>=0);assert.equal(this.isInputConnected(index),false);this.links.set(id,{id,target_slot:index});}
 disconnect(name){const index=this.inputs.findIndex(i=>i.name===name);const link=this.getInputLink(index);if(link)this.links.delete(link.id);}
 mapping(){return Object.fromEntries([...this.links.values()].map(l=>[l.id,this.inputs[l.target_slot]?.name]));}
 serialize(){return JSON.parse(JSON.stringify(this.inputs));}
 setDirtyCanvas(){}
}
for(const layout of [saved.inputs,[...saved.inputs.filter(i=>!i.widget),...saved.inputs.filter(i=>i.widget)]]){
 let node=new Node(layout);const expected=node.mapping();
 for(let round=0;round<50;round++){
  reconcileReferences(node);assert.deepEqual(node.mapping(),expected);
  const before=node.inputs.map(i=>i.name);reconcileReferences(node);assert.deepEqual(node.inputs.map(i=>i.name),before);
  node=new Node(node.serialize());assert.deepEqual(node.mapping(),expected);
 }
 // Grow every reference family to its limit, then shrink it, preserving existing cables.
 let id=10000;
 for(const [prefix,limit] of [['ref_image',9],['ref_video',3],['ref_video_audio',3],['ref_audio',3]]){
  const added=[];
  for(let i=0;i<limit;i++){
   const name=`${prefix}_${i}`;const index=node.inputs.findIndex(p=>p.name===name);assert.ok(index>=0);
   if(!node.isInputConnected(index)){node.connect(name,++id);added.push(name);}
   const mapping=node.mapping();reconcileReferences(node);assert.deepEqual(node.mapping(),mapping);
   node=new Node(node.serialize());assert.deepEqual(node.mapping(),mapping);
  }
  for(const name of added.reverse()){node.disconnect(name);const mapping=node.mapping();reconcileReferences(node);assert.deepEqual(node.mapping(),mapping);}
 }
 assert.deepEqual(node.mapping(),expected);
}
const legacy=new Node([{name:'model',type:'MODEL',link:1},{name:'source_video',type:'IMAGE',link:2},{name:'ref_video_0',type:'IMAGE',link:null},{name:'source_audio',type:'AUDIO',link:3}]);
reconcileReferences(legacy);assert.deepEqual(legacy.mapping(),{1:'model',2:'ref_video_0',3:'ref_video_audio_0'});
console.log('PASS: node 6600, two input layouts, 100 reload cycles, all autogrow limits, shrink, and legacy migration preserve links.');
