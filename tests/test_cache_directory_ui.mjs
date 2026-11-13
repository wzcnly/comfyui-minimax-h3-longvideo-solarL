import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
let result = {path:'D:/cache/test',cancelled:false};
const messages=[];
globalThis.__cacheApp={registerExtension(){},ui:{dialog:{show(m){messages.push(m);}}}};
globalThis.__cacheApi={async fetchApi(url,request){
    assert.equal(request.method,'POST');
    return {ok:!result.error, async json(){return result;}};
}};
const source=(await readFile(new URL('../web/cache_directory.js',import.meta.url),'utf8'))
    .replace("import { app } from '../../scripts/app.js';",'const app=globalThis.__cacheApp;')
    .replace("import { api } from '../../scripts/api.js';",'const api=globalThis.__cacheApi;');
const {addCacheButtons}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const field={name:'cache_directory',value:''};
const node={widgets:[field],addWidget(type,name,value,callback,options){
    const w={type,name,value,callback,options};this.widgets.push(w);return w;
}};
addCacheButtons(node);addCacheButtons(node);
assert.equal(node.widgets.length,3);
assert.equal(node.widgets[1].serialize,false);
await node.widgets[1].callback();assert.equal(field.value,'D:/cache/test');
result={path:null,cancelled:true};
await node.widgets[1].callback();assert.equal(field.value,'D:/cache/test');
node.widgets[2].callback();assert.equal(field.value,'');
result={error:'remote connection'};
await node.widgets[1].callback();assert.equal(field.value,'');
assert.deepEqual(messages,['remote connection']);
console.log('PASS: folder selection, cancellation, clear, errors and nonserialized buttons');
