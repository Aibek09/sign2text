const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function fixture({installed=true, runtimeReady=true, brokenRecorder=false, liveResults=[]}={}) {
  class Element {
    constructor() { this.hidden=false; this.disabled=false; this.style={}; this.textContent=''; this.lastChild={}; this.events={}; this.files=[]; this.children={}; this.classList={add(){},remove(){},toggle(){}}; this.size=0; }
    addEventListener(name, cb){ (this.events[name]??=[]).push(cb); }
    async fire(name){ for (const cb of this.events[name]??[]) await cb({}); }
    querySelector(name){ return this.children[name]??=new Element(); }
    removeAttribute(){} setAttribute(name,value){this[name]=value;} load(){} async play(){} click(){}
    replaceChildren(){this.childNodes=[];} append(...children){(this.childNodes??=[]).push(...children);}
    getContext(){ return {drawImage(){},getImageData(){return {data:new Uint8Array(160*90*4)}}}; }
    toBlob(callback){ callback(new Blob(['jpeg'],{type:'image/jpeg'})); }
  }
  const elements = new Map();
  const get = id => { if(!elements.has(id)) elements.set(id,new Element()); return elements.get(id); };
  let stopped=0, time=0;
  let nextInterval=0;
  const intervals=new Map();
  const stream={getTracks:()=>[{stop:()=>stopped++}]};
  const recordings=[];
  class Recorder {
    static isTypeSupported(){ return true; }
    constructor(){ if(brokenRecorder) throw Error('unsupported'); this.state='inactive'; this.mimeType='video/webm'; recordings.push(this); }
    start(){ this.state='recording'; }
    stop(){ this.state='inactive'; this.ondataavailable({data:new Blob(['video'],{type:'video/webm'})}); return this.onstop(); }
  }
  const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[],createElement:()=>new Element()},
    navigator:{mediaDevices:{getUserMedia:async()=>stream},clipboard:{writeText:async()=>{throw Error('denied')}}},
    Blob, URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}}, MediaRecorder:Recorder,
    performance:{now:()=>time},Date:{now:()=>time},console,
    window:{MediaRecorder:Recorder,sign2textModel:{status:async()=>({installed,runtimeReady,aslLettersInstalled:true,aslLettersRuntimeReady:true}),translate:async()=>({text:'test',confidence:null,latencyMs:100}),live:async()=>liveResults.shift()??{gesture:null,text:null,confidence:null}},setTimeout:()=>1,clearTimeout(){},setInterval:(fn)=>{const id=++nextInterval;intervals.set(id,fn);return id;},clearInterval:(id)=>intervals.delete(id),addEventListener(){}}
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../app/app.js'),'utf8'),context);
  return {get, recordings, context, stopped:()=>stopped, advance:ms=>time+=ms,
    tick:async()=>{ time+=300; for(const fn of [...intervals.values()]) fn(); await flush(); }};
}
const flush = ()=>new Promise(resolve=>setImmediate(resolve));

test('missing weights disable translation while retaining save',async()=>{
  const f=fixture({installed:false}); await flush();
  f.get('fileInput').files=[Object.assign(new Blob(['video'],{type:'video/mp4'}),{name:'test.mp4'})];
  await f.get('fileInput').fire('change'); f.get('video').duration=3; await f.get('video').fire('loadedmetadata');
  assert.equal(f.get('translateButton').disabled,true);
  assert.equal(f.get('saveButton').disabled,false);
});
test('recording errors restore upload controls',async()=>{
  const f=fixture({brokenRecorder:true}); await flush();
  await f.get('cameraButton').fire('click'); await f.get('recordButton').fire('click');
  assert.equal(f.get('uploadButton').disabled,false);
});
test('completed recordings stop the camera and become reviewable',async()=>{
  const f=fixture(); await flush();
  await f.get('cameraButton').fire('click'); await f.get('recordButton').fire('click');
  f.advance(3000); await f.get('recordButton').fire('click');
  assert.equal(f.stopped(),1);
  assert.equal(f.get('video').controls,true);
  assert.equal(f.get('video').src,'blob:test');
  assert.equal(f.get('translateButton').disabled,false);
  assert.equal(f.get('saveButton').disabled,false);
});
test('a new clip clears stale output and rejects long clips',async()=>{
  const f=fixture(); await flush();
  f.get('translationText').textContent='old result';
  f.get('fileInput').files=[new Blob(['video'],{type:'video/mp4'})];
  await f.get('fileInput').fire('change');
  assert.equal(f.get('translationText').textContent,'');
  f.get('video').duration=30; await f.get('video').fire('loadedmetadata');
  assert.equal(f.get('translateButton').disabled,true);
  assert.match(f.get('clipStatus').textContent,/1–10/);
});
test('copy errors produce feedback, not an unhandled rejection',async()=>{
  const f=fixture(); await flush(); await f.get('copyButton').fire('click');
  assert.match(f.get('toast').textContent,/Copy unavailable/);
});
test('camera displays a stable live gesture without recording',async()=>{
  const f=fixture({liveResults:[
    {gesture:'Thumb_Up',text:'Yes',confidence:.8},
    {gesture:null,text:null,confidence:null},
    {gesture:'Thumb_Up',text:'Yes',confidence:.9},
    {gesture:'Thumb_Up',text:'Yes',confidence:.9},
  ]});
  await flush();
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click');
  await flush();
  assert.equal(f.get('resultOutput').hidden,true);
  await f.tick();
  assert.equal(f.get('resultOutput').hidden,true);
  await f.tick();
  assert.equal(f.get('resultOutput').hidden,true);
  await f.tick();
  assert.equal(f.get('translationText').textContent,'Yes');
  assert.equal(f.get('resultOutput').hidden,false);
  await f.get('cameraButton').fire('click');
  assert.equal(f.get('resultOutput').hidden,true);
});
test('all gestures view collects changes but not a continuously held hand',async()=>{
  const yes={gesture:'Thumb_Up',text:'Yes',confidence:.8};
  const no={gesture:'Thumb_Down',text:'No',confidence:.8};
  const blank={gesture:null,text:null,confidence:null,handPresent:false};
  const f=fixture({liveResults:[yes,yes,yes,yes,no,no,no,blank,blank,blank,blank,no,no,no]});
  await flush();
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click');
  await flush();
  await f.get('historyTab').fire('click');
  assert.equal(f.get('historyPanel').hidden,false);
  for(let i=0;i<13;i++) await f.tick();
  assert.equal(f.get('historyText').textContent,'Yes No No');
  await f.get('clearHistoryButton').fire('click');
  assert.equal(f.get('historyText').textContent,'');
  assert.equal(f.get('historyEmpty').hidden,false);
});
test('an uncertain but still visible hand does not duplicate its word',async()=>{
  const yes={gesture:'Thumb_Up',text:'Yes',confidence:.8,handPresent:true};
  const uncertain={gesture:null,text:null,confidence:null,handPresent:true};
  const f=fixture({liveResults:[yes,yes,yes,uncertain,uncertain,uncertain,uncertain,yes,yes,yes]});
  await flush();
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click');
  await flush();
  for(let i=0;i<9;i++) await f.tick();
  assert.equal(f.get('historyText').textContent,'Yes');
});
test('two simultaneous gestures appear together and form one history event',async()=>{
  const pair={gesture:'Open_Palm|Thumb_Up',text:'Stop + Yes',confidence:null,handPresent:true};
  const f=fixture({liveResults:[pair,pair,pair,pair]});
  await flush();
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click');
  await flush();
  for(let i=0;i<3;i++) await f.tick();
  assert.equal(f.get('translationText').textContent,'Stop + Yes');
  assert.equal(f.get('historyText').textContent,'Stop + Yes');
});
test('switching recognition modes changes guide and keeps word histories separate',async()=>{
  const f=fixture({liveResults:[
    {gesture:'Thumb_Up',text:'Yes',confidence:.8},
    {gesture:'Thumb_Up',text:'Yes',confidence:.8},
    {gesture:'Thumb_Up',text:'Yes',confidence:.8},
    {gesture:'ASL_3',text:'Three',confidence:null},
    {gesture:'ASL_3',text:'Three',confidence:null},
    {gesture:'ASL_3',text:'Three',confidence:null},
    {gesture:'ASL_3',text:'Three',confidence:null},
  ]});
  await flush();
  assert.equal(f.get('gestureGuide').childNodes.length,26);
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click'); await flush();
  await f.tick(); await f.tick();
  assert.equal(f.get('historyText').textContent,'Yes');
  await f.get('aslMode').fire('click');
  assert.equal(f.get('gestureGuide').childNodes.length,9);
  assert.equal(f.get('historyText').textContent,'');
  await f.tick(); await f.tick(); await f.tick();
  assert.equal(f.get('historyText').textContent,'Three');
  await f.get('gesturesMode').fire('click');
  assert.equal(f.get('historyText').textContent,'Yes');
});
test('new gesture cards show distinct finger patterns and matching counts',async()=>{
  const f=fixture(); await flush();
  const cards=f.get('gestureGuide').childNodes.slice(16);
  assert.equal(cards.length,10);
  const expected=['10100','10010','10110','10101','10011','01010','01101','01011','00111','00101'];
  cards.forEach((card,index)=>{
    const [icon,key,label,detail]=card.childNodes;
    assert.match(icon.innerHTML,/<svg/);
    assert.equal(key.childNodes.map(finger=>finger.className==='raised'?'1':'0').join(''),expected[index]);
    const count=[...expected[index]].filter(finger=>finger==='1').length;
    assert.equal(detail.textContent,`${count===2?'Two':'Three'} fingers raised`);
    assert.ok(label.textContent.includes('+'));
  });
  assert.equal(new Set(expected).size,10);
});
test('ASL letters have a separate guide and history from ASL numbers',async()=>{
  const f=fixture(); await flush();
  await f.get('aslMode').fire('click');
  assert.equal(f.get('gestureGuide').childNodes.length,9);
  await f.get('aslLettersMode').fire('click');
  assert.equal(f.get('gestureGuide').childNodes.length,26);
  assert.equal(f.get('gestureGuide').childNodes[0].childNodes[1].textContent,'A');
  assert.equal(f.get('gestureGuide').childNodes[25].childNodes[1].textContent,'Z');
  assert.equal(f.get('wordBuilderActions').hidden,false);
  assert.equal(f.get('aslLettersMode')['aria-pressed'],'true');
  assert.equal(f.get('aslSubmode').hidden,false);
  await f.get('aslNumbersMode').fire('click');
  assert.equal(f.get('gestureGuide').childNodes.length,9);
});
test('ASL letters join into words with manual spacing and deletion',async()=>{
  const f=fixture({liveResults:[
    ...Array(3).fill({gesture:'LETTER_C',text:'C',confidence:.94,handPresent:true}),
    ...Array(3).fill({gesture:'LETTER_A',text:'A',confidence:.94,handPresent:true}),
    ...Array(3).fill({gesture:'LETTER_T',text:'T',confidence:.94,handPresent:true}),
  ]});
  await flush();
  await f.get('aslMode').fire('click');
  await f.get('aslLettersMode').fire('click');
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click'); await flush();
  for(let i=0;i<8;i++) await f.tick();
  assert.equal(f.get('historyText').textContent,'CAT');
  await f.get('wordSpaceButton').fire('click');
  assert.equal(f.get('historyText').textContent,'CAT ');
  await f.get('wordBackspaceButton').fire('click');
  assert.equal(f.get('historyText').textContent,'CAT');
  await f.get('wordRepeatButton').fire('click');
  assert.equal(f.get('historyText').textContent,'CATT');
  await f.get('aslNumbersMode').fire('click');
  assert.equal(f.get('wordBuilderActions').hidden,true);
});
test('recognized history changes language without losing the original words',async()=>{
  const yes={gesture:'Thumb_Up',text:'Yes',confidence:.9,handPresent:true};
  const f=fixture({liveResults:[yes,yes,yes]}); await flush();
  f.get('video').readyState=2;
  await f.get('cameraButton').fire('click'); await flush();
  await f.tick(); await f.tick();
  assert.equal(f.get('historyText').textContent,'Yes');
  f.context.window.sign2textI18n={translate:word=>word==='Yes'?'Да':word};
  f.context.window.sign2textRefreshLocale();
  assert.equal(f.get('historyText').textContent,'Да');
  f.context.window.sign2textI18n={translate:word=>word};
  f.context.window.sign2textRefreshLocale();
  assert.equal(f.get('historyText').textContent,'Yes');
});
