const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function adapter(responses) {
  const urls=[];
  const context=vm.createContext({window:{}, AbortController,
    setTimeout:(fn,ms)=> { if(ms===1200) queueMicrotask(fn); return 1; }, clearTimeout(){},
    fetch:async(url)=>{
      urls.push(url);
      const data=responses.shift();
      if(data instanceof Error) throw data;
      if(data===undefined) throw Error('Unexpected extra request');
      return {ok:!data.error,status:data.error?503:200,json:async()=>data};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../app/model-adapter.js'),'utf8'),context);
  return {model:context.window.sign2textModel,urls};
}

test('adapter polls progress then returns a completed result',async()=>{
  const result={text:'mock translation'};
  const f=adapter([{jobId:'123'},{status:'running',message:'Loading'},{status:'complete',result}]);
  const progress=[];
  assert.deepEqual(await f.model.translate({blob:{},onProgress:x=>progress.push(x)}),result);
  assert.deepEqual(progress,['Loading']);
  assert.deepEqual(f.urls,['/api/translate','/api/jobs/123','/api/jobs/123']);
});
test('adapter preserves server error instead of showing a fixture',async()=>{
  const f=adapter([{jobId:'123'},{status:'error',message:'No clear hands'}]);
  await assert.rejects(()=>f.model.translate({blob:{}}),/No clear hands/);
});
test('adapter recovers a temporary poll failure',async()=>{
  const result={text:'mock translation'};
  const f=adapter([{jobId:'123'},Error('offline'),{status:'complete',result}]);
  assert.deepEqual(await f.model.translate({blob:{}}),result);
});
test('unavailable model fails before polling',async()=>{
  const f=adapter([{error:'Model download incomplete'}]);
  await assert.rejects(()=>f.model.translate({blob:{}}),/download incomplete/);
  assert.equal(f.urls.length,1);
});
test('live recognition sends one frame and returns its result',async()=>{
  const result={gesture:'Thumb_Up',text:'Yes',confidence:.9};
  const f=adapter([result]);
  assert.deepEqual(await f.model.live({}),result);
  assert.deepEqual(f.urls,['/api/live']);
});
