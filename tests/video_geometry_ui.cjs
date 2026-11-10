const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Node {
  constructor(tag='div'){
    this.tag=tag;
    this.children=[];
    this.events={};
    this.attributes=new Map();
    this.dataset={};
    this.style={};
    this.classList=new Set();
    this.classList.toggle=(name,on)=>on?this.classList.add(name):this.classList.delete(name);
    this.classList.remove=name=>this.classList.delete(name);
    this.files=[];
    this.value='';
    this.currentTime=0;
    this.duration=NaN;
    this.textContent='';
  }
  addEventListener(type,fn){(this.events[type]??=[]).push(fn);}
  async dispatchEvent(event){
    if(!event.target)event.target=this;
    for(const fn of this.events[event.type]??[])await fn(event);
  }
  get childElementCount(){return this.children.length;}
  setAttribute(name,value){this.attributes.set(name,String(value));}
  hasAttribute(name){return this.attributes.has(name);}
  toggleAttribute(name,flag){flag?this.setAttribute(name,''):this.attributes.delete(name);}
  append(...children){for(const c of children)this.children.push(...(c.tag==='#fragment'?c.children:[c]));}
  replaceChildren(...children){this.children=[];this.append(...children);}
}
const html=fs.readFileSync('video/index.html','utf8');
const elements=new Map([...html.matchAll(/<(\w+)\b[^>]*\bid="([^"]+)"[^>]*>/g)].map(([match,tag,id])=>{
  const element=new Node(tag);
  for(const name of ['hidden','disabled']){
    if(new RegExp('\\b'+name+'(?=[\\s>])').test(match)){
      element[name]=true;
      element.setAttribute(name,'');
    }
  }
  return [id,element];
}));
const find=id=>elements.get(id);
const doc=new Node('document');
Object.assign(doc,{getElementById:find,createElement:tag=>new Node(tag),createElementNS:(_,tag)=>new Node(tag),createDocumentFragment:()=>new Node('#fragment')});
const raf=[];
const ctx=vm.createContext({
  document:doc,requestAnimationFrame:fn=>raf.push(fn),
  CustomEvent:class {constructor(type,{detail}){this.type=type;this.detail=detail;}},
});
for(const name of ['review.js','analysis.js','geometry.js'])vm.runInContext(fs.readFileSync('video/'+name,'utf8'),ctx,{filename:name});
const flush=()=>{while(raf.length)raf.shift()();};
const change=async(id,data,name=id+'.json')=>{
  const el=find(id);
  el.files=[{name,text:async()=>typeof data==='string'?data:JSON.stringify(data)}];
  await el.dispatchEvent({type:'change'});
  flush();
};
const publish=async(id,type)=>{await find(id).dispatchEvent({type});flush();};
const descendants=(el,tag)=>el.children.flatMap(c=>[...(c.tag===tag?[c]:[]),...descendants(c,tag)]);
const source={match_id:'m',video_id:'video',plan_id:'plan'};
const clip={id:'c',period:1,video_start_ms:1000,video_end_ms:2000,opportunities:[{
  id:'opp',reason:'regain_candidate',period:1,video_start_ms:1100,video_end_ms:1900,
  match_start_ms:0,match_end_ms:800,
}]};
const plan={...source,version:'fbrpro-video-plan-v1',clips:[clip],players:[{id:'p',name:'Alex'}],selected_video_ms:1000};
const seen={object_type:'player',observed:true,estimated:false,visibility:'observed'};
const frames=[
  {clip_id:'c',video_time_ms:1000,calibration_status:'calibrated',objects:[
    {...seen,track_id:'t1',player_id:'p',team_id:'a',x_m:11,y_m:20,bbox_xyxy:[12,40,24,80]},
    {...seen,track_id:'t2',x_m:52,y_m:20,bbox_xyxy:[40,12,50,34]},
    {...seen,track_id:'t3',observed:false,visibility:'missing',x_m:42,y_m:40},
  ]},
  {clip_id:'c',video_time_ms:1250,calibration_status:'uncalibrated',objects:[
    {...seen,track_id:'t4',x_m:null,y_m:null,bbox_xyxy:[82,60,100,120]},
  ]},
];
const tracking={...source,version:'fbrpro-video-tracking-v1',coordinate_length_m:105,coordinate_width_m:68,frame_width:1280,frame_height:720,frames};
const measurement={metric:'deep_to_advanced',label:'Deep-to-advanced runs',unit:'episodes',numerator:0,denominator:0,excluded:1,observations:[
  {...source,clip_id:'c',video_start_ms:1250,status:'unobservable',reason:'Player identity unresolved'},
]};
const metrics={...source,version:'fbrpro-video-metrics-v1',analysis_scope:'selected_clips',players:[{
  player_id:'p',name:'Alex',measures:[measurement],
}]};
async function evidenceButton(index=0){
  const details=descendants(find('metric-cards'),'details')[0];
  details.open=true;
  await details.dispatchEvent({type:'toggle'});
  return descendants(details,'button')[index];
}
(async()=>{
  await change('plan-file',plan);
  await change('tracking-file',tracking);
  assert.equal(find('geometry-apply').disabled,true);
  assert.match(find('geometry-status').textContent,/Rerun tracking/);
  const g={...tracking,pitch_segments:[{id:'s',start_ms:1000,end_ms:1250,period:1}],
    pitch_review:{segments:{s:{orientation:'normal',period:1,live_play:true}},directions:{'1':{'kit:team_a':1},'2':{'kit:team_a':-1}}},
    frames:frames.map(f=>({...f,pitch_geometry:{segment_id:'s',usable:true,continuity_status:'camera_checked',reviewed_orientation:'normal'},
      objects:f.objects.map(o=>({...o,team_id:null,inferred_team:'kit:team_a'}))})),
    spatial_summary:{players:[{track_id:'t1',team:'kit:team_a',observed_interval_ms:1000,usable_ms:800,coverage_pct:80,
      zones_pct:[25,25,0,0,0,0,0,0,0],boundary_uncertain_ms:400,forward_m:3,backward_m:0,lateral_m:1,
      directional_usable_ms:800,direction_uncertain_ms:200}]}};
  await change('tracking-file',g);
  assert.equal(find('geometry-apply').disabled,false);
  assert.equal(find('geometry-orientation').value,'normal');
  assert.equal(find('geometry-live').checked,true);
  assert.equal(find('direction-first').value,'1');
  assert.equal(find('direction-second').value,'-1');
  assert.equal(find('zone-grid').children.length,9);
  assert.match(find('spatial-coverage').textContent,/80% usable coverage/);
  assert.equal(find('spatial-cards').children.length,3);
  assert.match(find('geometry-current').textContent,/Camera movement checked/);
  vm.runInContext('let sent; fetch=async(url,options)=>{sent=JSON.parse(options.body);return {ok:true,json:async()=>sent.tracking};};',ctx);
  find('geometry-orientation').value='flipped';
  find('geometry-period').value='2';
  await find('geometry-apply').onclick(); flush();
  assert.equal(vm.runInContext('sent.review.segments.s.orientation',ctx),'flipped');
  assert.equal(vm.runInContext('sent.review.segments.s.period',ctx),2);
  assert.match(find('geometry-status').textContent,/Review applied/);
  vm.runInContext('clearGeometry()',ctx);
  assert.equal(find('geometry-panel').hidden,true);
  assert.equal(find('spatial-results').hidden,true);
  console.log('Pitch setup UI passed: legacy report, orientation, periods, team direction, zone grid, coverage and apply request.');
})().catch(error=>{console.error(error);process.exitCode=1;});
