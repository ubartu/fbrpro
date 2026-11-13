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
for(const name of ['review.js','analysis.js','statistics.js'])vm.runInContext(fs.readFileSync('video/'+name,'utf8'),ctx,{filename:name});
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
    {...seen,track_id:'staff',on_pitch:false,x_m:0,y_m:0},
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
const stats={...source,version:'fbrpro-video-stats-v1',coordinate_length_m:105,coordinate_width_m:68,notes:['Selected observable intervals only.'],
  lineup_snapshot:{t_ms:1000,track_ids:['t1','t2']},
  players:[{id:'t1',track_id:'t1',player_id:null,team:'a',coverage:{scope_ms:1000,usable_ms:800,coverage_pct:80},
    positioning:{status:'estimated',average_position_m:{x:20,y:30},width_m:10,depth_m:15,
      heatmap:{columns:6,rows:4,pct:[100,...Array(23).fill(0)]},zones:{columns:3,rows:3,pct:[100,...Array(8).fill(0)],uncertain_ms:0}},
    movement:{status:'estimated',relative_distance_pct:5.7,observed_distance_m:null},physical:{status:'unavailable',reason:'Live play and calibration review required',max_speed_m_s:null},
    relationships:{mean_nearest_teammate_m:8},off_ball:{support_runs:{count:null,status:'unavailable',reason:'Reviewed phase is missing'}},
    events:{pass_attempts:{value:null,confirmed_count:1,status:'partial',reason:'Event coverage is incomplete'}}},
    {id:'t2',track_id:'t2',team:null},{id:'t4',track_id:'t4',team:'b'}],
  teams:[{team:'a',coverage:{usable_ms:800},shape:{status:'estimated',width_m:30,length_m:40}}],
  event_candidates:[{type:'ball_proximity',start_ms:1000,end_ms:1200,status:'review_required'}]};
function allText(el){return String(el.textContent||'')+' '+el.children.map(allText).join(' ');}
(async()=>{
  await change('tracking-file',tracking);
  vm.runInContext(`footageLayer.toggleAttribute('hidden',false);
    const boundary=[[0,0],[100,0],[100,100]];
    drawOverlay({objects:[],pitch_overlay:{source:'stabilized_calibration',segments_xy:[[[0,5],[100,7]]]}},boundary);`,ctx);
  assert.equal(descendants(find('video-tracks'),'line').length,1);
  assert.equal(descendants(find('video-tracks'),'polygon').length,0);
  vm.runInContext('drawOverlay({objects:[],pitch_overlay:null},boundary)',ctx);
  assert.equal(descendants(find('video-tracks'),'line').length,0);
  assert.equal(descendants(find('video-tracks'),'polygon').length,0);
  vm.runInContext('drawOverlay({objects:[]},boundary)',ctx);
  assert.equal(descendants(find('video-tracks'),'polygon').length,1);
  assert.equal(find('statistics-calculate').disabled,false);
  assert.equal(find('statistics-results').hidden,true);
  vm.runInContext('let statisticsCalls=[]; let originalTracking=tracking; fetch=async(url,options)=>{statisticsCalls.push({url,payload:JSON.parse(options.body)}); return {ok:true,json:async()=>globalThis.testStats};};',ctx);
  ctx.testStats=stats;
  find('video').currentTime=1.1;
  const time=find('video').currentTime;
  await find('statistics-calculate').onclick();flush();
  assert.equal(vm.runInContext('tracking===originalTracking',ctx),true);
  assert.equal(find('video').currentTime,time);
  assert.equal(find('statistics-results').hidden,false);
  assert.equal(find('statistics-heatmap').children.length,25);
  assert.equal(find('statistics-zones').children.length,9);
  assert.match(allText(find('statistics-player-metrics')),/Not available/);
  assert.match(allText(find('statistics-player-metrics')),/Event coverage is incomplete/);
  assert.match(allText(find('statistics-player-metrics')),/Reviewed observations so far/);
  assert.match(allText(find('statistics-player-metrics')),/5\.7%/);
  assert.deepEqual(find('statistics-player').children.map(o=>o.value),['t1']);
  find('statistics-player-team').value='';find('statistics-player-team').onchange();
  assert.deepEqual(find('statistics-player').children.map(o=>o.value),['t2']);
  find('statistics-all-tracks').checked=true;find('statistics-all-tracks').onchange();
  find('statistics-player-team').value='all';find('statistics-player-team').onchange();
  assert.equal(find('statistics-player').children.length,3);
  find('statistics-all-tracks').checked=false;find('statistics-all-tracks').onchange();
  find('statistics-review-team').value='b';find('statistics-assign-team').onclick();
  assert.equal(vm.runInContext('statisticsContext.identities.t1.team_id',ctx),'b');
  const calls=vm.runInContext('statisticsCalls',ctx);
  assert.equal(calls.length,1);
  assert.equal(calls[0].url,'/api/statistics');
  assert.equal(Object.hasOwn(calls[0].payload.context,'live_play'),false);
  assert.equal(calls[0].payload.context.orientation_confirmed,false);
  assert.equal(calls[0].payload.context.calibration_reviewed,false);
  assert.equal(calls[0].payload.context.pitch_dimensions_verified,false);
  assert.equal(calls[0].payload.tracking.match_id,source.match_id);
  find('statistics-tab-team').onclick();
  assert.equal(find('statistics-team-view').hidden,false);
  assert.match(allText(find('statistics-team-metrics')),/30 m/);
  find('statistics-tab-candidates').onclick();
  assert.equal(find('statistics-candidate-view').hidden,false);
  assert.match(allText(find('statistics-candidates')),/Review required/);
  find('video').duration=3;find('video').currentTime=1;
  find('statistics-action-frame').onclick();
  assert.deepEqual(find('statistics-action-player').children.map(o=>o.value),['t1','t2']);
  find('statistics-action-type').value='pass';find('statistics-action-type').onchange();
  find('statistics-action-outcome').value='complete';find('statistics-save-action').onclick();
  const recorded=vm.runInContext('statisticsContext.events[0]',ctx);
  assert.equal(recorded.t_ms,1000);assert.equal(recorded.track_id,'t1');
  assert.equal(recorded.track_confirmed,true);assert.equal(recorded.outcome,'complete');
  assert.match(allText(find('statistics-reviewed-actions')),/Pass.*00:01\.0.*Complete/);
  const predictions={predictions:[{half:'1',position:'31250',label:'Pass',confidence:'0.8'}]};
  ctx.testEvents={version:'fbrpro-video-events-v1',predictions_in_scope:1,predictions_outside_scope:0,player_pass_attempts:{t1:1},
    events:[{t_ms:1250,type:'pass',label:'Pass',status:'predicted',source:'soccernet',action_score:0.8,track_id:'t1',actor_reason:'Ball moved away'}]};
  vm.runInContext('let eventRequest; const statsFetch=fetch; fetch=async(url,options)=>{if(url!=="/api/video-events")return statsFetch(url,options); eventRequest=JSON.parse(options.body); return {ok:true,json:async()=>globalThis.testEvents};};',ctx);
  await change('statistics-spotting-file',predictions);
  find('statistics-spotting-file').onchange();
  find('statistics-spotting-source').value='30';
  await find('statistics-spotting-run').onclick();
  assert.equal(vm.runInContext('eventRequest.alignment.source_ms',ctx),30000);
  assert.match(allText(find('statistics-spotting-events')),/Pass.*SoccerNet score 80%/);
  assert.match(allText(find('statistics-player-metrics')),/Automatic pass estimates/);
  descendants(find('statistics-spotting-events'),'button')[0].onclick();
  assert.equal(find('video').currentTime,1.25);
  vm.runInContext('seek(1250)',ctx);flush();
  assert.equal(vm.runInContext('statisticsCalls.length',ctx),1,'Replay must not calculate statistics');
  await change('statistics-context-file',{...source,live_play:true,calibration_reviewed:true,orientation_confirmed:true,pitch_dimensions_verified:true,
    directions:{'1':{a:1},'2':{a:-1}},events:[{id:'reviewed',type:'pass',reviewed:true}],event_coverage:[]});
  assert.equal(find('statistics-live').checked,true);
  find('statistics-direction-team').value='a';find('statistics-direction-period').value='2';find('statistics-direction-period').onchange();
  assert.equal(find('statistics-direction').value,'-1');
  find('statistics-half-start').value='1.2';find('statistics-half-start').onchange();
  find('statistics-pitch-length').value='102';find('statistics-pitch-width').value='67';
  await find('statistics-calculate').onclick();
  const context=vm.runInContext('statisticsCalls[1].payload.context',ctx);
  assert.equal(context.live_play,true);assert.equal(context.calibration_reviewed,true);
  assert.equal(context.period_intervals.length,2);assert.equal(context.period_intervals[1].start_ms,1200);
  assert.equal(context.pitch_length_m,102);assert.equal(context.pitch_width_m,67);
  assert.equal(context.events[0].id,'reviewed');
  find('statistics-end').value='0';
  await find('statistics-calculate').onclick();
  assert.match(find('statistics-status').textContent,/End time must be after/);
  assert.equal(vm.runInContext('statisticsCalls.length',ctx),2);
  find('statistics-end').value='';
  vm.runInContext('fetch=async()=>new Promise(resolve=>{globalThis.resolveStatistics=resolve;});',ctx);
  const pending=find('statistics-calculate').onclick();
  await change('tracking-file',{...tracking,video_id:'new-video'});
  ctx.resolveStatistics({ok:true,json:async()=>stats});
  await pending;flush();
  assert.equal(find('statistics-results').hidden,true,'Old response must not replace new session');
  assert.equal(find('statistics-calculate').disabled,false);
  await change('statistics-file',{...stats,video_id:'other-video'});
  assert.match(find('statistics-notes').textContent,/different video IDs/);
  assert.equal(descendants(find('statistics-candidates'),'button').length,0);
  await change('statistics-file',{version:'wrong'});
  assert.match(find('statistics-status').textContent,/Choose an FBRPRO video statistics report/);
  vm.runInContext('clearVideoStatistics()',ctx);
  assert.equal(find('statistics-results').hidden,true);
  assert.equal(find('statistics-export').disabled,true);
  console.log('Video statistics UI passed: explicit post-analysis only, scope/context, evidence gates, maps, team view, candidates, stale responses and session isolation.');
})().catch(error=>{console.error(error);process.exitCode=1;});
