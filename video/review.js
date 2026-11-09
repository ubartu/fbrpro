"use strict";
const $ = id => document.getElementById(id);
const video = $("video");
const reason = {regain_candidate: "Team regain", opposition_build_up_sample: "Opposition build-up",
  own_possession_sample: "Team possession candidate"};
const label = {observed:"Observed",not_observed:"Visible; not observed",unobservable:"Unobservable"};
let fileUrl, playbackEnd, clipPlan, segments = [], tags = [];
let videoRequest = 0, planRequest = 0;
const validMs = n => Number.isSafeInteger(n) && n >= 0;
function status(id,message,error=false){
  $(id).textContent=message; $(id).classList.toggle("error",error);
}
function option(select,value,content){
  const opt = document.createElement("option");
  opt.value=value; opt.textContent=content; select.append(opt);
}
function download(filename,data){
  const a=document.createElement("a");
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+"\n"],{type:"application/json"}));
  a.href=url; a.download=filename; a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function list(id,items){
  $(id).replaceChildren();
  for(const s of items){const li=document.createElement("li");li.textContent=s;$(id).append(li);}
}
function secs(ms){return (ms/1000).toFixed(2)+" s";}
function clip(){return clipPlan?.clips.find(c=>c.id===$("clip").value);}
function opportunity(){return clip()?.opportunities.find(o=>o.id===$("opportunity").value);}
function refresh(){
  const c=clip();
  $("opportunity").replaceChildren();
  for(const o of c?.opportunities||[])option($("opportunity"),o.id,reason[o.reason]??o.reason);
  $("opportunity").disabled=!c;
  $("clip-time").textContent=c?secs(c.video_start_ms)+" to "+secs(c.video_end_ms):"";
  $("play").disabled=!c||!Number.isFinite(video.duration);
  $("save-tag").disabled=!c||!$("player").value||!Number.isFinite(video.duration);
}
function setVideo(file){
  if(tags.length)return status("video-status","Export observations and reopen the page before changing footage.",true);
  const generation=++videoRequest;
  playbackEnd=undefined;
  if(fileUrl) URL.revokeObjectURL(fileUrl);
  fileUrl=URL.createObjectURL(file);
  video.src=fileUrl;
  video.onloadedmetadata=()=>{
    if(generation!==videoRequest)return;
    if(!$("video-id").value)$("video-id").value=file.name;
    status("video-status","Loaded "+file.name+" · "+secs(video.duration*1000));refresh();
  };
  video.onerror=()=>{status("video-status","This browser could not open the video.",true);refresh();};
  refresh();
}
$("video-file").addEventListener("change",e=>{if(e.target.files[0])setVideo(e.target.files[0]);});
video.addEventListener("timeupdate",()=>{
  if(playbackEnd!==undefined&&video.currentTime>=playbackEnd){video.pause();playbackEnd=undefined;}
});
for(const [button,input] of [["take-start","vs1"],["take-end","vs2"],["take-check","vs3"]]){
  $(button).onclick=()=>{
    if(!Number.isFinite(video.duration))return status("alignment-status","Open a video first.",true);
    $(input).value=video.currentTime.toFixed(3);
  };
}
function seconds(id){const s=$(id).value;return s.trim()===""?NaN:Math.round(Number(s)*1000);}
$("add-segment").onclick=()=>{
  const row={id:$("segment-id").value.trim()||"segment-"+(segments.length+1),period:Number($("period").value),
    match_start_ms:seconds("ms1"),match_end_ms:seconds("ms2"),
    video_start_ms:seconds("vs1"),video_end_ms:seconds("vs2")};
  const values=[row.match_start_ms,row.match_end_ms,row.video_start_ms,row.video_end_ms];
  if(!values.every(validMs)||!Number.isSafeInteger(row.period)||row.period<1||
    row.match_start_ms>=row.match_end_ms||row.video_start_ms>=row.video_end_ms)
    return status("alignment-status","Enter two increasing clock anchors and a valid half.",true);
  if(Number.isFinite(video.duration)&&row.video_end_ms>video.duration*1000+1)
    return status("alignment-status","The end anchor is beyond the video.",true);
  if($("ms3").value.trim()||$("vs3").value.trim()){
    const match_ms=seconds("ms3"),video_ms=seconds("vs3");
    if(!validMs(match_ms)||!validMs(video_ms)||match_ms<=row.match_start_ms||match_ms>=row.match_end_ms||
      video_ms<=row.video_start_ms||video_ms>=row.video_end_ms)
      return status("alignment-status","The check event needs both times and must fall between the anchors.",true);
    row.check_anchors=[{match_ms,video_ms}];
    const expected=row.video_start_ms+(match_ms-row.match_start_ms)*(row.video_end_ms-row.video_start_ms)/
      (row.match_end_ms-row.match_start_ms);
    if(Math.abs(Math.round(expected)-video_ms)>500)
      return status("alignment-status","Clock check differs by more than 0.5 seconds. Check the anchors or split the segment.",true);
  }
  const ordered=[...segments,row].sort((a,b)=>a.period-b.period||a.match_start_ms-b.match_start_ms);
  if(segments.some(s=>s.id===row.id)||ordered.some((s,i)=>i>0&&(
    ordered[i-1].video_end_ms>s.video_start_ms||
    (ordered[i-1].period===s.period&&ordered[i-1].match_end_ms>s.match_start_ms))))
    return status("alignment-status","Segments must have unique IDs and cannot overlap or reverse order.",true);
  segments=ordered;
  list("segments",segments.map(s=>s.id+" · half "+s.period+" · "+secs(s.video_start_ms)+" to "+secs(s.video_end_ms)+
    (s.check_anchors?" · clock checked":" · clock unchecked")));
  $("segment-id").value="";status("alignment-status","Segment added.");
};
$("download-alignment").onclick=()=>{
  const match_id=$("match-id").value.trim(),video_id=$("video-id").value.trim();
  if(!match_id||!video_id||!segments.length)
    return status("alignment-status","Enter the match and video IDs and add a segment.",true);
  download("video-alignment.json",{match_id,video_id,segments});
};
function parsePlan(data){
  if(data?.version!=="fbrpro-video-plan-v1"||typeof data.match_id!=="string"||
     typeof data.video_id!=="string"||typeof data.plan_id!=="string"||!Array.isArray(data.clips)||!Array.isArray(data.players))
    throw Error("Choose a clip plan generated by fbrpro.");
  for(const c of data.clips){
    if(!c||typeof c.id!=="string"||!validMs(c.video_start_ms)||!validMs(c.video_end_ms)||
       c.video_end_ms<=c.video_start_ms||!Array.isArray(c.opportunities)||!c.opportunities.length||
       c.opportunities.some(o=>!o||typeof o.id!=="string"||!validMs(o.video_start_ms)||
       !validMs(o.video_end_ms)||o.video_start_ms<c.video_start_ms||o.video_end_ms>c.video_end_ms||
       o.video_start_ms>=o.video_end_ms))
      throw Error("The plan contains an invalid clip or opportunity.");
  }
  const validRoster=roster=>Array.isArray(roster)&&roster.every(p=>p&&typeof p.id==="string"&&typeof p.name==="string");
  if(!validRoster(data.players)||(data.identity_roster!==undefined&&!validRoster(data.identity_roster)))
    throw Error("The plan contains an invalid player.");
  return data;
}
$("plan-file").addEventListener("change",async e=>{
  const file=e.target.files[0], generation=++planRequest;
  if(!file)return;
  try{
    const data=parsePlan(JSON.parse(await file.text()));
    if(generation!==planRequest)return;
    if(tags.length&&clipPlan.plan_id!==data.plan_id)
      throw Error("Export your current observations and reopen the page before switching clip plans.");
    clipPlan=data;
    $("clip").replaceChildren();$("player").replaceChildren();
    for(const c of data.clips) option($("clip"),c.id,c.id+" · half "+c.period+" · "+secs(c.video_start_ms));
    for(const p of data.players) option($("player"),p.id,p.name);
    $("clip").disabled=!data.clips.length;$("player").disabled=!data.players.length;
    status("plan-status",data.clips.length+" clips · "+secs(data.selected_video_ms)+" selected · video ID: "+data.video_id);
    refresh();
    document.dispatchEvent(new CustomEvent("fbrpro:plan",{detail:{name:file.name}}));
  }catch(err){if(generation===planRequest)status("plan-status",err.message,true);}
});
$("clip").onchange=refresh;$("player").onchange=refresh;
$("play").onclick=async()=>{
  const c=clip();
  if(!c||c.video_end_ms>video.duration*1000+1)
    return status("video-status","This clip falls outside the loaded video. Check the file and alignment.",true);
  video.currentTime=c.video_start_ms/1000;playbackEnd=c.video_end_ms/1000;
  try{await video.play();}catch(err){status("video-status","Playback could not start: "+err.message,true);}
};
$("save-tag").onclick=()=>{
  const c=clip(),o=opportunity(),player_id=$("player").value;
  if(!c||!o||!player_id||!Number.isFinite(video.duration)||o.video_end_ms>video.duration*1000+1)
    return status("tag-status","Select a player and an opportunity within the loaded video.",true);
  const entry={clip_id:c.id,opportunity_id:o.id,period:o.period,player_id,
    video_start_ms:o.video_start_ms,video_end_ms:o.video_end_ms,
    period_start_ms:o.match_start_ms,period_end_ms:o.match_end_ms,
    behaviour:$("behaviour").value,observation:$("observation").value,notes:$("notes").value.trim()};
  const index=tags.findIndex(t=>["clip_id","opportunity_id","player_id","behaviour"].every(k=>t[k]===entry[k]));
  if(index<0)tags.push(entry);else tags[index]=entry;
  list("tags",tags.map(t=>[t.opportunity_id,clipPlan.players.find(p=>p.id===t.player_id)?.name??t.player_id,
     t.behaviour.replaceAll("_"," "),label[t.observation]].join(" · ")));
  $("notes").value="";$("export-tags").disabled=false;
  status("tag-status",index<0?"Observation saved for export.":"Observation updated.");
};
$("export-tags").onclick=()=>download("video-observations.json",{
  version:"fbrpro-video-observations-v1",match_id:clipPlan.match_id,
  video_id:clipPlan.video_id,plan_id:clipPlan.plan_id,analysis_scope:"selected_clips",observations:tags
});
