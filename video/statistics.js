"use strict";

// Statistics consume a saved report only after an explicit request. They never
// replace tracking, seek the video automatically, or run in the replay loop.
let videoStatistics, statisticsContext={}, statisticsAppliedContext={};
let statisticsGeneration=0, statisticsBusy=false;
let statisticsHalfEdited=false;
let statisticsImportedNonLive=false;
const statisticsFlags={
  "statistics-live":"live_play", "statistics-calibration":"calibration_reviewed",
  "statistics-orientation":"orientation_confirmed", "statistics-dimensions":"pitch_dimensions_verified",
  "statistics-ground-ball":"ground_ball_reviewed",
};
const statisticsLabels={
  positioning:"Positioning",movement:"Observed movement",physical:"Physical estimates",relationships:"Distances and relationships",
  off_ball:"Off-ball evidence",events:"Reviewed match actions",shape:"Observed team shape",coverage:"Coverage",
  average_position_m:"Average position",observed_distance_m:"Distance during observable tracking",forward_m:"Forward displacement",
  backward_m:"Backward displacement",lateral_m:"Lateral displacement",mean_speed_m_s:"Average speed",max_speed_m_s:"Maximum speed",
  high_speed_distance_m:"High-speed running distance",high_speed_time_ms:"High-speed running time",sprint_bouts:"Sprint bouts",
  sprint_distance_m:"Sprint distance",max_acceleration_m_s2:"Maximum acceleration",max_deceleration_m_s2:"Maximum deceleration",
  acceleration_bouts:"Acceleration bouts",deceleration_bouts:"Deceleration bouts",below_noise_ms:"Time below movement resolution",
  mean_distance_to_ball_m:"Average distance to ground ball",mean_nearest_teammate_m:"Average nearest-teammate distance",
  mean_nearest_opponent_m:"Average nearest-opponent distance",width_m:"Width",depth_m:"Depth",length_m:"Length",
  support_runs:"Support-run candidates",overlaps:"Overlap candidates",recovery_runs:"Recovery-run candidates",
  passing_lane_availability:"Passing-lane availability",pressure_approaches:"Pressure-approach candidates",
  central_screening:"Central screening",deep_to_advanced_support:"Deep-to-advanced support",deep_passing_outlet:"Deep passing outlet",
  between_line_availability:"Between-line availability",between_line_receptions:"Reviewed between-line receptions",
  pass_attempts:"Pass attempts",passes_completed:"Completed passes",pass_completion_pct:"Pass completion",
  touches:"Touches",tackles:"Tackle attempts",tackles_won:"Tackles won",interceptions:"Interceptions",shots:"Shots",
  press_attempts:"Press attempts",successful_presses:"Successful presses",possession_ms:"Reviewed player control time",
  scope_ms:"Selected scope",observed_ms:"Observed time",usable_ms:"Usable time",coverage_pct:"Usable coverage",
  position_coverage_pct:"Position coverage",resolved_time_ms:"Resolved zone time",directional_time_ms:"Direction-reviewed time",
  usable_observed_pct:"Usable share of observed time",scope_coverage_pct:"Coverage of selected scope",physical_ms:"Time usable for physical estimates",
  reviewed_possession_ms:"Reviewed team possession",reviewed_possession_pct:"Reviewed possession share",convex_hull_area_m2:"Observed team area",
};
const statisticsStatusNames={unavailable:"Needs evidence",partial:"Partial coverage",estimated:"Estimate",estimate:"Estimate",reviewed:"Reviewed",available:"Available"};
function statisticsLabel(key){return statisticsLabels[key]||key.replaceAll("_"," ").replace(/\b\w/g,c=>c.toUpperCase());}
function statisticsTeamName(team){return team==="kit:team_a"?"Team A":team==="kit:team_b"?"Team B":team||"Team unconfirmed";}
function statisticsNumber(value,key=""){
  if(value===null||value===undefined||typeof value==="number"&&!finite(value))return "Not available";
  if(typeof value==="boolean")return value?"Confirmed":"Not confirmed";
  if(typeof value!=="number")return String(value).replaceAll("_"," ");
  if(/_ms$/.test(key))return fmt(value/1000)+" s";
  if(/_m_s2$/.test(key))return fmt(value)+" m/s²";
  if(/_m_s$/.test(key))return fmt(value)+" m/s";
  if(/_m2$/.test(key))return fmt(value)+" m²";
  if(/_pct$/.test(key)||key==="pct")return fmt(value)+"%";
  if(/_m$/.test(key)||key==="x"||key==="y")return fmt(value)+" m";
  return fmt(value);
}
function clearVideoStatistics(){
  statisticsGeneration++;statisticsBusy=false;videoStatistics=undefined;
  statisticsContext={};statisticsAppliedContext={};
  statisticsHalfEdited=false;
  statisticsImportedNonLive=false;
  $("statistics-results").hidden=true;$("statistics-export").disabled=true;
  $("statistics-calculate").disabled=!tracking;
  $("statistics-scope").textContent="No statistics yet";
  for(const id of Object.keys(statisticsFlags))$(id).checked=false;
  $("statistics-start").value="";$("statistics-end").value="";
  $("statistics-half-start").value="";
  status("statistics-status","Track a video or open a tracking report, then calculate statistics.");
  status("statistics-context-status","");
}
function setupVideoStatistics(){
  clearVideoStatistics();
  $("statistics-calculate").disabled=!tracking?.frames?.length;
  $("statistics-dimensions-label").textContent="The report uses the actual pitch dimensions ("+fmt(tracking?.coordinate_length_m)+" × "+fmt(tracking?.coordinate_width_m)+" m)";
  const teams=new Set(teamColours.keys());
  if(teams.size<2){
    teamsFromSummary:for(const clip of tracking?.role_inference?.clips||[])
      for(const shot of clip.shots||[])for(const group of shot.kit_groups||[]){
        if(group.inferred_team)teams.add(group.inferred_team);
        if(teams.size>=2)break teamsFromSummary;
      }
  }
  if(!teams.size){
    const frames=tracking?.frames||[];
    for(let i=0;i<Math.min(100,frames.length);i++){
      for(const o of frames[i].objects)if(o.object_type==="player"&&!probableReferee(o)&&(o.team_id||o.inferred_team))teams.add(o.team_id||o.inferred_team);
      if(teams.size>=2)break;
    }
  }
  const select=$("statistics-direction-team");select.replaceChildren();
  for(const team of teams)option(select,team,statisticsTeamName(team));
  select.value=[...teams][0]||"";$("statistics-direction-period").value="1";
  showStatisticsDirection();
  if(tracking?.frames?.length)status("statistics-status","Tracking ready. Calculate position maps now, or review the scope above to enable additional measurements.");
}
function showStatisticsDirection(){
  $("statistics-direction").value=String(statisticsContext.directions?.[$("statistics-direction-period").value]?.[$("statistics-direction-team").value]||"");
}
function collectStatisticsContext(){
  const context=JSON.parse(JSON.stringify(statisticsContext));
  for(const [id,key] of Object.entries(statisticsFlags)){
    if(key!=="live_play")context[key]=$(id).checked===true;
    else if($(id).checked===true)context.live_play=true;
    else if(statisticsImportedNonLive)context.live_play=false;
    else delete context.live_play;
  }
  for(const [id,key] of [["statistics-start","start_ms"],["statistics-end","end_ms"]]){
    const value=$(id).value.trim();
    if(value==="")delete context[key];
    else{
      const seconds=Number(value);
      if(!finite(seconds)||seconds<0)throw Error("Use a valid non-negative video time for the statistics scope.");
      context[key]=Math.round(seconds*1000);
    }
  }
  if(context.end_ms!==undefined&&context.end_ms<=(context.start_ms||0))throw Error("End time must be after the start time.");
  if(statisticsHalfEdited){
    const value=$("statistics-half-start").value.trim();
    if(value==="")delete context.period_intervals;
    else{
      const second=Number(value)*1000;
      const last=tracking?.frames?.[tracking.frames.length-1]?.video_time_ms;
      const end=finite(last)?Math.ceil(last+1000/(tracking.sample_fps||10)):context.end_ms;
      if(!finite(second)||second<0||!finite(end)||second>=end)throw Error("Set the second-half start within the tracking report.");
      const start=Math.round(second);
      context.period_intervals=[...(start>0?[{start_ms:0,end_ms:start,period:1}]:[]),{start_ms:start,end_ms:end,period:2}];
    }
  }
  for(const key of ["match_id","video_id","plan_id"])
    if(context[key]&&tracking?.[key]&&context[key]!==tracking[key])throw Error("The context file belongs to a different "+key.replaceAll("_"," ")+".");
  return context;
}
function applyStatisticsContext(context){
  if(!context||typeof context!=="object"||Array.isArray(context))throw Error("Choose a context JSON object.");
  statisticsContext=context;
  statisticsImportedNonLive=context.live_play===false;
  statisticsHalfEdited=false;
  const second=(context.period_intervals||[]).find(x=>x.period===2);
  $("statistics-half-start").value=finite(second?.start_ms)?String(second.start_ms/1000):"";
  for(const [id,key] of Object.entries(statisticsFlags))$(id).checked=context[key]===true;
  $("statistics-start").value=finite(context.start_ms)?String(context.start_ms/1000):"";
  $("statistics-end").value=finite(context.end_ms)?String(context.end_ms/1000):"";
  const teams=Object.values(context.directions||{}).flatMap(x=>x&&typeof x==="object"?Object.keys(x):[]);
  const select=$("statistics-direction-team"),known=new Set([...select.children].map(x=>x.value));
  for(const team of teams)if(!known.has(team)){option(select,team,statisticsTeamName(team));known.add(team);}
  if(!select.value)select.value=teams[0]||"";
  showStatisticsDirection();
}
function parseVideoStatistics(data){
  if(data?.version!=="fbrpro-video-stats-v1"||!Array.isArray(data.players)||!Array.isArray(data.teams)||
    data.players.some(p=>!p||typeof p.id!=="string")||data.teams.some(t=>!t||typeof t.team!=="string"))
    throw Error("Choose an FBRPRO video statistics report.");
  return data;
}
function statisticsRow(table,label,value,key,reason){
  const row=node("tr"),name=node("th",null,label),cell=node("td",value==null?"statistics-unavailable":null,statisticsNumber(value,key));
  name.setAttribute("scope","row");
  if(reason)cell.append(node("small",null,reason));
  row.append(name,cell);table.append(row);
}
function statisticsMetricRows(table,data,prefix="",depth=0,parentKey=""){
  if(!data||typeof data!=="object"||depth>3)return;
  const skipped=new Set(["status","reason","notes","evidence","heatmap","zones","id","track_id","player_id","team","config","definitions"]);
  for(const [key,value] of Object.entries(data)){
    if(skipped.has(key))continue;
    const label=(prefix?prefix+" · ":"")+statisticsLabel(key);
    if(Array.isArray(value)){
      if(key==="zone_transitions"){
        const total=value.reduce((n,x)=>n+(finite(x.count)?x.count:0),0);
        statisticsRow(table,label,value.length?total:null,"",value.length?"Observed transitions between resolved zones":"No resolved transition evidence");
      }
      continue;
    }
    if(value&&typeof value==="object"){
      if("value" in value||"count" in value){
        const metricValue="value" in value?value.value:value.count;
        const unitKey=key.endsWith("_ms")||key.endsWith("_pct")?key:"";
        statisticsRow(table,label,metricValue,unitKey,value.reason||statisticsStatusNames[value.status]);
        if(metricValue==null&&finite(value.confirmed_count)&&value.confirmed_count>0)
          statisticsRow(table,"Reviewed observations so far",value.confirmed_count,"","Partial evidence; not a complete event total");
        if(finite(value.coverage_pct))statisticsRow(table,"Coverage for "+statisticsLabel(key),value.coverage_pct,"coverage_pct");
        if(finite(value.observed_ms))statisticsRow(table,"Observed time for "+statisticsLabel(key),value.observed_ms,"observed_ms");
        if(finite(value.value_ms))statisticsRow(table,statisticsLabel(key)+" time",value.value_ms,"value_ms");
        if(finite(value.pct))statisticsRow(table,statisticsLabel(key)+" time share",value.pct,"pct");
      }else statisticsMetricRows(table,value,label,depth+1,key);
    }else statisticsRow(table,label,value,/_ms$|_pct$/.test(parentKey)?parentKey:key,value==null?data.reason:undefined);
  }
}
function statisticsSection(target,title,data,expanded=true){
  if(!data||typeof data!=="object")return;
  const article=node("details","statistics-section"),heading=node("summary",null,title);article.open=expanded;
  if(data.status)heading.append(node("span","statistics-section-state",statisticsStatusNames[data.status]||String(data.status).replaceAll("_"," ")));
  article.append(heading);
  if(data.reason)article.append(node("p","hint",data.reason));
  const table=node("table","statistics-table");statisticsMetricRows(table,data);
  if(table.childElementCount)article.append(table);else article.append(node("p","hint","No usable evidence for this measurement in the selected scope."));
  const evidence=Object.entries(data).flatMap(([key,value])=>value&&Array.isArray(value.evidence)?value.evidence.map(x=>({...x,kind:x.kind||key})):[]);
  if(evidence.length){
    const details=node("details","statistics-evidence"),summary=node("summary",null,"Review "+nf.format(evidence.length)+" evidence intervals");
    details.append(summary);
    let built=false;
    details.addEventListener("toggle",()=>{if(details.open&&!built){built=true;appendStatisticsEvidence(details,evidence);}});
    article.append(details);
  }
  target.append(article);
}
function statisticsMap(target,map,columns,rows,average){
  target.replaceChildren();target.style.gridTemplateColumns="repeat("+columns+", minmax(0,1fr))";
  const values=Array.isArray(map?.pct)?map.pct.slice(0,columns*rows):[];const max=Math.max(1,...values.filter(finite));
  for(let i=0;i<columns*rows;i++){
    const value=values[i],cell=node("div");
    cell.style.background=finite(value)?"rgba(135, 201, 130, "+(.12+.75*value/max)+")":"#244e3b";
    cell.title="Row "+(Math.floor(i/columns)+1)+", column "+(i%columns+1)+": "+(finite(value)?fmt(value)+"% of usable time":"No usable evidence");
    cell.append(node("strong",null,finite(value)?fmt(value)+"%":"—"));target.append(cell);
  }
  const length=videoStatistics?.coordinate_length_m||tracking?.coordinate_length_m;
  const width=videoStatistics?.coordinate_width_m||tracking?.coordinate_width_m;
  if(average&&finite(average.x)&&finite(average.y)&&positive(length)&&positive(width)){
    const dot=node("span","statistics-average");dot.style.left=Math.max(0,Math.min(100,average.x/length*100))+"%";
    dot.style.top=Math.max(0,Math.min(100,average.y/width*100))+"%";
    dot.title="Average position: "+fmt(average.x)+" m, "+fmt(average.y)+" m";target.append(dot);
  }
}
function renderStatisticsPlayer(){
  const p=videoStatistics?.players.find(x=>x.id===$("statistics-player").value);
  $("statistics-player-coverage").replaceChildren();$("statistics-player-metrics").replaceChildren();
  if(!p){statisticsMap($("statistics-heatmap"),null,6,4);statisticsMap($("statistics-zones"),null,3,3);return;}
  $("statistics-player-identity").textContent=(p.player_id?"Confirmed player: "+p.player_id:"Anonymous tracking identity; replay number is not a shirt number")+
    " · "+statisticsTeamName(p.team);
  statisticsSection($("statistics-player-coverage"),"Observed coverage",p.coverage);
  statisticsMap($("statistics-heatmap"),p.positioning?.heatmap,6,4,p.positioning?.average_position_m);
  statisticsMap($("statistics-zones"),p.positioning?.zones,3,3);
  $("statistics-zone-note").textContent="Percent of usable observed time · uncertain boundary time: "+statisticsNumber(p.positioning?.zones?.uncertain_ms,"uncertain_ms")+". Zone percentages can total less than 100%.";
  for(const key of ["positioning","movement","physical","relationships","off_ball","events"])
    statisticsSection($("statistics-player-metrics"),statisticsLabel(key),p[key],["positioning","movement"].includes(key));
}
function renderStatisticsTeam(){
  $("statistics-team-metrics").replaceChildren();
  const team=videoStatistics?.teams.find(t=>t.team===$("statistics-team").value);if(!team)return;
  statisticsSection($("statistics-team-metrics"),"Observed team coverage",team.coverage);
  statisticsSection($("statistics-team-metrics"),"Observed team shape",team.shape);
  const context=Object.fromEntries(Object.entries(team).filter(([key])=>!["team","coverage","shape"].includes(key)));
  if(Object.keys(context).length)statisticsSection($("statistics-team-metrics"),"Reviewed team context",context);
  const possession=videoStatistics?.possession?.teams?.find(t=>t.team===team.team);
  if(possession)statisticsSection($("statistics-team-metrics"),"Reviewed team possession",possession);
}
function appendStatisticsEvidence(target,items){
  for(const candidate of items.slice(0,100)){
    const el=node("article","statistics-candidate");
    const start=candidate.start_ms??candidate.t_ms??candidate.video_time_ms;
    const end=candidate.end_ms??start;
    el.append(node("p",null,statisticsLabel(candidate.kind||candidate.type||"Action candidate")+" · "+clock(start)+
      (finite(end)&&end!==start?"–"+clock(end):"")+" · "+(candidate.reason||"Review required")));
    if(finite(start)&&!getMismatch(videoStatistics,tracking)&&tracking){
      const button=node("button","secondary","Review in footage");button.type="button";button.onclick=()=>seek(Math.round(start));el.append(button);
    }
    target.append(el);
  }
  if(items.length>100)target.append(node("p","hint","Showing the first 100 intervals. Export the report for the complete evidence list."));
}
function renderVideoStatistics(){
  const data=videoStatistics;$("statistics-results").hidden=!data;$("statistics-export").disabled=!data;if(!data)return;
  const sourceDifference=getMismatch(data,tracking);
  $("statistics-scope").textContent=nf.format(data.players.length)+" tracked players";
  const notes=Array.isArray(data.notes)?data.notes.filter(x=>typeof x==="string"):[];
  $("statistics-notes").textContent=[sourceDifference?"This report belongs to different "+sourceDifference+"; footage evidence links are disabled.":"",
    "Stats describe observable tracking in the selected scope, not unseen full-match movement. Automatic calibration and anonymous identities remain estimates."].filter(Boolean).join(" ");
  $("statistics-method-notes").replaceChildren();for(const note of notes)$("statistics-method-notes").append(node("li",null,note));
  const player=$("statistics-player"),prior=player.value;player.replaceChildren();
  const ordered=[...data.players].sort((a,b)=>(b.coverage?.usable_ms||0)-(a.coverage?.usable_ms||0));
  ordered.forEach((p,i)=>option(player,p.id,(p.name||p.player_id||"Track #"+(playerNumbers.get(p.track_id||p.id)||i+1))+" · "+statisticsTeamName(p.team)));
  player.value=data.players.some(p=>p.id===prior)?prior:ordered[0]?.id||"";
  const team=$("statistics-team"),oldTeam=team.value;team.replaceChildren();for(const t of data.teams)option(team,t.team,statisticsTeamName(t.team));
  team.value=data.teams.some(t=>t.team===oldTeam)?oldTeam:data.teams[0]?.team||"";
  renderStatisticsPlayer();renderStatisticsTeam();
  $("statistics-candidates").replaceChildren();
  const candidates=Array.isArray(data.event_candidates)?data.event_candidates:[];
  if(candidates.length)appendStatisticsEvidence($("statistics-candidates"),candidates);
  else $("statistics-candidates").append(node("p","hint","No reviewable action candidates in this report. Missing ball evidence is not evidence that no actions occurred."));
}
$("statistics-calculate").onclick=async()=>{
  if(!tracking||statisticsBusy)return;
  const source=tracking,request=++statisticsGeneration;
  try{
    const context=collectStatisticsContext();statisticsBusy=true;$("statistics-calculate").disabled=true;
    status("statistics-status","Calculating from saved positions. The tracker and replay remain unchanged…");
    const result=await api("/api/statistics",{method:"POST",headers:{...trackHeaders,"Content-Type":"application/json"},
      body:JSON.stringify({tracking:source,context})}).then(r=>r.json());
    if(request!==statisticsGeneration||tracking!==source)return;
    videoStatistics=parseVideoStatistics(result);statisticsAppliedContext=context;statisticsContext=context;
    renderVideoStatistics();status("statistics-status","Statistics ready. Unavailable measurements explain the evidence they still need.");
  }catch(error){if(request===statisticsGeneration)status("statistics-status",error.message,true);}
  finally{if(request===statisticsGeneration){statisticsBusy=false;$("statistics-calculate").disabled=!tracking?.frames?.length;}}
};
$("statistics-direction-team").onchange=showStatisticsDirection;
$("statistics-half-start").onchange=()=>{statisticsHalfEdited=true;};
$("statistics-direction-period").onchange=showStatisticsDirection;
$("statistics-direction").onchange=()=>{
  const team=$("statistics-direction-team").value,period=$("statistics-direction-period").value;if(!team)return;
  statisticsContext.directions??={};statisticsContext.directions[period]??={};
  if($("statistics-direction").value)statisticsContext.directions[period][team]=Number($("statistics-direction").value);
  else delete statisticsContext.directions[period][team];
};
$("statistics-player").onchange=renderStatisticsPlayer;$("statistics-team").onchange=renderStatisticsTeam;
for(const view of ["player","team","candidates"])$("statistics-tab-"+view).onclick=()=>{
  for(const key of ["player","team","candidates"]){
    $("statistics-tab-"+key).setAttribute("aria-pressed",String(key===view));
    $("statistics-"+(key==="candidates"?"candidate":key)+"-view").hidden=key!==view;
  }
};
$("statistics-file").addEventListener("change",async e=>{
  const file=e.target.files[0];if(!file)return;const request=++statisticsGeneration;statisticsBusy=false;
  try{
    const data=parseVideoStatistics(JSON.parse(await file.text()));if(request!==statisticsGeneration)return;
    videoStatistics=data;statisticsAppliedContext=data.context||{};renderVideoStatistics();
    $("statistics-calculate").disabled=!tracking?.frames?.length;status("statistics-status","Statistics report opened. Calculations have not changed the active tracking report.");
  }catch(error){if(request===statisticsGeneration)status("statistics-status",error.message,true);}
});
$("statistics-context-file").addEventListener("change",async e=>{
  const file=e.target.files[0];if(!file)return;
  try{applyStatisticsContext(JSON.parse(await file.text()));status("statistics-context-status",file.name+" loaded. Check the scope, then calculate statistics.");}
  catch(error){status("statistics-context-status",error.message,true);}
});
$("statistics-export").onclick=()=>{if(videoStatistics)download("video-statistics.json",videoStatistics);};
$("statistics-context-export").onclick=()=>{try{download("video-statistics-context.json",collectStatisticsContext());}catch(error){status("statistics-context-status",error.message,true);}};
$("statistics-context-template").onclick=()=>download("video-statistics-context-template.json",{
  match_id:tracking?.match_id,video_id:tracking?.video_id,plan_id:tracking?.plan_id,
  calibration_reviewed:false,orientation_confirmed:false,pitch_dimensions_verified:false,ground_ball_reviewed:false,
  directions:{},identities:{},episodes:[],events:[],event_coverage:[],possession_intervals:[],
});
