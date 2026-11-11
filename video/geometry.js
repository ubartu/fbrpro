"use strict";

// Review operates on saved coordinates and never starts another inference job.
let pitchReview={segments:{},directions:{}};
let geometryRequest=0;
function clearGeometry(){
  geometryRequest++;
  pitchReview={segments:{},directions:{}};
  $("geometry-panel").hidden=true;$("spatial-results").hidden=true;
  status("geometry-current","");status("geometry-status","");
}
function geometryTeamName(team){
  return team==="kit:team_a"?"Team A":team==="kit:team_b"?"Team B":inferredTeams.get(team)?.title||team||"Team unconfirmed";
}
function geometrySettings(){
  const id=$("geometry-segment").value;
  const s=pitchReview.segments[id]||{};
  $("geometry-orientation").value=s.orientation||"unresolved";
  $("geometry-period").value=String(s.period||tracking.pitch_segments?.find(x=>x.id===id)?.period||1);
  $("geometry-live").checked=s.live_play===true;
}
function geometryDirections(){
  const team=$("geometry-team").value;
  for(const [period,id] of [[1,"direction-first"],[2,"direction-second"],[3,"direction-third"],[4,"direction-fourth"]])
    $(id).value=String(pitchReview.directions[String(period)]?.[team]||"");
}
function setupGeometry(){
  geometryRequest++;
  const available=tracking.pitch_segments?.length>0;
  $("geometry-panel").hidden=false;
  $("geometry-apply").disabled=!available;
  $("geometry-export").disabled=!available;
  status("geometry-status",available?
    "Confirm the segment orientation and live play, then set attacking directions for each team. Unreviewed time stays out of movement stats.":
    "Rerun tracking to add pitch continuity and position uncertainty to this older report.");
  pitchReview=JSON.parse(JSON.stringify(tracking.pitch_review||{segments:{},directions:{}}));
  const selected=$("geometry-segment").value;
  $("geometry-segment").replaceChildren();
  for(const s of tracking.pitch_segments||[])
    option($("geometry-segment"),s.id,clock(s.start_ms)+"–"+clock(s.end_ms));
  if(tracking.pitch_segments?.some(x=>x.id===selected))$("geometry-segment").value=selected;
  else $("geometry-segment").value=tracking.pitch_segments?.[0]?.id||"";
  const selectedTeam=$("geometry-team").value;
  const teams=new Set();
  for(const frame of tracking.frames)for(const o of frame.objects)
    if(o.object_type==="player"&&!probableReferee(o)&&(o.team_id||o.inferred_team))teams.add(o.team_id||o.inferred_team);
  $("geometry-team").replaceChildren();
  for(const team of teams)option($("geometry-team"),team,geometryTeamName(team));
  $("geometry-team").value=teams.has(selectedTeam)?selectedTeam:[...teams][0]||"";
  geometrySettings();geometryDirections();renderSpatial();
}
function showGeometryFrame(frame){
  const g=frame.pitch_geometry;
  if(!g){status("geometry-current","");return;}
  const statusName={rejected_jump:"Pitch estimate shown; movement stats withheld for mapping instability",mapping_unavailable:"No trusted pitch mapping",
    camera_checked:"Camera movement checked",new_segment:"New camera segment",insufficient_overlap:"New segment: limited overlap"};
  const errors=frame.objects.filter(o=>o.observed&&finite(o.position_uncertainty_m)).map(o=>o.position_uncertainty_m);
  status("geometry-current",[
    statusName[g.continuity_status]||"Pitch mapping estimate",
    g.reviewed_orientation==="normal"||g.reviewed_orientation==="flipped"?"Orientation confirmed":"Confirm orientation for movement stats",
    g.mirror_corrected?"Pitch flip corrected":"",
    errors.length?"Position sensitivity up to "+fmt(Math.max(...errors))+" m":"Position uncertainty unavailable",
  ].filter(Boolean).join(" · "));
}
function renderSpatial(){
  const players=tracking?.spatial_summary?.players||[];
  $("spatial-results").hidden=!players.length;
  const select=$("spatial-player"), prior=select.value;
  select.replaceChildren();
  for(const p of players)option(select,p.track_id,(p.player_id||"Track #"+(playerNumbers.get(p.track_id)||"?"))+" · "+
    geometryTeamName(p.team));
  select.value=players.some(p=>p.track_id===prior)?prior:players[0]?.track_id||"";
  renderSpatialPlayer();
}
function renderSpatialPlayer(){
  const p=tracking?.spatial_summary?.players.find(x=>x.track_id===$("spatial-player").value);
  $("zone-grid").replaceChildren();$("spatial-cards").replaceChildren();
  if(!p)return;
  $("spatial-coverage").textContent=fmt(p.coverage_pct)+"% usable coverage · "+fmt(p.usable_ms/1000)+
    " s usable / "+fmt(p.observed_interval_ms/1000)+" s contiguous observed track time. This is not full-match coverage.";
  for(let i=0;i<9;i++){
    const cell=node("div");
    cell.append(node("small",null,["Top","Middle","Bottom"][Math.floor(i/3)]+" · "+["Left","Centre","Right"][i%3]),
      node("strong",null,finite(p.zones_pct[i])?fmt(p.zones_pct[i])+"%":"—"));
    $("zone-grid").append(cell);
  }
  for(const [label,value] of [["Forward displacement",p.forward_m],["Backward displacement",p.backward_m],["Lateral displacement",p.lateral_m]]){
    const card=node("article","metric-card");
    card.append(node("h3",null,label),node("p","metric-value",finite(value)?fmt(value)+" m":"Not available"));
    $("spatial-cards").append(card);
  }
  $("spatial-note").textContent="Uncertain zone time: "+fmt(p.boundary_uncertain_ms/1000)+" s. Direction-reviewed time: "+
    fmt(p.directional_usable_ms/1000)+" s; movement below the error estimate: "+fmt(p.direction_uncertain_ms/1000)+
    " s. Displacements are conservative observed movement, not total distance; zero means no displacement resolved above the error estimate. "+
    (tracking.dimensions_assumed?"Pitch dimensions are assumed. ":"")+"Uncertainty is a geometric estimate, not a confidence interval.";
}
$("geometry-segment").onchange=()=>{
  geometrySettings();
  seek(tracking.pitch_segments.find(x=>x.id===$("geometry-segment").value)?.start_ms);
};
$("geometry-team").onchange=geometryDirections;
$("spatial-player").onchange=renderSpatialPlayer;
function saveGeometryDraft(){
  const id=$("geometry-segment").value,team=$("geometry-team").value;
  if(id)pitchReview.segments[id]={orientation:$("geometry-orientation").value,
    period:Number($("geometry-period").value),live_play:$("geometry-live").checked};
  if(team)for(const [period,field] of [[1,"direction-first"],[2,"direction-second"],[3,"direction-third"],[4,"direction-fourth"]]){
    pitchReview.directions[String(period)]??={};
    if($(field).value)pitchReview.directions[String(period)][team]=Number($(field).value);
    else delete pitchReview.directions[String(period)][team];
  }
  $("geometry-export").disabled=true;
  status("geometry-status","Changes pending. Apply review to update the replay and movement results.");
}
for(const id of ["geometry-orientation","geometry-period","geometry-live","direction-first","direction-second","direction-third","direction-fourth"])
  $(id).onchange=saveGeometryDraft;
$("geometry-apply").onclick=async()=>{
  if(!tracking)return;
  const source=tracking, request=++geometryRequest, ms=Math.round(video.currentTime*1000||replayMs);
  const identities=new Map(reviewedIdentities);
  if(Number.isFinite(video.duration))video.pause();
  const review=JSON.parse(JSON.stringify(pitchReview));
  review.segments[$("geometry-segment").value]={orientation:$("geometry-orientation").value,
    period:Number($("geometry-period").value),live_play:$("geometry-live").checked};
  const team=$("geometry-team").value;
  if(team)for(const [period,id] of [[1,"direction-first"],[2,"direction-second"],[3,"direction-third"],[4,"direction-fourth"]]){
    review.directions[String(period)]??={};
    if($(id).value)review.directions[String(period)][team]=Number($(id).value);
    else delete review.directions[String(period)][team];
  }
  $("geometry-apply").disabled=true;
  status("geometry-status","Checking reviewed intervals and calculating movement…");
  try{
    const result=await api("/api/geometry",{method:"POST",headers:{...trackHeaders,"Content-Type":"application/json"},
      body:JSON.stringify({tracking:source,review})}).then(r=>r.json());
    if(request!==geometryRequest||tracking!==source)return;
    setupTracking(parseTracking(result),$("tracking-name").textContent);
    reviewedIdentities=identities;setupIdentityRoster();
    exportedPreview=result;updateAnalysisControls();seek(ms);
    status("geometry-status","Review applied. Zones and movement use only trusted, reviewed intervals. Export to keep this review.");
  }catch(err){if(request===geometryRequest)status("geometry-status",err.message+" Open the local app if it is not running.",true);}
  finally{if(tracking===source)$("geometry-apply").disabled=false;}
};
$("geometry-export").onclick=()=>{if(tracking)download("video-tracking-reviewed.json",tracking);};
