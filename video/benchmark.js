"use strict";

let benchmarkResult, benchmarkLabels, benchmarkComparison, benchmarkFile, benchmarkLabelName, automaticReference;
let benchmarkPage=0, benchmarkLabelRequest=0, matchActionPage=0;
const benchmarkPageSize=20;

function benchmarkTime(ms){
  return clock(ms).slice(0,-1)+String(ms%1000).padStart(3,"0");
}

function comparePasses(predictions, truth, tolerance){
  // Pair the earliest compatible timestamps to maximize one-to-one matches.
  const expected=truth.events.filter(row=>row.type==="pass").slice().sort((a,b)=>a.t_ms-b.t_ms);
  const detected=predictions.filter(row=>truth.start_ms<=row.t_ms&&row.t_ms<truth.end_ms)
    .slice().sort((a,b)=>a.t_ms-b.t_ms);
  const moments=[];
  let i=0, j=0, matched=0, missed=0, extra=0;
  const fields=["passer_player_id","receiver_player_id","outcome"];
  const scores=Object.fromEntries(fields.map(field=>[field,{labeled:0,resolved:0,correct:0}]));
  const known=value=>Boolean(value)&&value!=="unknown";
  while(i<expected.length||j<detected.length){
    if(j===detected.length||i<expected.length&&expected[i].t_ms<detected[j].t_ms-tolerance){
      const reference=expected[i++];
      moments.push({status:"missed",reference_ms:reference.t_ms,reference});
      missed++;
    }else if(i===expected.length||detected[j].t_ms<expected[i].t_ms-tolerance){
      moments.push({status:"extra",detection:detected[j++]});
      extra++;
    }else{
      const detection=detected[j++], reference=expected[i++], reference_ms=reference.t_ms;
      moments.push({status:"matched",reference_ms,reference,detection,offset_ms:detection.t_ms-reference_ms});
      for(const field of fields){
        if(!known(reference[field]))continue;
        scores[field].labeled++;
        if(!known(detection[field]))continue;
        scores[field].resolved++;
        if(reference[field]===detection[field])scores[field].correct++;
      }
      matched++;
    }
  }
  return {
    start_ms:truth.start_ms,end_ms:truth.end_ms,tolerance_ms:tolerance,
    reference_passes:expected.length,detections:detected.length,matched,missed,extra,
    precision:detected.length?matched/detected.length:null,
    recall:expected.length?matched/expected.length:null,
    f1:expected.length+detected.length?2*matched/(expected.length+detected.length):null,
    outcome_coverage:detected.length?detected.filter(row=>known(row.outcome)).length/detected.length:null,
    attribution:scores,moments
  };
}

function readBenchmarkLabels(data,result){
  const validTime=value=>Number.isSafeInteger(value)&&value>=0;
  if(!data||!validTime(data.start_ms)||!validTime(data.end_ms)||data.end_ms<=data.start_ms||
      data.end_ms>result.duration_ms||!Array.isArray(data.events)){
    throw Error("The JSON needs start_ms, end_ms and an events array within the analyzed video.");
  }
  if(data.events.some(row=>!row||typeof row.type!=="string"||!row.type.trim()||
      !validTime(row.t_ms)||row.t_ms<data.start_ms||row.t_ms>=data.end_ms)){
    throw Error("Each event needs a type and an integer t_ms inside the labeled interval.");
  }
  if(data.source_sha256!==undefined&&data.source_sha256!==result.source_sha256){
    throw Error("These labels identify a different video file.");
  }
  return data;
}

function renderBenchmarkMoments(){
  const filter=$("benchmark-filter").value;
  const moments=benchmarkComparison?.moments||benchmarkResult?.events.map(detection=>({status:detection.status==="detected"?"detected":"candidate",detection}))||[];
  const rows=moments.filter(row=>!benchmarkComparison||filter==="all"||row.status===filter);
  const pageCount=Math.max(1,Math.ceil(rows.length/benchmarkPageSize));
  const target=$("benchmark-moments");
  target.replaceChildren();
  $("benchmark-pages").hidden=pageCount<=1;
  $("benchmark-prev").disabled=benchmarkPage===0;
  $("benchmark-next").disabled=benchmarkPage+1>=pageCount;
  $("benchmark-page").textContent="Page "+(benchmarkPage+1)+" of "+pageCount;
  if(!rows.length&&benchmarkResult)target.append(node("p","hint","No moments in this view."));
  for(const row of rows.slice(benchmarkPage*benchmarkPageSize,(benchmarkPage+1)*benchmarkPageSize)){
    const item=node("article","statistics-candidate benchmark-moment");
    const titles={matched:"Matched pass",missed:"Unmatched reference label",extra:"Unmatched detection",candidate:"Pass candidate",detected:"Detected pass"};
    item.append(node("strong",null,titles[row.status]));
    const times=node("div","row");
    for(const [label,time] of [["Ground truth",row.reference_ms],["FBRPRO",row.detection?.t_ms],["Next control",row.detection?.arrival_ms]]){
      if(time===undefined)continue;
      const button=node("button","secondary",label+" · "+benchmarkTime(time));
      button.type="button";
      button.onclick=()=>{
        if(activeVideoFile!==benchmarkFile)return;
        video.pause();
        playbackEnd=undefined;
        seek(time);
        $("video-stage").scrollIntoView({block:"center"});
      };
      times.append(button);
    }
    if(row.offset_ms!==undefined){
      const sign=row.offset_ms>0?"+":"";
      times.append(node("span","hint",sign+row.offset_ms+" ms"));
    }
    item.append(times);
    if(finite(row.detection?.pass_model_score))
      item.append(node("p","hint","Pass model score "+fmt(row.detection.pass_model_score*100)+"%"));
    const expected=row.reference?.outcome, detected=row.detection?.outcome;
    const actor=row.detection?.passer_player_id||
      (row.detection?.track_id&&statisticsPlayerName({track_id:row.detection.track_id}));
    if(actor)item.append(node("p","hint","Passer · "+actor));
    const player=row.detection?.receiver_player_id||
      (row.detection?.receiver_track_id&&statisticsPlayerName({track_id:row.detection.receiver_track_id}));
    if(player)item.append(node("p","hint","Receiver · "+player));
    if(expected||detected)item.append(node("small","hint",
      [expected&&"Ground truth "+expected,detected&&"FBRPRO "+detected].filter(Boolean).join(" · ")));
    target.append(item);
  }
}

function renderBenchmark(){
  const output=$("benchmark-summary");
  output.replaceChildren();
  $("benchmark-export-comparison").disabled=!benchmarkComparison;
  $("benchmark-review").hidden=!benchmarkResult;
  $("benchmark-filter").parentElement?.toggleAttribute("hidden",!benchmarkComparison);
  if(!benchmarkResult)return;
  const result=benchmarkComparison;
  const percent=value=>value===null?"—":fmt(value*100)+"%";
  const metrics=result?[
    ["Ground-truth passes",result.reference_passes],["FBRPRO detections",result.detections],
    ["Matched",result.matched],["Unmatched labels",result.missed],["Unmatched detections",result.extra],
    ["Label precision",percent(result.precision)],["Label recall",percent(result.recall)],["Label F1",percent(result.f1)],
    ["Outcome coverage",percent(result.outcome_coverage)]
  ]:[[benchmarkResult.pass_detector==="trained"?"Detected passes":"FBRPRO pass candidates",benchmarkResult.events.length],
    ...Object.entries(benchmarkResult.pass_outcomes||{}).filter(([key])=>["complete","incomplete","unknown"].includes(key))
      .map(([key,count])=>[key[0].toUpperCase()+key.slice(1),count])];
  if(result){
    for(const [field,label] of [["passer_player_id","Passer"],["receiver_player_id","Receiver"],["outcome","Outcome"]]){
      const score=result.attribution[field];
      if(score.labeled)metrics.push([label+" correct / resolved",score.correct+" / "+score.resolved],
        [label+" resolved / labeled",score.resolved+" / "+score.labeled]);
    }
  }
  for(const [label,value] of metrics){
    const card=node("div","benchmark-metric");
    card.append(node("span",null,label),node("strong",null,String(value)));
    output.append(card);
  }
  renderBenchmarkMoments();
}

function updatePassComparison(){
  benchmarkComparison=undefined;
  benchmarkPage=0;
  const input=$("benchmark-tolerance");
  if(!benchmarkLabels){
    status("benchmark-comparison-status","");
  }else if(!input.value||!input.validity.valid){
    status("benchmark-comparison-status","Enter a whole number from 0 to 5000 ms.",true);
  }else{
    benchmarkComparison=comparePasses(benchmarkResult.events,benchmarkLabels,Number(input.value));
    status("benchmark-comparison-status","Compared "+
      benchmarkTime(benchmarkComparison.start_ms)+" to "+benchmarkTime(benchmarkComparison.end_ms)+
      " · "+benchmarkLabelName);
  }
  renderBenchmark();
}

const possessionLabels={
  player_control:"Player control",in_transit:"Ball in transit",contested:"Contested",
  dead_ball:"Dead ball",unobservable:"Unobservable"
};
const contactLabels={
  controlled_touch:"Controlled touch",one_touch:"One-touch release",
  header_possible:"Possible header",deflection_possible:"Possible deflection"
};

function reviewedMatchActions(){
  if(benchmarkResult?.video_id!==tracking?.video_id)return [];
  return (statisticsContext.events||[]).filter(row=>row.reviewed===true);
}

function renderMatchActions(){
  const target=$("match-actions-rows");
  target.replaceChildren();
  if(!benchmarkResult)return;
  const view=$("match-actions-view").value||"automatic", filter=$("match-actions-filter").value;
  const possession=benchmarkResult.possession;
  $("match-actions-filter-label").hidden=view!=="automatic";
  const items=view==="possession"?possession?.intervals||[]:view==="reviewed"?reviewedMatchActions():
    benchmarkResult.automatic_events.filter(row=>!filter||row.type===filter);
  const total=Math.max(1,Math.ceil(items.length/benchmarkPageSize));
  matchActionPage=Math.min(matchActionPage,total-1);
  $("match-actions-pages").hidden=total<=1;
  $("match-actions-prev").disabled=matchActionPage===0;
  $("match-actions-next").disabled=matchActionPage+1>=total;
  $("match-actions-page").textContent="Page "+(matchActionPage+1)+" of "+total;
  if(!items.length)target.append(node("p","hint",view==="reviewed"?"No reviewed actions saved for this video.":"No moments in this view."));
  for(const row of items.slice(matchActionPage*benchmarkPageSize,(matchActionPage+1)*benchmarkPageSize)){
    const stamp=row.t_ms??row.start_ms;
    const item=node("article","statistics-candidate");
    const player=row.player_id||row.passer_player_id||(row.track_id&&statisticsPlayerName({track_id:row.track_id}));
    const title=possessionLabels[row.state]||contactLabels[row.contact_kind]||row.label||statisticsLabel(row.type);
    const carrier=row.carrier_player_id||(row.carrier_track_id&&statisticsPlayerName({track_id:row.carrier_track_id}));
    const detail=[view==="reviewed"?"Reviewed":null,player||(row.type?"Player unresolved":null),
      carrier&&"Ball carrier · "+carrier,
      row.next_control_team_id&&"Next control · "+statisticsTeamName(row.next_control_team_id),
      row.outcome&&statisticsLabel(row.outcome),row.actor_reason,row.outcome_reason].filter(Boolean).join(" · ");
    const time=clock(stamp)+(row.end_ms!==undefined?"–"+clock(row.end_ms):"");
    const button=node("button","secondary",time+" · "+title);
    button.type="button";
    button.onclick=()=>{
      if(activeVideoFile!==benchmarkFile||benchmarkResult.video_id!==tracking?.video_id)return;
      video.pause();
      playbackEnd=undefined;
      seek(Math.round(stamp));
      $("video-stage").scrollIntoView({block:"center"});
    };
    item.append(button);
    if(finite(row.regain_ms)){
      const regain=node("button","secondary","Regain · "+clock(row.regain_ms));
      regain.type="button";
      regain.onclick=()=>{if(activeVideoFile===benchmarkFile){video.pause();seek(Math.round(row.regain_ms));$("video-stage").scrollIntoView({block:"center"});}};
      item.append(regain);
    }
    if(detail)item.append(node("p","hint",detail));
    target.append(item);
  }
}

function showMatchActions(){
  const summary=$("match-actions-summary"), control=$("match-actions-control");
  summary.replaceChildren();control.replaceChildren();
  const possession=benchmarkResult.possession;
  const entries=[["Automatic actions",benchmarkResult.automatic_events.length]];
  if(possession){
    if(possession.scope_ms)entries.push(["Observable ball state",fmt(possession.observable_ms/possession.scope_ms*100)+"%"]);
    for(const [key,label] of Object.entries(possessionLabels))entries.push([label,secs(possession.states_ms[key])]);
    for(const team of possession.teams)control.append(node("p",null,statisticsTeamName(team.team_id)+
      " · "+secs(team.control_ms)+" · "+fmt(team.share_of_team_control*100)+"% of known team control"));
    if(!possession.teams.length)control.append(node("p","hint","No team control could be assigned."));
    if(possession.players.length){
      const label=node("label",null,"Player");const select=node("select");
      const duration=node("p","status");
      for(const [index,player] of possession.players.entries())
        option(select,String(index),player.player_id||statisticsPlayerName({track_id:player.track_ids[0]}));
      const update=()=>{duration.textContent="Observed control · "+secs(possession.players[Number(select.value||0)].control_ms);};
      select.onchange=update;
      update();
      label.append(select);
      control.append(label,duration);
    }
  }
  for(const [label,value] of entries){
    const card=node("div","benchmark-metric");
    card.append(node("span",null,label),node("strong",null,String(value)));
    summary.append(card);
  }
  $("match-actions").hidden=false;
  renderMatchActions();
}

for(const id of ["match-actions-view","match-actions-filter"])$(id).onchange=()=>{matchActionPage=0;renderMatchActions();};
$("match-actions-prev").onclick=()=>{matchActionPage--;renderMatchActions();};
$("match-actions-next").onclick=()=>{matchActionPage++;renderMatchActions();};
$("match-actions-export").onclick=()=>{
  if(benchmarkResult)download("fbrpro-match-actions.json",{
    version:"fbrpro-match-actions-v1",video_id:benchmarkResult.video_id,source_sha256:benchmarkResult.source_sha256,
    video_name:benchmarkResult.video_name,scope_spans:benchmarkResult.scope_spans,
    events:benchmarkResult.automatic_events,reviewed_events:reviewedMatchActions(),possession:benchmarkResult.possession
  });
};

function resetPassBenchmark(){
  benchmarkResult=benchmarkLabels=benchmarkComparison=benchmarkFile=benchmarkLabelName=automaticReference=undefined;
  benchmarkLabelRequest++;
  matchActionPage=0;
  $("match-actions").hidden=true;
  $("match-actions-view").value="automatic";
  $("match-actions-filter").value="";
  $("pass-benchmark").hidden=true;
  $("benchmark-controls").hidden=true;
  $("benchmark-labels").value="";
  $("benchmark-filter").value="all";
  status("benchmark-comparison-status","");
}

$("benchmark-analyze").onclick=()=>{
  if(analysisRequest||!activeVideoFile)return;
  resetPassBenchmark();
  return trackFromPage(true,(report,file,result)=>{
    benchmarkFile=file;
    benchmarkResult={
      version:"fbrpro-pass-test-v1",video_name:file.name,video_id:report.video_id,
      source_sha256:report.source_sha256,sample_fps:report.sample_fps,
      duration_ms:report.preview_plan.selected_video_ms,pass_outcomes:result.pass_outcomes,
      pass_detector:result.pass_detector,
      possession:result.possession,scope_spans:result.scope_spans,
      automatic_events:[...result.events,...(result.ball_actions||[])].sort((a,b)=>a.t_ms-b.t_ms),
      receptions:result.events.filter(event=>event.type==="reception"),
      events:result.events.filter(event=>event.type==="pass"&&event.source==="tracking")
    };
    $("benchmark-video").textContent=file.name;
    $("pass-benchmark").hidden=false;
    $("benchmark-controls").hidden=false;
    const detector=result.pass_detector==="trained"?"Trained pass model":"Pass analysis";
    try{
      if(result.ground_truth){
        const data=readBenchmarkLabels(result.ground_truth.data,benchmarkResult);
        automaticReference={data,filename:result.ground_truth.filename};
      }
    }catch(error){
      status("benchmark-status","Could not use the matching local labels. "+error.message,true);
    }
    if(automaticReference){
      benchmarkLabels=automaticReference.data;
      benchmarkLabelName=automaticReference.filename;
      status("benchmark-status",detector+" finished. Matching ground-truth labels loaded automatically.");
    }else if(!result.ground_truth){
      status("benchmark-status",detector+" finished. Open ground-truth labels to compare pass timing.");
    }
    analysisStatus("Match action analysis complete.");
    showMatchActions();
    updatePassComparison();
    $("match-actions").scrollIntoView({block:"start"});
  });
};
$("benchmark-labels").onchange=async event=>{
  const file=event.target.files[0], result=benchmarkResult, request=++benchmarkLabelRequest;
  benchmarkLabels=automaticReference?.data;
  benchmarkLabelName=automaticReference?.filename;
  updatePassComparison();
  if(!file||!result)return;
  try{
    if(file.size>20*1024*1024)throw Error("Choose a ground-truth JSON file under 20 MB.");
    const data=JSON.parse(await file.text());
    if(request!==benchmarkLabelRequest||result!==benchmarkResult||benchmarkFile!==activeVideoFile)return;
    benchmarkLabels=readBenchmarkLabels(data,result);
    benchmarkLabelName=file.name;
    updatePassComparison();
  }catch(error){
    if(request===benchmarkLabelRequest&&result===benchmarkResult){
      status("benchmark-comparison-status","Could not open labels. "+error.message,true);
    }
  }
};
$("benchmark-tolerance").oninput=updatePassComparison;
$("benchmark-filter").onchange=()=>{benchmarkPage=0;renderBenchmarkMoments();};
$("benchmark-prev").onclick=()=>{benchmarkPage--;renderBenchmarkMoments();};
$("benchmark-next").onclick=()=>{benchmarkPage++;renderBenchmarkMoments();};
$("benchmark-export-results").onclick=()=>{
  if(benchmarkResult)download("fbrpro-pass-results.json",benchmarkResult);
};
$("benchmark-export-comparison").onclick=()=>{
  if(benchmarkComparison)download("fbrpro-pass-comparison.json",{
    version:benchmarkResult.version,video_name:benchmarkResult.video_name,
    video_id:benchmarkResult.video_id,source_sha256:benchmarkResult.source_sha256,
    sample_fps:benchmarkResult.sample_fps,labels_file:benchmarkLabelName,...benchmarkComparison
  });
};
for(const event of ["emptied","loadedmetadata"])video.addEventListener(event,()=>{
  if(benchmarkFile&&activeVideoFile!==benchmarkFile)resetPassBenchmark();
});
