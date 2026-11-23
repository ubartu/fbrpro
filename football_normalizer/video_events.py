"""Align SoccerNet spotting predictions and estimate pass releases from tracking."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import hypot
from pathlib import Path

from .video_ball_actions import analyze_ball_actions, closest_contact as _closest_contact
from .video_ball_actions import redirects_ball as _redirects_ball, same_ball as _same_ball
from .video_stats_common import finite, prepare


VERSION = "fbrpro-video-events-v1"
PASS_LABELS = {"pass", "high pass", "cross"}
ATTRIBUTION_WINDOW_MS = 400
ATTRIBUTION_SEPARATION_MS = 100
MAX_RECEPTION_MS = 4000


def _number(value):
    if type(value) not in (int, float, str):
        raise ValueError("SoccerNet timestamps and confidence must be numeric")
    value = float(value)
    if not finite(value):
        raise ValueError("SoccerNet timestamps and confidence must be finite")
    return value


def align_predictions(document, alignment):
    """Use half-relative video positions and an explicit shared video anchor."""
    if not isinstance(document, dict) or not isinstance(document.get("predictions"), list):
        raise ValueError("Choose a SoccerNet results_spotting.json predictions file")
    if len(document["predictions"]) > 100_000:
        raise ValueError("Choose a SoccerNet prediction file for one match")
    if not isinstance(alignment, dict) or type(alignment.get("half")) is not int or alignment["half"] not in (1, 2):
        raise ValueError("Choose the prediction half to align")
    half = alignment["half"]
    source, target = (alignment.get(key) for key in ("source_ms", "video_ms"))
    if any(type(value) is not int or value < 0 for value in (source, target)):
        raise ValueError("Set the same moment in both videos using nonnegative timestamps")

    events = {}
    for row in document["predictions"]:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or not row["label"].strip():
            raise ValueError("Each prediction needs an action label")
        clock = row.get("gameTime")
        clock_half = clock.split(" - ", 1)[0] if isinstance(clock, str) and " - " in clock else None
        period = _number(row.get("half", clock_half))
        if period not in (1, 2) or clock_half is not None and _number(clock_half) != period:
            raise ValueError("SoccerNet prediction half and gameTime must agree")
        position, score = (_number(row.get(key)) for key in ("position", "confidence"))
        if position < 0 or not 0 <= score <= 1:
            raise ValueError("SoccerNet predictions need nonnegative positions and confidence from 0 to 1")
        if period != half:
            continue
        label = row["label"].strip()
        t_ms = round(position) - source + target
        action = label.lower().replace("-", " ")
        kind = "pass" if action in PASS_LABELS else action.replace(" ", "_")
        key = (t_ms, kind)
        if key not in events or score > events[key]["action_score"]:
            events[key] = {"source": "soccernet", "label": label,
                           "type": kind,
                           "source_ms": round(position), "t_ms": t_ms, "period": half, "action_score": score}
    return sorted(events.values(), key=lambda row: row["t_ms"])


def _box(row):
    box = row.get("bbox_xyxy")
    if (row.get("observed") is True and row.get("estimated") is not True and row.get("visibility") != "missing"
            and isinstance(box, (list, tuple)) and len(box) == 4 and all(finite(v) for v in box)
            and box[0] < box[2] and box[1] < box[3]):
        return box


@dataclass
class Sample:
    t_ms: int
    period: int
    shot: tuple
    ball_id: str
    ball: tuple
    players: dict
    teams: dict

    def ball_offset(self, track_id):
        box = self.players.get(track_id)
        if box is None:
            return None
        l, t, r, b = box
        return (self.ball[0]-(l+r)/2)/(b-t), (self.ball[1]-b)/(b-t)

    def foot_distance(self, track_id):
        offset = self.ball_offset(track_id)
        return hypot(*offset) if offset is not None else None


def _observations(tracking, prepared):
    raw = {frame["video_time_ms"]: frame for frame in tracking["frames"]}
    duplicate_times = {stamp for stamp, count in Counter(f["video_time_ms"] for f in tracking["frames"]).items() if count > 1}
    team_evidence = defaultdict(set)
    for frame in tracking["frames"]:
        for row in frame["objects"]:
            team = row.get("team_id") or row.get("inferred_team")
            if team and row.get("observed") is True and row.get("object_type") == "player":
                team_evidence[row["track_id"]].add(team)
    stable_teams = {track: next(iter(teams)) for track, teams in team_evidence.items() if len(teams) == 1}
    identities = prepared["context"].get("identities", {})
    excluded_epoch = 0
    dead = prepared["context"].get("dead_ball_intervals", [])
    for frame in prepared["frames"]:
        stamp = frame["t_ms"]
        source = raw[stamp]
        if (frame["live"] is False or stamp in duplicate_times or
                any(row.get("period", 1) == frame["period"] and row["start_ms"] <= stamp < row["end_ms"] for row in dead)):
            excluded_epoch += 1
            yield None
            continue
        balls, players, teams = [], {}, {}
        ids = Counter(o["track_id"] for o in source["objects"] if o.get("observed") is True)
        for row in source["objects"]:
            box = _box(row)
            if box is None or ids[row["track_id"]] != 1:
                continue
            if row.get("object_type") == "ball":
                balls.append((row["track_id"], ((box[0]+box[2])/2, (box[1]+box[3])/2)))
            elif (row.get("object_type") == "player" and row.get("on_pitch") is not False
                  and row.get("inferred_role") != "probable_referee" and row["track_id"] in prepared["tracks"]):
                players[row["track_id"]] = box
                track = row["track_id"]
                teams[track] = (identities.get(track, {}).get("team_id") or row.get("team_id")
                                or row.get("inferred_team") or stable_teams.get(track))
        if len(balls) == 1:
            ball_id, center = balls[0]
            yield Sample(stamp, frame["period"], (frame["segment_id"], source.get("segment_id"), excluded_epoch), ball_id, center, players, teams)
        else:
            yield None


def _release_candidates(samples, gap_ms):
    """Require sustained foot proximity followed by sustained separation."""
    releases = []
    for i in range(1, len(samples)-2):
        window = samples[i-1:i+3]
        if any(s is None for s in window):
            continue
        before, contact, leave, after = window
        if any((s.shot, s.period, s.ball_id) != (contact.shot, contact.period, contact.ball_id) for s in window):
            continue
        if any(not 0 < b.t_ms-a.t_ms <= gap_ms for a, b in zip(window, window[1:])):
            continue
        nearby = []
        for track_id in contact.players:
            at = contact.foot_distance(track_id)
            prior = before.foot_distance(track_id)
            if at <= .42 and prior is not None and prior <= .42:
                nearby.append(((at+prior)/2, track_id))
        if not nearby:
            continue
        nearby.sort()
        proximity, track_id = nearby[0]
        if len({s.teams[track_id] for s in window if s.teams.get(track_id)}) > 1:
            continue
        first, last = leave.foot_distance(track_id), after.foot_distance(track_id)
        if first is None or last is None or first-proximity < .20 or last-proximity < .60 or last < first-.10:
            continue
        margin = nearby[1][0]-proximity if len(nearby) > 1 else None
        releases.append({"t_ms": round((contact.t_ms+leave.t_ms)/2), "period": contact.period,
                         "actor_sample_ms": contact.t_ms, "track_id": track_id if margin is None or margin >= .15 else None,
                         "evidence_start_ms": before.t_ms, "evidence_end_ms": after.t_ms,
                         "actor_reason": "Ball moved away after two close samples" if margin is None or margin >= .15
                         else "Multiple players were close to the ball before release"})
    return releases


def _transfer_candidates(samples, gap_ms):
    """Look for the ball moving between distinct teammates' contact areas."""
    observed = [s for s in samples if s is not None]
    contacts = [_closest_contact(s) for s in observed]
    transfers = []
    for i in range(2, len(observed)-1):
        receiver = contacts[i]
        arrival = observed[i]
        before_arrival, after_arrival = observed[i-1], observed[i+1]
        if (not receiver or contacts[i-1] == receiver or not _same_ball(arrival, after_arrival, gap_ms)):
            continue
        if contacts[i+1] != receiver and not (_same_ball(before_arrival, arrival, gap_ms)
                                              and _redirects_ball(before_arrival, arrival, after_arrival, receiver)):
            continue
        j = i-1
        # A brief missing ball detection can occur exactly at the kick.
        while (j > 0 and contacts[j] is None and arrival.t_ms-observed[j].t_ms <= 1000
               and _same_ball(observed[j], observed[j+1], min(300, 2*gap_ms))):
            j -= 1
        passer = contacts[j]
        contact, departure = observed[j:j+2]
        if (j == 0 or not passer or passer == receiver
                or arrival.t_ms-contact.t_ms > 1000 or not _same_ball(observed[j-1], contact, gap_ms)
                or not _same_ball(contact, departure, min(300, 2*gap_ms))):
            continue
        if contacts[j-1] != passer and not _redirects_ball(observed[j-1], contact, departure, passer):
            continue
        team = contact.teams.get(passer)
        if not team or arrival.teams.get(receiver) != team:
            continue
        if any(s.teams.get(p) not in (None, team) for s in (observed[j-1], contact, departure, arrival, observed[i+1])
               for p in (passer, receiver)):
            continue
        from_receiver = contact.foot_distance(receiver)
        to_passer = arrival.foot_distance(passer)
        if (from_receiver is None or to_passer is None or from_receiver-arrival.foot_distance(receiver) < .25
                or to_passer-contact.foot_distance(passer) < .25):
            continue
        transfers.append({"t_ms": round((contact.t_ms+departure.t_ms)/2), "period": contact.period,
                          "actor_sample_ms": contact.t_ms, "track_id": passer, "recipient_candidate_track_id": receiver,
                          "evidence_start_ms": observed[j-1].t_ms, "evidence_end_ms": observed[i+1].t_ms,
                          "actor_reason": "Ball moved from one teammate’s feet toward another"})
    return transfers


def _controlled_arrival(samples, index, gap_ms):
    before, arrival, after = samples[index-1:index+2]
    if not all((before, arrival, after)) or not (
            _same_ball(before, arrival, gap_ms) and _same_ball(arrival, after, gap_ms)):
        return None
    player = _closest_contact(arrival)
    if player and _closest_contact(before) != player and (
            _closest_contact(after) == player or _redirects_ball(before, arrival, after, player)):
        return player


def _pass_outcome(release, samples, index, gap_ms, tracks):
    start = samples[index]
    passer = release.get("track_id")
    result = {"outcome": "unknown", "outcome_reason": "Passer could not be resolved"}
    if not passer:
        return result
    team = start.teams.get(passer)
    result["outcome_reason"] = "No controlled arrival observed after release"
    for i in range(index+1, len(samples)-1):
        arrival = samples[i]
        previous = samples[i-1]
        if arrival is None or not _same_ball(previous, arrival, gap_ms):
            result["outcome_reason"] = "Ball evidence ends at a gap, camera cut or track change"
            break
        if arrival.t_ms-start.t_ms > MAX_RECEPTION_MS:
            break
        player = _controlled_arrival(samples, i, gap_ms)
        if not player:
            continue
        player_id = tracks[player].get("player_id")
        passer_id = tracks[passer].get("player_id")
        if player == passer or (player_id and player_id == passer_id):
            result["outcome_reason"] = "The ball returned to the same player"
            break
        arrival_team = arrival.teams.get(player)
        after_team = samples[i+1].teams.get(player)
        if (not team or not arrival_team or team.startswith("kit:") != arrival_team.startswith("kit:")
                or after_team not in (None, arrival_team)
                or arrival.teams.get(passer) not in (None, team)):
            result["outcome_reason"] = "Teams could not be resolved at the first controlled arrival"
            break
        completed = arrival_team == team
        result.update(outcome="complete" if completed else "incomplete",
                      outcome_reason="Teammate controlled the ball" if completed else "Opponent controlled the ball",
                      arrival_ms=arrival.t_ms, next_control_track_id=player, next_control_player_id=player_id)
        if completed:
            result.update(receiver_track_id=player, receiver_player_id=player_id)
        break
    return result


def _attribute(prediction, releases, stamps, shot_boundaries, same_period):
    stamp = prediction["t_ms"]
    start = bisect_left(stamps, stamp-ATTRIBUTION_WINDOW_MS)
    end = bisect_right(stamps, stamp+ATTRIBUTION_WINDOW_MS)
    shot = bisect_right(shot_boundaries, stamp)
    matches = [row for row in releases[start:end] if (not same_period or row["period"] == prediction["period"])
               and bisect_right(shot_boundaries, row["actor_sample_ms"]) == shot]
    if not matches:
        return {"track_id": None, "actor_reason": "No continuous player and ball evidence at release"}
    best = min(matches, key=lambda row: abs(row["t_ms"]-stamp))
    closest = abs(best["t_ms"]-stamp)
    if best["track_id"] is None or any(row["track_id"] != best["track_id"]
                                       and abs(row["t_ms"]-stamp)-closest < ATTRIBUTION_SEPARATION_MS for row in matches):
        return {"track_id": None, "actor_reason": "Competing release evidence"}
    return {"release_ms": best["t_ms"], **{key: value for key, value in best.items() if key not in ("t_ms", "period")}}


def _shot_boundaries(tracking, gap_ms):
    boundaries = set()
    previous = None
    for frame in sorted(tracking["frames"], key=lambda row: row["video_time_ms"]):
        stamp = frame["video_time_ms"]
        if frame.get("camera_cut_suspected") or previous and (
                stamp-previous["video_time_ms"] > gap_ms or any(frame.get(key) != previous.get(key)
                                                               for key in ("clip_id", "segment_id", "period"))):
            boundaries.add(stamp)
        previous = frame
    return sorted(boundaries)


def analyze(tracking, context=None, predictions=None, alignment=None, *, include_tracking=True, attribute_predictions=True, pass_model=None):
    prepared = prepare(tracking, context)
    rate = tracking.get("sample_fps", 10)
    gap_ms = min(300, max(150, 1600/rate))
    samples = list(_observations(tracking, prepared))
    releases = _release_candidates(samples, gap_ms) if pass_model is None else pass_model.releases(samples, gap_ms)
    if pass_model is None:
        transfers = _transfer_candidates(samples, gap_ms)
        transfers_by_contact = {(row["period"], row["actor_sample_ms"]): row for row in transfers}
        for release in releases:
            transfer = transfers_by_contact.get((release["period"], release["actor_sample_ms"]))
            if transfer and transfer["track_id"] == release["track_id"]:
                release.update(transfer)
        seen_contacts = {(row["period"], row["actor_sample_ms"]) for row in releases}
        releases.extend(row for row in transfers if (row["period"], row["actor_sample_ms"]) not in seen_contacts)
    tracks = {key: dict(meta) for key, meta in prepared["tracks"].items()}
    conflicting_ids = set()
    for sample in samples:
        if sample is None:
            continue
        roster_ids = Counter(tracks[key]["player_id"] for key in sample.players if tracks[key]["player_id"])
        conflicting_ids.update(key for key in sample.players if roster_ids[tracks[key]["player_id"]] > 1)
    for key in conflicting_ids:
        tracks[key]["player_id"] = None
    sample_indices = {sample.t_ms: index for index, sample in enumerate(samples) if sample is not None}
    claimed_arrivals = set()
    for release in sorted(releases, key=lambda row: row["actor_sample_ms"], reverse=True):
        release["passer_player_id"] = tracks.get(release["track_id"], {}).get("player_id")
        outcome = _pass_outcome(release, samples, sample_indices[release["actor_sample_ms"]], gap_ms, tracks)
        arrival = (release["period"], outcome.get("arrival_ms"))
        if arrival[1] is not None:
            if arrival in claimed_arrivals:
                outcome = {"outcome": "unknown", "outcome_reason": "A later release already links to this arrival"}
            else:
                claimed_arrivals.add(arrival)
        release.update(outcome)
    releases.sort(key=lambda row: row["t_ms"])
    stamps = [row["t_ms"] for row in releases]
    spans = prepared["scope_spans"]
    known_periods = (bool(prepared["context"].get("period_intervals")) or any(f.get("period", 1) != 1 for f in tracking["frames"])
                     or any(segment.get("status") == "verified" for segment in tracking.get("sync", [])))
    cuts = _shot_boundaries(tracking, gap_ms)
    spotted = align_predictions(predictions, alignment) if predictions is not None else []
    live = prepared["context"].get("live_play")
    intervals = prepared["context"].get("live_intervals", [])

    events = []
    for prediction in spotted:
        stamp, period = prediction["t_ms"], prediction["period"]
        if not any(span["start_ms"] <= stamp < span["end_ms"] and (not known_periods or span["period"] == period) for span in spans):
            continue
        live_at_event = next((row.get("live_play") is True for row in intervals if row["start_ms"] <= stamp < row["end_ms"]
                              and (not known_periods or row.get("period", 1) == period)), live)
        if live_at_event is False:
            continue
        attribution = (_attribute(prediction, releases, stamps, cuts, known_periods)
                       if prediction["type"] == "pass" and attribute_predictions else
                       {"track_id": None, "actor_reason": "No tracking attribution requested for this action"})
        events.append({**prediction, **attribution, "status": "predicted"})

    by_release = defaultdict(list)
    for event in events:
        if "release_ms" in event:
            by_release[(event["period"], event["release_ms"])].append(event)
    for (_, release_ms), matches in by_release.items():
        best = min(matches, key=lambda row: (abs(row["t_ms"]-release_ms), -row["action_score"]))
        for event in matches:
            if event is best:
                continue
            event.update(track_id=None, actor_reason="This release already supports a closer prediction")
            for key in set(event) - {"source", "label", "type", "source_ms", "t_ms", "period",
                                     "action_score", "status", "track_id", "actor_reason"}:
                del event[key]

    matched_predictions = len(events)
    attributed_releases = {row["release_ms"] for row in events if "release_ms" in row}
    for release in releases if include_tracking else ():
        if release["t_ms"] in attributed_releases:
            continue
        label = "Possible pass between teammates" if release.get("recipient_candidate_track_id") else "Possible pass release"
        detected = "pass_model_score" in release
        events.append({**release, "type": "pass", "label": "Detected pass" if detected else label,
                       "source": "tracking", "status": "detected" if detected else "candidate"})

    receptions = []
    for row in events:
        row["id"] = f"{row['source']}:{row['period']}:{row['t_ms']}:{row['type']}"
        if row["type"] != "pass":
            continue
        row.setdefault("outcome", "unknown")
        row.setdefault("outcome_reason", "No attributed release to follow")
        if row["outcome"] == "complete":
            reception_id = f"{row['id']}:reception"
            row["reception_id"] = reception_id
            receptions.append({"id": reception_id, "type": "reception", "label": "Estimated reception",
                               "source": "tracking", "status": "estimated", "period": row["period"],
                               "t_ms": row["arrival_ms"], "track_id": row["receiver_track_id"],
                               "player_id": row["receiver_player_id"], "pass_id": row["id"],
                               "actor_reason": row["outcome_reason"]})
    passes = [row for row in events if row["type"] == "pass"]
    counts = Counter(row["track_id"] for row in passes if row["status"] == "predicted" and row["track_id"])
    if include_tracking:
        spotting = [row if known_periods else {**row, "period": 1}
                    for row in events if row["source"] == "soccernet"]
        ball_actions, possession = analyze_ball_actions(samples, gap_ms, prepared, tracks, releases, spotting)
    else:
        ball_actions, possession = [], None
    reviewed_events = [row for row in prepared["context"].get("events", []) if row.get("reviewed") is True and
                       any(span["period"] == row.get("period", 1) and span["start_ms"] <= row["t_ms"] < span["end_ms"]
                           for span in spans)]
    outcomes = Counter(row["outcome"] for row in passes)
    per_track = defaultdict(Counter)
    for row in passes:
        if row["track_id"]:
            per_track[row["track_id"]][row["outcome"]] += 1
        if row.get("receiver_track_id"):
            per_track[row["receiver_track_id"]]["receptions"] += 1
    return {"version": VERSION, **{key: tracking.get(key) for key in ("match_id", "video_id", "plan_id", "source_sha256")},
            "ball_actions": ball_actions, "possession": possession, "reviewed_events": reviewed_events,
            "pass_detector": "trained" if pass_model is not None else "rules",
            "prediction_game": predictions.get("UrlLocal") if predictions is not None else None,
            "alignment": alignment, "scope_spans": spans, "predictions_in_scope": matched_predictions,
            "predictions_outside_scope": len(spotted)-matched_predictions, "events": sorted(events + receptions, key=lambda row: row["t_ms"]),
            "player_pass_attempts": dict(counts), "pass_outcomes": _outcome_summary(outcomes),
            "track_pass_outcomes": {track: {**_outcome_summary(values), "receptions": values["receptions"]}
                                    for track, values in per_track.items()}}


def _outcome_summary(outcomes):
    complete, incomplete, unknown = (outcomes[key] for key in ("complete", "incomplete", "unknown"))
    resolved = complete + incomplete
    total = resolved + unknown
    return {"complete": complete, "incomplete": incomplete, "unknown": unknown,
            "completion_rate": complete/resolved if resolved else None,
            "outcome_coverage": resolved/total if total else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--half", type=int, choices=(1, 2))
    parser.add_argument("--source-ms", type=int, default=0)
    parser.add_argument("--video-ms", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze(json.loads(args.tracking.read_text()), json.loads(args.context.read_text()) if args.context else None,
                         json.loads(args.predictions.read_text()) if args.predictions else None,
                         {"half": args.half, "source_ms": args.source_ms, "video_ms": args.video_ms} if args.predictions else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"video_events: {exc}\n")


if __name__ == "__main__":
    main()
