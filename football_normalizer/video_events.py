"""Align SoccerNet spotting predictions and estimate pass releases from tracking."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import hypot
from pathlib import Path

from .video_stats_common import finite, prepare


VERSION = "fbrpro-video-events-v1"
PASS_LABELS = {"pass", "high pass", "cross"}
ATTRIBUTION_WINDOW_MS = 400
ATTRIBUTION_SEPARATION_MS = 100


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
        action = label.lower()
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
    shot = 0
    for frame in prepared["frames"]:
        stamp = frame["t_ms"]
        source = raw[stamp]
        if source.get("camera_cut_suspected"):
            shot += 1
        if frame["live"] is False or stamp in duplicate_times:
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
            yield Sample(stamp, frame["period"], (source.get("clip_id"), source.get("segment_id"), shot), ball_id, center, players, teams)
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


def _same_ball(a, b, gap_ms):
    return (0 < b.t_ms-a.t_ms <= gap_ms and (a.shot, a.period, a.ball_id) == (b.shot, b.period, b.ball_id))


def _closest_contact(sample):
    near = sorted((sample.foot_distance(track), track) for track in sample.players)
    if near and near[0][0] <= .43 and (len(near) == 1 or near[1][0]-near[0][0] >= .14):
        return near[0][1]


def _redirects_ball(before, contact, after, track):
    offsets = [sample.ball_offset(track) for sample in (before, contact, after)]
    if any(offset is None for offset in offsets):
        return False
    a, b, c = offsets
    if hypot(*b) > .22 or min(hypot(*a), hypot(*c))-hypot(*b) < .18:
        return False
    incoming = (b[0]-a[0], b[1]-a[1])
    outgoing = (c[0]-b[0], c[1]-b[1])
    return sum(u*v for u, v in zip(incoming, outgoing)) <= .5*hypot(*incoming)*hypot(*outgoing)


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


def analyze(tracking, context=None, predictions=None, alignment=None):
    prepared = prepare(tracking, context)
    rate = tracking.get("sample_fps", 10)
    gap_ms = min(300, max(150, 1600/rate))
    samples = list(_observations(tracking, prepared))
    releases = _release_candidates(samples, gap_ms)
    transfers = _transfer_candidates(samples, gap_ms)
    transfers_by_contact = {(row["period"], row["actor_sample_ms"]): row for row in transfers}
    for release in releases:
        transfer = transfers_by_contact.get((release["period"], release["actor_sample_ms"]))
        if transfer and transfer["track_id"] == release["track_id"]:
            release.update(transfer)
    seen_contacts = {(row["period"], row["actor_sample_ms"]) for row in releases}
    releases.extend(row for row in transfers if (row["period"], row["actor_sample_ms"]) not in seen_contacts)
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
                       if prediction["type"] == "pass" else {"track_id": None, "actor_reason": "Player attribution is available for passes"})
        events.append({**prediction, **attribution, "status": "predicted"})

    by_release = defaultdict(list)
    for event in events:
        if event["track_id"]:
            by_release[event["release_ms"]].append(event)
    for release_ms, matches in by_release.items():
        best = min(matches, key=lambda row: (abs(row["t_ms"]-release_ms), -row["action_score"]))
        for event in matches:
            if event is best:
                continue
            event.update(track_id=None, actor_reason="This release already supports a closer prediction")
            for key in ("release_ms", "actor_sample_ms", "evidence_start_ms", "evidence_end_ms", "recipient_candidate_track_id"):
                event.pop(key, None)

    matched_predictions = len(events)
    attributed_releases = {row["release_ms"] for row in events if "release_ms" in row}
    for release in releases:
        if release["t_ms"] in attributed_releases:
            continue
        label = "Possible pass between teammates" if release.get("recipient_candidate_track_id") else "Possible pass release"
        events.append({**release, "type": "pass", "label": label, "source": "tracking", "status": "candidate"})

    events.sort(key=lambda row: row["t_ms"])
    for row in events:
        row["id"] = f"{row['source']}:{row['period']}:{row['t_ms']}:{row['type']}"
    counts = Counter(row["track_id"] for row in events if row["type"] == "pass" and row["status"] == "predicted" and row["track_id"])
    return {"version": VERSION, **{key: tracking.get(key) for key in ("match_id", "video_id", "plan_id", "source_sha256")},
            "prediction_game": predictions.get("UrlLocal") if predictions is not None else None,
            "alignment": alignment, "scope_spans": spans, "predictions_in_scope": matched_predictions,
            "predictions_outside_scope": len(spotted)-matched_predictions, "events": events,
            "player_pass_attempts": dict(counts)}


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
