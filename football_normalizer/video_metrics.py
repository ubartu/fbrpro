"""Measure off-ball evidence on reviewed, synchronized video opportunities.

Outputs describe selected clips, not a full-match role. No interpolation,
imputed identity, inferred possession, or missing-as-zero scoring.
"""
import argparse
from collections import defaultdict
import json
from math import hypot, isfinite
from pathlib import Path
from statistics import median

from .midfield import GROUPS, intervals
from .video import read_rows, segments_from

VERSION = "fbrpro-video-metrics-v1"
DEFINITION = "fbrpro-off-ball-geometry-v1"
MEASURES = {
    "central_screening": ("Central screening time", "ms", {"settled_defence"}),
    "deep_to_advanced_support": ("Deep-to-advanced support", "episodes", {"transition"}),
    "deep_passing_outlet": ("Deep passing outlet", "episodes", {"build_up"}),
    "between_line_support": ("Between-line availability", "episodes", {"attack", "build_up"}),
    "between_line_reception": ("Between-line receptions", "episodes", {"attack", "build_up"}),
    "closing_down": ("Closing down the ball carrier", "episodes", {"settled_defence"}),
}
REASON_PHASE = {"regain_candidate": "transition", "opposition_build_up_sample": "settled_defence",
                "own_possession_sample": "build_up"}
DEFAULTS = {
    "max_gap_ms": 750, "transition_deep_x": .50, "transition_advanced_x": 2/3,
    "transition_min_advance_m": 12, "line_depth_spread_ratio": .12,
    "line_min_width_ratio": .15, "lane_clearance_m": 2, "approach_window_ms": 2500,
    "approach_gain_m": 4, "approach_end_distance_m": 5,
}
POSITION_GROUPS = set(GROUPS.values())
SCOPE_KEYS = ("competition_id", "season_id")


def finite(x):
    return type(x) in (int, float) and isfinite(x)


def fraction(x):
    return finite(x) and 0 < x < 1


def config(context):
    overrides = context.get("definitions", {})
    if not isinstance(overrides, dict) or set(overrides) - DEFAULTS.keys():
        raise ValueError("Unknown movement definition")
    out = {**DEFAULTS, **overrides}
    if not all(finite(v) and v > 0 for v in out.values()):
        raise ValueError("Movement definitions must be finite and positive")
    if not all(fraction(out[k]) for k in
               ("transition_deep_x", "transition_advanced_x", "line_depth_spread_ratio", "line_min_width_ratio")):
        raise ValueError("Pitch fractions must be between zero and one")
    if out["transition_deep_x"] >= out["transition_advanced_x"]:
        raise ValueError("Advanced zone must begin ahead of the deep zone")
    return out


def reviewed_id_list(value):
    return isinstance(value, list) and all(isinstance(x, str) and x for x in value) and len(set(value)) == len(value)


def observations(frames, start, end):
    """Group coincident samples by clock; ambiguous duplicate clocks are unusable."""
    by_time = defaultdict(list)
    for f in frames:
        t = f.get("period_time_ms")
        if type(t) is int and start <= t <= end:
            by_time[t].append(f)
    return [(t, fs[0] if len(fs) == 1 else {}) for t, fs in sorted(by_time.items())]


def point(obj, length, width, direction, *, ball_ground_reviewed=False):
    if (obj.get("observed") is not True or obj.get("estimated") is not False
            or obj.get("visibility") != "observed"):
        return None
    projection = obj.get("projection_status")
    if projection != "verified" and not (projection == "ball_ground_assumed" and ball_ground_reviewed
                                         and obj.get("object_type") == "ball"):
        return None
    x, y = obj.get("x_m"), obj.get("y_m")
    if not (finite(x) and finite(y) and 0 <= x <= length and 0 <= y <= width):
        return None
    return (x, y) if direction == 1 else (length-x, width-y)


def unique_object(frame, object_type, *, player_id=None, team_id=None):
    if frame.get("calibration_status") != "verified":
        return None
    matches = [o for o in frame.get("objects", []) if o.get("object_type") == object_type
               and (player_id is None or o.get("player_id") == player_id)
               and (team_id is None or o.get("team_id") == team_id)]
    return matches[0] if len(matches) == 1 else None


def line_at_x(points, y, length, width, cfg):
    if (len(points) < 2 or any(p is None for p in points)
            or max(p[0] for p in points)-min(p[0] for p in points) > cfg["line_depth_spread_ratio"]*length):
        return None
    ys = [p[1] for p in points]
    if max(ys)-min(ys) < cfg["line_min_width_ratio"]*width or not min(ys) <= y <= max(ys):
        return None
    return median(p[0] for p in points)


def clear_lane(ball, player, opponents, radius):
    dx, dy = player[0]-ball[0], player[1]-ball[1]
    norm2 = dx*dx+dy*dy
    if not norm2 or not opponents or any(p is None for p in opponents):
        return None
    for ox, oy in opponents:
        u = ((ox-ball[0])*dx+(oy-ball[1])*dy)/norm2
        if 0 <= u <= 1 and hypot(ox-ball[0]-u*dx, oy-ball[1]-u*dy) <= radius:
            return False
    return True


def covers(spans, period, start, end):
    covered = start
    for p, a, b in sorted(spans):
        if p == period and a <= covered <= b:
            covered = max(covered, b)
    return covered >= end


def context_for_plan(plan, bundle):
    """Derive known midfield eligibility and regain anchors, not line/ball identity."""
    if json.loads((bundle / "status.json").read_text()).get("status") != "accepted":
        raise ValueError("Expected an accepted normalized bundle")
    events = {e["event_id"]: e for e in read_rows(bundle, "events") if e["match_id"] == plan["match_id"]}
    if not events:
        raise ValueError("Bundle contains no events for this video plan")
    ends = {}
    for e in events.values():
        if e["type"] == "half_end":
            ends[e["period"]] = max(ends.get(e["period"], 0), e["period_time_ms"])
    windows = defaultdict(lambda: defaultdict(list))
    for row in read_rows(bundle, "appearances"):
        if row["match_id"] == plan["match_id"] and row["team_id"] == plan["team_id"]:
            for pos in row.get("position_intervals", []):
                if pos.get("position_id") in GROUPS:
                    # A malformed interval cannot establish on-pitch eligibility.
                    try:
                        parsed = list(intervals(pos, ends))
                    except (KeyError, ValueError, TypeError, AttributeError):
                        continue
                    windows[row["player_id"]][GROUPS[pos["position_id"]]].extend(parsed)
    out = {}
    for clip in plan["clips"]:
        for o in clip["opportunities"]:
            start, end, period = o["match_start_ms"], o["match_end_ms"], o["period"]
            info = {}
            if o["reason"] == "regain_candidate":
                tagged = [events[id_] for id_ in o.get("event_ids", []) if id_ in events
                          and events[id_]["period"] == period
                          and events[id_].get("possession_team_id") == plan["team_id"]]
                if tagged:
                    info["regain_ms"] = min(e["period_time_ms"] for e in tagged)
            eligible = []
            groups = {}
            for pid, by_group in windows.items():
                spans = [span for rows in by_group.values() for span in rows]
                if covers(spans, period, start, end):
                    eligible.append(pid)
                    full = [g for g, rows in by_group.items() if covers(rows, period, start, end)]
                    if len(full) == 1:
                        groups[pid] = full[0]
            if windows:
                info["player_ids"] = eligible
                info["position_groups"] = groups
            out[o["id"]] = info
    return out


def details(clip, opportunity, plan, start, end, segment):
    return {"match_id": plan["match_id"], "video_id": plan["video_id"], "plan_id": plan["plan_id"],
            "opportunity_id": opportunity["id"], "clip_id": clip["id"], "period": opportunity["period"],
            "video_start_ms": segment.map(start) if segment else opportunity["video_start_ms"],
            "video_end_ms": segment.map(end) if segment else opportunity["video_end_ms"],
            "period_start_ms": start, "period_end_ms": end, "event_ids": opportunity.get("event_ids", [])}


def summarize(rows):
    numerator = sum(o["numerator"] for o in rows)
    denominator = sum(o["denominator"] for o in rows)
    excluded = sum(o["excluded"] for o in rows)
    eligible = denominator + excluded
    return {"numerator": numerator, "denominator": denominator, "eligible": eligible, "excluded": excluded,
            "coverage_pct": round(100*denominator/eligible, 2) if eligible else None,
            "per_100": round(100*numerator/denominator, 2) if denominator else None}


def measure_summary(key, rows):
    label, unit, _ = MEASURES[key]
    return {"metric": key, "label": label, "unit": unit, **summarize(rows), "observations": rows}


def position_comparisons(players, *, comparable):
    groups = defaultdict(list)
    for player in players:
        evidence = defaultdict(lambda: defaultdict(list))
        for measure in player["measures"]:
            for row in measure["observations"]:
                group = row.get("position_group")
                if group in POSITION_GROUPS:
                    evidence[group][measure["metric"]].append(row)
        for group, measures in evidence.items():
            groups[group].append({**player, "measures": [measure_summary(key, measures[key]) for key in MEASURES]})
    for peers in groups.values():
        for index in range(len(MEASURES)):
            rows = [(p["player_id"], p["measures"][index]) for p in peers]
            for pid, row in rows:
                comparison = None
                if comparable and row["denominator"]:
                    comparison = dict(higher=0, tied=0, lower=0, missing=0)
                    for other_pid, other in rows:
                        if pid == other_pid:
                            continue
                        if not other["denominator"]:
                            comparison["missing"] += 1
                        else:
                            diff = (other["numerator"]*row["denominator"] - row["numerator"]*other["denominator"])
                            comparison["higher" if diff > 0 else "lower" if diff < 0 else "tied"] += 1
                row["comparison"] = comparison
    return [{"position_group": group, "players": groups[group]} for group in sorted(groups)]


def assess(metric, episode, samples, subject, team, length, width, direction, cfg, start, end):
    """Return numerator, observable denominator, exclusion explanation, evidence track IDs."""
    player = lambda frame: unique_object(frame, "player", player_id=subject, team_id=team)
    coords = lambda obj: point(obj or {}, length, width, direction,
                               ball_ground_reviewed=episode.get("ground_ball_reviewed") is True)
    max_gap = cfg["max_gap_ms"]
    tracks = set()

    def observed_player(obj):
        p = coords(obj)
        if p and obj and obj.get("track_id"):
            tracks.add(str(obj["track_id"]))
        return p

    def player_point(frame):
        return observed_player(player(frame))

    def members(frame, ids, side):
        return [coords(unique_object(frame, "player", player_id=id_, team_id=side)) for id_ in ids]

    def central(p):
        return width*.25 <= p[1] <= width*.75

    opp = episode.get("opponent_team_id")
    if opp == team or not isinstance(opp, str) or not opp:
        opp = None

    if metric == "central_screening":
        ids = episode.get("defender_ids", [])
        if not reviewed_id_list(ids) or len(ids) < 2 or subject in ids:
            return 0, 0, "Reviewed own defensive line is missing", []
        states = []
        for t, frame in samples:
            subject_track = player(frame)
            p, ball = observed_player(subject_track), coords(unique_object(frame, "ball"))
            line = line_at_x(members(frame, ids, team), p[1], length, width, cfg) if p and ball else None
            state = (line < p[0] < ball[0] and central(p)) if line is not None else None
            states.append((t, state, (subject_track or {}).get("track_id"), frame.get("camera_cut_suspected")))
        num = den = 0
        for (a, before, previous_id, _), (b, after, next_id, cut) in zip(states, states[1:]):
            if (before is not None and after is not None and 0 < b-a <= max_gap
                    and previous_id and previous_id == next_id and not cut):
                den += b-a
                # Conservative duration: no inference at state-changing edges.
                if before and after:
                    num += b-a
        return num, den, ("Player, ball, defensive line or contiguous samples missing"
                          if not den else "Some screening time was unobservable" if den < end-start else None), sorted(tracks)

    def continuity(series):
        return (len(series) >= 2 and 0 <= series[0][0]-start <= max_gap
                and 0 <= end-series[-1][0] <= max_gap
                and len(tracks) == 1 and all(x[1] is not None for x in series)
                and all(0 < b[0]-a[0] <= max_gap for a, b in zip(series, series[1:])))

    if metric == "deep_to_advanced_support":
        series = [(t, player_point(f)) for t, f in samples]
        if not continuity(series):
            return 0, 0, "Player trajectory is incomplete across the transition", sorted(tracks)
        origin = series[0][1][0]
        run = origin <= length*cfg["transition_deep_x"] and any(
            p[0] >= length*cfg["transition_advanced_x"] and p[0]-origin >= cfg["transition_min_advance_m"]
            for _, p in series[1:])
        return int(run), 1, None, sorted(tracks)

    if metric == "deep_passing_outlet":
        ids = episode.get("opponent_ids", [])
        if not opp or episode.get("lane_context_complete") is not True or not reviewed_id_list(ids) or not ids:
            return 0, 0, "Reviewed passing-lane context is missing", []
        series = []
        for t, f in samples:
            p, ball = player_point(f), coords(unique_object(f, "ball"))
            lane = clear_lane(ball, p, members(f, ids, opp), cfg["lane_clearance_m"]) if p and ball else None
            found = (lane and central(p) and p[0] <= length*.5
                     and 5 <= hypot(p[0]-ball[0], p[1]-ball[1]) <= 30) if lane is not None else None
            series.append((t, found))
        if not continuity(series):
            return 0, 0, "Ball, player or lane coverage is incomplete", sorted(tracks)
        return int(any(v for _, v in series)), 1, None, sorted(tracks)

    if metric in {"between_line_support", "between_line_reception"}:
        middle, back = episode.get("opponent_midfielder_ids", []), episode.get("opponent_defender_ids", [])
        if (not opp or not all(reviewed_id_list(v) and len(v) >= 2 for v in (middle, back))
                or set(middle) & set(back)):
            return 0, 0, "Reviewed opposition lines are missing", []
        series = []
        for t, f in samples:
            p = player_point(f)
            a = line_at_x(members(f, middle, opp), p[1], length, width, cfg) if p else None
            b = line_at_x(members(f, back, opp), p[1], length, width, cfg) if p else None
            series.append((t, (a < p[0] < b) if a is not None and b is not None and b > a else None))
        if not continuity(series):
            return 0, 0, "Player or opposition lines are not continuously observable", sorted(tracks)
        if metric == "between_line_support":
            return int(any(v for _, v in series)), 1, None, sorted(tracks)
        rec = episode.get("receptions")
        if (episode.get("receptions_complete") is not True or not isinstance(rec, list)
                or any(not isinstance(r, dict) or not isinstance(r.get("player_id"), str)
                       or type(r.get("period_time_ms")) is not int for r in rec)):
            return 0, 0, "Verified reception annotations are missing", sorted(tracks)
        unobservable_reception = False
        for r in rec:
            if r["player_id"] == subject and start <= r["period_time_ms"] <= end:
                t, between = min(series, key=lambda p: abs(p[0]-r["period_time_ms"]))
                if abs(t-r["period_time_ms"]) > max_gap/2:
                    unobservable_reception = True
                elif between:
                    return 1, 1, None, sorted(tracks)
        if unobservable_reception:
            return 0, 0, "Reception falls outside observable samples", sorted(tracks)
        return 0, 1, None, sorted(tracks)

    carrier = episode.get("ball_carrier_id")
    if not opp or not isinstance(carrier, str) or not carrier or carrier == subject:
        return 0, 0, "Reviewed ball carrier identity is missing", []
    series = []
    for t, f in samples:
        p, q = player_point(f), coords(unique_object(f, "player", player_id=carrier, team_id=opp))
        series.append((t, (p, q) if p and q else None))
    if not continuity(series):
        return 0, 0, "Player and ball carrier are not continuously observable", sorted(tracks)
    # Require the player to move toward the carrier's original location, not
    # simply a carrier running toward a stationary defender.
    for i, (t, (p, q)) in enumerate(series):
        before = hypot(p[0]-q[0], p[1]-q[1])
        for j in range(i+1, len(series)):
            u, (p2, q2) = series[j]
            if u-t > cfg["approach_window_ms"]:
                break
            after = hypot(p2[0]-q2[0], p2[1]-q2[1])
            if (after <= cfg["approach_end_distance_m"] and before-after >= cfg["approach_gain_m"]
                    and before-hypot(p2[0]-q[0], p2[1]-q[1]) >= cfg["approach_gain_m"]/2):
                return 1, 1, None, sorted(tracks)
    return 0, 1, None, sorted(tracks)


def analyze(plan, tracking, context, *, bundle=None):
    if plan.get("version") != "fbrpro-video-plan-v1" or tracking.get("version") != "fbrpro-video-tracking-v1":
        raise ValueError("Expected fbrpro video plan and tracking exports")
    for key in ("match_id", "video_id", "plan_id"):
        if not plan.get(key) or tracking.get(key) != plan[key] or context.get(key) != plan[key]:
            raise ValueError(f"Plan, tracking and context disagree on {key}")
    team = plan["team_id"]
    if context.get("team_id") not in (None, team):
        raise ValueError("Context team differs from the video plan")
    match = {}
    if bundle:
        match = next((r for r in read_rows(bundle, "matches") if r["match_id"] == plan["match_id"]), {})
        if not match:
            raise ValueError("Bundle contains no match for this video plan")
    scope = {}
    for key in SCOPE_KEYS:
        known_scope = match.get(key) or plan.get(key)
        reviewed_scope = context.get(key)
        if known_scope and reviewed_scope not in (None, known_scope):
            raise ValueError(f"Context disagrees with fixture on {key}")
        value = known_scope or reviewed_scope
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"Expected a canonical ID for {key}")
        scope[key] = value
    length, width = tracking["coordinate_length_m"], tracking["coordinate_width_m"]
    if (tracking.get("coordinate_frame") != "stadium_fixed"
            or not all(finite(v) and v > 0 for v in (length, width))):
        raise ValueError("Movement metrics need stadium-fixed pitch coordinates in metres")
    cfg = config(context)
    directions = context.get("attacking_direction", {})
    if not isinstance(directions, dict) or any(type(v) is not int or v not in {-1,1} for v in directions.values()):
        raise ValueError("Attacking directions must be 1 or -1 by period")
    roster = {p["id"]: p["name"] for p in plan["players"]}
    selected = context.get("players", list(roster))
    if not reviewed_id_list(selected) or set(selected) - roster.keys():
        raise ValueError("Select canonical player IDs from the plan roster")
    clip_opportunities = [(c, o) for c in plan["clips"] for o in c["opportunities"]]
    known = {o["id"] for _, o in clip_opportunities}
    if len(known) != len(clip_opportunities):
        raise ValueError("Plan repeats an opportunity")
    known_clips = {c["id"]: c for c in plan["clips"]}
    segments = {}
    if "alignment_segments" in plan:
        segments = {s.id: s for s in segments_from({"segments": plan["alignment_segments"]})}
    reviewed = {}
    supported_phases = set().union(*(phases for _, _, phases in MEASURES.values()))
    for e in context.get("episodes", []):
        if "phase" in e and e["phase"] not in supported_phases:
            raise ValueError(f"Unknown reviewed phase: {e['phase']}")
        id_ = e["opportunity_id"]
        if id_ not in known or id_ in reviewed:
            raise ValueError("Reviewed episodes must refer to unique selected opportunities")
        if "player_ids" in e and (not reviewed_id_list(e["player_ids"]) or set(e["player_ids"]) - roster.keys()):
            raise ValueError("Reviewed episode player_ids must be canonical IDs in the plan roster")
        groups = e.get("position_groups", {})
        if (not isinstance(groups, dict) or set(groups) - roster.keys()
                or any(not isinstance(g, str) or g not in POSITION_GROUPS for g in groups.values())):
            raise ValueError("Reviewed position groups must map plan player IDs to midfield groups")
        reviewed[id_] = e
    by_clip = defaultdict(list)
    for f in tracking["frames"]:
        c = known_clips.get(f.get("clip_id"))
        video_ms, period_ms = f.get("video_time_ms"), f.get("period_time_ms")
        if (c and f.get("period") == c["period"] and f.get("segment_id") == c["segment_id"]
                and type(video_ms) is int and type(period_ms) is int
                and c["video_start_ms"] <= video_ms <= c["video_end_ms"]):
            if segments:
                s = segments[c["segment_id"]]
                aligned = s.period_time(video_ms)
                if period_ms != aligned:
                    raise ValueError(f"{c['id']}: tracking frame clocks disagree with the video alignment")
            by_clip[c["id"]].append(f)
    derived = context_for_plan(plan, bundle) if bundle else {}
    rows = defaultdict(list)
    for clip, o in clip_opportunities:
        original = derived.get(o["id"], {})
        correction = reviewed.get(o["id"], {})
        episode = {**original, **correction}
        groups = {**original.get("position_groups", {}), **correction.get("position_groups", {})}
        phase = episode.get("phase", REASON_PHASE.get(o["reason"]))
        # "Long opposition possession" from the plan is a candidate, not
        # confirmation of settled defence. A reviewed context must confirm it.
        phase_verified = "phase" in episode or (phase == "transition" and "regain_ms" in original)
        start = episode.get("start_ms", episode.get("regain_ms", o["match_start_ms"])
                            if phase == "transition" else o["match_start_ms"])
        end = episode.get("end_ms", o["match_end_ms"])
        if (type(start) is not int or type(end) is not int
                or not o["match_start_ms"] <= start <= end <= o["match_end_ms"]):
            raise ValueError(f"{o['id']}: reviewed bounds must be inside the opportunity")
        samples = observations([f for f in by_clip[clip["id"]]
                                if o["video_start_ms"] <= f["video_time_ms"] <= o["video_end_ms"]], start, end)
        active = episode.get("player_ids")
        direction = directions.get(str(o["period"]))
        for pid in selected:
            if active is not None and pid not in active:
                continue
            for key, (_, unit, phases) in MEASURES.items():
                if phase not in phases:
                    continue
                eligible = end-start if unit == "ms" else 1
                if not eligible:
                    continue
                sync = segments.get(clip["segment_id"])
                error = ("Match clock has not passed an independent alignment check"
                         if "sync" in tracking and (sync is None or sync.sync_status != "verified") else
                         "Midfield playing interval has not been confirmed" if active is None else
                         "Phase has not been confirmed" if not phase_verified else
                         "Attacking direction has not been confirmed" if direction is None else
                         "Regain anchor has not been confirmed" if phase == "transition" and "regain_ms" not in episode
                         and "start_ms" not in episode else None)
                num, den, tracks = 0, 0, []
                if not error:
                    num, den, error, tracks = assess(key, episode, samples, pid, team, length, width,
                                                      direction, cfg, start, end)
                state = "unobservable" if not den else "partial" if den < eligible else "observed" if num else "not_observed"
                rows[pid, key].append({**details(clip, o, plan, start, end, segments.get(clip["segment_id"])),
                    "track_ids": tracks, "position_group": groups.get(pid), "phase": phase, "status": state, "reason": error,
                    "numerator": num, "denominator": den, "eligible": eligible, "excluded": eligible-den})
    players = [{"player_id": pid, "name": roster[pid], "team_id": team,
                "measures": [measure_summary(key, rows[pid, key]) for key in MEASURES]} for pid in selected]
    return {"version": VERSION, "definition_version": DEFINITION, "definitions": cfg,
            "match_id": plan["match_id"], "video_id": plan["video_id"], "plan_id": plan["plan_id"],
            **scope, "analysis_scope": "selected_clips", "players": players,
            "position_groups": position_comparisons(players, comparable=True),
            "notes": ["Rates apply to selected clips and observable opportunities only.",
                      "Measures describe movement evidence, not player roles, intent or execution quality.",
                      "An absent opportunity is not a negative observation."]}


def pool(reports):
    """Pool distinct match reports for the same definitions, by player/team/metric."""
    if not reports:
        raise ValueError("Provide at least one match metrics report")
    signature = (reports[0].get("definition_version"), reports[0].get("definitions"))
    scope = {key: reports[0].get(key) for key in SCOPE_KEYS}
    by_player = defaultdict(lambda: defaultdict(list))
    names = {}
    matches = set()
    for r in reports:
        if (r.get("version") != VERSION or (r.get("definition_version"), r.get("definitions")) != signature
                or r["match_id"] in matches):
            raise ValueError("Pool distinct matches with identical movement definitions")
        if any(r.get(key) != scope[key] for key in SCOPE_KEYS):
            raise ValueError("Pool reports from the same competition and season")
        matches.add(r["match_id"])
        for p in r["players"]:
            id_ = (p["player_id"], p["team_id"])
            names[id_] = p["name"]
            for m in p["measures"]:
                by_player[id_][m["metric"]].extend(m["observations"])
    players = [{"player_id": pid, "team_id": team, "name": names[pid, team],
                "measures": [measure_summary(key, rows) for key, rows in metrics.items()]}
               for (pid, team), metrics in by_player.items()]
    return {"version": VERSION, "definition_version": signature[0], "definitions": signature[1],
            **scope, "analysis_scope": "selected_clips_across_matches", "match_ids": sorted(matches), "players": players,
            "position_groups": position_comparisons(players, comparable=all(scope.values())),
            "notes": ["Pooled numerators and denominators, never averaged match percentages."]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path)
    p.add_argument("--tracking", type=Path)
    p.add_argument("--context", type=Path)
    p.add_argument("--bundle", type=Path, help="Optional accepted event bundle for playing intervals and regain anchors")
    p.add_argument("--pool", type=Path, nargs="+", help="Pool distinct match metrics reports instead")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        def load(path):
            return json.loads(path.read_text(encoding="utf-8"))
        if args.pool:
            if args.plan or args.tracking or args.context or args.bundle:
                raise ValueError("--pool cannot be combined with match inputs")
            report = pool([load(path) for path in args.pool])
        else:
            if not all((args.plan, args.tracking, args.context)):
                raise ValueError("Provide --plan, --tracking and --context")
            report = analyze(load(args.plan), load(args.tracking), load(args.context), bundle=args.bundle)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        p.exit(2, f"video_metrics: {exc}\n")
    print(f"{len(report['players'])} player movement profiles written.")


if __name__ == "__main__":
    main()
