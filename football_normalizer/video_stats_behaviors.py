"""Off-ball geometric evidence and reviewed events, separate from live tracking.

A proximity or movement candidate is never promoted to a confirmed football
 event. Missing contextual coverage is not interpreted as zero performance.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import hypot, isfinite
from statistics import median


RUN_NAMES = ("support_runs", "overlaps", "recovery_runs", "pressure_approaches", "deep_to_advanced_support")
STATE_NAMES = ("passing_lane_availability", "central_screening", "deep_passing_outlet", "between_line_availability")
EVENT_TYPES = ("pass", "touch", "tackle", "interception", "shot", "press")


def _finite(value):
    return type(value) in (int, float) and isfinite(value)


def _merge(spans):
    result = []
    for a, b in sorted((a, b) for a, b in spans if b > a):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def _intersections(left, right):
    return _merge([(max(a, x), min(b, y)) for a, b in left for x, y in right if min(b, y) > max(a, x)])


def _duration(spans):
    return sum(b-a for a, b in _merge(spans))


def _scopes(data):
    scopes = defaultdict(list)
    if data.get("scope_spans"):
        for row in data["scope_spans"]:
            scopes[row["period"]].append((row["start_ms"], row["end_ms"]))
    else:
        frames = data["frames"]
        for a, b in zip(frames, frames[1:]):
            if a["period"] == b["period"] and 0 < b["t_ms"]-a["t_ms"] <= data["max_gap_ms"]:
                scopes[a["period"]].append((a["t_ms"], b["t_ms"]))
    return {p: _merge(rows) for p, rows in scopes.items()}


def _subtract(spans, excluded):
    result = _merge(spans)
    for x, y in _merge(excluded):
        next_rows = []
        for a, b in result:
            if y <= a or x >= b:
                next_rows.append((a, b))
            else:
                if a < x:
                    next_rows.append((a, x))
                if y < b:
                    next_rows.append((y, b))
        result = next_rows
    return result


def _event_scopes(scopes, context):
    result = {}
    for period, spans in scopes.items():
        true = [(r["start_ms"], r["end_ms"]) for r in context.get("live_intervals", [])
                if r.get("period", 1) == period and r.get("live_play") is True]
        false = [(r["start_ms"], r["end_ms"]) for r in context.get("live_intervals", [])
                 if r.get("period", 1) == period and r.get("live_play") is False]
        result[period] = _intersections(spans, true) if context.get("live_play") is False else _subtract(spans, false)
    return result


def _resolve(value, players):
    if not isinstance(value, str) or not value:
        return None
    matches = [p for p in players if value in (p.get("id"), p.get("track_id"), p.get("player_id"))]
    return matches[0] if len(matches) == 1 else None


def _direction(data, period, team):
    value = data.get("directions", {}).get(str(period), {}).get(team)
    return value if type(value) is int and value in (-1, 1) else None


def _xy(player, direction=1):
    return player["x"]*direction, player["y"]


def _distance(a, b):
    return hypot(a["x"]-b["x"], a["y"]-b["y"])


def _connected(a, b, gap):
    return (a["period"] == b["period"] and a["segment_id"] == b["segment_id"]
            and not b.get("cut") and 0 < b["t_ms"]-a["t_ms"] <= gap)


def _episode_at(frame, episodes):
    matches = [e for e in episodes if e.get("period", 1) == frame["period"]
               and e["start_ms"] <= frame["t_ms"] < e["end_ms"]]
    return matches[0] if len(matches) == 1 else None


def _line(frame, ids, subject, direction, team):
    if not team or not isinstance(ids, list) or len(ids) < 2:
        return None
    members = [_resolve(id_, frame["players"]) for id_ in ids]
    if len({p["id"] for p in members if p is not None}) < 2:
        return None
    if any(p is None or p.get("team") != team for p in members):
        return None
    ys = [p["y"] for p in members]
    xs = [p["x"]*direction for p in members]
    if not min(ys) <= subject["y"] <= max(ys) or max(ys)-min(ys) < 6 or max(xs)-min(xs) > 15:
        return None
    return median(xs)


def _lane(frame, player, episode):
    ball = frame.get("ball")
    ids = episode.get("opponent_ids")
    opp = episode.get("opponent_team_id")
    if (not ball or not ball.get("ground") or episode.get("lane_context_complete") is not True
            or not isinstance(ids, list) or not ids or not opp or opp == player.get("team")):
        return None
    opponents = [_resolve(id_, frame["players"]) for id_ in ids]
    if any(p is None or p.get("team") != opp for p in opponents):
        return None
    dx, dy = player["x"]-ball["x"], player["y"]-ball["y"]
    norm = dx*dx+dy*dy
    if not 25 <= norm <= 35*35:
        return False
    for p in opponents:
        fraction = ((p["x"]-ball["x"])*dx+(p["y"]-ball["y"])*dy)/norm
        if 0 <= fraction <= 1 and hypot(p["x"]-ball["x"]-fraction*dx,
                                        p["y"]-ball["y"]-fraction*dy) <= 2+p.get("error_m", 0):
            return False
    return True


def _eligible(name, episode, team):
    phase, possession = episode.get("phase"), episode.get("possession_team_id")
    if name in ("support_runs", "overlaps", "passing_lane_availability"):
        return phase in ("attack", "build_up", "transition") and possession == team and bool(team)
    if name in ("deep_passing_outlet",):
        return phase == "build_up" and possession == team and bool(team)
    if name in ("between_line_availability", "between_line_receptions"):
        return phase in ("attack", "build_up") and possession == team and bool(team)
    if name == "deep_to_advanced_support":
        return phase == "transition" and possession == team and bool(team)
    if name == "recovery_runs":
        return phase in ("transition", "recovery") and bool(team) and bool(possession) and possession != team
    return phase in ("settled_defence", "recovery", "transition") and bool(team) and bool(possession) and possession != team


def _state(name, frame, player, episode, direction, data):
    if name in ("passing_lane_availability", "deep_passing_outlet"):
        value = _lane(frame, player, episode)
        if value is not None and name == "deep_passing_outlet":
            x = player["x"] if direction == 1 else data["length"]-player["x"]
            value = value and x <= data["length"]*.5 and data["width"]*.25 <= player["y"] <= data["width"]*.75
        return value
    if direction is None:
        return None
    if name == "central_screening":
        line = _line(frame, episode.get("defender_ids"), player, direction, player.get("team"))
        ball = frame.get("ball")
        if line is None or not ball or not ball.get("ground"):
            return None
        error = player.get("error_m", 0)
        return (line+error < player["x"]*direction < ball["x"]*direction-error
                and data["width"]*.25 <= player["y"] <= data["width"]*.75)
    middle = _line(frame, episode.get("opponent_midfielder_ids"), player, direction, episode.get("opponent_team_id"))
    back = _line(frame, episode.get("opponent_defender_ids"), player, direction, episode.get("opponent_team_id"))
    if middle is None or back is None or middle >= back:
        return None
    return middle+player.get("error_m", 0) < player["x"]*direction < back-player.get("error_m", 0)


def _state_continuity(name, before, after, subject, other, episode):
    ids = [subject["id"]]
    if name in ("passing_lane_availability", "deep_passing_outlet"):
        ids += episode.get("opponent_ids", [])
    elif name == "central_screening":
        ids += episode.get("defender_ids", [])
    else:
        ids += episode.get("opponent_midfielder_ids", [])+episode.get("opponent_defender_ids", [])
    dt = (after["t_ms"]-before["t_ms"])/1000
    for id_ in ids:
        a, b = _resolve(id_, before["players"]), _resolve(id_, after["players"])
        if (a is None or b is None or a["id"] != b["id"] or a.get("team") != b.get("team")
                or a.get("player_id") != b.get("player_id") or _distance(a, b) > 15*dt):
            return False
    return True


def _run_hit(name, rows, episode, direction, data):
    if len(rows) < 2 or rows[-1][0]["t_ms"]-rows[0][0]["t_ms"] < 800:
        return None
    if direction is None and name != "pressure_approaches":
        return None
    first, last = rows[0][1], rows[-1][1]
    error = first.get("error_m", 0)+last.get("error_m", 0)
    dx = (last["x"]-first["x"])*(direction or 1)
    if name == "deep_to_advanced_support":
        x0 = first["x"] if direction == 1 else data["length"]-first["x"]
        advanced = max(p["x"] if direction == 1 else data["length"]-p["x"] for _, p, _ in rows)
        return x0 <= data["length"]*.5 and advanced >= data["length"]*2/3 and advanced-x0 >= max(12, error)
    if name == "recovery_runs":
        return dx <= -max(5, error)
    if any(carrier is None for _, _, carrier in rows):
        return None
    carrier0, carrier1 = rows[0][2], rows[-1][2]
    if any(not carrier.get("team") or carrier.get("team") != episode.get("possession_team_id") for _, _, carrier in rows):
        return None
    same_team = all(p.get("team") == carrier.get("team") for _, p, carrier in rows)
    if name == "pressure_approaches":
        if same_team or any(p.get("team") == carrier.get("team") for _, p, carrier in rows):
            return None
        for index, (a, p, carrier) in enumerate(rows):
            for b, q, other in rows[index+1:]:
                if b["t_ms"]-a["t_ms"] > 3000:
                    break
                gain = _distance(p, carrier)-_distance(q, other)
                own_gain = _distance(p, carrier)-_distance(q, carrier)
                if gain >= max(4, error) and own_gain >= max(2, error/2) and _distance(q, other) <= 5:
                    return True
        return False
    if not same_team or first["id"] == carrier0["id"]:
        return None
    if name == "support_runs":
        return dx >= max(5, error) and 5 <= _distance(last, carrier1) <= 25
    ahead0 = (first["x"]-carrier0["x"])*direction
    ahead1 = (last["x"]-carrier1["x"])*direction
    outside = abs(last["y"]-data["width"]/2) > abs(carrier1["y"]-data["width"]/2)+1
    return (ahead0 <= -1 and ahead1 >= max(2, error) and dx >= max(4, error)
            and 2 <= abs(last["y"]-carrier1["y"]) <= 15 and outside)


def _off_ball(data, context, pid, meta, scopes):
    output = {}
    episodes = context.get("episodes", [])
    for name in RUN_NAMES+STATE_NAMES:
        eligible_ms = observed_ms = positive_ms = count = opportunities = observed_opportunities = 0
        evidence = []
        for episode in episodes:
            if not _eligible(name, episode, meta.get("team")):
                continue
            period = episode.get("period", 1)
            overlap = _intersections(scopes.get(period, []), [(episode["start_ms"], episode["end_ms"])])
            eligible_ms += _duration(overlap)
            if not overlap:
                continue
            opportunities += 1
            direction = _direction(data, period, meta.get("team"))
            if direction is None and name != "pressure_approaches":
                continue
            runs, run, states = [], [], []
            for frame in data["_episode_frames"].get(id(episode), []):
                if not any(a <= frame["t_ms"] <= b for a, b in overlap):
                    continue
                player = _resolve(pid, frame["players"])
                carrier = _resolve(episode.get("ball_carrier_id"), frame["players"])
                valid = frame.get("live") is True and player is not None and player.get("team") == meta.get("team")
                if name in ("support_runs", "overlaps", "pressure_approaches"):
                    valid = valid and carrier is not None and carrier["id"] != pid
                broken = False
                if run and valid:
                    previous_frame, previous_player, previous_carrier = run[-1]
                    dt = (frame["t_ms"]-previous_frame["t_ms"])/1000
                    broken = (not _connected(previous_frame, frame, data["max_gap_ms"])
                              or player.get("player_id") != previous_player.get("player_id")
                              or player.get("team") != previous_player.get("team")
                              or _distance(previous_player, player) > 15*dt)
                    if name in ("support_runs", "overlaps", "pressure_approaches"):
                        broken |= (carrier.get("id") != previous_carrier.get("id")
                                   or carrier.get("player_id") != previous_carrier.get("player_id")
                                   or carrier.get("team") != previous_carrier.get("team")
                                   or _distance(previous_carrier, carrier) > 15*dt)
                if not valid or broken:
                    if run:
                        runs.append(run)
                    run = []
                if valid:
                    run.append((frame, player, carrier))
                if name in STATE_NAMES:
                    states.append((frame, player, _state(name, frame, player, episode, direction, data) if valid else None))
            if run:
                runs.append(run)
            episode_observed = 0
            if name in RUN_NAMES:
                # Overlap/pressure are reviewed carrier-relative trajectories;
                # a switch of carrier identity also breaks the opportunity.
                for run in runs:
                    split = []
                    for row in run:
                        if split and (row[2] or {}).get("id") != (split[-1][2] or {}).get("id"):
                            hit = _run_hit(name, split, episode, direction, data)
                            if hit is not None:
                                episode_observed += split[-1][0]["t_ms"]-split[0][0]["t_ms"]
                                if hit:
                                    evidence.append(_evidence(name, pid, split[0][0], split[-1][0]))
                            split = []
                        split.append(row)
                    hit = _run_hit(name, split, episode, direction, data)
                    if hit is not None:
                        episode_observed += split[-1][0]["t_ms"]-split[0][0]["t_ms"]
                        if hit:
                            evidence.append(_evidence(name, pid, split[0][0], split[-1][0]))
            else:
                in_positive = False
                for (a, p, state), (b, q, next_state) in zip(states, states[1:]):
                    valid = (_connected(a, b, data["max_gap_ms"]) and state is not None and next_state is not None
                             and _state_continuity(name, a, b, p, q, episode))
                    if not valid:
                        in_positive = False
                        continue
                    dt = b["t_ms"]-a["t_ms"]
                    episode_observed += dt
                    if state and next_state:
                        positive_ms += dt
                        if not in_positive:
                            evidence.append(_evidence(name, pid, a, b))
                        else:
                            evidence[-1]["end_ms"] = b["t_ms"]
                        in_positive = True
                    else:
                        in_positive = False
            observed_ms += episode_observed
            observed_opportunities += bool(episode_observed)
        count = len(evidence)
        status = "unavailable" if not observed_ms else "partial" if observed_ms < eligible_ms else "estimated"
        output[name] = {"count": count if observed_ms else None, "observed_ms": observed_ms,
                        "eligible_ms": eligible_ms, "coverage_pct": round(100*observed_ms/eligible_ms, 2) if eligible_ms else None,
                        "opportunities": opportunities, "observed_opportunities": observed_opportunities,
                        "status": status, "reason": ("Requires live play, reviewed phase/team context, direction and continuously visible required players/ball/lines" if not observed_ms else
                                                       "Only observable opportunities contribute; these are geometric candidates, not intent or success"),
                        "evidence": evidence}
        if name in STATE_NAMES:
            output[name].update(value_ms=positive_ms if observed_ms else None,
                                pct=round(100*positive_ms/observed_ms, 2) if observed_ms else None)
    return output


def _evidence(name, pid, first, last):
    return {"kind": name, "track_id": pid, "period": first["period"], "start_ms": first["t_ms"],
            "end_ms": last["t_ms"], "segment_id": first["segment_id"], "status": "review_required"}


def _event_rows(context, data, pid):
    meta = data["tracks"][pid]
    observations = data.get("observed_by_track", {}).get(pid, [])
    rows, seen = [], set()
    for event in context.get("events", []):
        if event.get("reviewed") is not True or event.get("type") not in EVENT_TYPES+("reception",):
            continue
        if event.get("track_confirmed") is True:
            if event.get("track_id") != pid or not any(row["t_ms"] == event["t_ms"] and row["period"] == event.get("period", 1)
                                                      for row in observations):
                continue
        elif not meta.get("player_id"):
            continue
        identity = event.get("id")
        if identity and identity in seen:
            continue
        nearest = _nearest(data, event.get("period", 1), event["t_ms"])
        if nearest and abs(nearest["t_ms"]-event["t_ms"]) <= data["max_gap_ms"] and nearest.get("live") is False:
            continue
        player = _resolve(event.get("track_id"), nearest["players"]) if nearest else None
        if event.get("track_id") == pid or player and player["id"] == pid:
            rows.append(event)
            if identity:
                seen.add(identity)
    return rows


def _covered(context, scopes, event_type, pid):
    total = sum(_duration(rows) for rows in scopes.values())
    covered = 0
    for period, spans in scopes.items():
        review = [(r["start_ms"], r["end_ms"]) for r in context.get("event_coverage", [])
                  if r.get("complete") is True and r.get("period", 1) == period
                  and event_type in r.get("types", []) and r.get("track_id") in (None, pid)]
        covered += _duration(_intersections(spans, review))
    return covered, total > 0 and covered >= total


def _events(data, context, pid, scopes):
    rows = _event_rows(context, data, pid)
    meta = data["tracks"][pid]
    results = {}
    definitions = {
        "pass_attempts": ("pass", None), "passes_completed": ("pass", ("complete", "successful")),
        "touches": ("touch", None), "tackles": ("tackle", None),
        "tackles_won": ("tackle", ("successful", "won")), "interceptions": ("interception", None),
        "shots": ("shot", None), "receptions": ("reception", None), "press_attempts": ("press", None),
        "successful_presses": ("press", ("successful",)),
    }
    for name, (type_, outcomes) in definitions.items():
        selected = [r for r in rows if r["type"] == type_ and any(a <= r["t_ms"] < b for a, b in scopes.get(r.get("period", 1), []))]
        known = [r for r in selected if outcomes is None or r.get("outcome") in outcomes]
        count = len(known)
        covered, complete = _covered(context, scopes, type_, pid)
        known_outcomes = {"complete", "incomplete", "successful", "unsuccessful"} if type_ == "pass" else {"successful", "unsuccessful", "won", "lost"}
        missing_outcome = outcomes is not None and any(r.get("outcome") not in known_outcomes for r in selected)
        available = bool(meta.get("player_id")) and complete and not missing_outcome
        results[name] = {"value": count if available else None,
                         "confirmed_count": count if meta.get("player_id") or count else None,
                         "coverage_ms": covered, "status": "reviewed" if available else "partial" if count else "unavailable",
                         "reason": None if available else "Record actions from the footage. Complete totals require a confirmed player and full event review."}
    attempted, completed = results["pass_attempts"]["value"], results["passes_completed"]["value"]
    results["pass_completion_pct"] = {"value": round(100*completed/attempted, 2) if attempted and completed is not None else None,
                                      "status": "reviewed" if attempted and completed is not None else "unavailable",
                                      "reason": None if attempted and completed is not None else "No completely reviewed pass attempts/outcomes"}
    results["possession_ms"] = {"value": None, "status": "unavailable",
                                "reason": "Reviewed team possession does not establish individual ball control"}
    return results


def _receptions(data, context, pid, scopes, off_ball):
    base = {"count": None, "status": "unavailable", "observed_ms": 0, "eligible_ms": 0,
            "coverage_pct": None, "evidence": [],
            "reason": "Requires confirmed identity, complete reviewed reception arrivals, direction, phase and observable opposition lines"}
    meta = data["tracks"][pid]
    if not meta.get("player_id"):
        return base
    covered, complete = _covered(context, scopes, "reception", pid)
    episodes = [e for e in context.get("episodes", [])
                if _eligible("between_line_receptions", e, meta.get("team"))
                and _intersections(scopes.get(e.get("period", 1), []), [(e["start_ms"], e["end_ms"])])]
    if not complete or not episodes or any(_direction(data, e.get("period", 1), meta.get("team")) is None for e in episodes):
        return base
    evidence, observable, eligible = [], 0, 0
    for event in _event_rows(context, data, pid):
        if event.get("type") != "reception" or not any(a <= event["t_ms"] < b for a, b in scopes.get(event.get("period", 1), [])):
            continue
        frame = _nearest(data, event.get("period", 1), event["t_ms"])
        episode = _episode_at(frame, episodes) if frame else None
        if not episode:
            eligible += 1
            continue
        eligible += 1
        player = _resolve(pid, frame["players"])
        direction = _direction(data, frame["period"], meta.get("team"))
        if not player or frame.get("live") is not True or abs(frame["t_ms"]-event["t_ms"]) > data["max_gap_ms"]/2:
            continue
        value = _state("between_line_availability", frame, player, episode, direction, data)
        if value is None:
            continue
        observable += 1
        if value:
            evidence.append(_evidence("between_line_receptions", pid, frame, frame))
    base.update(count=len(evidence) if observable or not eligible else None,
                status="estimated" if observable == eligible else "partial", event_status="reviewed",
                observed_receptions=observable, eligible_receptions=eligible, coverage_ms=covered, evidence=evidence,
                reason="Reception arrival is reviewed; between-line classification uses estimated pitch geometry" if observable == eligible else base["reason"])
    return base


def _possession(data, context, scopes):
    by_team, all_spans = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for row in context.get("possession_intervals", []):
        if row.get("reviewed") is not True or not row.get("team_id"):
            continue
        period = row.get("period", 1)
        spans = _intersections(scopes.get(period, []), [(row["start_ms"], row["end_ms"])])
        by_team[row["team_id"]][period].extend(spans)
        all_spans[period].extend(spans)
    # Conflicting team intervals are not awarded to either team.
    conflict = defaultdict(list)
    teams = list(by_team)
    for i, team in enumerate(teams):
        for other in teams[i+1:]:
            for period in scopes:
                conflict[period].extend(_intersections(by_team[team][period], by_team[other][period]))
    totals = {}
    for team in teams:
        total = sum(_duration(rows)-_duration(_intersections(rows, conflict[period])) for period, rows in by_team[team].items())
        totals[team] = total
    reviewed = sum(totals.values())
    scope_ms = sum(_duration(rows) for rows in scopes.values())
    return {"teams": [{"team": team, "reviewed_possession_ms": value,
                       "possession_pct_of_reviewed_time": round(100*value/reviewed, 2) if reviewed else None}
                      for team, value in totals.items()],
            "reviewed_ms": reviewed, "scope_ms": scope_ms,
            "coverage_pct": round(100*reviewed/scope_ms, 2) if scope_ms else None,
            "conflicting_ms": sum(_duration(rows) for rows in conflict.values()),
            "status": "reviewed" if reviewed else "unavailable",
            "reason": None if reviewed else "Requires reviewed team-possession intervals; nearest player is not possession"}


def _transfer_candidates(data):
    candidates, previous, prior, run = [], None, None, []
    for frame in data["frames"]:
        ball = frame.get("ball")
        valid = frame.get("live") is True and ball and ball.get("ground") and frame["players"]
        continuous = previous is not None and _connected(previous, frame, data["max_gap_ms"])
        if not valid or previous is not None and not continuous:
            prior, run = None, []
        if not valid:
            previous = None
            continue
        previous = frame
        nearest = sorted(frame["players"], key=lambda p: _distance(p, ball))
        subject = nearest[0]
        near = (_distance(subject, ball) <= 2.5 and
                (len(nearest) == 1 or _distance(nearest[1], ball)-_distance(subject, ball) >= 1))
        if run and (not near or run[-1][1]["id"] != subject["id"]):
            if run[-1][0]["t_ms"]-run[0][0]["t_ms"] >= 300:
                prior = run[-1]
            run = []
        if not near:
            continue
        run.append((frame, subject, ball))
        if prior and frame["t_ms"]-prior[0]["t_ms"] > 3000:
            prior = None
        if (prior and run[-1][0]["t_ms"]-run[0][0]["t_ms"] >= 300
                and prior[1]["id"] != subject["id"] and prior[1].get("team")
                and prior[1]["team"] == subject.get("team") and _distance(prior[2], ball) >= 3):
            candidates.append({"kind": "ball_transfer_proximity_candidate", "status": "review_required",
                               "start_ms": prior[0]["t_ms"], "end_ms": frame["t_ms"], "period": frame["period"],
                               "from_track_id": prior[1]["id"], "to_track_id": subject["id"],
                               "reason": "Ball moved between player proximity zones; control, pass and completion are not established"})
            prior = None
    return candidates


def _nearest(data, period, stamp):
    frames, times = data["_period_frames"].get(period, ([], []))
    i = bisect_left(times, stamp)
    options = frames[max(0, i-1):i+1]
    return min(options, key=lambda f: abs(f["t_ms"]-stamp)) if options else None


def calculate(prepared, context=None):
    """Return explainable geometric opportunities, reviewed counts and candidates."""
    context = context or {}
    prepared = dict(prepared)
    by_period = defaultdict(list)
    for frame in prepared["frames"]:
        by_period[frame["period"]].append(frame)
    prepared["_period_frames"] = {p: (sorted(rows, key=lambda f: f["t_ms"]), sorted(f["t_ms"] for f in rows))
                                   for p, rows in by_period.items()}
    prepared["_episode_frames"] = {}
    for episode in context.get("episodes", []):
        rows, times = prepared["_period_frames"].get(episode.get("period", 1), ([], []))
        prepared["_episode_frames"][id(episode)] = rows[bisect_left(times, episode["start_ms"]):bisect_right(times, episode["end_ms"])]
    scopes = _scopes(prepared)
    event_scopes = _event_scopes(scopes, context)
    players = []
    for pid, meta in prepared["tracks"].items():
        off_ball = _off_ball(prepared, context, pid, meta, scopes)
        off_ball["between_line_receptions"] = _receptions(prepared, context, pid, event_scopes, off_ball)
        players.append({"id": pid, "off_ball": off_ball, "events": _events(prepared, context, pid, event_scopes)})
    return {"players": players, "event_candidates": _transfer_candidates(prepared),
            "possession": _possession(prepared, context, event_scopes),
            "notes": ["Off-ball outputs are geometric candidates within reviewed opportunities, not intent or success.",
                      "Event totals require confirmed identities and complete reviewed event coverage; partial reviewed counts are separate.",
                      "Proximity to the ball never establishes possession or a completed pass.",
                      "Reviewed events and possession exclude explicitly non-live video intervals; reception timing is never borrowed from pass release timing.",
                      "Distances use the supplied pitch scale; assumed dimensions make geometry approximate." if not prepared.get("metric_scale_verified") else
                      "Pitch scale was supplied as verified; geometric patterns still require contextual review."]}
