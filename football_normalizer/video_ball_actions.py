"""Estimate ball actions and possession from continuous on-screen observations."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from math import hypot
from statistics import median

from .video_pressing import estimate_presses


def same_ball(a, b, gap_ms):
    return bool(a and b and 0 < b.t_ms-a.t_ms <= gap_ms
                and (a.shot, a.period, a.ball_id) == (b.shot, b.period, b.ball_id))


def closest_contact(sample):
    near = sorted((sample.foot_distance(track), track) for track in sample.players)
    if near and near[0][0] <= .43 and (len(near) == 1 or near[1][0]-near[0][0] >= .14):
        return near[0][1]


def redirects_ball(before, contact, after, track, *, head=False):
    offsets = [sample.ball_offset(track) for sample in (before, contact, after)]
    if any(offset is None for offset in offsets):
        return False
    level = 1 if head else 0
    a, b, c = [(x, y+level) for x, y in offsets]
    if hypot(*b) > .22 or min(hypot(*a), hypot(*c))-hypot(*b) < .18:
        return False
    incoming = (b[0]-a[0], b[1]-a[1])
    outgoing = (c[0]-b[0], c[1]-b[1])
    return sum(u*v for u, v in zip(incoming, outgoing)) <= .5*hypot(*incoming)*hypot(*outgoing)


def _comparable_teams(a, b):
    return bool(a and b and a.startswith("kit:") == b.startswith("kit:"))


def _relative_motion(a, b, track):
    p, q = a.ball_offset(track), b.ball_offset(track)
    return hypot(p[0]-q[0], p[1]-q[1]) if p and q else None


def _contested(sample):
    nearby = [(dist, track) for track in sample.players if (dist := sample.foot_distance(track)) <= .55]
    if len(nearby) < 2:
        return False
    (a, first), (b, second) = sorted(nearby)[:2]
    ta, tb = sample.teams.get(first), sample.teams.get(second)
    return b-a < .22 and _comparable_teams(ta, tb) and ta != tb


def _state(a, b, first, second, gap_ms):
    if not same_ball(a, b, gap_ms) or not a.players or not b.players:
        return "unobservable", None, None, None
    if first and first == second and _relative_motion(a, b, first) <= .28:
        ta, tb = a.teams.get(first), b.teams.get(first)
        team = (ta or tb) if not ta or not tb or ta == tb else None
        return "player_control", first, team, a.shot
    if _contested(a) or _contested(b):
        return "contested", None, None, a.shot
    motion = []
    for player in a.players.keys() & b.players.keys():
        left, top, right, bottom = a.players[player]
        x, y, x2, y2 = b.players[player]
        motion.append(((x+x2-left-right)/2, y2-bottom, (bottom-top+y2-y)/2))
    if len(motion) >= 3:
        dx, dy, height = (median(column) for column in zip(*motion))
        if hypot(b.ball[0]-a.ball[0]-dx, b.ball[1]-a.ball[1]-dy)/height >= .23:
            return "in_transit", None, None, a.shot
    return "unobservable", None, None, None


def _event(kind, sample, track, tracks, reason, **details):
    return {"id": f"tracking:{sample.period}:{sample.t_ms}:{kind}", "type": kind,
            "label": f"Estimated {kind}", "source": "tracking", "status": "estimated",
            "t_ms": sample.t_ms, "period": sample.period, "track_id": track,
            "player_id": tracks[track].get("player_id"), "team_id": sample.teams.get(track),
            "actor_reason": reason, **details}


def _contacts(samples, nearest, gap_ms, tracks, release_rows):
    releases = {(row["period"], row["actor_sample_ms"], row["track_id"]) for row in release_rows}
    recent = {}
    events = []
    for i in range(1, len(samples)-1):
        before, contact, after = samples[i-1:i+2]
        if not (same_ball(before, contact, gap_ms) and same_ball(contact, after, gap_ms)):
            continue
        player = nearest[i]
        kind, reason = None, None
        if player and nearest[i-1] != player:
            if redirects_ball(before, contact, after, player):
                kind = "one_touch" if (contact.period, contact.t_ms, player) in releases else "deflection_possible"
                reason = "Ball redirected at the player's feet"
            elif nearest[i+1] == player and _relative_motion(contact, after, player) <= .28:
                kind, reason = "controlled_touch", "Observed ball motion stayed close to the player's feet"
        if kind is None:
            heads = sorted((hypot(x, y+1), track) for track in contact.players
                           for x, y in [contact.ball_offset(track)])
            if heads and heads[0][0] <= .22 and (len(heads) == 1 or heads[1][0]-heads[0][0] >= .14):
                track = heads[0][1]
                if redirects_ball(before, contact, after, track, head=True):
                    player, kind, reason = track, "header_possible", "Ball redirected near the player's head"
        if kind is None:
            continue
        key = (contact.shot, contact.period, player)
        if contact.t_ms-recent.get(key, -1000) > 220:
            events.append(_event("touch", contact, player, tracks, reason, contact_kind=kind))
            recent[key] = contact.t_ms
    return events


def _shots(samples, nearest, mapped, gap_ms, prepared, tracks):
    """Find fast calibrated releases aimed at a goal mouth."""
    events = []
    length, width = prepared["length"], prepared["width"]
    for i in range(1, len(samples)-2):
        before, kick, leave, after = samples[i-1:i+3]
        if not all(same_ball(a, b, gap_ms) for a, b in zip((before, kick, leave), (kick, leave, after))):
            continue
        actor = nearest[i]
        if not actor or nearest[i+1] or nearest[i+2]:
            continue
        if nearest[i-1] != actor and not redirects_ball(before, kick, leave, actor):
            continue
        points = [mapped.get(s.t_ms) for s in (kick, leave, after)]
        if any(not p or p["error_m"] > 2 for p in points):
            continue
        a, b, c = points
        dx, dy = c["x"]-a["x"], c["y"]-a["y"]
        if abs(dx) < 3 or hypot(dx, dy)*1000/(after.t_ms-kick.t_ms) < 12:
            continue
        goal = length if dx > 0 else 0
        direction = prepared["directions"].get(str(kick.period), {}).get(kick.teams.get(actor))
        if (direction and (direction > 0) != (dx > 0)) or abs(goal-a["x"]) > 35:
            continue
        goal_y = a["y"]+(goal-a["x"])*dy/dx
        line_error = abs(b["y"]-a["y"]-(b["x"]-a["x"])*dy/dx)
        if (abs(goal_y-width/2) <= 3.66 and line_error <= 1.25
                and (b["x"]-a["x"])*dx > 0 and (c["x"]-b["x"])*dx > 0):
            events.append(_event("shot", kick, actor, tracks, "Fast release projected toward the goal mouth"))
    return events


def _regains(pairs, tracks, pass_releases):
    result = []
    previous, bridge = None, []
    for first, second, state, actor, team, shot in pairs:
        if state == "unobservable":
            previous, bridge = None, []
            continue
        if state != "player_control":
            if previous and second.t_ms-previous[0].t_ms <= 1800:
                bridge.append(state)
            else:
                previous, bridge = None, []
            continue
        if previous:
            prior, prior_actor, prior_team, prior_shot = previous
            if (shot == prior_shot and _comparable_teams(team, prior_team) and team != prior_team
                    and 0 < first.t_ms-prior.t_ms <= 1600):
                distance = first.foot_distance(prior_actor)
                releases = pass_releases.get((first.period, prior_actor), ())
                release_index = bisect_left(releases, prior.t_ms-250)
                released = release_index < len(releases) and releases[release_index] <= first.t_ms
                if bridge.count("in_transit") >= 2 and "contested" not in bridge and released:
                    kind = "interception"
                    reason = "Opponent gained control of a tracked pass in transit"
                elif (bridge.count("in_transit") <= 1 and distance is not None and distance <= 1.10
                      and first.t_ms-prior.t_ms <= 400):
                    kind = "tackle"
                    reason = "Nearby opponent took control from the ball carrier"
                else:
                    kind = None
                if kind:
                    result.append(_event(kind, first, actor, tracks, reason, outcome="won",
                                         dispossessed_track_id=prior_actor,
                                         next_control_track_id=actor, next_control_team_id=team,
                                         next_control_player_id=tracks[actor].get("player_id")))
        previous, bridge = (second, actor, team, shot), []
    return result


def _possession(spans, pairs, dead_intervals, tracks):
    by_period = defaultdict(list)
    for first, second, state, actor, team, _ in pairs:
        by_period[first.period].append((first.t_ms, second.t_ms, state, actor, team))
    intervals, states, teams, players = [], Counter(), Counter(), {}
    for span in spans:
        start, end, period = span["start_ms"], span["end_ms"], span["period"]
        observed = by_period[period]
        stops = [(row["start_ms"], row["end_ms"]) for row in dead_intervals if row.get("period", 1) == period]
        stops.sort()
        bounds = {start, end}
        observed = [row for row in observed if row[0] < end and start < row[1]]
        for a, b, *_ in observed + stops:
            if a < end and start < b:
                bounds.update((max(start, a), min(end, b)))
        timestamps = sorted(bounds)
        stamps = [row[0] for row in observed]
        dead_end, stop_index = -1, 0
        for a, b in zip(timestamps, timestamps[1:]):
            while stop_index < len(stops) and stops[stop_index][0] <= a:
                dead_end = max(dead_end, stops[stop_index][1])
                stop_index += 1
            index = bisect_right(stamps, a)-1
            if dead_end >= b:
                state, actor, team = "dead_ball", None, None
            elif index >= 0 and observed[index][1] >= b:
                state, actor, team = observed[index][2:]
            else:
                state, actor, team = "unobservable", None, None
            player = tracks[actor].get("player_id") if actor else None
            row = {"period": period, "start_ms": a, "state": state, "track_id": actor,
                   "player_id": player, "team_id": team}
            if (intervals and intervals[-1]["end_ms"] == a
                    and all(intervals[-1][key] == value for key, value in row.items() if key != "start_ms")):
                intervals[-1]["end_ms"] = b
            else:
                intervals.append({**row, "end_ms": b})
            states[state] += b-a
            if actor:
                key = ("player", player) if player else ("track", actor)
                entry = players.setdefault(key, {"player_id": player, "track_ids": set(), "control_ms": 0})
                entry["track_ids"].add(actor)
                entry["control_ms"] += b-a
            if team and state == "player_control":
                teams[team] += b-a
    total = sum(states.values())
    known_team = sum(teams.values())
    return {"intervals": intervals, "scope_ms": total, "states_ms": {
                key: states[key] for key in ("player_control", "in_transit", "contested", "dead_ball", "unobservable")},
            "observable_ms": total-states["unobservable"],
            "team_control_ms": known_team,
            "teams": [{"team_id": team, "control_ms": ms, "share_of_team_control": ms/known_team}
                      for team, ms in sorted(teams.items())],
            "players": [{**row, "track_ids": sorted(row["track_ids"])}
                        for row in sorted(players.values(), key=lambda row: -row["control_ms"])]}


def _dead_intervals(reviewed, predictions, spans):
    """Use an observed-out prediction only when a restart supplies an end."""
    stops = list(reviewed)
    restart_types = {"throw_in", "free_kick", "corner", "goal_kick", "kick_off", "kickoff"}
    for span in spans:
        pending = None
        for row in predictions:
            if (row["period"] != span["period"] or row["action_score"] < .6
                    or not span["start_ms"] <= row["t_ms"] < span["end_ms"]):
                continue
            if row["type"] == "out":
                pending = row["t_ms"]
            elif row["type"] in restart_types and pending is not None:
                if 0 < row["t_ms"]-pending <= 30_000:
                    stops.append({"period": span["period"], "start_ms": pending, "end_ms": row["t_ms"],
                                  "source": "spotting"})
                pending = None
            elif row["type"] in {"pass", "shot", "drive", "header", "cross"}:
                pending = None
    return stops


def analyze_ball_actions(samples, gap_ms, prepared, tracks, releases, predictions=()):
    nearest = [closest_contact(s) if s else None for s in samples]
    # A missing sample breaks regain evidence, even if timestamps on either side are close.
    continuous_pairs = [(a, b, *_state(a, b, first, second, gap_ms)) if a and b
                        else (a, b, "unobservable", None, None, None)
                        for a, b, first, second in zip(samples, samples[1:], nearest, nearest[1:])]
    pass_releases = defaultdict(list)
    for row in releases:
        if row.get("track_id"):
            pass_releases[row["period"], row["track_id"]].append(row["actor_sample_ms"])
    for stamps in pass_releases.values():
        stamps.sort()
    mapped = {frame["t_ms"]: frame["ball"] for frame in prepared["frames"]}
    actions = (_contacts(samples, nearest, gap_ms, tracks, releases)
               + _shots(samples, nearest, mapped, gap_ms, prepared, tracks)
               + _regains(continuous_pairs, tracks, pass_releases))
    dead = _dead_intervals(prepared["context"].get("dead_ball_intervals", []), predictions,
                           prepared["scope_spans"])
    actions.extend(estimate_presses(continuous_pairs, prepared, tracks, dead))
    actions = [row for row in actions if not any(stop.get("period", 1) == row["period"] and
               stop["start_ms"] <= row["t_ms"] < stop["end_ms"] for stop in dead)]
    possession = _possession(
        prepared["scope_spans"], [row for row in continuous_pairs if row[0] and row[1] and row[0].period == row[1].period],
        dead, tracks)
    possession["dead_ball_evidence"] = dead
    return sorted(actions, key=lambda row: row["t_ms"]), possession
