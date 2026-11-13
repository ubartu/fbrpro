"""Time-weighted geometry and conservative physical estimates after tracking.

No detector, tracker or replay state is modified. Observations are never bridged
across cuts, missing tracks, calibration gaps or period changes. Physical values
are window estimates of visible motion, not full-match running totals.
"""
from collections import defaultdict
from math import hypot, isfinite
from statistics import median


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _rounded(value):
    return round(value, 3) if value is not None else None


def _continuous(a, b, limit):
    return (0 < b["t_ms"]-a["t_ms"] <= limit and not b.get("cut")
            and a.get("segment_id") == b.get("segment_id")
            and a.get("period") == b.get("period"))


def _same(a, b):
    return a.get("team") == b.get("team") and a.get("player_id") == b.get("player_id")


def _point(row):
    return row["x"], row["y"]


def _distance(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])


def _cell(row, columns, rows, length, width):
    x, y = row["x"], row["y"]
    error = max(0, row.get("error_m", 0))
    if any(abs(x-length*i/columns) <= error for i in range(1, columns)):
        return None
    if any(abs(y-width*i/rows) <= error for i in range(1, rows)):
        return None
    return min(rows-1, int(y/width*rows))*columns+min(columns-1, int(x/length*columns))


def _weighted_add(store, key, value, dt):
    if value is not None and _number(value):
        total, time = store.get(key, (0., 0))
        store[key] = total+value*dt, time+dt


def _weighted(store, key):
    total, time = store.get(key, (0, 0))
    return total/time if time else None


def _line_points(frame, team, name):
    mapping = frame.get(name) or {}
    members = mapping.get(team, [])
    if not isinstance(members, (list, tuple, set)):
        return []
    return [_point(p) for p in frame["players"] if p.get("team") == team and
            any(p.get(field) in members for field in ("id", "track_id", "player_id"))]


def _line_distance(point, points):
    if len(points) < 2:
        return None
    points = sorted(points, key=lambda p: p[1])
    if points[-1][1]-points[0][1] < 5:
        return None
    distances = []
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length2 = dx*dx+dy*dy
        if not length2:
            continue
        u = max(0, min(1, ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2))
        distances.append(_distance(point, (a[0]+u*dx, a[1]+u*dy)))
    return min(distances) if distances else None


def _relationship_values(frame, player):
    team = player.get("team")
    friends = [p for p in frame["players"] if team and p.get("team") == team and p["id"] != player["id"]]
    opponents = [p for p in frame["players"] if team and p.get("team") and p["team"] != team]
    point = _point(player)
    ball = frame.get("ball")
    other_teams = {p["team"] for p in opponents}
    defensive_distances = [_line_distance(point, _line_points(frame, t, "defensive_lines")) for t in other_teams]
    defensive_distances = [d for d in defensive_distances if d is not None]
    return {
        "mean_distance_to_ball_m": _distance(point, (ball["x"], ball["y"])) if ball and ball.get("ground") else None,
        "mean_nearest_teammate_m": min((_distance(point, _point(p)) for p in friends), default=None),
        "mean_nearest_opponent_m": min((_distance(point, _point(p)) for p in opponents), default=None),
        "mean_distance_to_own_defensive_line_m": _line_distance(point, _line_points(frame, team, "defensive_lines")) if team else None,
        "mean_distance_to_opposition_defensive_line_m": min(defensive_distances) if defensive_distances else None,
    }


def _hull_area(points):
    points = sorted(set(points))
    if len(points) < 3:
        return 0.
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1]+upper[:-1]
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(hull, hull[1:]+hull[:1])))/2


def _team_shape(frame, team, ids):
    players = [p for p in frame["players"] if p["id"] in ids and p.get("team") == team]
    if len(players) < 3:
        return None
    points = [_point(p) for p in players]
    centroid = tuple(sum(p[i] for p in points)/len(points) for i in (0, 1))
    reviewed_lines = (frame.get("team_lines") or {}).get(team, [])
    if isinstance(reviewed_lines, dict):
        reviewed_lines = list(reviewed_lines.values())
    depths = []
    if isinstance(reviewed_lines, list):
        for members in reviewed_lines:
            if not isinstance(members, (list, tuple, set)):
                continue
            line = [p for p in players if any(p.get(field) in members for field in ("id", "track_id", "player_id"))]
            if len(line) >= 2:
                depths.append(sum(p["x"] for p in line)/len(line))
    depths.sort()
    return {"centroid_x_m": centroid[0], "centroid_y_m": centroid[1],
            "width_m": max(p[1] for p in points)-min(p[1] for p in points),
            "length_m": max(p[0] for p in points)-min(p[0] for p in points),
            "compactness_m": sum(_distance(p, centroid) for p in points)/len(points),
            "convex_hull_area_m2": _hull_area(points),
            "mean_observed_players": len(players),
            "line_spacing_m": sum(b-a for a, b in zip(depths, depths[1:]))/(len(depths)-1) if len(depths) >= 2 else None}


def _new_player(id_, metadata):
    return {"id": id_, "track_id": metadata.get("track_id", id_), "player_id": metadata.get("player_id"),
            "team": metadata.get("team"), "_usable_ms": 0, "_position": {}, "_bounds": None,
            "_heatmap": [0]*24, "_zones": [0]*9, "_zone_uncertain_ms": 0,
            "_transitions": defaultdict(int), "_relations": {}, "_relative_windows": [], "_physical_windows": [],
            "_geometry_interval_ms": 0, "_rejected_interval_ms": 0}


def _regression_window(window, direction):
    """Robust median filtering followed by a one-second velocity fit."""
    if len(window) < 5 or window[-1][0]-window[0][0] < 1000:
        return None
    times = [(t-window[0][0])/1000 for t, _ in window]
    rows = [row for _, row in window]
    coords = [[row[k] for row in rows] for k in ("x", "y")]
    for values in coords:
        original = values[:]
        for i in range(1, len(values)-1):
            values[i] = median(original[i-1:i+2])
    center = sum(times)/len(times)
    denominator = sum((t-center)**2 for t in times)
    if denominator <= 0:
        return None
    slopes = [sum((t-center)*(v-sum(values)/len(values)) for t, v in zip(times, values))/denominator for values in coords]
    duration = times[-1]
    error = max(row.get("error_m", 0) for row in rows)*2
    displacement = hypot(*slopes)*duration
    below_noise = displacement <= error
    if below_noise:
        slopes = [0., 0.]
    speed = hypot(*slopes)
    if speed > 13:
        return None
    return {"start_ms": window[0][0], "end_ms": window[-1][0], "duration_ms": window[-1][0]-window[0][0],
            "center_ms": (window[0][0]+window[-1][0])/2, "speed": speed, "speed_error": error/duration,
            "distance": speed*duration, "dx": slopes[0]*duration, "dy": slopes[1]*duration,
            "direction": direction, "below_noise": below_noise}


def _physical_result(windows, config, gate_reason):
    keys = ("mean_speed_m_s", "max_speed_m_s", "high_speed_distance_m", "high_speed_time_ms", "sprint_bouts",
            "sprint_distance_m", "max_acceleration_m_s2", "max_deceleration_m_s2", "acceleration_bouts", "deceleration_bouts")
    result = {"status": "unavailable", "reason": gate_reason or "Need contiguous live, low-uncertainty observations for at least one second",
              "usable_ms": 0, "below_noise_ms": 0,
              "measurement": "Median-filtered velocity fits over contiguous windows of at least one second; speed maxima are window estimates, not instantaneous GPS maxima; acceleration compares adjacent window speeds",
              **{key: None for key in keys}}
    if not windows:
        return result
    total = sum(w["duration_ms"] for w in windows)
    high, sprint = config.get("high_speed_m_s", 5.5), config.get("sprint_m_s", 7)
    accel_threshold = config.get("accel_m_s2", 2.5)
    high_windows = [w for w in windows if w["speed"]-w["speed_error"] >= high]
    bouts, run = [], []
    for w in windows:
        if w["speed"]-w["speed_error"] >= sprint:
            if run and (w["start_ms"] != run[-1]["end_ms"] or w.get("chain") != run[-1].get("chain")):
                bouts.append(run)
                run = []
            run.append(w)
        elif run:
            bouts.append(run)
            run = []
    if run:
        bouts.append(run)
    bouts = [b for b in bouts if sum(w["duration_ms"] for w in b) >= config.get("min_sprint_ms", 1000)]
    accelerations = []
    for a, b in zip(windows, windows[1:]):
        if a["end_ms"] != b["start_ms"] or a.get("chain") != b.get("chain"):
            accelerations.append(None)
            continue
        dt = (b["center_ms"]-a["center_ms"])/1000
        difference = b["speed"]-a["speed"]
        if abs(difference) <= a["speed_error"]+b["speed_error"]:
            accelerations.append(0.)
        elif dt > 0 and abs(difference/dt) <= 8:
            accelerations.append(difference/dt)
        else:
            accelerations.append(None)
    resolved_accelerations = [v for v in accelerations if v is not None]
    result.update(status="estimate", reason=None, usable_ms=total,
        mean_speed_m_s=_rounded(sum(w["distance"] for w in windows)/(total/1000)),
        max_speed_m_s=_rounded(max(w["speed"] for w in windows)),
        high_speed_distance_m=_rounded(sum(w["distance"] for w in high_windows)),
        high_speed_time_ms=sum(w["duration_ms"] for w in high_windows), sprint_bouts=len(bouts),
        sprint_distance_m=_rounded(sum(w["distance"] for b in bouts for w in b)),
        max_acceleration_m_s2=_rounded(max([0.]+resolved_accelerations)) if resolved_accelerations else None,
        max_deceleration_m_s2=_rounded(abs(min([0.]+resolved_accelerations))) if resolved_accelerations else None,
        acceleration_bouts=sum(v is not None and v >= accel_threshold and (i == 0 or accelerations[i-1] is None or accelerations[i-1] < accel_threshold) for i, v in enumerate(accelerations)) if resolved_accelerations else None,
        deceleration_bouts=sum(v is not None and v <= -accel_threshold and (i == 0 or accelerations[i-1] is None or accelerations[i-1] > -accel_threshold) for i, v in enumerate(accelerations)) if resolved_accelerations else None,
        below_noise_ms=sum(w["duration_ms"] for w in windows if w["below_noise"]))
    return result


def calculate(prepared):
    """Return positional, relational, physical and observed-team shape sections."""
    length, width = prepared["length"], prepared["width"]
    frames = prepared.get("frames", [])
    limit = prepared.get("max_gap_ms", 750)
    scope = prepared.get("scope_ms", 0)
    config = prepared.get("config", {})
    players = {id_: _new_player(id_, metadata) for id_, metadata in prepared.get("tracks", {}).items()}
    identities, memberships = defaultdict(set), defaultdict(set)
    for frame in frames:
        for row in frame["players"]:
            if row.get("player_id"):
                identities[row["id"]].add(row["player_id"])
            if row.get("team"):
                memberships[row["id"]].add(row["team"])
    ambiguous = {id_ for id_ in identities.keys() | memberships.keys()
                 if len(identities[id_]) > 1 or len(memberships[id_]) > 1}
    for id_, item in players.items():
        if len(identities[id_]) == 1:
            item["player_id"] = next(iter(identities[id_]))
        elif len(identities[id_]) > 1:
            item["player_id"] = None
        if len(memberships[id_]) == 1:
            item["team"] = next(iter(memberships[id_]))
        elif len(memberships[id_]) > 1:
            item["team"] = None
    teams, pending = {}, {}
    scale_verified = prepared.get("metric_scale_verified") is True
    orientation_confirmed = prepared.get("orientation_confirmed") is True
    calibration_reviewed = prepared.get("calibration_reviewed") is True
    physical_gate = ("Confirm actual pitch dimensions before physical measurement" if not scale_verified else
                     "Confirm pitch orientation before physical measurement" if not orientation_confirmed else
                     "Review camera calibration before physical measurement" if not calibration_reviewed else None)
    def flush(key):
        rows, chain = pending.pop(key)
        window = _regression_window(rows, chain[-1])
        if window:
            window["chain"] = chain
            players[key[0]][key[1]].append(window)

    def extend(key, start, end, chain):
        if key in pending and pending[key][1] != chain:
            flush(key)
        samples, _ = pending.setdefault(key, ([start], chain))
        samples.append(end)
        if end[0]-samples[0][0] >= 1000:
            flush(key)

    previous = None
    for frame in frames:
        if ambiguous:
            frame = {**frame, "players": [p for p in frame["players"] if p["id"] not in ambiguous]}
        rows = {p["id"]: p for p in frame["players"] if p["id"] not in ambiguous}
        for id_, row in rows.items():
            if id_ not in players:
                players[id_] = _new_player(id_, row)
        extended = set()
        if previous is not None and _continuous(previous, frame, limit):
            earlier = {p["id"]: p for p in previous["players"] if p["id"] not in ambiguous}
            dt = frame["t_ms"]-previous["t_ms"]
            usable = {}
            for id_ in earlier.keys() & rows.keys():
                a, b, player = earlier[id_], rows[id_], players[id_]
                player["_geometry_interval_ms"] += dt
                if (not _same(a, b) or previous.get("live") is False or frame.get("live") is False
                        or _distance(_point(a), _point(b)) > 13*dt/1000+max(a.get("error_m", 0), b.get("error_m", 0))*2):
                    player["_rejected_interval_ms"] += dt
                    continue
                player["_usable_ms"] += dt
                usable[id_] = (a, b)
                _weighted_add(player["_position"], "x", (a["x"]+b["x"])/2, dt)
                _weighted_add(player["_position"], "y", (a["y"]+b["y"])/2, dt)
                bounds = player["_bounds"] or [a["x"], a["x"], a["y"], a["y"]]
                player["_bounds"] = [min(bounds[0], a["x"], b["x"]), max(bounds[1], a["x"], b["x"]),
                                     min(bounds[2], a["y"], b["y"]), max(bounds[3], a["y"], b["y"])]
                for columns, nrows, store in ((6, 4, "_heatmap"), (3, 3, "_zones")):
                    first = _cell(a, columns, nrows, length, width)
                    last = _cell(b, columns, nrows, length, width)
                    if first is not None and first == last:
                        player[store][first] += dt
                    elif store == "_zones":
                        player["_zone_uncertain_ms"] += dt
                        if first is not None and last is not None and first != last:
                            player["_transitions"][first, last] += 1
                direction = prepared.get("directions", {}).get(str(frame.get("period")), {}).get(b.get("team"))
                chain = (frame.get("segment_id"), frame.get("period"), b.get("team"), b.get("player_id"), direction)
                start, end = (previous["t_ms"], a), (frame["t_ms"], b)
                key = (id_, "_relative_windows")
                extend(key, start, end, chain)
                extended.add(key)
                if (physical_gate is None and previous.get("live") is True and frame.get("live") is True
                        and max(a.get("error_m", 99), b.get("error_m", 99)) < 1.5
                        and _distance(_point(a), _point(b)) <= 13*dt/1000):
                    key = (id_, "_physical_windows")
                    extend(key, start, end, chain)
                    extended.add(key)
            # Counterpart positions must pass the same interval validity checks
            # as the subject. A teleporting teammate is not a valid relation.
            first_view = {**previous, "players": [earlier[id_] for id_ in usable]}
            last_view = {**frame, "players": [rows[id_] for id_ in usable]}
            first_ball, last_ball = previous.get("ball"), frame.get("ball")
            if (first_ball and last_ball and _distance(_point(first_ball), _point(last_ball)) >
                    80*dt/1000+first_ball.get("error_m", 0)+last_ball.get("error_m", 0)):
                first_view["ball"] = last_view["ball"] = None
            for id_, (a, b) in usable.items():
                first_rel, last_rel = _relationship_values(first_view, a), _relationship_values(last_view, b)
                for key in first_rel:
                    if first_rel[key] is not None and last_rel[key] is not None:
                        _weighted_add(players[id_]["_relations"], key, (first_rel[key]+last_rel[key])/2, dt)
            team_ids = {row.get("team") for row in rows.values()}-{None}
            for team in team_ids:
                ids = {id_ for id_, (a, b) in usable.items() if a.get("team") == b.get("team") == team}
                before, after = _team_shape(previous, team, ids), _team_shape(frame, team, ids)
                if before and after:
                    item = teams.setdefault(team, {"team": team, "_shape": {}, "_time": 0})
                    item["_time"] += dt
                    for key in before:
                        if before[key] is not None and after[key] is not None:
                            _weighted_add(item["_shape"], key, (before[key]+after[key])/2, dt)
        for key in list(pending):
            if key not in extended:
                flush(key)
        previous = frame
    for key in list(pending):
        flush(key)
    observed = {}
    for id_, stamps in prepared.get("observed_by_track", {}).items():
        observed[id_] = sum(b["t_ms"]-a["t_ms"] for a, b in zip(stamps, stamps[1:]) if _continuous(a, b, limit))
    output = []
    relation_keys = ("mean_distance_to_ball_m", "mean_nearest_teammate_m", "mean_nearest_opponent_m",
                     "mean_distance_to_own_defensive_line_m", "mean_distance_to_opposition_defensive_line_m")
    for id_, player in players.items():
        time = player["_usable_ms"]
        observed_time = max(observed.get(id_, player["_geometry_interval_ms"]), time)
        bounds = player["_bounds"]
        relative = player["_relative_windows"]
        windows = player["_physical_windows"]
        resolved = [w for w in relative if not w["below_noise"]]
        physical = _physical_result(windows, config, physical_gate)
        if id_ in ambiguous:
            physical["reason"] = "Conflicting identities or teams on this track require review"
        directional = [w for w in windows if w["direction"] in (-1, 1)]
        movements = {"forward_m": 0., "backward_m": 0., "lateral_m": 0.}
        runs = {"forward": 0, "backward": 0, "lateral": 0, "stationary": 0}
        for w in directional:
            dx, dy = w["dx"]*w["direction"], w["dy"]
            movements["forward_m" if dx >= 0 else "backward_m"] += abs(dx)
            movements["lateral_m"] += abs(dy)
            kind = "stationary" if w["below_noise"] else "lateral" if abs(dy) > abs(dx) else "forward" if dx >= 0 else "backward"
            runs[kind] += w["duration_ms"]
        positioning = {"status": "estimate" if time else "unavailable", "reason": None if time else
                       "Conflicting identities or teams on this track require review" if id_ in ambiguous else "Need contiguous calibrated observations",
            "quality": "verified_pitch_dimensions" if scale_verified else "assumed_pitch_dimensions",
            "mapping_quality": "reviewed" if calibration_reviewed else "unreviewed_estimate",
            "average_position_m": {"x": _rounded(_weighted(player["_position"], "x")), "y": _rounded(_weighted(player["_position"], "y"))} if time else None,
            "width_m": _rounded(bounds[3]-bounds[2]) if bounds else None,
            "depth_m": _rounded(bounds[1]-bounds[0]) if bounds else None,
            "resolved_time_ms": time,
            "heatmap": {"columns": 6, "rows": 4, "time_ms": player["_heatmap"],
                        "pct": [round(100*v/time, 2) if time else None for v in player["_heatmap"]],
                        "uncertain_ms": time-sum(player["_heatmap"])},
            "zones": {"columns": 3, "rows": 3, "time_ms": player["_zones"],
                      "pct": [round(100*v/time, 2) if time else None for v in player["_zones"]],
                      "uncertain_ms": player["_zone_uncertain_ms"]}}
        relative_time = sum(w["duration_ms"] for w in relative)
        relative_distance = sum(w["distance"] for w in resolved)*100/length
        relative_directional = [w for w in resolved if w["direction"] in (-1, 1)]
        movement = {"status": "estimate" if resolved or windows else "unavailable",
                    "reason": None if resolved or windows else (
                        "Movement stayed within mapping uncertainty in the usable windows" if relative else
                        "Need at least one second of continuous pitch positions"),
                    "relative_distance_pct": _rounded(relative_distance) if resolved else None,
                    "relative_rate_pct_s": _rounded(relative_distance*1000/relative_time) if resolved else None,
                    "relative_usable_ms": relative_time,
                    "relative_unresolved_ms": sum(w["duration_ms"] for w in relative if w["below_noise"]),
                    "relative_forward_pct": _rounded(sum(max(0, w["dx"]*w["direction"]) for w in relative_directional)*100/length) if relative_directional else None,
                    "relative_backward_pct": _rounded(sum(max(0, -w["dx"]*w["direction"]) for w in relative_directional)*100/length) if relative_directional else None,
                    "relative_lateral_pct": _rounded(sum(abs(w["dy"]) for w in resolved)*100/width) if resolved else None,
                    "observed_distance_m": _rounded(sum(w["distance"] for w in windows)) if windows else None,
                    **{key: _rounded(v) if directional else None for key, v in movements.items()},
                    "directional_time_ms": sum(w["duration_ms"] for w in directional),
                    "run_directions_ms": runs if directional else {key: None for key in runs},
                    "direction_reason": None if relative_directional or directional else "Forward and backward movement need a confirmed attacking direction",
                    "zone_transitions": [{"from": a, "to": b, "count": n} for (a, b), n in sorted(player["_transitions"].items())],
                    "measurement": "Pitch-relative estimates use continuous observed positions and include only movement above mapping uncertainty."}
        relationships = {key: _rounded(_weighted(player["_relations"], key)) for key in relation_keys}
        relationships["coverage_ms"] = {key: player["_relations"].get(key, (0, 0))[1] for key in relation_keys}
        relationships["status"] = "estimate" if any(relationships[key] is not None for key in relation_keys) else "unavailable"
        relationships["reason"] = "Distances use only coincident visible objects; ground-ball review and reviewed line membership required where applicable"
        output.append({key: player[key] for key in ("id", "track_id", "player_id", "team")} | {
            "coverage": {"observed_ms": observed_time, "usable_ms": time, "scope_ms": scope,
                         "usable_observed_pct": round(100*time/observed_time, 2) if observed_time else None,
                         "scope_coverage_pct": round(100*time/scope, 2) if scope else None,
                         "physical_ms": physical["usable_ms"], "rejected_interval_ms": player["_rejected_interval_ms"]},
            "positioning": positioning, "movement": movement, "physical": physical, "relationships": relationships})
    team_output = []
    for team, item in sorted(teams.items()):
        shape = {key: _rounded(_weighted(item["_shape"], key)) for key in
                 ("centroid_x_m", "centroid_y_m", "width_m", "length_m", "compactness_m", "convex_hull_area_m2", "mean_observed_players", "line_spacing_m")}
        shape.update(status="estimate", scope="simultaneously_observed_players", reason="Observed subset only; this is not complete-team formation",
                     line_spacing_reason=None if shape["line_spacing_m"] is not None else "Review at least two team lines with two visible players each")
        team_output.append({"team": team, "coverage": {"usable_ms": item["_time"], "scope_ms": scope,
                            "scope_coverage_pct": round(100*item["_time"]/scope, 2) if scope else None}, "shape": shape})
    return {"players": output, "teams": team_output}
