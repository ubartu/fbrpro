"""Read-only preparation of stable tracking reports for optional statistics."""
from collections import Counter, defaultdict
from bisect import bisect_right
from math import hypot, isfinite
from statistics import median


def finite(value):
    return type(value) in (int, float) and isfinite(value)


def _spans(rows, name):
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a list")
    for row in rows:
        if (not isinstance(row, dict) or not finite(row.get("start_ms")) or not finite(row.get("end_ms"))
                or not 0 <= row["start_ms"] < row["end_ms"] or type(row.get("period", 1)) is not int
                or not 1 <= row.get("period", 1) <= 5):
            raise ValueError(f"{name} needs increasing video timestamps and valid periods")


def validate_context(context):
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("Statistics context must be a JSON object")
    for key in ("live_play", "pitch_dimensions_verified", "orientation_confirmed", "calibration_reviewed",
                "ground_ball_reviewed", "flip_pitch"):
        if key in context and type(context[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("start_ms", "end_ms", "pitch_length_m", "pitch_width_m"):
        if key in context and (not finite(context[key]) or context[key] < 0):
            raise ValueError(f"{key} must be a nonnegative finite number")
    directions = context.get("directions", {})
    if not isinstance(directions, dict):
        raise ValueError("Directions must be keyed by period and team")
    for period, teams in directions.items():
        if period not in {str(i) for i in range(1, 6)} or not isinstance(teams, dict):
            raise ValueError("Directions need periods 1 to 5")
        if any(not isinstance(k, str) or not k or type(v) is not int or v not in (-1, 1) for k, v in teams.items()):
            raise ValueError("Each attacking direction must be 1 or -1")
        if teams.get("kit:team_a") and teams.get("kit:team_a") == teams.get("kit:team_b"):
            raise ValueError("Team A and Team B must attack opposite goals")
    identities = context.get("identities", {})
    if not isinstance(identities, dict):
        raise ValueError("Identities must be keyed by track ID")
    for key, row in identities.items():
        if not isinstance(key, str) or not isinstance(row, dict):
            raise ValueError("Each identity needs a track ID and identity object")
        for field in ("player_id", "team_id"):
            if field in row and (not isinstance(row[field], str) or not row[field]):
                raise ValueError("Reviewed identity values must be nonempty strings")
    for key in ("episodes", "event_coverage", "possession_intervals", "ground_ball_intervals", "live_intervals", "period_intervals"):
        _spans(context.get(key, []), key)
    previous_end = -1
    for interval in sorted(context.get("period_intervals", []), key=lambda r: r["start_ms"]):
        if interval["start_ms"] < previous_end:
            raise ValueError("Period intervals cannot overlap in video time")
        previous_end = interval["end_ms"]
    for row in context.get("event_coverage", []):
        if not isinstance(row.get("types"), list) or any(t not in ("pass", "reception", "touch", "tackle", "interception", "shot", "press") for t in row["types"]):
            raise ValueError("Event coverage needs a list of recognised event types")
        if type(row.get("complete")) is not bool:
            raise ValueError("Event coverage completeness must be true or false")
    for key in ("episodes", "possession_intervals", "live_intervals"):
        previous = {}
        for row in sorted(context.get(key, []), key=lambda r: r["start_ms"]):
            period = row.get("period", 1)
            if row["start_ms"] < previous.get(period, -1):
                raise ValueError(f"Overlapping {key} make the analysis ambiguous")
            previous[period] = row["end_ms"]
    for row in context.get("episodes", []):
        for field in ("defender_ids", "opponent_ids", "opponent_midfielder_ids", "opponent_defender_ids"):
            if field in row and (not isinstance(row[field], list) or any(not isinstance(v, str) for v in row[field])):
                raise ValueError(f"{field} must contain player/track IDs")
    for owner in [context, *context.get("episodes", [])]:
        for field in ("defensive_lines", "team_lines"):
            lines = owner.get(field, {})
            if not isinstance(lines, dict):
                raise ValueError(f"{field} must be keyed by team")
            for team, members in lines.items():
                if not isinstance(team, str) or not team:
                    raise ValueError("Line memberships need a nonempty team ID")
                groups = [members] if field == "defensive_lines" else list(members.values()) if isinstance(members, dict) else members
                if not isinstance(groups, list) or any(not isinstance(group, list) or any(not isinstance(v, str) or not v for v in group) for group in groups):
                    raise ValueError("Line memberships must be lists of player/track IDs")
    events = context.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Events must be a list")
    ids = set()
    for event in events:
        if (not isinstance(event, dict) or not isinstance(event.get("id"), str) or not event["id"]
                or event["id"] in ids or not finite(event.get("t_ms")) or event["t_ms"] < 0
                or event.get("type") not in ("pass", "reception", "touch", "tackle", "interception", "shot", "press")
                or type(event.get("period", 1)) is not int or not 1 <= event.get("period", 1) <= 5
                or not isinstance(event.get("track_id"), str)):
            raise ValueError("Events need unique IDs, valid type/time/period and track_id")
        ids.add(event["id"])
        for flag in ("reviewed", "track_confirmed"):
            if flag in event and type(event[flag]) is not bool:
                raise ValueError(f"Event {flag} must be true or false")
    config = {"high_speed_m_s": 5.5, "sprint_m_s": 7., "accel_m_s2": 2.5, "min_sprint_ms": 1000}
    settings = context.get("config", {})
    if not isinstance(settings, dict) or set(settings)-set(config):
        raise ValueError("Unknown statistics configuration")
    config.update(settings)
    if any(not finite(x) or x <= 0 for x in config.values()) or config["sprint_m_s"] < config["high_speed_m_s"]:
        raise ValueError("Physical thresholds must be positive and sprint speed >= high speed")
    return context, config


def _union(spans):
    result = []
    for period in sorted({s["period"] for s in spans}):
        for span in sorted((s for s in spans if s["period"] == period), key=lambda s: s["start_ms"]):
            if result and result[-1]["period"] == period and span["start_ms"] <= result[-1]["end_ms"]:
                result[-1]["end_ms"] = max(result[-1]["end_ms"], span["end_ms"])
            else:
                result.append(dict(span))
    return result


def prepare(report, context=None):
    context, config = validate_context(context)
    if (not isinstance(report, dict) or report.get("version") != "fbrpro-video-tracking-v1"
            or not isinstance(report.get("frames"), list) or report.get("coordinate_frame", "stadium_fixed") != "stadium_fixed"):
        raise ValueError("Choose a stadium-fixed fbrpro video tracking report")
    for key in ("match_id", "video_id", "plan_id", "source_sha256"):
        if context.get(key) and report.get(key) and context[key] != report[key]:
            raise ValueError(f"Statistics context has a different {key}")
    for key in ("coordinate_length_m", "coordinate_width_m"):
        if not finite(report.get(key)) or not 0 < report[key] <= 500:
            raise ValueError("Tracking requires finite pitch dimensions in metres")
    source_length, source_width = report["coordinate_length_m"], report["coordinate_width_m"]
    length, width = context.get("pitch_length_m", source_length), context.get("pitch_width_m", source_width)
    if not 0 < length <= 500 or not 0 < width <= 500:
        raise ValueError("Pitch dimensions must be positive and at most 500 metres")
    rate = report.get("sample_fps", 10)
    if not finite(rate) or not 0 < rate <= 120:
        raise ValueError("Tracking sample rate must be positive and at most 120")
    raw = report["frames"]
    if len(raw) > 250000:
        raise ValueError("Select a shorter statistics scope (up to 250,000 samples)")
    for frame in raw:
        if (not isinstance(frame, dict) or not finite(frame.get("video_time_ms")) or frame["video_time_ms"] < 0
                or not isinstance(frame.get("objects"), list) or len(frame["objects"]) > 200
                or type(frame.get("period", 1)) is not int or not 1 <= frame.get("period", 1) <= 5):
            raise ValueError("Tracking contains invalid frame timestamps, periods or objects")
        if any(not isinstance(o, dict) or not isinstance(o.get("track_id"), str) or not o["track_id"] for o in frame["objects"]):
            raise ValueError("Each observation requires a track ID")
    raw = sorted(raw, key=lambda f: f["video_time_ms"])
    periods = context.get("period_intervals", [])
    if periods:
        raw = [{**f, "period": next((r["period"] for r in periods if r["start_ms"] <= f["video_time_ms"] < r["end_ms"]), f.get("period", 1))} for f in raw]
    grouped = defaultdict(list)
    for f in raw:
        grouped[f.get("clip_id")].append(f)
    scope = []
    clips = report.get("clips", [])
    if not isinstance(clips, list) or any(not isinstance(row, dict) for row in clips):
        raise ValueError("Tracking clips must be a list of clip objects")
    for row in clips:
        a, b = row.get("video_start_ms"), row.get("video_end_ms")
        if not finite(a) or not finite(b) or not 0 <= a < b or type(row.get("period", 1)) is not int or not 1 <= row.get("period", 1) <= 5:
            raise ValueError("Declared clips need increasing finite video times and valid periods")
        if finite(a) and finite(b) and 0 <= a < b:
            members = [f for f in grouped[row.get("clip_id", row.get("id"))] if a <= f["video_time_ms"] < b]
            period = members[0].get("period", 1) if members else row.get("period", 1)
            first = a
            for member in members[1:]:
                if member.get("period", 1) != period:
                    scope.append({"period": period, "start_ms": first, "end_ms": member["video_time_ms"]})
                    first, period = member["video_time_ms"], member.get("period", 1)
            scope.append({"period": period, "start_ms": first, "end_ms": b})
    if not scope and raw:
        by_clip = defaultdict(list)
        for f in raw:
            by_clip[f.get("clip_id", "video"), f.get("period", 1)].append(f["video_time_ms"])
        scope = [{"period": p, "start_ms": min(ts), "end_ms": max(ts)+1000/rate} for (_, p), ts in by_clip.items()]
    if periods:
        divided = []
        for s in scope:
            bounds = sorted({s["start_ms"], s["end_ms"], *[t for r in periods for t in (r["start_ms"], r["end_ms"]) if s["start_ms"] < t < s["end_ms"]]})
            for a, b in zip(bounds, bounds[1:]):
                p = next((r["period"] for r in periods if r["start_ms"] <= (a+b)/2 < r["end_ms"]), s["period"])
                divided.append({"period": p, "start_ms": a, "end_ms": b})
        scope = divided
    start = context.get("start_ms", min((s["start_ms"] for s in scope), default=0))
    end = context.get("end_ms", max((s["end_ms"] for s in scope), default=0))
    if end < start or (raw and end == start):
        raise ValueError("Statistics end must follow start in video time")
    scope = _union([{**s, "start_ms": max(start, s["start_ms"]), "end_ms": min(end, s["end_ms"])}
                    for s in scope if max(start, s["start_ms"]) < min(end, s["end_ms"])])
    period_scopes = defaultdict(list)
    for i, span in enumerate(scope):
        period_scopes[span["period"]].append((span["start_ms"], span["end_ms"], i))
    scope_starts = {p: [s[0] for s in spans] for p, spans in period_scopes.items()}
    def selected_span(frame):
        p, t = frame.get("period", 1), frame["video_time_ms"]
        index = bisect_right(scope_starts.get(p, []), t)-1
        spans = period_scopes[p]
        return spans[index][2] if index >= 0 and t < spans[index][1] else None
    raw = [f for f in raw if selected_span(f) is not None]
    stamps = Counter(f["video_time_ms"] for f in raw)
    excluded = Counter()
    identities = context.get("identities", {})
    canonical = defaultdict(set)
    referee_tracks = set()
    for f in raw:
        for row in f["objects"]:
            if row.get("player_id"):
                canonical[row["track_id"]].add(row["player_id"])
            if row.get("inferred_role") == "probable_referee" or row.get("object_type") == "referee":
                referee_tracks.add(row["track_id"])
    conflicts = {key for key, ids in canonical.items() if len(ids) > 1 and not identities.get(key, {}).get("player_id")}
    tracks, observed, output = {}, defaultdict(list), []
    max_gap = min(750., max(250., 2500/rate))
    epoch, previous, cohort, previous_selection = 0, None, {}, None
    for f in raw:
        t, period = f["video_time_ms"], f.get("period", 1)
        selection = selected_span(f)
        episode = next((r for r in context.get("episodes", []) if r.get("period", 1) == period and r["start_ms"] <= t < r["end_ms"]), None)
        live = context.get("live_play")
        for interval in context.get("live_intervals", []):
            if interval.get("period", 1) == period and interval["start_ms"] <= t < interval["end_ms"]:
                live = interval.get("live_play") is True
        ground = context.get("ground_ball_reviewed") is True or bool(episode and episode.get("ground_ball_reviewed") is True)
        ground |= any(r.get("period", 1) == period and r["start_ms"] <= t < r["end_ms"] and r.get("reviewed") is True for r in context.get("ground_ball_intervals", []))
        line_quality = f.get("calibration_line_quality")
        if not isinstance(line_quality, dict):
            line_quality = {}
        cut = bool(f.get("camera_cut_suspected") or line_quality.get("projection_reanchored")) or stamps[t] != 1
        if previous and (cut or selection != previous_selection or f.get("clip_id") != previous.get("clip_id") or period != previous.get("period", 1)
                         or not 0 < t-previous["video_time_ms"] <= max_gap):
            epoch += 1
            cohort = {}
        segment = f"{f.get('clip_id', 'video')}:{epoch}"
        players, ball = [], None
        counts = Counter(o["track_id"] for o in f["objects"] if o.get("observed"))
        ball_count = sum(o.get("object_type") == "ball" and o.get("observed") is True and o.get("estimated") is not True for o in f["objects"])
        for row in f["objects"]:
            key = row["track_id"]
            if row.get("observed") is not True or row.get("estimated") is True or row.get("visibility") == "missing":
                continue
            obj = row.get("object_type")
            identity = identities.get(key, {})
            if key in referee_tracks and not (identity.get("player_id") or row.get("player_id")):
                excluded["referee"] += 1
                continue
            if obj not in ("player", "ball"):
                continue
            if obj == "player" and row.get("on_pitch") is False:
                excluded["off_pitch"] += 1
                continue
            if obj == "ball" and ball_count != 1:
                excluded["ambiguous_ball"] += 1
                continue
            if counts[key] != 1 or key in conflicts or stamps[t] != 1:
                excluded["ambiguous_identity_or_timestamp"] += 1
                continue
            if obj == "player":
                meta = {"id": key, "track_id": key, "player_id": identity.get("player_id", row.get("player_id")),
                        "team": identity.get("team_id", row.get("team_id") or row.get("inferred_team"))}
                tracks.setdefault(key, meta)
                observed[key].append({"t_ms": t, "segment_id": segment, "period": period})
            x, y = row.get("x_m"), row.get("y_m")
            calibration_status, projection_status = f.get("calibration_status"), row.get("projection_status")
            compatible = (projection_status == calibration_status if obj == "player" else
                          projection_status == calibration_status or calibration_status == "verified" and projection_status == "ball_ground_assumed")
            if (f.get("calibration_status") not in ("verified", "automatic")
                    or not compatible
                    or not finite(x) or not finite(y) or not 0 <= x <= source_length or not 0 <= y <= source_width):
                excluded["missing_pitch_mapping"] += 1
                continue
            errors = [v for v in (f.get("calibration_fit_rms_m"), f.get("calibration_check_max_m")) if finite(v) and v >= 0]
            error = row.get("position_uncertainty_m")
            if not finite(error):
                error = max(errors)+.25 if errors else None
            disagreement = row.get("position_disagreement_m", line_quality.get("projection_disagreement_m"))
            if error is not None and error >= 0 and finite(disagreement) and disagreement > 0:
                error += disagreement
            if error is None or error < 0 or error > 3:
                excluded["uncertain_pitch_mapping"] += 1
                continue
            error *= max(length/source_length, width/source_width)
            if error > 3:
                excluded["uncertain_pitch_mapping"] += 1
                continue
            x, y = x*length/source_length, y*width/source_width
            if context.get("flip_pitch"):
                x, y = length-x, width-y
            point = {"x": x, "y": y, "error_m": error,
                     "quality": "verified" if f["calibration_status"] == "verified" else "automatic_estimate"}
            if obj == "player":
                players.append({**meta, **point})
            elif ball is None:
                ball = {**point, "ground": ground}
            else:
                ball = None
                excluded["ambiguous_ball"] += 1
        # Coherent movement of the whole visible cohort is a calibration alarm,
        # not a run by every player. Break analysis continuity, never live tracks.
        now = {p["id"]: p for p in players}
        if previous and cohort and not cut:
            common = now.keys() & cohort.keys()
            dt = (t-previous["video_time_ms"])/1000
            if len(common) >= 5 and dt > 0 and median(hypot(now[k]["x"]-cohort[k]["x"], now[k]["y"]-cohort[k]["y"]) for k in common)/dt > 15:
                epoch += 1
                segment = f"{f.get('clip_id', 'video')}:{epoch}"
                cut = True
                excluded["cohort_mapping_jump"] += 1
                for ledger in observed.values():
                    if ledger and ledger[-1]["t_ms"] == t:
                        ledger[-1]["segment_id"] = segment
        output.append({"t_ms": t, "period": period, "segment_id": segment, "cut": cut, "live": live,
                       "phase": episode.get("phase") if episode else None, "players": players, "ball": ball,
                       "defensive_lines": (episode or {}).get("defensive_lines", context.get("defensive_lines", {})),
                       "team_lines": (episode or {}).get("team_lines", context.get("team_lines", {}))})
        previous, cohort, previous_selection = f, now, selection
    verified = context.get("pitch_dimensions_verified") is True or (
        report.get("dimensions_assumed") is False and (length, width) == (source_length, source_width))
    mapping_reviewed = context.get("calibration_reviewed") is True or any(f["players"] for f in output) and all(
        p["quality"] == "verified" for f in output for p in f["players"])
    directions = context.get("directions", {}) if context.get("orientation_confirmed") is True else {}
    return {"frames": output, "tracks": tracks, "length": length, "width": width, "max_gap_ms": max_gap,
            "directions": directions, "config": config, "scope_ms": sum(s["end_ms"]-s["start_ms"] for s in scope),
            "scope_spans": scope, "observed_by_track": dict(observed), "context": context,
            "metric_scale_verified": verified, "calibration_reviewed": mapping_reviewed,
            "orientation_confirmed": context.get("orientation_confirmed") is True,
            "quality": {"source_frames": len(raw), "usable_position_frames": sum(bool(f["players"]) for f in output),
                        "excluded_observations": dict(excluded), "dimensions_assumed": not verified,
                        "mapping_reviewed": mapping_reviewed, "orientation_confirmed": context.get("orientation_confirmed") is True,
                        "uncertainty_method": "Stored sensitivity or frame fitting/check residual + 0.25 m, plus any temporal projection disagreement."}}
