"""Infer review hints from sampled shirt colors and independently verified pitch positions."""

from bisect import bisect_left
from collections import Counter, defaultdict
from math import isfinite

from .video_kits import KIT_RADIUS, distance, local_kit_groups, shirt_profiles, team_groups


VERSION = "fbrpro-video-role-hints-v1"
MIN_KIT_TRACKS = 3
MIN_KIT_SAMPLES = 3
MIN_KIT_SPAN_MS = 400
MAX_KIT_PROPAGATION_MS = 500
MIN_ROLE_SAMPLES = 5
MIN_ROLE_SPAN_MS = 800
MAX_ROLE_GAP_MS = 750


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _observed_player(row):
    return (row.get("object_type") == "player" and row.get("observed") is True
            and row.get("visibility") == "observed" and row.get("estimated") is False
            and isinstance(row.get("track_id"), str) and bool(row["track_id"]))


def _repeat(times, min_samples, min_span_ms):
    return len(times) >= min_samples and max(times)-min(times) >= min_span_ms


def _shirt_visible(row):
    return (_observed_player(row)
            and (row.get("projection_status") != "off_pitch" or row.get("on_pitch") is True))


def _stable_color(samples):
    colors = defaultdict(set)
    visible = set()
    for time_ms, _, row in samples:
        if not _shirt_visible(row):
            continue
        visible.add(time_ms)
        hue = row.get("kit_color_hint")
        if isinstance(hue, str) and hue:
            colors[hue].add(time_ms)
    if not colors:
        return None, []
    hue, times = max(colors.items(), key=lambda entry: len(entry[1]))
    colored = sum(map(len, colors.values()))
    if (not _repeat(times, MIN_KIT_SAMPLES, MIN_KIT_SPAN_MS)
            or len(times)*5 < colored*4 or len(times)*5 < len(visible)*2):
        return None, []
    return hue, sorted(times)


def _nearest_sample(times, time_ms):
    index = bisect_left(times, time_ms)
    return min(abs(t-time_ms) for t in times[max(0, index-1):index+1])


def _kit_pair(tracks, colors):
    by_time = defaultdict(lambda: defaultdict(set))
    for track, (hue, times) in colors.items():
        if hue:
            for time_ms in times:
                by_time[time_ms][hue].add(track)
    supported = defaultdict(set)
    for time_ms, same_frame in by_time.items():
        for hue, track_ids in same_frame.items():
            if len(track_ids) >= MIN_KIT_TRACKS:
                supported[hue].add(time_ms)
    recurring = sorted(hue for hue, times in supported.items()
                       if _repeat(times, MIN_KIT_SAMPLES, MIN_KIT_SPAN_MS))
    if len(recurring) > 2:
        return [], "multiple_recurring_kits"
    if len(recurring) < 2 or not _repeat(supported[recurring[0]] & supported[recurring[1]],
                                         MIN_KIT_SAMPLES, MIN_KIT_SPAN_MS):
        return [], "two_co_visible_kits_unconfirmed"

    reviewed_teams = defaultdict(set)
    for track, samples in tracks.items():
        hue, _ = colors[track]
        if hue in recurring:
            reviewed_teams[hue].update(row["team_id"] for _, _, row in samples if row.get("team_id"))
    a, b = (reviewed_teams[hue] for hue in recurring)
    if len(a) > 1 or len(b) > 1 or a & b:
        return [], "reviewed_teams_conflict_with_kits"
    return recurring, "inferred"


def _goal_depth(frame, row, length, width):
    if (not _observed_player(row) or frame.get("calibration_status") != "verified"
            or row.get("projection_status") != "verified"):
        return None
    x, y = row.get("x_m"), row.get("y_m")
    if not (_number(x) and _number(y) and 0 <= x <= length and 0 <= y <= width):
        return None
    return (x, length-x), abs(y-width/2)


def _goal_candidate(samples, length, width):
    positions = [(time_ms, _goal_depth(frame, row, length, width), row) for time_ms, frame, row in samples]
    checked = [position for _, position, _ in positions if position is not None]
    if len(checked) < MIN_ROLE_SAMPLES:
        return None

    for index, end in enumerate(("low_x", "high_x")):
        # A track that also ventures beyond the penalty area is weak evidence.
        if any(depth[index] > length*(16.5/105) for depth, _ in checked):
            continue
        near = [(time_ms, row) for time_ms, position, row in positions if position is not None
                and position[0][index] <= length*(8/105) and position[1] <= width*(8/68)]
        if len(near)*5 < len(checked)*4:
            continue
        runs = []
        run = []
        near_times = {time_ms for time_ms, _ in near}
        for time_ms, position, row in positions:
            if position is None or time_ms not in near_times or (run and time_ms-run[-1][0] > MAX_ROLE_GAP_MS):
                if run:
                    runs.append(run)
                run = []
            if position is None or time_ms not in near_times:
                continue
            if not run or time_ms > run[-1][0]:
                run.append((time_ms, row))
        if run:
            runs.append(run)
        qualifying = [r for r in runs if len(r) >= MIN_ROLE_SAMPLES and r[-1][0]-r[0][0] >= MIN_ROLE_SPAN_MS]
        if qualifying:
            best = max(qualifying, key=lambda r: r[-1][0]-r[0][0])
            return end, [sample for r in qualifying for sample in r], {
                "verified_observations": sum(map(len, qualifying)), "observed_span_ms": best[-1][0]-best[0][0]}
    return None


def _shirt_samples(tracks):
    return {track: {ms: row["_shirt_appearance"] for ms, _, row in rows
                    if "_shirt_appearance" in row and _shirt_visible(row)}
            for track, rows in tracks.items()}


def _shot_tracks(frames):
    tracks = defaultdict(list)
    for frame in frames:
        time_ms = frame["video_time_ms"]
        for row in frame["objects"]:
            if row.get("object_type") == "player":
                track = row.get("track_id")
                if isinstance(track, str) and track:
                    tracks[track].append((time_ms, frame, row))
    return tracks


def _touchline_evidence(samples, shirt_times, length, width):
    checked = []
    for stamp, frame, row in samples:
        x, y = row.get("x_m"), row.get("y_m")
        if (_observed_player(row) and frame.get("calibration_status") in ("verified", "automatic")
                and row.get("projection_status") in ("verified", "automatic")
                and _number(x) and _number(y) and 0 <= x <= length and 0 <= y <= width):
            checked.append((stamp, y))
    if len(checked) < MIN_ROLE_SAMPLES:
        return None
    for side in ("low_y", "high_y"):
        times = [stamp for stamp, y in checked
                 if (y if side == "low_y" else width-y) <= width*.03 and stamp in shirt_times]
        if len(times) < .85*len(checked):
            continue
        runs, run = [], []
        for stamp in times:
            if run and stamp-run[-1] > MAX_ROLE_GAP_MS:
                runs.append(run)
                run = []
            run.append(stamp)
        runs.append(run)
        qualifying = [run for run in runs if _repeat(run, MIN_ROLE_SAMPLES, MIN_ROLE_SPAN_MS)]
        if not qualifying:
            continue
        evidence = sorted(stamp for run in qualifying for stamp in run)
        return side, evidence, max(run[-1]-run[0] for run in qualifying)
    return None


def _annotate_referees(tracks, reference, length, width):
    """Flag distinct shirts with sustained touchline evidence for review."""
    if reference is None:
        return []
    import numpy as np

    profiles = shirt_profiles(np, _shirt_samples(tracks))
    candidates = []
    for track, (appearance, shirt_times) in profiles.items():
        samples = tracks[track]
        if any(row.get("player_id") or row.get("team_id") or row.get("inferred_team")
               or row.get("inferred_role") == "probable_goalkeeper" for _, _, row in samples):
            continue
        if distance(np, reference, appearance).min() <= KIT_RADIUS+6:
            continue
        evidence = _touchline_evidence(samples, shirt_times, length, width)
        if evidence is None:
            continue
        side, times, span = evidence
        annotated = 0
        for stamp, _, row in samples:
            if (_shirt_visible(row) and stamp in shirt_times
                    and _nearest_sample(times, stamp) <= MAX_KIT_PROPAGATION_MS):
                row["inferred_role"] = "probable_referee"
                row["inferred_referee_role"] = "assistant"
                annotated += 1
        candidates.append({"track_id": track, "inferred_role": "probable_referee",
                           "touchline": side, "annotated_observations": annotated,
                           "observed_span_ms": span})
    return candidates


def _annotate_shot(frames, length, width, usable_pitch, reference, shirt_groups):
    usable_pitch = usable_pitch and any(f.get("calibration_status") == "verified" for f in frames)
    tracks = _shot_tracks(frames)

    colors = {track: _stable_color(samples) for track, samples in tracks.items()}
    if shirt_groups:
        appearances = _shirt_samples(tracks)
        assignments, next_reference, team_status = team_groups(appearances, reference)
        pair = ("team_a", "team_b") if assignments else ()
        reviewed = defaultdict(set)
        for track, (group, _) in assignments.items():
            reviewed[group].update(row["team_id"] for _, _, row in tracks[track] if row.get("team_id"))
        a, b = reviewed["team_a"], reviewed["team_b"]
        if len(a) > 1 or len(b) > 1 or a & b:
            assignments, pair, team_status = {}, (), "reviewed_teams_conflict_with_kits"
        else:
            reference = next_reference
    else:
        pair, team_status = _kit_pair(tracks, colors)
        assignments = {track: (hue, times) for track, (hue, times) in colors.items() if hue in pair}
    counts = Counter()
    keepers = defaultdict(list)
    team_hues = {colors[track][0] for track in assignments}
    if pair:
        for track, samples in tracks.items():
            hue, _ = colors[track]
            if track in assignments:
                group, times = assignments[track]
                ordered_times = sorted(times)
                for time_ms, _, row in samples:
                    if shirt_groups:
                        consistent = time_ms in times or "_shirt_appearance" not in row
                    else:
                        consistent = row.get("kit_color_hint") in (None, hue)
                    if (_shirt_visible(row)
                            and consistent
                            and _nearest_sample(ordered_times, time_ms) <= MAX_KIT_PROPAGATION_MS):
                        row["inferred_team"] = f"kit:{group}"
                        counts[group] += 1
            elif hue and hue not in team_hues and usable_pitch:
                candidate = _goal_candidate(samples, length, width)
                if candidate:
                    end, evidence, summary = candidate
                    keepers[end].append((track, hue, evidence, summary))

    confirmed = []
    ambiguous = []
    for end, candidates in keepers.items():
        if len(candidates) != 1:
            ambiguous.append(end)
            continue
        track, hue, evidence, summary = candidates[0]
        for _, row in evidence:
            row["inferred_role"] = "probable_goalkeeper"
            row["inferred_goal_end"] = end
        confirmed.append({"track_id": track, "inferred_goal_end": end, "kit_color_hint": hue, **summary})
    referees = _annotate_referees(tracks, reference, length, width) if shirt_groups else []
    if shirt_groups and pair:
        protected = {r["track_id"] for r in referees + confirmed}
        for track, samples in tracks.items():
            if track in protected or any(row.get("team_id") or row.get("player_id") for _, _, row in samples):
                continue
            local = local_kit_groups(appearances[track], reference)
            for stamp, _, row in samples:
                if stamp in local and _shirt_visible(row):
                    prior = row.get("inferred_team")
                    if prior:
                        counts[prior.removeprefix("kit:")] -= 1
                    group = local[stamp]
                    row["inferred_team"] = f"kit:{group}"
                    counts[group] += 1
    return {"video_start_ms": frames[0]["video_time_ms"], "video_end_ms": frames[-1]["video_time_ms"],
            "team_status": team_status,
            "kit_groups": [{"inferred_team": f"kit:{group}", "kit_color_hint": None if shirt_groups else group,
                            "annotated_observations": counts[group]} for group in pair],
            "goalkeeper_status": ("inferred" if confirmed else "ambiguous" if ambiguous else
                                  "verified_pitch_required" if not usable_pitch else "insufficient_evidence"),
            "goalkeepers": confirmed, "ambiguous_goal_ends": sorted(ambiguous),
            "referees": referees}, reference


def annotate_video_roles(frames, *, coordinate_length_m, coordinate_width_m,
                         coordinate_frame="stadium_fixed", dimensions_assumed=False, shirt_groups=False):
    """Annotate review hints in place and summarize supporting shots, without writing canonical IDs."""
    length, width = coordinate_length_m, coordinate_width_m
    usable_pitch = (coordinate_frame == "stadium_fixed"
                    and _number(length) and length > 33 and _number(width) and width > 16)
    clips = defaultdict(list)
    for frame in frames:
        for row in frame["objects"]:
            if row.get("object_type") == "player":
                row.update(inferred_team=None, inferred_role=None, inferred_goal_end=None, inferred_referee_role=None)
        clips[frame["clip_id"]].append(frame)

    grouped_shots = {}
    for clip_id, clip_frames in clips.items():
        shot = []
        shots = []
        for frame in sorted(clip_frames, key=lambda frame: frame["video_time_ms"]):
            if shot and frame.get("camera_cut_suspected"):
                shots.append(shot)
                shot = []
            shot.append(frame)
        shots.append(shot)
        grouped_shots[clip_id] = shots
    reference = None
    if shirt_groups:
        for shots in grouped_shots.values():
            for shot in shots:
                tracks = _shot_tracks(shot)
                samples = _shirt_samples(tracks)
                _, reference, status = team_groups(samples, None)
                if status == "multiple_recurring_kits":
                    central = {track: appearances for track, appearances in samples.items()
                               if _touchline_evidence(tracks[track], set(appearances), length, width) is None}
                    _, reference, _ = team_groups(central, None)
                if reference is not None:
                    break
            if reference is not None:
                break
    summaries = []
    for clip_id, shots in grouped_shots.items():
        annotated = []
        for shot in shots:
            summary, reference = _annotate_shot(shot, length, width, usable_pitch, reference, shirt_groups)
            annotated.append(summary)
        clip_frames = clips[clip_id]
        for frame in clip_frames:
            for row in frame["objects"]:
                row.pop("_shirt_appearance", None)
        summaries.append({"clip_id": clip_id, "shots": annotated})
    return {"version": VERSION,
            "team_policy": "Shirt groups established in a wide view and reused across the video's shots; pitch projection is assessed separately.",
            "goalkeeper_policy": "Distinct kit and repeated verified positions near the same goal end; defended team is unknown.",
            "referee_policy": "Distinct shirt and repeated touchline positions identify assistant referee candidates.",
            "goalkeeper_pitch_dimensions_verified": usable_pitch and not dimensions_assumed,
            "goalkeeper_reference_dimensions": usable_pitch and dimensions_assumed, "clips": summaries}
