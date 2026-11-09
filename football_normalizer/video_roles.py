"""Infer review hints from sampled shirt colors and independently verified pitch positions."""

from bisect import bisect_left
from collections import Counter, defaultdict
from math import isfinite


VERSION = "fbrpro-video-role-hints-v1"
MIN_KIT_TRACKS = 3
MIN_KIT_SAMPLES = 3
MIN_KIT_SPAN_MS = 400
MAX_KIT_PROPAGATION_MS = 500
MIN_GOAL_SAMPLES = 5
MIN_GOAL_SPAN_MS = 800
MAX_GOAL_GAP_MS = 750


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _observed_player(row):
    return (row.get("object_type") == "player" and row.get("observed") is True
            and row.get("visibility") == "observed" and row.get("estimated") is False
            and isinstance(row.get("track_id"), str) and bool(row["track_id"]))


def _repeat(times, min_samples, min_span_ms):
    return len(times) >= min_samples and max(times)-min(times) >= min_span_ms


def _stable_color(samples):
    colors = defaultdict(set)
    visible = set()
    for time_ms, _, row in samples:
        if not _observed_player(row) or row.get("projection_status") == "off_pitch":
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
    if len(checked) < MIN_GOAL_SAMPLES:
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
            if position is None or time_ms not in near_times or (run and time_ms-run[-1][0] > MAX_GOAL_GAP_MS):
                if run:
                    runs.append(run)
                run = []
            if position is None or time_ms not in near_times:
                continue
            if not run or time_ms > run[-1][0]:
                run.append((time_ms, row))
        if run:
            runs.append(run)
        qualifying = [r for r in runs if len(r) >= MIN_GOAL_SAMPLES and r[-1][0]-r[0][0] >= MIN_GOAL_SPAN_MS]
        if qualifying:
            best = max(qualifying, key=lambda r: r[-1][0]-r[0][0])
            return end, [sample for r in qualifying for sample in r], {
                "verified_observations": sum(map(len, qualifying)), "observed_span_ms": best[-1][0]-best[0][0]}
    return None


def _annotate_shot(frames, length, width, usable_pitch):
    usable_pitch = usable_pitch and any(f.get("calibration_status") == "verified" for f in frames)
    tracks = defaultdict(list)
    for frame in frames:
        time_ms = frame["video_time_ms"]
        for row in frame["objects"]:
            if row.get("object_type") == "player":
                track = row.get("track_id")
                if isinstance(track, str) and track:
                    tracks[track].append((time_ms, frame, row))

    colors = {track: _stable_color(samples) for track, samples in tracks.items()}
    pair, team_status = _kit_pair(tracks, colors)
    counts = Counter()
    keepers = defaultdict(list)
    if pair:
        for track, samples in tracks.items():
            hue, times = colors[track]
            if hue in pair:
                for time_ms, _, row in samples:
                    if (_observed_player(row) and row.get("projection_status") != "off_pitch"
                            and row.get("kit_color_hint") in (None, hue)
                            and _nearest_sample(times, time_ms) <= MAX_KIT_PROPAGATION_MS):
                        row["inferred_team"] = f"kit:{hue}"
                        counts[hue] += 1
            elif hue and usable_pitch:
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
    return {"video_start_ms": frames[0]["video_time_ms"], "video_end_ms": frames[-1]["video_time_ms"],
            "team_status": team_status,
            "kit_groups": [{"inferred_team": f"kit:{hue}", "kit_color_hint": hue,
                            "annotated_observations": counts[hue]} for hue in pair],
            "goalkeeper_status": ("inferred" if confirmed else "ambiguous" if ambiguous else
                                  "verified_pitch_required" if not usable_pitch else "insufficient_evidence"),
            "goalkeepers": confirmed, "ambiguous_goal_ends": sorted(ambiguous)}


def annotate_video_roles(frames, *, coordinate_length_m, coordinate_width_m,
                         coordinate_frame="stadium_fixed", dimensions_assumed=False):
    """Annotate review hints in place and summarize supporting shots, without writing canonical IDs."""
    length, width = coordinate_length_m, coordinate_width_m
    usable_pitch = (coordinate_frame == "stadium_fixed"
                    and _number(length) and length > 33 and _number(width) and width > 16)
    clips = defaultdict(list)
    for frame in frames:
        for row in frame["objects"]:
            if row.get("object_type") == "player":
                row.update(inferred_team=None, inferred_role=None, inferred_goal_end=None)
        clips[frame["clip_id"]].append(frame)

    summaries = []
    for clip_id, clip_frames in clips.items():
        shot = []
        shots = []
        for frame in sorted(clip_frames, key=lambda frame: frame["video_time_ms"]):
            if shot and frame.get("camera_cut_suspected"):
                shots.append(_annotate_shot(shot, length, width, usable_pitch))
                shot = []
            shot.append(frame)
        shots.append(_annotate_shot(shot, length, width, usable_pitch))
        summaries.append({"clip_id": clip_id, "shots": shots})
    return {"version": VERSION,
            "team_policy": "Two recurring co-visible shirt groups per shot; officials can remain ambiguous.",
            "goalkeeper_policy": "Distinct kit and repeated verified positions near the same goal end; defended team is unknown.",
            "goalkeeper_pitch_dimensions_verified": usable_pitch and not dimensions_assumed,
            "goalkeeper_reference_dimensions": usable_pitch and dimensions_assumed, "clips": summaries}
