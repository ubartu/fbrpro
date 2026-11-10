"""Pitch continuity, explicit orientation review, and observable movement summaries.

Uncertainty is a geometric sensitivity estimate, not a calibrated confidence
interval. Automatic calibration never establishes team attacking direction.
"""
import argparse
from copy import deepcopy
import json
from math import hypot, isfinite
from pathlib import Path


VERSION = "fbrpro-pitch-geometry-v1"


def number(x):
    return type(x) in (int, float) and isfinite(x)


class PitchContinuity:
    """Compare fresh maps with the previous map transported by camera motion."""

    def __init__(self, cv2, np, length, width, prefix):
        from .video_motion import CameraMotion
        self.cv2, self.np = cv2, np
        self.length, self.width, self.prefix = length, width, prefix
        self.motion = CameraMotion(cv2, np)
        self.epoch = -1
        self.reset()

    def reset(self):
        self.motion.reset()
        self.previous = self.stamp = None
        self.rejected = 0
        self.boxes = ()
        self.epoch += 1

    def update(self, image, stamp, matrix, *, reviewed=False):
        cv2, np = self.cv2, self.np
        motion = self.motion.update(image, self.boxes)
        dt = stamp-self.stamp if self.stamp is not None else None
        self.stamp = stamp
        continuous = self.previous is not None and motion is not None and dt is not None and 0 < dt <= 750
        quality = {"version": VERSION, "segment_id": f"{self.prefix}:geometry:{self.epoch}",
                   "orientation_status": "reviewed" if reviewed else "unresolved",
                   "continuity_status": "new_segment", "mirror_corrected": False,
                   "drift_m": None, "usable": False}
        if not continuous and self.previous is not None:
            self.epoch += 1
            quality["segment_id"] = f"{self.prefix}:geometry:{self.epoch}"
        expected = self.previous @ np.linalg.inv(np.vstack((motion, [0, 0, 1.]))) if continuous else None
        if matrix is None:
            self.previous = expected
            quality["continuity_status"] = "mapping_unavailable"
            return None, quality
        if expected is not None:
            h, w = image.shape[:2]
            pixels = np.array([[[x, y]] for y in np.linspace(.2*h, .95*h, 7)
                               for x in np.linspace(.05*w, .95*w, 9)], np.float32)
            before = cv2.perspectiveTransform(pixels, expected)[:, 0]
            current = cv2.perspectiveTransform(pixels, matrix)[:, 0]
            inside = (np.isfinite(before).all(axis=1) & np.isfinite(current).all(axis=1)
                      & (before[:, 0] >= 0) & (before[:, 0] <= self.length)
                      & (before[:, 1] >= 0) & (before[:, 1] <= self.width))
            if inside.sum() >= 6 and np.ptp(before[inside, 0]) >= 5 and np.ptp(before[inside, 1]) >= 5:
                normal = float(np.quantile(np.linalg.norm(current[inside]-before[inside], axis=1), .8))
                mirrored = float(np.quantile(np.linalg.norm(
                    [self.length, self.width]-current[inside]-before[inside], axis=1), .8))
                # Reviewed maps define their own basis; automatic fits may be symmetric.
                if not reviewed and mirrored+.5 < normal:
                    matrix = np.array([[-1., 0, self.length], [0, -1., self.width], [0, 0, 1.]]) @ matrix
                    normal = mirrored
                    quality["mirror_corrected"] = True
                quality["drift_m"] = round(normal, 3)
                if normal > 3:
                    self.previous = expected
                    self.rejected += 1
                    if self.rejected >= 3:
                        # A stale reference must not reject all future maps.
                        # Reacquire a new, unreviewed segment on the next fit.
                        self.previous = None
                        self.epoch += 1
                    quality["continuity_status"] = "rejected_jump"
                    return None, quality
                quality["continuity_status"] = "camera_checked"
            else:
                self.epoch += 1
                quality["segment_id"] = f"{self.prefix}:geometry:{self.epoch}"
                quality["continuity_status"] = "insufficient_overlap"
        self.previous = matrix
        self.rejected = 0
        quality["usable"] = True
        return matrix, quality


def position_quality(np, row, calibration, line_quality, image_shape):
    """Local ground sensitivity plus marking residuals; no probability claim."""
    if not calibration or not number(row.get("x_m")) or row.get("object_type") != "player":
        return None
    base = [v for v in (calibration.rms_m, calibration.check_max_m) if number(v) and v >= 0]
    if not base:
        return None
    left, top, right, bottom = row["bbox_xyxy"]
    x, y = (left+right)/2, bottom
    pixel_error = max(2., .015*(bottom-top))
    if line_quality and number(line_quality.get("line_error_px")):
        resolution = line_quality.get("error_resolution", [640, 360])
        pixel_error += line_quality["line_error_px"]*max(image_shape[1]/resolution[0], image_shape[0]/resolution[1])
    origin = calibration.project((x, y))
    if origin is None:
        return None
    offsets = [calibration.project((x+dx, y+dy)) for dx, dy in
               ((pixel_error, 0), (-pixel_error, 0), (0, pixel_error), (0, -pixel_error))]
    if any(p is None for p in offsets):
        return None
    return round(max(base)+max(hypot(p[0]-origin[0], p[1]-origin[1]) for p in offsets), 3)


def segments(report):
    result = {}
    for frame in report["frames"]:
        geometry = frame.get("pitch_geometry") or {}
        key = geometry.get("segment_id")
        if not key:
            continue
        if key not in result:
            result[key] = {"id": key, "start_ms": frame["video_time_ms"], "end_ms": frame["video_time_ms"],
                           "period": frame.get("period", 1)}
        result[key]["end_ms"] = frame["video_time_ms"]
    return list(result.values())


def validate_review(review, report):
    if not isinstance(review, dict):
        raise ValueError("Pitch review must be an object")
    known = {s["id"] for s in segments(report)}
    settings = review.get("segments", {})
    directions = review.get("directions", {})
    if not isinstance(settings, dict) or set(settings)-known or not isinstance(directions, dict):
        raise ValueError("Pitch review contains unknown segments or invalid directions")
    for item in settings.values():
        if (not isinstance(item, dict) or item.get("orientation") not in ("normal", "flipped", "unresolved")
                or type(item.get("period")) is not int or not 1 <= item["period"] <= 5
                or type(item.get("live_play")) is not bool):
            raise ValueError("Review needs orientation, a period from 1 to 5, and live-play confirmation")
    for period, teams in directions.items():
        if period not in {str(p) for p in range(1, 6)} or not isinstance(teams, dict):
            raise ValueError("Attacking directions must be stored by period")
        if any(not isinstance(team, str) or not team or type(v) is not int or v not in (-1, 1)
               for team, v in teams.items()):
            raise ValueError("Each team direction must be 1 or -1")
        if {"kit:team_a", "kit:team_b"} <= teams.keys() and teams["kit:team_a"] == teams["kit:team_b"]:
            raise ValueError("Team A and Team B must attack opposite goals in the same period")
    return {"segments": settings, "directions": directions}


def reviewed_report(report, review=None):
    """Apply review idempotently to a copy, retaining source coordinates."""
    if (not isinstance(report, dict) or report.get("version") != "fbrpro-video-tracking-v1"
            or not isinstance(report.get("frames"), list)
            or not all(number(report.get(k)) and report[k] > 0 for k in ("coordinate_length_m", "coordinate_width_m"))):
        raise ValueError("Choose a tracking report with pitch dimensions")
    review = validate_review(review if review is not None else report.get("pitch_review", {}), report)
    result = deepcopy(report)
    length, width = result["coordinate_length_m"], result["coordinate_width_m"]
    for frame in result["frames"]:
        g = frame.get("pitch_geometry") or {}
        item = review["segments"].get(g.get("segment_id"), {})
        g["reviewed_orientation"] = item.get("orientation", "unresolved")
        g["live_play"] = item.get("live_play", False)
        if "source_image_to_pitch" not in g:
            g["source_image_to_pitch"] = g.get("image_to_pitch")
        matrix = g.get("source_image_to_pitch")
        if matrix is not None and item.get("orientation") == "flipped":
            g["image_to_pitch"] = [[-matrix[0][i]+length*matrix[2][i] for i in range(3)],
                                   [-matrix[1][i]+width*matrix[2][i] for i in range(3)], list(matrix[2])]
        else:
            g["image_to_pitch"] = matrix
        if g:
            frame["pitch_geometry"] = g
        frame["movement_period"] = item.get("period", frame.get("period", 1))
        for row in frame["objects"]:
            if "source_pitch_position" not in row:
                row["source_pitch_position"] = [row.get("x_m"), row.get("y_m")]
            x, y = row["source_pitch_position"]
            if item.get("orientation") == "flipped" and number(x) and number(y):
                x, y = length-x, width-y
            row["x_m"], row["y_m"] = x, y
    result["pitch_review"] = review
    result["pitch_segments"] = segments(result)
    result["spatial_summary"] = spatial_summary(result)
    return result


def spatial_summary(report):
    """Time-weighted observed intervals; no gap filling or match extrapolation."""
    length, width = report["coordinate_length_m"], report["coordinate_width_m"]
    directions = report.get("pitch_review", {}).get("directions", {})
    profiles, previous, windows = {}, None, {}
    rate = report.get("sample_fps", 10)
    if not number(rate) or rate <= 0:
        raise ValueError("Sample rate must be positive")
    max_gap = min(750, max(250, 2.5*1000/rate))
    def flush(key):
        window = windows.pop(key, None)
        if window is None:
            return
        start, end, a, b, direction = window
        p = profiles[key]
        dx, dy = (b["x_m"]-a["x_m"])*direction, b["y_m"]-a["y_m"]
        noise = a["position_uncertainty_m"]+b["position_uncertainty_m"]
        if abs(dx) > noise:
            p["forward_m" if dx > 0 else "backward_m"] += abs(dx)
        if abs(dy) > noise:
            p["lateral_m"] += abs(dy)
        if abs(dx) <= noise and abs(dy) <= noise:
            p["direction_uncertain_ms"] += end-start
    def seen(row):
        return (row.get("object_type") == "player" and row.get("observed") is True
                and row.get("estimated") is not True and row.get("visibility") != "missing"
                and row.get("inferred_role") != "probable_referee")
    def trusted(frame, row):
        g = frame.get("pitch_geometry") or {}
        return (g.get("usable") is True and g.get("reviewed_orientation") in ("normal", "flipped")
                and g.get("live_play") is True and row.get("projection_status") in ("verified", "automatic")
                and all(number(row.get(k)) for k in ("x_m", "y_m", "position_uncertainty_m"))
                and 0 <= row["position_uncertainty_m"] <= 3
                and 0 <= row["x_m"] <= length and 0 <= row["y_m"] <= width)
    def zone(row):
        x, y, error = row["x_m"], row["y_m"], row["position_uncertainty_m"]
        if any(abs(x-line) <= error for line in (length/3, 2*length/3)) or any(
                abs(y-line) <= error for line in (width/3, 2*width/3)):
            return None
        return min(2, int(y/(width/3)))*3+min(2, int(x/(length/3)))
    for frame in sorted(report["frames"], key=lambda f: f["video_time_ms"]):
        extended = set()
        current = {}
        duplicates = set()
        for row in filter(seen, frame["objects"]):
            key = row["track_id"]
            if key in current:
                duplicates.add(key)
            current[key] = row
            if key not in profiles:
                profiles[key] = {"track_id": key, "player_id": row.get("player_id"),
                                 "team": row.get("team_id") or row.get("inferred_team"),
                                 "observed_interval_ms": 0, "usable_ms": 0, "zone_resolved_ms": 0,
                                 "boundary_uncertain_ms": 0, "zones_ms": [0]*9,
                                 "directional_usable_ms": 0, "forward_m": 0., "backward_m": 0.,
                                 "lateral_m": 0., "direction_uncertain_ms": 0}
        if previous:
            before, earlier, prior_duplicates = previous
            dt = frame["video_time_ms"]-before["video_time_ms"]
            g, old = frame.get("pitch_geometry") or {}, before.get("pitch_geometry") or {}
            continuous = (0 < dt <= max_gap and frame.get("clip_id") == before.get("clip_id")
                          and not frame.get("camera_cut_suspected") and frame.get("period") == before.get("period")
                          and frame.get("movement_period") == before.get("movement_period")
                          and g.get("segment_id") and g.get("segment_id") == old.get("segment_id"))
            if continuous:
                for key in current.keys() & earlier.keys() - duplicates - prior_duplicates:
                    a, b, p = earlier[key], current[key], profiles[key]
                    p["observed_interval_ms"] += dt
                    if (a.get("player_id") != b.get("player_id") or a.get("team_id") != b.get("team_id")
                            or not trusted(before, a) or not trusted(frame, b)):
                        continue
                    if hypot(b["x_m"]-a["x_m"], b["y_m"]-a["y_m"]) > 15*dt/1000:
                        continue
                    p["usable_ms"] += dt
                    za, zb = zone(a), zone(b)
                    if za is not None and za == zb:
                        p["zones_ms"][za] += dt
                        p["zone_resolved_ms"] += dt
                    else:
                        p["boundary_uncertain_ms"] += dt
                    ta, tb = a.get("team_id") or a.get("inferred_team"), b.get("team_id") or b.get("inferred_team")
                    direction = directions.get(str(frame.get("movement_period", frame.get("period"))), {}).get(tb)
                    if ta != tb or direction not in (-1, 1):
                        continue
                    p["directional_usable_ms"] += dt
                    start, _, origin, _, _ = windows.get(key, (before["video_time_ms"], 0, a, None, direction))
                    windows[key] = (start, frame["video_time_ms"], origin, b, direction)
                    extended.add(key)
                    if frame["video_time_ms"]-start >= 1000:
                        flush(key)
        for key in list(windows):
            if key not in extended:
                flush(key)
        previous = frame, current, duplicates
    for key in list(windows):
        flush(key)
    for p in profiles.values():
        p["coverage_pct"] = round(100*p["usable_ms"]/p["observed_interval_ms"], 2) if p["observed_interval_ms"] else None
        p["zones_pct"] = [round(100*t/p["usable_ms"], 2) for t in p["zones_ms"]] if p["usable_ms"] else [None]*9
        for k in ("forward_m", "backward_m", "lateral_m"):
            p[k] = round(p[k], 3) if p["directional_usable_ms"] else None
    return {"version": VERSION, "scope": "observable_contiguous_track_intervals", "players": list(profiles.values()),
            "quality_policy": {"max_gap_ms": max_gap, "max_position_sensitivity_m": 3,
                               "max_interval_speed_m_s": 15, "movement_window_ms": 1000},
            "dimensions_assumed": report.get("dimensions_assumed", False),
            "notes": ["Coverage is relative to contiguous observed track time, not match minutes.",
                      "Zones use the fixed pitch grid: top row first, left to right; uncertain boundary time is separate.",
                      "Movement uses contiguous windows up to approximately one second (and shorter tails), exceeding the endpoint geometric error estimate; not total distance or speed.",
                      "Uncertainty is a sensitivity estimate, not a calibrated confidence interval."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = reviewed_report(json.loads(args.tracking.read_text()),
                                 json.loads(args.review.read_text()) if args.review else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"video_geometry: {exc}\n")


if __name__ == "__main__":
    main()
