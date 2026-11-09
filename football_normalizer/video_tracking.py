"""Run local football detection and pitch projection over selected video clips.

Uses OpenCV Zoo's YOLOX-s COCO ONNX export. Its person class yields anonymous
human candidates, which can include officials until a reviewer identifies them.
Pitch calibration is valid only within the operator-supplied camera interval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from .video import VERSION as PLAN_VERSION, segments_from
from .video_auto_calibration import PitchModel
from .video_inference import load_inference
from .video_motion import CameraMotion
from .video_pitch import detect_pitch
from .video_roles import annotate_video_roles

VERSION = "fbrpro-video-tracking-v1"
TRACK_ID_SCOPE = "pitch-v2"
MAX_FULL_VIDEO_MS = 6 * 60 * 60 * 1000


def vision():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ValueError("Tracking needs numpy and opencv-python-headless (install the tracking extra)") from exc
    return cv2, np


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Detection:
    object_type: str
    confidence: float
    box: tuple[float, float, float, float]
    kit_color: str | None = None

    def anchor(self):
        x1, y1, x2, y2 = self.box
        return (x1+x2)/2, (y1+y2)/2 if self.object_type == "ball" else y2


class YoloX:
    """Run the 640x640 OpenCV Zoo YOLOX model (person and sports ball)."""
    def __init__(self, model, confidence, *, inference_backend="opencv", inference_cache=None):
        cv2, np = vision()
        self.cv2, self.np = cv2, np
        self.confidence = confidence
        self.engine = load_inference(cv2, model, inference_backend, "the YOLOX detector", inference_cache)
        grids, strides = [], []
        for step in (8, 16, 32):
            y, x = np.indices((640//step, 640//step))
            grids.append(np.column_stack((x.ravel(), y.ravel())))
            strides.append(np.full((x.size, 1), step))
        self.grid = np.concatenate(grids)
        self.stride = np.concatenate(strides)

    def _detect(self, frame, *, people_only=False):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        ratio = min(640/w, 640/h)
        resized = cv2.resize(frame, (max(1, int(w*ratio)), max(1, int(h*ratio))))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        # Zoo demo uses RGB, then HWC -> CHW with raw 0..255 floats.
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0, swapRB=True)
        pred = np.asarray(self.engine.forward(blob)).reshape(-1, 85)
        if len(pred) != len(self.grid):
            raise ValueError("Expected the OpenCV Zoo 640x640 YOLOX COCO ONNX export")
        classes = np.argmax(pred[:, 5:], axis=1)
        conf = pred[:, 4] * pred[np.arange(len(pred)), classes+5]
        result = []
        classes_to_keep = ((0, "player"),) if people_only else ((0, "player"), (32, "ball"))
        for index, object_type in classes_to_keep:
            mask = (classes == index) & np.isfinite(conf) & (conf >= self.confidence)
            centers = (pred[mask, :2] + self.grid[mask]) * self.stride[mask] / ratio
            sizes = np.exp(np.clip(pred[mask, 2:4], -15, 15)) * self.stride[mask] / ratio
            corners = np.concatenate((centers - sizes/2, centers + sizes/2), axis=1)
            corners[:, ::2] = corners[:, ::2].clip(0, w - 1)
            corners[:, 1::2] = corners[:, 1::2].clip(0, h - 1)
            sizes = corners[:, 2:] - corners[:, :2]
            xywh = np.concatenate((corners[:, :2], sizes), axis=1)
            valid = np.isfinite(xywh).all(axis=1) & (sizes > 0).all(axis=1)
            boxes, scores, xywh = corners[valid], conf[mask][valid], xywh[valid]
            if len(scores):
                keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.confidence, .45)
                for j in np.asarray(keep).reshape(-1):
                    result.append(Detection(object_type, float(scores[j]), tuple(map(float, boxes[j]))))
        # A single visible ball is tracked; ambiguous ball detections are candidates.
        players = [d for d in result if d.object_type == "player"]
        balls = [d for d in result if d.object_type == "ball"]
        return players + ([max(balls, key=lambda d: d.confidence)] if balls else [])

    def detect(self, frame, region=None, *, accepts_person=None):
        """Combine the full image and overlapping, tighter views of a wide pitch."""
        def allowed(detection):
            return (detection.object_type != "player" or accepts_person is None
                    or accepts_person(detection.anchor()))

        detections = [d for d in self._detect(frame) if allowed(d)]
        if not region:
            return detections
        height, width = frame.shape[:2]
        xs, ys = zip(*region)
        padding = round(min(width, height)*.08)
        left, right = max(0, int(min(xs))-padding), min(width, int(max(xs))+padding)
        top, bottom = max(0, int(min(ys))-padding), min(height, int(max(ys))+padding)
        span = right-left
        tile_width = round(span*.61)
        if span < 640 or max(tile_width, bottom-top) >= .88*max(width, height) or bottom <= top:
            return detections
        people = [d for d in detections if d.object_type == "player"]
        balls = [d for d in detections if d.object_type == "ball"]
        for x in (left, right-tile_width):
            tile = frame[top:bottom, x:x+tile_width]
            for d in self._detect(tile, people_only=True):
                x1, y1, x2, y2 = d.box
                if (x > 0 and x1 < 4) or (x+tile_width < width and x2 > tile_width-5):
                    continue  # The other view covers this crop edge.
                shifted = replace(d, box=(x1+x, y1+top, x2+x, y2+top))
                if allowed(shifted):
                    people.append(shifted)
        if not people:
            return balls
        # Prefer a tighter high-confidence box to a redundant larger box.
        boxes = [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2 in (d.box for d in people)]
        indices = self.cv2.dnn.NMSBoxes(boxes, [d.confidence for d in people], self.confidence, .38)
        return [people[i] for i in self.np.asarray(indices).reshape(-1)] + balls


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0


@dataclass
class Track:
    id: str
    last: Detection
    seen_ms: int
    missing: bool = False
    previous: tuple[Detection, int] | None = None
    kit_sample: str | None = None


class Tracker:
    """Conservative one-to-one image association. No inferred missing positions."""
    def __init__(self, clip_id, max_gap_ms=800, *, id_scope=None):
        self.id_prefix = f"{clip_id}:{id_scope}" if id_scope else clip_id
        self.max_gap_ms = max_gap_ms
        self.active = []
        self.next_id = 1

    def reset(self):
        self.active = []

    def assign(self, options, detection_count):
        return sorted(options)

    def update(self, detections, time_ms, camera_transform=None):
        self.active = [t for t in self.active if time_ms-t.seen_ms <= self.max_gap_ms]
        if camera_transform is not None:
            def warp(detection):
                x1, y1, x2, y2 = detection.box
                points = [camera_transform @ (x, y, 1) for x in (x1, x2) for y in (y1, y2)]
                xs, ys = zip(*points)
                return replace(detection, box=(min(xs), min(ys), max(xs), max(ys)))

            for track in self.active:
                track.last = warp(track.last)
                if track.previous:
                    previous, seen = track.previous
                    track.previous = (warp(previous), seen)
        options = []
        for i, t in enumerate(self.active):
            dx = dy = 0.0
            elapsed = time_ms-t.seen_ms
            if t.previous and t.seen_ms > t.previous[1]:
                ax, ay = t.last.anchor()
                bx, by = t.previous[0].anchor()
                factor = min(2.0, elapsed/(t.seen_ms-t.previous[1]))
                dx, dy = (ax-bx)*factor, (ay-by)*factor
            b = t.last.box
            predicted = (b[0]+dx, b[1]+dy, b[2]+dx, b[3]+dy)
            scale = max(12, b[2]-b[0], b[3]-b[1])
            if t.last.object_type == "ball":
                scale = max(scale, 35)
            for j, d in enumerate(detections):
                if d.object_type != t.last.object_type:
                    continue
                da, db = d.anchor()
                ta, tb = t.last.anchor()
                distance = math.hypot(da-ta-dx, db-tb-dy)/scale
                overlap = iou(predicted, d.box)
                if overlap > .08 or distance <= .80:
                    kit_conflict = (t.kit_sample and t.kit_sample == t.last.kit_color
                                    and d.kit_color and t.kit_sample != d.kit_color)
                    options.append((distance + (1-overlap)*.60 + (.45 if kit_conflict else 0), i, j))
        used_tracks, used_detections = set(), set()
        for _, i, j in self.assign(options, len(detections)):
            if i in used_tracks or j in used_detections:
                continue
            t = self.active[i]
            t.previous = (t.last, t.seen_ms)
            t.kit_sample = t.last.kit_color
            t.last, t.seen_ms = detections[j], time_ms
            used_tracks.add(i)
            used_detections.add(j)
        for i, t in enumerate(self.active):
            t.missing = i not in used_tracks
        for j, detection in enumerate(detections):
            if j not in used_detections:
                self.active.append(Track(f"{self.id_prefix}:{self.next_id}", detection, time_ms))
                self.next_id += 1
        return self.active


@dataclass
class Calibration:
    segment_id: str
    start_ms: int | None
    end_ms: int | None
    matrix: object
    status: str
    rms_m: float | None
    check_max_m: float | None
    landmarks: tuple | None = None

    def applies(self, segment_id, time_ms):
        return (self.segment_id == segment_id and (self.start_ms is None
                or self.start_ms <= time_ms < self.end_ms))

    def project(self, point):
        u, v, w = self.matrix @ (*point, 1)
        if abs(w) < 1e-9:
            return None
        x, y = float(u/w), float(v/w)
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def fit_projection(cv2, np, pix, xy, cpx, cxy):
    matrix, _ = cv2.findHomography(pix, xy, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Could not fit a stable pitch mapping")

    def errors(src, expected):
        homogeneous = np.column_stack((src, np.ones(len(src)))) @ matrix.T
        if (np.abs(homogeneous[:, 2]) < 1e-9).any():
            raise ValueError("Projection is undefined at a landmark")
        return np.linalg.norm(homogeneous[:, :2]/homogeneous[:, 2:] - expected, axis=1)

    residuals = errors(pix, xy)
    rms = float(np.sqrt(np.mean(residuals**2)))
    check = float(np.max(errors(cpx, cxy))) if cpx is not None else None
    return matrix, rms, float(max(residuals)), check


def calibration_from(data, max_error_m):
    """Fit planar camera mappings from pixel-to-metre landmarks."""
    cv2, np = vision()
    if not isinstance(data, dict):
        raise ValueError("Calibration must be a JSON object")
    length = data.get("coordinate_length_m")
    width = data.get("coordinate_width_m")
    if not all(finite(v) and v > 0 for v in (length, width)):
        raise ValueError("Calibration needs positive coordinate_length_m and coordinate_width_m")
    rows = data.get("segments")
    if not isinstance(rows, list):
        raise ValueError("Calibration needs a segments array")
    result = []

    def points(raw, least):
        a = np.asarray(raw, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < least or not np.isfinite(a).all():
            raise ValueError("Invalid calibration landmark array")
        return a

    for row in rows:
        id_ = row.get("segment_id")
        if not isinstance(id_, str) or not id_:
            raise ValueError("Calibration segment_id must be nonempty")
        start, end = row.get("video_start_ms"), row.get("video_end_ms")
        if (start is not None or end is not None) and not (type(start) is type(end) is int and 0 <= start < end):
            raise ValueError(f"{id_}: supply both increasing camera interval timestamps")
        if any(c.segment_id == id_ and (start is None or c.start_ms is None
               or max(start, c.start_ms) < min(end, c.end_ms)) for c in result):
            raise ValueError(f"{id_}: calibration intervals overlap")
        pix, xy = points(row.get("image_points"), 4), points(row.get("pitch_points"), 4)
        if (len(pix) != len(xy) or np.linalg.matrix_rank(pix - pix.mean(axis=0)) != 2
                or np.linalg.matrix_rank(xy - xy.mean(axis=0)) != 2
                or len(np.unique(pix, axis=0)) != len(pix) or len(np.unique(xy, axis=0)) != len(xy)):
            raise ValueError(f"{id_}: need unique, non-collinear landmark pairs")
        if (xy < 0).any() or (xy > [length, width]).any():
            raise ValueError(f"{id_}: landmarks outside the declared pitch")
        cpx, cxy = row.get("check_image_points"), row.get("check_pitch_points")
        if cpx is not None or cxy is not None:
            cpx, cxy = points(cpx, 1), points(cxy, 1)
            if (len(cpx) != len(cxy) or (cxy < 0).any() or (cxy > [length, width]).any()
                    or (np.linalg.norm(cxy[:, None]-xy[None, :], axis=-1) < .05).any()
                    or (np.linalg.norm(cpx[:, None]-pix[None, :], axis=-1) < 1).any()):
                raise ValueError(f"{id_}: check landmarks must be independent paired points on the pitch")
        follow = row.get("follow_camera", False)
        if type(follow) is not bool or (follow and (start is None or cpx is None)):
            raise ValueError(f"{id_}: camera following requires a bounded interval and check landmarks")
        try:
            matrix, rms, error, check = fit_projection(cv2, np, pix, xy, cpx, cxy)
        except (ValueError, cv2.error) as exc:
            raise ValueError(f"{id_}: could not fit a stable pitch mapping") from exc
        status = ("rejected" if max(error, check or 0) > max_error_m else
                  "verified" if check is not None else "unverified")
        landmarks = (pix, xy, cpx, cxy) if follow else None
        result.append(Calibration(id_, start, end, matrix, status, rms, check, landmarks))
    return float(length), float(width), result


class CameraFollow:
    """Propagate manual field landmarks through a bounded, continuous shot."""
    def __init__(self, cv2, np, calibration, image, max_error_m):
        self.cv2, self.np = cv2, np
        self.calibration, self.max_error_m = calibration, max_error_m
        pix, self.pitch, check_pix, self.check_pitch = calibration.landmarks
        self.fit_count = len(pix)
        self.points = np.concatenate((pix, check_pix)).astype(np.float32).reshape(-1, 1, 2)
        self.previous = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.valid = self.in_frame(self.points)

    def in_frame(self, points):
        h, w = self.previous.shape
        return ((points[:, 0] >= 5) & (points[:, 0] < [w-5, h-5])).all(axis=1)

    def update(self, image):
        cv2, np = self.cv2, self.np
        current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        params = {"winSize": (41, 41), "maxLevel": 4}
        moved, forward, residual = cv2.calcOpticalFlowPyrLK(self.previous, current, self.points, None, **params)
        back, backward, _ = cv2.calcOpticalFlowPyrLK(
            current, self.previous, moved, self.points.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            winSize=(41, 41), maxLevel=0)
        drift = np.linalg.norm(back[:, 0] - self.points[:, 0], axis=1)
        self.valid &= ((forward[:, 0] == 1) & (backward[:, 0] == 1) & (drift < 1.5)
                       & (residual[:, 0] < 18) & self.in_frame(moved))
        self.previous, self.points = current, moved
        fit, check = self.valid[:self.fit_count], self.valid[self.fit_count:]
        cal = self.calibration
        # Lost points require a new operator anchor.
        if fit.sum() < 4 or not check.any() or cal.status != "verified":
            return replace(cal, status="rejected", rms_m=None, check_max_m=None)
        try:
            matrix, rms, error, check_max = fit_projection(
                cv2, np, moved[:self.fit_count, 0][fit], self.pitch[fit],
                moved[self.fit_count:, 0][check], self.check_pitch[check])
        except (ValueError, cv2.error):
            return replace(cal, status="rejected", rms_m=None, check_max_m=None)
        status = "verified" if max(error, check_max) <= self.max_error_m else "rejected"
        return replace(cal, matrix=matrix, rms_m=rms, check_max_m=check_max, status=status)


def identity_from(data, plan):
    """Resolve only an explicit operator mapping to known roster identities."""
    if not isinstance(data, dict):
        raise ValueError("Identity map must be an object keyed by track ID")
    target = plan.get("team_id")
    roster = {row["id"]: row.get("team_id", target)
              for row in plan.get("identity_roster", plan.get("players", []))}
    clean = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise ValueError("Malformed reviewed identity map")
        player, team = value.get("player_id"), value.get("team_id")
        if any(v is not None and (not isinstance(v, str) or not v) for v in (player, team)):
            raise ValueError(f"{key}: reviewed IDs must be nonempty strings or null")
        if player is not None:
            if player not in roster:
                raise ValueError(f"{key}: player ID is absent from the clip plan roster")
            if team is not None and roster[player] is not None and team != roster[player]:
                raise ValueError(f"{key}: team does not match the clip plan roster")
            team = team or roster[player]
        if not player and not team:
            raise ValueError(f"{key}: supply a reviewed player or team ID")
        clean[key] = (player, team)
    return clean


KIT_COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "red")


def kit_sample(cv2, np, frame, box):
    """Name a dominant shirt color when the torso is distinct from its background."""
    x1, y1, x2, y2 = map(round, box)
    width, height = x2 - x1, y2 - y1
    if width < 8 or height < 22:
        return None
    left = max(0, x1 - max(4, width // 4))
    right = min(frame.shape[1], x2 + max(4, width // 4))
    top, bottom = y1 + round(height * .24), y1 + round(height * .52)
    patch = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    inset = width // 4
    torso = patch[:, x1 + inset - left:x2 - inset - left].reshape(-1, 3)
    background = np.concatenate((patch[:, :x1-left], patch[:, x2-left:]), axis=1).reshape(-1, 3)
    if background.size:
        white = (torso[:, 1] < 54) & (torso[:, 2] > 162)
        if white.mean() >= .72 and np.median(torso[:, 2]) > np.median(background[:, 2]) + 24:
            return "white"
        dark = torso[:, 2] < 62
        if dark.mean() >= .80 and np.median(torso[:, 2]) < np.median(background[:, 2]) - 33:
            return "black"
    chromatic = (torso[:, 1] >= 85) & (torso[:, 2] >= 95)
    boundaries = (10, 20, 40, 84, 101, 131, 170)
    buckets = np.digitize(torso[:, 0], boundaries)
    counts = np.bincount(buckets[chromatic], minlength=len(KIT_COLORS))
    counts[0] += counts[-1]
    counts[-1] = 0
    index = int(np.argmax(counts))
    if counts[index] < torso.shape[0] * .60:
        return None
    if KIT_COLORS[index] == "yellow" and np.median(torso[:, 2]) < 160:
        return None
    if KIT_COLORS[index] == "green":
        grass = (background[:, 0] >= 25) & (background[:, 0] < 84) & (background[:, 1] >= 50)
        shirt_brightness = np.median(torso[chromatic & (buckets == index), 2])
        if grass.sum() < 10 or shirt_brightness < np.percentile(background[grass, 2], 85) + 18:
            return None
    return KIT_COLORS[index]


def calibrated_boundary(cv2, np, shape, calibration, length, width):
    """Return the visible portion of a verified rectangle in image pixels."""
    h, w = shape[:2]
    inverse = np.linalg.inv(calibration.matrix)
    projected = np.array([(0, 0, 1), (length, 0, 1), (length, width, 1), (0, width, 1)]) @ inverse.T
    depth = projected[:, 2]
    if np.any(np.abs(depth) < 1e-9) or np.min(depth)*np.max(depth) < 0:
        return None  # A full-pitch corner falls across the camera horizon.
    corners = projected[:, :2] / depth[:, None]
    if not np.isfinite(corners).all() or np.abs(corners).max() > 100*max(w, h):
        return None
    frame = np.array([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)], dtype=np.float32)
    area, polygon = cv2.intersectConvexConvex(corners.astype(np.float32), frame)
    if area <= 0 or polygon is None:
        return None
    return polygon.reshape(-1, 2).round(2).tolist()


def pitch_detections(cv2, np, frame, detector, calibration, length, width):
    """Gate person boxes by ground contact before track association."""
    auto = detect_pitch(cv2, np, frame)
    verified = calibration is not None and calibration.status == "verified"
    automatic = calibration is not None and calibration.status == "automatic"
    mapped = verified or automatic
    exact = calibrated_boundary(cv2, np, frame.shape, calibration, length, width) if mapped else None
    polygon = exact or (auto.polygon_xy if auto else None)
    source = ("verified_calibration" if verified else "automatic_landmarks") if exact else auto.source if auto else None
    boundary = {"polygon_xy": polygon, "source": source} if polygon else None
    status = "verified" if verified else "estimated" if auto or exact else "unavailable"
    if status == "unavailable":
        return [], boundary, status, 0
    rejected = 0

    def accepts_person(point):
        nonlocal rejected
        if mapped:
            xy = calibration.project(point)
            inside = xy is not None and 0 <= xy[0] <= length and 0 <= xy[1] <= width
        else:
            inside = auto.contains(point)
        rejected += not inside
        return inside

    detections = detector.detect(frame, polygon, accepts_person=accepts_person)
    return [replace(d, kit_color=kit_sample(cv2, np, frame, d.box))
            if d.object_type == "player" else d for d in detections], boundary, status, rejected


def object_row(track, calibration, length, width, identities):
    d = track.last
    player, team = identities.get(track.id, (None, None)) if d.object_type == "player" else (None, None)
    xy = None
    status = "missing" if track.missing else "uncalibrated"
    if not track.missing and calibration is not None:
        status = calibration.status
        if status != "rejected":
            xy = calibration.project(d.anchor())
            if not xy or not (0 <= xy[0] <= length and 0 <= xy[1] <= width):
                xy, status = None, "off_pitch"
            elif d.object_type == "ball" and status == "verified":
                status = "ball_ground_assumed"
    return {"track_id": track.id, "object_type": d.object_type,
            "player_id": player, "team_id": team, "observed": not track.missing,
            "estimated": False, "visibility": "missing" if track.missing else "observed",
            "confidence": None if track.missing else d.confidence,
            "bbox_xyxy": None if track.missing else d.box,
            "x_m": None if not xy else round(xy[0], 3), "y_m": None if not xy else round(xy[1], 3),
            "projection_status": status, "kit_color_hint": None,
            "on_pitch": d.object_type == "player" and not track.missing}


def small_frame(cv2, frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)


def run(video_path, plan, model, calibration_data, *, clip_ids=None, sample_fps=5.0,
        confidence=.35, max_error_m=2.0, identity_map=None, through_eof=False, on_progress=None,
        pitch_model=None, inference_backend="opencv", inference_cache=None):
    """Return a tracking manifest and pitch replay observations for selected clips."""
    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION:
        raise ValueError("Expected a versioned fbrpro clip plan")
    if not all(isinstance(plan.get(k), str) and plan[k] for k in ("match_id", "video_id", "plan_id")):
        raise ValueError("Clip plan is missing match_id, video_id, or plan_id")
    if not finite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError("sample-fps must be greater than zero and at most 30")
    if not finite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not finite(max_error_m) or max_error_m <= 0:
        raise ValueError("max-error-m must be positive")
    cv2, np = vision()
    length, width, calibrations = calibration_from(calibration_data, max_error_m)
    if calibration_data.get("video_id") not in (None, plan["video_id"]):
        raise ValueError("Calibration video_id does not match the clip plan")
    identities = identity_from({} if identity_map is None else identity_map, plan)
    aligned = {s.id: s for s in segments_from({"segments": plan["alignment_segments"]})}
    for c in calibrations:
        s = aligned.get(c.segment_id)
        if not s or (c.start_ms is not None and
                     not s.video_start_ms <= c.start_ms < c.end_ms <= s.video_end_ms):
            raise ValueError(f"{c.segment_id}: camera calibration lies outside the alignment")
    available = plan["clips"]
    known = {c["id"] for c in available}
    if len(known) != len(available) or any(not isinstance(id_, str) or not id_ for id_ in known):
        raise ValueError("Clip IDs must be unique nonempty strings")
    if clip_ids and set(clip_ids) - known:
        raise ValueError("Unknown clip ID in selection")
    clips = [c for c in available if not clip_ids or c["id"] in clip_ids]
    if not clips:
        raise ValueError("No selected clips in the plan")
    windows = []
    for clip in clips:
        segment = aligned.get(clip["segment_id"])
        start, end = clip["video_start_ms"], clip["video_end_ms"]
        if (not segment or segment.period != clip["period"] or type(start) is not int or type(end) is not int
                or not segment.video_start_ms <= start < end <= segment.video_end_ms):
            raise ValueError(f"{clip['id']}: clip lies outside the aligned video interval")
        windows.append((start, end, clip, segment))
    windows.sort(key=lambda window: window[0])
    if any(a[1] > b[0] for a, b in zip(windows, windows[1:])):
        raise ValueError("Selected clips overlap. Merge overlapping review windows before tracking.")
    if through_eof and (len(windows) != 1 or windows[0][0] != 0 or
                        windows[0][3].sync_status != "preview"):
        raise ValueError("Full video tracking needs one video-only window starting at zero")
    detector = YoloX(model, confidence, inference_backend=inference_backend, inference_cache=inference_cache)
    automatic_model = None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Unable to decode local video file")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if through_eof else None
        frames, summaries = [], []
        observed_ids = set()
        for start, end, clip, segment in windows:
            tracker = Tracker(clip["id"], max_gap_ms=max(800, round(2000/sample_fps)), id_scope=TRACK_ID_SCOPE)
            camera_motion = CameraMotion(cv2, np)
            if not through_eof and not cap.set(cv2.CAP_PROP_POS_MSEC, start):
                raise ValueError(f"{clip['id']}: decoder cannot seek by video time")
            # Do not invent clock from frame index/FPS when the decoder has no timestamps.
            last_ms, next_sample_ms, previous_img = -1, float(start), None
            last_step_ms = None
            interrupted, missing_frames, sampled, decoded = False, 0, 0, 0
            counters = Counter()
            previous_calibration = invalid_calibration = camera_follow = None
            pending_cut = False
            while True:
                ok, img = cap.read()
                if not ok:
                    interrupted = not through_eof
                    if through_eof and last_ms >= 0:
                        # Frame-count metadata is an estimate for variable-rate
                        # files. Bound coverage using the decoded presentation clock.
                        tail_ms = last_step_ms or (1000/fps if finite(fps) and fps > 0 else 1)
                        end = min(MAX_FULL_VIDEO_MS, max(math.floor(last_ms)+1, round(last_ms+tail_ms)))
                    break
                stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not finite(stamp) or stamp < 0 or (last_ms >= 0 and stamp <= last_ms):
                    raise ValueError(f"{clip['id']}: decoder did not supply monotonic frame timestamps")
                if last_ms >= 0:
                    last_step_ms = stamp-last_ms
                last_ms = stamp
                if through_eof and stamp >= MAX_FULL_VIDEO_MS:
                    raise ValueError("Full video tracking supports footage up to six hours")
                if not through_eof and stamp >= end:
                    break
                if stamp < start:
                    continue
                decoded += 1
                video_ms = round(stamp)
                current = small_frame(cv2, img)
                cut = previous_img is not None and float(np.mean(cv2.absdiff(previous_img, current))) > 44
                pending_cut |= cut
                previous_img = current
                selected = next((c for c in calibrations if c.applies(segment.id, video_ms)), None)
                if selected is not previous_calibration:
                    camera_follow = None
                    if selected and selected.landmarks is not None and video_ms == selected.start_ms:
                        camera_follow = CameraFollow(cv2, np, selected, img, max_error_m)
                elif cut:
                    invalid_calibration = selected
                cal = selected
                if selected is invalid_calibration:
                    cal = None
                elif selected and selected.landmarks is not None:
                    if not camera_follow:
                        cal = None  # Selected clips must include the reference timestamp.
                    elif selected is previous_calibration:
                        cal = camera_follow.update(img)
                    elif not camera_follow.valid.all():
                        cal = replace(selected, status="rejected", rms_m=None, check_max_m=None)
                previous_calibration = selected
                if stamp < next_sample_ms:
                    continue
                while next_sample_ms <= stamp:
                    next_sample_ms += 1000/sample_fps
                period_ms = video_ms if through_eof else segment.period_time(video_ms)
                if pending_cut:
                    tracker.reset()
                    camera_motion.reset()
                if pitch_model is not None and (cal is None or cal.status != "verified"):
                    if automatic_model is None:
                        automatic_model = PitchModel(cv2, np, pitch_model, length, width,
                                                     inference_backend=inference_backend,
                                                     inference_cache=inference_cache)
                    fit = automatic_model.fit(img)
                    if fit is not None:
                        matrix, rms, check = fit
                        cal = Calibration(segment.id, None, None, matrix, "automatic", rms, check)
                        counters["automatically_calibrated_frames"] += 1
                detections, boundary, pitch_status, off_pitch = pitch_detections(
                    cv2, np, img, detector, cal, length, width)
                counters["off_pitch_candidates"] += off_pitch
                counters["pitch_unavailable_frames"] += pitch_status == "unavailable"
                motion = camera_motion.update(img, (d.box for d in detections if d.object_type == "player"))
                tracks = tracker.update(detections, video_ms, motion)
                rows = [object_row(t, cal, length, width, identities) for t in tracks]
                for track, row in zip(tracks, rows):
                    color = track.last.kit_color
                    if not track.missing and color is not None and color == track.kit_sample:
                        row["kit_color_hint"] = color
                observed_ids.update(t.id for t in tracks if not t.missing and t.last.object_type == "player")
                seen_players = set()
                for row in rows:
                    pid = row["player_id"]
                    if row["observed"] and pid is not None:
                        if pid in seen_players:
                            raise ValueError(f"{clip['id']}: same reviewed player appears twice in a frame")
                        seen_players.add(pid)
                    if row["observed"]:
                        counters[row["object_type"]] += 1
                    else:
                        counters["missing"] += 1
                missing_frames += not any(r["observed"] for r in rows)
                frames.append({"clip_id": clip["id"], "segment_id": segment.id, "period": segment.period,
                               "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1,
                               "video_time_ms": video_ms, "period_time_ms": period_ms,
                               "calibration_status": cal.status if cal else "uncalibrated",
                               "calibration_fit_rms_m": cal.rms_m if cal else None,
                               "calibration_check_max_m": cal.check_max_m if cal else None,
                               "camera_cut_suspected": pending_cut,
                               "camera_motion_compensated": motion is not None,
                               "pitch_boundary_status": pitch_status, "pitch_boundary": boundary,
                               "off_pitch_candidates": off_pitch, "objects": rows})
                pending_cut = False
                sampled += 1
                if on_progress:
                    on_progress(video_ms)
            summaries.append({"clip_id": clip["id"], "sampled_frames": sampled,
                              "video_start_ms": start, "video_end_ms": end,
                              "decoded_frames": decoded,
                              "empty_frames": missing_frames, "observations": dict(counters),
                              "decode_ended_before_clip_end": interrupted})
        if not frames:
            raise ValueError("No video frames were decoded inside the selected clips")
    finally:
        cap.release()
    if set(identities) - observed_ids:
        raise ValueError("Reviewed identity map includes tracks not observed in this run. Regenerate it for the current report.")
    roles = annotate_video_roles(frames, coordinate_length_m=length, coordinate_width_m=width,
                                 dimensions_assumed=bool(calibration_data.get("dimensions_assumed", False)))
    with Path(video_path).open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    with Path(model).open("rb") as handle:
        model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"version": VERSION, "match_id": plan["match_id"], "video_id": plan["video_id"],
            "plan_id": plan["plan_id"], "coordinate_frame": "stadium_fixed",
            "coordinate_length_m": length, "coordinate_width_m": width,
            "dimensions_assumed": bool(calibration_data.get("dimensions_assumed", False)),
            "frame_width": frame_width, "frame_height": frame_height,
            "sample_fps": sample_fps, "confidence_threshold": confidence, "max_calibration_error_m": max_error_m,
            "detector": "opencv-zoo-yolox-s-coco", "source_sha256": source_hash, "model_sha256": model_hash,
            "inference": {"player": detector.engine.description(),
                          "pitch": automatic_model.engine.description() if automatic_model else None},
            "identity_policy": "reviewed mappings only; on-pitch detections can include officials",
            "tracker": "camera_compensated_motion_iou_kit",
            "pitch_policy": "person foot point must be in the verified or estimated pitch; no tracks if neither is available",
            "kit_color_policy": "review hint from two agreeing torso samples",
            "automatic_calibration": bool(pitch_model),
            "role_inference": roles,
            "sync": [{"segment_id": s.id, "status": s.sync_status,
                      "check_max_ms": s.check_max_ms} for s in aligned.values()],
            "calibration": [{"segment_id": c.segment_id, "status": c.status,
                             "video_start_ms": c.start_ms, "video_end_ms": c.end_ms,
                             "follow_camera": c.landmarks is not None,
                             "fit_rms_m": c.rms_m, "check_max_m": c.check_max_m} for c in calibrations],
            "clips": summaries, "frames": frames}


def canonical_rows(report):
    """Export reviewed players only, plus a clearly marked projected ball.

    Anonymous object IDs are kept in the sidecar because the common schema
    requires a canonical lineup ID for each player object.
    """
    verified_segments = {s["segment_id"] for s in report.get("sync", []) if s["status"] == "verified"}
    for i, frame in enumerate(report["frames"]):
        if "sync" in report and frame["segment_id"] not in verified_segments:
            continue
        rows = [r for r in frame["objects"] if r["object_type"] == "player" and r["player_id"]]
        # Prefer an observation over a stale missing track after re-identification.
        unique = {}
        for r in sorted(rows, key=lambda row: row["observed"]):
            unique[r["player_id"]] = r
        ball = max((r for r in frame["objects"] if r["object_type"] == "ball"),
                   key=lambda r: r["observed"], default=None)
        if ball is not None:
            unique["ball"] = ball
        for object_id, r in unique.items():
            usable = r["projection_status"] in {"verified", "ball_ground_assumed"}
            yield {"schema_version": "1.0.0", "tracking_id": f"{report['video_id']}:{i}:{object_id}",
                   "match_id": report["match_id"], "period": frame["period"],
                   "period_time_ms": frame["period_time_ms"], "frame_id": frame["frame_id"],
                   "object_id": object_id, "object_type": r["object_type"],
                   "team_id": r["team_id"], "player_id": r["player_id"],
                   "x_m": r["x_m"] if usable else None, "y_m": r["y_m"] if usable else None, "z_m": None,
                   "observed": r["observed"], "estimated": r["estimated"],
                   "coordinate_frame": report["coordinate_frame"],
                   "coordinate_length_m": report["coordinate_length_m"],
                   "coordinate_width_m": report["coordinate_width_m"],
                   "dimensions_assumed": report.get("dimensions_assumed", False),
                   "projection_status": r["projection_status"],
                   "source": {"provider": "local_video", "record_id": r["track_id"],
                              "file_sha256": report["source_sha256"], "row_number": None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    for arg in ("plan", "model", "calibration", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    parser.add_argument("--clip", action="append", dest="clip_ids")
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--max-error-m", type=float, default=2)
    parser.add_argument("--identity-map", type=Path, help="JSON object mapping reviewed track IDs to player/team IDs")
    parser.add_argument("--tracking-jsonl", type=Path, help="Optional normalized export of reviewed players and projected ball")
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    try:
        if args.output.exists() or (args.tracking_jsonl and args.tracking_jsonl.exists()):
            raise ValueError("Choose new output paths (tracking does not overwrite files)")
        if args.tracking_jsonl and args.output.resolve() == args.tracking_jsonl.resolve():
            raise ValueError("JSON manifest and JSONL export must be different paths")
        report = run(args.video, read(args.plan), args.model, read(args.calibration),
                     clip_ids=args.clip_ids, sample_fps=args.sample_fps, confidence=args.confidence,
                     max_error_m=args.max_error_m, identity_map=read(args.identity_map) if args.identity_map else None)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        if args.tracking_jsonl:
            with args.tracking_jsonl.open("x", encoding="utf-8") as handle:
                for row in canonical_rows(report):
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.exit(2, f"tracking: {exc}\n")
    print(f"Tracked {len(report['frames'])} sampled frames in {len(report['clips'])} clips.")


if __name__ == "__main__":
    main()
