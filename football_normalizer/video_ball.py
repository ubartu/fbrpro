"""Single-ball detection and camera-compensated association."""
from dataclasses import dataclass, replace
from math import hypot

from .video_inference import load_inference


class BallDetector:
    size = 1280
    confidence_floor = .14

    def __init__(self, cv2, np, path, *, inference_backend="opencv", inference_cache=None):
        self.cv2, self.np = cv2, np
        self.engine = load_inference(cv2, path, inference_backend, "the ball detector", inference_cache)

    def detect(self, frame, boundary):
        if boundary is None:
            return []
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        scale = self.size/max(h, w)
        width, height = round(w*scale), round(h*scale)
        x, y = (self.size-width)//2, (self.size-height)//2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[y:y+height, x:x+width] = cv2.resize(frame, (width, height))
        raw = self.engine.forward(cv2.dnn.blobFromImage(canvas, 1/255, swapRB=True))
        if raw.ndim != 3 or raw.shape[:2] != (1, 5):
            raise ValueError("The ball detector must be a single-class YOLO ONNX export with output [1, 5, anchors].")
        rows = raw[0].T
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= self.confidence_floor)]
        contour = np.asarray(boundary["polygon_xy"], dtype=np.float32)
        boxes, scores = [], []
        for cx, cy, bw, bh, score in rows:
            cx, cy = (float(cx)-x)/scale, (float(cy)-y)/scale
            bw, bh = float(bw)/scale, float(bh)/scale
            if (not 0 <= cx < w or not 0 <= cy < h or min(bw, bh) < 1
                    or max(bw, bh) > .045*max(h, w) or max(bw, bh) > 2.6*min(bw, bh)
                    or cv2.pointPolygonTest(contour, (cx, cy), True) < -.04*h):
                continue
            left, top = max(0, cx-bw/2), max(0, cy-bh/2)
            boxes.append((left, top, min(w, cx+bw/2)-left, min(h, cy+bh/2)-top))
            scores.append(float(score))
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_floor, .35) if boxes else ()
        result = []
        for i in np.asarray(indices, dtype=int).ravel()[:12]:
            left, top, bw, bh = boxes[i]
            result.append((scores[i], (left, top, left+bw, top+bh)))
        return result


@dataclass(frozen=True)
class BallTrack:
    id: str
    last: object
    missing: bool = False


class BallTracker:
    """Keep one observed ball and leave gaps when the evidence is ambiguous."""

    def __init__(self, clip_id, frame_width):
        self.clip_id, self.width = clip_id, frame_width
        self.sequence = 0
        self.reset()

    def reset(self):
        self.track = self.point = self.previous = None
        self.seen_ms = None

    def update(self, candidates, stamp, camera_transform=None):
        elapsed = stamp-self.seen_ms if self.seen_ms is not None else 0
        if elapsed > 650:
            self.reset()
        if self.track and camera_transform is not None:
            self.point = tuple(camera_transform @ (*self.point, 1))
            if self.previous is not None:
                previous, time = self.previous
                self.previous = (tuple(camera_transform @ (*previous, 1)), time)
        ranked = []
        if self.track:
            x, y = self.point
            if self.previous is not None and self.seen_ms > self.previous[1]:
                (px, py), time = self.previous
                factor = min(1.5, elapsed/(self.seen_ms-time))
                x += (x-px)*factor
                y += (y-py)*factor
            radius = self.width*(.035 + .36*min(.55, elapsed/1000))
            for d in candidates:
                cx, cy = d.anchor()
                distance = min(hypot(cx-x, cy-y), hypot(cx-self.point[0], cy-self.point[1])+.02*self.width)
                if distance <= radius:
                    ranked.append((d.confidence-.24*distance/radius, d))
        continuation = bool(ranked)
        if not ranked:
            floor = .35 if self.track else .22
            ranked = [(d.confidence, d) for d in candidates if d.confidence >= floor]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        ambiguous = (len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .045
                     and hypot(*(a-b for a, b in zip(ranked[0][1].anchor(), ranked[1][1].anchor()))) > .015*self.width)
        if not ranked or ambiguous:
            return replace(self.track, missing=True) if self.track else None
        chosen = ranked[0][1]
        if continuation:
            self.previous = (self.point, self.seen_ms)
            self.track = BallTrack(self.track.id, chosen)
        else:
            self.sequence += 1
            self.previous = None
            self.track = BallTrack(f"{self.clip_id}:ball:{self.sequence}", chosen)
        self.point = chosen.anchor()
        self.seen_ms = stamp
        return self.track
