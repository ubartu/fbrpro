"""Ground markings in SoccerNet's semantic line convention."""
import math
from itertools import combinations

from .video_inference import load_inference
from .video_motion import CameraMotion


def stabilize_projection(cv2, np, previous, current, motion, elapsed_ms, length, width):
    """Limit a refit's correction in ground metres after transporting camera motion."""
    transported = previous @ np.linalg.inv(np.vstack((motion, [0, 0, 1.])))
    pixels = np.array([(x, y) for y in np.linspace(36, 342, 7)
                       for x in np.linspace(32, 608, 9)], dtype=np.float64)
    before = cv2.perspectiveTransform(pixels[:, None], transported)[:, 0]
    fresh = cv2.perspectiveTransform(pixels[:, None], current)[:, 0]
    visible = (np.isfinite(before).all(axis=1) & np.isfinite(fresh).all(axis=1)
               & (before >= 0).all(axis=1) & (before <= [length, width]).all(axis=1)
               & (fresh >= 0).all(axis=1) & (fresh <= [length, width]).all(axis=1))
    if visible.sum() < 6:
        return None, None
    pixels, before, fresh = pixels[visible], before[visible], fresh[visible]
    if cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))) < 640*360*.01:
        return None, None
    shift = np.linalg.norm(fresh-before, axis=1)
    # A different pitch basis or unreliable camera match needs a new segment.
    if np.median(shift) > 5:
        return None, None
    limit = .002 * elapsed_ms
    if shift.max() <= limit:
        return current, 0.
    targets = before+(fresh-before)*(limit/shift.max())
    matrix, _ = cv2.findHomography(pixels, targets, 0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        return None, None
    stabilized = cv2.perspectiveTransform(pixels[:, None], matrix)[:, 0]
    if not np.isfinite(stabilized).all() or np.linalg.norm(stabilized-before, axis=1).max() > limit*1.1:
        return None, None
    disagreement = float(np.quantile(np.linalg.norm(stabilized-fresh, axis=1), .9))
    return matrix, disagreement


LINE_CLASSES = (
    "Big rect. left bottom", "Big rect. left main", "Big rect. left top",
    "Big rect. right bottom", "Big rect. right main", "Big rect. right top",
    "Circle central", "Circle left", "Circle right",
    "Goal left crossbar", "Goal left post left ", "Goal left post right",
    "Goal right crossbar", "Goal right post left", "Goal right post right",
    "Goal unknown", "Line unknown", "Middle line", "Side line bottom",
    "Side line left", "Side line right", "Side line top",
    "Small rect. left bottom", "Small rect. left main", "Small rect. left top",
    "Small rect. right bottom", "Small rect. right main", "Small rect. right top",
)


def pitch_lines(length, width):
    """Return ground-plane polylines in metres, with the top-left as origin."""
    middle = width/2
    lines = {
        "Side line top": [(0, 0), (length, 0)],
        "Side line bottom": [(0, width), (length, width)],
        "Side line left": [(0, 0), (0, width)],
        "Side line right": [(length, 0), (length, width)],
        "Middle line": [(length/2, 0), (length/2, width)],
    }
    for side, x, sign in (("left", 0, 1), ("right", length, -1)):
        for prefix, depth, half_width in (("Big", 16.5, 20.16), ("Small", 5.5, 9.16)):
            far = x+sign*depth
            top, bottom = middle-half_width, middle+half_width
            lines[f"{prefix} rect. {side} top"] = [(x, top), (far, top)]
            lines[f"{prefix} rect. {side} bottom"] = [(x, bottom), (far, bottom)]
            lines[f"{prefix} rect. {side} main"] = [(far, top), (far, bottom)]
        angle = math.acos(5.5/9.15)
        lines[f"Circle {side}"] = [
            (x+sign*(11+9.15*math.cos(-angle+2*angle*i/48)),
             middle+9.15*math.sin(-angle+2*angle*i/48)) for i in range(49)]
    lines["Circle central"] = [(length/2+9.15*math.cos(i*math.pi/64),
                               middle+9.15*math.sin(i*math.pi/64)) for i in range(129)]
    return lines


def polyline_errors(np, points, line):
    """Image-space distance to a finite semantic line, as used by SoccerNet."""
    first, direction = line[:-1], line[1:]-line[:-1]
    delta = points[:, None]-first[None]
    length = (direction**2).sum(axis=1)
    fraction = ((delta*direction[None]).sum(axis=2)/np.maximum(length, 1e-9)).clip(0, 1)
    nearest = first[None]+fraction[:, :, None]*direction[None]
    return np.linalg.norm(points[:, None]-nearest, axis=2).min(axis=1)


def fit_pitch_lines(cv2, np, observed, lines, shape):
    """Fit straight marking correspondences and check withheld markings."""
    height, width = shape[:2]
    pitch_length = lines["Side line top"][1][0]
    pitch_width = lines["Side line left"][1][1]
    field_scale = np.diag([1/pitch_length, 1/pitch_width, 1.])
    image_scale = np.diag([width, width, 1.])
    pairs = {}
    for name, pixels in observed.items():
        if name not in lines or len(lines[name]) != 2 or len(pixels) < 8:
            continue
        vx, vy, x, y = cv2.fitLine(pixels.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        normal = np.array([-vy, vx])
        if np.ptp(pixels @ [vx, vy]) < width*.045:
            continue
        if np.quantile(np.abs((pixels-[x, y]) @ normal), .8) > width*.006:
            continue
        a, b = normal
        c = -normal @ [x/width, y/width]
        equations, targets = [], []
        for px, py in np.asarray(lines[name])/[pitch_length, pitch_width]:
            equations.append([a*px, a*py, a, b*px, b*py, b, c*px, c*py])
            targets.append(-c)
        pairs[name] = (equations, targets)
    if len(pairs) < 4:
        return None

    def fit(names):
        design = np.concatenate([pairs[name][0] for name in names])
        target = np.concatenate([pairs[name][1] for name in names])
        values, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < 8:
            return None
        matrix = image_scale @ np.append(values, 1.).reshape(3, 3) @ field_scale
        if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            return None
        return matrix

    def errors(matrix):
        result = {}
        for name, pixels in observed.items():
            if name not in lines:
                continue
            field = np.asarray(lines[name], np.float32)
            projected = cv2.perspectiveTransform(field[:, None], matrix)[:, 0]
            if not np.isfinite(projected).all():
                continue
            result[name] = float(np.quantile(polyline_errors(np, pixels, projected), .8))
        return result

    subsets = list(combinations(pairs, 4))
    subsets = [tuple(pairs)]+subsets[::max(1, math.ceil(len(subsets)/64))]
    best = None
    threshold = width*.0125
    for names in subsets:
        matrix = fit(names)
        if matrix is None:
            continue
        residuals = errors(matrix)
        inliers = {name for name in pairs if residuals.get(name, math.inf) <= threshold}
        checks = {name: error for name, error in residuals.items() if name not in names and error <= threshold}
        if len(inliers) < max(4, .7*len(pairs)) or not checks:
            continue
        score = (len(inliers), -sum(residuals[name] for name in inliers)/len(inliers))
        if best is None or score > best[0]:
            best = (score, inliers, checks, matrix)
    if best is None:
        return None
    _, inliers, checks, check_matrix = best
    matrix = fit(sorted(inliers))
    if matrix is None:
        return None
    residuals = errors(matrix)
    if any(residuals.get(name, math.inf) > threshold for name in inliers):
        return None
    inverse = np.linalg.inv(matrix)
    metric_errors = []
    for name in inliers:
        field = cv2.perspectiveTransform(observed[name][:, None].astype(np.float32), inverse)[:, 0]
        metric_errors.extend(polyline_errors(np, field, np.asarray(lines[name])).tolist())
    rms_m = float(np.sqrt(np.mean(np.square(metric_errors))))
    withheld_errors = []
    check_inverse = np.linalg.inv(check_matrix)
    for name in checks:
        field = cv2.perspectiveTransform(observed[name][:, None].astype(np.float32), check_inverse)[:, 0]
        withheld_errors.append(float(np.quantile(polyline_errors(np, field, np.asarray(lines[name])), .8)))
    check_m = max(withheld_errors)
    if not np.isfinite(rms_m+check_m) or rms_m > 2 or check_m > 2.75:
        return None
    return inverse, rms_m, check_m, {
        "method": "semantic_lines", "matched_lines": len(inliers),
        "matched_line_names": sorted(inliers),
        "line_error_px": round(float(np.mean([residuals[name] for name in inliers])), 3),
        "withheld_line_error_px": round(max(checks.values()), 3),
        "error_resolution": [width, height],
    }


class LinePitchModel:
    """SoccerNet's DeepLabV3 29-class export with normalization in the graph."""

    def __init__(self, cv2, np, path, length, width, *, inference_backend="opencv", inference_cache=None,
                 landmark_model=None):
        self.cv2, self.np = cv2, np
        self.engine = load_inference(cv2, path, inference_backend, "the pitch line model", inference_cache)
        self.lines = pitch_lines(length, width)
        self.landmark = None
        if landmark_model is not None:
            from .video_auto_calibration import PitchModel
            self.landmark = PitchModel(cv2, np, landmark_model, length, width,
                                       inference_backend=inference_backend, inference_cache=inference_cache)
        self.last_quality = None
        self.motion = CameraMotion(cv2, np)
        self.reset()

    def reset(self):
        self.motion.reset()
        self.last_quality = None
        self.previous_fit = None
        self.previous_quality = None
        self.last_stamp = None
        self.anchor_stamp = None
        self.followed = 0
        self.frame_motion = None
        self.projection_fit = None
        self.projection_stamp = None

    def position_fit(self, fit, shape, stamp):
        """Produce replay coordinates without changing the fit used to detect objects."""
        if fit is None:
            self.projection_fit = self.projection_stamp = None
            return None, {}
        h, w = shape[:2]
        np = self.np
        matrix = fit[0] @ np.diag([w/640, h/360, 1.])
        previous = self.projection_fit
        quality = {}
        if previous is not None:
            elapsed = stamp-self.projection_stamp
            stabilized = None
            if self.frame_motion is not None and 0 < elapsed <= 250:
                stabilized, disagreement = stabilize_projection(
                    self.cv2, np, previous, matrix, self.frame_motion, elapsed,
                    self.lines["Side line top"][1][0], self.lines["Side line left"][1][1])
            if stabilized is None:
                quality["projection_reanchored"] = True
            else:
                matrix = stabilized
                quality["projection_disagreement_m"] = round(disagreement, 3)
        self.projection_fit, self.projection_stamp = matrix, stamp
        return matrix @ np.diag([640/w, 360/h, 1.]), quality

    def follow(self, image, matrix, quality=None):
        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ridge = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
        paint = ((hsv[:, :, 1] < 120) & (hsv[:, :, 2] > 120) & (ridge >= 12)).astype(np.uint8)
        distances = cv2.distanceTransform(1-paint, cv2.DIST_L2, 5)
        height, width = image.shape[:2]
        inverse = np.linalg.inv(matrix)
        supported = 0
        for name in (quality or self.previous_quality)["matched_line_names"]:
            points = np.asarray(self.lines[name], np.float32)
            if len(points) != 2:
                continue
            points = np.linspace(points[0], points[-1], 60).astype(np.float32)
            pixels = cv2.perspectiveTransform(points[:, None], inverse)[:, 0]
            inside = ((pixels[:, 0] >= 2) & (pixels[:, 0] < width-2)
                      & (pixels[:, 1] >= 2) & (pixels[:, 1] < height-2))
            pixels = pixels[inside]
            if len(pixels) < 8:
                continue
            x, y = np.rint(pixels).astype(int).T
            supported += np.mean(distances[y, x] <= 4) >= .65
        return supported >= 3

    def supports_display(self, frame, matrix, quality):
        h, w = frame.shape[:2]
        image = self.cv2.resize(frame, (640, 360))
        return self.follow(image, matrix @ self.np.diag([w/640, h/360, 1.]), quality)

    def checked_landmarks(self, frame, observed):
        if self.landmark is None or len(observed) < 3:
            return None
        cv2, np = self.cv2, self.np
        fit = self.landmark.fit(frame)
        if fit is None:
            return None
        h, w = frame.shape[:2]
        matrix = fit[0] @ np.diag([w/640, h/360, 1.])
        inverse = np.linalg.inv(matrix)
        supported = {}
        for name, pixels in observed.items():
            field = np.asarray(self.lines[name], np.float32)
            projected = cv2.perspectiveTransform(field[:, None], inverse)[:, 0]
            error = float(np.quantile(polyline_errors(np, pixels, projected), .8))
            if error <= 8:
                supported[name] = error
        if len(supported) < max(3, .8*len(observed)):
            return None
        return matrix, *fit[1:], {
            "method": "line_checked_landmarks", "matched_lines": len(supported),
            "matched_line_names": sorted(supported),
            "line_error_px": round(float(np.mean(list(supported.values()))), 3),
            "error_resolution": [640, 360],
        }

    def fit(self, frame, time_ms=None):
        cv2, np = self.cv2, self.np
        self.last_quality = None
        image = cv2.resize(frame, (640, 360))
        motion = self.motion.update(image, ()) if time_ms is not None else None
        self.frame_motion = motion
        elapsed = time_ms-self.last_stamp if time_ms is not None and self.last_stamp is not None else None
        self.last_stamp = time_ms
        followed_fit = None
        if self.previous_fit and elapsed is not None and 0 < elapsed <= 250 and motion is not None:
            candidate = self.previous_fit[0] @ np.linalg.inv(np.vstack((motion, [0, 0, 1.])))
            if self.follow(image, candidate):
                followed_fit = (candidate, *self.previous_fit[1:])
                if self.followed < 2:
                    self.followed += 1
                    self.previous_fit = followed_fit
                    self.last_quality = {**self.previous_quality, "method": "semantic_lines_flow",
                                         "propagated_frames": self.followed}
                    h, w = frame.shape[:2]
                    return candidate @ np.diag([640/w, 360/h, 1.]), *self.previous_fit[1:]
        previous_quality, propagated = self.previous_quality, self.followed
        self.previous_fit = None
        self.followed = 0
        raw = self.engine.forward(cv2.dnn.blobFromImage(image, 1/255, swapRB=False))
        if raw.shape != (1, 29, 360, 640):
            raise ValueError("The pitch line model must output [1, 29, 360, 640].")
        logits = raw[0]
        labels = logits.argmax(axis=0)
        peak = logits.max(axis=0)
        confidence = 1/np.exp(logits-peak[None]).sum(axis=0)
        observed = {}
        for index, name in enumerate(LINE_CLASSES, 1):
            if name not in self.lines:
                continue
            mask = ((labels == index) & (confidence >= .4)).astype(np.uint8)
            thickness = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            ridge = thickness >= cv2.dilate(thickness, np.ones((3, 3), np.uint8))-.05
            y, x = np.nonzero((thickness > 1) & ridge)
            if len(x) < 30:
                continue
            step = max(1, len(x)//160)
            observed[name] = np.column_stack((x[::step], y[::step])).astype(np.float32)
        result = fit_pitch_lines(cv2, np, observed, self.lines, image.shape)
        if result is None:
            result = self.checked_landmarks(frame, observed)
        if result is None:
            anchor = getattr(self, "anchor_stamp", None)
            if (followed_fit is None or anchor is None or time_ms is None
                    or not 0 < time_ms-anchor <= 1000):
                return None
            # A failed refresh must not erase an otherwise supported camera
            # mapping. Current painted lines and optical flow both have to
            # agree, and the original semantic anchor expires after one second.
            self.previous_fit = followed_fit
            self.previous_quality = previous_quality
            self.followed = propagated+1
            self.last_quality = {**previous_quality, "method": "semantic_lines_flow",
                                 "propagated_frames": self.followed,
                                 "refresh_failed": True, "anchor_age_ms": time_ms-anchor}
            h, w = frame.shape[:2]
            return followed_fit[0] @ np.diag([640/w, 360/h, 1.]), *followed_fit[1:]
        matrix, rms, check, quality = result
        self.previous_fit = (matrix, rms, check)
        self.previous_quality = quality
        self.anchor_stamp = time_ms
        h, w = frame.shape[:2]
        matrix = matrix @ np.diag([640/w, 360/h, 1.])
        self.last_quality = quality
        return matrix, rms, check
