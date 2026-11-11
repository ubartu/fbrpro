"""Estimate the visible field in a broadcast frame for on-screen review."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class PitchBoundary:
    """A screen-space approximation, with no pitch coordinates or identity claims."""

    polygon_xy: list[list[float]]
    _contour: object = field(repr=False)
    _distance: object = field(repr=False)
    source: str = "visible_turf_boundary"

    def contains(self, point):
        # Allow two original-frame pixels for a foot landing on a painted line.
        return self._distance(self._contour, tuple(map(float, point)), True) >= -2.0


def trim_along_line(np, polygon, start, end):
    """Clip the polygon to the side of a line containing its center."""
    dx, dy = end - start
    normal = np.array([-dy, dx], dtype=float) / math.hypot(dx, dy)
    if (polygon.mean(axis=0) - start) @ normal < 0:
        normal *= -1
    distances = (polygon - start) @ normal
    kept = []
    for i, (p, a) in enumerate(zip(polygon, distances)):
        j = (i + 1) % len(polygon)
        q, b = polygon[j], distances[j]
        if a >= 0:
            kept.append(p)
        if a * b < 0:
            kept.append(p + (q - p) * a / (a - b))
    return np.asarray(kept, dtype=np.float32)


def detect_pitch(cv2, np, frame):
    """Return a conservative visible turf polygon, or None if it is ambiguous.

    A single large, coherent green area with painted-line evidence is required.
    A long line close to its perimeter can separate the field from grass runoff.
    Faded markings, artificial colors, close-ups, obstructions and overlays can
    defeat the estimate. Use it for display and candidacy only, never calibration
    or movement evidence. Recompute on each sampled frame as the camera moves.
    """
    original_h, original_w = frame.shape[:2]
    if original_w < 160 or original_h < 90:
        return None
    width = min(original_w, 640)
    height = round(original_h * width / original_w)
    if height < 90:
        return None
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (27, 53, 36), (85, 255, 255))

    def ellipse(size):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    clean = cv2.morphologyEx(green, cv2.MORPH_OPEN, ellipse(5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, ellipse(13))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    ordered = sorted(contours, key=cv2.contourArea, reverse=True)
    turf = ordered[0]
    area = cv2.contourArea(turf)
    _, _, turf_width, turf_height = cv2.boundingRect(turf)
    if (area < width * height * .22 or turf_width < width * .47 or turf_height < height * .33
            or len(ordered) > 1 and cv2.contourArea(ordered[1]) > area * .30):
        return None
    hull = cv2.convexHull(turf)
    solid = area >= cv2.contourArea(hull) * .90
    if area < cv2.contourArea(hull) * .80:
        return None
    fill = cv2.drawContours(np.zeros_like(green), [turf], 0, 255, cv2.FILLED)
    saturation, brightness = hsv[:, :, 1], hsv[:, :, 2]
    ridge = cv2.morphologyEx(brightness, cv2.MORPH_TOPHAT, ellipse(7))
    marks = ((fill > 0) & (saturation <= 115) & (brightness >= 125) & (ridge >= 13)).astype("uint8") * 255
    lines = cv2.HoughLinesP(marks, 1, np.pi / 180, 22,
                            minLineLength=max(24, round(min(width, height) * .13)), maxLineGap=12)
    if lines is None:
        return None

    if not solid:
        # Sun/shadow boundaries can make a real pitch concave in the colour
        # mask. Require independent marking directions, then retain the turf
        # contour itself rather than filling its missing areas with a hull.
        long = [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines[:, 0]
                if math.hypot(x2-x1, y2-y1) >= width*.10]
        angles = [math.degrees(math.atan2(y2-y1, x2-x1)) % 180 for x1, y1, x2, y2 in long]
        diverse = any(20 <= abs(a-b) <= 160 for a in angles for b in angles)
        endpoints = np.asarray([(x1, y1) for x1, y1, _, _ in long]+
                               [(x2, y2) for _, _, x2, y2 in long], np.float32)
        if not diverse or cv2.contourArea(cv2.convexHull(endpoints)) < .02*width*height:
            return None

    reference = hull if solid else turf
    epsilon = .007 if solid else .004
    outline = cv2.approxPolyDP(reference, epsilon * cv2.arcLength(reference, True), True).reshape(-1, 2).astype(np.float32)
    polygon = outline
    minimum_trimmed_area = cv2.contourArea(outline) * .91
    # A visible near/far touchline may lie a little inside the contiguous turf.
    # Long, almost horizontal lines close to the hull can remove only a small
    # strip. Midfield, penalty-area markings, and isolated highlights stay inside.
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) < width * .32 or abs(y2 - y1) > abs(x2 - x1) * .46:
            continue
        start, end = np.array([x1, y1]), np.array([x2, y2])
        distances = [abs(cv2.pointPolygonTest(hull, tuple(map(float, point)), True))
                     for point in (start, (start + end) / 2, end)]
        if max(distances) > height * .08:
            continue
        trimmed = trim_along_line(np, polygon, start, end)
        if len(trimmed) >= 3 and cv2.contourArea(trimmed) >= minimum_trimmed_area:
            polygon = trimmed

    polygon *= [(original_w - 1) / (width - 1), (original_h - 1) / (height - 1)]
    return PitchBoundary(polygon.astype(float).round(2).tolist(), polygon, cv2.pointPolygonTest)
