"""Local football landmark inference and conservative ground-plane registration."""

from .video_inference import load_inference

def pitch_landmarks(length, width):
    """Roboflow's 32 landmark indices, using regulation marking dimensions."""
    mid_x, mid_y = length/2, width/2
    box_y = (mid_y-20.16, mid_y+20.16)
    goal_y = (mid_y-9.16, mid_y+9.16)
    end_y = (0, box_y[0], *goal_y, box_y[1], width)
    box_line_y = (box_y[0], *goal_y, box_y[1])
    return ([(0, y) for y in end_y] + [(5.5, y) for y in goal_y] + [(11, mid_y)]
            + [(16.5, y) for y in box_line_y]
            + [(mid_x, y) for y in (0, mid_y-9.15, mid_y+9.15, width)]
            + [(length-16.5, y) for y in box_line_y] + [(length-11, mid_y)]
            + [(length-5.5, y) for y in goal_y] + [(length, y) for y in end_y]
            + [(mid_x-9.15, mid_y), (mid_x+9.15, mid_y)])


def fit_landmarks(cv2, np, image_points, field_points, shape):
    """Return a fit only when spread, consensus and held-out points agree."""
    if len(image_points) < 6:
        return None
    matrix, mask = cv2.findHomography(image_points, field_points, cv2.RANSAC, 2.0)
    if matrix is None or mask is None:
        return None
    inliers = np.flatnonzero(mask.ravel())
    if len(inliers) < max(6, .7*len(image_points)):
        return None
    pixels, pitch = image_points[inliers], field_points[inliers]

    def spread(points, minimum):
        return cv2.contourArea(cv2.convexHull(points.astype(np.float32))) >= minimum

    if not spread(pixels, .007*shape[0]*shape[1]) or not spread(pitch, 175):
        return None
    # Fit from broadly spread anchors, testing the remainder before refitting.
    hull = cv2.convexHull(pitch.astype(np.float32), returnPoints=False).ravel()
    withheld = [i for i in range(len(pitch)) if i not in hull]
    if not withheld:
        withheld = [int(hull[0])]
        hull = hull[1:]
    if len(hull) < 4:
        # A triangular view still needs four non-collinear correspondences.
        hull = np.array(list(hull) + withheld[:4-len(hull)])
        withheld = [i for i in withheld if i not in hull]
    if not withheld:
        return None
    candidate, _ = cv2.findHomography(pixels[hull], pitch[hull], 0)
    if candidate is None:
        return None
    check = cv2.perspectiveTransform(pixels[withheld].reshape(-1, 1, 2), candidate).reshape(-1, 2)
    check_error = float(np.linalg.norm(check-pitch[withheld], axis=1).max())
    if not np.isfinite(check_error) or check_error > 2.75:
        return None
    matrix, _ = cv2.findHomography(pixels, pitch, 0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        return None
    projected = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected-pitch, axis=1)
    if errors.max() > 2.75:
        return None
    return matrix, float(np.sqrt(np.mean(errors**2))), check_error


class PitchModel:
    """Read the 640 px YOLO pose ONNX export with the 32 football landmarks."""
    size = 640

    def __init__(self, cv2, np, path, length, width, *, inference_backend="opencv", inference_cache=None):
        self.cv2, self.np = cv2, np
        self.engine = load_inference(cv2, path, inference_backend, "the pitch landmark model", inference_cache)
        self.field_points = np.asarray(pitch_landmarks(length, width), dtype=np.float32)

    def fit(self, frame):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        scale = self.size/max(h, w)
        width, height = round(w*scale), round(h*scale)
        x, y = (self.size-width)//2, (self.size-height)//2
        image = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        image[y:y+height, x:x+width] = cv2.resize(frame, (width, height))
        output = self.engine.forward(cv2.dnn.blobFromImage(image, 1/255, swapRB=True))
        if output.ndim != 3 or output.shape[:2] != (1, 101):
            raise ValueError("The pitch model must output one class and 32 landmark triples.")
        best = output[0, :, int(output[0, 4].argmax())]
        if not np.isfinite(best).all() or best[4] < .45:
            return None
        points = best[5:].reshape(32, 3)
        pixels = ((points[:, :2]-[x, y])/scale).astype(np.float32)
        usable = ((points[:, 2] >= .50) & (pixels[:, 0] >= 3) & (pixels[:, 0] < w-3)
                  & (pixels[:, 1] >= 3) & (pixels[:, 1] < h-3))
        return fit_landmarks(cv2, np, pixels[usable], self.field_points[usable], frame.shape)
