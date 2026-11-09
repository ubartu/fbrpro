"""Estimate frame-to-frame camera motion from static image features."""


class CameraMotion:
    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self.previous = None
        self.mask = None

    def reset(self):
        self.previous = self.mask = None

    def update(self, frame, boxes):
        if frame is None:
            self.reset()
            return None
        cv2, np = self.cv2, self.np
        height, width = frame.shape[:2]
        scale = min(1, 480/width)
        size = (round(width*scale), max(1, round(height*scale)))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        # Moving foreground cannot be used as camera reference.
        for left, top, right, bottom in boxes:
            a = (round(left*scale)-3, round(top*scale)-3)
            b = (round(right*scale)+3, round(bottom*scale)+3)
            cv2.rectangle(mask, a, b, 0, cv2.FILLED)
        previous, previous_mask = self.previous, self.mask
        self.previous, self.mask = gray, mask
        if previous is None or previous.shape != gray.shape:
            return None
        points = cv2.goodFeaturesToTrack(previous, 180, .02, 7, mask=previous_mask)
        if points is None or len(points) < 16:
            return None
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if moved is None:
            return None
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None)
        if back is None:
            return None
        valid = (forward.ravel() == 1) & (backward.ravel() == 1)
        valid &= np.linalg.norm((back-points)[:, 0], axis=1) < 1.5
        if valid.sum() < 16:
            return None
        matrix, consensus = cv2.estimateAffinePartial2D(
            points[valid], moved[valid], method=cv2.RANSAC, ransacReprojThreshold=2)
        if matrix is None or consensus is None or consensus.sum() < max(16, .6*valid.sum()):
            return None
        anchors = points[valid][consensus.ravel().astype(bool)]
        if cv2.contourArea(cv2.convexHull(anchors)) < .03*gray.size:
            return None
        zoom = np.hypot(matrix[0, 0], matrix[1, 0])
        if not .85 <= zoom <= 1.18 or abs(matrix[1, 0]) > .10:
            return None
        if np.linalg.norm(matrix[:, 2]) > .30*size[0]:
            return None
        matrix[:, 2] /= scale
        return matrix
