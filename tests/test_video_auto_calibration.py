import importlib.util
import unittest
from unittest.mock import Mock

from football_normalizer.video_auto_calibration import fit_landmarks, pitch_landmarks
from football_normalizer.video_pitch_lines import LinePitchModel, fit_pitch_lines, pitch_lines
from football_normalizer.video_tracking import Calibration, calibrated_boundary


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                     "optional vision dependencies not installed")
class AutomaticCalibrationTests(unittest.TestCase):
    def test_failed_semantic_refresh_keeps_line_supported_map_but_expires(self):
        import cv2
        import numpy as np

        model = LinePitchModel.__new__(LinePitchModel)
        model.cv2, model.np = cv2, np
        model.lines = pitch_lines(105, 68)
        model.landmark = None
        model.motion = Mock()
        model.motion.update.return_value = np.array([[1., 0, 0], [0, 1., 0]])
        model.previous_fit = (np.eye(3), .1, .2)
        model.previous_quality = {"method": "semantic_lines", "matched_lines": 5}
        model.last_stamp, model.anchor_stamp, model.followed = 200, 0, 2
        model.follow = Mock(return_value=True)
        model.engine = Mock()
        model.engine.forward.return_value = np.zeros((1, 29, 360, 640), np.float32)
        frame = np.zeros((360, 640, 3), np.uint8)
        for stamp in range(300, 1001, 100):
            fit = model.fit(frame, stamp)
            self.assertIsNotNone(fit)
            self.assertTrue(model.last_quality["refresh_failed"])
            self.assertEqual(model.last_quality["anchor_age_ms"], stamp)
        self.assertIsNone(model.fit(frame, 1100))
        self.assertIsNone(model.previous_fit)

    def test_failed_refresh_cannot_carry_mapping_without_current_painted_lines(self):
        import cv2
        import numpy as np

        model = LinePitchModel.__new__(LinePitchModel)
        model.cv2, model.np = cv2, np
        model.lines = pitch_lines(105, 68)
        model.landmark = None
        model.motion = Mock()
        model.motion.update.return_value = np.array([[1., 0, 0], [0, 1., 0]])
        model.previous_fit = (np.eye(3), .1, .2)
        model.previous_quality = {"method": "semantic_lines", "matched_lines": 5}
        model.last_stamp, model.anchor_stamp, model.followed = 200, 0, 2
        model.follow = Mock(return_value=False)
        model.engine = Mock()
        model.engine.forward.return_value = np.zeros((1, 29, 360, 640), np.float32)
        self.assertIsNone(model.fit(np.zeros((360, 640, 3), np.uint8), 300))

    def test_visible_pitch_is_clipped_when_distant_corners_cross_the_horizon(self):
        import cv2
        import numpy as np

        camera = np.array([[4., 0, 0], [0, 4., 0], [-.02, 0, 1.]])
        cal = Calibration("s", None, None, np.linalg.inv(camera), "automatic", 0, 0)
        boundary = calibrated_boundary(cv2, np, (360, 640, 3), cal, 105, 68)
        self.assertIsNotNone(boundary)
        self.assertGreater(cv2.contourArea(np.asarray(boundary, np.float32)), 200000)
        for pixel in boundary:
            self.assertTrue(0 <= pixel[0] < 640 and 0 <= pixel[1] < 360)
            x, y = cal.project(pixel)
            self.assertTrue(-.001 <= x <= 105.001 and -.001 <= y <= 68.001)

    def test_semantic_lines_fit_known_camera_and_reject_parallel_constraints(self):
        import cv2
        import numpy as np

        lines = pitch_lines(105, 68)
        camera = np.array([[8, 1, 35], [1, 6, 30], [.001, .002, 1.]])
        observed = {}
        for name, points in lines.items():
            field = np.asarray(points, np.float32)
            if len(field) == 2:
                field = np.linspace(field[0], field[1], 80).astype(np.float32)
            observed[name] = cv2.perspectiveTransform(field[:, None], camera)[:, 0]
        matrix, rms, check, quality = fit_pitch_lines(cv2, np, observed, lines, (1080, 1920))
        spot = np.array([[[24., 31.]]], np.float32)
        np.testing.assert_allclose(cv2.perspectiveTransform(cv2.perspectiveTransform(spot, camera), matrix),
                                   spot, atol=.01)
        self.assertLess(max(rms, check), .01)
        self.assertGreaterEqual(quality["matched_lines"], 4)
        parallel = {name: points for name, points in observed.items()
                    if name in ("Side line top", "Side line bottom", "Big rect. left top", "Big rect. left bottom")}
        self.assertIsNone(fit_pitch_lines(cv2, np, parallel, lines, (1080, 1920)))

    def test_line_following_uses_current_camera_motion_and_expires(self):
        import cv2
        import numpy as np

        model = LinePitchModel.__new__(LinePitchModel)
        model.cv2, model.np = cv2, np
        model.lines = pitch_lines(105, 68)
        model.landmark = None
        model.motion = Mock()
        model.motion.update.return_value = np.array([[1., 0, 4], [0, 1., 2]])
        model.previous_fit = (np.eye(3), .1, .2)
        model.previous_quality = {"method": "semantic_lines", "matched_lines": 5}
        model.last_stamp, model.followed = 0, 0
        model.follow = Mock(return_value=True)
        model.engine = Mock()
        frame = np.zeros((360, 640, 3), np.uint8)
        matrix, _, _ = model.fit(frame, 100)
        np.testing.assert_allclose(matrix, [[1, 0, -4], [0, 1, -2], [0, 0, 1]])
        self.assertEqual(model.last_quality["method"], "semantic_lines_flow")
        model.engine.forward.assert_not_called()
        model.engine.forward.return_value = np.zeros((1, 29, 360, 640), np.float32)
        self.assertIsNone(model.fit(frame, 600))
        self.assertIsNone(model.previous_fit)
        model.reset()
        self.assertIsNone(model.last_stamp)
        model.motion.reset.assert_called_once()

    def test_landmark_fallback_must_agree_with_visible_semantic_markings(self):
        import cv2
        import numpy as np

        model = LinePitchModel.__new__(LinePitchModel)
        model.cv2, model.np = cv2, np
        model.lines = pitch_lines(105, 68)
        model.landmark = Mock()
        camera = np.array([[4, .2, 15], [.3, 4, 20], [.0001, .0002, 1.]])
        observed = {}
        for name in ("Side line top", "Side line left", "Big rect. left main", "Circle left"):
            points = np.asarray(model.lines[name], np.float32)
            observed[name] = cv2.perspectiveTransform(points[:, None], camera)[:, 0]
        frame = np.zeros((360, 640, 3), np.uint8)
        model.landmark.fit.return_value = (np.linalg.inv(camera), .1, .2)
        self.assertIsNotNone(model.checked_landmarks(frame, observed))
        shifted = np.array([[1., 0, 100], [0, 1., 100], [0, 0, 1.]]) @ camera
        model.landmark.fit.return_value = (np.linalg.inv(shifted), .1, .2)
        self.assertIsNone(model.checked_landmarks(frame, observed))

    def test_fits_a_known_projection_and_rejects_collinear_points(self):
        import cv2
        import numpy as np

        landmarks = np.asarray(pitch_landmarks(105, 68), dtype=np.float32)
        self.assertEqual(landmarks.shape, (32, 2))
        field = landmarks[[0, 1, 4, 5, 6, 7, 9, 12, 13, 16]]
        pitch_to_image = np.array([[10, 2, 70], [1, 8, 50], [.002, .004, 1]])
        pixels = cv2.perspectiveTransform(field[:, None, :], pitch_to_image)[:, 0]
        matrix, rms, check = fit_landmarks(cv2, np, pixels, field, (1080, 1920))
        spot = np.array([[[24.0, 31.0]]], dtype=np.float32)
        recovered = cv2.perspectiveTransform(cv2.perspectiveTransform(spot, pitch_to_image), matrix)
        np.testing.assert_allclose(recovered, spot, atol=.01)
        self.assertLess(max(rms, check), .01)
        collinear = np.asarray([(10*i, 20*i) for i in range(8)], dtype=np.float32)
        self.assertIsNone(fit_landmarks(cv2, np, collinear, collinear, (1080, 1920)))
