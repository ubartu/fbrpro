import importlib.util
import unittest

from football_normalizer.video_auto_calibration import fit_landmarks, pitch_landmarks


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                     "optional vision dependencies not installed")
class AutomaticCalibrationTests(unittest.TestCase):
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
