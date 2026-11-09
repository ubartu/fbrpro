import importlib.util
import unittest

from football_normalizer.video_pitch import detect_pitch


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                     "optional vision dependencies not installed")
class PitchBoundaryTests(unittest.TestCase):
    def test_painted_touchline_excludes_grass_runoff(self):
        import cv2
        import numpy as np

        image = np.full((360, 640, 3), (36, 41, 53), dtype=np.uint8)
        turf = np.int32([[15, 350], [80, 116], [580, 119], [630, 350]])
        cv2.fillConvexPoly(image, turf, (34, 138, 68))
        cv2.line(image, (21, 329), (625, 329), (218, 223, 221), 2)
        cv2.line(image, (320, 119), (320, 329), (218, 223, 221), 2)
        boundary = detect_pitch(cv2, np, image)
        self.assertIsNotNone(boundary)
        self.assertTrue(boundary.contains((320, 240)))
        self.assertTrue(boundary.contains((310, 328)))
        self.assertFalse(boundary.contains((320, 340)))
        self.assertFalse(boundary.contains((320, 100)))
        self.assertTrue(all(0 <= x < 640 and 0 <= y < 360 for x, y in boundary.polygon_xy))

    def test_unmarked_and_fragmented_green_regions_stay_unknown(self):
        import cv2
        import numpy as np

        grass = (34, 138, 68)
        image = np.full((360, 640, 3), grass, dtype=np.uint8)
        self.assertIsNone(detect_pitch(cv2, np, image))
        image[:, 301:339] = (30, 30, 30)
        cv2.line(image, (0, 150), (300, 150), (220, 220, 220), 2)
        cv2.line(image, (339, 150), (639, 150), (220, 220, 220), 2)
        self.assertIsNone(detect_pitch(cv2, np, image))
