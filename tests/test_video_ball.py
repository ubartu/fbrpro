import importlib.util
import unittest
from unittest.mock import patch

from football_normalizer.video_ball import BallDetector, BallTracker
from football_normalizer.video_ball_benchmark import spotting_events
from football_normalizer.video_tracking import Detection, object_row


def ball(x, confidence=.6):
    return Detection("ball", confidence, (x-3, 97, x+3, 103))


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                     "optional vision dependencies not installed")
class BallCropTests(unittest.TestCase):
    def test_crop_boxes_are_mapped_back_to_the_original_pitch_boundary(self):
        import cv2
        import numpy as np

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        boundary = {"polygon_xy": [(300, 0), (639, 0), (639, 359), (300, 359)]}
        with patch("football_normalizer.video_ball.load_inference") as load:
            load.return_value.forward.return_value = np.array([[
                [640, 64], [640, 640], [48, 48], [48, 48], [.8, .9],
            ]])
            detector = BallDetector(cv2, np, "unused")
            rows = detector.detect(frame, boundary, region=(260, 100, 420, 260))
        self.assertEqual(rows, [(.8, (337, 177, 343, 183))])


class BallTrackingTests(unittest.TestCase):
    def test_crop_recovery_requires_recent_evidence_and_a_nearby_detection(self):
        tracker = BallTracker("clip", 1000)
        centers = []

        def recover(center):
            centers.append(center)
            return [ball(185, .5), ball(300, .9)]

        self.assertIsNone(tracker.update([], 0, recover=recover))
        initial = tracker.update([ball(150)], 200, recover=recover)
        restored = tracker.update([], 400, recover=recover)
        self.assertEqual(centers, [(150, 100)])
        self.assertEqual(restored.id, initial.id)
        self.assertEqual(restored.last.anchor(), (185, 100))
        self.assertTrue(restored.recovered)
        gap = tracker.update([], 800, recover=recover)
        self.assertTrue(gap.missing)
        self.assertFalse(gap.recovered)
        self.assertEqual(len(centers), 1)

    def test_crop_detection_cannot_steer_or_replace_full_frame_tracking(self):
        baseline = BallTracker("clip", 1000)
        enhanced = BallTracker("clip", 1000)
        for tracker in (baseline, enhanced):
            tracker.update([ball(150)], 0)
        self.assertTrue(baseline.update([], 200).missing)
        self.assertTrue(enhanced.update([], 200, recover=lambda _: [ball(230)]).recovered)
        candidates = [ball(150, .55), ball(300, .6)]

        def unexpected_crop(_):
            self.fail("A full-frame observation is available")

        self.assertEqual(enhanced.update(candidates, 400, recover=unexpected_crop),
                         baseline.update(candidates, 400))

    def test_camera_motion_and_continuity_win_over_a_distant_candidate(self):
        class Pan:
            def __matmul__(self, point):
                x, y, _ = point
                return x+100, y

        tracker = BallTracker("clip", 1000)
        initial = tracker.update([ball(150)], 0)
        track = tracker.update([ball(252, .19), ball(800, .9)], 200, Pan())
        self.assertEqual(track.id, initial.id)
        self.assertEqual(track.last.anchor(), (252, 100))

    def test_gaps_and_ambiguous_detections_do_not_publish_ball_positions(self):
        tracker = BallTracker("clip", 1000)
        self.assertIsNone(tracker.update([ball(150, .19)], 0))
        self.assertIsNone(tracker.update([ball(150), ball(450)], 200))
        track = tracker.update([ball(150)], 400)
        gap = tracker.update([], 600)
        self.assertEqual(gap.id, track.id)
        row = object_row(gap, None, 105, 68, {})
        self.assertFalse(row["observed"])
        self.assertIsNone(row["bbox_xyxy"])
        self.assertIsNone(row["x_m"])
        self.assertIsNone(tracker.update([], 1200))
        restarted = tracker.update([ball(150)], 1400)
        self.assertNotEqual(restarted.id, track.id)
        tracker.reset()
        self.assertNotEqual(tracker.update([ball(150)], 1600).id, restarted.id)


class SoccerNetSpottingTests(unittest.TestCase):
    def test_actions_use_the_aligned_half_and_video_position(self):
        data = {"annotations": [
            {"gameTime": "2 - 00:04", "position": "4801", "label": "Shot"},
            {"gameTime": "1 - 00:04", "position": "4801", "label": "Pass"},
            {"gameTime": "2 - 00:06", "position": "6082", "label": "Out", "visibility": "not shown"},
            {"gameTime": "2 - 00:03", "position": "3700", "label": "High Pass"},
        ]}
        self.assertEqual(spotting_events(data, 2, 4000), [(-300, "High Pass"), (801, "Shot")])
