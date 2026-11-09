import unittest

from football_normalizer.video_ball import BallTracker
from football_normalizer.video_tracking import Detection, object_row


def ball(x, confidence=.6):
    return Detection("ball", confidence, (x-3, 97, x+3, 103))


class BallTrackingTests(unittest.TestCase):
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
