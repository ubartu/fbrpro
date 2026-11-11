import importlib.util
import unittest
from unittest.mock import Mock

from football_normalizer.video_geometry import PitchContinuity, reviewed_report, position_quality
from football_normalizer.video_tracking import Calibration


def report(points, stamps=None, *, period=1):
    stamps = stamps or [i*200 for i in range(len(points))]
    return {"version": "fbrpro-video-tracking-v1", "coordinate_length_m": 105, "coordinate_width_m": 68,
            "sample_fps": 5, "frames": [
                {"video_time_ms": t, "clip_id": "c", "period": period,
                 "pitch_geometry": {"segment_id": "s", "usable": True},
                 "objects": [{"track_id": "t", "object_type": "player", "observed": True,
                              "visibility": "observed", "estimated": False, "team_id": "a",
                              "x_m": x, "y_m": y, "projection_status": "automatic",
                              "position_uncertainty_m": .1}]} for t, (x, y) in zip(stamps, points)]}


def review(orientation="normal", period=1, direction=1):
    return {"segments": {"s": {"orientation": orientation, "period": period, "live_play": True}},
            "directions": {str(period): {"a": direction}}}


class SpatialSummaryTests(unittest.TestCase):
    def test_unreviewed_or_non_live_positions_do_not_create_stats(self):
        data = report([(10, 10), (11, 10)])
        self.assertEqual(reviewed_report(data)["spatial_summary"]["players"][0]["usable_ms"], 0)
        settings = review()
        settings["segments"]["s"]["live_play"] = False
        p = reviewed_report(data, settings)["spatial_summary"]["players"][0]
        self.assertEqual(p["usable_ms"], 0)
        self.assertIsNone(p["forward_m"])

    def test_time_weighted_zones_do_not_count_frames_or_fill_gaps(self):
        data = report([(10, 10), (10, 10), (10, 10), (10, 10)], [0, 100, 500, 1700])
        p = reviewed_report(data, review())["spatial_summary"]["players"][0]
        self.assertEqual(p["usable_ms"], 500)
        self.assertEqual(p["observed_interval_ms"], 500)
        self.assertEqual(p["zones_ms"][0], 500)
        self.assertEqual(p["zones_pct"][0], 100)

    def test_zone_boundary_uncertainty_and_bad_positions_are_separate(self):
        data = report([(35, 10), (35.05, 10), (35.1, 10)])
        data["frames"][-1]["objects"][0]["position_uncertainty_m"] = 4
        p = reviewed_report(data, review())["spatial_summary"]["players"][0]
        self.assertEqual(p["usable_ms"], 200)
        self.assertEqual(p["boundary_uncertain_ms"], 200)
        self.assertEqual(sum(p["zones_pct"]), 0)
        self.assertEqual(p["coverage_pct"], 50)

    def test_team_direction_and_period_control_forward_backward(self):
        data = report([(10, 10), (11, 10), (12, 10)])
        first = reviewed_report(data, review())["spatial_summary"]["players"][0]
        second = reviewed_report(data, review(period=2, direction=-1))["spatial_summary"]["players"][0]
        self.assertEqual(first["forward_m"], 2)
        self.assertEqual(first["backward_m"], 0)
        self.assertEqual(second["forward_m"], 0)
        self.assertEqual(second["backward_m"], 2)
        unknown = review()
        unknown["directions"] = {}
        p = reviewed_report(data, unknown)["spatial_summary"]["players"][0]
        self.assertEqual(p["usable_ms"], 400)
        self.assertIsNone(p["forward_m"])

    def test_flipped_review_is_idempotent_and_preserves_source(self):
        data = report([(10, 10), (11, 10)])
        once = reviewed_report(data, review("flipped"))
        twice = reviewed_report(once, review("flipped"))
        self.assertEqual(once, twice)
        self.assertEqual(twice["frames"][0]["objects"][0]["x_m"], 95)
        self.assertEqual(data["frames"][0]["objects"][0]["x_m"], 10)
        restored = reviewed_report(twice, review())
        self.assertEqual(restored["frames"][0]["objects"][0]["x_m"], 10)

    def test_no_movement_across_cut_identity_switch_or_untrusted_map(self):
        for change in ("cut", "identity", "mapping", "segment"):
            with self.subTest(change=change):
                data = report([(10, 10), (11, 10)])
                f = data["frames"][-1]
                if change == "cut":
                    f["camera_cut_suspected"] = True
                elif change == "identity":
                    f["objects"][0]["player_id"] = "changed"
                elif change == "mapping":
                    f["pitch_geometry"]["usable"] = False
                else:
                    f["pitch_geometry"]["segment_id"] = "other"
                p = reviewed_report(data, review())["spatial_summary"]["players"][0]
                self.assertEqual(p["usable_ms"], 0)
                self.assertIsNone(p["forward_m"])

    def test_windowed_movement_exceeds_noise_without_summing_jitter(self):
        data = report([(10+i*.5, 10) for i in range(11)], [i*100 for i in range(11)])
        data["sample_fps"] = 10
        for f in data["frames"]:
            f["objects"][0]["position_uncertainty_m"] = .8
        p = reviewed_report(data, review())["spatial_summary"]["players"][0]
        self.assertEqual(p["forward_m"], 5)
        self.assertEqual(p["directional_usable_ms"], 1000)

    def test_referees_and_duplicate_tracks_do_not_supply_movement(self):
        data = report([(10, 10), (11, 10)])
        for f in data["frames"]:
            f["objects"][0]["inferred_role"] = "probable_referee"
        self.assertEqual(reviewed_report(data, review())["spatial_summary"]["players"], [])
        data = report([(10, 10), (11, 10)])
        data["frames"][0]["objects"] *= 2
        self.assertEqual(reviewed_report(data, review())["spatial_summary"]["players"][0]["usable_ms"], 0)

    def test_review_rejects_unknown_segments_and_invalid_direction(self):
        data = report([(10, 10), (11, 10)])
        settings = review()
        settings["directions"]["1"]["a"] = True
        with self.assertRaises(ValueError):
            reviewed_report(data, settings)
        settings = review()
        settings["directions"]["1"] = {"kit:team_a": 1, "kit:team_b": 1}
        with self.assertRaises(ValueError):
            reviewed_report(data, settings)
        settings = review()
        settings["segments"]["missing"] = settings["segments"].pop("s")
        with self.assertRaises(ValueError):
            reviewed_report(data, settings)


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"), "vision dependencies needed")
class PitchContinuityTests(unittest.TestCase):
    def test_existing_tracker_motion_is_reused_without_second_feature_pass(self):
        import cv2
        import numpy as np
        g = PitchContinuity(cv2, np, 105, 68, "clip")
        g.motion = Mock()
        image = np.zeros((680, 1050, 3), np.uint8)
        h = np.diag([.1, .1, 1.])
        g.update(image, 0, h, camera_transform=None)
        matrix, quality = g.update(image, 100, h, camera_transform=np.array([[1., 0, 0], [0, 1., 0]]))
        self.assertTrue(quality["usable"])
        self.assertIsNotNone(matrix)
        g.motion.update.assert_not_called()

    def test_known_pan_zoom_and_symmetric_solution_keep_ground_points_fixed(self):
        import cv2
        import numpy as np
        g = PitchContinuity(cv2, np, 105, 68, "clip")
        g.motion = Mock()
        g.motion.update.return_value = None
        image = np.zeros((680, 1050, 3), np.uint8)
        h0 = np.diag([.1, .1, 1.])
        g.update(image, 0, h0)
        movement = np.array([[1.05, 0, 12.], [0, 1.05, 7.]])
        g.motion.update.return_value = movement
        h1 = h0 @ np.linalg.inv(np.vstack((movement, [0, 0, 1.])))
        flip = np.array([[-1., 0, 105], [0, -1., 68], [0, 0, 1.]])
        corrected, quality = g.update(image, 100, flip @ h1)
        np.testing.assert_allclose(corrected, h1, atol=1e-8)
        self.assertTrue(quality["mirror_corrected"])
        self.assertLess(quality["drift_m"], .001)
        self.assertEqual(quality["orientation_status"], "unresolved")

    def test_jump_rejected_and_motion_failure_starts_unresolved_segment(self):
        import cv2
        import numpy as np
        g = PitchContinuity(cv2, np, 105, 68, "clip")
        g.motion = Mock()
        g.motion.update.return_value = None
        image = np.zeros((680, 1050, 3), np.uint8)
        h0 = np.diag([.1, .1, 1.])
        _, initial = g.update(image, 0, h0)
        g.motion.update.return_value = np.array([[1., 0, 0], [0, 1., 0]])
        wrong = h0.copy()
        wrong[0, 2] += 10
        matrix, quality = g.update(image, 100, wrong)
        self.assertIsNone(matrix)
        self.assertEqual(quality["continuity_status"], "rejected_jump")
        g.motion.update.return_value = None
        _, reset = g.update(image, 200, h0)
        self.assertNotEqual(reset["segment_id"], initial["segment_id"])
        self.assertEqual(reset["orientation_status"], "unresolved")
        g.reset()
        self.assertIsNone(g.previous)

    def test_local_position_sensitivity_increases_with_projection_scale(self):
        import numpy as np
        row = {"object_type": "player", "x_m": 10, "bbox_xyxy": [90, 30, 110, 100]}
        near = Calibration("s", None, None, np.diag([.1, .1, 1.]), "automatic", .1, .2)
        far = Calibration("s", None, None, np.diag([1., 1., 1.]), "automatic", .1, .2)
        a = position_quality(np, row, near, None, (680, 1050))
        b = position_quality(np, row, far, None, (680, 1050))
        self.assertGreater(b, a)

    def test_actual_feature_flow_preserves_ground_points_through_pan_and_zoom(self):
        import cv2
        import numpy as np
        image = np.random.default_rng(22).integers(20, 220, (270, 480, 3), np.uint8)
        g = PitchContinuity(cv2, np, 105, 68, "clip")
        h0 = np.diag([105/480, 68/270, 1.])
        g.update(image, 0, h0)
        camera = np.eye(3)
        errors = []
        ground = np.array([[[20., 20.]], [[50., 40.]], [[80., 30.]]], np.float32)
        for i in range(1, 7):
            step = np.array([[1.005, 0, 2.], [0, 1.005, 1.], [0, 0, 1.]])
            camera = step @ camera
            frame = cv2.warpPerspective(image, camera, (480, 270))
            expected = h0 @ np.linalg.inv(camera)
            fitted, quality = g.update(frame, i*100, expected)
            self.assertIsNotNone(fitted)
            self.assertEqual(quality["continuity_status"], "camera_checked")
            pixels = cv2.perspectiveTransform(ground, np.linalg.inv(expected))
            restored = cv2.perspectiveTransform(pixels, fitted)
            errors.extend(np.linalg.norm(restored-ground, axis=2).ravel())
        self.assertLess(max(errors), .001)

    def test_repeated_rejected_maps_reacquire_an_unreviewed_segment(self):
        import cv2
        import numpy as np
        g = PitchContinuity(cv2, np, 105, 68, "clip")
        g.motion = Mock()
        g.motion.update.return_value = np.array([[1., 0, 0], [0, 1., 0]])
        image = np.zeros((680, 1050, 3), np.uint8)
        h0 = np.diag([.1, .1, 1.])
        _, initial = g.update(image, 0, h0)
        shifted = h0.copy()
        shifted[0, 2] += 10
        for t in (100, 200, 300):
            result, _ = g.update(image, t, shifted)
            self.assertIsNone(result)
        result, quality = g.update(image, 400, shifted)
        self.assertIsNotNone(result)
        self.assertNotEqual(quality["segment_id"], initial["segment_id"])
        self.assertEqual(quality["orientation_status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
