import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from football_normalizer.video_motion import CameraMotion
from football_normalizer.video_tracking import (
    CameraFollow, Detection, Tracker, calibration_from, canonical_rows, identity_from, kit_sample, object_row,
    pitch_detections, follow_pitch_boundary, run,
)


def person(x):
    return Detection("player", .9, (x, 20, x+14, 60))


class TrackingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_short_outline_recovery_requires_camera_motion_turf_and_recent_anchor(self):
        import cv2
        import numpy as np

        frame = np.full((360, 640, 3), (34, 138, 68), np.uint8)
        boundary = {"polygon_xy": [[20, 40], [600, 40], [600, 330], [20, 330]]}
        motion = np.array([[1., 0, 4], [0, 1., 2]])
        recovered = follow_pitch_boundary(cv2, np, frame, boundary, motion, 400)
        self.assertEqual(recovered["source"], "tracked_pitch_outline")
        self.assertEqual(recovered["anchor_age_ms"], 400)
        self.assertAlmostEqual(min(p[0] for p in recovered["polygon_xy"]), 24)
        self.assertIsNone(follow_pitch_boundary(cv2, np, frame, boundary, motion, 751))
        self.assertIsNone(follow_pitch_boundary(cv2, np, frame, boundary, None, 100))
        self.assertIsNone(follow_pitch_boundary(cv2, np, frame*0, boundary, motion, 100))
        self.assertIsNone(follow_pitch_boundary(cv2, np, frame, None, motion, 100))
        concave = {"polygon_xy": [[20, 40], [600, 40], [600, 330], [400, 330],
                                  [400, 180], [240, 180], [240, 330], [20, 330]]}
        kept = follow_pitch_boundary(cv2, np, frame, concave, motion, 100)
        contour = np.asarray(kept["polygon_xy"], np.float32)
        self.assertLess(cv2.pointPolygonTest(contour, (324., 262.), False), 0)
    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_stable_live_tracker_never_calls_experimental_geometry(self):
        import cv2
        import numpy as np

        frame = np.zeros((100, 160, 3), np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame)]*3+[(False, None)]
        stamps = iter([0, 100, 200])
        cap.get.side_effect = lambda key: (next(stamps) if key == cv2.CAP_PROP_POS_MSEC else
            {cv2.CAP_PROP_FRAME_WIDTH: 160, cv2.CAP_PROP_FRAME_HEIGHT: 100,
             cv2.CAP_PROP_POS_FRAMES: 1}.get(key, 25))
        window = {"period": 1, "video_start_ms": 0, "video_end_ms": 300}
        plan = {"version": "fbrpro-video-plan-v1", "match_id": "m", "video_id": "v", "plan_id": "p",
                "alignment_segments": [{**window, "id": "s", "kind": "preview",
                                        "match_start_ms": 0, "match_end_ms": 300}],
                "clips": [{**window, "id": "c", "segment_id": "s"}]}
        geometry = Mock()
        geometry.update.side_effect = lambda *args, **kwargs: (None, {
            "segment_id": "c:geometry:0", "usable": False, "drift_m": 4.,
            "continuity_status": "rejected_jump", "orientation_status": "unresolved"})
        motion = Mock()
        motion.update.return_value = np.array([[1., 0, 0], [0, 1., 0]])
        model = Mock()
        model.fit.return_value = (np.eye(3), .1, .2)
        model.last_quality = None
        model.engine.description.return_value = {}
        detections = [person(10), Detection("ball", .9, (30, 30, 34, 34))]

        def detect(*args):
            self.assertIsNotNone(args[4], "Measurement rejection removed the detection boundary")
            return detections, {"polygon_xy": [[0, 0], [105, 0], [105, 68]], "source": "automatic_landmarks"}, "estimated", 0

        with TemporaryDirectory() as directory:
            source = Path(directory)/"video"
            source.write_bytes(b"test video")
            with (patch.object(cv2, "VideoCapture", return_value=cap),
                  patch("football_normalizer.video_tracking.YoloX"),
                  patch("football_normalizer.video_tracking.PitchModel", return_value=model),
                  patch("football_normalizer.video_tracking.CameraMotion", return_value=motion),
                  patch("football_normalizer.video_geometry.PitchContinuity", return_value=geometry),
                  patch("football_normalizer.video_tracking.pitch_detections", side_effect=detect)):
                data = run(source, plan, source, {"coordinate_length_m": 105, "coordinate_width_m": 68, "segments": []},
                           pitch_model=source)
        self.assertEqual(motion.update.call_count, 3)
        geometry.update.assert_not_called()
        self.assertEqual(len(data["frames"]), 3)
        self.assertNotIn("spatial_summary", data)
        ids = []
        for f in data["frames"]:
            self.assertNotIn("pitch_geometry", f)
            self.assertEqual(f["calibration_status"], "automatic")
            self.assertTrue(all(o["observed"] and o["x_m"] is not None for o in f["objects"]))
            self.assertEqual(len(f["objects"]), 2)
            ids.append([o["track_id"] for o in f["objects"]])
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_camera_motion_estimates_a_pan_and_resets_between_shots(self):
        import cv2
        import numpy as np

        texture = np.random.default_rng(9).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        source = cv2.resize(texture, (480, 270))
        target = cv2.warpAffine(source, np.float32([[1, 0, 9], [0, 1, 5]]), (480, 270))
        motion = CameraMotion(cv2, np)
        self.assertIsNone(motion.update(source, []))
        pan = motion.update(target, [(30, 50, 100, 140)])
        self.assertIsNotNone(pan)
        np.testing.assert_allclose(pan[:, 2], [9, 5], atol=1)
        motion.reset()
        self.assertIsNone(motion.update(target, []))

    def test_camera_pan_keeps_track_identity_when_boxes_no_longer_overlap(self):
        class Shift:
            def __matmul__(self, point):
                x, y, _ = point
                return x+100, y

        tracker = Tracker("pan")
        first, = tracker.update([person(10)], 0)
        moved, = tracker.update([person(110)], 200, Shift())
        self.assertEqual(moved.id, first.id)
        self.assertEqual(moved.last.box, person(110).box)
        self.assertFalse(moved.missing)

    def test_full_video_reads_past_duration_hint_until_decoder_eof(self):
        timestamps = [0, 40, 80, 250, 310, 420, 600]
        reads = iter(enumerate(timestamps))
        position, timestamp = -1, -1
        capture = Mock()
        capture.isOpened.return_value = True

        def read():
            nonlocal position, timestamp
            sample = next(reads, None)
            if sample is None:
                return False, None
            position, timestamp = sample
            return True, None

        capture.read.side_effect = read
        capture.get.side_effect = lambda key: {
            "width": 640, "height": 360, "fps": 25, "timestamp": timestamp, "frame": position+1,
        }[key]
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FRAME_WIDTH="width",
                              CAP_PROP_FRAME_HEIGHT="height", CAP_PROP_FPS="fps",
                              CAP_PROP_POS_MSEC="timestamp", CAP_PROP_POS_FRAMES="frame",
                              absdiff=lambda *_: 0)
        np = SimpleNamespace(mean=lambda x: x)
        window = {"period": 1, "video_start_ms": 0, "video_end_ms": 300}
        plan = {"version": "fbrpro-video-plan-v1", "match_id": "m", "video_id": "v", "plan_id": "p",
                "alignment_segments": [{**window, "id": "s", "kind": "preview",
                                        "match_start_ms": 0, "match_end_ms": 300}],
                "clips": [{**window, "id": "c", "segment_id": "s"}]}
        calibration = {"coordinate_length_m": 105, "coordinate_width_m": 68, "segments": []}
        progress = []
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.write_bytes(b"stub")
            with (patch("football_normalizer.video_tracking.vision", return_value=(cv2, np)),
                  patch("football_normalizer.video_tracking.YoloX"),
                  patch("football_normalizer.video_tracking.small_frame", return_value=0),
                  patch("football_normalizer.video_tracking.pitch_detections",
                        return_value=([], None, "unavailable", 0))):
                report = run(source, plan, source, calibration, through_eof=True, on_progress=progress.append)
                with self.assertRaisesRegex(ValueError, "different clips or tracking settings"):
                    run(source, plan, source, calibration, through_eof=True,
                        identity_map={"c:pitch-v2:1": {"team_id": "team"}})

        self.assertEqual(report["sample_fps"], 10)
        self.assertEqual(progress, [0, 250, 310, 420, 600])
        self.assertEqual([f["period_time_ms"] for f in report["frames"]], progress)
        self.assertEqual(report["clips"][0]["decoded_frames"], len(timestamps))
        self.assertEqual(report["clips"][0]["video_end_ms"], 780)
        self.assertFalse(report["clips"][0]["decode_ended_before_clip_end"])
        capture.set.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_kit_hint_distinguishes_neutral_and_colored_shirts_from_grass(self):
        import cv2
        import numpy as np

        frame = np.full((80, 80, 3), (25, 120, 60), dtype=np.uint8)
        box = (20, 10, 60, 70)
        self.assertIsNone(kit_sample(cv2, np, frame, box))
        frame[24:42, 30:50] = (20, 235, 105)
        self.assertEqual(kit_sample(cv2, np, frame, box), "green")
        frame[24:42, 30:50] = (20, 25, 220)
        self.assertEqual(kit_sample(cv2, np, frame, box), "red")
        frame[24:42, 30:40] = (235, 20, 105)
        self.assertIsNone(kit_sample(cv2, np, frame, box))
        frame[24:42, 30:50] = (25, 25, 25)
        self.assertEqual(kit_sample(cv2, np, frame, box), "black")
        frame[24:42, 30:50] = (220, 220, 220)
        self.assertEqual(kit_sample(cv2, np, frame, box), "white")

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_verified_pitch_gate_filters_off_field_people_before_tracking(self):
        import cv2
        import numpy as np

        frame = np.zeros((160, 230, 3), dtype=np.uint8)
        row = {"segment_id": "s1", "image_points": [[20,20],[125,20],[125,88],[20,88]],
               "pitch_points": [[0,0],[105,0],[105,68],[0,68]],
               "check_image_points": [[72,53]], "check_pitch_points": [[52,33]]}
        length, width, maps = calibration_from(
            {"coordinate_length_m": 105, "coordinate_width_m": 68, "segments": [row]}, 2)

        class Detector:
            def detect(self, frame, region, *, accepts_person):
                return [d for d in (person(26), person(175)) if accepts_person(d.anchor())]

        rows, boundary, status, rejected = pitch_detections(cv2, np, frame, Detector(), maps[0], length, width)
        self.assertEqual((status, rejected), ("verified", 1))
        self.assertEqual(boundary["source"], "verified_calibration")
        self.assertEqual([d.box for d in rows], [person(26).box])
        self.assertEqual(pitch_detections(cv2, np, frame, None, None, length, width),
                         ([], None, "unavailable", 0))

    def test_one_to_one_association_preserves_tracks_and_reports_missing(self):
        tracker = Tracker("clip-1", 450)
        tracks = tracker.update([person(10), person(65)], 0)
        ids = [t.id for t in tracks]
        tracks = tracker.update([person(66), person(13)], 200)
        self.assertEqual([t.id for t in tracks], ids)
        self.assertEqual([t.last.box[0] for t in tracks], [13, 66])
        tracks = tracker.update([person(16)], 400)
        row = object_row(tracks[1], None, 105, 68, {})
        self.assertFalse(row["observed"])
        self.assertFalse(row["estimated"])
        self.assertIsNone(row["bbox_xyxy"])
        self.assertIsNone(row["x_m"])
        self.assertEqual([t.id for t in tracker.update([person(20)], 800)], ids[:1])
        tracker.reset()
        self.assertNotIn(tracker.update([person(20)], 1000)[0].id, ids)

    def test_reviewed_mapping_rejects_unknown_or_wrong_team_player(self):
        plan = {"team_id": "home", "players": [{"id": "known"}]}
        self.assertEqual(identity_from({"clip:1": {"player_id": "known"}}, plan),
                         {"clip:1": ("known", "home")})
        self.assertEqual(identity_from({"clip:1": {"team_id": "away"}}, plan),
                         {"clip:1": (None, "away")})
        for mapping in ({"player_id": "missing"}, {"player_id": "known", "team_id": "away"}):
            with self.assertRaises(ValueError):
                identity_from({"clip:1": mapping}, plan)
        plan["identity_roster"] = [{"id": "known", "team_id": "home"}, {"id": "opponent", "team_id": "away"}]
        self.assertEqual(identity_from({"clip:2": {"player_id": "opponent"}}, plan),
                         {"clip:2": ("opponent", "away")})

    def test_only_reviewed_players_enter_the_canonical_table(self):
        tracks = Tracker("clip-1").update([person(2), person(60)], 0)
        objects = [object_row(t, None, 105, 68, {"clip-1:2": ("p2", "away")}) for t in tracks]
        report = {
            "match_id": "m", "video_id": "v", "source_sha256": "a"*64,
            "coordinate_frame": "stadium_fixed", "coordinate_length_m": 105,
            "coordinate_width_m": 68,
            "frames": [{"period": 1, "period_time_ms": 0, "frame_id": 0, "objects": objects}],
        }
        rows = list(canonical_rows(report))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["object_id"], "p2")
        self.assertEqual(rows[0]["source"]["record_id"], "clip-1:2")
        self.assertIsNone(rows[0]["x_m"])

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_projection_and_independent_check_gate(self):
        row = {"segment_id": "s1", "image_points": [[0,0],[210,0],[210,136],[0,136]],
               "pitch_points": [[0,0],[105,0],[105,68],[0,68]],
               "check_image_points": [[104,66]], "check_pitch_points": [[52,33]]}
        config = {"coordinate_length_m": 105, "coordinate_width_m": 68, "segments": [row]}
        length, width, maps = calibration_from(config, 2)
        self.assertEqual(maps[0].status, "verified")
        self.assertAlmostEqual(maps[0].project((104,66))[0], 52, places=5)
        track = Tracker("clip").update([person(20)], 0)[0]
        result = object_row(track, maps[0], length, width, {})
        self.assertAlmostEqual(result["x_m"], 13.5)
        self.assertAlmostEqual(result["y_m"], 30)
        self.assertIsNone(result["player_id"])
        row["check_pitch_points"] = [[61,33]]
        _, _, maps = calibration_from(config, 2)
        self.assertEqual(maps[0].status, "rejected")
        self.assertIsNone(object_row(track, maps[0], length, width, {})["x_m"])
        row["check_image_points"] = [[0,0]]
        row["check_pitch_points"] = [[0,0]]
        with self.assertRaises(ValueError):
            calibration_from(config, 2)

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "optional vision dependencies not installed")
    def test_camera_pan_updates_projection_and_blank_shot_is_rejected(self):
        import cv2
        import numpy as np

        pixels = [[60, 60], [270, 60], [270, 196], [60, 196]]
        config = {"coordinate_length_m": 105, "coordinate_width_m": 68, "segments": [{
            "segment_id": "s1", "follow_camera": True, "video_start_ms": 0, "video_end_ms": 1000,
            "image_points": pixels, "pitch_points": [[0, 0], [105, 0], [105, 68], [0, 68]],
            "check_image_points": [[164, 126]], "check_pitch_points": [[52, 33]],
        }]}
        _, _, maps = calibration_from(config, 2)
        texture = np.random.default_rng(3).integers(0, 256, (52, 68, 3), dtype=np.uint8)
        image = cv2.resize(texture, (340, 260), interpolation=cv2.INTER_CUBIC)
        pan = np.float32([[1, 0, 9], [0, 1, 6]])
        moved = cv2.warpAffine(image, pan, (340, 260))
        follow = CameraFollow(cv2, np, maps[0], image, 2)
        updated = follow.update(moved)
        self.assertEqual(updated.status, "verified")
        self.assertAlmostEqual(updated.project((173, 132))[0], 52, delta=.1)
        self.assertAlmostEqual(updated.project((173, 132))[1], 33, delta=.1)
        self.assertEqual(follow.update(np.zeros_like(moved)).status, "rejected")
