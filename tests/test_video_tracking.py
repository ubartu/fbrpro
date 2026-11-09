import importlib.util
import unittest

from football_normalizer.video_tracking import (
    CameraFollow, Detection, Tracker, calibration_from, canonical_rows, identity_from, kit_sample, object_row,
    pitch_detections,
)


def person(x):
    return Detection("player", .9, (x, 20, x+14, 60))


class TrackingTests(unittest.TestCase):
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
