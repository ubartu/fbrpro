from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from football_normalizer.video_statistics import analyze
from football_normalizer.video_stats_common import prepare


def tracking():
    return {"version": "fbrpro-video-tracking-v1", "match_id": "m", "video_id": "v", "plan_id": "p",
            "coordinate_frame": "stadium_fixed", "coordinate_length_m": 105, "coordinate_width_m": 68,
            "dimensions_assumed": False, "sample_fps": 10,
            "clips": [{"clip_id": "c", "video_start_ms": 0, "video_end_ms": 3000}],
            "frames": [{"video_time_ms": t, "period": 1, "clip_id": "c", "calibration_status": "verified",
                        "calibration_fit_rms_m": .05, "calibration_check_max_m": .1,
                        "objects": [{"object_type": "player", "track_id": f"p{i}", "player_id": f"person{i}",
                                     "team_id": "a" if i < 3 else "b", "observed": True, "estimated": False,
                                     "visibility": "observed", "projection_status": "verified",
                                     "x_m": 10+i*6+t*.003, "y_m": 10+i*5} for i in range(6)]}
                       for t in range(0, 3000, 100)]}


class VideoStatisticsTests(unittest.TestCase):
    def test_read_only_calculation_never_runs_detection(self):
        data = tracking()
        before = deepcopy(data)
        with patch("football_normalizer.video_tracking.run") as detector:
            result = analyze(data)
        detector.assert_not_called()
        self.assertEqual(data, before)
        self.assertEqual(result["version"], "fbrpro-video-stats-v1")
        self.assertEqual(len(result["players"]), 6)
        self.assertEqual(result["scope_ms"], 3000)
        self.assertIsNone(result["players"][0]["events"]["pass_attempts"]["value"])
        json.dumps(result, allow_nan=False)

    def test_actual_frame_values_are_part_of_analysis_fingerprint(self):
        data = tracking()
        a = analyze(data)
        data["frames"][0]["objects"][0]["x_m"] += .5
        b = analyze(data)
        self.assertNotEqual(a["analysis_fingerprint"], b["analysis_fingerprint"])
        data = tracking()
        data["dimensions_assumed"] = True
        self.assertNotEqual(a["analysis_fingerprint"], analyze(data)["analysis_fingerprint"])

    def test_period_direction_needs_explicit_orientation_review(self):
        data = tracking()
        context = {"directions": {"1": {"a": 1}}}
        self.assertEqual(prepare(data, context)["directions"], {})
        context["orientation_confirmed"] = True
        self.assertEqual(prepare(data, context)["directions"], context["directions"])

    def test_missing_calibration_is_counted_in_observability_not_as_zero(self):
        data = tracking()
        for f in data["frames"][10:20]:
            f["calibration_status"] = "uncalibrated"
        p = prepare(data)
        self.assertEqual(len(p["observed_by_track"]["p0"]), 30)
        self.assertEqual(sum(any(x["id"] == "p0" for x in f["players"]) for f in p["frames"]), 20)
        self.assertEqual(p["quality"]["excluded_observations"]["missing_pitch_mapping"], 60)

    def test_ambiguous_balls_identities_and_referees_are_excluded(self):
        data = tracking()
        for f in data["frames"]:
            f["objects"].append({**f["objects"][0], "track_id": "ref", "player_id": None, "inferred_role": "probable_referee"})
            for i in range(3):
                f["objects"].append({**f["objects"][0], "track_id": f"ball{i}", "object_type": "ball"})
        data["frames"][1]["objects"][0]["player_id"] = "switched-person"
        p = prepare(data, {"ground_ball_reviewed": True})
        self.assertNotIn("ref", p["tracks"])
        self.assertTrue(all(f["ball"] is None for f in p["frames"]))
        self.assertTrue(all(not any(x["id"] == "p0" for x in f["players"]) for f in p["frames"]))

    def test_scope_is_video_time_and_clipped_to_available_clip_spans(self):
        p = prepare(tracking(), {"start_ms": 500, "end_ms": 1600})
        self.assertEqual(p["scope_ms"], 1100)
        self.assertEqual([f["t_ms"] for f in p["frames"]], list(range(500, 1600, 100)))

    def test_video_only_recording_can_be_split_into_reviewed_halves(self):
        p = prepare(tracking(), {"period_intervals": [{"start_ms": 1500, "end_ms": 3000, "period": 2}]})
        self.assertEqual(p["scope_spans"], [{"period": 1, "start_ms": 0, "end_ms": 1500},
                                             {"period": 2, "start_ms": 1500, "end_ms": 3000}])
        self.assertEqual(p["frames"][14]["period"], 1)
        self.assertEqual(p["frames"][15]["period"], 2)
        self.assertNotEqual(p["frames"][14]["segment_id"], p["frames"][15]["segment_id"])

    def test_cohort_calibration_jump_breaks_statistics_not_source_tracks(self):
        data = tracking()
        for row in data["frames"][10]["objects"]:
            row["x_m"] += 20
        p = prepare(data)
        self.assertTrue(p["frames"][10]["cut"])
        self.assertNotEqual(p["frames"][9]["segment_id"], p["frames"][10]["segment_id"])
        self.assertTrue(all(len(f["objects"]) == 6 for f in data["frames"]))

    def test_frames_outside_declared_clip_spans_cannot_inflate_coverage(self):
        data = tracking()
        data["clips"] = [{"clip_id": "c", "video_start_ms": 0, "video_end_ms": 1000},
                         {"clip_id": "c", "video_start_ms": 2000, "video_end_ms": 3000}]
        p = prepare(data)
        self.assertEqual(p["scope_ms"], 2000)
        self.assertFalse(any(1000 <= f["t_ms"] < 2000 for f in p["frames"]))

    def test_disjoint_selected_spans_break_analysis_even_with_short_gap(self):
        data = tracking()
        data["clips"] = [{"clip_id": "c", "video_start_ms": 0, "video_end_ms": 1000},
                         {"clip_id": "c", "video_start_ms": 1050, "video_end_ms": 3000}]
        p = prepare(data)
        self.assertNotEqual(p["frames"][9]["segment_id"], p["frames"][10]["segment_id"])

    def test_source_period_transition_splits_one_clip_scope(self):
        data = tracking()
        for f in data["frames"][15:]:
            f["period"] = 2
        p = prepare(data)
        self.assertEqual([s["period"] for s in p["scope_spans"]], [1, 2])
        self.assertEqual(p["scope_spans"][0]["end_ms"], 1500)

    def test_incompatible_object_projection_cannot_claim_verified_calibration(self):
        data = tracking()
        for f in data["frames"]:
            for row in f["objects"]:
                row["projection_status"] = "ball_ground_assumed"
        p = prepare(data)
        self.assertFalse(p["calibration_reviewed"])
        self.assertFalse(any(f["players"] for f in p["frames"]))

    def test_invalid_and_overlapping_review_context_is_rejected(self):
        for context in [{"directions": {"1": {"a": True}}}, {"live_play": "yes"},
                        {"match_id": "another-match"},
                        {"defensive_lines": ["a"]}, {"team_lines": {"a": "not a line"}},
                        {"episodes": [{"period": 1, "start_ms": 0, "end_ms": 1000},
                                      {"period": 1, "start_ms": 500, "end_ms": 1500}]},
                        {"events": [{"id": "e", "type": "invented", "t_ms": 0, "track_id": "p0"}]}]:
            with self.subTest(context=context), self.assertRaises(ValueError):
                analyze(tracking(), context)

