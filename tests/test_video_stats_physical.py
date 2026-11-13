from copy import deepcopy
import unittest

from football_normalizer.video_stats_physical import calculate


def prepared(points, *, stamps=None, live=True, direction=1, error=.05):
    stamps = stamps if stamps is not None else [i*100 for i in range(len(points))]
    return {
        "length": 105., "width": 68., "scope_ms": stamps[-1]-stamps[0] if stamps else 0,
        "metric_scale_verified": True, "orientation_confirmed": True, "calibration_reviewed": True,
        "max_gap_ms": 500, "config": {}, "directions": {"1": {"A": direction}},
        "tracks": {"p": {"track_id": "p", "player_id": "P", "team": "A"}},
        "frames": [{"t_ms": t, "period": 1, "segment_id": "s", "cut": False, "live": live,
                    "phase": None, "players": [{"id": "p", "track_id": "p", "player_id": "P", "team": "A",
                    "x": x, "y": y, "error_m": error, "quality": "verified"}], "ball": None}
                   for t, (x, y) in zip(stamps, points)]}


def player(data):
    return calculate(data)["players"][0]


class VideoPhysicalStatsTests(unittest.TestCase):
    def test_raw_tracking_adapter_contract_without_changing_tracking(self):
        from football_normalizer.video_stats_common import prepare

        raw = {"version": "fbrpro-video-tracking-v1", "coordinate_length_m": 105,
               "coordinate_width_m": 68, "coordinate_frame": "stadium_fixed", "dimensions_assumed": False,
               "sample_fps": 10, "clips": [{"clip_id": "c", "video_start_ms": 0, "video_end_ms": 2100}],
               "frames": [{"video_time_ms": i*100, "period": 1, "clip_id": "c", "calibration_status": "verified",
                           "calibration_fit_rms_m": .1, "calibration_check_max_m": .2,
                           "objects": [{"track_id": "p", "player_id": "P", "team_id": "A", "object_type": "player",
                                        "observed": True, "estimated": False, "visibility": "observed",
                                        "projection_status": "verified", "x_m": 10+i*.4, "y_m": 10}]}
                          for i in range(21)]}
        before = deepcopy(raw)
        context = {"live_play": True, "orientation_confirmed": True, "directions": {"1": {"A": 1}}}
        result = player(prepare(raw, context))
        self.assertEqual(raw, before)
        self.assertEqual(result["movement"]["observed_distance_m"], 8)
        self.assertEqual(result["coverage"]["observed_ms"], 2000)
        self.assertEqual(result["coverage"]["scope_ms"], 2100)
        self.assertEqual(result["physical"]["mean_speed_m_s"], 4)

    def test_constant_speed_path_has_observed_distance_and_direction(self):
        data = prepared([(10+i*.4, 10) for i in range(21)])
        before = deepcopy(data)
        result = player(data)
        self.assertEqual(data, before)
        self.assertEqual(result["physical"]["status"], "estimate")
        self.assertEqual(result["physical"]["usable_ms"], 2000)
        self.assertAlmostEqual(result["movement"]["observed_distance_m"], 8)
        self.assertAlmostEqual(result["physical"]["mean_speed_m_s"], 4)
        self.assertAlmostEqual(result["movement"]["forward_m"], 8)
        self.assertEqual(result["movement"]["backward_m"], 0)
        data["directions"]["1"]["A"] = -1
        other = player(data)
        self.assertEqual(other["movement"]["forward_m"], 0)
        self.assertAlmostEqual(other["movement"]["backward_m"], 8)

    def test_missing_direction_does_not_invent_forward_movement(self):
        data = prepared([(10+i*.4, 10) for i in range(11)])
        data["directions"] = {}
        result = player(data)
        self.assertEqual(result["movement"]["observed_distance_m"], 4)
        self.assertIsNone(result["movement"]["forward_m"])
        self.assertEqual(result["movement"]["directional_time_ms"], 0)

    def test_physical_requires_live_dimensions_orientation_and_calibration_review(self):
        for gate in ("metric_scale_verified", "orientation_confirmed", "calibration_reviewed", "live"):
            with self.subTest(gate=gate):
                data = prepared([(10+i*.4, 10) for i in range(11)])
                if gate == "live":
                    for frame in data["frames"]:
                        frame["live"] = None
                else:
                    data[gate] = False
                result = player(data)
                self.assertIsNone(result["physical"]["mean_speed_m_s"])
                self.assertIsNone(result["movement"]["observed_distance_m"])
                self.assertAlmostEqual(result["movement"]["relative_distance_pct"], 4/105*100, places=3)
                self.assertEqual(result["movement"]["relative_usable_ms"], 1000)
                self.assertGreater(result["positioning"]["resolved_time_ms"], 0)

    def test_relative_windows_tolerate_small_projection_jitter(self):
        points = [(10+i*.4+(.55 if i % 2 else -.55), 10) for i in range(11)]
        result = player(prepared(points, error=.6))
        self.assertGreater(result["movement"]["relative_distance_pct"], 2)
        self.assertIsNone(result["physical"]["mean_speed_m_s"])

    def test_time_weighted_zone_occupancy_and_raw_observed_denominator(self):
        data = prepared([(10, 10)]*4, stamps=[0, 100, 200, 500])
        data["observed_by_track"] = {"p": [
            {"t_ms": f["t_ms"], "period": 1, "segment_id": "s"} for f in data["frames"]]}
        data["frames"][2]["players"] = []
        result = player(data)
        self.assertEqual(result["coverage"]["observed_ms"], 500)
        self.assertEqual(result["coverage"]["usable_ms"], 100)
        self.assertEqual(result["coverage"]["usable_observed_pct"], 20)
        self.assertEqual(result["positioning"]["zones"]["time_ms"][0], 100)
        self.assertEqual(result["positioning"]["zones"]["pct"][0], 100)

    def test_zone_boundaries_remain_uncertain_and_known_transitions_are_counted(self):
        result = player(prepared([(35, 10)]*11, error=.2))
        self.assertEqual(result["positioning"]["zones"]["uncertain_ms"], 1000)
        self.assertEqual(sum(result["positioning"]["zones"]["time_ms"]), 0)
        result = player(prepared([(34, 10), (36, 10)], stamps=[0, 500]))
        self.assertEqual(result["movement"]["zone_transitions"], [{"from": 0, "to": 1, "count": 1}])

    def test_cuts_segments_and_missing_samples_are_not_bridged(self):
        for change in ("cut", "segment", "missing"):
            with self.subTest(change=change):
                data = prepared([(10+i*.4, 10) for i in range(11)])
                if change == "cut":
                    data["frames"][5]["cut"] = True
                elif change == "segment":
                    for f in data["frames"][5:]:
                        f["segment_id"] = "new"
                else:
                    data["frames"][5]["players"] = []
                result = player(data)
                self.assertIsNone(result["movement"]["observed_distance_m"])

    def test_conflicting_track_identity_is_unavailable(self):
        data = prepared([(10+i*.4, 10) for i in range(21)])
        for f in data["frames"][10:]:
            f["players"][0]["player_id"] = "Q"
        result = player(data)
        self.assertEqual(result["positioning"]["status"], "unavailable")
        self.assertIn("Conflicting", result["positioning"]["reason"])
        self.assertIsNone(result["movement"]["observed_distance_m"])

    def test_sub_error_jitter_does_not_become_running_distance(self):
        data = prepared([(10+(.2 if i % 2 else -.2), 10) for i in range(21)], error=.5)
        result = player(data)
        self.assertEqual(result["movement"]["observed_distance_m"], 0)
        self.assertEqual(result["physical"]["below_noise_ms"], 2000)
        self.assertEqual(result["physical"]["sprint_bouts"], 0)

    def test_large_outlier_does_not_become_sprint(self):
        points = [(10+i*.2, 10) for i in range(11)]
        points[5] = (70, 10)
        result = player(prepared(points))
        self.assertIsNone(result["physical"]["max_speed_m_s"])
        self.assertGreater(result["coverage"]["rejected_interval_ms"], 0)

    def test_sprint_bouts_and_acceleration_are_window_estimates(self):
        result = player(prepared([(10+i*.9, 10) for i in range(21)]))
        self.assertEqual(result["physical"]["sprint_bouts"], 1)
        self.assertEqual(result["physical"]["sprint_distance_m"], 18)
        self.assertEqual(result["physical"]["high_speed_time_ms"], 2000)
        points, x = [(10, 10)], 10
        for speed in [2]*10+[4]*10+[7]*10:
            x += speed*.1
            points.append((x, 10))
        result = player(prepared(points))
        self.assertAlmostEqual(result["physical"]["max_acceleration_m_s2"], 3)
        self.assertEqual(result["physical"]["acceleration_bouts"], 1)

    def test_relationships_require_coincident_objects_and_ground_ball(self):
        data = prepared([(10, 10)]*11)
        for f in data["frames"]:
            f["ball"] = {"x": 13, "y": 14, "ground": True}
            f["players"] += [
                {"id": "friend", "player_id": "F", "team": "A", "x": 10, "y": 15, "error_m": .1},
                {"id": "opponent", "player_id": "O", "team": "B", "x": 13, "y": 10, "error_m": .1}]
        result = player(data)
        self.assertEqual(result["relationships"]["mean_distance_to_ball_m"], 5)
        self.assertEqual(result["relationships"]["mean_nearest_teammate_m"], 5)
        self.assertEqual(result["relationships"]["mean_nearest_opponent_m"], 3)
        for f in data["frames"]:
            f["ball"]["ground"] = False
        self.assertIsNone(player(data)["relationships"]["mean_distance_to_ball_m"])

    def test_team_shape_is_observed_subset_and_reviewed_line_spacing(self):
        data = prepared([(10, 10)]*11)
        for f in data["frames"]:
            f["players"] += [
                {"id": "a2", "team": "A", "x": 10, "y": 30, "error_m": .1},
                {"id": "a3", "team": "A", "x": 30, "y": 10, "error_m": .1},
                {"id": "a4", "team": "A", "x": 30, "y": 30, "error_m": .1}]
            f["team_lines"] = {"A": [["p", "a2"], ["a3", "a4"]]}
            f["defensive_lines"] = {"A": ["a3", "a4"]}
        result = calculate(data)
        shape = result["teams"][0]["shape"]
        self.assertEqual(shape["scope"], "simultaneously_observed_players")
        self.assertEqual(shape["width_m"], 20)
        self.assertEqual(shape["length_m"], 20)
        self.assertEqual(shape["convex_hull_area_m2"], 400)
        self.assertEqual(shape["line_spacing_m"], 20)
        self.assertEqual(result["players"][0]["relationships"]["mean_distance_to_own_defensive_line_m"], 20)

    def test_teleporting_counterpart_cannot_supply_relationship_distance(self):
        data = prepared([(10, 10)]*2, stamps=[0, 100])
        for i, f in enumerate(data["frames"]):
            f["players"].append({"id": "opponent", "team": "B", "player_id": "O",
                                 "x": 20 if i == 0 else 90, "y": 10, "error_m": .1})
            f["ball"] = {"x": 20 if i == 0 else 90, "y": 10, "ground": True}
        relations = player(data)["relationships"]
        self.assertIsNone(relations["mean_nearest_opponent_m"])
        self.assertEqual(relations["coverage_ms"]["mean_nearest_opponent_m"], 0)
        self.assertIsNone(relations["mean_distance_to_ball_m"])

    def test_acceleration_bouts_reset_after_unobserved_gap(self):
        points, stamps = [], []
        for start_ms, origin in ((0, 10), (4000, 25)):
            x = origin
            points.append((x, 10))
            stamps.append(start_ms)
            for i, speed in enumerate([2]*10+[5]*10, 1):
                x += speed*.1
                points.append((x, 10))
                stamps.append(start_ms+i*100)
        physical = player(prepared(points, stamps=stamps))["physical"]
        self.assertEqual(physical["acceleration_bouts"], 2)
        self.assertEqual(physical["max_acceleration_m_s2"], 3)
        self.assertIn("window estimates", physical["measurement"])


if __name__ == "__main__":
    unittest.main()
