"""Exercise geometric candidates, review gates, cuts and event completeness."""
import copy
import unittest

from football_normalizer.video_stats_behaviors import calculate


def player(id_, x, y, team="home", named=True):
    return {"id": id_, "track_id": id_, "player_id": "player-"+id_ if named else None,
            "team": team, "x": x, "y": y, "error_m": .1, "quality": "verified"}


def fixture(rows, *, phase="attack", possession="home", **episode):
    frames = [{"t_ms": i*500, "period": 1, "segment_id": "s", "cut": False, "live": True,
               "phase": phase, "players": items, "ball": {"x": 50, "y": 30, "ground": True}}
              for i, items in enumerate(rows)]
    data = {"frames": frames, "tracks": {p["id"]: {k: p[k] for k in ("track_id", "player_id", "team")}
                                          for items in rows for p in items},
            "length": 100., "width": 60., "max_gap_ms": 750, "directions": {"1": {"home": 1, "away": -1}},
            "config": {}, "scope_ms": (len(rows)-1)*500,
            "scope_spans": [{"period": 1, "start_ms": 0, "end_ms": (len(rows)-1)*500}],
            "metric_scale_verified": True}
    context = {"episodes": [{"start_ms": 0, "end_ms": (len(rows)-1)*500, "period": 1,
                             "phase": phase, "possession_team_id": possession, **episode}]}
    return data, context


def stats(report, id_="runner"):
    return next(p for p in report["players"] if p["id"] == id_)


class VideoStatsBehaviorsTests(unittest.TestCase):
    def test_support_and_overlap_are_candidates_with_observable_denominators(self):
        data, context = fixture([[player("runner", x, 15), player("carrier", 50, 30)]
                                 for x in [45, 48, 50, 52, 55]], ball_carrier_id="carrier")
        out = calculate(data, context)
        p = stats(out)
        self.assertEqual(p["off_ball"]["support_runs"]["count"], 1)
        self.assertEqual(p["off_ball"]["overlaps"]["count"], 1)
        self.assertEqual(p["off_ball"]["support_runs"]["observed_ms"], 2000)
        self.assertEqual(p["off_ball"]["support_runs"]["status"], "estimated")
        self.assertEqual(p["off_ball"]["support_runs"]["evidence"][0]["status"], "review_required")
        self.assertIsNone(p["events"]["passes_completed"]["value"])

    def test_missing_direction_carrier_or_live_review_remains_unavailable(self):
        data, context = fixture([[player("runner", x, 15), player("carrier", 50, 30)]
                                 for x in [45, 48, 50, 52, 55]], ball_carrier_id="carrier")
        for missing in ("direction", "carrier", "live"):
            d, c = copy.deepcopy(data), copy.deepcopy(context)
            if missing == "direction":
                d["directions"] = {}
            elif missing == "carrier":
                c["episodes"][0].pop("ball_carrier_id")
            else:
                for f in d["frames"]:
                    f["live"] = None
            metric = stats(calculate(d, c))["off_ball"]["support_runs"]
            self.assertIsNone(metric["count"], missing)
            self.assertEqual(metric["status"], "unavailable")

    def test_recovery_and_pressure_need_appropriate_opposition_context(self):
        data, context = fixture([[player("runner", x, 30), player("carrier", 52, 30, "away")]
                                 for x in [40, 42, 44, 46, 48]],
                                phase="settled_defence", possession="away", ball_carrier_id="carrier")
        metric = stats(calculate(data, context))["off_ball"]["pressure_approaches"]
        self.assertEqual(metric["count"], 1)
        data, context = fixture([[player("runner", x, 30)] for x in [60, 58, 56, 54, 52]],
                                phase="recovery", possession="away")
        metric = stats(calculate(data, context))["off_ball"]["recovery_runs"]
        self.assertEqual(metric["count"], 1)

    def test_pressure_cannot_assign_opposition_to_unknown_carrier_team(self):
        data, context = fixture([[player("runner", x, 30), player("carrier", 52, 30, None)]
                                 for x in [40, 42, 44, 46, 48]],
                                phase="settled_defence", possession="away", ball_carrier_id="carrier")
        self.assertIsNone(stats(calculate(data, context))["off_ball"]["pressure_approaches"]["count"])

    def test_camera_cut_or_impossible_jump_cannot_create_a_run(self):
        for cut in (True, False):
            data, context = fixture([[player("runner", x, 15), player("carrier", 70, 30)]
                                     for x in [30, 30, 60, 60, 60]], ball_carrier_id="carrier")
            data["frames"][2]["cut"] = cut
            metric = stats(calculate(data, context))["off_ball"]["support_runs"]
            self.assertEqual(metric["count"], 0)
            self.assertLess(metric["observed_ms"], metric["eligible_ms"])

    def test_carrier_teleport_or_identity_switch_cannot_create_overlap(self):
        for change in ("teleport", "identity"):
            data, context = fixture([[player("runner", x, 15), player("carrier", 50, 30)]
                                     for x in [45, 46, 47, 48, 49]], ball_carrier_id="carrier")
            for frame in data["frames"][2:]:
                carrier = frame["players"][1]
                carrier["x"] = 30
                if change == "identity":
                    carrier["player_id"] = "different-player"
            metric = stats(calculate(data, context))["off_ball"]["overlaps"]
            self.assertEqual(metric["count"], 0)
            self.assertLess(metric["observed_ms"], metric["eligible_ms"])

    def test_lane_availability_needs_ground_ball_and_complete_visible_opponents(self):
        rows = [[player("runner", 20, 30), player("opponent", 35, 45, "away")] for _ in range(5)]
        data, context = fixture(rows, phase="build_up", opponent_team_id="away",
                                opponent_ids=["opponent"], lane_context_complete=True)
        output = stats(calculate(data, context))["off_ball"]
        self.assertEqual(output["passing_lane_availability"]["value_ms"], 2000)
        self.assertEqual(output["deep_passing_outlet"]["pct"], 100)
        context["episodes"][0]["lane_context_complete"] = False
        self.assertIsNone(stats(calculate(data, context))["off_ball"]["passing_lane_availability"]["value_ms"])
        context["episodes"][0]["lane_context_complete"] = True
        for f in data["frames"]:
            f["ball"]["ground"] = False
        self.assertIsNone(stats(calculate(data, context))["off_ball"]["passing_lane_availability"]["value_ms"])

    def test_missing_middle_observation_does_not_bridge_state_time(self):
        rows = [[player("runner", 20, 30), player("opponent", 35, 45, "away")] for _ in range(5)]
        data, context = fixture(rows, phase="build_up", opponent_team_id="away",
                                opponent_ids=["opponent"], lane_context_complete=True)
        data["frames"][2]["players"] = [data["frames"][2]["players"][1]]
        metric = stats(calculate(data, context))["off_ball"]["passing_lane_availability"]
        self.assertEqual(metric["observed_ms"], 1000)
        self.assertEqual(metric["coverage_pct"], 50)

    def test_screening_and_between_line_geometry(self):
        rows = [[player("runner", 35, 30), player("d1", 25, 10), player("d2", 25, 50)] for _ in range(5)]
        data, context = fixture(rows, phase="settled_defence", possession="away", defender_ids=["d1", "d2"])
        metric = stats(calculate(data, context))["off_ball"]["central_screening"]
        self.assertEqual(metric["value_ms"], 2000)
        rows = [[player("runner", 50, 30), player("m1", 40, 10, "away"), player("m2", 40, 50, "away"),
                 player("d1", 60, 10, "away"), player("d2", 60, 50, "away")] for _ in range(5)]
        data, context = fixture(rows, opponent_team_id="away", opponent_midfielder_ids=["m1", "m2"],
                                opponent_defender_ids=["d1", "d2"])
        metric = stats(calculate(data, context))["off_ball"]["between_line_availability"]
        self.assertEqual(metric["pct"], 100)

    def test_events_require_named_identity_review_and_complete_scope(self):
        data, context = fixture([[player("runner", 30, 30)] for _ in range(5)])
        context["events"] = [{"id": "p1", "type": "pass", "t_ms": 500, "period": 1,
                              "track_id": "runner", "reviewed": True, "outcome": "complete"}]
        events = stats(calculate(data, context))["events"]
        self.assertIsNone(events["pass_attempts"]["value"])
        self.assertEqual(events["pass_attempts"]["confirmed_count"], 1)
        context["event_coverage"] = [{"period": 1, "start_ms": 0, "end_ms": 2000,
                                      "types": ["pass", "touch", "tackle", "interception", "shot", "press"], "complete": True}]
        events = stats(calculate(data, context))["events"]
        self.assertEqual(events["pass_attempts"]["value"], 1)
        self.assertEqual(events["passes_completed"]["value"], 1)
        self.assertEqual(events["pass_completion_pct"]["value"], 100)
        self.assertEqual(events["shots"]["value"], 0)
        self.assertIsNone(events["possession_ms"]["value"])
        data["tracks"]["runner"]["player_id"] = None
        self.assertIsNone(stats(calculate(data, context))["events"]["pass_attempts"]["value"])

    def test_incomplete_outcomes_and_out_of_scope_events_are_not_false_success(self):
        data, context = fixture([[player("runner", 30, 30)] for _ in range(5)])
        context["event_coverage"] = [{"period": 1, "start_ms": 0, "end_ms": 2000, "types": ["pass", "press"], "complete": True}]
        context["events"] = [{"id": "p1", "type": "pass", "t_ms": 500, "track_id": "runner", "reviewed": True},
                             {"id": "p2", "type": "pass", "t_ms": 2500, "track_id": "runner", "reviewed": True, "outcome": "complete"},
                             {"id": "p3", "type": "press", "t_ms": 1000, "track_id": "runner", "reviewed": True, "outcome": "successful"},
                             {"id": "p4", "type": "press", "t_ms": 1500, "track_id": "runner", "reviewed": False, "outcome": "successful"}]
        events = stats(calculate(data, context))["events"]
        self.assertEqual(events["pass_attempts"]["value"], 1)
        self.assertIsNone(events["passes_completed"]["value"])
        self.assertEqual(events["successful_presses"]["value"], 1)
        context["event_coverage"][0]["end_ms"] = 1000
        self.assertIsNone(stats(calculate(data, context))["events"]["successful_presses"]["value"])

    def test_confirmed_visible_track_actions_produce_partial_counts(self):
        data, context = fixture([[player("runner", 30, 30, named=False)] for _ in range(5)])
        data["observed_by_track"] = {"runner": [{"t_ms": 500, "period": 1, "segment_id": "s"}]}
        event = {"id": "s1", "type": "shot", "t_ms": 500, "period": 1, "track_id": "runner",
                 "reviewed": True, "track_confirmed": True}
        context["events"] = [event, {**event, "id": "s2", "t_ms": 1000}, {**event, "id": "s3", "track_confirmed": False}]
        shots = stats(calculate(data, context))["events"]["shots"]
        self.assertEqual(shots["confirmed_count"], 1)
        self.assertEqual(shots["status"], "partial")
        self.assertIsNone(shots["value"])

    def test_explicit_non_live_intervals_do_not_count_reviewed_actions(self):
        data, context = fixture([[player("runner", 30, 30)] for _ in range(5)])
        context["events"] = [{"id": "a", "type": "shot", "t_ms": 500, "track_id": "runner", "reviewed": True},
                             {"id": "b", "type": "shot", "t_ms": 1500, "track_id": "runner", "reviewed": True}]
        context["event_coverage"] = [{"period": 1, "start_ms": 0, "end_ms": 2000, "types": ["shot"], "complete": True}]
        context["live_intervals"] = [{"start_ms": 500, "end_ms": 1500, "period": 1, "live_play": False}]
        self.assertEqual(stats(calculate(data, context))["events"]["shots"]["value"], 1)
        context["live_play"] = False
        self.assertIsNone(stats(calculate(data, context))["events"]["shots"]["value"])

    def test_reception_classification_uses_reviewed_arrival_not_pass_release(self):
        rows = [[player("runner", 50, 30), player("m1", 40, 10, "away"), player("m2", 40, 50, "away"),
                 player("d1", 60, 10, "away"), player("d2", 60, 50, "away")] for _ in range(5)]
        data, context = fixture(rows, opponent_team_id="away", opponent_midfielder_ids=["m1", "m2"],
                                opponent_defender_ids=["d1", "d2"])
        context["event_coverage"] = [{"period": 1, "start_ms": 0, "end_ms": 2000, "types": ["pass"], "complete": True}]
        context["events"] = [{"id": "p", "type": "pass", "t_ms": 500, "track_id": "m1",
                              "recipient_track_id": "runner", "reviewed": True, "outcome": "complete"}]
        metric = stats(calculate(data, context))["off_ball"]["between_line_receptions"]
        self.assertIsNone(metric["count"])
        context["events"].append({"id": "r", "type": "reception", "t_ms": 1000,
                                   "track_id": "runner", "reviewed": True})
        context["event_coverage"][0]["types"].append("reception")
        metric = stats(calculate(data, context))["off_ball"]["between_line_receptions"]
        self.assertEqual(metric["count"], 1)
        self.assertEqual(metric["evidence"][0]["start_ms"], 1000)
        self.assertEqual(stats(calculate(data, context))["events"]["receptions"]["value"], 1)
        data["directions"] = {}
        self.assertIsNone(stats(calculate(data, context))["off_ball"]["between_line_receptions"]["count"])

    def test_possession_is_reviewed_team_time_never_nearest_ball(self):
        data, context = fixture([[player("runner", 50, 30)] for _ in range(5)])
        self.assertEqual(calculate(data, context)["possession"]["status"], "unavailable")
        context["possession_intervals"] = [{"start_ms": 0, "end_ms": 1000, "period": 1, "team_id": "home", "reviewed": True},
                                            {"start_ms": 1000, "end_ms": 2000, "period": 1, "team_id": "away", "reviewed": True}]
        possession = calculate(data, context)["possession"]
        self.assertEqual(possession["reviewed_ms"], 2000)
        self.assertEqual([t["possession_pct_of_reviewed_time"] for t in possession["teams"]], [50, 50])

    def test_proximity_transfer_is_review_required_not_a_completed_pass(self):
        data, context = fixture([[player("runner", 30, 30), player("other", 40, 30)] for _ in range(5)])
        for i, f in enumerate(data["frames"]):
            f["ball"]["x"] = 30 if i < 2 else 40
        report = calculate(data, context)
        self.assertEqual(len(report["event_candidates"]), 1)
        self.assertEqual(report["event_candidates"][0]["status"], "review_required")
        self.assertIsNone(stats(report)["events"]["passes_completed"]["value"])
        data["frames"][2]["cut"] = True
        self.assertEqual(calculate(data, context)["event_candidates"], [])

    def test_root_analyzer_orientation_gate_and_scope_filter(self):
        from football_normalizer.video_statistics import analyze
        raw = {"version": "fbrpro-video-tracking-v1", "coordinate_length_m": 100, "coordinate_width_m": 60,
               "dimensions_assumed": False, "sample_fps": 2, "match_id": "m", "video_id": "v", "plan_id": "p",
               "clips": [{"clip_id": "c", "video_start_ms": 0, "video_end_ms": 2000}], "frames": []}
        for i, x in enumerate([45, 48, 50, 52, 55]):
            objects = [{"object_type": "player", "track_id": p["id"], "player_id": p["player_id"], "team_id": p["team"],
                        "x_m": p["x"], "y_m": p["y"], "observed": True, "estimated": False, "visibility": "observed",
                        "projection_status": "verified"} for p in (player("runner", x, 15), player("carrier", 50, 30))]
            raw["frames"].append({"video_time_ms": i*500, "period": 1, "clip_id": "c", "calibration_status": "verified",
                                  "calibration_fit_rms_m": .1, "calibration_check_max_m": .2, "objects": objects})
        context = {"live_play": True, "directions": {"1": {"home": 1}},
                   "episodes": [{"period": 1, "start_ms": 0, "end_ms": 2000, "phase": "attack",
                                 "possession_team_id": "home", "ball_carrier_id": "carrier"}]}
        unchanged = copy.deepcopy(raw)
        report = analyze(raw, context)
        self.assertIsNone(stats(report)["off_ball"]["support_runs"]["count"])
        context["orientation_confirmed"] = True
        self.assertEqual(stats(analyze(raw, context))["off_ball"]["support_runs"]["count"], 1)
        context.update(start_ms=1000, end_ms=2000)
        self.assertIsNone(stats(analyze(raw, context))["off_ball"]["support_runs"]["count"])
        self.assertEqual(raw, unchanged)


if __name__ == "__main__":
    unittest.main()
