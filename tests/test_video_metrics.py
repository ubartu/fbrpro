"""Cover measured behaviour, observable negatives, and missing evidence."""
import copy
import unittest

from football_normalizer.video_metrics import analyze, pool
from football_normalizer.video_validate import compare


def player(id_, x, y, team="home", **changes):
    return {"track_id": id_, "object_type": "player", "player_id": id_, "team_id": team,
            "x_m": x, "y_m": y, "observed": True, "estimated": False,
            "visibility": "observed", "projection_status": "verified", **changes}


def ball(x, y):
    return {**player("ball", x, y), "object_type": "ball", "player_id": None,
            "team_id": None, "projection_status": "ball_ground_assumed"}


def fixture(rows, phase="settled_defence", reason="opposition_build_up_sample", **ctx):
    ids = {"match_id": "m", "video_id": "v", "plan_id": "p"}
    o = {"id": "o1", "reason": reason, "period": 1, "match_start_ms": 0,
         "match_end_ms": 2000, "video_start_ms": 1000, "video_end_ms": 3000,
         "event_ids": ["e1"]}
    p = {**ids, "version": "fbrpro-video-plan-v1", "team_id": "home",
         "players": [{"id": "p1", "name": "Player 1"}],
         "alignment_segments": [{"id": "s1", "period": 1, "match_start_ms": 0, "match_end_ms": 2000,
                                 "video_start_ms": 1000, "video_end_ms": 3000}],
         "clips": [{"id": "c1", "segment_id": "s1", "period": 1, "video_start_ms": 1000,
                    "video_end_ms": 3000, "opportunities": [o]}]}
    t = {**ids, "version": "fbrpro-video-tracking-v1", "coordinate_frame": "stadium_fixed",
         "coordinate_length_m": 100, "coordinate_width_m": 60,
         "frames": [{"clip_id": "c1", "segment_id": "s1", "period": 1, "frame_id": i,
                     "period_time_ms": i*500, "video_time_ms": 1000+i*500,
                     "calibration_status": "verified", "objects": objs}
                    for i, objs in enumerate(rows)]}
    c = {**ids, "attacking_direction": {"1": 1},
         "episodes": [{"opportunity_id": "o1", "phase": phase, "player_ids": ["p1"], **ctx}]}
    return p, t, c


def measure(report, metric):
    return next(m for m in report["players"][0]["measures"] if m["metric"] == metric)


class VideoMetricsTests(unittest.TestCase):
    def test_unchecked_clocks_and_independent_episode_reviews(self):
        rows = [[player("p1", x, 30)] for x in [40, 50, 63, 70, 69]]
        p, t, c = fixture(rows, "transition", "regain_candidate", regain_ms=0)
        t["sync"] = [{"segment_id": "s1", "status": "unchecked"}]
        m = measure(analyze(p, t, c), "deep_to_advanced_support")
        self.assertIn("independent alignment", m["observations"][0]["reason"])
        p["alignment_segments"][0]["check_anchors"] = [{"match_ms": 1000, "video_ms": 2010}]
        t["sync"][0]["status"] = "verified"
        report = analyze(p, t, c)
        review = {k: report[k] for k in ("match_id", "video_id", "plan_id")}
        review.update(version="fbrpro-video-observations-v1", observations=[
            {"clip_id": "c1", "opportunity_id": "o1", "player_id": "p1",
             "behaviour": "deep_to_advanced", "observation": "observed"}])
        other = copy.deepcopy(review)
        check = compare(report, [review, other])["measures"][0]
        self.assertEqual((check["comparable"], check["precision"], check["recall"]), (1, 1, 1))
        other["observations"][0]["observation"] = "not_observed"
        check = compare(report, [review, other])["measures"][0]
        self.assertEqual(check["counts"], {"analyst_disagreement": 1})

    def test_screening_positive_and_missing_line_does_not_count_as_negative(self):
        rows = [[player("p1", 35, 30), ball(65, 30), player("d1", 26, 17), player("d2", 28, 43)]
                for _ in range(5)]
        p, t, c = fixture(rows, defender_ids=["d1", "d2"], ground_ball_reviewed=True)
        m = measure(analyze(p, t, c), "central_screening")
        self.assertEqual((m["numerator"], m["denominator"], m["per_100"]), (2000, 2000, 100))
        self.assertEqual(m["observations"][0]["event_ids"], ["e1"])
        # Lose a line defender in the middle; the adjacent intervals become
        # unobservable instead of lowering the observed screening percentage.
        t["frames"][2]["objects"] = t["frames"][2]["objects"][:-1]
        m = measure(analyze(p, t, c), "central_screening")
        self.assertEqual((m["numerator"], m["denominator"], m["excluded"]), (1000, 1000, 1000))
        self.assertEqual(m["observations"][0]["status"], "partial")
        c["episodes"][0].pop("ground_ball_reviewed")
        m = measure(analyze(p, t, c), "central_screening")
        self.assertEqual(m["denominator"], 0)
        self.assertIsNone(m["per_100"])

    def test_screening_excludes_cross_track_time_and_checks_video_clock(self):
        rows = [[player("p1", 35, 30), ball(65, 30), player("d1", 26, 17), player("d2", 28, 43)]
                for _ in range(5)]
        p, t, c = fixture(rows, defender_ids=["d1", "d2"], ground_ball_reviewed=True)
        for f in t["frames"][2:]:
            f["objects"][0]["track_id"] = "p1-second-track"
        m = measure(analyze(p, t, c), "central_screening")
        self.assertEqual((m["denominator"], m["excluded"]), (1500, 500))
        c["episodes"][0].update(start_ms=1000, end_ms=1500)
        m = measure(analyze(p, t, c), "central_screening")
        self.assertEqual((m["observations"][0]["video_start_ms"], m["observations"][0]["video_end_ms"]),
                         (2000, 2500))
        t["frames"][0]["period_time_ms"] = 250
        with self.assertRaisesRegex(ValueError, "clocks disagree"):
            analyze(p, t, c)

    def test_transition_complete_run_observed_negative_and_incomplete(self):
        rows = [[player("p1", x, 30)] for x in [40, 50, 63, 70, 69]]
        p, t, c = fixture(rows, "transition", "regain_candidate", regain_ms=0)
        m = measure(analyze(p, t, c), "deep_to_advanced_support")
        self.assertEqual((m["numerator"], m["denominator"], m["excluded"]), (1, 1, 0))
        self.assertEqual(m["observations"][0]["track_ids"], ["p1"])
        for f in t["frames"]:
            f["objects"][0]["x_m"] = 40
        m = measure(analyze(p, t, c), "deep_to_advanced_support")
        self.assertEqual((m["numerator"], m["denominator"], m["per_100"]), (0, 1, 0))
        t["frames"][2]["objects"][0]["estimated"] = True
        m = measure(analyze(p, t, c), "deep_to_advanced_support")
        self.assertEqual((m["denominator"], m["excluded"], m["per_100"]), (0, 1, None))

    def test_between_line_support_and_reception_are_separate(self):
        rows = [[player("p1", 57, 30), player("m1", 50, 17, "away"),
                 player("m2", 51, 43, "away"), player("d1", 65, 17, "away"),
                 player("d2", 66, 43, "away")] for _ in range(5)]
        p, t, c = fixture(rows, "attack", opponent_team_id="away",
                          opponent_midfielder_ids=["m1", "m2"], opponent_defender_ids=["d1", "d2"])
        r = analyze(p, t, c)
        self.assertEqual(measure(r, "between_line_support")["numerator"], 1)
        self.assertIsNone(measure(r, "between_line_reception")["per_100"])
        c["episodes"][0].update(receptions_complete=True, receptions=[{"player_id": "p1", "period_time_ms": 1000}])
        self.assertEqual(measure(analyze(p, t, c), "between_line_reception")["numerator"], 1)

    def test_outlet_lane_and_closing_down(self):
        rows = [[player("p1", 40, 30), ball(30, 30), player("a1", 35, 40, "away")]
                for _ in range(5)]
        p, t, c = fixture(rows, "build_up", opponent_team_id="away",
                          lane_context_complete=True, opponent_ids=["a1"], ground_ball_reviewed=True)
        self.assertEqual(measure(analyze(p, t, c), "deep_passing_outlet")["numerator"], 1)
        for f in t["frames"]:
            f["objects"][-1]["y_m"] = 30
        self.assertEqual(measure(analyze(p, t, c), "deep_passing_outlet")["numerator"], 0)

        rows = [[player("p1", x, 30), player("a1", 58, 30, "away")] for x in [45, 48, 52, 54, 55]]
        p, t, c = fixture(rows, opponent_team_id="away", ball_carrier_id="a1")
        self.assertEqual(measure(analyze(p, t, c), "closing_down")["numerator"], 1)

    def test_pool_sums_counts_and_refuses_duplicate_matches(self):
        rows = [[player("p1", x, 30)] for x in [40, 50, 63, 70, 69]]
        p, t, c = fixture(rows, "transition", "regain_candidate", regain_ms=0)
        first = analyze(p, t, c)
        second = copy.deepcopy(first)
        second["match_id"] = "m2"
        m = measure(second, "deep_to_advanced_support")
        m["observations"][0].update(match_id="m2", numerator=0, status="not_observed")
        combined = measure(pool([first, second]), "deep_to_advanced_support")
        self.assertEqual((combined["numerator"], combined["denominator"], combined["per_100"]), (1, 2, 50))
        with self.assertRaises(ValueError):
            pool([first, first])

    def test_position_comparison_and_pooled_scope(self):
        rows = [[player("p1", x, 30), player("p2", 40, 30), player("p3", x, 30)]
                for x in [40, 50, 63, 70, 69]]
        groups = {"p1": "central", "p2": "central", "p3": "defensive", "p4": "central"}
        p, t, c = fixture(rows, "transition", "regain_candidate", regain_ms=0, position_groups=groups)
        p["players"].extend({"id": pid, "name": pid} for pid in ("p2", "p3", "p4"))
        c.update(competition_id="c", season_id="s")
        c["episodes"][0]["player_ids"] = list(groups)
        first = analyze(p, t, c)
        central, defensive = first["position_groups"]
        self.assertEqual(central["position_group"], "central")
        self.assertEqual(defensive["position_group"], "defensive")
        first_measure = measure(central, "deep_to_advanced_support")
        self.assertEqual(first_measure["comparison"], {"higher": 0, "tied": 0, "lower": 1, "missing": 1})
        self.assertEqual(first_measure["observations"][0]["position_group"], "central")

        for obj in (p, t, c):
            obj.update(match_id="m2", plan_id="p2", video_id="v2")
        groups["p1"] = "attacking"
        second = analyze(p, t, c)
        merged = pool([first, second])
        summaries = {g["position_group"]: g for g in merged["position_groups"]}
        self.assertEqual(measure(merged, "deep_to_advanced_support")["denominator"], 2)
        self.assertEqual(measure(summaries["central"], "deep_to_advanced_support")["denominator"], 1)
        self.assertEqual(measure(summaries["attacking"], "deep_to_advanced_support")["denominator"], 1)
        second["season_id"] = "another"
        with self.assertRaisesRegex(ValueError, "same competition and season"):
            pool([first, second])

    def test_wrong_video_and_unknown_direction(self):
        p, t, c = fixture([[player("p1", 40, 30)] for _ in range(5)], "transition", regain_ms=0)
        c["attacking_direction"] = {}
        m = measure(analyze(p, t, c), "deep_to_advanced_support")
        self.assertIn("direction", m["observations"][0]["reason"])
        t["video_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "video_id"):
            analyze(p, t, c)


if __name__ == "__main__":
    unittest.main()
