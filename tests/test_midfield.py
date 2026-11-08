import unittest

from football_normalizer.midfield import assess, intervals, match_metrics, rate


def event(eid, kind="pass", player="p", team="h", ms=1000, start=35,
          end=50, pattern="Regular Play", attrs=None, possession="h"):
    return {
        "type": kind, "event_id": eid, "match_id": "m", "player_id": player,
        "team_id": team, "period": 1, "period_time_ms": ms,
        "x_m": start, "y_m": 34, "end_x_m": end, "end_y_m": 34,
        "coordinate_frame": "team_attacking_left_to_right",
        "coordinate_length_m": 105, "coordinate_width_m": 68,
        "play_pattern": pattern, "outcome": "incomplete" if eid == "a" else "complete",
        "provider_attributes": attrs or {}, "source": {"record_id": eid},
        "possession_team_id": possession, "possession_id": 1 if possession == "h" else 2,
    }


class MidfieldTest(unittest.TestCase):
    def test_intervals_keep_first_half_stoppage_and_split_at_break(self):
        pos = {"clock_basis": "provider_match_clock", "from_period": 1,
               "to_period": 2, "from": "00:00", "to": "45:00"}
        self.assertEqual(list(intervals(pos, {1: 2850000, 2: 2900000})),
                         [(1, 0, 2850000)])

    def test_progressive_attempts_include_failures_and_exclude_restart_and_missing_end(self):
        events = [event("a"), event("b", end=None), event("c", pattern="From Corner"),
                  event("d", ms=4000), event("opp", team="a", possession="a"),
                  event("teammate", player="q")]
        metrics = match_metrics(events, "p", "h", [(1, 0, 3000)])
        self.assertEqual((metrics["progressive_passing"]["count"],
                          metrics["progressive_passing"]["denominator"],
                          metrics["progressive_passing"]["excluded"]), (1, 1, 1))
        self.assertEqual(metrics["progressive_pass_completion"]["per_100"], 0)
        self.assertEqual(metrics["deep_distribution"]["denominator"], 3)
        self.assertEqual(metrics["deep_distribution"]["count"], 2)

    def test_chance_links_resolve_to_shot_and_defensive_exposure_is_separate(self):
        events = [event("a", attrs={"assisted_shot_id": "shot"}),
                  event("b", attrs={"assisted_shot_id": "missing"}),
                  event("shot", kind="shot", player="q"),
                  event("opp", team="a", possession="a"),
                  event("press1", kind="pressure", possession="a"),
                  event("press2", kind="pressure", possession="a")]
        m = match_metrics(events, "p", "h", [(1, 0, 3000)])
        self.assertEqual((m["chance_creation"]["count"], m["chance_creation"]["excluded"]), (1, 1))
        self.assertEqual((m["pressing"]["count"], m["pressing"]["denominator"]), (2, 1))

    def test_benchmark_requires_sample_and_peers_and_never_confirms_movement(self):
        from football_normalizer.midfield import ROLE_METRICS
        keys = {k for group in ROLE_METRICS.values() for k in group}
        profile = {"matches": [1], "estimated_minutes": 90,
                   "metrics": {k: rate([{"match_id": m, "event_id": str(i)}
                                         for i, m in enumerate(["m1", "m1", "m2"])], 30) for k in keys}}
        assess(profile, [], True)
        self.assertTrue(all(r["status"] == "unassessed" for r in profile["roles"].values()))
        profile.update(matches=list(range(5)), estimated_minutes=450)
        peer = {"matches": list(range(5)), "estimated_minutes": 450, "metrics": {k: rate([], 30) for k in keys}}
        assess(profile, [peer] * 29, True)
        self.assertEqual(profile["roles"]["deep_lying_playmaker"]["status"], "unassessed")
        assess(profile, [peer] * 30, False)
        self.assertEqual(profile["roles"]["deep_lying_playmaker"]["status"], "benchmark_available")
        self.assertEqual(profile["roles"]["ball_winning_midfielder"]["status"], "unassessed")
        self.assertEqual(profile["roles"]["box_to_box_midfielder"]["status"], "unassessed")

    def test_joint_cohort_and_single_match_checks(self):
        from football_normalizer.role_assessment import RULES
        keys = {k for features, _ in RULES.values() for k in features}
        actions = [{"match_id": m, "event_id": str(i)} for i, m in enumerate(["a", "a", "b"])]
        profile = {"matches": list(range(5)), "estimated_minutes": 450,
                   "metrics": {key: rate(actions, 30) for key in keys}}
        peers = [
            {"matches": list(range(5)), "estimated_minutes": 450,
             "metrics": {key: rate([], 0 if key == missing else 30) for key in keys}}
            for missing in ("deep_distribution", "deep_progressive_passing") for _ in range(30)
        ]
        assess(profile, peers, True)
        dlp = profile["roles"]["deep_lying_playmaker"]
        self.assertEqual(dlp["status"], "unassessed")
        self.assertEqual(dlp["peer_count"], 0)
        match = {"metrics": profile["metrics"]}
        assess(match, [], True, match_only=True)
        self.assertEqual(match["roles"]["deep_lying_playmaker"]["status"], "match_observation")
        self.assertEqual(match["roles"]["deep_lying_playmaker"]["signals"][0]["count"], 3)

    def test_match_scores_are_continuous_and_do_not_need_peers(self):
        from football_normalizer.role_assessment import RULES, assess_match
        keys = {k for features, _ in RULES.values() for k in features}
        actions = [{"match_id": "m", "event_id": str(i)} for i in range(3)]
        profile = {"estimated_minutes": 55, "metrics": {k: rate(actions, 20) for k in keys}}
        role = "deep_lying_playmaker"
        assess_match(profile, False)
        first = profile["roles"][role]
        self.assertAlmostEqual(first["fit_score_pct"], 50)
        self.assertIsNone(profile["roles"]["ball_winning_midfielder"]["fit_score_pct"])
        self.assertIsNone(profile["roles"]["holding_midfielder"]["fit_score_pct"])
        profile["estimated_minutes"] = 12
        assess_match(profile, False)
        self.assertAlmostEqual(profile["roles"][role]["fit_score_pct"], 50)
        self.assertLess(profile["roles"][role]["evidence_strength_pct"], first["evidence_strength_pct"])
        profile["metrics"]["deep_progressive_passing"] = rate([], 20)
        assess_match(profile, False)
        self.assertEqual(profile["roles"][role]["fit_score_pct"], 0)
        profile["metrics"]["deep_progressive_passing"] = rate([], 0)
        assess_match(profile, False)
        self.assertIsNone(profile["roles"][role]["fit_score_pct"])
