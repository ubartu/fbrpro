import unittest

from football_normalizer.video_ball_actions import analyze_ball_actions
from football_normalizer.video_events import Sample, analyze
from test_video_events import tracking


def samples(positions, *, teams=None):
    boxes = {"a": (90, 60, 110, 100), "b": (150, 60, 170, 100), "c": (230, 60, 250, 100)}
    teams = teams or {"a": "A", "b": "B", "c": "B"}
    return [Sample(i*100, 1, ("clip", "shot"), "ball", (p, 99) if isinstance(p, int) else p,
                   boxes, teams) if p is not None else None for i, p in enumerate(positions)]


def run(observed, *, releases=(), mapped=(), dead=(), predictions=()):
    tracks = {"a": {"player_id": "p1"}, "b": {"player_id": "p2"}, "c": {"player_id": None}}
    prepared = {"length": 105, "width": 68, "directions": {},
                "context": {"dead_ball_intervals": dead},
                "scope_spans": [{"start_ms": 0, "end_ms": len(observed)*100, "period": 1}],
                "frames": [{"t_ms": t, "players": [], "ball": {"x": x, "y": y, "error_m": .5}} for t, x, y in mapped]}
    return analyze_ball_actions(observed, 160, prepared, tracks, releases, predictions)


class BallActionsTests(unittest.TestCase):
    def test_distinct_foot_contact_redirection_and_header(self):
        redirection = samples([(100, 65), (100, 98), (132, 98), (137, 100)])
        events, _ = run(redirection, releases=[{"period": 1, "actor_sample_ms": 100, "track_id": "a"}])
        self.assertEqual([(e["contact_kind"], e["track_id"]) for e in events], [("one_touch", "a")])
        events, _ = run(redirection)
        self.assertEqual([e["contact_kind"] for e in events], ["deflection_possible"])
        headers = samples([(100, 35), (100, 59), (122, 59)])
        events, _ = run(headers)
        self.assertEqual([(e["contact_kind"], e["player_id"]) for e in events], [("header_possible", "p1")])

    def test_goalward_shot_requires_motion_and_continuous_evidence(self):
        observed = samples([99, 102, 124, 136])
        mapping = [(100, 87, 32), (200, 89, 32.3), (300, 91, 32.6)]
        shot, = [event for event in run(observed, mapped=mapping)[0] if event["type"] == "shot"]
        self.assertEqual((shot["track_id"], shot["t_ms"]), ("a", 100))
        observed[2].shot = observed[3].shot = ("clip", "new-shot")
        self.assertFalse(any(event["type"] == "shot" for event in run(observed, mapped=mapping)[0]))

    def test_competing_players_qc_and_follow_on_control(self):
        observed = samples([99, 102, 132, 160, 161])
        for sample in observed:
            sample.players = {**sample.players, "b": (118, 60, 138, 100)}
        # A closely marked carrier loses the ball directly to the challenger.
        observed[2].ball = (126, 99)
        observed[3].ball = (128, 99)
        observed[4].ball = (128, 99)
        events, _ = run(observed)
        tackle, = [e for e in events if e["type"] == "tackle"]
        self.assertEqual((tackle["track_id"], tackle["next_control_team_id"]), ("b", "B"))
        observed = samples([99, 102, 124, 146, 160, 161])
        releases = [{"period": 1, "actor_sample_ms": 100, "track_id": "a"}]
        events, _ = run(observed, releases=releases)
        interception, = [e for e in events if e["type"] == "interception"]
        self.assertEqual((interception["dispossessed_track_id"], interception["track_id"]), ("a", "b"))
        observed[2] = None
        self.assertFalse(any(e["type"] in {"tackle", "interception"} for e in run(observed, releases=releases)[0]))

    def test_press_records_carrier_team_regain_and_unknown_on_cut(self):
        from football_normalizer.video_pressing import estimate_presses

        observed = samples([100]*10)
        frames = [{"t_ms": s.t_ms, "players": [
            {"id": "a", "team": "A", "x": 40, "y": 30, "error_m": .15},
            {"id": "b", "team": "B", "x": max(44, 51-s.t_ms/100), "y": 30, "error_m": .15},
        ]} for s in observed]
        prepared = {"frames": frames, "context": {"press_outcome_window_ms": 250}}
        tracks = {"a": {"player_id": "carrier"}, "b": {"player_id": "defender"}}
        pairs = [(a,b,"player_control","a","A",a.shot) for a,b in zip(observed[:8],observed[1:8])]
        regain = (observed[7], observed[8], "player_control", "b", "B", observed[7].shot)
        press, = estimate_presses([*pairs, regain], prepared, tracks, [])
        self.assertEqual((press["player_id"], press["carrier_player_id"], press["regain_ms"]), ("defender", "carrier", 700))
        self.assertEqual(press["outcome"], "successful")
        gap = (observed[7], None, "unobservable", None, None, None)
        press, = estimate_presses([*pairs, gap], prepared, tracks, [])
        self.assertEqual(press["outcome"], "unknown")
        transit = [(a, b, "in_transit", None, None, a.shot)
                   for a, b in zip(observed[7:], observed[8:])]
        prepared["context"]["press_outcome_window_ms"] = 150
        press, = estimate_presses([*pairs, *transit], prepared, tracks, [])
        self.assertEqual(press["outcome"], "unsuccessful")

    def test_possession_covers_exact_scope_with_unknown_and_stops(self):
        observed = samples([100, 101, None, 160, 161])
        reviewed = [{"id": "saved", "type": "touch", "track_id": "passer", "t_ms": 100, "reviewed": True}]
        self.assertEqual(analyze(tracking(), {"events": reviewed})["reviewed_events"], reviewed)
        _, possession = run(observed, dead=[{"start_ms": 200, "end_ms": 300}])
        self.assertEqual(possession["states_ms"], {"player_control": 200, "in_transit": 0,
                         "contested": 0, "dead_ball": 100, "unobservable": 200})
        self.assertEqual(possession["scope_ms"], 500)
        self.assertEqual(possession["team_control_ms"], 200)
        self.assertEqual({team["team_id"]: team["control_ms"] for team in possession["teams"]}, {"A": 100, "B": 100})
        self.assertEqual(sum(p["control_ms"] for p in possession["players"]), 200)
        spots = [{"period": 1, "action_score": .9, "type": "out", "t_ms": 180},
                 {"period": 1, "action_score": .9, "type": "throw_in", "t_ms": 350}]
        self.assertEqual(run(observed, predictions=spots)[1]["states_ms"]["dead_ball"], 170)
