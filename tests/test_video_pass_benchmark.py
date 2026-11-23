import unittest

from football_normalizer.video_pass_benchmark import _metrics, compare, footpass_predictions, identity_continuity


class PassBenchmarkTests(unittest.TestCase):
    def test_counts_timing_attribution_and_outcome_coverage_separately(self):
        truth = {"start_ms": 0, "end_ms": 5000, "events": [
            {"type": "pass", "t_ms": 1000, "passer_player_id": "1:10", "receiver_player_id": "1:6", "outcome": "complete"},
            {"type": "pass", "t_ms": 2000},
        ]}
        events = [{"type": "pass", "t_ms": 1200, "passer_player_id": "1:10", "outcome": "unknown"},
                  {"type": "pass", "t_ms": 3000, "outcome": "complete"},
                  {"type": "reception", "t_ms": 2100},
                  {"type": "pass", "t_ms": 5000}]
        counts, moments = compare(events, truth, 480)
        result = _metrics(counts)
        self.assertEqual((result["matched"], result["missed"], result["extra"]), (1, 1, 1))
        self.assertEqual(result["outcome_coverage"], .5)
        self.assertEqual(result["passer_player_id"]["accuracy"], 1)
        self.assertEqual(result["receiver_player_id"]["coverage"], 0)
        self.assertIsNone(result["outcome"]["accuracy"])
        self.assertEqual(moments[0]["offset_ms"], 200)

    def test_footpass_conversion_and_roster_continuity_across_a_cut(self):
        predictions = {"events": {"game_H1": [[1503, 1, 9, 2, .9], [1505, 0, 8, 3, .9], [1507, 0, 6, 2, .1]]}}
        self.assertEqual([(row["t_ms"], row["passer_player_id"]) for row in
                          footpass_predictions(predictions, "game_H1", 60000)], [(120, "1:9")])
        box = [0, 0, 20, 40]
        frames = [{"video_time_ms": stamp, "objects": [{"track_id": track, "player_id": "a", "observed": True,
                   "bbox_xyxy": box, "object_type": "player"}]} for stamp, track in ((100, "t1"), (500, "t2"))]
        cases = [{"kind": "camera_cut", "same_player": True, "before": {"t_ms": 100, "bbox_xyxy": box},
                  "after": {"t_ms": 500, "bbox_xyxy": box}}]
        result = identity_continuity({"frames": frames}, cases)["camera_cut"]
        self.assertEqual((result["track_id_correct"], result["player_id_correct"]), (0, 1))
