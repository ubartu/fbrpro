from copy import deepcopy
import unittest

from football_normalizer.video_midfield import ROLE_SIGNALS, VIDEO_KEYS, enrich
from test_video_statistics import tracking


def fixture():
    match_ids = ("match-one", "match-two")
    metrics = {key for signals in ROLE_SIGNALS.values() for key in signals if key not in VIDEO_KEYS}
    profiles = []
    for index in range(3):
        rate = 60 - 20*index
        match_rows = [{"match_id": match, "estimated_minutes": 1,
                       "intervals": [[1, 1000, 61000]],
                       "metrics": {key: {"per_100": rate, "denominator": 10, "coverage": 1}
                                   for key in metrics}} for match in match_ids]
        profiles.append({"player_id": f"person{index}", "team_id": "a",
                         "scope": ["comp", "season", "central"], "matches": match_rows})
    entries = []
    for match in match_ids:
        video = tracking()
        video.update(match_id=match, sync=[{"segment_id": "sync", "status": "verified"}])
        for frame in video["frames"]:
            frame.update(segment_id="sync", period_time_ms=frame["video_time_ms"]+500)
            for index, player in enumerate(frame["objects"]):
                player.update(y_m=34, x_m=(20 if index==0 else 50 if index==1 else 83))
        context = {"orientation_confirmed": True, "directions": {"1": {"a": 1}},
                   "events": [{"type": "press", "id": "reviewed-press", "track_id": "p0",
                               "t_ms": 1000, "reviewed": True, "track_confirmed": True},
                              {"type": "press", "id": "automatic-press", "track_id": "p0", "t_ms": 1100}]}
        entries.append((video, context))
    return {"version": "midfield-v0.1", "defensive_coverage_confirmed": True,
            "matches": [{"match_id": mid} for mid in match_ids], "profiles": profiles}, entries


class VideoMidfieldTests(unittest.TestCase):
    def test_join_and_rank_across_two_matches(self):
        report, entries = fixture()
        before = deepcopy(report)
        result = enrich(report, entries)
        self.assertEqual(before, report)
        p = result["profiles"][0]
        for row in p["matches"]:
            video = row["video_role_evidence"]
            self.assertEqual(video["usable_ms"], 2400)
            self.assertEqual(video["zone_time_ms"][3], 2400)
            self.assertEqual(video["reviewed_actions"], {"press": 1})
            self.assertEqual(video["roles"]["deep_lying_playmaker"]["comparison_pct"], 100)
            self.assertIn("deep_lying_playmaker", video["relative_labels"])
        summary = p["video_role_summary"]
        self.assertEqual((summary["matches_with_video"], summary["usable_ms"]), (2, 4800))
        self.assertEqual(summary["roles"]["deep_lying_playmaker"]["scored_matches"], 2)

    def test_missing_alignment_or_orientation_and_defensive_coverage(self):
        report, entries = fixture()
        entries[0][0]["sync"][0]["status"] = "unchecked"
        with self.assertRaisesRegex(ValueError, "verified match-clock"):
            enrich(report, entries)
        report["defensive_coverage_confirmed"] = False
        entries[0][0]["sync"][0]["status"] = "verified"
        entries[1][1]["orientation_confirmed"] = False
        result = enrich(report, entries)
        p = result["profiles"][0]
        self.assertEqual(p["video_role_summary"]["matches_with_video"], 1)
        self.assertEqual(p["video_role_summary"]["roles"]["ball_winning_midfielder"]["scored_matches"], 0)
