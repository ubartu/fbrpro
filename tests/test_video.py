import unittest

from football_normalizer.video import opportunities, plan, segments_from


def event(t, p, side, period=1, pattern="Regular Play", kind="pass"):
    return {"event_id": f"{period}-{t}", "period": period, "period_time_ms": t,
            "type": kind, "possession_id": p, "possession_team_id": side, "play_pattern": pattern}


def alignment(*ranges):
    return {"segments": [
        {"id": str(i), "period": period, "match_start_ms": start, "match_end_ms": end,
         "video_start_ms": vstart, "video_end_ms": vend}
        for i, (period,start,end,vstart,vend) in enumerate(ranges)]}


class VideoTests(unittest.TestCase):
    def test_regain_and_build_up_merge_without_losing_their_denominators(self):
        ev = [event(1000, 1, "away"), event(11000, 1, "away"),
              event(12000, 2, "home"), event(20000, 2, "home")]
        r = plan(ev, alignment((1,0,30000,10000,40000)), "home")
        self.assertEqual(len(r["clips"]), 1)
        clip = r["clips"][0]
        self.assertEqual((clip["video_start_ms"], clip["video_end_ms"]), (14000, 30000))
        self.assertEqual([o["reason"] for o in clip["opportunities"]],
                         ["opposition_build_up_sample", "regain_candidate"])
        self.assertEqual(r["selected_video_ms"], 16000)

    def test_cut_or_unsynced_span_is_excluded_not_interpolated(self):
        ev = [event(0, 1, "away"), event(10000, 1, "away")]
        r = plan(ev, alignment((1,0,8000,0,8000),(1,8000,20000,10000,22000)), "home")
        self.assertEqual(r["clips"], [])
        self.assertEqual(r["counts"]["unaligned"], {"opposition_build_up_sample": 1})

    def test_period_clocks_and_sampling(self):
        ev = [event(0,1,"away",1), event(15000,1,"away",1),
              event(0,2,"home",2), event(10000,2,"home",2),
              event(12000,3,"away",2), event(28000,3,"away",2)]
        r = plan(ev, alignment((1,0,40000,0,40000),(2,0,40000,60000,100000)),
                 "home",build_up_limit=1)
        self.assertEqual(r["counts"]["candidate"], {"opposition_build_up_sample": 2})
        self.assertEqual(r["clips"][0]["period"], 2)
        self.assertEqual(r["clips"][0]["video_start_ms"], 75000)
        self.assertEqual(len(r["not_selected"]), 1)
        self.assertEqual(plan(ev,alignment((2,0,40000,0,40000)),"home",build_up_limit=0)["clips"],[])

    def test_restart_and_unknown_possession_do_not_count_as_regains(self):
        e = [event(1000,1,"away"), event(2000,2,None), event(3000,3,"home"),
             event(4000,4,"away"), event(5000,5,"home",pattern="From Throw In")]
        self.assertEqual(opportunities(e,"home"),[])

    def test_invalid_alignment_and_drift_mapping(self):
        with self.assertRaises(ValueError):
            segments_from(alignment((1,0,1000,0,1000),(2,0,1000,500,1500)))
        with self.assertRaises(ValueError):
            segments_from(alignment((1,0,1000,1000,2000),(2,0,1000,0,1000)))
        self.assertEqual(segments_from(alignment((1,0,10000,2000,12100)))[0].map(5000),7050)
