import unittest
from football_normalizer.relative_roles import annotate
from football_normalizer.role_assessment import RULES
from football_normalizer.statsbomb import receipt_clock_anomalies


def sample(rate, match=None):
    m = {"estimated_minutes": 10, "metrics": {
        k: {"per_100": rate, "denominator": 5, "count": 1}
        for keys, _ in RULES.values() for k in keys}}
    if match:
        m["match_id"] = match
    return m


class RelativeTest(unittest.TestCase):
    def test_match_and_window_can_reverse_with_no_minimum_minutes(self):
        a = dict(sample(30), player_id="a", scope=["league","season","central"],
                 matches=[sample(10, "m")])
        b = dict(sample(20), player_id="b", scope=a["scope"], matches=[sample(25, "m")])
        annotate([a,b])
        role = "deep_lying_playmaker"
        self.assertEqual(a["role_comparison"][role]["comparison_pct"], 100)
        self.assertEqual(a["matches"][0]["role_comparison"][role]["comparison_pct"], 0)
        self.assertTrue(a["role_comparison"][role]["measures"][0]["leads"])
        self.assertIsNone(a["role_comparison"]["holding_midfielder"]["comparison_pct"])

    def test_tied_rates_do_not_assign_a_label(self):
        a = dict(sample(10), player_id="a", scope=["l","s","central"], matches=[])
        b = dict(sample(10), player_id="b", scope=a["scope"], matches=[])
        annotate([a,b])
        self.assertEqual(a["role_comparison"]["deep_lying_playmaker"]["comparison_pct"], 50)
        self.assertEqual(a["relative_labels"], [])

    def test_half_end_receipt_excludes_only_linked_anomaly(self):
        def event(id_,kind,stamp,links=()):
            return {"id":id_,"type":{"name":kind},"timestamp":stamp,"period":1,"related_events":links}
        pass_ = event("p","Pass","00:46:57.531")
        end = event("e","Half End","00:47:03.040")
        self.assertEqual(list(receipt_clock_anomalies(
            [pass_,event("r","Ball Receipt*","00:00:00.440",["p"]),end])), ["r"])
        self.assertEqual(list(receipt_clock_anomalies(
            [pass_,event("r","Ball Receipt*","00:00:00.440",["other"]),end])), [])
