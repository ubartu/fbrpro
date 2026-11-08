"""Compare observed midfield behaviours, without role-assignment thresholds."""
from collections import defaultdict
from math import isfinite

from .role_assessment import RULES, MOVEMENT

VERSION = "midfield-relative-v1"
TITLES = {
    "deep_lying_playmaker": "Deep build-up distributor",
    "advanced_playmaker": "Advanced chance provider",
    "ball_winning_midfielder": "Midfield disruptor",
    "holding_midfielder": "Holding midfielder",
    "box_to_box_midfielder": "Box-to-box midfielder",
}
DESCRIPTION = {
    "deep_lying_playmaker": "Involvement in deep central build-up and attempts to advance the ball from there. Evidence for a deep-lying playmaker.",
    "advanced_playmaker": "Involvement in advanced central possession and passes from there directly linked to shots. Evidence for an advanced playmaker.",
    "ball_winning_midfielder": "Recorded pressure and tackles or interceptions against opponent possession. This measures disruption, not successful regains.",
    "holding_midfielder": "Screens the space in front of the back line. Event counts alone cannot show screening.",
    "box_to_box_midfielder": "Repeatedly moves between defensive and attacking areas. Tracking or reviewed video is needed.",
}


def usable(metric):
    rate = metric.get("per_100")
    return metric["denominator"] > 0 and isinstance(rate, (int, float)) and isfinite(rate)


def compare(rows, defensive_coverage):
    """Each row is (player ID, output record) within one comparable scope."""
    for role, (keys, _) in RULES.items():
        unavailable = ("Needs tracking or reviewed video" if role in MOVEMENT else
                       "Defensive event coverage unconfirmed" if role == "ball_winning_midfielder"
                       and not defensive_coverage else None)
        # Keep one reference per player after transfers. Prefer the longest
        # usable sample; ties are resolved by deterministic row order.
        available = {}
        for pid, row in sorted(rows, key=lambda pair: pair[1]["estimated_minutes"]):
            if all(usable(row["metrics"][k]) for k in keys):
                available[pid] = row
        for pid, row in rows:
            peers = [other for id_, other in available.items() if id_ != pid]
            result = {
                "label": TITLES[role], "definition": DESCRIPTION[role], "comparison_pct": None,
                "peer_count": len(peers), "measures": [],
                "reason": unavailable,
            }
            row.setdefault("role_comparison", {})[role] = result
            if unavailable:
                continue
            if not all(usable(row["metrics"][k]) for k in keys):
                result["reason"] = "Missing opportunities for at least one behaviour"
                continue
            if not peers:
                result["reason"] = "No other player with both measures in this position group"
                continue
            for k in keys:
                m = row["metrics"][k]
                rate = m["per_100"]
                values = [p["metrics"][k]["per_100"] for p in peers]
                higher = sum(rate > v for v in values)
                tied = sum(rate == v for v in values)
                result["measures"].append({
                    "metric": k, "per_100": rate, "count": m["count"], "opportunities": m["denominator"],
                    "higher_than": higher, "tied_with": tied,
                    "lower_than": len(peers) - higher - tied,
                    "comparison_pct": 100 * (higher + .5 * tied) / len(peers),
                    "leads": higher > 0 and higher + tied == len(peers),
                })
            result["comparison_pct"] = sum(x["comparison_pct"] for x in result["measures"]) / len(keys)
    for _, row in rows:
        results = row["role_comparison"]
        scored = {k: v["comparison_pct"] for k,v in results.items()
                  if v["comparison_pct"] is not None and any(
                      x["higher_than"] for x in v["measures"])}
        best = max(scored.values(), default=None)
        row["relative_labels"] = [k for k,v in scored.items() if v == best]


def annotate(profiles, defensive_coverage=False):
    """Compare match peers separately from freshly pooled window rates."""
    matches, windows = defaultdict(list), defaultdict(list)
    for p in profiles:
        scope = tuple(p["scope"])
        if None not in scope:
            windows[scope].append((p["player_id"], p))
        for m in p["matches"]:
            matches[(m["match_id"], scope[2])].append((p["player_id"], m))
    for group in (*matches.values(), *windows.values()):
        compare(group, defensive_coverage)
