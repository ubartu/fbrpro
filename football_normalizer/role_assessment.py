"""Match role scores and optional peer benchmarks for midfield metrics."""
from math import isfinite, prod

MIN_MATCHES, MIN_MINUTES = 5, 450
MIN_OPPORTUNITIES, MIN_COVERAGE, MIN_PEERS = 30, .90, 30

RULES = {
    "deep_lying_playmaker": (
        ("deep_distribution", "deep_progressive_passing"),
        "Compare deep central involvement and how often the player advances play from there.",
    ),
    "advanced_playmaker": (
        ("advanced_chance_creation", "advanced_distribution"),
        "Compare advanced central involvement and passes from that area linked to shots.",
    ),
    "ball_winning_midfielder": (
        ("pressing", "midfield_interventions"),
        "Compare pressure and tackle/interception activity relative to opponent possessions.",
    ),
    "holding_midfielder": (
        ("deep_interventions", "deep_distribution"),
        "Use video or tracking to verify screening and spacing in front of the back line.",
    ),
    "box_to_box_midfielder": (
        ("deep_interventions", "progressive_carrying", "box_involvement"),
        "Use video or tracking to verify repeated movement across phases and box arrivals.",
    ),
}
MOVEMENT = {"holding_midfielder", "box_to_box_midfielder"}
DEFENSIVE = {"pressing", "midfield_interventions", "deep_interventions"}


def sufficient(profile):
    return (len(profile["matches"]) >= MIN_MATCHES
            and profile["estimated_minutes"] >= MIN_MINUTES)


def eligible(metric):
    value = metric.get("per_100")
    return (metric["denominator"] >= MIN_OPPORTUNITIES
            and (metric.get("coverage") or 0) >= MIN_COVERAGE
            and isinstance(value, (int, float)) and isfinite(value))


def check(label, actual, minimum):
    return {"label": label, "actual": actual, "minimum": minimum,
            "met": actual is not None and actual >= minimum}


def assess(profile, peers, defensive_coverage, *, baseline_defensive_coverage=True, match_only=False):
    """Attach diagnostics; compare each role against one joint peer cohort."""
    profile["roles"] = {}
    for role, (features, explanation) in RULES.items():
        movement = role in MOVEMENT
        defensive = role == "ball_winning_midfielder"
        blockers = []
        if movement:
            blockers.append("Requires tracking or video review")
        if defensive and not defensive_coverage:
            blockers.append("Provider defensive-event coverage not confirmed")
        if not match_only and not movement and not sufficient(profile):
            blockers.append("Pilot requires at least five matches and 450 estimated position minutes")

        comparable = [] if (movement or match_only or
                            (defensive and not (defensive_coverage and baseline_defensive_coverage))) else [
            p for p in peers if sufficient(p)
            and all(k in p["metrics"] and eligible(p["metrics"][k]) for k in features)
        ]
        if not match_only and not movement and len(comparable) < MIN_PEERS:
            blockers.append(f"{len(comparable)} eligible peers; {MIN_PEERS} needed for this role")

        signals = []
        percentiles = {}
        for key in features:
            metric = profile["metrics"][key]
            observed_matches = len({e["match_id"] for e in metric.get("evidence", [])})
            gates = [
                check("Opportunities", metric["denominator"], MIN_OPPORTUNITIES),
                check("Field coverage (%)", None if metric.get("coverage") is None else 100 * metric["coverage"], 100 * MIN_COVERAGE),
            ]
            if not movement and not match_only and not eligible(metric):
                blockers.append(f"Insufficient data for {key.replace('_', ' ')}")
            if not movement and not match_only:
                if len(comparable) >= MIN_PEERS and eligible(metric):
                    target = metric["per_100"]
                    values = [p["metrics"][key]["per_100"] for p in comparable]
                    rank = 100 * sum((v < target) + .5 * (v == target) for v in values) / len(values)
                    percentiles[key] = {"midrank": rank, "peer_count": len(values)}
            signals.append({
                "metric": key, "count": metric["count"], "denominator": metric["denominator"],
                "per_100": metric["per_100"], "coverage": metric.get("coverage"),
                "evidence_matches": observed_matches, "checks": gates,
                "feed_confirmed": key not in DEFENSIVE or defensive_coverage,
            })

        status = "match_observation" if match_only else "unassessed"
        if not blockers and not match_only:
            status = "benchmark_available"
        profile["roles"][role] = {
            "status": status, "features": list(features), "percentiles": percentiles,
            "explanation": explanation, "signals": signals, "blockers": blockers,
            "reason": blockers[0] if blockers else None,
            "peer_count": len(comparable),
            "sample_checks": [] if match_only or movement else [
                check("Selected matches", len(profile["matches"]), MIN_MATCHES),
                check("Estimated position minutes", profile["estimated_minutes"], MIN_MINUTES),
            ],
        }


# Soft anchors: a rate equal to its anchor scores 50, with no pass/fail cutoff.
MATCH_SCORE_VERSION = "midfield-match-score-v1"
MATCH_ANCHORS = {
    "deep_lying_playmaker": {"deep_distribution": (15, 12), "deep_progressive_passing": (15, 6)},
    "advanced_playmaker": {"advanced_distribution": (15, 10), "advanced_chance_creation": (8, 6)},
    "ball_winning_midfielder": {"pressing": (20, 12), "midfield_interventions": (8, 12)},
}


def geometric(values):
    return prod(values) ** (1 / len(values))


def assess_match(profile, defensive_coverage):
    """Score independent roles from one match; sample strength never changes fit."""
    minutes = profile["estimated_minutes"]
    roles = {}
    for role, (features, explanation) in RULES.items():
        anchors = MATCH_ANCHORS.get(role)
        blockers = []
        if anchors is None:
            blockers.append("Requires tracking or video review")
        if role == "ball_winning_midfielder" and not defensive_coverage:
            blockers.append("Provider defensive-event coverage not confirmed")
        signals = []
        for key in features:
            m = profile["metrics"][key]
            signal = {
                "metric": key, "count": m["count"], "denominator": m["denominator"],
                "per_100": m["per_100"], "coverage": m.get("coverage"),
                "feed_confirmed": key not in DEFENSIVE or defensive_coverage,
                "fit_score_pct": None, "evidence_strength_pct": None,
            }
            if anchors is not None:
                anchor, opportunity_scale = anchors[key]
                rate = m["per_100"]
                coverage = m.get("coverage")
                n = m["denominator"]
                signal["anchor_per_100"] = anchor
                if (n > 0 and rate is not None and isfinite(rate) and rate >= 0
                        and coverage is not None and 0 < coverage <= 1 and signal["feed_confirmed"]):
                    signal["fit_score_pct"] = 100 * rate / (rate + anchor)
                    signal["evidence_strength_pct"] = 100 * coverage * n / (n + opportunity_scale)
                elif signal["feed_confirmed"]:
                    blockers.append(f"Missing usable evidence for {key.replace('_', ' ')}")
            signals.append(signal)
        fit = strength = None
        if not blockers:
            fit = geometric([s["fit_score_pct"] for s in signals])
            strength = geometric([s["evidence_strength_pct"] for s in signals]) * minutes / (minutes + 30)
        roles[role] = {
            "status": "scored" if fit is not None else "unassessed",
            "features": list(features), "explanation": explanation,
            "signals": signals, "blockers": blockers, "reason": blockers[0] if blockers else None,
            "fit_score_pct": fit, "evidence_strength_pct": strength,
            "score_version": MATCH_SCORE_VERSION,
        }
    profile["roles"] = roles
