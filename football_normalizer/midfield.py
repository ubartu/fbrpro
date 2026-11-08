"""Selected-match midfield evidence from accepted StatsBomb bundles."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .relative_roles import VERSION as COMPARISON_VERSION, annotate
from .role_assessment import MATCH_SCORE_VERSION, RULES, assess, assess_match, sufficient

VERSION = "midfield-v0.1"
GROUPS = {
    9: "defensive", 10: "defensive", 11: "defensive",
    13: "central", 14: "central", 15: "central",
    18: "attacking", 19: "attacking", 20: "attacking",
}
OPEN = {"Regular Play", "From Counter"}
ROLE_METRICS = {role: features for role, (features, _) in RULES.items()}


def rows(path, table):
    with (path / f"{table}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clock(value, period):
    minute, second = map(int, value.split(":"))
    result = (minute * 60 + second - (period - 1) * 2700) * 1000
    if not 0 <= second < 60 or result < 0:
        raise ValueError(f"Unsupported StatsBomb clock: {period=} {value=}")
    return result


def intervals(position, ends):
    """Split regulation intervals at the actual end of each half."""
    if position.get("clock_basis") != "provider_match_clock":
        raise ValueError("Unsupported position clock")
    first = position["from_period"]
    last = position["to_period"] or max(ends)
    if first not in ends or last not in ends or first > last:
        raise ValueError("Invalid or unsupported position periods")
    if position["to"] is None and position.get("end_reason") != "Final Whistle":
        raise ValueError("Position interval has no supported endpoint")
    for period in range(first, last + 1):
        start = clock(position["from"], first) if period == first else 0
        end = clock(position["to"], last) if period == last and position["to"] else ends[period]
        # Lineup clocks are whole seconds, unlike half-end event clocks.
        if end > ends[period] + 1000 or start > end:
            raise ValueError("Position interval falls outside the period")
        if start < min(end, ends[period]):
            yield period, start, min(end, ends[period])


def location(event, end=False):
    prefix = "end_" if end else ""
    x, y = event.get(prefix + "x_m"), event.get(prefix + "y_m")
    length, width = event.get("coordinate_length_m"), event.get("coordinate_width_m")
    if (event.get("coordinate_frame") == "team_attacking_left_to_right"
            and x is not None and y is not None and length and width
            and 0 <= x <= length and 0 <= y <= width):
        return x / length, y / width
    return None


def deep(point):
    return point[0] < .5 and .25 <= point[1] <= .75


def advanced(point):
    return point[0] >= 2/3 and .25 <= point[1] <= .75


def box(point):
    # StatsBomb standardized pitch markings (120 x 80 source grid).
    return point[0] >= 102/120 and 18/80 <= point[1] <= 62/80


def rate(samples, denominator, excluded=0):
    return {
        "count": len(samples), "denominator": denominator,
        "excluded": excluded,
        "coverage": denominator / (denominator + excluded) if denominator + excluded else None,
        "per_100": 100 * len(samples) / denominator if denominator else None,
        "evidence": [{"match_id": e["match_id"], "event_id": e["event_id"]} for e in samples],
    }


def match_metrics(events, player, team, windows):
    def active(e):
        return any(e["period"] == period and start <= e["period_time_ms"] < end
                   for period, start, end in windows)

    match_events = [e for e in events if active(e)]
    own = [e for e in match_events if e["player_id"] == player and e["team_id"] == team]
    open_own = [e for e in own if e.get("play_pattern") in OPEN]
    passes = [e for e in open_own if e["type"] == "pass"]
    carries = [e for e in open_own if e["type"] == "carry"]
    team_passes = [e for e in match_events if e["team_id"] == team
                   and e["type"] == "pass" and e.get("play_pattern") in OPEN]

    def progression(attempts, zone=None):
        starts = [e for e in attempts if location(e) and (zone is None or zone(location(e)))]
        eligible = [e for e in starts if location(e, True)]
        excluded = len(starts) - len(eligible) + sum(location(e) is None for e in attempts)
        return [e for e in eligible if location(e, True)[0] - location(e)[0] >= .10 - 1e-9], len(eligible), excluded

    def distribution(zone):
        eligible = [e for e in team_passes if location(e) and zone(location(e))]
        return rate([e for e in eligible if e["player_id"] == player], len(eligible),
                    sum(location(e) is None for e in team_passes))

    shots = {str(e["source"]["record_id"]): e for e in events if e["type"] == "shot"}

    def chance_creation(attempts, excluded=0):
        linked = []
        broken = 0
        for e in attempts:
            attrs = e["provider_attributes"]
            shot = shots.get(str(attrs.get("assisted_shot_id")))
            if shot and shot["team_id"] == team and shot.get("play_pattern") in OPEN:
                linked.append(e)
            elif attrs.get("assisted_shot_id") is not None or attrs.get("shot_assist") or attrs.get("goal_assist"):
                broken += 1
        return rate(linked, len(attempts) - broken, excluded + broken)

    def possessions(for_team):
        return len({(e["period"], e["possession_id"]) for e in match_events
                    if e.get("possession_id") is not None and e.get("possession_team_id") == for_team})

    opponents = {e["team_id"] for e in events if e.get("team_id") and e["team_id"] != team}
    if len(opponents) != 1:
        raise ValueError("Expected exactly two teams")
    opp = next(iter(opponents))
    opponent_possessions = possessions(opp)
    defending = [e for e in own if e.get("possession_team_id") == opp]
    interventions = [e for e in defending if e["type"] == "interception" or
                     (e["type"] == "duel" and (e["provider_attributes"].get("type") or {}).get("name") == "Tackle")]
    prog, n, skipped = progression(passes)
    known = [e for e in prog if e.get("outcome") in {"complete", "incomplete", "out", "pass_offside"}]
    features = {
        "progressive_passing": rate(prog, n, skipped),
        "progressive_pass_completion": rate([e for e in known if e["outcome"] == "complete"], len(known), len(prog)-len(known)),
        "deep_progressive_passing": rate(*progression(passes, deep)),
        "progressive_carrying": rate(*progression(carries)),
        "deep_distribution": distribution(deep),
        "advanced_distribution": distribution(advanced),
        "chance_creation": chance_creation(passes),
        "advanced_chance_creation": chance_creation([e for e in passes if location(e) and advanced(location(e))],
                                                   sum(location(e) is None for e in passes)),
        "pressing": rate([e for e in defending if e["type"] == "pressure"], opponent_possessions),
        "midfield_interventions": rate([e for e in interventions if location(e) and 1/3 <= location(e)[0] < 2/3], opponent_possessions),
        "deep_interventions": rate([e for e in interventions if location(e) and deep(location(e))], opponent_possessions),
        "box_involvement": rate([e for e in open_own if e["type"] in {"shot", "carry"} and
                                 (point := location(e, end=e["type"] == "carry")) and box(point)], possessions(team)),
    }
    return features


def build(bundles, match_ids, baseline=None, defensive_coverage=False):
    profiles = {}
    selected = set()
    match_records = []
    for path in bundles:
        if json.loads((path / "status.json").read_text()).get("status") != "accepted":
            raise ValueError(f"Bundle is not accepted: {path}")
        matches = [m for m in rows(path, "matches") if not match_ids or m["match_id"] in match_ids
                   or str(m["source"]["record_id"]) in match_ids]
        if not matches:
            continue
        players = {p["player_id"]: p["name"] for p in rows(path, "players")}
        event_rows = rows(path, "events")
        appearances = rows(path, "appearances")
        for match in matches:
            mid = match["match_id"]
            if mid in selected:
                raise ValueError(f"Duplicate selected match: {mid}")
            if match["source"]["provider"] != "statsbomb":
                raise ValueError("Midfield v0.1 supports StatsBomb bundles only")
            selected.add(mid)
            match_records.append({"match_id": mid, "source_match_id": match["source"]["record_id"],
                                  "date": match.get("date")})
            events = [e for e in event_rows if e["match_id"] == mid]
            ends = {e["period"]: e["period_time_ms"] for e in events if e["type"] == "half_end"}
            if set(ends) != {1, 2}:
                raise ValueError("Pilot requires both regulation half-end events and no extra time")
            for a in appearances:
                if a["match_id"] != mid:
                    continue
                by_group = defaultdict(list)
                for position in a["position_intervals"]:
                    group = GROUPS.get(position["position_id"])
                    if group:
                        by_group[group].extend(intervals(position, ends))
                for group, windows in by_group.items():
                    windows.sort()
                    if any(p == q and end > start for (p, _, end), (q, start, _) in zip(windows, windows[1:])):
                        raise ValueError("Overlapping intervals in position group")
                    if not windows:
                        continue
                    pid, team = a["player_id"], a["team_id"]
                    scope = (match["competition_id"], match["season_id"], group)
                    profile = profiles.setdefault((pid, team, scope), {
                        "player_id": pid, "player_name": players[pid], "team_id": team, "scope": list(scope),
                        "matches": [], "estimated_minutes": 0, "metrics": {},
                    })
                    metrics = match_metrics(events, pid, team, windows)
                    minutes = sum(end-start for _, start, end in windows) / 60000
                    profile["estimated_minutes"] += minutes
                    profile["matches"].append({"match_id": mid, "estimated_minutes": minutes,
                                               "intervals": windows, "metrics": metrics})
    missing = set(match_ids) - {str(m[k]) for m in match_records for k in ("match_id", "source_match_id")}
    if missing or not selected:
        raise ValueError(f"No match found for selector(s): {', '.join(sorted(missing)) or '(none)'}")
    refs = []
    if baseline is not None:
        if baseline["version"] != VERSION:
            raise ValueError("Baseline metric version differs")
        if selected & {m["match_id"] for m in baseline["matches"]}:
            raise ValueError("Baseline and selected matches must be disjoint")
        refs = baseline["profiles"]
    for profile in profiles.values():
        for key in profile["matches"][0]["metrics"]:
            metrics = [m["metrics"][key] for m in profile["matches"]]
            num = sum(m["count"] for m in metrics)
            den = sum(m["denominator"] for m in metrics)
            exc = sum(m["excluded"] for m in metrics)
            profile["metrics"][key] = {
                "count": num, "denominator": den, "excluded": exc,
                "coverage": den / (den + exc) if den + exc else None,
                "per_100": 100 * num / den if den else None,
                "evidence": [e for m in metrics for e in m["evidence"]],
            }
        # One eligible observation per peer; if transferred, use the largest eligible sample.
        unique = {}
        for p in sorted(refs, key=lambda p: p["estimated_minutes"]):
            if p["scope"] == profile["scope"] and None not in p["scope"] and sufficient(p) and p["player_id"] != profile["player_id"]:
                unique[p["player_id"]] = p
        peers = list(unique.values())
        assess(profile, peers, defensive_coverage,
               baseline_defensive_coverage=bool(baseline and baseline.get("defensive_coverage_confirmed")))
        for match_profile in profile["matches"]:
            assess_match(match_profile, defensive_coverage)
        profile["match_score_summary"] = {}
        for role in RULES:
            scored = [m for m in profile["matches"] if m["roles"][role]["status"] == "scored"]
            minutes = sum(m["estimated_minutes"] for m in scored)
            def mean(key):
                return sum(m["estimated_minutes"] * m["roles"][role][key] for m in scored) / minutes if minutes else None
            profile["match_score_summary"][role] = {
                "fit_score_pct": mean("fit_score_pct"),
                "evidence_strength_pct": mean("evidence_strength_pct"),
                "scored_matches": len(scored),
                "total_matches": len(profile["matches"]),
            }
    annotate(profiles.values(), defensive_coverage)
    return {"version": VERSION, "comparison_version": COMPARISON_VERSION, "match_score_version": MATCH_SCORE_VERSION, "defensive_coverage_confirmed": defensive_coverage, "matches": match_records,
            "profiles": sorted(profiles.values(), key=lambda p: (p["player_name"], p["scope"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--match", action="append", default=[], help="Canonical or provider match ID")
    parser.add_argument("--baseline", type=Path, help="Disjoint midfield report for peer comparisons")
    parser.add_argument("--confirm-defensive-coverage", action="store_true", help="Assert the feed consistently logs pressures, tackles and interceptions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        result = build(args.bundles, args.match, baseline, args.confirm_defensive_coverage)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"midfield: {exc}\n")
    print(f"Wrote {len(result['profiles'])} midfield profiles across {len(result['matches'])} matches.")


if __name__ == "__main__":
    main()
