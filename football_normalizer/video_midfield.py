"""Add identity-linked video evidence to a normalized multi-match midfield report."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from .midfield import VERSION
from .relative_roles import TITLES, usable
from .role_assessment import DEFENSIVE
from .video_run import read, save
from .video_stats_common import finite, prepare
from .video_stats_physical import calculate

# Rankings use a shared set of measures for every eligible peer in a match.
ROLE_SIGNALS = {
    "deep_lying_playmaker": ("deep_distribution", "deep_progressive_passing", "deep_central_share"),
    "advanced_playmaker": ("advanced_distribution", "advanced_chance_creation", "advanced_central_share"),
    "ball_winning_midfielder": ("pressing", "midfield_interventions", "middle_third_share"),
    "holding_midfielder": ("deep_distribution", "deep_interventions", "deep_central_share"),
    "box_to_box_midfielder": ("progressive_carrying", "box_involvement", "end_thirds_balance", "movement_rate"),
}
VIDEO_KEYS = {"deep_central_share", "advanced_central_share", "middle_third_share", "end_thirds_balance", "movement_rate"}


def _profiles_by_match(report):
    by_match = defaultdict(list)
    for profile in report["profiles"]:
        for row in profile["matches"]:
            by_match[row["match_id"]].append((profile, row))
    return by_match


def _video_evidence(tracking, context, targets):
    verified = {row["segment_id"] for row in tracking.get("sync", []) if row["status"] == "verified"}
    if not verified:
        raise ValueError("Video evidence requires a verified match-clock alignment")
    prepared = prepare(tracking, context)
    clocks = {}
    for frame in tracking["frames"]:
        if frame.get("segment_id") in verified and finite(frame.get("period_time_ms")):
            stamp = frame["video_time_ms"]
            clocks[stamp] = (frame.get("period", 1), frame["period_time_ms"], frame["segment_id"]) if stamp not in clocks else None

    groups = defaultdict(list)
    for profile, row in targets:
        groups[tuple(profile["scope"])].append((profile, row))

    for members in groups.values():
        records = {(p["player_id"], p["team_id"]): row for p, row in members}
        if len(records) != len(members):
            raise ValueError("Multiple midfield profiles for one player in the same match and position group")
        frames = []
        by_sample = {}
        for frame in prepared["frames"]:
            stamp = frame["t_ms"]
            clock = clocks.get(stamp)
            players = []
            if clock and frame["live"] is not False:
                period, match_time, segment = clock
                identities = Counter((p["player_id"], p["team"]) for p in frame["players"] if p["player_id"])
                directions = prepared["directions"].get(str(period), {})
                for player in frame["players"]:
                    ident = player["player_id"], player["team"]
                    row = records.get(ident)
                    direction = directions.get(player["team"])
                    if (row is None or identities[ident] != 1
                            or not any(p == period and start <= match_time < end for p, start, end in row["intervals"])):
                        continue
                    # Role zones always face the direction of attack.
                    if direction == -1:
                        player = {**player, "x": prepared["length"]-player["x"],
                                  "y": prepared["width"]-player["y"]}
                    if direction in (-1, 1):
                        players.append(player)
                    by_sample[(stamp, player["id"], period)] = ident
            if players:
                # Force a break if the raw verified segment or orientation changes.
                frames.append({**frame, "players": players, "ball": None, "segment_id": (
                    frame["segment_id"], period, segment)})
            else:
                frames.append({**frame, "players": [], "ball": None})

        data = {**prepared, "frames": frames, "directions": {}, "observed_by_track": {}}
        results = defaultdict(list)
        for result in calculate(data)["players"]:
            if result["coverage"]["usable_ms"]:
                results[result["player_id"], result["team"]].append(result)
        counts = defaultdict(Counter)
        for event in prepared["context"].get("events", []):
            if event.get("reviewed") is True and event.get("track_confirmed") is True:
                identity = by_sample.get((event["t_ms"], event["track_id"], event.get("period", 1)))
                if identity:
                    counts[identity][event["type"]] += 1

        for identity, row in records.items():
            usable_ms = motion_ms = 0
            distance_pct = 0
            zones = [0]*9
            for result in results[identity]:
                usable_ms += result["coverage"]["usable_ms"]
                motion = result["movement"]
                motion_ms += motion["relative_usable_ms"]
                distance_pct += motion["relative_distance_pct"] or 0
                for index, value in enumerate(result["positioning"]["zones"]["time_ms"]):
                    zones[index] += value
            resolved = sum(zones)
            own, middle, final = (sum(zones[i::3]) for i in range(3))
            role_inputs = {
                "deep_central_share": zones[3]/resolved if resolved else None,
                "advanced_central_share": zones[5]/resolved if resolved else None,
                "middle_third_share": middle/resolved if resolved else None,
                "end_thirds_balance": 2*min(own, final)/resolved if resolved else None,
                "movement_rate": distance_pct*60_000/motion_ms if motion_ms else None,
            }
            row["video_role_evidence"] = {
                "usable_ms": usable_ms, "resolved_zone_ms": resolved, "movement_ms": motion_ms,
                "midfield_position_ms": round(row["estimated_minutes"]*60_000),
                "zone_time_ms": zones, "reviewed_actions": dict(counts[identity]),
                "video_id": tracking.get("video_id"), "source_sha256": tracking.get("source_sha256"),
                "measures": role_inputs, "roles": {}, "relative_labels": [],
            }


def _compare(members, defensive_coverage):
    for role, keys in ROLE_SIGNALS.items():
        cohort = []
        for profile, row in members:
            video = row["video_role_evidence"]
            signals = []
            for key in keys:
                if key in VIDEO_KEYS:
                    duration = video["movement_ms"] if key == "movement_rate" else video["resolved_zone_ms"]
                    signals.append({"metric": key, "source": "video", "value": video["measures"][key], "usable_ms": duration})
                else:
                    metric = row["metrics"].get(key)
                    value = metric["per_100"] if metric and usable(metric) and (key not in DEFENSIVE or defensive_coverage) else None
                    signals.append({"metric": key, "source": "event_feed", "value": value,
                                    "coverage": metric.get("coverage") if metric else None,
                                    "opportunities": metric.get("denominator") if metric else None})
            available = all(signal["value"] is not None for signal in signals)
            result = {"label": TITLES[role], "comparison_pct": None, "peer_count": 0, "signals": signals,
                      "reason": "Needs aligned video and usable event evidence for each measure" if not available else
                      "Needs another midfielder with the same measures"}
            video["roles"][role] = result
            if available:
                cohort.append((profile["player_id"], result))
        for player_id, row in cohort:
            peers = [other for peer, other in cohort if peer != player_id]
            row["peer_count"] = len(peers)
            if not peers:
                continue
            for i, signal in enumerate(row["signals"]):
                values = [p["signals"][i]["value"] for p in peers]
                target = signal["value"]
                higher = sum(target > value for value in values)
                signal["higher_than"] = higher
                signal["comparison_pct"] = 100*(higher+.5*values.count(target))/len(values)
            row["comparison_pct"] = sum(s["comparison_pct"] for s in row["signals"])/len(keys)
            row["reason"] = None

    for _, row in members:
        video = row["video_role_evidence"]
        candidates = {role: v["comparison_pct"] for role, v in video["roles"].items()
                      if v["comparison_pct"] is not None and any(s["higher_than"] for s in v["signals"])}
        best = max(candidates.values(), default=None)
        video["relative_labels"] = [key for key, score in candidates.items() if score == best]


def enrich(report, entries):
    if report.get("version") != VERSION:
        raise ValueError("Choose a normalized midfield-v0.1 report")
    report = deepcopy(report)
    by_match = _profiles_by_match(report)
    for profile in report["profiles"]:
        profile.pop("video_role_summary", None)
        for row in profile["matches"]:
            row.pop("video_role_evidence", None)
    seen = set()
    for tracking, context in entries:
        match_id = tracking.get("match_id")
        if match_id not in by_match or match_id in seen:
            raise ValueError("Each video report needs a distinct match from the loaded midfield report")
        seen.add(match_id)
        targets = by_match[match_id]
        _video_evidence(tracking, context, targets)
        cohorts = defaultdict(list)
        for profile, row in targets:
            cohorts[tuple(profile["scope"])].append((profile, row))
        for members in cohorts.values():
            _compare(members, report.get("defensive_coverage_confirmed") is True)
    for profile in report["profiles"]:
        matches = [row["video_role_evidence"] for row in profile["matches"] if "video_role_evidence" in row]
        if not matches:
            continue
        profile["video_role_summary"] = {
            "matches_with_video": sum(row["usable_ms"] > 0 for row in matches),
            "total_matches": len(profile["matches"]),
            "usable_ms": sum(row["usable_ms"] for row in matches),
            "roles": {role: {"label": TITLES[role],
                             "scored_matches": sum(row["roles"][role]["comparison_pct"] is not None for row in matches),
                             "strongest_matches": sum(role in row["relative_labels"] for row in matches),
                             "usable_ms": sum(row["usable_ms"] for row in matches if row["roles"][role]["comparison_pct"] is not None)}
                      for role in ROLE_SIGNALS},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midfield_report", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with a videos list of tracking and optional context paths")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        base = args.manifest.parent
        entries = [(read(base / item["tracking"]), read(base / item["context"]) if item.get("context") else None)
                   for item in read(args.manifest)["videos"]]
        save(args.output, enrich(read(args.midfield_report), entries))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"video midfield: {exc}\n")


if __name__ == "__main__":
    main()
