"""Optional video-statistics report. Does not run or modify tracking inference."""
import argparse
import hashlib
import json
from pathlib import Path

from .video_stats_common import prepare


VERSION = "fbrpro-video-stats-v1"


def analyze(tracking, context=None):
    # Imports are deliberately lazy: the stable inference pipeline does not
    # import these modules, and statistics can fail independently of tracking.
    from .video_stats_physical import calculate as calculate_physical
    from .video_stats_behaviors import calculate as calculate_behaviors

    prepared = prepare(tracking, context)
    physical = calculate_physical(prepared)
    behaviors = calculate_behaviors(prepared, prepared["context"])
    players = {p["id"]: p for p in physical["players"]}
    for p in behaviors.get("players", []):
        players.setdefault(p["id"], {**prepared["tracks"][p["id"]]}).update(p)
    digest = hashlib.sha256(json.dumps({"source_sha256": tracking.get("source_sha256"),
        "model_sha256": tracking.get("model_sha256"), "plan_id": tracking.get("plan_id"),
        "context": prepared["context"], "definition_version": "fbrpro-observed-video-statistics-v1",
        "measurement_metadata": {key: tracking.get(key) for key in ("coordinate_frame", "coordinate_length_m", "coordinate_width_m", "dimensions_assumed", "sample_fps", "clips")}}, sort_keys=True, allow_nan=False).encode())
    for frame in tracking["frames"]:
        digest.update(json.dumps(frame, sort_keys=True, allow_nan=False).encode())
    fingerprint = digest.hexdigest()
    return {"version": VERSION, "definition_version": "fbrpro-observed-video-statistics-v1",
            **{key: tracking.get(key) for key in ("match_id", "video_id", "plan_id", "source_sha256")},
            "analysis_scope": "selected_observed_video", "analysis_fingerprint": fingerprint,
            "coordinate_length_m": prepared["length"], "coordinate_width_m": prepared["width"],
            "scope_spans": prepared["scope_spans"], "scope_ms": prepared["scope_ms"],
            "quality": prepared["quality"], "config": prepared["config"], "context": prepared["context"],
            "players": list(players.values()), "teams": physical.get("teams", []),
            "event_candidates": behaviors.get("event_candidates", []), "possession": behaviors.get("possession", {}),
            "notes": ["Calculated separately from tracking. Source detections, IDs and replay coordinates are unchanged.",
                      "Statistics describe observable video scope, never extrapolated full-match totals.",
                      "Track IDs are not shirt numbers; identities and teams need review.",
                      "Automatic mappings and physical values are estimates, not validated GPS measurements.",
                      "Off-ball geometry and ball proximity candidates do not prove tactical intent or possession.",
                      "Unavailable event metrics need reviewed identities, events and complete coverage annotations.",
                      *behaviors.get("notes", [])]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze(json.loads(args.tracking.read_text()), json.loads(args.context.read_text()) if args.context else {})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"video_statistics: {exc}\n")
    print(f"{len(result['players'])} observed-track profiles and {len(result['teams'])} team summaries written.")


if __name__ == "__main__":
    main()
