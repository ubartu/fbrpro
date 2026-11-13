"""Select review clips from normalized events and explicit video clock anchors."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

VERSION = "fbrpro-video-plan-v1"


@dataclass(frozen=True)
class Segment:
    id: str
    period: int
    match_start_ms: int
    match_end_ms: int
    video_start_ms: int
    video_end_ms: int
    kind: str = "live"
    check_max_ms: int | None = None

    def map(self, match_ms):
        span = self.match_end_ms - self.match_start_ms
        return round(self.video_start_ms + (match_ms - self.match_start_ms)
                     * (self.video_end_ms - self.video_start_ms) / span)

    def period_time(self, video_ms):
        span = self.video_end_ms - self.video_start_ms
        return round(self.match_start_ms + (video_ms - self.video_start_ms)
                     * (self.match_end_ms - self.match_start_ms) / span)

    @property
    def sync_status(self):
        if self.kind != "live":
            return self.kind
        if self.check_max_ms is None:
            return "unchecked"
        return "verified" if self.check_max_ms <= 500 else "rejected"


def segments_from(data):
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("Alignment must provide a segments array")
    out = []
    for i, row in enumerate(data["segments"], 1):
        s = Segment(str(row.get("id") or f"segment-{i}"), *(
            row[k] for k in ("period", "match_start_ms", "match_end_ms",
                             "video_start_ms", "video_end_ms")))
        vals = (s.period, s.match_start_ms, s.match_end_ms, s.video_start_ms, s.video_end_ms)
        if (any(type(v) is not int or v < 0 for v in vals) or s.period < 1
                or s.match_start_ms >= s.match_end_ms or s.video_start_ms >= s.video_end_ms):
            raise ValueError(f"{s.id}: invalid clock interval")
        kind = row.get("kind", "live")
        if kind not in ("live", "replay", "close_up", "break", "preview"):
            raise ValueError(f"{s.id}: unknown camera segment kind")
        checks = row.get("check_anchors", [])
        if not isinstance(checks, list):
            raise ValueError(f"{s.id}: check_anchors must be a list")
        errors = []
        for check in checks:
            if not isinstance(check, dict):
                raise ValueError(f"{s.id}: invalid independent clock anchor")
            match_ms, video_ms = check.get("match_ms"), check.get("video_ms")
            if (type(match_ms) is not int or type(video_ms) is not int
                    or not s.match_start_ms < match_ms < s.match_end_ms
                    or not s.video_start_ms < video_ms < s.video_end_ms):
                raise ValueError(f"{s.id}: independent anchors must be inside the segment")
            errors.append(abs(s.map(match_ms) - video_ms))
        out.append(Segment(s.id, *vals, kind, max(errors, default=None)))
    if not out or len({s.id for s in out}) != len(out):
        raise ValueError("Alignment requires at least one segment with unique IDs")
    ordered = sorted(out, key=lambda s: (s.period, s.match_start_ms))
    for a, b in zip(ordered, ordered[1:]):
        if (a.period == b.period and a.match_end_ms > b.match_start_ms
                or a.video_end_ms > b.video_start_ms):
            raise ValueError("Match alignment segments overlap or reverse video order")
    return ordered


def possessions(events):
    """Consecutive tagged possessions, excluding administrative events."""
    blocks = []
    for e in sorted(events, key=lambda e: (e["period"], e["period_time_ms"], e.get("provider_index") or 0)):
        if e["type"] in {"half_start", "half_end", "starting_xi", "tactical_shift", "substitution"}:
            continue
        team, possession = e.get("possession_team_id"), e.get("possession_id")
        if team is None or possession is None:
            if blocks and not blocks[-1].get("barrier"):
                blocks.append({"barrier": True})
            continue
        key = (e["period"], possession, team)
        if not blocks or blocks[-1].get("key") != key:
            blocks.append({"key": key, "start": e["period_time_ms"], "end": e["period_time_ms"],
                           "pattern": e.get("play_pattern"), "event_ids": [e["event_id"]]})
        else:
            blocks[-1]["end"] = e["period_time_ms"]
            blocks[-1]["event_ids"].append(e["event_id"])
    return blocks


def opportunities(events, team, *, include_in_possession=False):
    """Return candidate sequences, not confirmed off-ball behaviours."""
    blocks = possessions(events)
    out = []
    for i, b in enumerate(blocks):
        if "key" not in b:
            continue
        period, _, side = b["key"]
        a = blocks[i-1] if i else {}
        open_play = b["pattern"] in {"Regular Play", "From Counter"}
        if not open_play:
            continue
        if (side == team and "key" in a and a["key"][0] == period and a["key"][2] != team
                and 0 <= b["start"] - a["end"] <= 4000):
            # Last event bounds the observed phase; do not manufacture a longer run.
            start, end = max(0, b["start"]-3000), min(b["end"], b["start"]+12000)
            reason = "regain_candidate"
        elif side != team and b["end"]-b["start"] >= 10000:
            start, end = b["start"]+3000, min(b["end"], b["start"]+13000)
            reason = "opposition_build_up_sample"
        elif include_in_possession and side == team and b["end"] - b["start"] >= 6000:
            start, end = b["start"], min(b["end"], b["start"]+15000)
            reason = "own_possession_sample"
        else:
            continue
        out.append({"id": f"opportunity-{len(out)+1}", "reason": reason, "period": period,
                    "match_start_ms": start, "match_end_ms": end, "event_ids": b["event_ids"]})
    return out


def plan(events, alignment, team, *, build_up_limit=10, in_possession_limit=0):
    if any(type(n) is not int or n < 0 for n in (build_up_limit, in_possession_limit)):
        raise ValueError("Sample sizes must be non-negative integers")
    segments = segments_from(alignment)
    available, unaligned = [], []
    for o in opportunities(events, team, include_in_possession=bool(in_possession_limit)):
        s = next((s for s in segments if s.period == o["period"] and s.sync_status in {"verified", "unchecked"}
                  and s.match_start_ms <= o["match_start_ms"]
                  and o["match_end_ms"] <= s.match_end_ms), None)
        if not s:
            unaligned.append({**o, "exclusion": "No uninterrupted aligned video span"})
            continue
        start, end = s.map(o["match_start_ms"]), s.map(o["match_end_ms"])
        if start >= end:
            unaligned.append({**o, "exclusion": "No positive playback duration"})
            continue
        available.append({**o, "segment_id": s.id, "video_start_ms": start, "video_end_ms": end,
                          "sync_status": s.sync_status, "sync_check_max_ms": s.check_max_ms})
    # Midpoints of equally sized chronological buckets, deterministic across reruns.
    sample_ids = set()
    for reason, limit in (("opposition_build_up_sample", build_up_limit),
                          ("own_possession_sample", in_possession_limit)):
        candidates = [o for o in available if o["reason"] == reason]
        n = min(len(candidates), limit)
        sample_ids.update(candidates[(2*i+1)*len(candidates)//(2*n)]["id"] for i in range(n))
    chosen = [o for o in available if o["reason"] == "regain_candidate" or o["id"] in sample_ids]
    clips = []
    for o in sorted(chosen, key=lambda o: (o["video_start_ms"], o["video_end_ms"])):
        if (clips and clips[-1]["segment_id"] == o["segment_id"]
                and o["video_start_ms"] <= clips[-1]["video_end_ms"]):
            clips[-1]["video_end_ms"] = max(clips[-1]["video_end_ms"], o["video_end_ms"])
            clips[-1]["opportunities"].append(o)
        else:
            clips.append({"id": f"clip-{len(clips)+1}", "segment_id": o["segment_id"],
                          "period": o["period"], "video_start_ms": o["video_start_ms"],
                          "video_end_ms": o["video_end_ms"], "opportunities": [o]})
    counts = lambda rows: dict(Counter(o["reason"] for o in rows))
    return {"version": VERSION, "team_id": team, "clips": clips,
            "counts": {"candidate": counts(available + unaligned), "aligned": counts(available),
                       "selected": counts(chosen), "unaligned": counts(unaligned)},
            "unaligned": unaligned,
            "not_selected": [o for o in available if o["reason"] != "regain_candidate" and o["id"] not in sample_ids],
            "selected_video_ms": sum(c["video_end_ms"]-c["video_start_ms"] for c in clips),
            "analysis_scope": "selected_clips"}


def read_rows(root, name):
    with (root / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def from_bundle(root, selector, alignment, side, limit, *, in_possession_limit=0):
    if json.loads((root / "status.json").read_text()).get("status") != "accepted":
        raise ValueError("Expected an accepted normalized bundle")
    if not isinstance(alignment.get("video_id"), str) or not alignment["video_id"].strip():
        raise ValueError("Alignment requires a video_id")
    matches = [m for m in read_rows(root, "matches") if selector in
               {m["match_id"], str(m.get("source", {}).get("record_id"))}]
    if len(matches) != 1:
        raise ValueError("Match selector must resolve to exactly one match")
    match = matches[0]
    mid = match["match_id"]
    if alignment.get("match_id") not in {mid, str(match.get("source", {}).get("record_id"))}:
        raise ValueError("Alignment match_id does not match the selected fixture")
    team = match[f"{side}_team_id"]
    events = [e for e in read_rows(root, "events") if e["match_id"] == mid]
    if not events:
        raise ValueError("No normalized events for this match")
    report = plan(events, alignment, team, build_up_limit=limit, in_possession_limit=in_possession_limit)
    players = {p["player_id"]: p["name"] for p in read_rows(root, "players")}
    lineup = {(a["player_id"], a["team_id"]): a.get("shirt_number") for a in read_rows(root, "appearances")
              if a["match_id"] == mid and a.get("participated")}
    roster = [{"id": pid, "name": players.get(pid, pid), "team_id": tid,
               **({"shirt_number": number} if number is not None else {})} for (pid, tid), number in sorted(lineup.items())]
    report.update(match_id=mid, source_match_id=match.get("source", {}).get("record_id"),
                  competition_id=match.get("competition_id"), season_id=match.get("season_id"),
                  video_id=alignment["video_id"], team_side=side,
                  players=[p for p in roster if p["team_id"] == team], identity_roster=roster,
                  alignment_segments=alignment["segments"])
    content = json.dumps(report, sort_keys=True, ensure_ascii=False).encode("utf-8")
    report["plan_id"] = hashlib.sha256(content).hexdigest()
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--match", required=True)
    p.add_argument("--alignment", type=Path, required=True)
    p.add_argument("--team", choices=("home", "away"), required=True)
    p.add_argument("--build-up-limit", type=int, default=10)
    p.add_argument("--in-possession-limit", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
        report = from_bundle(args.bundle, args.match, alignment, args.team, args.build_up_limit,
                             in_possession_limit=args.in_possession_limit)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        p.exit(2, f"video: {exc}\n")
    print(f"{len(report['clips'])} clips, {report['selected_video_ms']/60000:.1f} min selected for review.")


if __name__ == "__main__":
    main()
