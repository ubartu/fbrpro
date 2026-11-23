"""Evaluate pass detection and attribution on labeled held-out video intervals."""
import argparse
from collections import Counter
from pathlib import Path

from .video_events import analyze
from .video_pass_model import PassModel
from .video_run import read, save
from .video_stats_common import finite
from .video_tracking import iou


def footpass_predictions(document, game, source_start_ms=0, score_min=.15):
    """Convert published TAAD predictions from 25 fps frames to clip time."""
    rows = document["events"][game]
    if type(source_start_ms) is not int or source_start_ms < 0 or not 0 <= score_min <= 1:
        raise ValueError("FOOTPASS needs a nonnegative source offset and a score from 0 to 1")
    result = []
    for frame, team, jersey, action, score in rows:
        if action != 2 or score < score_min or 40*frame < source_start_ms:
            continue
        result.append({"type": "pass", "t_ms": 40*frame-source_start_ms, "action_score": score,
                       "passer_player_id": f"{team}:{jersey}", "source": "footpass_taad"})
    return result


def _ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def _metrics(counts):
    matched, extra, missed = (counts[k] for k in ("matched", "extra", "missed"))
    result = {k: counts[k] for k in ("matched", "extra", "missed")}
    result.update(precision=_ratio(matched, matched+extra), recall=_ratio(matched, matched+missed),
                  f1=_ratio(2*matched, 2*matched+extra+missed))
    for field in ("passer_player_id", "receiver_player_id", "outcome"):
        labeled, resolved, correct = (counts[f"{field}_{k}"] for k in ("labeled", "resolved", "correct"))
        result[field] = {"labeled": labeled, "resolved": resolved, "correct": correct,
                         "coverage": _ratio(resolved, labeled), "accuracy": _ratio(correct, resolved)}
    result["outcome_coverage"] = _ratio(counts["resolved_outcomes"], matched+extra)
    return result


def compare(events, truth, tolerance_ms):
    start, end = truth["start_ms"], truth["end_ms"]
    if not finite(start) or not finite(end) or not 0 <= start < end or not finite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("Provide a valid labeled interval and nonnegative timing tolerance")
    reference = truth["events"]
    if any(not finite(row.get("t_ms")) or not start <= row["t_ms"] < end for row in reference):
        raise ValueError("Ground-truth event times must fall inside the labeled interval")
    expected = sorted((r for r in reference if r["type"] == "pass"), key=lambda r: r["t_ms"])
    detected = sorted((r for r in events if r["type"] == "pass" and start <= r["t_ms"] < end),
                      key=lambda r: r["t_ms"])
    counts = Counter(resolved_outcomes=sum(r.get("outcome") in ("complete", "incomplete") for r in detected))
    moments = []
    i = j = 0
    while i < len(expected) or j < len(detected):
        if j == len(detected) or (i < len(expected) and expected[i]["t_ms"] < detected[j]["t_ms"]-tolerance_ms):
            moments.append({"status": "missed", "reference": expected[i]})
            counts["missed"] += 1
            i += 1
        elif i == len(expected) or detected[j]["t_ms"] < expected[i]["t_ms"]-tolerance_ms:
            moments.append({"status": "extra", "detection": detected[j]})
            counts["extra"] += 1
            j += 1
        else:
            label, prediction = expected[i], detected[j]
            moments.append({"status": "matched", "reference": label, "detection": prediction,
                            "offset_ms": prediction["t_ms"]-label["t_ms"]})
            counts["matched"] += 1
            for field in ("passer_player_id", "receiver_player_id", "outcome"):
                if label.get(field) in (None, "", "unknown"):
                    continue
                counts[f"{field}_labeled"] += 1
                if prediction.get(field) not in (None, "", "unknown"):
                    counts[f"{field}_resolved"] += 1
                    counts[f"{field}_correct"] += prediction[field] == label[field]
            i += 1
            j += 1
    return counts, moments


def identity_continuity(tracking, cases):
    """Measure tracker IDs and roster IDs separately using reviewed box pairs."""
    frames = {}
    for frame in tracking["frames"]:
        stamp = frame["video_time_ms"]
        frames[stamp] = frame if stamp not in frames else None

    def match(point):
        frame = frames.get(point["t_ms"])
        if not frame:
            return None
        candidates = [row for row in frame["objects"] if row.get("object_type") == "player"
                      and row.get("observed") is True and not row.get("estimated") and row.get("bbox_xyxy")
                      and iou(row["bbox_xyxy"], point["bbox_xyxy"]) >= .5]
        return candidates[0] if len(candidates) == 1 else None

    summary = {}
    for case in cases:
        kind = case["kind"]
        if kind not in ("overlap", "reentry", "camera_cut") or type(case["same_player"]) is not bool:
            raise ValueError("Identity cases need a supported kind and a same_player boolean")
        counts = summary.setdefault(kind, Counter())
        counts["labeled"] += 1
        first, second = (match(case[key]) for key in ("before", "after"))
        if first is None or second is None:
            continue
        for key in ("track_id", "player_id"):
            if first.get(key) and second.get(key):
                counts[f"{key}_resolved"] += 1
                counts[f"{key}_correct"] += (first[key] == second[key]) == case["same_player"]
    return {kind: dict(counts) for kind, counts in summary.items()}


def run(manifest, base, tolerance_ms):
    totals, matches = {}, []
    seen = {}
    for item in manifest["matches"]:
        match_id = item["match_id"]
        tracking = read(base / item["tracking"])
        truth = read(base / item["labels"])
        source = truth.get("source_sha256")
        if source is not None and source != tracking.get("source_sha256"):
            raise ValueError(f"{match_id}: labels identify a different video")
        start, end = truth["start_ms"], truth["end_ms"]
        context = read(base / item["context"]) if "context" in item else None
        baseline = analyze(tracking, context)
        if not any(span["start_ms"] <= start < end <= span["end_ms"] for span in baseline["scope_spans"]):
            raise ValueError(f"{match_id}: labels must be inside one analyzed interval")
        intervals = seen.setdefault(source or tracking.get("source_sha256") or item["tracking"], [])
        if any(start < previous_end and previous_start < end for previous_start, previous_end in intervals):
            raise ValueError(f"{match_id}: evaluation intervals overlap")
        intervals.append((start, end))
        methods = {"tracking": baseline["events"]}
        if "pass_model" in item:
            model_data = read(base / item["pass_model"])
            if any(row["source_sha256"] == tracking.get("source_sha256")
                   and start < row["end_ms"] and row["start_ms"] < end
                   for row in model_data.get("training_sources", [])):
                raise ValueError(f"{match_id}: model training overlaps the evaluation interval")
            methods["trained"] = analyze(tracking, context, pass_model=PassModel.from_json(model_data))["events"]
        if "predictions" in item:
            spotting = read(base / item["predictions"])
            methods["soccernet"] = analyze(tracking, context, predictions=spotting, alignment=item["alignment"],
                                            include_tracking=False, attribute_predictions=False)["events"]
            methods["combined"] = analyze(tracking, context, predictions=spotting, alignment=item["alignment"])["events"]
        if "footpass" in item:
            config = item["footpass"]
            methods["footpass_taad"] = footpass_predictions(read(base / config["predictions"]), config["game"],
                                                           config.get("source_start_ms", 0), config.get("score_min", .15))
        evaluated = {}
        for method, events in methods.items():
            counts, moments = compare(events, truth, tolerance_ms)
            aggregate = totals.setdefault(method, {"counts": Counter(), "matches": set(), "intervals": 0})
            aggregate["counts"].update(counts)
            aggregate["matches"].add(match_id)
            aggregate["intervals"] += 1
            evaluated[method] = {**_metrics(counts), "moments": moments}
        matches.append({"match_id": match_id, "start_ms": start, "end_ms": end, "methods": evaluated,
                        "identity": identity_continuity(tracking, truth.get("identity_cases", []))})
    return {"version": "fbrpro-video-pass-benchmark-v1", "tolerance_ms": tolerance_ms, "matches": matches,
            "aggregate": {name: {**_metrics(row["counts"]), "match_ids": sorted(row["matches"]),
                                  "intervals": row["intervals"]} for name, row in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tolerance-ms", type=int, default=480)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        save(args.output, run(read(args.manifest), args.manifest.parent, args.tolerance_ms))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.exit(2, f"pass benchmark: {exc}\n")


if __name__ == "__main__":
    main()
