"""Compare automatic clip observations with independent analyst exports."""
import argparse
from collections import Counter
from pathlib import Path

from .video_metrics import MEASURES, VERSION
from .video_run import read, save

REVIEW_VERSION = "fbrpro-video-observations-v1"
LEGACY = {"screening": "central_screening", "deep_to_advanced": "deep_to_advanced_support"}
OUTCOMES = {"observed", "not_observed", "unobservable"}


def key(row, player_id, metric):
    return row["clip_id"], row["opportunity_id"], player_id, metric


def review_index(review, report):
    if review.get("version") != REVIEW_VERSION or any(
            review.get(k) != report[k] for k in ("match_id", "video_id", "plan_id")):
        raise ValueError("Each analyst export must reference the same match, video and plan as the movement report")
    rows = {}
    for row in review["observations"]:
        metric = LEGACY.get(row["behaviour"], row["behaviour"])
        identity = key(row, row["player_id"], metric)
        if metric not in MEASURES or row["observation"] not in OUTCOMES or identity in rows:
            raise ValueError("Unknown or duplicated analyst observation")
        rows[identity] = row["observation"]
    return rows


def compare(report, reviews):
    if report.get("version") != VERSION or report.get("analysis_scope") != "selected_clips":
        raise ValueError("Validation needs an individual match movement report")
    by_reviewer = [review_index(review, report) for review in reviews]
    if len(by_reviewer) < 2 or not all(by_reviewer):
        raise ValueError("Provide at least two nonempty, independent analyst exports")
    automatic = {
        key(row, player["player_id"], measure["metric"]): row
        for player in report["players"] for measure in player["measures"] for row in measure["observations"]
    }
    evidence = []
    counts = {metric: Counter() for metric in MEASURES}
    for identity in sorted(set().union(*by_reviewer)):
        clip, opportunity, player, metric = identity
        votes = [review.get(identity) for review in by_reviewer]
        row = automatic.get(identity)
        if None in votes:
            result = "missing_review"
        elif len(set(votes)) != 1:
            result = "analyst_disagreement"
        elif votes[0] == "unobservable":
            result = "analyst_unobservable"
        elif not row or row["status"] not in {"observed", "not_observed"}:
            result = "automatic_unobservable"
        else:
            actual, predicted = votes[0] == "observed", row["status"] == "observed"
            result = ("true_positive" if actual else "false_positive") if predicted else (
                      "false_negative" if actual else "true_negative")
        counts[metric][result] += 1
        evidence.append({"clip_id": clip, "opportunity_id": opportunity, "player_id": player,
                         "metric": metric, "reviews": votes,
                         "automatic_status": row["status"] if row else None, "result": result})
    measures = []
    for metric, c in counts.items():
        if not c:
            continue
        tp, fp, fn = (c[k] for k in ("true_positive", "false_positive", "false_negative"))
        comparable = tp + fp + fn + c["true_negative"]
        measures.append({"metric": metric, "counts": dict(c), "comparable": comparable,
                         "precision": tp/(tp+fp) if tp+fp else None,
                         "recall": tp/(tp+fn) if tp+fn else None})
    return {"version": "fbrpro-video-validation-v1", "match_id": report["match_id"],
            "video_id": report["video_id"], "plan_id": report["plan_id"],
            "reviewers": len(reviews), "measures": measures, "observations": evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--review", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if len({p.resolve() for p in args.review}) != len(args.review):
            raise ValueError("Choose one distinct file per reviewer")
        if args.output.exists():
            raise ValueError("Choose a new output path for the validation report")
        report = compare(read(args.report), [read(path) for path in args.review])
        save(args.output, report)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.exit(2, f"video validation: {exc}\n")
    print(f"Compared {len(report['observations'])} reviewed opportunities.")


if __name__ == "__main__":
    main()
