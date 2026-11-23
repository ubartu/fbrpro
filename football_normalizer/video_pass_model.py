"""Train a pass release detector on player and ball tracks."""
import argparse
from bisect import bisect_left
from dataclasses import dataclass
from math import exp, hypot
from pathlib import Path

VERSION = "fbrpro-pass-model-v1"


def examples(samples, gap_ms):
    """Score every observed player near the ball over consecutive samples."""
    for i in range(1, len(samples)-2):
        window = samples[i-1:i+3]
        if any(sample is None for sample in window):
            continue
        before, contact, leave, after = window
        if any(not 0 < b.t_ms-a.t_ms <= gap_ms or
               (a.shot, a.period, a.ball_id) != (b.shot, b.period, b.ball_id)
               for a, b in zip(window, window[1:])):
            continue
        nearby = [(contact.foot_distance(track), track) for track in contact.players]
        if not nearby:
            continue
        distance, track = min(nearby)
        if distance > 1:
            continue
        offsets = [sample.ball_offset(track) for sample in window]
        if any(offset is None for offset in offsets):
            continue
        distances = [min(4, hypot(*offset)) for offset in offsets]
        displacement = [min(4, hypot(b[0]-a[0], b[1]-a[1])) for a, b in zip(offsets, offsets[1:])]
        other = min((leave.foot_distance(key) for key in leave.players if key != track), default=4)
        features = distances + displacement + [min(4, other)]
        yield {"t_ms": round((contact.t_ms+leave.t_ms)/2), "period": contact.period,
               "actor_sample_ms": contact.t_ms, "track_id": track,
               "evidence_start_ms": before.t_ms, "evidence_end_ms": after.t_ms}, features


@dataclass(frozen=True)
class PassModel:
    means: tuple
    scales: tuple
    weights: tuple
    bias: float
    threshold: float

    @classmethod
    def from_json(cls, value):
        from .video_stats_common import finite

        if value["version"] != VERSION:
            raise ValueError("Unsupported pass model version")
        means, scales, weights = (tuple(value[key]) for key in ("means", "scales", "weights"))
        bias, threshold = value["bias"], value["threshold"]
        if (not len(means) == len(scales) == len(weights) == 8
                or not all(finite(x) for x in (*means, *scales, *weights, bias, threshold))
                or min(scales) <= 0 or not 0 < threshold < 1):
            raise ValueError("Invalid pass model parameters")
        return cls(means, scales, weights, bias, threshold)

    def releases(self, samples, gap_ms):
        scored = []
        for row, features in examples(samples, gap_ms):
            logit = self.bias + sum(w*(x-m)/s for x, m, s, w in zip(features, self.means, self.scales, self.weights))
            score = 1/(1+exp(-max(-60, min(60, logit))))
            if score >= self.threshold:
                scored.append({**row, "pass_model_score": score, "actor_reason": "Learned player and ball movement pattern"})
        # Keep the peak when several adjacent windows describe the same release.
        peaks = []
        for row in scored:
            if peaks and row["period"] == peaks[-1]["period"] and row["t_ms"]-peaks[-1]["t_ms"] <= 250:
                if row["pass_model_score"] > peaks[-1]["pass_model_score"]:
                    peaks[-1] = row
            else:
                peaks.append(row)
        return peaks


def train(rows, labels, threshold):
    import numpy as np

    if set(labels) != {0, 1}:
        raise ValueError("Training needs both observable labeled passes and negative examples")
    x, y = np.array(rows), np.array(labels)
    means, scales = x.mean(axis=0), x.std(axis=0).clip(.01)
    x = (x-means)/scales
    weights = np.zeros(x.shape[1])
    bias = 0.
    importance = np.where(y, .5/y.sum(), .5/(1-y).sum())
    for _ in range(600):
        residual = (1/(1+np.exp(-np.clip(x @ weights+bias, -60, 60)))-y)*importance
        weights -= .1*(x.T @ residual+.03*weights)
        bias -= .1*residual.sum()
    return {"version": VERSION, "means": means.tolist(), "scales": scales.tolist(),
            "weights": weights.tolist(), "bias": float(bias), "threshold": threshold}


def main():
    from .video_events import _observations
    from .video_run import read, save
    from .video_stats_common import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.65)
    args = parser.parse_args()
    rows, targets, sources = [], [], []
    try:
        if not 0 < args.threshold < 1:
            raise ValueError("Threshold must be between zero and one")
        for item in read(args.manifest)["matches"]:
            tracking, truth = (read(args.manifest.parent / item[key]) for key in ("tracking", "labels"))
            if not tracking.get("source_sha256") or truth.get("source_sha256") != tracking["source_sha256"]:
                raise ValueError("Training labels must identify the exact video by source_sha256")
            start, end = truth["start_ms"], truth["end_ms"]
            prepared = prepare(tracking, {"start_ms": start, "end_ms": end})
            if not any(span["start_ms"] <= start < end <= span["end_ms"] for span in prepared["scope_spans"]):
                raise ValueError("Training labels must be inside an analyzed interval")
            stamps = sorted(row["t_ms"] for row in truth["events"] if row["type"] == "pass")
            if any(type(stamp) is not int or not start <= stamp < end for stamp in stamps):
                raise ValueError("Pass timestamps must be integer milliseconds within the labeled interval")
            positive_stamps = set()
            samples = list(_observations(tracking, prepared))
            gap = min(300, max(150, 1600/tracking.get("sample_fps", 10)))
            for row, features in examples(samples, gap):
                stamp = row["t_ms"]
                i = bisect_left(stamps, stamp)
                near = min(stamps[max(0,i-1):i+1], key=lambda t: abs(t-stamp), default=None)
                distance = abs(near-stamp) if near is not None else float("inf")
                if 160 < distance <= 500:
                    continue
                positive = distance <= 160
                if positive:
                    positive_stamps.add(near)
                rows.append(features)
                targets.append(int(positive))
            sources.append({"source_sha256": tracking["source_sha256"], "start_ms": start, "end_ms": end,
                            "labeled_passes": len(stamps), "observable_passes": len(positive_stamps)})
        model = train(rows, targets, args.threshold)
        save(args.output, {**model, "training_sources": sources, "positive_windows": sum(targets),
                           "negative_windows": len(targets)-sum(targets)})
        print(f"Trained on {sum(targets)} positive and {len(targets)-sum(targets)} negative windows.")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"pass training: {exc}\n")


if __name__ == "__main__":
    main()
