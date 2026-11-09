"""Compare ball localization in SoccerNet footage, including around spotted actions."""
import argparse
from bisect import bisect_left
from collections import Counter
import configparser
import json
from math import hypot, isfinite
from pathlib import Path
from time import perf_counter

from .video_ball import BallDetector, BallTracker
from .video_benchmark import samples
from .video_inference import BACKENDS
from .video_motion import CameraMotion
from .video_tracking import Detection, YoloX, pitch_detections, vision


def spotting_events(data, half, clip_start_ms):
    """Map sn-spotting's half-relative video positions to the matching clip."""
    events = []
    for row in data["annotations"]:
        if int(row["gameTime"].split(" - ", 1)[0]) != half or row.get("visibility") == "not shown":
            continue
        position = float(row["position"])
        if not isfinite(position) or position < 0:
            raise ValueError("A SoccerNet action position must be a non-negative video timestamp.")
        events.append((round(position)-clip_start_ms, row["label"]))
    return sorted(set(events))


def tracking_events(sequence):
    config = configparser.ConfigParser()
    config.read(sequence / "gameinfo.ini")
    row = config["Sequence"]
    return [(row.getint("actionPosition")-row.getint("clipStart"), row["actionClass"])]


def score(counts, track, truth, tolerance):
    counts["sampled"] += 1
    counts["labelled"] += bool(truth)
    observed = track is not None and not track.missing
    matched = False
    if observed:
        x, y = track.last.anchor()
        matched = any(hypot(x-tx, y-ty) <= tolerance for tx, ty in (d.anchor() for _, d in truth))
    counts["matched"] += matched
    counts["unmatched"] += observed and not matched


def compare(sequence, people, detector, events, stride, window_ms, tolerance):
    cv2, np = detector.cv2, detector.np
    fps, frames = samples(sequence, stride, object_type="ball")
    if not frames:
        raise ValueError(f"{sequence}: no video frames")
    last_sample_ms = round((frames[-1][0]-1)*1000/fps)
    events = [(ms, label) for ms, label in events if -window_ms <= ms <= last_sample_ms+window_ms]
    if not events:
        raise ValueError(f"{sequence}: no action windows overlap the clip. Check the half and video offset.")
    times = [ms for ms, _ in events]
    motion = CameraMotion(cv2, np)
    baseline = enhanced = None
    scores = {k: Counter(sampled=0, labelled=0, matched=0, unmatched=0)
              for k in ("baseline", "enhanced", "baseline_near_actions", "enhanced_near_actions")}
    full_seconds = crop_seconds = 0.
    crop_calls = 0
    for fid, path, truth in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Missing sampled image {path}")
        h, w = frame.shape[:2]
        if baseline is None:
            baseline, enhanced = BallTracker(sequence.name, w), BallTracker(sequence.name, w)
        players, boundary, _, _ = pitch_detections(cv2, np, frame, people, None, 105, 68)
        transform = motion.update(frame, (d.box for d in players if d.object_type == "player"))
        stamp = round((fid-1)*1000/fps)
        start = perf_counter()
        candidates = [Detection("ball", s, b) for s, b in detector.detect(frame, boundary)]
        full_seconds += perf_counter()-start

        def recover(center):
            nonlocal crop_calls, crop_seconds
            if boundary is None or not (0 <= center[0] < w and 0 <= center[1] < h):
                return []
            crop_calls += 1
            start = perf_counter()
            rows = [Detection("ball", s, b) for s, b in detector.recover(frame, boundary, center)]
            crop_seconds += perf_counter()-start
            return rows

        tracks = (baseline.update(candidates, stamp, transform),
                  enhanced.update(candidates, stamp, transform, recover=recover))
        idx = bisect_left(times, stamp-window_ms)
        for name, track in zip(("baseline", "enhanced"), tracks):
            score(scores[name], track, truth, tolerance)
            if idx < len(times) and times[idx] <= stamp+window_ms:
                score(scores[name+"_near_actions"], track, truth, tolerance)
    return {"sequence": sequence.name, "sample_fps": fps/stride, "events": events, **scores,
            "extra_crop_inferences": crop_calls, "full_ball_seconds": round(full_seconds, 3),
            "crop_ball_seconds": round(crop_seconds, 3)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True, help="Person detector ONNX")
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--inference-backend", choices=BACKENDS, default="auto")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=2000, help="Time on either side of an action")
    parser.add_argument("--tolerance-px", type=int, default=12, help="Maximum ball-center error in source pixels")
    parser.add_argument("--spotting-labels", type=Path, help="Labels-ball.json for the exact same broadcast")
    parser.add_argument("--half", type=int)
    parser.add_argument("--clip-start-ms", type=int, help="Clip start on the SoccerNet half's video timeline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.stride, args.window_ms, args.tolerance_px) < 1:
        parser.error("Stride, window and tolerance must be positive.")
    if args.spotting_labels:
        if (len(args.sequences) != 1 or args.half not in (1, 2)
                or args.clip_start_ms is None or args.clip_start_ms < 0):
            parser.error("Spotting labels require one sequence, its half and its clip start in milliseconds.")
        labels = spotting_events(json.loads(args.spotting_labels.read_text()), args.half, args.clip_start_ms)
    elif args.half is not None or args.clip_start_ms is not None:
        parser.error("Use --half and --clip-start-ms with --spotting-labels.")
    cv2, np = vision()
    people = YoloX(args.model, .25, inference_backend=args.inference_backend)
    detector = BallDetector(cv2, np, args.ball_model, inference_backend=args.inference_backend)
    results = []
    for sequence in args.sequences:
        events = labels if args.spotting_labels else tracking_events(sequence)
        result = compare(sequence, people, detector, events, args.stride, args.window_ms, args.tolerance_px)
        results.append(result)
        print(f"{sequence.name}: {result['baseline']['matched']} → {result['enhanced']['matched']} matches", flush=True)
    report = {"event_source": "sn-spotting" if args.spotting_labels else "tracking_gameinfo",
              "window_ms": args.window_ms, "tolerance_px": args.tolerance_px, "clips": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
