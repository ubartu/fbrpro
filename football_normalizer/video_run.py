"""Run or resume selected-clip tracking and movement analysis."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from . import video, video_geometry, video_inference, video_kits, video_metrics, video_motion, video_pitch, video_pitch_lines, video_roles, video_tracking
from .export import parquet


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
    try:
        writer(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def save(path, value):
    write_atomic(path, lambda pending: pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def write_tracking(path, report):
    with path.open("w", encoding="utf-8") as stream:
        for row in video_tracking.canonical_rows(report):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run(args):
    plan, calibration = read(args.plan), read(args.calibration)
    identities = read(args.identity_map) if args.identity_map else {}
    options = {"clip_ids": sorted(set(args.clip or [])), "sample_fps": args.sample_fps,
               "confidence": args.confidence, "max_error_m": args.max_error_m}
    line_model = getattr(args, "pitch_lines_model", None)
    if line_model is not None:
        options["pitch_lines_model"] = str(line_model)
    cv2, np = video_tracking.vision()
    source, model = file_hash(args.video), file_hash(args.model)
    tracking_inputs = {
        "source_sha256": source, "model_sha256": model, "plan": json_hash(plan),
        "calibration": json_hash(calibration), "reviewed_identities": json_hash(identities),
        "tracker_code": file_hash(Path(video_tracking.__file__)),
        "pitch_geometry_code": file_hash(Path(video_geometry.__file__)),
        "inference_code": file_hash(Path(video_inference.__file__)),
        "camera_motion_code": file_hash(Path(video_motion.__file__)),
        "pitch_code": file_hash(Path(video_pitch.__file__)),
        "kit_code": file_hash(Path(video_kits.__file__)),
        "pitch_lines_code": file_hash(Path(video_pitch_lines.__file__)),
        "pitch_lines_model": file_hash(line_model) if line_model is not None else None,
        "role_inference_code": file_hash(Path(video_roles.__file__)),
        "alignment_code": file_hash(Path(video.__file__)), "opencv": cv2.__version__,
        "numpy": np.__version__, **options,
    }
    tracking_key = json_hash(tracking_inputs)
    tracking_dir = args.output / "tracking" / tracking_key
    report_path = tracking_dir / "video-tracking.json"
    receipt_path = tracking_dir / "receipt.json"
    reused = False
    if receipt_path.is_file() and report_path.is_file():
        try:
            reused = read(receipt_path) == {"key": tracking_key, "sha256": file_hash(report_path)}
        except (OSError, ValueError):
            pass
    if reused:
        tracking = read(report_path)
    else:
        tracking = video_tracking.run(args.video, plan, args.model, calibration,
                                      identity_map=identities, **options)
        if (tracking["source_sha256"], tracking["model_sha256"]) != (source, model):
            raise ValueError("Video or model changed while tracking. Run again with stable inputs.")
        save(report_path, tracking)
        save(receipt_path, {"key": tracking_key, "sha256": file_hash(report_path)})
    canonical_path = tracking_dir / "tracking.jsonl"
    if not reused or not canonical_path.is_file():
        write_atomic(canonical_path, lambda pending: write_tracking(pending, tracking))
    result = {"version": "fbrpro-video-run-v1", "match_id": tracking["match_id"],
              "video_id": tracking["video_id"], "plan_id": tracking["plan_id"],
              "tracking_key": tracking_key, "tracking_reused": reused,
              "tracking_inputs": tracking_inputs,
              "outputs": {"tracking": str(report_path.relative_to(args.output)),
                          "canonical_tracking": str(canonical_path.relative_to(args.output))}}
    if args.parquet and canonical_path.stat().st_size:
        parquet_path = tracking_dir / "tracking.parquet"
        if not reused or not parquet_path.is_file():
            with TemporaryDirectory(dir=tracking_dir) as temporary:
                staging = Path(temporary)
                (staging / "tracking.jsonl").symlink_to(canonical_path.resolve())
                parquet(staging)
                (staging / "tracking.parquet").replace(parquet_path)
        result["outputs"]["parquet_tracking"] = str(parquet_path.relative_to(args.output))
    if args.context:
        context = read(args.context)
        bundle_inputs = {name: file_hash(args.bundle / name) for name in
                         ("status.json", "matches.jsonl", "events.jsonl", "appearances.jsonl")} if args.bundle else {}
        metric_key = json_hash({"tracking": file_hash(report_path), "context": json_hash(context),
                                "bundle": bundle_inputs, "code": file_hash(Path(video_metrics.__file__))})
        metrics = video_metrics.analyze(plan, tracking, context, bundle=args.bundle)
        metrics_path = args.output / "metrics" / metric_key / "video-metrics.json"
        save(metrics_path, metrics)
        result["outputs"]["metrics"] = str(metrics_path.relative_to(args.output))
        result["metrics_key"] = metric_key
    save(args.output / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    for arg in ("plan", "model", "calibration", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    for arg in ("identity-map", "context", "bundle"):
        parser.add_argument("--" + arg, type=Path)
    parser.add_argument("--clip", action="append", help="Only track this clip ID (repeatable)")
    parser.add_argument("--sample-fps", type=float, default=video_tracking.DEFAULT_SAMPLE_FPS)
    parser.add_argument("--pitch-lines-model", type=Path, help="SoccerNet pitch line ONNX model for automatic calibration")
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--max-error-m", type=float, default=2)
    parser.add_argument("--parquet", action="store_true", help="Export nonempty canonical tracking as Parquet")
    args = parser.parse_args(argv)
    if args.bundle and not args.context:
        parser.error("--bundle requires --context")
    try:
        result = run(args)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.exit(2, f"video run: {exc}\n")
    print("Reused tracking." if result["tracking_reused"] else "Finished tracking.")
    for stage, path in result["outputs"].items():
        print(f"{stage}: {args.output / path}")


if __name__ == "__main__":
    main()
