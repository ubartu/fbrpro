"""Compare calibration on SoccerNet images with annotated ground markings."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from .video_auto_calibration import PitchModel
from .video_pitch_lines import LinePitchModel, pitch_lines, polyline_errors
from .video_tracking import vision


def annotation_errors(cv2, np, matrix, labels, shape, lines, mirrored):
    inverse = np.linalg.inv(matrix)
    errors = []
    for name, points in labels.items():
        if name not in lines or not points:
            continue
        field = np.asarray(lines[name], np.float32)
        if mirrored:
            field = np.asarray([105., 68.], np.float32)-field
        projected = cv2.perspectiveTransform(field[:, None], inverse)[:, 0]
        pixels = np.asarray([[point["x"]*shape[1], point["y"]*shape[0]] for point in points])
        errors.extend(polyline_errors(np, pixels, projected).tolist())
    return errors


def summarize(np, rows, key):
    errors = [value for row in rows for value in row.get(key, [])]
    return {"frames": len(rows), "available": sum(key in row for row in rows),
            "annotated_points": len(errors),
            "median_error_px": float(np.median(errors)) if errors else None,
            "mean_error_px": float(np.mean(errors)) if errors else None,
            "within_5px_pct": float(np.mean(np.asarray(errors) <= 5)*100) if errors else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory of matched JPEG and SoccerNet JSON files")
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--line-model", type=Path, required=True)
    parser.add_argument("--partition", choices=("all", "development", "validation"), default="all",
                        help="Development uses every third sorted image, validation uses the rest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cv2, np = vision()
    landmark = PitchModel(cv2, np, args.landmark_model, 105, 68, inference_backend="auto")
    semantic = LinePitchModel(cv2, np, args.line_model, 105, 68, inference_backend="auto",
                             landmark_model=args.landmark_model)
    lines = pitch_lines(105, 68)
    images = sorted(args.images.glob("*.jpg"))
    results = []
    for index, image in enumerate(images):
        if args.partition == "development" and index % 3 != 0:
            continue
        if args.partition == "validation" and index % 3 == 0:
            continue
        labels = json.loads(image.with_suffix(".json").read_text())
        frame = cv2.imread(str(image))
        if frame is None:
            parser.error(f"Cannot decode {image}")
        row = {"image": image.name, "resolution": [frame.shape[1], frame.shape[0]]}
        for name, model in (("landmarks", landmark), ("semantic_lines", semantic)):
            start = perf_counter()
            fit = model.fit(frame)
            row[name+"_seconds"] = round(perf_counter()-start, 4)
            if fit is None:
                continue
            # SoccerNet evaluates the 180-degree pitch symmetry as well.
            variants = [annotation_errors(cv2, np, fit[0], labels, frame.shape, lines, mirrored)
                        for mirrored in (False, True)]
            usable = [errors for errors in variants if errors]
            if usable:
                row[name] = min(usable, key=np.mean)
        row["line_quality"] = semantic.last_quality
        results.append(row)
        print(f"{image.name}: landmarks {'landmarks' in row}, lines {'semantic_lines' in row}", flush=True)
    if not results:
        parser.error("No matching images in this partition.")
    paired = [row for row in results if "landmarks" in row and "semantic_lines" in row]
    report = {
        "source": "https://github.com/SoccerNet/sn-calibration",
        "protocol": "Ground-marking annotation distance in source pixels, including 180-degree pitch symmetry",
        "partition": args.partition, "reference_dimensions_m": [105, 68],
        "all_frames": {key: summarize(np, results, key) for key in ("landmarks", "semantic_lines")},
        "paired_frames": {key: summarize(np, paired, key) for key in ("landmarks", "semantic_lines")},
        "frames": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
