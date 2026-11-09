"""Compare trackers on sampled SoccerNet MOT sequences and export their tracks."""
import argparse
from collections import defaultdict
import configparser
import csv
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from .video_motion import CameraMotion
from .video_pitch import detect_pitch
from .video_tracking import Detection, Tracker, YoloX, kit_sample, vision


class FootballYolo(YoloX):
    """640px YOLO ONNX export with ball, goalkeeper, player, referee classes."""

    def __init__(self, model, confidence, *, coreml=False):
        self.cv2, self.np = vision()
        self.confidence = confidence
        if coreml:
            import onnxruntime as ort
            if "CoreMLExecutionProvider" not in ort.get_available_providers():
                raise ValueError("This ONNX Runtime installation does not provide CoreML.")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(model), sess_options=options, providers=[
                ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                                             "RequireStaticInputShapes": "1"}), "CPUExecutionProvider"])
        else:
            self.session = None
            self.net = self.cv2.dnn.readNetFromONNX(str(model))

    def _detect(self, frame, *, people_only=False):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        scale = 640/max(h, w)
        width, height = round(w*scale), round(h*scale)
        x, y = (640-width)//2, (640-height)//2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[y:y+height, x:x+width] = cv2.resize(frame, (width, height))
        blob = cv2.dnn.blobFromImage(canvas, 1/255, swapRB=True)
        if self.session is None:
            self.net.setInput(blob)
            output = self.net.forward()
        else:
            output, = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        if output.ndim != 3 or output.shape[1] != 8:
            raise ValueError("Expected a four-class YOLO ONNX output [1, 8, anchors].")
        pred = output[0].T
        classes = pred[:, 4:].argmax(axis=1)
        confidence = pred[np.arange(len(pred)), 4+classes]
        use = np.isfinite(pred).all(axis=1) & (confidence >= self.confidence)
        if people_only:
            use &= classes != 0
        pred, confidence, classes = pred[use], confidence[use], classes[use]
        half = pred[:, 2:4]/2
        first = ((pred[:, :2]-half-[x, y])/scale).clip([0, 0], [w-1, h-1])
        last = ((pred[:, :2]+half-[x, y])/scale).clip([0, 0], [w-1, h-1])
        boxes = np.concatenate((first, last-first), axis=1)
        result = []
        for ball in (False, True):
            keep = np.flatnonzero(((classes == 0) == ball) & (boxes[:, 2:] > 0).all(axis=1))
            if not len(keep):
                continue
            selected = cv2.dnn.NMSBoxes(boxes[keep].tolist(), confidence[keep].tolist(), self.confidence, .45)
            indices = keep[np.asarray(selected, dtype=int).reshape(-1)]
            if ball:
                indices = indices[:1]
            result.extend(Detection("ball" if ball else "player", float(confidence[i]),
                                    tuple(map(float, (*first[i], *last[i])))) for i in indices)
        return result


def samples(sequence, stride):
    meta = configparser.ConfigParser()
    meta.read(sequence / "seqinfo.ini")
    info = meta["Sequence"]
    game = configparser.ConfigParser()
    game.read(sequence / "gameinfo.ini")
    person_ids = {int(k.removeprefix("trackletid_")) for k, v in game["Sequence"].items()
                  if k.startswith("trackletid_") and v.split(";")[0].strip() != "ball"}
    labels = defaultdict(list)
    with (sequence / "gt" / "gt.txt").open() as stream:
        for frame, id_, x, y, w, h, score, *_ in csv.reader(stream):
            f, key = int(frame), int(id_)
            if key in person_ids and float(score) > 0 and (f-1) % stride == 0:
                x, y, w, h = map(float, (x, y, w, h))
                labels[f].append((key, Detection("player", 1.0, (x, y, x+w, y+h))))
    return info.getfloat("frameRate"), [
        (f, sequence / info["imDir"] / f"{f:06d}{info['imExt']}", labels[f])
        for f in range(1, info.getint("seqLength")+1, stride)
    ]


def tracker_output(frames, kind, fps, transforms=None):
    import numpy as np
    if kind == "bytetrack":
        from supervision import ByteTrack, Detections
        tracker = ByteTrack(track_activation_threshold=.23, lost_track_buffer=24, frame_rate=round(fps))
        for detections in frames:
            boxes = np.asarray([d.box for d in detections], dtype=float).reshape(-1, 4)
            result = tracker.update_with_detections(Detections(
                xyxy=boxes, confidence=np.asarray([d.confidence for d in detections], dtype=float)))
            yield list(zip(result.tracker_id, result.xyxy))
    else:
        from scipy.optimize import linear_sum_assignment

        class GlobalTracker(Tracker):
            def assign(self, options, detection_count):
                if not options:
                    return []
                size = len(self.active)
                costs = np.full((size, detection_count+size), 1e6)
                costs[:, detection_count:] = 0
                for cost, i, j in options:
                    costs[i, j] = cost - 4
                rows, columns = linear_sum_assignment(costs)
                return [(costs[i, j], i, j) for i, j in zip(rows, columns) if costs[i, j]<0]

        factory = GlobalTracker if kind in {"global", "camera_global"} else Tracker
        tracker = factory("benchmark", max_gap_ms=max(800, round(2000/fps)))
        for i, detections in enumerate(frames):
            camera = transforms[i] if transforms is not None else None
            yield [(t.id, t.last.box) for t in tracker.update(detections, round(i*1000/fps), camera) if not t.missing]


def evaluate(truth, predictions, hota, clear):
    import numpy as np
    from trackeval.datasets._base_dataset import _BaseDataset

    def split(frames):
        ids = {id_: i for i, id_ in enumerate(dict.fromkeys(id_ for frame in frames for id_, _ in frame))}
        return (len(ids), [np.asarray([ids[id_] for id_, _ in frame], dtype=int) for frame in frames],
                [np.asarray([b for _, b in frame], dtype=float).reshape(-1, 4) for frame in frames])

    ng, gt_ids, gt_boxes = split(truth)
    nt, tr_ids, tr_boxes = split(predictions)
    data = {"num_gt_ids": ng, "num_tracker_ids": nt, "gt_ids": gt_ids, "tracker_ids": tr_ids,
            "num_gt_dets": sum(map(len, gt_ids)), "num_tracker_dets": sum(map(len, tr_ids)),
            "num_timesteps": len(truth), "similarity_scores": [
                _BaseDataset._calculate_box_ious(a, b, box_format="x0y0x1y1")
                for a, b in zip(gt_boxes, tr_boxes, strict=True)]}
    return hota.eval_sequence(data), clear.eval_sequence(data)


def export_mot(path, frames):
    """Use consecutive 1-based sample indexes in the MOT text export."""
    ids = {}
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for frame, objects in enumerate(frames, 1):
            for id_, (x, y, r, b) in objects:
                key = ids.setdefault(id_, len(ids)+1)
                writer.writerow((frame, key, x, y, r-x, b-y, 1, -1, -1, -1))


def summary(hota, clear):
    return {**{k: round(float(hota[k].mean()*100), 2) for k in ("HOTA", "DetA", "AssA")},
            **{k: int(clear[k]) for k in ("IDSW", "CLR_TP", "CLR_FP", "CLR_FN")}}


def annotate_color(cv2, np, image, detection):
    return replace(detection, kit_color=kit_sample(cv2, np, image, detection.box))


def collect(sequence, stride, detector):
    cv2, np = vision()
    fps, source = samples(sequence, stride)
    if any(not path.is_file() for _, path, _ in source):
        raise ValueError(f"{sequence.name} is missing sampled images")
    gt, inputs, seconds, frames = [], [], 0.0, []
    for frame, path, labels in source:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode {path}")
        boundary = detect_pitch(cv2, np, image)
        gt.append([(id_, annotate_color(cv2, np, image, d)) for id_, d in labels
                   if boundary and boundary.contains(d.anchor())])
        start = perf_counter()
        detections = detector.detect(image, boundary.polygon_xy, accepts_person=boundary.contains) if boundary else []
        seconds += perf_counter()-start
        inputs.append([annotate_color(cv2, np, image, d) for d in detections if d.object_type=="player"])
        frames.append(frame)
    return fps/stride, frames, gt, inputs, seconds


def camera_transforms(sequence, stride, inputs):
    cv2, np = vision()
    motion = CameraMotion(cv2, np)
    _, source = samples(sequence, stride)
    return [motion.update(cv2.imread(str(path)), (d.box for d in frame))
            for (_, path, _), frame in zip(source, inputs, strict=True)]


def main(argv=None):
    from trackeval.metrics import CLEAR, HOTA

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequences", nargs="+", type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--detector", choices=("yolox", "football"), default="yolox")
    p.add_argument("--coreml", action="store_true", help="Run the football detector with ONNX Runtime and CoreML")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.stride < 1:
        p.error("The sample stride must be positive.")
    if args.coreml and args.detector != "football":
        p.error("--coreml requires --detector football")
    hota, clear = HOTA(), CLEAR({"THRESHOLD": .5, "PRINT_CONFIG": False})
    detector = (YoloX(args.model, .23) if args.detector == "yolox" else
                FootballYolo(args.model, .23, coreml=args.coreml))
    args.output.mkdir()
    scores = defaultdict(dict)
    results = {"detector": args.detector, "stride": args.stride,
               "backend": "CoreML/ONNX Runtime" if args.coreml else "OpenCV",
               "scope": "person boxes inside the same estimated pitch mask, balls excluded",
               "sequences": {}}
    for sequence in args.sequences:
        fps, frames, gt, detections, seconds = collect(sequence, args.stride, detector)
        gt_boxes = [[(id_, d.box) for id_, d in rows] for rows in gt]
        row = {"sample_fps": fps, "frames": len(frames), "gt_boxes": sum(map(len, gt)),
               "detector_seconds": round(seconds, 2), "runs": {}}
        out = args.output / sequence.name
        out.mkdir()
        export_mot(out / "gt.txt", gt_boxes)
        (out / "source-frames.json").write_text(json.dumps(frames))
        for name, inputs in (("annotated_boxes", [[d for _, d in rows] for rows in gt]), ("detector", detections)):
            transforms = camera_transforms(sequence, args.stride, inputs)
            for kind in ("current", "bytetrack", "global", "camera", "camera_global"):
                key = f"{name}/{kind}"
                start = perf_counter()
                predicted = list(tracker_output(inputs, kind, fps, transforms if kind.startswith("camera") else None))
                row["runs"][key] = {"tracker_seconds": round(perf_counter()-start, 3)}
                h, c = evaluate(gt_boxes, predicted, hota, clear)
                scores[key][sequence.name] = (h, c)
                row["runs"][key].update(summary(h, c))
                export_mot(out / (key.replace("/", "-")+".txt"), predicted)
        results["sequences"][sequence.name] = row
        print(sequence.name, json.dumps(row), flush=True)
    results["combined"] = {k: summary(
        hota.combine_sequences({n: h for n, (h, _) in seq.items()}),
        clear.combine_sequences({n: c for n, (_, c) in seq.items()})) for k, seq in scores.items()}
    (args.output / "results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results["combined"], indent=2))


if __name__ == "__main__":
    main()
