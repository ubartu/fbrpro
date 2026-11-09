"""Serve the video workbench and track uploaded footage locally."""
import argparse
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
from shutil import disk_usage
from tempfile import TemporaryDirectory
from threading import Event, Lock
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

from . import video, video_tracking


WEB_ROOT = Path(__file__).resolve().parent.parent
STATIC = {
    "/video/": "video/index.html",
    "/video/index.html": "video/index.html",
    "/video/review.css": "video/review.css",
    "/video/review.js": "video/review.js",
    "/video/analysis.js": "video/analysis.js",
    "/dashboard/": "dashboard/index.html",
    "/dashboard/index.html": "dashboard/index.html",
}
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_FULL_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
DISK_RESERVE_BYTES = 512 * 1024 * 1024
MAX_WINDOW_MS = 15_000
MAX_VIDEO_MS = video_tracking.MAX_FULL_VIDEO_MS
CHUNK_SIZE = 1024 * 1024


def preview_plan(start, end, scope="video_preview"):
    key = uuid4().hex
    window = {"video_start_ms": start, "video_end_ms": end,
              "match_start_ms": start, "match_end_ms": end, "period": 1}
    clip = {**window, "id": "preview-clip", "segment_id": "preview-segment", "opportunities": []}
    return {"version": video.VERSION, "plan_id": f"preview-plan-{key}",
            "video_id": f"preview-video-{key}", "match_id": f"local-preview-{key}",
            "analysis_scope": scope, "clock_source": "video_only_unverified",
            "players": [], "counts": {"selected": {}}, "unaligned": [],
            "selected_video_ms": end - start, "clips": [clip],
            "alignment_segments": [{**window, "id": "preview-segment", "kind": "preview"}]}


def estimate_duration_ms(path):
    """Use decoder metadata as a progress estimate, never as an EOF boundary."""
    cv2, _ = video_tracking.vision()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Unable to open the uploaded video.")
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if math.isfinite(count) and math.isfinite(fps) and count > 0 and fps > 0:
            milliseconds = 1000 * (count / fps)
            if math.isfinite(milliseconds):
                return math.ceil(milliseconds)
        return None
    finally:
        cap.release()


def job_id(fields):
    values = fields.get("job", [])
    if len(values) != 1 or len(values[0]) != 36:
        raise ValueError("Use a valid tracking job identifier.")
    return str(UUID(values[0]))


class JobCancelled(ValueError):
    pass


@dataclass
class FullVideoJob:
    id: str | None
    total_ms: int | None = None
    processed_ms: int = 0
    stage: str = "uploading"
    cancelled: Event = field(default_factory=Event)

    def progress(self, video_ms=None):
        if self.cancelled.is_set():
            raise JobCancelled("Tracking was cancelled.")
        if video_ms is not None:
            if video_ms > MAX_VIDEO_MS:
                raise ValueError("Full-video tracking supports footage up to six hours.")
            self.processed_ms = video_ms

    def snapshot(self):
        return {"processed_ms": self.processed_ms, "total_ms": self.total_ms,
                "stage": "cancelling" if self.cancelled.is_set() else self.stage}


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, port, model, demo=None):
        super().__init__(("127.0.0.1", port), ReviewHandler)
        self.model = model
        self.demo = demo or {}
        self.job = Lock()
        self.active_job = None
        self.hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}


class ReviewHandler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(30)
        super().setup()

    def send_headers(self, status, mime, size, **headers):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_headers(status, "application/json; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def error(self, status, message):
        self.send_json(status, {"error": message})

    def same_origin(self):
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if (host not in self.server.hosts or origin not in (None, f"http://{host}")
                or self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none")):
            self.error(HTTPStatus.FORBIDDEN, "Open this workbench directly on its local address.")
            return False
        return True

    def requested_job(self, target):
        try:
            fields = parse_qs(target.query, strict_parsing=True, max_num_fields=1)
            id_ = job_id(fields)
        except ValueError:
            self.error(HTTPStatus.BAD_REQUEST, "Use a valid tracking job identifier.")
            return None
        job = self.server.active_job
        if job is None or job.id != id_:
            self.error(HTTPStatus.NOT_FOUND, "This tracking job is no longer active.")
            return None
        return job

    def full_report(self, directory, report, job):
        # Whole-match manifests can be large. Encode to a temporary file to
        # avoid a second in-memory copy of the JSON payload.
        path = directory / "tracking.json"
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with path.open("w", encoding="utf-8") as stream:
            for index, fragment in enumerate(encoder.iterencode(report)):
                if index % 4096 == 0:
                    job.progress()
                stream.write(fragment)
        job.progress()
        self.file(path, mime="application/json; charset=utf-8")

    def file(self, path, *, ranges=False, mime=None):
        try:
            stream = path.open("rb")
        except OSError:
            self.error(HTTPStatus.NOT_FOUND, "The requested workbench file is unavailable.")
            return
        with stream:
            size = path.stat().st_size
            start, end = 0, size - 1
            status, headers = HTTPStatus.OK, {}
            if ranges:
                headers["Accept-Ranges"] = "bytes"
            requested = self.headers.get("Range") if ranges else None
            if requested is not None:
                match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", requested)
                if match and any(match.groups()):
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                if not match or not any(match.groups()) or not 0 <= start <= end < size:
                    self.send_headers(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "application/octet-stream", 0,
                                      **{"Content-Range": f"bytes */{size}"})
                    return
                status = HTTPStatus.PARTIAL_CONTENT
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            length = end - start + 1
            mime = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_headers(status, mime, length, **headers)
            if self.command == "HEAD":
                return
            stream.seek(start)
            while length:
                block = stream.read(min(length, CHUNK_SIZE))
                if not block:
                    break
                self.wfile.write(block)
                length -= len(block)

    def do_GET(self):
        if not self.same_origin():
            return
        target = urlsplit(self.path)
        path = target.path
        if path in {"/", "/video", "/dashboard"}:
            self.send_headers(HTTPStatus.FOUND, "text/plain", 0,
                              Location="/dashboard/" if path == "/dashboard" else "/video/")
        elif path in STATIC:
            self.file(WEB_ROOT / STATIC[path])
        elif path == "/api/status":
            demo = self.server.demo
            self.send_json(HTTPStatus.OK, {"demo_available": "video" in demo and "tracking" in demo,
                                           "demo_plan_available": "plan" in demo,
                                           "max_upload_bytes": MAX_UPLOAD_BYTES, "max_window_ms": MAX_WINDOW_MS,
                                           "max_full_upload_bytes": MAX_FULL_UPLOAD_BYTES,
                                           "max_full_video_ms": MAX_VIDEO_MS})
        elif path == "/api/progress":
            job = self.requested_job(target)
            if job:
                self.send_json(HTTPStatus.OK, job.snapshot())
        elif path.startswith("/api/demo/") and path.removeprefix("/api/demo/") in self.server.demo:
            name = path.removeprefix("/api/demo/")
            asset = self.server.demo[name]
            mime = "application/json; charset=utf-8"
            if name == "video":
                guessed = mimetypes.guess_type(asset.name)[0]
                mime = guessed if guessed and guessed.startswith("video/") else "application/octet-stream"
            self.file(asset, ranges=name == "video", mime=mime)
        else:
            self.error(HTTPStatus.NOT_FOUND, "This workbench route does not exist.")

    do_HEAD = do_GET

    def do_POST(self):
        if not self.same_origin():
            return
        target = urlsplit(self.path)
        if target.path not in {"/api/track", "/api/cancel"}:
            self.error(HTTPStatus.NOT_FOUND, "This workbench route does not exist.")
            return
        if self.headers.get("X-FBRPRO-Video") != "1":
            self.error(HTTPStatus.FORBIDDEN, "Submit the file from the local video workbench.")
            return
        if target.path == "/api/cancel":
            job = self.requested_job(target)
            if job:
                job.cancelled.set()
                self.send_json(HTTPStatus.ACCEPTED, {"stage": "cancelling"})
            return
        full = None
        try:
            fields = parse_qs(target.query, strict_parsing=True, max_num_fields=2)
            if fields.get("mode") == ["full"] and set(fields) <= {"mode", "job"}:
                full = FullVideoJob(job_id(fields) if "job" in fields else None)
            elif set(fields) != {"start_ms", "end_ms"} or any(
                    len(v) != 1 or not re.fullmatch(r"\d{1,8}", v[0]) for v in fields.values()):
                raise ValueError
            else:
                start, end = (int(fields[k][0]) for k in ("start_ms", "end_ms"))
                if not 0 <= start < end <= MAX_VIDEO_MS or end - start > MAX_WINDOW_MS:
                    raise ValueError
        except ValueError:
            self.error(HTTPStatus.BAD_REQUEST, "Choose a video window up to 15 seconds or full-video tracking.")
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not re.fullmatch(r"\d{1,15}", lengths[0]) or self.headers.get("Transfer-Encoding"):
            self.error(HTTPStatus.LENGTH_REQUIRED, "Send a video file with a known size.")
            return
        size = int(lengths[0])
        if not 0 < size <= (MAX_FULL_UPLOAD_BYTES if full else MAX_UPLOAD_BYTES):
            limit = "4 GiB" if full else "256 MiB"
            self.error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Choose a nonempty video file up to {limit}.")
            return
        if not self.server.job.acquire(blocking=False):
            self.error(HTTPStatus.TOO_MANY_REQUESTS, "Tracking is already running. Try again when it finishes.")
            return
        try:
            self.server.active_job = full
            with TemporaryDirectory(prefix="fbrpro-video-") as work:
                directory = Path(work)
                if full and disk_usage(directory).free < size + DISK_RESERVE_BYTES:
                    self.error(HTTPStatus.INSUFFICIENT_STORAGE, "There is not enough free temporary space for this video.")
                    return
                source = directory / "upload.video"
                with source.open("wb") as output:
                    while size:
                        if full:
                            full.progress()
                        block = self.rfile.read(min(size, CHUNK_SIZE))
                        if not block:
                            raise ValueError("The video upload ended early.")
                        output.write(block)
                        size -= len(block)
                scope = "full_video" if full else "video_preview"
                options = {}
                if full:
                    full.progress()
                    full.total_ms = estimate_duration_ms(source)
                    if full.total_ms is not None and full.total_ms > MAX_VIDEO_MS:
                        full.total_ms = None  # The decoder's timestamps govern the actual limit.
                    start, end = 0, full.total_ms or MAX_VIDEO_MS
                    full.stage = "tracking"
                    options = {"through_eof": True, "on_progress": full.progress}
                plan = preview_plan(start, end, scope)
                report = video_tracking.run(source, plan, self.server.model, {
                    "coordinate_length_m": 105, "coordinate_width_m": 68,
                    "dimensions_assumed": True, "segments": [],
                }, confidence=.23, **options)
                source.unlink()
                if full:
                    actual_end = report["clips"][0]["video_end_ms"]
                    if not 0 < actual_end <= MAX_VIDEO_MS:
                        raise ValueError("Full-video tracking supports footage up to six hours.")
                    plan["selected_video_ms"] = actual_end
                    for window in (plan["clips"][0], plan["alignment_segments"][0]):
                        window.update(video_end_ms=actual_end, match_end_ms=actual_end)
                    full.total_ms = actual_end
                    full.stage = "preparing_result"
                report["preview_plan"] = plan
                report["analysis_scope"] = scope
                if full:
                    self.full_report(directory, report, full)
                else:
                    self.send_json(HTTPStatus.OK, report)
        except (BrokenPipeError, ConnectionResetError):
            return
        except TimeoutError:
            self.error(HTTPStatus.REQUEST_TIMEOUT, "The video upload timed out.")
        except JobCancelled as exc:
            self.error(HTTPStatus.CONFLICT, str(exc))
        except ValueError as exc:
            self.error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except Exception as exc:
            self.log_error("Tracking failed: %s", exc)
            self.error(HTTPStatus.INTERNAL_SERVER_ERROR, "Local tracking could not complete.")
        finally:
            self.server.active_job = None
            self.server.job.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", required=True, type=Path, help="OpenCV Zoo YOLOX-s ONNX weights")
    parser.add_argument("--demo-video", type=Path)
    parser.add_argument("--demo-tracking", type=Path)
    parser.add_argument("--demo-plan", type=Path)
    args = parser.parse_args(argv)
    demo = {name: value for name in ("video", "tracking", "plan")
            if (value := getattr(args, "demo_" + name)) is not None}
    if demo and not {"video", "tracking"} <= demo.keys():
        parser.error("Supply --demo-video and --demo-tracking together.")
    if not 1 <= args.port <= 65535:
        parser.error("Choose a port between 1 and 65535.")
    if not all(path.is_file() for path in (args.model, *demo.values())):
        parser.error("The model and any supplied demo paths must be readable files.")
    if not (WEB_ROOT / STATIC["/video/"]).is_file():
        parser.error("Run the workbench from a project checkout with the video and dashboard pages.")
    try:
        video_tracking.vision()
        with ReviewServer(args.port, args.model, demo) as server:
            print(f"Open http://127.0.0.1:{server.server_port}/video/", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        parser.exit(2, f"video workbench: {exc}\n")


if __name__ == "__main__":
    main()
