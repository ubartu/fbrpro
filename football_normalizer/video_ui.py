"""Serve the video review workbench and run short, local tracking previews."""
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

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
MAX_WINDOW_MS = 15_000
MAX_VIDEO_MS = 6 * 60 * 60 * 1000
CHUNK_SIZE = 1024 * 1024


def preview_plan(start, end):
    key = uuid4().hex
    window = {"video_start_ms": start, "video_end_ms": end,
              "match_start_ms": start, "match_end_ms": end, "period": 1}
    clip = {**window, "id": "preview-clip", "segment_id": "preview-segment", "opportunities": []}
    return {"version": video.VERSION, "plan_id": f"preview-plan-{key}",
            "video_id": f"preview-video-{key}", "match_id": f"local-preview-{key}",
            "analysis_scope": "video_preview", "clock_source": "video_only_unverified",
            "players": [], "counts": {"selected": {}}, "unaligned": [],
            "selected_video_ms": end - start, "clips": [clip],
            "alignment_segments": [{**window, "id": "preview-segment", "kind": "preview"}]}


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, port, model, demo=None):
        super().__init__(("127.0.0.1", port), ReviewHandler)
        self.model = model
        self.demo = demo or {}
        self.job = Lock()
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
        path = urlsplit(self.path).path
        if path in {"/", "/video", "/dashboard"}:
            self.send_headers(HTTPStatus.FOUND, "text/plain", 0,
                              Location="/dashboard/" if path == "/dashboard" else "/video/")
        elif path in STATIC:
            self.file(WEB_ROOT / STATIC[path])
        elif path == "/api/status":
            demo = self.server.demo
            self.send_json(HTTPStatus.OK, {"demo_available": "video" in demo and "tracking" in demo,
                                           "demo_plan_available": "plan" in demo,
                                           "max_upload_bytes": MAX_UPLOAD_BYTES, "max_window_ms": MAX_WINDOW_MS})
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
        if target.path != "/api/track":
            self.error(HTTPStatus.NOT_FOUND, "This workbench route does not exist.")
            return
        if self.headers.get("X-FBRPRO-Video") != "1":
            self.error(HTTPStatus.FORBIDDEN, "Submit the file from the local video workbench.")
            return
        try:
            fields = parse_qs(target.query, strict_parsing=True, max_num_fields=2)
            if set(fields) != {"start_ms", "end_ms"} or any(
                    len(v) != 1 or not re.fullmatch(r"\d{1,8}", v[0]) for v in fields.values()):
                raise ValueError
            start, end = (int(fields[k][0]) for k in ("start_ms", "end_ms"))
            if not 0 <= start < end <= MAX_VIDEO_MS or end - start > MAX_WINDOW_MS:
                raise ValueError
        except ValueError:
            self.error(HTTPStatus.BAD_REQUEST, "Choose a positive video window of up to 15 seconds.")
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not re.fullmatch(r"\d{1,15}", lengths[0]) or self.headers.get("Transfer-Encoding"):
            self.error(HTTPStatus.LENGTH_REQUIRED, "Send a video file with a known size.")
            return
        size = int(lengths[0])
        if not 0 < size <= MAX_UPLOAD_BYTES:
            self.error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Choose a video file up to 256 MB.")
            return
        if not self.server.job.acquire(blocking=False):
            self.error(HTTPStatus.TOO_MANY_REQUESTS, "A preview is running. Try again when it finishes.")
            return
        try:
            with TemporaryDirectory(prefix="fbrpro-preview-") as work:
                source = Path(work) / "upload.video"
                with source.open("wb") as output:
                    while size:
                        block = self.rfile.read(min(size, CHUNK_SIZE))
                        if not block:
                            raise ValueError("The video upload ended early.")
                        output.write(block)
                        size -= len(block)
                plan = preview_plan(start, end)
                report = video_tracking.run(source, plan, self.server.model, {
                    "coordinate_length_m": 105, "coordinate_width_m": 68,
                    "dimensions_assumed": True, "segments": [],
                }, confidence=.23)
            report["preview_plan"] = plan
            report["analysis_scope"] = "video_preview"
            self.send_json(HTTPStatus.OK, report)
        except (BrokenPipeError, ConnectionResetError):
            return
        except TimeoutError:
            self.error(HTTPStatus.REQUEST_TIMEOUT, "The video upload timed out.")
        except ValueError as exc:
            self.error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except Exception as exc:
            self.log_error("Preview failed: %s", exc)
            self.error(HTTPStatus.INTERNAL_SERVER_ERROR, "The local tracking preview could not complete.")
        finally:
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
