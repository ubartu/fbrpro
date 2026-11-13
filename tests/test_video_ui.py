from http.client import parse_headers
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

from football_normalizer import video_ui


class InMemoryHandler(video_ui.ReviewHandler):
    def setup(self):
        self.rfile = BytesIO(self.request)
        self.wfile = BytesIO()

    def finish(self):
        pass

    def log_message(self, *args):
        pass


class ReviewServerTests(unittest.TestCase):
    def test_statistics_are_separate_read_only_requests_with_origin_guard(self):
        from test_video_statistics import tracking
        data = json.dumps({"tracking": tracking(), "context": {}}).encode()
        headers = {"X-FBRPRO-Video": "1", "Content-Type": "application/json"}
        with patch.object(video_ui.video_tracking, "run") as detector:
            status, _, body = self.call("POST", "/api/statistics", headers=headers, body=data)
        detector.assert_not_called()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["version"], "fbrpro-video-stats-v1")
        self.assertEqual(self.call("POST", "/api/statistics", body=data)[0], 403)
        self.assertEqual(self.call("POST", "/api/statistics", headers=headers, body=b"{}")[0], 400)
        self.server.job.acquire()
        try:
            self.assertEqual(self.call("POST", "/api/statistics", headers=headers, body=data)[0], 409)
        finally:
            self.server.job.release()

    def test_geometry_review_endpoint_recalculates_without_running_detector(self):
        tracking = {"version": "fbrpro-video-tracking-v1", "coordinate_length_m": 105,
                    "coordinate_width_m": 68, "frames": []}
        data = json.dumps({"tracking": tracking, "review": {}}).encode()
        headers = {"X-FBRPRO-Video": "1", "Content-Type": "application/json"}
        with patch.object(video_ui.video_tracking, "run") as run:
            status, _, body = self.call("POST", "/api/geometry", headers=headers, body=data)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["spatial_summary"]["players"], [])
            run.assert_not_called()
        self.assertEqual(self.call("POST", "/api/geometry", body=data)[0], 403)
        self.assertEqual(self.call("POST", "/api/geometry", headers=headers, body=b"{}")[0], 400)
        self.assertEqual(self.call("GET", "/video/geometry.js")[0], 404)
        status, _, body = self.call("GET", "/api/status")
        self.assertEqual(json.loads(body)["tracking_pipeline"], "stable-0244d67-recovery")

    def test_event_analysis_aligns_and_attributes_without_running_detector(self):
        from test_video_events import ALIGN, predictions, tracking

        data = json.dumps({"tracking": tracking(), "predictions": predictions(), "alignment": ALIGN}).encode()
        headers = {"X-FBRPRO-Video": "1", "Content-Type": "application/json"}
        with patch.object(video_ui.video_tracking, "run") as detector:
            status, _, body = self.call("POST", "/api/video-events", headers=headers, body=data)
        detector.assert_not_called()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["player_pass_attempts"], {"passer": 1})
        self.assertEqual(self.call("POST", "/api/video-events", body=data)[0], 403)

    def setUp(self):
        scratch = self.enterContext(TemporaryDirectory())
        video = Path(scratch) / "sample.mp4"
        video.write_bytes(b"0123456789")
        tracking = Path(scratch) / "tracking.json"
        tracking.write_text("{}")
        self.server = SimpleNamespace(
            hosts={"127.0.0.1:8765"}, model=Path(scratch) / "model.onnx", pitch_model=None,
            inference_backend="auto", inference_cache={}, ball_model=None,
            sample_fps=video_ui.video_tracking.DEFAULT_SAMPLE_FPS, pitch_lines_model=None,
            demo={"video": video, "tracking": tracking}, demo_calibration=None, job=Lock(), active_job=None)

    def call(self, method, path, *, headers=None, body=None):
        fields = {"Host": "127.0.0.1:8765"}
        if body is not None:
            fields["Content-Length"] = str(len(body))
        fields.update(headers or {})
        request = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields.items())
        handler = InMemoryHandler(request.encode("ascii") + b"\r\n" + (body or b""),
                                  ("127.0.0.1", 12345), self.server)
        header, response = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        status, raw_fields = header.split(b"\r\n", 1)
        return int(status.split()[1]), parse_headers(BytesIO(raw_fields + b"\r\n\r\n")), response

    def test_demo_discovery_ranges_and_routes_are_restricted(self):
        status, _, body = self.call("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["demo_available"])
        self.assertFalse(json.loads(body)["demo_plan_available"])
        self.assertEqual(json.loads(body)["max_full_upload_bytes"], video_ui.MAX_FULL_UPLOAD_BYTES)
        self.assertEqual(json.loads(body)["max_full_video_ms"], video_ui.MAX_VIDEO_MS)
        self.assertEqual(json.loads(body)["sample_fps"], 10)
        self.assertEqual(self.call("GET", "/api/demo/plan")[0], 404)
        status, headers, body = self.call("GET", "/api/demo/video", headers={"Range": "bytes=3-5"})
        self.assertEqual((status, headers["Content-Range"], body), (206, "bytes 3-5/10", b"345"))
        self.assertEqual(self.call("GET", "/api/demo/video", headers={"Range": "bytes=30-"})[0], 416)
        self.assertEqual(self.call("GET", "/football_normalizer/video_ui.py")[0], 404)
        self.assertEqual(self.call("GET", "/api/status", headers={"Host": "example.invalid"})[0], 403)
        self.assertEqual(self.call("GET", "/api/status", headers={"Origin": "https://example.invalid"})[0], 403)

    def test_upload_is_bounded_and_runs_without_fixture_evidence(self):
        path = "/api/track?start_ms=1000&end_ms=2500"
        guard = {"X-FBRPRO-Video": "1"}
        self.assertEqual(self.call("POST", path, body=b"x")[0], 403)
        self.assertEqual(self.call("POST", "/api/track?start_ms=0&end_ms=16000", headers=guard, body=b"x")[0], 400)
        self.assertEqual(self.call("POST", path, headers={
            **guard, "Content-Length": str(video_ui.MAX_UPLOAD_BYTES + 1)})[0], 413)

        captured = []

        def track(source, plan, model, calibration, **kwargs):
            self.assertEqual(kwargs["sample_fps"], 10)
            self.assertEqual(source.read_bytes(), b"raw video bytes")
            self.assertEqual(calibration["segments"], [])
            self.assertEqual(plan["alignment_segments"][0]["kind"], "preview")
            self.assertEqual(plan["clips"][0]["opportunities"], [])
            self.assertEqual(plan["players"], [])
            captured.append(source)
            return {"plan_id": plan["plan_id"], "frames": []}

        with patch.object(video_ui.video_tracking, "run", side_effect=track):
            status, _, body = self.call("POST", path, headers=guard, body=b"raw video bytes")
        self.assertEqual(status, 200)
        report = json.loads(body)
        self.assertEqual(report["analysis_scope"], "video_preview")
        self.assertEqual(report["preview_plan"]["plan_id"], report["plan_id"])
        self.assertEqual(report["preview_plan"]["clock_source"], "video_only_unverified")
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0].exists())

    def test_full_video_uses_decoded_end_and_reports_progress(self):
        job = uuid4()
        progress_url = f"/api/progress?job={job}"
        self.assertEqual(self.call("GET", progress_url)[0], 404)

        captured = []

        def track(source, plan, model, calibration, *, through_eof, on_progress, **kwargs):
            self.assertEqual(kwargs["sample_fps"], 10)
            self.assertTrue(through_eof)
            self.assertEqual(plan["analysis_scope"], "full_video")
            self.assertEqual(plan["selected_video_ms"], 20_000)
            self.assertEqual(plan["clips"][0]["video_start_ms"], 0)
            self.assertEqual(len(plan["clips"]), 1)
            self.assertEqual(source.read_bytes(), b"video")
            on_progress(22_000)
            status, _, body = self.call("GET", progress_url)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body),
                             {"processed_ms": 22_000, "total_ms": 20_000, "stage": "tracking"})
            captured.append(source)
            return {"plan_id": plan["plan_id"], "frames": [], "clips": [{"video_end_ms": 22_100}]}

        with (patch.object(video_ui, "disk_usage", return_value=SimpleNamespace(free=1 << 32)),
              patch.object(video_ui, "estimate_duration_ms", return_value=20_000),
              patch.object(video_ui.video_tracking, "run", side_effect=track)):
            status, _, body = self.call("POST", f"/api/track?mode=full&job={job}",
                                        headers={"X-FBRPRO-Video": "1"}, body=b"video")
        self.assertEqual(status, 200)
        report = json.loads(body)
        self.assertEqual(report["analysis_scope"], "full_video")
        plan = report["preview_plan"]
        self.assertEqual(plan["clock_source"], "video_only_unverified")
        self.assertEqual(plan["selected_video_ms"], 22_100)
        for window in (plan["clips"][0], plan["alignment_segments"][0]):
            self.assertEqual((window["video_end_ms"], window["match_end_ms"]), (22_100, 22_100))
        self.assertFalse(captured[0].exists())
        self.assertEqual(self.call("GET", progress_url)[0], 404)

    def test_calibration_is_reused_only_for_matching_video_or_explicit_upload(self):
        guard = {"X-FBRPRO-Video": "1"}
        demo_bytes = self.server.demo["video"].read_bytes()
        self.server.demo_size = len(demo_bytes)
        self.server.demo_hash = hashlib.sha256(demo_bytes).hexdigest()
        self.server.demo_calibration = {
            "coordinate_length_m": 105, "coordinate_width_m": 68,
            "segments": [{"segment_id": "source-shot", "video_start_ms": 0, "video_end_ms": 4000,
                          "follow_camera": True}],
        }
        calls = []

        def track(source, plan, model, calibration, **kwargs):
            calls.append(calibration)
            return {"frames": [], "clips": [{"video_end_ms": 30_000}]}

        with (patch.object(video_ui, "disk_usage", return_value=SimpleNamespace(free=1 << 32)),
              patch.object(video_ui, "estimate_duration_ms", return_value=30_000),
              patch.object(video_ui.video_tracking, "run", side_effect=track)):
            self.assertEqual(self.call("POST", "/api/track?mode=full", headers=guard, body=demo_bytes)[0], 200)
            self.assertEqual(self.call("POST", "/api/track?mode=full", headers=guard, body=b"abcdefghij")[0], 200)
            self.assertEqual(self.call("POST", "/api/track?start_ms=1000&end_ms=5000",
                                       headers=guard, body=demo_bytes)[0], 200)
            own = {"coordinate_length_m": 100, "coordinate_width_m": 60, "segments": [{"segment_id": "fixed"}]}
            headers = {**guard, "X-FBRPRO-Calibration": quote(json.dumps(own))}
            self.assertEqual(self.call("POST", "/api/track?mode=full", headers=headers, body=b"abcdefghij")[0], 200)
        self.assertEqual(calls[0]["segments"][0]["segment_id"], "preview-segment")
        self.assertEqual(calls[1]["segments"], [])
        self.assertEqual(calls[2]["segments"], [])
        self.assertEqual(calls[3]["coordinate_length_m"], 100)

    def test_full_video_rejects_limits_and_honors_cancellation(self):
        job = uuid4()
        upload = f"/api/track?mode=full&job={job}"
        cancel = f"/api/cancel?job={job}"
        guard = {"X-FBRPRO-Video": "1"}
        self.assertEqual(self.call("POST", upload, headers={
            **guard, "Content-Length": str(video_ui.MAX_FULL_UPLOAD_BYTES + 1)})[0], 413)
        with patch.object(video_ui, "disk_usage", return_value=SimpleNamespace(free=0)):
            self.assertEqual(self.call("POST", upload, headers=guard, body=b"video")[0], 507)

        def track(*args, on_progress, **kwargs):
            self.assertEqual(self.call("POST", cancel)[0], 403)
            self.assertEqual(self.call("POST", cancel, headers=guard)[0], 202)
            on_progress(1_000)

        with (patch.object(video_ui, "disk_usage", return_value=SimpleNamespace(free=1 << 32)),
              patch.object(video_ui, "estimate_duration_ms", return_value=None),
              patch.object(video_ui.video_tracking, "run", side_effect=track)):
            status, _, body = self.call("POST", upload, headers=guard, body=b"video")
        self.assertEqual(status, 409)
        self.assertIn("cancelled", json.loads(body)["error"])
        self.assertIsNone(self.server.active_job)


if __name__ == "__main__":
    unittest.main()
