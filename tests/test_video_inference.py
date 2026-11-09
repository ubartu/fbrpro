from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from football_normalizer.video_inference import ModelInference, load_inference


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.cpu = Mock()
        self.cpu.forward.return_value = "result"
        self.cv2 = SimpleNamespace(dnn=SimpleNamespace(
            readNetFromONNX=Mock(return_value=self.cpu), DNN_BACKEND_OPENCV=1, DNN_TARGET_CPU=0),
            error=RuntimeError)

    def test_auto_falls_back_when_coreml_is_unavailable(self):
        with patch.object(ModelInference, "open_coreml", side_effect=ImportError):
            model = ModelInference(self.cv2, "weights.onnx", "auto", "player")
            self.assertEqual(model.forward("tensor"), "result")
            self.assertEqual(model.description(), {"backend": "opencv", "fallback": "ImportError"})
            with self.assertRaisesRegex(ValueError, "Core ML could not load player"):
                ModelInference(self.cv2, "weights.onnx", "coreml", "player")

    def test_auto_switches_to_opencv_when_coreml_fails_during_inference(self):
        def open_coreml(model):
            model.net = Mock(run=Mock(side_effect=RuntimeError))
            model.input_name = "input"
            model.backend = "coreml"

        with patch.object(ModelInference, "open_coreml", open_coreml):
            model = ModelInference(self.cv2, "weights.onnx", "auto", "pitch")
        self.assertEqual(model.forward("tensor"), "result")
        self.assertEqual(model.forward("next"), "result")
        self.assertEqual(model.description(), {"backend": "opencv", "fallback": "RuntimeError"})
        self.cv2.dnn.readNetFromONNX.assert_called_once()
        self.cpu.setInput.assert_called_with("next")

    def test_cache_reuses_loaded_weights_and_reloads_changed_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.onnx"
            path.write_bytes(b"first")
            cache = {}
            first = load_inference(self.cv2, path, "opencv", "player", cache)
            self.assertIs(load_inference(self.cv2, path, "opencv", "player", cache), first)
            path.write_bytes(b"replacement")
            self.assertIsNot(load_inference(self.cv2, path, "opencv", "player", cache), first)
        self.assertEqual(self.cv2.dnn.readNetFromONNX.call_count, 2)
