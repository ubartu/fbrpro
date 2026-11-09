"""Local ONNX inference with an optional Core ML accelerator."""
from pathlib import Path


BACKENDS = ("auto", "opencv", "coreml")


class ModelInference:
    def __init__(self, cv2, path, backend, name):
        if backend not in BACKENDS:
            raise ValueError("Choose auto, opencv, or coreml for inference.")
        self.cv2, self.path, self.name = cv2, path, name
        self.requested = backend
        self.fallback = None
        if backend == "opencv":
            self.open_opencv()
        else:
            try:
                self.open_coreml()
            except Exception as exc:
                if backend == "coreml":
                    raise ValueError(f"Core ML could not load {name}: {exc}") from exc
                self.fallback = type(exc).__name__
                self.open_opencv()

    def open_opencv(self):
        cv2 = self.cv2
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise ValueError(f"OpenCV could not load {self.name}.") from exc
        self.backend = "opencv"

    def open_coreml(self):
        import onnxruntime as ort

        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Install an ONNX Runtime build with Core ML support.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram",
                      "MLComputeUnits": "CPUAndGPU", "RequireStaticInputShapes": "1"}),
                     "CPUExecutionProvider"]
        session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers)
        if "CoreMLExecutionProvider" not in session.get_providers():
            raise RuntimeError("Core ML was not enabled for this model.")
        self.input_name = session.get_inputs()[0].name
        self.net = session
        self.backend = "coreml"

    def forward(self, blob):
        if self.backend == "coreml":
            try:
                return self.net.run(None, {self.input_name: blob})[0]
            except Exception as exc:
                if self.requested == "coreml":
                    raise ValueError(f"Core ML inference failed for {self.name}: {exc}") from exc
                self.fallback = type(exc).__name__
                self.open_opencv()
        self.net.setInput(blob)
        return self.net.forward()

    def description(self):
        return {"backend": self.backend, "fallback": self.fallback}


def load_inference(cv2, path, backend, name, cache=None):
    if cache is None:
        return ModelInference(cv2, path, backend, name)
    path = Path(path).resolve()
    stat = path.stat()
    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    key = (path, backend)
    entry = cache.get(key)
    if entry is None or entry[0] != signature:
        entry = (signature, ModelInference(cv2, path, backend, name))
        cache[key] = entry
    return entry[1]
