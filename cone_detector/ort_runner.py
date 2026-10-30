"""ONNX Runtime runner: CUDA execution provider when it works, CPU otherwise.

Pascal GPUs (e.g. GTX 1050) need a CUDA 12 build and cuDNN <= 9.7, newer ones dropped sm_61:
    pip install "onnxruntime-gpu[cuda,cudnn]==1.23.2" "nvidia-cudnn-cu12==9.7.1.26"
"""
import numpy as np
import onnxruntime as ort


class OrtRunner:
    def __init__(self, model_path, device_id=0):
        ort.set_default_logger_severity(3)  # errors only; device discovery warnings are noise
        if hasattr(ort, 'preload_dlls'):  # CUDA/cuDNN from the nvidia-* pip wheels, if present
            ort.preload_dlls()
        self.cuda_error = None
        self.session = None
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            try:
                self.load(model_path, [('CUDAExecutionProvider', {'device_id': device_id})])
            except Exception as e:  # missing libs or kernels for this GPU arch
                self.cuda_error = str(e).splitlines()[0]
                self.session = None
        if self.session is None:
            self.load(model_path, ['CPUExecutionProvider'])
        self.provider = self.session.get_providers()[0]

    def load(self, model_path, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = tuple(inp.shape)
        # Warm up: unsupported GPUs often only fail on the first run, not at load.
        self(np.zeros(inp.shape, np.float32))

    def __call__(self, x):
        return self.session.run(None, {self.input_name: x})[0]
