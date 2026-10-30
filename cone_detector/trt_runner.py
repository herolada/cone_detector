"""Minimal TensorRT engine runner.

GPU memory is managed through libcudart via ctypes, so no pycuda / cuda-python is needed.
"""
import ctypes
import ctypes.util

import numpy as np
import tensorrt as trt

H2D, D2H = 1, 2  # cudaMemcpyKind


def load_cudart():
    for name in (ctypes.util.find_library('cudart'), 'libcudart.so',
                 '/usr/local/cuda/lib64/libcudart.so'):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        p = ctypes.c_void_p
        lib.cudaMalloc.argtypes = [ctypes.POINTER(p), ctypes.c_size_t]
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(p)]
        lib.cudaStreamSynchronize.argtypes = [p]
        lib.cudaMemcpyAsync.argtypes = [p, p, ctypes.c_size_t, ctypes.c_int, p]
        return lib
    raise OSError('libcudart not found')


class TrtRunner:
    def __init__(self, engine_path):
        self.cudart = load_cudart()
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to load TensorRT engine {engine_path}')
        self.context = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        self.check(self.cudart.cudaStreamCreate(ctypes.byref(self.stream)))

        # One preallocated host + device buffer per I/O tensor (shapes are static).
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            host = np.empty(tuple(self.engine.get_tensor_shape(name)),
                            trt.nptype(self.engine.get_tensor_dtype(name)))
            dev = ctypes.c_void_p()
            self.check(self.cudart.cudaMalloc(ctypes.byref(dev), host.nbytes))
            self.context.set_tensor_address(name, dev.value)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((host, dev))
            else:
                self.outputs.append((host, dev))
        self.input_shape = self.inputs[0][0].shape

    @staticmethod
    def check(err):
        if err != 0:
            raise RuntimeError(f'CUDA error {err}')

    def memcpy(self, dst, src, nbytes, kind):
        self.check(self.cudart.cudaMemcpyAsync(dst, src, nbytes, kind, self.stream))

    def __call__(self, x):
        host, dev = self.inputs[0]
        np.copyto(host, x)
        self.memcpy(dev, host.ctypes.data, host.nbytes, H2D)
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError('TensorRT inference failed')
        host, dev = self.outputs[0]
        self.memcpy(host.ctypes.data, dev, host.nbytes, D2H)
        self.check(self.cudart.cudaStreamSynchronize(self.stream))
        return host
