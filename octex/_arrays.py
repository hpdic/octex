# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Array handling without framework dependencies: inputs from numpy, CuPy, PyTorch, or any array with DLPack or
__cuda_array_interface__ go to the CUDA core without copies, and outputs come back in the framework of the input.

numpy is the only required dependency. CuPy and PyTorch are used only when the caller passes their arrays, and then
only through the module that the array itself belongs to (sys.modules), so that importing octex imports neither.
"""

import sys

import numpy as np

_KDL_CPU, _KDL_CUDA, _KDL_CUDA_HOST, _KDL_CUDA_MANAGED = 1, 2, 3, 13


def _module(a):
    return type(a).__module__.partition(".")[0]


def _typestr(dtype):
    return np.dtype(dtype).name  # "float32", "int64", ...


class Batch:
    """An input matrix ready for the core (`array`), with what is needed to answer in its framework."""

    def __init__(self, kind, array, original, device=None, stream=0, shape=None):
        self.kind = kind  # "numpy", "torch-cpu", "torch", "cupy", "cai", "dlpack"
        self.array = array  # what _core receives (anything nanobind takes as an ndarray)
        self.original = original
        self.device = device  # CUDA device id for device arrays
        self.stream = stream  # CUDA stream the input was produced on
        self._shape = tuple(int(n) for n in (shape if shape is not None else array.shape))

    @property
    def on_device(self):
        return self.kind not in ("numpy", "torch-cpu")

    @property
    def shape(self):
        return self._shape

    def empty(self, rows, cols, dtype):
        """An output matrix in the framework of the input; returns (user_array, core_array)."""
        if self.kind == "numpy":
            out = np.empty((rows, cols), dtype=dtype)
            return out, out
        if self.kind == "torch-cpu":
            torch = sys.modules["torch"]
            out = np.empty((rows, cols), dtype=dtype)
            return torch.from_numpy(out), out
        if self.kind == "torch":
            torch = sys.modules["torch"]
            out = torch.empty((rows, cols), dtype=getattr(torch, _typestr(dtype)), device=self.original.device)
            return out, out
        if self.kind == "cupy":
            cp = sys.modules["cupy"]
            with cp.cuda.Device(self.device):
                out = cp.empty((rows, cols), dtype=dtype)
            return out, out
        if self.kind == "cai" and _module(self.original) == "pylibraft":
            from pylibraft.common import device_ndarray

            out = device_ndarray.empty((rows, cols), dtype=dtype)
            return out, _wrap_cai(out)
        from . import _core

        out = _core.empty(rows, cols, _typestr(dtype), self.device or 0)
        return out, out


def _wrap_cai(a, dtype=None):
    """A nanobind ndarray over the memory of an object with __cuda_array_interface__ (no copy)."""
    from . import _core

    cai = a.__cuda_array_interface__
    shape = tuple(cai["shape"])
    if len(shape) != 2:
        raise ValueError("octex: expected a matrix [n, dim]")
    dt = np.dtype(cai["typestr"])
    strides = cai.get("strides")
    contiguous = strides is None or tuple(strides) == (shape[1] * dt.itemsize, dt.itemsize) or shape[0] <= 1
    if not contiguous or (dtype is not None and dt != np.dtype(dtype)):
        return None
    return _core.wrap(int(cai["data"][0]), shape[0], shape[1], dt.name, a)


def _cai_stream(a):
    s = a.__cuda_array_interface__.get("stream")
    return int(s) if s else 0


def batch(a, dtypes=(np.float32,), name="queries"):
    """Prepare a matrix for the core; arrays of other dtypes are converted to dtypes[0] in their own framework."""
    want = [np.dtype(t) for t in dtypes]
    mod = _module(a)
    if isinstance(a, np.ndarray) or (mod != "torch" and mod != "cupy" and not _is_device(a)):
        x = np.asarray(a)
        if x.ndim == 1:
            x = x[None, :]
        if x.dtype not in want:
            x = x.astype(want[0])
        return Batch("numpy", np.ascontiguousarray(x), a)
    if mod == "torch":
        torch = sys.modules["torch"]
        t = a.detach()
        if t.ndim == 1:
            t = t[None, :]
        if _torch_dtype(t) is None or _torch_dtype(t) not in want:
            t = t.to(getattr(torch, want[0].name))
        t = t.contiguous()
        if t.device.type == "cuda":
            return Batch("torch", t, a, device=t.device.index, stream=torch.cuda.current_stream(t.device).cuda_stream)
        if t.device.type != "cpu":
            raise TypeError(f"octex: {name} must live in host or CUDA device memory, not on {t.device}")
        return Batch("torch-cpu", t.numpy(), a)
    if mod == "cupy":
        cp = sys.modules["cupy"]
        x = a if a.ndim == 2 else a.reshape(1, -1)
        if x.dtype not in want:
            x = x.astype(want[0])
        x = cp.ascontiguousarray(x)
        return Batch("cupy", x, a, device=x.device.id, stream=cp.cuda.get_current_stream().ptr)
    if hasattr(a, "__cuda_array_interface__"):
        w = _wrap_cai(a)
        if w is None or np.dtype(a.__cuda_array_interface__["typestr"]) not in want:
            cp = _cupy(name)
            x = cp.ascontiguousarray(cp.asarray(a), dtype=want[0])
            return Batch("cupy", x, a, device=x.device.id, stream=cp.cuda.get_current_stream().ptr)
        return Batch(
            "cai",
            w,
            a,
            device=w.__dlpack_device__()[1],
            stream=_cai_stream(a),
            shape=a.__cuda_array_interface__["shape"],
        )
    # an array that offers only DLPack, on a GPU
    from . import _core

    device = a.__dlpack_device__()[1]
    return Batch("dlpack", a, a, device=device, stream=0, shape=_core.shape(a))


def _torch_dtype(t):
    """The numpy dtype of a torch tensor, or None (bfloat16 and other types numpy lacks)."""
    try:
        return np.dtype(str(t.dtype).removeprefix("torch."))
    except TypeError:
        return None


def _is_device(a):
    if hasattr(a, "__cuda_array_interface__"):
        return True
    if hasattr(a, "__dlpack_device__"):
        return a.__dlpack_device__()[0] in (_KDL_CUDA, _KDL_CUDA_MANAGED)
    return False


def _cupy(name):
    try:
        import cupy

        return cupy
    except ImportError as e:
        raise TypeError(f"octex: {name} must be C-contiguous float32 (CuPy converts other layouts)") from e


def to_host(a, dtypes=(np.float32,)):
    """A C-contiguous host numpy copy or view of any matrix (for accumulating Faiss-style adds)."""
    mod = _module(a)
    want = [np.dtype(t) for t in dtypes]
    if mod == "torch":
        torch = sys.modules["torch"]
        t = a.detach()
        if _torch_dtype(t) is None or _torch_dtype(t) not in want:
            t = t.to(getattr(torch, want[0].name))
        x = t.cpu().numpy()
    elif mod == "cupy":
        x = a.get()
    elif hasattr(a, "__cuda_array_interface__"):
        x = _cupy("vectors").asarray(a).get()
    elif hasattr(a, "__dlpack_device__") and a.__dlpack_device__()[0] != _KDL_CPU:
        x = _cupy("vectors").from_dlpack(a).get()
    else:
        x = np.asarray(a)
    if x.ndim == 1:
        x = x[None, :]
    if x.dtype not in want:
        x = x.astype(want[0])
    return np.ascontiguousarray(x)


def copy_into(dst, src):
    """dst[...] = src across frameworks (casts dtypes); used for caller-supplied output arrays."""
    dmod, smod = _module(dst), _module(src)
    if isinstance(dst, np.ndarray):
        dst[...] = to_host(src, (dst.dtype,))
    elif dmod == "torch":
        torch = sys.modules["torch"]
        if smod == "torch":
            dst.copy_(src)
        elif isinstance(src, np.ndarray):
            dst.copy_(torch.from_numpy(src))
        else:
            dst.copy_(torch.from_dlpack(src))
    else:
        cp = _cupy("outputs")
        view = cp.asarray(dst) if dmod != "cupy" else dst
        if isinstance(src, np.ndarray):
            view[...] = cp.asarray(src)
        elif smod == "torch" or hasattr(src, "__cuda_array_interface__"):
            view[...] = cp.asarray(src)
        else:
            view[...] = cp.from_dlpack(src)
    return dst
