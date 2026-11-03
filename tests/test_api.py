# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Shapes, dtypes, and frameworks of the results for numpy, CuPy, PyTorch, __cuda_array_interface__ and DLPack
inputs, and the errors of invalid calls.

    python -m pytest tests/test_api.py
"""

import numpy as np
import pytest

import octex
from octex import cuvs as cagra
from octex import faiss

pytestmark = pytest.mark.gpu
K = 10


def check(D, I, m=100, k=K):
    D, I = np.asarray(D), np.asarray(I)
    assert D.shape == (m, k) and I.shape == (m, k)
    assert D.dtype == np.float32 and I.dtype == np.int64
    assert (np.diff(D, axis=1) >= -1e-3 * np.abs(D[:, 1:]) - 1e-3).all(), "distances not ascending"
    assert ((I >= 0) & (I < 1_000_000)).all()


def test_properties(light):
    assert light.size == len(light) == 1_000_000
    assert light.dim == 128
    assert light.copies == 16
    assert light.devices == [0]
    assert len(light.device_bytes) == 1 and light.device_bytes[0] > 0
    assert "octex.Index" in repr(light)


def test_numpy(light, sift_queries):
    D, I = light.search(sift_queries[:100], k=K)
    assert isinstance(D, np.ndarray) and isinstance(I, np.ndarray)
    check(D, I)


def test_host_conversions(light, sift_queries):
    D, I = light.search(sift_queries[:100], k=K)
    D64, I64 = light.search(sift_queries[:100].astype(np.float64), k=K)  # converted to float32
    assert np.array_equal(I, I64)
    Dl, Il = light.search(sift_queries[:3].tolist(), k=K)
    assert np.array_equal(Il, I[:3])
    D1, I1 = light.search(sift_queries[0], k=K)  # one vector: a batch of one
    assert I1.shape == (1, K) and np.array_equal(I1[0], I[0])
    Ds, Is = light.search(np.asfortranarray(sift_queries[:100]), k=K)
    assert np.array_equal(Is, I)


def test_cupy(light, sift_queries):
    cp = pytest.importorskip("cupy")
    D_ref, I_ref = light.search(sift_queries[:100], k=K)
    q = cp.asarray(sift_queries[:100])
    D, I = light.search(q, k=K)
    assert isinstance(D, cp.ndarray) and isinstance(I, cp.ndarray)
    assert D.device.id == q.device.id
    check(D.get(), I.get())
    assert np.array_equal(I.get(), I_ref)
    assert np.allclose(D.get(), D_ref, rtol=1e-5, atol=1e-3)
    D16, I16 = light.search(q.astype(cp.float16), k=K)  # converted to float32 in CuPy
    assert I16.shape == (100, K)


def test_cupy_outputs_in_place(light, sift_queries):
    cp = pytest.importorskip("cupy")
    q = cp.asarray(sift_queries[:100])
    I_out, D_out = cp.empty((100, K), cp.int64), cp.empty((100, K), cp.float32)
    D, I = light.search(q, k=K, neighbors=I_out, distances=D_out)
    assert D is D_out and I is I_out
    I32 = cp.empty((100, K), cp.uint32)  # the CAGRA neighbor dtype: cast into it
    _, I2 = light.search(q, k=K, neighbors=I32)
    assert I2 is I32 and cp.array_equal(I32.astype(cp.int64), I_out)
    host_I = np.empty((100, K), np.int64)
    _, I3 = light.search(sift_queries[:100], k=K, neighbors=host_I)
    assert I3 is host_I and np.array_equal(host_I, I_out.get())


def test_torch(light, sift_queries):
    torch = pytest.importorskip("torch")
    D_ref, I_ref = light.search(sift_queries[:100], k=K)
    q = torch.from_numpy(sift_queries[:100]).cuda()
    D, I = light.search(q, k=K)
    assert isinstance(D, torch.Tensor) and D.is_cuda and D.device == q.device
    assert D.dtype == torch.float32 and I.dtype == torch.int64
    assert np.array_equal(I.cpu().numpy(), I_ref)
    Dc, Ic = light.search(torch.from_numpy(sift_queries[:100]), k=K)  # CPU tensors in, CPU tensors out
    assert isinstance(Ic, torch.Tensor) and not Ic.is_cuda and np.array_equal(Ic.numpy(), I_ref)
    Dh, Ih = light.search(q.half(), k=K)
    assert Ih.shape == (100, K)


def test_cuda_array_interface(light, sift_queries):
    pytest.importorskip("pylibraft")
    from pylibraft.common import device_ndarray

    D_ref, I_ref = light.search(sift_queries[:100], k=K)
    D, I = light.search(device_ndarray(sift_queries[:100]), k=K)
    assert isinstance(D, device_ndarray) and isinstance(I, device_ndarray)
    assert np.array_equal(I.copy_to_host(), I_ref)


class DLPackOnly:
    """A device array that offers nothing but DLPack."""

    def __init__(self, a):
        self._a = a

    def __dlpack__(self, **kw):
        return self._a.__dlpack__(**kw)

    def __dlpack_device__(self):
        return self._a.__dlpack_device__()


def test_dlpack(light, sift_queries):
    cp = pytest.importorskip("cupy")
    D_ref, I_ref = light.search(sift_queries[:100], k=K)
    D, I = light.search(DLPackOnly(cp.asarray(sift_queries[:100])), k=K)
    assert hasattr(I, "__dlpack__")
    assert np.array_equal(cp.from_dlpack(I).get(), I_ref)
    assert np.allclose(cp.from_dlpack(D).get(), D_ref, rtol=1e-5, atol=1e-3)


def test_search_params(light, sift_queries):
    D, I = light.search(sift_queries[:100], k=5, read=8, starts=32, waves=0)
    check(D, I, k=5)
    D, I = light.search(sift_queries[:100], k=K, batch=37)  # batches split and padded
    check(D, I)
    D0, I0 = light.search(sift_queries[:0], k=K)
    assert I0.shape == (0, K)


def test_errors(light, sift_queries, sift_path):
    with pytest.raises(ValueError):
        light.search(sift_queries[:10, :64], k=K)  # wrong dimension
    with pytest.raises(ValueError):
        light.search(sift_queries[:10], k=0)
    with pytest.raises(TypeError):
        light.search(sift_queries[:10], k=K, bogus=1)
    with pytest.raises(ValueError):
        octex.build(sift_queries, metric="inner_product")
    with pytest.raises(ValueError):
        cagra.IndexParams(metric="cosine")
    with pytest.raises(ValueError):
        faiss.IndexOctex(128, faiss.METRIC_INNER_PRODUCT)
    with pytest.raises(TypeError):
        octex.build(sift_queries, not_a_parameter=3)
    with pytest.raises(FileNotFoundError):
        octex.load(sift_path + "/does-not-exist")
    with pytest.raises(ValueError):
        octex.load(sift_path, placement="mirror")
    with pytest.raises(ValueError):
        cagra.search(cagra.SearchParams(), cagra.Index(), sift_queries[:10], K)  # empty index
