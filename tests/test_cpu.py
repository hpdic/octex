# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Checks that need neither a GPU nor the CUDA extension: the package imports, the adapters reject unsupported
metrics and unknown parameters, and host arrays are prepared and copied correctly.

    python -m pytest tests/test_cpu.py
"""

import numpy as np
import pytest

import octex
from octex import _arrays, faiss
from octex import cuvs as cagra


def test_version():
    assert octex.__version__ == "0.1.0"


def test_metrics():
    for metric in ("sqeuclidean", "l2", "L2"):
        octex.index.check_metric(metric)
        cagra.IndexParams(metric=metric)
    for metric in ("inner_product", "cosine", "euclidean"):
        with pytest.raises(ValueError):
            cagra.IndexParams(metric=metric)
    with pytest.raises(ValueError):
        faiss.IndexOctex(16, faiss.METRIC_INNER_PRODUCT)
    with pytest.raises(ValueError):
        faiss.GpuIndexCagra(faiss.StandardGpuResources(), 16, faiss.METRIC_INNER_PRODUCT)


def test_adapter_parameters():
    p = cagra.IndexParams(copies=32, graph_degree=64, placement="replicate")  # CAGRA fields are ignored
    assert p.params == {"copies": 32} and p.load_args == {"placement": "replicate"}
    s = cagra.SearchParams(max_queries=500, itopk_size=64, read=8)
    assert s.params == {"read": 8, "batch": 500}
    with pytest.raises(TypeError):
        cagra.IndexParams(not_a_parameter=1)
    with pytest.raises(TypeError):
        cagra.SearchParams(not_a_parameter=1)
    with pytest.raises(TypeError):
        faiss.IndexOctex(16, not_a_parameter=1)
    index = faiss.IndexOctex(16)
    index.add(np.zeros((5, 16), np.float64))
    index.add(np.zeros((3, 16), np.float32))
    assert index.ntotal == 8 and index.d == 16 and index.is_trained
    with pytest.raises(ValueError):
        index.add(np.zeros((3, 8), np.float32))
    index.reset()
    assert index.ntotal == 0


def test_host_batches():
    b = _arrays.batch(np.arange(12, dtype=np.float64).reshape(3, 4))
    assert b.kind == "numpy" and b.array.dtype == np.float32 and b.shape == (3, 4) and not b.on_device
    assert _arrays.batch([1.0, 2.0, 3.0]).shape == (1, 3)
    f = np.asfortranarray(np.ones((4, 5), np.float32))
    assert _arrays.batch(f).array.flags.c_contiguous
    h = _arrays.batch(np.ones((2, 2), np.float16), (np.float32, np.float16))
    assert h.array.dtype == np.float16
    out, core = b.empty(3, 2, np.int64)
    assert out is core and out.shape == (3, 2) and out.dtype == np.int64


def test_copy_into_host():
    dst = np.zeros((2, 3), np.uint32)
    _arrays.copy_into(dst, np.arange(6, dtype=np.int64).reshape(2, 3))
    assert dst.tolist() == [[0, 1, 2], [3, 4, 5]]
