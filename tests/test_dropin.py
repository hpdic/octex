# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""The adapters are drop-in replacements: the calls of cuvs.neighbors.cagra and of Faiss GPU indexes run unchanged
on octex and give the results of the native interface.

    python -m pytest tests/test_dropin.py
"""

import gc

import numpy as np
import pytest

import octex
from octex import cuvs as cagra
from octex import faiss

pytestmark = pytest.mark.gpu
K = 10


def test_cuvs_search_matches_native(light, sift_queries):
    D_ref, I_ref = light.search(sift_queries[:200], k=K, read=16, starts=64)
    index = cagra.Index(light)
    assert index.trained and index.dim == 128
    sp = cagra.SearchParams(read=16, starts=64, itopk_size=64)  # CAGRA knobs are accepted and have no effect
    D, I = cagra.search(sp, index, sift_queries[:200], K)
    assert np.array_equal(I, I_ref) and np.allclose(D, D_ref)
    cp = pytest.importorskip("cupy")
    neighbors, distances = cp.empty((200, K), cp.uint32), cp.empty((200, K), cp.float32)
    D2, I2 = cagra.search(sp, index, cp.asarray(sift_queries[:200]), K, neighbors=neighbors, distances=distances)
    assert I2 is neighbors and D2 is distances
    assert np.array_equal(neighbors.get().astype(np.int64), I_ref)


def test_cuvs_load_forms(sift_path, sift_queries):
    a = cagra.load(sift_path, devices=[0], copies_limit=4)
    b = cagra.load(cagra.Index(), sift_path, devices=[0], copies_limit=4)  # the cuVS 26.10 form
    assert a.size == b.size == 1_000_000
    Da, Ia = cagra.search(cagra.SearchParams(), a, sift_queries[:50], K)
    Db, Ib = cagra.search(cagra.SearchParams(), b, sift_queries[:50], K)
    assert np.array_equal(Ia, Ib)
    del a, b
    gc.collect()


def test_faiss_read_index_matches_native(sift_path, sift_queries):
    native = octex.load(sift_path, devices=[0], copies_limit=4)
    D_ref, I_ref = native.search(sift_queries[:100], k=K, read=8, starts=32, waves=1)
    del native
    gc.collect()
    index = faiss.read_index(sift_path, devices=[0], copies_limit=4, read=8, starts=32, waves=1)
    assert index.ntotal == 1_000_000 and index.d == 128 and index.is_trained
    D, I = index.search(sift_queries[:100], K)  # the Faiss call
    assert D.dtype == np.float32 and I.dtype == np.int64 and D.shape == (100, K)
    assert np.array_equal(I, I_ref)
    del index
    gc.collect()


def _data(n=20_000, m=200, d=64, seed=0):
    from conftest import clustered

    return clustered(n, m, d, seed=seed)


def test_cuvs_build_same_calls_as_cagra():
    """The same four lines run on cuvs.neighbors.cagra and on octex.cuvs."""
    from conftest import recall

    cp = pytest.importorskip("cupy")
    X, Q, gt = _data()
    Xd, Qd = cp.asarray(X), cp.asarray(Q)
    modules = [cagra]
    try:
        from cuvs.neighbors import cagra as real

        modules.append(real)
    except ImportError:
        pass
    for lib in modules:
        index = lib.build(lib.IndexParams(metric="sqeuclidean"), Xd)
        distances, neighbors = lib.search(lib.SearchParams(), index, Qd, K)
        I = cp.asarray(neighbors).get()
        assert I.shape == (200, K) and cp.asarray(distances).shape == (200, K)
        if lib is cagra:  # the reference runs the same calls; its recall is its own
            assert recall(I, gt) >= 0.9


def test_faiss_add_then_search():
    from conftest import recall

    X, Q, gt = _data(seed=1)
    index = faiss.IndexOctex(X.shape[1])
    index.train(X)  # no-op, as for flat indexes
    index.add(X[:12_000])
    index.add(X[12_000:])  # ids continue in the order of the adds
    assert index.ntotal == len(X)
    D, I = index.search(Q, K)
    assert D.dtype == np.float32 and I.dtype == np.int64 and I.shape == (len(Q), K)
    assert recall(I, gt) >= 0.9
    cagra_like = faiss.GpuIndexCagra(faiss.StandardGpuResources(), X.shape[1], faiss.METRIC_L2)
    cagra_like.train(X)  # GpuIndexCagra builds in train, as in Faiss
    D2, I2 = cagra_like.search(Q, K)
    assert recall(I2, gt) >= 0.9


def test_faiss_against_faiss():
    real = pytest.importorskip("faiss")
    from conftest import recall

    X, Q, gt = _data(seed=2)
    ref, ours = real.IndexFlatL2(X.shape[1]), faiss.IndexOctex(X.shape[1])
    for index in (ref, ours):  # the same calls on both libraries
        index.add(X)
    D_ref, I_ref = ref.search(Q, K)
    D, I = ours.search(Q, K)
    assert D.dtype == D_ref.dtype and I.dtype == I_ref.dtype and D.shape == D_ref.shape
    assert ours.ntotal == ref.ntotal
    assert recall(I, I_ref) >= 0.9
