# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""octex in three interfaces: build an index of random clustered vectors, search it natively, then make the same
search through the calls of cuVS CAGRA and of Faiss. Recall is measured against exhaustive search.

    python examples/python/quickstart.py
"""

import tempfile
import time

import numpy as np

import octex
from octex import cuvs as cagra  # was: from cuvs.neighbors import cagra
from octex import faiss  # was: import faiss


def data(n=50_000, m=1_000, d=96, clusters=100, seed=0):
    g = np.random.default_rng(seed)
    centers = 4 * g.standard_normal((clusters, d), dtype=np.float32)
    xb = centers[g.integers(clusters, size=n)] + g.standard_normal((n, d), dtype=np.float32)
    xq = centers[g.integers(clusters, size=m)] + g.standard_normal((m, d), dtype=np.float32)
    return xb.astype(np.float32), xq.astype(np.float32)


def exact_knn(xb, xq, k):
    d2 = (xq**2).sum(1)[:, None] - 2 * xq @ xb.T + (xb**2).sum(1)[None, :]
    return np.argsort(d2, axis=1)[:, :k]


def recall(I, gt):
    I, gt = np.asarray(I), np.asarray(gt)
    k = gt.shape[1]
    return np.mean([len(set(a[:k]) & set(b)) / k for a, b in zip(I, gt, strict=True)])


def main():
    k = 10
    xb, xq = data()
    gt = exact_knn(xb, xq, k)

    # native: build writes an index directory and loads it onto the GPUs
    t = time.perf_counter()
    index = octex.build(xb, copies=64)
    print(f"built {index} in {time.perf_counter() - t:.1f} s")
    D, I = index.search(xq, k=k)  # squared distances and int64 ids, nearest first
    print(f"native     recall@{k} = {recall(I, gt):.4f}  (defaults)")
    D, I = index.search(xq, k=k, read=8, starts=32, waves=1)  # a lighter search: fewer list entries and start points
    print(f"native     recall@{k} = {recall(I, gt):.4f}  (read 8, starts 32, waves 1)")

    with tempfile.TemporaryDirectory() as tmp:  # an index is a directory: save copies it, load reads it
        index.save(f"{tmp}/index")
        again = octex.load(f"{tmp}/index", placement="replicate")
        assert np.array_equal(again.search(xq, k=k)[1], index.search(xq, k=k)[1])
        del again

    # cuVS: the calls of cuvs.neighbors.cagra
    cindex = cagra.build(cagra.IndexParams(metric="sqeuclidean", copies=64), xb)
    distances, neighbors = cagra.search(cagra.SearchParams(), cindex, xq, k)
    print(f"cuvs.cagra recall@{k} = {recall(neighbors, gt):.4f}")

    # Faiss: the calls of a Faiss GPU index
    findex = faiss.IndexOctex(xb.shape[1], copies=64)
    findex.add(xb)
    D, I = findex.search(xq, k)
    print(f"faiss      recall@{k} = {recall(I, gt):.4f}  (ntotal {findex.ntotal})")

    try:  # device arrays stay on the GPU: CuPy in, CuPy out
        import cupy as cp

        D, I = index.search(cp.asarray(xq), k=k)
        print(f"cupy       recall@{k} = {recall(I.get(), gt):.4f}  ({type(I).__module__}.{type(I).__name__})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
