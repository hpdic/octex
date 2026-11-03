# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Shared fixtures. The SIFT1M index directory comes from OCTEX_SIFT1M (default /home/cc/data/octex/sift1m) and
holds queries.npy and groundtruth.npy next to the index; tests that need it are skipped when it is missing."""

import os

import numpy as np
import pytest

SIFT1M = os.environ.get("OCTEX_SIFT1M", "/home/cc/data/octex/sift1m")


def sift_dir():
    if not os.path.isfile(os.path.join(SIFT1M, "meta.json")):
        pytest.skip(f"no SIFT1M index directory at {SIFT1M} (set OCTEX_SIFT1M)")
    return SIFT1M


@pytest.fixture(scope="session")
def sift_path():
    return sift_dir()


@pytest.fixture(scope="session")
def sift_queries(sift_path):
    return np.ascontiguousarray(np.load(os.path.join(sift_path, "queries.npy")), dtype=np.float32)


@pytest.fixture(scope="session")
def sift_groundtruth(sift_path):
    return np.load(os.path.join(sift_path, "groundtruth.npy"))


@pytest.fixture(scope="session")
def light(sift_path):
    """The SIFT1M index with its first 16 copies on GPU 0: enough for API tests, small in memory."""
    import octex

    return octex.load(sift_path, devices=[0], copies_limit=16)


def recall(I, gt, k=10):
    I, gt = np.asarray(I)[:, :k], np.asarray(gt)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(I, gt, strict=True)]))


def clustered(n=20_000, m=200, d=64, clusters=50, seed=0):
    """A Gaussian mixture dataset, its queries, and the exact 10 nearest neighbors of the queries."""
    g = np.random.default_rng(seed)
    centers = g.standard_normal((clusters, d), dtype=np.float32) * 4
    X = (centers[g.integers(clusters, size=n)] + g.standard_normal((n, d), dtype=np.float32)).astype(np.float32)
    Q = (centers[g.integers(clusters, size=m)] + g.standard_normal((m, d), dtype=np.float32)).astype(np.float32)
    d2 = (Q**2).sum(1)[:, None] - 2 * Q @ X.T + (X**2).sum(1)[None, :]
    gt = np.argsort(d2, axis=1)[:, :10]
    return X, Q, gt
