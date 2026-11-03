# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Recall of the SIFT1M index exported by the research prototype, on one GPU.

python -m pytest tests/test_recall.py
"""

import numpy as np
import pytest
from conftest import recall

import octex

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def sift(sift_path):
    index = octex.load(sift_path, devices=[0])
    yield index
    del index


def test_recall_light_settings(sift, sift_queries, sift_groundtruth):
    D, I = sift.search(sift_queries, k=10, read=8, starts=32, waves=1)
    r = recall(I, sift_groundtruth)
    print(f"SIFT1M recall@10 (read 8, starts 32, waves 1): {r:.4f}")
    assert r >= 0.99


def test_recall_defaults(sift, sift_queries, sift_groundtruth):
    D, I = sift.search(sift_queries, k=10)
    r = recall(I, sift_groundtruth)
    print(f"SIFT1M recall@10 (defaults): {r:.4f}")
    assert r >= 0.99
    # distances are the squared Euclidean distances of the returned ids (fp16 vectors: small relative error)
    base = octex.index.stored_vectors(sift.path)
    rows = np.asarray(base[I[:20].ravel()], dtype=np.float32).reshape(20, 10, -1)
    exact = ((rows - sift_queries[:20, None, :]) ** 2).sum(-1)
    assert np.allclose(D[:20], exact, rtol=1e-2, atol=1.0)
