# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Build then search on random clustered data: recall, input types, and the index directory round trip.

python -m pytest tests/test_build.py
"""

import json
import os

import numpy as np
import pytest
from conftest import clustered, recall

import octex

pytestmark = pytest.mark.gpu
K = 10


@pytest.fixture(scope="module")
def data():
    return clustered()


@pytest.fixture(scope="module")
def built(data, tmp_path_factory):
    X, Q, gt = data
    path = tmp_path_factory.mktemp("octex") / "index"
    return octex.build(X, path, devices=[0], copies=64, seed=1), path


def test_build_writes_directory(built, data):
    index, path = built
    meta = json.load(open(os.path.join(path, "meta.json")))
    assert meta["format"] == 1 and meta["n"] == len(data[0]) and meta["dim"] == data[0].shape[1]
    assert meta["copies"] == 64 and meta["metric"] == "sqeuclidean"
    assert index.size == len(data[0]) and index.dim == data[0].shape[1] and index.copies == 64


def test_build_recall(built, data):
    index, _ = built
    X, Q, gt = data
    D, I = index.search(Q, k=K)
    assert recall(I, gt) >= 0.95
    exact = ((X[I] - Q[:, None, :]) ** 2).sum(-1)
    assert np.allclose(D, exact, rtol=1e-2, atol=0.5)


def test_save_load_round_trip(built, data, tmp_path):
    index, _ = built
    X, Q, gt = data
    D, I = index.search(Q, k=K)
    index.save(tmp_path / "copy")
    with pytest.raises(FileExistsError):
        index.save(tmp_path / "copy")
    again = octex.load(tmp_path / "copy", devices=[0], placement="replicate")
    D2, I2 = again.search(Q, k=K)
    assert np.array_equal(I, I2)


def test_build_temporary_directory(data):
    X, Q, gt = data
    index = octex.build(X[:5000], copies=16, devices=[0])  # no path: a temporary directory
    assert os.path.isfile(os.path.join(index.path, "meta.json"))
    path = index.path
    del index
    assert not os.path.exists(path)


def test_build_float16_and_device_inputs(data):
    X, Q, gt = data
    half = octex.build(X.astype(np.float16), copies=32, devices=[0])
    assert recall(half.search(Q, k=K)[1], gt) >= 0.9
    cp = pytest.importorskip("cupy")
    dev = octex.build(cp.asarray(X), copies=32, devices=[0])  # device datasets are copied to the host
    D, I = dev.search(cp.asarray(Q), k=K)
    assert isinstance(I, cp.ndarray) and recall(I.get(), gt) >= 0.9
