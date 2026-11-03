# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Faiss-style adapter: GPU indexes with the interface of faiss.Index, answered by octex.

    from octex import faiss                       # was: import faiss
    index = faiss.IndexOctex(d)                   # or faiss.GpuIndexCagra(faiss.StandardGpuResources(), d)
    index.train(xb); index.add(xb)                # train is a no-op for IndexOctex; the build runs on first search
    D, I = index.search(xq, k)                    # float32 squared distances, int64 ids, nearest first
    faiss.write_index(index, "index_dir"); index = faiss.read_index("index_dir")

It does not import faiss. Vectors are added in batches (numpy, CuPy, or PyTorch) and the index is built when it is
first searched after an add, or by an explicit build(); ids are the positions in the order of the adds. Results come
back in the framework of the queries (numpy for numpy queries, as in Faiss). Only METRIC_L2 is supported.
"""

import os

import numpy as np

from . import _arrays
from . import index as _index

METRIC_INNER_PRODUCT, METRIC_L2 = 0, 1  # the values of faiss.METRIC_INNER_PRODUCT / METRIC_L2


def _check_metric(metric):
    if metric != METRIC_L2:
        raise ValueError("octex.faiss: only METRIC_L2 is supported (octex searches by squared Euclidean distance)")


class SearchParametersOctex:
    """Per-call search parameters (fields of cuvs::neighbors::octex::search_params); CAGRA fields are ignored."""

    def __init__(self, **kwargs):
        self.params = {k: v for k, v in kwargs.items() if k in _index.SEARCH_PARAMS}


SearchParametersCagra = SearchParametersOctex


class IndexOctex:
    """An octex index with the faiss.Index interface (d, ntotal, is_trained, metric_type, train, add, search, reset).

    Keyword arguments: the octex index parameters (copies, list_len, table_len, seed, devices, ...), the load
    arguments (placement, vectors_on_host, ...), and the default search parameters (read, starts, waves, ...), which
    are also attributes of `search_params`.
    """

    def __init__(self, d, metric=METRIC_L2, **kwargs):
        _check_metric(metric)
        self.d, self.metric_type, self.is_trained = int(d), metric, True
        load_names = ("placement", "vectors_on_host", "shard_vectors", "table_on_host", "sketch")
        self.index_params = {k: v for k, v in kwargs.items() if k in _index.INDEX_PARAMS}
        self.load_args = {k: v for k, v in kwargs.items() if k in load_names}
        self.search_params = {k: v for k, v in kwargs.items() if k in _index.SEARCH_PARAMS}
        unknown = set(kwargs) - set(self.index_params) - set(self.load_args) - set(self.search_params)
        if unknown:
            raise TypeError(f"octex.faiss: unknown parameters {sorted(unknown)}")
        self.reset()

    @property
    def ntotal(self):
        built = 0 if self._built is None else self._built.size
        return built + sum(len(x) for x in self._pending)

    def reset(self):
        self._parts = []  # host batches of all built adds, in order
        self._pending = []
        self._built = None

    def train(self, x):
        pass

    def add(self, x):
        x = _arrays.to_host(x, (np.float32, np.float16))
        if x.shape[1] != self.d:
            raise ValueError(f"octex.faiss: vectors have dimension {x.shape[1]}, the index {self.d}")
        self._pending.append(x)

    def build(self):
        """Build the index of all vectors added so far (otherwise done by the first search after an add)."""
        if not self._pending and self._built is not None:
            return self
        if self._built is not None and not self._parts:  # read from disk: start from its stored vectors
            self._parts = [np.asarray(_index.stored_vectors(self._built.path))]
        parts = self._parts + self._pending
        if not parts:
            raise RuntimeError("octex.faiss: the index is empty")
        data = parts[0] if len(parts) == 1 else np.concatenate([p.astype(np.float32, copy=False) for p in parts])
        self._built = _index.build(data, **self.load_args, **self.index_params)
        self._parts, self._pending = parts, []
        return self

    def search(self, x, k, *, params=None, D=None, I=None):
        if self._pending or self._built is None:
            self.build()
        sp = dict(self.search_params)
        if params is not None:
            sp.update(getattr(params, "params", {}))
        return self._built.search(x, k, distances=D, neighbors=I, **sp)

    @property
    def octex(self):
        """The underlying octex.Index (built on access)."""
        return self.build()._built


class StandardGpuResources:
    """Accepted for faiss.StandardGpuResources compatibility; octex manages its own CUDA resources."""

    def noTempMemory(self):
        pass

    def setTempMemory(self, size):
        pass


class GpuIndexCagraConfig:
    """Accepted for faiss.GpuIndexCagraConfig compatibility: `device` selects the GPU, other fields are ignored."""

    def __init__(self):
        self.device = None
        self.graph_degree = 64
        self.intermediate_graph_degree = 128


class GpuIndexCagra(IndexOctex):
    """faiss.GpuIndexCagra(res, d, metric, config): train(x) builds the index of x, as in Faiss."""

    def __init__(self, res=None, d=None, metric=METRIC_L2, config=None, **kwargs):
        if d is None and isinstance(res, int):  # GpuIndexCagra(d)
            res, d = None, res
        if config is not None and getattr(config, "device", None) is not None:
            kwargs.setdefault("devices", [int(config.device)])
        super().__init__(d, metric, **kwargs)
        self.is_trained = False

    def train(self, x):
        self.reset()
        self.add(x)
        self.build()
        self.is_trained = True

    def add(self, x):
        super().add(x)
        self.is_trained = True


def write_index(index, path):
    """Write the index directory of an IndexOctex (built first if needed) to `path`."""
    index.octex.save(os.fspath(path))


def read_index(path, **kwargs):
    """An IndexOctex over an index directory (placement and search defaults as keyword arguments)."""
    loaded = _index.load(os.fspath(path), **{k: v for k, v in kwargs.items() if k not in _index.SEARCH_PARAMS})
    index = IndexOctex(loaded.dim, **{k: v for k, v in kwargs.items() if k in _index.SEARCH_PARAMS})
    index._built = loaded
    return index
