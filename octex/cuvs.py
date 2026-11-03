# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""cuVS-style adapter: the calls of cuvs.neighbors.cagra, answered by octex.

    from octex import cuvs as cagra                                     # was: from cuvs.neighbors import cagra
    index = cagra.build(cagra.IndexParams(), dataset)
    distances, neighbors = cagra.search(cagra.SearchParams(), index, queries, k)
    cagra.save("index_dir", index); index = cagra.load("index_dir")

IndexParams and SearchParams take the octex parameters (copies, list_len, ... and read, starts, waves, ...) and accept
the CAGRA parameters (graph_degree, itopk_size, ...), which have no effect; `max_queries` sets the batch size. Only
metric="sqeuclidean" is supported (ValueError otherwise). Inputs may be CuPy, PyTorch, numpy, or any array with
__cuda_array_interface__ or DLPack; outputs come back in the framework of the queries (int64 neighbors, float32
squared distances). An index is a directory: save copies it, load reads it.
"""

import os

from . import index as _index

_CAGRA_INDEX_PARAMS = (
    "intermediate_graph_degree",
    "graph_degree",
    "build_algo",
    "nn_descent_niter",
    "ivf_pq_build_params",
    "ivf_pq_search_params",
    "ace_params",
    "refinement_rate",
    "compression",
    "guarantee_connectivity",
    "add_data_on_build",
)
_CAGRA_SEARCH_PARAMS = (
    "itopk_size",
    "max_iterations",
    "algo",
    "team_size",
    "search_width",
    "min_iterations",
    "thread_block_size",
    "hashmap_mode",
    "hashmap_min_bitlen",
    "hashmap_max_fill_rate",
    "num_random_samplings",
    "rand_xor_mask",
    "persistent",
    "persistent_lifetime",
    "persistent_device_usage",
)
_LOAD_ARGS = ("placement", "vectors_on_host", "shard_vectors", "table_on_host", "sketch")


def _split(kw, own, ignored, what):
    mine = {}
    for name, value in kw.items():
        if name in own:
            mine[name] = value
        elif name not in ignored:
            raise TypeError(f"octex.cuvs: unknown {what} parameter {name!r}")
    return mine


class IndexParams:
    """Build parameters: metric plus the fields of cuvs::neighbors::octex::index_params and the load placement."""

    def __init__(self, metric="sqeuclidean", **kwargs):
        _index.check_metric(metric)
        self.metric = metric
        params = _split(kwargs, _index.INDEX_PARAMS + _LOAD_ARGS, _CAGRA_INDEX_PARAMS, "index")
        self.load_args = {k: params.pop(k) for k in list(params) if k in _LOAD_ARGS}
        self.params = params

    def __repr__(self):
        return f"octex.cuvs.IndexParams(metric={self.metric!r}, {self.params}, {self.load_args})"


class SearchParams:
    """Search parameters: the fields of cuvs::neighbors::octex::search_params; max_queries sets the batch size."""

    def __init__(self, max_queries=0, **kwargs):
        self.params = _split(kwargs, _index.SEARCH_PARAMS, _CAGRA_SEARCH_PARAMS, "search")
        if max_queries:
            self.params.setdefault("batch", int(max_queries))

    def __repr__(self):
        return f"octex.cuvs.SearchParams({self.params})"


class Index:
    """A cuVS-style index handle around octex.Index (empty until build or load)."""

    def __init__(self, octex_index=None):
        self._index = octex_index

    @property
    def trained(self):
        return self._index is not None

    @property
    def dim(self):
        return self._index.dim if self._index is not None else 0

    @property
    def size(self):
        return self._index.size if self._index is not None else 0

    def __len__(self):
        return self.size

    @property
    def octex(self):
        """The underlying octex.Index."""
        return self._require()

    def _require(self):
        if self._index is None:
            raise ValueError("octex.cuvs: the index is empty (build or load it first)")
        return self._index

    def __repr__(self):
        return f"octex.cuvs.Index({self._index!r})"


def build(index_params, dataset, resources=None):
    """Build an index of dataset [n, dim] (float32 or float16) with IndexParams."""
    if index_params is None:
        index_params = IndexParams()
    return Index(_index.build(dataset, metric=index_params.metric, **index_params.load_args, **index_params.params))


def search(search_params, index, queries, k, neighbors=None, distances=None, resources=None, filter=None):
    """k nearest neighbors of the queries: returns (distances, neighbors), written into the given arrays if any."""
    if filter is not None:
        raise ValueError("octex.cuvs: filtered search is not supported")
    if search_params is None:
        search_params = SearchParams()
    return index._require().search(queries, k, neighbors=neighbors, distances=distances, **search_params.params)


def save(filename, index, include_dataset=True, resources=None):
    """Copy the index directory to `filename` (a new directory). The vectors are always included."""
    index._require().save(filename)


def load(*args, resources=None, **load_args):
    """load(path) or load(index, path) (the cuVS 26.10 form, filling an empty Index); returns the index.

    Keyword arguments are those of octex.load (devices, placement, vectors_on_host, ...).
    """
    if len(args) == 1:
        target, path = Index(), args[0]
    elif len(args) == 2 and isinstance(args[0], Index):
        target, path = args
    elif len(args) == 2:
        path, target = args[0], Index()
    else:
        raise TypeError("octex.cuvs.load(path) or load(index, path)")
    target._index = _index.load(os.fspath(path), **load_args)
    return target
