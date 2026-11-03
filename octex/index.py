# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""The octex index: build() writes an index directory and loads it, load() places a directory on the GPUs, and
Index.search() answers batches of queries.

    import octex
    index = octex.build(xb)                        # numpy, CuPy, or PyTorch; float32 or float16
    D, I = index.search(xq, k=10)                  # squared Euclidean distances and int64 ids, nearest first
    index.save("sift.octex")                       # a directory; octex.load("sift.octex") restores it

Results follow Faiss and cuVS: (distances, neighbors). They come back in the framework of the queries: numpy for
host arrays (numpy, lists, CPU tensors give CPU tensors), and arrays on the same GPU for CuPy, PyTorch, or any array
with DLPack or __cuda_array_interface__.
"""

import json
import os
import shutil
import tempfile

import numpy as np

from . import _arrays

METRICS = ("sqeuclidean", "l2")

INDEX_PARAMS = (
    "copies",
    "blocks",
    "pca_dims",
    "spacings",
    "list_len",
    "min_cell",
    "log_buckets",
    "sketch_width",
    "table_len",
    "reverse_edges",
    "exact_lists",
    "list_probes",
    "member_lists",
    "exact_table_max",
    "table_passes",
    "table_sketch_order",
    "write_behind",
    "seed",
    "devices",
)
SEARCH_PARAMS = ("read", "starts", "waves", "rerank", "wave_rerank", "batch", "sketch_rank", "final_rerank")
LOAD_PARAMS = (
    "devices",
    "placement",
    "vectors_on_host",
    "shard_vectors",
    "table_on_host",
    "sketch",
    "parts",
    "copies_per_part",
    "copies_limit",
    "list_limit",
)
PLACEMENTS = ("shard", "replicate")
SKETCH_PLACEMENTS = ("replicate", "shard", "host")


def _core():
    try:
        from . import _core as core
    except ImportError as e:
        raise ImportError(
            "octex: the CUDA extension octex._core is missing or cannot load its libraries (pip install octex "
            "builds it; it needs CUDA 12 and the libcuvs-cu12 wheel)"
        ) from e
    return core


def check_metric(metric):
    if str(metric).lower() not in METRICS:
        raise ValueError(
            f"octex: metric {metric!r} is not supported; octex searches by squared Euclidean distance "
            "(metric='sqeuclidean')"
        )


def _set(obj, values, names, what):
    for name, value in values.items():
        if name not in names:
            raise TypeError(f"octex: unknown {what} parameter {name!r}; valid: {', '.join(names)}")
        if value is None:
            continue
        if name in ("spacings",):
            value = [float(v) for v in value]
        elif name == "devices":
            value = [int(v) for v in ([value] if isinstance(value, int) else value)]
        setattr(obj, name, value)
    return obj


def index_params(**kw):
    """octex._core.IndexParams from keyword arguments (names of cuvs::neighbors::octex::index_params)."""
    return _set(_core().IndexParams(), kw, INDEX_PARAMS, "index")


def search_params(**kw):
    """octex._core.SearchParams from keyword arguments (names of cuvs::neighbors::octex::search_params)."""
    return _set(_core().SearchParams(), kw, SEARCH_PARAMS, "search")


def load_params(placement="shard", sketch="replicate", **kw):
    core = _core()
    if placement not in PLACEMENTS:
        raise ValueError(f"octex: placement must be one of {PLACEMENTS}, not {placement!r}")
    if sketch not in SKETCH_PLACEMENTS:
        raise ValueError(f"octex: sketch must be one of {SKETCH_PLACEMENTS}, not {sketch!r}")
    p = _set(core.LoadParams(), kw, LOAD_PARAMS, "load")
    p.mode = getattr(core.Placement, placement)
    p.sketch = getattr(core.SketchPlacement, sketch)
    return p


def _meta(path):
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def stored_vectors(path):
    """The fp16 vectors [N, D] of an index directory, memory mapped."""
    f = _meta(path).get("files", {}).get("vectors", os.path.join(path, "vectors.npy"))
    return np.load(f if os.path.isabs(f) else os.path.join(path, f), mmap_mode="r")


class Index:
    """An octex index placed on one or more GPUs. Create it with octex.build() or octex.load()."""

    def __init__(self, handle, path, tmpdir=None, load_args=None):
        self._handle = handle
        self._path = os.fspath(path)
        self._tmpdir = tmpdir  # a temporary build directory lives as long as the index
        self._load_args = dict(load_args or {})

    # ---------------------------------------------------------------- properties
    @property
    def size(self):
        """Number of indexed vectors."""
        return int(self._handle.size)

    @property
    def dim(self):
        return int(self._handle.dim)

    @property
    def copies(self):
        """Randomly projected copies loaded."""
        return int(self._handle.copies)

    @property
    def devices(self):
        """CUDA devices that hold the index."""
        return list(self._handle.devices)

    @property
    def device_bytes(self):
        """Device memory per GPU, in bytes."""
        return list(self._handle.device_bytes)

    @property
    def path(self):
        """The index directory the index was loaded from."""
        return self._path

    @property
    def metric(self):
        return "sqeuclidean"

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"octex.Index(size={self.size}, dim={self.dim}, copies={self.copies}, devices={self.devices})"

    # ---------------------------------------------------------------- search
    def search(
        self,
        queries,
        k=10,
        *,
        read=32,
        starts=128,
        waves=2,
        rerank=128,
        wave_rerank=128,
        batch=10000,
        sketch_rank=False,
        final_rerank=32,
        neighbors=None,
        distances=None,
    ):
        """Approximate k nearest neighbors of a batch of queries [m, dim].

        Returns (distances, neighbors): float32 squared Euclidean distances and int64 ids [m, k], nearest first, in
        the framework (and on the GPU) of the queries. `read` list entries per copy, `starts` first-round hits, and
        `waves` rounds over the neighbor table trade speed for recall. Optional `neighbors` / `distances` arrays
        receive the results in place (any dtype; cast if needed) and are returned.
        """
        k = int(k)
        if k <= 0:
            raise ValueError("octex: k must be positive")
        params = search_params(
            read=read,
            starts=starts,
            waves=waves,
            rerank=rerank,
            wave_rerank=wave_rerank,
            batch=batch,
            sketch_rank=sketch_rank,
            final_rerank=final_rerank,
        )
        q = _arrays.batch(queries, (np.float32,), "queries")
        m, d = q.shape
        if d != self.dim:
            raise ValueError(f"octex: queries have dimension {d}, the index {self.dim}")
        I, I_core = self._out(q, neighbors, m, k, np.int64)
        D, D_core = self._out(q, distances, m, k, np.float32)
        _core().search(params, self._handle, q.array, I_core, D_core, q.stream)
        if neighbors is not None and I is not neighbors:
            I = _arrays.copy_into(neighbors, I)
        if distances is not None and D is not distances:
            D = _arrays.copy_into(distances, D)
        return D, I

    @staticmethod
    def _out(q, given, m, k, dtype):
        """An output array: the caller's when the core can write it directly, else a fresh one."""
        if given is not None:
            try:
                g = _arrays.batch(given, (dtype,), "output")
            except Exception:
                g = None
            if (
                g is not None
                and g.kind == q.kind
                and g.array is not None
                and g.shape == (m, k)
                and _same_memory(g.array, given)
                and g.device == q.device
            ):
                return given, g.array
            if tuple(given.shape) != (m, k):
                raise ValueError(f"octex: output arrays must have shape {(m, k)}, not {tuple(given.shape)}")
        return q.empty(m, k, dtype)

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        """Copy the index directory to `path` (a new directory); octex.load(path) restores the index.

        Arrays that meta.json refers to by absolute path (its "files" map) stay where they are and are not copied.
        """
        path = os.fspath(path)
        if os.path.exists(path) and (not os.path.isdir(path) or os.listdir(path)):
            raise FileExistsError(f"octex: {path} exists and is not an empty directory")
        if os.path.abspath(path) == os.path.abspath(self._path):
            return path
        shutil.copytree(self._path, path, dirs_exist_ok=True)
        return path


def _same_memory(core_array, given):
    """True when the array handed to the core is the caller's own memory (no conversion copy was made)."""
    if core_array is given:
        return True
    try:
        if isinstance(given, np.ndarray):
            return np.shares_memory(core_array, given) and core_array.dtype == given.dtype
        return _ptr(core_array) == _ptr(given)
    except Exception:
        return False


def _ptr(a):
    if hasattr(a, "data_ptr"):
        return a.data_ptr()
    if hasattr(a, "__cuda_array_interface__"):
        return a.__cuda_array_interface__["data"][0]
    if hasattr(a, "__array_interface__"):
        return a.__array_interface__["data"][0]
    return id(a)


def load(
    path,
    devices=None,
    placement="shard",
    vectors_on_host=False,
    *,
    shard_vectors=False,
    table_on_host=False,
    sketch="replicate",
    parts=0,
    copies_per_part=0,
    copies_limit=0,
    list_limit=0,
):
    """Load an index directory onto the GPUs (devices: list of CUDA ids, default all visible).

    placement "shard" splits the copies and the neighbor table over the GPUs, "replicate" gives every GPU the whole
    index and a slice of the queries. vectors_on_host / table_on_host keep those arrays in pinned host memory, and
    shard_vectors splits the vectors by rows over the GPUs. sketch is "replicate" (a copy per GPU), "shard" (row
    ranges over the GPUs), or "host" (pinned host memory). parts: the copies were saved in this many consecutive
    parts (dealt to the GPUs round robin); copies_per_part / copies_limit / list_limit load a smaller index.
    """
    path = os.fspath(path)
    if not os.path.isfile(os.path.join(path, "meta.json")):
        raise FileNotFoundError(f"octex: {path} is not an index directory (no meta.json)")
    check_metric(_meta(path).get("metric", "sqeuclidean"))
    args = dict(
        devices=devices,
        placement=placement,
        vectors_on_host=vectors_on_host,
        shard_vectors=shard_vectors,
        table_on_host=table_on_host,
        sketch=sketch,
        parts=parts,
        copies_per_part=copies_per_part,
        copies_limit=copies_limit,
        list_limit=list_limit,
    )
    handle = _core().load(path, load_params(**args))
    return Index(handle, path, load_args=args)


def build(
    dataset,
    path=None,
    *,
    metric="sqeuclidean",
    placement="shard",
    vectors_on_host=False,
    table_on_host=False,
    shard_vectors=False,
    sketch="replicate",
    **index_params_,
):
    """Build an index of `dataset` [n, dim] (float32 or float16; host or GPU) and load it.

    The index directory is written to `path`, or to a temporary directory that lives as long as the returned index
    (Index.save keeps it). Keyword arguments are the fields of cuvs::neighbors::octex::index_params (copies, blocks,
    list_len, table_len, seed, devices, ...); `devices` also selects the GPUs the index is loaded on.
    """
    check_metric(metric)
    params = index_params(**index_params_)
    x = _arrays.batch(dataset, (np.float32, np.float16), "dataset")
    tmp = None
    if path is None:
        tmp = tempfile.TemporaryDirectory(prefix="octex-")
        path = tmp.name
    path = os.fspath(path)
    os.makedirs(path, exist_ok=True)
    _core().build(params, x.array, path)
    idx = load(
        path,
        devices=index_params_.get("devices"),
        placement=placement,
        vectors_on_host=vectors_on_host,
        table_on_host=table_on_host,
        shard_vectors=shard_vectors,
        sketch=sketch,
    )
    idx._tmpdir = tmp
    return idx
