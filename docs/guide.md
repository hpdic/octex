# octex user guide

octex answers approximate k-nearest-neighbor queries by squared Euclidean distance on one or more NVIDIA GPUs. An
index is a directory on disk. `build` writes the directory, `load` places it on the GPUs, and `search` answers a batch
of queries with one CUDA graph per GPU. [design.md](design.md) describes the algorithm and the directory format.

## Installation

```bash
pip install .            # from a checkout: Linux, Python >= 3.11, CUDA 12, compute capability >= 8.0
```

The build compiles the CUDA library and the Python extension against the `libcuvs-cu12` 26.10 wheel, which pip
installs together with RAFT, RMM, and the CUDA math libraries. At run time the extension finds these libraries in
site-packages, so no `LD_LIBRARY_PATH` is needed. The package needs only numpy. CuPy, PyTorch, and any other array
with DLPack or `__cuda_array_interface__` are accepted without copies when they are installed. For development:

```bash
pip install -e ".[test]"
pytest -m gpu            # needs a GPU; pytest -m "not gpu" runs the rest
```

The C++ library alone builds with CMake from `cpp/` against the same wheel (see the comment at the top of
`cpp/CMakeLists.txt`). `OCTEX_BUILD_EXAMPLES=ON` (the default) adds `examples/cpp/search.cu` and, with the C API,
`examples/c/search.c`. `OCTEX_BUILD_C_API=ON` (the default) adds the C API below.

## Three interfaces

All three return `(distances, neighbors)`: float32 squared distances and int64 ids `[m, k]`, nearest first. Results
come back in the framework of the queries. Host arrays (numpy, lists, CPU tensors) give numpy arrays (CPU tensors
for CPU tensors). Device arrays give arrays on the same GPU: CuPy for CuPy, PyTorch for PyTorch, pylibraft
`device_ndarray` for `device_ndarray`, and a DLPack array for anything else. Queries must be float32; float16 and
float64 queries are converted in their own framework. Only `metric="sqeuclidean"` is supported, and any other metric
raises `ValueError`.

### Native

```python
import octex

index = octex.build(xb, "index_dir", copies=200)       # xb: [n, d] float32 or float16, host or GPU
index = octex.load("index_dir", devices=[0, 1], placement="shard")
D, I = index.search(xq, k=10, read=32, starts=128, waves=2)
index.save("copy_dir")                                  # copies the directory
index.size, index.dim, index.copies, index.devices, index.device_bytes
```

Without a path, `build` writes to a temporary directory that is deleted with the index; `save` keeps it. The
`neighbors=` and `distances=` arguments of `search` take preallocated outputs. Arrays of the right dtype on the
right device are written in place, and other arrays (for example uint32 neighbors) receive a cast copy.

### cuVS

```python
from octex import cuvs as cagra                         # was: from cuvs.neighbors import cagra

index = cagra.build(cagra.IndexParams(metric="sqeuclidean", copies=200), dataset)
distances, neighbors = cagra.search(cagra.SearchParams(read=16), index, queries, k=10)
cagra.save("index_dir", index)
index = cagra.load("index_dir")                         # or cagra.load(cagra.Index(), "index_dir")
```

`IndexParams` takes the build parameters and the placement arguments of `load` below. `SearchParams` takes the
search parameters. Both accept the CAGRA parameters (`graph_degree`, `itopk_size`, ...) and ignore them.
`SearchParams(max_queries=...)` sets the batch size.

### Faiss

```python
from octex import faiss                                 # was: import faiss

index = faiss.IndexOctex(d, copies=200, read=16)        # build and search parameters as keyword arguments
index.add(xb)                                           # adds accumulate; the build runs on the next search
D, I = index.search(xq, 10)
faiss.write_index(index, "index_dir")
index = faiss.read_index("index_dir", devices=[0])
```

`train` is a no-op and `ntotal`, `d`, `is_trained`, `metric_type`, and `reset` behave as in Faiss. Ids are positions
in the order of the adds. `GpuIndexCagra(res, d, metric, config)` builds in `train`, as Faiss does, and
`config.device` selects the GPU. `StandardGpuResources` is accepted and does nothing.

## C API

The C API follows the cuVS C API (`cuvsBruteForce*`, `cuvsCagra*`): parameter structs with `Create` and `Destroy`
and the defaults of the C++ API, an opaque index handle, DLPack tensors for every array, a `cuvsResources_t` handle
from `cuvsResourcesCreate`, and a `cuvsError_t` result whose message comes from `cuvsGetLastErrorText`. The header is
`c/include/cuvs/neighbors/octex.h`; the library `liboctex_c` links against `libcuvs_c`.

```c
#include <cuvs/neighbors/octex.h>

cuvsResources_t res;
cuvsResourcesCreate(&res);

cuvsOctexIndexParams_t params;
cuvsOctexIndexParamsCreate(&params);           /* defaults of the C++ API: 200 copies, all visible GPUs */
params->copies = 64;
cuvsOctexBuild(res, params, &dataset, "index_dir");  /* float32 or float16 [n, d], host or device */

cuvsOctexLoadParams_t load_params;
cuvsOctexLoadParamsCreate(&load_params);       /* or NULL in cuvsOctexLoad: all visible GPUs, shard */
load_params->placement = OCTEX_PLACEMENT_REPLICATE;
cuvsOctexIndex_t index;
cuvsOctexIndexCreate(&index);
cuvsOctexLoad(res, "index_dir", load_params, index);

cuvsOctexSearchParams_t sp;
cuvsOctexSearchParamsCreate(&sp);              /* read 32, starts 128, waves 2 */
cuvsOctexSearch(res, sp, index, &queries, &neighbors, &distances);
cuvsStreamSync(res);                           /* device results are ordered on the stream of res */

cuvsOctexIndexDestroy(index);
```

The fields of the structs are the build, load, and search parameters of the tables below. The arrays `spacings` and
`devices` are a pointer and a count (`n_spacings`, `n_devices`) owned by the caller; NULL keeps the default. The
placements are `OCTEX_PLACEMENT_SHARD` and `OCTEX_PLACEMENT_REPLICATE`, and the sketch placements are
`OCTEX_SKETCH_REPLICATE`, `OCTEX_SKETCH_SHARD`, and `OCTEX_SKETCH_HOST`. A dataset in device memory is copied to
the host before the build. Queries are float32 `[m, d]`, neighbors int64 `[m, k]`, and distances float32 `[m, k]`
(squared distances, nearest first), with k the second extent of `neighbors`. The three tensors live either all in
host memory or all in device memory (any GPU). Host results are written when `cuvsOctexSearch` returns. The getters
are `cuvsOctexIndexGetSize`, `cuvsOctexIndexGetDim`, `cuvsOctexIndexGetCopies`, `cuvsOctexIndexGetNumDevices`,
`cuvsOctexIndexGetDevices`, and `cuvsOctexIndexGetDeviceBytes` (one entry per GPU).

The test `octex_c_test` (`c/tests/neighbors/octex_c.cu`, whose driver `c/tests/neighbors/run_octex_c.c` compiles
as C) builds small indexes through the API and checks their recall against exhaustive search. `examples/c/search.c`
is a complete C program on this API (build, load, search on device buffers from `cudaMalloc`, and the recall on a
sample of queries); the C++ build compiles it as `build/octex_example_c`. The Python wheel does not include the C
API.

## Build parameters

| parameter | default | what it trades |
|---|---|---|
| `copies` | 200 | Randomly projected copies. More copies raise recall at a given `read`, and every copy costs memory and search time. |
| `blocks` | 4 | E8 blocks per copy (each copy projects into `8 * blocks` dimensions). More blocks make cells finer and lists more selective. |
| `pca_dims` | 64 | Principal directions that the projections rotate. Raise it for data whose variance is spread over many directions. |
| `spacings` | 2.0 ... 5.0 | Cell spacings, fine to coarse. A query reads the finest level whose cell has a list, so coarse levels catch queries in empty regions. |
| `list_len` | 64 | Vectors stored per cell. It is the upper bound of `read`, and list memory grows linearly in it. |
| `min_cell` | 1 | Cells smaller than this get no list (except at the coarsest level). Raising it saves memory on sparse levels. |
| `log_buckets` | 20 | log2 of the hash buckets per copy and level. It sets the fixed CSR size per copy. |
| `sketch_width` | 0 (auto) | int8 sketch coordinates per vector (a multiple of 16, at most 512). 0 takes the dimension rounded down to a multiple of 16, at most 128. Wider sketches rank candidates better before the exact rerank; narrower ones save memory at billion scale. |
| `table_len` | 64 | Neighbor-table row length. The waves expand these rows, and 0 builds no table (first round only). |
| `reverse_edges` | true | Merge reverse edges into the table. This helps vectors that are no one's forward neighbor. |
| `exact_lists`, `list_probes` | false, 16 | Exact (brute force) or IVF-Flat list construction, and its probes. This is build time against list quality. |
| `member_lists` | false | Cell lists from the members of each cell: vectors sorted by their distance to the center of their cell, the `list_len` nearest kept. No IVF-Flat index is built, which makes billion-scale lists about 40x faster. Use it when every listed cell holds at least `list_len` vectors (`min_cell >= list_len`). |
| `exact_table_max` | 20,000,000 | Up to this many vectors the table is exact (brute force), and above it octex builds the table by its own search. |
| `table_passes` | 3 | Search passes of a searched table (1 to 3): the first round alone, then two and three waves over the previous rows. |
| `table_sketch_order` | false | The table searches keep their rows in sketch order instead of reranking them with the fp16 vectors (no host reads at billion scale). |
| `write_behind` | false | Keep the lists, keys, offsets, and sketch in host memory and write them while the table is built. It costs about their size in extra host memory. |
| `seed` | 0 | Seed of the projections and shifts. |
| `devices` | all visible | GPUs used by the build, and in Python also the GPUs that the built index is loaded on. |

The dimension must be even.

## Building at billion scale

`octex_build --fast` sets `member_lists`, `table_passes = 2`, `table_sketch_order`, and `write_behind`, hard-links an
fp16 `.npy` base as `vectors.npy` (same file system) instead of rewriting it, and page-locks the base in place for the
table build instead of copying it into new pinned memory. It builds the Deep1B index of the paper in 44 minutes and
SIFT1B in 47 minutes on four H100 GPUs:

```
octex_build base_fp16.npy deep1B_index --copies 256 --list-len 16 --min-cell 100 --sketch-width 96 \
    --table-len 32 --fast
```

The member lists need cells that hold at least `list_len` vectors, as `min_cell 100` with `list_len 16` gives at one
billion vectors; datasets with sparse cells keep the default IVF-Flat lists. Host memory: the fp16 base, the forward
and merged table rows (`4 * 1.5 * table_len` and `4 * table_len` bytes per vector), and with `write_behind` the lists
and the sketch: the builds above peak at 677 GB (Deep1B) and 721 GB (SIFT1B) of resident host memory.

## Load parameters and placements

| parameter | default | effect |
|---|---|---|
| `devices` | all visible | GPUs that hold the index. |
| `placement` | `"shard"` | `"shard"` splits the copies over the GPUs. Every GPU runs the first round for all queries over its copies, and the waves for its slice of the queries, reading the other GPUs through peer pointers. `"replicate"` gives every GPU the whole index and a slice of the queries, with no peer traffic. |
| `vectors_on_host` | false | Keep the fp16 vectors in pinned host memory. The exact rerank then reads them over PCIe, which saves `2 * D` bytes per vector on every GPU. |
| `shard_vectors` | false | Split the fp16 vectors by rows over the GPUs instead of a full copy per GPU. |
| `table_on_host` | false | Keep the neighbor table in pinned host memory (`4 * table_len` bytes per vector). |
| `sketch` | `"replicate"` | Where the int8 sketch lives: `"replicate"` (a copy per GPU), `"shard"` (row ranges over the GPUs, read through peer pointers), or `"host"` (pinned host memory, read over PCIe). |
| `parts` | 0 | The copies were saved in this many consecutive parts. Shard placement deals the parts to the GPUs round robin. |
| `copies_per_part`, `copies_limit` | 0 | Load only the first copies of every part, or of the whole index (0: all). |
| `list_limit` | 0 | Keep only the first entries of every cell list (0: all). |

`copies_limit` and `list_limit` load a smaller index from the same directory, so the effect of fewer copies or
shorter lists can be measured without a rebuild. Shard placement needs at least as many copies (or parts) as GPUs.

## Search parameters

| parameter | default | what it trades |
|---|---|---|
| `read` | 32 | List entries read per copy (at most `list_len`). Candidates grow as `copies * read`. This is the main recall/speed knob of the first round. |
| `starts` | 128 | Best first-round hits kept as start points of the waves (`max(starts, k) <= 512`). |
| `waves` | 2 | Rounds over the neighbor table. 0 runs the first round only. Each wave expands `starts` rows of `table_len` ids. |
| `rerank` | 128 | Sketch-nearest candidates of the first round that get an exact fp16 distance (`<= 512`, and `max(starts, k) + rerank <= 1024`). |
| `wave_rerank` | 128 | The same for every wave. |
| `batch` | 10,000 | Queries per captured CUDA graph. Larger batches use the GPU better and need more workspace; the last partial batch is padded. |
| `sketch_rank` | false | Billion-scale mode: rank start points by the sketch and rerank exactly only at the end. It reads fewer fp16 vectors, which matters when they live on the host. |
| `final_rerank` | 32 | Sketch-rank mode: kept entries reranked exactly at the end (`k <= final_rerank <= 512`). |

Limits that the search checks: `copies_per_gpu * min(read, list_len) * 1.5 <= 32768` (the per-query hash set lives in
shared memory), `starts * table_len * 1.5 <= 32768`, `starts * (waves - 1) <= 4096`, and with shard placement
`devices * max(starts, k) <= 4096`.

## Memory sizing

Per vector (N vectors of dimension D), on every GPU that holds the array:

| array | bytes per vector |
|---|---|
| fp16 vectors | `2 * D` |
| int8 sketch | `sketch_width` |
| neighbor table | `4 * table_len` |

Per copy (on the GPU that holds the copy), with L = `len(spacings)` and T = `2 ** log_buckets`:

| array | bytes per copy |
|---|---|
| cell lists | `4 * list_len * (populated cells, summed over the L levels)` |
| cell keys | `8 * L * (populated cells of the most populated level)` |
| bucket offsets | `4 * L * (T + 1)`, about 21 MB at the defaults |
| projection and shift | `4 * 8 * blocks * (D + 1)` |

A level has at most N populated cells, and coarse levels have far fewer, so the lists are bounded by
`4 * list_len * L * N` bytes per copy and are usually a small multiple of `4 * list_len * N`. The lists dominate the
index. `list_len`, `copies`, and `min_cell` are the levers. `Index.device_bytes` reports the actual bytes per GPU
after a load, and the `.npy` sizes in the index directory give the same numbers before one. With shard placement each
GPU holds `copies / devices` copies; vectors, sketch, and table are replicated unless they are sharded or kept on the
host.

## Choosing parameters for a recall target

1. Build once with generous settings (the defaults, or more `copies` and `list_len` than you expect to need). Keep a
   few thousand held-out queries with exact neighbors, for example from `cuvs.neighbors.brute_force` or from the
   ground truth of a benchmark.
2. Search with the defaults (`read=32, starts=128, waves=2`) and check that recall is above the target. If it is not,
   raise `waves` and `starts` first (cheap, no rebuild), then rebuild with a longer `table_len` or more `copies`.
3. Lower `read` step by step (32, 16, 8, ...) until recall falls under the target, and keep the last value above it.
   `read` sets the size of the first round, which dominates the search time.
4. Lower `starts`, then `waves`, the same way. With a good first round, one wave or none often suffices.
5. To save memory, reload with `copies_limit` and `list_limit` and repeat steps 2 to 4. When a smaller index meets the
   target, rebuild with those `copies` and `list_len`, or keep loading with the limits.
6. Measure throughput with the batch size of the application (`batch`): a CUDA graph is captured per batch size on
   the first search, so time the second call.

Recall depends on the data: the same parameters give different recall on different datasets, so tune on held-out
queries from the target distribution.
