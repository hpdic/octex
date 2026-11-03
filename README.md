# octex

[![CI](https://github.com/hpdic/octex/actions/workflows/ci.yml/badge.svg)](https://github.com/hpdic/octex/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/octex)](https://pypi.org/project/octex/)
[![OctEx paper](https://img.shields.io/badge/OctEx%20paper-arXiv%3A2611.02556-b31b1b.svg)](https://arxiv.org/abs/2611.02556)

octex (Octa Explosion) is billion-scale approximate nearest-neighbor search on NVIDIA GPUs by precomputed lattice
neighbor lists. It holds hundreds of randomly projected copies of the dataset in GPU memory. Every copy rounds its
vectors to the eight-dimensional E8 lattice and stores, for every populated cell, the list of vectors nearest to its
center. A query reads the lists of its cells in all copies at once and then, in a fixed number of waves, the precomputed
neighbor lists of its best hits. An int8 sketch ranks every candidate before one exact fp16 rerank, and the whole search
of a batch runs as CUDA graphs over all GPUs of a server. octex is pure CUDA C++ laid out as a cuVS module
(`cuvs::neighbors::octex`), with a Python package that keeps the calls of cuVS and Faiss.

## Benchmarks

Throughput at recall@10 >= 0.99 on four H100 GPUs (NVLink), batches of 10,000 queries (1,000 for GIST1M, 6,980 for
MS MARCO), k = 10, against the best CAGRA configuration of cuVS 26.10 on the same GPUs at the same recall:

| Dataset | Vectors | Dimension | Queries per second | Over CAGRA |
|---|---|---|---|---|
| SIFT1M | 1M | 128 | 9.44 M | 6.3x |
| GIST1M | 1M | 960 | 1.03 M | 14.2x |
| DINOv2 (ImageNet) | 1.28M | 768 | 2.53 M | 12.8x |
| MS MARCO passages | 8.8M | 768 | 2.20 M | 32.6x |
| Deep10M | 10M | 96 | 3.82 M | 3.5x |
| Deep100M | 100M | 96 | 1.97 M | 2.3x |
| SIFT100M | 100M | 128 | 2.52 M | 2.6x |
| Deep1B | 1B | 96 | 297 k | 1.9x |
| SIFT1B | 1B | 128 | 437 k | 1.5x |

Both systems are timed with the queries resident on the GPUs, median of five runs after a warm-up. Deep1B and SIFT1B keep
their fp16 vectors in pinned host memory. The runs are `bench/paper_frontier.sh`; the paper has the full recall and
throughput fronts and the other baselines.

## Installation

```bash
pip install octex        # Linux, Python >= 3.11, CUDA 12, an NVIDIA GPU of compute capability >= 8.0
pip install .            # or from a checkout
```

The wheels on PyPI hold the CUDA library for compute capabilities 8.0 to 12.0 and take cuVS, RAFT, RMM, and the CUDA
math libraries from the `libcuvs-cu12` 26.10 wheel and its dependencies. A build from a checkout compiles the CUDA
library against the same wheel. The package needs only numpy; CuPy and PyTorch arrays are accepted when installed.

## Usage

Native:

```python
import octex

index = octex.build(xb)                       # numpy, CuPy, or PyTorch; float32 or float16
D, I = index.search(xq, k=10)                 # squared distances and int64 ids, nearest first
index.save("index_dir"); index = octex.load("index_dir", devices=[0, 1], placement="shard")
```

cuVS (`cuvs.neighbors.cagra` calls):

```python
from octex import cuvs as cagra               # was: from cuvs.neighbors import cagra

index = cagra.build(cagra.IndexParams(), dataset)
distances, neighbors = cagra.search(cagra.SearchParams(), index, queries, k=10)
```

Faiss (GPU index calls):

```python
from octex import faiss                       # was: import faiss

index = faiss.IndexOctex(d)                   # or faiss.GpuIndexCagra(faiss.StandardGpuResources(), d)
index.add(xb)
D, I = index.search(xq, 10)
```

Results are `(distances, neighbors)` everywhere, as in Faiss (`D, I`) and cuVS: float32 squared Euclidean distances
and int64 ids, nearest first, in the framework of the queries (numpy for host arrays; arrays on the same GPU for CuPy,
PyTorch, or anything with DLPack or `__cuda_array_interface__`). Inputs are read without copies. The search knobs
are `read` (list entries per copy), `starts` (first-round hits kept), and `waves` (rounds over the neighbor table):

```python
D, I = index.search(xq, k=10, read=8, starts=32, waves=1)
```

Only the squared Euclidean metric is supported; other metrics raise `ValueError`.

Billion-scale builds use `octex_build --fast`, which builds the Deep1B index in 44 minutes on four H100 GPUs
([guide](docs/guide.md#building-at-billion-scale)).

## Index directory

An index is a directory of `.npy` arrays plus `meta.json` (format in [docs/design.md](docs/design.md)). `build`
writes it, `load` places it on one or more GPUs: `placement="shard"` splits the copies and the neighbor table over
the GPUs, `placement="replicate"` gives every GPU the whole index and a slice of the queries, and
`vectors_on_host=True` keeps the fp16 vectors in pinned host memory.

## C++

```cpp
#include <cuvs/neighbors/octex.hpp>
namespace octex = cuvs::neighbors::octex;

octex::build(res, octex::index_params{}, dataset_view, "index_dir");
auto index = octex::load(res, "index_dir");
octex::search(res, octex::search_params{}, index, queries_view, neighbors_view, distances_view);
```

## Documentation and examples

- [docs/guide.md](docs/guide.md): installation, the three interfaces, every build, load, and search parameter with
  what it trades, placements, memory sizing, and how to choose parameters for a recall target.
- [docs/design.md](docs/design.md): the algorithm and the index directory format.
- [examples/python/quickstart.py](examples/python/quickstart.py): build and search, natively and through the cuVS and
  Faiss calls.
- [examples/cpp/search.cu](examples/cpp/search.cu): load an index directory and search it from C++
  (`cmake --build build --target octex_example_search`).
- [examples/c/search.c](examples/c/search.c): build, load, and search through the C API (`cuvsOctex*`, the
  conventions of the cuVS C API; `cmake --build build --target octex_example_c`).

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests
```

## Citation

```bibtex
@article{zhao2026octex,
  title   = {{OctEx}: Billion-Scale Vector Search by Precomputed Lattice Neighbor Lists on {GPUs}},
  author  = {Zhao, Dongfang},
  journal = {arXiv preprint arXiv:2611.02556},
  year    = {2026}
}
```

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
