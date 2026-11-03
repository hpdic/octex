# Changelog

All notable changes to octex are recorded here. The format follows Keep a Changelog, and versions follow semantic
versioning.

## [Unreleased]

## [0.1.0] - 2026-11-02

The first public release, with the paper (arXiv:2611.02556).

### Added
- The CUDA C++ library as a cuVS module (`cuvs::neighbors::octex`): `build` writes an index directory, `load` places
  it on one or more GPUs, and `search` answers a batch of queries with one CUDA graph per GPU, on host or device
  arrays.
- Placements: copies sharded over the GPUs or the whole index replicated, the fp16 vectors, the neighbor table, and
  the int8 sketch in device or pinned host memory, and a sketch-rank search mode for billion-scale indexes.
- The Python package `octex`: `octex.build`, `octex.load`, and `octex.Index.search` on numpy, CuPy, PyTorch, and any
  array with DLPack or `__cuda_array_interface__`, without copies, with results in the framework of the queries.
- Adapters with the calls of cuVS CAGRA (`from octex import cuvs as cagra`) and Faiss GPU indexes
  (`from octex import faiss`).
- The C API (`cuvsOctex*`, `c/include/cuvs/neighbors/octex.h`) with the conventions of the cuVS C API: build, load,
  and search on DLPack tensors in host or device memory, with the library `liboctex_c`, its test, and the example
  `examples/c/search.c`.
- A billion-scale build (`octex_build --fast`; `index_params::member_lists`, `table_passes`, `table_sketch_order`,
  `write_behind`): cell lists by sorting the members of every cell, two sketch-order passes of the neighbor table on
  one resident index, and index files written in the background; Deep1B builds in 44 minutes on four H100 GPUs.
- Examples (`examples/python/quickstart.py`, `examples/cpp/search.cu`), a user guide (`docs/guide.md`), and the
  design and index format (`docs/design.md`).
