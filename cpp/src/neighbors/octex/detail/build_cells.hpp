/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Internal entry points of the index build, exported for the tests: the cell structures of given
// projections, the PCA, and the projections themselves.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <raft/core/resources.hpp>

#include <cstdint>
#include <vector>

namespace cuvs::neighbors::octex::detail {

/** What the cell structures depend on (a subset of index_params). */
struct cell_params {
  int blocks = 4;
  std::vector<float> spacings{2.0f, 2.5f, 3.2f, 4.0f, 5.0f};
  int list_len     = 64;
  int min_cell     = 1;
  int log_buckets  = 20;
  bool exact_lists  = false;
  int list_probes   = 16;
  bool member_lists = false;  // lists from the members of each cell (index_params::member_lists)
};

/** The cell structures of one copy, all levels. */
struct copy_cells {
  std::vector<int32_t> lists;  // [rows * l]: the lists of all levels, nearest first
  std::vector<int64_t> keys;   // the cell hashes of all levels, each level in (bucket, hash) order
  std::vector<int32_t> offsets;  // [L * (T + 1)]: CSR over buckets, per level
  std::vector<int32_t> base;     // [L]: first cell row of every level
  std::vector<int64_t> cells;    // [L]: populated cells per level
};

/**
 * Build the cells of copies [c0, c1) on the current device from given projections.
 *
 * Two paths with the same output: the whole projection [n, 8 * blocks] on the device, or, when
 * that does not fit in 60% of the free device memory, a streaming path (hashes of all levels in
 * one pass over the host rows, centers from the first members, lists by one index per row range
 * merged to the best l). OCTEX_BUILD_STREAM=1 / =0 forces a path, OCTEX_BUILD_RANGES the number
 * of row ranges of the streaming path.
 *
 * @param x host fp16 dataset [n, d]
 * @param mean [d]; proj [C, d, 8 * blocks]; shift [C, 8 * blocks] (host, indexed by copy)
 * @param out out[c - c0] receives copy c
 */
void build_cells(raft::resources const& res,
                 cell_params const& params,
                 const half* x,
                 int64_t n,
                 int64_t d,
                 const float* mean,
                 const float* proj,
                 const float* shift,
                 int c0,
                 int c1,
                 std::vector<copy_cells>& out);

}  // namespace cuvs::neighbors::octex::detail
