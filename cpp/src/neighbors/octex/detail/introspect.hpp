/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Measurements for benchmarks: per-GPU phase times of one batch and device memory by index part.

#include <cuvs/core/cuda_fp16.hpp>

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/resources.hpp>

#include <cstdint>
#include <vector>

namespace cuvs::neighbors::octex::detail {

struct gpu_phase {
  int device           = 0;
  float first_round_ms = 0.f;  // first-round graph (replicate placement: the whole search graph)
  float waves_ms       = 0.f;  // wave graph (shard placement only)
  int64_t search_buffer_bytes = 0;  // batch buffers of this search configuration
  int64_t search_mem_delta =
    0;  // device memory taken when the configuration was set up (buffers, graphs)
};

struct gpu_memory {
  int device      = 0;
  int64_t copies  = 0;  // lists, keys, offsets, base
  int64_t sketch  = 0;
  int64_t table   = 0;
  int64_t vectors = 0;
  int64_t other   = 0;  // projections, sketch basis, pointer tables, cuBLAS workspace
};

/** One timed batch: the first min(batch, nq) queries (host or device memory), events around each
 * graph. */
auto timed_batch(raft::resources const& res,
                 search_params const& params,
                 index const& idx,
                 const float* queries,
                 int64_t nq,
                 int64_t k) -> std::vector<gpu_phase>;

/**
 * One search of the batch that the previous search of the same parameters left on the GPUs: the
 * queries are already in place on every GPU and the results stay there (the timing protocol of GPU
 * baselines that keep their queries resident). Needs nq <= batch.
 */
auto resident_search(raft::resources const& res,
                     search_params const& params,
                     index const& idx,
                     int64_t nq,
                     int64_t k) -> void;

/** Device memory per GPU by index part, as loaded. */
auto memory_report(index const& idx) -> std::vector<gpu_memory>;

}  // namespace cuvs::neighbors::octex::detail
