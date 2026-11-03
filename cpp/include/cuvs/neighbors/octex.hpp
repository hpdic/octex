/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cuvs/core/export.hpp>

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors::octex {

/**
 * @defgroup octex_cpp_index_params octex index build parameters
 * @{
 */

/** @brief Parameters of an octex index (squared Euclidean distance). */
struct index_params : cuvs::neighbors::index_params {
  /** Randomly projected copies of the dataset. */
  int copies = 200;
  /** Eight-dimensional E8 blocks per copy: each copy projects into 8 * blocks dimensions. */
  int blocks = 4;
  /** Principal directions that the random projections rotate. */
  int pca_dims = 64;
  /** Cell spacings in projected standard deviations, fine to coarse. */
  std::vector<float> spacings = {2.0f, 2.5f, 3.2f, 4.0f, 5.0f};
  /** Vectors stored per populated cell (nearest first). */
  int list_len = 64;
  /** Lists only for cells holding at least this many vectors (all spacings but the coarsest). */
  int min_cell = 1;
  /** log2 of the hash buckets per copy and spacing. */
  int log_buckets = 20;
  /**
   * int8 sketch coordinates per vector (top principal directions). 0 picks the dimension rounded
   * down to a multiple of 16, at most 128: a sketch that keeps most of the variance ranks the
   * candidates well enough for the default rerank depth. A narrower sketch saves memory at billion
   * scale.
   */
  int sketch_width = 0;
  /** Neighbor-table row length; 0 builds no table (first round only). */
  int table_len = 64;
  /** Merge reverse edges into the neighbor table. */
  bool reverse_edges = true;
  /** Exact cell lists (brute force) instead of IVF-Flat on the projection. */
  bool exact_lists = false;
  /** IVF-Flat probes of the list construction. */
  int list_probes = 16;
  /**
   * Cell lists from the members of each cell: every vector is sorted by its distance to the center
   * of its cell, and a cell keeps its list_len nearest members. No index over the projection is
   * built, which makes the lists of a billion vectors about 40x faster to build than with
   * IVF-Flat; a cell with fewer than list_len members repeats its nearest one (min_cell >=
   * list_len avoids that).
   */
  bool member_lists = false;
  /** Up to this many vectors the neighbor table is exact (brute force). */
  int64_t exact_table_max = 20000000;
  /**
   * Search passes of a searched neighbor table, 1 to 3: the first round alone, then 2 and 3 waves
   * over the rows of the previous pass.
   */
  int table_passes = 3;
  /**
   * The searches of a searched table keep their rows in sketch order instead of reranking them
   * with the full vectors, which they would read from host memory at billion scale.
   */
  bool table_sketch_order = false;
  /**
   * The lists, keys, offsets, and sketch stay in host memory for the neighbor table while a
   * background thread writes them, so that the writes overlap the table passes (more host memory
   * at billion scale).
   */
  bool write_behind = false;
  /** Seed of the projections and shifts. */
  uint64_t seed = 0;
  /** GPUs used by the build (empty: all visible). */
  std::vector<int> devices{};
};

/** @} */

/**
 * @defgroup octex_cpp_search_params octex search parameters
 * @{
 */

struct search_params : cuvs::neighbors::search_params {
  /** List entries read per copy in the first round. */
  int read = 32;
  /** Best first-round hits kept as start points of the waves. */
  int starts = 128;
  /** Secondary waves over the neighbor table (0: first round only). */
  int waves = 2;
  /** Exact rerank depth of the first round (sketch-nearest candidates). */
  int rerank = 128;
  /** Exact rerank depth of every wave. */
  int wave_rerank = 128;
  /** Queries per captured batch. */
  int batch = 10000;
  /** Rank start points by the sketch and rerank exactly only at the end (billion scale). */
  bool sketch_rank = false;
  /** Sketch-rank mode: kept entries reranked exactly at the end (0: the sketch order, no rerank). */
  int final_rerank = 32;
  /**
   * Ablation: the gather splits the entries of all lists of a query evenly over the block (false:
   * every warp walks whole lists, one fragment list or neighbor row at a time).
   */
  bool balanced_gather = true;
  /**
   * Ablation: the kept list and the start points are merged by fused kernels in shared memory
   * (false: the candidate keys are sorted in global memory by CUB, and separate launches take the
   * kept list and the start points).
   */
  bool fused_merge = true;
  /**
   * Ablation: every batch replays CUDA graphs ordered by device-side events (false: direct
   * launches, with a host synchronization of every GPU between the first round and the waves).
   */
  bool use_graph = true;
};

/** Where the parts of an index live. */
enum class placement : int {
  /** Every GPU holds the whole index and answers a slice of the queries. */
  replicate = 0,
  /** Copies and table rows split over the GPUs, read across GPUs through peer pointers. */
  shard = 1,
};

/** Where the int8 sketch (N x sketch_width bytes) lives. */
enum class sketch_placement : int {
  /** A full copy on every GPU. */
  replicate = 0,
  /** Row ranges split over the GPUs, read through peer pointers (billion scale). */
  shard = 1,
  /** Pinned host memory, read in place over PCIe. */
  host = 2,
};

struct load_params {
  /** GPUs that hold the index (empty: all visible). */
  std::vector<int> devices{};
  placement mode = placement::shard;
  /** Keep the fp16 vectors in pinned host memory (exact rerank over PCIe). */
  bool vectors_on_host = false;
  /** Split the fp16 vectors by rows over the GPUs instead of replicating them. */
  bool shard_vectors = false;
  /** Keep the neighbor table in pinned host memory. */
  bool table_on_host = false;
  /** Where the int8 sketch lives. */
  sketch_placement sketch = sketch_placement::replicate;
  /** The copies were saved in this many equal consecutive parts (0: one). Shard placement deals the
   * parts to the GPUs round robin (one part: equal consecutive ranges). */
  int parts = 0;
  /** Load only the first copies_per_part copies of every part (0: all). */
  int copies_per_part = 0;
  /** Load only the first copies_limit copies (0: all; one part only). */
  int copies_limit = 0;
  /** Keep only the first list_limit entries of every cell list (0: all). */
  int list_limit = 0;
};

/** @} */

namespace detail {
struct index_impl;
}

/**
 * @brief An octex index placed on one or more GPUs.
 */
class index : public cuvs::neighbors::index {
 public:
  explicit index(std::shared_ptr<detail::index_impl> impl);
  index(index&&) noexcept;
  auto operator=(index&&) noexcept -> index&;
  ~index();

  [[nodiscard]] auto size() const noexcept -> int64_t;
  [[nodiscard]] auto dim() const noexcept -> int64_t;
  [[nodiscard]] auto copies() const noexcept -> int;
  [[nodiscard]] auto devices() const noexcept -> std::vector<int> const&;
  /** Device memory per GPU, in bytes. */
  [[nodiscard]] auto device_bytes() const -> std::vector<int64_t>;
  [[nodiscard]] auto impl() const noexcept -> detail::index_impl&;

 private:
  std::shared_ptr<detail::index_impl> impl_;
};

/**
 * @brief Build an index of a dataset and write it to a directory.
 *
 * @param[in] res raft resources
 * @param[in] params index parameters
 * @param[in] dataset host matrix [N, D]
 * @param[in] path index directory (created)
 */
void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           std::string const& path);

void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           std::string const& path);

/** @brief Load an index directory onto the GPUs. */
auto load(raft::resources const& res, std::string const& path, load_params const& params = {})
  -> index;

/**
 * @brief Search the k nearest neighbors of a batch of queries.
 *
 * Queries, neighbors and distances may live in host or device memory (any GPU); results are squared
 * distances.
 *
 * @param[in] res raft resources
 * @param[in] params search parameters
 * @param[in] idx the index
 * @param[in] queries [n_queries, D] row major
 * @param[out] neighbors [n_queries, k]
 * @param[out] distances [n_queries, k]
 */
void search(raft::resources const& res,
            search_params const& params,
            index const& idx,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& res,
            search_params const& params,
            index const& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

}  // namespace neighbors::octex
}  // namespace CUVS_EXPORT cuvs
