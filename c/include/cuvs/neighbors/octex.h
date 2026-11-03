/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup octex_c_index_params octex C-API index build parameters
 * @{
 */

/**
 * @brief Parameters of an octex index (cuvs::neighbors::octex::index_params), squared Euclidean
 * distance.
 *
 * The arrays `spacings` and `devices` belong to the caller and are read only during cuvsOctexBuild;
 * a NULL array (or a count of 0) keeps the default of the C++ API.
 */
struct cuvsOctexIndexParams {
  /** Randomly projected copies of the dataset */
  int32_t copies;
  /** Eight-dimensional E8 blocks per copy: each copy projects into 8 * blocks dimensions */
  int32_t blocks;
  /** Principal directions that the random projections rotate */
  int32_t pca_dims;
  /** Cell spacings in projected standard deviations, fine to coarse (NULL: 2.0, 2.5, 3.2, 4.0, 5.0)
   */
  const float* spacings;
  /** Number of entries of `spacings` */
  int32_t n_spacings;
  /** Vectors stored per populated cell (nearest first) */
  int32_t list_len;
  /** Lists only for cells holding at least this many vectors (all spacings but the coarsest) */
  int32_t min_cell;
  /** log2 of the hash buckets per copy and spacing */
  int32_t log_buckets;
  /** int8 sketch coordinates per vector (0: the dimension rounded down to a multiple of 16, at most
   * 128) */
  int32_t sketch_width;
  /** Neighbor-table row length; 0 builds no table (first round only) */
  int32_t table_len;
  /** Merge reverse edges into the neighbor table */
  bool reverse_edges;
  /** Exact cell lists (brute force) instead of IVF-Flat on the projection */
  bool exact_lists;
  /** IVF-Flat probes of the list construction */
  int32_t list_probes;
  /** Cell lists from the members of each cell, without an index (billion-scale builds) */
  bool member_lists;
  /** Up to this many vectors the neighbor table is exact (brute force) */
  int64_t exact_table_max;
  /** Search passes of a searched neighbor table, 1 to 3 */
  int32_t table_passes;
  /** Searched-table rows in sketch order, without the exact rerank */
  bool table_sketch_order;
  /** Write the copies and the sketch while the neighbor table is built (more host memory) */
  bool write_behind;
  /** Seed of the projections and shifts */
  uint64_t seed;
  /** GPUs used by the build (NULL: all visible) */
  const int32_t* devices;
  /** Number of entries of `devices` */
  int32_t n_devices;
};

typedef struct cuvsOctexIndexParams* cuvsOctexIndexParams_t;

/** @brief Allocate index parameters with the defaults of the C++ API. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexParamsCreate(cuvsOctexIndexParams_t* params);

/** @brief Free index parameters. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexParamsDestroy(cuvsOctexIndexParams_t params);

/**
 * @}
 */

/**
 * @defgroup octex_c_load_params octex C-API load parameters
 * @{
 */

/** @brief Where the parts of an index live. */
typedef enum {
  /** Every GPU holds the whole index and answers a slice of the queries */
  OCTEX_PLACEMENT_REPLICATE = 0,
  /** Copies and table rows split over the GPUs, read across GPUs through peer pointers */
  OCTEX_PLACEMENT_SHARD = 1
} cuvsOctexPlacement;

/** @brief Where the int8 sketch lives. */
typedef enum {
  /** A full copy on every GPU */
  OCTEX_SKETCH_REPLICATE = 0,
  /** Row ranges split over the GPUs, read through peer pointers (billion scale) */
  OCTEX_SKETCH_SHARD = 1,
  /** Pinned host memory, read in place over PCIe */
  OCTEX_SKETCH_HOST = 2
} cuvsOctexSketchPlacement;

/**
 * @brief Parameters of a load (cuvs::neighbors::octex::load_params). The array `devices` belongs to
 * the caller and is read only during cuvsOctexLoad.
 */
struct cuvsOctexLoadParams {
  /** GPUs that hold the index (NULL: all visible) */
  const int32_t* devices;
  /** Number of entries of `devices` */
  int32_t n_devices;
  /** Copies sharded over the GPUs or the whole index on every GPU */
  cuvsOctexPlacement placement;
  /** Keep the fp16 vectors in pinned host memory (exact rerank over PCIe) */
  bool vectors_on_host;
  /** Split the fp16 vectors by rows over the GPUs instead of replicating them */
  bool shard_vectors;
  /** Keep the neighbor table in pinned host memory */
  bool table_on_host;
  /** Where the int8 sketch lives */
  cuvsOctexSketchPlacement sketch;
  /** The copies were saved in this many equal consecutive parts (0: one) */
  int32_t parts;
  /** Load only the first copies_per_part copies of every part (0: all) */
  int32_t copies_per_part;
  /** Load only the first copies_limit copies (0: all; one part only) */
  int32_t copies_limit;
  /** Keep only the first list_limit entries of every cell list (0: all) */
  int32_t list_limit;
};

typedef struct cuvsOctexLoadParams* cuvsOctexLoadParams_t;

/** @brief Allocate load parameters with the defaults of the C++ API (all visible GPUs, shard
 * placement). */
CUVS_EXPORT cuvsError_t cuvsOctexLoadParamsCreate(cuvsOctexLoadParams_t* params);

/** @brief Free load parameters. */
CUVS_EXPORT cuvsError_t cuvsOctexLoadParamsDestroy(cuvsOctexLoadParams_t params);

/**
 * @}
 */

/**
 * @defgroup octex_c_search_params octex C-API search parameters
 * @{
 */

/** @brief Parameters of an octex search (cuvs::neighbors::octex::search_params). */
struct cuvsOctexSearchParams {
  /** List entries read per copy in the first round */
  int32_t read;
  /** Best first-round hits kept as start points of the waves */
  int32_t starts;
  /** Secondary waves over the neighbor table (0: first round only) */
  int32_t waves;
  /** Exact rerank depth of the first round (sketch-nearest candidates) */
  int32_t rerank;
  /** Exact rerank depth of every wave */
  int32_t wave_rerank;
  /** Queries per captured batch */
  int32_t batch;
  /** Rank start points by the sketch and rerank exactly only at the end (billion scale) */
  bool sketch_rank;
  /** Sketch-rank mode: kept entries reranked exactly at the end */
  int32_t final_rerank;
  /** Ablation: load-balanced gather (false: every warp walks whole lists) */
  bool balanced_gather;
  /** Ablation: fused merge kernels (false: CUB sorts in global memory and separate launches) */
  bool fused_merge;
  /** Ablation: one CUDA graph per batch (false: direct launches with host synchronization) */
  bool use_graph;
};

typedef struct cuvsOctexSearchParams* cuvsOctexSearchParams_t;

/** @brief Allocate search parameters with the defaults of the C++ API. */
CUVS_EXPORT cuvsError_t cuvsOctexSearchParamsCreate(cuvsOctexSearchParams_t* params);

/** @brief Free search parameters. */
CUVS_EXPORT cuvsError_t cuvsOctexSearchParamsDestroy(cuvsOctexSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup octex_c_index octex C-API index
 * @{
 */

/** @brief An opaque octex index placed on one or more GPUs. */
typedef struct {
  uintptr_t addr;
} cuvsOctexIndex;

typedef cuvsOctexIndex* cuvsOctexIndex_t;

/** @brief Allocate an empty index handle. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexCreate(cuvsOctexIndex_t* index);

/** @brief Free an index handle and its index (the device memory on every GPU). */
CUVS_EXPORT cuvsError_t cuvsOctexIndexDestroy(cuvsOctexIndex_t index);

/** @brief Number of vectors of the index. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexGetSize(cuvsOctexIndex_t index, int64_t* size);

/** @brief Dimension of the vectors of the index. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexGetDim(cuvsOctexIndex_t index, int64_t* dim);

/** @brief Number of copies loaded. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexGetCopies(cuvsOctexIndex_t index, int32_t* copies);

/** @brief Number of GPUs that hold the index. */
CUVS_EXPORT cuvsError_t cuvsOctexIndexGetNumDevices(cuvsOctexIndex_t index, int32_t* n_devices);

/** @brief The GPUs that hold the index: `devices` has room for cuvsOctexIndexGetNumDevices entries.
 */
CUVS_EXPORT cuvsError_t cuvsOctexIndexGetDevices(cuvsOctexIndex_t index, int32_t* devices);

/** @brief Device memory per GPU in bytes: `bytes` has room for cuvsOctexIndexGetNumDevices entries.
 */
CUVS_EXPORT cuvsError_t cuvsOctexIndexGetDeviceBytes(cuvsOctexIndex_t index, int64_t* bytes);

/**
 * @}
 */

/**
 * @defgroup octex_c_index_build octex C-API build and load
 * @{
 */

/**
 * @brief Build an index of a dataset [n_rows, dim] of float32 or float16 and write it to the
 * directory `path` (created). A dataset in device memory is copied to the host first; the build
 * places its own work on the GPUs of `params->devices`. The dimension must be even.
 *
 * @code{.c}
 *   cuvsResources_t res;
 *   cuvsResourcesCreate(&res);
 *   cuvsOctexIndexParams_t params;
 *   cuvsOctexIndexParamsCreate(&params);
 *   cuvsOctexBuild(res, params, &dataset, "index_dir");
 *   cuvsOctexIndex_t index;
 *   cuvsOctexIndexCreate(&index);
 *   cuvsOctexLoad(res, "index_dir", NULL, index);
 * @endcode
 */
CUVS_EXPORT cuvsError_t cuvsOctexBuild(cuvsResources_t res,
                                       cuvsOctexIndexParams_t params,
                                       DLManagedTensor* dataset,
                                       const char* path);

/** @brief Load an index directory onto the GPUs into an empty index handle (`params` NULL: the
 * defaults). */
CUVS_EXPORT cuvsError_t cuvsOctexLoad(cuvsResources_t res,
                                      const char* path,
                                      cuvsOctexLoadParams_t params,
                                      cuvsOctexIndex_t index);

/**
 * @}
 */

/**
 * @defgroup octex_c_index_search octex C-API search
 * @{
 */

/**
 * @brief Approximate k nearest neighbors of every query, k = the second extent of `neighbors`.
 *
 * queries [n_queries, dim] float32, neighbors [n_queries, k] int64, and distances [n_queries, k]
 * float32 (squared Euclidean distances, nearest first). The three tensors live either all in host
 * memory (CPU or pinned) or all in device memory (any GPU). Host results are written when the call
 * returns; device results are ordered on the stream of `res` (cuvsStreamSync waits for them).
 */
CUVS_EXPORT cuvsError_t cuvsOctexSearch(cuvsResources_t res,
                                        cuvsOctexSearchParams_t params,
                                        cuvsOctexIndex_t index,
                                        DLManagedTensor* queries,
                                        DLManagedTensor* neighbors,
                                        DLManagedTensor* distances);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
