/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>

#include "npy.hpp"

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/error.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cuvs::neighbors::octex::detail {

/** Device memory owned on one GPU (freed on that GPU). */
struct dbuf {
  void* ptr   = nullptr;
  size_t size = 0;
  int device  = -1;

  dbuf() = default;
  dbuf(int dev, size_t bytes) : size(bytes), device(dev)
  {
    if (bytes == 0) return;
    RAFT_CUDA_TRY(cudaSetDevice(dev));
    RAFT_CUDA_TRY(cudaMalloc(&ptr, bytes));
  }
  dbuf(dbuf const&)                    = delete;
  auto operator=(dbuf const&) -> dbuf& = delete;
  dbuf(dbuf&& o) noexcept { *this = std::move(o); }
  auto operator=(dbuf&& o) noexcept -> dbuf&
  {
    std::swap(ptr, o.ptr);
    std::swap(size, o.size);
    std::swap(device, o.device);
    return *this;
  }
  ~dbuf()
  {
    if (ptr != nullptr) {
      int cur = 0;
      cudaGetDevice(&cur);
      cudaSetDevice(device);
      cudaFree(ptr);
      cudaSetDevice(cur);
    }
  }
  template <typename T>
  [[nodiscard]] auto as() const -> T*
  {
    return static_cast<T*>(ptr);
  }
};

/** Pinned, mapped host memory readable by kernels on every GPU. */
struct hbuf {
  void* ptr   = nullptr;
  size_t size = 0;

  hbuf() = default;
  explicit hbuf(size_t bytes) : size(bytes)
  {
    if (bytes > 0)
      RAFT_CUDA_TRY(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
  }
  hbuf(hbuf const&)                    = delete;
  auto operator=(hbuf const&) -> hbuf& = delete;
  hbuf(hbuf&& o) noexcept { *this = std::move(o); }
  auto operator=(hbuf&& o) noexcept -> hbuf&
  {
    std::swap(ptr, o.ptr);
    std::swap(size, o.size);
    return *this;
  }
  ~hbuf()
  {
    if (ptr != nullptr) cudaFreeHost(ptr);
  }
};

/**
 * An array read by kernels in place from host memory: an anonymous pinned copy
 * (cudaHostAlloc) filled in chunks via ordinary buffered file I/O.
 */
struct host_array {
  std::unique_ptr<npy_file> file;
  bool registered = false;
  hbuf copy;
  const void* ptr = nullptr;

  host_array()                                = default;
  host_array(host_array const&)               = delete;
  host_array(host_array&&)                    = default;
  auto operator=(host_array&&) -> host_array& = default;
  ~host_array()
  {
    if (registered && file) cudaHostUnregister(file->map_);
  }
};

/** The part of the index on one GPU. */
struct device_part {
  int device     = 0;
  int copies     = 0;  // copies held here (shard: a range of the copies; replicate: all)
  int copy0      = 0;  // first copy
  int64_t stride = 0;  // entries per copy in lists
  int64_t mmax   = 0;  // keys per (copy, level)
  dbuf lists, keys, offsets,
    base;          // the copies: i32 [C, stride], i64 [C, L, mmax], i32 [C, L, T + 1], i32 [C, L]
  dbuf ucat;       // f16 [D, C * 8b]: projections of all copies side by side
  dbuf shift;      // f16 [C * 8b]
  dbuf inv_f;      // f32 [L]: inverse spacings
  dbuf mean;       // f32 [D]
  dbuf basis;      // f32 [D, S]
  dbuf scale;      // f32 [S]
  dbuf sketch;     // i8 [N, S] (replicated) or this GPU's row range (sharded); empty on the host
  dbuf vectors;    // f16 rows (all, or this GPU's row range), empty when on the host
  dbuf table;      // i32 rows (all, or this GPU's row range), empty when on the host
  dbuf xptr;       // i64 [*]: pointers of the vector row ranges
  dbuf tptr;       // i64 [*]: pointers of the table row ranges
  dbuf cptr;       // i64 [*]: pointers of the sketch row ranges (sharded or host sketch)
  dbuf workspace;  // cuBLAS workspace
  cudaStream_t stream   = nullptr;
  cublasHandle_t cublas = nullptr;
  int64_t bytes         = 0;

  device_part()                   = default;
  device_part(device_part const&) = delete;
  ~device_part()
  {
    if (cublas != nullptr) {
      cudaSetDevice(device);
      cublasDestroy(cublas);
    }
    if (stream != nullptr) {
      cudaSetDevice(device);
      cudaStreamDestroy(stream);
    }
  }
};

struct plan;  // captured search graphs and buffers of one (batch, search parameters, k); search.cu

struct index_impl {
  // meta.json
  int64_t n = 0, dim = 0;
  int copies = 0, blocks = 4, list_len = 0, log_buckets = 20, sketch_width = 0, table_len = 0;
  std::vector<float> spacings;
  std::string path;

  load_params params;
  std::vector<int> devices;
  std::vector<std::unique_ptr<device_part>> parts;
  host_array host_vectors, host_table, host_sketch;
  int64_t xrows = 0;  // rows per vector range behind xptr
  int64_t trows = 0;  // rows per table range behind tptr
  int64_t crows = 0;  // rows per sketch range behind cptr (0: a full sketch on every GPU)

  std::mutex mu;
  std::map<std::string, std::shared_ptr<plan>> plans;

  index_impl() = default;
  ~index_impl();
};

/**
 * Neighbor rows [n, k] (host) into the device table of a loaded index (allocated on first use, row
 * ranges as load places them); captured plans are dropped. For the table build, which keeps one
 * loaded index across its passes.
 */
void set_table(index const& idx, const int32_t* rows, int k);
/** Free the device table of a loaded index (table_len 0); captured plans are dropped. */
void drop_table(index const& idx);

}  // namespace cuvs::neighbors::octex::detail
