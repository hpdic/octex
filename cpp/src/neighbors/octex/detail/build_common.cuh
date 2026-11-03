/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Shared pieces of the index build: progress output, timers, the fp16 dataset on a device (resident
// or streamed from host memory in chunks), one host thread per GPU.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace cuvs::neighbors::octex::detail {

/** Progress lines on stderr when OCTEX_VERBOSE is set. */
inline auto build_verbose() -> bool
{
  static const bool v = std::getenv("OCTEX_VERBOSE") != nullptr;
  return v;
}

inline void build_log(const char* fmt, ...)
{
  if (!build_verbose()) return;
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fflush(stderr);
}

inline auto seconds_since(std::chrono::steady_clock::time_point t0) -> double
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/**
 * The stream-ordered CUDA allocator as the current device resource while the scope lives (cuVS
 * allocates and frees per call; plain cudaMalloc/cudaFree would dominate the build). The previous
 * resource returns at the end.
 */
class async_memory_scope {
 public:
  async_memory_scope()
    : old_(rmm::mr::set_current_device_resource(rmm::mr::cuda_async_memory_resource{}))
  {
  }
  async_memory_scope(async_memory_scope const&)                    = delete;
  auto operator=(async_memory_scope const&) -> async_memory_scope& = delete;
  ~async_memory_scope() { rmm::mr::set_current_device_resource(std::move(old_)); }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> old_;
};

/** Run f(rank, device) on one host thread per device; the first exception is rethrown. */
inline void per_device(std::vector<int> const& devices, std::function<void(int, int)> const& f)
{
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(devices.size());
  for (size_t r = 0; r < devices.size(); r++) {
    threads.emplace_back([&, r] {
      try {
        RAFT_CUDA_TRY(cudaSetDevice(devices[r]));
        async_memory_scope scope;
        f(static_cast<int>(r), devices[r]);
      } catch (...) {
        errors[r] = std::current_exception();
      }
    });
  }
  for (auto& t : threads)
    t.join();
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
}

/** f(i0, i1) over [0, n) split across the host cores. */
inline void parallel_for(int64_t n, std::function<void(int64_t, int64_t)> const& f)
{
  int64_t t =
    std::max<int64_t>(1, std::min<int64_t>(std::thread::hardware_concurrency(), n / 65536 + 1));
  std::vector<std::thread> threads;
  for (int64_t r = 0; r < t; r++)
    threads.emplace_back([&, r] { f(n * r / t, n * (r + 1) / t); });
  for (auto& th : threads)
    th.join();
}

/** Free device memory of the current device, in bytes. */
inline auto device_free_bytes() -> size_t
{
  size_t free_b = 0, total_b = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));
  return free_b;
}

/**
 * The fp16 dataset on the current device: resident when it fits in `budget` bytes, else rows are
 * staged from host memory chunk by chunk.
 */
class device_rows {
 public:
  device_rows(
    const half* host, int64_t n, int64_t d, int64_t chunk, size_t budget, cudaStream_t stream)
    : host_(host), n_(n), d_(d), stream_(stream)
  {
    size_t bytes = static_cast<size_t>(n) * d * sizeof(half);
    if (bytes <= budget) {
      resident_.emplace(static_cast<size_t>(n * d), stream);
      RAFT_CUDA_TRY(
        cudaMemcpyAsync(resident_->data(), host, bytes, cudaMemcpyHostToDevice, stream));
    } else {
      stage_.emplace(static_cast<size_t>(chunk * d), stream);
    }
  }
  [[nodiscard]] auto resident() const -> bool { return resident_.has_value(); }
  [[nodiscard]] auto data() const -> const half* { return resident_ ? resident_->data() : nullptr; }
  /** Device pointer to rows [r0, r0 + rows); staged rows stay valid until the next call. */
  auto rows(int64_t r0, int64_t rows) -> const half*
  {
    if (resident_) return resident_->data() + r0 * d_;
    RAFT_EXPECTS(static_cast<size_t>(rows * d_) <= stage_->size(),
                 "octex: staging chunk too small");
    RAFT_CUDA_TRY(cudaMemcpyAsync(stage_->data(),
                                  host_ + r0 * d_,
                                  static_cast<size_t>(rows * d_) * sizeof(half),
                                  cudaMemcpyHostToDevice,
                                  stream_));
    return stage_->data();
  }

 private:
  const half* host_;
  int64_t n_, d_;
  cudaStream_t stream_;
  std::optional<rmm::device_uvector<half>> resident_;
  std::optional<rmm::device_uvector<half>> stage_;
};

/** out[i, j] = float(x[i, j]) - mu[j] (mu may be null). */
static __global__ void center_kernel(
  const half* x, const float* mu, float* out, int64_t rows, int64_t d)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= rows * d) return;
  float v = __half2float(x[i]);
  out[i]  = mu != nullptr ? v - mu[i % d] : v;
}

inline void center(
  const half* x, const float* mu, float* out, int64_t rows, int64_t d, cudaStream_t s)
{
  int64_t total = rows * d;
  if (total == 0) return;
  center_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, s>>>(x, mu, out, rows, d);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::octex::detail
