/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The neighbor table: the k nearest neighbors of every vector (the vector itself excluded), exact
// by cuVS brute force (then reverse edges merged in, nearest first), or, above exact_table_max, by
// octex's own search over the written index directory: a first round over the lists, then waves
// over the previous table, reverse edges merged in after every pass.

#include "build_cells.cuh"
#include "build_common.cuh"
#include "index_impl.hpp"
#include "npy.hpp"

#include <cuvs/neighbors/octex.hpp>

#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <cub/device/device_radix_sort.cuh>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <fcntl.h>
#include <optional>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cuvs::neighbors::octex::detail {

constexpr int64_t kTableChunk = int64_t(1) << 16;  // query rows per brute-force call

/** Keep the first k of k + 1 neighbors that are not the row itself; distances clamped at zero. */
static __global__ void drop_self_kernel(
  const int64_t* nb, const float* dist, int64_t m, int k, int64_t r0, int32_t* ids, float* out_dist)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  int t = 0;
  for (int j = 0; j <= k && t < k; j++) {
    int64_t v = nb[i * (k + 1) + j];
    if (v == r0 + i) continue;
    float dd            = dist[i * (k + 1) + j];
    ids[i * k + t]      = v < 0 ? -1 : static_cast<int32_t>(v);
    out_dist[i * k + t] = v < 0 ? INFINITY : (dd > 0.f ? dd : 0.f);
    t++;
  }
}

/** Exact k nearest neighbors of every row, rows split over the devices. */
inline void exact_table(std::vector<int> const& devices,
                        const half* x,
                        int64_t n,
                        int64_t d,
                        int k,
                        int32_t* ids,
                        float* dist)
{
  per_device(devices, [&](int rank, int) {
    raft::device_resources res;
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    int64_t r0          = n * rank / int64_t(devices.size());
    int64_t r1          = n * (rank + 1) / int64_t(devices.size());
    size_t free         = device_free_bytes();
    bool fp32 =
      size_t(n) * d * sizeof(float) * 2 <= free;  // fp32 when it takes at most half the memory
    RAFT_EXPECTS(fp32 || size_t(n) * d * sizeof(half) * 3 <= free * 2,
                 "octex: dataset too large for the exact table on one GPU; lower exact_table_max");
    int64_t qc = std::min(kTableChunk, std::max<int64_t>(r1 - r0, 1));
    rmm::device_uvector<int64_t> nb(qc * (k + 1), stream);
    rmm::device_uvector<float> nd(qc * (k + 1), stream), od(qc * k, stream);
    rmm::device_uvector<int32_t> oi(qc * k, stream);
    cuvs::neighbors::brute_force::index_params ip;
    ip.metric = cuvs::distance::DistanceType::L2Expanded;
    auto run  = [&](auto const& index, auto const* data) {
      using T = std::remove_cv_t<std::remove_pointer_t<decltype(data)>>;
      for (int64_t q0 = r0; q0 < r1; q0 += qc) {
        int64_t m = std::min(qc, r1 - q0);
        auto qv   = raft::make_device_matrix_view<const T, int64_t>(data + q0 * d, m, d);
        cuvs::neighbors::brute_force::search(
          res,
          cuvs::neighbors::brute_force::search_params{},
          index,
          qv,
          raft::make_device_matrix_view<int64_t, int64_t>(nb.data(), m, k + 1),
          raft::make_device_matrix_view<float, int64_t>(nd.data(), m, k + 1));
        drop_self_kernel<<<grid(m), 256, 0, stream>>>(
          nb.data(), nd.data(), m, k, q0, oi.data(), od.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          ids + q0 * k, oi.data(), m * k * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          dist + q0 * k, od.data(), m * k * sizeof(float), cudaMemcpyDeviceToHost, stream));
        raft::resource::sync_stream(res);
        if (rank == 0 && ((q0 - r0) / qc) % 64 == 0)
          build_log(
            "  table: %lld / %lld rows on GPU 0\n", (long long)(q0 - r0 + m), (long long)(r1 - r0));
      }
    };
    device_rows rows(x, n, d, kProjectChunk, fp32 ? 0 : SIZE_MAX, stream);
    if (fp32) {
      rmm::device_uvector<float> xf(n * d, stream);
      for (int64_t a = 0; a < n; a += kProjectChunk) {
        int64_t m = std::min(kProjectChunk, n - a);
        center(rows.rows(a, m), nullptr, xf.data() + a * d, m, d, stream);
      }
      auto index = cuvs::neighbors::brute_force::build(
        res, ip, raft::make_device_matrix_view<const float, int64_t>(xf.data(), n, d));
      run(index, static_cast<const float*>(xf.data()));
    } else {
      auto index = cuvs::neighbors::brute_force::build(
        res, ip, raft::make_device_matrix_view<const half, int64_t>(rows.data(), n, d));
      run(index, rows.data());
    }
  });
}

constexpr uint64_t kNoEdge = ~uint64_t(0);

static __device__ __forceinline__ void bitonic_sort(uint64_t* s, int p)
{
  for (int size = 2; size <= p; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int t = threadIdx.x; t < p; t += blockDim.x) {
        int o = t ^ stride;
        if (o > t) {
          bool up = (t & size) == 0;
          if ((s[t] > s[o]) == up) {
            uint64_t tmp = s[t];
            s[t]         = s[o];
            s[o]         = tmp;
          }
        }
      }
    }
  __syncthreads();
}

/**
 * Which entries a merged row keeps: its first kf forward entries (of kin), then up to kr vectors v
 * that list the row at a rank below ks (lowest rank first, then lowest v), then its remaining
 * forward entries; duplicates dropped (the first occurrence stays), cut to kout.
 */
struct reverse_rule {
  int kin;
  int kf;
  int ks;
  int kr;
  int kout;
};

/**
 * Row u of the reverse-edge merge: the candidates in priority order, duplicates dropped, the first
 * kout kept; with distances ordered nearest first, else kept in priority order (-1 padded).
 * Shared memory: p keys, then the p candidate distances by slot.
 */
static __global__ void merge_rows_kernel(const int32_t* fids,
                                         const float* fdist,
                                         const uint64_t* rev,
                                         int64_t rows,
                                         int64_t a,
                                         int kin,
                                         int kout,
                                         int kf,
                                         int kr,
                                         int p,
                                         int32_t* out)
{
  extern __shared__ uint64_t s[];
  auto* sd  = reinterpret_cast<float*>(s + p);
  int64_t r = blockIdx.x;
  if (r >= rows) return;
  int64_t u = a + r;
  for (int t = threadIdx.x; t < p; t += blockDim.x) {
    int64_t v = -1;
    float dv  = 0.f;
    if (t < kf || (t >= kf + kr && t < kin + kr)) {
      int c = t < kf ? t : t - kr;
      v     = fids[r * kin + c];
      dv    = fdist != nullptr ? fdist[r * kin + c] : 0.f;
    } else if (t < kf + kr) {
      uint64_t rv = rev[r * kr + (t - kf)];
      if (rv != kNoEdge) {
        v  = int64_t(rv >> 32);
        dv = __uint_as_float(uint32_t(rv & 0xffffffffULL));
      }
    }
    s[t]  = v >= 0 && v != u ? (uint64_t(v) << 32) | uint64_t(t) : kNoEdge;
    sd[t] = dv > 0.f ? dv : 0.f;
  }
  bitonic_sort(s, p);  // by id, then slot
  uint64_t mine[8];    // p <= 8 * blockDim.x
  int cnt = 0;
  for (int t = threadIdx.x; t < p; t += blockDim.x) {
    uint64_t e = s[t];
    if (e != kNoEdge && t > 0 && (s[t - 1] >> 32) == (e >> 32))
      e = kNoEdge;  // keep the first slot of an id
    mine[cnt++] = e == kNoEdge ? kNoEdge : ((e & 0xffffffffULL) << 32) | (e >> 32);
  }
  __syncthreads();
  cnt = 0;
  for (int t = threadIdx.x; t < p; t += blockDim.x)
    s[t] = mine[cnt++];
  bitonic_sort(s, p);  // by slot (priority), then id
  if (fdist == nullptr) {
    for (int t = threadIdx.x; t < kout; t += blockDim.x)
      out[r * kout + t] = s[t] == kNoEdge ? -1 : static_cast<int32_t>(s[t] & 0xffffffffULL);
    return;
  }
  cnt = 0;
  for (int t = threadIdx.x; t < p; t += blockDim.x) {
    uint64_t e  = s[t];
    mine[cnt++] = (e == kNoEdge || t >= kout)
                    ? kNoEdge
                    : (uint64_t(__float_as_uint(sd[e >> 32])) << 32) | (e & 0xffffffffULL);
  }
  __syncthreads();
  cnt = 0;
  for (int t = threadIdx.x; t < p; t += blockDim.x)
    s[t] = mine[cnt++];
  bitonic_sort(s, p);  // the kept entries by distance, then id
  for (int t = threadIdx.x; t < kout; t += blockDim.x)
    out[r * kout + t] = s[t] == kNoEdge ? -1 : static_cast<int32_t>(s[t] & 0xffffffffULL);
}

/** rev[u, t] = the t-th sorted reverse entry of row u (t < kr). */
static __global__ void scatter_rev_kernel(
  const uint64_t* vals, const int64_t* starts, int64_t rows, int kr, uint64_t* rev)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= rows * kr) return;
  int64_t u = i / kr, t = i % kr;
  int64_t s = starts[u] + t;
  rev[i]    = s < starts[u + 1] ? vals[s] : kNoEdge;
}

/** counts[g] += reverse entries (rank < ks, not the row itself) of rows v0 .. v0 + m into range g.
 */
static __global__ void count_ranges_kernel(const int32_t* ids,
                                           int64_t m,
                                           int64_t v0,
                                           int kin,
                                           int ks,
                                           int64_t rrows,
                                           int ranges,
                                           unsigned long long* counts)
{
  extern __shared__ unsigned long long hist[];
  for (int g = threadIdx.x; g < ranges; g += blockDim.x)
    hist[g] = 0;
  __syncthreads();
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < m * kin;
       i += int64_t(gridDim.x) * blockDim.x) {
    int64_t u = ids[i];
    if (i % kin < ks && u >= 0 && u != v0 + i / kin) atomicAdd(&hist[u / rrows], 1ULL);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < ranges; g += blockDim.x)
    if (hist[g] != 0) atomicAdd(&counts[g], hist[g]);
}

/**
 * Merge reverse edges into the forward table fids [n, kin] (fdist the matching distances, or null:
 * rows then stay in priority order) by the rule rr, into table [n, kout] (host). Rows are processed
 * in ranges dealt round robin over the devices: one counting pass over the forward table per
 * device, then one collecting pass per range. Every device streams the forward table from host
 * memory in chunks (pinned for full speed).
 */
inline void merge_reverse(std::vector<int> const& devices,
                          const int32_t* fids,
                          const float* fdist,
                          int64_t n,
                          reverse_rule const& rr,
                          int32_t* table)
{
  const int kin = rr.kin, kf = rr.kf, ks = rr.ks, kr = rr.kr, kout = rr.kout;
  RAFT_EXPECTS(kin >= 1 && kf >= 0 && kf <= kin && ks >= 1 && ks <= kin && kr >= 1 && kout >= 1 &&
                 kout <= kin + kr,
               "octex: bad reverse-edge rule");
  int p = 1;
  while (p < kin + kr)
    p <<= 1;
  const int threads = std::max(32, std::min(1024, p));
  RAFT_EXPECTS(p <= 8 * threads, "octex: table_len too large");
  const bool has_dist = fdist != nullptr;
  const size_t row_bytes =
    size_t(kin) * (sizeof(int32_t) + (has_dist ? sizeof(float) : 0));  // forward bytes per row
  const int64_t kStream = int64_t(1) << 20;                            // streamed rows per chunk
  const int G           = static_cast<int>(devices.size());

  // memory that an earlier phase freed into the stream-ordered pool goes back to the device first
  size_t min_free = SIZE_MAX;
  for (int dev : devices) {
    RAFT_CUDA_TRY(cudaSetDevice(dev));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    cudaMemPool_t pool;
    RAFT_CUDA_TRY(cudaDeviceGetDefaultMemPool(&pool, dev));
    RAFT_CUDA_TRY(cudaMemPoolTrimTo(pool, 0));
    min_free = std::min(min_free, device_free_bytes());
  }
  // forward rows stream through two chunk buffers (a resident copy of all rows hung at 1e7 rows)
  const size_t fixed = 2 * kStream * row_bytes +                          // forward rows
                       size_t(kStream) * kin * sizeof(int64_t) +          // compaction
                       size_t(kStream) * (kin + kout) * sizeof(int32_t);  // merged rows
  size_t avail = min_free > fixed + (size_t(2) << 30) ? min_free - fixed - (size_t(2) << 30) : 0;
  avail /=
    2;  // headroom for the radix-sort scratch and pool fragmentation (34 GB ranges failed at 1e9)
  // an entry in flight: 32-bit key and 64-bit value, twice (sort buffers), plus the reverse rows
  // (kr per row, 8 bytes); ks entries per row on average, 25% above for uneven ranges
  const double per_row = 1.25 * ks * 2 * (sizeof(uint32_t) + sizeof(uint64_t)) + kr * 8.0;
  int64_t rrows        = std::max<int64_t>(1, int64_t(double(avail) / per_row));
  rrows = std::min<int64_t>(rrows, int64_t(UINT32_MAX / uint32_t(ks)));  // 32-bit keys
  if (n >= int64_t(G) * kStream) rrows = std::min<int64_t>(rrows, (n + G - 1) / G);
  rrows            = std::min(rrows, n);
  const int ranges = static_cast<int>((n + rrows - 1) / rrows);
  RAFT_EXPECTS(ranges <= 6144, "octex: too little device memory for the reverse edges");
  per_device(devices, [&](int rank, int dev) {
    if (rank >= ranges) return;
    raft::device_resources res;
    cudaStream_t stream  = raft::resource::get_cuda_stream(res).get();
    auto policy          = raft::resource::get_thrust_policy(res);
    const int64_t fchunk = std::min<int64_t>(n, kStream);
    rmm::device_uvector<int32_t> di(fchunk * kin * 2, stream);
    rmm::device_uvector<float> dd(has_dist ? fchunk * kin * 2 : 0, stream);
    cudaStream_t copy_stream;
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    cudaEvent_t loaded[2], used[2];
    for (int s = 0; s < 2; s++) {
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&loaded[s], cudaEventDisableTiming));
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&used[s], cudaEventDisableTiming));
      RAFT_CUDA_TRY(cudaEventRecord(used[s], stream));
    }
    auto copy_rows = [&](int slot, int64_t v0, int64_t m, cudaStream_t s) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(di.data() + slot * fchunk * kin,
                                    fids + v0 * kin,
                                    m * kin * sizeof(int32_t),
                                    cudaMemcpyHostToDevice,
                                    s));
      if (has_dist)
        RAFT_CUDA_TRY(cudaMemcpyAsync(dd.data() + slot * fchunk * kin,
                                      fdist + v0 * kin,
                                      m * kin * sizeof(float),
                                      cudaMemcpyHostToDevice,
                                      s));
    };
    auto for_chunks = [&](int64_t v_lo, int64_t v_hi, auto&& f) {
      int64_t nch = (v_hi - v_lo + fchunk - 1) / fchunk;
      auto issue  = [&](int64_t c) {
        int slot = int(c & 1);
        RAFT_CUDA_TRY(cudaStreamWaitEvent(copy_stream, used[slot], 0));
        int64_t v0 = v_lo + c * fchunk;
        copy_rows(slot, v0, std::min(fchunk, v_hi - v0), copy_stream);
        RAFT_CUDA_TRY(cudaEventRecord(loaded[slot], copy_stream));
      };
      if (nch > 0) issue(0);
      for (int64_t c = 0; c < nch; c++) {
        if (c + 1 < nch) issue(c + 1);
        int slot = int(c & 1);
        RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, loaded[slot], 0));
        int64_t v0 = v_lo + c * fchunk;
        f(v0,
          std::min(fchunk, v_hi - v0),
          di.data() + slot * fchunk * kin,
          has_dist ? dd.data() + slot * fchunk * kin : nullptr);
        RAFT_CUDA_TRY(cudaEventRecord(used[slot], stream));
      }
    };

    // entries per range (every device counts over all rows, all devices at once)
    std::vector<unsigned long long> counts(ranges);
    {
      rmm::device_uvector<unsigned long long> dc(ranges, stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(dc.data(), 0, ranges * sizeof(unsigned long long), stream));
      for_chunks(0, n, [&](int64_t v0, int64_t m, const int32_t* ids, const float*) {
        count_ranges_kernel<<<static_cast<unsigned>(std::min<int64_t>(grid(m * kin), 2048)),
                              256,
                              ranges * sizeof(unsigned long long),
                              stream>>>(ids, m, v0, kin, ks, rrows, ranges, dc.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      });
      RAFT_CUDA_TRY(cudaMemcpyAsync(counts.data(),
                                    dc.data(),
                                    ranges * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      raft::resource::sync_stream(res);
    }
    rmm::device_uvector<int64_t> flat(fchunk * kin, stream);  // compaction of one chunk
    for (int g = rank; g < ranges; g += G) {
      const int64_t a = int64_t(g) * rrows, b = std::min(n, a + rrows), R = b - a;
      const int64_t total = static_cast<int64_t>(counts[g]);
      // collect the entries (v, rank) with a <= u = row v [rank] < b, in order of v
      rmm::device_uvector<uint32_t> keys(total, stream), skeys(total, stream);
      rmm::device_uvector<uint64_t> vals(total, stream), svals(total, stream);
      int64_t at = 0;
      for_chunks(0, n, [&](int64_t v0, int64_t m, const int32_t* ids, const float* ds) {
        uint32_t* kp = keys.data() + at;
        uint64_t* vp = vals.data() + at;
        // stream compaction keeps the order of v: equal keys then sort by v (radix sort is
        // stable)
        auto pred = [=] __device__(int64_t i) -> bool {
          int64_t u = ids[i];
          return i % kin < ks && u >= a && u < b && u != v0 + i / kin;
        };
        auto e            = thrust::copy_if(policy,
                                 thrust::make_counting_iterator<int64_t>(0),
                                 thrust::make_counting_iterator<int64_t>(m * kin),
                                 flat.data(),
                                 pred);
        int64_t c         = e - flat.data();
        const int64_t* fl = flat.data();
        thrust::for_each_n(
          policy, thrust::make_counting_iterator<int64_t>(0), c, [=] __device__(int64_t j) -> void {
            int64_t i  = fl[j];
            int64_t u  = ids[i];
            kp[j]      = static_cast<uint32_t>((u - a) * ks + i % kin);
            uint32_t w = ds != nullptr ? __float_as_uint(ds[i]) : 0u;
            vp[j]      = (uint64_t(uint32_t(v0 + i / kin)) << 32) | w;
          });
        at += c;
      });
      RAFT_EXPECTS(at == total, "octex: reverse-edge count mismatch");
      int bits = 1;
      while ((int64_t(1) << bits) < R * ks)
        bits++;
      size_t tmp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      tmp_bytes,
                                      keys.data(),
                                      skeys.data(),
                                      vals.data(),
                                      svals.data(),
                                      total,
                                      0,
                                      bits,
                                      stream);
      rmm::device_buffer tmp(tmp_bytes, stream);
      cub::DeviceRadixSort::SortPairs(tmp.data(),
                                      tmp_bytes,
                                      keys.data(),
                                      skeys.data(),
                                      vals.data(),
                                      svals.data(),
                                      total,
                                      0,
                                      bits,
                                      stream);
      tmp.resize(0, stream);
      tmp.shrink_to_fit(stream);
      keys.release();
      vals.release();
      rmm::device_uvector<int64_t> starts(R + 1, stream);
      auto row_key = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int64_t>(0),
        [ks] __device__(int64_t u) -> uint32_t { return static_cast<uint32_t>(u * ks); });
      thrust::lower_bound(policy, skeys.begin(), skeys.end(), row_key, row_key + R, starts.begin());
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        starts.data() + R, &total, sizeof(int64_t), cudaMemcpyHostToDevice, stream));
      rmm::device_uvector<uint64_t> rev(R * kr, stream);
      scatter_rev_kernel<<<grid(R * kr), 256, 0, stream>>>(
        svals.data(), starts.data(), R, kr, rev.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      skeys.release();
      svals.release();
      starts.release();

      // forward rows of the range, merged with their reverse entries
      rmm::device_uvector<int32_t> out(std::min<int64_t>(R, kStream) * kout, stream);
      int64_t r_at = a;
      for_chunks(a, b, [&](int64_t v0, int64_t m, const int32_t* fi, const float* fd) {
        merge_rows_kernel<<<static_cast<unsigned>(m),
                            threads,
                            p * (sizeof(uint64_t) + sizeof(float)),
                            stream>>>(
          fi, fd, rev.data() + (v0 - a) * kr, m, v0, kin, kout, kf, kr, p, out.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
        RAFT_CUDA_TRY(cudaMemcpyAsync(table + v0 * kout,
                                      out.data(),
                                      m * kout * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost,
                                      stream));
        raft::resource::sync_stream(res);  // out is reused
        r_at = v0 + m;
      });
      RAFT_EXPECTS(r_at == b, "octex: merged rows");
      build_log("  reverse edges: range %d / %d on GPU %d, %lld entries\n",
                g + 1,
                ranges,
                dev,
                (long long)total);
    }
    raft::resource::sync_stream(res);
    RAFT_CUDA_TRY(cudaStreamSynchronize(copy_stream));
    RAFT_CUDA_TRY(cudaStreamDestroy(copy_stream));
    for (int s = 0; s < 2; s++) {
      cudaEventDestroy(loaded[s]);
      cudaEventDestroy(used[s]);
    }
  });
}

/** The prototype's rule with kf = ks = kr = k / 2 on forward rows of k. */
inline auto half_rule(int k) -> reverse_rule
{
  int h = std::max(1, k / 2);
  return reverse_rule{k, h, h, h, k};
}

/** The exact neighbor table [n, k] (host), -1 padded, reverse edges merged in nearest first. */
inline void build_table(std::vector<int> const& devices,
                        const half* x,
                        int64_t n,
                        int64_t d,
                        int k,
                        bool reverse,
                        int32_t* table)
{
  RAFT_EXPECTS(k >= 1 && k < n, "octex: table_len must be in [1, n)");
  std::vector<int32_t> ids(size_t(n) * k);
  std::vector<float> dist(size_t(n) * k);
  exact_table(devices, x, n, d, k, ids.data(), dist.data());
  if (reverse) {
    merge_reverse(devices, ids.data(), dist.data(), n, half_rule(k), table);
  } else {
    std::memcpy(table, ids.data(), ids.size() * sizeof(int32_t));
  }
}

// ---- the neighbor table built by octex's own search (above exact_table_max)

/** MemAvailable of /proc/meminfo, in bytes (0 when unreadable). */
inline auto host_available_bytes() -> size_t
{
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (f == nullptr) return 0;
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr)
    if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  std::fclose(f);
  return kb * 1024;
}

/** Anonymous pinned host memory (cudaHostAlloc; never a mapped file): full-speed copies to and
 * from the devices. */
template <typename T>
class pinned_rows {
 public:
  explicit pinned_rows(size_t count) : buf_(count * sizeof(T)) {}
  [[nodiscard]] auto data() const -> T* { return static_cast<T*>(buf_.ptr); }

 private:
  hbuf buf_;
};

/** One search pass of the table build: first round only (waves 0), or waves over the last table. */
struct table_pass {
  int waves;
  int starts;
};

/**
 * The passes, rows and reverse rule of the searched table (rows of k): a first round only, then 2
 * and 3 waves over the merged table of the previous pass. Forward rows of 3k / 2 (k + 1 searched,
 * the vector itself dropped); merged rows keep k / 4 forward entries first and up to 3k / 4 reverse
 * entries (8-48-24 for k = 32).
 */
struct table_plan {
  std::vector<table_pass> passes{{0, 64}, {2, 128}, {3, 128}};
  int read         = 64;
  int final_rerank = 64;
  int batch        = 10000;
  reverse_rule rule{};
  int64_t chunk = int64_t(1) << 20;  // base rows per search call

  /** passes: the first 1 to 3 passes; sketch_order: rows in sketch order, no exact rerank. */
  explicit table_plan(int k, int npasses = 3, bool sketch_order = false)
  {
    int kin      = k + k / 2;
    int kf       = std::max(1, k / 4);
    rule         = reverse_rule{kin, kf, kin, std::max(1, k - kf), k};
    final_rerank = sketch_order ? 0 : std::max(final_rerank, kin + 1);
    passes.resize(std::clamp(npasses, 1, 3));
    for (auto& p : passes)
      p.starts = std::max(p.starts, kin + 1);
  }
};

/** Planned peak host bytes of the searched table: the loaded index, forward and merged rows. */
inline auto searched_table_host_bytes(int64_t n, int64_t d, int k, bool vectors_pinned = false)
  -> size_t
{
  table_plan tp(k);
  const int kin   = tp.rule.kin;
  const int64_t q = std::min(n, tp.chunk);
  size_t vectors  = size_t(n) * d * sizeof(half);       // pinned, vectors on host
  size_t forward  = size_t(n) * kin * sizeof(int32_t);  // pinned forward rows
  if (vectors_pinned) vectors = 0;                      // the base, locked in place by its owner
  size_t merged  = size_t(n) * k * sizeof(int32_t);     // merged rows
  size_t buffers = 2 * size_t(q) * d * (sizeof(half) + sizeof(float))            // two query chunks
                   + size_t(q) * (kin + 1) * (sizeof(int64_t) + sizeof(float));  // search output
  return vectors + forward + merged + buffers;  // one load for all passes: all held at the merges
}

/** Rows [r0, r0 + m) of an fp16 .npy matrix as float, by buffered sequential reads (pread). */
inline void read_rows_as_float(
  int fd, size_t data_offset, int64_t r0, int64_t m, int64_t d, std::vector<half>& hbuf, float* out)
{
  hbuf.resize(size_t(m) * d);
  parallel_pread(fd,
                 reinterpret_cast<char*>(hbuf.data()),
                 size_t(m) * d * sizeof(half),
                 data_offset + size_t(r0) * d * sizeof(half));
  const half* h = hbuf.data();
  parallel_for(m * d, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; i++)
      out[i] = __half2float(h[i]);
  });
}

/**
 * The neighbor table [n, k] by octex's own search, written to table_path: every pass loads the
 * index directory (stage(table_len) returns a loadable directory over its files, with or without
 * the table) on all devices (copies and table rows sharded, fp16 vectors in pinned host memory, the
 * sketch replicated when it fits, else sharded), searches every base vector (read from vectors_path
 * in chunks), drops the vector itself, merges reverse edges and writes table_path.
 * phase(name) records the time since the previous call.
 */
inline void searched_table(std::vector<int> const& devices,
                           std::string const& vectors_path,
                           int64_t n,
                           int64_t d,
                           int k,
                           int sketch_width,
                           std::function<std::string(int)> const& stage,
                           std::string const& table_path,
                           std::function<void(std::string const&)> const& phase,
                           bool reverse_edges = true,
                           int passes         = 3,
                           bool sketch_order  = false)
{
  table_plan tp(k, passes, sketch_order);
  const reverse_rule rr = tp.rule;
  const int kin = rr.kin, kq = kin + 1;
  RAFT_EXPECTS(k >= 1 && kq < n, "octex: table_len too large for the dataset");

  // host memory: refuse a plan above 70% of the available memory
  auto vm           = find_mem_array(vectors_path);
  size_t plan_bytes = searched_table_host_bytes(n, d, k, vm && vm->pinned);
  size_t avail      = host_available_bytes();
  build_log(
    "table: searched, n %lld, rows %d (forward %d, rule %d-%d-%d), planned peak host %.1f "
    "GiB of %.1f GiB available\n",
    (long long)n,
    k,
    kin,
    rr.kf,
    rr.ks,
    rr.kr,
    plan_bytes / double(1 << 30),
    avail / double(1 << 30));
  if (avail > 0 && double(plan_bytes) > 0.7 * double(avail))
    RAFT_FAIL(
      "octex: the searched neighbor table needs %.1f GiB of host memory (pinned fp16 vectors %.1f, "
      "forward rows %.1f, merged rows %.1f), above 70%% of the %.1f GiB available; free host "
      "memory "
      "or lower table_len",
      plan_bytes / double(1 << 30),
      double(n) * d * sizeof(half) / double(1 << 30),
      double(n) * kin * sizeof(int32_t) / double(1 << 30),
      double(n) * k * sizeof(int32_t) / double(1 << 30),
      avail / double(1 << 30));

  auto vhdr = read_npy_header(vectors_path);
  RAFT_EXPECTS(
    vhdr.descr == "<f2" && vhdr.shape.size() == 2 && vhdr.shape[0] == n && vhdr.shape[1] == d,
    "octex: vectors.npy must be fp16 [n, d]");
  const char* dump = std::getenv("OCTEX_TABLE_DUMP");  // tests: every pass's rows to this directory

  size_t min_free = SIZE_MAX;
  for (int dev : devices) {
    RAFT_CUDA_TRY(cudaSetDevice(dev));
    min_free = std::min(min_free, device_free_bytes());
  }
  load_params lp;
  lp.devices         = devices;
  lp.mode            = placement::shard;
  lp.vectors_on_host = true;
  lp.sketch          = size_t(n) * sketch_width <= min_free / 4 ? sketch_placement::replicate
                                                                : sketch_placement::shard;
  if (const char* e = std::getenv("OCTEX_TABLE_SKETCH"))
    lp.sketch = std::string(e) == "shard"  ? sketch_placement::shard
                : std::string(e) == "host" ? sketch_placement::host
                                           : sketch_placement::replicate;

  pinned_rows<int32_t> fwd(size_t(n) * kin);  // forward rows, read by every merge range
  // one load for all passes: each pass gets the merged rows of the previous one in place
  // (set_table), the device table is freed for each merge, and only the last rows are written
  std::vector<int32_t, no_init_allocator<int32_t>> table;  // the merged rows of the last pass
  std::optional<raft::device_resources> rres;
  std::optional<cuvs::neighbors::octex::index> ridx;
  for (size_t v = 0; v < tp.passes.size(); v++) {
    auto const& pass      = tp.passes[v];
    const std::string tag = "table_v" + std::to_string(v + 1);
    {
      RAFT_CUDA_TRY(cudaSetDevice(devices[0]));
      if (!ridx) {
        rres.emplace();
        ridx.emplace(cuvs::neighbors::octex::load(*rres, stage(0), lp));
      }
      if (pass.waves > 0) set_table(*ridx, table.data(), k);
      raft::device_resources& res = *rres;
      auto& idx                   = *ridx;
      search_params sp;
      sp.read         = tp.read;
      sp.starts       = pass.starts;
      sp.waves        = pass.waves;
      sp.batch        = tp.batch;
      sp.sketch_rank  = true;
      sp.final_rerank = tp.final_rerank;
      build_log("%s: loaded (%s sketch), %d waves, %d starts\n",
                tag.c_str(),
                lp.sketch == sketch_placement::shard ? "sharded" : "replicated",
                pass.waves,
                pass.starts);
      phase(tag + "_load");

      int fd = ::open(vectors_path.c_str(), O_RDONLY);
      RAFT_EXPECTS(fd >= 0, "octex: cannot open %s", vectors_path.c_str());
      const int64_t q = std::min(n, tp.chunk);
      // pinned queries and results: copies from pageable memory would block the host on every
      // batch and serialize the GPUs
      hbuf qfb[2]  = {hbuf(size_t(q) * d * sizeof(float)), hbuf(size_t(q) * d * sizeof(float))};
      float* qf[2] = {static_cast<float*>(qfb[0].ptr), static_cast<float*>(qfb[1].ptr)};
      std::vector<half> hb[2];
      hbuf nbb(size_t(q) * kq * sizeof(int64_t)), ndb(size_t(q) * kq * sizeof(float));
      auto* nb = static_cast<int64_t*>(nbb.ptr);
      auto* nd = static_cast<float*>(ndb.ptr);
      std::thread reader;
      std::exception_ptr rerr;
      auto read_chunk = [&](int64_t r0, int slot) {
        try {
          read_rows_as_float(fd, vhdr.header_len, r0, std::min(q, n - r0), d, hb[slot], qf[slot]);
        } catch (...) {
          rerr = std::current_exception();
        }
      };
      read_chunk(0, 0);
      auto t0  = std::chrono::steady_clock::now();
      int slot = 0;
      for (int64_t r0 = 0; r0 < n; r0 += q, slot ^= 1) {
        if (reader.joinable()) reader.join();
        if (rerr) std::rethrow_exception(rerr);
        const int64_t m = std::min(q, n - r0);
        if (r0 + q < n) reader = std::thread(read_chunk, r0 + q, slot ^ 1);
        cuvs::neighbors::octex::search(
          res,
          sp,
          idx,
          raft::make_host_matrix_view<const float, int64_t>(qf[slot], m, d),
          raft::make_host_matrix_view<int64_t, int64_t>(nb, m, kq),
          raft::make_host_matrix_view<float, int64_t>(nd, m, kq));
        // drop the vector itself, keep the first kin
        parallel_for(m, [&](int64_t i0, int64_t i1) {
          for (int64_t i = i0; i < i1; i++) {
            int32_t* o = fwd.data() + (r0 + i) * kin;
            int t      = 0;
            for (int j = 0; j < kq && t < kin; j++) {
              int64_t u = nb[i * kq + j];
              if (u == r0 + i) continue;
              o[t++] = u < 0 ? -1 : static_cast<int32_t>(u);
            }
          }
        });
        int64_t done = r0 + m;
        if ((r0 / q) % 16 == 0 || done == n)
          build_log("  %s: %lld / %lld rows, %.0f rows/s\n",
                    tag.c_str(),
                    (long long)done,
                    (long long)n,
                    done / std::max(seconds_since(t0), 1e-9));
      }
      if (reader.joinable()) reader.join();
      ::close(fd);
      if (rerr) std::rethrow_exception(rerr);
      if (pass.waves > 0) drop_table(idx);  // the merge needs the HBM of the table
    }
    phase(tag + "_search");
    if (dump != nullptr)
      write_npy(std::string(dump) + "/forward_v" + std::to_string(v + 1) + ".npy",
                fwd.data(),
                {n, int64_t(kin)});
    {
      table.resize(size_t(n) * k);
      auto tm = std::chrono::steady_clock::now();
      if (reverse_edges) {
        merge_reverse(devices, fwd.data(), nullptr, n, rr, table.data());
        build_log("  %s: reverse edges merged in %.1f s\n", tag.c_str(), seconds_since(tm));
      } else {  // the first k forward neighbors of every row
        parallel_for(n, [&](int64_t i0, int64_t i1) {
          for (int64_t i = i0; i < i1; i++)
            std::copy_n(fwd.data() + i * kin, k, table.data() + i * k);
        });
        build_log("  %s: forward rows only, kept in %.1f s\n", tag.c_str(), seconds_since(tm));
      }
      if (v + 1 == tp.passes.size()) write_npy(table_path, table.data(), {n, int64_t(k)});
      if (dump != nullptr)
        write_npy(std::string(dump) + "/table_v" + std::to_string(v + 1) + ".npy",
                  table.data(),
                  {n, int64_t(k)});
    }
    phase(tag + "_reverse");
  }
}

}  // namespace cuvs::neighbors::octex::detail
