/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The cells of one copy: project the dataset, hash every vector at every spacing, keep the
// populated cells, list the l vectors nearest to every cell center, sort the cells by (bucket,
// hash). Billion-row datasets take the streaming path (build_cells_stream): device memory O(n) only
// for the hashes, host memory O(cells) per copy.

#include "build_cells.hpp"
#include "build_common.cuh"
#include "e8.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>

#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <thread>

namespace cuvs::neighbors::octex::detail {

constexpr int64_t kProjectChunk = int64_t(1) << 18;  // rows per projection GEMM, as the prototype
constexpr int64_t kListChunk    = int64_t(1) << 20;  // cell centers per kNN call
constexpr int kMaxBlocks        = 8;

/** h[i] = hash of the E8^b cell of Y[i] / f; idx[i] = i. */
static __global__ void cell_hash_kernel(
  const float* y, int64_t n, int b, float f, int64_t* h, int32_t* idx)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  float x[8 * kMaxBlocks];
  const float* r = y + i * 8 * b;
  for (int k = 0; k < 8 * b; k++)
    x[k] = r[k] / f;  // division, as the prototype; the hash then scales by exactly 1
  h[i]   = e8b_hash64(x, b, 1.0f);
  idx[i] = static_cast<int32_t>(i);
}

/** first[i] = the smallest row of run i (rows are sorted within a run). */
static __global__ void run_first_kernel(const int32_t* sorted_idx,
                                        const int32_t* start,
                                        int64_t m,
                                        int32_t* first)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < m) first[i] = sorted_idx[start[i]];
}

/** z[i] = f * E8^b point of Y[first[i]] / f: the center of cell i. */
static __global__ void cell_center_kernel(
  const float* y, const int32_t* first, int64_t m, int b, float f, float* z)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  const float* r = y + int64_t(first[i]) * 8 * b;
  float* o       = z + i * 8 * b;
  for (int j = 0; j < b; j++) {
    float x[8], p[8];
#pragma unroll
    for (int k = 0; k < 8; k++)
      x[k] = r[8 * j + k] / f;
    e8_point(x, p);
#pragma unroll
    for (int k = 0; k < 8; k++)
      o[8 * j + k] = p[k] * f;
  }
}

static __global__ void add_shift_kernel(float* y, const float* shift, int64_t n, int w)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n * w) y[i] += shift[i % w];
}

/** out row r = in row perm[r] (rows of l int32). */
static __global__ void gather_rows_kernel(
  const int32_t* in, const int32_t* perm, int64_t m, int l, int32_t* out)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < m * l) out[i] = in[int64_t(perm[i / l]) * l + i % l];
}

static __global__ void narrow_kernel(const int64_t* in, int64_t n, int32_t* out)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = static_cast<int32_t>(in[i]);
}

inline auto grid(int64_t n, int t = 256) -> unsigned
{
  return static_cast<unsigned>((n + t - 1) / t);
}

// ---- lists from cell members (cell_params::member_lists): no index over the projection ----

constexpr int kMaxMemberList = 64;

/** The cells of one spacing: rows sorted by cell, unique cell hashes ascending, run starts. */
struct level_cells {
  rmm::device_uvector<int32_t> sidx;
  rmm::device_uvector<int64_t> ukeys;
  rmm::device_uvector<int32_t> start, pop;
};

/** Cell of row i of z / f: [s, s + c) of the sorted rows, c = 0 when the cell is empty. */
static __global__ void cell_lookup_kernel(const float* z,
                                          int64_t m,
                                          int b,
                                          float f,
                                          const int64_t* ukeys,
                                          int64_t u,
                                          const int32_t* start,
                                          const int32_t* pop,
                                          int32_t* s,
                                          int32_t* c)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  float x[8 * kMaxBlocks];
  for (int k = 0; k < 8 * b; k++)
    x[k] = z[i * 8 * b + k] / f;
  int64_t key = e8b_hash64(x, b, 1.0f);
  int64_t lo = 0, hi = u;
  while (lo < hi) {
    int64_t mid = (lo + hi) / 2;
    if (ukeys[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  bool hit = lo < u && ukeys[lo] == key;
  s[i]     = hit ? start[lo] : 0;
  c[i]     = hit ? pop[lo] : 0;
}

/**
 * One warp per center i: the l rows nearest to it among the union of K sorted-row ranges
 * [s[k M + i], + c[k M + i]) of sidx[k], without repeats, padded by repeating the nearest one.
 */
static __global__ void member_list_kernel(const float* y,
                                          int w,
                                          const float* z,
                                          int64_t m,
                                          int K,
                                          const int32_t* const* sidx,
                                          const int32_t* s,
                                          const int32_t* c,
                                          int l,
                                          int32_t* lists)
{
  int64_t i = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  int lane  = threadIdx.x % 32;
  if (i >= m) return;
  float zc[8 * kMaxBlocks];
  for (int k = 0; k < w; k++)
    zc[k] = z[i * w + k];
  float bd[kMaxMemberList];
  int32_t bi[kMaxMemberList];
  for (int k = 0; k < l; k++) {
    bd[k] = INFINITY;
    bi[k] = -1;
  }
  for (int q = 0; q < K; q++) {
    const int32_t* rows = sidx[q] + s[q * m + i];
    int cnt             = c[q * m + i];
    for (int t = lane; t < cnt; t += 32) {
      int32_t r      = rows[t];
      const float* v = y + int64_t(r) * w;
      float dd       = 0.f;
      for (int k = 0; k < w; k++) {
        float e = v[k] - zc[k];
        dd += e * e;
      }
      if (dd < bd[l - 1]) {
        bool seen = false;
        for (int k = 0; k < l && bd[k] <= dd; k++)
          seen |= bi[k] == r;
        if (seen) continue;
        int p = l - 1;
        while (p > 0 && bd[p - 1] > dd) {
          bd[p] = bd[p - 1];
          bi[p] = bi[p - 1];
          p--;
        }
        bd[p] = dd;
        bi[p] = r;
      }
    }
  }
  // merge the 32 sorted lane lists: rounds of a warp argmin over the heads, repeats skipped
  int head        = 0;
  int32_t* out    = lists + i * l;
  int32_t nearest = -1, last = -1;
  int k = 0;
  for (int round = 0; round < 32 * l && k < l; round++) {
    float hv = head < l ? bd[head] : INFINITY;
    int hl   = lane;
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, hv, o);
      int ol   = __shfl_xor_sync(0xffffffffu, hl, o);
      if (ov < hv || (ov == hv && ol < hl)) {
        hv = ov;
        hl = ol;
      }
    }
    if (hv == INFINITY) break;
    int32_t id = __shfl_sync(0xffffffffu, head < l ? bi[head] : -1, hl);
    if (lane == hl) head++;
    if (id < 0 || id == last) continue;
    if (k == 0) nearest = id;
    if (lane == 0) out[k] = id;
    last = id;
    k++;
  }
  for (; k < l; k++)
    if (lane == 0) out[k] = nearest;
}

inline auto make_level_cells(raft::resources const& res, const float* y, int64_t n, int b, float f)
  -> level_cells
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  rmm::device_uvector<int64_t> h(n, stream), hs(n, stream);
  rmm::device_uvector<int32_t> idx(n, stream);
  level_cells lc{rmm::device_uvector<int32_t>(n, stream),
                 rmm::device_uvector<int64_t>(n, stream),
                 rmm::device_uvector<int32_t>(n, stream),
                 rmm::device_uvector<int32_t>(n, stream)};
  cell_hash_kernel<<<grid(n), 256, 0, stream>>>(y, n, b, f, h.data(), idx.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  size_t tb = 0;
  cub::DeviceRadixSort::SortPairs(
    nullptr, tb, h.data(), hs.data(), idx.data(), lc.sidx.data(), n, 0, 64, stream);
  rmm::device_buffer tmp(tb, stream);
  cub::DeviceRadixSort::SortPairs(
    tmp.data(), tb, h.data(), hs.data(), idx.data(), lc.sidx.data(), n, 0, 64, stream);
  rmm::device_scalar<int64_t> d_runs(stream);
  size_t rb = 0;
  cub::DeviceRunLengthEncode::Encode(
    nullptr, rb, hs.data(), lc.ukeys.data(), lc.pop.data(), d_runs.data(), n, stream);
  rmm::device_buffer rtmp(rb, stream);
  cub::DeviceRunLengthEncode::Encode(
    rtmp.data(), rb, hs.data(), lc.ukeys.data(), lc.pop.data(), d_runs.data(), n, stream);
  int64_t u = d_runs.value(stream);
  size_t sb = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, sb, lc.pop.data(), lc.start.data(), u, stream);
  rmm::device_buffer stmp(sb, stream);
  cub::DeviceScan::ExclusiveSum(stmp.data(), sb, lc.pop.data(), lc.start.data(), u, stream);
  lc.ukeys.resize(u, stream);
  lc.start.resize(u, stream);
  lc.pop.resize(u, stream);
  return lc;
}

/**
 * Lists [M, l] of the M centers z from cell members: the union over the given levels of the rows
 * of the cell that holds each center at that level.
 */
inline void member_lists(raft::resources const& res,
                         const float* y,
                         int b,
                         const float* z,
                         int64_t M,
                         int l,
                         std::vector<const level_cells*> const& lv,
                         std::vector<float> const& fs,
                         int32_t* lists)
{
  RAFT_EXPECTS(l <= kMaxMemberList, "octex: member lists hold at most 64 entries");
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int w         = 8 * b;
  const int K         = static_cast<int>(lv.size());
  rmm::device_uvector<int32_t> s(K * M, stream), c(K * M, stream);
  std::vector<const int32_t*> hp(K);
  for (int q = 0; q < K; q++) {
    cell_lookup_kernel<<<grid(M), 256, 0, stream>>>(z,
                                                    M,
                                                    b,
                                                    fs[q],
                                                    lv[q]->ukeys.data(),
                                                    lv[q]->ukeys.size(),
                                                    lv[q]->start.data(),
                                                    lv[q]->pop.data(),
                                                    s.data() + q * M,
                                                    c.data() + q * M);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    hp[q] = lv[q]->sidx.data();
  }
  rmm::device_uvector<const int32_t*> dp(K, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    dp.data(), hp.data(), K * sizeof(const int32_t*), cudaMemcpyHostToDevice, stream));
  member_list_kernel<<<grid(M * 32, 128), 128, 0, stream>>>(
    y, w, z, M, K, dp.data(), s.data(), c.data(), l, lists);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(res);
}

template <typename T>
void to_host(std::vector<T>& dst, const T* src, int64_t n, cudaStream_t s)
{
  size_t at = dst.size();
  dst.resize(at + n);
  if (n > 0)
    RAFT_CUDA_TRY(cudaMemcpyAsync(dst.data() + at, src, n * sizeof(T), cudaMemcpyDeviceToHost, s));
}

/** Project rows of the dataset: y = (x - mean) @ u + shift (fp32, cuBLAS SGEMM in chunks of 2^18
 * rows). */
inline void project_all(raft::resources const& res,
                        device_rows& rows,
                        int64_t n,
                        int64_t d,
                        const float* d_mean,
                        const float* d_u,
                        const float* d_shift,
                        int w,
                        float* a_buf,
                        float* y)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto handle         = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  const float one = 1.f, zero = 0.f;
  for (int64_t r0 = 0; r0 < n; r0 += kProjectChunk) {
    int64_t m = std::min(kProjectChunk, n - r0);
    center(rows.rows(r0, m), d_mean, a_buf, m, d, stream);
    // row major y [m, w] = a [m, d] @ u [d, w], i.e. column major y^T = u^T a^T
    RAFT_CUBLAS_TRY(cublasSgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                w,
                                static_cast<int>(m),
                                static_cast<int>(d),
                                &one,
                                d_u,
                                w,
                                a_buf,
                                static_cast<int>(d),
                                &zero,
                                y + r0 * w,
                                w));
  }
  add_shift_kernel<<<grid(n * w), 256, 0, stream>>>(y, d_shift, n, w);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Append one level to cc: sort its M cells (keys in ascending hash order, lists [M, l] in the same
 * order) by (bucket, hash) with a stable sort by bucket, CSR offsets over the buckets.
 */
inline void finish_level(raft::resources const& res,
                         cell_params const& params,
                         const int64_t* keys,
                         const int32_t* lists,
                         int64_t M,
                         int64_t& row,
                         copy_cells& cc)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto policy         = raft::resource::get_thrust_policy(res);
  const int l         = params.list_len;
  const int64_t T     = int64_t(1) << params.log_buckets;
  rmm::device_uvector<int32_t> bucket(M, stream), sbucket(M, stream), seq(M, stream),
    perm(M, stream);
  int64_t tmask = T - 1;
  thrust::transform(
    policy, keys, keys + M, bucket.begin(), [tmask] __device__(int64_t k) -> int32_t {
      return static_cast<int32_t>(k & tmask);
    });
  thrust::sequence(policy, seq.begin(), seq.end());
  size_t bs_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  bs_bytes,
                                  bucket.data(),
                                  sbucket.data(),
                                  seq.data(),
                                  perm.data(),
                                  M,
                                  0,
                                  params.log_buckets,
                                  stream);
  rmm::device_buffer bs_tmp(bs_bytes, stream);
  cub::DeviceRadixSort::SortPairs(bs_tmp.data(),
                                  bs_bytes,
                                  bucket.data(),
                                  sbucket.data(),
                                  seq.data(),
                                  perm.data(),
                                  M,
                                  0,
                                  params.log_buckets,
                                  stream);
  rmm::device_uvector<int64_t> skeys(M, stream);
  thrust::gather(policy, perm.begin(), perm.end(), keys, skeys.begin());
  rmm::device_uvector<int32_t> slists(M * l, stream);
  gather_rows_kernel<<<grid(M * l), 256, 0, stream>>>(lists, perm.data(), M, l, slists.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  rmm::device_uvector<int32_t> offs(T + 1, stream);
  thrust::lower_bound(policy,
                      sbucket.begin(),
                      sbucket.end(),
                      thrust::make_counting_iterator<int32_t>(0),
                      thrust::make_counting_iterator<int32_t>(static_cast<int32_t>(T + 1)),
                      offs.begin());

  to_host(cc.lists, slists.data(), M * l, stream);
  to_host(cc.keys, skeys.data(), M, stream);
  to_host(cc.offsets, offs.data(), T + 1, stream);
  cc.base.push_back(static_cast<int32_t>(row));
  cc.cells.push_back(M);
  row += M;
  raft::resource::sync_stream(res);
}

/** Cells of copies [c0, c1) with the whole projection on the device (y [n, 8b] fp32). */
inline void build_cells_full(raft::resources const& res,
                             cell_params const& params,
                             const half* x,
                             int64_t n,
                             int64_t d,
                             const float* mean,
                             const float* proj,
                             const float* shift,
                             int c0,
                             int c1,
                             std::vector<copy_cells>& out)
{
  const int b = params.blocks;
  const int w = 8 * b;
  const int l = params.list_len;
  const int L = static_cast<int>(params.spacings.size());
  RAFT_EXPECTS(b >= 1 && b <= kMaxBlocks, "octex: blocks must be in [1, 8]");
  RAFT_EXPECTS(L >= 1, "octex: at least one spacing");
  RAFT_EXPECTS(n >= l, "octex: fewer vectors than the list length");
  RAFT_EXPECTS(n < (int64_t(1) << 31), "octex: at most 2^31 - 1 vectors");
  RAFT_EXPECTS(params.log_buckets >= 1 && params.log_buckets <= 30,
               "octex: log_buckets in [1, 30]");

  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto policy         = raft::resource::get_thrust_policy(res);
  RAFT_CUBLAS_TRY(cublasSetMathMode(raft::resource::get_cublas_handle(res), CUBLAS_DEFAULT_MATH));

  // the dataset stays on the device when it takes at most a third of the free memory
  device_rows rows(x, n, d, kProjectChunk, device_free_bytes() / 3, stream);
  rmm::device_uvector<float> d_mean(d, stream), d_u(d * w, stream), d_shift(w, stream);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_mean.data(), mean, d * sizeof(float), cudaMemcpyHostToDevice, stream));
  rmm::device_uvector<float> a_buf(std::min(kProjectChunk, n) * d, stream);
  rmm::device_uvector<float> y(n * w, stream);
  rmm::device_uvector<int64_t> h(n, stream), hs(n, stream);
  rmm::device_uvector<int32_t> idx(n, stream), is(n, stream);
  out.assign(c1 - c0, copy_cells{});
  for (int c = c0; c < c1; c++) {
    auto& cc = out[c - c0];
    auto tc  = std::chrono::steady_clock::now();
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_u.data(),
                                  proj + int64_t(c) * d * w,
                                  d * w * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_shift.data(), shift + int64_t(c) * w, w * sizeof(float), cudaMemcpyHostToDevice, stream));
    project_all(
      res, rows, n, d, d_mean.data(), d_u.data(), d_shift.data(), w, a_buf.data(), y.data());
    auto yv = raft::make_device_matrix_view<const float, int64_t>(y.data(), n, w);

    std::optional<cuvs::neighbors::brute_force::index<float, float>> bfi;
    std::optional<cuvs::neighbors::ivf_flat::index<float, int64_t>> ivf;
    if (params.member_lists) {
      // lists from cell members, no index
    } else if (params.exact_lists) {
      cuvs::neighbors::brute_force::index_params ip;
      ip.metric = cuvs::distance::DistanceType::L2Expanded;
      bfi.emplace(cuvs::neighbors::brute_force::build(res, ip, yv));
    } else {
      cuvs::neighbors::ivf_flat::index_params ip;
      ip.metric  = cuvs::distance::DistanceType::L2Expanded;
      ip.n_lists = std::max<uint32_t>(64, static_cast<uint32_t>(4 * std::sqrt(double(n))));
      ip.kmeans_trainset_fraction = n <= 4000000 ? 0.5 : 0.1;
      ivf.emplace(cuvs::neighbors::ivf_flat::build(res, ip, yv));
    }

    int64_t row = 0;
    for (int j = 0; j < L; j++) {
      const float f = params.spacings[j];
      cell_hash_kernel<<<grid(n), 256, 0, stream>>>(y.data(), n, b, f, h.data(), idx.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // unique cells (ascending signed hash), population and first member
      size_t tmp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(
        nullptr, tmp_bytes, h.data(), hs.data(), idx.data(), is.data(), n, 0, 64, stream);
      rmm::device_buffer tmp(tmp_bytes, stream);
      cub::DeviceRadixSort::SortPairs(
        tmp.data(), tmp_bytes, h.data(), hs.data(), idx.data(), is.data(), n, 0, 64, stream);
      // h and idx are free again: reuse h for the unique hashes, idx for the populations
      rmm::device_scalar<int64_t> d_runs(stream);
      int64_t* uh      = h.data();
      int32_t* pop     = idx.data();
      size_t rle_bytes = 0;
      cub::DeviceRunLengthEncode::Encode(
        nullptr, rle_bytes, hs.data(), uh, pop, d_runs.data(), n, stream);
      rmm::device_buffer rle_tmp(rle_bytes, stream);
      cub::DeviceRunLengthEncode::Encode(
        rle_tmp.data(), rle_bytes, hs.data(), uh, pop, d_runs.data(), n, stream);
      int64_t M = d_runs.value(stream);

      rmm::device_uvector<int32_t> start(M, stream), first(M, stream);
      size_t scan_bytes = 0;
      cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, pop, start.data(), M, stream);
      rmm::device_buffer scan_tmp(scan_bytes, stream);
      cub::DeviceScan::ExclusiveSum(scan_tmp.data(), scan_bytes, pop, start.data(), M, stream);
      run_first_kernel<<<grid(M), 256, 0, stream>>>(is.data(), start.data(), M, first.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      rmm::device_uvector<int64_t> keys(M, stream);
      if (params.min_cell > 1 &&
          j < L - 1) {  // lists only for populated cells; the coarsest level keeps all
        int mc  = params.min_cell;
        auto in = thrust::make_zip_iterator(thrust::make_tuple(uh, first.data()));
        auto o  = thrust::make_zip_iterator(thrust::make_tuple(keys.data(), start.data()));
        auto e  = thrust::copy_if(
          policy, in, in + M, pop, o, [mc] __device__(int32_t p) -> bool { return p >= mc; });
        M = e - o;
        keys.resize(M, stream);
        first.resize(M, stream);
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          first.data(), start.data(), M * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
      } else {
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(keys.data(), uh, M * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
      }

      // the list of every cell: the l projected vectors nearest to its center
      rmm::device_uvector<int32_t> lists(M * l, stream);
      if (params.member_lists && M > 0) {
        rmm::device_uvector<float> z(M * w, stream);
        cell_center_kernel<<<grid(M), 256, 0, stream>>>(y.data(), first.data(), M, b, f, z.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
        auto own = make_level_cells(res, y.data(), n, b, f);
        member_lists(res, y.data(), b, z.data(), M, l, {&own}, {f}, lists.data());
        raft::resource::sync_stream(res);
      } else {
        int64_t zc = std::min<int64_t>(kListChunk, std::max<int64_t>(M, 1));
        rmm::device_uvector<float> z(zc * w, stream), dist(zc * l, stream);
        rmm::device_uvector<int64_t> nb(zc * l, stream);
        for (int64_t z0 = 0; z0 < M; z0 += zc) {
          int64_t mz = std::min(zc, M - z0);
          cell_center_kernel<<<grid(mz), 256, 0, stream>>>(
            y.data(), first.data() + z0, mz, b, f, z.data());
          RAFT_CUDA_TRY(cudaPeekAtLastError());
          auto zv = raft::make_device_matrix_view<const float, int64_t>(z.data(), mz, w);
          auto nv = raft::make_device_matrix_view<int64_t, int64_t>(nb.data(), mz, l);
          auto dv = raft::make_device_matrix_view<float, int64_t>(dist.data(), mz, l);
          if (bfi) {
            cuvs::neighbors::brute_force::search(
              res, cuvs::neighbors::brute_force::search_params{}, *bfi, zv, nv, dv);
          } else {
            cuvs::neighbors::ivf_flat::search_params sp;
            sp.n_probes = params.list_probes;
            cuvs::neighbors::ivf_flat::search(res, sp, *ivf, zv, nv, dv);
          }
          narrow_kernel<<<grid(mz * l), 256, 0, stream>>>(nb.data(), mz * l, lists.data() + z0 * l);
          RAFT_CUDA_TRY(cudaPeekAtLastError());
        }
      }

      finish_level(res, params, keys.data(), lists.data(), M, row, cc);
    }
    raft::resource::sync_stream(res);
    build_log("  copy %d: %.2f s, cells per level", c, seconds_since(tc));
    for (auto m : cc.cells)
      build_log(" %lld", static_cast<long long>(m));
    build_log("\n");
  }
}

// ---- streaming path: no array of all n projected rows on the device, no O(n) host buffer ----

constexpr int64_t kStreamChunk  = kProjectChunk * 8;  // rows per host-to-device transfer
constexpr int kMaxLevels        = 16;
constexpr int64_t kMaxRangeRows = int64_t(1) << 26;  // rows per list index (default)

/**
 * Host rows to the device through two anonymous pinned staging buffers (cudaMallocHost; the
 * dataset itself is only read by the CPU, never pinned or registered). The CPU copy of the next
 * chunk overlaps the device work on the previous one; device slot k % 2 is reused only by work
 * ordered after the kernels that read it (one stream).
 */
class stream_stager {
 public:
  stream_stager(const half* host, int64_t d, int64_t rows, cudaStream_t stream)
    : host_(host), d_(d), rows_(rows), stream_(stream)
  {
    for (int k = 0; k < 2; k++) {
      RAFT_CUDA_TRY(cudaMallocHost(&pinned_[k], static_cast<size_t>(rows * d) * sizeof(half)));
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&copied_[k], cudaEventDisableTiming));
      dev_[k].emplace(static_cast<size_t>(rows * d), stream);
    }
  }
  stream_stager(stream_stager const&)                    = delete;
  auto operator=(stream_stager const&) -> stream_stager& = delete;
  ~stream_stager()
  {
    cudaStreamSynchronize(stream_);
    for (int k = 0; k < 2; k++) {
      cudaEventDestroy(copied_[k]);
      cudaFreeHost(pinned_[k]);
    }
  }
  [[nodiscard]] auto capacity() const -> int64_t { return rows_; }

  /** Device copy of host rows [r0, r0 + m). */
  auto rows(int64_t r0, int64_t m) -> const half*
  {
    return put(m, m, [&](half* dst, int64_t i0, int64_t i1) {
      std::memcpy(dst + i0 * d_, host_ + (r0 + i0) * d_, static_cast<size_t>((i1 - i0) * d_) * 2);
    });
  }

  /** Device copy of host rows idx[0..m), zero rows up to `pad` rows. */
  auto gather(const int32_t* idx, int64_t m, int64_t pad) -> const half*
  {
    return put(m, std::max(m, pad), [&](half* dst, int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; i++)
        std::memcpy(dst + i * d_, host_ + int64_t(idx[i]) * d_, static_cast<size_t>(d_) * 2);
    });
  }

 private:
  template <typename Fill>
  auto put(int64_t m, int64_t total, Fill const& fill) -> const half*
  {
    RAFT_EXPECTS(total <= rows_, "octex: staging chunk too small");
    int k = slot_;
    slot_ ^= 1;
    RAFT_CUDA_TRY(cudaEventSynchronize(copied_[k]));  // the previous transfer out of slot k is done
    half* dst = pinned_[k];
    int t     = static_cast<int>(std::clamp<int64_t>(m / 16384, 1, 16));
    std::vector<std::thread> th;
    for (int r = 1; r < t; r++)
      th.emplace_back([&, r] { fill(dst, m * r / t, m * (r + 1) / t); });
    fill(dst, 0, m / t);
    for (auto& x : th)
      x.join();
    if (total > m) std::memset(dst + m * d_, 0, static_cast<size_t>((total - m) * d_) * 2);
    RAFT_CUDA_TRY(cudaMemcpyAsync(dev_[k]->data(),
                                  dst,
                                  static_cast<size_t>(total * d_) * sizeof(half),
                                  cudaMemcpyHostToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaEventRecord(copied_[k], stream_));
    return dev_[k]->data();
  }

  const half* host_;
  int64_t d_, rows_;
  cudaStream_t stream_;
  half* pinned_[2]{nullptr, nullptr};
  cudaEvent_t copied_[2]{};
  std::optional<rmm::device_uvector<half>> dev_[2];
  int slot_ = 0;
};

/**
 * y [m, w] = (x - mean) @ u + shift for m device rows, in GEMMs of kProjectChunk rows (the same
 * calls as project_all when the rows start at a multiple of kProjectChunk, hence the same bits).
 */
inline void stream_project(raft::resources const& res,
                           const half* x,
                           int64_t m,
                           int64_t d,
                           const float* d_mean,
                           const float* d_u,
                           const float* d_shift,
                           int w,
                           float* a_buf,
                           float* y)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto handle         = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  const float one = 1.f, zero = 0.f;
  for (int64_t r0 = 0; r0 < m; r0 += kProjectChunk) {
    int64_t mm = std::min(kProjectChunk, m - r0);
    center(x + r0 * d, d_mean, a_buf, mm, d, stream);
    RAFT_CUBLAS_TRY(cublasSgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                w,
                                static_cast<int>(mm),
                                static_cast<int>(d),
                                &one,
                                d_u,
                                w,
                                a_buf,
                                static_cast<int>(d),
                                &zero,
                                y + r0 * w,
                                w));
  }
  add_shift_kernel<<<grid(m * w), 256, 0, stream>>>(y, d_shift, m, w);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
void drop(rmm::device_uvector<T>& v, cudaStream_t s)  // free the memory now
{
  v = rmm::device_uvector<T>(0, s);
}

struct stream_levels {
  int count;
  float f[kMaxLevels];
  int64_t* h[kMaxLevels];
  float* dist[kMaxLevels];  // squared distance to the cell center, or nullptr
};

/** h_g[r0 + i] = hash of the E8^b cell of Y[i] / f_g for the levels g of one pass. */
static __global__ void stream_hash_kernel(
  const float* y, int64_t m, int b, int64_t r0, stream_levels lv)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  const float* r = y + i * 8 * b;
  for (int g = 0; g < lv.count; g++) {
    float x[8 * kMaxBlocks];
    const float f = lv.f[g];
    for (int k = 0; k < 8 * b; k++)
      x[k] = r[k] / f;  // as cell_hash_kernel
    lv.h[g][r0 + i] = e8b_hash64(x, b, 1.0f);
    if (lv.dist[g] != nullptr) {
      float dd = 0.f;
      for (int j = 0; j < b; j++) {
        float p[8];
        e8_point(x + 8 * j, p);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          float e = r[8 * j + k] - p[k] * f;
          dd += e * e;
        }
      }
      lv.dist[g][r0 + i] = dd;
    }
  }
}

/** lists[i, t] = row t of cell i in sorted order, or its first row when the cell holds fewer. */
static __global__ void run_list_kernel(
  const int32_t* sidx, const int32_t* start, const int32_t* pop, int64_t m, int l, int32_t* lists)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m * l) return;
  int64_t c = i / l;
  int t     = static_cast<int>(i % l);
  lists[i]  = sidx[start[c] + (t < pop[c] ? t : 0)];
}

/** z[i] = f * E8^b point of Y[i] / f (as cell_center_kernel with first[i] = i). */
static __global__ void stream_center_kernel(const float* y, int64_t m, int b, float f, float* z)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  const float* r = y + i * 8 * b;
  float* o       = z + i * 8 * b;
  for (int j = 0; j < b; j++) {
    float x[8], p[8];
#pragma unroll
    for (int k = 0; k < 8; k++)
      x[k] = r[8 * j + k] / f;
    e8_point(x, p);
#pragma unroll
    for (int k = 0; k < 8; k++)
      o[8 * j + k] = p[k] * f;
  }
}

/**
 * Merge the l results of one row range (ascending distance, ids local to the range, invalid ids
 * outside [0, rows)) into the best l so far (ascending): out = best l of both, ids global.
 */
static __global__ void stream_merge_kernel(const float* bd,
                                           const int32_t* bi,
                                           const float* nd,
                                           const int64_t* nn,
                                           int64_t m,
                                           int l,
                                           int64_t a,
                                           int64_t rows,
                                           float* od,
                                           int32_t* oi)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  const float* x    = bd + i * l;
  const int32_t* xi = bi + i * l;
  const float* y    = nd + i * l;
  const int64_t* yi = nn + i * l;
  float* o          = od + i * l;
  int32_t* oo       = oi + i * l;
  const float inf   = __int_as_float(0x7f800000);
  int p = 0, q = 0;
  for (int k = 0; k < l; k++) {
    // skip invalid entries of the range
    while (q < l && (yi[q] < 0 || yi[q] >= rows))
      q++;
    float dy = q < l ? y[q] : inf;
    float dx = x[p];
    if (q < l && dy < dx) {  // ties keep the earlier range; p < l always holds here
      o[k]  = dy;
      oo[k] = static_cast<int32_t>(yi[q] + a);
      q++;
    } else {
      o[k]  = dx;
      oo[k] = xi[p];
      p++;
    }
  }
}

/** Device bytes of the whole-projection path for n rows (projection, hash and sort buffers, the
 * list index). */
inline auto full_build_bytes(int64_t n, int64_t d, int w, bool exact_lists, size_t free_b) -> size_t
{
  size_t per_row = size_t(w) * 4                     // y
                   + 8 + 8 + 4 + 4 + 12              // h, hs, idx, is, sort scratch
                   + (exact_lists ? 4 : w * 4 + 8);  // brute-force norms or the IVF-Flat copy
  size_t data = size_t(n) * d * sizeof(half);
  return size_t(n) * per_row + (data <= free_b / 3 ? data : 0);
}

/**
 * The cells of one copy with member lists (cell_params::member_lists), streamed: every row's hash
 * and squared distance to its cell center at every level in passes over the dataset, then per level
 * the rows sorted by distance and stably by hash, so that each run of a cell starts with its l
 * rows nearest to the center. No index over the projection, no centers, no gathers.
 */
inline void stream_member_copy(raft::resources const& res,
                               cell_params const& params,
                               std::function<const half*(int64_t, int64_t)> const& rows,
                               int64_t cap,
                               int64_t n,
                               int64_t d,
                               const float* d_mean,
                               const float* d_u,
                               const float* d_shift,
                               float* a_buf,
                               float* ychunk,
                               int& G,
                               copy_cells& cc)
{
  const int b         = params.blocks;
  const int w         = 8 * b;
  const int l         = params.list_len;
  const int L         = static_cast<int>(params.spacings.size());
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto policy         = raft::resource::get_thrust_policy(res);
  raft::resource::sync_stream(res);

  // sort buffers of one level (20 bytes per row), 12 bytes per row per level of a pass
  rmm::device_uvector<int64_t> alt_k(n, stream);
  rmm::device_uvector<int32_t> idx(n, stream), alt_i(n, stream);
  rmm::device_uvector<float> alt_d(n, stream);
  size_t sort_bytes = 0, fb = 0;
  {
    cub::DoubleBuffer<int64_t> kb(alt_k.data(), alt_k.data());
    cub::DoubleBuffer<int32_t> vb(idx.data(), alt_i.data());
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, kb, vb, n, 0, 64, stream);
    cub::DoubleBuffer<float> db(alt_d.data(), alt_d.data());
    cub::DeviceRadixSort::SortPairs(nullptr, fb, db, vb, n, 0, 32, stream);
    sort_bytes = std::max(sort_bytes, fb);
  }
  rmm::device_buffer sort_tmp(sort_bytes, stream);
  raft::resource::sync_stream(res);
  // levels per pass: sized once, at the first copy (later copies reuse the pooled memory, which
  // the free-memory count no longer shows)
  if (G == 0) {
    // per level of a pass: hash and distance (12 bytes per row); once: the run starts (up to 4
    // bytes per row), the kept cells and the scan and run-length scratch
    int64_t per = n * 12;
    G           = static_cast<int>(std::clamp<int64_t>(
      (int64_t(0.92 * double(device_free_bytes())) - n * 4 - (int64_t(1) << 30)) / per, 1, L));
    build_log("  member lists: %d level(s) per pass\n", G);
  }

  int64_t row = 0;
  for (int g0 = 0; g0 < L; g0 += G) {
    int g1 = std::min(L, g0 + G);
    std::vector<rmm::device_uvector<int64_t>> h;
    std::vector<rmm::device_uvector<float>> dist;
    h.reserve(g1 - g0);
    dist.reserve(g1 - g0);
    stream_levels lv{};
    lv.count = g1 - g0;
    for (int j = g0; j < g1; j++) {
      h.emplace_back(n, stream);
      dist.emplace_back(n, stream);
      lv.f[j - g0]    = params.spacings[j];
      lv.h[j - g0]    = h.back().data();
      lv.dist[j - g0] = dist.back().data();
    }
    for (int64_t r0 = 0; r0 < n; r0 += cap) {
      int64_t m      = std::min(cap, n - r0);
      const half* xr = rows(r0, m);
      stream_project(res, xr, m, d, d_mean, d_u, d_shift, w, a_buf, ychunk);
      stream_hash_kernel<<<grid(m), 256, 0, stream>>>(ychunk, m, b, r0, lv);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    for (int j = g0; j < g1; j++) {
      auto& hj = h[j - g0];
      auto& dj = dist[j - g0];
      thrust::sequence(policy, idx.begin(), idx.end());
      cub::DoubleBuffer<float> db(dj.data(), alt_d.data());
      cub::DoubleBuffer<int32_t> vb(idx.data(), alt_i.data());
      size_t tb = sort_bytes;
      cub::DeviceRadixSort::SortPairs(sort_tmp.data(), tb, db, vb, n, 0, 32, stream);
      drop(dj, stream);
      // hashes in distance order, then a stable sort by hash
      thrust::gather(policy, vb.Current(), vb.Current() + n, hj.data(), alt_k.data());
      cub::DoubleBuffer<int64_t> kb(alt_k.data(), hj.data());
      cub::DoubleBuffer<int32_t> pb(vb.Current(), vb.Alternate());
      tb = sort_bytes;
      cub::DeviceRadixSort::SortPairs(sort_tmp.data(), tb, kb, pb, n, 0, 64, stream);
      const int32_t* sidx = pb.Current();
      int64_t* uh         = kb.Alternate();
      int32_t* pop        = pb.Alternate();
      rmm::device_scalar<int64_t> d_runs(stream);
      size_t rle_bytes = 0;
      cub::DeviceRunLengthEncode::Encode(
        nullptr, rle_bytes, kb.Current(), uh, pop, d_runs.data(), n, stream);
      rmm::device_buffer rle_tmp(rle_bytes, stream);
      cub::DeviceRunLengthEncode::Encode(
        rle_tmp.data(), rle_bytes, kb.Current(), uh, pop, d_runs.data(), n, stream);
      int64_t M = d_runs.value(stream);
      rmm::device_uvector<int32_t> start(M, stream);
      {
        size_t scan_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, pop, start.data(), M, stream);
        rmm::device_buffer scan_tmp(scan_bytes, stream);
        cub::DeviceScan::ExclusiveSum(scan_tmp.data(), scan_bytes, pop, start.data(), M, stream);
      }
      // the kept cells (all at the coarsest level), counted first: M can reach hundreds of
      // millions at fine spacings, where few cells are kept
      const bool filter = params.min_cell > 1 && j < L - 1;
      const int mc      = params.min_cell;
      auto keep         = [mc] __device__(int32_t p) -> bool { return p >= mc; };
      int64_t Mk        = filter ? thrust::count_if(policy, pop, pop + M, keep) : M;
      rmm::device_uvector<int64_t> keys(Mk, stream);
      rmm::device_uvector<int32_t> ks(Mk, stream), kp(Mk, stream);
      auto in = thrust::make_zip_iterator(thrust::make_tuple(uh, start.data(), pop));
      auto o  = thrust::make_zip_iterator(thrust::make_tuple(keys.data(), ks.data(), kp.data()));
      if (filter) {
        thrust::copy_if(policy, in, in + M, pop, o, keep);
      } else {
        thrust::copy(policy, in, in + M, o);
      }
      rmm::device_uvector<int32_t> lists(Mk * l, stream);
      run_list_kernel<<<grid(Mk * l), 256, 0, stream>>>(
        sidx, ks.data(), kp.data(), Mk, l, lists.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      drop(hj, stream);
      finish_level(res, params, keys.data(), lists.data(), Mk, row, cc);
    }
  }
}

inline void build_cells_stream(raft::resources const& res,
                               cell_params const& params,
                               const half* x,
                               int64_t n,
                               int64_t d,
                               const float* mean,
                               const float* proj,
                               const float* shift,
                               int c0,
                               int c1,
                               std::vector<copy_cells>& out)
{
  const int b = params.blocks;
  const int w = 8 * b;
  const int l = params.list_len;
  const int L = static_cast<int>(params.spacings.size());
  RAFT_EXPECTS(L <= kMaxLevels, "octex: at most 16 spacings");

  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto policy         = raft::resource::get_thrust_policy(res);
  RAFT_CUBLAS_TRY(cublasSetMathMode(raft::resource::get_cublas_handle(res), CUBLAS_DEFAULT_MATH));

  stream_stager st(x, d, std::min(kStreamChunk, std::max<int64_t>(n, kProjectChunk)), stream);
  rmm::device_uvector<float> d_mean(d, stream), d_u(d * w, stream), d_shift(w, stream);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_mean.data(), mean, d * sizeof(float), cudaMemcpyHostToDevice, stream));
  rmm::device_uvector<float> a_buf(kProjectChunk * d, stream);
  rmm::device_uvector<float> ychunk(st.capacity() * w, stream);
  const char* env_ranges                                   = std::getenv("OCTEX_BUILD_RANGES");
  const bool member_mode                                   = params.member_lists;
  int member_G                                             = 0;
  const int64_t member_cap                                 = st.capacity();
  std::function<const half*(int64_t, int64_t)> member_rows = [&](int64_t r0, int64_t m) {
    return st.rows(r0, m);
  };

  out.assign(c1 - c0, copy_cells{});
  for (int c = c0; c < c1; c++) {
    auto& cc = out[c - c0];
    auto tc  = std::chrono::steady_clock::now();
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_u.data(),
                                  proj + int64_t(c) * d * w,
                                  d * w * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_shift.data(), shift + int64_t(c) * w, w * sizeof(float), cudaMemcpyHostToDevice, stream));

    if (member_mode) {
      stream_member_copy(res,
                         params,
                         member_rows,
                         member_cap,
                         n,
                         d,
                         d_mean.data(),
                         d_u.data(),
                         d_shift.data(),
                         a_buf.data(),
                         ychunk.data(),
                         member_G,
                         cc);
      build_log(
        "  copy %d (streamed, member lists): %.1f s; cells per level", c, seconds_since(tc));
      for (auto m : cc.cells)
        build_log(" %lld", static_cast<long long>(m));
      build_log("\n");
      continue;
    }

    // per level: populated cells (ascending hash) and their centers
    std::vector<rmm::device_uvector<int64_t>> keys;
    std::vector<rmm::device_uvector<float>> centers;
    keys.reserve(L);
    centers.reserve(L);
    {
      raft::resource::sync_stream(res);
      // sort buffers of one level (16 bytes per row) and at most 8 bytes per row of runs
      size_t free_b = device_free_bytes();
      int64_t fixed = n * (8 + 4 + 4 + 8) + (int64_t(256) << 20);
      int64_t per   = n * 8;
      int G =
        static_cast<int>(std::clamp<int64_t>((int64_t(0.85 * double(free_b)) - fixed) / per, 1, L));
      rmm::device_uvector<int64_t> alt_k(n, stream);
      rmm::device_uvector<int32_t> idx(n, stream), alt_i(n, stream);
      size_t sort_bytes = 0;
      {
        cub::DoubleBuffer<int64_t> kb(alt_k.data(), alt_k.data());
        cub::DoubleBuffer<int32_t> vb(idx.data(), alt_i.data());
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, kb, vb, n, 0, 64, stream);
      }
      rmm::device_buffer sort_tmp(sort_bytes, stream);

      for (int g0 = 0; g0 < L; g0 += G) {
        int g1 = std::min(L, g0 + G);
        std::vector<rmm::device_uvector<int64_t>> h;
        h.reserve(g1 - g0);
        stream_levels lv{};
        lv.count = g1 - g0;
        for (int j = g0; j < g1; j++) {
          h.emplace_back(n, stream);
          lv.f[j - g0] = params.spacings[j];
          lv.h[j - g0] = h.back().data();
        }
        // one pass over the dataset: project each chunk, hash it at the levels of this pass
        for (int64_t r0 = 0; r0 < n; r0 += st.capacity()) {
          int64_t m      = std::min(st.capacity(), n - r0);
          const half* xr = st.rows(r0, m);
          stream_project(res,
                         xr,
                         m,
                         d,
                         d_mean.data(),
                         d_u.data(),
                         d_shift.data(),
                         w,
                         a_buf.data(),
                         ychunk.data());
          stream_hash_kernel<<<grid(m), 256, 0, stream>>>(ychunk.data(), m, b, r0, lv);
          RAFT_CUDA_TRY(cudaPeekAtLastError());
        }

        for (int j = g0; j < g1; j++) {
          const float f = params.spacings[j];
          auto& hj      = h[j - g0];
          thrust::sequence(policy, idx.begin(), idx.end());
          cub::DoubleBuffer<int64_t> kb(hj.data(), alt_k.data());
          cub::DoubleBuffer<int32_t> vb(idx.data(), alt_i.data());
          size_t tb = sort_bytes;
          cub::DeviceRadixSort::SortPairs(sort_tmp.data(), tb, kb, vb, n, 0, 64, stream);
          // unique cells into the free halves of the double buffers
          int64_t* uh         = kb.Alternate();
          int32_t* pop        = vb.Alternate();
          const int32_t* sidx = vb.Current();
          rmm::device_scalar<int64_t> d_runs(stream);
          size_t rle_bytes = 0;
          cub::DeviceRunLengthEncode::Encode(
            nullptr, rle_bytes, kb.Current(), uh, pop, d_runs.data(), n, stream);
          rmm::device_buffer rle_tmp(rle_bytes, stream);
          cub::DeviceRunLengthEncode::Encode(
            rle_tmp.data(), rle_bytes, kb.Current(), uh, pop, d_runs.data(), n, stream);
          int64_t M = d_runs.value(stream);
          rmm::device_uvector<int32_t> first(M, stream);
          {
            rmm::device_uvector<int32_t> start(M, stream);
            size_t scan_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, pop, start.data(), M, stream);
            rmm::device_buffer scan_tmp(scan_bytes, stream);
            cub::DeviceScan::ExclusiveSum(
              scan_tmp.data(), scan_bytes, pop, start.data(), M, stream);
            run_first_kernel<<<grid(M), 256, 0, stream>>>(sidx, start.data(), M, first.data());
            RAFT_CUDA_TRY(cudaPeekAtLastError());
          }
          if (params.min_cell > 1 && j < L - 1) {  // the coarsest level keeps all cells
            int mc     = params.min_cell;
            auto keep  = [mc] __device__(int32_t p) -> bool { return p >= mc; };
            int64_t Mk = thrust::count_if(policy, pop, pop + M, keep);
            keys.emplace_back(Mk, stream);
            rmm::device_uvector<int32_t> fk(Mk, stream);
            auto in = thrust::make_zip_iterator(thrust::make_tuple(uh, first.data()));
            auto o  = thrust::make_zip_iterator(thrust::make_tuple(keys.back().data(), fk.data()));
            thrust::copy_if(policy, in, in + M, pop, o, keep);
            first = std::move(fk);
            M     = Mk;
          } else {
            keys.emplace_back(M, stream);
            RAFT_CUDA_TRY(cudaMemcpyAsync(
              keys.back().data(), uh, M * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
          }
          drop(hj, stream);  // this level's hashes are done

          // centers: the first members re-projected from the host rows (GEMMs of kProjectChunk
          // rows, as in the hash pass), decoded to the E8^b point
          std::vector<int32_t> hf(M);
          if (M > 0)
            RAFT_CUDA_TRY(cudaMemcpyAsync(
              hf.data(), first.data(), M * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
          raft::resource::sync_stream(res);
          drop(first, stream);
          centers.emplace_back(M * w, stream);
          for (int64_t z0 = 0; z0 < M; z0 += kProjectChunk) {
            int64_t mz     = std::min(kProjectChunk, M - z0);
            const half* xr = st.gather(hf.data() + z0, mz, kProjectChunk);
            stream_project(res,
                           xr,
                           kProjectChunk,
                           d,
                           d_mean.data(),
                           d_u.data(),
                           d_shift.data(),
                           w,
                           a_buf.data(),
                           ychunk.data());
            stream_center_kernel<<<grid(mz), 256, 0, stream>>>(
              ychunk.data(), mz, b, f, centers.back().data() + z0 * w);
            RAFT_CUDA_TRY(cudaPeekAtLastError());
          }
          raft::resource::sync_stream(res);
        }
      }
    }
    double t_cells = seconds_since(tc);

    // lists: the l nearest projected rows of every center, by one index per row range, merged
    std::vector<int64_t> Ms(L);
    int64_t Mmax = 1;
    for (int j = 0; j < L; j++) {
      Ms[j] = static_cast<int64_t>(keys[j].size());
      Mmax  = std::max(Mmax, Ms[j]);
    }
    std::vector<rmm::device_uvector<float>> bd;
    std::vector<rmm::device_uvector<int32_t>> bi;
    bd.reserve(L);
    bi.reserve(L);
    for (int j = 0; j < L; j++) {
      bd.emplace_back(Ms[j] * l, stream);
      bi.emplace_back(Ms[j] * l, stream);
      thrust::fill(
        policy, bd.back().begin(), bd.back().end(), std::numeric_limits<float>::infinity());
      thrust::fill(policy, bi.back().begin(), bi.back().end(), -1);
    }
    const int64_t zc = std::min<int64_t>(kListChunk, Mmax);
    rmm::device_uvector<float> dist(zc * l, stream), od(zc * l, stream);
    rmm::device_uvector<int64_t> nb(zc * l, stream);
    rmm::device_uvector<int32_t> oi(zc * l, stream);
    raft::resource::sync_stream(res);

    int64_t K = 0;
    if (env_ranges != nullptr && std::atoi(env_ranges) > 0) {
      K = std::atoi(env_ranges);
    } else {
      // projection of the range, its IVF-Flat copy, the k-means trainset and labels
      double per_row = params.exact_lists ? double(w) * 4 + 8 : 2.1 * w * 4 + 16;
      int64_t rmax   = static_cast<int64_t>(0.6 * double(device_free_bytes()) / per_row);
      // index build time grows as R^1.5 per range, the search of all centers as sqrt(R) per range:
      // ranges of at most 2^26 rows balance the two at a billion rows
      rmax = std::min(rmax, kMaxRangeRows);
      K    = (n + std::max<int64_t>(rmax, 1) - 1) / std::max<int64_t>(rmax, 1);
    }
    K = std::clamp<int64_t>(K, 1, std::max<int64_t>(1, n / std::max(l, 1)));
    // range bounds at multiples of kProjectChunk (the GEMMs of the hash pass)
    int64_t rlen = (n + K - 1) / K;
    rlen         = std::min(n, (rlen + kProjectChunk - 1) / kProjectChunk * kProjectChunk);
    double t_ivf = 0, t_search = 0;
    for (int64_t a = 0; a < n; a += rlen) {
      int64_t R = std::min(rlen, n - a);
      auto t1   = std::chrono::steady_clock::now();
      std::optional<rmm::device_uvector<float>> yr;
      yr.emplace(R * w, stream);
      for (int64_t r0 = 0; r0 < R; r0 += st.capacity()) {
        int64_t m      = std::min(st.capacity(), R - r0);
        const half* xr = st.rows(a + r0, m);
        stream_project(res,
                       xr,
                       m,
                       d,
                       d_mean.data(),
                       d_u.data(),
                       d_shift.data(),
                       w,
                       a_buf.data(),
                       yr->data() + r0 * w);
      }
      auto yv = raft::make_device_matrix_view<const float, int64_t>(yr->data(), R, w);
      std::optional<cuvs::neighbors::brute_force::index<float, float>> bfi;
      std::optional<cuvs::neighbors::ivf_flat::index<float, int64_t>> ivf;
      if (params.exact_lists) {
        cuvs::neighbors::brute_force::index_params ip;
        ip.metric = cuvs::distance::DistanceType::L2Expanded;
        bfi.emplace(cuvs::neighbors::brute_force::build(res, ip, yv));
      } else {
        cuvs::neighbors::ivf_flat::index_params ip;  // the settings of the whole-projection path
        ip.metric  = cuvs::distance::DistanceType::L2Expanded;
        ip.n_lists = std::max<uint32_t>(64, static_cast<uint32_t>(4 * std::sqrt(double(R))));
        ip.kmeans_trainset_fraction       = R <= 4000000 ? 0.5 : 0.1;
        ip.conservative_memory_allocation = true;  // memory only, the lists are the same
        ivf.emplace(cuvs::neighbors::ivf_flat::build(res, ip, yv));
        yr.reset();  // the index holds its own copy
      }
      raft::resource::sync_stream(res);
      auto t2 = std::chrono::steady_clock::now();
      t_ivf += std::chrono::duration<double>(t2 - t1).count();
      for (int j = 0; j < L; j++) {
        for (int64_t z0 = 0; z0 < Ms[j]; z0 += zc) {
          int64_t mz = std::min(zc, Ms[j] - z0);
          auto zv =
            raft::make_device_matrix_view<const float, int64_t>(centers[j].data() + z0 * w, mz, w);
          auto nv = raft::make_device_matrix_view<int64_t, int64_t>(nb.data(), mz, l);
          auto dv = raft::make_device_matrix_view<float, int64_t>(dist.data(), mz, l);
          if (bfi) {
            cuvs::neighbors::brute_force::search(
              res, cuvs::neighbors::brute_force::search_params{}, *bfi, zv, nv, dv);
          } else {
            cuvs::neighbors::ivf_flat::search_params sp;
            sp.n_probes = params.list_probes;
            cuvs::neighbors::ivf_flat::search(res, sp, *ivf, zv, nv, dv);
          }
          stream_merge_kernel<<<grid(mz, 128), 128, 0, stream>>>(bd[j].data() + z0 * l,
                                                                 bi[j].data() + z0 * l,
                                                                 dist.data(),
                                                                 nb.data(),
                                                                 mz,
                                                                 l,
                                                                 a,
                                                                 R,
                                                                 od.data(),
                                                                 oi.data());
          RAFT_CUDA_TRY(cudaPeekAtLastError());
          RAFT_CUDA_TRY(cudaMemcpyAsync(bd[j].data() + z0 * l,
                                        od.data(),
                                        mz * l * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        stream));
          RAFT_CUDA_TRY(cudaMemcpyAsync(bi[j].data() + z0 * l,
                                        oi.data(),
                                        mz * l * sizeof(int32_t),
                                        cudaMemcpyDeviceToDevice,
                                        stream));
        }
      }
      raft::resource::sync_stream(res);
      t_search += seconds_since(t2);
    }
    centers.clear();

    int64_t row = 0;
    for (int j = 0; j < L; j++) {
      finish_level(res, params, keys[j].data(), bi[j].data(), Ms[j], row, cc);
      drop(keys[j], stream);
      drop(bd[j], stream);
      drop(bi[j], stream);
    }
    build_log(
      "  copy %d (streamed, %lld ranges): cells %.1f s, list indexes %.1f s, list search "
      "%.1f s, total %.1f s; cells per level",
      c,
      static_cast<long long>((n + rlen - 1) / rlen),
      t_cells,
      t_ivf,
      t_search,
      seconds_since(tc));
    for (auto m : cc.cells)
      build_log(" %lld", static_cast<long long>(m));
    build_log("\n");
  }
}

/**
 * The cells of copies [c0, c1): with the whole projection on the device when it fits in 60% of
 * the free memory, else streamed (OCTEX_BUILD_STREAM=1 forces the streaming path, =0 the other).
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
                 std::vector<copy_cells>& out)
{
  const int b = params.blocks;
  const int L = static_cast<int>(params.spacings.size());
  RAFT_EXPECTS(b >= 1 && b <= kMaxBlocks, "octex: blocks must be in [1, 8]");
  RAFT_EXPECTS(L >= 1, "octex: at least one spacing");
  RAFT_EXPECTS(n >= params.list_len, "octex: fewer vectors than the list length");
  RAFT_EXPECTS(n < (int64_t(1) << 31), "octex: at most 2^31 - 1 vectors");
  RAFT_EXPECTS(params.log_buckets >= 1 && params.log_buckets <= 30,
               "octex: log_buckets in [1, 30]");
  bool stream;
  const char* env = std::getenv("OCTEX_BUILD_STREAM");
  if (env != nullptr && *env != '\0') {
    stream = std::atoi(env) != 0;
  } else {
    size_t free_b = device_free_bytes();
    stream = double(full_build_bytes(n, d, 8 * b, params.exact_lists, free_b)) > 0.6 * free_b;
  }
  build_log("build_cells: %s path for %lld rows\n",
            stream ? "streaming" : "whole-projection",
            static_cast<long long>(n));
  if (stream) {
    build_cells_stream(res, params, x, n, d, mean, proj, shift, c0, c1, out);
  } else {
    build_cells_full(res, params, x, n, d, mean, proj, shift, c0, c1, out);
  }
}

}  // namespace cuvs::neighbors::octex::detail
