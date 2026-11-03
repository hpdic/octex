/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "detail/index_impl.hpp"
#include "detail/introspect.hpp"
#include "detail/search_kernels.cuh"

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/cublas_macros.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cublasLt.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace cuvs::neighbors::octex {

namespace detail {

/** Buffers and graphs of one GPU for one batch shape and search parameters. */
struct gpu_plan {
  device_part* part = nullptr;
  int m1 = 0, q1 = 0;          // first round: m1 queries starting at batch row q1
  int lo = 0, hi = 0, mw = 0;  // waves: batch rows [lo, hi)
  int W = 0, logW = 0, W2 = 0, logW2 = 0;
  dbuf qbuf, qc, qh, Y, qs, seg, cand, dist, nin, si1, sd1;
  dbuf si2, sd2, done, seg2, cand2, dist2, n2, bi, bd, srci, srcd;
  long long* si      = nullptr;  // kept list of the waves (si2, or si1 without a cross-GPU merge)
  float* sd          = nullptr;
  cudaGraphExec_t g1 = nullptr, g2 = nullptr;
  cudaEvent_t ev1 = nullptr, ev2 = nullptr;
  int64_t buffer_bytes = 0;  // batch buffers of this plan
  int64_t mem_delta    = 0;  // device memory taken by creating the plan (buffers, graphs, cuBLASLt)
  // projection Y = qh @ ucat + shift: one cuBLASLt matmul with the shift as bias epilogue
  cublasLtHandle_t lt       = nullptr;
  cublasLtMatmulDesc_t op   = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  // unfused merge (ablation): candidate keys and pools (two buffers each for the sorts), their
  // segments, the sorted ids of earlier start points with their segments, fresh marks, CUB storage
  dbuf ukey, ukey2, upool, upool2, ubeg, uend, uoff1, uoff2, doff, dend, dsort, fresh, utmp;

  ~gpu_plan()
  {
    if (part == nullptr) return;
    cudaSetDevice(part->device);
    if (la != nullptr) cublasLtMatrixLayoutDestroy(la);
    if (lb != nullptr) cublasLtMatrixLayoutDestroy(lb);
    if (lc != nullptr) cublasLtMatrixLayoutDestroy(lc);
    if (op != nullptr) cublasLtMatmulDescDestroy(op);
    if (lt != nullptr) cublasLtDestroy(lt);
    if (g1 != nullptr) cudaGraphExecDestroy(g1);
    if (g2 != nullptr) cudaGraphExecDestroy(g2);
    if (ev1 != nullptr) cudaEventDestroy(ev1);
    if (ev2 != nullptr) cudaEventDestroy(ev2);
  }
};

struct plan {
  search_params sp;
  int m = 0, k = 0, KO = 0, lr = 0, R1 = 0, R2 = 0, FR = 0, P = 0;
  int pool1 = 0, pool2 = 0;  // unfused merge: keys per query of the first-round and wave pools
  bool shard = false;
  std::vector<std::unique_ptr<gpu_plan>> gpus;
  std::mutex mu;
};

namespace {

auto log2_ceil(int64_t v) -> int
{
  int e = 0;
  while ((int64_t{1} << e) < v)
    e++;
  return e;
}

auto grid(int64_t total, int threads) -> unsigned
{
  return static_cast<unsigned>((total + threads - 1) / threads);
}

template <typename T>
auto alloc(device_part const& p, int64_t count, int fill = -2) -> dbuf
{
  dbuf b(p.device, static_cast<size_t>(std::max<int64_t>(count, 1)) * sizeof(T));
  if (fill != -2) RAFT_CUDA_TRY(cudaMemset(b.ptr, fill, b.size));
  return b;
}

void launch_probe(index_impl const& ix, device_part const& p, gpu_plan& g, int lr, cudaStream_t s)
{
  const int L       = static_cast<int>(ix.spacings.size());
  const int64_t tot = static_cast<int64_t>(g.m1) * p.copies;
  const int64_t T1  = (int64_t{1} << ix.log_buckets) + 1;
  auto args         = std::make_tuple(g.Y.as<const __half>(),
                              g.m1,
                              p.copies,
                              L,
                              p.inv_f.as<const float>(),
                              p.offsets.as<const int>(),
                              T1,
                              p.keys.as<const long long>(),
                              p.mmax,
                              p.base.as<const int>(),
                              ix.list_len,
                              lr,
                              ix.log_buckets,
                              g.seg.as<int2>());
  auto go           = [&](auto kernel) {
    std::apply([&](auto... a) { kernel<<<grid(tot, 256), 256, 0, s>>>(a...); }, args);
  };
  switch (ix.blocks) {
    case 1: go(probe_kernel<1>); break;
    case 2: go(probe_kernel<2>); break;
    case 3: go(probe_kernel<3>); break;
    case 4: go(probe_kernel<4>); break;
    case 5: go(probe_kernel<5>); break;
    case 6: go(probe_kernel<6>); break;
    case 8: go(probe_kernel<8>); break;
    default: RAFT_FAIL("octex: blocks must be 1-6 or 8");
  }
}

void setup_projection(gpu_plan& g, device_part const& p, int D, int wdt)
{
  const int m1 = std::max(g.m1, 1);
  RAFT_CUBLAS_TRY(cublasLtCreate(&g.lt));
  RAFT_CUBLAS_TRY(cublasLtMatmulDescCreate(&g.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  const void* bias       = p.shift.ptr;
  cudaDataType_t btype   = CUDA_R_16F;
  RAFT_CUBLAS_TRY(
    cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  RAFT_CUBLAS_TRY(
    cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
    g.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype)));
  RAFT_CUBLAS_TRY(cublasLtMatrixLayoutCreate(&g.la, CUDA_R_16F, wdt, D, wdt));
  RAFT_CUBLAS_TRY(cublasLtMatrixLayoutCreate(&g.lb, CUDA_R_16F, D, m1, D));
  RAFT_CUBLAS_TRY(cublasLtMatrixLayoutCreate(&g.lc, CUDA_R_16F, wdt, m1, wdt));
  cublasLtMatmulPreference_t pref = nullptr;
  RAFT_CUBLAS_TRY(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = p.workspace.size;
  RAFT_CUBLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
    pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t res{};
  int found = 0;
  RAFT_CUBLAS_TRY(
    cublasLtMatmulAlgoGetHeuristic(g.lt, g.op, g.la, g.lb, g.lc, g.lc, pref, 1, &res, &found));
  cublasLtMatmulPreferenceDestroy(pref);
  RAFT_EXPECTS(found > 0, "octex: no cuBLASLt algorithm for the projection");
  g.algo = res.algo;
}

/** The sketch of a gather: one full local copy, or row ranges behind pointers (sharded or host). */
auto sketch_code(index_impl const& ix, device_part const& p) -> const signed char*
{
  return ix.crows == 0 ? p.sketch.as<const signed char>() : nullptr;
}
auto sketch_ptrs(index_impl const& ix, device_part const& p) -> const long long*
{
  return ix.crows == 0 ? nullptr : p.cptr.as<const long long>();
}

auto gather_smem(int W, int sw, int C) -> size_t
{
  return static_cast<size_t>(W + 2 * sw + 2 * C + 1) * sizeof(int);
}

auto gather_warp_smem(int W, int sw) -> size_t
{
  return static_cast<size_t>(W + 2 * sw) * sizeof(int);
}

/** CUB segmented radix sort of 64-bit keys between buffers a and b; returns the sorted ones. */
auto sort_keys(gpu_plan const& g,
               dbuf const& a,
               dbuf const& b,
               int64_t items,
               int nseg,
               const int* beg,
               const int* end,
               cudaStream_t s) -> const uint64_t*
{
  cub::DoubleBuffer<uint64_t> keys(a.as<uint64_t>(), b.as<uint64_t>());
  size_t bytes = g.utmp.size;
  RAFT_CUDA_TRY(cub::DeviceSegmentedRadixSort::SortKeys(
    g.utmp.ptr, bytes, keys, items, nseg, beg, end, 0, 64, s));
  return keys.Current();
}

/**
 * Kept lists of mq queries merged with their candidates without fused kernels (ablation of
 * rerank_keep_kernel and sketch_keep_kernel): all candidate keys sorted in global memory, a pool of
 * the kept list and the R best candidates (exact distances, or sketch distances in sketch-rank
 * mode), the pools sorted, and their first KO distinct keys kept.
 */
void unfused_keep(index_impl const& ix,
                  plan const& pl,
                  gpu_plan& g,
                  int mq,
                  const int* cand,
                  const float* dist,
                  const int* nin,
                  int W,
                  const float* Q,
                  int R,
                  int pool,
                  const int* pool_off,
                  long long* si,
                  float* sd,
                  bool fresh,
                  cudaStream_t s)
{
  const int D      = static_cast<int>(ix.dim);
  const bool exact = !pl.sp.sketch_rank;
  cand_keys_kernel<<<mq, 256, 0, s>>>(
    cand, dist, nin, W, g.ukey.as<uint64_t>(), g.ubeg.as<int>(), g.uend.as<int>());
  const uint64_t* sorted = sort_keys(
    g, g.ukey, g.ukey2, int64_t{mq} * W, mq, g.ubeg.as<const int>(), g.uend.as<const int>(), s);
  pool_keys_kernel<<<mq, 256, exact ? D * sizeof(float) : 0, s>>>(
    sorted,
    g.ubeg.as<const int>(),
    g.uend.as<const int>(),
    exact ? g.part->xptr.as<const long long>() : nullptr,
    ix.xrows,
    D,
    Q,
    exact ? R : pl.KO,
    si,
    sd,
    pl.KO,
    fresh,
    pool,
    g.upool.as<uint64_t>());
  const uint64_t* pooled =
    sort_keys(g, g.upool, g.upool2, int64_t{mq} * pool, mq, pool_off, pool_off + 1, s);
  keep_distinct_kernel<<<grid(mq, 128), 128, 0, s>>>(pooled, pool, pl.KO, mq, si, sd);
}

/**
 * Start points of wave wv without the fused selection (ablation of pick_starts_kernel): the ids
 * that earlier waves expanded sorted by CUB, then one launch marks the fresh kept entries by binary
 * search and one takes the first S of them.
 */
void unfused_pick(index_impl const& ix, plan const& pl, gpu_plan& g, int wv, cudaStream_t s)
{
  const int S = pl.sp.starts, E = S * pl.sp.waves, off = wv * S;
  if (off > 0) {
    size_t bytes = g.utmp.size;
    RAFT_CUDA_TRY(
      cub::DeviceSegmentedRadixSort::SortKeys(g.utmp.ptr,
                                              bytes,
                                              g.done.as<const long long>(),
                                              g.dsort.as<long long>(),
                                              int64_t{g.mw} * E,
                                              g.mw,
                                              g.doff.as<const int>(),
                                              g.dend.as<const int>() + int64_t{wv} * g.mw,
                                              0,
                                              64,
                                              s));
  }
  mark_fresh_kernel<<<grid(int64_t{g.mw} * pl.KO, 256), 256, 0, s>>>(
    g.si, pl.KO, g.dsort.as<const long long>(), E, off, g.mw, g.fresh.as<unsigned char>());
  take_starts_kernel<<<grid(g.mw, 128), 128, 0, s>>>(g.si,
                                                     g.fresh.as<const unsigned char>(),
                                                     pl.KO,
                                                     g.done.as<long long>(),
                                                     E,
                                                     off,
                                                     S,
                                                     ix.table_len,
                                                     g.mw,
                                                     g.seg2.as<int2>());
}

/** First round of a GPU: m1 queries over its copies, the best KO kept in si1 / sd1. */
void first_round(index_impl const& ix, plan const& pl, gpu_plan& g, cudaStream_t s)
{
  auto& p          = *g.part;
  const int D      = static_cast<int>(ix.dim);
  const int sw     = ix.sketch_width;
  const int64_t mD = static_cast<int64_t>(g.m1) * D;
  center_kernel<<<grid(mD, 256), 256, 0, s>>>(
    g.qbuf.as<float>(), p.mean.as<float>(), mD, D, g.qc.as<float>(), g.qh.as<__half>());
  const float one = 1.f, zero = 0.f;
  // row major Y [m1, wdt] = qh [m1, D] @ ucat [D, wdt] + shift (column major: Y^T = ucat^T qh^T +
  // shift)
  RAFT_CUBLAS_TRY(cublasLtMatmul(g.lt,
                                 g.op,
                                 &one,
                                 p.ucat.ptr,
                                 g.la,
                                 g.qh.ptr,
                                 g.lb,
                                 &zero,
                                 g.Y.ptr,
                                 g.lc,
                                 g.Y.ptr,
                                 g.lc,
                                 &g.algo,
                                 p.workspace.ptr,
                                 p.workspace.size,
                                 s));
  RAFT_CUBLAS_TRY(cublasSetStream(p.cublas, s));
  // qs [m1, sw] = qc [m1, D] @ basis [D, sw]
  RAFT_CUBLAS_TRY(cublasGemmEx(p.cublas,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               sw,
                               g.m1,
                               D,
                               &one,
                               p.basis.ptr,
                               CUDA_R_32F,
                               sw,
                               g.qc.ptr,
                               CUDA_R_32F,
                               D,
                               &zero,
                               g.qs.ptr,
                               CUDA_R_32F,
                               sw,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT));
  launch_probe(ix, p, g, pl.lr, s);
  auto gather = [&](auto kernel, size_t smem) {
    kernel<<<g.m1, 256, smem, s>>>(g.seg.as<const int2>(),
                                   p.copies,
                                   p.lists.as<const int>(),
                                   p.stride,
                                   nullptr,
                                   1,
                                   sketch_code(ix, p),
                                   sketch_ptrs(ix, p),
                                   std::max<int64_t>(ix.crows, 1),
                                   p.scale.as<const float>(),
                                   g.qs.as<const float>(),
                                   sw,
                                   g.W,
                                   g.logW,
                                   g.cand.as<int>(),
                                   g.dist.as<float>(),
                                   g.nin.as<int>());
  };
  if (pl.sp.balanced_gather)
    gather(gather_kernel<false>, gather_smem(g.W, sw, p.copies));
  else
    gather(gather_warp_kernel<false>, gather_warp_smem(g.W, sw));
  if (!pl.sp.fused_merge) {
    unfused_keep(ix,
                 pl,
                 g,
                 g.m1,
                 g.cand.as<const int>(),
                 g.dist.as<const float>(),
                 g.nin.as<const int>(),
                 g.W,
                 g.qbuf.as<const float>(),
                 pl.R1,
                 pl.pool1,
                 g.uoff1.as<const int>(),
                 g.si1.as<long long>(),
                 g.sd1.as<float>(),
                 true,
                 s);
  } else if (pl.sp.sketch_rank) {
    sketch_keep_kernel<<<g.m1, 512, 0, s>>>(g.cand.as<const int>(),
                                            g.dist.as<const float>(),
                                            g.nin.as<const int>(),
                                            g.W,
                                            pl.KO,
                                            g.si1.as<long long>(),
                                            g.sd1.as<float>(),
                                            true);
  } else {
    rerank_keep_kernel<<<g.m1, 256, D * sizeof(float), s>>>(g.cand.as<const int>(),
                                                            g.dist.as<const float>(),
                                                            g.nin.as<const int>(),
                                                            g.W,
                                                            p.xptr.as<const long long>(),
                                                            ix.xrows,
                                                            D,
                                                            g.qbuf.as<const float>(),
                                                            pl.R1,
                                                            pl.KO,
                                                            g.si1.as<long long>(),
                                                            g.sd1.as<float>(),
                                                            true);
  }
}

/** Waves of a GPU over batch rows [lo, hi): merge of the kept lists (sharded), waves, best k into
 * bi / bd. */
void wave_round(index_impl const& ix, plan const& pl, gpu_plan& g, cudaStream_t s)
{
  auto& p          = *g.part;
  const int D      = static_cast<int>(ix.dim);
  const int sw     = ix.sketch_width;
  const int S      = pl.sp.starts;
  const int nw     = pl.sp.waves;
  const float* Qw  = g.qbuf.as<const float>() + static_cast<int64_t>(g.lo - g.q1) * D;
  const float* qsw = g.qs.as<const float>() + static_cast<int64_t>(g.lo - g.q1) * sw;
  if (g.mw == 0) return;
  if (pl.shard) {
    merge_kept_kernel<<<g.mw, std::clamp(pl.P / 2, 64, 512), pl.P * sizeof(uint64_t), s>>>(
      g.srci.as<const long long>(),
      g.srcd.as<const long long>(),
      static_cast<int>(ix.parts.size()),
      g.lo,
      pl.KO,
      pl.P,
      g.si,
      g.sd);
  }
  auto gather = [&](auto kernel, size_t smem) {
    kernel<<<g.mw, 512, smem, s>>>(g.seg2.as<const int2>(),
                                   S,
                                   nullptr,
                                   ix.table_len,
                                   p.tptr.as<const long long>(),
                                   ix.trows,
                                   sketch_code(ix, p),
                                   sketch_ptrs(ix, p),
                                   std::max<int64_t>(ix.crows, 1),
                                   p.scale.as<const float>(),
                                   qsw,
                                   sw,
                                   g.W2,
                                   g.logW2,
                                   g.cand2.as<int>(),
                                   g.dist2.as<float>(),
                                   g.n2.as<int>());
  };
  for (int wv = 0; wv < nw; wv++) {
    if (pl.sp.fused_merge)
      pick_starts_kernel<<<g.mw, 256, 0, s>>>(
        g.si, pl.KO, g.done.as<long long>(), S * nw, wv * S, S, ix.table_len, g.seg2.as<int2>());
    else
      unfused_pick(ix, pl, g, wv, s);
    if (pl.sp.balanced_gather)
      gather(gather_kernel<true>, gather_smem(g.W2, sw, S));
    else
      gather(gather_warp_kernel<true>, gather_warp_smem(g.W2, sw));
    if (!pl.sp.fused_merge) {
      unfused_keep(ix,
                   pl,
                   g,
                   g.mw,
                   g.cand2.as<const int>(),
                   g.dist2.as<const float>(),
                   g.n2.as<const int>(),
                   g.W2,
                   Qw,
                   pl.R2,
                   pl.pool2,
                   g.uoff2.as<const int>(),
                   g.si,
                   g.sd,
                   false,
                   s);
    } else if (pl.sp.sketch_rank) {
      sketch_keep_kernel<<<g.mw, 512, 0, s>>>(g.cand2.as<const int>(),
                                              g.dist2.as<const float>(),
                                              g.n2.as<const int>(),
                                              g.W2,
                                              pl.KO,
                                              g.si,
                                              g.sd,
                                              false);
    } else {
      rerank_keep_kernel<<<g.mw, 256, D * sizeof(float), s>>>(g.cand2.as<const int>(),
                                                              g.dist2.as<const float>(),
                                                              g.n2.as<const int>(),
                                                              g.W2,
                                                              p.xptr.as<const long long>(),
                                                              ix.xrows,
                                                              D,
                                                              Qw,
                                                              pl.R2,
                                                              pl.KO,
                                                              g.si,
                                                              g.sd,
                                                              false);
    }
  }
  if (pl.sp.sketch_rank && pl.FR > 0) {
    final_rerank_kernel<<<g.mw, 256, D * sizeof(float), s>>>(g.si,
                                                             pl.KO,
                                                             pl.FR,
                                                             p.xptr.as<const long long>(),
                                                             ix.xrows,
                                                             D,
                                                             Qw,
                                                             pl.k,
                                                             g.bi.as<long long>(),
                                                             g.bd.as<float>());
  } else {
    take_k_kernel<<<grid(int64_t{g.mw} * pl.k, 256), 256, 0, s>>>(
      g.si, g.sd, pl.KO, pl.k, g.mw, g.bi.as<long long>(), g.bd.as<float>());
  }
}

void set_smem_limits(gpu_plan const& g, int sw, int C, int S, bool balanced)
{
  if (!balanced) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(gather_warp_kernel<false>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(gather_warp_smem(g.W, sw))));
    RAFT_CUDA_TRY(cudaFuncSetAttribute(gather_warp_kernel<true>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(gather_warp_smem(g.W2, sw))));
  }
  RAFT_CUDA_TRY(cudaFuncSetAttribute(gather_kernel<false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(gather_smem(g.W, sw, C))));
  RAFT_CUDA_TRY(cudaFuncSetAttribute(gather_kernel<true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(gather_smem(g.W2, sw, S))));
  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    merge_kept_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096 * sizeof(uint64_t)));
}

auto capture(cudaStream_t s, std::function<void()> const& body) -> cudaGraphExec_t
{
  cudaGraph_t graph = nullptr;
  RAFT_CUDA_TRY(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  body();
  RAFT_CUDA_TRY(cudaStreamEndCapture(s, &graph));
  cudaGraphExec_t exec = nullptr;
  RAFT_CUDA_TRY(cudaGraphInstantiate(&exec, graph, 0));
  RAFT_CUDA_TRY(cudaGraphDestroy(graph));
  return exec;
}

/** Buffers of the unfused merge and selection (ablation) of a GPU: m1 first-round and mw wave
 * queries. */
void setup_unfused(plan const& pl, gpu_plan& g, int m1, int mw)
{
  auto& p             = *g.part;
  const int nw        = pl.sp.waves;
  const int64_t E     = int64_t{pl.sp.starts} * nw;
  const int64_t nkey  = std::max(int64_t{m1} * g.W, nw > 0 ? int64_t{mw} * g.W2 : 0);
  const int64_t npool = std::max(int64_t{m1} * pl.pool1, nw > 0 ? int64_t{mw} * pl.pool2 : 0);
  RAFT_EXPECTS(nkey < INT_MAX && npool < INT_MAX && int64_t{mw} * E < INT_MAX,
               "octex: batch too large for the 32-bit segments of the unfused merge");
  g.ukey      = alloc<uint64_t>(p, nkey);
  g.ukey2     = alloc<uint64_t>(p, nkey);
  g.upool     = alloc<uint64_t>(p, npool);
  g.upool2    = alloc<uint64_t>(p, npool);
  g.ubeg      = alloc<int>(p, std::max(m1, mw));
  g.uend      = alloc<int>(p, std::max(m1, mw));
  auto upload = [&](std::vector<int> const& h) {
    dbuf b = alloc<int>(p, static_cast<int64_t>(h.size()));
    RAFT_CUDA_TRY(cudaMemcpy(b.ptr, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
    return b;
  };
  std::vector<int> o1, o2, od, oe;
  for (int q = 0; q <= m1; q++)
    o1.push_back(q * pl.pool1);
  for (int q = 0; q <= mw; q++)
    o2.push_back(q * pl.pool2);
  g.uoff1 = upload(o1);
  g.uoff2 = upload(o2);
  // CUB temporary storage: the largest of all sorts
  size_t tmp = 0, bytes = 0;
  auto need = [&](int64_t items, int nseg) {
    cub::DoubleBuffer<uint64_t> keys(nullptr, nullptr);
    RAFT_CUDA_TRY(cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr, bytes, keys, items, nseg, g.ubeg.as<const int>(), g.uend.as<const int>(), 0, 64));
    tmp = std::max(tmp, bytes);
  };
  need(int64_t{m1} * g.W, m1);
  need(int64_t{m1} * pl.pool1, m1);
  if (nw > 0) {
    need(int64_t{mw} * g.W2, mw);
    need(int64_t{mw} * pl.pool2, mw);
    // start points of a query: done[q E, q E + wv S) sorted before wave wv
    for (int q = 0; q <= mw; q++)
      od.push_back(static_cast<int>(q * E));
    for (int wv = 0; wv < nw; wv++)
      for (int q = 0; q < mw; q++)
        oe.push_back(static_cast<int>(q * E + int64_t{wv} * pl.sp.starts));
    g.doff  = upload(od);
    g.dend  = upload(oe);
    g.dsort = alloc<long long>(p, int64_t{mw} * E);
    g.fresh = alloc<unsigned char>(p, int64_t{mw} * pl.KO);
    RAFT_CUDA_TRY(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                          bytes,
                                                          static_cast<const long long*>(nullptr),
                                                          static_cast<long long*>(nullptr),
                                                          int64_t{mw} * E,
                                                          mw,
                                                          g.doff.as<const int>(),
                                                          g.dend.as<const int>(),
                                                          0,
                                                          64));
    tmp = std::max(tmp, bytes);
  }
  g.utmp = alloc<char>(p, static_cast<int64_t>(tmp));
}

auto make_plan(index_impl& ix, search_params const& sp, int m, int k) -> std::shared_ptr<plan>
{
  auto pl      = std::make_shared<plan>();
  pl->sp       = sp;
  pl->m        = m;
  pl->k        = k;
  pl->KO       = std::max(sp.starts, k);
  pl->lr       = std::min(sp.read, ix.list_len);
  pl->R1       = sp.rerank;
  pl->R2       = sp.wave_rerank;
  pl->FR       = std::min(sp.final_rerank, pl->KO);
  pl->shard    = ix.params.mode == placement::shard && ix.parts.size() > 1;
  pl->pool1    = pl->KO + (sp.sketch_rank ? pl->KO : pl->R1);
  pl->pool2    = pl->KO + (sp.sketch_rank ? pl->KO : pl->R2);
  const int ng = static_cast<int>(ix.parts.size());
  const int D  = static_cast<int>(ix.dim);
  const int sw = ix.sketch_width;
  const int S  = sp.starts;
  RAFT_EXPECTS(sp.read >= 1 && sp.starts >= 1 && sp.waves >= 0,
               "octex: read, starts >= 1 and waves >= 0");
  RAFT_EXPECTS(sp.waves == 0 || ix.table_len > 0, "octex: waves need the neighbor table");
  // exact ranking keeps the start points next to the reranked entries (512 each); sketch ranking
  // keeps up to 1024
  RAFT_EXPECTS(pl->KO <= (sp.sketch_rank ? kMaxSketchKO : kMaxRerank),
               "octex: max(starts, k) must be <= 512 (1024 with sketch_rank)");
  RAFT_EXPECTS(sp.sketch_rank ||
                 (sp.rerank >= 1 && sp.rerank <= kMaxRerank && pl->KO + sp.rerank <= kMaxKeepSort),
               "octex: rerank in [1, 512] and max(starts, k) + rerank <= 1024");
  RAFT_EXPECTS(sp.sketch_rank || sp.waves == 0 ||
                 (sp.wave_rerank >= 1 && sp.wave_rerank <= kMaxRerank &&
                  pl->KO + sp.wave_rerank <= kMaxKeepSort),
               "octex: wave_rerank in [1, 512] and max(starts, k) + wave_rerank <= 1024");
  RAFT_EXPECTS(!sp.sketch_rank || pl->FR == 0 || (k <= pl->FR && pl->FR <= kMaxRerank),
               "octex: final_rerank 0 (sketch order) or k <= final_rerank <= 512");
  RAFT_EXPECTS(sp.waves <= 1 || S * (sp.waves - 1) <= kMaxExpanded,
               "octex: starts * (waves - 1) <= 6144");
  RAFT_EXPECTS(!pl->shard || ng * pl->KO <= 4096, "octex: devices * max(starts, k) <= 4096");
  pl->P = 2;
  while (pl->P < ng * pl->KO)
    pl->P <<= 1;
  std::vector<size_t> free0(ng);
  for (int gi = 0; gi < ng; gi++) {
    size_t total = 0;
    RAFT_CUDA_TRY(cudaSetDevice(ix.parts[gi]->device));
    RAFT_CUDA_TRY(cudaMemGetInfo(&free0[gi], &total));
  }

  for (int gi = 0; gi < ng; gi++) {
    auto gp      = std::make_unique<gpu_plan>();
    auto& g      = *gp;
    auto& p      = *ix.parts[gi];
    g.part       = &p;
    g.lo         = static_cast<int>(int64_t{m} * gi / ng);
    g.hi         = static_cast<int>(int64_t{m} * (gi + 1) / ng);
    g.mw         = g.hi - g.lo;
    g.q1         = pl->shard ? 0 : g.lo;
    g.m1         = pl->shard ? m : g.mw;
    const int m1 = std::max(g.m1, 1), mw = std::max(g.mw, 1);
    // hash sets of 1.5 x the entries read, capped at the shared-memory maximum: a set at least as
    // large as the entries read never overflows, it only probes longer when nearly full
    const int log_max = log2_ceil(kMaxHash);
    RAFT_EXPECTS(int64_t{p.copies} * pl->lr <= kMaxHash,
                 "octex: copies x read too large for the shared-memory hash set");
    g.logW = std::clamp(log2_ceil(static_cast<int64_t>(std::ceil(p.copies * pl->lr * kHashGrow))),
                        kMinLogHash,
                        log_max);
    g.W    = 1 << g.logW;
    if (sp.waves > 0) {
      RAFT_EXPECTS(int64_t{S} * ix.table_len <= kMaxHash,
                   "octex: starts x table_len too large for the shared-memory hash set");
      g.logW2 = std::clamp(log2_ceil(static_cast<int64_t>(std::ceil(S * ix.table_len * kHashGrow))),
                           kMinLogHash,
                           log_max);
      g.W2    = 1 << g.logW2;
    }
    RAFT_CUDA_TRY(cudaSetDevice(p.device));
    const int wdt = 8 * ix.blocks * p.copies;
    g.qbuf        = alloc<float>(p, int64_t{m1} * D, 0);
    g.qc          = alloc<float>(p, int64_t{m1} * D);
    g.qh          = alloc<__half>(p, int64_t{m1} * D);
    g.Y           = alloc<__half>(p, int64_t{m1} * wdt);
    g.qs          = alloc<float>(p, int64_t{m1} * sw);
    g.seg         = alloc<int2>(p, int64_t{m1} * p.copies);
    g.cand        = alloc<int>(p, int64_t{m1} * g.W);
    g.dist        = alloc<float>(p, int64_t{m1} * g.W);
    g.nin         = alloc<int>(p, m1);
    g.si1         = alloc<long long>(p, int64_t{m1} * pl->KO, 0xff);
    g.sd1         = alloc<float>(p, int64_t{m1} * pl->KO);
    if (pl->shard) {
      g.si2 = alloc<long long>(p, int64_t{mw} * pl->KO, 0xff);
      g.sd2 = alloc<float>(p, int64_t{mw} * pl->KO);
      g.si  = g.si2.as<long long>();
      g.sd  = g.sd2.as<float>();
    } else {
      g.si = g.si1.as<long long>();
      g.sd = g.sd1.as<float>();
    }
    if (sp.waves > 0) {
      g.done  = alloc<long long>(p, int64_t{mw} * S * sp.waves);
      g.seg2  = alloc<int2>(p, int64_t{mw} * S);
      g.cand2 = alloc<int>(p, int64_t{mw} * g.W2);
      g.dist2 = alloc<float>(p, int64_t{mw} * g.W2);
      g.n2    = alloc<int>(p, mw);
    }
    g.bi = alloc<long long>(p, int64_t{mw} * k);
    g.bd = alloc<float>(p, int64_t{mw} * k);
    if (!sp.fused_merge) setup_unfused(*pl, g, m1, mw);
    for (dbuf const* b :
         {&g.qbuf,  &g.qc,   &g.qh,   &g.Y,     &g.qs,    &g.seg,    &g.cand,  &g.dist,  &g.nin,
          &g.si1,   &g.sd1,  &g.si2,  &g.sd2,   &g.done,  &g.seg2,   &g.cand2, &g.dist2, &g.n2,
          &g.bi,    &g.bd,   &g.ukey, &g.ukey2, &g.upool, &g.upool2, &g.ubeg,  &g.uend,  &g.uoff1,
          &g.uoff2, &g.doff, &g.dend, &g.dsort, &g.fresh, &g.utmp})
      g.buffer_bytes += static_cast<int64_t>(b->size);
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&g.ev1, cudaEventDisableTiming));
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&g.ev2, cudaEventDisableTiming));
    setup_projection(g, p, D, wdt);
    pl->gpus.push_back(std::move(gp));
  }
  // kept lists of every GPU, read by the merge of every GPU
  if (pl->shard) {
    std::vector<long long> pi, pd;
    for (auto& g : pl->gpus) {
      pi.push_back(reinterpret_cast<long long>(g->si1.ptr));
      pd.push_back(reinterpret_cast<long long>(g->sd1.ptr));
    }
    for (auto& g : pl->gpus) {
      RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
      g->srci = alloc<long long>(*g->part, ng);
      g->srcd = alloc<long long>(*g->part, ng);
      RAFT_CUDA_TRY(
        cudaMemcpy(g->srci.ptr, pi.data(), ng * sizeof(long long), cudaMemcpyHostToDevice));
      RAFT_CUDA_TRY(
        cudaMemcpy(g->srcd.ptr, pd.data(), ng * sizeof(long long), cudaMemcpyHostToDevice));
    }
  }
  // one eager run (module loading, cuBLAS heuristics), then capture: first round and waves in one
  // graph, or two graphs when the waves read the kept lists of all GPUs
  for (auto& g : pl->gpus) {
    RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
    set_smem_limits(*g, sw, g->part->copies, S, sp.balanced_gather);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    first_round(ix, *pl, *g, g->part->stream);
  }
  for (auto& g : pl->gpus) {
    RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
    RAFT_CUDA_TRY(cudaStreamSynchronize(g->part->stream));
  }
  for (auto& g : pl->gpus) {
    RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
    wave_round(ix, *pl, *g, g->part->stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(g->part->stream));
  }
  // without graphs (ablation), every batch launches the kernels directly
  for (auto& g : pl->gpus) {
    if (!sp.use_graph) break;
    RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
    cudaStream_t s = g->part->stream;
    if (pl->shard) {
      g->g1 = capture(s, [&] { first_round(ix, *pl, *g, s); });
      g->g2 = capture(s, [&] { wave_round(ix, *pl, *g, s); });
    } else {
      g->g1 = capture(s, [&] {
        first_round(ix, *pl, *g, s);
        wave_round(ix, *pl, *g, s);
      });
    }
  }
  for (int gi = 0; gi < ng; gi++) {
    size_t fr = 0, total = 0;
    RAFT_CUDA_TRY(cudaSetDevice(ix.parts[gi]->device));
    RAFT_CUDA_TRY(cudaMemGetInfo(&fr, &total));
    pl->gpus[gi]->mem_delta = static_cast<int64_t>(free0[gi]) - static_cast<int64_t>(fr);
  }
  return pl;
}

auto get_plan(index_impl& ix, search_params const& sp, int m, int k) -> std::shared_ptr<plan>
{
  std::string key = std::to_string(m) + "/" + std::to_string(k) + "/" + std::to_string(sp.read) +
                    "/" + std::to_string(sp.starts) + "/" + std::to_string(sp.waves) + "/" +
                    std::to_string(sp.rerank) + "/" + std::to_string(sp.wave_rerank) + "/" +
                    std::to_string(sp.sketch_rank) + "/" + std::to_string(sp.final_rerank) + "/" +
                    std::to_string(sp.balanced_gather) + std::to_string(sp.fused_merge) +
                    std::to_string(sp.use_graph);
  std::lock_guard<std::mutex> lock(ix.mu);
  auto it = ix.plans.find(key);
  if (it != ix.plans.end()) return it->second;
  // one cached configuration: the buffers and graphs of the previous one are freed before the new
  // ones are allocated, so a sweep over search parameters never holds more than one set in device
  // memory
  ix.plans.clear();
  auto pl       = make_plan(ix, sp, m, k);
  ix.plans[key] = pl;
  return pl;
}

void search_impl(raft::resources const& res,
                 search_params const& sp,
                 index const& idx,
                 const float* queries,
                 int64_t nq,
                 int64_t qdim,
                 int64_t* neighbors,
                 float* distances,
                 int64_t k,
                 bool host_out,
                 std::vector<gpu_phase>* phases = nullptr,
                 bool resident                  = false)
{
  auto& ix = idx.impl();
  RAFT_EXPECTS(qdim == ix.dim, "octex: query dimension differs from the index");
  RAFT_EXPECTS(k >= 1, "octex: k >= 1");
  if (nq == 0) return;
  int cur = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cur));
  cudaStream_t caller = raft::resource::get_cuda_stream(res).get();
  const int m         = static_cast<int>(std::min<int64_t>(std::max(sp.batch, 1), nq));
  auto pl             = get_plan(ix, sp, m, static_cast<int>(k));
  std::lock_guard<std::mutex> lock(pl->mu);
  const int D = static_cast<int>(ix.dim);

  cudaEvent_t start = nullptr;
  RAFT_CUDA_TRY(cudaSetDevice(cur));
  RAFT_CUDA_TRY(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
  RAFT_CUDA_TRY(cudaEventRecord(start, caller));
  for (auto& g : pl->gpus) {
    RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(g->part->stream, start, 0));
  }
  // timing of the first batch per GPU: events around the first-round graph and the wave graph
  const int ng = static_cast<int>(pl->gpus.size());
  std::vector<cudaEvent_t> te(phases ? 4 * ng : 0, nullptr);
  for (int gi = 0; gi < static_cast<int>(te.size()) / 4; gi++) {
    RAFT_CUDA_TRY(cudaSetDevice(pl->gpus[gi]->part->device));
    for (int j = 0; j < 4; j++)
      RAFT_CUDA_TRY(cudaEventCreate(&te[4 * gi + j]));
  }
  auto mark = [&](int gi, int j, int64_t b0, cudaStream_t s) {
    if (phases && b0 == 0) RAFT_CUDA_TRY(cudaEventRecord(te[4 * gi + j], s));
  };
  for (int64_t b0 = 0; b0 < nq; b0 += m) {
    const int n = static_cast<int>(std::min<int64_t>(m, nq - b0));
    int gi      = 0;
    for (auto& g : pl->gpus) {
      cudaStream_t s = g->part->stream;
      RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
      if (pl->shard)  // the kept lists are rewritten: every GPU's waves of the previous batch are
                      // done
        for (auto& h : pl->gpus)
          if (h != g) RAFT_CUDA_TRY(cudaStreamWaitEvent(s, h->ev2, 0));
      int have = std::clamp(n - g->q1, 0, g->m1);
      auto* qb = g->qbuf.as<float>();
      if (resident) {  // the queries of the previous search of this batch are still in place
        have = g->m1;
      } else if (have > 0) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          qb, queries + (b0 + g->q1) * D, sizeof(float) * have * D, cudaMemcpyDefault, s));
      } else {
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(qb, queries + b0 * D, sizeof(float) * D, cudaMemcpyDefault, s));
        have = 1;
      }
      if (have < g->m1)
        pad_rows_kernel<<<grid(int64_t{g->m1 - have} * D, 256), 256, 0, s>>>(qb, have, g->m1, D);
      mark(gi, 0, b0, s);
      if (sp.use_graph)
        RAFT_CUDA_TRY(cudaGraphLaunch(g->g1, s));
      else
        first_round(ix, *pl, *g, s);
      mark(gi, 1, b0, s);
      RAFT_CUDA_TRY(cudaEventRecord(g->ev1, s));
      gi++;
    }
    gi = 0;
    if (!sp.use_graph) {
      // without graphs (ablation): a host barrier on every GPU between the first round and the
      // waves, then direct launches of the waves
      for (auto& g : pl->gpus) {
        RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
        RAFT_CUDA_TRY(cudaStreamSynchronize(g->part->stream));
      }
      for (auto& g : pl->gpus) {
        cudaStream_t s = g->part->stream;
        RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
        mark(gi, 2, b0, s);
        wave_round(ix, *pl, *g, s);
        mark(gi, 3, b0, s);
        gi++;
      }
    } else if (pl->shard) {
      for (auto& g : pl->gpus) {
        cudaStream_t s = g->part->stream;
        RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
        for (auto& h : pl->gpus)
          if (h != g) RAFT_CUDA_TRY(cudaStreamWaitEvent(s, h->ev1, 0));
        mark(gi, 2, b0, s);
        RAFT_CUDA_TRY(cudaGraphLaunch(g->g2, s));
        mark(gi, 3, b0, s);
        gi++;
      }
    }
    for (auto& g : pl->gpus) {
      cudaStream_t s = g->part->stream;
      RAFT_CUDA_TRY(cudaSetDevice(g->part->device));
      int rows = std::clamp(n - g->lo, 0, g->mw);
      if (rows > 0 && !resident) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(neighbors + (b0 + g->lo) * k,
                                      g->bi.ptr,
                                      sizeof(int64_t) * rows * k,
                                      cudaMemcpyDefault,
                                      s));
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          distances + (b0 + g->lo) * k, g->bd.ptr, sizeof(float) * rows * k, cudaMemcpyDefault, s));
      }
      RAFT_CUDA_TRY(cudaEventRecord(g->ev2, s));
    }
  }
  RAFT_CUDA_TRY(cudaSetDevice(cur));
  for (auto& g : pl->gpus)
    RAFT_CUDA_TRY(cudaStreamWaitEvent(caller, g->ev2, 0));
  if (host_out) RAFT_CUDA_TRY(cudaStreamSynchronize(caller));
  RAFT_CUDA_TRY(cudaEventDestroy(start));
  if (phases) {
    phases->clear();
    for (int gi = 0; gi < ng; gi++) {
      auto& g = *pl->gpus[gi];
      RAFT_CUDA_TRY(cudaSetDevice(g.part->device));
      RAFT_CUDA_TRY(cudaStreamSynchronize(g.part->stream));
      gpu_phase ph;
      ph.device = g.part->device;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&ph.first_round_ms, te[4 * gi], te[4 * gi + 1]));
      if (pl->shard || !sp.use_graph)
        RAFT_CUDA_TRY(cudaEventElapsedTime(&ph.waves_ms, te[4 * gi + 2], te[4 * gi + 3]));
      ph.search_buffer_bytes = g.buffer_bytes;
      ph.search_mem_delta    = g.mem_delta;
      phases->push_back(ph);
      for (int j = 0; j < 4; j++)
        cudaEventDestroy(te[4 * gi + j]);
    }
    RAFT_CUDA_TRY(cudaSetDevice(cur));
  }
}

}  // namespace

auto timed_batch(raft::resources const& res,
                 search_params const& sp,
                 index const& idx,
                 const float* queries,
                 int64_t nq,
                 int64_t k) -> std::vector<gpu_phase>
{
  const int64_t n = std::min<int64_t>(std::max(sp.batch, 1), nq);
  std::vector<int64_t> ni(n * k);
  std::vector<float> nd(n * k);
  auto p = sp;
  p.batch =
    static_cast<int>(std::min<int64_t>(std::max(sp.batch, 1), nq));  // the plan of the full search
  std::vector<gpu_phase> out;
  search_impl(res, p, idx, queries, n, idx.dim(), ni.data(), nd.data(), k, true, &out);
  return out;
}

auto resident_search(raft::resources const& res,
                     search_params const& sp,
                     index const& idx,
                     int64_t nq,
                     int64_t k) -> void
{
  auto& ix = idx.impl();
  RAFT_EXPECTS(nq <= std::max(sp.batch, 1),
               "octex: resident search needs the whole query set in one batch");
  search_impl(res, sp, idx, nullptr, nq, ix.dim, nullptr, nullptr, k, false, nullptr, true);
  for (auto const& p : ix.parts) {
    RAFT_CUDA_TRY(cudaSetDevice(p->device));
    RAFT_CUDA_TRY(cudaStreamSynchronize(p->stream));
  }
}

auto memory_report(index const& idx) -> std::vector<gpu_memory>
{
  std::vector<gpu_memory> out;
  for (auto const& pp : idx.impl().parts) {
    auto const& p = *pp;
    gpu_memory g;
    g.device  = p.device;
    g.copies  = static_cast<int64_t>(p.lists.size + p.keys.size + p.offsets.size + p.base.size);
    g.sketch  = static_cast<int64_t>(p.sketch.size);
    g.table   = static_cast<int64_t>(p.table.size);
    g.vectors = static_cast<int64_t>(p.vectors.size);
    g.other   = static_cast<int64_t>(p.ucat.size + p.shift.size + p.inv_f.size + p.mean.size +
                                   p.basis.size + p.scale.size + p.xptr.size + p.tptr.size +
                                   p.cptr.size + p.workspace.size);
    out.push_back(g);
  }
  return out;
}

}  // namespace detail

void search(raft::resources const& res,
            search_params const& params,
            index const& idx,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances)
{
  RAFT_EXPECTS(neighbors.extent(0) == queries.extent(0) &&
                 distances.extent(0) == queries.extent(0) &&
                 distances.extent(1) == neighbors.extent(1),
               "octex: output shapes");
  detail::search_impl(res,
                      params,
                      idx,
                      queries.data_handle(),
                      queries.extent(0),
                      queries.extent(1),
                      neighbors.data_handle(),
                      distances.data_handle(),
                      neighbors.extent(1),
                      true);
}

void search(raft::resources const& res,
            search_params const& params,
            index const& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  RAFT_EXPECTS(neighbors.extent(0) == queries.extent(0) &&
                 distances.extent(0) == queries.extent(0) &&
                 distances.extent(1) == neighbors.extent(1),
               "octex: output shapes");
  detail::search_impl(res,
                      params,
                      idx,
                      queries.data_handle(),
                      queries.extent(0),
                      queries.extent(1),
                      neighbors.data_handle(),
                      distances.data_handle(),
                      neighbors.extent(1),
                      false);
}

}  // namespace cuvs::neighbors::octex
