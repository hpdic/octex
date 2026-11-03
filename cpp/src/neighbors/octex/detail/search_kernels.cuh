/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Search kernels of octex: raw pointers, one block per query for every per-query step.

#include "e8.cuh"

#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <cstdint>

namespace cuvs::neighbors::octex::detail {

constexpr int kMinLogHash  = 11;     // smallest hash set per query: 2^11 slots
constexpr int kMaxHash     = 32768;  // largest hash set (shared memory)
constexpr float kHashGrow  = 1.5f;   // hash set >= 1.5 x the entries read
constexpr int kMaxRerank   = 512;    // exact rerank depth per round
constexpr int kMaxKeepSort = 1024;   // kept + reranked keys sorted together
constexpr int kMaxSketchKO = 1024;   // kept entries of the sketch merge
constexpr int kPickHash    = 8192;   // hash set of the expanded start points
constexpr int kMaxExpanded = 6144;   // start points expanded by earlier waves (75% of the hash set)
constexpr unsigned kIdHash = 2654435761u;
constexpr uint64_t kEmpty  = ~0ULL;

/** log2 of the hash set of a query reading `total` entries: ceil(log2(1.5 total)) clamped to [11,
 * log_cap]. */
__host__ __device__ inline int hash_log(int total, int log_cap)
{
  if (total <= 0) return kMinLogHash;
  int e = static_cast<int>(ceilf(log2f(static_cast<float>(total) * kHashGrow)));
  return e < kMinLogHash ? kMinLogHash : (e > log_cap ? log_cap : e);
}

__device__ __forceinline__ const signed char* sketch_row(
  const signed char* code, const long long* cptr, int64_t crows, int id, int S)
{
  if (cptr) return reinterpret_cast<const signed char*>(cptr[id / crows]) + (id % crows) * S;
  return code + static_cast<int64_t>(id) * S;
}

__device__ __forceinline__ const __half* vector_row(const long long* xptr,
                                                    int64_t xrows,
                                                    int id,
                                                    int D)
{
  int64_t s = id / xrows;
  return reinterpret_cast<const __half*>(xptr[s]) + (id - s * xrows) * D;
}

/** qc = q - mu (fp32) and its fp16 copy. */
RAFT_KERNEL center_kernel(const float* __restrict__ q,
                          const float* __restrict__ mu,
                          int64_t total,
                          int D,
                          float* qc,
                          __half* qh)
{
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total) return;
  float v = q[t] - mu[t % D];
  qc[t]   = v;
  qh[t]   = __float2half(v);
}

/** Rows [from, m) of q become copies of row 0 (padding of a partial batch). */
RAFT_KERNEL pad_rows_kernel(float* __restrict__ q, int from, int m, int D)
{
  int64_t t     = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(m - from) * D;
  if (t >= total) return;
  q[static_cast<int64_t>(from) * D + t] = q[t % D];
}

/**
 * Per (query, copy): the finest level whose cell holds a list; seg = (first entry, entries read) in
 * the copy's lists, (0, 0) when no level has one.
 */
template <int B>
RAFT_KERNEL probe_kernel(const __half* __restrict__ Y,
                         int m,
                         int C,
                         int L,
                         const float* __restrict__ inv_f,
                         const int* __restrict__ off,
                         int64_t T1,
                         const long long* __restrict__ keys,
                         int64_t Mmax,
                         const int* __restrict__ base,
                         int l,
                         int lr,
                         int logT,
                         int2* __restrict__ seg)
{
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(m) * C) return;
  int c = static_cast<int>(t % C);
  float y[8 * B];
  auto yp = reinterpret_cast<const __half2*>(Y + t * 8 * B);
#pragma unroll
  for (int i = 0; i < 4 * B; i++) {
    float2 v     = __half22float2(yp[i]);
    y[2 * i]     = v.x;
    y[2 * i + 1] = v.y;
  }
  int2 sg = make_int2(0, 0);
  for (int j = 0; j < L; j++) {
    long long h         = e8b_hash64(y, B, inv_f[j]);
    int bk              = static_cast<int>(h & ((1LL << logT) - 1));
    const int* o        = off + (static_cast<int64_t>(c) * L + j) * T1 + bk;
    const long long* kk = keys + (static_cast<int64_t>(c) * L + j) * Mmax;
    bool found          = false;
    for (int p = o[0]; p < o[1]; p++)
      if (kk[p] == h) {
        sg    = make_int2((base[c * L + j] + p) * l, lr);
        found = true;
        break;
      }
    if (found) break;
  }
  seg[t] = sg;
}

/**
 * Distinct ids of all segments of a query with their int8 sketch distances. The entries of all
 * segments are spread evenly over the block (exclusive prefix of the segment lengths, binary search
 * per entry); a hash set in shared memory sized from the entries read drops duplicates.
 *
 * kRows = false: segment (start, len) indexes the lists of copy c (row stride `stride`).
 * kRows = true: segment (row, len) is a row of the neighbor table, split in ranges of rows_per rows
 * behind tptr.
 */
template <bool kRows>
RAFT_KERNEL gather_kernel(const int2* __restrict__ seg,
                          int C,
                          const int* __restrict__ lists,
                          int64_t stride,
                          const long long* __restrict__ tptr,
                          int64_t rows_per,
                          const signed char* __restrict__ code,
                          const long long* __restrict__ cptr,
                          int64_t crows,
                          const float* __restrict__ scale,
                          const float* __restrict__ qs,
                          int S,
                          int W,
                          int logW,
                          int* __restrict__ cand,
                          float* __restrict__ dist,
                          int* __restrict__ nout)
{
  const int q = blockIdx.x;
  extern __shared__ int sm[];
  float* qv  = reinterpret_cast<float*>(sm);
  float* sc  = qv + S;
  int* pre   = reinterpret_cast<int*>(sc + S);  // [C + 1] exclusive prefix
  int* st    = pre + C + 1;                     // [C] segment starts
  int* table = st + C;
  __shared__ int cnt;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    qv[i] = qs[static_cast<int64_t>(q) * S + i];
    sc[i] = scale[i];
  }
  if (threadIdx.x == 0) cnt = 0;
  if (warp == 0) {
    int carry = 0;
    for (int c0 = 0; c0 < C; c0 += 32) {
      int c   = c0 + lane;
      int2 sl = c < C ? seg[static_cast<int64_t>(q) * C + c] : make_int2(0, 0);
      int v   = sl.y;
      for (int o = 1; o < 32; o <<= 1) {
        int u = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) v += u;
      }
      if (c < C) {
        pre[c] = carry + v - sl.y;
        st[c]  = sl.x;
      }
      carry += __shfl_sync(0xffffffff, v, 31);
    }
    if (lane == 0) pre[C] = carry;
  }
  __syncthreads();
  const int total = pre[C];
  const int HT    = 1 << hash_log(total, logW);
  for (int i = threadIdx.x; i < HT; i += blockDim.x)
    table[i] = -1;
  __syncthreads();
  int* crow = cand + static_cast<int64_t>(q) * W;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    int lo = 0, hi = C - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (pre[mid] <= i)
        lo = mid;
      else
        hi = mid - 1;
    }
    int v;
    if constexpr (kRows) {
      int64_t row = st[lo];
      int64_t sh  = row / rows_per;
      v = reinterpret_cast<const int*>(tptr[sh])[(row - sh * rows_per) * stride + (i - pre[lo])];
    } else {
      v = lists[lo * stride + st[lo] + (i - pre[lo])];
    }
    if (v < 0) continue;  // padding of a table row
    unsigned s = (static_cast<unsigned>(v) * kIdHash) & (HT - 1);
    for (int p = 0; p < HT; p++) {
      int prev = atomicCAS(&table[s], -1, v);
      if (prev == -1) {
        int slot = atomicAdd(&cnt, 1);
        if (slot < W) crow[slot] = v;
        break;
      }
      if (prev == v) break;
      s = (s + 1) & (HT - 1);
    }
  }
  __syncthreads();
  const int n = min(cnt, W);
  int Lr      = 1;  // lanes per row: the next power of two >= S / 16, at most 32
  while (Lr < S / 16 && Lr < 32)
    Lr <<= 1;
  const int G = 32 / Lr, sub = lane % Lr;
  float* drow = dist + static_cast<int64_t>(q) * W;
  for (int j0 = warp * G; j0 < n; j0 += nw * G) {
    int j   = j0 + lane / Lr;
    float d = 0.f;
    if (j < n) {
      const signed char* x = sketch_row(code, cptr, crows, crow[j], S);
      for (int k = sub * 16; k < S; k += Lr * 16) {
        int4 w                = *reinterpret_cast<const int4*>(x + k);
        const signed char* bb = reinterpret_cast<const signed char*>(&w);
#pragma unroll
        for (int t = 0; t < 16; t++) {
          float u = static_cast<float>(bb[t]) * sc[k + t] - qv[k + t];
          d += u * u;
        }
      }
    }
    for (int o = Lr >> 1; o; o >>= 1)
      d += __shfl_xor_sync(0xffffffff, d, o);
    if (j < n && sub == 0) drow[j] = d;
  }
  if (threadIdx.x == 0) nout[q] = n;
}

/**
 * Radix select (8 bits per pass) of the R smallest float keys among n: their ids (and keys) go to
 * sid / skey, their count to nsel. All threads of the block call it.
 */
__device__ __forceinline__ void select_smallest(const unsigned* kd,
                                                const int* cid,
                                                int n,
                                                int R,
                                                int* sid,
                                                float* skey,
                                                unsigned* hist,
                                                unsigned& prefix,
                                                unsigned& kk,
                                                int& nsel,
                                                int& ntie)
{
  if (threadIdx.x == 0) {
    prefix = 0;
    kk     = min(R, n);
    nsel   = 0;
    ntie   = 0;
  }
  __syncthreads();
  if (n > R) {
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = threadIdx.x; i < 256; i += blockDim.x)
        hist[i] = 0;
      __syncthreads();
      unsigned pf = prefix;
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned k = kd[i];
        if ((k & mask) == pf) atomicAdd(&hist[(k >> shift) & 255], 1u);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        unsigned cum = 0, k = kk;
        int d = 0;
        for (; d < 256; d++) {
          if (cum + hist[d] >= k) break;
          cum += hist[d];
        }
        kk     = k - cum;
        prefix = pf | (static_cast<unsigned>(d) << shift);
      }
      mask |= 255u << shift;
      __syncthreads();
    }
    unsigned T = prefix, ties = kk;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      unsigned k = kd[i];
      bool take  = k < T;
      if (!take && k == T) take = atomicAdd(&ntie, 1) < static_cast<int>(ties);
      if (take) {
        int s  = atomicAdd(&nsel, 1);
        sid[s] = cid[i];
        if (skey) skey[s] = __uint_as_float(k);
      }
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      sid[i] = cid[i];
      if (skey) skey[i] = __uint_as_float(kd[i]);
    }
    if (threadIdx.x == 0) nsel = n;
  }
  __syncthreads();
}

/** Squared distances of the rows sid[0, ns) to the fp32 query sq (one warp per row). */
__device__ __forceinline__ void exact_distances(
  const int* sid, int ns, const long long* xptr, int64_t xrows, int D, const float* sq, float* sd)
{
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int j = warp; j < ns; j += nw) {
    auto x    = reinterpret_cast<const __half2*>(vector_row(xptr, xrows, sid[j], D));
    float acc = 0.f;
    for (int k = lane; k < D / 2; k += 32) {
      float2 v = __half22float2(x[k]);
      float a = v.x - sq[2 * k], b = v.y - sq[2 * k + 1];
      acc += a * a + b * b;
    }
    for (int o = 16; o; o >>= 1)
      acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) sd[j] = acc;
  }
  __syncthreads();
}

/** Ascending bitonic sort of P (power of two) 64-bit keys in shared memory. */
__device__ __forceinline__ void bitonic_sort(uint64_t* key, int P)
{
  for (int size = 2; size <= P; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < P; i += blockDim.x) {
        int j = i ^ stride;
        if (j > i) {
          bool up    = (i & size) == 0;
          uint64_t a = key[i], b = key[j];
          if ((a > b) == up) {
            key[i] = b;
            key[j] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ uint64_t dist_key(float d, int64_t id)
{
  return (static_cast<uint64_t>(__float_as_uint(d)) << 32) | static_cast<unsigned>(id);
}

/** Load the kept list (or nothing when fresh) and n new (distance, id) pairs as keys; P keys in
 * total. */
__device__ __forceinline__ void load_keys(uint64_t* key,
                                          int P,
                                          const long long* pb,
                                          const float* pd,
                                          int KO,
                                          bool fresh,
                                          const int* sid,
                                          const float* sdv,
                                          int ns)
{
  for (int i = threadIdx.x; i < P; i += blockDim.x) {
    uint64_t k = kEmpty;
    if (i < KO) {
      if (!fresh && pb[i] >= 0) k = dist_key(pd[i], pb[i]);
    } else if (i - KO < ns) {
      k = dist_key(sdv[i - KO], sid[i - KO]);
    }
    key[i] = k;
  }
  __syncthreads();
}

/** Write the first KO distinct sorted keys (the same id carries the same distance: equal keys
 * collapse). */
__device__ __forceinline__ void store_keys(
  const uint64_t* key, int P, int KO, long long* pb, float* pd)
{
  if (threadIdx.x != 0) return;
  int o         = 0;
  uint64_t prev = kEmpty - 1;
  for (int i = 0; i < P && o < KO; i++) {
    uint64_t k = key[i];
    if (k == kEmpty) break;
    if (k == prev) continue;
    prev  = k;
    pb[o] = static_cast<long long>(static_cast<unsigned>(k & 0xffffffffULL));
    pd[o] = __uint_as_float(static_cast<unsigned>(k >> 32));
    o++;
  }
  for (; o < KO; o++) {
    pb[o] = -1;
    pd[o] = INFINITY;
  }
}

/**
 * Exact rerank of the R sketch-nearest candidates of a query, merged into its kept list of the best
 * KO (ids, squared distances, best first). fresh: the kept list starts empty.
 */
RAFT_KERNEL rerank_keep_kernel(const int* __restrict__ cand,
                               const float* __restrict__ dist,
                               const int* __restrict__ nin,
                               int W,
                               const long long* __restrict__ xptr,
                               int64_t xrows,
                               int D,
                               const float* __restrict__ Q,
                               int R,
                               int KO,
                               long long* __restrict__ bi,
                               float* __restrict__ bd,
                               bool fresh)
{
  const int q = blockIdx.x;
  extern __shared__ float sq[];
  __shared__ unsigned hist[256];
  __shared__ unsigned prefix, kk;
  __shared__ int nsel, ntie;
  __shared__ int sid[kMaxRerank];
  __shared__ float sd[kMaxRerank];
  __shared__ uint64_t key[kMaxKeepSort];
  const int n = nin[q];
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    sq[i] = Q[static_cast<int64_t>(q) * D + i];
  select_smallest(reinterpret_cast<const unsigned*>(dist + static_cast<int64_t>(q) * W),
                  cand + static_cast<int64_t>(q) * W,
                  n,
                  R,
                  sid,
                  nullptr,
                  hist,
                  prefix,
                  kk,
                  nsel,
                  ntie);
  const int ns = nsel;
  exact_distances(sid, ns, xptr, xrows, D, sq, sd);
  long long* pb = bi + static_cast<int64_t>(q) * KO;
  float* pd     = bd + static_cast<int64_t>(q) * KO;
  int P         = 2;  // the next power of two >= KO + R
  while (P < KO + R)
    P <<= 1;
  load_keys(key, P, pb, pd, KO, fresh, sid, sd, ns);
  bitonic_sort(key, P);
  store_keys(key, P, KO, pb, pd);
}

/** The KO sketch-nearest candidates of a query merged into its kept list by sketch distance (no
 * exact rerank). */
RAFT_KERNEL sketch_keep_kernel(const int* __restrict__ cand,
                               const float* __restrict__ dist,
                               const int* __restrict__ nin,
                               int W,
                               int KO,
                               long long* __restrict__ si,
                               float* __restrict__ sd,
                               bool fresh)
{
  const int q = blockIdx.x;
  __shared__ unsigned hist[256];
  __shared__ unsigned prefix, kk;
  __shared__ int nsel, ntie;
  __shared__ int sid[kMaxSketchKO];
  __shared__ float sdv[kMaxSketchKO];
  __shared__ uint64_t key[2 * kMaxSketchKO];
  const int n = min(nin[q], W);
  select_smallest(reinterpret_cast<const unsigned*>(dist + static_cast<int64_t>(q) * W),
                  cand + static_cast<int64_t>(q) * W,
                  n,
                  KO,
                  sid,
                  sdv,
                  hist,
                  prefix,
                  kk,
                  nsel,
                  ntie);
  const int ns  = nsel;
  long long* pb = si + static_cast<int64_t>(q) * KO;
  float* pd     = sd + static_cast<int64_t>(q) * KO;
  int P         = 2;
  while (P < KO + ns)
    P <<= 1;
  load_keys(key, P, pb, pd, KO, fresh, sid, sdv, ns);
  bitonic_sort(key, P);
  store_keys(key, P, KO, pb, pd);
}

/**
 * Merge of the kept lists of nsrc sources (any GPUs, read through their pointers): row row0 + q of
 * every
 * [*, KO] source; out = the best KO distinct ids by distance. Two sorts in shared memory: by (id,
 * distance) to keep the nearest copy of every id, then by (distance, id). Dynamic shared memory: P
 * keys, P = next power of two of nsrc * KO.
 */
RAFT_KERNEL merge_kept_kernel(const long long* __restrict__ src_i,
                              const long long* __restrict__ src_d,
                              int nsrc,
                              int64_t row0,
                              int KO,
                              int P,
                              long long* __restrict__ si,
                              float* __restrict__ sd)
{
  const int q = blockIdx.x;
  extern __shared__ uint64_t mk[];
  const int total = nsrc * KO;
  for (int i = threadIdx.x; i < P; i += blockDim.x) {
    uint64_t k = kEmpty;
    if (i < total) {
      int h       = i / KO;
      int64_t sr  = (row0 + q) * KO + i % KO;
      long long v = reinterpret_cast<const long long*>(src_i[h])[sr];
      float d     = reinterpret_cast<const float*>(src_d[h])[sr];
      if (v >= 0) k = (static_cast<uint64_t>(static_cast<unsigned>(v)) << 32) | __float_as_uint(d);
    }
    mk[i] = k;
  }
  __syncthreads();
  bitonic_sort(mk, P);
  // the first key of every id run holds its smallest distance; re-key as (distance, id)
  uint64_t kept[8];
  int nk = 0;
  for (int i = threadIdx.x; i < P; i += blockDim.x) {
    uint64_t k = mk[i], out = kEmpty;
    if (k != kEmpty && (i == 0 || (mk[i - 1] >> 32) != (k >> 32)))
      out = (static_cast<uint64_t>(static_cast<unsigned>(k & 0xffffffffULL)) << 32) | (k >> 32);
    kept[nk++] = out;  // P / blockDim.x <= 8
  }
  __syncthreads();
  nk = 0;
  for (int i = threadIdx.x; i < P; i += blockDim.x)
    mk[i] = kept[nk++];
  __syncthreads();
  bitonic_sort(mk, P);
  long long* pb = si + static_cast<int64_t>(q) * KO;
  float* pd     = sd + static_cast<int64_t>(q) * KO;
  for (int i = threadIdx.x; i < KO; i += blockDim.x) {
    uint64_t k = mk[i];
    pb[i] = k == kEmpty ? -1 : static_cast<long long>(static_cast<unsigned>(k & 0xffffffffULL));
    pd[i] = k == kEmpty ? INFINITY : __uint_as_float(static_cast<unsigned>(k >> 32));
  }
}

/**
 * Start points of a wave: the first S entries of the kept list (best first) that no earlier wave of
 * the query expanded (done[:, :off]), -1 when fewer remain; written to done[:, off:off + S] and to
 * the gather segments seg2 [S] = (row, rowlen).
 */
RAFT_KERNEL pick_starts_kernel(const long long* __restrict__ si,
                               int KO,
                               long long* __restrict__ done,
                               int E,
                               int off,
                               int S,
                               int rowlen,
                               int2* __restrict__ seg2)
{
  const int q = blockIdx.x;
  __shared__ int ht[kPickHash];
  __shared__ unsigned char fresh[kMaxSketchKO];
  const long long* ps = si + static_cast<int64_t>(q) * KO;
  long long* pd       = done + static_cast<int64_t>(q) * E;
  for (int i = threadIdx.x; i < kPickHash; i += blockDim.x)
    ht[i] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < off; i += blockDim.x) {
    long long v = pd[i];
    if (v < 0) continue;
    unsigned h = (static_cast<unsigned>(v) * kIdHash) & (kPickHash - 1);
    while (true) {
      int prev = atomicCAS(&ht[h], -1, static_cast<int>(v));
      if (prev == -1 || prev == static_cast<int>(v)) break;
      h = (h + 1) & (kPickHash - 1);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < KO; i += blockDim.x) {
    long long v = ps[i];
    bool f      = v >= 0;
    if (f) {
      unsigned h = (static_cast<unsigned>(v) * kIdHash) & (kPickHash - 1);
      while (true) {
        int t = ht[h];
        if (t == -1) break;
        if (t == static_cast<int>(v)) {
          f = false;
          break;
        }
        h = (h + 1) & (kPickHash - 1);
      }
    }
    fresh[i] = f;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int o = 0;
    for (int i = 0; i < KO && o < S; i++) {
      if (!fresh[i]) continue;
      long long v                           = ps[i];
      pd[off + o]                           = v;
      seg2[static_cast<int64_t>(q) * S + o] = make_int2(static_cast<int>(v), rowlen);
      o++;
    }
    for (; o < S; o++) {
      pd[off + o]                           = -1;
      seg2[static_cast<int64_t>(q) * S + o] = make_int2(0, 0);
    }
  }
}

/** Exact rerank of the first R kept entries (sketch-rank mode) into the best k. */
RAFT_KERNEL final_rerank_kernel(const long long* __restrict__ si,
                                int KO,
                                int R,
                                const long long* __restrict__ xptr,
                                int64_t xrows,
                                int D,
                                const float* __restrict__ Q,
                                int k,
                                long long* __restrict__ bi,
                                float* __restrict__ bd)
{
  const int q = blockIdx.x;
  extern __shared__ float sq[];
  __shared__ int sid[kMaxRerank];
  __shared__ float sd[kMaxRerank];
  __shared__ uint64_t key[kMaxRerank];
  __shared__ int nsel;
  const long long* ps = si + static_cast<int64_t>(q) * KO;
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    sq[i] = Q[static_cast<int64_t>(q) * D + i];
  if (threadIdx.x == 0) {
    int n = 0;
    for (int i = 0; i < R; i++)
      if (ps[i] >= 0) sid[n++] = static_cast<int>(ps[i]);
    nsel = n;
  }
  __syncthreads();
  const int ns = nsel;
  exact_distances(sid, ns, xptr, xrows, D, sq, sd);
  int P = 2;
  while (P < R)
    P <<= 1;
  load_keys(key, P, nullptr, nullptr, 0, true, sid, sd, ns);
  bitonic_sort(key, P);
  store_keys(key, P, k, bi + static_cast<int64_t>(q) * k, bd + static_cast<int64_t>(q) * k);
}

/** bi, bd [m, k] = the first k columns of si, sd [m, KO]. */
RAFT_KERNEL take_k_kernel(const long long* __restrict__ si,
                          const float* __restrict__ sd,
                          int KO,
                          int k,
                          int m,
                          long long* bi,
                          float* bd)
{
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(m) * k) return;
  int64_t q = t / k, j = t % k;
  bi[t] = si[q * KO + j];
  bd[t] = sd[q * KO + j];
}

// ---- Ablation baselines (search_params::balanced_gather, fused_merge = false): the
// straightforward kernels that the optimized ones replace, with the same results.

/**
 * gather_kernel without the load balance: warp w walks the segments w, w + warps, ... whole, one
 * entry per lane, so the longest segments of a query set the time of its block. The same hash set,
 * the same inserts and the same sketch distances. Dynamic shared memory: 2 S floats and W ints.
 */
template <bool kRows>
RAFT_KERNEL gather_warp_kernel(const int2* __restrict__ seg,
                               int C,
                               const int* __restrict__ lists,
                               int64_t stride,
                               const long long* __restrict__ tptr,
                               int64_t rows_per,
                               const signed char* __restrict__ code,
                               const long long* __restrict__ cptr,
                               int64_t crows,
                               const float* __restrict__ scale,
                               const float* __restrict__ qs,
                               int S,
                               int W,
                               int logW,
                               int* __restrict__ cand,
                               float* __restrict__ dist,
                               int* __restrict__ nout)
{
  const int q = blockIdx.x;
  extern __shared__ int sm[];
  float* qv  = reinterpret_cast<float*>(sm);
  float* sc  = qv + S;
  int* table = reinterpret_cast<int*>(sc + S);
  __shared__ int cnt, total;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    qv[i] = qs[static_cast<int64_t>(q) * S + i];
    sc[i] = scale[i];
  }
  if (threadIdx.x == 0) {
    cnt   = 0;
    total = 0;
  }
  __syncthreads();
  int part = 0;
  for (int c = threadIdx.x; c < C; c += blockDim.x)
    part += seg[static_cast<int64_t>(q) * C + c].y;
  if (part > 0) atomicAdd(&total, part);
  __syncthreads();
  const int HT = 1 << hash_log(total, logW);
  for (int i = threadIdx.x; i < HT; i += blockDim.x)
    table[i] = -1;
  __syncthreads();
  int* crow = cand + static_cast<int64_t>(q) * W;
  for (int c = warp; c < C; c += nw) {
    const int2 sl = seg[static_cast<int64_t>(q) * C + c];
    for (int e = lane; e < sl.y; e += 32) {
      int v;
      if constexpr (kRows) {
        int64_t row = sl.x;
        int64_t sh  = row / rows_per;
        v           = reinterpret_cast<const int*>(tptr[sh])[(row - sh * rows_per) * stride + e];
      } else {
        v = lists[c * stride + sl.x + e];
      }
      if (v < 0) continue;  // padding of a table row
      unsigned s = (static_cast<unsigned>(v) * kIdHash) & (HT - 1);
      for (int p = 0; p < HT; p++) {
        int prev = atomicCAS(&table[s], -1, v);
        if (prev == -1) {
          int slot = atomicAdd(&cnt, 1);
          if (slot < W) crow[slot] = v;
          break;
        }
        if (prev == v) break;
        s = (s + 1) & (HT - 1);
      }
    }
  }
  __syncthreads();
  const int n = min(cnt, W);
  int Lr      = 1;  // lanes per row: the next power of two >= S / 16, at most 32
  while (Lr < S / 16 && Lr < 32)
    Lr <<= 1;
  const int G = 32 / Lr, sub = lane % Lr;
  float* drow = dist + static_cast<int64_t>(q) * W;
  for (int j0 = warp * G; j0 < n; j0 += nw * G) {
    int j   = j0 + lane / Lr;
    float d = 0.f;
    if (j < n) {
      const signed char* x = sketch_row(code, cptr, crows, crow[j], S);
      for (int k = sub * 16; k < S; k += Lr * 16) {
        int4 w                = *reinterpret_cast<const int4*>(x + k);
        const signed char* bb = reinterpret_cast<const signed char*>(&w);
#pragma unroll
        for (int t = 0; t < 16; t++) {
          float u = static_cast<float>(bb[t]) * sc[k + t] - qv[k + t];
          d += u * u;
        }
      }
    }
    for (int o = Lr >> 1; o; o >>= 1)
      d += __shfl_xor_sync(0xffffffff, d, o);
    if (j < n && sub == 0) drow[j] = d;
  }
  if (threadIdx.x == 0) nout[q] = n;
}

/**
 * Unfused merge, step 1: the (sketch distance, id) keys of the n candidates of every query in
 * global memory, and the segment [q W, q W + n) of the query for a CUB segmented sort.
 */
RAFT_KERNEL cand_keys_kernel(const int* __restrict__ cand,
                             const float* __restrict__ dist,
                             const int* __restrict__ nin,
                             int W,
                             uint64_t* __restrict__ key,
                             int* __restrict__ beg,
                             int* __restrict__ end)
{
  const int q     = blockIdx.x;
  const int n     = min(nin[q], W);
  const int64_t o = static_cast<int64_t>(q) * W;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    key[o + j] = dist_key(dist[o + j], cand[o + j]);
  if (threadIdx.x == 0) {
    beg[q] = static_cast<int>(o);
    end[q] = static_cast<int>(o + n);
  }
}

/**
 * Unfused merge, step 2 (after the sort): the pool of a query in global memory, [pool] keys: its
 * kept list (none when fresh), then its first R sorted candidates with exact distances (xptr), or
 * with their sketch distances (xptr null, sketch-rank mode). Unused keys are empty.
 */
RAFT_KERNEL pool_keys_kernel(const uint64_t* __restrict__ sorted,
                             const int* __restrict__ beg,
                             const int* __restrict__ end,
                             const long long* __restrict__ xptr,
                             int64_t xrows,
                             int D,
                             const float* __restrict__ Q,
                             int R,
                             const long long* __restrict__ bi,
                             const float* __restrict__ bd,
                             int KO,
                             bool fresh,
                             int pool,
                             uint64_t* __restrict__ out)
{
  const int q = blockIdx.x;
  extern __shared__ float sq[];
  __shared__ int sid[kMaxRerank];
  __shared__ float sd[kMaxRerank];
  const uint64_t* sk  = sorted + beg[q];
  const int ns        = min(R, end[q] - beg[q]);
  const long long* pb = bi + static_cast<int64_t>(q) * KO;
  const float* pd     = bd + static_cast<int64_t>(q) * KO;
  uint64_t* po        = out + static_cast<int64_t>(q) * pool;
  for (int i = threadIdx.x; i < KO; i += blockDim.x)
    po[i] = !fresh && pb[i] >= 0 ? dist_key(pd[i], pb[i]) : kEmpty;
  for (int i = KO + ns + threadIdx.x; i < pool; i += blockDim.x)
    po[i] = kEmpty;
  if (xptr == nullptr) {
    for (int i = threadIdx.x; i < ns; i += blockDim.x)
      po[KO + i] = sk[i];
    return;
  }
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    sq[i] = Q[static_cast<int64_t>(q) * D + i];
  for (int i = threadIdx.x; i < ns; i += blockDim.x)
    sid[i] = static_cast<int>(sk[i] & 0xffffffffULL);
  __syncthreads();
  exact_distances(sid, ns, xptr, xrows, D, sq, sd);
  for (int i = threadIdx.x; i < ns; i += blockDim.x)
    po[KO + i] = dist_key(sd[i], sid[i]);
}

/** Unfused merge, step 3 (after the second sort): the first KO distinct keys of every sorted pool,
 * one thread per query. */
RAFT_KERNEL keep_distinct_kernel(
  const uint64_t* __restrict__ sorted, int pool, int KO, int m, long long* bi, float* bd)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= m) return;
  const uint64_t* key = sorted + static_cast<int64_t>(q) * pool;
  long long* pb       = bi + static_cast<int64_t>(q) * KO;
  float* pd           = bd + static_cast<int64_t>(q) * KO;
  int o               = 0;
  uint64_t prev       = kEmpty - 1;
  for (int i = 0; i < pool && o < KO; i++) {
    uint64_t k = key[i];
    if (k == kEmpty) break;
    if (k == prev) continue;
    prev  = k;
    pb[o] = static_cast<long long>(static_cast<unsigned>(k & 0xffffffffULL));
    pd[o] = __uint_as_float(static_cast<unsigned>(k >> 32));
    o++;
  }
  for (; o < KO; o++) {
    pb[o] = -1;
    pd[o] = INFINITY;
  }
}

/**
 * Unfused selection, step 1 (after a CUB segmented sort of done[:, :off]): a kept entry is fresh
 * when a binary search does not find it among the ids that earlier waves expanded. One thread per
 * entry.
 */
RAFT_KERNEL mark_fresh_kernel(const long long* __restrict__ si,
                              int KO,
                              const long long* __restrict__ sorted_done,
                              int E,
                              int off,
                              int m,
                              unsigned char* __restrict__ fresh)
{
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(m) * KO) return;
  const long long v = si[t];
  bool f            = v >= 0;
  if (f && off > 0) {
    const long long* d = sorted_done + (t / KO) * E;
    int lo = 0, hi = off;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (d[mid] < v)
        lo = mid + 1;
      else
        hi = mid;
    }
    f = !(lo < off && d[lo] == v);
  }
  fresh[t] = f;
}

/** Unfused selection, step 2: the first S fresh kept entries of every query as its start points, as
 * pick_starts_kernel writes them. One thread per query. */
RAFT_KERNEL take_starts_kernel(const long long* __restrict__ si,
                               const unsigned char* __restrict__ fresh,
                               int KO,
                               long long* __restrict__ done,
                               int E,
                               int off,
                               int S,
                               int rowlen,
                               int m,
                               int2* __restrict__ seg2)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= m) return;
  const long long* ps     = si + static_cast<int64_t>(q) * KO;
  const unsigned char* fr = fresh + static_cast<int64_t>(q) * KO;
  long long* pd           = done + static_cast<int64_t>(q) * E;
  int2* sg                = seg2 + static_cast<int64_t>(q) * S;
  int o                   = 0;
  for (int i = 0; i < KO && o < S; i++) {
    if (!fr[i]) continue;
    pd[off + o] = ps[i];
    sg[o]       = make_int2(static_cast<int>(ps[i]), rowlen);
    o++;
  }
  for (; o < S; o++) {
    pd[off + o] = -1;
    sg[o]       = make_int2(0, 0);
  }
}

}  // namespace cuvs::neighbors::octex::detail
