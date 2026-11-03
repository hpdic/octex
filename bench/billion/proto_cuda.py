"""CUDA kernels of the research prototype: E8^b decoding of the projected queries, and the per-query gather of the hit union
with sketch distances.

decode(Y [m, C * 8b] fp32, b, logT) -> bucket [m, C] int32, bit-identical to the prototype e8_bucket (one thread per
(query, copy)).

gather(bucket [m, C], off [C, T + 1], ids [C, N], code [N, S] int8, scale [S], qs [m, S], cap, HT, maxc)
-> (cand [m, maxc] int32, dist [m, maxc] fp32, n [m] int32, reads [1] int64): one thread block per query. Every warp
walks copies; its lanes read the ids of the query's bucket in the copy (at most cap, coalesced) and insert them into
an open-addressing hash set of HT slots (shared memory up to 32768 slots, global memory beyond). Then every distinct id gets its sketch distance
sum_k (code[id, k] * scale[k] - qs[k])^2 and is appended to the output row (unordered; the rest of the row is +inf).
No hit counts: the union is the candidate set.

decode2(Y, b, logT, off, cap) -> (seg [m, C, 2] = (start, length <= cap) of every bucket, tot [m] = ids to read), and
gather2(seg, ids, code, scale, qs, HT, maxc, threads): the same gather with one copy per thread (independent random
reads in flight) and the bucket segments precomputed by decode2.

decode3 = decode2 on fp16 projections (b <= 4); gather3 = gather2 with a per-query hash-set size (one launch for
queries of different weights; each block initializes and scans only its own slots). gather4 = gather3 with the
ids of 32 buckets spread evenly over the lanes of a warp, new ids appended to the row at insertion (no table scan),
and S/16 lanes per candidate for the sketch distance.

Fixed-shape versions for CUDA graphs (every launch covers the whole query batch; blocks of other queries return):
decode4 (templated on b, skips inactive queries), gather5 and select_rerank2 (a query works only if it has a row
in this class's preallocated buffers: rowof[q] in [0, rows)). gather_global: gather5 for heavy queries with
`split` blocks per query (init, insert, sketch kernels; global hash set and counters). gather6 / gather_global2:
the same with hit counts packed into the table entry ((id << 6) | hits, ids < 2^25) and the sketch read only for
candidates hit >= tau times (others get +inf); tau = 1 reproduces gather5 / gather_global.

probe(Y, 4, inv_f [L], off, keys, base, l, lr, logT, seg, tot, lvl): design B (stored ripples): per (query, copy)
the list of the finest probe level whose cell holds one (64-bit key verified in the level's bucket), as the segment
(row * l, lr) into the copy's flat list array (lists of length l, the first lr read); feeds gather5 like decode4.

gather7: gather5 over rows of ONE shared table addressed in 64 bits (seg.x = row, start = row * rowlen): the
second, Huygens wave, where every seed re-emits its row of the neighbor table. select_rerank3: select_rerank2 that
keeps the best KO (bitonic sort of (distance, id) keys), for the seeds of the second wave.
gather8: gather7 over a table split in row ranges whose device pointers may live on peer GPUs (CUDA IPC, NVLink).

select_rerank(cand, dist, n, X fp16 [N, D], Q [m, D], R, bi [m, 10] int64, bd [m, 10]): per query, radix-select the
R smallest sketch distances of its row, compute their exact squared distances, and merge them into its current
best 10 (in place).
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

__constant__ long long PR[8] = {1000003LL, 998244353LL, 1000000007LL, 2147483647LL, 1099511627791LL,
                                4398042316799LL, 70368744177643LL, 1125899906842597LL};
#define MIXC 0x9E3779B97F4A7C15ULL

__device__ __forceinline__ void d8(const float* x, float* f) {
    // nearest point of D8 (integer vectors with an even sum), as in e8.d8
    float s = 0.f; int im = 0; float em = -1.f, ei = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        f[i] = rintf(x[i]); s += f[i];
        float e = x[i] - f[i];
        if (fabsf(e) > em) { em = fabsf(e); im = i; ei = e; }
    }
    if (fmodf(s, 2.f) != 0.f) f[im] += (ei >= 0.f ? 1.f : -1.f);
}

__global__ void decode_kernel(const float* __restrict__ Y, int m, int C, int b, int logT, int* __restrict__ out) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)m * C) return;
    const float* y = Y + t * 8 * b;
    unsigned long long h = 0;
    for (int j = 0; j < b; j++) {
        float x[8], a[8], c[8], xs[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) { x[i] = y[8 * j + i]; xs[i] = x[i] - 0.5f; }
        d8(x, a);
        d8(xs, c);
        float da = 0.f, dc = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { c[i] += 0.5f; da += (x[i] - a[i]) * (x[i] - a[i]); dc += (x[i] - c[i]) * (x[i] - c[i]); }
        unsigned long long hb = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            long long p2 = (long long)(2.f * (da <= dc ? a[i] : c[i]));
            hb += (unsigned long long)p2 * (unsigned long long)PR[i];
        }
        h = (j == 0) ? hb : h * MIXC + hb;
    }
    long long hs = (long long)h;
    hs = (long long)((unsigned long long)(hs ^ (hs >> 31)) * MIXC);
    hs = hs ^ (hs >> 29);
    out[t] = (int)(hs & ((1LL << logT) - 1));
}

__device__ __forceinline__ long long e8b_hash64(const float* y, int b, float inv_f) {
    // 64-bit hash of the nearest point of E8^b to y * inv_f, identical to nest.phash(e8b_point(y * inv_f))
    unsigned long long h = 0;
    for (int j = 0; j < b; j++) {
        float x[8], a[8], c[8], xs[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) { x[i] = y[8 * j + i] * inv_f; xs[i] = x[i] - 0.5f; }
        d8(x, a);
        d8(xs, c);
        float da = 0.f, dc = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { c[i] += 0.5f; da += (x[i] - a[i]) * (x[i] - a[i]); dc += (x[i] - c[i]) * (x[i] - c[i]); }
        unsigned long long hb = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            long long p2 = (long long)(2.f * (da <= dc ? a[i] : c[i]));
            hb += (unsigned long long)p2 * (unsigned long long)PR[i];
        }
        h = (j == 0) ? hb : h * MIXC + hb;
    }
    long long hs = (long long)h;
    hs = (long long)((unsigned long long)(hs ^ (hs >> 31)) * MIXC);
    return hs ^ (hs >> 29);
}

template <int B>
__global__ void probe_kernel(const __half* __restrict__ Y, int m, int C, int L, const float* __restrict__ inv_f,
                             const int* __restrict__ off, long T1, const long long* __restrict__ keys, long Mmax,
                             const int* __restrict__ base, int l, int lr, int logT, int2* __restrict__ seg,
                             int* __restrict__ tot, int* __restrict__ lvl) {
    // per (query, copy): the finest probe level whose cell holds a list; seg = (row * l, l) into the copy's lists
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    bool valid = t < (long)m * C;
    int q = valid ? (int)(t / C) : -1 - lane;
    int len = 0;
    if (valid) {
        int c = t % C;
        float y[8 * B];
        const __half2* yp = reinterpret_cast<const __half2*>(Y + t * 8 * B);
        #pragma unroll
        for (int i = 0; i < 4 * B; i++) { float2 v = __half22float2(yp[i]); y[2 * i] = v.x; y[2 * i + 1] = v.y; }
        int2 sg = make_int2(0, 0);
        int found = -1;
        for (int j = 0; j < L && found < 0; j++) {
            long long h = e8b_hash64(y, B, inv_f[j]);
            int bk = (int)(h & ((1LL << logT) - 1));
            const int* o = off + ((long)c * L + j) * T1 + bk;
            const long long* kk = keys + ((long)c * L + j) * Mmax;
            for (int p = o[0]; p < o[1]; p++)
                if (kk[p] == h) { sg = make_int2((base[c * L + j] + p) * l, lr); found = j; break; }
        }
        seg[t] = sg;
        len = sg.y;
        if (lvl) lvl[t] = found;
    }
    int sum = len;
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_down_sync(0xffffffff, sum, o);
        int qq = __shfl_down_sync(0xffffffff, q, o);
        if (lane + o < 32 && qq == q) sum += v;
    }
    int qp = __shfl_up_sync(0xffffffff, q, 1);
    if (valid && (lane == 0 || qp != q) && sum) atomicAdd(&tot[q], sum);
}

__device__ __forceinline__ int e8b_bucket(const float* y, int b, int logT) {
    unsigned long long h = 0;
    for (int j = 0; j < b; j++) {
        float x[8], a[8], c[8], xs[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) { x[i] = y[8 * j + i]; xs[i] = x[i] - 0.5f; }
        d8(x, a);
        d8(xs, c);
        float da = 0.f, dc = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { c[i] += 0.5f; da += (x[i] - a[i]) * (x[i] - a[i]); dc += (x[i] - c[i]) * (x[i] - c[i]); }
        unsigned long long hb = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            long long p2 = (long long)(2.f * (da <= dc ? a[i] : c[i]));
            hb += (unsigned long long)p2 * (unsigned long long)PR[i];
        }
        h = (j == 0) ? hb : h * MIXC + hb;
    }
    long long hs = (long long)h;
    hs = (long long)((unsigned long long)(hs ^ (hs >> 31)) * MIXC);
    hs = hs ^ (hs >> 29);
    return (int)(hs & ((1LL << logT) - 1));
}

__global__ void decode2_kernel(const float* __restrict__ Y, int m, int C, int b, int logT, const int* __restrict__ off,
                               long T1, int cap, int2* __restrict__ seg, int* __restrict__ tot) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    bool valid = t < (long)m * C;
    int q = valid ? (int)(t / C) : -1 - lane;          // invalid lanes match nobody
    int len = 0;
    if (valid) {
        int c = t % C;
        int h = e8b_bucket(Y + t * 8 * b, b, logT);
        const int* o = off + (long)c * T1 + h;
        int st = o[0];
        len = min(o[1] - st, cap);
        seg[t] = make_int2(st, len);
    }
    // the queries of a warp are contiguous runs: segmented sum, the head of every run adds it once
    int sum = len;
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_down_sync(0xffffffff, sum, o);
        int qq = __shfl_down_sync(0xffffffff, q, o);
        if (lane + o < 32 && qq == q) sum += v;
    }
    int qp = __shfl_up_sync(0xffffffff, q, 1);
    if (valid && (lane == 0 || qp != q) && sum) atomicAdd(&tot[q], sum);
}

__global__ void decode3_kernel(const __half* __restrict__ Y, int m, int C, int b, int logT,
                               const int* __restrict__ off, long T1, int cap, int2* __restrict__ seg,
                               int* __restrict__ tot) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    bool valid = t < (long)m * C;
    int q = valid ? (int)(t / C) : -1 - lane;
    int len = 0;
    if (valid) {
        int c = t % C;
        float y[32];
        const __half2* yp = reinterpret_cast<const __half2*>(Y + t * 8 * b);
        for (int i = 0; i < 4 * b; i++) { float2 v = __half22float2(yp[i]); y[2 * i] = v.x; y[2 * i + 1] = v.y; }
        int h = e8b_bucket(y, b, logT);
        const int* o = off + (long)c * T1 + h;
        int st = o[0];
        len = min(o[1] - st, cap);
        seg[t] = make_int2(st, len);
    }
    int sum = len;
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_down_sync(0xffffffff, sum, o);
        int qq = __shfl_down_sync(0xffffffff, q, o);
        if (lane + o < 32 && qq == q) sum += v;
    }
    int qp = __shfl_up_sync(0xffffffff, q, 1);
    if (valid && (lane == 0 || qp != q) && sum) atomicAdd(&tot[q], sum);
}

__global__ void gather3_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts, int C, long N, int S,
                               int maxc, int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               int* __restrict__ gtable, long gstride) {
    // gather2 with a per-query hash-set size hts[q] (power of two): a block initializes and scans only its own
    // slots, so light queries stay cheap in a launch sized for the heaviest
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    const int q = blockIdx.x, HT = hts[q];
    int* table = gtable ? gtable + (long)q * gstride : sm + 2 * S;
    __shared__ int cnt;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int2 sl = seg[(long)q * C + c];
        const int* idc = ids + (long)c * N + sl.x;
        for (int p = 0; p < sl.y; p++) {
            int v = idc[p];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1 || prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < HT; i += blockDim.x) {
        int v = table[i];
        if (v < 0) continue;
        const signed char* x = code + (long)v * S;
        float d = 0.f;
        for (int k = 0; k < S; k += 16) {
            int4 w = *reinterpret_cast<const int4*>(x + k);
            const signed char* bb = reinterpret_cast<const signed char*>(&w);
            #pragma unroll
            for (int j = 0; j < 16; j++) { float t = (float)bb[j] * sc[k + j] - qv[k + j]; d += t * t; }
        }
        int slot = atomicAdd(&cnt, 1);
        if (slot < maxc) { cand[(long)q * maxc + slot] = v; dist[(long)q * maxc + slot] = d; }
    }
    __syncthreads();
    if (threadIdx.x == 0) nout[q] = min(cnt, maxc);
}

__global__ void gather4_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts, int C, long N, int S,
                               int maxc, int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               int* __restrict__ gtable, long gstride) {
    // Insert: every warp takes 32 copies at a time, scans their bucket lengths, and spreads the concatenated ids
    // evenly over its lanes. A new id (first CAS into the query's hash set) is appended to the candidate row.
    // Sketch: S/16 lanes per candidate, 16 int8 coordinates each (one 16-byte load), reduced by shuffles.
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* wbuf = (int*)(sc + S);                        // per warp: 32 exclusive offsets, 32 starts, 32 copies
    const int nw = blockDim.x >> 5;
    int* table = gtable ? gtable + (long)blockIdx.x * gstride : wbuf + 96 * nw;
    __shared__ int cnt;
    const int q = blockIdx.x, HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    int* crow = cand + (long)q * maxc;
    for (int c0 = warp * 32; c0 < C; c0 += nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;                       // the last k with wex[k] <= i
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            int v = ids[(long)wcp[lo] * N + wst[lo] + (i - wex[lo])];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1) { int slot = atomicAdd(&cnt, 1); if (slot < maxc) crow[slot] = v; break; }
                if (prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
        __syncwarp();
    }
    __syncthreads();
    const int n = min(cnt, maxc);
    const int L = min(32, S / 16), G = 32 / L, sub = lane % L;
    float* drow = dist + (long)q * maxc;
    for (int j0 = warp * G; j0 < n; j0 += nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = code + (long)crow[j] * S;
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (threadIdx.x == 0) nout[q] = n;
}

template <int B>
__global__ void decode4_kernel(const __half* __restrict__ Y, const bool* __restrict__ active, int m, int C, int logT,
                               const int* __restrict__ off, long T1, int cap, int2* __restrict__ seg,
                               int* __restrict__ tot) {
    // decode3 for a fixed number of blocks B, skipping inactive queries (their tot stays 0)
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    bool valid = t < (long)m * C;
    int q = valid ? (int)(t / C) : -1 - lane;
    if (valid && !active[q]) valid = false;
    int len = 0;
    if (valid) {
        int c = t % C;
        float y[8 * B];
        const __half2* yp = reinterpret_cast<const __half2*>(Y + t * 8 * B);
        #pragma unroll
        for (int i = 0; i < 4 * B; i++) { float2 v = __half22float2(yp[i]); y[2 * i] = v.x; y[2 * i + 1] = v.y; }
        int h = e8b_bucket(y, B, logT);
        const int* o = off + (long)c * T1 + h;
        int st = o[0];
        len = min(o[1] - st, cap);
        seg[t] = make_int2(st, len);
    }
    int sum = len;
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_down_sync(0xffffffff, sum, o);
        int qq = __shfl_down_sync(0xffffffff, q, o);
        if (lane + o < 32 && qq == q) sum += v;
    }
    int qp = __shfl_up_sync(0xffffffff, q, 1);
    if (valid && (lane == 0 || qp != q) && sum) atomicAdd(&tot[q], sum);
}

__global__ void gather5_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, int S, int W,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               int* __restrict__ gtable) {
    // gather4 over all queries of the batch: block q works only if rowof[q] in [0, rows), writing row rowof[q]
    // (width W) of this class's buffers; gtable (rows x W) holds the hash sets when they do not fit shared memory
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* wbuf = (int*)(sc + S);
    const int nw = blockDim.x >> 5;
    int* table = gtable ? gtable + (long)r * W : wbuf + 96 * nw;
    __shared__ int cnt;
    const int HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    int* crow = cand + (long)r * W;
    for (int c0 = warp * 32; c0 < C; c0 += nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            int v = ids[(long)wcp[lo] * N + wst[lo] + (i - wex[lo])];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1) { int slot = atomicAdd(&cnt, 1); if (slot < W) crow[slot] = v; break; }
                if (prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
        __syncwarp();
    }
    __syncthreads();
    const int n = min(cnt, W);
    const int L = min(32, S / 16), G = 32 / L, sub = lane % L;
    float* drow = dist + (long)r * W;
    for (int j0 = warp * G; j0 < n; j0 += nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = code + (long)crow[j] * S;
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (threadIdx.x == 0) nout[r] = n;
}

__device__ __forceinline__ const signed char* srow(const signed char* code, const long long* cptr, long crows,
                                                  int id, int S) {
    // sketch row id: code itself, or row ranges of crows rows behind pointers (peer GPUs or pinned host memory)
    if (cptr) return reinterpret_cast<const signed char*>(cptr[id / crows]) + (long)(id % crows) * S;
    return code + (long)id * S;
}

__global__ void gather9_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, int S, int W,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               const long long* __restrict__ cptr, long crows) {
    // gather5 with the list entries of ALL copies spread evenly over all threads of the block (gather5 gives each
    // warp 32 whole copies: with ~60 copies per GPU most warps idle at the barrier). Warp 0 builds the exclusive
    // prefix of the segment lengths in shared memory; every thread binary-searches its entry's copy.
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* pre = (int*)(sc + S);  // [C + 1] exclusive prefix
    int* st = pre + C + 1;      // [C] segment starts
    int* table = st + C;
    __shared__ int cnt;
    const int HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    if (warp == 0) {
        int carry = 0;
        for (int c0 = 0; c0 < C; c0 += 32) {
            int c = c0 + lane;
            int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
            int v = sl.y;
            for (int o = 1; o < 32; o <<= 1) { int u = __shfl_up_sync(0xffffffff, v, o); if (lane >= o) v += u; }
            if (c < C) { pre[c] = carry + v - sl.y; st[c] = sl.x; }
            carry += __shfl_sync(0xffffffff, v, 31);
        }
        if (lane == 0) pre[C] = carry;
    }
    __syncthreads();
    const int total = pre[C];
    int* crow = cand + (long)r * W;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int lo = 0, hi = C - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (pre[mid] <= i) lo = mid; else hi = mid - 1; }
        int v = ids[(long)lo * N + st[lo] + (i - pre[lo])];
        unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
        for (int probe = 0; probe < HT; probe++) {
            int prev = atomicCAS(&table[s], -1, v);
            if (prev == -1) { int slot = atomicAdd(&cnt, 1); if (slot < W) crow[slot] = v; break; }
            if (prev == v) break;
            s = (s + 1) & (HT - 1);
        }
    }
    __syncthreads();
    const int n = min(cnt, W);
    int L = 1;  // lanes per row: the next power of two >= S / 16 (lanes beyond S idle), at most 32
    while (L < S / 16 && L < 32) L <<= 1;
    const int G = 32 / L, sub = lane % L;
    float* drow = dist + (long)r * W;
    for (int j0 = warp * G; j0 < n; j0 += nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = srow(code, cptr, crows, crow[j], S);
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (threadIdx.x == 0) nout[r] = n;
}

__global__ void gather10_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, int S, int W,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                                long rowlen, const long long* __restrict__ tptr, long rows_per,
                                const long long* __restrict__ cptr, long crows) {
    // gather8 balanced like gather9: the table rows of all seeds (seg.x = row, seg.y = length; the table split in
    // ranges of rows_per rows behind tptr, peer GPUs allowed) spread evenly over the block's threads
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* pre = (int*)(sc + S);  // [C + 1] exclusive prefix
    int* st = pre + C + 1;      // [C] segment starts
    int* table = st + C;
    __shared__ int cnt;
    const int HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    if (warp == 0) {
        int carry = 0;
        for (int c0 = 0; c0 < C; c0 += 32) {
            int c = c0 + lane;
            int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
            int v = sl.y;
            for (int o = 1; o < 32; o <<= 1) { int u = __shfl_up_sync(0xffffffff, v, o); if (lane >= o) v += u; }
            if (c < C) { pre[c] = carry + v - sl.y; st[c] = sl.x; }
            carry += __shfl_sync(0xffffffff, v, 31);
        }
        if (lane == 0) pre[C] = carry;
    }
    __syncthreads();
    const int total = pre[C];
    int* crow = cand + (long)r * W;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int lo = 0, hi = C - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (pre[mid] <= i) lo = mid; else hi = mid - 1; }
        long row = st[lo];
        long sh = row / rows_per;
        int v = reinterpret_cast<const int*>(tptr[sh])[(row - sh * rows_per) * rowlen + (i - pre[lo])];
        unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
        for (int probe = 0; probe < HT; probe++) {
            int prev = atomicCAS(&table[s], -1, v);
            if (prev == -1) { int slot = atomicAdd(&cnt, 1); if (slot < W) crow[slot] = v; break; }
            if (prev == v) break;
            s = (s + 1) & (HT - 1);
        }
    }
    __syncthreads();
    const int n = min(cnt, W);
    int L = 1;  // lanes per row: the next power of two >= S / 16 (lanes beyond S idle), at most 32
    while (L < S / 16 && L < 32) L <<= 1;
    const int G = 32 / L, sub = lane % L;
    float* drow = dist + (long)r * W;
    for (int j0 = warp * G; j0 < n; j0 += nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = srow(code, cptr, crows, crow[j], S);
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (threadIdx.x == 0) nout[r] = n;
}

__global__ void gather7_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, int S, int W,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               int* __restrict__ gtable, long rowlen) {
    // gather5 where seg.x is a row of one shared table (ids + seg.x * rowlen, 64-bit), seg.y its length
    // gather4 over all queries of the batch: block q works only if rowof[q] in [0, rows), writing row rowof[q]
    // (width W) of this class's buffers; gtable (rows x W) holds the hash sets when they do not fit shared memory
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* wbuf = (int*)(sc + S);
    const int nw = blockDim.x >> 5;
    int* table = gtable ? gtable + (long)r * W : wbuf + 96 * nw;
    __shared__ int cnt;
    const int HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    int* crow = cand + (long)r * W;
    for (int c0 = warp * 32; c0 < C; c0 += nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            int v = ids[(long)wst[lo] * rowlen + (i - wex[lo])];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1) { int slot = atomicAdd(&cnt, 1); if (slot < W) crow[slot] = v; break; }
                if (prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
        __syncwarp();
    }
    __syncthreads();
    const int n = min(cnt, W);
    const int L = min(32, S / 16), G = 32 / L, sub = lane % L;
    float* drow = dist + (long)r * W;
    for (int j0 = warp * G; j0 < n; j0 += nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = code + (long)crow[j] * S;
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (threadIdx.x == 0) nout[r] = n;
}

__global__ void gather8_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, int S, int W,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               int* __restrict__ gtable, long rowlen, const long long* __restrict__ tptr, long rows_per) {
    // gather7 over a table split in row ranges of rows_per rows, range s at device pointer tptr[s] (peer GPUs via
    // CUDA IPC and NVLink)
    // gather5 where seg.x is a row of one shared table (ids + seg.x * rowlen, 64-bit), seg.y its length
    // gather4 over all queries of the batch: block q works only if rowof[q] in [0, rows), writing row rowof[q]
    // (width W) of this class's buffers; gtable (rows x W) holds the hash sets when they do not fit shared memory
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* wbuf = (int*)(sc + S);
    const int nw = blockDim.x >> 5;
    int* table = gtable ? gtable + (long)r * W : wbuf + 96 * nw;
    __shared__ int cnt;
    const int HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    int* crow = cand + (long)r * W;
    for (int c0 = warp * 32; c0 < C; c0 += nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            long row = wst[lo];
            long sh = row / rows_per;
            int v = reinterpret_cast<const int*>(tptr[sh])[(row - sh * rows_per) * rowlen + (i - wex[lo])];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1) { int slot = atomicAdd(&cnt, 1); if (slot < W) crow[slot] = v; break; }
                if (prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
        __syncwarp();
    }
    __syncthreads();
    const int n = min(cnt, W);
    const int L = min(32, S / 16), G = 32 / L, sub = lane % L;
    float* drow = dist + (long)r * W;
    for (int j0 = warp * G; j0 < n; j0 += nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = code + (long)crow[j] * S;
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (threadIdx.x == 0) nout[r] = n;
}

__global__ void ginit_kernel(const int* __restrict__ rowof, int rows, const int* __restrict__ hts,
                             int* __restrict__ gtable, long W, int* __restrict__ cnt) {
    const int q = blockIdx.x, r = rowof[q];
    if (r < 0 || r >= rows) return;
    const int HT = hts[q];
    for (long i = (long)blockIdx.y * blockDim.x + threadIdx.x; i < HT; i += (long)gridDim.y * blockDim.x)
        gtable[(long)r * W + i] = -1;
    if (blockIdx.y == 0 && threadIdx.x == 0) cnt[r] = 0;
}

__global__ void ginsert_kernel(const int2* __restrict__ seg, const int* __restrict__ ids, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, long W,
                               int* __restrict__ cand, int* __restrict__ gtable, int* __restrict__ cnt) {
    // gather5's insert phase for heavy queries, spread over gridDim.y blocks per query (global hash set)
    const int q = blockIdx.x, r = rowof[q];
    if (r < 0 || r >= rows) return;
    __shared__ int wbuf[96 * 32];
    const int nw = blockDim.x >> 5, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int HT = hts[q];
    int* table = gtable + (long)r * W;
    int* crow = cand + (long)r * W;
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    for (int c0 = (blockIdx.y * nw + warp) * 32; c0 < C; c0 += gridDim.y * nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            int v = ids[(long)wcp[lo] * N + wst[lo] + (i - wex[lo])];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1) { int slot = atomicAdd(&cnt[r], 1); if (slot < W) crow[slot] = v; break; }
                if (prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
        __syncwarp();
    }
}

__global__ void gsketch_kernel(const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ rowof, int rows, int S, long W,
                               const int* __restrict__ cand, float* __restrict__ dist, const int* __restrict__ cnt,
                               int* __restrict__ nout) {
    // gather5's sketch phase for heavy queries, spread over gridDim.y blocks per query
    const int q = blockIdx.x, r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ float sm2[];
    float* qv = sm2;
    float* sc = qv + S;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    __syncthreads();
    const int n = (int)min((long)cnt[r], W);
    const int nw = blockDim.x >> 5, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int L = min(32, S / 16), G = 32 / L, sub = lane % L;
    const int* crow = cand + (long)r * W;
    float* drow = dist + (long)r * W;
    for (int j0 = (blockIdx.y * nw + warp) * G; j0 < n; j0 += gridDim.y * nw * G) {
        int j = j0 + lane / L;
        float d = 0.f;
        if (j < n) {
            const signed char* x = code + (long)crow[j] * S;
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (j < n && sub == 0) drow[j] = d;
    }
    if (blockIdx.y == 0 && threadIdx.x == 0) nout[r] = n;
}

__device__ __forceinline__ void insert_counted(int* table, int HT, int v, int* crow, int* cnt, int W) {
    // table entries: -1 empty, else (id << 6) | hits (saturating at 63); a new id appends its SLOT to crow
    const int key = v << 6;
    unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
    for (int probe = 0; probe < HT; probe++) {
        int prev = atomicCAS(&table[s], -1, key | 1);
        if (prev == -1) { int slot = atomicAdd(cnt, 1); if (slot < W) crow[slot] = (int)s; return; }
        if ((prev >> 6) == v) {
            int old = prev;
            while ((old & 63) < 63) { int r = atomicCAS(&table[s], old, old + 1); if (r == old) break; old = r; }
            return;
        }
        s = (s + 1) & (HT - 1);
    }
}

__device__ __forceinline__ void sketch_counted(const int* table, int* crow, float* drow, int n, int tau,
                                               const signed char* __restrict__ code, const float* qv,
                                               const float* sc, int S, int j0, int jstep) {
    // for every listed slot: id = entry >> 6; sketch distance if hits >= tau, else +inf (no code read);
    // the row gets the id in place of the slot
    const int lane = threadIdx.x & 31;
    const int L = min(32, S / 16), sub = lane % L;
    for (int jb = j0; jb < n; jb += jstep) {
        int j = jb + lane / L;
        float d = 0.f;
        int e = j < n ? table[crow[j]] : 0;
        bool use = j < n && (e & 63) >= tau;
        if (use) {
            const signed char* x = code + (long)(e >> 6) * S;
            for (int k = sub * 16; k < S; k += L * 16) {
                int4 w = *reinterpret_cast<const int4*>(x + k);
                const signed char* bb = reinterpret_cast<const signed char*>(&w);
                #pragma unroll
                for (int t = 0; t < 16; t++) { float u = (float)bb[t] * sc[k + t] - qv[k + t]; d += u * u; }
            }
        }
        for (int o = L >> 1; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        __syncwarp();
        if (j < n && sub == 0) { crow[j] = e >> 6; drow[j] = use ? d : INFINITY; }
    }
}

__global__ void gather6_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, const int* __restrict__ hts,
                               const int* __restrict__ rowof, int rows, int C, long N, int S, int W, int tau,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout) {
    // gather5 (shared-memory hash set) with hit counts in the table and the sketch only for hits >= tau
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* wbuf = (int*)(sc + S);
    const int nw = blockDim.x >> 5;
    int* table = wbuf + 96 * nw;
    __shared__ int cnt;
    const int HT = hts[q];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    int* crow = cand + (long)r * W;
    for (int c0 = warp * 32; c0 < C; c0 += nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            insert_counted(table, HT, ids[(long)wcp[lo] * N + wst[lo] + (i - wex[lo])], crow, &cnt, W);
        }
        __syncwarp();
    }
    __syncthreads();
    const int n = min(cnt, W);
    const int L = min(32, S / 16), G = 32 / L;
    sketch_counted(table, crow, dist + (long)r * W, n, tau, code, qv, sc, S, warp * G, nw * G);
    if (threadIdx.x == 0) nout[r] = n;
}

__global__ void ginsert2_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                                const int* __restrict__ hts, const int* __restrict__ rowof, int rows, int C, long N,
                                long W, int* __restrict__ cand, int* __restrict__ gtable, int* __restrict__ cnt) {
    const int q = blockIdx.x, r = rowof[q];
    if (r < 0 || r >= rows) return;
    __shared__ int wbuf[96 * 32];
    const int nw = blockDim.x >> 5, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int HT = hts[q];
    int* table = gtable + (long)r * W;
    int* crow = cand + (long)r * W;
    int* wex = wbuf + 96 * warp;
    int* wst = wex + 32;
    int* wcp = wex + 64;
    for (int c0 = (blockIdx.y * nw + warp) * 32; c0 < C; c0 += gridDim.y * nw * 32) {
        int c = c0 + lane;
        int2 sl = c < C ? seg[(long)q * C + c] : make_int2(0, 0);
        int pre = sl.y;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
        int total = __shfl_sync(0xffffffff, pre, 31);
        wex[lane] = pre - sl.y; wst[lane] = sl.x; wcp[lane] = c;
        __syncwarp();
        for (int i = lane; i < total; i += 32) {
            int lo = 0, hi = 31;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (wex[mid] <= i) lo = mid; else hi = mid - 1; }
            insert_counted(table, HT, ids[(long)wcp[lo] * N + wst[lo] + (i - wex[lo])], crow, &cnt[r], (int)W);
        }
        __syncwarp();
    }
}

__global__ void gsketch2_kernel(const signed char* __restrict__ code, const float* __restrict__ scale,
                                const float* __restrict__ qs, const int* __restrict__ rowof, int rows, int S, long W,
                                int tau, int* __restrict__ cand, float* __restrict__ dist,
                                const int* __restrict__ gtable, const int* __restrict__ cnt, int* __restrict__ nout) {
    const int q = blockIdx.x, r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ float sm2[];
    float* qv = sm2;
    float* sc = qv + S;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    __syncthreads();
    const int n = (int)min((long)cnt[r], W);
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5;
    const int L = min(32, S / 16), G = 32 / L;
    sketch_counted(gtable + (long)r * W, cand + (long)r * W, dist + (long)r * W, n, tau, code, qv, sc, S,
                   (blockIdx.y * nw + warp) * G, gridDim.y * nw * G);
    if (blockIdx.y == 0 && threadIdx.x == 0) nout[r] = n;
}

__device__ __forceinline__ const __half* xrow(const __half* X, const long long* xptr, long xrows, int id, int D) {
    // row id of the fp16 vectors: X itself, or row ranges of xrows rows behind device pointers (peer GPUs allowed)
    if (xptr) return reinterpret_cast<const __half*>(xptr[id / xrows]) + (long)(id % xrows) * D;
    return X + (long)id * D;
}

__global__ void select_rerank3_kernel(const int* __restrict__ cand, const float* __restrict__ dist,
                                      const int* __restrict__ nin, const int* __restrict__ rowof, int rows, int W,
                                      const __half* __restrict__ X, int D, const float* __restrict__ Q, int R, int KO,
                                      long long* __restrict__ bi, float* __restrict__ bd,
                                      const long long* __restrict__ xptr, long xrows) {
    // select_rerank2 keeping the best KO: radix-select the R sketch-nearest, exact distances, then a bitonic sort of
    // the (distance, id) keys of the current KO and the R new ones; equal keys (the same id twice) collapse
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ float sq[];
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, kk;
    __shared__ int nsel, ntie;
    __shared__ int sid[512];
    __shared__ float sd[512];
    __shared__ unsigned long long key[1024];
    const int n = nin[r];
    const unsigned* kd = reinterpret_cast<const unsigned*>(dist + (long)r * W);
    const int* cid = cand + (long)r * W;
    for (int i = threadIdx.x; i < D; i += blockDim.x) sq[i] = Q[(long)q * D + i];
    if (threadIdx.x == 0) { prefix = 0; kk = min(R, n); nsel = 0; ntie = 0; }
    __syncthreads();
    unsigned mask = 0;
    if (n > R) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
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
                for (; d < 256; d++) { if (cum + hist[d] >= k) break; cum += hist[d]; }
                kk = k - cum;
                prefix = pf | ((unsigned)d << shift);
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        unsigned T = prefix, ties = kk;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned k = kd[i];
            if (k < T) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; }
            else if (k == T) { int t = atomicAdd(&ntie, 1); if (t < (int)ties) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; } }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) sid[i] = cid[i];
        if (threadIdx.x == 0) nsel = n;
    }
    __syncthreads();
    const int ns = nsel;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = warp; j < ns; j += nw) {
        const __half2* x = reinterpret_cast<const __half2*>(xrow(X, xptr, xrows, sid[j], D));
        float acc = 0.f;
        for (int k = lane; k < D / 2; k += 32) {
            float2 v = __half22float2(x[k]);
            float a = v.x - sq[2 * k], b = v.y - sq[2 * k + 1];
            acc += a * a + b * b;
        }
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (lane == 0) sd[j] = acc;
    }
    __syncthreads();
    long long* pb = bi + (long)q * KO;
    float* pd = bd + (long)q * KO;
    int P = 2;  // sort only the next power of two >= KO + R keys (not a fixed 1024)
    while (P < KO + R) P <<= 1;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        unsigned long long k = ~0ULL;
        if (i < KO) {
            if (pb[i] >= 0) k = ((unsigned long long)__float_as_uint(pd[i]) << 32) | (unsigned)pb[i];
        } else if (i - KO < ns) {
            k = ((unsigned long long)__float_as_uint(sd[i - KO]) << 32) | (unsigned)sid[i - KO];
        }
        key[i] = k;
    }
    __syncthreads();
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < P; i += blockDim.x) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = (i & size) == 0;
                    unsigned long long a = key[i], b = key[j];
                    if ((a > b) == up) { key[i] = b; key[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        int o = 0;
        unsigned long long prev = ~0ULL - 1;
        for (int i = 0; i < P && o < KO; i++) {
            unsigned long long k = key[i];
            if (k == ~0ULL) break;
            if (k == prev) continue;
            prev = k;
            pb[o] = (long long)(unsigned)(k & 0xffffffffULL);
            pd[o] = __uint_as_float((unsigned)(k >> 32));
            o++;
        }
        for (; o < KO; o++) { pb[o] = -1; pd[o] = INFINITY; }
    }
}

__global__ void peer_gather_kernel(const long long* __restrict__ pi, const long long* __restrict__ pd, int ng,
                                   long lo, int m, int KO, long long* __restrict__ oi, float* __restrict__ od) {
    // out[q, h * KO + k] = source h's row lo + q, column k (sources may live on peer GPUs)
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)m * ng * KO;
    if (t >= total) return;
    int k = t % KO;
    int h = (t / KO) % ng;
    long q = t / ((long)KO * ng);
    long src = (lo + q) * KO + k;
    oi[t] = reinterpret_cast<const long long*>(pi[h])[src];
    od[t] = reinterpret_cast<const float*>(pd[h])[src];
}

__global__ void gather2_kernel(const int2* __restrict__ seg, const int* __restrict__ ids,
                               const signed char* __restrict__ code, const float* __restrict__ scale,
                               const float* __restrict__ qs, int C, long N, int S, int HT, int maxc,
                               int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                               int* __restrict__ gtable) {
    // one thread block per query; every thread walks copies (one copy per thread at a time)
    extern __shared__ int sm[];
    float* qv = (float*)sm;
    float* sc = qv + S;
    int* table = gtable ? gtable + (long)blockIdx.x * HT : sm + 2 * S;
    __shared__ int cnt;
    const int q = blockIdx.x;
    if (!gtable) for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int2 sl = seg[(long)q * C + c];
        const int* idc = ids + (long)c * N + sl.x;
        for (int p = 0; p < sl.y; p++) {
            int v = idc[p];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1 || prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < HT; i += blockDim.x) {
        int v = table[i];
        if (v < 0) continue;
        const signed char* x = code + (long)v * S;
        float d = 0.f;
        for (int k = 0; k < S; k += 16) {
            int4 w = *reinterpret_cast<const int4*>(x + k);
            const signed char* bb = reinterpret_cast<const signed char*>(&w);
            #pragma unroll
            for (int j = 0; j < 16; j++) { float t = (float)bb[j] * sc[k + j] - qv[k + j]; d += t * t; }
        }
        int slot = atomicAdd(&cnt, 1);
        if (slot < maxc) { cand[(long)q * maxc + slot] = v; dist[(long)q * maxc + slot] = d; }
    }
    __syncthreads();
    if (threadIdx.x == 0) nout[q] = min(cnt, maxc);
}

__global__ void select_rerank_kernel(const int* __restrict__ cand, const float* __restrict__ dist,
                                     const int* __restrict__ nin, int W, const __half* __restrict__ X, int D,
                                     const float* __restrict__ Q, int R, long long* __restrict__ bi,
                                     float* __restrict__ bd) {
    // per query (block): radix-select the R smallest sketch distances of its row, exact distances of those R,
    // merge with its current best 10 (bi, bd, in place; ids may repeat across steps)
    extern __shared__ float sq[];                      // D floats: the query
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, kk;
    __shared__ int nsel, ntie;
    __shared__ int sid[512];
    __shared__ float sd[512];
    const int q = blockIdx.x, n = nin[q];
    const unsigned* key = reinterpret_cast<const unsigned*>(dist + (long)q * W);
    const int* cid = cand + (long)q * W;
    for (int i = threadIdx.x; i < D; i += blockDim.x) sq[i] = Q[(long)q * D + i];
    if (threadIdx.x == 0) { prefix = 0; kk = min(R, n); nsel = 0; ntie = 0; }
    __syncthreads();
    unsigned mask = 0;
    if (n > R) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            unsigned pf = prefix;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                unsigned k = key[i];
                if ((k & mask) == pf) atomicAdd(&hist[(k >> shift) & 255], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned cum = 0, k = kk;
                int d = 0;
                for (; d < 256; d++) { if (cum + hist[d] >= k) break; cum += hist[d]; }
                kk = k - cum;
                prefix = pf | ((unsigned)d << shift);
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        // keys < prefix are in; kk of the keys == prefix are in
        unsigned T = prefix, ties = kk;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned k = key[i];
            if (k < T) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; }
            else if (k == T) { int t = atomicAdd(&ntie, 1); if (t < (int)ties) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; } }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) sid[i] = cid[i];
        if (threadIdx.x == 0) nsel = n;
    }
    __syncthreads();
    const int ns = nsel;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = warp; j < ns; j += nw) {
        const __half2* x = reinterpret_cast<const __half2*>(X + (long)sid[j] * D);
        float acc = 0.f;
        for (int k = lane; k < D / 2; k += 32) {
            float2 v = __half22float2(x[k]);
            float a = v.x - sq[2 * k], b = v.y - sq[2 * k + 1];
            acc += a * a + b * b;
        }
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (lane == 0) sd[j] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        long long oi[10]; float od[10];
        long long* pb = bi + (long)q * 10; float* pd = bd + (long)q * 10;
        for (int t = 0; t < 10; t++) { oi[t] = pb[t]; od[t] = pd[t]; }
        for (int j = 0; j < ns; j++) {
            float d = sd[j]; long long v = sid[j];
            if (!(d < od[9])) continue;
            bool dup = false;
            for (int t = 0; t < 10; t++) if (oi[t] == v) { dup = true; break; }
            if (dup) continue;
            int t = 9;
            while (t > 0 && od[t - 1] > d) { od[t] = od[t - 1]; oi[t] = oi[t - 1]; t--; }
            od[t] = d; oi[t] = v;
        }
        for (int t = 0; t < 10; t++) { pb[t] = oi[t]; pd[t] = od[t]; }
    }
}

__global__ void select_rerank2_kernel(const int* __restrict__ cand, const float* __restrict__ dist,
                                      const int* __restrict__ nin, const int* __restrict__ rowof, int rows, int W,
                                      const __half* __restrict__ X, int D, const float* __restrict__ Q, int R,
                                      long long* __restrict__ bi, float* __restrict__ bd,
                                      const long long* __restrict__ xptr, long xrows) {
    // select_rerank over all queries of the batch: block q works only if rowof[q] in [0, rows), reading row
    // rowof[q] of the class buffers and merging into row q of bi, bd
    const int q = blockIdx.x;
    const int r = rowof[q];
    if (r < 0 || r >= rows) return;
    extern __shared__ float sq[];
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, kk;
    __shared__ int nsel, ntie;
    __shared__ int sid[512];
    __shared__ float sd[512];
    const int n = nin[r];
    const unsigned* key = reinterpret_cast<const unsigned*>(dist + (long)r * W);
    const int* cid = cand + (long)r * W;
    for (int i = threadIdx.x; i < D; i += blockDim.x) sq[i] = Q[(long)q * D + i];
    if (threadIdx.x == 0) { prefix = 0; kk = min(R, n); nsel = 0; ntie = 0; }
    __syncthreads();
    unsigned mask = 0;
    if (n > R) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            unsigned pf = prefix;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                unsigned k = key[i];
                if ((k & mask) == pf) atomicAdd(&hist[(k >> shift) & 255], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned cum = 0, k = kk;
                int d = 0;
                for (; d < 256; d++) { if (cum + hist[d] >= k) break; cum += hist[d]; }
                kk = k - cum;
                prefix = pf | ((unsigned)d << shift);
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        unsigned T = prefix, ties = kk;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned k = key[i];
            if (k < T) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; }
            else if (k == T) { int t = atomicAdd(&ntie, 1); if (t < (int)ties) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; } }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) sid[i] = cid[i];
        if (threadIdx.x == 0) nsel = n;
    }
    __syncthreads();
    const int ns = nsel;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = warp; j < ns; j += nw) {
        const __half2* x = reinterpret_cast<const __half2*>(xrow(X, xptr, xrows, sid[j], D));
        float acc = 0.f;
        for (int k = lane; k < D / 2; k += 32) {
            float2 v = __half22float2(x[k]);
            float a = v.x - sq[2 * k], b = v.y - sq[2 * k + 1];
            acc += a * a + b * b;
        }
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (lane == 0) sd[j] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        long long oi[10]; float od[10];
        long long* pb = bi + (long)q * 10; float* pd = bd + (long)q * 10;
        for (int t = 0; t < 10; t++) { oi[t] = pb[t]; od[t] = pd[t]; }
        for (int j = 0; j < ns; j++) {
            float d = sd[j]; long long v = sid[j];
            if (!(d < od[9])) continue;
            bool dup = false;
            for (int t = 0; t < 10; t++) if (oi[t] == v) { dup = true; break; }
            if (dup) continue;
            int t = 9;
            while (t > 0 && od[t - 1] > d) { od[t] = od[t - 1]; oi[t] = oi[t - 1]; t--; }
            od[t] = d; oi[t] = v;
        }
        for (int t = 0; t < 10; t++) { pb[t] = oi[t]; pd[t] = od[t]; }
    }
}

__global__ void gather_kernel(const int* __restrict__ bucket, const int* __restrict__ off, const int* __restrict__ ids,
                              const signed char* __restrict__ code, const float* __restrict__ scale,
                              const float* __restrict__ qs, int C, long T1, long N, int S, int cap, int HT, int maxc,
                              int* __restrict__ cand, float* __restrict__ dist, int* __restrict__ nout,
                              unsigned long long* __restrict__ reads, int* __restrict__ gtable) {
    extern __shared__ int sm[];
    float* qv = (float*)sm;                            // S
    float* sc = qv + S;                                // S
    int* table = gtable ? gtable + (long)blockIdx.x * HT : sm + 2 * S;   // HT slots, shared or global
    __shared__ int cnt;
    __shared__ unsigned long long rd;
    const int q = blockIdx.x;
    if (!gtable) for (int i = threadIdx.x; i < HT; i += blockDim.x) table[i] = -1;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { qv[i] = qs[(long)q * S + i]; sc[i] = scale[i]; }
    if (threadIdx.x == 0) { cnt = 0; rd = 0; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned long long myrd = 0;
    for (int c = warp; c < C; c += nw) {
        int h = bucket[(long)q * C + c];
        const int* o = off + (long)c * T1 + h;
        int st = o[0];
        int en = min(o[1], st + cap);
        myrd += (lane == 0) ? (unsigned long long)(en - st) : 0ULL;
        const int* idc = ids + (long)c * N;
        for (int p = st + lane; p < en; p += 32) {
            int v = idc[p];
            unsigned s = ((unsigned)v * 2654435761u) & (HT - 1);
            for (int probe = 0; probe < HT; probe++) {
                int prev = atomicCAS(&table[s], -1, v);
                if (prev == -1 || prev == v) break;
                s = (s + 1) & (HT - 1);
            }
        }
    }
    if (lane == 0 && myrd) atomicAdd(&rd, myrd);
    __syncthreads();
    for (int i = threadIdx.x; i < HT; i += blockDim.x) {
        int v = table[i];
        if (v < 0) continue;
        const signed char* x = code + (long)v * S;
        float d = 0.f;
        for (int k = 0; k < S; k += 16) {
            int4 w = *reinterpret_cast<const int4*>(x + k);
            const signed char* bb = reinterpret_cast<const signed char*>(&w);
            #pragma unroll
            for (int j = 0; j < 16; j++) { float t = (float)bb[j] * sc[k + j] - qv[k + j]; d += t * t; }
        }
        int slot = atomicAdd(&cnt, 1);
        if (slot < maxc) { cand[(long)q * maxc + slot] = v; dist[(long)q * maxc + slot] = d; }
    }
    __syncthreads();
    if (threadIdx.x == 0) { nout[q] = min(cnt, maxc); atomicAdd(reads, rd); }
}

std::vector<torch::Tensor> decode2(torch::Tensor Y, int64_t b, int64_t logT, torch::Tensor off, int64_t cap) {
    const c10::cuda::OptionalCUDAGuard device_guard_(Y.device());
    int m = Y.size(0); int C = Y.size(1) / (8 * b);
    auto seg = torch::empty({m, C, 2}, Y.options().dtype(torch::kInt32));
    auto tot = torch::zeros({m}, Y.options().dtype(torch::kInt32));
    long n = (long)m * C;
    int th = 256;
    decode2_kernel<<<(n + th - 1) / th, th, 0, c10::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(), m, C, (int)b, (int)logT, off.data_ptr<int>(), off.size(1), (int)cap,
        (int2*)seg.data_ptr<int>(), tot.data_ptr<int>());
    return {seg, tot};
}

std::vector<torch::Tensor> gather2(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale,
                                   torch::Tensor qs, int64_t HT, int64_t maxc, int64_t threads) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    auto cand = torch::full({m, maxc}, -1, seg.options());
    auto dist = torch::full({m, maxc}, INFINITY, qs.options());
    auto nout = torch::empty({m}, seg.options());
    bool global = HT > 32768;
    torch::Tensor gt;
    if (global) gt = torch::full({m, HT}, -1, seg.options());
    size_t smem = (global ? 0 : HT * sizeof(int)) + 2 * S * sizeof(float);
    cudaFuncSetAttribute(gather2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather2_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), C, ids.size(1), S, (int)HT, (int)maxc,
        cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(), global ? gt.data_ptr<int>() : nullptr);
    return {cand, dist, nout};
}

void select_rerank(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor X, torch::Tensor Q,
                   int64_t R, torch::Tensor bi, torch::Tensor bd) {
    const c10::cuda::OptionalCUDAGuard device_guard_(cand.device());
    int m = cand.size(0), W = cand.size(1), D = X.size(1);
    TORCH_CHECK(R <= 512 && D % 2 == 0 && bi.size(1) == 10);
    select_rerank_kernel<<<m, 256, D * sizeof(float), c10::cuda::getCurrentCUDAStream()>>>(
        cand.data_ptr<int>(), dist.data_ptr<float>(), n.data_ptr<int>(), W, (const __half*)X.data_ptr<at::Half>(), D,
        Q.data_ptr<float>(), (int)R, (long long*)bi.data_ptr<int64_t>(), bd.data_ptr<float>());
}

std::vector<torch::Tensor> decode3(torch::Tensor Y, int64_t b, int64_t logT, torch::Tensor off, int64_t cap) {
    const c10::cuda::OptionalCUDAGuard device_guard_(Y.device());
    TORCH_CHECK(Y.scalar_type() == at::kHalf && b <= 4);
    int m = Y.size(0); int C = Y.size(1) / (8 * b);
    auto seg = torch::empty({m, C, 2}, Y.options().dtype(torch::kInt32));
    auto tot = torch::zeros({m}, Y.options().dtype(torch::kInt32));
    long n = (long)m * C;
    int th = 256;
    decode3_kernel<<<(n + th - 1) / th, th, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const __half*)Y.data_ptr<at::Half>(), m, C, (int)b, (int)logT, off.data_ptr<int>(), off.size(1), (int)cap,
        (int2*)seg.data_ptr<int>(), tot.data_ptr<int>());
    return {seg, tot};
}

std::vector<torch::Tensor> gather3(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale,
                                   torch::Tensor qs, torch::Tensor hts, int64_t HTmax, int64_t threads) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // hts [m]: per-query hash-set sizes (powers of two) <= HTmax; HTmax <= 32768 -> shared memory, else global.
    // Rows of the output are HTmax wide.
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    auto cand = torch::empty({m, HTmax}, seg.options());   // only the first nout[q] entries of a row are written
    auto dist = torch::empty({m, HTmax}, qs.options());
    auto nout = torch::empty({m}, seg.options());
    bool global = HTmax > 32768;
    torch::Tensor gt;
    if (global) gt = torch::empty({m, HTmax}, seg.options());
    size_t smem = (global ? 0 : HTmax * sizeof(int)) + 2 * S * sizeof(float);
    cudaFuncSetAttribute(gather3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather3_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), C, ids.size(1), S, (int)HTmax,
        cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(), global ? gt.data_ptr<int>() : nullptr,
        HTmax);
    return {cand, dist, nout};
}

std::vector<torch::Tensor> gather4(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale,
                                   torch::Tensor qs, torch::Tensor hts, int64_t HTmax, int64_t threads) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // as gather3; rows are HTmax wide, the first nout[q] entries hold the distinct ids and their sketch distances
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512 (lanes reduce by xor shuffles)");
    TORCH_CHECK(S % 16 == 0 && threads % 32 == 0);
    auto cand = torch::empty({m, HTmax}, seg.options());
    auto dist = torch::empty({m, HTmax}, qs.options());
    auto nout = torch::empty({m}, seg.options());
    bool global = HTmax > 32768;
    torch::Tensor gt;
    if (global) gt = torch::empty({m, HTmax}, seg.options());
    size_t smem = (global ? 0 : HTmax * sizeof(int)) + 2 * S * sizeof(float) + 96 * (threads / 32) * sizeof(int);
    cudaFuncSetAttribute(gather4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather4_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), C, ids.size(1), S, (int)HTmax,
        cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(), global ? gt.data_ptr<int>() : nullptr,
        HTmax);
    return {cand, dist, nout};
}

void decode4(torch::Tensor Y, torch::Tensor active, int64_t b, int64_t logT, torch::Tensor off, int64_t cap,
             torch::Tensor seg, torch::Tensor tot) {
    const c10::cuda::OptionalCUDAGuard device_guard_(Y.device());
    // writes seg [m, C, 2] and adds to tot [m] (zeroed by the caller) for the active queries
    TORCH_CHECK(Y.scalar_type() == at::kHalf);
    int m = Y.size(0); int C = Y.size(1) / (8 * b);
    long n = (long)m * C;
    int th = 256, gr = (n + th - 1) / th;
    auto st = c10::cuda::getCurrentCUDAStream();
    const __half* y = (const __half*)Y.data_ptr<at::Half>();
    const bool* a = active.data_ptr<bool>();
    int2* sg = (int2*)seg.data_ptr<int>();
    int* tt = tot.data_ptr<int>();
    long T1 = off.size(1);
    switch (b) {
        case 1: decode4_kernel<1><<<gr, th, 0, st>>>(y, a, m, C, (int)logT, off.data_ptr<int>(), T1, (int)cap, sg, tt); break;
        case 2: decode4_kernel<2><<<gr, th, 0, st>>>(y, a, m, C, (int)logT, off.data_ptr<int>(), T1, (int)cap, sg, tt); break;
        case 3: decode4_kernel<3><<<gr, th, 0, st>>>(y, a, m, C, (int)logT, off.data_ptr<int>(), T1, (int)cap, sg, tt); break;
        case 4: decode4_kernel<4><<<gr, th, 0, st>>>(y, a, m, C, (int)logT, off.data_ptr<int>(), T1, (int)cap, sg, tt); break;
        case 5: decode4_kernel<5><<<gr, th, 0, st>>>(y, a, m, C, (int)logT, off.data_ptr<int>(), T1, (int)cap, sg, tt); break;
        case 6: decode4_kernel<6><<<gr, th, 0, st>>>(y, a, m, C, (int)logT, off.data_ptr<int>(), T1, (int)cap, sg, tt); break;
        default: TORCH_CHECK(false, "b must be 1..6");
    }
}

void gather5(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
             torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads, bool shared,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor gtable) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // all buffers preallocated (fixed shapes, CUDA-graph friendly): cand, dist [rows, W]; nout [rows];
    // gtable [rows, W] when not shared
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512 (lanes reduce by xor shuffles)");
    size_t smem = (shared ? W * sizeof(int) : 0) + 2 * S * sizeof(float) + 96 * (threads / 32) * sizeof(int);
    cudaFuncSetAttribute(gather5_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather5_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C,
        ids.size(1), S, (int)W, cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(),
        shared ? nullptr : gtable.data_ptr<int>());
}

void gather9(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
             torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor cptr, int64_t crows) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // gather5 with the list entries of all copies spread evenly over the block's threads (hash set in shared memory)
    int m = seg.size(0), C = seg.size(1), S = code.size(1);  // code: the sketch, or a local shard (width S) with cptr
    TORCH_CHECK(S % 16 == 0 && S / 16 <= 32, "sketch width must be a multiple of 16, at most 512");
    size_t smem = W * sizeof(int) + 2 * S * sizeof(float) + (2 * C + 1) * sizeof(int);
    cudaFuncSetAttribute(gather9_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather9_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C,
        ids.size(1), S, (int)W, cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(),
        cptr.numel() ? (const long long*)cptr.data_ptr<int64_t>() : nullptr, (long)crows);
}

void select_rerank2(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                    torch::Tensor X, torch::Tensor Q, int64_t R, torch::Tensor bi, torch::Tensor bd) {
    const c10::cuda::OptionalCUDAGuard device_guard_(cand.device());
    int m = Q.size(0), W = cand.size(1), D = X.size(1);
    TORCH_CHECK(R <= 512 && D % 2 == 0 && bi.size(1) == 10);
    select_rerank2_kernel<<<m, 256, D * sizeof(float), c10::cuda::getCurrentCUDAStream()>>>(
        cand.data_ptr<int>(), dist.data_ptr<float>(), n.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, W,
        (const __half*)X.data_ptr<at::Half>(), D, Q.data_ptr<float>(), (int)R, (long long*)bi.data_ptr<int64_t>(),
        bd.data_ptr<float>(), nullptr, 1L);
}

void select_rerank2s(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                     torch::Tensor X, torch::Tensor xptr, int64_t xrows, torch::Tensor Q, int64_t R, torch::Tensor bi,
                     torch::Tensor bd) {
    // select_rerank2 reading the vectors from row ranges (xptr int64 [shards] device pointers, xrows rows each)
    const c10::cuda::OptionalCUDAGuard device_guard_(cand.device());
    int m = Q.size(0), W = cand.size(1), D = X.size(1);
    TORCH_CHECK(R <= 512 && D % 2 == 0 && bi.size(1) == 10);
    select_rerank2_kernel<<<m, 256, D * sizeof(float), c10::cuda::getCurrentCUDAStream()>>>(
        cand.data_ptr<int>(), dist.data_ptr<float>(), n.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, W,
        (const __half*)X.data_ptr<at::Half>(), D, Q.data_ptr<float>(), (int)R, (long long*)bi.data_ptr<int64_t>(),
        bd.data_ptr<float>(), (const long long*)xptr.data_ptr<int64_t>(), (long)xrows);
}

void gather_global(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
                   torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t split, int64_t threads,
                   torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor gtable,
                   torch::Tensor cnt) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // gather5 for heavy queries: split blocks per query, global hash set (gtable [rows, W]), counters cnt [rows]
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512 (lanes reduce by xor shuffles)");
    TORCH_CHECK(threads <= 1024 && threads % 32 == 0);
    auto st = c10::cuda::getCurrentCUDAStream();
    dim3 grid(m, split);
    ginit_kernel<<<grid, threads, 0, st>>>(rowof.data_ptr<int>(), (int)rows, hts.data_ptr<int>(),
                                           gtable.data_ptr<int>(), W, cnt.data_ptr<int>());
    ginsert_kernel<<<grid, threads, 0, st>>>((const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(),
                                             hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C, ids.size(1),
                                             W, cand.data_ptr<int>(), gtable.data_ptr<int>(), cnt.data_ptr<int>());
    gsketch_kernel<<<grid, threads, 2 * S * sizeof(float), st>>>(
        (const signed char*)code.data_ptr<int8_t>(), scale.data_ptr<float>(), qs.data_ptr<float>(),
        rowof.data_ptr<int>(), (int)rows, S, W, cand.data_ptr<int>(), dist.data_ptr<float>(), cnt.data_ptr<int>(),
        nout.data_ptr<int>());
}

void gather6(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
             torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads, int64_t tau,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    TORCH_CHECK(ids.size(1) < (1L << 25), "ids must fit 25 bits (hit counts share the table entry)");
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512 (lanes reduce by xor shuffles)");
    size_t smem = W * sizeof(int) + 2 * S * sizeof(float) + 96 * (threads / 32) * sizeof(int);
    cudaFuncSetAttribute(gather6_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather6_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C,
        ids.size(1), S, (int)W, (int)tau, cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>());
}

void gather_global2(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
                    torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t split, int64_t threads,
                    int64_t tau, torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor gtable,
                    torch::Tensor cnt) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    TORCH_CHECK(ids.size(1) < (1L << 25));
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512 (lanes reduce by xor shuffles)");
    auto st = c10::cuda::getCurrentCUDAStream();
    dim3 grid(m, split);
    ginit_kernel<<<grid, threads, 0, st>>>(rowof.data_ptr<int>(), (int)rows, hts.data_ptr<int>(),
                                           gtable.data_ptr<int>(), W, cnt.data_ptr<int>());
    ginsert2_kernel<<<grid, threads, 0, st>>>((const int2*)seg.data_ptr<int>(), ids.data_ptr<int>(),
                                              hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C, ids.size(1),
                                              W, cand.data_ptr<int>(), gtable.data_ptr<int>(), cnt.data_ptr<int>());
    gsketch2_kernel<<<grid, threads, 2 * S * sizeof(float), st>>>(
        (const signed char*)code.data_ptr<int8_t>(), scale.data_ptr<float>(), qs.data_ptr<float>(),
        rowof.data_ptr<int>(), (int)rows, S, W, (int)tau, cand.data_ptr<int>(), dist.data_ptr<float>(),
        gtable.data_ptr<int>(), cnt.data_ptr<int>(), nout.data_ptr<int>());
}

void probe(torch::Tensor Y, int64_t b, torch::Tensor inv_f, torch::Tensor off, torch::Tensor keys, torch::Tensor base,
           int64_t l, int64_t lr, int64_t logT, torch::Tensor seg, torch::Tensor tot, torch::Tensor lvl) {
    const c10::cuda::OptionalCUDAGuard device_guard_(Y.device());
    // Y [m, C * 8b] fp16; off [C, L, T+1] int32; keys [C, L, Mmax] int64; base [C, L] int32; seg [m, C, 2];
    // tot [m] (zeroed by the caller); lvl [m, C] int32 (the level used, -1 none) or an empty tensor
    TORCH_CHECK(Y.scalar_type() == at::kHalf && b == 4, "probe: fp16 projections, b = 4");
    int m = Y.size(0), C = off.size(0), L = off.size(1);
    long n = (long)m * C;
    int th = 256;
    probe_kernel<4><<<(n + th - 1) / th, th, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const __half*)Y.data_ptr<at::Half>(), m, C, L, inv_f.data_ptr<float>(), off.data_ptr<int>(), off.size(2),
        (const long long*)keys.data_ptr<int64_t>(), keys.size(2), base.data_ptr<int>(), (int)l, (int)lr, (int)logT,
        (int2*)seg.data_ptr<int>(), tot.data_ptr<int>(), lvl.numel() ? lvl.data_ptr<int>() : nullptr);
}

void gather7(torch::Tensor seg, torch::Tensor table, int64_t rowlen, torch::Tensor code, torch::Tensor scale,
             torch::Tensor qs, torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // gather5 over rows of one shared table: seg [m, S, 2] = (row, length), ids table[row * rowlen + k]
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512");
    size_t smem = W * sizeof(int) + 2 * S * sizeof(float) + 96 * (threads / 32) * sizeof(int);
    cudaFuncSetAttribute(gather7_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather7_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), table.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C,
        0L, S, (int)W, cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(), nullptr, (long)rowlen);
}

void select_rerank3(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                    torch::Tensor X, torch::Tensor Q, int64_t R, torch::Tensor bi, torch::Tensor bd) {
    const c10::cuda::OptionalCUDAGuard device_guard_(cand.device());
    int m = Q.size(0), W = cand.size(1), D = X.size(1), KO = bi.size(1);
    TORCH_CHECK(R <= 512 && KO <= 512 && R + KO <= 1024 && D % 2 == 0);
    select_rerank3_kernel<<<m, 256, D * sizeof(float), c10::cuda::getCurrentCUDAStream()>>>(
        cand.data_ptr<int>(), dist.data_ptr<float>(), n.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, W,
        (const __half*)X.data_ptr<at::Half>(), D, Q.data_ptr<float>(), (int)R, KO,
        (long long*)bi.data_ptr<int64_t>(), bd.data_ptr<float>(), nullptr, 1L);
}

void select_rerank3s(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                     torch::Tensor X, torch::Tensor xptr, int64_t xrows, torch::Tensor Q, int64_t R, torch::Tensor bi,
                     torch::Tensor bd) {
    // select_rerank3 reading the vectors from row ranges (xptr int64 [shards] device pointers, xrows rows each)
    const c10::cuda::OptionalCUDAGuard device_guard_(cand.device());
    int m = Q.size(0), W = cand.size(1), D = X.size(1), KO = bi.size(1);
    TORCH_CHECK(R <= 512 && KO <= 512 && R + KO <= 1024 && D % 2 == 0);
    select_rerank3_kernel<<<m, 256, D * sizeof(float), c10::cuda::getCurrentCUDAStream()>>>(
        cand.data_ptr<int>(), dist.data_ptr<float>(), n.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, W,
        (const __half*)X.data_ptr<at::Half>(), D, Q.data_ptr<float>(), (int)R, KO,
        (long long*)bi.data_ptr<int64_t>(), bd.data_ptr<float>(), (const long long*)xptr.data_ptr<int64_t>(),
        (long)xrows);
}

void gather8(torch::Tensor seg, torch::Tensor tptr, int64_t rows_per, int64_t rowlen, torch::Tensor code,
             torch::Tensor scale, torch::Tensor qs, torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W,
             int64_t threads, torch::Tensor cand, torch::Tensor dist, torch::Tensor nout) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // gather7 over a table split in row ranges; tptr int64 [shards] holds the device pointers of the ranges
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && ((S / 16) & (S / 16 - 1)) == 0 && S / 16 <= 32, "sketch width must be 16 x a power of two <= 512");
    size_t smem = W * sizeof(int) + 2 * S * sizeof(float) + 96 * (threads / 32) * sizeof(int);
    cudaFuncSetAttribute(gather8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather8_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), nullptr, (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C,
        0L, S, (int)W, cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(), nullptr, (long)rowlen,
        (const long long*)tptr.data_ptr<int64_t>(), (long)rows_per);
}

void gather10(torch::Tensor seg, torch::Tensor tptr, int64_t rows_per, int64_t rowlen, torch::Tensor code,
              torch::Tensor scale, torch::Tensor qs, torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W,
              int64_t threads, torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor cptr,
              int64_t crows) {
    const c10::cuda::OptionalCUDAGuard device_guard_(seg.device());
    // gather8 with the table rows of all seeds spread evenly over the block's threads
    int m = seg.size(0), C = seg.size(1), S = code.size(1);
    TORCH_CHECK(S % 16 == 0 && S / 16 <= 32, "sketch width must be a multiple of 16, at most 512");
    size_t smem = W * sizeof(int) + 2 * S * sizeof(float) + (2 * C + 1) * sizeof(int);
    cudaFuncSetAttribute(gather10_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather10_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const int2*)seg.data_ptr<int>(), nullptr, (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), hts.data_ptr<int>(), rowof.data_ptr<int>(), (int)rows, C,
        0L, S, (int)W, cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(), (long)rowlen,
        (const long long*)tptr.data_ptr<int64_t>(), (long)rows_per,
        cptr.numel() ? (const long long*)cptr.data_ptr<int64_t>() : nullptr, (long)crows);
}

void peer_gather(torch::Tensor pi, torch::Tensor pd, int64_t lo, torch::Tensor oi, torch::Tensor od) {
    // oi int64 [m, ng * KO], od float [m, ng * KO]; pi / pd int64 [ng]: pointers to [*, KO] lists on any GPU
    const c10::cuda::OptionalCUDAGuard device_guard_(oi.device());
    int ng = pi.size(0), m = oi.size(0), KO = oi.size(1) / ng;
    long total = (long)m * ng * KO;
    peer_gather_kernel<<<(total + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const long long*)pi.data_ptr<int64_t>(), (const long long*)pd.data_ptr<int64_t>(), ng, (long)lo, m, KO,
        (long long*)oi.data_ptr<int64_t>(), od.data_ptr<float>());
}

torch::Tensor decode(torch::Tensor Y, int64_t b, int64_t logT) {
    const c10::cuda::OptionalCUDAGuard device_guard_(Y.device());
    int m = Y.size(0); int C = Y.size(1) / (8 * b);
    auto out = torch::empty({m, C}, Y.options().dtype(torch::kInt32));
    long n = (long)m * C;
    int th = 256;
    decode_kernel<<<(n + th - 1) / th, th, 0, c10::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(), m, C, (int)b, (int)logT, out.data_ptr<int>());
    return out;
}

std::vector<torch::Tensor> gather(torch::Tensor bucket, torch::Tensor off, torch::Tensor ids, torch::Tensor code,
                                  torch::Tensor scale, torch::Tensor qs, int64_t cap, int64_t HT, int64_t maxc,
                                  int64_t threads) {
    const c10::cuda::OptionalCUDAGuard device_guard_(bucket.device());
    // HT <= 32768: hash set in shared memory; larger: in global memory (one HT-slot table per query)
    int m = bucket.size(0), C = bucket.size(1), S = code.size(1);
    auto cand = torch::full({m, maxc}, -1, bucket.options());
    auto dist = torch::full({m, maxc}, INFINITY, qs.options());
    auto nout = torch::empty({m}, bucket.options());
    auto reads = torch::zeros({1}, bucket.options().dtype(torch::kInt64));
    bool global = HT > 32768;
    torch::Tensor gt;
    if (global) gt = torch::full({m, HT}, -1, bucket.options());
    size_t smem = (global ? 0 : HT * sizeof(int)) + 2 * S * sizeof(float);
    cudaFuncSetAttribute(gather_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gather_kernel<<<m, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        bucket.data_ptr<int>(), off.data_ptr<int>(), ids.data_ptr<int>(), (const signed char*)code.data_ptr<int8_t>(),
        scale.data_ptr<float>(), qs.data_ptr<float>(), C, off.size(1), ids.size(1), S, (int)cap, (int)HT, (int)maxc,
        cand.data_ptr<int>(), dist.data_ptr<float>(), nout.data_ptr<int>(),
        (unsigned long long*)reads.data_ptr<int64_t>(), global ? gt.data_ptr<int>() : nullptr);
    return {cand, dist, nout, reads};
}
__global__ void sketch_merge_kernel(const int* __restrict__ cand, const float* __restrict__ dist,
                                    const int* __restrict__ nin, int W, int KO, long long* __restrict__ si,
                                    float* __restrict__ sd) {
    // sketch_merge in one kernel per query: radix-select the KO sketch-nearest of this round's n candidates, then a
    // bitonic sort of the (distance, id) keys of the kept KO and the selected ones; the same id carries the same
    // sketch distance, so duplicates are equal keys and collapse
    const int q = blockIdx.x;
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, kk;
    __shared__ int nsel, ntie;
    __shared__ int sid[1024];
    __shared__ float sdv[1024];
    __shared__ unsigned long long key[2048];
    const int n = min(nin[q], W);
    const unsigned* kd = reinterpret_cast<const unsigned*>(dist + (long)q * W);
    const int* cid = cand + (long)q * W;
    if (threadIdx.x == 0) { prefix = 0; kk = min(KO, n); nsel = 0; ntie = 0; }
    __syncthreads();
    unsigned mask = 0;
    if (n > KO) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
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
                for (; d < 256; d++) { if (cum + hist[d] >= k) break; cum += hist[d]; }
                kk = k - cum;
                prefix = pf | ((unsigned)d << shift);
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        unsigned T = prefix, ties = kk;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned k = kd[i];
            if (k < T) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; sdv[s] = __uint_as_float(k); }
            else if (k == T) {
                int t = atomicAdd(&ntie, 1);
                if (t < (int)ties) { int s = atomicAdd(&nsel, 1); sid[s] = cid[i]; sdv[s] = __uint_as_float(k); }
            }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) { sid[i] = cid[i]; sdv[i] = __uint_as_float(kd[i]); }
        if (threadIdx.x == 0) nsel = n;
    }
    __syncthreads();
    const int ns = nsel;
    long long* pb = si + (long)q * KO;
    float* pd = sd + (long)q * KO;
    int P = 2;
    while (P < KO + ns) P <<= 1;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        unsigned long long k = ~0ULL;
        if (i < KO) {
            if (pb[i] >= 0) k = ((unsigned long long)__float_as_uint(pd[i]) << 32) | (unsigned)pb[i];
        } else if (i - KO < ns) {
            k = ((unsigned long long)__float_as_uint(sdv[i - KO]) << 32) | (unsigned)sid[i - KO];
        }
        key[i] = k;
    }
    __syncthreads();
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < P; i += blockDim.x) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = (i & size) == 0;
                    unsigned long long a = key[i], b = key[j];
                    if ((a > b) == up) { key[i] = b; key[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        int o = 0;
        unsigned long long prev = ~0ULL - 1;
        for (int i = 0; i < P && o < KO; i++) {
            unsigned long long k = key[i];
            if (k == ~0ULL) break;
            if (k == prev) continue;
            prev = k;
            pb[o] = (long long)(unsigned)(k & 0xffffffffULL);
            pd[o] = __uint_as_float((unsigned)(k >> 32));
            o++;
        }
        for (; o < KO; o++) { pb[o] = -1; pd[o] = INFINITY; }
    }
}

void sketch_merge(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor si, torch::Tensor sd) {
    const c10::cuda::OptionalCUDAGuard device_guard_(cand.device());
    int m = cand.size(0), W = cand.size(1), KO = si.size(1);
    TORCH_CHECK(KO <= 1024 && si.size(0) == m);
    TORCH_CHECK(cand.is_contiguous() && dist.is_contiguous() && si.is_contiguous() && sd.is_contiguous());
    sketch_merge_kernel<<<m, 512, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cand.data_ptr<int>(), dist.data_ptr<float>(), n.data_ptr<int>(), W, KO, (long long*)si.data_ptr<int64_t>(),
        sd.data_ptr<float>());
}

__global__ void pick_starts_kernel(const long long* __restrict__ si, int KO, long long* __restrict__ done, int E,
                                   int off, int S, const int* __restrict__ rowlen, int* __restrict__ seg2) {
    // the start points of a wave: the first S entries of the kept list si (best first) that no earlier wave of this
    // query expanded (done[:, :off]), -1 when fewer remain; written to done[:, off:off + S] and to the gather
    // segments seg2 [S, 2] = (row, length); a hash set of the expanded ids in shared memory replaces a sort
    const int q = blockIdx.x;
    __shared__ int ht[8192];
    __shared__ unsigned char fresh[1024];
    const long long* ps = si + (long)q * KO;
    long long* pd = done + (long)q * E;
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) ht[i] = -1;
    __syncthreads();
    for (int i = threadIdx.x; i < off; i += blockDim.x) {
        long long v = pd[i];
        if (v < 0) continue;
        unsigned h = ((unsigned)v * 2654435761u) & 8191u;
        while (true) {
            int prev = atomicCAS(&ht[h], -1, (int)v);
            if (prev == -1 || prev == (int)v) break;
            h = (h + 1) & 8191u;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < KO; i += blockDim.x) {
        long long v = ps[i];
        bool f = v >= 0;
        if (f) {
            unsigned h = ((unsigned)v * 2654435761u) & 8191u;
            while (true) {
                int t = ht[h];
                if (t == -1) break;
                if (t == (int)v) { f = false; break; }
                h = (h + 1) & 8191u;
            }
        }
        fresh[i] = f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0;
        for (int i = 0; i < KO && o < S; i++) {
            if (!fresh[i]) continue;
            long long v = ps[i];
            pd[off + o] = v;
            seg2[((long)q * S + o) * 2] = (int)v;
            seg2[((long)q * S + o) * 2 + 1] = rowlen[o];
            o++;
        }
        for (; o < S; o++) {
            pd[off + o] = -1;
            seg2[((long)q * S + o) * 2] = 0;
            seg2[((long)q * S + o) * 2 + 1] = 0;
        }
    }
}

void pick_starts(torch::Tensor si, torch::Tensor done, int64_t off, int64_t S, torch::Tensor rowlen, torch::Tensor seg2) {
    const c10::cuda::OptionalCUDAGuard device_guard_(si.device());
    int m = si.size(0), KO = si.size(1), E = done.size(1);
    TORCH_CHECK(KO <= 1024 && off <= 4096 && off + S <= E && seg2.size(1) == S);
    TORCH_CHECK(si.is_contiguous() && done.is_contiguous() && seg2.is_contiguous() && rowlen.is_contiguous());
    pick_starts_kernel<<<m, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const long long*)si.data_ptr<int64_t>(), KO, (long long*)done.data_ptr<int64_t>(), E, (int)off, (int)S,
        rowlen.data_ptr<int>(), seg2.data_ptr<int>());
}

"""

CPP_SRC = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> decode2(torch::Tensor Y, int64_t b, int64_t logT, torch::Tensor off, int64_t cap);
std::vector<torch::Tensor> gather2(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale,
                                   torch::Tensor qs, int64_t HT, int64_t maxc, int64_t threads);
torch::Tensor decode(torch::Tensor Y, int64_t b, int64_t logT);
std::vector<torch::Tensor> gather(torch::Tensor bucket, torch::Tensor off, torch::Tensor ids, torch::Tensor code,
                                  torch::Tensor scale, torch::Tensor qs, int64_t cap, int64_t HT, int64_t maxc,
                                  int64_t threads);
void select_rerank(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor X, torch::Tensor Q,
                   int64_t R, torch::Tensor bi, torch::Tensor bd);
std::vector<torch::Tensor> decode3(torch::Tensor Y, int64_t b, int64_t logT, torch::Tensor off, int64_t cap);
std::vector<torch::Tensor> gather3(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale,
                                   torch::Tensor qs, torch::Tensor hts, int64_t HTmax, int64_t threads);
std::vector<torch::Tensor> gather4(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale,
                                   torch::Tensor qs, torch::Tensor hts, int64_t HTmax, int64_t threads);
void decode4(torch::Tensor Y, torch::Tensor active, int64_t b, int64_t logT, torch::Tensor off, int64_t cap,
             torch::Tensor seg, torch::Tensor tot);
void gather5(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
             torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads, bool shared,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor gtable);
void gather9(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
             torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor cptr, int64_t crows);
void select_rerank2(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                    torch::Tensor X, torch::Tensor Q, int64_t R, torch::Tensor bi, torch::Tensor bd);
void gather_global(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
                   torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t split, int64_t threads,
                   torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor gtable,
                   torch::Tensor cnt);
void gather6(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
             torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads, int64_t tau,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout);
void gather_global2(torch::Tensor seg, torch::Tensor ids, torch::Tensor code, torch::Tensor scale, torch::Tensor qs,
                    torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t split, int64_t threads,
                    int64_t tau, torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor gtable,
                    torch::Tensor cnt);
void probe(torch::Tensor Y, int64_t b, torch::Tensor inv_f, torch::Tensor off, torch::Tensor keys, torch::Tensor base,
           int64_t l, int64_t lr, int64_t logT, torch::Tensor seg, torch::Tensor tot, torch::Tensor lvl);
void gather7(torch::Tensor seg, torch::Tensor table, int64_t rowlen, torch::Tensor code, torch::Tensor scale,
             torch::Tensor qs, torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W, int64_t threads,
             torch::Tensor cand, torch::Tensor dist, torch::Tensor nout);
void select_rerank3(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                    torch::Tensor X, torch::Tensor Q, int64_t R, torch::Tensor bi, torch::Tensor bd);
void gather8(torch::Tensor seg, torch::Tensor tptr, int64_t rows_per, int64_t rowlen, torch::Tensor code,
             torch::Tensor scale, torch::Tensor qs, torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W,
             int64_t threads, torch::Tensor cand, torch::Tensor dist, torch::Tensor nout);
void peer_gather(torch::Tensor pi, torch::Tensor pd, int64_t lo, torch::Tensor oi, torch::Tensor od);
void gather10(torch::Tensor seg, torch::Tensor tptr, int64_t rows_per, int64_t rowlen, torch::Tensor code,
              torch::Tensor scale, torch::Tensor qs, torch::Tensor hts, torch::Tensor rowof, int64_t rows, int64_t W,
              int64_t threads, torch::Tensor cand, torch::Tensor dist, torch::Tensor nout, torch::Tensor cptr,
              int64_t crows);
void select_rerank3s(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                     torch::Tensor X, torch::Tensor xptr, int64_t xrows, torch::Tensor Q, int64_t R, torch::Tensor bi,
                     torch::Tensor bd);
void select_rerank2s(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor rowof, int64_t rows,
                     torch::Tensor X, torch::Tensor xptr, int64_t xrows, torch::Tensor Q, int64_t R, torch::Tensor bi,
                     torch::Tensor bd);
void sketch_merge(torch::Tensor cand, torch::Tensor dist, torch::Tensor n, torch::Tensor si, torch::Tensor sd);
void pick_starts(torch::Tensor si, torch::Tensor done, int64_t off, int64_t S, torch::Tensor rowlen, torch::Tensor seg2);
"""

_ext = None


def enable_peer_access():
    """Let kernels on every visible GPU read every other GPU's memory (needed for tables shared by CUDA IPC)."""
    import ctypes
    import glob
    import torch
    lib = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib",
                                 "libcudart.so*"))
    rt = ctypes.CDLL(lib[0] if lib else "libcudart.so")
    cur = torch.cuda.current_device()
    n = torch.cuda.device_count()
    for a in range(n):
        torch.cuda.set_device(a)  # switch through torch, so torch's notion of the current device stays true
        torch.zeros(1, device=f"cuda:{a}")
        for b in range(n):
            if a != b:
                rt.cudaDeviceEnablePeerAccess(b, 0)  # 704 (already enabled) is fine
    torch.cuda.set_device(cur)


def ext():
    global _ext
    if _ext is None:
        _ext = load_inline("proto_cuda", CPP_SRC, cuda_sources=CUDA_SRC, functions=["decode", "gather", "decode2", "gather2", "select_rerank", "decode3", "gather3", "gather4", "decode4", "gather5", "select_rerank2", "gather_global", "gather6", "gather_global2", "probe", "gather7", "select_rerank3", "gather8", "peer_gather", "select_rerank3s", "select_rerank2s", "gather9", "gather10", "sketch_merge", "pick_starts"],
                           extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_%d%d" % torch.cuda.get_device_capability()],
                           build_directory=_build_dir(), verbose=False)
    return _ext


def _build_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_cuda")
    if torch.cuda.get_device_capability() != (9, 0):  # one build per architecture (sm_90 keeps the old directory)
        d += "_sm%d%d" % torch.cuda.get_device_capability()
    os.makedirs(d, exist_ok=True)
    return d
