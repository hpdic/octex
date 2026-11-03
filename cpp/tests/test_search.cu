/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Search kernels on small random data against CPU references, the ablation switches of the search
// against the optimized search on a small synthetic index, then an end-to-end recall check on an
// index directory (OCTEX_TEST_INDEX, default /home/cc/data/octex/sift1m; skipped when absent).

#include <cuvs/core/cuda_fp16.hpp>

#include "neighbors/octex/detail/npy.hpp"
#include "neighbors/octex/detail/search_kernels.cuh"

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace cuvs::neighbors::octex::detail;

namespace {

int failures = 0;

#define EXPECT(c, ...)                                 \
  do {                                                 \
    if (!(c)) {                                        \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                        \
      std::printf("\n");                               \
      failures++;                                      \
    }                                                  \
  } while (0)

#define CK(x)                                                                     \
  do {                                                                            \
    cudaError_t e_ = (x);                                                         \
    if (e_ != cudaSuccess) {                                                      \
      std::printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

template <typename T>
auto to_dev(std::vector<T> const& h) -> T*
{
  T* d = nullptr;
  CK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
  if (!h.empty()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
auto to_host(const T* d, size_t n) -> std::vector<T>
{
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

std::mt19937 rng(7);

// ---- probe: the E8 cell of a projection is found through the bucket CSR and the key array

__global__ void hash_kernel(const float* y, int n, int b, float inv_f, long long* out)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = e8b_hash64(y + t * 8 * b, b, inv_f);
}

void test_probe()
{
  const int m = 7, C = 3, L = 2, b = 4, logT = 4, l = 5, lr = 3, w = 8 * b;
  const int64_t T1           = (1 << logT) + 1;
  std::vector<float> spacing = {1.0f, 4.0f};
  std::normal_distribution<float> nd(0.f, 2.f);
  std::vector<float> y(m * C * w);
  for (auto& v : y)
    v = __half2float(__float2half(nd(rng)));
  std::vector<__half> yh(y.size());
  for (size_t i = 0; i < y.size(); i++)
    yh[i] = __float2half(y[i]);
  // hashes of every (query, copy, level) on the device, as the builder sees them
  std::vector<long long> h(m * C * L);
  float* dy = to_dev(y);
  long long* dh;
  CK(cudaMalloc(&dh, m * C * sizeof(long long)));
  for (int j = 0; j < L; j++) {
    hash_kernel<<<1, 64>>>(dy, m * C, b, 1.f / spacing[j], dh);
    auto hj = to_host(dh, m * C);
    for (int t = 0; t < m * C; t++)
      h[t * L + j] = hj[t];
  }
  // index: per copy and level, the cells of every second query (fine level) and all queries (coarse
  // level)
  int64_t Mmax = m;
  std::vector<long long> keys(C * L * Mmax, 0);
  std::vector<int> off(C * L * T1, 0), base(C * L);
  std::map<std::tuple<int, int, long long>, int> row;  // (copy, level, hash) -> cell row
  int rows = 0;
  for (int c = 0; c < C; c++)
    for (int j = 0; j < L; j++) {
      base[c * L + j] = rows;
      std::set<std::pair<int, long long>> cells;  // (bucket, hash)
      for (int q = 0; q < m; q++)
        if (j == 1 || q % 2 == 0) {
          long long hh = h[(q * C + c) * L + j];
          cells.insert({static_cast<int>(hh & ((1LL << logT) - 1)), hh});
        }
      int p = 0;
      std::vector<int> cnt(1 << logT, 0);
      for (auto [bk, hh] : cells) {
        keys[(c * L + j) * Mmax + p] = hh;
        row[{c, j, hh}]              = rows + p;
        cnt[bk]++;
        p++;
      }
      for (int t = 0; t < (1 << logT); t++)
        off[(c * L + j) * T1 + t + 1] = off[(c * L + j) * T1 + t] + cnt[t];
      rows += p;
    }
  std::vector<float> inv = {1.f / spacing[0], 1.f / spacing[1]};
  int2* dseg;
  CK(cudaMalloc(&dseg, m * C * sizeof(int2)));
  auto* dyh = to_dev(yh);
  probe_kernel<4><<<1, 64>>>(dyh,
                             m,
                             C,
                             L,
                             to_dev(inv),
                             to_dev(off),
                             T1,
                             to_dev(keys),
                             Mmax,
                             to_dev(base),
                             l,
                             lr,
                             logT,
                             dseg);
  auto seg = to_host(dseg, m * C);
  for (int q = 0; q < m; q++)
    for (int c = 0; c < C; c++) {
      int j    = q % 2 == 0 ? 0 : 1;
      int want = row[{c, j, h[(q * C + c) * L + j]}] * l;
      EXPECT(seg[q * C + c].x == want && seg[q * C + c].y == lr,
             "probe q=%d c=%d: (%d, %d), expected (%d, %d)",
             q,
             c,
             seg[q * C + c].x,
             seg[q * C + c].y,
             want,
             lr);
    }
  std::printf("probe: done\n");
}

// ---- gather: distinct ids of all segments with sketch distances

float sketch_dist(
  const std::vector<int8_t>& code, const std::vector<float>& sc, const float* qs, int id, int S)
{
  float d = 0.f;
  for (int k = 0; k < S; k++) {
    float u = code[static_cast<size_t>(id) * S + k] * sc[k] - qs[k];
    d += u * u;
  }
  return d;
}

void test_gather(bool rows, bool split_sketch)
{
  const int m = 5, C = 37, S = 64, N = 3000, stride = 400, rowlen = 24;
  std::uniform_int_distribution<int> idd(0, N - 1), cd(-127, 127);
  std::vector<int8_t> code(static_cast<size_t>(N) * S);
  for (auto& v : code)
    v = static_cast<int8_t>(cd(rng));
  std::vector<float> sc(S), qs(m * S);
  for (auto& v : sc)
    v = 0.01f + 0.001f * (rng() % 50);
  for (auto& v : qs)
    v = (static_cast<int>(rng() % 200) - 100) * 0.01f;
  std::vector<int> lists(C * stride), table(static_cast<size_t>(N) * rowlen);
  for (auto& v : lists)
    v = idd(rng) % 700;  // many duplicates
  for (auto& v : table)
    v = idd(rng) % 900;
  for (int i = 0; i < 50; i++)
    table[rng() % table.size()] = -1;  // padding is skipped
  std::vector<int2> seg(m * C);
  for (int q = 0; q < m; q++)
    for (int c = 0; c < C; c++) {
      int len        = rng() % 3 == 0 ? 0 : 1 + rng() % (rows ? rowlen : 32);
      seg[q * C + c] = rows ? make_int2(idd(rng), len) : make_int2(rng() % (stride - 32), len);
    }
  int total_max = 0;
  for (int q = 0; q < m; q++) {
    int t = 0;
    for (int c = 0; c < C; c++)
      t += seg[q * C + c].y;
    total_max = std::max(total_max, t);
  }
  int logW = std::max(kMinLogHash, static_cast<int>(std::ceil(std::log2(total_max * 1.5)))),
      W    = 1 << logW;
  // table split in two row ranges behind pointers
  int rows_per = (N + 1) / 2;
  int* t0 = to_dev(std::vector<int>(table.begin(), table.begin() + int64_t(rows_per) * rowlen));
  int* t1 = to_dev(std::vector<int>(table.begin() + int64_t(rows_per) * rowlen, table.end()));
  long long* tptr = to_dev(
    std::vector<long long>{reinterpret_cast<long long>(t0), reinterpret_cast<long long>(t1)});
  int *dcand, *dn;
  float* ddist;
  CK(cudaMalloc(&dcand, m * W * sizeof(int)));
  CK(cudaMalloc(&ddist, m * W * sizeof(float)));
  CK(cudaMalloc(&dn, m * sizeof(int)));
  size_t smem = (W + 2 * S + 2 * C + 1) * sizeof(int);
  auto kern   = rows ? gather_kernel<true> : gather_kernel<false>;
  CK(cudaFuncSetAttribute(
    kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  // the sketch whole, or split in row ranges behind pointers (sharded or host sketch)
  const int crows = 1100;
  std::vector<long long> cp;
  for (int r0 = 0; r0 < N; r0 += crows)
    cp.push_back(reinterpret_cast<long long>(to_dev(std::vector<int8_t>(
      code.begin() + int64_t(r0) * S, code.begin() + int64_t(std::min(N, r0 + crows)) * S))));
  kern<<<m, 256, smem>>>(to_dev(seg),
                         C,
                         to_dev(lists),
                         rows ? rowlen : stride,
                         tptr,
                         rows_per,
                         split_sketch ? nullptr : to_dev(code),
                         split_sketch ? to_dev(cp) : nullptr,
                         split_sketch ? crows : 1,
                         to_dev(sc),
                         to_dev(qs),
                         S,
                         W,
                         logW,
                         dcand,
                         ddist,
                         dn);
  CK(cudaDeviceSynchronize());
  auto cand = to_host(dcand, m * W);
  auto dist = to_host(ddist, m * W);
  auto n    = to_host(dn, m);
  for (int q = 0; q < m; q++) {
    std::set<int> want;
    for (int c = 0; c < C; c++) {
      auto sg = seg[q * C + c];
      for (int i = 0; i < sg.y; i++) {
        int v = rows ? table[int64_t(sg.x) * rowlen + i] : lists[c * stride + sg.x + i];
        if (v >= 0) want.insert(v);
      }
    }
    std::set<int> got(cand.begin() + q * W, cand.begin() + q * W + n[q]);
    EXPECT(n[q] == static_cast<int>(want.size()) && got == want,
           "gather(%s) q=%d: %d ids, expected %zu",
           rows ? "rows" : "lists",
           q,
           n[q],
           want.size());
    for (int j = 0; j < n[q]; j++) {
      float e = sketch_dist(code, sc, qs.data() + q * S, cand[q * W + j], S);
      EXPECT(std::fabs(dist[q * W + j] - e) <= 1e-3f * (1.f + e),
             "gather sketch distance %f vs %f",
             dist[q * W + j],
             e);
    }
  }
  std::printf("gather (%s, %s sketch): done\n",
              rows ? "table rows" : "lists",
              split_sketch ? "split" : "whole");
}

// ---- rerank: exact rerank of the R sketch-nearest merged into the kept KO; sketch merge; final
// rerank

void test_rerank()
{
  const int m = 4, W = 2048, D = 32, N = 5000, R = 64, KO = 40;
  std::vector<__half> X(static_cast<size_t>(N) * D);
  std::vector<float> Xf(X.size()), Q(m * D);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (size_t i = 0; i < X.size(); i++) {
    X[i]  = __float2half(nd(rng));
    Xf[i] = __half2float(X[i]);
  }
  for (auto& v : Q)
    v = nd(rng);
  auto exact = [&](int q, int id) {
    double d = 0;
    for (int k = 0; k < D; k++) {
      double u = Xf[static_cast<size_t>(id) * D + k] - Q[q * D + k];
      d += u * u;
    }
    return d;
  };
  std::vector<int> cand(m * W), n(m);
  std::vector<float> dist(m * W);
  std::vector<long long> bi(m * KO, -1);
  std::vector<float> bd(m * KO, INFINITY);
  for (int q = 0; q < m; q++) {
    n[q] = 300 + 400 * q;
    std::vector<int> ids(N);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int> rank(n[q]);
    std::iota(rank.begin(), rank.end(), 1);
    std::shuffle(rank.begin(), rank.end(), rng);
    for (int j = 0; j < n[q]; j++) {
      cand[q * W + j] = ids[j];
      dist[q * W + j] = static_cast<float>(rank[j]) * 0.01f;  // distinct sketch distances
    }
    // a kept list of other ids (the same id always carries the same distance in a search)
    std::vector<std::pair<double, int>> kept;
    for (int j = 0; j < KO / 2; j++)
      kept.push_back({exact(q, ids[n[q] + j]), ids[n[q] + j]});
    std::sort(kept.begin(), kept.end());
    for (size_t j = 0; j < kept.size(); j++) {
      bi[q * KO + j] = kept[j].second;
      bd[q * KO + j] = static_cast<float>(kept[j].first);
    }
  }
  long long* xptr = to_dev(std::vector<long long>{reinterpret_cast<long long>(to_dev(X))});
  auto* dbi       = to_dev(bi);
  auto* dbd       = to_dev(bd);
  rerank_keep_kernel<<<m, 256, D * sizeof(float)>>>(
    to_dev(cand), to_dev(dist), to_dev(n), W, xptr, N, D, to_dev(Q), R, KO, dbi, dbd, false);
  auto gi = to_host(dbi, m * KO);
  auto gd = to_host(dbd, m * KO);
  for (int q = 0; q < m; q++) {
    // reference: R sketch-nearest of the candidates, exact distances, merged with the kept list,
    // best KO distinct
    std::vector<std::pair<float, int>> sk;
    for (int j = 0; j < n[q]; j++)
      sk.push_back({dist[q * W + j], cand[q * W + j]});
    std::sort(sk.begin(), sk.end());
    std::map<int, double> pool;
    for (int j = 0; j < std::min(R, n[q]); j++)
      pool[sk[j].second] = exact(q, sk[j].second);
    for (int j = 0; j < KO; j++)
      if (bi[q * KO + j] >= 0)
        pool[static_cast<int>(bi[q * KO + j])] = exact(q, static_cast<int>(bi[q * KO + j]));
    std::vector<std::pair<double, int>> all;
    for (auto [id, d] : pool)
      all.push_back({d, id});
    std::sort(all.begin(), all.end());
    for (int j = 0; j < KO; j++) {
      long long want = j < static_cast<int>(all.size()) ? all[j].second : -1;
      EXPECT(gi[q * KO + j] == want,
             "rerank_keep q=%d j=%d: %lld, expected %lld",
             q,
             j,
             gi[q * KO + j],
             want);
      if (want >= 0)
        EXPECT(std::fabs(gd[q * KO + j] - all[j].first) <= 1e-3 * (1 + all[j].first),
               "rerank distance");
    }
  }

  // sketch merge: the KO sketch-nearest, by sketch distance, into an empty list
  sketch_keep_kernel<<<m, 512>>>(to_dev(cand), to_dev(dist), to_dev(n), W, KO, dbi, dbd, true);
  gi = to_host(dbi, m * KO);
  gd = to_host(dbd, m * KO);
  for (int q = 0; q < m; q++) {
    std::vector<std::pair<float, int>> sk;
    for (int j = 0; j < n[q]; j++)
      sk.push_back({dist[q * W + j], cand[q * W + j]});
    std::sort(sk.begin(), sk.end());
    for (int j = 0; j < KO; j++)
      EXPECT(gd[q * KO + j] == sk[j].first,
             "sketch_keep q=%d j=%d: %f, expected %f",
             q,
             j,
             gd[q * KO + j],
             sk[j].first);
  }

  // final rerank of the first 24 kept entries into the best 10
  const int k = 10, FR = 24;
  long long* fi;
  float* fd;
  CK(cudaMalloc(&fi, m * k * sizeof(long long)));
  CK(cudaMalloc(&fd, m * k * sizeof(float)));
  final_rerank_kernel<<<m, 256, D * sizeof(float)>>>(dbi, KO, FR, xptr, N, D, to_dev(Q), k, fi, fd);
  auto ri = to_host(fi, m * k);
  for (int q = 0; q < m; q++) {
    std::vector<std::pair<double, int>> all;
    for (int j = 0; j < FR; j++)
      all.push_back({exact(q, static_cast<int>(gi[q * KO + j])), static_cast<int>(gi[q * KO + j])});
    std::sort(all.begin(), all.end());
    for (int j = 0; j < k; j++)
      EXPECT(ri[q * k + j] == all[j].second, "final_rerank q=%d j=%d", q, j);
  }
  std::printf("rerank, sketch merge, final rerank: done\n");
}

// ---- merge of the kept lists of several sources: distinct ids, best KO

void test_merge()
{
  const int m = 6, KO = 50, ng = 3, row0 = 2, rows = m + row0;
  std::vector<std::vector<long long>> si(ng, std::vector<long long>(rows * KO));
  std::vector<std::vector<float>> sd(ng, std::vector<float>(rows * KO));
  auto dist_of = [](int q, int id) {
    return static_cast<float>((id * 7919 + q * 31) % 1000) + 0.5f;
  };
  for (int h = 0; h < ng; h++)
    for (int q = 0; q < rows; q++) {
      int cnt = KO - 5 * h;  // shorter lists end in -1
      std::vector<std::pair<float, int>> l;
      std::set<int> used;
      while (static_cast<int>(l.size()) < cnt) {
        int id = rng() % 200;
        if (used.insert(id).second) l.push_back({dist_of(q, id), id});
      }
      std::sort(l.begin(), l.end());
      for (int j = 0; j < KO; j++) {
        si[h][q * KO + j] = j < cnt ? l[j].second : -1;
        sd[h][q * KO + j] = j < cnt ? l[j].first : INFINITY;
      }
    }
  std::vector<long long> pi, pd;
  for (int h = 0; h < ng; h++) {
    pi.push_back(reinterpret_cast<long long>(to_dev(si[h])));
    pd.push_back(reinterpret_cast<long long>(to_dev(sd[h])));
  }
  int P = 2;
  while (P < ng * KO)
    P <<= 1;
  long long* oi;
  float* od;
  CK(cudaMalloc(&oi, m * KO * sizeof(long long)));
  CK(cudaMalloc(&od, m * KO * sizeof(float)));
  merge_kept_kernel<<<m, 512, P * sizeof(uint64_t)>>>(
    to_dev(pi), to_dev(pd), ng, row0, KO, P, oi, od);
  auto gi = to_host(oi, m * KO);
  for (int q = 0; q < m; q++) {
    std::map<int, float> best;
    for (int h = 0; h < ng; h++)
      for (int j = 0; j < KO; j++) {
        long long v = si[h][(row0 + q) * KO + j];
        if (v >= 0) best[static_cast<int>(v)] = sd[h][(row0 + q) * KO + j];
      }
    std::vector<std::pair<float, int>> all;
    for (auto [id, d] : best)
      all.push_back({d, id});
    std::sort(all.begin(), all.end());
    for (int j = 0; j < KO; j++) {
      long long want = j < static_cast<int>(all.size()) ? all[j].second : -1;
      EXPECT(
        gi[q * KO + j] == want, "merge q=%d j=%d: %lld, expected %lld", q, j, gi[q * KO + j], want);
    }
  }
  std::printf("merge of kept lists: done\n");
}

// ---- pick starts: unexpanded kept entries in order

void test_pick_starts()
{
  const int m = 3, KO = 30, S = 8, waves = 3, E = S * waves, rowlen = 16;
  std::vector<long long> si(m * KO), done(m * E, -1);
  for (int q = 0; q < m; q++)
    for (int j = 0; j < KO; j++)
      si[q * KO + j] = j < KO - 4 ? 100 * q + j * 3 : -1;
  for (int q = 0; q < m; q++)
    for (int j = 0; j < S; j++)
      done[q * E + j] = 100 * q + j * 6;  // every second kept entry expanded already
  auto* ddone = to_dev(done);
  int2* seg2;
  CK(cudaMalloc(&seg2, m * S * sizeof(int2)));
  pick_starts_kernel<<<m, 256>>>(to_dev(si), KO, ddone, E, S, S, rowlen, seg2);
  auto g  = to_host(ddone, m * E);
  auto sg = to_host(seg2, m * S);
  for (int q = 0; q < m; q++) {
    std::set<long long> seen(done.begin() + q * E, done.begin() + q * E + S);
    std::vector<long long> want;
    for (int j = 0; j < KO && static_cast<int>(want.size()) < S; j++)
      if (si[q * KO + j] >= 0 && !seen.count(si[q * KO + j])) want.push_back(si[q * KO + j]);
    for (int j = 0; j < S; j++) {
      long long w = j < static_cast<int>(want.size()) ? want[j] : -1;
      EXPECT(g[q * E + S + j] == w, "pick_starts q=%d j=%d", q, j);
      EXPECT(sg[q * S + j].y == (w >= 0 ? rowlen : 0), "pick_starts segment length");
    }
  }
  std::printf("pick starts: done\n");
}

// ---- ablations: every combination of the switches returns the neighbors of the optimized search,
// on a small synthetic index (replicated on one GPU; sharded over two when present)

void test_ablation()
{
  namespace ox    = cuvs::neighbors::octex;
  const int64_t n = 20000, d = 48, nq = 600;
  const int k = 10;
  std::mt19937 gen(11);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> x(n * d), qv(nq * d);
  for (int64_t i = 0; i < n; i++) {
    float c = float(i % 37);  // clusters
    for (int64_t j = 0; j < d; j++)
      x[i * d + j] = g(gen) + (j < 4 ? c : 0.f);
  }
  for (int64_t i = 0; i < nq; i++)  // queries near data points
    for (int64_t j = 0; j < d; j++)
      qv[i * d + j] = x[(i * 31 % n) * d + j] + 0.3f * g(gen);
  std::string dir = "/home/cc/tmp/octex_test_search_ablation";
  {
    ox::index_params p;
    p.copies       = 8;
    p.blocks       = 2;
    p.pca_dims     = 32;
    p.list_len     = 16;
    p.log_buckets  = 12;
    p.sketch_width = 32;
    p.table_len    = 16;
    raft::resources res;
    ox::build(res, p, raft::make_host_matrix_view<const float, int64_t>(x.data(), n, d), dir);
  }
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  for (int shard = 0; shard < (ndev > 1 ? 2 : 1); shard++) {
    raft::resources res;
    ox::load_params lp;
    lp.devices = shard ? std::vector<int>{0, 1} : std::vector<int>{0};
    lp.mode    = shard ? ox::placement::shard : ox::placement::replicate;
    auto idx   = ox::load(res, dir, lp);
    auto run   = [&](ox::search_params const& sp, std::vector<int64_t>& I, std::vector<float>& Dd) {
      I.assign(nq * k, -1);
      Dd.assign(nq * k, 0.f);
      ox::search(res,
                 sp,
                 idx,
                 raft::make_host_matrix_view<const float, int64_t>(qv.data(), nq, d),
                 raft::make_host_matrix_view<int64_t, int64_t>(I.data(), nq, k),
                 raft::make_host_matrix_view<float, int64_t>(Dd.data(), nq, k));
      CK(cudaDeviceSynchronize());
      CK(cudaGetLastError());
    };
    for (int sketch = 0; sketch < 2; sketch++) {
      ox::search_params base;
      base.read        = 8;
      base.starts      = 32;
      base.waves       = 2;
      base.rerank      = 64;
      base.wave_rerank = 64;
      base.batch       = 256;  // several batches, the last one partial
      base.sketch_rank = sketch == 1;
      std::vector<int64_t> I0, I;
      std::vector<float> D0, Dd;
      run(base, I0, D0);
      for (int v = 1; v < 8; v++) {
        auto sp            = base;
        sp.balanced_gather = (v & 1) == 0;
        sp.fused_merge     = (v & 2) == 0;
        sp.use_graph       = (v & 4) == 0;
        run(sp, I, Dd);
        int64_t diff = 0, ddiff = 0;
        for (int64_t i = 0; i < nq * k; i++) {
          diff += I[i] != I0[i];
          ddiff += std::fabs(Dd[i] - D0[i]) > 1e-4f * (1.f + std::fabs(D0[i]));
        }
        // the same neighbors up to the order among equal sketch distances, which the fused
        // selection breaks in any order
        EXPECT(diff <= nq * k / 1000 && ddiff <= nq * k / 1000,
               "ablation (%s, %s, balanced_gather=%d fused_merge=%d use_graph=%d): %ld ids and %ld "
               "distances differ",
               shard ? "shard" : "replicate",
               sketch ? "sketch rank" : "exact rank",
               sp.balanced_gather,
               sp.fused_merge,
               sp.use_graph,
               static_cast<long>(diff),
               static_cast<long>(ddiff));
      }
    }
    std::printf("ablations (%s): done\n", shard ? "shard over 2 GPUs" : "replicate");
  }
}

// ---- end to end: recall@10 on an index directory

void test_end_to_end()
{
  const char* env  = std::getenv("OCTEX_TEST_INDEX");
  std::string path = env ? env : "/home/cc/data/octex/sift1m";
  struct stat st{};
  if (::stat((path + "/meta.json").c_str(), &st) != 0) {
    std::printf("end to end: skipped (no index at %s)\n", path.c_str());
    return;
  }
  namespace ox = cuvs::neighbors::octex;
  npy_file qf(path + "/queries.npy"), gf(path + "/groundtruth.npy");
  const int64_t nq = std::min<int64_t>(2000, qf.shape[0]), D = qf.shape[1], gk = gf.shape[1];
  raft::resources res;
  ox::load_params lp;
  lp.devices = {0};
  lp.mode    = ox::placement::replicate;
  // 50 copies: a quarter of the index keeps the test light
  lp.copies_limit = 50;
  auto idx        = ox::load(res, path, lp);
  ox::search_params sp;
  sp.read     = 8;
  sp.starts   = 32;
  sp.waves    = 1;
  sp.batch    = 777;  // several batches, the last one partial
  const int k = 10;
  std::vector<int64_t> I(nq * k);
  std::vector<float> Dd(nq * k);
  ox::search(res,
             sp,
             idx,
             raft::make_host_matrix_view<const float, int64_t>(qf.as<float>(), nq, D),
             raft::make_host_matrix_view<int64_t, int64_t>(I.data(), nq, k),
             raft::make_host_matrix_view<float, int64_t>(Dd.data(), nq, k));
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  int64_t hit = 0;
  auto g      = gf.as<int64_t>();
  for (int64_t q = 0; q < nq; q++)
    for (int i = 0; i < k; i++)
      for (int j = 0; j < 10; j++)
        hit += I[q * k + i] == g[q * gk + j];
  double recall = double(hit) / double(nq * 10);
  std::printf("end to end: 50 copies, read 8, starts 32, waves 1: recall@10 %.4f\n", recall);
  EXPECT(recall > 0.94, "end-to-end recall %.4f", recall);
  for (int64_t q = 0; q < nq; q++)
    for (int i = 1; i < k; i++)
      EXPECT(Dd[q * k + i - 1] <= Dd[q * k + i], "distances not sorted");
}

}  // namespace

int main()
{
  CK(cudaSetDevice(0));
  test_probe();
  test_gather(false, false);
  test_gather(true, false);
  test_gather(false, true);
  test_gather(true, true);
  test_rerank();
  test_merge();
  test_pick_starts();
  test_ablation();
  test_end_to_end();
  std::printf(failures == 0 ? "all passed\n" : "%d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
