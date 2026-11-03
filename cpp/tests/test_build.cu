/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests of the index build.
//   (a) E8 decoding: the device decoder returns a point of E8 that no Voronoi-relevant vector (the
//   240
//       roots) improves, and its 64-bit cell hash equals a CPU reference.
//   (b) a small build end to end: shapes of the directory, the neighbor table against CPU brute
//   force. (c) equivalence with a reference index directory (default /home/cc/data/octex/sift1m):
//   from its
//       mean, projections and shifts, the cells of the first copies (exact lists) must reproduce
//       keys, offsets, base and cells exactly and the lists up to distance ties.
//
//   test_build [reference dir] [copies to compare, default 2] [ivf: IVF-Flat lists, reports their
//   overlap]

#include <cuvs/neighbors/octex.hpp>

#include "neighbors/octex/detail/build_cells.hpp"
#include "neighbors/octex/detail/e8.cuh"
#include "neighbors/octex/detail/npy.hpp"

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace octex = cuvs::neighbors::octex;
using octex::detail::npy_file;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) failures++;
}

// ---- (a) E8 ----

__global__ void e8_kernel(const float* x, int n, int b, float inv_f, float* p, long long* h)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int j = 0; j < b; j++) {
    float xs[8];
    for (int k = 0; k < 8; k++)
      xs[k] = x[i * 8 * b + 8 * j + k] * inv_f;
    octex::detail::e8_point(xs, p + i * 8 * b + 8 * j);
  }
  h[i] = octex::detail::e8b_hash64(x + i * 8 * b, b, inv_f);
}

const long long kPrimesHost[8] = {1000003LL,
                                  998244353LL,
                                  1000000007LL,
                                  2147483647LL,
                                  1099511627791LL,
                                  4398042316799LL,
                                  70368744177643LL,
                                  1125899906842597LL};

auto hash_ref(const float* p, int b) -> long long
{
  unsigned long long h = 0, mix = 0x9E3779B97F4A7C15ULL;
  for (int j = 0; j < b; j++) {
    unsigned long long hb = 0;
    for (int k = 0; k < 8; k++)
      hb += (unsigned long long)(long long)std::lround(2.0 * p[8 * j + k]) *
            (unsigned long long)kPrimesHost[k];
    h = j == 0 ? hb : h * mix + hb;
  }
  long long s = (long long)h;
  s           = (long long)((unsigned long long)(s ^ (s >> 31)) * mix);
  return s ^ (s >> 29);
}

void test_e8()
{
  const int n = 200000, b = 4, w = 8 * b;
  const float f = 2.5f;
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> u(-6.f, 6.f);
  std::vector<float> x(size_t(n) * w);
  for (auto& v : x)
    v = u(gen);
  // some points exactly on half-integer ties
  for (int i = 0; i < 1000; i++)
    x[i] = std::round(x[i] * 2.f) * 0.5f;
  float *dx, *dp;
  long long* dh;
  cudaMalloc(&dx, x.size() * 4);
  cudaMalloc(&dp, x.size() * 4);
  cudaMalloc(&dh, n * 8);
  cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
  e8_kernel<<<(n + 255) / 256, 256>>>(dx, n, b, 1.f / f, dp, dh);
  std::vector<float> p(x.size());
  std::vector<long long> h(n);
  cudaMemcpy(p.data(), dp, p.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h.data(), dh, n * 8, cudaMemcpyDeviceToHost);
  cudaFree(dx);
  cudaFree(dp);
  cudaFree(dh);

  // the 240 roots of E8
  std::vector<std::array<double, 8>> roots;
  for (int i = 0; i < 8; i++)
    for (int j = i + 1; j < 8; j++)
      for (int si : {-1, 1})
        for (int sj : {-1, 1}) {
          std::array<double, 8> r{};
          r[i] = si;
          r[j] = sj;
          roots.push_back(r);
        }
  for (int m = 0; m < 256; m++)
    if (__builtin_popcount(m) % 2 == 0) {
      std::array<double, 8> r{};
      for (int k = 0; k < 8; k++)
        r[k] = (m >> k & 1) ? -0.5 : 0.5;
      roots.push_back(r);
    }
  int64_t bad_lattice = 0, bad_nearest = 0, bad_hash = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < b; j++) {
      const float* pp = p.data() + i * w + 8 * j;
      double xs[8], sum = 0;
      bool all_int = true, all_half = true;
      for (int k = 0; k < 8; k++) {
        xs[k] = double(x[i * w + 8 * j + k] * (1.f / f));
        sum += pp[k];
        double fr = pp[k] - std::floor(pp[k]);
        all_int &= fr == 0.0;
        all_half &= fr == 0.5;
      }
      if (!(all_int || all_half) || std::fmod(std::fabs(sum), 2.0) != 0.0) bad_lattice++;
      double d0 = 0;
      for (int k = 0; k < 8; k++)
        d0 += (xs[k] - pp[k]) * (xs[k] - pp[k]);
      for (auto const& r : roots) {
        double d1 = 0;
        for (int k = 0; k < 8; k++)
          d1 += (xs[k] - pp[k] - r[k]) * (xs[k] - pp[k] - r[k]);
        if (d1 < d0 - 1e-6) {
          bad_nearest++;
          break;
        }
      }
    }
    if (hash_ref(p.data() + i * w, b) != h[i]) bad_hash++;
  }
  std::printf("E8: %d points x %d blocks: not in E8 %lld, not nearest %lld, hash mismatches %lld\n",
              n,
              b,
              (long long)bad_lattice,
              (long long)bad_nearest,
              (long long)bad_hash);
  check(bad_lattice == 0 && bad_nearest == 0 && bad_hash == 0,
        "E8 decode and hash match the CPU reference");
}

// ---- (b) small build ----

void test_small_build()
{
  const int64_t n = 20000, d = 48;
  std::mt19937 gen(3);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> x(size_t(n) * d);
  for (int64_t i = 0; i < n; i++) {
    float c = float(i % 37);  // clusters
    for (int64_t k = 0; k < d; k++)
      x[i * d + k] = g(gen) + (k < 4 ? c : 0.f);
  }
  std::string dir = "/home/cc/tmp/octex_test_build_small";
  octex::index_params p;
  p.copies        = 6;
  p.blocks        = 2;
  p.pca_dims      = 32;
  p.list_len      = 16;
  p.min_cell      = 2;
  p.log_buckets   = 12;
  p.sketch_width  = 24;
  p.table_len     = 10;
  p.reverse_edges = false;
  p.exact_lists   = false;
  raft::device_resources res;
  octex::build(res, p, raft::make_host_matrix_view<const float, int64_t>(x.data(), n, d), dir);
  auto f  = [&](const char* name) { return (std::filesystem::path(dir) / name).string(); };
  bool ok = std::filesystem::exists(f("meta.json"));
  npy_file proj(f("proj.npy")), shift(f("shift.npy")), lists(f("lists.npy")), keys(f("keys.npy")),
    offs(f("offsets.npy")), base(f("base.npy")), cells(f("cells.npy")), sk(f("sketch.npy")),
    skb(f("sketch_basis.npy")), sks(f("sketch_scale.npy")), table(f("table.npy")),
    vec(f("vectors.npy")), mean(f("mean.npy"));
  ok &= proj.shape == std::vector<int64_t>{6, d, 16} && shift.shape == std::vector<int64_t>{6, 16};
  ok &= keys.shape.size() == 3 && keys.shape[0] == 6 && keys.shape[1] == 5;
  ok &= offs.shape == std::vector<int64_t>{6, 5, 4097} && base.shape == std::vector<int64_t>{6, 5};
  ok &= sk.shape == std::vector<int64_t>{n, 24} && skb.shape == std::vector<int64_t>{d, 24};
  ok &= sks.shape == std::vector<int64_t>{24} && table.shape == std::vector<int64_t>{n, 10};
  ok &= vec.shape == std::vector<int64_t>{n, d} && vec.descr == "<f2" &&
        mean.shape == std::vector<int64_t>{d};
  // CSR consistency: offsets end at the cell count, keys in their buckets
  const int64_t* kk = keys.as<int64_t>();
  const int32_t* oo = offs.as<int32_t>();
  const int64_t* cc = cells.as<int64_t>();
  for (int c = 0; c < 6; c++)
    for (int j = 0; j < 5; j++) {
      const int32_t* o = oo + (c * 5 + j) * 4097;
      ok &= o[4096] == cc[c * 5 + j];
      for (int t = 0; t < 4096; t++)
        for (int e = o[t]; e < o[t + 1]; e++)
          ok &= (kk[(c * 5 + j) * keys.shape[2] + e] & 4095) == t;
    }
  check(ok, "small build: shapes and CSR consistency");

  // the table against CPU brute force over the fp16 vectors (distance of the k-th neighbor)
  const half* xh   = vec.as<half>();
  const int32_t* t = table.as<int32_t>();
  int64_t worse = 0, total = 0;
  for (int64_t i = 0; i < n; i += 97) {
    std::vector<std::pair<double, int>> all;
    for (int64_t j = 0; j < n; j++) {
      if (j == i) continue;
      double s = 0;
      for (int64_t k = 0; k < d; k++) {
        double v = double(__half2float(xh[i * d + k])) - double(__half2float(xh[j * d + k]));
        s += v * v;
      }
      all.emplace_back(s, int(j));
    }
    std::nth_element(all.begin(), all.begin() + 9, all.end());
    double kth = all[9].first;
    for (int e = 0; e < 10; e++) {
      int v    = t[i * 10 + e];
      double s = 0;
      for (int64_t k = 0; k < d; k++) {
        double dv = double(__half2float(xh[i * d + k])) - double(__half2float(xh[v * d + k]));
        s += dv * dv;
      }
      total++;
      if (v == i || s > kth * (1 + 1e-4) + 1e-4) worse++;
    }
  }
  std::printf("table: %lld of %lld sampled entries outside the exact top 10\n",
              (long long)worse,
              (long long)total);
  check(worse == 0, "small build: exact neighbor table");
}

// ---- (b2) neighbor table with reverse edges, exact and approximate ----

void test_table_modes()
{
  const int64_t n = 6000, d = 32;
  const int k = 12;
  std::mt19937 gen(5);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> x(size_t(n) * d);
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < d; j++)
      x[i * d + j] = g(gen) + (j < 3 ? float(i % 23) : 0.f);
  // exact top-k (self excluded) of every row over the fp16 vectors
  std::vector<half> xh(x.size());
  for (size_t i = 0; i < x.size(); i++)
    xh[i] = __float2half(x[i]);
  auto dist = [&](int64_t a, int64_t b) {
    double s = 0;
    for (int64_t j = 0; j < d; j++) {
      double v = double(__half2float(xh[a * d + j])) - double(__half2float(xh[b * d + j]));
      s += v * v;
    }
    return s;
  };
  std::vector<std::vector<int>> exact(n);
  std::vector<double> kth(n);
  for (int64_t i = 0; i < n; i++) {
    std::vector<std::pair<double, int>> all;
    for (int64_t j = 0; j < n; j++)
      if (j != i) all.emplace_back(dist(i, j), int(j));
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    for (int e = 0; e < k; e++)
      exact[i].push_back(all[e].second);
    kth[i] = all[k - 1].first;
  }
  // the expected rows (prototype rule): first kf forward, up to kr reverse (rank < ks), rest
  // forward
  const int kf = k / 2, ks = k / 2, kr = k / 2;
  std::vector<std::vector<std::pair<int, int>>> rv(n);
  for (int64_t v = 0; v < n; v++)
    for (int r = 0; r < ks; r++)
      rv[exact[v][r]].emplace_back(r, int(v));
  std::vector<std::vector<int>> expect(n);
  for (int64_t u = 0; u < n; u++) {
    std::sort(rv[u].begin(), rv[u].end());
    std::vector<int> c(exact[u].begin(), exact[u].begin() + kf);
    for (int t = 0; t < kr && t < int(rv[u].size()); t++)
      c.push_back(rv[u][t].second);
    c.insert(c.end(), exact[u].begin() + kf, exact[u].end());
    for (int v : c)
      if (int(expect[u].size()) < k &&
          std::find(expect[u].begin(), expect[u].end(), v) == expect[u].end())
        expect[u].push_back(v);
  }

  for (int mode = 0; mode < 2; mode++) {
    std::string dir = "/home/cc/tmp/octex_test_build_table" + std::to_string(mode);
    octex::index_params p;
    p.copies          = 2;
    p.blocks          = 2;
    p.pca_dims        = 16;
    p.list_len        = 8;
    p.log_buckets     = 10;
    p.sketch_width    = 16;
    p.table_len       = k;
    p.reverse_edges   = true;
    p.exact_table_max = mode == 0 ? n : 0;  // mode 1: approximate graph
    raft::device_resources res;
    octex::build(res, p, raft::make_host_matrix_view<const float, int64_t>(x.data(), n, d), dir);
    npy_file table((std::filesystem::path(dir) / "table.npy").string());
    const int32_t* t   = table.as<int32_t>();
    int64_t bad_struct = 0, outside = 0, in_exact = 0, entries = 0;
    for (int64_t i = 0; i < n; i++) {
      double prev = -1;
      bool pad    = false;
      std::vector<int> seen;
      for (int e = 0; e < k; e++) {
        int v = t[i * k + e];
        if (v < 0) {
          pad = true;
          continue;
        }
        entries++;
        double dv = dist(i, v);
        // searched rows carry no distances and stay in priority order (merge_reverse), so only
        // exact rows are sorted by distance
        bool unsorted = mode == 0 && dv < prev - 1e-3 * (1 + prev);
        if (pad || v == i || v >= n || unsorted ||
            std::find(seen.begin(), seen.end(), v) != seen.end())
          bad_struct++;
        prev = dv;
        seen.push_back(v);
        bool ex = std::find(exact[i].begin(), exact[i].end(), v) != exact[i].end() ||
                  dv <= kth[i] * (1 + 1e-4);
        in_exact += ex;
        if (mode == 0 && std::find(expect[i].begin(), expect[i].end(), v) == expect[i].end())
          outside++;
      }
    }
    std::printf(
      "table (%s, reverse edges): %lld entries, malformed %lld, in the exact top %d %.4f%s\n",
      mode == 0 ? "exact" : "approximate",
      (long long)entries,
      (long long)bad_struct,
      k,
      double(in_exact) / entries,
      mode == 0 ? (", not in the expected row " + std::to_string(outside)).c_str() : "");
    if (mode == 0) {
      check(bad_struct == 0 && outside <= entries / 1000 && entries == n * k,
            "table with reverse edges (exact)");
    } else {
      check(bad_struct == 0 && entries == n * k, "table with reverse edges (approximate graph)");
    }
  }
}

// ---- (c) equivalence with a reference index ----

void test_equivalence(std::string const& dir, int ncopies, bool ivf)
{
  auto f = [&](const char* name) { return (std::filesystem::path(dir) / name).string(); };
  if (!std::filesystem::exists(f("proj.npy"))) {
    std::printf("SKIP: no reference index at %s\n", dir.c_str());
    return;
  }
  npy_file vec(f("vectors.npy")), mean(f("mean.npy")), proj(f("proj.npy")), shift(f("shift.npy")),
    lists(f("lists.npy")), keys(f("keys.npy")), offs(f("offsets.npy")), base(f("base.npy")),
    cells(f("cells.npy"));
  // parameters of the reference (from meta.json: spacings, list length, buckets)
  std::ifstream mf(f("meta.json"));
  std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
  auto num = [&](const char* key) {
    auto at = meta.find(std::string("\"") + key + "\"");
    return std::stod(meta.substr(meta.find(':', at) + 1));
  };
  octex::detail::cell_params cp;
  cp.blocks      = int(num("blocks"));
  cp.list_len    = int(num("list_len"));
  cp.log_buckets = int(num("log_buckets"));
  cp.min_cell    = 1;
  cp.exact_lists = !ivf;
  cp.spacings.clear();
  {
    auto s        = meta.find('[', meta.find("\"spacings\""));
    auto e        = meta.find(']', s);
    std::string v = meta.substr(s + 1, e - s - 1);
    size_t pos    = 0;
    while (pos < v.size()) {
      size_t q = v.find(',', pos);
      if (q == std::string::npos) q = v.size();
      cp.spacings.push_back(std::stof(v.substr(pos, q - pos)));
      pos = q + 1;
    }
  }
  const int64_t n = vec.shape[0], d = vec.shape[1];
  const int L = int(cp.spacings.size()), l = cp.list_len;
  const int64_t T = int64_t(1) << cp.log_buckets;
  raft::device_resources res;
  std::vector<octex::detail::copy_cells> out;
  auto t0 = std::chrono::steady_clock::now();
  octex::detail::build_cells(res,
                             cp,
                             vec.as<half>(),
                             n,
                             d,
                             mean.as<float>(),
                             proj.as<float>(),
                             shift.as<float>(),
                             0,
                             ncopies,
                             out);
  std::printf("built %d copies of %lld vectors in %.1f s\n",
              ncopies,
              (long long)n,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  const int64_t mmax = keys.shape[2], lmax = lists.shape[1];
  // cells are matched by hash: a vector whose projection rounds across a cell boundary (fp32 GEMM
  // order) can move one cell, which would shift every later position
  int64_t total_cells = 0, total_diff = 0, total_entries = 0, total_same = 0, total_same_set = 0,
          bucket_bad = 0;
  int64_t count_diff = 0;
  for (int c = 0; c < ncopies; c++) {
    auto const& cc    = out[c];
    const int32_t* rl = lists.as<int32_t>() + int64_t(c) * lmax;
    int64_t at = 0, diff = 0;
    for (int j = 0; j < L; j++) {
      int64_t mr = cells.as<int64_t>()[c * L + j], mm = cc.cells[j];
      count_diff += std::abs(mm - mr);
      const int64_t* rk = keys.as<int64_t>() + (int64_t(c) * L + j) * mmax;
      const int32_t* ro = offs.as<int32_t>() + (int64_t(c) * L + j) * (T + 1);
      int64_t rbase     = base.as<int32_t>()[c * L + j];
      for (int64_t t = 0; t < T; t++) {
        int32_t m0 = cc.offsets[j * (T + 1) + t], m1 = cc.offsets[j * (T + 1) + t + 1];
        int32_t r0 = ro[t], r1 = ro[t + 1];
        bucket_bad += (m1 - m0) != (r1 - r0);
        // both sides hold the bucket's hashes in ascending order: merge
        int32_t x = m0, y = r0;
        while (x < m1 || y < r1) {
          int64_t kx = x < m1 ? cc.keys[at + x] : INT64_MAX, ky = y < r1 ? rk[y] : INT64_MAX;
          if (x < m1 && y < r1 && kx == ky) {
            const int32_t* a = cc.lists.data() + (cc.base[j] + int64_t(x)) * l;
            const int32_t* b = rl + (rbase + int64_t(y)) * l;
            std::vector<int32_t> sa(a, a + l), sb(b, b + l), both;
            for (int e = 0; e < l; e++)
              total_same += a[e] == b[e];
            std::sort(sa.begin(), sa.end());
            std::sort(sb.begin(), sb.end());
            std::set_intersection(
              sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(both));
            total_same_set += both.size();
            total_entries += l;
            x++;
            y++;
          } else if (kx < ky) {
            diff++;
            x++;
          } else {
            diff++;
            y++;
          }
        }
      }
      at += mm;
      total_cells += mr;
    }
    total_diff += diff;
    std::printf("copy %d: cells %lld, cells present on one side only %lld\n",
                c,
                (long long)(cc.lists.size() / l),
                (long long)diff);
  }
  double frac     = total_entries ? double(total_same) / total_entries : 0.0;
  double frac_set = total_entries ? double(total_same_set) / total_entries : 0.0;
  std::printf(
    "%d copies: %lld cells, %lld on one side only (%.2e), buckets with another count %lld; "
    "lists of the common cells: %.5f identical entries, %.5f identical as row sets\n",
    ncopies,
    (long long)total_cells,
    (long long)total_diff,
    double(total_diff) / total_cells,
    (long long)bucket_bad,
    frac,
    frac_set);
  check(count_diff <= total_diff, "equivalence: cells per level (up to the boundary cells)");
  check(double(total_diff) / total_cells <= 1e-4,
        "equivalence: keys and offsets (boundary vectors only)");
  if (!ivf) check(frac >= 0.999, "equivalence: lists >= 99.9% identical entries");
}

}  // namespace

int main(int argc, char** argv)
{
  std::string dir = argc > 1 ? argv[1] : "/home/cc/data/octex/sift1m";
  int copies      = argc > 2 ? std::atoi(argv[2]) : 2;
  test_e8();
  test_small_build();
  test_table_modes();
  test_equivalence(dir, copies, argc > 3 && std::string(argv[3]) == "ivf");
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
  return failures ? 1 : 0;
}
