/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The streaming cell build (no projection of all rows on the device, the lists merged over row
// ranges) against the whole-projection build, on a prefix of a dataset with the projections of a
// reference index:
//   - keys, offsets, base and cells identical;
//   - exact lists (brute force per range, merged): identical up to distance ties;
//   - IVF-Flat lists: fraction of identical entries, and the recall@10 of the union of the
//     first-round candidates (per copy the list of the finest level that has the query's cell)
//     against brute force; streaming may lose at most 0.005.
//
//   test_build_stream [reference dir, default /home/cc/data/octex/deep10M] [rows, default 2000000]
//                     [copies, default 16] [queries, default 1000]
//
// Files are read with buffered reads of the needed rows only (no memory maps).

#include <cuvs/neighbors/brute_force.hpp>

#include "neighbors/octex/detail/build_cells.hpp"
#include "neighbors/octex/detail/e8.cuh"
#include "neighbors/octex/detail/npy.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

namespace octex = cuvs::neighbors::octex;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) failures++;
}

/** The first `rows` rows of an .npy matrix (all rows if rows <= 0), by buffered reads. */
template <typename T>
auto read_rows(std::string const& path, int64_t rows, std::vector<int64_t>& shape) -> std::vector<T>
{
  auto h = octex::detail::read_npy_header(path);
  shape  = h.shape;
  if (rows > 0 && rows < shape[0]) shape[0] = rows;
  int64_t count = 1;
  for (auto s : shape)
    count *= s;
  std::vector<T> v(count);
  std::ifstream f(path, std::ios::binary);
  f.seekg(static_cast<std::streamoff>(h.header_len));
  const size_t step = size_t(1) << 26;
  char* dst         = reinterpret_cast<char*>(v.data());
  size_t total = count * sizeof(T), done = 0;
  while (done < total) {
    size_t m = std::min(step, total - done);
    f.read(dst + done, static_cast<std::streamsize>(m));
    if (!f) throw std::runtime_error("short read: " + path);
    done += m;
  }
  return v;
}

__global__ void hash_kernel(const float* y, int n, int b, float f, long long* h)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x[64];
  for (int k = 0; k < 8 * b; k++)
    x[k] = y[i * 8 * b + k] / f;
  h[i] = octex::detail::e8b_hash64(x, b, 1.0f);
}

struct setup {
  int64_t n = 0, d = 0;
  std::vector<half> x;
  std::vector<float> mean, proj, shift, q;
  int64_t nq = 0;
  octex::detail::cell_params cp;
};

auto build(setup const& s, octex::detail::cell_params const& cp, int copies, const char* stream)
  -> std::vector<octex::detail::copy_cells>
{
  setenv("OCTEX_BUILD_STREAM", stream, 1);
  raft::device_resources res;
  std::vector<octex::detail::copy_cells> out;
  auto t0 = std::chrono::steady_clock::now();
  octex::detail::build_cells(
    res, cp, s.x.data(), s.n, s.d, s.mean.data(), s.proj.data(), s.shift.data(), 0, copies, out);
  std::printf("  %s path: %d copies of %lld rows in %.1f s\n",
              stream[0] == '1' ? "streaming" : "whole-projection",
              copies,
              (long long)s.n,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return out;
}

/** Structure equal; returns the fraction of identical list entries and of identical row sets. */
void compare(std::vector<octex::detail::copy_cells> const& a,
             std::vector<octex::detail::copy_cells> const& b,
             int l,
             double& same,
             double& same_set,
             bool& structure)
{
  structure   = a.size() == b.size();
  int64_t tot = 0, eq = 0, eqs = 0;
  for (size_t c = 0; c < a.size() && structure; c++) {
    structure = structure && a[c].keys == b[c].keys && a[c].offsets == b[c].offsets &&
                a[c].base == b[c].base && a[c].cells == b[c].cells &&
                a[c].lists.size() == b[c].lists.size();
    if (!structure) break;
    for (size_t r = 0; r < a[c].lists.size() / l; r++) {
      const int32_t* x = a[c].lists.data() + r * l;
      const int32_t* y = b[c].lists.data() + r * l;
      std::vector<int32_t> sx(x, x + l), sy(y, y + l), both;
      for (int e = 0; e < l; e++)
        eq += x[e] == y[e];
      std::sort(sx.begin(), sx.end());
      std::sort(sy.begin(), sy.end());
      std::set_intersection(sx.begin(), sx.end(), sy.begin(), sy.end(), std::back_inserter(both));
      eqs += both.size();
      tot += l;
    }
  }
  same     = tot ? double(eq) / tot : 0;
  same_set = tot ? double(eqs) / tot : 0;
}

/** Per query and copy the hashes of every level (fp32 projection on the CPU, hash on the GPU). */
auto query_hashes(setup const& s, int copies) -> std::vector<long long>
{
  const int b = s.cp.blocks, w = 8 * b, L = int(s.cp.spacings.size());
  const int64_t nq = s.nq, d = s.d;
  std::vector<float> y(size_t(nq) * copies * w);
  for (int64_t i = 0; i < nq; i++)
    for (int c = 0; c < copies; c++)
      for (int k = 0; k < w; k++) {
        double acc = 0;
        for (int64_t r = 0; r < d; r++)
          acc += double(s.q[i * d + r] - s.mean[r]) * s.proj[(int64_t(c) * d + r) * w + k];
        y[(i * copies + c) * w + k] = float(acc) + s.shift[int64_t(c) * w + k];
      }
  int m = int(nq * copies);
  raft::device_resources res;
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  rmm::device_uvector<float> dy(y.size(), stream);
  rmm::device_uvector<long long> dh(m, stream);
  raft::copy(dy.data(), y.data(), y.size(), stream);
  std::vector<long long> h(size_t(m) * L);
  for (int j = 0; j < L; j++) {
    hash_kernel<<<(m + 127) / 128, 128, 0, stream>>>(dy.data(), m, b, s.cp.spacings[j], dh.data());
    raft::copy(h.data() + size_t(j) * m, dh.data(), m, stream);
  }
  raft::resource::sync_stream(res);
  return h;  // [L][nq * copies]
}

/** recall@10 of the union of the probed lists (finest level whose cell has a list, per copy). */
auto candidate_recall(setup const& s,
                      std::vector<octex::detail::copy_cells> const& cells,
                      std::vector<long long> const& qh,
                      std::vector<int64_t> const& gt,
                      double& mean_cand) -> double
{
  const int L = int(s.cp.spacings.size()), l = s.cp.list_len, copies = int(cells.size());
  const int64_t T = int64_t(1) << s.cp.log_buckets;
  const int64_t m = s.nq * copies;
  int64_t hit = 0, ncand = 0;
  for (int64_t i = 0; i < s.nq; i++) {
    std::unordered_set<int32_t> u;
    for (int c = 0; c < copies; c++) {
      auto const& cc = cells[c];
      for (int j = 0; j < L; j++) {
        long long h    = qh[size_t(j) * m + i * copies + c];
        int64_t bucket = h & (T - 1);
        int32_t o0 = cc.offsets[j * (T + 1) + bucket], o1 = cc.offsets[j * (T + 1) + bucket + 1];
        const int64_t* k = cc.keys.data() + cc.base[j];
        int64_t at       = -1;
        for (int32_t p = o0; p < o1; p++)
          if (k[p] == h) at = p;
        if (at < 0) continue;
        const int32_t* lst = cc.lists.data() + (int64_t(cc.base[j]) + at) * l;
        for (int e = 0; e < l; e++)
          if (lst[e] >= 0) u.insert(lst[e]);
        break;
      }
    }
    ncand += u.size();
    for (int e = 0; e < 10; e++)
      hit += u.count(static_cast<int32_t>(gt[i * 10 + e]));
  }
  mean_cand = double(ncand) / s.nq;
  return double(hit) / (s.nq * 10.0);
}

auto ground_truth(setup const& s) -> std::vector<int64_t>
{
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<float> xf(s.x.size());
  for (size_t i = 0; i < xf.size(); i++)
    xf[i] = __half2float(s.x[i]);
  rmm::device_uvector<float> dx(xf.size(), stream), dq(s.q.size(), stream), dd(s.nq * 10, stream);
  rmm::device_uvector<int64_t> di(s.nq * 10, stream);
  raft::copy(dx.data(), xf.data(), xf.size(), stream);
  raft::copy(dq.data(), s.q.data(), s.q.size(), stream);
  cuvs::neighbors::brute_force::index_params ip;
  ip.metric = cuvs::distance::DistanceType::L2Expanded;
  auto idx  = cuvs::neighbors::brute_force::build(
    res, ip, raft::make_device_matrix_view<const float, int64_t>(dx.data(), s.n, s.d));
  cuvs::neighbors::brute_force::search(
    res,
    cuvs::neighbors::brute_force::search_params{},
    idx,
    raft::make_device_matrix_view<const float, int64_t>(dq.data(), s.nq, s.d),
    raft::make_device_matrix_view<int64_t, int64_t>(di.data(), s.nq, 10),
    raft::make_device_matrix_view<float, int64_t>(dd.data(), s.nq, 10));
  std::vector<int64_t> gt(s.nq * 10);
  raft::copy(gt.data(), di.data(), gt.size(), stream);
  raft::resource::sync_stream(res);
  return gt;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string dir = argc > 1 ? argv[1] : "/home/cc/data/octex/deep10M";
  int64_t rows    = argc > 2 ? std::atoll(argv[2]) : 2000000;
  int copies      = argc > 3 ? std::atoi(argv[3]) : 16;
  int64_t nq      = argc > 4 ? std::atoll(argv[4]) : 1000;
  auto f          = [&](const char* name) { return (std::filesystem::path(dir) / name).string(); };
  if (!std::filesystem::exists(f("vectors.npy")) || !std::filesystem::exists(f("proj.npy"))) {
    std::printf("SKIP: no reference index at %s\n", dir.c_str());
    return 0;
  }
  if (std::getenv("OCTEX_BUILD_RANGES") == nullptr) setenv("OCTEX_BUILD_RANGES", "4", 1);

  setup s;
  std::vector<int64_t> sh;
  s.x    = read_rows<half>(f("vectors.npy"), rows, sh);
  s.n    = sh[0];
  s.d    = sh[1];
  s.mean = read_rows<float>(f("mean.npy"), 0, sh);
  s.proj = read_rows<float>(f("proj.npy"), copies, sh);
  RAFT_EXPECTS(sh[0] == copies, "reference has fewer copies");
  s.cp.blocks   = int(sh[2] / 8);
  s.shift       = read_rows<float>(f("shift.npy"), copies, sh);
  s.q           = read_rows<float>(f("queries.npy"), nq, sh);
  s.nq          = sh[0];
  s.cp.spacings = {2.0f, 2.5f, 3.2f, 4.0f, 5.0f};
  s.cp.list_len = 16;
  s.cp.min_cell = 100;
  std::printf("%lld rows, %d copies, %lld queries, %s row ranges in the streaming path\n",
              (long long)s.n,
              copies,
              (long long)s.nq,
              std::getenv("OCTEX_BUILD_RANGES"));

  const int l = s.cp.list_len;
  double same = 0, same_set = 0;
  bool structure = false;

  // exact lists: the merge over row ranges must reproduce the whole-range lists
  {
    auto cp        = s.cp;
    cp.exact_lists = true;
    int ce         = std::min(copies, 2);
    std::printf("exact lists:\n");
    auto a = build(s, cp, ce, "0");
    auto b = build(s, cp, ce, "1");
    compare(a, b, l, same, same_set, structure);
    std::printf("  identical entries %.6f, identical row sets %.6f\n", same, same_set);
    check(structure, "exact lists: keys, offsets, base, cells identical");
    check(same_set >= 0.999, "exact lists: >= 99.9% identical row sets (distance ties only)");
  }

  // IVF-Flat lists
  std::printf("IVF-Flat lists (list_probes %d):\n", s.cp.list_probes);
  auto a = build(s, s.cp, copies, "0");
  auto b = build(s, s.cp, copies, "1");
  compare(a, b, l, same, same_set, structure);
  int64_t cells = 0;
  for (auto const& cc : a)
    cells += cc.lists.size() / l;
  std::printf("  %lld cells; identical entries %.4f, identical as row sets %.4f\n",
              (long long)cells,
              same,
              same_set);
  check(structure, "IVF lists: keys, offsets, base, cells identical");

  auto gt   = ground_truth(s);
  auto qh   = query_hashes(s, copies);
  double ca = 0, cb = 0;
  double ra = candidate_recall(s, a, qh, gt, ca);
  double rb = candidate_recall(s, b, qh, gt, cb);
  std::printf(
    "  first-round candidate recall@10 (union of %d copies): whole-projection %.4f (%.1f "
    "candidates), streaming %.4f (%.1f candidates)\n",
    copies,
    ra,
    ca,
    rb,
    cb);
  check(rb >= ra - 0.005, "IVF lists: streaming candidate recall within 0.005");
  std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
  return failures ? 1 : 0;
}
