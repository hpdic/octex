/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Quality of neighbor tables: the share of the exact top-k (the row itself excluded) of sampled
// rows that a table row holds. The exact neighbors come from cuVS brute force over the first N rows
// of an fp16 .npy matrix, read in chunks by buffered reads.
//
//   octex_table_check <base fp16 .npy> <rows N> <samples> <k> <table .npy> [<table .npy> ...]

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "neighbors/octex/detail/npy.hpp"

#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace detail = cuvs::neighbors::octex::detail;

namespace {

/** Rows [r0, r0 + m) of an fp16 .npy matrix as float, by pread. */
void read_rows(int fd, size_t off, int64_t r0, int64_t m, int64_t d, std::vector<float>& out)
{
  std::vector<half> h(size_t(m) * d);
  auto* dst  = reinterpret_cast<char*>(h.data());
  size_t rem = h.size() * sizeof(half), at = off + size_t(r0) * d * sizeof(half);
  while (rem > 0) {
    ssize_t got = ::pread(fd, dst, std::min(rem, size_t(64) << 20), at);
    if (got <= 0) throw std::runtime_error("short read");
    dst += got;
    at += size_t(got);
    rem -= size_t(got);
  }
  out.resize(h.size());
  for (size_t i = 0; i < h.size(); i++)
    out[i] = __half2float(h[i]);
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 6) {
    std::fprintf(
      stderr, "usage: %s <base fp16 .npy> <rows> <samples> <k> <table .npy>...\n", argv[0]);
    return 2;
  }
  std::string base = argv[1];
  int64_t n        = std::stoll(argv[2]);
  int64_t ns       = std::stoll(argv[3]);
  int k            = std::stoi(argv[4]);
  auto hdr         = detail::read_npy_header(base);
  if (hdr.descr != "<f2" || hdr.shape.size() != 2) throw std::runtime_error("expected fp16 [N, D]");
  const int64_t d = hdr.shape[1];
  n               = std::min(n, hdr.shape[0]);
  int fd          = ::open(base.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + base);

  std::vector<int64_t> rows(ns);
  for (int64_t i = 0; i < ns; i++)
    rows[i] = (n - 1) * i / std::max<int64_t>(ns - 1, 1);
  std::vector<float> q(size_t(ns) * d), tmp;
  for (int64_t i = 0; i < ns; i++) {
    read_rows(fd, hdr.header_len, rows[i], 1, d, tmp);
    std::copy(tmp.begin(), tmp.end(), q.begin() + i * d);
  }

  raft::device_resources res;
  auto stream     = raft::resource::get_cuda_stream(res);
  const int kk    = k + 1;  // the row itself may come back
  const int64_t m = int64_t(1) << 22;
  rmm::device_uvector<float> dq(q.size(), stream), dx(size_t(m) * d, stream), dd(ns * kk, stream);
  rmm::device_uvector<int64_t> di(ns * kk, stream);
  raft::copy(dq.data(), q.data(), q.size(), stream);
  std::vector<std::vector<std::pair<float, int64_t>>> best(ns);
  std::vector<float> hd(ns * kk);
  std::vector<int64_t> hi(ns * kk);
  for (int64_t r0 = 0; r0 < n; r0 += m) {
    int64_t c = std::min(m, n - r0);
    read_rows(fd, hdr.header_len, r0, c, d, tmp);
    raft::copy(dx.data(), tmp.data(), tmp.size(), stream);
    cuvs::neighbors::brute_force::index_params ip;
    ip.metric = cuvs::distance::DistanceType::L2Expanded;
    auto idx  = cuvs::neighbors::brute_force::build(
      res, ip, raft::make_device_matrix_view<const float, int64_t>(dx.data(), c, d));
    cuvs::neighbors::brute_force::search(
      res,
      cuvs::neighbors::brute_force::search_params{},
      idx,
      raft::make_device_matrix_view<const float, int64_t>(dq.data(), ns, d),
      raft::make_device_matrix_view<int64_t, int64_t>(di.data(), ns, kk),
      raft::make_device_matrix_view<float, int64_t>(dd.data(), ns, kk));
    raft::copy(hd.data(), dd.data(), hd.size(), stream);
    raft::copy(hi.data(), di.data(), hi.size(), stream);
    res.sync_stream();
    for (int64_t i = 0; i < ns; i++)
      for (int j = 0; j < kk; j++)
        if (hi[i * kk + j] >= 0) best[i].emplace_back(hd[i * kk + j], hi[i * kk + j] + r0);
  }
  ::close(fd);
  std::vector<std::vector<int64_t>> exact(ns);
  for (int64_t i = 0; i < ns; i++) {
    std::sort(best[i].begin(), best[i].end());
    for (auto const& e : best[i])
      if (e.second != rows[i] && int(exact[i].size()) < k) exact[i].push_back(e.second);
  }
  std::printf("exact top-%d of %lld rows of the first %lld\n", k, (long long)ns, (long long)n);
  for (int a = 5; a < argc; a++) {
    detail::npy_file t(argv[a]);
    const int64_t L = t.shape[1];
    if (t.shape[0] != n)
      std::printf("  (table rows %lld != %lld)\n", (long long)t.shape[0], (long long)n);
    int64_t hit = 0, pad = 0, self = 0;
    for (int64_t i = 0; i < ns; i++) {
      const int32_t* row = t.as<int32_t>() + rows[i] * L;
      for (int64_t j = 0; j < L; j++) {
        pad += row[j] < 0;
        self += row[j] == rows[i];
      }
      for (int64_t e : exact[i])
        hit += std::find(row, row + L, int32_t(e)) != row + L;
    }
    std::printf("%s: rows of %lld, top-%d per row %.4f, padding %.4f, self %lld\n",
                argv[a],
                (long long)L,
                k,
                double(hit) / double(ns * k),
                double(pad) / double(ns * L),
                (long long)self);
  }
  return 0;
}
