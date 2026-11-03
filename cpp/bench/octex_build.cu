/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Build an octex index directory from a vector file and report the time of every phase and the peak
// device and host memory.
//
//   octex_build <base .fvecs|.bvecs|.npy> <out dir> [--rows N] [--copies 200] [--blocks 4]
//               [--pca-dims 64] [--spacings 2.0,2.5,3.2,4.0,5.0] [--list-len 64] [--min-cell 1]
//               [--log-buckets 20] [--sketch-width 64] [--table-len 64] [--no-reverse]
//               [--exact-lists]
//               [--list-probes 16] [--exact-table-max 20000000] [--seed 0] [--devices 0,1,...]
//               [--queries <.fvecs|.bvecs|.npy>] [--gt <.ivecs|.npy>]
//               [--member-lists] [--table-passes 3] [--table-sketch-order] [--write-behind]
//               [--link-vectors] [--reuse-base] [--fast]
//
// Billion-scale build (--fast = all of the following): --member-lists builds the cell lists from
// the members of each cell instead of IVF-Flat; --table-passes 2 and --table-sketch-order make the
// searched neighbor table two passes in sketch order; --write-behind writes the copies and the
// sketch while the table is built; --link-vectors hard-links an fp16 .npy base as vectors.npy
// instead of rewriting it; --reuse-base page-locks the base in place for the table build instead of
// freeing it and pinning a new copy.
//
// With --queries / --gt the directory also gets queries.npy (f32) and groundtruth.npy (i64).

#include <cuvs/neighbors/octex.hpp>

#include "neighbors/octex/detail/npy.hpp"

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>

#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace octex = cuvs::neighbors::octex;
using octex::detail::npy_file;
using octex::detail::write_npy;

namespace {

auto ends_with(std::string const& s, std::string const& e) -> bool
{
  return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

/** Rows of a .fvecs / .bvecs / .ivecs file (element type E) as T, at most `limit` rows (0: all). */
template <typename E, typename T>
auto read_vecs(std::string const& path, int64_t limit, int64_t& n, int64_t& d) -> std::vector<T>
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int32_t dim = 0;
  f.read(reinterpret_cast<char*>(&dim), 4);
  f.seekg(0, std::ios::end);
  int64_t bytes = f.tellg();
  int64_t row   = 4 + int64_t(dim) * sizeof(E);
  n             = bytes / row;
  if (limit > 0) n = std::min(n, limit);
  d = dim;
  std::vector<T> out(size_t(n) * d);
  std::vector<E> buf(static_cast<size_t>(d));
  f.seekg(0);
  for (int64_t i = 0; i < n; i++) {
    f.seekg(4, std::ios::cur);
    f.read(reinterpret_cast<char*>(buf.data()), d * sizeof(E));
    for (int64_t k = 0; k < d; k++)
      out[i * d + k] = static_cast<T>(buf[k]);
  }
  return out;
}

/** Vectors as float (or half when the .npy is f16 and want_half). */
struct dataset {
  std::vector<float> f;
  std::vector<half, cuvs::neighbors::octex::detail::no_init_allocator<half>> h;
  int64_t n = 0, d = 0;
};

auto read_dataset(std::string const& path, int64_t limit, bool keep_half) -> dataset
{
  dataset ds;
  if (ends_with(path, ".fvecs")) {
    ds.f = read_vecs<float, float>(path, limit, ds.n, ds.d);
  } else if (ends_with(path, ".bvecs")) {
    ds.f = read_vecs<uint8_t, float>(path, limit, ds.n, ds.d);
  } else if (ends_with(path, ".npy")) {
    npy_file a(path);
    if (a.shape.size() != 2) throw std::runtime_error(path + ": expected a matrix");
    ds.n     = limit > 0 ? std::min(limit, a.shape[0]) : a.shape[0];
    ds.d     = a.shape[1];
    size_t m = size_t(ds.n) * ds.d;
    if (a.descr == "<f2" && keep_half) {  // buffered sequential reads: no file pages stay mapped
      auto hdr = cuvs::neighbors::octex::detail::read_npy_header(path);
      ds.h.resize(m);
      int fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0) throw std::runtime_error(path + ": cannot open");
      cuvs::neighbors::octex::detail::parallel_pread(
        fd, reinterpret_cast<char*>(ds.h.data()), m * sizeof(half), hdr.header_len);
      ::close(fd);
    } else {
      ds.f.resize(m);
      if (a.descr == "<f4") {
        std::memcpy(ds.f.data(), a.as<float>(), m * 4);
      } else if (a.descr == "<f2") {
        for (size_t i = 0; i < m; i++)
          ds.f[i] = __half2float(a.as<half>()[i]);
      } else if (a.descr == "|u1") {
        for (size_t i = 0; i < m; i++)
          ds.f[i] = a.as<uint8_t>()[i];
      } else if (a.descr == "|i1") {
        for (size_t i = 0; i < m; i++)
          ds.f[i] = a.as<int8_t>()[i];
      } else {
        throw std::runtime_error(path + ": unsupported dtype " + a.descr);
      }
    }
  } else {
    throw std::runtime_error("unknown vector file type: " + path);
  }
  return ds;
}

auto split_floats(std::string const& s) -> std::vector<float>
{
  std::vector<float> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    v.push_back(std::stof(t));
  return v;
}

auto split_ints(std::string const& s) -> std::vector<int>
{
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    v.push_back(std::stoi(t));
  return v;
}

/** Peak used memory per visible device, sampled every 20 ms. */
class memory_monitor {
 public:
  memory_monitor()
  {
    cudaGetDeviceCount(&count_);
    base_.resize(count_);
    peak_ = std::vector<std::atomic<size_t>>(count_);
    for (int g = 0; g < count_; g++) {
      cudaSetDevice(g);
      size_t f = 0, t = 0;
      cudaMemGetInfo(&f, &t);
      base_[g] = t - f;
      peak_[g] = 0;
    }
    thread_ = std::thread([this] {
      while (!stop_) {
        for (int g = 0; g < count_; g++) {
          cudaSetDevice(g);
          size_t f = 0, t = 0;
          if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
            size_t used = t - f > base_[g] ? t - f - base_[g] : 0;
            if (used > peak_[g]) peak_[g] = used;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
  }
  ~memory_monitor()
  {
    stop_ = true;
    thread_.join();
  }
  [[nodiscard]] auto count() const -> int { return count_; }
  [[nodiscard]] auto peak(int g) const -> size_t { return peak_[g]; }

 private:
  int count_ = 0;
  std::vector<size_t> base_;
  std::vector<std::atomic<size_t>> peak_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <base .fvecs|.bvecs|.npy> <out dir> [options]\n", argv[0]);
    return 2;
  }
  std::string in = argv[1], out = argv[2], queries, gt;
  int64_t rows      = 0;
  bool link_vectors = false, reuse_base = false;
  octex::index_params p;
  for (int i = 3; i < argc; i++) {
    std::string a = argv[i];
    auto next     = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value of " + a);
      return argv[++i];
    };
    if (a == "--rows")
      rows = std::stoll(next());
    else if (a == "--copies")
      p.copies = std::stoi(next());
    else if (a == "--blocks")
      p.blocks = std::stoi(next());
    else if (a == "--pca-dims")
      p.pca_dims = std::stoi(next());
    else if (a == "--spacings")
      p.spacings = split_floats(next());
    else if (a == "--list-len")
      p.list_len = std::stoi(next());
    else if (a == "--min-cell")
      p.min_cell = std::stoi(next());
    else if (a == "--log-buckets")
      p.log_buckets = std::stoi(next());
    else if (a == "--sketch-width")
      p.sketch_width = std::stoi(next());
    else if (a == "--table-len")
      p.table_len = std::stoi(next());
    else if (a == "--no-reverse")
      p.reverse_edges = false;
    else if (a == "--exact-lists")
      p.exact_lists = true;
    else if (a == "--list-probes")
      p.list_probes = std::stoi(next());
    else if (a == "--exact-table-max")
      p.exact_table_max = std::stoll(next());
    else if (a == "--seed")
      p.seed = std::stoull(next());
    else if (a == "--devices")
      p.devices = split_ints(next());
    else if (a == "--member-lists")
      p.member_lists = true;
    else if (a == "--table-passes")
      p.table_passes = std::stoi(next());
    else if (a == "--table-sketch-order")
      p.table_sketch_order = true;
    else if (a == "--write-behind")
      p.write_behind = true;
    else if (a == "--link-vectors")
      link_vectors = true;
    else if (a == "--reuse-base")
      reuse_base = true;
    else if (a == "--fast") {
      p.member_lists       = true;
      p.table_passes       = 2;
      p.table_sketch_order = true;
      p.write_behind       = true;
      link_vectors         = true;
      reuse_base           = true;
    } else if (a == "--queries")
      queries = next();
    else if (a == "--gt")
      gt = next();
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  setenv("OCTEX_VERBOSE", "1", 0);

  // OCTEX_TABLE_ONLY: the directory holds every array but the table, whose searches read their
  // queries from vectors.npy; the base is not read at all
  auto t0               = std::chrono::steady_clock::now();
  const bool table_only = std::getenv("OCTEX_TABLE_ONLY") != nullptr;
  dataset ds;
  if (table_only) {
    auto hdr = cuvs::neighbors::octex::detail::read_npy_header(in);
    ds.n     = rows > 0 ? std::min(rows, hdr.shape[0]) : hdr.shape[0];
    ds.d     = hdr.shape[1];
  } else {
    ds = read_dataset(in, rows, true);
  }
  std::printf("read %lld x %lld from %s in %.1f s\n",
              (long long)ds.n,
              (long long)ds.d,
              in.c_str(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  std::fflush(stdout);
  // --link-vectors: an fp16 .npy base of the full dataset becomes vectors.npy by a hard link
  if (link_vectors && ends_with(in, ".npy") && rows == 0 && !ds.h.empty())
    setenv("OCTEX_VECTORS_FROM", in.c_str(), 1);

  memory_monitor mon;
  raft::device_resources res;
  auto t1 = std::chrono::steady_clock::now();
  if (table_only) {
    octex::build(
      res, p, raft::make_host_matrix_view<const half, int64_t>(nullptr, ds.n, ds.d), out);
  } else if (!ds.h.empty() && p.table_len > 0 && ds.n > p.exact_table_max) {
    // a searched table: build everything else, free the base (179 GiB at 1e9 x 96), then the table
    auto p0      = p;
    p0.table_len = 0;
    setenv("OCTEX_TABLE_FOLLOWS", std::to_string(p.table_len).c_str(), 1);
    octex::build(
      res, p0, raft::make_host_matrix_view<const half, int64_t>(ds.h.data(), ds.n, ds.d), out);
    unsetenv("OCTEX_TABLE_FOLLOWS");
    // --reuse-base: the base (anonymous memory) is page-locked in place and serves the table
    // build as its host vectors, instead of being freed and read again into a new pinned buffer
    const std::string vpath = (std::filesystem::path(out) / "vectors.npy").string();
    const bool reuse        = reuse_base;
    if (reuse) {
      auto tr      = std::chrono::steady_clock::now();
      size_t bytes = size_t(ds.n) * ds.d * sizeof(half);
      RAFT_CUDA_TRY(
        cudaHostRegister(ds.h.data(), bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
      cuvs::neighbors::octex::detail::register_mem_array(
        vpath, ds.h.data(), {ds.n, ds.d}, "<f2", bytes, true);
      std::printf("base locked in place in %.1f s\n",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count());
      std::fflush(stdout);
    } else {
      decltype(ds.h)().swap(ds.h);
    }
    setenv("OCTEX_TABLE_ONLY", "1", 1);
    octex::build(
      res, p, raft::make_host_matrix_view<const half, int64_t>(nullptr, ds.n, ds.d), out);
    unsetenv("OCTEX_TABLE_ONLY");
    if (reuse) {
      cuvs::neighbors::octex::detail::unregister_mem_array(vpath);
      cudaHostUnregister(ds.h.data());
      decltype(ds.h)().swap(ds.h);
    }
  } else if (!ds.h.empty()) {
    octex::build(
      res, p, raft::make_host_matrix_view<const half, int64_t>(ds.h.data(), ds.n, ds.d), out);
  } else {
    octex::build(
      res, p, raft::make_host_matrix_view<const float, int64_t>(ds.f.data(), ds.n, ds.d), out);
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  std::printf("build: %.1f s\n", secs);
  for (int g = 0; g < mon.count(); g++)
    std::printf(
      "peak device memory (visible GPU %d): %.2f GiB\n", g, mon.peak(g) / double(1 << 30));
  struct rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  std::printf("peak host memory (RSS): %.2f GiB\n", ru.ru_maxrss / double(1 << 20));

  if (!queries.empty()) {
    auto q = read_dataset(queries, 0, false);
    write_npy((std::filesystem::path(out) / "queries.npy").string(), q.f.data(), {q.n, q.d});
  }
  if (!gt.empty()) {
    int64_t n = 0, k = 0;
    std::vector<int64_t> g;
    if (ends_with(gt, ".ivecs")) {
      g = read_vecs<int32_t, int64_t>(gt, 0, n, k);
    } else {
      npy_file a(gt);
      n = a.shape[0];
      k = a.shape[1];
      g.resize(size_t(n) * k);
      for (int64_t i = 0; i < n * k; i++)
        g[i] = a.descr == "<i8" ? a.as<int64_t>()[i] : a.as<int32_t>()[i];
    }
    write_npy((std::filesystem::path(out) / "groundtruth.npy").string(), g.data(), {n, k});
  }
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
