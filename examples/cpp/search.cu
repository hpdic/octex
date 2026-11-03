/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The C++ module, with the calls of the cuVS index modules: load an index directory onto the GPUs
// and search the 10 nearest neighbors of a batch of queries held in device memory.
//
//   cmake --build build --target octex_example_search
//   build/octex_example_search INDEX_DIR [QUERIES.npy [GROUNDTRUTH.npy]]
//
// QUERIES.npy is float32 [m, dim]; without it the program searches 1,000 random queries.
// GROUNDTRUTH.npy is int32 or int64 [m, >= 10] and gives the recall@10.

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** A two-dimensional little-endian C-order .npy array: its dtype code ("<f4", "<i8", ...), shape,
 * bytes. */
struct npy {
  std::string dtype;
  int64_t rows = 0, cols = 0;
  std::vector<char> data;
};

auto read_npy(const std::string& path) -> npy
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("cannot open " + path); }
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) { throw std::runtime_error(path + " is not .npy"); }
  uint32_t header_len = 0;
  if (magic[6] == 1) {
    uint16_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 2);
    header_len = n;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, ' ');
  f.read(header.data(), header_len);
  npy a;
  auto d  = header.find("'descr': '") + 10;
  a.dtype = header.substr(d, header.find('\'', d) - d);
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error(path + " is not in C order");
  }
  auto s = header.find('(', header.find("'shape':")) + 1;
  std::sscanf(header.c_str() + s, "%ld, %ld", &a.rows, &a.cols);
  const int64_t item = a.dtype.back() - '0';
  a.data.resize(a.rows * a.cols * item);
  f.read(a.data.data(), a.data.size());
  return a;
}

}  // namespace

int main(int argc, char** argv)
{
  using namespace cuvs::neighbors;
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s INDEX_DIR [QUERIES.npy [GROUNDTRUTH.npy]]\n", argv[0]);
    return 1;
  }
  const int k = 10;
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  octex::load_params load_params;  // all visible GPUs, copies sharded over them
  auto index   = octex::load(res, argv[1], load_params);
  const auto d = index.dim();
  std::printf("loaded %ld vectors of dimension %ld, %d copies, on %zu GPU(s)\n",
              long(index.size()),
              long(d),
              index.copies(),
              index.devices().size());

  std::vector<float> query;
  int64_t nq = 1000;
  if (argc > 2) {
    auto q = read_npy(argv[2]);
    if (q.dtype != "<f4" || q.cols != d) {
      throw std::runtime_error("queries must be float32 [m, dim]");
    }
    nq = q.rows;
    query.resize(nq * d);
    std::memcpy(query.data(), q.data.data(), q.data.size());
  } else {
    query.resize(nq * d);
    std::mt19937 gen(0);
    std::normal_distribution<float> normal;
    for (auto& v : query)
      v = normal(gen);
  }

  auto dq        = raft::make_device_matrix<float, int64_t>(res, nq, d);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, nq, k);
  raft::copy(dq.data_handle(), query.data(), nq * d, stream);

  octex::search_params search_params;  // read 32, starts 128, waves 2
  for (int run = 0; run < 2; ++run) {  // the first run captures the CUDA graph
    res.sync_stream();
    const auto t0 = std::chrono::steady_clock::now();
    octex::search(res,
                  search_params,
                  index,
                  raft::make_const_mdspan(dq.view()),
                  neighbors.view(),
                  distances.view());
    res.sync_stream();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("run %d: %ld queries in %.3f s\n", run, long(nq), s);
  }

  std::vector<int64_t> ids(nq * k);
  std::vector<float> dist(nq * k);
  raft::copy(ids.data(), neighbors.data_handle(), nq * k, stream);
  raft::copy(dist.data(), distances.data_handle(), nq * k, stream);
  res.sync_stream();
  std::printf("nearest neighbors of query 0:");
  for (int t = 0; t < k; ++t)
    std::printf(" %ld (%.1f)", long(ids[t]), dist[t]);
  std::printf("\n");

  if (argc > 3) {
    auto gt = read_npy(argv[3]);
    if (gt.rows != nq || gt.cols < k) {
      throw std::runtime_error("ground truth must be [m, >= 10]");
    }
    int64_t hits = 0;
    for (int64_t i = 0; i < nq; ++i) {
      std::set<int64_t> truth;
      for (int t = 0; t < k; ++t) {
        const auto j = i * gt.cols + t;
        truth.insert(gt.dtype == "<i8" ? reinterpret_cast<const int64_t*>(gt.data.data())[j]
                                       : reinterpret_cast<const int32_t*>(gt.data.data())[j]);
      }
      for (int t = 0; t < k; ++t)
        hits += truth.count(ids[i * k + t]);
    }
    std::printf("recall@10: %.4f\n", double(hits) / double(nq * k));
  }
  return 0;
}
