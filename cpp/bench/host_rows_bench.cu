/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Throughput of GPU reads of random rows from page-locked host memory (anonymous, cudaHostAlloc)
// against the same reads from HBM: the cost model behind keeping the rows that the waves read in
// HBM.
//
//   host_rows_bench [--rows-gib 16] [--row-bytes 128] [--reads 320000,10000000] [--devices 0]
//   [--reps 5]
//
// One warp reads one row with 16-byte loads; every GPU of --devices reads its own random rows at
// the same time.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK_CUDA(x)                                                                 \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

namespace {

__global__ void read_rows(const int4* __restrict__ base,
                          const int64_t* __restrict__ ids,
                          int64_t n,
                          int words,
                          unsigned long long* __restrict__ sink)
{
  const int lane   = threadIdx.x & 31;
  int64_t w        = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int64_t nw = (int64_t(gridDim.x) * blockDim.x) >> 5;
  unsigned acc     = 0;
  for (; w < n; w += nw) {
    const int4* row = base + ids[w] * words;
    for (int i = lane; i < words; i += 32) {
      int4 v = row[i];
      acc ^= unsigned(v.x) ^ unsigned(v.y) ^ unsigned(v.z) ^ unsigned(v.w);
    }
  }
  if (acc == 0x12345678u) atomicAdd(sink, 1ull);  // keeps the loads alive
}

auto ints(std::string const& s) -> std::vector<int64_t>
{
  std::vector<int64_t> out;
  size_t p = 0;
  while (p < s.size()) {
    size_t e = s.find(',', p);
    if (e == std::string::npos) e = s.size();
    out.push_back(std::atoll(s.substr(p, e - p).c_str()));
    p = e + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv)
{
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2)
    opt[std::string(argv[i]).substr(2)] = argv[i + 1];
  auto get = [&](std::string const& k, std::string const& d) { return opt.count(k) ? opt[k] : d; };
  const double gib     = std::atof(get("rows-gib", "16").c_str());
  const int row_bytes  = std::atoi(get("row-bytes", "128").c_str());
  const auto reads     = ints(get("reads", "320000,10000000"));
  const auto devices   = ints(get("devices", "0"));
  const int reps       = std::atoi(get("reps", "5").c_str());
  const int words      = row_bytes / 16;
  const int64_t nrows  = int64_t(gib * double(1ull << 30)) / row_bytes;
  const int64_t max_rd = *std::max_element(reads.begin(), reads.end());

  // anonymous page-locked rows, mapped for every GPU
  void* host = nullptr;
  CHECK_CUDA(
    cudaHostAlloc(&host, size_t(nrows) * row_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
  std::memset(host, 1, size_t(nrows) * row_bytes);
  std::printf(
    "rows: %lld x %d B (%.1f GiB) in page-locked host memory\n", (long long)nrows, row_bytes, gib);

  struct dev {
    int id;
    int64_t* ids;
    int4* hbm;
    unsigned long long* sink;
    cudaStream_t s;
  };
  std::vector<dev> ds;
  std::mt19937_64 rng(1);
  std::vector<int64_t> hid(max_rd);
  for (auto d : devices) {
    dev x{int(d)};
    CHECK_CUDA(cudaSetDevice(x.id));
    CHECK_CUDA(cudaStreamCreate(&x.s));
    for (auto& v : hid)
      v = int64_t(rng() % uint64_t(nrows));
    CHECK_CUDA(cudaMalloc(&x.ids, max_rd * sizeof(int64_t)));
    CHECK_CUDA(cudaMemcpy(x.ids, hid.data(), max_rd * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&x.hbm, size_t(nrows) * row_bytes));
    CHECK_CUDA(cudaMemset(x.hbm, 1, size_t(nrows) * row_bytes));
    CHECK_CUDA(cudaMalloc(&x.sink, sizeof(unsigned long long)));
    ds.push_back(x);
  }
  for (int where = 0; where < 2; where++) {
    for (auto n : reads) {
      double best = 1e30;
      for (int r = 0; r < reps + 1; r++) {
        for (auto& x : ds) {
          CHECK_CUDA(cudaSetDevice(x.id));
          CHECK_CUDA(cudaDeviceSynchronize());
        }
        auto t0 = std::chrono::steady_clock::now();
        for (auto& x : ds) {
          CHECK_CUDA(cudaSetDevice(x.id));
          const int4* base = where == 0 ? static_cast<const int4*>(host) : x.hbm;
          read_rows<<<1056, 256, 0, x.s>>>(base, x.ids, n, words, x.sink);
        }
        for (auto& x : ds) {
          CHECK_CUDA(cudaSetDevice(x.id));
          CHECK_CUDA(cudaStreamSynchronize(x.s));
        }
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r > 0) best = std::min(best, t);  // the first round warms up
      }
      double per_gpu = double(n) / best;
      std::printf(
        "%s, %zu GPU(s), %lld random %d-byte rows per GPU: %.2f M rows/s per GPU (%.2f GB/s)\n",
        where == 0 ? "host" : "HBM ",
        ds.size(),
        (long long)n,
        row_bytes,
        per_gpu / 1e6,
        per_gpu * row_bytes / 1e9);
    }
  }
  for (auto& x : ds) {
    CHECK_CUDA(cudaSetDevice(x.id));
    cudaFree(x.ids);
    cudaFree(x.hbm);
    cudaFree(x.sink);
  }
  cudaFreeHost(host);
  return 0;
}
