/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The octex C API on a small Gaussian mixture (20,000 x 64, 8 copies, GPU 0) against exhaustive
// search in double precision: device and host datasets, float32 and float16, device and host
// queries, shard and replicate placement. Then the defaults of the parameter structs and the errors
// of the calls.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuvs/neighbors/octex.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

extern "C" int run_octex(int64_t n_rows,
                         int64_t n_queries,
                         int64_t dim,
                         int64_t k,
                         void* dataset,
                         int32_t dataset_on_device,
                         int32_t dataset_half,
                         float* queries,
                         int32_t search_on_device,
                         int32_t copies,
                         int32_t placement,
                         const char* path,
                         int64_t* neighbors,
                         float* distances);

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

constexpr int64_t n = 20000, nq = 200, d = 64, k = 10;
constexpr int32_t copies = 8;

struct Case {
  const char* name;
  bool dataset_on_device, dataset_half, search_on_device;
  cuvsOctexPlacement placement;
};

/** A Gaussian mixture of 50 clusters (centers scaled by 4), its queries, as in tests/conftest.py.
 */
struct Data {
  std::vector<float> x, q;
  Data() : x(n * d), q(nq * d)
  {
    std::mt19937 gen(0);
    std::normal_distribution<float> normal;
    std::uniform_int_distribution<int> pick(0, 49);
    std::vector<float> centers(50 * d);
    for (auto& v : centers)
      v = 4.0f * normal(gen);
    for (int64_t i = 0; i < n; ++i) {
      const int c = pick(gen);
      for (int64_t j = 0; j < d; ++j)
        x[i * d + j] = centers[c * d + j] + normal(gen);
    }
    for (int64_t i = 0; i < nq; ++i) {
      const int c = pick(gen);
      for (int64_t j = 0; j < d; ++j)
        q[i * d + j] = centers[c * d + j] + normal(gen);
    }
  }
};

auto dist(Data const& data, std::vector<float> const& x, int64_t qi, int64_t xi) -> double
{
  double s = 0;
  for (int64_t j = 0; j < d; ++j) {
    const double t = double(data.q[qi * d + j]) - double(x[xi * d + j]);
    s += t * t;
  }
  return s;
}

void check(Data const& data, Case c)
{
  // the dataset the index sees: float16 rounds the vectors
  std::vector<float> x = data.x;
  std::vector<__half> xh;
  if (c.dataset_half) {
    xh.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      xh[i] = __float2half(x[i]);
      x[i]  = __half2float(xh[i]);
    }
  }
  const size_t x_bytes = c.dataset_half ? xh.size() * 2 : x.size() * 4;
  const void* x_host   = c.dataset_half ? static_cast<const void*>(xh.data()) : x.data();

  void* dx = const_cast<void*>(x_host);
  if (c.dataset_on_device) {
    CK(cudaMalloc(&dx, x_bytes));
    CK(cudaMemcpy(dx, x_host, x_bytes, cudaMemcpyHostToDevice));
  }
  std::vector<float> q = data.q, dist_out(nq * k);
  std::vector<int64_t> ids(nq * k, -1);
  float* dq   = q.data();
  float* dd   = dist_out.data();
  int64_t* dn = ids.data();
  if (c.search_on_device) {
    CK(cudaMalloc(&dq, q.size() * 4));
    CK(cudaMalloc(&dd, nq * k * 4));
    CK(cudaMalloc(&dn, nq * k * 8));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
  }

  const char* tmp        = std::getenv("TMPDIR");
  const std::string path = std::string(tmp != nullptr ? tmp : "/tmp") + "/octex_c_test_" + c.name;
  std::filesystem::remove_all(path);
  const int rc = run_octex(n,
                           nq,
                           d,
                           k,
                           dx,
                           c.dataset_on_device,
                           c.dataset_half,
                           dq,
                           c.search_on_device,
                           copies,
                           c.placement,
                           path.c_str(),
                           dn,
                           dd);
  EXPECT(rc == 0, "%s: run_octex returned %d", c.name, rc);
  if (c.search_on_device) {
    CK(cudaMemcpy(ids.data(), dn, ids.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dist_out.data(), dd, dist_out.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(dq));
    CK(cudaFree(dd));
    CK(cudaFree(dn));
  }
  if (c.dataset_on_device) { CK(cudaFree(dx)); }
  std::filesystem::remove_all(path);
  if (rc != 0) { return; }

  // recall@10 against exhaustive search, and the returned distances against exact ones
  int64_t hits = 0, bad_dist = 0, bad_id = 0;
  std::vector<std::pair<double, int64_t>> all(n);
  for (int64_t qi = 0; qi < nq; ++qi) {
    for (int64_t i = 0; i < n; ++i)
      all[i] = {dist(data, x, qi, i), i};
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    std::set<int64_t> truth;
    for (int64_t j = 0; j < k; ++j)
      truth.insert(all[j].second);
    for (int64_t j = 0; j < k; ++j) {
      const int64_t id = ids[qi * k + j];
      if (id < 0 || id >= n) {
        bad_id++;
        continue;
      }
      hits += truth.count(id);
      const double e = dist(data, x, qi, id);
      if (std::abs(dist_out[qi * k + j] - e) > 1e-2 * e + 0.5) { bad_dist++; }
    }
  }
  const double recall = double(hits) / double(nq * k);
  std::printf("%-24s recall@10 %.4f\n", c.name, recall);
  EXPECT(bad_id == 0, "%s: %ld ids out of range", c.name, long(bad_id));
  EXPECT(bad_dist == 0, "%s: %ld distances differ from the exact ones", c.name, long(bad_dist));
  EXPECT(recall >= 0.9, "%s: recall@10 %.4f < 0.9", c.name, recall);
}

/** Parameter defaults and errors that need no GPU: they fail before any CUDA call. */
void check_api()
{
  cuvsOctexIndexParams_t ip;
  EXPECT(cuvsOctexIndexParamsCreate(&ip) == CUVS_SUCCESS, "index params");
  EXPECT(ip->copies == 200 && ip->blocks == 4 && ip->list_len == 64 && ip->table_len == 64 &&
           ip->reverse_edges && ip->spacings == nullptr && ip->devices == nullptr,
         "index params defaults");
  cuvsOctexLoadParams_t lp;
  EXPECT(cuvsOctexLoadParamsCreate(&lp) == CUVS_SUCCESS, "load params");
  EXPECT(lp->placement == OCTEX_PLACEMENT_SHARD && lp->sketch == OCTEX_SKETCH_REPLICATE &&
           !lp->vectors_on_host,
         "load params defaults");
  cuvsOctexSearchParams_t sp;
  EXPECT(cuvsOctexSearchParamsCreate(&sp) == CUVS_SUCCESS, "search params");
  EXPECT(sp->read == 32 && sp->starts == 128 && sp->waves == 2 && sp->batch == 10000,
         "search params defaults");

  cuvsOctexIndex_t index;
  EXPECT(cuvsOctexIndexCreate(&index) == CUVS_SUCCESS, "index create");
  int64_t size = 0;
  EXPECT(cuvsOctexIndexGetSize(index, &size) == CUVS_ERROR, "the size of an empty index");
  EXPECT(std::strstr(cuvsGetLastErrorText(), "empty") != nullptr,
         "error text: %s",
         cuvsGetLastErrorText());

  double x64[4]      = {};
  int64_t shape[2]   = {2, 2};
  DLManagedTensor x  = {};
  x.dl_tensor.data   = x64;
  x.dl_tensor.device = {kDLCPU, 0};
  x.dl_tensor.ndim   = 2;
  x.dl_tensor.dtype  = {kDLFloat, 64, 1};
  x.dl_tensor.shape  = shape;
  EXPECT(cuvsOctexBuild(0, ip, &x, "unused") == CUVS_ERROR, "a float64 dataset");
  EXPECT(std::strstr(cuvsGetLastErrorText(), "float32 or float16") != nullptr,
         "error text: %s",
         cuvsGetLastErrorText());
  EXPECT(cuvsOctexBuild(0, ip, &x, nullptr) == CUVS_ERROR, "a NULL directory");

  lp->placement = static_cast<cuvsOctexPlacement>(7);
  EXPECT(cuvsOctexLoad(0, "unused", lp, index) == CUVS_ERROR, "an unknown placement");
  EXPECT(std::strstr(cuvsGetLastErrorText(), "placement") != nullptr,
         "error text: %s",
         cuvsGetLastErrorText());

  EXPECT(cuvsOctexIndexDestroy(index) == CUVS_SUCCESS, "index destroy");
  EXPECT(cuvsOctexSearchParamsDestroy(sp) == CUVS_SUCCESS, "search params destroy");
  EXPECT(cuvsOctexLoadParamsDestroy(lp) == CUVS_SUCCESS, "load params destroy");
  EXPECT(cuvsOctexIndexParamsDestroy(ip) == CUVS_SUCCESS, "index params destroy");
}

}  // namespace

int main()
{
  check_api();
  // OCTEX_C_API_ONLY=1: the parameter and error checks only, without a GPU
  const char* api_only = std::getenv("OCTEX_C_API_ONLY");
  if (api_only == nullptr || std::strcmp(api_only, "1") != 0) {
    const Data data;
    const Case cases[] = {
      {"device_f32_device", true, false, true, OCTEX_PLACEMENT_SHARD},
      {"host_f32_host", false, false, false, OCTEX_PLACEMENT_SHARD},
      {"host_f16_device", false, true, true, OCTEX_PLACEMENT_REPLICATE},
      {"device_f16_host", true, true, false, OCTEX_PLACEMENT_REPLICATE},
    };
    for (auto const& c : cases)
      check(data, c);
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
  return failures ? 1 : 0;
}
