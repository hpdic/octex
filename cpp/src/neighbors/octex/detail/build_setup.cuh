/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Mean and principal directions from a sample of the dataset, the random projections of the copies,
// and the int8 sketch.

#include "build_cells.cuh"
#include "build_common.cuh"

#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/linalg/eig.cuh>

#include <rmm/device_uvector.hpp>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace cuvs::neighbors::octex::detail {

constexpr int64_t kSampleRows  = 200000;
constexpr int64_t kSketchChunk = int64_t(1) << 20;

struct pca_result {
  std::vector<float> mean;     // [d]
  std::vector<float> vectors;  // [d, d] row major: column k is the k-th principal direction
  std::vector<double> values;  // [d] eigenvalues of the sample scatter matrix, descending
  int64_t samples = 0;
};

/** Mean and principal directions of every max(1, n / 200000)-th row (fp64 scatter matrix,
 * cuSOLVER). */
inline auto sample_pca(raft::resources const& res, const half* x, int64_t n, int64_t d)
  -> pca_result
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  int64_t step        = std::max<int64_t>(1, n / kSampleRows);
  int64_t ns          = (n + step - 1) / step;
  pca_result out;
  out.samples = ns;
  std::vector<double> mu(d, 0.0);
  for (int64_t i = 0; i < ns; i++)
    for (int64_t k = 0; k < d; k++)
      mu[k] += __half2float(x[i * step * d + k]);
  out.mean.resize(d);
  for (int64_t k = 0; k < d; k++)
    out.mean[k] = static_cast<float>(mu[k] / double(ns));
  std::vector<double> a(ns * d);
  for (int64_t i = 0; i < ns; i++)
    for (int64_t k = 0; k < d; k++)
      a[i * d + k] = double(__half2float(x[i * step * d + k]) - out.mean[k]);

  rmm::device_uvector<double> da(ns * d, stream), g(d * d, stream), vec(d * d, stream),
    val(d, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    da.data(), a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
  auto handle = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  const double one = 1.0, zero = 0.0;
  // g = a^T a: a row major [ns, d] is column major [d, ns]
  RAFT_CUBLAS_TRY(cublasDgemm(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_T,
                              static_cast<int>(d),
                              static_cast<int>(d),
                              static_cast<int>(ns),
                              &one,
                              da.data(),
                              static_cast<int>(d),
                              da.data(),
                              static_cast<int>(d),
                              &zero,
                              g.data(),
                              static_cast<int>(d)));
  raft::linalg::eig_dc(
    res,
    raft::make_device_matrix_view<const double, int64_t, raft::col_major>(g.data(), d, d),
    raft::make_device_matrix_view<double, int64_t, raft::col_major>(vec.data(), d, d),
    raft::make_device_vector_view<double, int64_t>(val.data(), d));
  std::vector<double> hv(d * d), hval(d);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(hv.data(), vec.data(), d * d * sizeof(double), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(hval.data(), val.data(), d * sizeof(double), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);
  // ascending eigenvalues, column major vectors -> descending, row major [d, d]
  out.vectors.resize(d * d);
  out.values.resize(d);
  for (int64_t k = 0; k < d; k++) {
    int64_t src   = d - 1 - k;
    out.values[k] = hval[src];
    for (int64_t r = 0; r < d; r++)
      out.vectors[r * d + k] = static_cast<float>(hv[src * d + r]);
  }
  return out;
}

inline auto splitmix64(uint64_t z) -> uint64_t
{
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/**
 * Projections [C, d, 8b] and shifts [C, 8b]: copy c rotates the top-p principal subspace into 8b
 * dimensions by the Q factor of a Gaussian [p, 8b] matrix and divides by the mean projected
 * standard deviation; its shift is uniform in [0, 2). Copy c draws from its own generator (seed,
 * c).
 */
inline void make_projections(pca_result const& pca,
                             int64_t d,
                             int p,
                             int w,
                             int copies,
                             uint64_t seed,
                             std::vector<float>& proj,
                             std::vector<float>& shift)
{
  RAFT_EXPECTS(p >= w, "octex: pca_dims must be at least 8 * blocks");
  RAFT_EXPECTS(p <= d, "octex: pca_dims exceeds the dimension");
  proj.assign(int64_t(copies) * d * w, 0.f);
  shift.assign(int64_t(copies) * w, 0.f);
  std::vector<double> q(int64_t(p) * w);
  for (int c = 0; c < copies; c++) {
    std::mt19937_64 gen(splitmix64(seed * 0x100000001B3ULL + uint64_t(c)));
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 2.0);
    for (auto& v : q)
      v = normal(gen);
    // modified Gram-Schmidt on the columns of q [p, w]
    for (int s = 0; s < w; s++) {
      for (int t = 0; t < s; t++) {
        double dot = 0;
        for (int r = 0; r < p; r++)
          dot += q[r * w + s] * q[r * w + t];
        for (int r = 0; r < p; r++)
          q[r * w + s] -= dot * q[r * w + t];
      }
      double nrm = 0;
      for (int r = 0; r < p; r++)
        nrm += q[r * w + s] * q[r * w + s];
      nrm = std::sqrt(nrm);
      for (int r = 0; r < p; r++)
        q[r * w + s] /= nrm;
    }
    // projected sample standard deviation of column s: sqrt(sum_r lambda_r q[r, s]^2 / (ns - 1))
    double sd = 0;
    for (int s = 0; s < w; s++) {
      double v = 0;
      for (int r = 0; r < p; r++)
        v += pca.values[r] * q[r * w + s] * q[r * w + s];
      sd += std::sqrt(v / double(std::max<int64_t>(pca.samples - 1, 1)));
    }
    sd /= w;
    float* u = proj.data() + int64_t(c) * d * w;
    for (int64_t k = 0; k < d; k++)
      for (int s = 0; s < w; s++) {
        double v = 0;
        for (int r = 0; r < p; r++)
          v += double(pca.vectors[k * d + r]) * q[r * w + s];
        u[k * w + s] = static_cast<float>(v / sd);
      }
    for (int s = 0; s < w; s++)
      shift[int64_t(c) * w + s] = static_cast<float>(uni(gen));
  }
}

static __global__ void amax_kernel(const float* v, int64_t n, int s, unsigned* amax)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n * s) atomicMax(amax + i % s, __float_as_uint(fabsf(v[i])));
}

static __global__ void quantize_kernel(
  const float* v, const float* scale, int64_t n, int s, int8_t* out)
{
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= n * s) return;
  float q = rintf(v[i] / scale[i % s]);  // round half to even, as torch.round
  out[i]  = static_cast<int8_t>(fminf(127.f, fmaxf(-127.f, q)));
}

/** codes = round(((x - mean) @ basis) / scale), scale = amax / 127 over the dataset (two passes).
 */
inline void make_sketch(raft::resources const& res,
                        const half* x,
                        int64_t n,
                        int64_t d,
                        const float* mean,
                        const float* basis,  // [d, s]
                        int s,
                        std::vector<float>& scale,
                        int8_t* codes)  // host [n, s]
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto handle         = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  RAFT_CUBLAS_TRY(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  device_rows rows(x, n, d, kSketchChunk, device_free_bytes() / 3, stream);
  int64_t chunk = std::min(kSketchChunk, n);
  rmm::device_uvector<float> d_mean(d, stream), d_basis(d * s, stream), a(chunk * d, stream),
    v(chunk * s, stream);
  rmm::device_uvector<unsigned> amax(s, stream);
  rmm::device_uvector<float> d_scale(s, stream);
  rmm::device_uvector<int8_t> q(chunk * s, stream);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_mean.data(), mean, d * sizeof(float), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_basis.data(), basis, d * s * sizeof(float), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(amax.data(), 0, s * sizeof(unsigned), stream));
  const float one = 1.f, zero = 0.f;
  auto project = [&](int64_t r0, int64_t m) {
    center(rows.rows(r0, m), d_mean.data(), a.data(), m, d, stream);
    RAFT_CUBLAS_TRY(cublasSgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                s,
                                static_cast<int>(m),
                                static_cast<int>(d),
                                &one,
                                d_basis.data(),
                                s,
                                a.data(),
                                static_cast<int>(d),
                                &zero,
                                v.data(),
                                s));
  };
  for (int64_t r0 = 0; r0 < n; r0 += chunk) {
    int64_t m = std::min(chunk, n - r0);
    project(r0, m);
    amax_kernel<<<grid(m * s), 256, 0, stream>>>(v.data(), m, s, amax.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  std::vector<unsigned> ha(s);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(ha.data(), amax.data(), s * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);
  scale.resize(s);
  for (int k = 0; k < s; k++) {
    float am;
    std::memcpy(&am, &ha[k], sizeof(float));
    scale[k] = am > 0.f ? am / 127.f : 1.f;
  }
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_scale.data(), scale.data(), s * sizeof(float), cudaMemcpyHostToDevice, stream));
  for (int64_t r0 = 0; r0 < n; r0 += chunk) {
    int64_t m = std::min(chunk, n - r0);
    project(r0, m);
    quantize_kernel<<<grid(m * s), 256, 0, stream>>>(v.data(), d_scale.data(), m, s, q.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cudaMemcpyAsync(codes + r0 * s, q.data(), m * s, cudaMemcpyDeviceToHost, stream));
    raft::resource::sync_stream(res);
  }
}

}  // namespace cuvs::neighbors::octex::detail
