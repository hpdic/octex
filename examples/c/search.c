/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

/* The C API, with the conventions of the cuVS C API: build an index directory of 200,000 clustered
 * vectors, load it on all visible GPUs, search the 10 nearest neighbors of 1,000 queries held in
 * device memory, and measure the recall@10 on a sample of queries against exhaustive search in
 * double precision.
 *
 *   cmake --build build --target octex_example_c && build/octex_example_c [INDEX_DIR]
 *
 * INDEX_DIR (default octex_c_index) is created by the build and left in place.
 */

#include <cuvs/neighbors/octex.h>

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                               \
  do {                                                            \
    if ((call) != CUVS_SUCCESS) {                                 \
      fprintf(stderr, "%s: %s\n", #call, cuvsGetLastErrorText()); \
      return 1;                                                   \
    }                                                             \
  } while (0)

/* A DLPack view of a row-major matrix in host (kDLCPU) or device (kDLCUDA) memory. */
static void matrix(
  DLManagedTensor* t, void* data, int64_t* shape, DLDeviceType where, uint8_t code, uint8_t bits)
{
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data               = data;
  t->dl_tensor.device.device_type = where;
  t->dl_tensor.ndim               = 2;
  t->dl_tensor.dtype.code         = code;
  t->dl_tensor.dtype.bits         = bits;
  t->dl_tensor.dtype.lanes        = 1;
  t->dl_tensor.shape              = shape;
}

static uint64_t rng = 88172645463325252ull;
static double uniform(void)
{
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return ((double)(rng >> 11) + 0.5) / 9007199254740992.0;
}
static float normal(void) /* Box-Muller */
{
  return (float)(sqrt(-2.0 * log(uniform())) * cos(6.283185307179586 * uniform()));
}

/* Vectors around one of 100 random centers. */
static void clustered(float* out, int64_t rows, int64_t d, const float* centers)
{
  for (int64_t i = 0; i < rows; ++i) {
    const int64_t c = (int64_t)(uniform() * 100) % 100;
    for (int64_t j = 0; j < d; ++j)
      out[i * d + j] = centers[c * d + j] + normal();
  }
}

int main(int argc, char** argv)
{
  const char* dir = argc > 1 ? argv[1] : "octex_c_index";
  const int64_t n = 200000, nq = 1000, d = 64, k = 10;
  float* centers = malloc(sizeof(float) * 100 * d);
  float* base    = malloc(sizeof(float) * n * d);
  float* query   = malloc(sizeof(float) * nq * d);
  for (int64_t i = 0; i < 100 * d; ++i)
    centers[i] = 4.0f * normal();
  clustered(base, n, d, centers);
  clustered(query, nq, d, centers);

  cuvsResources_t res;
  CHECK(cuvsResourcesCreate(&res));

  /* build: the dataset may live in host or device memory; the build reads it from the host */
  int64_t x_shape[2] = {n, d};
  DLManagedTensor x;
  matrix(&x, base, x_shape, kDLCPU, kDLFloat, 32);
  cuvsOctexIndexParams_t params; /* 200 copies, all visible GPUs */
  CHECK(cuvsOctexIndexParamsCreate(&params));
  CHECK(cuvsOctexBuild(res, params, &x, dir));

  /* load: all visible GPUs, copies sharded over them */
  cuvsOctexIndex_t index;
  CHECK(cuvsOctexIndexCreate(&index));
  CHECK(cuvsOctexLoad(res, dir, NULL, index));
  int64_t size   = 0;
  int32_t copies = 0, n_devices = 0;
  CHECK(cuvsOctexIndexGetSize(index, &size));
  CHECK(cuvsOctexIndexGetCopies(index, &copies));
  CHECK(cuvsOctexIndexGetNumDevices(index, &n_devices));
  printf("loaded %ld vectors, %d copies, on %d GPU(s)\n", (long)size, copies, n_devices);

  /* search: queries and results in device memory */
  float *dq, *dd;
  int64_t* dn;
  cudaMalloc((void**)&dq, sizeof(float) * nq * d);
  cudaMalloc((void**)&dn, sizeof(int64_t) * nq * k);
  cudaMalloc((void**)&dd, sizeof(float) * nq * k);
  cudaMemcpy(dq, query, sizeof(float) * nq * d, cudaMemcpyHostToDevice);
  int64_t q_shape[2] = {nq, d}, o_shape[2] = {nq, k};
  DLManagedTensor q, nb, ds;
  matrix(&q, dq, q_shape, kDLCUDA, kDLFloat, 32);
  matrix(&nb, dn, o_shape, kDLCUDA, kDLInt, 64);
  matrix(&ds, dd, o_shape, kDLCUDA, kDLFloat, 32);

  cuvsOctexSearchParams_t search_params; /* read 32, starts 128, waves 2 */
  CHECK(cuvsOctexSearchParamsCreate(&search_params));
  CHECK(cuvsOctexSearch(res, search_params, index, &q, &nb, &ds));
  CHECK(cuvsStreamSync(res));

  int64_t* ids = malloc(sizeof(int64_t) * nq * k);
  cudaMemcpy(ids, dn, sizeof(int64_t) * nq * k, cudaMemcpyDeviceToHost);

  /* recall@10 on a sample: the 10 smallest exact distances by a partial selection */
  int64_t hits = 0, checked = 0;
  double* all = malloc(sizeof(double) * n);
  for (int64_t i = 0; i < nq; i += 20, ++checked) {
    for (int64_t j = 0; j < n; ++j) {
      double s = 0;
      for (int64_t c = 0; c < d; ++c) {
        const double t = (double)query[i * d + c] - (double)base[j * d + c];
        s += t * t;
      }
      all[j] = s;
    }
    int64_t truth[10];
    for (int64_t t = 0; t < k; ++t) {
      int64_t best = -1;
      for (int64_t j = 0; j < n; ++j)
        if (all[j] >= 0 && (best < 0 || all[j] < all[best])) best = j;
      truth[t]  = best;
      all[best] = -1;
    }
    for (int64_t t = 0; t < k; ++t)
      for (int64_t u = 0; u < k; ++u)
        hits += ids[i * k + t] == truth[u];
  }
  printf("nearest neighbors of query 0:");
  for (int64_t t = 0; t < k; ++t)
    printf(" %ld", (long)ids[t]);
  printf("\nrecall@10 on %ld sampled queries: %.4f\n",
         (long)checked,
         (double)hits / (double)(checked * k));

  CHECK(cuvsOctexSearchParamsDestroy(search_params));
  CHECK(cuvsOctexIndexDestroy(index));
  CHECK(cuvsOctexIndexParamsDestroy(params));
  CHECK(cuvsResourcesDestroy(res));
  cudaFree(dq);
  cudaFree(dn);
  cudaFree(dd);
  free(centers);
  free(base);
  free(query);
  free(ids);
  free(all);
  return 0;
}
