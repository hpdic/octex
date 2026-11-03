/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// A plain C caller of the octex C API (compiled as C): build an index directory from a float32 or
// float16 dataset in host or device memory, load it on GPU 0, check the getters, and search host or
// device queries.

#include <cuvs/neighbors/octex.h>

#include <stdint.h>
#include <stdio.h>

static void matrix(
  DLManagedTensor* t, void* data, int64_t* shape, int device, uint8_t code, uint8_t bits)
{
  t->dl_tensor.data               = data;
  t->dl_tensor.device.device_type = device ? kDLCUDA : kDLCPU;
  t->dl_tensor.device.device_id   = 0;
  t->dl_tensor.ndim               = 2;
  t->dl_tensor.dtype.code         = code;
  t->dl_tensor.dtype.bits         = bits;
  t->dl_tensor.dtype.lanes        = 1;
  t->dl_tensor.shape              = shape;
  t->dl_tensor.strides            = NULL;
  t->dl_tensor.byte_offset        = 0;
  t->manager_ctx                  = NULL;
  t->deleter                      = NULL;
}

#define CHECK(call)                                               \
  do {                                                            \
    if ((call) != CUVS_SUCCESS) {                                 \
      fprintf(stderr, "%s: %s\n", #call, cuvsGetLastErrorText()); \
      return 1;                                                   \
    }                                                             \
  } while (0)

int run_octex(int64_t n_rows,
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
              float* distances)
{
  cuvsResources_t res;
  CHECK(cuvsResourcesCreate(&res));

  int64_t x_shape[2] = {n_rows, dim}, q_shape[2] = {n_queries, dim}, o_shape[2] = {n_queries, k};
  DLManagedTensor x, q, nb, ds;
  matrix(&x, dataset, x_shape, dataset_on_device, kDLFloat, dataset_half ? 16 : 32);
  matrix(&q, queries, q_shape, search_on_device, kDLFloat, 32);
  matrix(&nb, neighbors, o_shape, search_on_device, kDLInt, 64);
  matrix(&ds, distances, o_shape, search_on_device, kDLFloat, 32);

  int32_t gpu0[1] = {0};
  cuvsOctexIndexParams_t ip;
  CHECK(cuvsOctexIndexParamsCreate(&ip));
  ip->copies    = copies;
  ip->seed      = 1;
  ip->devices   = gpu0;
  ip->n_devices = 1;
  CHECK(cuvsOctexBuild(res, ip, &x, path));

  cuvsOctexLoadParams_t lp;
  CHECK(cuvsOctexLoadParamsCreate(&lp));
  lp->devices   = gpu0;
  lp->n_devices = 1;
  lp->placement = (cuvsOctexPlacement)placement;
  cuvsOctexIndex_t index;
  CHECK(cuvsOctexIndexCreate(&index));
  CHECK(cuvsOctexLoad(res, path, lp, index));

  int64_t size = 0, d = 0, bytes = 0;
  int32_t c = 0, n_devices = 0, device = -1;
  CHECK(cuvsOctexIndexGetSize(index, &size));
  CHECK(cuvsOctexIndexGetDim(index, &d));
  CHECK(cuvsOctexIndexGetCopies(index, &c));
  CHECK(cuvsOctexIndexGetNumDevices(index, &n_devices));
  if (size != n_rows || d != dim || c != copies || n_devices != 1) { return 2; }
  CHECK(cuvsOctexIndexGetDevices(index, &device));
  CHECK(cuvsOctexIndexGetDeviceBytes(index, &bytes));
  if (device != 0 || bytes <= 0) { return 2; }

  cuvsOctexSearchParams_t sp;
  CHECK(cuvsOctexSearchParamsCreate(&sp));
  CHECK(cuvsOctexSearch(res, sp, index, &q, &nb, &ds));
  CHECK(cuvsStreamSync(res));

  CHECK(cuvsOctexSearchParamsDestroy(sp));
  CHECK(cuvsOctexIndexDestroy(index));
  CHECK(cuvsOctexLoadParamsDestroy(lp));
  CHECK(cuvsOctexIndexParamsDestroy(ip));
  CHECK(cuvsResourcesDestroy(res));
  return 0;
}
