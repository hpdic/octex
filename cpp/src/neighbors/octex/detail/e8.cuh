/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::octex::detail {

// primes of the per-block hash and the 64-bit mixer; builder and search share them
__constant__ long long kPrimes[8] = {1000003LL,
                                     998244353LL,
                                     1000000007LL,
                                     2147483647LL,
                                     1099511627791LL,
                                     4398042316799LL,
                                     70368744177643LL,
                                     1125899906842597LL};
constexpr unsigned long long kMix = 0x9E3779B97F4A7C15ULL;

/** Nearest point of D8 (integer vectors with an even sum). */
__device__ __forceinline__ void d8(const float* x, float* f)
{
  float s  = 0.f;
  int im   = 0;
  float em = -1.f, ei = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    f[i] = rintf(x[i]);
    s += f[i];
    float e = x[i] - f[i];
    if (fabsf(e) > em) {
      em = fabsf(e);
      im = i;
      ei = e;
    }
  }
  if (fmodf(s, 2.f) != 0.f) f[im] += (ei >= 0.f ? 1.f : -1.f);
}

/** Nearest point p of E8 to the eight coordinates x (p in Z/2). */
__device__ __forceinline__ void e8_point(const float* x, float* p)
{
  float a[8], c[8], xs[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    xs[i] = x[i] - 0.5f;
  d8(x, a);
  d8(xs, c);
  float da = 0.f, dc = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    c[i] += 0.5f;
    da += (x[i] - a[i]) * (x[i] - a[i]);
    dc += (x[i] - c[i]) * (x[i] - c[i]);
  }
#pragma unroll
  for (int i = 0; i < 8; i++)
    p[i] = da <= dc ? a[i] : c[i];
}

/** 64-bit hash of the nearest point of E8^b to y * inv_f (y: 8b coordinates). */
__device__ __forceinline__ long long e8b_hash64(const float* y, int b, float inv_f)
{
  unsigned long long h = 0;
  for (int j = 0; j < b; j++) {
    float x[8], p[8];
#pragma unroll
    for (int i = 0; i < 8; i++)
      x[i] = y[8 * j + i] * inv_f;
    e8_point(x, p);
    unsigned long long hb = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      long long p2 = (long long)(2.f * p[i]);
      hb += (unsigned long long)p2 * (unsigned long long)kPrimes[i];
    }
    h = (j == 0) ? hb : h * kMix + hb;
  }
  long long hs = (long long)h;
  hs           = (long long)((unsigned long long)(hs ^ (hs >> 31)) * kMix);
  return hs ^ (hs >> 29);
}

}  // namespace cuvs::neighbors::octex::detail
