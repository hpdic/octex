/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "detail/index_impl.hpp"
#include "detail/npy.hpp"

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/error.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

namespace cuvs::neighbors::octex {

namespace {

using detail::dbuf;
using detail::device_part;
using detail::hbuf;
using detail::host_array;
using detail::index_impl;
using detail::npy_file;

// ---- meta.json: a flat object of numbers, one array of numbers and one map of strings

auto read_text(std::string const& path) -> std::string
{
  std::ifstream f(path);
  RAFT_EXPECTS(f.good(), "octex: cannot open %s", path.c_str());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

auto json_value_pos(std::string const& s, std::string const& key) -> size_t
{
  auto p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return p;
  p = s.find(':', p);
  return p == std::string::npos ? p : p + 1;
}

auto json_number(std::string const& s, std::string const& key, double dflt) -> double
{
  auto p = json_value_pos(s, key);
  if (p == std::string::npos) return dflt;
  return std::strtod(s.c_str() + p, nullptr);
}

auto json_numbers(std::string const& s, std::string const& key) -> std::vector<double>
{
  std::vector<double> out;
  auto p = json_value_pos(s, key);
  if (p == std::string::npos) return out;
  auto a = s.find('[', p), b = s.find(']', a);
  std::string body = s.substr(a + 1, b - a - 1);
  const char* c    = body.c_str();
  char* end        = nullptr;
  while (*c != 0) {
    double v = std::strtod(c, &end);
    if (end == c) {
      c++;
      continue;
    }
    out.push_back(v);
    c = end;
  }
  return out;
}

auto json_strings(std::string const& s, std::string const& key)
  -> std::map<std::string, std::string>
{
  std::map<std::string, std::string> out;
  auto p = json_value_pos(s, key);
  if (p == std::string::npos) return out;
  auto a = s.find('{', p), b = s.find('}', a);
  size_t i = a + 1;
  while (true) {
    auto k1 = s.find('"', i);
    if (k1 == std::string::npos || k1 > b) break;
    auto k2                            = s.find('"', k1 + 1);
    auto v1                            = s.find('"', s.find(':', k2) + 1);
    auto v2                            = s.find('"', v1 + 1);
    out[s.substr(k1 + 1, k2 - k1 - 1)] = s.substr(v1 + 1, v2 - v1 - 1);
    i                                  = v2 + 1;
  }
  return out;
}

// ---- host to device copies through two pinned staging buffers

struct stager {
  static constexpr size_t kChunk = size_t{64} << 20;
  int device;
  cudaStream_t stream;
  hbuf buf[2];
  cudaEvent_t ev[2];
  int next = 0;

  stager(int dev, cudaStream_t s) : device(dev), stream(s), buf{hbuf(kChunk), hbuf(kChunk)}
  {
    RAFT_CUDA_TRY(cudaSetDevice(dev));
    for (auto& e : ev) {
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      RAFT_CUDA_TRY(cudaEventRecord(e, stream));
    }
  }
  ~stager()
  {
    cudaSetDevice(device);
    cudaStreamSynchronize(stream);
    for (auto& e : ev)
      cudaEventDestroy(e);
  }

  void copy(void* dst, const void* src, size_t bytes)
  {
    RAFT_CUDA_TRY(cudaSetDevice(device));
    auto* d       = static_cast<char*>(dst);
    auto const* s = static_cast<const char*>(src);
    for (size_t o = 0; o < bytes; o += kChunk) {
      size_t len = std::min(kChunk, bytes - o);
      RAFT_CUDA_TRY(cudaEventSynchronize(ev[next]));
      std::memcpy(buf[next].ptr, s + o, len);
      RAFT_CUDA_TRY(cudaMemcpyAsync(d + o, buf[next].ptr, len, cudaMemcpyHostToDevice, stream));
      RAFT_CUDA_TRY(cudaEventRecord(ev[next], stream));
      next ^= 1;
    }
  }
};

auto upload(device_part& p, stager& st, const void* src, size_t bytes) -> dbuf
{
  dbuf b(p.device, bytes);
  st.copy(b.ptr, src, bytes);
  p.bytes += static_cast<int64_t>(bytes);
  return b;
}

/**
 * An array for kernels to read in place from host memory: an anonymous pinned buffer
 * (cudaHostAlloc) filled in sequential chunks via ordinary buffered file I/O.
 * Never maps or registers file pages with CUDA.
 */
auto map_host(std::string const& path) -> host_array
{
  host_array h;
  if (auto m = detail::find_mem_array(path)) {  // held in memory by the build
    if (m->pinned) {                            // already page-locked by its owner: read in place
      h.ptr        = m->data;
      h.registered = false;
      return h;
    }
    h.copy = hbuf(m->bytes);
    detail::parallel_memcpy(h.copy.ptr, m->data, m->bytes);
    h.ptr        = h.copy.ptr;
    h.registered = false;
    return h;
  }
  auto hdr = detail::read_npy_header(path);
  h.copy   = hbuf(hdr.data_bytes);
  int fd   = ::open(path.c_str(), O_RDONLY);
  RAFT_EXPECTS(fd >= 0, "octex: cannot open %s", path.c_str());

  detail::parallel_pread(fd, static_cast<char*>(h.copy.ptr), hdr.data_bytes, hdr.header_len);
  ::close(fd);

  h.ptr        = h.copy.ptr;
  h.registered = false;
  return h;
}

void enable_peer_access(std::vector<int> const& devs)
{
  for (int a : devs)
    for (int b : devs) {
      if (a == b) continue;
      int ok = 0;
      RAFT_CUDA_TRY(cudaDeviceCanAccessPeer(&ok, a, b));
      if (!ok) continue;
      RAFT_CUDA_TRY(cudaSetDevice(a));
      auto e = cudaDeviceEnablePeerAccess(b, 0);
      if (e == cudaErrorPeerAccessAlreadyEnabled)
        cudaGetLastError();
      else
        RAFT_CUDA_TRY(e);
    }
}

}  // namespace

auto load(raft::resources const& res, std::string const& path, load_params const& params) -> index
{
  (void)res;
  int cur_dev = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cur_dev));
  auto impl = std::make_shared<index_impl>();
  auto& ix  = *impl;
  ix.path   = path;
  ix.params = params;
  auto meta = read_text(path + "/meta.json");
  RAFT_EXPECTS(static_cast<int>(json_number(meta, "format", 1)) == 1,
               "octex: unknown index format");
  ix.n            = static_cast<int64_t>(json_number(meta, "n", 0));
  ix.dim          = static_cast<int64_t>(json_number(meta, "dim", 0));
  int all_copies  = static_cast<int>(json_number(meta, "copies", 0));
  ix.blocks       = static_cast<int>(json_number(meta, "blocks", 4));
  ix.list_len     = static_cast<int>(json_number(meta, "list_len", 0));
  ix.log_buckets  = static_cast<int>(json_number(meta, "log_buckets", 20));
  ix.sketch_width = static_cast<int>(json_number(meta, "sketch_width", 0));
  ix.table_len    = static_cast<int>(json_number(meta, "table_len", 0));
  auto spacings   = json_numbers(meta, "spacings");
  for (double f : spacings)
    ix.spacings.push_back(static_cast<float>(f));
  auto files = json_strings(meta, "files");
  auto file  = [&](std::string const& name) {
    auto it = files.find(name);
    if (it == files.end()) return path + "/" + name + ".npy";
    return it->second[0] == '/' ? it->second : path + "/" + it->second;
  };
  RAFT_EXPECTS(ix.dim % 2 == 0, "octex: the dimension must be even");
  RAFT_EXPECTS(ix.sketch_width % 16 == 0 && ix.sketch_width <= 512,
               "octex: sketch width: a multiple of 16, <= 512");

  int ll = params.list_limit > 0 ? std::min(params.list_limit, ix.list_len) : ix.list_len;

  if (params.devices.empty()) {
    int nd = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&nd));
    for (int d = 0; d < nd; d++)
      ix.devices.push_back(d);
  } else {
    ix.devices = params.devices;
  }
  const int ng     = static_cast<int>(ix.devices.size());
  const bool shard = params.mode == placement::shard;
  RAFT_EXPECTS(ng >= 1, "octex: no device");
  // copies: the saved parts (equal consecutive ranges), each cut to copies_per_part; shard
  // placement deals the parts to the GPUs round robin (one part: equal consecutive ranges)
  const int nparts = std::max(1, params.parts);
  std::vector<std::vector<int>> part_copies(nparts);
  for (int q = 0; q < nparts; q++) {
    int lo = static_cast<int>(int64_t{all_copies} * q / nparts),
        hi = static_cast<int>(int64_t{all_copies} * (q + 1) / nparts);
    if (params.copies_per_part > 0) hi = std::min(hi, lo + params.copies_per_part);
    if (nparts == 1 && params.copies_limit > 0) hi = std::min(hi, lo + params.copies_limit);
    for (int c = lo; c < hi; c++)
      part_copies[q].push_back(c);
  }
  std::vector<std::vector<int>> gpu_copies(ng);
  ix.copies = 0;
  for (auto const& pc : part_copies)
    ix.copies += static_cast<int>(pc.size());
  for (int g = 0; g < ng; g++) {
    if (!shard) {
      for (auto const& pc : part_copies)
        gpu_copies[g].insert(gpu_copies[g].end(), pc.begin(), pc.end());
    } else if (nparts == 1) {
      auto const& pc = part_copies[0];
      gpu_copies[g].assign(pc.begin() + pc.size() * g / ng, pc.begin() + pc.size() * (g + 1) / ng);
    } else {
      for (int q = g; q < nparts; q += ng)
        gpu_copies[g].insert(gpu_copies[g].end(), part_copies[q].begin(), part_copies[q].end());
    }
    RAFT_EXPECTS(!gpu_copies[g].empty(),
                 "octex: a device holds no copies (fewer copies or parts than devices)");
  }
  enable_peer_access(ix.devices);

  npy_file lists(file("lists")), keys(file("keys")), offsets(file("offsets")), base(file("base"));
  npy_file proj(file("proj")), shift(file("shift")), mean(file("mean"));
  npy_file basis(file("sketch_basis")), scale(file("sketch_scale"));

  const auto sk = params.sketch;
  std::unique_ptr<npy_file> sketch;
  if (sk != sketch_placement::host) { sketch = std::make_unique<npy_file>(file("sketch")); }

  std::unique_ptr<npy_file> vectors;
  std::vector<int64_t> vectors_shape;
  if (params.vectors_on_host) {
    auto vhdr     = detail::read_npy_header(file("vectors"));
    vectors_shape = vhdr.shape;
  } else {
    vectors       = std::make_unique<npy_file>(file("vectors"));
    vectors_shape = vectors->shape;
  }

  std::unique_ptr<npy_file> table;
  if (ix.table_len > 0 && !params.table_on_host) {
    table = std::make_unique<npy_file>(file("table"));
  }

  const int L     = static_cast<int>(ix.spacings.size());
  const int width = 8 * ix.blocks;
  const int64_t D = ix.dim, N = ix.n, S = ix.sketch_width;
  const int64_t T1 = offsets.shape[2];
  RAFT_EXPECTS(lists.shape[0] == all_copies && keys.shape[1] == L && proj.shape[2] == width,
               "octex: bad shapes");
  RAFT_EXPECTS(vectors_shape.size() >= 2 && vectors_shape[0] == N && vectors_shape[1] == D,
               "octex: vectors shape");
  const int64_t rows_per = (N + ng - 1) / ng;

  // the mapped files into the page cache by parallel reads before the uploads read them
  {
    std::vector<std::string> pre{file("lists"), file("keys"), file("offsets")};
    if (sketch) pre.push_back(file("sketch"));
    if (vectors) pre.push_back(file("vectors"));
    if (table) pre.push_back(file("table"));
    for (auto const& f : pre)
      detail::prefetch_file(f);
  }

  // host arrays first: every GPU reads them through one pointer
  if (params.vectors_on_host) ix.host_vectors = map_host(file("vectors"));
  if (ix.table_len > 0 && params.table_on_host) ix.host_table = map_host(file("table"));
  if (sk == sketch_placement::host) ix.host_sketch = map_host(file("sketch"));

  ix.parts.resize(ng);
  std::vector<std::exception_ptr> errs(ng);
  std::vector<std::thread> threads;
  for (int g = 0; g < ng; g++) {
    threads.emplace_back([&, g] {
      try {
        auto part = std::make_unique<device_part>();
        auto& p   = *part;
        p.device  = ix.devices[g];
        RAFT_CUDA_TRY(cudaSetDevice(p.device));
        RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
        stager st(p.device, p.stream);
        auto const& cl = gpu_copies[g];  // this GPU's copies, in order
        p.copy0        = cl.front();
        p.copies       = static_cast<int>(cl.size());
        const int Cg   = p.copies;
        // copy runs of consecutive copies, per_copy bytes each, into the device array in list order
        auto put = [&](dbuf& dst, const char* src, size_t per_copy) {
          dst = dbuf(p.device, per_copy * Cg);
          p.bytes += static_cast<int64_t>(dst.size);
          for (int i = 0; i < Cg;) {
            int j = i + 1;
            while (j < Cg && cl[j] == cl[j - 1] + 1)
              j++;
            st.copy(dst.as<char>() + per_copy * i, src + per_copy * cl[i], per_copy * (j - i));
            i = j;
          }
        };

        // lists (the first ll entries of every cell row)
        const int64_t stride0 = lists.shape[1];
        if (ll == ix.list_len) {
          p.stride = stride0;
          put(p.lists, lists.data, stride0 * sizeof(int32_t));
        } else {
          int64_t rows = stride0 / ix.list_len;
          p.stride     = rows * ll;
          p.lists      = dbuf(p.device, static_cast<size_t>(Cg * p.stride) * sizeof(int32_t));
          p.bytes += static_cast<int64_t>(p.lists.size);
          std::vector<int32_t> tmp(p.stride);
          for (int c = 0; c < Cg; c++) {
            auto const* src = lists.as<int32_t>() + cl[c] * stride0;
            for (int64_t r = 0; r < rows; r++)
              std::memcpy(tmp.data() + r * ll, src + r * ix.list_len, ll * sizeof(int32_t));
            st.copy(p.lists.as<int32_t>() + c * p.stride, tmp.data(), p.stride * sizeof(int32_t));
            RAFT_CUDA_TRY(cudaStreamSynchronize(p.stream));  // tmp is reused
          }
        }
        p.mmax = keys.shape[2];
        put(p.keys, keys.data, L * p.mmax * sizeof(int64_t));
        put(p.offsets, offsets.data, L * T1 * sizeof(int32_t));
        put(p.base, base.data, L * sizeof(int32_t));

        // projections side by side [D, Cg * 8b] and shifts, in fp16
        std::vector<half> uc(static_cast<size_t>(D) * Cg * width),
          sh(static_cast<size_t>(Cg) * width);
        auto const* pr = proj.as<float>();
        auto const* sp = shift.as<float>();
        for (int c = 0; c < Cg; c++) {
          for (int64_t d = 0; d < D; d++)
            for (int s = 0; s < width; s++)
              uc[(d * Cg + c) * width + s] = __float2half(pr[(int64_t{cl[c]} * D + d) * width + s]);
          for (int s = 0; s < width; s++)
            sh[c * width + s] = __float2half(sp[cl[c] * width + s]);
        }
        std::vector<float> inv(L);
        for (int j = 0; j < L; j++)
          inv[j] = static_cast<float>(1.0 / spacings[j]);
        p.ucat  = upload(p, st, uc.data(), uc.size() * sizeof(half));
        p.shift = upload(p, st, sh.data(), sh.size() * sizeof(half));
        p.inv_f = upload(p, st, inv.data(), inv.size() * sizeof(float));
        p.mean  = upload(p, st, mean.as<float>(), D * sizeof(float));
        p.basis = upload(p, st, basis.as<float>(), D * S * sizeof(float));
        p.scale = upload(p, st, scale.as<float>(), S * sizeof(float));
        if (sk == sketch_placement::replicate) {
          p.sketch = upload(p, st, sketch->data, static_cast<size_t>(N * S));
        } else if (sk == sketch_placement::shard) {
          int64_t r0 = std::min(N, g * rows_per), r1 = std::min(N, r0 + rows_per);
          p.sketch = upload(p, st, sketch->data + r0 * S, static_cast<size_t>((r1 - r0) * S));
        }

        // vectors: host, row ranges, or the whole matrix
        if (!params.vectors_on_host) {
          int64_t r0 = params.shard_vectors ? g * rows_per : 0;
          int64_t r1 = params.shard_vectors ? std::min(N, r0 + rows_per) : N;
          p.vectors  = upload(
            p, st, vectors->as<half>() + r0 * D, static_cast<size_t>((r1 - r0) * D) * sizeof(half));
        }
        if (table && !params.table_on_host) {
          int64_t r0 = shard ? g * rows_per : 0;
          int64_t r1 = shard ? std::min(N, r0 + rows_per) : N;
          p.table    = upload(p,
                           st,
                           table->as<int32_t>() + r0 * ix.table_len,
                           static_cast<size_t>((r1 - r0) * ix.table_len) * sizeof(int32_t));
        }
        RAFT_CUDA_TRY(cudaStreamSynchronize(p.stream));

        RAFT_CUBLAS_TRY(cublasCreate(&p.cublas));
        p.workspace = dbuf(p.device, size_t{32} << 20);
        RAFT_CUBLAS_TRY(cublasSetWorkspace(p.cublas, p.workspace.ptr, p.workspace.size));
        ix.parts[g] = std::move(part);
      } catch (...) {
        errs[g] = std::current_exception();
      }
    });
  }
  for (auto& t : threads)
    t.join();
  for (auto& e : errs)
    if (e) std::rethrow_exception(e);

  // pointer tables of the vector and table row ranges
  std::vector<long long> xp, tp;
  if (params.vectors_on_host) {
    xp       = {reinterpret_cast<long long>(ix.host_vectors.ptr)};
    ix.xrows = N;
  } else if (params.shard_vectors) {
    for (auto& p : ix.parts)
      xp.push_back(reinterpret_cast<long long>(p->vectors.ptr));
    ix.xrows = rows_per;
  } else {
    ix.xrows = N;
  }
  if (ix.table_len > 0) {  // a host table has no device copy and no file handle left open
    if (params.table_on_host) {
      tp       = {reinterpret_cast<long long>(ix.host_table.ptr)};
      ix.trows = N;
    } else if (shard) {
      for (auto& p : ix.parts)
        tp.push_back(reinterpret_cast<long long>(p->table.ptr));
      ix.trows = rows_per;
    } else {
      ix.trows = N;
    }
  }
  std::vector<long long> cp;
  if (sk == sketch_placement::host) {
    cp       = {reinterpret_cast<long long>(ix.host_sketch.ptr)};
    ix.crows = N;
  } else if (sk == sketch_placement::shard) {
    for (auto& p : ix.parts)
      cp.push_back(reinterpret_cast<long long>(p->sketch.ptr));
    ix.crows = rows_per;
  }
  for (auto& p : ix.parts) {
    if (!cp.empty()) {
      p->cptr = dbuf(p->device, cp.size() * sizeof(long long));
      RAFT_CUDA_TRY(cudaMemcpy(p->cptr.ptr, cp.data(), p->cptr.size, cudaMemcpyHostToDevice));
    }
    auto own_x =
      xp.empty() ? std::vector<long long>{reinterpret_cast<long long>(p->vectors.ptr)} : xp;
    auto own_t =
      tp.empty() ? std::vector<long long>{reinterpret_cast<long long>(p->table.ptr)} : tp;
    p->xptr = dbuf(p->device, own_x.size() * sizeof(long long));
    p->tptr = dbuf(p->device, own_t.size() * sizeof(long long));
    RAFT_CUDA_TRY(cudaMemcpy(p->xptr.ptr, own_x.data(), p->xptr.size, cudaMemcpyHostToDevice));
    RAFT_CUDA_TRY(cudaMemcpy(p->tptr.ptr, own_t.data(), p->tptr.size, cudaMemcpyHostToDevice));
  }
  ix.list_len = ll;
  RAFT_CUDA_TRY(cudaSetDevice(cur_dev));
  return index(impl);
}

namespace detail {

void set_table(index const& idx, const int32_t* rows, int k)
{
  auto& ix = idx.impl();
  RAFT_EXPECTS(!ix.params.table_on_host, "octex: set_table needs a device table");
  int cur = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cur));
  const int ng           = static_cast<int>(ix.parts.size());
  const int64_t N        = ix.n;
  const bool shard       = ix.params.mode == placement::shard;
  const int64_t rows_per = (N + ng - 1) / ng;
  std::vector<std::thread> th;
  std::vector<std::exception_ptr> errs(ng);
  for (int g = 0; g < ng; g++)
    th.emplace_back([&, g] {
      try {
        auto& p    = *ix.parts[g];
        int64_t r0 = shard ? std::min(N, g * rows_per) : 0;
        int64_t r1 = shard ? std::min(N, r0 + rows_per) : N;
        size_t b   = static_cast<size_t>((r1 - r0) * k) * sizeof(int32_t);
        RAFT_CUDA_TRY(cudaSetDevice(p.device));
        if (p.table.size != b) {
          p.bytes -= static_cast<int64_t>(p.table.size);
          p.table = dbuf();
          p.table = dbuf(p.device, b);
          p.bytes += static_cast<int64_t>(b);
        }
        RAFT_CUDA_TRY(cudaMemcpy(p.table.ptr, rows + r0 * k, b, cudaMemcpyHostToDevice));
      } catch (...) {
        errs[g] = std::current_exception();
      }
    });
  for (auto& t : th)
    t.join();
  for (auto& e : errs)
    if (e) std::rethrow_exception(e);
  std::vector<long long> tp;
  if (shard)
    for (auto& p : ix.parts)
      tp.push_back(reinterpret_cast<long long>(p->table.ptr));
  ix.trows = shard ? rows_per : N;
  for (auto& p : ix.parts) {
    auto own_t =
      tp.empty() ? std::vector<long long>{reinterpret_cast<long long>(p->table.ptr)} : tp;
    p->tptr = dbuf(p->device, own_t.size() * sizeof(long long));
    RAFT_CUDA_TRY(cudaMemcpy(p->tptr.ptr, own_t.data(), p->tptr.size, cudaMemcpyHostToDevice));
  }
  ix.table_len = k;
  {
    std::lock_guard<std::mutex> lock(ix.mu);
    ix.plans.clear();
  }
  RAFT_CUDA_TRY(cudaSetDevice(cur));
}

void drop_table(index const& idx)
{
  auto& ix = idx.impl();
  {
    std::lock_guard<std::mutex> lock(ix.mu);
    ix.plans.clear();
  }
  for (auto& p : ix.parts) {
    p->bytes -= static_cast<int64_t>(p->table.size);
    p->table = dbuf();
  }
  ix.table_len = 0;
}

}  // namespace detail

}  // namespace cuvs::neighbors::octex
