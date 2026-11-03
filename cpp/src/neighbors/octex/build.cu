/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Index build: PCA of a sample, random projections, the cells of every copy (copies split over the
// GPUs, one host thread each), the neighbor table and the int8 sketch, written as an index
// directory (format 1, docs/design.md).

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "detail/build_cells.cuh"
#include "detail/build_common.cuh"
#include "detail/build_io.hpp"
#include "detail/build_setup.cuh"
#include "detail/build_table.cuh"
#include "detail/npy.hpp"

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace cuvs::neighbors::octex {

namespace {

template <typename T>
auto json_float(T v) -> std::string  // shortest form that reads back as the same T
{
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), v);
  std::string s(buf, r.ptr);
  if (s.find_first_of(".en") == std::string::npos) s += ".0";
  return s;
}

/** The arrays a build writes behind (index_params::write_behind), kept until the writes join. */
struct write_behind_t {
  std::vector<int64_t, detail::no_init_allocator<int64_t>> keys;
  std::vector<int32_t, detail::no_init_allocator<int32_t>> offs, lists;
  std::vector<int8_t, detail::no_init_allocator<int8_t>> codes;
  std::thread writer;
  std::exception_ptr err;
};
auto write_behind() -> write_behind_t&
{
  static write_behind_t w;
  return w;
}

/**
 * meta.json of an index directory: table_len rows of the neighbor table, files the arrays at other
 * paths (name -> path), build_seconds the phase times (none: omitted).
 */
auto meta_json(index_params const& params,
               int64_t n,
               int64_t d,
               int S,
               int table_len,
               std::vector<std::pair<std::string, std::string>> const& files,
               std::vector<std::pair<std::string, double>> const* phases,
               double total) -> std::string
{
  const int L = static_cast<int>(params.spacings.size());
  std::ostringstream m;
  m << "{\n \"format\": 1,\n \"n\": " << n << ",\n \"dim\": " << d
    << ",\n \"copies\": " << params.copies << ",\n \"blocks\": " << params.blocks
    << ",\n \"spacings\": [";
  for (int j = 0; j < L; j++)
    m << (j ? ", " : "") << json_float(params.spacings[j]);
  m << "],\n \"list_len\": " << params.list_len << ",\n \"log_buckets\": " << params.log_buckets
    << ",\n \"sketch_width\": " << S << ",\n \"table_len\": " << table_len
    << ",\n \"metric\": \"sqeuclidean\",\n \"files\": {";
  for (size_t i = 0; i < files.size(); i++)
    m << (i ? ", " : "") << "\"" << files[i].first << "\": \"" << files[i].second << "\"";
  m << "}";
  if (phases != nullptr) {
    m << ",\n \"build_seconds\": {";
    for (size_t i = 0; i < phases->size(); i++)
      m << (i ? ", " : "") << "\"" << (*phases)[i].first
        << "\": " << json_float((*phases)[i].second);
    m << ", \"total\": " << json_float(total) << "}";
  }
  m << "\n}\n";
  return m.str();
}

/**
 * Build into the directory path. own: the builder's own fp16 copy of the dataset (x points into
 * it), released before a searched neighbor table, which reads the vectors from vectors.npy.
 */
void build_impl(index_params const& params,
                const half* x,
                int64_t n,
                int64_t d,
                std::string const& path,
                std::vector<half>* own = nullptr)
{
  using namespace detail;
  const int C     = params.copies;
  const int b     = params.blocks;
  const int w     = 8 * b;
  const int L     = static_cast<int>(params.spacings.size());
  const int l     = params.list_len;
  const int64_t T = int64_t(1) << params.log_buckets;
  const int S     = params.sketch_width > 0
                      ? static_cast<int>(std::min<int64_t>(params.sketch_width, d))
                      : static_cast<int>(std::max<int64_t>(std::min<int64_t>(d / 16 * 16, 128),
                                                       std::min<int64_t>(d, 16)));
  const int k     = params.table_len;
  RAFT_EXPECTS(C >= 1, "octex: at least one copy");
  RAFT_EXPECTS(n >= 2 && d >= 1, "octex: empty dataset");
  RAFT_EXPECTS(S >= 1, "octex: sketch_width must be positive");
  std::vector<int> devices = params.devices;
  if (devices.empty()) {
    int count = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
    for (int g = 0; g < count; g++)
      devices.push_back(g);
  }
  RAFT_EXPECTS(!devices.empty(), "octex: no GPU");
  if (int(devices.size()) > C) devices.resize(C);
  std::filesystem::create_directories(path);
  auto file = [&](const char* name) { return (std::filesystem::path(path) / name).string(); };
  auto t0   = std::chrono::steady_clock::now();
  std::vector<std::pair<std::string, double>> phases;
  auto phase = [&](const char* name, std::chrono::steady_clock::time_point& t) {
    double s = seconds_since(t);
    phases.emplace_back(name, s);
    build_log("%s: %.1f s\n", name, s);
    t = std::chrono::steady_clock::now();
  };
  auto t = t0;
  // OCTEX_TABLE_ONLY=1: the directory already holds every array but table.npy (an earlier build
  // that stopped in its table phase); only the neighbor table and meta.json are built
  const bool table_only = std::getenv("OCTEX_TABLE_ONLY") != nullptr;
  if (table_only)
    for (const char* f : {"mean.npy",
                          "proj.npy",
                          "shift.npy",
                          "base.npy",
                          "cells.npy",
                          "keys.npy",
                          "offsets.npy",
                          "lists.npy",
                          "sketch_basis.npy",
                          "sketch_scale.npy",
                          "sketch.npy",
                          "vectors.npy"})
      RAFT_EXPECTS(std::filesystem::exists(file(f)) || find_mem_array(file(f)),
                   "octex: OCTEX_TABLE_ONLY needs %s",
                   file(f).c_str());  // or the array, still in memory while written behind
  // write_behind: keys, offsets, lists and the sketch stay in memory, registered under their
  // paths for the table's loads, while a background thread writes them (joined before meta.json)
  // OCTEX_TABLE_FOLLOWS (set by the command-line tool): a table-only build follows this one in
  // the process, after the tool frees the base; the writes behind are joined by that build
  const char* follows_env = std::getenv("OCTEX_TABLE_FOLLOWS");
  const int follows       = follows_env != nullptr ? std::atoi(follows_env) : 0;
  const bool behind       = params.write_behind;
  auto& wb                = write_behind();
  auto& mkeys             = wb.keys;
  auto& moffs             = wb.offs;
  auto& mlists            = wb.lists;
  auto& mcodes            = wb.codes;
  auto& writer            = wb.writer;
  auto& werr              = wb.err;
  if (!table_only) {
    // mean, principal directions, projections
    pca_result pca;
    per_device({devices[0]}, [&](int, int) {
      raft::device_resources res;
      pca = sample_pca(res, x, n, d);
    });
    int p = std::min<int>(params.pca_dims, static_cast<int>(d));
    std::vector<float> proj, shift;
    make_projections(pca, d, p, w, C, params.seed, proj, shift);
    phase("setup", t);

    // the cells of every copy; GPU r builds copies [C r / G, C (r + 1) / G)
    cell_params cp;
    cp.blocks      = b;
    cp.spacings    = params.spacings;
    cp.list_len    = l;
    cp.min_cell    = params.min_cell;
    cp.log_buckets = params.log_buckets;
    cp.exact_lists  = params.exact_lists;
    cp.list_probes  = params.list_probes;
    cp.member_lists = params.member_lists;
    std::vector<copy_cells> cells(C);
    const int G = static_cast<int>(devices.size());
    per_device(devices, [&](int rank, int dev) {
      raft::device_resources res;
      int c0 = C * rank / G, c1 = C * (rank + 1) / G;
      std::vector<copy_cells> part;
      auto tc = std::chrono::steady_clock::now();
      build_cells(res, cp, x, n, d, pca.mean.data(), proj.data(), shift.data(), c0, c1, part);
      for (int c = c0; c < c1; c++)
        cells[c] = std::move(part[c - c0]);
      build_log("GPU %d: copies %d..%d in %.1f s\n", dev, c0, c1 - 1, seconds_since(tc));
    });
    phase("copies", t);

    write_npy(file("mean.npy"), pca.mean.data(), {d});
    write_npy(file("proj.npy"), proj.data(), {C, d, w});
    write_npy(file("shift.npy"), shift.data(), {C, w});
    int64_t lmax = 0, mmax = 0;
    std::vector<int32_t> base(int64_t(C) * L);
    std::vector<int64_t> ncells(int64_t(C) * L);
    for (int c = 0; c < C; c++) {
      lmax = std::max<int64_t>(lmax, cells[c].lists.size());
      for (int j = 0; j < L; j++) {
        mmax              = std::max(mmax, cells[c].cells[j]);
        base[c * L + j]   = cells[c].base[j];
        ncells[c * L + j] = cells[c].cells[j];
      }
    }
    write_npy(file("base.npy"), base.data(), {C, L});
    write_npy(file("cells.npy"), ncells.data(), {C, L});
    if (behind) {
      mkeys.resize(size_t(C) * L * mmax);
      moffs.resize(size_t(C) * L * (T + 1));
      mlists.resize(size_t(C) * lmax);
      const int nt = std::min<int>(C, 16);
      std::vector<std::thread> th;
      for (int r = 0; r < nt; r++)
        th.emplace_back([&, r] {
        for (int c = r; c < C; c += nt) {
          int64_t at = 0;
          for (int j = 0; j < L; j++) {
            int64_t* o = mkeys.data() + (size_t(c) * L + j) * mmax;
            std::copy_n(cells[c].keys.data() + at, cells[c].cells[j], o);
            std::fill(o + cells[c].cells[j], o + mmax, int64_t{0});
            at += cells[c].cells[j];
          }
          std::copy_n(cells[c].offsets.data(),
                      int64_t(L) * (T + 1),
                      moffs.data() + size_t(c) * L * (T + 1));
          int32_t* lo = mlists.data() + size_t(c) * lmax;
          std::copy_n(cells[c].lists.data(), cells[c].lists.size(), lo);
          std::fill(lo + cells[c].lists.size(), lo + lmax, int32_t{0});
          std::vector<int64_t>().swap(cells[c].keys);
          std::vector<int32_t>().swap(cells[c].offsets);
          std::vector<int32_t>().swap(cells[c].lists);
        }
        });
      for (auto& x : th)
        x.join();
      register_mem_array(
        file("keys.npy"), mkeys.data(), {C, L, mmax}, "<i8", mkeys.size() * sizeof(int64_t));
      register_mem_array(
        file("offsets.npy"), moffs.data(), {C, L, T + 1}, "<i4", moffs.size() * sizeof(int32_t));
      register_mem_array(
        file("lists.npy"), mlists.data(), {C, lmax}, "<i4", mlists.size() * sizeof(int32_t));
    } else {
      npy_writer<int64_t> keys(file("keys.npy"), {C, L, mmax});
      npy_writer<int32_t> offs(file("offsets.npy"), {C, L, T + 1});
      for (int c = 0; c < C; c++) {
        int64_t at = 0;
        for (int j = 0; j < L; j++) {
          keys.write(cells[c].keys.data() + at, cells[c].cells[j]);
          keys.zeros(mmax - cells[c].cells[j]);
          at += cells[c].cells[j];
        }
        offs.write(cells[c].offsets.data(), int64_t(L) * (T + 1));
        std::vector<int64_t>().swap(cells[c].keys);
        std::vector<int32_t>().swap(cells[c].offsets);
      }
      keys.close();
      offs.close();
    }
    if (!behind) {
      npy_writer<int32_t> lists(file("lists.npy"), {C, lmax});
      for (int c = 0; c < C; c++) {
        lists.write(cells[c].lists.data(), cells[c].lists.size());
        lists.zeros(lmax - int64_t(cells[c].lists.size()));
        std::vector<int32_t>().swap(cells[c].lists);
      }
      lists.close();
    }
    phase("write", t);

    {
      std::vector<float> basis(size_t(d) * S), scale;
      for (int64_t r = 0; r < d; r++)
        for (int s = 0; s < S; s++)
          basis[r * S + s] = pca.vectors[r * d + s];
      std::vector<int8_t, no_init_allocator<int8_t>> codes(size_t(n) * S);
      per_device({devices[0]}, [&](int, int) {
        raft::device_resources res;
        make_sketch(res, x, n, d, pca.mean.data(), basis.data(), S, scale, codes.data());
      });
      write_npy(file("sketch_basis.npy"), basis.data(), {d, int64_t(S)});
      write_npy(file("sketch_scale.npy"), scale.data(), {int64_t(S)});
      if (behind) {
        mcodes = std::move(codes);
        register_mem_array(file("sketch.npy"), mcodes.data(), {n, int64_t(S)}, "|i1", mcodes.size());
        // the paths by value: with a table build to follow, this build returns before the writes
        writer = std::thread([&wb,
                              fk = file("keys.npy"),
                              fo = file("offsets.npy"),
                              fl = file("lists.npy"),
                              fs = file("sketch.npy"),
                              C, L, mmax, T, lmax, n, S] {
          try {
            write_npy(fk, wb.keys.data(), {C, L, mmax});
            write_npy(fo, wb.offs.data(), {C, L, T + 1});
            write_npy(fl, wb.lists.data(), {C, lmax});
            write_npy(fs, wb.codes.data(), {n, int64_t(S)});
          } catch (...) {
            wb.err = std::current_exception();
          }
        });
      } else {
        write_npy(file("sketch.npy"), codes.data(), {n, int64_t(S)});
      }
    }
    // OCTEX_VECTORS_FROM (the command-line tool's --link-vectors): an fp16 [n, d] .npy holding
    // these vectors on the same file system, linked instead of rewritten
    const char* vfrom = std::getenv("OCTEX_VECTORS_FROM");
    if (vfrom != nullptr) {
      auto h = read_npy_header(vfrom);
      RAFT_EXPECTS(h.descr == "<f2" && h.shape.size() == 2 && h.shape[0] == n && h.shape[1] == d,
                   "octex: OCTEX_VECTORS_FROM must be fp16 [n, d]");
      std::filesystem::remove(file("vectors.npy"));
      std::error_code ec;
      std::filesystem::create_hard_link(vfrom, file("vectors.npy"), ec);
      if (ec) write_npy(file("vectors.npy"), x, {n, d});  // another file system: written
    } else {
      write_npy(file("vectors.npy"), x, {n, d});
    }
    phase("sketch", t);
  }  // !table_only

  // the directory is complete but for table.npy and meta.json
  if (k > 0 && n <= params.exact_table_max) {
    RAFT_EXPECTS(x != nullptr,
                 "octex: the exact table needs the vectors (OCTEX_TABLE_ONLY reads none)");
    std::vector<int32_t, no_init_allocator<int32_t>> table(size_t(n) * k);
    build_table(devices, x, n, d, k, params.reverse_edges, table.data());
    write_npy(file("table.npy"), table.data(), {n, int64_t(k)});
    phase("table", t);
  } else if (k > 0) {
    if (own != nullptr) {  // the searches read their queries from vectors.npy
      std::vector<half>().swap(*own);
      x = nullptr;
    }
    // every pass loads a directory whose meta.json names the files of this one (absolute paths)
    auto abs_dir = std::filesystem::absolute(path);
    auto stage   = (abs_dir / ".table").string();
    std::filesystem::create_directories(stage);
    std::vector<std::pair<std::string, std::string>> names;
    for (const char* f : {"mean",
                          "proj",
                          "shift",
                          "base",
                          "keys",
                          "offsets",
                          "lists",
                          "sketch_basis",
                          "sketch_scale",
                          "sketch",
                          "vectors",
                          "table"})
      names.emplace_back(f, (abs_dir / (std::string(f) + ".npy")).string());
    auto staged = [&](int table_len) {
      std::ofstream((std::filesystem::path(stage) / "meta.json").string())
        << meta_json(params, n, d, S, table_len, names, nullptr, 0.0);
      return stage;
    };
    auto tt = t;
    searched_table(
      devices,
      file("vectors.npy"),
      n,
      d,
      k,
      S,
      staged,
      file("table.npy"),
      [&](std::string const& name) {
        phases.emplace_back(name, seconds_since(tt));
        build_log("%s: %.1f s\n", name.c_str(), phases.back().second);
        tt = std::chrono::steady_clock::now();
      },
      params.reverse_edges,
      params.table_passes,
      params.table_sketch_order);
    std::filesystem::remove_all(stage);
    phase("table", t);
  }

  if (writer.joinable() && follows == 0) {
    auto tw = std::chrono::steady_clock::now();
    writer.join();
    build_log("write behind: joined after %.1f s more\n", seconds_since(tw));
  }
  if (follows == 0) {
    clear_mem_arrays();
    auto err = werr;  // before the reset, which clears it
    wb       = write_behind_t{};
    if (err) std::rethrow_exception(err);
  }

  // meta.json last: a directory without it is incomplete
  std::ofstream(file("meta.json")) << meta_json(params, n, d, S, k, {}, &phases, seconds_since(t0));
  build_log("total: %.1f s\n", seconds_since(t0));
}

}  // namespace

void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           std::string const& path)
{
  (void)res;  // every GPU gets its own resources, one host thread each
  int64_t n = dataset.extent(0), d = dataset.extent(1);
  std::vector<half> xh(size_t(n) *
                       d);  // the index is built from (and reranks with) the fp16 vectors
  const float* src = dataset.data_handle();
  detail::parallel_for(n * d, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; i++)
      xh[i] = __float2half(src[i]);
  });
  build_impl(params, xh.data(), n, d, path, &xh);
}

void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           std::string const& path)
{
  (void)res;
  build_impl(params, dataset.data_handle(), dataset.extent(0), dataset.extent(1), path);
}

}  // namespace cuvs::neighbors::octex
