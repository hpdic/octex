/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Recall@10 and QPS of octex search over a grid of search parameters.
//
//   octex_bench <index dir> [--devices 0,1] [--placement shard|replicate] [--vectors-on-host]
//   [--shard-vectors]
//               [--table-on-host] [--parts P] [--copies-per-part K] [--copies-limit C]
//               [--list-limit l] [--read 4,8,16] [--starts 32,64]
//               [--waves 1,2] [--rerank 128] [--wave-rerank 128] [--batch 10000] [--sketch-rank]
//               [--final-rerank 32] [--k 10] [--reps 5] [--host] [--queries q.npy] [--groundtruth
//               gt.npy]
//               [--nq N] [--sketch replicate|shard|host] [--dataset name] [--json out.jsonl]
//               [--no-balanced-gather] [--no-fused-merge] [--no-graph]
//
// Queries (f32 [nq, D]) and ground truth (i64 or i32 [nq, >= 10]) default to queries.npy and
// groundtruth.npy in the index directory. QPS: best of --reps searches of the whole query set after
// a warm-up, wall clock across all GPUs (queries and results on the first device, or on the host
// with --host). --resident times searches whose queries are already on every GPU and whose results
// stay there, as GPU baselines with resident queries are timed (the whole query set must fit one
// batch).
//
// Ablations of the search (search_params): --no-balanced-gather walks whole lists per warp,
// --no-fused-merge merges by CUB sorts in global memory and separate launches, --no-graph launches
// the kernels directly with a host synchronization between the first round and the waves.

#include <cuvs/core/cuda_fp16.hpp>

#include "neighbors/octex/detail/introspect.hpp"
#include "neighbors/octex/detail/npy.hpp"

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace octex = cuvs::neighbors::octex;

namespace {

auto ints(std::string const& s) -> std::vector<int>
{
  std::vector<int> out;
  size_t p = 0;
  while (p < s.size()) {
    size_t e = s.find(',', p);
    if (e == std::string::npos) e = s.size();
    out.push_back(std::atoi(s.substr(p, e - p).c_str()));
    p = e + 1;
  }
  return out;
}

#define CHECK_CUDA(x)                                                                 \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <index dir> [options] (see the source header)\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  std::map<std::string, std::string> opt;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) continue;
    a         = a.substr(2);
    bool flag = a == "vectors-on-host" || a == "shard-vectors" || a == "table-on-host" ||
                a == "sketch-rank" || a == "host" || a == "resident" || a == "no-balanced-gather" ||
                a == "no-fused-merge" || a == "no-graph";
    if (flag)
      opt[a] = "1";
    else if (i + 1 < argc)
      opt[a] = argv[++i];
  }
  auto get = [&](std::string const& k, std::string const& d) { return opt.count(k) ? opt[k] : d; };

  octex::load_params lp;
  lp.devices          = ints(get("devices", "0"));
  lp.mode             = get("placement", "shard") == "replicate" ? octex::placement::replicate
                                                                 : octex::placement::shard;
  lp.vectors_on_host  = opt.count("vectors-on-host") > 0;
  lp.shard_vectors    = opt.count("shard-vectors") > 0;
  lp.table_on_host    = opt.count("table-on-host") > 0;
  lp.copies_limit     = std::atoi(get("copies-limit", "0").c_str());
  lp.list_limit       = std::atoi(get("list-limit", "0").c_str());
  lp.parts            = std::atoi(get("parts", "0").c_str());
  lp.copies_per_part  = std::atoi(get("copies-per-part", "0").c_str());
  std::string skp     = get("sketch", "replicate");
  lp.sketch           = skp == "shard"  ? octex::sketch_placement::shard
                        : skp == "host" ? octex::sketch_placement::host
                                        : octex::sketch_placement::replicate;
  std::string dataset = get("dataset", "");
  if (dataset.empty()) {
    std::string d = dir;
    while (!d.empty() && d.back() == '/')
      d.pop_back();
    dataset = d.substr(d.find_last_of('/') + 1);
  }
  const int k         = std::atoi(get("k", "10").c_str());
  const int reps      = std::atoi(get("reps", "5").c_str());
  const bool resident = opt.count("resident") > 0;
  const bool host     = opt.count("host") > 0;
  const bool sketch   = opt.count("sketch-rank") > 0;
  const bool balanced = opt.count("no-balanced-gather") == 0;
  const bool fused    = opt.count("no-fused-merge") == 0;
  const bool graph    = opt.count("no-graph") == 0;
  std::string jpath   = get("json", "");
  // ablated optimizations, printed on every result line
  std::string ablate = std::string(balanced ? "" : " no_balanced_gather") +
                       (fused ? "" : " no_fused_merge") + (graph ? "" : " no_graph");

  cuvs::neighbors::octex::detail::npy_file qf(get("queries", dir + "/queries.npy"));
  cuvs::neighbors::octex::detail::npy_file gf(get("groundtruth", dir + "/groundtruth.npy"));
  // --nq N: the first N queries are the whole query set, one batch of N unless --batch says
  // otherwise
  const int64_t nq_opt = std::atoll(get("nq", "0").c_str());
  const int64_t nq     = nq_opt > 0 ? std::min<int64_t>(nq_opt, qf.shape[0]) : qf.shape[0];
  const int64_t D = qf.shape[1], gk = gf.shape[1];
  if (nq_opt > 0 && !opt.count("batch")) opt["batch"] = std::to_string(nq);
  std::vector<int64_t> gt(nq * gk);
  if (gf.descr == "<i8")
    std::memcpy(gt.data(), gf.as<int64_t>(), gt.size() * sizeof(int64_t));  // the first nq rows
  else
    for (size_t i = 0; i < gt.size(); i++)
      gt[i] = gf.as<int32_t>()[i];

  CHECK_CUDA(cudaSetDevice(lp.devices[0]));
  raft::resources res;
  auto t0       = std::chrono::steady_clock::now();
  auto idx      = octex::load(res, dir, lp);
  double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("loaded %s: n=%ld dim=%ld copies=%d on %zu GPU(s) (%s) in %.1f s; GiB per GPU:",
              dir.c_str(),
              (long)idx.size(),
              (long)idx.dim(),
              idx.copies(),
              idx.devices().size(),
              lp.mode == octex::placement::shard ? "shard" : "replicate",
              load_s);
  for (auto b : idx.device_bytes())
    std::printf(" %.1f", b / double(1 << 30));
  std::printf("\n");
  std::fflush(stdout);

  CHECK_CUDA(cudaSetDevice(lp.devices[0]));
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  std::vector<int64_t> hi(nq * k);
  std::vector<float> hd(nq * k);
  float* dq      = nullptr;
  int64_t* di    = nullptr;
  float* dd      = nullptr;
  auto const* hq = qf.as<float>();
  if (!host) {
    CHECK_CUDA(cudaMalloc(&dq, nq * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&di, nq * k * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&dd, nq * k * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dq, hq, nq * D * sizeof(float), cudaMemcpyHostToDevice));
  }
  auto run = [&](octex::search_params const& sp) {
    if (host) {
      octex::search(res,
                    sp,
                    idx,
                    raft::make_host_matrix_view<const float, int64_t>(hq, nq, D),
                    raft::make_host_matrix_view<int64_t, int64_t>(hi.data(), nq, k),
                    raft::make_host_matrix_view<float, int64_t>(hd.data(), nq, k));
    } else {
      octex::search(res,
                    sp,
                    idx,
                    raft::make_device_matrix_view<const float, int64_t>(dq, nq, D),
                    raft::make_device_matrix_view<int64_t, int64_t>(di, nq, k),
                    raft::make_device_matrix_view<float, int64_t>(dd, nq, k));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
  };

  FILE* jf = jpath.empty() ? nullptr : std::fopen(jpath.c_str(), "a");
  for (int read : ints(get("read", "8")))
    for (int starts : ints(get("starts", "64")))
      for (int waves : ints(get("waves", "1")))
        for (int r1 : ints(get("rerank", "128")))
          for (int r2 : ints(get("wave-rerank", "128")))
            for (int batch : ints(get("batch", "10000")))
              for (int fr : ints(get("final-rerank", "32"))) {
                octex::search_params sp;
                sp.read            = read;
                sp.starts          = starts;
                sp.waves           = waves;
                sp.rerank          = r1;
                sp.wave_rerank     = r2;
                sp.batch           = batch;
                sp.sketch_rank     = sketch;
                sp.final_rerank    = fr;
                sp.balanced_gather = balanced;
                sp.fused_merge     = fused;
                sp.use_graph       = graph;
                auto c0            = std::chrono::steady_clock::now();
                run(sp);  // captures the graphs
                double cap_s =
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
                run(sp);
                std::vector<double> times;
                for (int r = 0; r < reps; r++) {
                  auto a = std::chrono::steady_clock::now();
                  if (resident) {  // queries stay on the GPUs from the previous search, results
                                   // stay there
                    cuvs::neighbors::octex::detail::resident_search(res, sp, idx, nq, k);
                  } else {
                    run(sp);
                  }
                  times.push_back(
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
                }
                auto sorted = times;
                std::sort(sorted.begin(), sorted.end());
                double best = sorted.front(), median = sorted[sorted.size() / 2];
                if (!host)
                  CHECK_CUDA(
                    cudaMemcpy(hi.data(), di, nq * k * sizeof(int64_t), cudaMemcpyDeviceToHost));
                const int kk = std::min<int>(10, std::min<int64_t>(k, gk));
                int64_t hit  = 0;
                for (int64_t q = 0; q < nq; q++)
                  for (int i = 0; i < kk; i++)
                    for (int j = 0; j < kk; j++)
                      hit += hi[q * k + i] == gt[q * gk + j];
                double recall = double(hit) / double(nq * kk);
                double qps    = nq / best;
                std::printf(
                  "read=%d starts=%d waves=%d rerank=%d wave_rerank=%d batch=%d%s%s recall@10=%.4f "
                  "qps=%.0f "
                  "(median %.0f) time=%.2fms (capture %.2fs)\n",
                  read,
                  starts,
                  waves,
                  r1,
                  r2,
                  batch,
                  sketch ? (" sketch_rank final_rerank=" + std::to_string(fr)).c_str() : "",
                  ablate.c_str(),
                  recall,
                  qps,
                  nq / median,
                  best * 1e3,
                  cap_s);
                std::fflush(stdout);
                if (jf != nullptr) {
                  std::string dl, tl;
                  for (size_t i = 0; i < lp.devices.size(); i++)
                    dl += (i ? ", " : "") + std::to_string(lp.devices[i]);
                  for (size_t i = 0; i < times.size(); i++) {
                    char b[32];
                    std::snprintf(b, sizeof(b), "%s%.6f", i ? ", " : "", times[i]);
                    tl += b;
                  }
                  auto tf = [](bool v) { return v ? "true" : "false"; };
                  // one extra timed batch: per GPU the first-round and wave graph times; memory per
                  // GPU by part
                  auto ph = cuvs::neighbors::octex::detail::timed_batch(
                    res, sp, idx, host ? hq : dq, nq, k);
                  auto mem = cuvs::neighbors::octex::detail::memory_report(idx);
                  std::string sl, ml;
                  for (size_t i = 0; i < ph.size(); i++) {
                    char b[160];
                    std::snprintf(
                      b,
                      sizeof(b),
                      "%s{\"device\": %d, \"first_round_ms\": %.4f, \"waves_ms\": %.4f}",
                      i ? ", " : "",
                      ph[i].device,
                      ph[i].first_round_ms,
                      ph[i].waves_ms);
                    sl += b;
                  }
                  for (size_t i = 0; i < mem.size(); i++) {
                    char b[400];
                    int64_t sb = i < ph.size() ? ph[i].search_buffer_bytes : 0;
                    int64_t sd = i < ph.size() ? ph[i].search_mem_delta : 0;
                    std::snprintf(
                      b,
                      sizeof(b),
                      "%s{\"device\": %d, \"copies\": %ld, \"sketch\": %ld, \"table\": %ld, "
                      "\"vectors\": %ld, \"other\": %ld, \"search_buffers\": %ld, "
                      "\"search_setup_delta\": %ld}",
                      i ? ", " : "",
                      mem[i].device,
                      (long)mem[i].copies,
                      (long)mem[i].sketch,
                      (long)mem[i].table,
                      (long)mem[i].vectors,
                      (long)mem[i].other,
                      (long)sb,
                      (long)sd);
                    ml += b;
                  }
                  std::fprintf(
                    jf,
                    "{\"dataset\": \"%s\", \"index\": \"%s\", \"n\": %ld, \"dim\": %ld, "
                    "\"copies\": %d, "
                    "\"devices\": [%s], \"placement\": \"%s\", \"vectors_on_host\": %s, "
                    "\"shard_vectors\": %s, "
                    "\"table_on_host\": %s, \"sketch\": \"%s\", \"parts\": %d, "
                    "\"copies_per_part\": %d, \"copies_limit\": %d, \"list_limit\": %d, "
                    "\"read\": %d, \"starts\": %d, \"waves\": %d, \"rerank\": %d, \"wave_rerank\": "
                    "%d, "
                    "\"batch\": %d, \"sketch_rank\": %s, \"final_rerank\": %d, \"k\": %d, "
                    "\"host_io\": %s, \"resident\": %s, \"balanced_gather\": %s, "
                    "\"fused_merge\": %s, \"use_graph\": %s, "
                    "\"nq\": %ld, \"recall\": %.6f, \"qps\": %.1f, \"qps_median\": %.1f, "
                    "\"seconds\": %.6f, "
                    "\"times\": [%s], \"capture_s\": %.4f, \"split_ms\": [%s], \"memory\": [%s]}\n",
                    dataset.c_str(),
                    dir.c_str(),
                    (long)idx.size(),
                    (long)idx.dim(),
                    idx.copies(),
                    dl.c_str(),
                    lp.mode == octex::placement::shard ? "shard" : "replicate",
                    tf(lp.vectors_on_host),
                    tf(lp.shard_vectors),
                    tf(lp.table_on_host),
                    skp.c_str(),
                    lp.parts,
                    lp.copies_per_part,
                    lp.copies_limit,
                    lp.list_limit,
                    read,
                    starts,
                    waves,
                    r1,
                    r2,
                    batch,
                    tf(sketch),
                    fr,
                    k,
                    tf(host),
                    tf(resident),
                    tf(balanced),
                    tf(fused),
                    tf(graph),
                    (long)nq,
                    recall,
                    qps,
                    nq / median,
                    best,
                    tl.c_str(),
                    cap_s,
                    sl.c_str(),
                    ml.c_str());
                  std::fflush(jf);
                }
              }
  if (jf != nullptr) std::fclose(jf);
  if (!host) {
    cudaFree(dq);
    cudaFree(di);
    cudaFree(dd);
  }
  return 0;
}
