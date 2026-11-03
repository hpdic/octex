// IVF-RaBitQ (cuVS 26.10 C++) baseline (adapted from certex/baselines): the base and the
// queries are read from .fbin files (8-byte header) -- base rows [b0, b0 + n) (a shard; written ids are global),
// query rows [q0, q0 + nq).
// usage: rabitq_bench base.fbin b0 n d query.fbin q0 nq k n_lists bits "probes,..." "ratios,..." outdir
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/copy.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

static std::vector<float> load(const std::string& p, size_t count, size_t skip = 0) {
  std::vector<float> v(count);
  std::ifstream f(p, std::ios::binary);
  f.seekg(8 + skip * sizeof(float));  // .fbin: int32 n, int32 d, then the rows
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
  if (!f) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
  return v;
}
static std::vector<int> parse(const std::string& s) {
  std::vector<int> out; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) out.push_back(std::stoi(t));
  return out;
}
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 14) { fprintf(stderr, "usage\n"); return 1; }
  std::string bpath = argv[1]; int64_t b0 = atoll(argv[2]); int64_t n = atoll(argv[3]); int64_t d = atoll(argv[4]);
  std::string qpath = argv[5]; int64_t q0 = atoll(argv[6]); int64_t nq = atoll(argv[7]); int64_t k = atoll(argv[8]);
  uint32_t n_lists = atoi(argv[9]); uint32_t bits = atoi(argv[10]);
  auto probes = parse(argv[11]); auto ratios = parse(argv[12]); std::string outdir = argv[13];

  raft::device_resources res;
  auto hb = load(bpath, n * d, b0 * d); auto hq = load(qpath, nq * d, q0 * d);
  // the base stays on the host: build streams it (a device copy of 100M x 96 fp32 is 38 GB, and the build itself
  // allocates ~35 GB more, beyond one 94 GB H100)
  auto dq = raft::make_device_matrix<float, int64_t>(res, nq, d);
  raft::copy(res, dq.view(), raft::make_host_matrix_view<const float, int64_t>(hq.data(), nq, d));
  res.sync_stream();

  size_t free0, total; cudaMemGetInfo(&free0, &total);
  cuvs::neighbors::ivf_rabitq::index_params ip; ip.n_lists = n_lists; ip.bits_per_dim = bits;
  double t0 = now();
  auto index = cuvs::neighbors::ivf_rabitq::build(res, ip, raft::make_host_matrix_view<const float, int64_t>(hb.data(), n, d));
  res.sync_stream(); double build_s = now() - t0;
  size_t free1; cudaMemGetInfo(&free1, &total);
  printf("{\"event\":\"build\",\"n\":%ld,\"d\":%ld,\"n_lists\":%u,\"bits\":%u,\"build_s\":%.4f,\"index_gpu_bytes\":%zu}\n",
         n, d, n_lists, bits, build_s, free0 - free1);
  fflush(stdout);

  for (int r : ratios) {
    int64_t kp = k * r;
    auto cand = raft::make_device_matrix<int64_t, int64_t>(res, nq, kp);
    auto cdist = raft::make_device_matrix<float, int64_t>(res, nq, kp);
    auto out = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto odist = raft::make_device_matrix<float, int64_t>(res, nq, k);
    for (int p : probes) {
      cuvs::neighbors::ivf_rabitq::search_params sp; sp.n_probes = p;
      int64_t qb = getenv("RABITQ_QBATCH") ? atoll(getenv("RABITQ_QBATCH")) : nq;   // bounds the search workspace
      for (int rep = 0; rep < 2; ++rep) {   // first pass warms up
        res.sync_stream(); double s0 = now();
        for (int64_t q0 = 0; q0 < nq; q0 += qb) {
          int64_t b = std::min(qb, nq - q0);
          auto qv = raft::make_device_matrix_view<const float, int64_t>(dq.data_handle() + q0 * d, b, d);
          auto cv = raft::make_device_matrix_view<int64_t, int64_t>(cand.data_handle() + q0 * kp, b, kp);
          auto dv = raft::make_device_matrix_view<float, int64_t>(cdist.data_handle() + q0 * kp, b, kp);
          cuvs::neighbors::ivf_rabitq::search(res, sp, index, qv, cv, dv);
        }
        res.sync_stream(); double el = now() - s0;
        if (rep == 1) {
          // search-only timing; the candidates (k * ratio per query) are reranked by the caller
          std::vector<int64_t> h(nq * kp);
          raft::copy(res, raft::make_host_matrix_view<int64_t, int64_t>(h.data(), nq, kp), cand.view());
          res.sync_stream();
          for (auto& x : h) if (x >= 0) x += b0;  // global ids
          char fn[512]; snprintf(fn, sizeof fn, "%s/ids_b%ld_q%ld_p%d_r%d.i64", outdir.c_str(), b0, q0, p, r);
          std::ofstream(fn, std::ios::binary).write(reinterpret_cast<char*>(h.data()), h.size() * sizeof(int64_t));
          printf("{\"event\":\"search\",\"n_probes\":%d,\"refine_ratio\":%d,\"kprime\":%ld,\"search_s\":%.6f,\"ids\":\"%s\"}\n", p, r, kp, el, fn);
          fflush(stdout);
        }
      }
    }
  }
  return 0;
}
