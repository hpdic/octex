/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Python binding of the CUDA C++ library (cuvs::neighbors::octex): octex._core. Arrays arrive by
// DLPack (numpy, CuPy, PyTorch, ...) without copies; outputs are written into arrays that the
// caller allocates in the framework of its queries, so that the binding owns no memory besides the
// index. Device arrays that offer only __cuda_array_interface__ are wrapped by wrap(), and device
// outputs for frameworks the package does not know come from empty(). Build and search release the
// GIL.

#include <cuvs/neighbors/octex.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

namespace octex = cuvs::neighbors::octex;

using any_array = nb::ndarray<nb::c_contig>;

void check(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("octex: ") + what + ": " + cudaGetErrorString(e));
  }
}

/** Makes a device current for the lifetime of the guard (a negative device changes nothing). */
class device_guard {
 public:
  explicit device_guard(int device)
  {
    check(cudaGetDevice(&previous_), "cudaGetDevice");
    if (device >= 0 && device != previous_) { check(cudaSetDevice(device), "cudaSetDevice"); }
  }
  ~device_guard() { cudaSetDevice(previous_); }
  device_guard(const device_guard&)                    = delete;
  auto operator=(const device_guard&) -> device_guard& = delete;

 private:
  int previous_ = 0;
};

/** raft::resources of the calling thread and the current device, running on the given stream. */
auto resources(uintptr_t stream) -> raft::resources&
{
  thread_local std::map<int, std::unique_ptr<raft::resources>> per_device;
  int device = 0;
  check(cudaGetDevice(&device), "no CUDA device");
  auto& r = per_device[device];
  if (!r) { r = std::make_unique<raft::resources>(); }
  raft::resource::set_cuda_stream(*r, cuda::stream_ref(reinterpret_cast<cudaStream_t>(stream)));
  return *r;
}

auto is_f32(const any_array& a) -> bool { return a.dtype() == nb::dtype<float>(); }

auto is_f16(const any_array& a) -> bool
{
  const auto t = a.dtype();
  return t.code == uint8_t(nb::dlpack::dtype_code::Float) && t.bits == 16 && t.lanes == 1;
}

auto on_device(const any_array& a) -> bool
{
  if (a.device_type() == nb::device::cuda::value) { return true; }
  if (a.device_type() == nb::device::cpu::value ||
      a.device_type() == nb::device::cuda_host::value) {
    return false;
  }
  throw nb::type_error("octex: arrays must live in host or CUDA device memory");
}

void require_matrix(const any_array& a, const char* name)
{
  if (a.ndim() != 2) {
    throw nb::value_error((std::string("octex: ") + name + " must be a matrix [n, dim]").c_str());
  }
}

auto parse_dtype(const std::string& s) -> nb::dlpack::dtype
{
  if (s == "float32" || s == "<f4") { return nb::dtype<float>(); }
  if (s == "int64" || s == "<i8") { return nb::dtype<int64_t>(); }
  if (s == "int32" || s == "<i4") { return nb::dtype<int32_t>(); }
  if (s == "uint32" || s == "<u4") { return nb::dtype<uint32_t>(); }
  if (s == "float16" || s == "<f2") {
    return nb::dlpack::dtype{uint8_t(nb::dlpack::dtype_code::Float), 16, 1};
  }
  throw nb::type_error(("octex: unsupported dtype " + s).c_str());
}

auto itemsize(nb::dlpack::dtype t) -> size_t { return (t.bits * t.lanes + 7) / 8; }

/** Build: the dataset may live on the host or on a GPU (then copied to the host, as the builder
 * reads it there). */
void build(const octex::index_params& params, any_array dataset, const std::string& path)
{
  require_matrix(dataset, "dataset");
  const bool f32 = is_f32(dataset);
  if (!f32 && !is_f16(dataset)) {
    throw nb::type_error("octex: the dataset must be float32 or float16");
  }
  const auto rows  = int64_t(dataset.shape(0));
  const auto cols  = int64_t(dataset.shape(1));
  const auto bytes = size_t(rows) * size_t(cols) * (f32 ? 4 : 2);
  const bool dev   = on_device(dataset);
  const int dev_id = dev ? int(dataset.device_id()) : -1;

  nb::gil_scoped_release nogil;
  std::unique_ptr<uint8_t[]> staged;
  const void* data = dataset.data();
  if (dev) {
    staged.reset(new uint8_t[bytes > 0 ? bytes : 1]);
    device_guard g(dev_id);
    check(cudaMemcpy(staged.get(), data, bytes, cudaMemcpyDeviceToHost),
          "copying the dataset to the host");
    data = staged.get();
  }
  auto& res = resources(0);
  if (f32) {
    octex::build(res,
                 params,
                 raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
                   static_cast<const float*>(data), rows, cols),
                 path);
  } else {
    octex::build(res,
                 params,
                 raft::make_host_matrix_view<const half, int64_t, raft::row_major>(
                   static_cast<const half*>(data), rows, cols),
                 path);
  }
}

/** Search: queries, neighbors and distances all on the host, or all on one GPU. */
void search(const octex::search_params& params,
            const octex::index& idx,
            any_array queries,
            any_array neighbors,
            any_array distances,
            uintptr_t stream)
{
  require_matrix(queries, "queries");
  require_matrix(neighbors, "neighbors");
  require_matrix(distances, "distances");
  if (!is_f32(queries)) { throw nb::type_error("octex: queries must be float32"); }
  if (neighbors.dtype() != nb::dtype<int64_t>()) {
    throw nb::type_error("octex: neighbors must be int64");
  }
  if (!is_f32(distances)) { throw nb::type_error("octex: distances must be float32"); }
  const auto m = int64_t(queries.shape(0));
  const auto d = int64_t(queries.shape(1));
  const auto k = int64_t(neighbors.shape(1));
  if (d != idx.dim()) {
    throw nb::value_error(("octex: queries have dimension " + std::to_string(d) + ", the index " +
                           std::to_string(idx.dim()))
                            .c_str());
  }
  if (int64_t(neighbors.shape(0)) != m || int64_t(distances.shape(0)) != m ||
      int64_t(distances.shape(1)) != k) {
    throw nb::value_error("octex: neighbors and distances must both be [n_queries, k]");
  }
  const bool dev = on_device(queries);
  if (on_device(neighbors) != dev || on_device(distances) != dev) {
    throw nb::value_error(
      "octex: queries, neighbors and distances must all be on the host or all on the GPU");
  }
  if (dev && (neighbors.device_id() != queries.device_id() ||
              distances.device_id() != queries.device_id())) {
    throw nb::value_error("octex: queries, neighbors and distances must be on the same GPU");
  }
  if (m == 0 || k == 0) { return; }

  nb::gil_scoped_release nogil;
  if (dev) {
    device_guard g(int(queries.device_id()));
    auto& res = resources(stream);
    // the queries were produced on the caller's stream; the results are complete on return
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
          "synchronizing the caller's stream");
    octex::search(res,
                  params,
                  idx,
                  raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
                    static_cast<const float*>(queries.data()), m, d),
                  raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
                    static_cast<int64_t*>(neighbors.data()), m, k),
                  raft::make_device_matrix_view<float, int64_t, raft::row_major>(
                    static_cast<float*>(distances.data()), m, k));
    raft::resource::sync_stream(res);
    check(cudaDeviceSynchronize(), "search");
  } else {
    auto& res = resources(0);
    octex::search(res,
                  params,
                  idx,
                  raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
                    static_cast<const float*>(queries.data()), m, d),
                  raft::make_host_matrix_view<int64_t, int64_t, raft::row_major>(
                    static_cast<int64_t*>(neighbors.data()), m, k),
                  raft::make_host_matrix_view<float, int64_t, raft::row_major>(
                    static_cast<float*>(distances.data()), m, k));
  }
}

/** A device matrix allocated with cudaMalloc, freed when the last DLPack consumer releases it. */
auto empty(int64_t rows, int64_t cols, const std::string& dtype, int device)
  -> nb::ndarray<nb::array_api>
{
  const auto t     = parse_dtype(dtype);
  const auto bytes = size_t(rows) * size_t(cols) * itemsize(t);
  void* p          = nullptr;
  {
    device_guard g(device);
    check(cudaMalloc(&p, bytes > 0 ? bytes : 1), "cudaMalloc");
  }
  nb::capsule owner(p, [](void* q) noexcept { cudaFree(q); });
  const size_t shape[2] = {size_t(rows), size_t(cols)};
  return nb::ndarray<nb::array_api>(
    p, 2, shape, owner, nullptr, t, nb::device::cuda::value, device);
}

/** A C-contiguous device matrix given by a raw pointer (__cuda_array_interface__), kept alive by
 * owner. */
auto wrap(uintptr_t ptr, int64_t rows, int64_t cols, const std::string& dtype, nb::handle owner)
  -> nb::ndarray<nb::array_api>
{
  const auto t = parse_dtype(dtype);
  int device   = 0;
  if (ptr != 0) {
    cudaPointerAttributes attr{};
    check(cudaPointerGetAttributes(&attr, reinterpret_cast<void*>(ptr)),
          "cudaPointerGetAttributes");
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
      throw nb::value_error("octex: __cuda_array_interface__ pointer is not device memory");
    }
    device = attr.device;
  } else {
    check(cudaGetDevice(&device), "cudaGetDevice");
  }
  const size_t shape[2] = {size_t(rows), size_t(cols)};
  return nb::ndarray<nb::array_api>(
    reinterpret_cast<void*>(ptr), 2, shape, owner, nullptr, t, nb::device::cuda::value, device);
}

auto load(const std::string& path, const octex::load_params& params) -> octex::index
{
  nb::gil_scoped_release nogil;
  return octex::load(resources(0), path, params);
}

}  // namespace

NB_MODULE(_core, m)
{
  m.doc() = "CUDA C++ core of octex (cuvs::neighbors::octex)";
  nb::set_leak_warnings(false);

  using ip = octex::index_params;
  nb::class_<ip>(m, "IndexParams")
    .def(nb::init<>())
    .def_rw("copies", &ip::copies)
    .def_rw("blocks", &ip::blocks)
    .def_rw("pca_dims", &ip::pca_dims)
    .def_rw("spacings", &ip::spacings)
    .def_rw("list_len", &ip::list_len)
    .def_rw("min_cell", &ip::min_cell)
    .def_rw("log_buckets", &ip::log_buckets)
    .def_rw("sketch_width", &ip::sketch_width)
    .def_rw("table_len", &ip::table_len)
    .def_rw("reverse_edges", &ip::reverse_edges)
    .def_rw("exact_lists", &ip::exact_lists)
    .def_rw("list_probes", &ip::list_probes)
    .def_rw("member_lists", &ip::member_lists)
    .def_rw("exact_table_max", &ip::exact_table_max)
    .def_rw("table_passes", &ip::table_passes)
    .def_rw("table_sketch_order", &ip::table_sketch_order)
    .def_rw("write_behind", &ip::write_behind)
    .def_rw("seed", &ip::seed)
    .def_rw("devices", &ip::devices);

  using sp = octex::search_params;
  nb::class_<sp>(m, "SearchParams")
    .def(nb::init<>())
    .def_rw("read", &sp::read)
    .def_rw("starts", &sp::starts)
    .def_rw("waves", &sp::waves)
    .def_rw("rerank", &sp::rerank)
    .def_rw("wave_rerank", &sp::wave_rerank)
    .def_rw("batch", &sp::batch)
    .def_rw("sketch_rank", &sp::sketch_rank)
    .def_rw("final_rerank", &sp::final_rerank)
    .def_rw("balanced_gather", &sp::balanced_gather)
    .def_rw("fused_merge", &sp::fused_merge)
    .def_rw("use_graph", &sp::use_graph);

  nb::enum_<octex::placement>(m, "Placement")
    .value("replicate", octex::placement::replicate)
    .value("shard", octex::placement::shard);

  nb::enum_<octex::sketch_placement>(m, "SketchPlacement")
    .value("replicate", octex::sketch_placement::replicate)
    .value("shard", octex::sketch_placement::shard)
    .value("host", octex::sketch_placement::host);

  using lp = octex::load_params;
  nb::class_<lp>(m, "LoadParams")
    .def(nb::init<>())
    .def_rw("devices", &lp::devices)
    .def_rw("mode", &lp::mode)
    .def_rw("vectors_on_host", &lp::vectors_on_host)
    .def_rw("shard_vectors", &lp::shard_vectors)
    .def_rw("table_on_host", &lp::table_on_host)
    .def_rw("sketch", &lp::sketch)
    .def_rw("parts", &lp::parts)
    .def_rw("copies_per_part", &lp::copies_per_part)
    .def_rw("copies_limit", &lp::copies_limit)
    .def_rw("list_limit", &lp::list_limit);

  nb::class_<octex::index>(m, "Index")
    .def_prop_ro("size", &octex::index::size)
    .def_prop_ro("dim", &octex::index::dim)
    .def_prop_ro("copies", &octex::index::copies)
    .def_prop_ro("devices", [](const octex::index& i) { return std::vector<int>(i.devices()); })
    .def_prop_ro("device_bytes", &octex::index::device_bytes);

  m.def("build", &build, nb::arg("params"), nb::arg("dataset"), nb::arg("path"));
  m.def("load", &load, nb::arg("path"), nb::arg("params"));
  m.def("search",
        &search,
        nb::arg("params"),
        nb::arg("index"),
        nb::arg("queries"),
        nb::arg("neighbors").noconvert(),
        nb::arg("distances").noconvert(),
        nb::arg("stream") = 0);
  m.def(
    "shape",
    [](any_array a) {
      std::vector<int64_t> s;
      for (size_t i = 0; i < a.ndim(); ++i) {
        s.push_back(int64_t(a.shape(i)));
      }
      return s;
    },
    nb::arg("array"));
  m.def("empty", &empty, nb::arg("rows"), nb::arg("cols"), nb::arg("dtype"), nb::arg("device"));
  m.def("wrap",
        &wrap,
        nb::arg("ptr"),
        nb::arg("rows"),
        nb::arg("cols"),
        nb::arg("dtype"),
        nb::arg("owner"));
}
