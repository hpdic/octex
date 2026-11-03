/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// C API of cuvs::neighbors::octex, in the layout of the cuVS C API (c/src/neighbors).

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cstdint>
#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <string>
#include <vector>

#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/octex.h>
#include <cuvs/neighbors/octex.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

namespace octex = cuvs::neighbors::octex;

auto to_cpp(cuvsOctexIndexParams const& p) -> octex::index_params
{
  octex::index_params out;
  out.copies   = p.copies;
  out.blocks   = p.blocks;
  out.pca_dims = p.pca_dims;
  if (p.spacings != nullptr && p.n_spacings > 0) {
    out.spacings.assign(p.spacings, p.spacings + p.n_spacings);
  }
  out.list_len           = p.list_len;
  out.min_cell           = p.min_cell;
  out.log_buckets        = p.log_buckets;
  out.sketch_width       = p.sketch_width;
  out.table_len          = p.table_len;
  out.reverse_edges      = p.reverse_edges;
  out.exact_lists        = p.exact_lists;
  out.list_probes        = p.list_probes;
  out.member_lists       = p.member_lists;
  out.exact_table_max    = p.exact_table_max;
  out.table_passes       = p.table_passes;
  out.table_sketch_order = p.table_sketch_order;
  out.write_behind       = p.write_behind;
  out.seed               = p.seed;
  if (p.devices != nullptr && p.n_devices > 0) {
    out.devices.assign(p.devices, p.devices + p.n_devices);
  }
  return out;
}

auto to_cpp(cuvsOctexLoadParams const& p) -> octex::load_params
{
  RAFT_EXPECTS(p.placement == OCTEX_PLACEMENT_REPLICATE || p.placement == OCTEX_PLACEMENT_SHARD,
               "octex: unknown placement");
  RAFT_EXPECTS(p.sketch == OCTEX_SKETCH_REPLICATE || p.sketch == OCTEX_SKETCH_SHARD ||
                 p.sketch == OCTEX_SKETCH_HOST,
               "octex: unknown sketch placement");
  octex::load_params out;
  if (p.devices != nullptr && p.n_devices > 0) {
    out.devices.assign(p.devices, p.devices + p.n_devices);
  }
  out.mode            = static_cast<octex::placement>(int(p.placement));
  out.vectors_on_host = p.vectors_on_host;
  out.shard_vectors   = p.shard_vectors;
  out.table_on_host   = p.table_on_host;
  out.sketch          = static_cast<octex::sketch_placement>(int(p.sketch));
  out.parts           = p.parts;
  out.copies_per_part = p.copies_per_part;
  out.copies_limit    = p.copies_limit;
  out.list_limit      = p.list_limit;
  return out;
}

auto to_cpp(cuvsOctexSearchParams const& p) -> octex::search_params
{
  octex::search_params out;
  out.read            = p.read;
  out.starts          = p.starts;
  out.waves           = p.waves;
  out.rerank          = p.rerank;
  out.wave_rerank     = p.wave_rerank;
  out.batch           = p.batch;
  out.sketch_rank     = p.sketch_rank;
  out.final_rerank    = p.final_rerank;
  out.balanced_gather = p.balanced_gather;
  out.fused_merge     = p.fused_merge;
  out.use_graph       = p.use_graph;
  return out;
}

auto as_index(cuvsOctexIndex const& index) -> octex::index*
{
  RAFT_EXPECTS(index.addr != 0, "octex: the index is empty (load it first)");
  return reinterpret_cast<octex::index*>(index.addr);
}

/** Device-only memory (kDLCUDA); pinned and managed memory is read in place from the host. */
auto device_only(DLTensor const& t) -> bool { return t.device.device_type == kDLCUDA; }

template <typename T>
void build_typed(raft::resources const& res,
                 octex::index_params const& p,
                 DLManagedTensor* x,
                 std::string const& path)
{
  using view = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  if (device_only(x->dl_tensor)) {  // the build reads the dataset from the host
    auto dv =
      cuvs::core::from_dlpack<raft::device_matrix_view<const T, int64_t, raft::row_major>>(x);
    std::vector<T> host(static_cast<size_t>(dv.extent(0) * dv.extent(1)));
    RAFT_CUDA_TRY(
      cudaMemcpy(host.data(), dv.data_handle(), sizeof(T) * host.size(), cudaMemcpyDefault));
    octex::build(res, p, view(host.data(), dv.extent(0), dv.extent(1)), path);
  } else {
    octex::build(res, p, cuvs::core::from_dlpack<view>(x), path);
  }
}

}  // namespace

extern "C" cuvsError_t cuvsOctexIndexParamsCreate(cuvsOctexIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    octex::index_params d;
    *params = new cuvsOctexIndexParams{d.copies,
                                       d.blocks,
                                       d.pca_dims,
                                       nullptr,
                                       0,
                                       d.list_len,
                                       d.min_cell,
                                       d.log_buckets,
                                       d.sketch_width,
                                       d.table_len,
                                       d.reverse_edges,
                                       d.exact_lists,
                                       d.list_probes,
                                       d.member_lists,
                                       d.exact_table_max,
                                       d.table_passes,
                                       d.table_sketch_order,
                                       d.write_behind,
                                       d.seed,
                                       nullptr,
                                       0};
  });
}

extern "C" cuvsError_t cuvsOctexIndexParamsDestroy(cuvsOctexIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsOctexLoadParamsCreate(cuvsOctexLoadParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    octex::load_params d;
    *params = new cuvsOctexLoadParams{nullptr,
                                      0,
                                      static_cast<cuvsOctexPlacement>(int(d.mode)),
                                      d.vectors_on_host,
                                      d.shard_vectors,
                                      d.table_on_host,
                                      static_cast<cuvsOctexSketchPlacement>(int(d.sketch)),
                                      d.parts,
                                      d.copies_per_part,
                                      d.copies_limit,
                                      d.list_limit};
  });
}

extern "C" cuvsError_t cuvsOctexLoadParamsDestroy(cuvsOctexLoadParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsOctexSearchParamsCreate(cuvsOctexSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    octex::search_params d;
    *params = new cuvsOctexSearchParams{d.read,
                                        d.starts,
                                        d.waves,
                                        d.rerank,
                                        d.wave_rerank,
                                        d.batch,
                                        d.sketch_rank,
                                        d.final_rerank,
                                        d.balanced_gather,
                                        d.fused_merge,
                                        d.use_graph};
  });
}

extern "C" cuvsError_t cuvsOctexSearchParamsDestroy(cuvsOctexSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsOctexIndexCreate(cuvsOctexIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsOctexIndex{}; });
}

extern "C" cuvsError_t cuvsOctexIndexDestroy(cuvsOctexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->addr != 0) { delete reinterpret_cast<octex::index*>(index->addr); }
    delete index;
  });
}

extern "C" cuvsError_t cuvsOctexIndexGetSize(cuvsOctexIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] { *size = as_index(*index)->size(); });
}

extern "C" cuvsError_t cuvsOctexIndexGetDim(cuvsOctexIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] { *dim = as_index(*index)->dim(); });
}

extern "C" cuvsError_t cuvsOctexIndexGetCopies(cuvsOctexIndex_t index, int32_t* copies)
{
  return cuvs::core::translate_exceptions([=] { *copies = as_index(*index)->copies(); });
}

extern "C" cuvsError_t cuvsOctexIndexGetNumDevices(cuvsOctexIndex_t index, int32_t* n_devices)
{
  return cuvs::core::translate_exceptions(
    [=] { *n_devices = static_cast<int32_t>(as_index(*index)->devices().size()); });
}

extern "C" cuvsError_t cuvsOctexIndexGetDevices(cuvsOctexIndex_t index, int32_t* devices)
{
  return cuvs::core::translate_exceptions([=] {
    auto const& d = as_index(*index)->devices();
    for (size_t i = 0; i < d.size(); ++i) {
      devices[i] = d[i];
    }
  });
}

extern "C" cuvsError_t cuvsOctexIndexGetDeviceBytes(cuvsOctexIndex_t index, int64_t* bytes)
{
  return cuvs::core::translate_exceptions([=] {
    const auto b = as_index(*index)->device_bytes();
    for (size_t i = 0; i < b.size(); ++i) {
      bytes[i] = b[i];
    }
  });
}

extern "C" cuvsError_t cuvsOctexBuild(cuvsResources_t res,
                                      cuvsOctexIndexParams_t params,
                                      DLManagedTensor* dataset,
                                      const char* path)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(path != nullptr, "octex: the index directory is NULL");
    auto* res_ptr    = reinterpret_cast<raft::resources*>(res);
    const auto p     = to_cpp(*params);
    const auto dtype = dataset->dl_tensor.dtype;
    const std::string dir(path);
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      build_typed<float>(*res_ptr, p, dataset, dir);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      build_typed<half>(*res_ptr, p, dataset, dir);
    } else {
      RAFT_FAIL("octex: the dataset must be float32 or float16");
    }
  });
}

extern "C" cuvsError_t cuvsOctexLoad(cuvsResources_t res,
                                     const char* path,
                                     cuvsOctexLoadParams_t params,
                                     cuvsOctexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index->addr == 0, "octex: the index handle already holds an index");
    RAFT_EXPECTS(path != nullptr, "octex: the index directory is NULL");
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    const auto p  = params != nullptr ? to_cpp(*params) : octex::load_params{};
    index->addr =
      reinterpret_cast<uintptr_t>(new octex::index(octex::load(*res_ptr, std::string(path), p)));
  });
}

extern "C" cuvsError_t cuvsOctexSearch(cuvsResources_t res,
                                       cuvsOctexSearchParams_t params,
                                       cuvsOctexIndex_t index,
                                       DLManagedTensor* queries,
                                       DLManagedTensor* neighbors,
                                       DLManagedTensor* distances)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr  = reinterpret_cast<raft::resources*>(res);
    const auto p   = to_cpp(*params);
    auto const& ix = *as_index(*index);
    const auto t   = queries->dl_tensor.device.device_type;
    if (t == kDLCUDA || t == kDLCUDAManaged) {
      using q_view = raft::device_matrix_view<const float, int64_t, raft::row_major>;
      using n_view = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
      using d_view = raft::device_matrix_view<float, int64_t, raft::row_major>;
      octex::search(*res_ptr,
                    p,
                    ix,
                    cuvs::core::from_dlpack<q_view>(queries),
                    cuvs::core::from_dlpack<n_view>(neighbors),
                    cuvs::core::from_dlpack<d_view>(distances));
    } else {
      using q_view = raft::host_matrix_view<const float, int64_t, raft::row_major>;
      using n_view = raft::host_matrix_view<int64_t, int64_t, raft::row_major>;
      using d_view = raft::host_matrix_view<float, int64_t, raft::row_major>;
      octex::search(*res_ptr,
                    p,
                    ix,
                    cuvs::core::from_dlpack<q_view>(queries),
                    cuvs::core::from_dlpack<n_view>(neighbors),
                    cuvs::core::from_dlpack<d_view>(distances));
    }
  });
}
