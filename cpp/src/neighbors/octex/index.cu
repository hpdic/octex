/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "detail/index_impl.hpp"

#include <cuvs/neighbors/octex.hpp>

namespace cuvs::neighbors::octex {

namespace detail {
index_impl::~index_impl()
{
  plans.clear();  // graphs and batch buffers before the index arrays
  parts.clear();
}
}  // namespace detail

index::index(std::shared_ptr<detail::index_impl> impl) : impl_(std::move(impl)) {}
index::index(index&&) noexcept                    = default;
auto index::operator=(index&&) noexcept -> index& = default;
index::~index()                                   = default;

auto index::size() const noexcept -> int64_t { return impl_->n; }
auto index::dim() const noexcept -> int64_t { return impl_->dim; }
auto index::copies() const noexcept -> int { return impl_->copies; }
auto index::devices() const noexcept -> std::vector<int> const& { return impl_->devices; }
auto index::impl() const noexcept -> detail::index_impl& { return *impl_; }

auto index::device_bytes() const -> std::vector<int64_t>
{
  std::vector<int64_t> out;
  for (auto const& p : impl_->parts)
    out.push_back(p->bytes);
  return out;
}

}  // namespace cuvs::neighbors::octex
