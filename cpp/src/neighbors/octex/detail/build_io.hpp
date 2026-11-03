/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Streaming .npy writer (the array is written in pieces, then checked against its shape) for the
// large arrays of the index directory.

#include "npy.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvs::neighbors::octex::detail {

template <typename T>
class npy_writer {
 public:
  npy_writer(std::string const& path, std::vector<int64_t> const& shape) : path_(path)
  {
    std::string sh = "(";
    want_          = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
      if (i + 1 < shape.size()) sh += " ";
      want_ *= shape[i];
    }
    sh += ")";
    std::string hdr = std::string("{'descr': '") + npy_descr<T>() +
                      "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + hdr.size() + 1;
    hdr.append((64 - total % 64) % 64, ' ');
    hdr += '\n';
    f_ = std::fopen(path.c_str(), "wb");
    if (f_ == nullptr) throw std::runtime_error("octex: cannot write " + path);
    const char magic[] = "\x93NUMPY\x01\x00";
    std::fwrite(magic, 1, 8, f_);
    auto hl = static_cast<uint16_t>(hdr.size());
    std::fwrite(&hl, 2, 1, f_);
    std::fwrite(hdr.data(), 1, hdr.size(), f_);
    std::fflush(f_);
    pos_ = 10 + hdr.size();
  }
  npy_writer(npy_writer const&)                    = delete;
  auto operator=(npy_writer const&) -> npy_writer& = delete;
  ~npy_writer()
  {
    if (f_ != nullptr) std::fclose(f_);
  }

  void write(const T* data, int64_t count)
  {
    if (count <= 0) return;
    size_t bytes = size_t(count) * sizeof(T);
    try {
      parallel_pwrite(::fileno(f_), reinterpret_cast<const char*>(data), bytes, pos_);
    } catch (...) {
      throw std::runtime_error("octex: short write " + path_);
    }
    pos_ += bytes;
    done_ += count;
  }
  void zeros(int64_t count)
  {
    static const std::vector<T> z(int64_t(1) << 16, T{});
    while (count > 0) {
      int64_t m = std::min<int64_t>(count, z.size());
      write(z.data(), m);
      count -= m;
    }
  }
  void close()
  {
    if (done_ != want_) throw std::runtime_error("octex: wrong element count in " + path_);
    if (std::fclose(f_) != 0) throw std::runtime_error("octex: cannot close " + path_);
    f_ = nullptr;
  }

 private:
  std::string path_;
  FILE* f_      = nullptr;
  int64_t want_ = 0, done_ = 0;
  size_t pos_   = 0;  // file offset of the next element
};

}  // namespace cuvs::neighbors::octex::detail
