/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Minimal .npy (format 1.0 / 2.0) reader and writer for the index directory: little-endian, C
// order. Reading maps the file; the data pointer stays valid while the npy_file lives.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cuvs::neighbors::octex::detail {

/**
 * Leaves elements uninitialized: a vector of hundreds of GB is then not zeroed on one thread before
 * parallel readers or GPU copies fill it.
 */
template <typename T>
struct no_init_allocator : std::allocator<T> {
  template <typename U>
  struct rebind {
    using other = no_init_allocator<U>;
  };
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args)
  {
    if constexpr (sizeof...(Args) == 0) {
      ::new (static_cast<void*>(p)) U;
    } else {
      ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
  }
};

/**
 * Arrays of an index directory that the build still holds in memory while a background thread
 * writes them (OCTEX_WRITE_BEHIND): opening such a path reads the memory instead of the file.
 */
struct mem_array {
  const char* data = nullptr;
  std::vector<int64_t> shape;
  std::string descr;
  size_t bytes = 0;
  bool pinned  = false;  // page-locked by its owner (cudaHostRegister of anonymous memory)
};
inline auto mem_key(std::string const& path) -> std::string
{
  return std::filesystem::absolute(path).lexically_normal().string();
}
inline auto mem_registry() -> std::pair<std::mutex, std::map<std::string, mem_array>>&
{
  static std::pair<std::mutex, std::map<std::string, mem_array>> r;
  return r;
}
inline void register_mem_array(std::string const& path,
                               const void* data,
                               std::vector<int64_t> const& shape,
                               std::string const& descr,
                               size_t bytes,
                               bool pinned = false)
{
  auto& r = mem_registry();
  std::lock_guard<std::mutex> lock(r.first);
  r.second[mem_key(path)] = mem_array{static_cast<const char*>(data), shape, descr, bytes, pinned};
}
inline void unregister_mem_array(std::string const& path)
{
  auto& r = mem_registry();
  std::lock_guard<std::mutex> lock(r.first);
  r.second.erase(mem_key(path));
}
inline void clear_mem_arrays()
{
  auto& r = mem_registry();
  std::lock_guard<std::mutex> lock(r.first);
  r.second.clear();
}
inline auto find_mem_array(std::string const& path) -> std::optional<mem_array>
{
  auto& r = mem_registry();
  std::lock_guard<std::mutex> lock(r.first);
  if (r.second.empty()) return std::nullopt;
  auto it = r.second.find(mem_key(path));
  if (it == r.second.end()) return std::nullopt;
  return it->second;
}

/** dst = src (bytes) on several threads. */
inline void parallel_memcpy(void* dst, const void* src, size_t bytes, int threads = 16)
{
  std::vector<std::thread> th;
  for (int t = 0; t < threads; t++)
    th.emplace_back([=] {
      size_t a = bytes * t / threads, b = bytes * (t + 1) / threads;
      std::memcpy(static_cast<char*>(dst) + a, static_cast<const char*>(src) + a, b - a);
    });
  for (auto& x : th)
    x.join();
}

/**
 * bytes of fd at offset into dst by ordinary buffered reads on several threads: a network block
 * device reads about 8x faster with eight requests in flight than with one.
 */
inline void parallel_pread(int fd, char* dst, size_t bytes, size_t offset, int threads = 16)
{
  constexpr size_t kChunk = size_t{64} << 20;
  size_t chunks           = (bytes + kChunk - 1) / kChunk;
  threads = static_cast<int>(std::max<size_t>(1, std::min<size_t>(threads, chunks)));
  std::vector<std::thread> th;
  std::vector<int> bad(threads, 0);
  for (int t = 0; t < threads; t++)
    th.emplace_back([&, t] {
      for (size_t c = t; c < chunks; c += threads) {
        size_t at = c * kChunk, end = std::min(bytes, at + kChunk);
        while (at < end) {
          ssize_t n = ::pread(fd, dst + at, end - at, offset + at);
          if (n <= 0) {
            bad[t] = 1;
            return;
          }
          at += static_cast<size_t>(n);
        }
      }
    });
  for (auto& x : th)
    x.join();
  for (int v : bad)
    if (v) throw std::runtime_error("octex: short parallel read");
}

/** bytes of src to fd at offset by buffered writes on several threads (as parallel_pread). */
inline void parallel_pwrite(int fd, const char* src, size_t bytes, size_t offset, int threads = 16)
{
  constexpr size_t kChunk = size_t{64} << 20;
  size_t chunks           = (bytes + kChunk - 1) / kChunk;
  threads = static_cast<int>(std::max<size_t>(1, std::min<size_t>(threads, chunks)));
  std::vector<std::thread> th;
  std::vector<int> bad(threads, 0);
  for (int t = 0; t < threads; t++)
    th.emplace_back([&, t] {
      for (size_t c = t; c < chunks; c += threads) {
        size_t at = c * kChunk, end = std::min(bytes, at + kChunk);
        while (at < end) {
          ssize_t n = ::pwrite(fd, src + at, end - at, offset + at);
          if (n <= 0) {
            bad[t] = 1;
            return;
          }
          at += static_cast<size_t>(n);
        }
      }
    });
  for (auto& x : th)
    x.join();
  for (int v : bad)
    if (v) throw std::runtime_error("octex: short parallel write");
}

/** The whole file into the page cache by parallel buffered reads (later mapped reads hit it). */
inline void prefetch_file(std::string const& path, int threads = 16)
{
  if (find_mem_array(path)) return;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  struct stat st{};
  ::fstat(fd, &st);
  size_t bytes            = static_cast<size_t>(st.st_size);
  constexpr size_t kChunk = size_t{64} << 20;
  size_t chunks           = (bytes + kChunk - 1) / kChunk;
  threads = static_cast<int>(std::max<size_t>(1, std::min<size_t>(threads, chunks)));
  std::vector<std::thread> th;
  for (int t = 0; t < threads; t++)
    th.emplace_back([&, t] {
      std::vector<char> buf(kChunk);
      for (size_t c = t; c < chunks; c += threads) {
        size_t at = c * kChunk, end = std::min(bytes, at + kChunk);
        while (at < end) {
          ssize_t n = ::pread(fd, buf.data(), end - at, at);
          if (n <= 0) break;
          at += static_cast<size_t>(n);
        }
      }
    });
  for (auto& x : th)
    x.join();
  ::close(fd);
}

template <typename T>
constexpr auto npy_descr() -> const char*
{
  if constexpr (std::is_same_v<T, float>) return "<f4";
  if constexpr (std::is_same_v<T, double>) return "<f8";
  if constexpr (std::is_same_v<T, int8_t>) return "|i1";
  if constexpr (std::is_same_v<T, uint8_t>) return "|u1";
  if constexpr (std::is_same_v<T, int32_t>) return "<i4";
  if constexpr (std::is_same_v<T, int64_t>) return "<i8";
  if constexpr (sizeof(T) == 2) return "<f2";  // half
  return "";
}

struct npy_header {
  std::string descr;
  std::vector<int64_t> shape;
  size_t header_len = 0;
  size_t data_bytes = 0;
};

inline auto read_npy_header(std::string const& path) -> npy_header
{
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("octex: cannot open " + path);
  struct stat st{};
  ::fstat(fd, &st);
  char buf[4096];
  ssize_t nr = ::read(fd, buf, sizeof(buf));
  if (nr < 10 || std::memcmp(buf, "\x93NUMPY", 6) != 0) {
    ::close(fd);
    throw std::runtime_error("octex: not an .npy file: " + path);
  }
  int major     = static_cast<unsigned char>(buf[6]);
  size_t hlen   = major == 1 ? (uint8_t(buf[8]) | (uint8_t(buf[9]) << 8))
                             : (uint8_t(buf[8]) | (uint8_t(buf[9]) << 8) | (uint8_t(buf[10]) << 16) |
                              (size_t(uint8_t(buf[11])) << 24));
  size_t hstart = major == 1 ? 10 : 12;
  std::string hdr;
  if (hstart + hlen <= static_cast<size_t>(nr)) {
    hdr.assign(buf + hstart, hlen);
  } else {
    std::vector<char> big_hdr(hlen);
    ssize_t n = ::pread(fd, big_hdr.data(), hlen, hstart);
    if (n < static_cast<ssize_t>(hlen)) {
      ::close(fd);
      throw std::runtime_error("octex: truncated header: " + path);
    }
    hdr.assign(big_hdr.data(), hlen);
  }
  ::close(fd);
  if (hdr.find("'fortran_order': True") != std::string::npos)
    throw std::runtime_error("octex: Fortran order not supported: " + path);
  npy_header out;
  out.header_len = hstart + hlen;
  out.data_bytes = static_cast<size_t>(st.st_size) > out.header_len
                     ? static_cast<size_t>(st.st_size) - out.header_len
                     : 0;
  auto d         = hdr.find("'descr':");
  auto q1        = hdr.find('\'', d + 8);
  auto q2        = hdr.find('\'', q1 + 1);
  out.descr      = hdr.substr(q1 + 1, q2 - q1 - 1);
  auto s         = hdr.find('(', hdr.find("'shape':"));
  auto e         = hdr.find(')', s);
  std::string sh = hdr.substr(s + 1, e - s - 1);
  size_t pos     = 0;
  while (pos < sh.size()) {
    while (pos < sh.size() && (sh[pos] == ' ' || sh[pos] == ','))
      pos++;
    if (pos >= sh.size()) break;
    size_t end = pos;
    while (end < sh.size() && sh[end] != ',')
      end++;
    out.shape.push_back(std::stoll(sh.substr(pos, end - pos)));
    pos = end;
  }
  return out;
}

struct npy_file {
  std::string descr;
  std::vector<int64_t> shape;
  const char* data = nullptr;
  size_t bytes     = 0;
  void* map_       = nullptr;
  size_t map_len_  = 0;

  npy_file() = default;
  explicit npy_file(std::string const& path)
  {
    if (auto m = find_mem_array(path)) {  // held in memory by the build (no mapping to undo)
      descr = m->descr;
      shape = m->shape;
      data  = m->data;
      bytes = m->bytes;
      return;
    }
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("octex: cannot open " + path);
    struct stat st{};
    ::fstat(fd, &st);
    map_len_ = static_cast<size_t>(st.st_size);
    map_     = ::mmap(nullptr, map_len_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) throw std::runtime_error("octex: cannot map " + path);
    auto* p = static_cast<const char*>(map_);
    if (map_len_ < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0)
      throw std::runtime_error("octex: not an .npy file: " + path);
    int major     = p[6];
    size_t hlen   = major == 1 ? (uint8_t(p[8]) | (uint8_t(p[9]) << 8))
                               : (uint8_t(p[8]) | (uint8_t(p[9]) << 8) | (uint8_t(p[10]) << 16) |
                                (size_t(uint8_t(p[11])) << 24));
    size_t hstart = major == 1 ? 10 : 12;
    std::string hdr(p + hstart, hlen);
    if (hdr.find("'fortran_order': True") != std::string::npos)
      throw std::runtime_error("octex: Fortran order not supported: " + path);
    auto d         = hdr.find("'descr':");
    auto q1        = hdr.find('\'', d + 8);
    auto q2        = hdr.find('\'', q1 + 1);
    descr          = hdr.substr(q1 + 1, q2 - q1 - 1);
    auto s         = hdr.find('(', hdr.find("'shape':"));
    auto e         = hdr.find(')', s);
    std::string sh = hdr.substr(s + 1, e - s - 1);
    size_t pos     = 0;
    while (pos < sh.size()) {
      while (pos < sh.size() && (sh[pos] == ' ' || sh[pos] == ','))
        pos++;
      if (pos >= sh.size()) break;
      size_t end = pos;
      while (end < sh.size() && sh[end] != ',')
        end++;
      shape.push_back(std::stoll(sh.substr(pos, end - pos)));
      pos = end;
    }
    data  = p + hstart + hlen;
    bytes = map_len_ - (hstart + hlen);
  }
  npy_file(npy_file const&)                    = delete;
  auto operator=(npy_file const&) -> npy_file& = delete;
  npy_file(npy_file&& o) noexcept { *this = std::move(o); }
  auto operator=(npy_file&& o) noexcept -> npy_file&
  {
    std::swap(descr, o.descr);
    std::swap(shape, o.shape);
    std::swap(data, o.data);
    std::swap(bytes, o.bytes);
    std::swap(map_, o.map_);
    std::swap(map_len_, o.map_len_);
    return *this;
  }
  ~npy_file()
  {
    if (map_ != nullptr && map_ != MAP_FAILED) ::munmap(map_, map_len_);
  }

  [[nodiscard]] auto numel() const -> int64_t
  {
    int64_t n = 1;
    for (auto v : shape)
      n *= v;
    return n;
  }
  template <typename T>
  [[nodiscard]] auto as() const -> const T*
  {
    if (descr != npy_descr<T>())
      throw std::runtime_error("octex: .npy dtype " + descr + ", expected " + npy_descr<T>());
    return reinterpret_cast<const T*>(data);
  }
};

/** Write a C-order array of the given shape (format 1.0, header padded to 64 bytes). */
template <typename T>
void write_npy(std::string const& path, const T* data, std::vector<int64_t> const& shape)
{
  std::string sh = "(";
  int64_t n      = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    if (i + 1 < shape.size()) sh += " ";
    n *= shape[i];
  }
  sh += ")";
  std::string hdr = std::string("{'descr': '") + npy_descr<T>() +
                    "', 'fortran_order': False, 'shape': " + sh + ", }";
  size_t total = 10 + hdr.size() + 1;
  hdr.append((64 - total % 64) % 64, ' ');
  hdr += '\n';
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("octex: cannot write " + path);
  const char magic[] = "\x93NUMPY\x01\x00";
  std::fwrite(magic, 1, 8, f);
  uint16_t hl = static_cast<uint16_t>(hdr.size());
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(hdr.data(), 1, hdr.size(), f);
  size_t want = static_cast<size_t>(n) * sizeof(T);
  std::fflush(f);
  try {
    if (want > 0)
      parallel_pwrite(::fileno(f), reinterpret_cast<const char*>(data), want, 10 + hdr.size());
  } catch (...) {
    std::fclose(f);
    throw std::runtime_error("octex: short write " + path);
  }
  std::fclose(f);
}

}  // namespace cuvs::neighbors::octex::detail
