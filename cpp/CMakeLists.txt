# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# octex for cuVS: the CUDA C++ implementation, laid out as a cuVS module (include/cuvs/neighbors,
# src/neighbors/octex) so that it can move into cuVS unchanged. It builds against an installed cuVS,
# e.g. the libcuvs pip wheel:
#
#   SP=$(python -c "import site; print(site.getsitepackages()[0])")
#   cmake -S cpp -B build -DCMAKE_PREFIX_PATH="$SP/libcuvs/lib64/cmake;$SP/libraft/lib64/cmake;..."
#   cmake --build build -j

cmake_minimum_required(VERSION 3.26)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()
project(octex_cuvs VERSION 0.1.0 LANGUAGES C CXX CUDA)

option(OCTEX_BUILD_TESTS "Build the tests and benchmarks" ON)

find_package(cuvs REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(OCTEX_NVJITLINK "" CACHE FILEPATH "nvJitLink library required by the installed cuVS")
set(OCTEX_RPATH "" CACHE STRING "extra run-time search paths (; separated)")
set(CMAKE_BUILD_RPATH "${OCTEX_RPATH}")
if(OCTEX_RPATH)
  add_link_options("LINKER:--disable-new-dtags")
endif()

# every translation unit of the module: search, load and build each live in their own files
file(GLOB OCTEX_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/octex/*.cu"
                                          "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/octex/*.cpp")
add_library(octex_cuvs SHARED ${OCTEX_SOURCES})
set_target_properties(octex_cuvs PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                            CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)
target_include_directories(octex_cuvs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(octex_cuvs PRIVATE
                       $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda -lineinfo>)
target_link_libraries(octex_cuvs PUBLIC ${OCTEX_NVJITLINK} cuvs::cuvs PRIVATE CUDA::cublas)

# C API (c/, in the layout of the cuVS C API): liboctex_c on cuvs::c_api for the resource handle
# and the error text; the DLPack header from the system or the pinned release
option(OCTEX_BUILD_C_API "Build the C API (c/)" ON)
if(OCTEX_BUILD_C_API)
  find_package(cuvs REQUIRED COMPONENTS c_api)
  find_path(OCTEX_DLPACK_INCLUDE dlpack/dlpack.h)
  if(NOT OCTEX_DLPACK_INCLUDE)
    include(FetchContent)
    # SOURCE_SUBDIR without a CMakeLists.txt: download only
    FetchContent_Declare(dlpack GIT_REPOSITORY https://github.com/dmlc/dlpack.git
                                GIT_TAG v1.0 GIT_SHALLOW TRUE SOURCE_SUBDIR headers-only)
    FetchContent_MakeAvailable(dlpack)
    set(OCTEX_DLPACK_INCLUDE "${dlpack_SOURCE_DIR}/include" CACHE PATH "" FORCE)
  endif()
  add_library(octex_c SHARED "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/neighbors/octex.cpp")
  set_target_properties(octex_c PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
                                           POSITION_INDEPENDENT_CODE ON)
  target_include_directories(octex_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../c/include"
                                            "${OCTEX_DLPACK_INCLUDE}")
  target_link_libraries(octex_c PUBLIC cuvs::c_api CUDA::cudart PRIVATE octex_cuvs)
endif()

# Examples (examples/cpp and examples/c): the smallest complete programs on the C++ module and the C API
option(OCTEX_BUILD_EXAMPLES "Build the C++ and C examples" ON)
if(OCTEX_BUILD_EXAMPLES)
  add_executable(octex_example_search "${CMAKE_CURRENT_SOURCE_DIR}/../examples/cpp/search.cu")
  target_link_libraries(octex_example_search PRIVATE octex_cuvs)
  set_target_properties(octex_example_search PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(octex_example_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(OCTEX_BUILD_C_API)
    add_executable(octex_example_c "${CMAKE_CURRENT_SOURCE_DIR}/../examples/c/search.c")
    target_link_libraries(octex_example_c PRIVATE octex_c m)
  endif()
endif()

if(NOT OCTEX_BUILD_TESTS)
  return()
endif()

enable_testing()
file(GLOB OCTEX_TOOLS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
                                        "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu")
foreach(src IN LISTS OCTEX_TOOLS)  # one executable per file
  get_filename_component(name "${src}" NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE octex_cuvs CUDA::cublas)
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set_target_properties(${name} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
  if(src MATCHES "/tests/")
    add_test(NAME ${name} COMMAND ${name})
  endif()
endforeach()

# the C API against exhaustive search; its driver c/tests/neighbors/run_octex_c.c compiles as C
if(OCTEX_BUILD_C_API)
  add_executable(octex_c_test "${CMAKE_CURRENT_SOURCE_DIR}/../c/tests/neighbors/octex_c.cu"
                              "${CMAKE_CURRENT_SOURCE_DIR}/../c/tests/neighbors/run_octex_c.c")
  target_link_libraries(octex_c_test PRIVATE octex_c)
  set_target_properties(octex_c_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME octex_c_test COMMAND octex_c_test)
endif()
