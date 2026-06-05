cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-float-conversion -O3 -std=c++20)

# Keep the three executable variants independent at configure time. The PBS
# job intentionally configures two separate build trees: the baseline container
# enables matrix_based and matrix_free_no_simd, while the AVX512 container only
# enables matrix_free_simd. This prevents a target from being accidentally
# compiled with the wrong compiler/runtime environment.
option(MFSOLVER_BUILD_MATRIX_BASED "Build the matrix-based executable" ON)
option(MFSOLVER_BUILD_MATRIX_FREE_NO_SIMD "Build the non-SIMD matrix-free executable" ON)
option(MFSOLVER_BUILD_MATRIX_FREE_SIMD "Build the SIMD matrix-free executable" ON)

# The run scripts execute ./matrix_based, ./matrix_free_no_simd, and
# ./matrix_free_simd from the repository root. Keep that behavior even when
# CMake is configured out-of-source in build/baseline or build/avx512.
set(MFSOLVER_RUNTIME_OUTPUT_DIR
    "${CMAKE_SOURCE_DIR}"
    CACHE PATH "Directory where solver executables are written")

if(MFSOLVER_BUILD_MATRIX_BASED)
  add_executable(matrix_based src/MB_main.cpp)
  set_target_properties(matrix_based PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${MFSOLVER_RUNTIME_OUTPUT_DIR}")
  deal_ii_setup_target(matrix_based)
endif()

if(MFSOLVER_BUILD_MATRIX_FREE_NO_SIMD)
  add_executable(matrix_free_no_simd src/MF_main.cpp)
  set_target_properties(matrix_free_no_simd PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${MFSOLVER_RUNTIME_OUTPUT_DIR}")
  deal_ii_setup_target(matrix_free_no_simd)
endif()

if(MFSOLVER_BUILD_MATRIX_FREE_SIMD)
  add_executable(matrix_free_simd src/MF_main.cpp)
  set_target_properties(matrix_free_simd PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${MFSOLVER_RUNTIME_OUTPUT_DIR}")
  target_compile_options(matrix_free_simd PUBLIC "-march=native")
  deal_ii_setup_target(matrix_free_simd)
endif()
