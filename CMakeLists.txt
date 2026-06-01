cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-float-conversion -O3 -std=c++20)

# Matrix - based files
add_executable(matrix_based src/MB_main.cpp)
deal_ii_setup_target(matrix_based)

# Matrix - free files
add_executable(matrix_free_simd src/MF_main.cpp)
target_compile_options(matrix_free_simd PUBLIC "-march=native")
deal_ii_setup_target(matrix_free_simd)

add_executable(matrix_free_no_simd src/MF_main.cpp)
deal_ii_setup_target(matrix_free_no_simd)