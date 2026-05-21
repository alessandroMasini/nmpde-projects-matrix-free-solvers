cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-float-conversion -march=native -O2 -std=c++20)

# Matrix - based files
add_executable(MB_advanced src/MB_advanced.cpp)
deal_ii_setup_target(MB_advanced)

add_executable(MB_lab_02 src/MB_lab_02.cpp)
deal_ii_setup_target(MB_lab_02)

add_executable(MB_lab_03 src/MB_lab_03.cpp)
deal_ii_setup_target(MB_lab_03)

add_executable(MB_parabolic src/MB_parabolic.cpp)
deal_ii_setup_target(MB_parabolic)

add_executable(MB_transient src/MB_transient.cpp)
deal_ii_setup_target(MB_transient)

# Matrix - free files
add_executable(MF_advanced src/MF_advanced.cpp)
deal_ii_setup_target(MF_advanced)

add_executable(MF_lab_02 src/MF_lab_02.cpp)
deal_ii_setup_target(MF_lab_02)

add_executable(MF_lab_03 src/MF_lab_03.cpp)
deal_ii_setup_target(MF_lab_03)

add_executable(MF_parabolic src/MF_parabolic.cpp)
deal_ii_setup_target(MF_parabolic)

add_executable(MF_transient src/MF_transient.cpp)
deal_ii_setup_target(MF_transient)