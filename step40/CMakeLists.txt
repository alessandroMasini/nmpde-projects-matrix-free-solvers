cmake_minimum_required(VERSION 3.12.0)
project(MatrixBasedSolver)

include(../common/cmake-common.cmake)

# At some point we should also use -Wall -Wextra -Wpedantic
# add_compile_options(-std=c++20 -O0 -g3 -fno-inline -fno-inline-functions -fsanitize=address)
# add_link_options(-fsanitize=address)
add_compile_options(-std=c++20 -Og -g -fno-inline -fno-inline-functions -fno-omit-frame-pointer)

add_executable(MB_advanced MB_advanced.cpp)
deal_ii_setup_target(MB_advanced)

add_executable(MB_lab_02 MB_lab_02.cpp)
deal_ii_setup_target(MB_lab_02)

add_executable(MB_lab_03 MB_lab_03.cpp)
deal_ii_setup_target(MB_lab_03)

add_executable(MB_parabolic MB_parabolic.cpp)
deal_ii_setup_target(MB_parabolic)

add_executable(MB_transient MB_transient.cpp)
deal_ii_setup_target(MB_transient)