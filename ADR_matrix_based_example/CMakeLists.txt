cmake_minimum_required(VERSION 3.12.0)
project(MatrixBasedSolver)

include(../common/cmake-common.cmake)

add_compile_options(-Og -g -fno-inline -fno-inline-functions -fsanitize=address)

add_executable(MatrixBasedTest MatrixBasedTest.cpp)
deal_ii_setup_target(MatrixBasedTest)