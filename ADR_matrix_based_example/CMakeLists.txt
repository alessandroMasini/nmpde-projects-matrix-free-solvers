cmake_minimum_required(VERSION 3.12.0)
project(MatrixBasedSolver)

include(../common/cmake-common.cmake)

add_executable(MatrixBasedTest MatrixBasedTest.cpp)
deal_ii_setup_target(MatrixBasedTest)