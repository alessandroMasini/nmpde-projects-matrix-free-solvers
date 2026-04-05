cmake_minimum_required(VERSION 3.12.0)
project(step37 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(step37 step37.cpp)
deal_ii_setup_target(step37)