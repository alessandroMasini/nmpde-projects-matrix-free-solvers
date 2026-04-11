cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -march=native -std=c++20 -Wno-deprecated-declarations)

add_executable(mfsolver src/main.cpp)
deal_ii_setup_target(mfsolver)