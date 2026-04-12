cmake_minimum_required(VERSION 3.12.0)
project(MatrixBasedSolver)

include(./common/cmake-common.cmake)

# At some point we should also use -Wall -Wextra -Wpedantic
add_compile_options(-std=c++20 -O0 -g3 -fno-inline -fno-inline-functions -Wall -Wextra)
add_link_options(-fsanitize=address)
# add_compile_options(-std=c++20 -Og -g -fno-inline -fno-inline-functions  -fno-omit-frame-pointer)

add_executable(MatrixBasedTest src/MatrixBasedTest.cpp)
deal_ii_setup_target(MatrixBasedTest)