cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -march=native -std=c++20 -g -O0 -Wno-deprecated-declarations -Wno-float-conversion)

add_executable(mfsolver src/main.cpp)
deal_ii_setup_target(mfsolver)

target_precompile_headers(mfsolver
PRIVATE
    <exception>
    <fstream>
    <functional>
    <iostream>
    <memory>
    <stdexcept>
    <string>
    <unordered_map>
    
    <deal.II/base/conditional_ostream.h>
    <deal.II/base/function.h>
    <deal.II/base/mg_level_object.h>
    <deal.II/base/point.h>
    <deal.II/base/quadrature_lib.h>
    <deal.II/base/tensor_function.h>
    <deal.II/base/tensor.h>
    <deal.II/base/timer.h>
    <deal.II/base/types.h>
    
    <deal.II/distributed/tria.h>
    <deal.II/distributed/fully_distributed_tria.h>

    <deal.II/dofs/dof_tools.h>
    
    <deal.II/fe/mapping_q1.h>
    <deal.II/fe/fe_q.h>
    <deal.II/fe/fe_simplex_p.h>
    
    <deal.II/grid/grid_generator.h>
    <deal.II/grid/grid_in.h>
    <deal.II/grid/grid_tools.h>
    <deal.II/grid/tria.h>    
    
    <deal.II/lac/affine_constraints.h>
    <deal.II/lac/la_parallel_vector.h>
    <deal.II/lac/precondition.h>
    <deal.II/lac/solver_cg.h>
    <deal.II/lac/solver_control.h>
    <deal.II/lac/solver_gmres.h>
    <deal.II/lac/trilinos_precondition.h>
    <deal.II/lac/trilinos_sparse_matrix.h>
    <deal.II/lac/trilinos_vector.h>
    <deal.II/lac/vector.h>
    
    <deal.II/matrix_free/fe_evaluation.h>
    <deal.II/matrix_free/matrix_free.h>
    <deal.II/matrix_free/operators.h>
    <deal.II/matrix_free/tools.h>
    
    <deal.II/multigrid/mg_coarse.h>
    <deal.II/multigrid/mg_constrained_dofs.h>
    <deal.II/multigrid/mg_matrix.h>
    <deal.II/multigrid/mg_smoother.h>
    <deal.II/multigrid/mg_transfer_matrix_free.h>
    <deal.II/multigrid/mg_tools.h>
    <deal.II/multigrid/multigrid.h>
    
    <deal.II/numerics/data_out.h>
    <deal.II/numerics/matrix_tools.h>
    <deal.II/numerics/vector_tools.h>
)
