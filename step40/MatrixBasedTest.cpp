// #include "../src/mfsolver.hpp"

// using namespace MFSolver;
// // Main function.
// // TODO: actually use real:
// // - boundaries, mesh, multigrid implementation
// // - the number of quadrature points depends on the degree
// //   of the solver. this is known after the solver is instantiated.
// //   how can we know it before instantiating it?


// template <int dim>
// class TrigonometricF : public RealFunction<dim>
// {   

// public:
//     TrigonometricF() : RealFunction<dim>() {}

//     double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
//         {
//             return value<double>(p, component);
//         }

//     template <typename Number>
//     Number value(const dealii::Point<dim, Number> & p, const unsigned int /*component*/ = 0) const {
//         return 2*( p[1]*(1-p[1]) + p[0]*(1-p[0]) ) +  (1-2*p[1])*p[0]*(1-p[0]) + (1-2*p[0])*p[1]*(1-p[1]) 
//             + p[0]*(1-p[0])*p[1]*(1-p[1]);
//     }
// };

// int
// main (int argc, char* argv [ ]) {
//     // TODO: decide what to do with the number of threads
//     Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv, 1);

//     ADR::ProblemData<2, 1> data;
//     // data.fe_degree = 1;
//     // data.refinement_level = 5;

//     data.mu = std::make_shared<ADR::ConstantRealFunction<2>>(1.0);
//     data.beta = std::make_shared<ADR::ConstantVectorFunctionWithGradient<2>>(1.0);
//     data.gamma = std::make_shared<ADR::ConstantRealFunction<2>>(1.0);

//     data.forcing_term = std::make_shared<TrigonometricF<2>>();
    

//     data.num_quadrature_points = 1 + 1;
//     data.solver_max_iterations = 10000;
//     data.solver_tolerance_factor = 1.0e-16;

//     // Currently not used
//     data.dirichlet_boundary_value = std::make_shared<ADR::ConstantRealFunction<2>>(0.0);
//     data.neumann_boundary_value = std::make_shared<ADR::ConstantRealFunction<2>>(0.0);
//     data.mesh_filename = "input.msh";
//     data.num_levels = 1;    // TODO: What is this?
//     data.lv0_smoothing_range = 1.e-3;
//     data.lvgt0_smoothing_range = 15;
//     data.lvgt0_smoothing_degree = 5;
//     data.lvgt0_smoothing_eigenvalue_max_iterations = 10;
//     data.refinement_coefficient_per_level = 1;

//     MatrixBasedADRSolver<2, 1> solver(data);
//     solver.run();
//     return 0;
// }

#include "../src/mfsolver.hpp"
#include "../src/ProblemData.hpp"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    // Available Test Cases (3D)
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::standard_test_case();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::advanced_test_case();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_neumann_fix();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_parabolic();

    ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::lab_02_poisson();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::lab_03_dr_eq();

    //ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::standard_test_case();
    MFSolver::MatrixBasedADRSolver<2, 2> solver(data);

    solver.run();

    return 0;
}