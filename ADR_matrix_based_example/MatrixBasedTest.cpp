#include "../src/mfsolver.hpp"

using namespace MFSolver;
// Main function.
// TODO: actually use real:
// - boundaries, mesh, multigrid implementation
// - the number of quadrature points depends on the degree
//   of the solver. this is known after the solver is instantiated.
//   how can we know it before instantiating it?
int
main (int argc, char* argv [ ]) {

    Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv);

    constexpr double half_pi = M_PI * 0.5;

    ADRProblem<1> problem = {
        // Functions
        RealFunction<1>([](const Point<1>& /*p*/) { return 1.0; }),
        VectorFunction<1>([](const Point<1>& p, const unsigned int i) { return p[i] - 1.0; }),
        RealFunction<1>([](const Point<1>& /*p*/) { return 1.0; }),
        // TODO: check if just removing time dependencies will make everything fine
        RealFunction<1>([](const Point<1>& p) {
        return half_pi * std::sin (half_pi * p[0]);
        }),

        // Not used boundaries
        DirichletBoundaries<1>{},  
        NeumannBoundaries<1>{},

        // Mesh
        std::string{},

        // Not used multigrid levels
        1,

        // Quadrature points
        2+1,
        
        // Not used multigrid settings
        1,
        1.0,
        1.0,
        1.0,
        1.0,

        // Solver settings
        10000,
        1.0e-16
    };

    MatrixBasedADRSolver<1, 2> solver(problem);
    solver.run();

    ///////////////////////////////////////////////////////
    // OLD IMPLEMENTATION 
    ///////////////////////////////////////////////////////

    
    // constexpr unsigned int dim = ADRProblem::dim;

    // Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv);

    // constexpr double half_pi = M_PI * 0.5;

    // const auto mu = [](const Point<dim>& /*p*/) { return 1.0; };
    // const auto b = [](const Point<dim>& p) { return p[0] - 1.0; };
    // const auto k = [](const Point<dim>& /*p*/) { return 1.0; };
    // const auto f = [](const Point<dim>& p, const double& t) {
    //     return half_pi * std::sin (half_pi * p[0]) * std::cos (half_pi * t) +
    //         (half_pi * half_pi + 2.0) * std::sin (half_pi * p[0]) * std::sin (half_pi * t) +
    //         half_pi * (p[0] - 1.0) * std::cos (half_pi * p[0]) * std::sin (half_pi * t);
    // };

    // ADRProblem problem (/*N_el = */ 40,
    //      /* degree = */ 2,
    //     /* T = */ 1.0,
    //     /* theta = */ 0.5,
    //     /* delta_t = */ 0.1,
    //     mu,
    //     b,
    //     k,
    //     f);

    // problem.run ();



    return 0;
    }