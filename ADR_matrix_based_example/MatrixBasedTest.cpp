#include "../src/mfsolver.hpp"

// Main function.
int
main (int argc, char* argv [ ]) {
    ADRProblem<1> problem = {
        RealFunction<1>([](const Point<dim>& /*p*/) { return 1.0; }),
        VectorFunction<1>([](const Point<dim>& p) { return p[0] - 1.0; }),
        RealFunction<1>([](const Point<dim>& /*p*/) { return 1.0; }),
        RealFunction<1>([](const Point<dim>& p, const double& t) {
        return half_pi * std::sin (half_pi * p[0]) * std::cos (half_pi * t) +
            (half_pi * half_pi + 2.0) * std::sin (half_pi * p[0]) * std::sin (half_pi * t) +
            half_pi * (p[0] - 1.0) * std::cos (half_pi * p[0]) * std::sin (half_pi * t);
        }),
        


    }

    constexpr unsigned int dim = ADRProblem::dim;

    Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv);

    constexpr double half_pi = M_PI * 0.5;

    const auto mu = [](const Point<dim>& /*p*/) { return 1.0; };
    const auto b = [](const Point<dim>& p) { return p[0] - 1.0; };
    const auto k = [](const Point<dim>& /*p*/) { return 1.0; };
    const auto f = [](const Point<dim>& p, const double& t) {
        return half_pi * std::sin (half_pi * p[0]) * std::cos (half_pi * t) +
            (half_pi * half_pi + 2.0) * std::sin (half_pi * p[0]) * std::sin (half_pi * t) +
            half_pi * (p[0] - 1.0) * std::cos (half_pi * p[0]) * std::sin (half_pi * t);
    };

    ADRProblem problem (/*N_el = */ 40,
         /* degree = */ 2,
        /* T = */ 1.0,
        /* theta = */ 0.5,
        /* delta_t = */ 0.1,
        mu,
        b,
        k,
        f);

    problem.run ();

    return 0;
    }