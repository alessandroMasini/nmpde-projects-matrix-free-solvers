#include "../src/mfsolver.hpp"

using namespace MFSolver;
// Main function.
// TODO: actually use real:
// - boundaries, mesh, multigrid implementation
// - the number of quadrature points depends on the degree
//   of the solver. this is known after the solver is instantiated.
//   how can we know it before instantiating it?


template <int dim>
class TrigonometricF : public RealFunction<dim>
{   

public:
    TrigonometricF() : RealFunction<dim>() {}

    double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return value<double>(p, component);
        }

    template <typename Number>
    Number value(const dealii::Point<dim, Number> & p, const unsigned int /*component*/ = 0) const {
        return 2*1.0e-2*(M_PI*M_PI) * std::sin (M_PI * p[0]) * std::sin (M_PI * p[1])
        + M_PI * std::cos (M_PI * p[0]) * std::sin (M_PI * p[1])
        + M_PI * std::cos (M_PI * p[1]) * std::sin (M_PI * p[0])
        + std::sin (M_PI * p[0]) * std::sin (M_PI * p[1]);
    }
};

int
main (int argc, char* argv [ ]) {

    Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv);

    ADR::ProblemData<2, 2> data;
    // data.fe_degree = 1;
    // data.refinement_level = 5;

    data.mu = std::make_shared<ADR::ConstantRealFunction<2>>(1.0e-2);
    data.beta = std::make_shared<ADR::ConstantVectorFunctionWithGradient<2>>(1.0);
    data.gamma = std::make_shared<ADR::ConstantRealFunction<2>>(1.0);

    data.forcing_term = std::make_shared<TrigonometricF<2>> ();

    data.num_quadrature_points = 2 + 1;
    data.solver_max_iterations = 10000;
    data.solver_tolerance_factor = 1.0e-16;

    // Currently not used
    data.dirichlet_boundary_value = std::make_shared<ADR::ConstantRealFunction<2>>(0.0);
    data.dirichlet_boundary_value = std::make_shared<ADR::ConstantRealFunction<2>>(0.0);
    data.mesh_filename = "input.msh";
    data.num_levels = 5;
    data.lv0_smoothing_range = 1.e-3;
    data.lvgt0_smoothing_range = 15;
    data.lvgt0_smoothing_degree = 5;
    data.lvgt0_smoothing_eigenvalue_max_iterations = 10;
    data.refinement_coefficient_per_level = 4;

    MatrixBasedADRSolver<2, 2> solver(data);
    solver.run();
    return 0;
    }