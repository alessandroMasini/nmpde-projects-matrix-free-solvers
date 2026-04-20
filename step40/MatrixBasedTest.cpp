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
    // Old forcing term
    /*
    template <typename Number>
    Number value(const dealii::Point<dim, Number> & p, const unsigned int /*component*//* = 0) const {
        return 2*( p[1]*(1-p[1]) + p[0]*(1-p[0]) ) +  (1-2*p[1])*p[0]*(1-p[0]) + (1-2*p[0])*p[1]*(1-p[1]) 
            + p[0]*(1-p[0])*p[1]*(1-p[1]);
    }
    */
   template <typename Number>
    Number value(const dealii::Point<dim, Number> & p, const unsigned int /*component*/ = 0) const {
    if (p[0] < 0.5)
        return 100.0;
    else
        return 1.0;
    }
};

int
main (int argc, char* argv [ ]) {
    // TODO: decide what to do with the number of threads
    if (argc < 2){
        cout << "Please insert the number of desired threads" << std::endl;
        return 1;
    }

    Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv, std::stoi(argv[1]));

    ADR::ProblemData<3, 1> data;
    // data.fe_degree = 1;
    // data.refinement_level = 5;

    data.mu = std::make_shared<ADR::ConstantRealFunction<3>>(1.0);
    data.beta = std::make_shared<ADR::ConstantVectorFunctionWithGradient<3>>(0.0);
    data.gamma = std::make_shared<ADR::ConstantRealFunction<3>>(1.0);

    data.forcing_term = std::make_shared<TrigonometricF<3>>();
    

    data.num_quadrature_points = 1 + 1;
    data.solver_max_iterations = 10000;
    data.solver_tolerance_factor = 1.0e-16;

    // Currently not used
    data.dirichlet_boundary_value = std::make_shared<ADR::ConstantRealFunction<3>>(0.0);
    data.dirichlet_boundary_value = std::make_shared<ADR::ConstantRealFunction<3>>(0.0);
    data.mesh_filename = "mesh-cube-5.msh";
    data.num_levels = 5;
    data.lv0_smoothing_range = 1.e-3;
    data.lvgt0_smoothing_range = 15;
    data.lvgt0_smoothing_degree = 5;
    data.lvgt0_smoothing_eigenvalue_max_iterations = 10;
    data.refinement_coefficient_per_level = 4;

    MatrixBasedADRSolver<3, 1> solver(data);
    // solver.create_test_grid();
    solver.run();
    return 0;
}