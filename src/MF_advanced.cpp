#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    // Available Test Cases (3D)
    // ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::standard_test_case();
    ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::advanced_test_case();

    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_neumann_fix();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_parabolic();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_comprehensive_transient();

    // ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::lab_02_poisson();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::lab_03_dr_eq();

    // ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::see_miro_for_problem_definition_andrea_knows();

    // Refinement level from input
    if (argc >= 5)
    {
        data.refinement_level += std::stoi(argv[3]);
        data.solver_max_iterations = std::stoi(argv[4]);
        data.solver_tolerance_factor = std::stod(argv[5]);
    }
    else
    {
        throw std::invalid_argument(
            "Usage: mpirun -n <n_cores> ./MB_advanced <n_threads> <simd> <n_additional_refinements> <max_iters> <max_err>");
    }

    MFSolver::MatrixFreeADRSolver<3, 2> solver(data);

    solver.run();
    solver.output_to_file();

    return 0;
}