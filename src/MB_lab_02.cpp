#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    if (argc < 4)
    {
       throw std::invalid_argument("Usage: mpirun -n <n_cores> <program> <n_threads> <n_additional_refinements> <max_iters> <max_err>");
    }

    ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::lab_02_poisson();
    data.refinement_level += std::stoi(argv[2]);
    data.solver_max_iterations = std::stoi(argv[3]);
    data.solver_tolerance_factor = std::stod(argv[4]);

    MFSolver::MatrixBasedADRSolver<2, 2> solver(data);

    solver.run();
    solver.output_to_file();

    return 0;
}