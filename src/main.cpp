#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{
    // dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::lab_02_poisson();

    MFSolver::MatrixFreeADRSolver<2, 2> solver(data);

    solver.run();

    return 0;
}