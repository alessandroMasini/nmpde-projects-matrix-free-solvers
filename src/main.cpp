#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{
    // dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ADR::ProblemData<3, 3> data = ADR::ProblemData<3, 3>::lab_03_dr_eq();

    MFSolver::MatrixFreeADRSolver<3, 3> solver(data);

    solver.run();

    return 0;
}