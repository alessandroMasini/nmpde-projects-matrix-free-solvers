#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    
    // Using advanced_test_case to test Dirichlet + Neumann logic
    ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::advanced_test_case();

    MFSolver::MatrixFreeADRSolver<3, 2> solver(data);

    solver.run();

    return 0;
}