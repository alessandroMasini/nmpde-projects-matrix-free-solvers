#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Available Test Cases (3D)
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::standard_test_case();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::advanced_test_case();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_neumann_fix();
    //  ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::test_case_parabolic();

    // ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::lab_02_poisson();
    // ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::lab_03_dr_eq();

    ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::see_miro_for_problem_definition_andrea_knows();
    MFSolver::MatrixFreeADRSolver<2, 2> solver(data);

    solver.run();

    return 0;
}