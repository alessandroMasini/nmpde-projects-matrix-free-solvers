#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main()
{
    ADR::ProblemData<3, 2> data = ADR::ProblemData<3, 2>::standard_test_case();

    MFSolver::MatrixFreeADRSolver<3, 2> solver(data);

    // solver.setup_system();
    // solver.assemble();

    return 0;
}