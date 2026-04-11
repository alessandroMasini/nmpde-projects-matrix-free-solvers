#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main()
{
    ADR::ProblemData<3, 2, ADR::ConstantRealFunction, ADR::ConstantVectorFunctionWithGradient, ADR::ConstantRealFunction> data = ADR::ProblemData<3, 2, ADR::ConstantRealFunction, ADR::ConstantVectorFunctionWithGradient, ADR::ConstantRealFunction>::standard_test_case();

    MFSolver::MatrixFreeADRSolver<3, 2, ADR::ConstantRealFunction, ADR::ConstantVectorFunctionWithGradient, ADR::ConstantRealFunction> solver(data);

    solver.run();

    return 0;
}