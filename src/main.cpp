#include "mfsolver.hpp"

int main()
{
    constexpr int dim = 3;

    MFSolver::RealFunction<dim> rf([](const dealii::Point<dim> &)
                                   { return 3.0; });

    MFSolver::VectorFunctionWithGradient<dim> vf([](const dealii::Point<dim> &)
                                                 { return dealii::Tensor<1, dim, double>({1.0, 2.0, 3.0}); },
                                                 [](const dealii::Point<dim> &)
                                                 {
                                                     return dealii::Tensor<2, dim, double>({{1., 0., 0.}, {2., 0., 0.}, {3., 0., 0.}});
                                                 });

    dealii::Point<dim> p(1.0, 2.0, 3.0);
    std::cout << vf.divergence(p) << std::endl;

    MFSolver::ADROperator<3, 1, double> o;

    o.clear();
        o.compute_diagonal();

    o.clear();
        o.evaluate_coefficients(rf, vf, rf);

    
}