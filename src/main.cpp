#include "mfsolver.hpp"

namespace MyFunctions
{

    using namespace dealii;

    template <int dim>
    class MyFunction : public MFSolver::RealFunction<dim>
    {
    public:
        // Do not modify this. It is needed in order to have working template polymorphism.
        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            return this->value<double>(p, component);
        }

        // This methos should contain the actual formula that this RealFunction should compute.
        template <typename Number>
        Number value(const Point<dim, Number> &p, const unsigned int component = 0) const
        {
            return 3.;
        }
    };

    template <int dim>
    class MyVectorFunction : public MFSolver::VectorFunction<dim>
    {
    public:
        using Super = typename MFSolver::VectorFunction<dim>;

        virtual typename Super::value_type<double> value(const Point<dim> &p) const override
        {
            return value<double>(p);
        }

        template <typename Number>
        typename Super::value_type<Number> value(const Point<dim, Number> &p) const
        {
            return 2. * p;
        }
    };

    template <int dim>
    class MyVectorWithGradientFunction : public MFSolver::VectorFunctionWithGradient<dim>
    {
    public:
        using Super = typename MFSolver::VectorFunctionWithGradient<dim>;

        virtual typename Super::value_type<double> value(const Point<dim> &p) const override
        {
            return value<double>(p);
        }

        virtual typename Super::gradient_type<double> gradient(const Point<dim> &p) const override
        {
            return gradient<double>(p);
        }

        // TODO: is this OK or does this break SIMD context?
        virtual double divergence(const Point<dim> &p) const override
        {
            return divergence<double>(p);
        }

        template <typename Number>
        typename Super::value_type<Number> value(const Point<dim, Number> &p) const
        {
            return 2. * p;
        }

        template <typename Number>
        typename Super::gradient_type<Number> gradient(const Point<dim, Number> &p) const
        {
            return Tensor<2, dim, Number>({{1., 0., 0.}, {2., 0., 0.}, {3., 0., 0.}});
        }

        template <typename Number>
        Number divergence(const Point<dim, Number> &p) const
        {
            return trace(gradient(p));
        }
    };
};

int main()
{
    std::cout << "Begin" << std::endl;
    constexpr int dim = 3;

    MyFunctions::MyFunction<dim> rf;

    MyFunctions::MyVectorWithGradientFunction<dim> vf;

    // MFSolver::VectorFunctionWithGradient<dim> vf([](const dealii::Point<dim> &)
    //                                              { return dealii::Tensor<1, dim, double>({1.0, 2.0, 3.0}); },
    //                                              [](const dealii::Point<dim> &)
    //                                              {
    //                                                  return dealii::Tensor<2, dim, double>({{1., 0., 0.}, {2., 0., 0.}, {3., 0., 0.}});
    //                                              });

    dealii::Point<dim> p(1.0, 2.0, 3.0);

    std::cout << "End decl." << std::endl;

    std::cout << vf.divergence(p) << std::endl;

    std::cout << "End div" << std::endl;

    MFSolver::ADROperator<3, 1, double> o;

    std::cout << "After operator" << std::endl;

    o.clear();

    std::cout << "After clear" << std::endl;

    o.evaluate_coefficients(rf, vf, rf);

    std::cout << "After coeff" << std::endl;

    o.compute_diagonal();

    std::cout << "After diagonal" << std::endl;
}