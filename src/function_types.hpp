#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <exception>
#include <stdexcept>

namespace MFSolver
{
    using namespace dealii;

    /**
     * \brief Represents a function that takes a `dim`-dimensional vector and returns a real number.
     * \tparam dim The dimensionality of the input vector.
     *
     * In order to overcome C++ limitations (we already tried to use `std::function`s miserably failing because of the non-polymorphism of the return type of the lambda wrapped Callable),
     * this class must be extended each time a different RealFunction is needed.
     *
     * Inheritance is not straightforward either. Here follows a commented snippet to readily copy & paste whenever needed.
     *
     * ```cpp
     * template <int dim>
     * class MyFunction : public MFSolver::RealFunction<dim> {
     * public:
     *     // Do not modify this. It is needed in order to have working template polymorphism.
     *     virtual double value(const Point<dim> &p, const unsigned int component = 0) const override {
     *         return value<double>(p, component);
     *     }
     *
     *     // This methos should contain the actual formula that this RealFunction should compute.
     *     // Yes, this is correct, it does not need to be marked as `override`.
     *     template <typename Number>
     *     Number value(const Point<dim, Number> &p, const unsigned int component = 0) const {
     *         return ...;
     *     }
     * };
     * ```
     */
    template <int dim>
    class RealFunction : public Function<dim>
    {
    public:
        /**
         * \brief Constructs a new instance of RealFunction.
         */
        RealFunction() : Function<dim>() {}

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override = 0;

        template <typename Number>
        Number value(const Point<dim, Number> &p, const unsigned int component = 0) const
        {
            throw std::logic_error("If you see this, you have not extended this class (RealFunction - value) correctly. See docs for more information.");
        }
    };

    /**
     * \brief Represents a function that takes a `dim`-dimensional vector and returns another `dim`-dimensional vector.
     * \tparam dim The dimensionality of the input and output vector.
     *
     * In order to overcome C++ limitations (we already tried to use `std::function`s miserably failing because of the non-polymorphism of the return type of the lambda wrapped Callable),
     * this class must be extended each time a different VectorFunction is needed.
     *
     * Inheritance is not straightforward either. Here follows a commented snippet to readily copy & paste whenever needed.
     *
     * ```cpp
     * template <int dim>
     * class MyFunction : public MFSolver::VectorFunction<dim> {
     * public:
     *     using Super = typename MFSolver::VectorFunction<dim>;
     *
     *     // Do not modify this. It is needed in order to have working template polymorphism.
     *     virtual typename Super::value_type<double> value(const Point<dim> &p) const override {
     *         return value<double>(p);
     *     }
     *
     *     // This methos should contain the actual formula that this VectorFunction should compute.
     *     // Yes, this is correct, it does not need to be marked as `override`.
     *     template <typename Number>
     *     typename Super::value_type<Number> value(const Point<dim, Number> &p) const {
     *         return ...;
     *     }
     * };
     * ```
     */
    template <int dim>
    class VectorFunction : public TensorFunction<1, dim, double>
    {
    public:
        // Using dealii's definition of value_type if we instantiated it with Number

        template <typename Number>
        using value_type = typename TensorFunction<1, dim, Number>::value_type;

        /**
         * \brief Constructs a new instance of VectorFunction.
         */
        VectorFunction() : TensorFunction<1, dim, double>() {}

        virtual value_type<double> value(const Point<dim> &p) const override = 0;

        template <typename Number>
        value_type<Number> value(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunction - value) correctly. See docs for more information.");
        }
    };

    /**
     * \brief Represents a function that takes a `dim` dimensional vector and returns anothr `dim`-dimensional vector. Moreover, the represented function must be differentiable and it's gradientmust also be provided.
     * \tparam dim The dimensionality of the input and output vector.
     *
     * In order to overcome C++ limitations (we already tried to use `std::function`s miserably failing because of the non-polymorphism of the return type of the lambda wrapped Callable),
     * this class must be extended each time a different VectorFunctionWithGradient is needed.
     *
     * Inheritance is not straightforward either. Here follows a commented snippet to readily copy & paste whenever needed.
     *
     * ```cpp
     * template <int dim>
     * class MyVectorWithGradientFunction : public MFSolver::VectorFunctionWithGradient<dim>
     * {
     * public:
     *     using Super = typename MFSolver::VectorFunctionWithGradient<dim>;
     *
     *     virtual typename Super::value_type<double> value(const Point<dim> &p) const override
     *     {
     *         return value<double>(p);
     *     }
     *
     *     virtual typename Super::gradient_type<double> gradient(const Point<dim> &p) const override
     *     {
     *         return gradient<double>(p);
     *     }
     *
     *     // TODO: is this OK or does this break SIMD context?
     *     virtual double divergence(const Point<dim> &p) const override
     *     {
     *         return divergence<double>(p);
     *     }
     *
     *     template <typename Number>
     *     typename Super::value_type<Number> value(const Point<dim, Number> &p) const
     *     {
     *         return 2. * p;
     *     }
     *
     *     template <typename Number>
     *     typename Super::gradient_type<Number> gradient(const Point<dim, Number> &p) const
     *     {
     *         return Tensor<2, dim, Number>({{1., 0., 0.}, {2., 0., 0.}, {3., 0., 0.}});
     *     }
     *
     *     template <typename Number>
     *     Number divergence(const Point<dim, Number> &p) const
     *     {
     *         return trace(gradient(p));
     *     }
     * };
     * ```
     *
     * \warning There are no checks that assert that the implementation of the gradient method effectively computes the gradient of the value method. Otherwise we would have just used that instead of having the user to implement it by itself.
     */
    template <int dim>
    class VectorFunctionWithGradient : public VectorFunction<dim>
    {
    public:
        /**
         * \brief Type alias representing the type of the value returned by this function.
         */
        template <typename Number>
        using value_type = typename VectorFunction<dim>::template value_type<Number>;

        /**
         * \brief Type alias representing the type of the gradient of this function.
         */
        template <typename Number>
        using gradient_type = typename TensorFunction<1, dim, Number>::gradient_type;

        /**
         * \brief Constructs a new instance of VectorFunctionWithGradient.
         */
        VectorFunctionWithGradient() : VectorFunction<dim>() {}

        virtual double divergence(const Point<dim> &p) const = 0;

        template <typename Number>
        Number divergence(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunctionWithGradient - divergence) correctly. See docs for more information.");
        }

        virtual gradient_type<double> gradient(const Point<dim> &p) const override = 0;

        template <typename Number>
        gradient_type<Number> gradient(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunctionWithGradient - gradient) correctly. See docs for more information.");
        }
    };

} // namespace MFSolver
