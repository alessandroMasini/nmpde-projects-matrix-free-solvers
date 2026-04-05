#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <exception>
#include <stdexcept>

namespace MFSolver {

    using namespace dealii;

    template <int dim>
    class RealFunction : public Function<dim>
    {
    public:
        RealFunction() : Function<dim>() {}

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override = 0;

        template <typename Number>
        Number value(const Point<dim, Number> &p, const unsigned int component = 0) const
        {
            throw std::logic_error("If you see this, you have not extended this class (RealFunction - value) correctly. See docs for more information.");
        }
    };

    template <int dim>
    class VectorFunction : public TensorFunction<1, dim, double>
    {
    public:
        // Using dealii's definition of value_type if we instantiated it with Number
        template <typename Number>
        using value_type = typename TensorFunction<1, dim, Number>::value_type;

        VectorFunction() : TensorFunction<1, dim, double>() {}

        virtual value_type<double> value(const Point<dim> &p) const override = 0;

        template <typename Number>
        value_type<Number> value(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunction - value) correctly. See docs for more information.");
        }
    };

    template <int dim>
    class VectorFunctionWithGradient : public VectorFunction<dim>
    {
    public:
        template <typename Number>
        using value_type = typename VectorFunction<dim>::template value_type<Number>;

        template <typename Number>
        using gradient_type = typename TensorFunction<1, dim, Number>::gradient_type;

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
