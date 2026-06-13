/**
 * @file function_types.hpp
 * @brief Polymorphic scalar, vector, and gradient-aware function interfaces.
 *
 * @details The matrix-free kernels evaluate coefficients in both scalar and
 * vectorized deal.II point types. These interfaces make that contract explicit
 * for coefficients, exact solutions, and manufactured forcing terms.
 */

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
     * @brief Represents a function that takes a `dim`-dimensional vector and returns a real number.
     * @tparam dim The dimensionality of the input vector.
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
     *     // Implement the formula once for scalar and vectorized number types.
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
         * @brief Constructs a new instance of RealFunction.
         */
        RealFunction() : Function<dim>() {}

        /**
         * @brief Evaluate the scalar function at a standard deal.II point.
         * @param p Evaluation point.
         * @param component Component requested by the deal.II Function interface.
         * @return Scalar function value.
         */
        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override = 0;

        /**
         * @brief Evaluate the scalar function at a float-vectorized point.
         * @param p Vectorized evaluation point.
         * @param component Component requested by the deal.II Function interface.
         * @return Vectorized scalar function value.
         */
        virtual VectorizedArray<float> value(const Point<dim, VectorizedArray<float>> &p, const unsigned int component = 0) const = 0;

        /**
         * @brief Evaluate the scalar function at a double-vectorized point.
         * @param p Vectorized evaluation point.
         * @param component Component requested by the deal.II Function interface.
         * @return Vectorized scalar function value.
         */
        virtual VectorizedArray<double> value(const Point<dim, VectorizedArray<double>> &p, const unsigned int component = 0) const = 0;
    };

    /**
     * @brief Represents a function that takes a `dim`-dimensional vector and returns another `dim`-dimensional vector.
     * @tparam dim The dimensionality of the input and output vector.
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
     *     // Implement the formula once for scalar and vectorized number types.
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
        /// Alias deal.II's vector-valued return type for the selected number type.

        template <typename Number>
        using value_type = typename TensorFunction<1, dim, Number>::value_type;

        /**
         * @brief Constructs a new instance of VectorFunction.
         */
        VectorFunction() : TensorFunction<1, dim, double>() {}

        /**
         * @brief Evaluate the vector field at a standard deal.II point.
         * @param p Evaluation point.
         * @return Vector-valued function value.
         */
        virtual value_type<double> value(const Point<dim> &p) const override = 0;

        /**
         * @brief Evaluate the vector field at a float-vectorized point.
         * @param p Vectorized evaluation point.
         * @return Vectorized vector-field value.
         */
        virtual value_type<VectorizedArray<float>> value(const Point<dim, VectorizedArray<float>> &p) const = 0;

        /**
         * @brief Evaluate the vector field at a double-vectorized point.
         * @param p Vectorized evaluation point.
         * @return Vectorized vector-field value.
         */
        virtual value_type<VectorizedArray<double>> value(const Point<dim, VectorizedArray<double>> &p) const = 0;
    };

    /**
     * @brief Represents a function that takes a `dim` dimensional vector and returns anothr `dim`-dimensional vector. Moreover, the represented function must be differentiable and it's gradientmust also be provided.
     * @tparam dim The dimensionality of the input and output vector.
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
     *     // Forward the scalar divergence override to the templated implementation.
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
     * @warning There are no checks that assert that the implementation of the gradient method effectively computes the gradient of the value method. Otherwise we would have just used that instead of having the user to implement it by itself.
     */
    template <int dim>
    class VectorFunctionWithGradient : public VectorFunction<dim>
    {
    public:
        /**
         * @brief Type alias representing the type of the value returned by this function.
         */
        template <typename Number>
        using value_type = typename VectorFunction<dim>::template value_type<Number>;

        /**
         * @brief Type alias representing the type of the gradient of this function.
         */
        template <typename Number>
        using gradient_type = typename TensorFunction<1, dim, Number>::gradient_type;

        /**
         * @brief Constructs a new instance of VectorFunctionWithGradient.
         */
        VectorFunctionWithGradient() : VectorFunction<dim>() {}

        /**
         * @brief Evaluate the divergence at a standard deal.II point.
         * @param p Evaluation point.
         * @return Divergence of the vector field.
         */
        virtual double divergence(const Point<dim> &p) const = 0;

        /**
         * @brief Evaluate the divergence at a float-vectorized point.
         * @param p Vectorized evaluation point.
         * @return Vectorized divergence value.
         */
        virtual VectorizedArray<float> divergence(const Point<dim, VectorizedArray<float>> &p) const = 0;

        /**
         * @brief Evaluate the divergence at a double-vectorized point.
         * @param p Vectorized evaluation point.
         * @return Vectorized divergence value.
         */
        virtual VectorizedArray<double> divergence(const Point<dim, VectorizedArray<double>> &p) const = 0;

        /**
         * @brief Evaluate the Jacobian of the vector field at a standard point.
         * @param p Evaluation point.
         * @return Gradient tensor of the vector field.
         */
        virtual gradient_type<double> gradient(const Point<dim> &p) const override = 0;

        /**
         * @brief Evaluate the Jacobian at a float-vectorized point.
         * @param p Vectorized evaluation point.
         * @return Vectorized gradient tensor.
         */
        virtual gradient_type<VectorizedArray<float>> gradient(const Point<dim, VectorizedArray<float>> &p) const = 0;

        /**
         * @brief Evaluate the Jacobian at a double-vectorized point.
         * @param p Vectorized evaluation point.
         * @return Vectorized gradient tensor.
         */
        virtual gradient_type<VectorizedArray<double>> gradient(const Point<dim, VectorizedArray<double>> &p) const = 0;
    };

} // namespace MFSolver
