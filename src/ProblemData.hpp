/**
 * @file ProblemData.hpp
 * @brief Benchmark problem definitions and coefficient functions for ADR runs.
 *
 * @details Provides reusable coefficient, boundary, exact-solution, and forcing
 * function objects together with the ProblemData factory methods used by the
 * matrix-based and matrix-free driver executables.
 */

#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>

#include "function_types.hpp"
#include "boundaries.hpp"

#include <memory>

namespace ADR
{

    /**
     * @brief Constant scalar coefficient or source term.
     * @tparam dim Spatial dimension of the evaluation point.
     */
    template <int dim>
    class ConstantRealFunction : public MFSolver::RealFunction<dim>
    {
    private:
        double val;

    public:
        ConstantRealFunction(double v) : MFSolver::RealFunction<dim>(), val(v) {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return do_compute_value<double>(p, component);
        }

        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int component = 0) const override
        {
            return do_compute_value<dealii::VectorizedArray<float>>(p, component);
        }

        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int component = 0) const override
        {
            return do_compute_value<dealii::VectorizedArray<double>>(p, component);
        }

        template <typename Number>
        Number do_compute_value(const dealii::Point<dim, Number> & /*p*/, const unsigned int /*component*/ = 0) const
        {
            return Number(val);
        }
    };

    /**
     * @brief Constant vector coefficient with zero gradient and divergence.
     * @tparam dim Spatial dimension of the vector field.
     */
    template <int dim>
    class ConstantVectorFunctionWithGradient : public MFSolver::VectorFunctionWithGradient<dim>
    {
    private:
        double val;

    public:
        using Super = typename MFSolver::VectorFunctionWithGradient<dim>;

        ConstantVectorFunctionWithGradient(double v) : val(v) {}

        virtual typename Super::template value_type<double> value(const dealii::Point<dim> &p) const override
        {
            return do_compute_value<double>(p);
        }

        virtual typename Super::template value_type<dealii::VectorizedArray<float>> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p) const override
        {
            return do_compute_value<dealii::VectorizedArray<float>>(p);
        }

        virtual typename Super::template value_type<dealii::VectorizedArray<double>> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p) const override
        {
            return do_compute_value<dealii::VectorizedArray<double>>(p);
        }

        template <typename Number>
        typename Super::template value_type<Number> do_compute_value(const dealii::Point<dim, Number> & /*p*/) const
        {
            dealii::Tensor<1, dim, Number> b;
            for (unsigned int d = 0; d < dim; ++d)
                b[d] = Number(val);
            return b;
        }

        virtual double divergence(const dealii::Point<dim> &p) const override
        {
            return do_compute_divergence<double>(p);
        }

        virtual dealii::VectorizedArray<float> divergence(const dealii::Point<dim, dealii::VectorizedArray<float>> &p) const override
        {
            return do_compute_divergence<dealii::VectorizedArray<float>>(p);
        }

        virtual dealii::VectorizedArray<double> divergence(const dealii::Point<dim, dealii::VectorizedArray<double>> &p) const override
        {
            return do_compute_divergence<dealii::VectorizedArray<double>>(p);
        }

        template <typename Number>
        Number do_compute_divergence(const dealii::Point<dim, Number> & /*p*/) const
        {
            return Number(0.0);
        }

        virtual typename Super::template gradient_type<double> gradient(const dealii::Point<dim> &p) const override
        {
            return do_compute_gradient<double>(p);
        }

        virtual typename Super::template gradient_type<dealii::VectorizedArray<float>> gradient(const dealii::Point<dim, dealii::VectorizedArray<float>> &p) const override
        {
            return do_compute_gradient<dealii::VectorizedArray<float>>(p);
        }

        virtual typename Super::template gradient_type<dealii::VectorizedArray<double>> gradient(const dealii::Point<dim, dealii::VectorizedArray<double>> &p) const override
        {
            return do_compute_gradient<dealii::VectorizedArray<double>>(p);
        }

        template <typename Number>
        typename Super::template gradient_type<Number> do_compute_gradient(const dealii::Point<dim, Number> & /*p*/) const
        {
            return typename Super::template gradient_type<Number>();
        }
    };

    /**
     * @brief Constant Dirichlet boundary value.
     * @tparam dim Spatial dimension of the boundary.
     */
    template <int dim>
    class ConstantDirichletBoundary : public MFSolver::DirichletBoundary<dim>
    {
    private:
        double storage;

    public:
        ConstantDirichletBoundary(double _storage) : MFSolver::DirichletBoundary<dim>(), storage(_storage) {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return storage;
        }
    };

    /**
     * @brief Dirichlet function returning the sum of point coordinates.
     * @tparam dim Spatial dimension of the boundary.
     */
    template <int dim>
    class SumReductionFunction : public MFSolver::DirichletBoundary<dim>
    {
    public:
        SumReductionFunction() : MFSolver::DirichletBoundary<dim>() {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int = 0) const override
        {
            double sum = 0.0;
            for (unsigned int d = 0; d < dim; ++d)
                sum += p[d];
            return sum;
        }
    };

    /**
     * @brief Neumann function returning one selected coordinate.
     * @tparam dim Spatial dimension of the boundary.
     */
    template <int dim>
    class NthCoordFunction : public MFSolver::NeumannBoundary<dim>
    {
    private:
        size_t n;

    public:
        NthCoordFunction(size_t _n) : MFSolver::NeumannBoundary<dim>(), n(_n) {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int = 0) const override
        {
            return do_compute_value<double>(p);
        }

        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int = 0) const override
        {
            return do_compute_value<dealii::VectorizedArray<float>>(p);
        }

        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int = 0) const override
        {
            return do_compute_value<dealii::VectorizedArray<double>>(p);
        }

        template <typename Number>
        Number do_compute_value(const dealii::Point<dim, Number> &p) const
        {
            return p[n];
        }
    };

    /**
     * @brief Piecewise diffusion coefficient used by the lab_03 benchmark.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    class Lab03Mu : public MFSolver::RealFunction<dim>
    {
    public:
        Lab03Mu() : MFSolver::RealFunction<dim>() {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return (p[0] < 0.5 ? 100.0 : 1.0);
        }

        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int component = 0) const override
        {
            dealii::VectorizedArray<float> result;
            for (unsigned int i = 0; i < dealii::VectorizedArray<float>::size(); ++i)
            {
                result[i] = (p[0][i] < 0.5f ? 100.0f : 1.0f);
            }
            return result;
        }

        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int component = 0) const override
        {
            dealii::VectorizedArray<double> result;
            for (unsigned int i = 0; i < dealii::VectorizedArray<double>::size(); ++i)
            {
                result[i] = (p[0][i] < 0.5 ? 100.0 : 1.0);
            }
            return result;
        }
    };

    /**
     * @brief Advection field used by the Miro benchmark definition.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    class beta_term_of_miro_problem : public MFSolver::VectorFunctionWithGradient<dim>
    {
    public:
        beta_term_of_miro_problem() : MFSolver::VectorFunctionWithGradient<dim>() {}

        virtual typename MFSolver::VectorFunctionWithGradient<dim>::template value_type<double> value(const dealii::Point<dim> &p) const override
        {
            dealii::Tensor<1, dim> b;
            b[0] = p[0];
            b[1] = p[1];
            return b;
        }

        virtual typename MFSolver::VectorFunctionWithGradient<dim>::template value_type<dealii::VectorizedArray<float>> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p) const override
        {
            dealii::Tensor<1, dim, dealii::VectorizedArray<float>> b;
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int i = 0; i < dealii::VectorizedArray<float>::size(); ++i)
                    b[d][i] = p[d][i];
            return b;
        }

        virtual typename MFSolver::VectorFunctionWithGradient<dim>::template value_type<dealii::VectorizedArray<double>> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p) const override
        {
            dealii::Tensor<1, dim, dealii::VectorizedArray<double>> b;
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int i = 0; i < dealii::VectorizedArray<double>::size(); ++i)
                    b[d][i] = p[d][i];
            return b;
        }

        virtual double divergence(const dealii::Point<dim> &p) const override
        {
            return trace(gradient(p));
        }

        virtual dealii::VectorizedArray<float> divergence(const dealii::Point<dim, dealii::VectorizedArray<float>> &p) const override
        {
            dealii::VectorizedArray<float> result;
            for (unsigned int i = 0; i < dealii::VectorizedArray<float>::size(); ++i)
            {
                dealii::Point<dim> point_i;
                for (unsigned int d = 0; d < dim; ++d)
                    point_i[d] = p[d][i];
                result[i] = trace(gradient(point_i));
            }
            return result;
        }

        virtual dealii::VectorizedArray<double> divergence(const dealii::Point<dim, dealii::VectorizedArray<double>> &p) const override
        {
            dealii::VectorizedArray<double> result;
            for (unsigned int i = 0; i < dealii::VectorizedArray<double>::size(); ++i)
            {
                dealii::Point<dim> point_i;
                for (unsigned int d = 0; d < dim; ++d)
                    point_i[d] = p[d][i];
                result[i] = trace(gradient(point_i));
            }
            return result;
        }

        virtual typename MFSolver::VectorFunctionWithGradient<dim>::template gradient_type<double> gradient(const dealii::Point<dim> &p) const override
        {
            dealii::Tensor<2, dim> grad;
            grad[0][0] = 1.0;
            grad[1][1] = 1.0;
            return grad;
        }

        virtual typename MFSolver::VectorFunctionWithGradient<dim>::template gradient_type<dealii::VectorizedArray<float>> gradient(const dealii::Point<dim, dealii::VectorizedArray<float>> &p) const override
        {
            dealii::Tensor<2, dim, dealii::VectorizedArray<float>> grad;
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int i = 0; i < dealii::VectorizedArray<float>::size(); ++i)
                    grad[d][d][i] = 1.0f;
            return grad;
        }

        virtual typename MFSolver::VectorFunctionWithGradient<dim>::template gradient_type<dealii::VectorizedArray<double>> gradient(const dealii::Point<dim, dealii::VectorizedArray<double>> &p) const override
        {
            dealii::Tensor<2, dim, dealii::VectorizedArray<double>> grad;
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int i = 0; i < dealii::VectorizedArray<double>::size(); ++i)
                    grad[d][d][i] = 1.0;
            return grad;
        }
    };

    /**
     * @brief Forcing term used by the Miro benchmark definition.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    class forcing_term_of_miro_problem : public MFSolver::RealFunction<dim>
    {
    public:
        forcing_term_of_miro_problem() : MFSolver::RealFunction<dim>() {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return p[0] + 2 * p[1] + p[0] * p[0] + 3 * p[0] * p[1] + 2 * p[1] * p[1];
        }

        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int component = 0) const override
        {
            dealii::VectorizedArray<float> result;
            for (unsigned int i = 0; i < dealii::VectorizedArray<float>::size(); ++i)
            {
                result[i] = p[0][i] + 2 * p[1][i] + p[0][i] * p[0][i] + 3 * p[0][i] * p[1][i] + 2 * p[1][i] * p[1][i];
            }
            return result;
        }

        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int component = 0) const override
        {
            dealii::VectorizedArray<double> result;
            for (unsigned int i = 0; i < dealii::VectorizedArray<double>::size(); ++i)
            {
                result[i] = p[0][i] + 2 * p[1][i] + p[0][i] * p[0][i] + 3 * p[0][i] * p[1][i] + 2 * p[1][i] * p[1][i];
            }
            return result;
        }
    };

    /**
     * @brief Gaussian initial condition used by transient benchmark problems.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    class GaussianFunction : public MFSolver::RealFunction<dim>
    {
    public:
        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return do_compute_value<double>(p);
        }

        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int component = 0) const override
        {
            return do_compute_value<dealii::VectorizedArray<float>>(p);
        }

        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int component = 0) const override
        {
            return do_compute_value<dealii::VectorizedArray<double>>(p);
        }

    private:
        template <typename Number>
        Number do_compute_value(const dealii::Point<dim, Number> &p) const
        {
            Number r2 = (p[0] - 0.2) * (p[0] - 0.2) + (p[1] - 0.2) * (p[1] - 0.2);
            if constexpr (dim > 2)
                r2 += (p[2] - 0.2) * (p[2] - 0.2);

            return std::exp(-75.0 * r2);
        }
    };

    /**
     * @brief Exact solution for the manufactured-solution benchmark.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    class MMSExactSolution : public MFSolver::RealFunction<dim> {
    public:
        virtual double value(const dealii::Point<dim> &p, const unsigned int = 0) const override {
            double val = 1.0;
            for (unsigned int d = 0; d < dim; ++d) val *= std::sin(M_PI * p[d]);
            return val;
        }
        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int = 0) const override {
            dealii::VectorizedArray<float> val = dealii::make_vectorized_array<float>(1.0);
            for (unsigned int d = 0; d < dim; ++d) val *= std::sin(static_cast<float>(M_PI) * p[d]);
            return val;
        }
        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int = 0) const override {
            dealii::VectorizedArray<double> val = dealii::make_vectorized_array<double>(1.0);
            for (unsigned int d = 0; d < dim; ++d) val *= std::sin(M_PI * p[d]);
            return val;
        }
    };

    /**
     * @brief Forcing term consistent with the manufactured exact solution.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    class MMSForcingTerm : public MFSolver::RealFunction<dim> {
    public:
        virtual double value(const dealii::Point<dim> &p, const unsigned int = 0) const override {
            double val = 1.0;
            for (unsigned int d = 0; d < dim; ++d) val *= std::sin(M_PI * p[d]);
            return (dim * M_PI * M_PI + 1.0) * val;
        }
        virtual dealii::VectorizedArray<float> value(const dealii::Point<dim, dealii::VectorizedArray<float>> &p, const unsigned int = 0) const override {
            dealii::VectorizedArray<float> val = dealii::make_vectorized_array<float>(1.0);
            for (unsigned int d = 0; d < dim; ++d) val *= std::sin(static_cast<float>(M_PI) * p[d]);
            return (static_cast<float>(dim * M_PI * M_PI + 1.0)) * val;
        }
        virtual dealii::VectorizedArray<double> value(const dealii::Point<dim, dealii::VectorizedArray<double>> &p, const unsigned int = 0) const override {
            dealii::VectorizedArray<double> val = dealii::make_vectorized_array<double>(1.0);
            for (unsigned int d = 0; d < dim; ++d) val *= std::sin(M_PI * p[d]);
            return (dim * M_PI * M_PI + 1.0) * val;
        }
    };

    /**
     * @brief A common structure to hold the algebraic and analytical data
     * required defining the Advection-Diffusion-Reaction (ADR) problem.
     *
     * This ensures that both the Matrix-Based and Matrix-Free solvers
     * solve the exact same mathematical problem.
     */
    /**
     * @brief Complete runtime description of one ADR benchmark problem.
     * @tparam dim Spatial dimension of the problem.
     */
    template <int dim>
    struct ProblemData
    {
        std::string mesh_filename; /**< Filename from which to load the mesh. */
        std::string problem_name;  /**< Name that the problem solution will saved with. */
        unsigned int fe_degree = 1; /**< Degree of the FE_Q finite element used by both solver implementations. */

        unsigned int num_levels; /**< Number of multigrid levels in the V-cycle. */

        /// @todo Verify whether this parameter is still used by the solvers.
        unsigned int num_quadrature_points; /**< Number of quadrature points. */

        double lv0_smoothing_range; /**< The range between the largest and the smaller eigenvalue for the lower level of the multigrid V-Cycle. */
        /// @note The level-0 smoothing degree is intentionally unset because
        /// the lowest level is used as a solver rather than a preconditioner.
        /// @note The level-0 eigenvalue iteration count is intentionally unset
        /// because the lowest-level row count is used instead.

        double lvgt0_smoothing_range;                     /**< The range between the largest and the smaller eigenvalue for all but the lower level of the multigrid V-Cycle. */
        double lvgt0_smoothing_degree;                    /**< The number of smoothing iterations for all but the lower level of the multigrid V-Cycle. */
        double lvgt0_smoothing_eigenvalue_max_iterations; /**< The maximum number of iterations used to find the maximum eigenvalue forall but the lower level of the multigrid V-Cycle. */

        unsigned int solver_max_iterations; /**< Maximum number of iterations when solving the algebraic system. */
        double solver_tolerance_factor;     /**< Factor to multiply to the l2 norm of the rhs of the algebraic system in order to get the absolute tolerance. */

        unsigned int refinement_level = 3; /**< Total number of times the grid is refined. This should be changed for problem to problem. */
        unsigned int n_additional_refinements = 0; /**< Extra global refinements requested on top of the problem default. */
        /// @todo Verify where this refinement coefficient is consumed.
        unsigned int refinement_coefficient_per_level = 4; /**< Mesh refinement level (if generating a hyper_cube/hyper_ball) */

        /// @name PDE coefficients
        /// @{

        std::shared_ptr<MFSolver::RealFunction<dim>> mu;                 /**< Diffusion coefficient function: mu(x) */
        std::shared_ptr<MFSolver::VectorFunctionWithGradient<dim>> beta; /**< Advection coefficient function: beta(x) (velocity field) */
        std::shared_ptr<MFSolver::RealFunction<dim>> gamma;              /**< Reaction coefficient function: gamma(x) (or k in some notations) */

        std::shared_ptr<MFSolver::RealFunction<dim>> forcing_term; /**< Forcing term: f(x) */
        std::shared_ptr<MFSolver::RealFunction<dim>> exact_solution = nullptr;          /**< Exact solution: u(x) */
        /// @}

        /// @name Time dependency parameters
        /// @{
        bool is_time_dependent = false;                                 /**< Flag to explicitly mark this problem as unsteady/transient. */
        double delta_t = 0.0;                                           /**< The size of the time step. */
        double end_time = 0.0;                                          /**< The final simulation time. */
        std::shared_ptr<MFSolver::RealFunction<dim>> initial_condition; /**< u(x, t=0): The initial state of the domain. */
        MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;        /**< Dirichlet boundaries. */
        MFSolver::NeumannBoundaries<dim> neumann_boundaries;            /**< Neumann boundaries. */
        /// @}

        /**
         * @brief Build the generic steady test case.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> standard_test_case(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (int i = 0; i < 5; i++)
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(static_cast<double>(i));

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(1.0);

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "standard",
                .fe_degree = fe_degree,
                .num_levels = 5,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 1000,
                .solver_tolerance_factor = 1e-10,

                .mu = std::make_shared<ConstantRealFunction<dim>>(2.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        /**
         * @brief Build the steady advanced benchmark.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> advanced_test_case(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Side 0 value. */
            dirichlet_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(1.0); /**< Side 1 value. */

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            /// The other four faces use zero-flux Neumann insulation.
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "advanced",
                .fe_degree = fe_degree,
                .num_levels = 5,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                .mu = std::make_shared<ConstantRealFunction<dim>>(2.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),

                /// Set zero forcing for a boundary-driven steady Laplace solution.
                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(0.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        /**
         * @brief Build a Neumann-boundary regression test case.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> test_case_neumann_fix(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(10.0); /**< Exact flux on face 1. */
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "neumann",
                .fe_degree = fe_degree,
                .num_levels = 5,
                .num_quadrature_points = fe_degree + 1,
                .lv0_smoothing_range = 1.e-3,
                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,
                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                /// @note IMPORTANT: We set mu = 5.0 to trigger the bug.
                .mu = std::make_shared<ConstantRealFunction<dim>>(5.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),
                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(0.0),
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        /**
         * @brief Build the manufactured-solution benchmark.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> mms_test_case(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (int i = 0; i < 2 * dim; i++)
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;

            ProblemData<dim> data{
                .mesh_filename = "none",
                .problem_name = "mms",
                .fe_degree = fe_degree,
                .num_levels = 5,
                .num_quadrature_points = fe_degree + 1,
                .lv0_smoothing_range = 1.e-3,
                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,
                .solver_max_iterations = 1000,
                .solver_tolerance_factor = 1e-12,
                .refinement_level = 3,
                .refinement_coefficient_per_level = 4,
                .mu = std::make_shared<ConstantRealFunction<dim>>(1.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(1.0),
                .forcing_term = std::make_shared<MMSForcingTerm<dim>>(),
                .exact_solution = std::make_shared<MMSExactSolution<dim>>(),
                .is_time_dependent = false,
                .delta_t = 0.0,
                .end_time = 0.0,
                .initial_condition = nullptr,
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        /**
         * @brief Build a simple parabolic benchmark.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> test_case_parabolic(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Side 0 value. */
            dirichlet_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Side 1 value. */

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "parabolic",
                .fe_degree = fe_degree,
                .num_levels = 5,
                .num_quadrature_points = fe_degree + 1,
                .lv0_smoothing_range = 1.e-3,
                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,
                .solver_max_iterations = 1000,
                .solver_tolerance_factor = 1e-10,

                .mu = std::make_shared<ConstantRealFunction<dim>>(1.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),

                /// Uniformly heat the domain with a constant forcing term.
                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(6.0),
                .is_time_dependent = true,
                .delta_t = 0.1,
                .end_time = 1.0,
                .initial_condition = std::make_shared<ConstantRealFunction<dim>>(0.0),
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        /**
         * @brief Build a transient benchmark exercising all ADR terms.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> test_case_comprehensive_transient(const unsigned int fe_degree)
        {
            /// Exercise diffusion, advection, reaction, mixed boundaries, and time dependence.
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Left boundary. */
            dirichlet_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Right boundary. */
            dirichlet_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Bottom boundary. */
            dirichlet_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Top boundary. */

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Front zero-flux boundary. */
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0); /**< Back zero-flux boundary. */

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "transient",
                .fe_degree = fe_degree,
                .num_levels = 5,
                .num_quadrature_points = fe_degree + 1,
                .lv0_smoothing_range = 1.e-3,
                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,
                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                /// @name Physics stress-test coefficients
                /// @{
                .mu = std::make_shared<ConstantRealFunction<dim>>(0.02),                /**< Small diffusion so the initial blob spreads slowly. */
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.8), /**< Diagonal advection velocity. (0.8, 0.8, 0.8) */
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.5),              /**< Gradual reaction decay. */
                /// @}

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(0.0), /**< No external source; only the drifting blob. */
                .is_time_dependent = true,
                .delta_t = 0.005,
                .end_time = 0.5,
                .initial_condition = std::make_shared<GaussianFunction<dim>>(), /**< Initial hot sphere near (0.2, 0.2, 0.2). */
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        /**
         * @brief Build the heated-wall transient benchmark.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> test_case_heated_wall(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(50.0);
            dirichlet_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(20.0);

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(10.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(-10.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "heated_wall",
                .fe_degree = fe_degree,
                .num_levels = 5,
                .num_quadrature_points = fe_degree + 1,
                .lv0_smoothing_range = 1.e-3,
                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,
                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                /// @name Heated-wall physics coefficients
                /// @{
                .mu = std::make_shared<ConstantRealFunction<dim>>(0.02),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.8),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0),
                /// @}

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0),
                .is_time_dependent = true,
                .delta_t = 0.005,
                .end_time = 0.5,
                .initial_condition = std::make_shared<GaussianFunction<dim>>(),
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        /**
         * @brief Build the two-dimensional lab_02 Poisson benchmark.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> lab_02_poisson(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<SumReductionFunction<dim>>();
            dirichlet_boundaries[1] = std::make_shared<SumReductionFunction<dim>>();

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[2] = std::make_shared<NthCoordFunction<dim>>(1);
            neumann_boundaries[3] = std::make_shared<NthCoordFunction<dim>>(1);

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "lab_02",
                .fe_degree = fe_degree,
                .num_levels = 5,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                .mu = std::make_shared<ConstantRealFunction<dim>>(1.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(-5.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        /**
         * @brief Build the lab_03 diffusion-reaction benchmark.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> lab_03_dr_eq(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (size_t i = 0; i < 6; ++i)
            {
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            }

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "lab_03",
                .fe_degree = fe_degree,
                .num_levels = 5,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                .mu = std::make_shared<Lab03Mu<dim>>(),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(1.0),

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        /**
         * @brief Build the Miro benchmark problem.
         * @param fe_degree Degree of the FE_Q finite element.
         * @return Fully initialized problem descriptor.
         */
        static ProblemData<dim> see_miro_for_problem_definition_andrea_knows(const unsigned int fe_degree)
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (size_t i = 0; i < 4; ++i)
            {
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            }

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;

            ProblemData<dim> data{
                .mesh_filename = "input.msh",
                .problem_name = "andrea",
                .fe_degree = fe_degree,
                .num_levels = 5,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                .mu = std::make_shared<ConstantRealFunction<dim>>(0.0),
                .beta = std::make_shared<beta_term_of_miro_problem<dim>>(),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),

                .forcing_term = std::make_shared<forcing_term_of_miro_problem<dim>>(),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }
    };

} // namespace ADR
