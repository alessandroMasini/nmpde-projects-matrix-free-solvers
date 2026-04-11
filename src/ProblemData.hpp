#ifndef PROBLEMDATA_HPP
#define PROBLEMDATA_HPP

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>

#include "function_types.hpp"
#include "boundaries.hpp"

#include <memory>

namespace ADR
{

    template <int dim>
    class ConstantRealFunction : public MFSolver::RealFunction<dim>
    {
    private:
        double val;

    public:
        ConstantRealFunction(double v) : val(v) {}

        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            return value<double>(p, component);
        }

        template <typename Number>
        Number value(const dealii::Point<dim, Number> & /*p*/, const unsigned int /*component*/ = 0) const
        {
            return Number(val);
        }
    };

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
            return value<double>(p);
        }

        template <typename Number>
        typename Super::template value_type<Number> value(const dealii::Point<dim, Number> & /*p*/) const
        {
            dealii::Tensor<1, dim, Number> b;
            for (unsigned int d = 0; d < dim; ++d)
                b[d] = Number(val);
            return b;
        }

        virtual double divergence(const dealii::Point<dim> &p) const override
        {
            return divergence<double>(p);
        }

        template <typename Number>
        Number divergence(const dealii::Point<dim, Number> & /*p*/) const
        {
            return Number(0.0);
        }

        virtual typename Super::template gradient_type<double> gradient(const dealii::Point<dim> &p) const override
        {
            return gradient<double>(p);
        }

        template <typename Number>
        typename Super::template gradient_type<Number> gradient(const dealii::Point<dim, Number> & /*p*/) const
        {
            return typename Super::template gradient_type<Number>();
        }
    };

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
     * @brief A common structure to hold the algebraic and analytical data
     * required defining the Advection-Diffusion-Reaction (ADR) problem.
     *
     * This ensures that both the Matrix-Based and Matrix-Free solvers
     * solve the exact same mathematical problem.
     */
    template <int dim, int fe_degree, template <int> typename MuCoeffType, template <int> typename BetaCoeffType, template <int> typename GammaCoeffType>
    struct ProblemData
    {
        // using MuCoeffType = _MuCoeffType;
        // using BetaCoeffType = _BetaCoeffType;
        // using GammaCoeffType = _GammaCoeffType;

        std::string mesh_filename; /**< Filename from which to load the mesh. */

        unsigned int num_levels; /**< Number of multigrid levels in the V-cycle. */

        // TODO: is this actually used?
        unsigned int num_quadrature_points; /**< Number of quadrature points. */

        double lv0_smoothing_range; /**< The range between the largest and the smaller eigenvalue for the lower level of the multigrid V-Cycle. */
        // double lv0_smoothing_degree; // Unset as we use invalid int to make this a solver instead of a preconditioner. See PreconditionChebyshev documentation
        // double lv0_smoothing_eigenvalue_max_iterations; // Unset as we use the number of rows of the lowest level matrix in muligrid V-Cycle

        double lvgt0_smoothing_range;                     /**< The range between the largest and the smaller eigenvalue for all but the lower level of the multigrid V-Cycle. */
        double lvgt0_smoothing_degree;                    /**< The number of smoothing iterations for all but the lower level of the multigrid V-Cycle. */
        double lvgt0_smoothing_eigenvalue_max_iterations; /**< The maximum number of iterations used to find the maximum eigenvalue forall but the lower level of the multigrid V-Cycle. */

        unsigned int solver_max_iterations; /**< Maximum number of iterations when solving the algebraic system. */
        double solver_tolerance_factor;     /**< Factor to multiply to the l2 norm of the rhs of the algebraic system in order to get the absolute tolerance. */

        // TODO: where is this used???
        unsigned int refinement_coefficient_per_level = 4; /**< Mesh refinement level (if generating a hyper_cube/hyper_ball) */

        // --- PDE Coefficients ---

        MuCoeffType<dim> mu;       /**< Diffusion coefficient function: mu(x) */
        BetaCoeffType<dim> beta;   /**< Advection coefficient function: beta(x) (velocity field) */
        GammaCoeffType<dim> gamma; /**< Reaction coefficient function: gamma(x) (or k in some notations) */

        std::shared_ptr<MFSolver::RealFunction<dim>> forcing_term; /**< Forcing term: f(x) */

        MFSolver::DirichletBoundaries<dim> dirichlet_boundaries; /**< Dirichlet boundaries. */
        MFSolver::NeumannBoundaries<dim> neumann_boundaries;     /**< Neumann boundaries. */

        /**
         * @brief Helper to initialize with some default test-case values
         */
        static ProblemData<dim, fe_degree, ConstantRealFunction, ConstantVectorFunctionWithGradient, ConstantRealFunction> standard_test_case()
        {
            ConstantDirichletBoundary<dim> cdb(1.0);

            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = &cdb;

            ProblemData<dim, fe_degree, ConstantRealFunction, ConstantVectorFunctionWithGradient, ConstantRealFunction> data{
                .mesh_filename = "input.msh",
                .num_levels = 5,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                .mu = ConstantRealFunction<dim>(1.0),
                .beta = ConstantVectorFunctionWithGradient<dim>(1.0),
                .gamma = ConstantRealFunction<dim>(0.0),

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = MFSolver::NeumannBoundaries<dim>(),
            };

            return data;
        }
    };

} // namespace ADR

#endif // PROBLEMDATA_HPP
