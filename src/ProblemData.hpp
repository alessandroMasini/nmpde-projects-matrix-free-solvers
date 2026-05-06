// 
// #ifndef PROBLEMDATA_HPP
// #define PROBLEMDATA_HPP

// // #include <memory>

// // #include <deal.II/base/tensor_function.h>
// // #include <deal.II/base/function.h>

// #include "function_types.hpp"

// namespace ADR
// {

//     template <int dim>
//     class ConstantRealFunction : public MFSolver::RealFunction<dim>
//     {
//     private:
//         double val;

//     public:
//         ConstantRealFunction(double v) : val(v) {}

//         virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
//         {
//             return value<double>(p, component);
//         }

//         template <typename Number>
//         Number value(const dealii::Point<dim, Number> & /*p*/, const unsigned int /*component*/ = 0) const
//         {
//             return Number(val);
//         }
//     };

//     template <int dim>
//     class ConstantVectorFunctionWithGradient : public MFSolver::VectorFunctionWithGradient<dim>
//     {
//     private:
//         double val;

//     public:
//         using Super = typename MFSolver::VectorFunctionWithGradient<dim>;

//         ConstantVectorFunctionWithGradient(double v) : val(v) {}

//         virtual typename Super::template value_type<double> value(const dealii::Point<dim> &p) const override
//         {
//             return value<double>(p);
//         }

//         template <typename Number>
//         typename Super::template value_type<Number> value(const dealii::Point<dim, Number> & /*p*/) const
//         {
//             dealii::Tensor<1, dim, Number> b;
//             for (unsigned int d = 0; d < dim; ++d)
//                 b[d] = Number(val);
//             return b;
//         }

//         virtual double divergence(const dealii::Point<dim> &p) const override
//         {
//             return divergence<double>(p);
//         }

//         template <typename Number>
//         Number divergence(const dealii::Point<dim, Number> & /*p*/) const
//         {
//             return Number(0.0);
//         }

//         virtual typename Super::template gradient_type<double> gradient(const dealii::Point<dim> &p) const override
//         {
//             return gradient<double>(p);
//         }

//         template <typename Number>
//         typename Super::template gradient_type<Number> gradient(const dealii::Point<dim, Number> & /*p*/) const
//         {
//             return typename Super::template gradient_type<Number>();
//         }
//     };

//     /**
//      * @brief A common structure to hold the algebraic and analytical data
//      * required defining the Advection-Diffusion-Reaction (ADR) problem.
//      *
//      * This ensures that both the Matrix-Based and Matrix-Free solvers
//      * solve the exact same mathematical problem.
//      */
//     template <int dim, int fe_degree>
//     struct ProblemData
//     {
//         std::string mesh_filename; /**< Filename from which to load the mesh. */

//         unsigned int num_levels; /**< Number of multigrid levels in the V-cycle. */

//         // TODO: is this actually used?
//         unsigned int num_quadrature_points; /**< Number of quadrature points. */

//         double lv0_smoothing_range; /**< The range between the largest and the smaller eigenvalue for the lower level of the multigrid V-Cycle. */
//         // double lv0_smoothing_degree; // Unset as we use invalid int to make this a solver instead of a preconditioner. See PreconditionChebyshev documentation
//         // double lv0_smoothing_eigenvalue_max_iterations; // Unset as we use the number of rows of the lowest level matrix in muligrid V-Cycle

//         double lvgt0_smoothing_range;                     /**< The range between the largest and the smaller eigenvalue for all but the lower level of the multigrid V-Cycle. */
//         double lvgt0_smoothing_degree;                    /**< The number of smoothing iterations for all but the lower level of the multigrid V-Cycle. */
//         double lvgt0_smoothing_eigenvalue_max_iterations; /**< The maximum number of iterations used to find the maximum eigenvalue forall but the lower level of the multigrid V-Cycle. */

//         unsigned int solver_max_iterations; /**< Maximum number of iterations when solving the algebraic system. */
//         double solver_tolerance_factor;     /**< Factor to multiply to the l2 norm of the rhs of the algebraic system in order to get the absolute tolerance. */

//         // TODO: where is this used???
//         unsigned int refinement_coefficient_per_level = 1; /**< Mesh refinement level (if generating a hyper_cube/hyper_ball) */

//         // Number of elements in each direction (if using a subdivision)
//         // unsigned int elements_per_direction = 10;

//         // --- PDE Coefficients ---

//         std::shared_ptr<MFSolver::RealFunction<dim>> mu;                 /**< Diffusion coefficient function: mu(x) */
//         std::shared_ptr<MFSolver::VectorFunctionWithGradient<dim>> beta; /**< Advection coefficient function: beta(x) (velocity field) */
//         std::shared_ptr<MFSolver::RealFunction<dim>> gamma;              /**< Reaction coefficient function: gamma(x) (or k in some notations) */

//         std::shared_ptr<MFSolver::RealFunction<dim>> forcing_term; /**< Forcing term: f(x) */

//         std::shared_ptr<MFSolver::RealFunction<dim>> dirichlet_boundary_value; /**< Dirichlet boundary condition: g(x) for the general lifting */
//         std::shared_ptr<MFSolver::RealFunction<dim>> neumann_boundary_value;   /**< Neumann boundary conditions: h(g) */

//         /**
//          * @brief Helper to initialize with some default test-case values
//          */
//         static ProblemData<dim, fe_degree> standard_test_case()
//         {
//             ProblemData<dim, fe_degree> data;

//             // data.fe_degree = 1;
//             // data.refinement_level = 5;

//             data.mu = std::make_shared<ConstantRealFunction<dim>>(1.0);
//             data.beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(1.0);
//             data.gamma = std::make_shared<ConstantRealFunction<dim>>(0.0);
//             data.forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0);
//             data.dirichlet_boundary_value = std::make_shared<ConstantRealFunction<dim>>(0.0);

//             data.mesh_filename = "input.msh";
//             data.num_levels = 5;
//             data.num_quadrature_points = 3;

//             data.lv0_smoothing_range = 1.e-3;

//             data.lvgt0_smoothing_range = 15;
//             data.lvgt0_smoothing_degree = 5;
//             data.lvgt0_smoothing_eigenvalue_max_iterations = 10;

//             data.solver_max_iterations = 100;
//             data.solver_tolerance_factor = 1e-12;

//             return data;
//         }
//     };

// } // namespace ADR

// #endif // PROBLEMDATA_HPP


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
     * @brief A common structure to hold the algebraic and analytical data
     * required defining the Advection-Diffusion-Reaction (ADR) problem.
     *
     * This ensures that both the Matrix-Based and Matrix-Free solvers
     * solve the exact same mathematical problem.
     */
    template <int dim, int fe_degree>
    struct ProblemData
    {
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

        std::shared_ptr<MFSolver::RealFunction<dim>> mu;                 /**< Diffusion coefficient function: mu(x) */
        std::shared_ptr<MFSolver::VectorFunctionWithGradient<dim>> beta; /**< Advection coefficient function: beta(x) (velocity field) */
        std::shared_ptr<MFSolver::RealFunction<dim>> gamma;              /**< Reaction coefficient function: gamma(x) (or k in some notations) */

        std::shared_ptr<MFSolver::RealFunction<dim>> forcing_term; /**< Forcing term: f(x) */

        MFSolver::DirichletBoundaries<dim> dirichlet_boundaries; /**< Dirichlet boundaries. */
        MFSolver::NeumannBoundaries<dim> neumann_boundaries;     /**< Neumann boundaries. */

        /**
         * @brief Helper to initialize with some default test-case values
         */
        static ProblemData<dim, fe_degree> standard_test_case()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (int i = 0; i < 2 * dim; i++)
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(static_cast<double>(i));

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(1.0);

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
                // TODO: reset this to 5
                .num_levels = 0,

                .num_quadrature_points = fe_degree + 1,

                .lv0_smoothing_range = 1.e-3,

                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,

                .solver_max_iterations = 1000,
                .solver_tolerance_factor = 1e-12,

                .mu = std::make_shared<ConstantRealFunction<dim>>(2.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        static ProblemData<dim, fe_degree> advanced_test_case()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0); // Side 0: 0.0
            dirichlet_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(1.0); // Side 1: 1.0

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            // The other 4 faces are zero-flux Neumann insulation
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
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

                // Set forcing term to 0 for a generic laplace steady state solution driven by boundaries
                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(0.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        static ProblemData<dim, fe_degree> test_case_neumann_fix()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(10.0); // Flux is exactly 10.0 on face 1
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
                .num_levels = 5,
                .num_quadrature_points = fe_degree + 1,
                .lv0_smoothing_range = 1.e-3,
                .lvgt0_smoothing_range = 15,
                .lvgt0_smoothing_degree = 5,
                .lvgt0_smoothing_eigenvalue_max_iterations = 10,
                .solver_max_iterations = 100,
                .solver_tolerance_factor = 1e-12,

                // IMPORTANT: We set mu = 5.0 to trigger the bug.
                .mu = std::make_shared<ConstantRealFunction<dim>>(5.0),
                .beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(0.0),
                .gamma = std::make_shared<ConstantRealFunction<dim>>(0.0),
                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(0.0),
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        static ProblemData<dim, fe_degree> test_case_parabolic()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<ConstantRealFunction<dim>>(0.0); // Side 0: 0.0
            dirichlet_boundaries[1] = std::make_shared<ConstantRealFunction<dim>>(0.0); // Side 1: 0.0

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[2] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[3] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[4] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            neumann_boundaries[5] = std::make_shared<ConstantRealFunction<dim>>(0.0);

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
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

                // We heat the whole domain up uniformly with a forcing term of 6.0
                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(6.0),
                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };
            return data;
        }

        static ProblemData<dim, fe_degree> lab_02_poisson()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            dirichlet_boundaries[0] = std::make_shared<SumReductionFunction<dim>>();
            dirichlet_boundaries[1] = std::make_shared<SumReductionFunction<dim>>();

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;
            neumann_boundaries[2] = std::make_shared<NthCoordFunction<dim>>(1);
            neumann_boundaries[3] = std::make_shared<NthCoordFunction<dim>>(1);

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
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

                .forcing_term = std::make_shared<ConstantRealFunction<dim>>(-5.0),

                .dirichlet_boundaries = dirichlet_boundaries,
                .neumann_boundaries = neumann_boundaries,
            };

            return data;
        }

        static ProblemData<dim, fe_degree> lab_03_dr_eq()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (size_t i = 0; i < 6; ++i)
            {
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            }

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
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

        static ProblemData<dim, fe_degree> see_miro_for_problem_definition_andrea_knows()
        {
            MFSolver::DirichletBoundaries<dim> dirichlet_boundaries;
            for (size_t i = 0; i < 4; ++i)
            {
                dirichlet_boundaries[i] = std::make_shared<ConstantRealFunction<dim>>(0.0);
            }

            MFSolver::NeumannBoundaries<dim> neumann_boundaries;

            ProblemData<dim, fe_degree> data{
                .mesh_filename = "input.msh",
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

#endif // PROBLEMDATA_HPP