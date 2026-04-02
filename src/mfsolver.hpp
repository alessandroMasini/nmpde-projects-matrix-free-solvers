#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

/**
 * \brief Namespace containing all the methods and type definitions used in the project.
 */
namespace MFSolver
{
    using namespace dealii;

    /**
     * \brief Like a vector, but distributed.
     * \tparam T The type of elements stored in the vector.
     */
    template <typename T>
    using DVector = LinearAlgebra::distributed::Vector<T>;

    /**
     * \brief Represents a function that takes a `dim`-dimensional vector and returns a real number.
     * \tparam dim The dimensionality of the input vector.
     */
    template <int dim>
    class RealFunction : public Function<dim>
    {
    public:
        /**
         * \brief Constructs a new instance of RealFunction.
         * \param _lambda The closure that will be used to compute the value of the function.
         */
        RealFunction(const std::function<double(const Point<dim> &)> &_lambda) : Function<dim>(), lambda(_lambda)
        {
        }

    private:
        /**
         * \brief The closure that will be used to compute the value of the function.
         */
        std::function<double(const Point<dim> &)> lambda;

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            return lambda(p);
        }
    };

    /**
     * \brief Represents a function that takes a `dim`-dimensional vector and returns another `dim`-dimensional vector.
     * \tparam dim The dimensionality of the input and output vector.
     */
    template <int dim>
    class VectorFunction : public Function<dim>
    {
    public:
        /**
         * \brief Constructs a new instance of VectorFunction.
         * \param _lambda The closure that will be used to compute the value of the function.
         *
         * \note Since the `Function::vector_value` default implementation calls `Function::value` for each vector component, avoid computing the entire vector and then returning the requested component inside `value` to improve performance.
         */
        VectorFunction(const std::function<Tensor<1, dim>(const Point<dim> &, const unsigned int)> &_lambda) : Function<dim>(), lambda(_lambda)
        {
        }

    private:
        /**
         * \brief The closure that will be used to compute the value of the function.
         */
        std::function<Tensor<1, dim>(const Point<dim> &, const unsigned int)> lambda;

        virtual double value(const Point<dim> &p, const unsigned int component) const override
        {
            return lambda(p, component);
        }
    };

    /**
     * \brief Represents a function that takes a `dim` dimensional vector and returns anothr `dim`-dimensional vector. Moreover, the represented function must be differentiable and it's gradientmust also be provided.
     * \tparam dim The dimensionality of the input and output vector.
     */
    template <int dim>
    class VectorFunctionWithGradient : public VectorFunction<dim>
    {
    public:
        VectorFunctionWithGradient(const std::function<Tensor<1, dim>(const Point<dim> &, const unsigned int)> &_lambda, const std::function<Tensor<1, dim>(const Point<dim> &, const unsigned int)> &_gradient_lambda) : VectorFunction<dim>(_lambda), gradient_lambda(_gradient_lambda)
        {
        }

        // TODO: This is not correct, I don't know how I have proven this but this is definitely not correct.
        double divergence(const Point<dim> &p)
        {
            double trace = 1.;

            for (unsigned int i = 0; i < dim; ++i)
            {
                trace *= gradient(p, i);
            }

            return trace;
        }

    private:
        std::function<Tensor<1, dim>(const Point<dim> &, const unsigned int)> gradient_lambda;

        virtual double gradient(const Point<dim> &p, const unsigned int component) override
        {
            return gradient_lambda(p, component);
        }
    };

    /**
     * \brief Represents a function that describes a Dirichlet boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using DirichletBoundary = RealFunction<dim>;

    /**
     * \brief Represents a function tht describes a Neumann boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using NeumannBoundary = RealFunction<dim>;

    /**
     * \brief Represents a mapping between boundaries (identified by boundary IDs) and the corresponding boundary condition.
     * \tparam T The type of boundary condition.
     */
    template <typename T>
    using Boundaries = std::unordered_map<types::boundary_id, T>;

    /**
     * \brief Represents a mapping between boundaries (represented by boundary IDs) and the corresponding Dirichlet boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using DirichletBoundaries = Boundaries<DirichletBoundary<dim>>;

    /**
     * \brief Represents a mapping between boundaries (represented by boundary IDs) and the corresponding Neumann boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using NeumannBoundaries = Boundaries<NeumannBoundary<dim>>;

    using Range = std::pair<unsigned int, unsigned int>;

    /**
     * \brief Represents a generic ADR problem.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    struct ADRProblem
    {
        RealFunction<dim> mu;     /**< Diffusion coefficient. */
        VectorFunction<dim> beta; /**< Advection coefficient. */
        RealFunction<dim> gamma;  /**< Reaction coefficient. */

        RealFunction<dim> f; /**< Forcing term. */

        // RealFunction<dim> g; /**< Dirichlet boundary. */
        // RealFunction<dim> h; /**< Neumann boundary. */

        DirichletBoundaries<dim> dirichlet_boundaries; /**< Dirichlet boundaries. */
        NeumannBoundaries<dim> neumann_boundaries;     /**< Neumann boundaries. */

        std::string mesh_filename; /**< Filename from which to load the mesh. */

        // unsigned int degree; // Unset as it needs to be a template argument

        unsigned int num_levels; /**< Number of multigrid levels in the V-cycle. */

        unsigned int num_quadrature_points; /**< Number of quadrature points. */

        unsigned int refinement_coefficient; /**< How many times the mesh is refined each time it gets refined. */

        double lv0_smoothing_range; /**< The range between the largest and the smaller eigenvalue for the lower level of the multigrid V-Cycle. */
        // double lv0_smoothing_degree; // Unset as we use invalid int to make this a solver instead of a preconditioner. See PreconditionChebyshev documentation
        // double lv0_smoothing_eigenvalue_max_iterations; // Unset as we use the number of rows of the lowest level matrix in muligrid V-Cycle

        double lvgt0_smoothing_range;                     /**< The range between the largest and the smaller eigenvalue for all but the lower level of the multigrid V-Cycle. */
        double lvgt0_smoothing_degree;                    /**< The number of smoothing iterations for all but the lower level of the multigrid V-Cycle. */
        double lvgt0_smoothing_eigenvalue_max_iterations; /**< The maximum number of iterations used to find the maximum eigenvalue forall but the lower level of the multigrid V-Cycle. */

        unsigned int solver_max_iterations; /**< Maximum number of iterations when solving the algebraic system. */
        double solver_tolerance_factor;     /**< Factor to multiply to the l2 norm of the rhs of the algebraic system in order to get the absolute tolerance. */
    };

    /**
     * \brief Abstract class used to keep a common interface between the matrix-free solver (MatrixFreeADRSolver) and the matrix-based solver (MatrixBasedADRSolver).
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     * \tparam fe_degree The degree of the finite elements used to solve the problem.
     */
    template <int dim, int fe_degree>
    class ADRSolver
    {
    public:
        /**
         * \brief Constructs a new instance of ADRSolver
         * \param _problem The problem this solver will solve.
         */
        ADRSolver(const ADRProblem<dim> &_problem)
            : problem(_problem)
        {
        }

        /**
         * \brief Destructor for ADRProblem.
         */
        virtual ~ADRSolver() = 0;

        /**
         * \brief Actually solve the ADRProblem.
         */
        virtual void run() = 0;

    protected:
        /**
         * \brief Sets up the algebraic system corresponding to the problem.
         */
        virtual void setup_system() = 0;

        /**
         * \brief Assembles the rhs of the algebraic system corresponding to the problem.
         */
        virtual void assemble_rhs() = 0;

        /**
         * \brief Solves the algebraic system corresponding to the problem.
         */
        virtual void solve() = 0;

        /**
         * \brief writes the result of the computation.
         */
        virtual void output_results() = 0;

        /**
         * \brief The problem this solver will solve.
         */
        ADRProblem<dim> problem;
    };

    template <int dim, int fe_degree, typename Number>
    class ADROperator : MatrixFreeOperators::Base<dim, DVector<Number>>
    {
    public:
        ADROperator() : Super()
        {
        }

        void clear() override
        {
            mu_coeff.reinit(0, 0);
            beta_coeff.reinit(0, 0);
            div_beta_coeff.reinit(0, 0);
            gamma_coeff.reinit(0, 0);

            Super::clear();
        }

        void evaluate_coefficients(
            const RealFunction<dim> &mu_coeff_function,
            const VectorFunctionWithGradient<dim> &beta_coeff_function,
            const RealFunction<dim> &gamma_coeff_function)
        {
            const unsigned int n_cells = this->data->n_cell_batches();
            Phi phi(*this->data);

            mu_coeff.reinit(n_cells, phi.n_q_points);
            beta_coeff.reinit(n_cells, phi.n_q_points);
            div_beta_coeff.reinit(n_cells, phi.n_q_points);
            gamma_coeff.reinit(n_cells, phi.n_q_points);

            for (unsigned int cell = 0; cell < n_cells; ++cell)
            {
                phi.reinit(cell);
                for (const unsigned int q : phi.quadrature_point_indices())
                {
                    Point<dim, Number> quadrature_point = phi.quadrature_point(q);

                    mu_coeff(cell, q) = mu_coeff_function.value(quadrature_point);
                    beta_coeff(cell, q) = beta_coeff_function.value(quadrature_point);
                    div_beta_coeff(cell, q) = beta_coeff_function.divergence(quadrature_point);
                    gamma_coeff(cell, q) = gamma_coeff_function.value(quadrature_point);
                }
            }
        }

        virtual void compute_diagonal() override
        {
            this->inverse_diagonal_entries.reset(new DiagonalMatrix<DVector<Number>>());

            DVector<Number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
            this->data->initialize_dof_vector(inverse_diagonal);

            MatrixFreeTools::compute_diagonal(*this->data, inverse_diagonal, &ADROperator::local_compute_diagonal, this);

            this->set_constrained_entries_to_one(inverse_diagonal);

            for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
            {
                Assert(inverse_diagonal.local_element(i) > 0., ExcMessage("Error: non-positive entry found. Operator must be positive definite."));

                inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
            }
        }

    private:
        using Super = MatrixFreeOperators::Base<dim, DVector<Number>>;

        using Phi = FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>;

        void rhs_computation(Phi &phi, const unsigned int cell, const unsigned int q) // This code is extracted and reused by `local_apply` and `local_compute_diagonal`.
        // According to Step-37, it seems that they are the same but I have not found proof for it.
        {
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            for (const unsigned int q : phi.quadrature_point_indices())
            {
                Tensor<1, dim, VectorizedArray<Number>> gradient_of_u = phi.get_gradient(q);
                VectorizedArray<Number> value_of_u = phi.get_value(q);

                VectorizedArray<Number> mu = mu_coeff(cell, q);
                Tensor<1, dim, VectorizedArray<Number>> beta = beta_coeff(cell, q);
                VectorizedArray<Number> div_beta = div_beta_coeff(cell, q);
                VectorizedArray<Number> gamma = gamma_coeff(cell, q);

                phi.submit_gradient(mu * gradient_of_u, q);
                phi.submit_value(scalar_product(gradient_of_u, beta) + (div_beta + gamma) * value_of_u, q);
            }

            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        }

        void local_apply(const MatrixFree<dim, Number> &data, DVector<Number> &dst, const DVector<Number> &src, const Range &cell_range) const
        {
            Phi phi(data);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
                AssertDimension(coefficient.size(0), data.n_cell_batches());
                AssertDimension(coefficient.size(1), phi.n_q_points);

                phi.reinit(cell);
                phi.read_dof_values(src);

                rhs_computation(phi, cell, q);

                phi.distribute_local_to_global(dst);
            }
        }

        void local_compute_diagonal(Phi &phi) const
        {
            const unsigned int cell = phi.get_current_cell_index();
            rhs_computation(phi, cell, q);
        }

        virtual void apply_add(DVector<Number> &dst, const DVector<Number> &src) const override
        {
            this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
        }

        Table<2, VectorizedArray<Number>> mu_coeff;
        Table<2, Tensor<1, dim, VectorizedArray<Number>>> beta_coeff; // Is this OK?
        Table<2, VectorizedArray<Number>> div_beta_coeff;             // We precalculate the divergences of beta in order to not break the SIMD context inside the for loops
        Table<2, VectorizedArray<Number>> gamma_coeff;
    };

    /**
     * \brief Solver class that will solve an ADR problem using matrix-free techniques.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     * \tparam fe_degree The degree of the finite elements used to solve the problem.
     */
    template <int dim, int fe_degree>
    class MatrixFreeADRSolver : public ADRSolver<dim, fe_degree>
    {
    public:
        MatrixFreeADRSolver(const ADRProblem<dim> &_problem) : ADRSolver<dim, fe_degree>(_problem)
        {
        }

        void run() override {}

    private:
        void setup_system() override {}
        void assemble_rhs() override {}
        void solve() override {}
        void output_results() override {}

#ifdef DEAL_II_WITH_P4EST
        parallel::distributed::Triangulation<dim> triangulation;
#else
        Triangulation<dim> triangulation;
#endif

        const FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        const MappingQ1<dim> mapping;

        AffineConstraints<double> constraints;

        using SystemMatrixType = ADROperator<dim, fe_degree, double>;
        SystemMatrixType system_matrix;

        MGConstrainedDoFs mg_constrained_dofs;

        using LevelMatrixType = ADROperator<dim, fe_degree, float>;
        MGLevelObject<LevelMatrixType> mg_matrices;

        DVector<double> solution;
        DVector<double> system_rhs;

        double setup_time;
        ConditionalOStream pcout;
        ConditionalOStream time_details;
    };

    /**
     * \brief Solver class that will solve an ADR problem using matrix-based techniques.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     * \tparam fe_degree The degree of the finite elements used to solve the problem.
     */
    template <int dim, int fe_degree>
    class MatrixBasedADRSolver : public ADRSolver<dim, fe_degree>
    {
    public:
        MatrixBasedADRSolver(const ADRProblem<dim> &_problem) : ADRSolver<dim, fe_degree>(_problem)
        {
        }

        void run() override {}

    private:
        void setup_system() override {}
        void assemble_rhs() override {}
        void solve() override {}
        void output_results() override {}

        // Number of MPI processes.
        const unsigned int mpi_size;

        // Rank of the current MPI process.
        const unsigned int mpi_rank;

        // Triangulation.
        // TODO: clarify difference with MatrixFreeADRSolver mesh types
        parallel::fullydistributed::Triangulation<dim> mesh;

        // Finite element space.
        // TODO: clarify difference with MatrixFreeADRSolver fe non-pointer
        std::unique_ptr<FiniteElement<dim>> fe;

        // TODO: should we add
        // - mapping
        // - affine constraints
        // here?

        // Quadrature formula.
        std::unique_ptr<Quadrature<dim>> quadrature;

        // DoF handler.
        DoFHandler<dim> dof_handler;

        // System matrix.
        TrilinosWrappers::SparseMatrix system_matrix;

        // System right-hand side.
        TrilinosWrappers::MPI::Vector system_rhs;

        // System solution, without ghost elements.
        TrilinosWrappers::MPI::Vector solution_owned;

        // System solution, with ghost elements.
        TrilinosWrappers::MPI::Vector solution;

        // Output stream for process 0.
        ConditionalOStream pcout;
    };
};
