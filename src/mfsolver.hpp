#pragma once

// TODO: reorder imports in a neat way

#include <unordered_map>
#include <string>
#include <functional>

#include <deal.II/base/types.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/la_parallel_vector.h>

// TODO: deal.II libraries: did we actually need these?
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>


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
        VectorFunction(const std::function<Point<dim>(const Point<dim> &, size_t)> &_lambda) : Function<dim>(), lambda(_lambda)
        {
        }

    private:
        /**
         * \brief The closure that will be used to compute the value of the function.
         */
        std::function<Point<dim>(const Point<dim> &, size_t)> lambda;

        virtual double value(const Point<dim> &p, const unsigned int component) const override
        {
            return lambda(p, component);
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
        // The second "dim" is needed in case the spatial dimension
        // is different than the FE dimension
        parallel::distributed::Triangulation<dim, dim> triangulation;
#else
        Triangulation<dim> triangulation;
#endif

        const FE_Q<dim> fe;
        DoFHandler<dim, dim> dof_handler;

        const MappingQ1<dim, dim> mapping;

        AffineConstraints<double> constraints;
        
        // TODO: again, the number of quadrature points in 1D
        // is known at runtime (part of the problem object), however
        // the laplace operator type needs it a compile time

        // TODO: last 4 arguments were set considering deal.II documentation; shall it be our implementation choice?
        using SystemMatrixType = MatrixFreeOperators::LaplaceOperator<dim, fe_degree>;
        SystemMatrixType system_matrix;

        MGConstrainedDoFs mg_constrained_dofs;

        using LevelMatrixType = MatrixFreeOperators::LaplaceOperator<dim, fe_degree>;
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
        parallel::fullydistributed::Triangulation<dim, dim> mesh;

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
