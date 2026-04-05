#pragma once

// TODO: reorder imports in a neat way

#include <exception>
#include <stdexcept>
#include <functional>
#include <string>
#include <unordered_map>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

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
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "function_types.hpp"
#include "ProblemData.hpp"

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
            throw std::logic_error("If you see this, you have not extended this class (RealFunction - value) correctly. See docs fore more information.");
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
        template <typename Number>
        using value_type = typename TensorFunction<1, dim, Number>::value_type;

        /**
         * \brief Constructs a new instance of VectorFunction.
         */
        VectorFunction() : TensorFunction<1, dim, double>()
        {
        }

        virtual value_type<double> value(const Point<dim> &p) const override = 0;

        template <typename Number>
        value_type<Number> value(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunction - value) correctly. See docs fore more information.");
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
        using value_type = typename VectorFunction<dim>::value_type<Number>;

        /**
         * \brief Type alias representing the type of the gradient of this function.
         */
        template <typename Number>
        using gradient_type = typename TensorFunction<1, dim, Number>::gradient_type;

        /**
         * \brief Constructs a new instance of VectorFunctionWithGradient.
         */
        VectorFunctionWithGradient() : VectorFunction<dim>()
        {
        }

        virtual double divergence(const Point<dim> &p) const = 0;

        template <typename Number>
        Number divergence(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunctionWithGradient - divergence) correctly. See docs fore more information.");
        }

        virtual gradient_type<double> gradient(const Point<dim> &p) const override = 0;

        template <typename Number>
        gradient_type<Number> gradient(const Point<dim, Number> &p) const
        {
            throw std::logic_error("If you see this, you have not extended this class (VectorFunctionWithGradient - gradient) correctly. See docs fore more information.");
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
     * \brief Represents a range of cells.
     */
    using Range = std::pair<unsigned int, unsigned int>;

    /**
     * \brief Represents a generic ADR problem.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    struct ADRProblem
    {
        RealFunction<dim> mu;                 /**< Diffusion coefficient. */
        VectorFunctionWithGradient<dim> beta; /**< Advection coefficient. */
        RealFunction<dim> gamma;              /**< Reaction coefficient. */

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
    template <unsigned int dim, unsigned int fe_degree>
    class ADRSolver
    {
    public:
        /**
         * \brief Constructs a new instance of ADRSolver
         * \param _problem The problem this solver will solve.
         */
        ADRSolver(const ADR::ProblemData<dim> &_problem)
            : problem(_problem)
        {
        }

        /**
         * \brief Destructor for ADRProblem.
         */
        virtual ~ADRSolver() {};

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
        virtual void assemble() = 0;

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
        ADR::ProblemData<dim> problem;
    };

    /**
     * \brief Class representing the Advection-Diffusion-Reaction operator.
     * \tparam dim The dimensionality of the space the ADR problem lives in.
     * \tparam fe_degree The degree used in the finite element approximation
     * \tparam Number The data type used to represent coordinates in the space the ADR problem lives in.
     *
     * The ADR operator is built to represent an operator \f( L \f) such that the problem to solve can be expressed as \f[ Lu := -\nabla \cdot (\mu \nabla u) + \nabla \cdot (\beta u) + \gamma u = f \f]
     */
    template <int dim, int fe_degree, typename Number>
    class ADROperator : MatrixFreeOperators::Base<dim, DVector<Number>>
    {
    public:
        /**
         * \brief Constructs a new instance of ADROperator.
         */
        ADROperator() : Super()
        {
        }

        /**
         * \brief Resets the ADROperator.
         */
        void clear() override
        {
            mu_coeff.reinit(0, 0);
            beta_coeff.reinit(0, 0);
            div_beta_coeff.reinit(0, 0);
            gamma_coeff.reinit(0, 0);

            Super::clear();
        }

        /**
         * \brief Precomputes all the coefficients.
         * \param mu_coeff_function Instance of RealFunction representing the diffusion coefficient of the problem to be solved.
         * \param beta_coeff_function Instance of VectorFunctionWithGradient representing the advection coefficient of the problem to be solved.
         * \param gamma_coeff_function Instance of RealFunction representing the reaction coefficient of the problem to be solved.
         *
         * A call to this method is needed in order to have them ready in SIMD vectors (without having to break the SIMD context) when used while solving the associated algebraic system.
         * There is no failsafe implemented that is activated when this method is not called. In case this method is not called before the coefficients are used, the program will most likely crash with a segmentation fault.
         *
         * \note This method assumes that the ADROperator was correctly initialized (see Deal.II tutorial step-37 for reference).
         * If this is not true, this method will just crash with a segmentation fault trying to access non initialized pointers.
         */
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
                    Point<dim, VectorizedArray<Number>> quadrature_point = phi.quadrature_point(q);

                    mu_coeff(cell, q) = mu_coeff_function.value(quadrature_point);
                    beta_coeff(cell, q) = beta_coeff_function.value(quadrature_point);
                    div_beta_coeff(cell, q) = beta_coeff_function.divergence(quadrature_point);
                    gamma_coeff(cell, q) = gamma_coeff_function.value(quadrature_point);
                }
            }
        }

        /**
         * \brief Computes the diagonal of the ADROperator.
         *
         * \note This method assumes that the ADROperator was correctly initialized (see Deal.II tutorial step-37 for reference).
         * If this is not true, this method will just crash with a segmentation fault trying to access non initialized pointers.
         */
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
        /**
         * \brief Type alias used as a shorthand to get to the base class.
         */
        using Super = MatrixFreeOperators::Base<dim, DVector<Number>>;

        /**
         * \brief Type alias used as a shorthand to use FEEvaluation with the correct template parameters
         */
        using Phi = FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>;

        /**
         * \brief Computes the lhs for a given cell.
         * \param phi The FEEvaluation object representing the finite element approximation.
         * \param cell The cell for which to compute the lhs.
         *
         * \note This code is extracted and reused by `local_apply` and `local_compute_diagonal`.
         * According to Step-37, it seems that they are the same but I have not found proof for it (TODO: check).
         * Class methods gets automatically inlined by the compiler, therefore there should not be any performance loss due to the function call.
         */
        void lhs_computation(Phi &phi, const unsigned int cell) const
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

        /**
         * \brief Applies the ADROperator to a range of cells.
         * \param data The MatrixFree object containing all the information needed by the FEEvaluation to evaluate.
         * \param dst The vector of DoFs in which the result of the application is saved.
         * \param src The vector of DoFs to which the operator is applied.
         * \param cell_range Describes the range of cells to which apply the operator.
         */
        void local_apply(const MatrixFree<dim, Number> &data, DVector<Number> &dst, const DVector<Number> &src, const Range &cell_range) const
        {
            Phi phi(data);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
                AssertDimension(mu_coeff.size(0), data.n_cell_batches());
                AssertDimension(mu_coeff.size(1), phi.n_q_points);

                AssertDimension(beta_coeff.size(0), data.n_cell_batches());
                AssertDimension(beta_coeff.size(1), phi.n_q_points);

                AssertDimension(div_beta_coeff.size(0), data.n_cell_batches());
                AssertDimension(div_beta_coeff.size(1), phi.n_q_points);

                AssertDimension(gamma_coeff.size(0), data.n_cell_batches());
                AssertDimension(gamma_coeff.size(1), phi.n_q_points);

                phi.reinit(cell);
                phi.read_dof_values(src);

                lhs_computation(phi, cell);

                phi.distribute_local_to_global(dst);
            }
        }

        /**
         * \brief Used as cell operation to compute the diagonal of the operator.
         * \param phi The FEEvaluation to be used in the computation.
         */
        void local_compute_diagonal(Phi &phi) const
        {
            const unsigned int cell = phi.get_current_cell_index();
            lhs_computation(phi, cell);
        }

        /**
         * \brief Applies the operator to a given vector of DoFs.
         * \param dst The vector of DoFs in which the result of the application is saved.
         * \param src The vector of DoFs to which the operator is applied.
         */
        virtual void apply_add(DVector<Number> &dst, const DVector<Number> &src) const override
        {
            this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
        }

        /**
         * \brief Cache used to store precomputed values for the diffusion coefficient.
         */
        Table<2, VectorizedArray<Number>> mu_coeff;

        /**
         * \brief Cache used to store precomputed values for the advection coefficient.
         */
        Table<2, Tensor<1, dim, VectorizedArray<Number>>> beta_coeff; // Is this OK?

        /**
         * \brief Cache used to store precomputed values for the divergence of the advection coefficient.
         *
         * Here, the divergence of beta is treated exactly like a coefficient in order to have it precomputed when needed in order to not break the SIMD contexts in which they are used (that would cause significand slowdowns).
         */
        Table<2, VectorizedArray<Number>> div_beta_coeff;

        /**
         * \brief Cache used to store precomputed values for the reaction coefficient.
         */
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
        MatrixFreeADRSolver(const ADR::ProblemData<dim> &_problem) : ADRSolver<dim, fe_degree>(_problem)
        {
        }
        ~MatrixFreeADRSolver() override {};

        void run() override;

    private:
        void setup_system() override;
        void assemble() override;
        void solve() override;
        void output_results() override;

#ifdef DEAL_II_WITH_P4EST
        //         // The second "dim" is needed in case the spatial dimension
        //         // is different than the FE dimension
        parallel::distributed::Triangulation<dim, dim> triangulation();
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
        MatrixBasedADRSolver(const ADRProblem<dim> &_problem) : ADRSolver<dim, fe_degree>(_problem),
                                                                mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
                                                                mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
                                                                mesh(MPI_COMM_WORLD),
                                                                pcout(std::cout, mpi_rank == 0)
        {
        }
        ~MatrixBasedADRSolver() override {};

        void run() override;

    private:
        void setup_system() override;
        void assemble() override;
        void solve() override;
        void output_results() override;

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

// Including template function implementations
#include "adr_matrix_based/MatrixBasedADRSolver.tpp"