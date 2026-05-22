#include <exception>
#include <stdexcept>
#include <functional>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include "function_types.hpp"
#include "ProblemData.hpp"

/**
 * \brief Namespace containing all the methods and type definitions used in the project.
 */
namespace MFSolver
{
    using namespace dealii;

    namespace LA
    {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
        using namespace LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
        using namespace LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
    } // namespace LA

    /**
     * \brief Like a vector, but distributed.
     * \tparam T The type of elements stored in the vector.
     */
    template <typename T>
    using DVector = LinearAlgebra::distributed::Vector<T>;

    /**
     * \brief Represents a range of cells.
     */
    using Range = std::pair<unsigned int, unsigned int>;

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
        ADRSolver(const ADR::ProblemData<dim, fe_degree> &_problem)
            : problem(_problem)
        {
        }

        /**
         * \brief Destructor for the solver.
         */
        virtual ~ADRSolver() {};

        /**
         * \brief Actually solve the problem.
         */
        virtual void run() = 0;

        // /**
        //  * \brief writes the result of the computation in a structured manner on file.
        //  */
        virtual void output_to_file() = 0;

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
         * \brief writes the result of the computation on terminal.
         */
        virtual void output_results() = 0;

        /**
         * \brief The problem this solver will solve.
         */
        ADR::ProblemData<dim, fe_degree> problem;

        unsigned int timestep_number = 0;

        /**
         * Attributes needed for logging info.
         */
        double start_time = 0;
        double end_time = 0;

        std::vector<std::vector<double>> conv_history;
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
    class ADROperator : public MatrixFreeOperators::Base<dim, DVector<Number>>
    {
    public:
        /**
         * \brief Constructs a new instance of ADROperator.
         */
        ADROperator() : Super(), delta_t(0.0)
        {
        }

        /**
         * \brief Sets the time step size for transient problems.
         */
        void set_time_step(double dt)
        {
            delta_t = dt;
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
                Assert(inverse_diagonal.local_element(i) > 0., ExcMessage(std::format("Error: non-positive entry found. Operator must be positive definite. ({} = {})", i, inverse_diagonal.local_element(i))));
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

                VectorizedArray<Number> mass_term_coeff = make_vectorized_array<Number>(delta_t > 0.0 ? 1.0 / delta_t : 0.0);

                phi.submit_gradient(mu * gradient_of_u, q);
                phi.submit_value(scalar_product(gradient_of_u, beta) + (div_beta + gamma + mass_term_coeff) * value_of_u, q);
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
         * \brief The time step size. If > 0, the operator shifts from steady-state to time-dependent (adds Mass Matrix component).
         */
        double delta_t;

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
        MatrixFreeADRSolver(const ADR::ProblemData<dim, fe_degree> &_problem, bool _simd_flag)
            : ADRSolver<dim, fe_degree>(_problem)
#ifdef DEAL_II_WITH_P4EST
              ,
              triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices, parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
#else
              ,
              triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
              ,
              fe(fe_degree), dof_handler(triangulation), simd_flag(_simd_flag), mapping(), setup_time(0.0), pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0), time_details(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
        }

        ~MatrixFreeADRSolver() override {};

        void run() override;
        void output_to_file() override;

    private:
        void setup_system() override;
        void assemble() override; // <-- this one assembles the RHS, the LHS initialization was already performed somewhere else
        void solve() override;
        void output_results() override;

#ifdef DEAL_II_WITH_P4EST
        //         // The second "dim" is needed in case the spatial dimension
        //         // is different than the FE dimension
        parallel::distributed::Triangulation<dim, dim> triangulation;
#else
        Triangulation<dim> triangulation;
#endif
        const FE_Q<dim> fe;
        DoFHandler<dim, dim> dof_handler;

        const bool simd_flag;
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
        DVector<double> old_solution;
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
        MatrixBasedADRSolver(const ADR::ProblemData<dim, fe_degree> &_problem)
            : ADRSolver<dim, fe_degree>(_problem),
              mpi_communicator(MPI_COMM_WORLD),
              triangulation(mpi_communicator,
                            typename Triangulation<dim>::MeshSmoothing(
                                Triangulation<dim>::smoothing_on_refinement |
                                Triangulation<dim>::smoothing_on_coarsening),
                            parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
              fe(fe_degree),
              dof_handler(triangulation),
              mapping(),
              pcout(std::cout,
                    (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
              computing_timer(mpi_communicator,
                              pcout,
                              TimerOutput::never,
                              TimerOutput::wall_times)
        {
        }
        ~MatrixBasedADRSolver() override {};

        void run() override;

        void output_to_file() override;

    private:
        void setup_system() override;
        void assemble() override;
        void solve() override;
        void output_results() override;

        MPI_Comm mpi_communicator;

        parallel::distributed::Triangulation<dim> triangulation;

        const FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;
        const MappingQ1<dim, dim> mapping;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        AffineConstraints<double> constraints;

        LA::MPI::SparseMatrix system_matrix;

        // TODO: use the correct vector type
        LA::MPI::Vector completely_distributed_solution;
        LA::MPI::Vector locally_relevant_solution;
        LA::MPI::Vector system_rhs;
        LA::MPI::Vector old_solution;

        // Multigrid
        MGLevelObject<LA::MPI::SparseMatrix> mg_matrices;
        MGConstrainedDoFs mg_constrained_dofs;
        MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;

        ConditionalOStream pcout;
        TimerOutput computing_timer;

        double time = 0.0;
        // const double theta = 1.0;

    };
};

// Including template function implementations
#include "MatrixBasedADRSolver.tpp"
#include "MatrixFreeADRSolver.tpp"