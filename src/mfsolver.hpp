/**
 * @file mfsolver.hpp
 * @brief Core ADR solver interfaces and solver class declarations.
 *
 * @details Declares the shared solver base class, the matrix-free ADR operator,
 * and the matrix-based and matrix-free solver front ends. Template
 * implementation files are included at the end of this header.
 */

#include <exception>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <functional>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <iomanip>

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
#include <deal.II/base/work_stream.h>

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
#include <deal.II/lac/precondition.h>
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
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
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
 * @brief Namespace containing all solver types and helper aliases used by the project.
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
     * @brief Distributed vector alias used by the solver implementations.
     * @tparam T Element type stored in the vector.
     */
    template <typename T>
    using DVector = LinearAlgebra::distributed::Vector<T>;

    /**
     * @brief Represents a range of cells
     */
    using Range = std::pair<unsigned int, unsigned int>;

    /**
     * @brief Common interface shared by the matrix-free and matrix-based solvers.
     * @tparam dim Spatial dimension of the ADR problem.
     */
    template <unsigned int dim>
    class ADRSolver
    {
    public:
        /**
         * @brief Construct a solver for one benchmark problem definition.
         * @param _problem Problem data copied into this solver instance.
         */
        ADRSolver(const ADR::ProblemData<dim> &_problem)
            : problem(_problem)
        {
        }

        /**
         * @brief Virtual destructor for derived solver implementations.
         */
        virtual ~ADRSolver() {};

        /**
         * @brief Execute the full setup, solve, output, and logging workflow.
         */
        virtual void run() = 0;

        /**
         * @brief Write the collected run data in the project log format.
         */
        virtual void output_to_file() = 0;

    protected:
        /**
         * @brief Set up the algebraic system corresponding to the problem.
         */
        virtual void setup_system() = 0;

        /**
         * @brief Assemble the algebraic right-hand side.
         */
        virtual void assemble() = 0;

        /**
         * @brief Solve the assembled algebraic system.
         */
        virtual void solve() = 0;

        /**
         * @brief Write solution output and terminal status information.
         */
        virtual void output_results() = 0;

        /**
         * @brief Problem definition solved by this object.
         */
        ADR::ProblemData<dim> problem;

        unsigned int timestep_number = 0;

        /**
         * @brief Timing and convergence data collected during one solver run.
         *
         * The same solver object writes both the visualization files and the
         * final log.txt summary. Keeping the selected output directory here
         * makes that relationship explicit: all artifacts produced by one run
         * must land in the same test_N folder, even when several MPI ranks are
         * contributing different VTU pieces.
         */
        double start_time = 0;
        double end_time = 0;

        std::vector<std::vector<double>> conv_history;
        std::vector<double> solver_tolerances;

        /**
         * @brief Directory reserved for this run.
         *
         * It is intentionally solver state, not recomputed from the filesystem
         * later.
         */
        std::string output_dir;
    };

    /**
     * @brief Matrix-free Advection-Diffusion-Reaction operator.
     * @tparam dim Spatial dimension of the ADR problem.
     * @tparam Number Scalar type used by the matrix-free evaluator.
     *
     * The ADR operator is built to represent an operator \f( L \f) such that the problem to solve can be expressed as \f[ Lu := -\nabla \cdot (\mu \nabla u) + \nabla \cdot (\beta u) + \gamma u = f \f]
     */
    template <int dim, typename Number>
    class ADROperator : public MatrixFreeOperators::Base<dim, DVector<Number>>
    {
    public:
        /**
         * @brief Construct an empty ADR operator.
         */
        ADROperator() : Super(), delta_t(0.0)
        {
        }

        /**
         * @brief Set the time step size for transient problems.
         * @param dt Time-step size; zero selects the steady-state operator.
         */
        void set_time_step(double dt)
        {
            delta_t = dt;
        }

        /**
         * @brief Reset coefficient pointers and base matrix-free state.
         */
        void clear() override
        {
            mu_func = nullptr;
            beta_func = nullptr;
            gamma_func = nullptr;

            Super::clear();
        }

        /**
         * @brief Store the coefficient functions used by operator applications.
         * @param mu_coeff_function Diffusion coefficient function.
         * @param beta_coeff_function Advection coefficient function with divergence.
         * @param gamma_coeff_function Reaction coefficient function.
         *
         * A call to this method is needed in order to have them ready in SIMD vectors (without having to break the SIMD context) when used while solving the associated algebraic system.
         * There is no failsafe implemented that is activated when this method is not called. In case this method is not called before the coefficients are used, the program will most likely crash with a segmentation fault.
         *
         * @note This method assumes that the ADROperator was correctly initialized (see deal.II tutorial step-37 for reference).
         * If this is not true, this method will just crash with a segmentation fault trying to access non initialized pointers.
         */
        void evaluate_coefficients(
            const RealFunction<dim> &mu_coeff_function,
            const VectorFunctionWithGradient<dim> &beta_coeff_function,
            const RealFunction<dim> &gamma_coeff_function)
        {
            mu_func = &mu_coeff_function;
            beta_func = &beta_coeff_function;
            gamma_func = &gamma_coeff_function;
        }

        /**
         * @todo Is this correct? Doesn't it compute the inverse diagonal?
         * @brief Compute the diagonal of the ADR operator.
         *
         * @note This method assumes that the ADROperator was correctly initialized (see deal.II tutorial step-37 for reference).
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
         * @brief Base class alias for the matrix-free operator implementation.
         */
        using Super = MatrixFreeOperators::Base<dim, DVector<Number>>;

        /**
         * @brief FEEvaluation alias using runtime element degree and quadrature size.
         *
         * The FE degree is controlled at runtime through FE_Q(problem.fe_degree).
         * deal.II supports FEEvaluation<dim, -1, 0, ...> for this mode: -1
         * tells the evaluator to read the element degree from MatrixFree, and
         * 0 does the same for the 1D quadrature size.
         */
        using Phi = FEEvaluation<dim, -1, 0, 1, Number>;

        /**
         * @brief Computes the lhs for a given cell.
         * @param phi The FEEvaluation object representing the finite element approximation.
         * @param cell The cell for which to compute the lhs.
         *
         * This code is extracted and reused by `local_apply` and
         * `local_compute_diagonal`.
         * @todo Confirm that the same local action is appropriate for diagonal
         * extraction in all ADR configurations.
         * Class methods gets automatically inlined by the compiler, therefore there should not be any performance loss due to the function call.
         */
        void lhs_computation(Phi &phi, const unsigned int cell) const
        {
            (void)cell; ///< Cell index no longer needed since coefficients are evaluated on the fly.
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            for (const unsigned int q : phi.quadrature_point_indices())
            {
                Tensor<1, dim, VectorizedArray<Number>> gradient_of_u = phi.get_gradient(q);
                VectorizedArray<Number> value_of_u = phi.get_value(q);

                Point<dim, VectorizedArray<Number>> quadrature_point = phi.quadrature_point(q);

                VectorizedArray<Number> mu = mu_func->value(quadrature_point);
                Tensor<1, dim, VectorizedArray<Number>> beta = beta_func->value(quadrature_point);
                VectorizedArray<Number> div_beta = beta_func->divergence(quadrature_point);
                VectorizedArray<Number> gamma = gamma_func->value(quadrature_point);

                VectorizedArray<Number> mass_term_coeff = make_vectorized_array<Number>(delta_t > 0.0 ? 1.0 / delta_t : 0.0);

                phi.submit_gradient(mu * gradient_of_u, q);
                phi.submit_value(scalar_product(gradient_of_u, beta) + (div_beta + gamma + mass_term_coeff) * value_of_u, q);
            }

            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        }

        /**
         * @brief the ADROperator to a range of cells.
         * @param data The MatrixFree object containing all the information needed by the FEEvaluation to evaluate.
         * @param dst The vector of DoFs in which the result of the application is saved.
         * @param src The vector of DoFs to which the operator is applied.
         * @param cell_range Describes the range of cells to which apply the operator.
         */
        void local_apply(const MatrixFree<dim, Number> &data, DVector<Number> &dst, const DVector<Number> &src, const Range &cell_range) const
        {
            Phi phi(data);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
                Assert(mu_func != nullptr, ExcMessage("mu coefficient function was not set."));
                Assert(beta_func != nullptr, ExcMessage("beta coefficient function was not set."));
                Assert(gamma_func != nullptr, ExcMessage("gamma coefficient function was not set."));

                phi.reinit(cell);
                phi.read_dof_values(src);

                lhs_computation(phi, cell);

                phi.distribute_local_to_global(dst);
            }
        }

        /**
         * @brief Cell operation used to compute diagonal entries.
         * @param phi FEEvaluation object positioned on the current cell.
         */
        void local_compute_diagonal(Phi &phi) const
        {
            const unsigned int cell = phi.get_current_cell_index();
            lhs_computation(phi, cell);
        }

        /**
         * @brief Applies the operator to a given vector of DoFs.
         * @param dst The vector of DoFs in which the result of the application is saved.
         * @param src The vector of DoFs to which the operator is applied.
         */
        virtual void apply_add(DVector<Number> &dst, const DVector<Number> &src) const override
        {
            this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
        }

        /**
         * @brief Time-step size; positive values add the transient mass term.
         */
        double delta_t;

        /**
         * @brief Pointer to the diffusion coefficient function.
         */
        const RealFunction<dim> *mu_func = nullptr;

        /**
         * @brief Pointer to the advection coefficient function.
         */
        const VectorFunctionWithGradient<dim> *beta_func = nullptr;

        /**
         * @brief Pointer to the reaction coefficient function.
         */
        const RealFunction<dim> *gamma_func = nullptr;
    };

    /**
     * @brief ADR solver implementation based on matrix-free operator application.
     * @tparam dim Spatial dimension of the ADR problem.
     */
    template <int dim>
    class MatrixFreeADRSolver : public ADRSolver<dim>
    {
    public:
        MatrixFreeADRSolver(const ADR::ProblemData<dim> &_problem, bool _simd_flag)
            : ADRSolver<dim>(_problem)
#ifdef DEAL_II_WITH_P4EST
              ,
              triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices, parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
#else
              ,
              triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
              ,
              fe(_problem.fe_degree), dof_handler(triangulation), simd_flag(_simd_flag), mapping(), setup_time(0.0), pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0), time_details(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
        }

        ~MatrixFreeADRSolver() override {};

        void run() override;
        void output_to_file() override;

    private:
        void setup_system() override;
        void assemble() override; ///< Assemble the RHS; LHS initialization is handled TODO: where?.
        void solve() override;
        void output_results() override;
        void compute_error();
        void local_assemble_cell(const MatrixFree<dim, double> &data,
                                 DVector<double> &dst,
                                 const DVector<double> &src,
                                 const std::pair<unsigned int, unsigned int> &cell_range) const;

#ifdef DEAL_II_WITH_P4EST
        /// Distributed triangulation with explicit spacedim equal to dim.
        parallel::distributed::Triangulation<dim, dim> triangulation;
#else
        Triangulation<dim> triangulation;
#endif
        const FE_Q<dim> fe;
        DoFHandler<dim, dim> dof_handler;

        const bool simd_flag;
        const MappingQ1<dim, dim> mapping;

        AffineConstraints<double> constraints;

        /// The matrix-free operator reads degree and quadrature size from the
        /// MatrixFree object at runtime through FEEvaluation<dim, -1, 0, ...>.
        using SystemMatrixType = ADROperator<dim, double>;
        SystemMatrixType system_matrix;

        MGConstrainedDoFs mg_constrained_dofs;

        using LevelMatrixType = ADROperator<dim, float>;
        MGLevelObject<LevelMatrixType> mg_matrices;

        DVector<double> solution;
        DVector<double> old_solution;
        DVector<double> system_rhs;

        double setup_time;
        double l2_error = 0.0;
        double h1_error = 0.0;
        double linfty_error = 0.0;
        bool converged = false;
        ConditionalOStream pcout;
        ConditionalOStream time_details;

        bool mf_setup_initialized = false;
        std::shared_ptr<MatrixFree<dim, double>> inhomogeneous_mf_storage;
        std::shared_ptr<ADROperator<dim, double>> inhomogeneous_operator;

        std::shared_ptr<MGTransferMatrixFree<dim, float>> mg_transfer;
        using SmootherType = PreconditionChebyshev<LevelMatrixType, DVector<float>>;
        std::shared_ptr<mg::SmootherRelaxation<SmootherType, DVector<float>>> mg_smoother;
        std::shared_ptr<MGCoarseGridApplySmoother<DVector<float>>> mg_coarse;
        std::shared_ptr<mg::Matrix<DVector<float>>> mg_matrix;
        std::shared_ptr<MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>> mg_interface_matrices;
        std::shared_ptr<mg::Matrix<DVector<float>>> mg_interface;
        std::shared_ptr<Multigrid<DVector<float>>> mg;
        std::shared_ptr<PreconditionMG<dim, DVector<float>, MGTransferMatrixFree<dim, float>>> preconditioner;
    };

    /**
     * @brief Per-cell local matrix and vector data copied into global objects.
     * @tparam dim Spatial dimension of the finite element cell.
     */
    template <int dim>
    struct PerTaskData {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> dof_indices;
        unsigned int cell_level = numbers::invalid_unsigned_int;
        bool cell_is_locally_owned = false;
        bool assemble_matrix = true;

        PerTaskData (const FiniteElement<dim> &fe)
                    :
                    cell_matrix (fe.dofs_per_cell, fe.dofs_per_cell),
                    cell_rhs (fe.dofs_per_cell),
                    dof_indices (fe.dofs_per_cell)
            {}

        /**
         * @brief Copy constructor used by WorkStream worker-local data pools.
         *
         * WorkStream keeps a pool of CopyData objects and constructs that pool
         * by copying this sample object. Copy only the allocated shape, not the
         * transient values from a previous cell. The worker resets the fields
         * again before every use, but keeping the copy constructor explicit
         * documents the required contract: each copy is an independent local
         * matrix/vector buffer that can be filled by one worker thread and then
         * consumed by the sequential copier.
         */
        PerTaskData(const PerTaskData &data)
            :
            cell_matrix(data.cell_matrix.m(), data.cell_matrix.n()),
            cell_rhs(data.cell_rhs.size()),
            dof_indices(data.dof_indices.size()),
            cell_level(numbers::invalid_unsigned_int),
            cell_is_locally_owned(false),
            assemble_matrix(data.assemble_matrix)
            {}
    };

    /**
     * @brief Per-thread scratch objects used during matrix-based assembly.
     * @tparam dim Spatial dimension of the finite element cell.
     */
    template <int dim>
    struct ScratchData {
        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_face_values;
        std::vector<double> old_solution_values;
        
        ScratchData (const FiniteElement<dim> &fe,
                    const Quadrature<dim>    &quadrature,
                    const Quadrature<dim - 1> &face_quadrature,
                    const UpdateFlags         update_flags,
                    const UpdateFlags         face_update_flags)
                    :
                    fe_values (fe, quadrature, update_flags),
                    fe_face_values(fe, face_quadrature, face_update_flags),
                    old_solution_values(quadrature.size())
            {}
        
        /**
         * @brief Copy constructor that creates private evaluator caches.
         *
         * FEValues and FEFaceValues own mutable caches that are changed by
         * reinit(). Sharing them across threads would be a data race, so a
         * ScratchData copy must construct fresh evaluator objects using the
         * same finite element, quadrature, and update flags as the sample.
         * WorkStream then gives each worker one of these private scratch
         * objects.
         */
        ScratchData (const ScratchData &scratch)
                    :
                    fe_values (scratch.fe_values.get_fe(),
                                scratch.fe_values.get_quadrature(),
                                scratch.fe_values.get_update_flags()),
                    fe_face_values(scratch.fe_face_values.get_fe(),
                                   scratch.fe_face_values.get_quadrature(),
                                   scratch.fe_face_values.get_update_flags()),
                    old_solution_values(scratch.old_solution_values.size())
            {}
    };

    /**
     * @brief ADR solver implementation based on an assembled sparse matrix.
     * @tparam dim Spatial dimension of the ADR problem.
     */
    template <int dim>
    class MatrixBasedADRSolver : public ADRSolver<dim>
    {
    public:
        MatrixBasedADRSolver(const ADR::ProblemData<dim> &_problem)
            : ADRSolver<dim>(_problem),
              mpi_communicator(MPI_COMM_WORLD),
              triangulation(mpi_communicator,
                            typename Triangulation<dim>::MeshSmoothing(
                                Triangulation<dim>::smoothing_on_refinement |
                                Triangulation<dim>::smoothing_on_coarsening)),
              fe(_problem.fe_degree),
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
        void assemble_on_one_cell(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            ScratchData<dim> &scratch,
            PerTaskData<dim> &data);
        void copy_local_to_global(const PerTaskData<dim> &data);
        void assemble() override;
        void solve() override;
        void output_results() override;
        void compute_error();

        MPI_Comm mpi_communicator;

        parallel::distributed::Triangulation<dim> triangulation;

        const FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;
        const MappingQ1<dim, dim> mapping;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        AffineConstraints<double> constraints;

        LA::MPI::SparseMatrix system_matrix;

        LA::MPI::Vector completely_distributed_solution;
        LA::MPI::Vector locally_relevant_solution;
        LA::MPI::Vector system_rhs;
        LA::MPI::Vector old_solution;

        ConditionalOStream pcout;
        TimerOutput computing_timer;

        /**
         * @brief Serializes the narrow old-solution read section.
         *
         * PETSc MPI vectors are not a safe object to sample concurrently from
         * several WorkStream workers. Transient RHS assembly reads
         * old_solution through FEValues::get_function_values(), so serialize
         * that narrow read section while leaving the quadrature work parallel.
         */
        std::mutex old_solution_read_mutex;

        double time = 0.0;
        double l2_error = 0.0;
        double h1_error = 0.0;
        double linfty_error = 0.0;
        bool converged = false;

        bool assemble_matrix_flag = true;
        std::shared_ptr<LA::MPI::PreconditionAMG> preconditioner_amg;
    };
};

/// @brief Include template function implementations for the declarations above.
#include "SolverOutputUtilities.tpp"
#include "MatrixBasedADRSolver.tpp"
#include "MatrixFreeADRSolver.tpp"
