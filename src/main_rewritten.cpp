#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <fstream>

namespace MatrixFreeSolver
{
    using namespace dealii;

    const unsigned int degree_finite_element = 2;
    const unsigned int dimension = 3;

    // This is one of the coefficients of the equation, we should probably add more in order to solve a full ADR problem
    template <int dim>
    class Coefficient : public Function<dim>
    {
    public:
        template <typename Number>
        Number value(const Point<dim, Number> &p, const unsigned int = 0) const
        {
            return 1. / (0.05 + 2. * p.square());
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            return value<double>(p, component);
        }
    };

    template <typename Number>
    using DVector = LinearAlgebra::distributed::Vector<Number>;

    // This already exists in matrix_free/operators.h, should we check if all the other operators used in the ADR already exist? Spiler: they don't
    template <int dim, int fe_degree, typename Number>
    class LaplaceOperator : public MatrixFreeOperators::Base<dim, DVector<Number>>
    {
    public:
        using value_type = Number;
        using Super = MatrixFreeOperators::Base<dim, DVector<Number>>;      // Base per tutti gli operatori MatrixFree.
        using Phi = FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>; // Tipo FEValues, serve per calcolare i valori delle funzioni sui punti di quadratura.

        LaplaceOperator() : Super()
        {
        }

        void clear() override
        {
            coefficient.reinit(0, 0); // Tabella 0x0
            Super::clear();           // Svuota tutto
        }

        // Rende `coefficient` una tabella (numero celle)x(punti di quadratura per cella) che contiene i valori del coefficiente di cui sopra per ogni punto di quadratura
        void evaluate_coefficient(const Coefficient<dim> &coefficient_function)
        {
            const unsigned int n_cells = this->data->n_cell_batches(); // 1 batch = un gruppo di celle, raggruppate come conto vettorizzato
            Phi phi(*this->data);

            coefficient.reinit(n_cells, phi.n_q_points); // la tabella diventa (numero celle)x(punti di quadratura per cella)

            for (unsigned int cell = 0; cell < n_cells; ++cell)
            {
                phi.reinit(cell);
                for (const unsigned int q : phi.quadrature_point_indices())
                {
                    coefficient(cell, q) = coefficient_function.value(phi.quadrature_point(q));
                }
            }
        }

        // this->data è un MatrixFree

        // Calcola e memorizza la diagonale inversa dell'operatore
        virtual void compute_diagonal() override
        {
            this->inverse_diagonal_entries.reset(new DiagonalMatrix<DVector<Number>>());
            DVector<Number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
            this->data->initialize_dof_vector(inverse_diagonal); // Alloca abbastanza memoria per contenere tutti i DoF di qualcosa, la diagonale presumo ma non riesco a capire. Dove caspita viene passato il DoFHandler???

            MatrixFreeTools::compute_diagonal(*this->data, inverse_diagonal, &LaplaceOperator::local_compute_diagonal, this); // Calcola la matrice diagonale (il nome è fuorviante, viene poi invertito sotto, credo) dell'operatore Laplace utilizzando "local_compute_diagonal"

            // MEtte a 1 tutti i nodi soggetti a vincoli, probabilmente per facilitare l'inversione della matrice, vedere la doc di compute_diagonal
            this->set_constrained_entries_to_one(inverse_diagonal);

            // Fino a prima di questo loop, era la diagonale, ora verrà invertita per ottenere la diagonale inversa
            // Itera solo sugli elementi che ti appartengono
            for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
            {
                Assert(inverse_diagonal.local_element(i) > 0., ExcMessage("No diagonal entry in a positive definite operator should be zero"));
                inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
            }
        }

    private:
        // dst += Op * src ??? In pratica chiama "local_apply" su tutte le celle in parallelo
        virtual void apply_add(DVector<Number> &dst, const DVector<Number> &src) const override
        {
            this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
        }

        // Applica qualcosa ad un range di celle, realisticamente "locali"
        void local_apply(const MatrixFree<dim, Number> &data, DVector<Number> &dst, const DVector<Number> &src, const std::pair<unsigned int, unsigned int> &cell_range) const
        {
            Phi phi(data);

            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
                AssertDimension(coefficient.size(0), data.n_cell_batches());
                AssertDimension(coefficient.size(1), phi.n_q_points);

                phi.reinit(cell);
                phi.read_dof_values(src);
                phi.evaluate(EvaluationFlags::gradients);

                for (const unsigned int q : phi.quadrature_point_indices())
                {
                    phi.submit_gradient(coefficient(cell, q) * phi.get_gradient(q), q);
                }

                phi.integrate(EvaluationFlags::gradients);
                phi.distribute_local_to_global(dst);
            }
        }

        // Ma questa calcola la weak form o no???
        void local_compute_diagonal(Phi &phi) const
        {
            const unsigned int cell = phi.get_current_cell_index();

            phi.evaluate(EvaluationFlags::gradients);

            for (const unsigned int q : phi.quadrature_point_indices())
            {
                phi.submit_gradient(coefficient(cell, q) * phi.get_gradient(q), q);
            }
            phi.integrate(EvaluationFlags::gradients);
        }

        // Tabella bidimensionale di VectorizedArrays
        Table<2, VectorizedArray<Number>> coefficient;
    };

    template <int dim>
    class LaplaceProblem
    {
    public:
        LaplaceProblem()
#ifdef DEAL_II_WITH_P4EST
            : triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices, parallel::distributed::Triangulation<dim>::constructmultigrid_hierarchy),
#else
            : triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
#endif
              fe(degree_finite_element),
              dof_handler(triangulation),
              setup_time(0.),
              pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
              time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
        }

        void run()
        {
            {
                {
                    pcout << "Number of MPI ranks:            "
                          << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
                    pcout << "Number of threads on each rank: "
                          << MultithreadInfo::n_threads() << std::endl;
                    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
                    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
                    pcout << "Vectorization over " << n_vect_doubles
                          << " doubles = " << n_vect_bits << " bits ("
                          << Utilities::System::get_current_vectorization_level() << ')'
                          << std::endl
                          << std::endl;
                }

                const unsigned int n_vect_doubles = VectorizedArray<double>::size();
                const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

                pcout << "Vectorization over " << n_vect_doubles
                      << " doubles = " << n_vect_bits << " bits ("
                      << Utilities::System::get_current_vectorization_level() << ')'
                      << std::endl;
            }

            // Do we solve an increasingly less coarse system? Multigrid?
            for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
            {
                pcout << "Cycle " << cycle << std::endl;

                if (cycle == 0)
                {
                    GridGenerator::hyper_cube(triangulation, 0., 1.);
                    triangulation.refine_global(3 - dim); // Why do we do that? Is this just to have some material to work on at the first iterations?
                }

                triangulation.refine_global(1);
                setup_system();
                assemble_rhs();
                solve();
                output_results(cycle);
                pcout << std::endl;
            }
        }

    private:
        void setup_system()
        {
            Timer time;
            setup_time = 0;

            {
                system_matrix.clear();
                mg_matrices.clear_elements();

                dof_handler.distribute_dofs(fe);
                dof_handler.distribute_mg_dofs();

                pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

                constraints.clear();
                constraints.reinit(dof_handler.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dof_handler));
                DoFTools::make_hanging_node_constraints(dof_handler, constraints);

                VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints); // Qui stiamo andando ad imporre cosa, di preciso? Dirichlet? BCs a zero ovunque???
                constraints.close();
            }

            setup_time += time.wall_time();
            time_details << "Distribute DoFs and B.Cs. (cpu/wall): " << time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;
            time.restart();

            {
                { // Initialize the system matrix with the correct settings
                    typename MatrixFree<dim, double>::AdditionalData additional_data;
                    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_color;
                    additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;

                    std::shared_ptr<MatrixFree<dim, double>>
                        system_mf_storage(new MatrixFree<dim, double>());
                    system_mf_storage->reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);
                    system_matrix.initialize(system_mf_storage);
                }

                system_matrix.evaluate_coefficient(Coefficient<dim>());
                system_matrix.initialize_dof_vector(solution);
                system_matrix.initialize_dof_vector(system_rhs);
            }

            setup_time += time.wall_time();
            time_details << "Setup mf system (cpu/wall): " << time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;
            time.restart();

            { // Initialize all multigrid matrices with correct data
                const unsigned int nlevels = triangulation.n_global_levels();
                mg_matrices.resize(0, nlevels - 1);

                const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
                mg_constrained_dofs.initialize(dof_handler);
                mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

                for (unsigned int level = 0; level < nlevels; ++level)
                {
                    AffineConstraints<double> level_constraints(dof_handler.locally_owned_mg_dofs(level), DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

                    for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
                    {
                        level_constraints.constrain_dof_to_zero(dof_index);
                    }

                    level_constraints.close();

                    typename MatrixFree<dim, float>::AdditionalData additional_data;
                    additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::TasksParallelScheme::partition_color;
                    additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
                    additional_data.mg_level = level;
                    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level = std::make_shared<MatrixFree<dim, float>>();
                    mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);

                    mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
                    mg_matrices[level].evaluate_coefficient(Coefficient<dim>());
                }
            }

            setup_time += time.wall_time();
            time_details << "Setup mf levels (cpu/wall): " << time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;
        }

        void assemble_rhs()
        {
            Timer time;

            system_rhs = 0;

            FEEvaluation<dim, degree_finite_element> phi(*system_matrix.get_matrix_free());

            for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell)
            {
                phi.reinit(cell);

                for (const unsigned int q : phi.quadrature_point_indices())
                {
                    phi.submit_value(make_vectorized_array<double>(1.0), q);
                }

                phi.integrate(EvaluationFlags::values);
                phi.distribute_local_to_global(system_rhs);
            }

            system_rhs.compress(VectorOperation::add);

            setup_time += time.wall_time();
            time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
        }

        void solve()
        {
            Timer time;
            // Why do we use float instead of double???
            MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
            mg_transfer.build(dof_handler);

            setup_time += time.wall_time();
            time_details << "MG build transfer time     (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
            time.restart();

            using SmootherType = PreconditionChebyshev<LevelMatrixType, DVector<float>>;
            mg::SmootherRelaxation<SmootherType, DVector<float>> mg_smoother;

            MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
            smoother_data.resize(0, triangulation.n_global_levels() - 1);

            for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            {
                if (level > 0)
                {
                    smoother_data[level].smoothing_range = 15.;
                    smoother_data[level].degree = 5;
                    smoother_data[level].eig_cg_n_iterations = 10;
                }
                else
                {
                    smoother_data[0].smoothing_range = 1e-3;
                    smoother_data[0].degree = numbers::invalid_unsigned_int;
                    smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
                }

                mg_matrices[level].compute_diagonal();
                smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
            }

            mg_smoother.initialize(mg_matrices, smoother_data);

            MGCoarseGridApplySmoother<DVector<float>> mg_coarse;
            mg_coarse.initialize(mg_smoother);

            mg::Matrix<DVector<float>> mg_matrix(mg_matrices);

            MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
            mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
            for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            {
                mg_interface_matrices[level].initialize(mg_matrices[level]);
            }

            mg::Matrix<DVector<float>> mg_interface(mg_interface_matrices);

            Multigrid<DVector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
            mg.set_edge_matrices(mg_interface, mg_interface);

            PreconditionMG<dim, DVector<float>, MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);

            SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
            SolverCG<DVector<double>> cg(solver_control);

            setup_time += time.wall_time();
            time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time()
                         << "s/" << time.wall_time() << "s\n";
            pcout << "Total setup time               (wall) " << setup_time << "s\n";

            time.reset();
            time.start();
            constraints.set_zero(solution);
            cg.solve(system_matrix, solution, system_rhs, preconditioner);

            constraints.distribute(solution);

            pcout << "Time solve (" << solver_control.last_step() << " iterations)"
                  << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) "
                  << time.cpu_time() << "s/" << time.wall_time() << "s\n";
        }

        void output_results(const unsigned int cycle) const
        {
            Timer time;
            if (triangulation.n_global_active_cells() > 1000000)
                return;

            DataOut<dim> data_out;

            solution.update_ghost_values();
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches(mapping);

            DataOutBase::VtkFlags flags;
            flags.compression_level = DataOutBase::CompressionLevel::best_speed;
            data_out.set_flags(flags);
            data_out.write_vtu_with_pvtu_record(
                "./", "solution", cycle, MPI_COMM_WORLD, 3);

            time_details << "Time write output          (CPU/wall) " << time.cpu_time()
                         << "s/" << time.wall_time() << "s\n";
        }

#ifdef DEAL_II_WITH_P4EST
        parallel::distributed::Triangulation<dim> triangulation;
#else
        Triangulation<dim> triangulation;
#endif
        // Split space in cubic cells
        const FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        // Mappa da una reference cell a QUALCOSA? MA PERCHé?
        const MappingQ1<dim> mapping;

        // WHAT
        AffineConstraints<double> constraints;

        using SystemMatrixType = LaplaceOperator<dim, degree_finite_element, double>; // In che senso usiamo un operatore come matrice? Sono stupido
        SystemMatrixType system_matrix;

        // WHAT pt.2
        MGConstrainedDoFs mg_constrained_dofs;

        using LevelMatrixType = LaplaceOperator<dim, degree_finite_element, float>; // Should we implement the ADR operator??? Why do we use float here???
        MGLevelObject<LevelMatrixType> mg_matrices;

        DVector<double> solution;
        DVector<double> system_rhs;

        double setup_time;
        ConditionalOStream pcout;
        ConditionalOStream time_details;
    };
};

int main(int argc, char *argv[])
{
    try
    {
        using namespace MatrixFreeSolver;

        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, numbers::invalid_unsigned_int);

        LaplaceProblem<dimension> laplace_problem;
        laplace_problem.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}