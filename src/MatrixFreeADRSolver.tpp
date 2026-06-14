#pragma once

/**
 * @file MatrixFreeADRSolver.tpp
 * @brief Template implementations for MatrixFreeADRSolver.
 *
 * @details Contains setup, matrix-free assembly, multigrid preconditioner
 * construction, solve, output, and error-computation routines for the
 * matrix-free ADR solver backend.
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>

namespace MFSolver
{
    template <int dim>
    void MatrixFreeADRSolver<dim>::setup_system()
    {
        dealii::Timer timer;
        setup_time = 0;
        this->mf_setup_initialized = false;

        {
            system_matrix.clear();
            mg_matrices.clear_elements();

            /// Distribute degrees of freedom for the fine mesh and hierarchy.
            dof_handler.distribute_dofs(fe);
            dof_handler.distribute_mg_dofs();

            pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

            pcout << "  Initialize constraints..." << std::endl;
            /// Handle hanging nodes created by adaptive refinement to ensure continuity.
            constraints.clear();
            constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);

            pcout << "  Interpolating boundary values..." << std::endl;
            /// Interpolate Dirichlet boundary conditions from the ProblemData map.
            for (const auto &[boundary_id, function] : this->problem.dirichlet_boundaries)
            {
                /// Interpolate the selected boundary function onto this boundary id.
                VectorTools::interpolate_boundary_values(mapping, dof_handler, boundary_id, *function, constraints);
            }

            constraints.close();

            pcout << "  Setup vectors..." << std::endl;
        }

        setup_time += timer.wall_time();
        time_details << "Distribute DoFs and B.Cs. (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        timer.restart();

        {
            {
                /// Set up the fine-level MatrixFree storage with all update flags
                /// needed by ADR cell and boundary evaluations.
                typename MatrixFree<dim, double>::AdditionalData additional_data;
                additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_color;
                additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points | update_values;
                additional_data.mapping_update_flags_boundary_faces = update_gradients | update_JxW_values | update_quadrature_points | update_values | update_normal_vectors;

                std::shared_ptr<MatrixFree<dim, double>>
                    system_mf_storage(new MatrixFree<dim, double>());

                system_mf_storage->reinit(mapping, dof_handler, constraints, QGauss<1>(fe.degree + 1), additional_data);

                system_matrix.initialize(system_mf_storage);
            }

            system_matrix.initialize_dof_vector(solution);
            system_matrix.initialize_dof_vector(old_solution);
            system_matrix.initialize_dof_vector(system_rhs);
        }

        setup_time += timer.wall_time();
        time_details << "Setup mf system (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        timer.restart();

        {
            /// Repeat matrix-free initialization for every multigrid level.
            /// Float level operators reduce memory traffic during smoothing.
            const unsigned int nlevels = triangulation.n_global_levels();
            mg_matrices.resize(0, nlevels - 1);

            std::set<types::boundary_id> dirichlet_boundary_ids;
            for (const auto &[boundary_id, function] : this->problem.dirichlet_boundaries)
            {
                dirichlet_boundary_ids.insert(boundary_id);
            }

            mg_constrained_dofs.initialize(dof_handler);
            mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

            for (unsigned int level = 0; level < nlevels; ++level)
            {
                AffineConstraints<double> level_constraints(DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

                for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
                {
                    level_constraints.add_line(dof_index);
                }

                level_constraints.close();

                typename MatrixFree<dim, float>::AdditionalData additional_data;
                additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::TasksParallelScheme::partition_color;
                additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points | update_values;
                additional_data.mg_level = level;

                std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level = std::make_shared<MatrixFree<dim, float>>();

                mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);

                mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
            }
        }

        setup_time += timer.wall_time();
        time_details << "Setup matrix-free levels   (CPU/wall) " << timer.cpu_time()
                     << "s/" << timer.wall_time() << 's' << std::endl;
    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::local_assemble_cell(const MatrixFree<dim, double> &data,
                                                      DVector<double> &dst,
                                                      const DVector<double> &src,
                                                      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, -1, 0, 1, double> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);

            if (this->problem.is_time_dependent && this->problem.delta_t > 0.0)
            {
                phi.read_dof_values(src);
                phi.evaluate(EvaluationFlags::values);
            }

            for (const unsigned int q : phi.quadrature_point_indices())
            {
                Point<dim, VectorizedArray<double>> quadrature_point = phi.quadrature_point(q);
                VectorizedArray<double> value_of_f = this->problem.forcing_term->value(quadrature_point);

                if (this->problem.is_time_dependent && this->problem.delta_t > 0.0)
                {
                    value_of_f += phi.get_value(q) / this->problem.delta_t;
                }

                phi.submit_value(value_of_f, q);
            }
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(dst);
        }
    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::assemble()
    {
        Timer timer;

        if (!this->mf_setup_initialized)
        {
            system_matrix.set_time_step(this->problem.is_time_dependent ? this->problem.delta_t : 0.0);
            system_matrix.evaluate_coefficients(*(this->problem.mu), *(this->problem.beta), *(this->problem.gamma));
            system_matrix.compute_diagonal();

            const unsigned int nlevels = triangulation.n_global_levels();
            for (unsigned int level = 0; level < nlevels; ++level)
            {
                mg_matrices[level].set_time_step(this->problem.is_time_dependent ? this->problem.delta_t : 0.0);
                mg_matrices[level].evaluate_coefficients(*(this->problem.mu), *(this->problem.beta), *(this->problem.gamma));
                mg_matrices[level].compute_diagonal();
            }

            AffineConstraints<double> no_constraints;
            no_constraints.close();

            typename MatrixFree<dim, double>::AdditionalData additional_data;
            additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_color;
            additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points | update_values;
            this->inhomogeneous_mf_storage = std::make_shared<MatrixFree<dim, double>>();

            this->inhomogeneous_mf_storage->reinit(mapping, dof_handler, no_constraints, QGauss<1>(fe.degree + 1), additional_data);
            
            this->inhomogeneous_operator = std::make_shared<ADROperator<dim, double>>();
            this->inhomogeneous_operator->initialize(this->inhomogeneous_mf_storage);
            this->inhomogeneous_operator->evaluate_coefficients(*(this->problem.mu), *(this->problem.beta), *(this->problem.gamma));
        }

        system_rhs = 0;

        solution = 0;
        constraints.distribute(solution);
        this->inhomogeneous_operator->vmult(system_rhs, solution);
        system_rhs *= -1.0;

        this->inhomogeneous_operator->get_matrix_free()->cell_loop(
            &MatrixFreeADRSolver::local_assemble_cell, this, system_rhs, old_solution);

        system_rhs.compress(VectorOperation::add);

        /// Face terms use the same dynamic degree/quadrature convention as the
        /// cell evaluator above.
        FEFaceEvaluation<dim, -1, 0, 1, double> face_phi(*system_matrix.get_matrix_free());

        for (unsigned int face = 0; face < system_matrix.get_matrix_free()->n_boundary_face_batches(); ++face)
        {
            const unsigned int boundary_id = system_matrix.get_matrix_free()->get_boundary_id(face);

            if (this->problem.neumann_boundaries.find(boundary_id) != this->problem.neumann_boundaries.end())
            {
                face_phi.reinit(face);

                const auto &neumann = this->problem.neumann_boundaries.at(boundary_id);

                for (const unsigned int q : face_phi.quadrature_point_indices())
                {
                    Point<dim, VectorizedArray<double>> quadrature_point = face_phi.quadrature_point(q);
                    VectorizedArray<double> neumann_value = neumann->value(quadrature_point);

                    face_phi.submit_value(neumann_value, q);
                }

                face_phi.integrate(EvaluationFlags::values);
                face_phi.distribute_local_to_global(system_rhs);
            }
        }

        constraints.distribute(system_rhs);

        std::cout << "Boundary face batches: " << system_matrix.get_matrix_free()->n_boundary_face_batches() << "\n";
        system_rhs.compress(VectorOperation::add);

        constraints.set_zero(system_rhs);

        setup_time += timer.wall_time();
        time_details << "Assemble right hand side   (CPU/wall) " << timer.cpu_time()
                     << "s/" << timer.wall_time() << 's' << std::endl;
    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::solve()
    {
        Timer timer;

        if (!this->mf_setup_initialized)
        {
            /// Build transfer operators for restriction and prolongation.
            this->mg_transfer = std::make_shared<MGTransferMatrixFree<dim, float>>(mg_constrained_dofs);
            this->mg_transfer->build(dof_handler);

            /// Use Chebyshev smoothing because it relies only on matrix-vector products.
            this->mg_smoother = std::make_shared<mg::SmootherRelaxation<SmootherType, DVector<float>>>();
            MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
            smoother_data.resize(0, triangulation.n_global_levels() - 1);

            for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            {
                if (level > 0)
                {
                    /// Configure the smoothing sweep for intermediate and fine levels.
                    smoother_data[level].smoothing_range = this->problem.lvgt0_smoothing_range;
                    smoother_data[level].degree = this->problem.lvgt0_smoothing_degree;
                    smoother_data[level].eig_cg_n_iterations = this->problem.lvgt0_smoothing_eigenvalue_max_iterations;
                }
                else
                {
                    smoother_data[0].smoothing_range = this->problem.lv0_smoothing_range;
                    smoother_data[0].degree = numbers::invalid_unsigned_int;
                    smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
                }
                /// Reuse cached inverse diagonals computed during assemble().
                smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
            }
            this->mg_smoother->initialize(mg_matrices, smoother_data);

            /// Use the configured level-0 smoother as the coarse-grid solver.
            this->mg_coarse = std::make_shared<MGCoarseGridApplySmoother<DVector<float>>>();
            this->mg_coarse->initialize(*(this->mg_smoother));

            this->mg_matrix = std::make_shared<mg::Matrix<DVector<float>>>(mg_matrices);

            /// Interface operators account for discontinuities introduced by h-refinement.
            this->mg_interface_matrices = std::make_shared<MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>>();
            this->mg_interface_matrices->resize(0, triangulation.n_global_levels() - 1);
            for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
                (*(this->mg_interface_matrices))[level].initialize(mg_matrices[level]);

            this->mg_interface = std::make_shared<mg::Matrix<DVector<float>>>(*(this->mg_interface_matrices));

            /// Assemble the full multigrid V-cycle preconditioner structure.
            this->mg = std::make_shared<Multigrid<DVector<float>>>(*(this->mg_matrix), *(this->mg_coarse), *(this->mg_transfer), *(this->mg_smoother), *(this->mg_smoother));
            this->mg->set_edge_matrices(*(this->mg_interface), *(this->mg_interface));

            this->preconditioner = std::make_shared<PreconditionMG<dim, DVector<float>, MGTransferMatrixFree<dim, float>>>(dof_handler, *(this->mg), *(this->mg_transfer));
            
            this->mf_setup_initialized = true;
        }

        /// Use GMRES because advection makes the ADR operator non-symmetric.
        const double adjusted_solver_tolerance =
            this->problem.solver_tolerance_factor * system_rhs.l2_norm();
        SolverControl solver_control(this->problem.solver_max_iterations, adjusted_solver_tolerance);
        solver_control.enable_history_data();
        SolverGMRES<DVector<double>> gmres(solver_control);

        /// Zero out constraints before solving so the GMRES internal vectors aren't corrupted,
        /// then distribute the exact boundary values back at the end.
        constraints.set_zero(solution);
        try
        {
            gmres.solve(system_matrix, solution, system_rhs, *(this->preconditioner));
            converged = (solver_control.last_check() == SolverControl::State::success);
        }
        catch (const SolverControl::NoConvergence &)
        {
            converged = false;
            pcout << "Solver did not converge within "
                  << this->problem.solver_max_iterations
                  << " iterations." << std::endl;
        }
        this->conv_history.emplace_back(solver_control.get_history_data());
        this->solver_tolerances.emplace_back(adjusted_solver_tolerance);
        constraints.distribute(solution);

        pcout << "Solved in " << solver_control.last_step() << " iterations." << std::endl;
        time_details << "Time solve (CPU/wall) " << timer.cpu_time() << "s/" << timer.wall_time() << "s\n";
    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::output_results()
    {
        Timer timer;
        dealii::DataOut<dim> data_out;

        /// Matrix-free vectors need their ghost entries synchronized before
        /// DataOut samples the solution on cells owned by this rank.
        this->solution.update_ghost_values();
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(this->solution, "solution");
        data_out.build_patches(mapping);

        /// Prefer cheap compression so frequent VTU output does not dominate timings.
        dealii::DataOutBase::VtkFlags flags;
        flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
        data_out.set_flags(flags);

        /**
         * The first visualization output reserves the run directory. After that
         * all ranks keep reusing the same path, so rank-local VTU files, the
         * PVTU record, and log.txt describe one coherent run.
         */
        if (this->output_dir.empty())
            create_saving_directory_mf<dim>(this->problem,
                                            this->simd_flag,
                                            this->output_dir);

        const char* disable_vtu_env = std::getenv("MFSOLVER_DISABLE_VTU");
        const bool disable_vtu = (disable_vtu_env != nullptr && (std::string(disable_vtu_env) == "1" || std::string(disable_vtu_env) == "true"));

        if (!disable_vtu)
        {
            time_details << "Creating solution output (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
            timer.restart();

            /// All ranks participate; the shared output_dir keeps VTU pieces
            /// and the PVTU index file attached to one run.
            data_out.write_vtu_with_pvtu_record(
                this->output_dir, "/solution", this->timestep_number, MPI_COMM_WORLD);

            time_details << "Writing solution output (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        }

    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::compute_error()
    {
        if (this->problem.exact_solution == nullptr)
            return;

        this->solution.update_ghost_values();
        dealii::Vector<double> difference_per_cell(triangulation.n_active_cells());

        dealii::VectorTools::integrate_difference(
            mapping,
            dof_handler,
            this->solution,
            *(this->problem.exact_solution),
            difference_per_cell,
            dealii::QGauss<dim>(fe.degree + 1),
            dealii::VectorTools::L2_norm);

        this->l2_error = dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, dealii::VectorTools::L2_norm);

        dealii::VectorTools::integrate_difference(
            mapping,
            dof_handler,
            this->solution,
            *(this->problem.exact_solution),
            difference_per_cell,
            dealii::QGauss<dim>(fe.degree + 1),
            dealii::VectorTools::H1_norm);

        this->h1_error = dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, dealii::VectorTools::H1_norm);

        dealii::VectorTools::integrate_difference(
            mapping,
            dof_handler,
            this->solution,
            *(this->problem.exact_solution),
            difference_per_cell,
            dealii::QGauss<dim>(fe.degree + 1),
            dealii::VectorTools::Linfty_norm);

        this->linfty_error = dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, dealii::VectorTools::Linfty_norm);

        pcout << "   L2 Error vs Exact Solution: " << this->l2_error << std::endl;
        pcout << "   H1 Error vs Exact Solution: " << this->h1_error << std::endl;
        pcout << "   L_infty Error vs Exact Solution: " << this->linfty_error << std::endl;
    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::run()
    {
        pcout << "===========================================" << std::endl;
        pcout << "   Matrix-Free ADR Solver                  " << std::endl;
        pcout << "===========================================" << std::endl;

        pcout << "Number of MPI ranks:            "
              << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;

        const unsigned int n_vect_doubles = dealii::VectorizedArray<double>::size();
        const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
        pcout << "Vectorization over " << n_vect_doubles
              << " doubles = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ')'
              << std::endl
              << std::endl;

        this->start_time = MPI_Wtime();
        GridGenerator::hyper_cube(triangulation, 0., 1., true); ///< Colorize boundaries with deal.II's default hypercube ids.
        triangulation.refine_global(this->problem.refinement_level);

        if (this->problem.is_time_dependent)
        {
            pcout << "--- Time-Dependent Simulation ---" << std::endl;
            setup_system();

            if (this->problem.initial_condition != nullptr)
            {
                VectorTools::interpolate(dof_handler, *(this->problem.initial_condition), old_solution);
            }
            else
            {
                old_solution = 0;
            }
            solution = old_solution;

            this->timestep_number = 0;
            for (double time = 0.0; time <= this->problem.end_time; time += this->problem.delta_t, ++this->timestep_number)
            {
                pcout << "\nTime step " << this->timestep_number << " at t = " << time << std::endl;

                pcout << "   Assembling..." << std::endl;
                assemble();

                pcout << "   Solving..." << std::endl;
                solve();

                pcout << "   Outputting results..." << std::endl;
                output_results();
                compute_error();
                if (this->problem.exact_solution != nullptr)
                {
                    pcout << "   L2 Error vs Exact Solution: " << this->l2_error << std::endl;
                    pcout << "   H1 Error vs Exact Solution: " << this->h1_error << std::endl;
                    pcout << "   L_infty Error vs Exact Solution: " << this->linfty_error << std::endl;
                }

                old_solution = solution;
            }
            pcout << "===========================================" << std::endl;
        }
        else
        {
            setup_system();

            pcout << "   Assembling..." << std::endl;
            assemble();

            pcout << "   Solving..." << std::endl;
            solve();

            pcout << "   Outputting results..." << std::endl;
            output_results();
            compute_error();
            if (this->problem.exact_solution != nullptr)
            {
                pcout << "   L2 Error vs Exact Solution: " << this->l2_error << std::endl;
                pcout << "   H1 Error vs Exact Solution: " << this->h1_error << std::endl;
                pcout << "   L_infty Error vs Exact Solution: " << this->linfty_error << std::endl;
            }

        }

        this->end_time = MPI_Wtime();
    }

    template <int dim>
    void MatrixFreeADRSolver<dim>::output_to_file()
    {
        /// Only rank 0 writes the scalar log file.
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
            return;

        /**
         * log.txt is the metadata companion of the VTU/PVTU files. It must be
         * written into the directory selected during output_results(), not into
         * whatever happens to be the newest test_N folder at the end of the run.
         */
        AssertThrow(!this->output_dir.empty(),
                    ExcMessage("output_results() must be called before output_to_file()"));

        /**
         * The table layout is shared with the matrix-based solver. This wrapper
         * only supplies the matrix-free run state that is private to this
         * concrete class: error norms, convergence flag, and elapsed time.
         */
        write_solver_log_file(this->output_dir,
                              this->problem,
                              this->conv_history,
                              this->solver_tolerances,
                              this->end_time - this->start_time,
                              this->l2_error,
                              this->h1_error,
                              this->linfty_error,
                              converged);
    }
}
