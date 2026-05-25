#pragma once

// Implementation of MatrixFreeADRSolver methods
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
    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::setup_system()
    {
        dealii::Timer timer;
        setup_time = 0;

        {
            system_matrix.clear();
            mg_matrices.clear_elements();

            // Distribute degrees of freedom for the fine mesh and the multigrid hierarchy
            dof_handler.distribute_dofs(fe);
            dof_handler.distribute_mg_dofs();

            pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

            pcout << "  Initialize constraints..." << std::endl;
            // Handle hanging nodes (created by adaptive h-refinement) to ensure solution continuity
            constraints.clear();
            constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);

            pcout << "  Interpolating boundary values..." << std::endl;
            // Interpolate the Dirichlet (essential) boundary conditions from our ProblemData map
            for (const auto &[boundary_id, function] : this->problem.dirichlet_boundaries)
            {
                // Interpolates the specific function onto the nodes belonging to boundary_id
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
                // Set up the core Matrix-Free storage.
                // We ask it to compute gradients, JxW (Jacobian * quadrature weights), and x-y-z points on the fly.
                typename MatrixFree<dim, double>::AdditionalData additional_data;
                additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_color;
                additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points | update_values;
                additional_data.mapping_update_flags_boundary_faces = update_gradients | update_JxW_values | update_quadrature_points | update_values | update_normal_vectors;

                std::shared_ptr<MatrixFree<dim, double>>
                    system_mf_storage(new MatrixFree<dim, double>());

                // Since the quadrature type decides whether simd is used or not, the choice of the former needs to depend on the latter
                // TODO: Andrea sa
                // if (simd_flag){
                //     system_mf_storage->reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);
                // } else {
                    system_mf_storage->reinit(mapping, dof_handler, constraints, QGauss<1>(fe.degree + 1), additional_data);
                //}

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
            // Now repeat the matrix-free initialization for every single level of the multigrid hierarchy.
            // We use 'float' instead of 'double' here to save memory bandwidth during the coarse grid iterations.
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

                // Since the quadrature type decides whether simd is used or not, the choice of the former needs to depend on the latter
                // TODO: Andrea sa
                // if (simd_flag){
                //     mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);
                // } else {
                    mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);
                //}

                mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
            }
        }

        setup_time += timer.wall_time();
        time_details << "Setup matrix-free levels   (CPU/wall) " << timer.cpu_time()
                     << "s/" << timer.wall_time() << 's' << std::endl;
    }

    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::assemble()
    {
        Timer timer;

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

        system_rhs = 0;

        AffineConstraints<double> no_constraints;
        no_constraints.close();

        ADROperator<dim, fe_degree, double> inhomogeneous_operator;

        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
        std::shared_ptr<MatrixFree<dim, double>> inhomogeneous_mf_storage(new MatrixFree<dim, double>());
        
        // Since the quadrature type decides whether simd is used or not, the choice of the former needs to depend on the latter
        // TODO: Andrea sa
        // if (simd_flag){
        //     inhomogeneous_mf_storage->reinit(mapping, dof_handler, no_constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);
        // } else {
            inhomogeneous_mf_storage->reinit(mapping, dof_handler, no_constraints, QGauss<1>(fe.degree + 1), additional_data);
        // }
        inhomogeneous_operator.initialize(inhomogeneous_mf_storage);

        solution = 0;
        constraints.distribute(solution);
        inhomogeneous_operator.evaluate_coefficients(*(this->problem.mu), *(this->problem.beta), *(this->problem.gamma));
        inhomogeneous_operator.vmult(system_rhs, solution);
        system_rhs *= -1.0;

        FEEvaluation<dim, fe_degree> phi(*inhomogeneous_operator.get_matrix_free());

        for (unsigned int cell = 0; cell < inhomogeneous_operator.get_matrix_free()->n_cell_batches(); ++cell)
        {
            phi.reinit(cell);

            if (this->problem.is_time_dependent && this->problem.delta_t > 0.0)
            {
                phi.read_dof_values(old_solution);
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
            phi.distribute_local_to_global(system_rhs);
        }

        system_rhs.compress(VectorOperation::add);

        FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, double> face_phi(*system_matrix.get_matrix_free());

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

                    face_phi.submit_value(neumann_value, q); // TODO: we need to check whether here we need mu again or not. See proof on miro
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

    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::solve()
    {
        Timer timer;

        // Grid Transfer: builds interpolation weights to move residual data to coarser grids (Restriction)
        // and move algebraic corrections back up to finer grids (Prolongation).
        MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
        mg_transfer.build(dof_handler);

        // Smoother: Chebyshev iteration squashes high-frequency errors. It's mathematically
        // perfect for matrix-free because it entirely relies on matrix-vector multiplications.
        using SmootherType = PreconditionChebyshev<LevelMatrixType, DVector<float>>;
        mg::SmootherRelaxation<SmootherType, DVector<float>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_global_levels() - 1);

        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
            if (level > 0)
            {
                // For intermediate and fine levels, do a quick 5-degree polynomial smoothing sweep
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
            // Inject the cached inverse diagonals we extracted during assemble()
            smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(mg_matrices, smoother_data);

        // Tell the coarse solver to just run the level 0 smoother we just configured above
        MGCoarseGridApplySmoother<DVector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<DVector<float>> mg_matrix(mg_matrices);

        // Hanging node interfaces: when transferring residual data between levels, these
        // special operators correctly account for the spatial discontinuities where h-refinement occurred.
        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(mg_matrices[level]);

        mg::Matrix<DVector<float>> mg_interface(mg_interface_matrices);

        // Assemble the full Multigrid V-Cycle Preconditioner structure
        Multigrid<DVector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, DVector<float>, MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);

        // Outer Iterative Krylov Solver: Since our ADR equation has an asymmetric advection term,
        // standard Conjugate Gradient (CG) could fail here. We use GMRES instead.
        // pcout << this->problem.solver_max_iterations << std::endl;
        // pcout << this->problem.solver_tolerance_factor << std::endl;
        SolverControl solver_control(this->problem.solver_max_iterations, this->problem.solver_tolerance_factor * system_rhs.l2_norm());
        solver_control.enable_history_data();
        SolverGMRES<DVector<double>> gmres(solver_control);

        // Zero out constraints before solving so the GMRES internal vectors aren't corrupted,
        // then distribute the exact boundary values back at the end
        constraints.set_zero(solution);
        gmres.solve(system_matrix, solution, system_rhs, preconditioner);
        this->conv_history.emplace_back(solver_control.get_history_data());
        converged = (solver_control.last_check() == SolverControl::State::success);
        constraints.distribute(solution);

        pcout << "Solved in " << solver_control.last_step() << " iterations." << std::endl;
        time_details << "Time solve (CPU/wall) " << timer.cpu_time() << "s/" << timer.wall_time() << "s\n";
    }

    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::output_results()
    {   
        Timer timer;
        // static unsigned int cycle = 0; // Using an internal counter since the method takes no arguments

        dealii::DataOut<dim> data_out;

        // Matrix-free vectors need their ghost entries synchronized before
        // DataOut samples the solution on cells owned by this rank.
        this->solution.update_ghost_values();
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(this->solution, "solution");
        data_out.build_patches(mapping);

        // Prefer cheap compression because these files are produced often in
        // time-dependent runs; solver timings should not be dominated by I/O.
        dealii::DataOutBase::VtkFlags flags;
        flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
        data_out.set_flags(flags);

        /*
         * The first visualization output reserves the run directory. After that
         * all ranks keep reusing the same path, so rank-local VTU files, the
         * PVTU record, and log.txt describe one coherent run.
         */
        if (this->output_dir.empty())
            create_saving_directory_mf<dim, fe_degree>(this->problem,
                                                       this->simd_flag,
                                                       this->output_dir);

        time_details << "Creating solution output (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        timer.restart();
        
        // All ranks participate here. The shared output_dir ensures their VTU
        // pieces and the PVTU index file describe one run rather than several.
        data_out.write_vtu_with_pvtu_record(
            this->output_dir, "/solution", this->timestep_number, MPI_COMM_WORLD);
        
        time_details << "Writing solution output (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
    }

    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::compute_error()
    {
        if (this->problem.exact_solution == nullptr) return;

        this->solution.update_ghost_values();
        dealii::Vector<double> difference_per_cell(triangulation.n_active_cells());
        
        dealii::VectorTools::integrate_difference(
            mapping,
            dof_handler,
            this->solution,
            *(this->problem.exact_solution),
            difference_per_cell,
            dealii::QGauss<dim>(fe.degree + 1),
            dealii::VectorTools::L2_norm
        );

        this->l2_error = dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, dealii::VectorTools::L2_norm);
        
        dealii::VectorTools::integrate_difference(
            mapping,
            dof_handler,
            this->solution,
            *(this->problem.exact_solution),
            difference_per_cell,
            dealii::QGauss<dim>(fe.degree + 1),
            dealii::VectorTools::H1_norm
        );

        this->h1_error = dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, dealii::VectorTools::H1_norm);
        
        dealii::VectorTools::integrate_difference(
            mapping,
            dof_handler,
            this->solution,
            *(this->problem.exact_solution),
            difference_per_cell,
            dealii::QGauss<dim>(fe.degree + 1),
            dealii::VectorTools::Linfty_norm
        );

        this->linfty_error = dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, dealii::VectorTools::Linfty_norm);
        
        pcout << "   L2 Error vs Exact Solution: " << this->l2_error << std::endl;
        pcout << "   H1 Error vs Exact Solution: " << this->h1_error << std::endl;
        pcout << "   L_infty Error vs Exact Solution: " << this->linfty_error << std::endl;
    }

    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::run()
    {
        pcout << "===========================================" << std::endl;
        pcout << "   Matrix-Free ADR Solver                  " << std::endl;
        pcout << "===========================================" << std::endl;

        pcout << "Number of MPI ranks:            "
              << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;

        // TODO: check this output actually reflects the employed vectorization
        const unsigned int n_vect_doubles = dealii::VectorizedArray<double>::size();
        const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
        pcout << "Vectorization over " << n_vect_doubles
              << " doubles = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ')'
              << std::endl
              << std::endl;

        this->start_time = MPI_Wtime();
        GridGenerator::hyper_cube(triangulation, 0., 1., true); // `true` colorizes the boundaries: 0=left, 1=right, 2=bottom, 3=top, 4=back, 5=front
        triangulation.refine_global(this->problem.refinement_level);

        if (this->problem.is_time_dependent)
        {
            pcout << "--- Time-Dependent Simulation ---" << std::endl;
            // triangulation.refine_global(2); // refine it a bit for the simulation
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
                if (this->problem.exact_solution != nullptr) {
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
            // for (unsigned int cycle = 0; cycle < 3; ++cycle) // let's do 3 cycles for the test
            // {
            //     pcout << "Cycle " << cycle << std::endl;
            //     if (cycle > 0)
            //         triangulation.refine_global(1);

            setup_system();

            pcout << "   Assembling..." << std::endl;
            assemble();

            pcout << "   Solving..." << std::endl;
            solve();

            pcout << "   Outputting results..." << std::endl;
            output_results();
            compute_error();
            if (this->problem.exact_solution != nullptr) {
                pcout << "   L2 Error vs Exact Solution: " << this->l2_error << std::endl;
                pcout << "   H1 Error vs Exact Solution: " << this->h1_error << std::endl;
                pcout << "   L_infty Error vs Exact Solution: " << this->linfty_error << std::endl;
            }

            //     pcout << "===========================================" << std::endl;
            // }
        }

        this->end_time = MPI_Wtime();
    }

    template <int dim, int fe_degree>
    void MatrixFreeADRSolver<dim, fe_degree>::output_to_file()
    {
        // Only rank 0 should write to file.
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
            return;

        /*
         * log.txt is the metadata companion of the VTU/PVTU files. It must be
         * written into the directory selected during output_results(), not into
         * whatever happens to be the newest test_N folder at the end of the run.
         */
        AssertThrow(!this->output_dir.empty(),
                    ExcMessage("output_results() must be called before output_to_file()"));

        /*
         * The table layout is shared with the matrix-based solver. This wrapper
         * only supplies the matrix-free run state that is private to this
         * concrete class: error norms, convergence flag, and elapsed time.
         */
        write_solver_log_file(this->output_dir,
                              this->problem,
                              this->conv_history,
                              this->end_time - this->start_time,
                              this->l2_error,
                              this->h1_error,
                              this->linfty_error,
                              converged);
    }
}
