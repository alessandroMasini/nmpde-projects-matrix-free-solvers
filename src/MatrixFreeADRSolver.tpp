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

namespace MFSolver
{
    template <int dim, int fe_degree, template <int> typename MuCoeffFunc, template <int> typename BetaCoeffFunc, template <int> typename GammaCoeffFunc>
    void MatrixFreeADRSolver<dim, fe_degree, MuCoeffFunc, BetaCoeffFunc, GammaCoeffFunc>::setup_system()
    {
        dealii::Timer timer;
        setup_time = 0;

        {
            system_matrix.clear();
            mg_matrices.clear_elements();

            dof_handler.distribute_dofs(fe);
            dof_handler.distribute_mg_dofs();

            pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

            constraints.clear();
            constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);

            // Interpolate the Dirichlet boundary conditions from our ProblemData map
            for (const auto &[boundary_id, function] : this->problem.dirichlet_boundaries)
            {
                // Interpolates the specific function onto the nodes belonging to boundary_id
                VectorTools::interpolate_boundary_values(mapping, dof_handler, boundary_id, *function, constraints);
            }

            constraints.close();
        }

        setup_time += timer.wall_time();
        time_details << "Distribute DoFs and B.Cs. (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        timer.restart();

        {
            {
                // Initialize the system matrix with the correct settings
                typename MatrixFree<dim, double>::AdditionalData additional_data;
                additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_color;
                additional_data.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points | update_values;

                std::shared_ptr<MatrixFree<dim, double>>
                    system_mf_storage(new MatrixFree<dim, double>());

                system_mf_storage->reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);

                system_matrix.initialize(system_mf_storage);
            }

            system_matrix.initialize_dof_vector(solution);
            system_matrix.initialize_dof_vector(system_rhs);
        }

        setup_time += timer.wall_time();
        time_details << "Setup mf system (cpu/wall): " << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        timer.restart();

        {
            // Initialize all multigrid matrices with correct data
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
                mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGaussLobatto<1>(fe.degree + 1), additional_data);

                mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
            }
        }

        setup_time += timer.wall_time();
        time_details << "Setup matrix-free levels   (CPU/wall) " << timer.cpu_time()
                     << "s/" << timer.wall_time() << 's' << std::endl;
    }

    template <int dim, int fe_degree, template <int> typename MuCoeffFunc, template <int> typename BetaCoeffFunc, template <int> typename GammaCoeffFunc>
    void MatrixFreeADRSolver<dim, fe_degree, MuCoeffFunc, BetaCoeffFunc, GammaCoeffFunc>::assemble()
    {
        Timer timer;

        system_matrix.evaluate_coefficients(
            this->problem.mu,
            this->problem.beta,
            this->problem.gamma);
        system_matrix.compute_diagonal();

        const unsigned int nlevels = triangulation.n_global_levels();
        for (unsigned int level = 0; level < nlevels; ++level)
        {
            mg_matrices[level].evaluate_coefficients(
                this->problem.mu,
                this->problem.beta,
                this->problem.gamma);
            mg_matrices[level].compute_diagonal();
        }

        system_rhs = 0;

        FEEvaluation<dim, fe_degree> phi(*system_matrix.get_matrix_free());
        FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, double> face_phi(*system_matrix.get_matrix_free());

        for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell)
        {
            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
            {
                Point<dim, VectorizedArray<double>> quadrature_point = phi.quadrature_point(q);
                VectorizedArray<double> value_of_f = this->problem.forcing_term->value(quadrature_point);
                phi.submit_value(value_of_f, q);
            }
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(system_rhs);
        }

        for (unsigned int face = 0; face < system_matrix.get_matrix_free()->n_boundary_face_batches(); ++face)
        {
            face_phi.reinit(face);

            const unsigned int boundary_id = system_matrix.get_matrix_free()->get_boundary_id(face);

            if (this->problem.neumann_boundaries.find(boundary_id) != this->problem.neumann_boundaries.end())
            {
                const auto &neumann = this->problem.neumann_boundaries.at(boundary_id);

                for (const unsigned int q : face_phi.quadrature_point_indices())
                {
                    Point<dim, VectorizedArray<double>> quadrature_point = face_phi.quadrature_point(q);
                    VectorizedArray<double> neumann_value = neumann->value(quadrature_point);
                    VectorizedArray<double> mu = this->problem.mu.value(quadrature_point);

                    face_phi.submit_value(neumann_value * mu, q);
                }
            }

            face_phi.integrate(EvaluationFlags::values);
            face_phi.distribute_local_to_global(system_rhs);
        }

        system_rhs.compress(VectorOperation::add);

        setup_time += timer.wall_time();
        time_details << "Assemble right hand side   (CPU/wall) " << timer.cpu_time()
                     << "s/" << timer.wall_time() << 's' << std::endl;
    }

    template <int dim, int fe_degree, template <int> typename MuCoeffFunc, template <int> typename BetaCoeffFunc, template <int> typename GammaCoeffFunc>
    void MatrixFreeADRSolver<dim, fe_degree, MuCoeffFunc, BetaCoeffFunc, GammaCoeffFunc>::solve()
    {
        Timer timer;

        MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
        mg_transfer.build(dof_handler);

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
            smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<DVector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<DVector<float>> mg_matrix(mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(mg_matrices[level]);

        mg::Matrix<DVector<float>> mg_interface(mg_interface_matrices);

        Multigrid<DVector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, DVector<float>, MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);

        SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
        SolverGMRES<DVector<double>> gmres(solver_control);

        constraints.set_zero(solution);
        gmres.solve(system_matrix, solution, system_rhs, preconditioner);
        constraints.distribute(solution);

        pcout << "Solved in " << solver_control.last_step() << " iterations." << std::endl;
        time_details << "Time solve (CPU/wall) " << timer.cpu_time() << "s/" << timer.wall_time() << "s\n";
    }

    template <int dim, int fe_degree, template <int> typename MuCoeffFunc, template <int> typename BetaCoeffFunc, template <int> typename GammaCoeffFunc>
    void MatrixFreeADRSolver<dim, fe_degree, MuCoeffFunc, BetaCoeffFunc, GammaCoeffFunc>::output_results() {}

    template <int dim, int fe_degree, template <int> typename MuCoeffFunc, template <int> typename BetaCoeffFunc, template <int> typename GammaCoeffFunc>
    void MatrixFreeADRSolver<dim, fe_degree, MuCoeffFunc, BetaCoeffFunc, GammaCoeffFunc>::run()
    {
        {
            const unsigned int n_vect_doubles = VectorizedArray<double>::size();
            const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

            pcout << "Vectorization over " << n_vect_doubles
                  << " doubles = " << n_vect_bits << " bits ("
                  << Utilities::System::get_current_vectorization_level() << ')'
                  << std::endl;
        }

        GridGenerator::hyper_cube(triangulation, 0., 1.);
        triangulation.refine_global(3 - dim);
        triangulation.refine_global(1);

        setup_system();
        assemble();
        solve();
    }
}
