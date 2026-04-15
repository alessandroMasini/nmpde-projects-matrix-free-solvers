// TODO: employ SIMD vectorization

namespace MFSolver{
  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::setup_system () {
    TimerOutput::Scope t(computing_timer, "setup");

    // TODO: use actual grid
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(5);
 
    dof_handler.distribute_dofs(fe);
 
    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;
 
 
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
 
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
 
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // TODO: use actual boundary values
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();
 
    DynamicSparsityPattern dsp(locally_relevant_dofs);
 
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
 
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::assemble () {
    TimerOutput::Scope t(computing_timer, "assembly");
 
    const QGauss<dim> quadrature_formula(this->problem.num_quadrature_points);
 
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      
    // TODO: implement ADR with actual functions
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
 
          cell_matrix = 0.;
          cell_rhs    = 0.;
 
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const double rhs_value =
                (fe_values.quadrature_point(q_point)[1] >
                     0.5 +
                       0.25 * std::sin(4.0 * numbers::PI *
                                       fe_values.quadrature_point(q_point)[0]) ?
                   1. :
                   -1.);
 
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                         fe_values.shape_grad(j, q_point) *
                                         fe_values.JxW(q_point);
 
                  cell_rhs(i) += rhs_value *                         
                                 fe_values.shape_value(i, q_point) * 
                                 fe_values.JxW(q_point);
                }
            }
            
          // TODO: apply b.c.
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    pcout<<"After for loop I'm still alive :)"<<std::endl;
 
    system_matrix.compress(VectorOperation::add);

    pcout<<"After matrix is compressed"<<std::endl;
    system_rhs.compress(VectorOperation::add);


  }

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::solve () {
    TimerOutput::Scope t(computing_timer, "solve");
 
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
 
    SolverControl solver_control(this->problem.solver_max_iterations,
                                 this->problem.solver_tolerance_factor * system_rhs.l2_norm());
    LA::SolverCG  solver(solver_control);
 
 
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, data);
 
    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);
 
    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;
 
    constraints.distribute(completely_distributed_solution);
 
    locally_relevant_solution = completely_distributed_solution;
  }

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::output_results () {
    TimerOutput::Scope t(computing_timer, "output");
 
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");
 
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
 
    data_out.build_patches();
    
    // TODO: inquire this hardwired numbers
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", 0, mpi_communicator, 2, 8);
  }

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::run () {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    setup_system ();
    pcout<<"Finished setup"<<std::endl;
    assemble ();
    pcout<<"Finished assemble"<<std::endl;
    solve ();
    pcout<<"Finished solve"<<std::endl;
    output_results ();

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }
}