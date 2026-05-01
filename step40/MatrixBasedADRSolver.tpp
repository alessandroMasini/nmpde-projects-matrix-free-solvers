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
  void MatrixBasedADRSolver<dim, fe_degree>::assemble_on_one_cell (
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim> &scratch,
    PerTaskData<dim> &data) {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = scratch.fe_values.n_quadrature_points;
 
    double mu_loc;
    Tensor<1, dim, double> b_loc;
    double b_div;
    double k_loc;
    double f_loc;
    
    // TODO: is this check useful?
    if (cell->is_locally_owned())
      {
        scratch.fe_values.reinit(cell);

        data.cell_matrix = 0.;
        data.cell_rhs    = 0.;
    
        for (unsigned int q = 0; q < n_q_points; ++q) {
          mu_loc = this->problem.mu->value (scratch.fe_values.quadrature_point (q));
          b_loc = this->problem.beta->value (scratch.fe_values.quadrature_point (q));
          b_div = this->problem.beta->divergence(scratch.fe_values.quadrature_point (q));
          k_loc = this->problem.gamma->value (scratch.fe_values.quadrature_point (q));
          f_loc = this->problem.forcing_term->value (scratch.fe_values.quadrature_point (q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              // Diffusion.
              data.cell_matrix (i, j) +=
                mu_loc *                             //
                scratch.fe_values.shape_grad (i, q) *  //
                scratch.fe_values.shape_grad (j, q) * //
                scratch.fe_values.JxW (q);

              // Advection
              data.cell_matrix (i, j) += b_loc *
                scratch.fe_values.shape_grad (j, q) *
                scratch.fe_values.shape_value (i, q) *
                scratch.fe_values.JxW (q);

              // Reaction
              data.cell_matrix (i, j) += (k_loc + b_div)*
                scratch.fe_values.shape_value (i, q) *
                scratch.fe_values.shape_value (j, q) *
                scratch.fe_values.JxW (q);
          }

          // Forcing term.
          data.cell_rhs (i) += f_loc * //
            scratch.fe_values.shape_value (i, q) *                     //
            scratch.fe_values.JxW (q);
        }
      }

      cell->get_dof_indices(data.dof_indices);
    }
  }
    
  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::copy_local_to_global(const PerTaskData<dim> &data)
    {     
      constraints.distribute_local_to_global(data.cell_matrix,
                                              data.cell_rhs,
                                              data.dof_indices,
                                              system_matrix,
                                              system_rhs);
  }

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::assemble () {
    TimerOutput::Scope t(computing_timer, "assembly");

    PerTaskData per_task_data(fe);

    const QGauss<dim> quadrature_formula(this->problem.num_quadrature_points);
    ScratchData scratch_data(fe,
                             quadrature_formula,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);


    WorkStream::run (dof_handler.begin_active(),
                 dof_handler.end(),
                 *this,
                 &MatrixBasedADRSolver<dim, fe_degree>::assemble_on_one_cell,
                 &MatrixBasedADRSolver<dim, fe_degree>::copy_local_to_global,
                 scratch_data,
                 per_task_data);

    pcout<<"After local assembly I'm still alive :)"<<std::endl;

    // TODO: can compression be done in multi-threaded way?
    system_matrix.compress(VectorOperation::add);
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