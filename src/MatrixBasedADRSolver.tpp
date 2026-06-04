namespace MFSolver
{
  template <int dim>
  void MatrixBasedADRSolver<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    // idk if needed
    system_matrix.clear();
    system_rhs.clear();

    GridGenerator::hyper_cube(triangulation);

    for (auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (!cell->face(f)->at_boundary())
          continue;

        const auto p = cell->face(f)->center();

        for (int i = 0, j = 0; i < dim; ++i, j = j + 2)
        {
          if (std::abs(p[i] - 0.0) < 1e-12)
          {
            cell->face(f)->set_boundary_id(j);
            break;
          }
          if (std::abs(p[i] - 1.0) < 1e-12)
          {
            cell->face(f)->set_boundary_id(j + 1);
            break;
          }
        }
      }
    }

    // This is a property of the solver.
    // It should be set according to the level of refinement desired.
    triangulation.refine_global(this->problem.refinement_level);

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

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
    old_solution.reinit(locally_owned_dofs, mpi_communicator);

    // Handle hanging nodes (created by adaptive h-refinement) to ensure solution continuity
    constraints.clear();
    constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Interpolate the Dirichlet (essential) boundary conditions from our ProblemData map
    for (const auto &[boundary_id, function] : this->problem.dirichlet_boundaries)
    {
      // Interpolates the specific function onto the nodes belonging to boundary_id
      VectorTools::interpolate_boundary_values(mapping, dof_handler, boundary_id, *function, constraints);
    }

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

    // Multigrid
    const unsigned int n_levels = triangulation.n_levels();

    mg_matrices.resize(0, n_levels - 1);

    for (unsigned int level = 0; level < n_levels; ++level)
    {

      DynamicSparsityPattern dsp(dof_handler.n_dofs(level));

      MGTools::make_sparsity_pattern(dof_handler, dsp, level);

      mg_matrices[level].reinit(
          dof_handler.locally_owned_mg_dofs(level),
          dof_handler.locally_owned_mg_dofs(level),
          dsp,
          mpi_communicator);
    }

    mg_constrained_dofs.initialize(dof_handler);
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::assemble_on_one_cell (
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim> &scratch,
    PerTaskData<dim> &data) {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = scratch.fe_values.n_quadrature_points;

    /*
     * WorkStream reuses CopyData objects. Always reset every field that the
     * copier may inspect, including the skip flag, before returning control to
     * the framework.
     */
    data.cell_matrix = 0.;
    data.cell_rhs = 0.;
    data.cell_level = numbers::invalid_unsigned_int;
    data.cell_is_locally_owned = false;

    /*
     * In a distributed triangulation, every rank can iterate over cells that
     * exist only as ghosts or artificial cells. Those cells must not contribute
     * to this rank's PETSc matrix/vector. The copier still runs for the item,
     * but the flag above makes it a cheap no-op.
     */
    if (!cell->is_locally_owned())
      return;

    data.cell_is_locally_owned = true;
    data.cell_level = cell->level();

    scratch.fe_values.reinit(cell);

    /*
     * Time-dependent problems need the previous solution in the mass-term
     * contribution to the right-hand side. This vector lives in ScratchData so
     * each worker thread has private storage for FEValues to fill.
     */
    if (this->problem.is_time_dependent)
      scratch.fe_values.get_function_values(old_solution,
                                            scratch.old_solution_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const Point<dim> &quadrature_point =
          scratch.fe_values.quadrature_point(q);
      const double mu_loc = this->problem.mu->value(quadrature_point);
      const Tensor<1, dim, double> b_loc =
          this->problem.beta->value(quadrature_point);
      const double b_div =
          this->problem.beta->divergence(quadrature_point);
      const double k_loc = this->problem.gamma->value(quadrature_point);
      const double f_loc =
          this->problem.forcing_term->value(quadrature_point);
      const double dx = scratch.fe_values.JxW(q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = scratch.fe_values.shape_value(i, q);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const double phi_j = scratch.fe_values.shape_value(j, q);

          /*
           * Mass term.
           */
          if (this->problem.is_time_dependent)
            data.cell_matrix(i, j) +=
                (1.0 / this->problem.delta_t) * phi_i * phi_j * dx;

          // Diffusion: mu grad(phi_i) . grad(phi_j).
          data.cell_matrix(i, j) +=
              mu_loc *
              scratch.fe_values.shape_grad(i, q) *
              scratch.fe_values.shape_grad(j, q) *
              dx;

          // Advection part beta . grad(phi_j), tested against phi_i.
          data.cell_matrix(i, j) +=
              b_loc *
              scratch.fe_values.shape_grad(j, q) *
              phi_i *
              dx;

          // Reaction plus div(beta) term from the conservative formulation.
          data.cell_matrix(i, j) +=
              (k_loc + b_div) * phi_i * phi_j * dx;
        }

        if (this->problem.is_time_dependent)
          data.cell_rhs(i) +=
              (1.0 / this->problem.delta_t) *
              phi_i *
              scratch.old_solution_values[q] *
              dx;

        // Forcing term.
        data.cell_rhs(i) += f_loc * phi_i * dx;
      }
    }

    /*
     * Neumann data are face-local, so the face FEValues object also belongs in
     * ScratchData. Only faces explicitly listed in ProblemData contribute; all
     * other boundary faces are either Dirichlet or natural zero-flux.
     */
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (!cell->face(f)->at_boundary())
        continue;

      /**
       * Checking if the face boundary id is in the Neumann boundary map.
       * If not, it means that this face is either a Dirichlet boundary or
       * a natural zero-flux boundary, so we can skip it.
       */
      const auto boundary_id = cell->face(f)->boundary_id();
      auto it = this->problem.neumann_boundaries.find(boundary_id);
      if (it == this->problem.neumann_boundaries.end())
        continue;

      /**
       * It is necessary to specify the index of the face in the reinit call
       * to set up the correct face quadrature formula. This is particularly
       * relevant if the cell degrees differ.
       */
      scratch.fe_face_values.reinit(cell, f);

      for (unsigned int q = 0;
           q < scratch.fe_face_values.n_quadrature_points;
           ++q)
      {
        const double g =
            it->second->value(scratch.fe_face_values.quadrature_point(q));
        const double dx = scratch.fe_face_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          data.cell_rhs(i) +=
              g *
              scratch.fe_face_values.shape_value(i, q) *
              dx;
      }
    }

    cell->get_dof_indices(data.dof_indices);
  }
    
  template <int dim>
  void MatrixBasedADRSolver<dim>::copy_local_to_global(const PerTaskData<dim> &data)
  {
    /*
     * WorkStream guarantees that this copier is never executed concurrently
     * with another copier invocation and that calls happen in iterator order.
     * That is why the PETSc matrix/vector writes below need no explicit mutex.
     */
    if (!data.cell_is_locally_owned)
      return;

    constraints.distribute_local_to_global(data.cell_matrix,
                                           data.cell_rhs,
                                           data.dof_indices,
                                           system_matrix,
                                           system_rhs);

    /*
     * Every local cell matrix is also scattered to the level matrix associated with the cell's
     * level. The scatter remains in the copier for the same reason as the
     * global system scatter: sparse-matrix writes are shared mutable state.
     */
    constraints.distribute_local_to_global(data.cell_matrix,
                                           data.dof_indices,
                                           mg_matrices[data.cell_level]);
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::assemble_multithreaded()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    /*
     * The solver can assemble once for steady problems and many times for
     * transient problems. Start every assembly from a clean algebraic state.
     */
    {
      TimerOutput::Scope t(computing_timer, "assembly: zero algebraic objects");
      system_matrix = 0;
      system_rhs = 0;
      for (unsigned int l = 0; l < triangulation.n_levels(); ++l)
        mg_matrices[l] = 0;
    }

    TimerOutput::Scope workstream_setup_timer(computing_timer,
                                              "assembly: workstream setup");
    const QGauss<dim> quadrature_formula(this->problem.num_quadrature_points);
    const QGauss<dim - 1> quadrature_boundary(
        this->problem.num_quadrature_points);

    PerTaskData<dim> per_task_data(fe);
    ScratchData<dim> scratch_data(fe,
                                  quadrature_formula,
                                  quadrature_boundary,
                                  update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values,
                                  update_values |
                                      update_quadrature_points | update_JxW_values);
    workstream_setup_timer.stop();

    /*
     * WorkStream runs assemble_on_one_cell() in parallel with private scratch
     * and copy buffers, then runs copy_local_to_global() sequentially. This is
     * the deal.II pattern intended for local finite-element assembly: the
     * expensive quadrature loop is parallel, while the non-thread-safe sparse
     * matrix/vector insertion is serialized by the framework.
     */
    {
      TimerOutput::Scope t(computing_timer, "assembly: workstream run");
      WorkStream::run(
          dof_handler.begin_active(),
          dof_handler.end(),
          *this,
          &MatrixBasedADRSolver<dim>::assemble_on_one_cell,
          &MatrixBasedADRSolver<dim>::copy_local_to_global,
          scratch_data,
          per_task_data);
    }

    /*
     * PETSc accumulates off-process entries lazily. Compression is collective
     * over MPI ranks and must happen after all local WorkStream copy operations
     * have finished.
     */
    {
      TimerOutput::Scope t(computing_timer, "assembly: compress system");
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    }

    {
      TimerOutput::Scope t(computing_timer, "assembly: compress mg matrices");
      for (unsigned int l = 0; l < triangulation.n_levels(); ++l)
        mg_matrices[l].compress(VectorOperation::add);
    }
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::assemble()
  {
    assemble_multithreaded();
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    {
      TimerOutput::Scope t(computing_timer, "solve: solution reinit");
      completely_distributed_solution.reinit(locally_owned_dofs,
                                             mpi_communicator);
    }

    double solver_tolerance = 0.0;
    {
      TimerOutput::Scope t(computing_timer, "solve: rhs l2 norm");
      solver_tolerance =
          this->problem.solver_tolerance_factor * system_rhs.l2_norm();
    }

    /*
     * Keep the multigrid and GMRES objects in one lexical scope so the
     * preconditioner remains alive for the complete Krylov solve, while still
     * timing the expensive construction steps separately.
     */
    {
      // Smoother
      MGSmootherPrecondition<LA::MPI::SparseMatrix,
                             PETScWrappers::PreconditionJacobi,
                             LA::MPI::Vector>
          mg_smoother;
      {
        TimerOutput::Scope t(computing_timer,
                             "solve: mg smoother initialize");
        mg_smoother.initialize(mg_matrices);
        mg_smoother.set_steps(2);
      }

      // Coarse grid solver and preconditioner
      SolverControl coarse_control(1000, 1e-8);
      PETScWrappers::SolverCG coarse_solver(coarse_control, mpi_communicator);

      PETScWrappers::PreconditionJacobi coarse_prec;
      {
        TimerOutput::Scope t(computing_timer,
                             "solve: coarse jacobi initialize");
        coarse_prec.initialize(mg_matrices[0]);
      }

      MGCoarseGridApplySmoother<LA::MPI::Vector> mg_coarse;
      mg::Matrix<LA::MPI::Vector> mg_matrix(mg_matrices);
      {
        TimerOutput::Scope t(computing_timer,
                             "solve: coarse smoother initialize");
        mg_coarse.initialize(mg_smoother);
      }

      // Multigrid
      Multigrid<LA::MPI::Vector> mg(mg_matrix,
                                    mg_coarse,
                                    mg_transfer,
                                    mg_smoother,
                                    mg_smoother);

      PreconditionMG<dim, LA::MPI::Vector, MGTransferPrebuilt<LA::MPI::Vector>>
          preconditioner(dof_handler, mg, mg_transfer);

      // Solver and preconditioner
      SolverControl solver_control(this->problem.solver_max_iterations,
                                   solver_tolerance);
      SolverGMRES<LA::MPI::Vector> solver(solver_control);
      {
        TimerOutput::Scope t(computing_timer, "solve: gmres initialize");
        solver_control.enable_history_data();
      }

      // AMG test
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.symmetric_operator = true;
      LA::MPI::PreconditionAMG preconditioner_amg;
      {
        TimerOutput::Scope t(computing_timer, "solve: amg initialize");
        preconditioner_amg.initialize(system_matrix, data);
      }

      {
        TimerOutput::Scope t(computing_timer, "solve: gmres iterations");
        solver.solve(system_matrix,
                     completely_distributed_solution,
                     system_rhs,
                     preconditioner_amg);
      }

      {
        TimerOutput::Scope t(computing_timer, "solve: convergence bookkeeping");
        this->conv_history.emplace_back(solver_control.get_history_data());
        converged = (solver_control.last_check() ==
                     SolverControl::State::success);
        pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;
      }
    }

    {
      TimerOutput::Scope t(computing_timer, "solve: constraints distribute");
      constraints.distribute(completely_distributed_solution);
    }

    {
      TimerOutput::Scope t(computing_timer,
                           "solve: copy to locally relevant solution");
      locally_relevant_solution = completely_distributed_solution;
    }
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::output_results()
  {
    TimerOutput::Scope t(computing_timer, "output");

    // DataOut owns the conversion from the distributed finite-element solution
    // to VTU/PVTU files. The directory logic below only decides where those
    // collective files belong.
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");

    // Writing the subdomain id beside the solution makes it easier to inspect
    // parallel decompositions when a run is distributed across several ranks.
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    /*
     * Reserve the run folder at the first output point and then reuse it for
     * every following time step. This keeps solution_0.*, solution_1.*, ...
     * together even for time-dependent problems.
     */
    if (this->output_dir.empty())
      create_saving_directory_mb<dim>(this->problem,
                                      this->mpi_communicator,
                                      this->output_dir);
    
    // Every time step writes into the same reserved run folder. The time-step
    // number appears in the filename, not in the directory name.
    data_out.write_vtu_with_pvtu_record(
        this->output_dir, "/solution", this->timestep_number, mpi_communicator);
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on:\n"
          << Utilities::MPI::n_mpi_processes(mpi_communicator) << " MPI rank(s)\n"
          << MultithreadInfo::n_threads() << " threads"
          << std::endl;

    if (!this->problem.is_time_dependent)
    {
      pcout << "Solving a time independent problem" << std::endl;

      this->start_time = MPI_Wtime();
      setup_system();
      pcout << "Finished setup" << std::endl;
      assemble_multithreaded();
      pcout << "Finished assemble" << std::endl;
      solve();
      pcout << "Finished solve" << std::endl;
      output_results();
    }
    else
    {
      pcout << "Solving a time dependent problem" << std::endl;
      setup_system();

      VectorTools::interpolate(dof_handler, *(this->problem.initial_condition), old_solution);
      locally_relevant_solution = old_solution;
      completely_distributed_solution = old_solution;

      output_results();

      while (time < this->problem.end_time - 0.5 * this->problem.delta_t)
      {
        time += this->problem.delta_t;
        ++this->timestep_number;

        pcout << "TIMESTEP " << this->timestep_number << std::endl;
        assemble_multithreaded();
        pcout << "   Finished assemble" << std::endl;
        solve();
        pcout << "   Finished solve" << std::endl;

        old_solution = completely_distributed_solution;
        locally_relevant_solution = completely_distributed_solution;

        output_results();
      }
    }

    this->end_time = MPI_Wtime();
    computing_timer.print_summary();
    computing_timer.reset();

    compute_error();
    if (this->problem.exact_solution != nullptr) {
        pcout << "   L2 Error vs Exact Solution: " << this->l2_error << std::endl;
        pcout << "   H1 Error vs Exact Solution: " << this->h1_error << std::endl;
        pcout << "   L_infty Error vs Exact Solution: " << this->linfty_error << std::endl;
    }

    pcout << std::endl;
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::compute_error()
  {
      if (this->problem.exact_solution == nullptr) return;

      dealii::Vector<double> difference_per_cell(triangulation.n_active_cells());
      
      dealii::VectorTools::integrate_difference(
          mapping,
          dof_handler,
          locally_relevant_solution,
          *(this->problem.exact_solution),
          difference_per_cell,
          dealii::QGauss<dim>(fe.degree + 1),
          dealii::VectorTools::L2_norm
      );
      
      this->l2_error = dealii::VectorTools::compute_global_error(
          triangulation,
          difference_per_cell,
          dealii::VectorTools::L2_norm
      );

      dealii::VectorTools::integrate_difference(
          mapping,
          dof_handler,
          locally_relevant_solution,
          *(this->problem.exact_solution),
          difference_per_cell,
          dealii::QGauss<dim>(fe.degree + 1),
          dealii::VectorTools::H1_norm
      );
      
      this->h1_error = dealii::VectorTools::compute_global_error(
          triangulation,
          difference_per_cell,
          dealii::VectorTools::H1_norm
      );

      dealii::VectorTools::integrate_difference(
          mapping,
          dof_handler,
          locally_relevant_solution,
          *(this->problem.exact_solution),
          difference_per_cell,
          dealii::QGauss<dim>(fe.degree + 1),
          dealii::VectorTools::Linfty_norm
      );
      
      this->linfty_error = dealii::VectorTools::compute_global_error(
          triangulation,
          difference_per_cell,
          dealii::VectorTools::Linfty_norm
      );
  }

  template <int dim>
  void MatrixBasedADRSolver<dim>::output_to_file()
  {
    // Only rank 0 should write to file.
    if (Utilities::MPI::this_mpi_process(this->mpi_communicator) != 0)
      return;

    /*
     * log.txt is the metadata companion of the VTU/PVTU files. It must be
     * written into the directory selected during output_results(), not into
     * whatever happens to be the newest test_N folder at the end of the run.
     */
    AssertThrow(!this->output_dir.empty(),
                ExcMessage("output_results() must be called before output_to_file()"));

    /*
     * The table layout is shared with the matrix-free solver. This wrapper only
     * supplies the matrix-based run state that is private to this concrete
     * class: error norms, convergence flag, and elapsed time.
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
