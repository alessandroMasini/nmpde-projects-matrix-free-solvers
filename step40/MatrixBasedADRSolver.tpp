// TODO: employ SIMD vectorization

namespace MFSolver{
  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::setup_system () {
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

        for(int i = 0, j = 0; i<dim; ++i, j = j+2){
          if (std::abs(p[i] - 0.0) < 1e-12){
            cell->face(f)->set_boundary_id(j);
            break;
          }
          if (std::abs(p[i] - 1.0) < 1e-12){
            cell->face(f)->set_boundary_id(j+1);
            break;
          }
        }
      }
    }

    triangulation.refine_global(3);
    
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

    for (unsigned int level = 0; level < n_levels; ++level){

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

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::assemble () {
    TimerOutput::Scope t(computing_timer, "assembly");
 
    const QGauss<dim> quadrature_formula(this->problem.num_quadrature_points);
    const QGauss<dim-1> quadrature_boundary(this->problem.num_quadrature_points);
 
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);
    
    FEFaceValues<dim> fe_values_boundary(fe,
                                     quadrature_boundary,
                                     update_values |
                                      update_quadrature_points | update_JxW_values);

 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    double mu_loc;
    Tensor<1, dim, double> b_loc;
    double b_div;
    double k_loc;
    double f_loc;

    std::vector<double> old_solution_values (n_q_points);

    system_matrix = 0;
    system_rhs = 0;
    for (unsigned int l = 0; l < triangulation.n_levels(); ++l)
      mg_matrices[l] = 0;
      
    // TODO: implement ADR with actual functions
    for (const auto &cell : dof_handler.active_cell_iterators()){
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
 
          cell_matrix = 0.;
          cell_rhs    = 0.;

          fe_values.get_function_values(old_solution, old_solution_values);
 
        for (unsigned int q = 0; q < n_q_points; ++q) {
          mu_loc = this->problem.mu->value (fe_values.quadrature_point (q));
          b_loc = this->problem.beta->value (fe_values.quadrature_point (q));
          b_div = this->problem.beta->divergence(fe_values.quadrature_point (q));
          k_loc = this->problem.gamma->value (fe_values.quadrature_point (q));
          f_loc = this->problem.forcing_term->value (fe_values.quadrature_point (q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              if(this->problem.is_time_dependent){
                cell_matrix (i, j) += (1.0 / this->problem.delta_t) *
                  fe_values.shape_value (i, q) * 
                  fe_values.shape_value (j, q) *
                  fe_values.JxW (q);
              }
              

              // Diffusion.
              cell_matrix (i, j) +=
                // (this->problem.is_time_dependent ? theta : 1.0) * assuming theta = 1.0
                mu_loc *                             //
                fe_values.shape_grad (i, q) *  //
                fe_values.shape_grad (j, q) * //
                fe_values.JxW (q);

              // Advection
              cell_matrix (i, j) += 
                // (this->problem.is_time_dependent ? theta : 1.0) * assuming theta = 1.0
                b_loc *
                fe_values.shape_grad (j, q) *
                fe_values.shape_value (i, q) *
                fe_values.JxW (q);

              // Reaction
              cell_matrix (i, j) += 
                // (this->problem.is_time_dependent ? theta : 1.0) * assuming theta = 1.0
                (k_loc + b_div)*
                fe_values.shape_value (i, q) *
                fe_values.shape_value (j, q) *
                fe_values.JxW (q);
          }
          
          if(this->problem.is_time_dependent){
            cell_rhs (i) += (1.0 / this->problem.delta_t) *             //
              fe_values.shape_value (i, q) * //
              old_solution_values[q] *      //
              fe_values.JxW (q);
          }
          // Forcing term.
          cell_rhs (i) += f_loc * //
            fe_values.shape_value (i, q) *                     //
            fe_values.JxW (q);
        }
      }

      //Neumann
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f){
        if(!cell->face(f)->at_boundary())
          continue;

        const auto boundary_id = cell->face(f)->boundary_id();
        auto it = this->problem.neumann_boundaries.find(boundary_id);
        if(it == this->problem.neumann_boundaries.end())
          continue;

        const auto &function = it->second;

        fe_values_boundary.reinit(cell, f);
        
        for(unsigned int q = 0; q < quadrature_boundary.size(); ++q){
          const double g = function->value(fe_values_boundary.quadrature_point(q));
          for(unsigned int i = 0; i < dofs_per_cell; ++i){
            cell_rhs(i) += g * 
              fe_values_boundary.shape_value (i, q) *
              fe_values_boundary.JxW (q);
          }
        }
      }

            
      
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);

      constraints.distribute_local_to_global(cell_matrix,
                                              local_dof_indices,
                                              mg_matrices[cell->level()]);
      }
    }


 
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    for (unsigned int l = 0; l < triangulation.n_levels(); ++l){
      mg_matrices[l].compress(VectorOperation::add);
    }
  }

  template <int dim, int fe_degree>
  void MatrixBasedADRSolver<dim, fe_degree>::solve () {
    TimerOutput::Scope t(computing_timer, "solve");
 
    completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
 
    
    // Smoother
    MGSmootherPrecondition<LA::MPI::SparseMatrix, PETScWrappers::PreconditionJacobi, LA::MPI::Vector> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);

    // Coarse grid solver and preconditioner
    SolverControl coarse_control(1000, 1e-8);
    PETScWrappers::SolverCG coarse_solver(coarse_control, mpi_communicator);

    PETScWrappers::PreconditionJacobi coarse_prec;
    coarse_prec.initialize(mg_matrices[0]);

    MGCoarseGridApplySmoother<LA::MPI::Vector> mg_coarse;
    mg_coarse.initialize(mg_smoother);
    mg::Matrix<LA::MPI::Vector> mg_matrix(mg_matrices);    

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
                                 this->problem.solver_tolerance_factor * system_rhs.l2_norm());
    SolverGMRES<LA::MPI::Vector> solver(solver_control);

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
    
    // TODO: inquire these hardwired numbers
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", timestep_number, mpi_communicator, 2, 8);

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

    if(!this->problem.is_time_dependent){
      pcout << "Solving a time independent problem" << std::endl;

      setup_system ();
      pcout<<"Finished setup"<<std::endl;
      assemble ();
      pcout<<"Finished assemble"<<std::endl;
      solve ();
      pcout<<"Finished solve"<<std::endl;
      output_results ();
    }
    else{
      pcout << "Solving a time dependent problem" << std::endl;
      setup_system();

      VectorTools::interpolate(dof_handler, *(this->problem.initial_condition), old_solution);
      locally_relevant_solution = old_solution;
      completely_distributed_solution = old_solution;

      output_results();

      while(time < this->problem.end_time - 0.5 * this->problem.delta_t){
        time += this->problem.delta_t;
        ++timestep_number;

        pcout << "TIMESTEP " << timestep_number << std::endl;
        assemble();
        pcout<<"   Finished assemble"<<std::endl;
        solve();
        pcout<<"   Finished solve"<<std::endl;

        old_solution = completely_distributed_solution;
        locally_relevant_solution = completely_distributed_solution;        

        output_results();
      }
    }

    computing_timer.print_summary();
    computing_timer.reset();
 
    pcout << std::endl;
  }

}