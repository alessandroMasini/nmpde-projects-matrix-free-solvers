#include "../src/mfsolver.hpp"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_simplex_p.h>


// TODO: employ SIMD vectorization

using namespace MFSolver;

template <int dim, int fe_degree>
void MatrixBasedADRSolver<dim, fe_degree>::setup_system () {
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  // TODO: adapt this so it actually uses the problem's mesh
  {
    pcout << "Initializing the mesh" << std::endl;
    // Copy the serial mesh into the parallel one.
    {
      Triangulation<dim> mesh_serial;
      GridGenerator::subdivided_hyper_cube (mesh_serial, 40, 0.0, 1.0, /* colorize = */true);
      std::cout << "  Number of elements = " << mesh_serial.n_active_cells ()
        << std::endl;


      GridTools::partition_triangulation (mpi_size, mesh_serial);

      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation (mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation (construction_data);
    }


    pcout << "  Number of elements = " << mesh.n_global_active_cells ()
      << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>> (fe_degree);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
      << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>> (this->problem.num_quadrature_points);

    pcout << "  Quadrature points per cell = " << quadrature->size ()
      << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit (mesh);
    dof_handler.distribute_dofs (*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs () << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs ();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs (dof_handler);

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity (locally_owned_dofs,
      MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern (dof_handler, sparsity);
    sparsity.compress ();

    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit (sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit (locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit (locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit (locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

template <int dim, int fe_degree>
void MatrixBasedADRSolver<dim, fe_degree>::assemble_rhs () {
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size ();

  FEValues<dim> fe_values (*fe,
    *quadrature,
    update_values | update_gradients |
    update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices (dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs = 0.0;

  // Evaluation of the old solution on quadrature nodes of current cell.
  std::vector<double> solution_old_values (n_q);

  // Evaluation of the gradient of the old solution on quadrature nodes of
  // current cell.
  std::vector<Tensor<1, dim>> solution_old_grads (n_q);

  for (const auto& cell : dof_handler.active_cell_iterators ()) {
    if (!cell->is_locally_owned ())
      continue;

    fe_values.reinit (cell);

    cell_matrix = 0.0;
    cell_rhs = 0.0;

    /*
      Evaluate the "solution" value and gradient at each quadrature nodes and stores the result into
      solution_old_values and solution_old_grads respectively before overwriting the solution vector
    */

    fe_values.get_function_values (solution, solution_old_values);
    fe_values.get_function_gradients (solution, solution_old_grads);

    for (unsigned int q = 0; q < n_q; ++q) {
      const double mu_loc = mu (fe_values.quadrature_point (q));
      const double b_loc = b (fe_values.quadrature_point (q));

      const double k_loc = k (fe_values.quadrature_point (q));


      const double f_old_loc =
        f (fe_values.quadrature_point (q), time - delta_t);
      const double f_new_loc = f (fe_values.quadrature_point (q), time);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          // Time derivative.
          cell_matrix (i, j) += (1.0 / delta_t) *             //
            fe_values.shape_value (i, q) * //
            fe_values.shape_value (j, q) * //
            fe_values.JxW (q);

          // Diffusion.
          cell_matrix (i, j) +=
            theta * mu_loc *                             //
            fe_values.shape_grad (i, q) *  //
            fe_values.shape_grad (j, q) * //
            fe_values.JxW (q);

          cell_matrix (i, j) -= theta * b_loc *
            fe_values.shape_value (j, q) *
            fe_values.shape_grad (i, q)[0] *
            fe_values.JxW (q);

          cell_matrix (i, j) += theta * k_loc *
            fe_values.shape_value (i, q) *
            fe_values.shape_value (j, q) *
            fe_values.JxW (q);
        }

        // Time derivative.
        cell_rhs (i) += (1.0 / delta_t) *             //
          fe_values.shape_value (i, q) * //
          solution_old_values[q] *      //
          fe_values.JxW (q);

        // Diffusion.
        cell_rhs (i) -= (1.0 - theta) * mu_loc *                   //
          fe_values.shape_grad (i, q) * //
          solution_old_grads[q] *    //
          fe_values.JxW (q);

        cell_rhs (i) += (1.0 - theta) * b_loc *
          fe_values.shape_grad (i, q)[0] *
          solution_old_values[q] *
          fe_values.JxW (q);

        cell_rhs (i) -= (1.0 - theta) * k_loc *
          fe_values.shape_value (i, q) *
          solution_old_values[q] *
          fe_values.JxW (q);

        // Forcing term.
        cell_rhs (i) +=
          (theta * f_new_loc + (1.0 - theta) * f_old_loc) * //
          fe_values.shape_value (i, q) *                     //
          fe_values.JxW (q);
      }
    }

    cell->get_dof_indices (dof_indices);

    system_matrix.add (dof_indices, cell_matrix);
    system_rhs.add (dof_indices, cell_rhs);
  }

  system_matrix.compress (VectorOperation::add);
  system_rhs.compress (VectorOperation::add);

  {
    std::map<types::global_dof_index, double> boundary_values;
    Functions::ZeroFunction<dim> bc_function;
    std::map<types::boundary_id, const Function<dim>*> boundary_functions;
    boundary_functions[0] = &bc_function;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values (dof_handler,
      boundary_functions,
      boundary_values);

    MatrixTools::apply_boundary_values (
      boundary_values, system_matrix, solution, system_rhs, true);
  }

}

template <int dim, int fe_degree>
void MatrixBasedADRSolver<dim, fe_degree>::solve () {
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize (
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData (1.0));

  SolverControl solver_control (/* maxiter = */ problem.solver_max_iterations,
    /* tolerance = */ problem.solver_tolerance_factor);

  SolverCG<TrilinosWrappers::MPI::Vector> solver (solver_control);

  solver.solve (system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << solver_control.last_step () << " CG iterations" << std::endl;
}

template <int dim, int fe_degree>
void MatrixBasedADRSolver<dim, fe_degree>::output_results () const {
  DataOut<dim> data_out;

  data_out.add_data_vector (dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int (mesh.n_active_cells ());
  GridTools::get_subdomain_association (mesh, partition_int);
  const Vector<double> partitioning (partition_int.begin (), partition_int.end ());
  data_out.add_data_vector (partitioning, "partitioning");

  data_out.build_patches ();

  //const std::filesystem::path mesh_path (mesh_file_name);
  const std::string output_file_name = "ex1.5-output - ";

  data_out.write_vtu_with_pvtu_record (/* folder = */ "./",
    /* basename = */ output_file_name,
    /* index = */ timestep_number,
    MPI_COMM_WORLD);
}

template <int dim, int fe_degree>
void MatrixBasedADRSolver<dim, fe_degree>::run () {
  setup ();
  assemble ();
  solve_linear_system ();
  output ();

}