/*
   --------------------------------------------------------------------------
                         INCLUDE ALL NECESSARY LIBRARIES
   --------------------------------------------------------------------------
*/

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

namespace Step37 {
    using namespace dealii;

    /*
       --------------------------------------------------------------------------
                              DEFINE THE EQUATION DATA
       --------------------------------------------------------------------------
    */

    const unsigned int degree_finite_element = 2;
    const unsigned int dimension = 3;



    template <int dim>
    class Coefficient : public Function<dim> {                          // definition of a(x)
    public:
        virtual double value (const Point<dim>& p,
            const unsigned int component = 0) const override;

        template <typename number>
        number value (const Point<dim, number>& p,
            const unsigned int        component = 0) const;
    };



    template <int dim>
    template <typename number>
    number Coefficient<dim>::value (const Point<dim, number>& p,        // generic (templated) implementation of the coefficient function
        const unsigned int /*component*/) const {                       // (the computation is independent of the component index, which
        return static_cast<number>(1.) / (0.05 + 2. * p.square ());     // is unused)
    }



    template <int dim>
    double Coefficient<dim>::value (const Point<dim>& p,               // wrapper for the double version
        const unsigned int component) const {
        return value<double> (p, component);
    }


    /*
       --------------------------------------------------------------------------
                                 LAPLACE OPERATOR
       --------------------------------------------------------------------------
    */


    /*
        The operator is implemented to appear as a matrix externally, but does not store any matrix internally.
        It inherits functions like m(), n(), vmult() and Tvmult() from the class
        MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>

        The size of the "matrix" (i.e., number of rows and columns) is determined during initialization via the MatrixFree
        object, and corresponds to the number of degrees of freedom (DoFs) in the problem.

        The vmult() function provided by the base class computes dst = A * src by  first zeroing dst and then calling
        apply_add(dst, src), which is implemented in this class and defines the actual operator action.

        The Tvmult() works in a similar way to vmult().

        The only actual data stored in this class is the coefficient (which is kept private) and is of type
        Table<2, VectorizedArray<number>>

    */
    template <int dim, int fe_degree, typename number>
    class LaplaceOperator
        : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
    public:
        using value_type = number;

        LaplaceOperator ();                 // constructor

        void clear () override;

        void evaluate_coefficient (const Coefficient<dim>& coefficient_function);

        virtual void compute_diagonal () override;

    private:
        virtual void apply_add (
            LinearAlgebra::distributed::Vector<number>& dst,
            const LinearAlgebra::distributed::Vector<number>& src) const override;

        void
            local_apply (const MatrixFree<dim, number>& data,
                LinearAlgebra::distributed::Vector<number>& dst,
                const LinearAlgebra::distributed::Vector<number>& src,
                const std::pair<unsigned int, unsigned int>& cell_range) const;

        void local_compute_diagonal (
            FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>& integrator) const;

        Table<2, VectorizedArray<number>> coefficient;
    };


    /*
        The constructor of the LaplaceOperator class just defaults to the MatrixFreeOperators::Base
        default constructor, which will initialize parameters to make methods such as m() and n()
        work properly
    */
    template <int dim, int fe_degree, typename number>
    LaplaceOperator<dim, fe_degree, number>::LaplaceOperator ()
        : MatrixFreeOperators::Base<dim,
        LinearAlgebra::distributed::Vector<number>>() { }


    /*
        The clear method of the LaplaceOperator class resets the coefficient and calls the clear method
        of the MatrixFreeOperators::Base class.
    */
    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::clear () {
        coefficient.reinit (0, 0);
        MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
            clear ();
    }


    /*
        To initalize the coefficient, we directly give the Coefficient class defined above
        and then select the method coefficient_function.value with vectorized number
    */
    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::evaluate_coefficient (
        const Coefficient<dim>& coefficient_function) {
        const unsigned int n_cells = this->data->n_cell_batches ();
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi (*this->data);

        coefficient.reinit (n_cells, phi.n_q_points);
        for (unsigned int cell = 0; cell < n_cells; ++cell) {
            phi.reinit (cell);
            for (const unsigned int q : phi.quadrature_point_indices ())
                coefficient (cell, q) =
                coefficient_function.value (phi.quadrature_point (q));
        }
    }



    /*
        This is the core of the logic!
        The arguments it takes as inputs are:
         -  data        The MatrixFree object holding all mesh, DoF, quadrature, and mapping info
         -  dst         The destination vector
         -  src         The source vector
         -  cell_range  A range of cells to work on
        This function will be called in parallel by MatrixFree::cell_loop (see apply_add)
    */
    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_apply (
        const MatrixFree<dim, number>& data,
        LinearAlgebra::distributed::Vector<number>& dst,
        const LinearAlgebra::distributed::Vector<number>& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const {

        // the FEEvaluation object holds all the inprotant info (shape function evaluations and grads, quadrature results)
        // FEEvaluation is the matrix-free equivalent of FEvalues
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi (data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {      // iterating over cells
            AssertDimension (coefficient.size (0), data.n_cell_batches ());
            AssertDimension (coefficient.size (1), phi.n_q_points);

            phi.reinit (cell);                                                              // prepare phi
            phi.read_dof_values (src);                                                      // gather local dofs (basicalli u_cell)
            phi.evaluate (EvaluationFlags::gradients);                                      // evaluate gradients
            for (const unsigned int q : phi.quadrature_point_indices ()) {                  // for each quad point
                /*
                    get_gradient applies the Jacobian and returns the gradient in real space. This is then multiplied by
                    the coefficient.
                    The function submit_gradient will then apply the second, the quadrature weight and the Jacobian
                    determinant (aka JxW).
                    Note that the submitted gradient is stored in the same data field as where it is read from in get_gradient.
                    Therefore, you need to make sure to not read from the same quadrature point again after having called
                    submit_gradient on that particular quadrature point.
                */
                phi.submit_gradient (coefficient (cell, q) * phi.get_gradient (q), q);
            }
            /*
                phi.integrate sums over all quad points (which is equivalent to assembling the local constribution of A*x
                without actually storing the matrix)
            */
            phi.integrate (EvaluationFlags::gradients);
            phi.distribute_local_to_global (dst);                           // add local contributions to global destination vector
        }
    }


    /*
        calls the function MatrixBase::cell_loop that splits all cells in independent chuncks and calls the function local_apply
        in parallel
    */
    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::apply_add (
        LinearAlgebra::distributed::Vector<number>& dst,
        const LinearAlgebra::distributed::Vector<number>& src) const {
        /*
            In the base class, there is a member: std::shared_ptr<MatrixFree<dim, number>> data;
            which is a pointer or reference to the MatrixFree object.

            So when writing this->data->cell_loop(...), we are calling cell_loop on the MatrixFree
            object stored in the base class.
        */
        this->data->cell_loop (&LaplaceOperator::local_apply, this, dst, src);
    }

    /*
        This function calculates the inverse of the diagonal part of the matrix A, which is often needed
        for preconditioners that will be used when solving the linear system.
    */
    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::compute_diagonal () {
        /*
            allocate a new diagonal matrix object
            inverse_diagonal_entries is a shared pointer to a DiagonalMatrix
        */
        this->inverse_diagonal_entries.reset (
            new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>> ());


        // get a referenct to the vector storing the diagonal of the matrix initialized above
        LinearAlgebra::distributed::Vector<number>& inverse_diagonal =
            this->inverse_diagonal_entries->get_vector ();

        // initializes the vector with the correct parallel layout and size.
        this->data->initialize_dof_vector (inverse_diagonal);

        /*
            The function local_compute_diagonal (defined below) tells us how the operator acts
            on a single cell.
            We use the compute_diagonal helper function to fill the vector
        */
        MatrixFreeTools::compute_diagonal (*this->data,
            inverse_diagonal,
            &LaplaceOperator::local_compute_diagonal,
            this);

        // we fix all constrained entries to one in order to avoid problems when inverting the values
        this->set_constrained_entries_to_one (inverse_diagonal);

        // invert the diagonal
        for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size (); ++i) {
            Assert (inverse_diagonal.local_element (i) > 0.,
                ExcMessage ("No diagonal entry in a positive definite operator "
                    "should be zero"));
            inverse_diagonal.local_element (i) =
                static_cast<number>(1.) / inverse_diagonal.local_element (i);
        }
    }



    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_compute_diagonal (
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>& phi) const {

        const unsigned int cell = phi.get_current_cell_index ();

        phi.evaluate (EvaluationFlags::gradients);

        /*
            get_gradient applies the Jacobian and returns the gradient in real space. This is then multiplied by
            the coefficient.
            The function submit_gradient will then apply the second, the quadrature weight and the Jacobian
            determinant (aka JxW).
            Note that the submitted gradient is stored in the same data field as where it is read from in get_gradient.
            Therefore, you need to make sure to not read from the same quadrature point again after having called
            submit_gradient on that particular quadrature point.
        */
        for (const unsigned int q : phi.quadrature_point_indices ()) {
            phi.submit_gradient (coefficient (cell, q) * phi.get_gradient (q), q);
        }
        phi.integrate (EvaluationFlags::gradients);
    }


    /*
       --------------------------------------------------------------------------
                                 LAPLACE PROBLEM
       --------------------------------------------------------------------------
    */
    template <int dim>
    class LaplaceProblem {
    public:
        LaplaceProblem ();
        void run ();

    private:
        void setup_system ();
        void assemble_rhs ();
        void solve ();
        void output_results (const unsigned int cycle) const;

        // two possible ways of defining the triangulaton based on deal.ii options

#ifdef DEAL_II_WITH_P4EST
        parallel::distributed::Triangulation<dim> triangulation;
#else
        Triangulation<dim> triangulation;
#endif

        const FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        const MappingQ1<dim> mapping;

        AffineConstraints<double> constraints;
        using SystemMatrixType =
            LaplaceOperator<dim, degree_finite_element, double>;
        SystemMatrixType system_matrix;

        MGConstrainedDoFs mg_constrained_dofs;
        using LevelMatrixType = LaplaceOperator<dim, degree_finite_element, float>;
        MGLevelObject<LevelMatrixType> mg_matrices;

        LinearAlgebra::distributed::Vector<double> solution;
        LinearAlgebra::distributed::Vector<double> system_rhs;

        double             setup_time;
        ConditionalOStream pcout;
        ConditionalOStream time_details;
    };



    template <int dim>
    LaplaceProblem<dim>::LaplaceProblem ()
#ifdef DEAL_II_WITH_P4EST
        : triangulation (MPI_COMM_WORLD,
            Triangulation<dim>::limit_level_difference_at_vertices,
            parallel::distributed::Triangulation<
            dim>::construct_multigrid_hierarchy)
#else
        : triangulation (Triangulation<dim>::limit_level_difference_at_vertices)
#endif
        , fe (degree_finite_element)
        , dof_handler (triangulation)
        , setup_time (0.)
        , pcout (std::cout, Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
        , time_details (std::cout,
            false &&                            // this disables the stream that prints time details, by removing it we get them printed
            Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0) { }





    template <int dim>
    void LaplaceProblem<dim>::setup_system () {
        Timer time;
        setup_time = 0;

        {
            // Setting up the dof handler
            system_matrix.clear ();
            mg_matrices.clear_elements ();

            dof_handler.distribute_dofs (fe);
            dof_handler.distribute_mg_dofs ();  // assigning dofs   for each multigrid level

            pcout << "Number of degrees of freedom: " << dof_handler.n_dofs ()
                << std::endl;

            // Initiazing the constraints

            // 1. removing old constraints
            constraints.clear ();
            // 2. making sure that the constraints know the locally relevant dofs (aka locally owned + ghost dofs) 
            //    to ensure correctness in parallel without unnecessary global storage

            // [CHANGES IN THE CODE] bc deal.ii 9.5 does not allow two arguments in reinit
            // constraints.reinit (dof_handler.locally_owned_dofs (),
            //     DoFTools::extract_locally_relevant_dofs (dof_handler));
            constraints.reinit (dof_handler.locally_owned_dofs ());
            // [END OF CHANGES]

            // 3. adding constraints for hanging nodes
            DoFTools::make_hanging_node_constraints (dof_handler, constraints);
            // 4. imposing homogeneous Dirichlet boundary conditions
            VectorTools::interpolate_boundary_values (
                mapping, dof_handler, 0, Functions::ZeroFunction<dim> (), constraints);
            // 5. finalizing constraint asssembly
            constraints.close ();
        }
        setup_time += time.wall_time ();
        time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << 's' << std::endl;
        time.restart ();
        {
            {
                // setting up the MatrixFree instance for the problem
                typename MatrixFree<dim, double>::AdditionalData additional_data;
                additional_data.tasks_parallel_scheme =
                    MatrixFree<dim, double>::AdditionalData::none;
                additional_data.mapping_update_flags =
                    (update_gradients | update_JxW_values | update_quadrature_points);
                // the MatrixFree class (which is the base for the LaplaceOperator) is 
                // initialized with a shared pointer to MatrixFree object
                std::shared_ptr<MatrixFree<dim, double>> system_mf_storage (
                    new MatrixFree<dim, double> ());
                system_mf_storage->reinit (mapping,
                    dof_handler,
                    constraints,
                    QGauss<1> (fe.degree + 1),
                    additional_data);
                system_matrix.initialize (system_mf_storage);
            }

            // initializing the coefficient
            system_matrix.evaluate_coefficient (Coefficient<dim> ());

            // initializing src and dst vectors
            system_matrix.initialize_dof_vector (solution);
            system_matrix.initialize_dof_vector (system_rhs);
        }
        setup_time += time.wall_time ();
        time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << 's' << std::endl;
        time.restart ();

        {
            // initializing matrices for the multigrid method over all levels
            const unsigned int nlevels = triangulation.n_global_levels ();
            mg_matrices.resize (0, nlevels - 1);

            // mg_constrained_dofs // keeps information about the indices subject to boundary conditions as well 
            // as the indices on edges between different refinement levels
            const std::set<types::boundary_id> dirichlet_boundary_ids = { 0 };
            mg_constrained_dofs.initialize (dof_handler);
            mg_constrained_dofs.make_zero_boundary_constraints (
                dof_handler, dirichlet_boundary_ids);

            for (unsigned int level = 0; level < nlevels; ++level) { // for each level we initialize the matrices
                // [CHANGES IN THE CODE] bc in deal.ii 9.5 this constructor does not exist
                // AffineConstraints<double> level_constraints (
                //     dof_handler.locally_owned_mg_dofs (level),
                //     DoFTools::extract_locally_relevant_level_dofs (dof_handler, level));
                AffineConstraints<double> level_constraints;
                level_constraints.reinit (dof_handler.locally_owned_mg_dofs (level));
                    // [END OF CHANGES]

                    for (const types::global_dof_index dof_index :
                            mg_constrained_dofs.get_boundary_indices (level)) {
                        // [CHANGES IN THE CODE] bc in deal.ii 9.5 this method does not exist
                        // level_constraints.constrain_dof_to_zero (dof_index);
                        level_constraints.add_line(dof_index);
                        // [END OF CHANGES]
                    }
                level_constraints.close ();

                typename MatrixFree<dim, float>::AdditionalData additional_data;
                additional_data.tasks_parallel_scheme =
                    MatrixFree<dim, float>::AdditionalData::none;
                additional_data.mapping_update_flags =
                    (update_gradients | update_JxW_values | update_quadrature_points);
                additional_data.mg_level = level;
                std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level =
                    std::make_shared<MatrixFree<dim, float>> ();
                mg_mf_storage_level->reinit (mapping,
                    dof_handler,
                    level_constraints,
                    QGauss<1> (fe.degree + 1),
                    additional_data);

                mg_matrices[level].initialize (mg_mf_storage_level,
                    mg_constrained_dofs,
                    level);
                mg_matrices[level].evaluate_coefficient (Coefficient<dim> ());
            }
        }
        setup_time += time.wall_time ();
        time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << 's' << std::endl;
    }



    // in the assemble phase we just need to assemble the rhs of the system, since we are not storing the matrix
    template <int dim>
    void LaplaceProblem<dim>::assemble_rhs () {
        Timer time;

        system_rhs = 0;
        FEEvaluation<dim, degree_finite_element> phi (
            *system_matrix.get_matrix_free ());
        for (unsigned int cell = 0;
            cell < system_matrix.get_matrix_free ()->n_cell_batches ();
            ++cell) {
            phi.reinit (cell);
            for (const unsigned int q : phi.quadrature_point_indices ()) // for each quad point
                phi.submit_value (make_vectorized_array<double> (1.0), q);
            phi.integrate (EvaluationFlags::values);
            phi.distribute_local_to_global (system_rhs);
        }
        system_rhs.compress (VectorOperation::add);

        setup_time += time.wall_time ();
        time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << 's' << std::endl;
    }




    template <int dim>
    void LaplaceProblem<dim>::solve () {
        Timer                            time;
        MGTransferMatrixFree<dim, float> mg_transfer (mg_constrained_dofs);
        mg_transfer.build (dof_handler);
        setup_time += time.wall_time ();
        time_details << "MG build transfer time     (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << "s\n";
        time.restart ();

        using SmootherType =
            PreconditionChebyshev<LevelMatrixType,
            LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType,
            LinearAlgebra::distributed::Vector<float>>
            mg_smoother;                               // Chebyshev smoother (great for parallelization)
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize (0, triangulation.n_global_levels () - 1);
        // we initialize the smoother with all level matrices
        for (unsigned int level = 0; level < triangulation.n_global_levels ();
            ++level) {
            if (level > 0) {
                smoother_data[level].smoothing_range = 15.; // smoothing the range [1.2 * L / 15, 1.2 * L] (L = max eigval)
                smoother_data[level].degree = 5; // high degree bc matrix-vector products are cheap (higher degree = better smoothing)
                smoother_data[level].eig_cg_n_iterations = 10;  // cg iterations used to estimate max eigval
            }
            else {
                // on level 0 we initialize the smoother differently:
                // 1. smoothing range is reinterpreted as relative tolerance
                smoother_data[0].smoothing_range = 1e-3;
                // 2. the following line switches Chebyshev from smoother to solver
                smoother_data[0].degree = numbers::invalid_unsigned_int;
                // 3. the number of iterations is high bc we need to solve, not just to smooth
                smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m ();
            }
            // since the internal preconditioner is the inverse diagonal of the matrix, 
            // we are practically using Jacobi iteration
            mg_matrices[level].compute_diagonal ();
            smoother_data[level].preconditioner =
                mg_matrices[level].get_matrix_diagonal_inverse ();
        }
        mg_smoother.initialize (mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
            mg_coarse;
        mg_coarse.initialize (mg_smoother);

        // setup the interface matrices needed to deal with hanging nodes
        // (which appear at the interface between multigrid levels)
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix (
            mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
            mg_interface_matrices;
        mg_interface_matrices.resize (0, triangulation.n_global_levels () - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels ();
            ++level)
            mg_interface_matrices[level].initialize (mg_matrices[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface (
            mg_interface_matrices);

        // we use Chebyshev both for pre and post smoothing
        Multigrid<LinearAlgebra::distributed::Vector<float>> mg (
            mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices (mg_interface, mg_interface);

        PreconditionMG<dim,
            LinearAlgebra::distributed::Vector<float>,
            MGTransferMatrixFree<dim, float>>
            preconditioner (dof_handler, mg, mg_transfer);

        // actually solving the system
        SolverControl solver_control (100, 1e-12 * system_rhs.l2_norm ());
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg (solver_control);
        setup_time += time.wall_time ();
        time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << "s\n";
        pcout << "Total setup time               (wall) " << setup_time << "s\n";

        time.reset ();
        time.start ();
        constraints.set_zero (solution);
        cg.solve (system_matrix, solution, system_rhs, preconditioner);

        constraints.distribute (solution);

        pcout << "Time solve (" << solver_control.last_step () << " iterations)"
            << (solver_control.last_step () < 10 ? "  " : " ") << "(CPU/wall) "
            << time.cpu_time () << "s/" << time.wall_time () << "s\n";
    }




    template <int dim>
    void LaplaceProblem<dim>::output_results (const unsigned int cycle) const {
        Timer time;
        if (triangulation.n_global_active_cells () > 1000000)
            return;

        DataOut<dim> data_out;

        solution.update_ghost_values ();
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");
        data_out.build_patches (mapping);

        DataOutBase::VtkFlags flags;
        flags.compression_level = DataOutBase::CompressionLevel::best_speed;
        data_out.set_flags (flags);
        data_out.write_vtu_with_pvtu_record (
            "./", "solution", cycle, MPI_COMM_WORLD, 3);

        time_details << "Time write output          (CPU/wall) " << time.cpu_time ()
            << "s/" << time.wall_time () << "s\n";
    }




    template <int dim>
    void LaplaceProblem<dim>::run () {
        {
            const unsigned int n_vect_doubles = VectorizedArray<double>::size ();
            const unsigned int n_vect_bits = 8 * sizeof (double) * n_vect_doubles;

            pcout << "Vectorization over " << n_vect_doubles
                << " doubles = " << n_vect_bits << " bits ("
                << Utilities::System::get_current_vectorization_level () << ')'
                << std::endl;
        }

        for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle) {
            pcout << "Cycle " << cycle << std::endl;

            if (cycle == 0) {
                GridGenerator::hyper_cube (triangulation, 0., 1.);
                triangulation.refine_global (3 - dim);
            }
            triangulation.refine_global (1);
            setup_system ();
            assemble_rhs ();
            solve ();
            output_results (cycle);
            pcout << std::endl;
        };
    }
} // namespace Step37




int main (int argc, char* argv [ ]) {
    try {
        using namespace Step37;

        Utilities::MPI::MPI_InitFinalize mpi_init (argc, argv, 1);

        LaplaceProblem<dimension> laplace_problem;
        laplace_problem.run ();
    }
    catch (std::exception& exc) {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what () << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    catch (...) {
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