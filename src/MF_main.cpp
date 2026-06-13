/**
 * @file MF_main.cpp
 * @brief Command-line entry point for the matrix-free ADR solver executable.
 *
 * @details Parses the experiment parameters produced by the batch scripts,
 * including the SIMD implementation selector, selects a benchmark problem,
 * applies runtime mesh and solver controls, and dispatches to
 * MatrixFreeADRSolver in either two or three spatial dimensions.
 */

#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{   
    /// Validate the expected matrix-free command-line interface.
    if (argc < 8)
    {
       throw std::invalid_argument("Usage: mpirun -n <n_cores> <program> <n_threads> <simd> <problem> <fe_deg> <n_additional_refinements> <delta_t> <max_iters> <tol>");
    }

    /// Validate the benchmark problem and retain its index for dispatch.
    std::string problems[6] = {"advanced", "lab_02", "lab_03", "parabolic", "transient", "mms"};
    int test_idx = std::find(problems, problems + 6, argv[3]) - problems;
    if (test_idx >= 6){
        throw std::invalid_argument("<problem> needs to be one of 'advanced', 'lab_02', 'lab_03', 'parabolic', 'transient', 'mms'");
    }

    /// Initialize MPI and deal.II threading using the requested thread count.
    unsigned int max_n_threads = std::stoi(argv[1]);
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_n_threads);
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "DEBUG: max_n_threads passed = " << max_n_threads << ", MultithreadInfo::n_threads() = " << dealii::MultithreadInfo::n_threads() << std::endl;

    /// Read the runtime finite element degree used by MatrixFree.
    int parsed_fe_degree = std::stoi(argv[4]);
    if (parsed_fe_degree < 1)
        throw std::invalid_argument("<fe_deg> must be a positive integer");
    unsigned int fe_degree = static_cast<unsigned int>(parsed_fe_degree);
    
    /// Dispatch lab_02 to the two-dimensional problem specialization.
    if (test_idx == 1){
        ADR::ProblemData<2> data = ADR::ProblemData<2>::lab_02_poisson(fe_degree);
        data.n_additional_refinements = std::stoi(argv[5]);
        data.refinement_level += data.n_additional_refinements;

        /// Interpret delta_t: zero keeps the problem default; positive values
        /// in (0, 1) override default settings.
        double delta_t = std::stod(argv[6]);
        if (!data.is_time_dependent && delta_t != 0){
            if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "WARNING: in steady problems, the delta_t argument defaults to 0" << std::endl;
        }
        else if (delta_t > 0 && delta_t < 1)
            data.delta_t = delta_t;
        else if (delta_t != 0)
            throw std::invalid_argument("The time step must be a number between 0 and 1");

        data.solver_max_iterations = std::stoi(argv[7]);
        data.solver_tolerance_factor = std::stod(argv[8]);

        bool simd_flag = MFSolver::to_bool(argv[2]);
        MFSolver::MatrixFreeADRSolver<2> solver(data, simd_flag);
        solver.run();
        solver.output_to_file();
    
    } else {
        ADR::ProblemData<3> data;

        /// Instantiate the selected three-dimensional benchmark problem.
        switch (test_idx){
            case 0:
            data = ADR::ProblemData<3>::advanced_test_case(fe_degree);
            break;
            
            case 2:
            data = ADR::ProblemData<3>::lab_03_dr_eq(fe_degree);
            break;

            case 3:
            data = ADR::ProblemData<3>::test_case_parabolic(fe_degree);
            break;

            case 4:
            data = ADR::ProblemData<3>::test_case_comprehensive_transient(fe_degree);
            break;

            case 5:
            data = ADR::ProblemData<3>::mms_test_case(fe_degree);
            break;
        }

        data.n_additional_refinements = std::stoi(argv[5]);
        data.refinement_level += data.n_additional_refinements;

        /// Interpret delta_t: zero keeps the problem default; positive values
        /// in (0, 1) override default settings.
        double delta_t = std::stod(argv[6]);
        if (!data.is_time_dependent && delta_t != 0){
            if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "WARNING: in steady problems, the delta_t argument defaults to 0" << std::endl;
        }
        else if (delta_t > 0 && delta_t < 1)
            data.delta_t = delta_t;
        else if (delta_t != 0)
            throw std::invalid_argument("The time step must be a number between 0 and 1");

        data.solver_max_iterations = std::stoi(argv[7]);
        data.solver_tolerance_factor = std::stod(argv[8]);
        
        bool simd_flag = MFSolver::to_bool(argv[2]);
        MFSolver::MatrixFreeADRSolver<3> solver(data, simd_flag);
        solver.run();
        solver.output_to_file();
    }

    return 0;
}
