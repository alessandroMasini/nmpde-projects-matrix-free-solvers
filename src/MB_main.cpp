#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{   
    // Checking that there are enough inputs
    if (argc < 5)
    {
       throw std::invalid_argument("Usage: mpirun -n <n_cores> <program> <n_threads> <problem> <n_additional_refinements> <max_iters> <tol>");
    }

    // Checking that the considered problem is valid
    // Problems will now be addressed through test_idx, the index relative to the following array
    std::string problems[5] = {"advanced", "lab_02", "lab_03", "parabolic", "transient"};
    int test_idx = std::find(problems, problems + 5, argv[2]) - problems;
    if (test_idx >= 5){
        throw std::invalid_argument("<problem> needs to be one of 'advanced', 'lab_02', 'lab_03', 'parabolic', 'transient'");
    }

    // Initializing MPI
    unsigned int max_n_threads = std::stoi(argv[1]);
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_n_threads);
    
    // We first distinguish between 2d and 3d case, and then specialize
    if (test_idx == 1){
        ADR::ProblemData<2, 2> data = ADR::ProblemData<2, 2>::lab_02_poisson();
        data.refinement_level += std::stoi(argv[3]);
        data.solver_max_iterations = std::stoi(argv[4]);
        data.solver_tolerance_factor = std::stod(argv[5]);

        MFSolver::MatrixBasedADRSolver<2, 2> solver(data);
        solver.run();
        solver.output_to_file();
    
    } else {
        ADR::ProblemData<3, 2> data;

        // Switching based on the test case to be created
        switch (test_idx){
            case 0:
            data = ADR::ProblemData<3, 2>::advanced_test_case();
            break;
            
            case 2:
            data = ADR::ProblemData<3, 2>::lab_03_dr_eq();
            break;

            case 3:
            data = ADR::ProblemData<3, 2>::test_case_parabolic();
            break;

            case 4:
            data = ADR::ProblemData<3, 2>::test_case_comprehensive_transient();
            break;
        }

        data.refinement_level += std::stoi(argv[3]);
        data.solver_max_iterations = std::stoi(argv[4]);
        data.solver_tolerance_factor = std::stod(argv[5]);
        
        MFSolver::MatrixBasedADRSolver<3, 2> solver(data);
        solver.run();
        solver.output_to_file();
    }

    return 0;
}