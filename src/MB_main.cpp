#include "mfsolver.hpp"
#include "ProblemData.hpp"

int main(int argc, char **argv)
{   
    // Checking that there are enough inputs
    if (argc < 7)
    {
       throw std::invalid_argument("Usage: mpirun -n <n_cores> <program> <n_threads> <problem> <fe_deg> <n_additional_refinements> <delta_t> <max_iters> <tol>");
    }

    // Checking that the considered problem is valid
    // Problems will now be addressed through test_idx, the index relative to the following array
    std::string problems[6] = {"advanced", "lab_02", "lab_03", "parabolic", "transient", "mms"};
    int test_idx = std::find(problems, problems + 6, argv[2]) - problems;
    if (test_idx >= 6){
        throw std::invalid_argument("<problem> needs to be one of 'advanced', 'lab_02', 'lab_03', 'parabolic', 'transient', 'mms'");
    }

    // Initializing MPI
    unsigned int max_n_threads = std::stoi(argv[1]);
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_n_threads);

    // fe_deg is a runtime experiment parameter shared with the matrix-free
    // executable. The matrix-based solver simply constructs FE_Q(fe_deg).
    int parsed_fe_degree = std::stoi(argv[3]);
    if (parsed_fe_degree < 1)
        throw std::invalid_argument("<fe_deg> must be a positive integer");
    unsigned int fe_degree = static_cast<unsigned int>(parsed_fe_degree);
    
    // We first distinguish between 2d and 3d case, and then specialize
    if (test_idx == 1){
        ADR::ProblemData<2> data = ADR::ProblemData<2>::lab_02_poisson(fe_degree);
        data.n_additional_refinements = std::stoi(argv[4]);
        data.refinement_level += data.n_additional_refinements;

        // A time step of 0 simply means getting the problem's default
        // A value different then [0, 1] for a time-indepedent gets simply ignored since 
        // the is_time_dependent flag checks for that
        double delta_t = std::stod(argv[5]);
        if (!data.is_time_dependent && delta_t != 0){
            if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "WARNING: in steady problems, the delta_t argument defaults to 0" << std::endl;
        }
        else if (delta_t > 0 && delta_t < 1)
            data.delta_t = delta_t;
        else if (delta_t != 0)
            throw std::invalid_argument("The time step must be a number between 0 and 1");

        data.solver_max_iterations = std::stoi(argv[6]);
        data.solver_tolerance_factor = std::stod(argv[7]);

        MFSolver::MatrixBasedADRSolver<2> solver(data);
        solver.run();
        solver.output_to_file();
    
    } else {
        ADR::ProblemData<3> data;

        // Switching based on the test case to be created
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

        data.n_additional_refinements = std::stoi(argv[4]);
        data.refinement_level += data.n_additional_refinements;

        // A time step of 0 simply means getting the problem's default
        // A value different then [0, 1] for a time-indepedent gets simply ignored since 
        // the is_time_dependent flag checks for that
        double delta_t = std::stod(argv[5]);
        if (!data.is_time_dependent && delta_t != 0){
            if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "WARNING: in steady problems, the delta_t argument defaults to 0" << std::endl;
        }
        else if (delta_t > 0 && delta_t < 1)
            data.delta_t = delta_t;
        else if (delta_t != 0)
            throw std::invalid_argument("The time step must be a number between 0 and 1");

        data.solver_max_iterations = std::stoi(argv[6]);
        data.solver_tolerance_factor = std::stod(argv[7]);
        
        MFSolver::MatrixBasedADRSolver<3> solver(data);
        solver.run();
        solver.output_to_file();
    }

    return 0;
}
