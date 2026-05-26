#pragma once

namespace MFSolver
{
  inline bool to_bool(const std::string &x)
  {
    // Command-line solver selection passes the SIMD flag as 0/1. Keep the
    // accepted values narrow so an accidental string cannot silently choose the
    // non-vectorized path.
    assert(x == "0" || x == "1");
    return x == "1";
  }

  /**
   * Find the highest run number already present in a parameter directory.
   *
   * A parameter directory may contain holes such as test_0, test_2, test_4
   * after failed or manually removed runs. For choosing the next run folder,
   * the meaningful value is therefore the largest numeric suffix, not the
   * number of folders currently present.
   */
  inline int get_max_test_number(const std::filesystem::path &dir)
  {
    /*
     * A missing parameter directory means no run has produced output for this
     * configuration yet, so the caller should start numbering from test_0.
     */
    if (!std::filesystem::exists(dir))
      return -1;

    int max_test_number = -1;
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
      // Ignore non-directories such as temporary files or notes left near the
      // results. Only test_N folders participate in run numbering.
      if (!entry.is_directory())
        continue;

      const std::string name = entry.path().filename().string();
      const std::string prefix = "test_";

      // The parameter directory may contain unrelated folders. Treat them as
      // out-of-band metadata, not as solver runs.
      if (name.rfind(prefix, 0) != 0)
        continue;

      try
      {
        /*
         * A valid run folder is exactly test_<integer>. The parsed_length check
         * keeps names like test_3_backup from being mistaken for a real run.
         */
        std::size_t parsed_length = 0;
        const int test_number = std::stoi(name.substr(prefix.size()), &parsed_length);
        if (parsed_length == name.size() - prefix.size())
          max_test_number = std::max(max_test_number, test_number);
      }
      catch (const std::exception &)
      {
        // Malformed test_* names are ignored so one bad manual folder does not
        // prevent the next simulation from reserving a clean directory.
        continue;
      }
    }

    return max_test_number;
  }

  /**
   * Share a rank-0 directory decision with the whole MPI communicator.
   *
   * Directory selection produces strings rather than plain numbers: the chosen
   * test_N path when setup succeeds, or an error message when it does not.
   * Broadcasting the length first gives every rank enough information to receive
   * the same text and follow the same control flow.
   */
  inline void broadcast_string_from_root(std::string &value,
                                         const MPI_Comm &mpi_communicator)
  {
    // MPI needs to know how many characters will follow before non-root ranks
    // can size their receive buffer.
    int value_size = static_cast<int>(value.size());
    MPI_Bcast(&value_size, 1, MPI_INT, 0, mpi_communicator);

    // On non-root ranks this resize creates the exact buffer that will receive
    // the text chosen by rank 0.
    value.resize(value_size);

    // Empty strings are meaningful here: they represent "no error" or "no
    // selected directory yet". Avoid broadcasting a zero-length data pointer.
    if (value_size > 0)
      MPI_Bcast(value.data(), value_size, MPI_CHAR, 0, mpi_communicator);
  }

  inline void create_saving_directory(const std::filesystem::path &base_dir,
                                      const MPI_Comm &mpi_communicator,
                                      std::string &output_dir)
  {
    /*
     * A solver run is represented by exactly one test_N folder.
     *
     * The solution writer is collective: every MPI rank participates and writes
     * its own piece of the output. If every rank independently inspected the
     * filesystem and chose a test_N name, different ranks could split one run
     * across multiple folders. Instead, rank 0 reserves the folder and then
     * broadcasts that decision to everyone else.
     */
    static std::mutex directory_mutex;

    // selected_dir is the normal result; error_message is the failure result.
    // Broadcasting both lets every rank reach the same AssertThrow below.
    std::string selected_dir;
    std::string error_message;

    // Only rank 0 talks to the filesystem for the reservation. Other ranks
    // wait for the chosen path, which prevents one MPI run from splitting into
    // rank-specific folders.
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      /*
       * This mutex protects multiple solver objects in the same process. The
       * filesystem create_directory call below protects separate MPI launches
       * of this project that happen to write the same parameter directory at
       * the same time.
       */
      std::lock_guard<std::mutex> lock(directory_mutex);

      try
      {
        // The parameter directory itself is shared by many test_N folders and
        // may not exist yet for a new combination of solver settings.
        std::filesystem::create_directories(base_dir);

        /*
         * Reserve the run directory by creating it, not merely by counting
         * existing folders. If another independent run claims the candidate
         * first, we simply try the next number.
         */
        for (int test_number = get_max_test_number(base_dir) + 1;; ++test_number)
        {
          const std::filesystem::path candidate =
              base_dir / ("test_" + std::to_string(test_number));

          // create_directory is the actual reservation operation. It succeeds
          // only for the process that first claims this exact test_N path.
          std::error_code error_code;
          if (std::filesystem::create_directory(candidate, error_code))
          {
            selected_dir = candidate.string();
            break;
          }

          // If the candidate already exists, another run got there first; try
          // the next integer rather than failing the whole batch.
          if (std::filesystem::exists(candidate))
            continue;

          // Any other failure means the filesystem is not giving us a usable
          // run directory, so all ranks should stop with the same message.
          error_message = "Could not create " + candidate.string() + ": " +
                          error_code.message();
          break;
        }
      }
      catch (const std::exception &exception)
      {
        // Convert filesystem exceptions into a broadcastable error so non-root
        // ranks do not keep running after rank 0 has failed setup.
        error_message = exception.what();
      }
    }

    /*
     * Publish both the successful path and the failure state. Non-root ranks did
     * not attempt the reservation themselves, so this is their only source of
     * truth about where this run belongs, or why no valid output directory
     * exists.
     */
    broadcast_string_from_root(selected_dir, mpi_communicator);
    broadcast_string_from_root(error_message, mpi_communicator);

    AssertThrow(error_message.empty(), ExcMessage(error_message));
    AssertThrow(!selected_dir.empty(), ExcMessage("No output directory was selected"));

    // Store the agreed directory in the solver object through the reference
    // parameter. Later log writing uses this exact path, not a filesystem query.
    output_dir = selected_dir;

    /*
     * Non-root ranks should see the directory before DataOut starts writing
     * their VTU pieces. The barrier keeps the collective output phase from
     * racing ahead of directory visibility on shared filesystems.
     */
    std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_communicator);
  }

  template <int dim, int fe_degree>
  void create_saving_directory_mb(ADR::ProblemData<dim, fe_degree> &problem,
                                  MPI_Comm &mpi_communicator,
                                  std::string &output_dir)
  {
    /*
     * Matrix-based runs use the same directory taxonomy as matrix-free runs.
     * The final component is fixed to 0 because this solver has no SIMD/non-SIMD
     * variant, but keeping the slot makes the test tree uniform for plotting.
     */
    std::filesystem::path save_dir =
        std::filesystem::path("tests") /
        "matrix_based" /
        // Problem name and refinement describe the numerical case.
        problem.problem_name /
        std::to_string(problem.refinement_level) /
        // MPI ranks and threads describe the parallel execution shape.
        std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator)) /
        std::to_string(MultithreadInfo::n_threads()) / // TODO: restore actual multithreading
        // The final slot is the SIMD flag in the matrix-free tree. Matrix-based
        // output uses 0 so both solvers keep the same directory depth.
        "0";

    create_saving_directory(save_dir, mpi_communicator, output_dir);
  }

  template <int dim, int fe_degree>
  void create_saving_directory_mf(ADR::ProblemData<dim, fe_degree> &problem,
                                  const bool &simd_flag,
                                  std::string &output_dir)
  {
    /*
     * Matrix-free runs split the output tree by SIMD setting so that plots and
     * summaries can compare vectorized and non-vectorized executions without
     * inspecting executable names.
     */
    std::filesystem::path save_dir =
        std::filesystem::path("tests") /
        "matrix_free" /
        // Problem name and refinement identify the numerical experiment.
        problem.problem_name /
        std::to_string(problem.refinement_level) /
        // MPI ranks and threads identify the parallel execution shape.
        std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) /
        std::to_string(MultithreadInfo::n_threads()) /
        // The last level distinguishes vectorized and non-vectorized runs.
        (simd_flag ? "1" : "0");

    create_saving_directory(save_dir, MPI_COMM_WORLD, output_dir);
  }

  inline void register_latest_run_output_dir(const std::string &output_dir)
  {
    const char *manifest_path = std::getenv("MFSOLVER_LATEST_RUN_MANIFEST");
    if (manifest_path == nullptr || std::string(manifest_path).empty())
      return;

    std::ofstream manifest_file(manifest_path, std::ios::app);
    AssertThrow(manifest_file,
                ExcMessage("Could not open latest-run manifest " +
                           std::string(manifest_path)));

    manifest_file << output_dir << '\n';
  }

  template <int dim, int fe_degree>
  void write_solver_log_file(const std::string &output_dir,
                             const ADR::ProblemData<dim, fe_degree> &problem,
                             const std::vector<std::vector<double>> &conv_history,
                             const double total_time,
                             const double l2_error,
                             const double h1_error,
                             const double linfty_error,
                             const bool converged)
  {
    LogStream deallog;
    std::ofstream log_file(output_dir + "/log.txt");

    // deallog keeps file writes serialized in the same spirit as the rest of
    // deal.II logging; the attached stream pins the destination to this run.
    deallog.attach(log_file, false);

    /*
     * The log is meant to be read by humans and by plots.py/summarize_tests.py.
     * The field widths make the table easy to scan, while the explicit spaces
     * are the real parsing contract: even if a value fills its whole field, the
     * next column still starts as a separate token.
     *
     * Rows are ended with std::endl instead of '\n'. For LogStream this is not
     * just a flush preference: std::endl tells deallog to emit the buffered row
     * as a complete log record, so the DEAL:: prefix is applied consistently to
     * every row.
     */
    constexpr unsigned int float_width = 14;
    constexpr unsigned int int_width = 8;
    constexpr unsigned int step_width = 10;
    constexpr unsigned int converged_width = 9;

    deallog << std::right
            // Fixed-width headers mirror the numeric rows below, so opening a
            // log by hand still gives a readable table.
            << std::setw(float_width) << "delta_t" << ' '
            << std::setw(int_width) << "max_iter" << ' '
            << std::setw(float_width) << "tol" << ' '
            << std::setw(step_width) << "rel_t_step" << ' '
            << std::setw(int_width) << "it_n" << ' '
            << std::setw(float_width) << "err" << ' '
            << std::setw(float_width) << "total_t" << ' '
            << std::setw(float_width) << "l2_error" << ' '
            << std::setw(float_width) << "h1_error" << ' '
            << std::setw(float_width) << "linfty_error" << ' '
            << std::setw(converged_width) << "converged" << std::endl;

    const auto write_history = [&](const double rel_t_step,
                                   const std::vector<double> &history) {
      /*
       * Each row describes one solver-control checkpoint for a representative
       * time position. The convergence flag is repeated on every row because it
       * is a run-level answer: the scripts can safely read the last row without
       * having to reconstruct solver state from the residual history.
       */
      for (size_t i = 0; i < history.size(); i++)
      {
        /*
         * Each insertion is followed by an explicit space. That space is not
         * decoration: it is what makes split()-based parsing reliable when a
         * scientific-notation value exactly fills its formatted width.
         *
         * std::endl completes the deallog record for this row. That keeps the
         * visual prefix column identical between the header and every data row,
         * which is what makes the log.txt table line up in a text editor.
         */
        deallog << std::right
                << std::setw(float_width) << std::setprecision(6) << std::scientific << problem.delta_t << ' '
                << std::setw(int_width) << problem.solver_max_iterations << ' '
                << std::setw(float_width) << std::setprecision(6) << std::scientific << problem.solver_tolerance_factor << ' '
                << std::setw(step_width) << std::fixed << std::setprecision(1) << rel_t_step << ' '
                << std::setw(int_width) << i << ' '
                << std::setw(float_width) << std::setprecision(6) << std::scientific << history[i] << ' '
                << std::setw(float_width) << std::setprecision(6) << std::scientific << total_time << ' '
                << std::setw(float_width) << std::setprecision(6) << std::scientific << l2_error << ' '
                << std::setw(float_width) << std::setprecision(6) << std::scientific << h1_error << ' '
                << std::setw(float_width) << std::setprecision(6) << std::scientific << linfty_error << ' '
                << std::setw(converged_width) << (converged ? 1 : 0) << std::endl;
      }
    };

    AssertThrow(!conv_history.empty(),
                ExcMessage("No solver convergence history is available"));

    write_history(0.0, conv_history[0]);

    if (conv_history.size() > 1)
    {
      /*
       * For time-dependent runs, the full history can be large. Logging the
       * first, middle, and last time slices keeps enough shape for summaries and
       * plots without turning log.txt into a second solution output file.
       */
      size_t mid_step = conv_history.size() / 2;
      write_history(0.5, conv_history[mid_step]);
    }

    if (conv_history.size() > 2)
    {
      // The last slice captures the final solver behavior and is the row most
      // scripts naturally inspect when they only need one status value.
      size_t last_step = conv_history.size() - 1;
      write_history(1.0, conv_history[last_step]);
    }

    // Every table row was already emitted with std::endl. Detaching is enough
    // here; an extra std::flush would create a prefix-only DEAL:: line because
    // LogStream treats the empty buffer as another log record.
    deallog.detach();
    log_file.close();

    register_latest_run_output_dir(output_dir);
  }
}
