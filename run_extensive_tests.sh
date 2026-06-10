#!/bin/bash

# Default Parameters
SOLVER="mb mf"
PROBLEM="advanced lab_02 lab_03 parabolic transient mms"
FE_DEG="2"
N_TESTS=5
N_ADDITIONAL_REFINEMENTS="0 3 6"
N_RANKS="1 2"
N_THREADS="1 2 4 8"
SIMD="0 1"
DELTA_T="0"
MAX_ITERS="1000"
TOL="0.000001 0.00000001 0.0000000001 0.000000000001"
RUN_TIMEOUT_SECONDS=60
USE_SCRATCH_LOCAL=0
SCRATCH_LOCAL_ROOT=""
SCRATCH_GLOBAL_ROOT=""
DEFAULT_CONTAINER="$HOME/amsc_mk_2025.sif"
AVX512_CONTAINER="$HOME/dealii-avx512.sif"
APPTAINER_BIND=""

# Function Definitions

show_help() {
    echo "This script performs extensive testing over the various problems.

Usage:
  ./run_extensive_tests.sh [options]

Options:
  --solver <list>                     List of solvers to test (mb, mf).
                                      Default: "all"
  --problem <list>                    List of problems to test, space-separated or "all".
                                      Default: "all"
  --fe_deg <list>                     List of finite-element polynomial degrees to use.
                                      Default: "2"
  --n_tests <int>                     Number of runs for each parameter combination.
                                      Default: 5
  --n_additional_refinements <list>   List of additional global mesh refinements (on top of problem default).
                                      Default: "0 3 6"
  --n_ranks <list>                    List of MPI rank counts to use.
                                      Default: "1 2"
  --n_threads <list>                  List of numbers of threads to use.
                                      Default: "1 4 8"
  --simd <list>                       List of values of SIMD to use.
                                      Default: "0 1"
  --delta_t <list>                    List of timesteps to use.
                                      Default: 0 (i.e. problem default; this may vary across problems)
  --max_iters <list>                  List of maximum iterations.
                                      Default: "100"
  --tol <list>                        List of tolerances (per time step).
                                      Default: "1e-6 1-8 1e-10 1e-12"
  --run_timeout_seconds <int>         Mark a single run as overtime if it lasts
                                      longer than this many seconds. Use 0 to disable.
                                      Default: 60
  --use_scratch_local                 Write tests to /scratch_local and copy
                                      the completed tests folder to
                                      /scratch_global at the end.
  --scratch_local_root <path>         Root used with --use_scratch_local.
                                      Default: /scratch_local/$USER/<repo>
  --scratch_global_root <path>        Copy destination root used with
                                      --use_scratch_local.
                                      Default: /scratch_global/$USER/<repo>
  --default_container <path>          Container for matrix_based and
                                      matrix_free with SIMD=0.
                                      Default: ~/amsc_mk_2025.sif
  --avx512_container <path>           Container for matrix_free with SIMD=1.
                                      Default: ~/dealii-avx512.sif
  --apptainer_bind <paths>            Comma-separated bind paths passed to
                                      apptainer exec, e.g.
                                      /scratch_local,/scratch_global.
  --help                              Show this help message.

Output:
  Test results are saved in the ./tests/ directory unless --use_scratch_local is set.
  A summary table is printed at the end showing aggregated statistics for each unique parameter combination.
  Results saved under:
  tests/<solver>/<problem>/<fe_deg>/<n_add_ref>/<n_ranks>/<n_threads>/<simd>/test_N"
  exit
}

# Argument Parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver) shift; SOLVER=""; while [[ $# -gt 0 && "$1" != --* ]]; do SOLVER+="$1 "; shift; done;;
        --problem) shift; PROBLEM=""; while [[ $# -gt 0 && "$1" != --* ]]; do PROBLEM+="$1 "; shift; done;;
        --fe_deg) shift; FE_DEG=""; while [[ $# -gt 0 && "$1" != --* ]]; do FE_DEG+="$1 "; shift; done;;
        --n_tests) N_TESTS="$2"; shift 2;; 
        --n_additional_refinements) shift; N_ADDITIONAL_REFINEMENTS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_ADDITIONAL_REFINEMENTS+="$1 "; shift; done;;
        --n_ranks) shift; N_RANKS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_RANKS+="$1 "; shift; done;;
        --n_threads) shift; N_THREADS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_THREADS+="$1 "; shift; done;;
        --simd) shift; SIMD=""; while [[ $# -gt 0 && "$1" != --* ]]; do SIMD+="$1 "; shift; done;;
        --delta_t) shift; DELTA_T=""; while [[ $# -gt 0 && "$1" != --* ]]; do DELTA_T+="$1 "; shift; done;;
        --max_iters) shift; MAX_ITERS=""; while [[ $# -gt 0 && "$1" != --* ]]; do MAX_ITERS+="$1 "; shift; done;;
        --tol) shift; TOL=""; while [[ $# -gt 0 && "$1" != --* ]]; do TOL+="$1 "; shift; done;;
        --run_timeout_seconds) RUN_TIMEOUT_SECONDS="$2"; shift 2;;
        --use_scratch_local) USE_SCRATCH_LOCAL=1; shift;;
        --scratch_local_root) SCRATCH_LOCAL_ROOT="$2"; USE_SCRATCH_LOCAL=1; shift 2;;
        --scratch_global_root) SCRATCH_GLOBAL_ROOT="$2"; USE_SCRATCH_LOCAL=1; shift 2;;
        --default_container) DEFAULT_CONTAINER="$2"; shift 2;;
        --avx512_container) AVX512_CONTAINER="$2"; shift 2;;
        --apptainer_bind) APPTAINER_BIND="$2"; shift 2;;
        --help) show_help;; 
        *) 
            echo "Unknown parameter: $1"
            show_help;; 
    esac
done

if ! [[ "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Error: --run_timeout_seconds must be a non-negative integer" >&2
    exit 2
fi

if [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]] && ! command -v timeout >/dev/null 2>&1; then
    echo "Error: --run_timeout_seconds requires the timeout command" >&2
    exit 2
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "Error: run_extensive_tests.sh requires the apptainer command" >&2
    exit 2
fi

if [[ ! -f "$DEFAULT_CONTAINER" ]]; then
    echo "Error: default container not found: $DEFAULT_CONTAINER" >&2
    exit 2
fi

if [[ ! -f "$AVX512_CONTAINER" ]]; then
    echo "Error: AVX512 container not found: $AVX512_CONTAINER" >&2
    exit 2
fi

# Run tests
echo "=== Running tests ==="

# Absolute path to the directory containing this script. This keeps the
# manifest location stable even if the script is launched from another folder.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$SCRIPT_DIR")"
USER_NAME="${USER:-$(id -un)}"

if [[ "$USE_SCRATCH_LOCAL" -eq 1 ]]; then
    if [[ -z "$SCRATCH_LOCAL_ROOT" ]]; then
        SCRATCH_LOCAL_ROOT="/scratch_local/$USER_NAME/$REPO_NAME"
    fi

    if [[ -z "$SCRATCH_GLOBAL_ROOT" ]]; then
        SCRATCH_GLOBAL_ROOT="/scratch_global/$USER_NAME/$REPO_NAME"
    fi

    TESTS_DIR="$SCRATCH_LOCAL_ROOT/tests"
else
    TESTS_DIR="$SCRIPT_DIR/tests"
fi

copy_scratch_results() {
    # Capture the status that made the script exit. The EXIT trap runs this
    # function after normal completion and after failures, so we preserve that
    # original status and return it at the end.
    local exit_status=$?
    local copy_status=0

    # In the default mode, results already live in the repository tests/
    # directory. Only the scratch-local mode needs a final transfer.
    if [[ "$USE_SCRATCH_LOCAL" -ne 1 ]]; then
        exit "$exit_status"
    fi

    echo ""
    echo "Copying scratch-local tests to scratch_global..."
    echo "  from: $TESTS_DIR"
    echo "  to:   $SCRATCH_GLOBAL_ROOT/tests"

    mkdir -p "$SCRATCH_GLOBAL_ROOT"

    # Prefer rsync when available because it handles existing destination
    # trees cleanly. Fall back to cp for simpler cluster/container images.
    # The || command is an or command. copy_status=$? reports the status
    # of the latest command; since || runs the right argument only if the 
    # left fails, this essentially sets copy_status to the error of rsync.
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$TESTS_DIR" "$SCRATCH_GLOBAL_ROOT/" || copy_status=$?
    else
        cp -a "$TESTS_DIR" "$SCRATCH_GLOBAL_ROOT/" || copy_status=$?
    fi

    # If the solver run succeeded but the copy failed, make the script fail so
    # the PBS job reports the transfer problem. If the solver already failed,
    # keep the original failure status.
    if [[ "$copy_status" -ne 0 ]]; then
        echo "Error: copy to scratch_global failed with status $copy_status" >&2
        if [[ "$exit_status" -eq 0 ]]; then
            exit_status="$copy_status"
        fi
    fi

    # End the script with the correct status. Returning from an EXIT trap can be
    # subtle, so this function exits explicitly.
    exit "$exit_status"
}

trap copy_scratch_results EXIT

# File used as the manifest for this batch. Each completed solver run appends
# its test_N output directory here.
LATEST_RUN_MANIFEST="$TESTS_DIR/latest_run_tests.txt"
LATEST_RUN_MANIFEST="$TESTS_DIR/latest_run_tests.txt"

# Ensure the tests directory exists before truncating/creating the manifest.
mkdir -p "$TESTS_DIR"
mkdir -p "$TESTS_DIR"

# Start this extensive run with an empty manifest. The ':' command does
# nothing; the redirection is the useful part, truncating or creating the file.
: > "$LATEST_RUN_MANIFEST"

# Child processes inherit exported variables. The C++ output helpers read
# these paths when choosing the tests root and appending completed runs.
export MFSOLVER_TESTS_DIR="$TESTS_DIR"
export MFSOLVER_LATEST_RUN_MANIFEST="$LATEST_RUN_MANIFEST"
export MFSOLVER_DISABLE_VTU=1

if [[ "$USE_SCRATCH_LOCAL" -eq 1 ]]; then
    export MFSOLVER_SCRATCH_GLOBAL_ROOT="$SCRATCH_GLOBAL_ROOT"
    export MFSOLVER_SCRATCH_GLOBAL_TESTS_DIR="$SCRATCH_GLOBAL_ROOT/tests"
fi


test_base_dir_for_run() {
    # Reconstruct the exact parameter directory where this solver invocation
    # will create its test_N folder. On timeout, cleanup is scoped to this
    # directory so unrelated runs with other settings are never touched.
    local solver="$1"
    local problem="$2"
    local fe_deg="$3"
    local n_additional_refinements="$4"
    local n_ranks="$5"
    local n_threads="$6"
    local simd="$7"
    local solver_dir

    if [[ "$solver" == "mf" ]]; then
        solver_dir="matrix_free"
    else
        solver_dir="matrix_based"
    fi

    # Directory hierarchy:
    # tests/<solver>/<problem>/<fe_deg>/<n_add_ref>/<n_ranks>/<n_threads>/<simd>
    # FE degree is placed above refinement and parallel settings because it is
    # part of the discretization, not a repetition or execution-shape detail.
    printf '%s/%s/%s/%s/%s/%s/%s/%s\n' \
        "$TESTS_DIR" \
        "$solver_dir" \
        "$problem" \
        "$fe_deg" \
        "$n_additional_refinements" \
        "$n_ranks" \
        "$n_threads" \
        "$simd"
}

list_test_dirs() {
    # Only real test_<integer> directories count as solver attempts. This
    # mirrors the C++ naming convention and ignores helper files or backups.
    local base_dir="$1"
    local candidate
    local name

    [[ -d "$base_dir" ]] || return 0

    for candidate in "$base_dir"/test_*; do
        [[ -d "$candidate" ]] || continue
        name="$(basename "$candidate")"
        [[ "$name" =~ ^test_[0-9]+$ ]] || continue
        printf '%s\n' "$candidate"
    done
}

list_new_test_dirs() {
    # Compare the parameter directory before and after the command. Any new
    # test_N directory belongs to the just-finished attempt.
    local base_dir="$1"
    local before_file="$2"
    local candidate

    [[ -d "$base_dir" ]] || return 0

    while IFS= read -r candidate; do
        if ! grep -Fxq "$candidate" "$before_file"; then
            printf '%s\n' "$candidate"
        fi
    done < <(list_test_dirs "$base_dir")
}

join_lines_with_commas() {
    local first=1
    local line

    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        if [[ "$first" -eq 1 ]]; then
            printf '%s' "$line"
            first=0
        else
            printf ',%s' "$line"
        fi
    done
}

save_command_output_log() {
    local base_dir="$1"
    local new_dirs="$2"
    local command_output_file="$3"
    local run_label="$4"
    local status_kind="$5"
    local saved_log_path=""
    local run_dir
    local safe_label
    local fallback_dir

    safe_label="${run_label//[^[:alnum:]_.-]/_}"

    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        cp "$command_output_file" "$run_dir/runner_output.log"
        saved_log_path="$run_dir/runner_output.log"
    done <<< "$new_dirs"

    if [[ -z "$saved_log_path" && "$status_kind" != "ok" ]]; then
        fallback_dir="$base_dir/_attempt_logs"
        mkdir -p "$fallback_dir"
        saved_log_path="$fallback_dir/${safe_label}_$(date +%Y%m%dT%H%M%S%N).log"
        cp "$command_output_file" "$saved_log_path"
    fi

    printf '%s\n' "$saved_log_path"
}

manifest_solver_name() {
    local solver="$1"

    if [[ "$solver" == "mf" ]]; then
        printf '%s\n' "matrix_free"
    else
        printf '%s\n' "matrix_based"
    fi
}

record_attempt_manifest_entry() {
    local status_kind="$1"
    local solver="$2"
    local problem="$3"
    local fe_deg="$4"
    local n_additional_refinements="$5"
    local n_ranks="$6"
    local n_threads="$7"
    local simd="$8"
    local delta_t="$9"
    shift 9
    local max_iters="$1"
    local tol="$2"
    local exit_code="$3"
    local output_dirs="$4"
    local log_path="$5"

    printf 'ATTEMPT status=%s solver=%s problem=%s fe_deg=%s n_add_ref=%s n_ranks=%s n_threads=%s simd=%s delta_t=%s max_iters=%s tol=%s exit_code=%s output_dirs=%s log=%s\n' \
        "$status_kind" \
        "$(manifest_solver_name "$solver")" \
        "$problem" \
        "$fe_deg" \
        "$n_additional_refinements" \
        "$n_ranks" \
        "$n_threads" \
        "$simd" \
        "$delta_t" \
        "$max_iters" \
        "$tol" \
        "$exit_code" \
        "$output_dirs" \
        "$log_path" >> "$LATEST_RUN_MANIFEST"
}

run_solver_command() {
    # Execute exactly one solver command, optionally under timeout.
    #
    # Arguments:
    #   $1 = human-readable label printed on timeout, e.g. "run 0"
    #   $2 = parameter directory where this command may create test_N folders
    #   $3... = command to execute, including all its arguments
    #
    # The C++ output helper creates a new test_N directory during the run.
    # Timeout and failure cases are still useful evidence, so this function
    # snapshots the existing test_N directories before launching the command,
    # keeps any new ones, and saves stdout/stderr into them when possible.
    local run_label="$1"
    local base_dir="$2"
    shift 2
    local before_file
    local command_output_file
    local new_dirs
    local status_kind
    local saved_log_path
    local status

    RUN_SOLVER_COMMAND_STATUS="fail"
    RUN_SOLVER_COMMAND_EXIT_CODE=1
    RUN_SOLVER_COMMAND_OUTPUT_DIRS=""
    RUN_SOLVER_COMMAND_LOG_PATH=""

    # Store the pre-run directory list in a temporary file instead of a Bash
    # array. This keeps the comparison simple and avoids quoting problems with
    # paths when the test root is under scratch.
    before_file="$(mktemp "${TMPDIR:-/tmp}/mfsolver-test-dirs-before.XXXXXX")" || return 1
    list_test_dirs "$base_dir" > "$before_file"
    command_output_file="$(mktemp "${TMPDIR:-/tmp}/mfsolver-run-output.XXXXXX")" || {
        rm -f "$before_file"
        return 1
    }

    # timeout --foreground lets MPI children receive terminal-related signals
    # correctly. The -k grace period sends SIGKILL if the process group does
    # not exit after the initial timeout signal.
    if [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]]; then
        timeout --foreground -k 10s "${RUN_TIMEOUT_SECONDS}s" "$@" > "$command_output_file" 2>&1
    else
        "$@" > "$command_output_file" 2>&1
    fi
    status=$?
    cat "$command_output_file"

    status_kind="ok"

    # GNU timeout normally returns 124. MPI launchers may instead surface 137
    # or 143 when the killed process reports SIGKILL/SIGTERM. Treat all three
    # as overtime attempts: preserve partial output, print a note, and return 0
    # so the sweep continues with the remaining parameter combinations.
    if [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]] && [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
        status_kind="overtime"
        echo "Overtime $run_label: exceeded ${RUN_TIMEOUT_SECONDS}s"
    elif [[ "$status" -ne 0 ]]; then
        status_kind="fail"
    fi

    new_dirs="$(list_new_test_dirs "$base_dir" "$before_file")"
    saved_log_path="$(save_command_output_log "$base_dir" "$new_dirs" "$command_output_file" "$run_label" "$status_kind")"

    RUN_SOLVER_COMMAND_STATUS="$status_kind"
    RUN_SOLVER_COMMAND_EXIT_CODE="$status"
    RUN_SOLVER_COMMAND_OUTPUT_DIRS="$(printf '%s\n' "$new_dirs" | join_lines_with_commas)"
    RUN_SOLVER_COMMAND_LOG_PATH="$saved_log_path"

    rm -f "$before_file"
    rm -f "$command_output_file"
    return 0
}

container_for_run() {
    # Only matrix-free SIMD=1 runs belong in the AVX512 image. Matrix-based
    # has no SIMD variant in this benchmark tree, and matrix-free SIMD=0 must
    # stay in the baseline image to keep those timings comparable.
    local solver="$1"
    local simd="$2"

    if [[ "$solver" == "mf" && "$simd" == "1" ]]; then
        printf '%s\n' "$AVX512_CONTAINER"
    else
        printf '%s\n' "$DEFAULT_CONTAINER"
    fi
}

run_solver_command_for_variant() {
    # Run a solver command through the Apptainer image selected for this
    # solver/SIMD pair.
    #
    # Arguments:
    #   $1 = human-readable run label
    #   $2 = parameter directory used by run_solver_command for cleanup
    #   $3 = solver short name: "mb" or "mf"
    #   $4 = SIMD flag for this run: "0" or "1"
    #   $5... = solver command to run inside the selected environment
    local run_label="$1"
    local base_dir="$2"
    local solver="$3"
    local simd="$4"
    shift 4

    # container_for_run encodes the policy: matrix_free SIMD=1 -> AVX512
    # image; every other run -> baseline image.
    local container
    container="$(container_for_run "$solver" "$simd")"
    echo "  container: $container"

    # Build Apptainer arguments as an array so an optional bind string remains
    # one argument even if it contains commas or paths with shell-sensitive
    # characters.
    local apptainer_args=(exec)
    if [[ -n "$APPTAINER_BIND" ]]; then
        apptainer_args+=(--bind "$APPTAINER_BIND")
    fi

    local container_script

    if [[ "$solver" == "mf" && "$simd" == "1" ]]; then
        # The AVX512 image is expected to be self-contained: its compiler/MPI
        # and deal.II runtime paths are already part of the image environment.
        # Do not call module here, because that container may not ship the
        # cluster module command at all.
        container_script='
set -euo pipefail
exec "$@"
'
    else
        # The baseline image follows the original cluster setup and needs the
        # deal.II module environment before running matrix_based or
        # matrix_free_no_simd.
        container_script='
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
    if [[ -r /u/sw/lmod/8.5.8/init/bash ]]; then
        source /u/sw/lmod/8.5.8/init/bash
    fi
fi
export MODULEPATH="${MODULEPATH:-/u/sw/modules}"
module load toolchains/gcc-glibc/11.2.0 2>/dev/null || module load gcc-glibc
module load dealii
exec "$@"
'
    fi

    # The outer run_solver_command still owns timeout handling and cleanup.
    # Inside the container, bash -lc receives the setup script, "_" becomes
    # bash's $0, and the original solver command becomes "$@" for the final
    # exec line in container_script.
    run_solver_command "$run_label" "$base_dir" \
        apptainer "${apptainer_args[@]}" "$container" \
        bash -lc "$container_script" _ "$@"
}

run_solver_attempt() {
    local run_label="$1"
    local base_dir="$2"
    local solver="$3"
    local simd="$4"
    local problem="$5"
    local fe_deg="$6"
    local n_additional_refinements="$7"
    local n_ranks="$8"
    local n_threads="$9"
    local delta_t="${10}"
    local max_iters="${11}"
    local tol="${12}"
    shift 12

    run_solver_command_for_variant "$run_label" "$base_dir" "$solver" "$simd" "$@"

    record_attempt_manifest_entry \
        "$RUN_SOLVER_COMMAND_STATUS" \
        "$solver" \
        "$problem" \
        "$fe_deg" \
        "$n_additional_refinements" \
        "$n_ranks" \
        "$n_threads" \
        "$simd" \
        "$delta_t" \
        "$max_iters" \
        "$tol" \
        "$RUN_SOLVER_COMMAND_EXIT_CODE" \
        "$RUN_SOLVER_COMMAND_OUTPUT_DIRS" \
        "$RUN_SOLVER_COMMAND_LOG_PATH"
}

for solver in $SOLVER; do
    echo "--- Solver: $solver ---"
    
    for problem in $PROBLEM; do
        for fe_deg in $FE_DEG; do
            for n_additional_refinements in $N_ADDITIONAL_REFINEMENTS; do
                for n_ranks in $N_RANKS; do
                    for n_threads in $N_THREADS; do
                        for simd in $SIMD; do
                            for delta_t in $DELTA_T; do
                                for max_iters in $MAX_ITERS; do
                                    for tol in $TOL; do
                                        for ((i=0; i<N_TESTS; i++)); do
                                            echo "RUN $i: solver=$solver problem=$problem fe_deg=$fe_deg n_additional_refinements=$n_additional_refinements n_ranks=$n_ranks n_threads=$n_threads simd=$simd max_iters=$max_iters tol=$tol"

                                            if [[ "$solver" == "mf" ]]; then
                                                # MATRIX-FREE
                                                test_base_dir="$(test_base_dir_for_run "$solver" "$problem" "$fe_deg" "$n_additional_refinements" "$n_ranks" "$n_threads" "$simd")"
                                                case "$simd" in
                                                    0)  run_solver_attempt "run $i" "$test_base_dir" "$solver" "$simd" "$problem" "$fe_deg" "$n_additional_refinements" "$n_ranks" "$n_threads" "$delta_t" "$max_iters" "$tol" mpirun --report-bindings -n "$n_ranks" --bind-to core --map-by package:PE="$n_threads" ./matrix_free_no_simd "$n_threads" "$simd" "$problem" "$fe_deg" "$n_additional_refinements" "$delta_t" "$max_iters" "$tol";;
                                                    1)  run_solver_attempt "run $i" "$test_base_dir" "$solver" "$simd" "$problem" "$fe_deg" "$n_additional_refinements" "$n_ranks" "$n_threads" "$delta_t" "$max_iters" "$tol" mpirun --report-bindings -n "$n_ranks" --bind-to core --map-by package:PE="$n_threads" ./matrix_free_simd "$n_threads" "$simd" "$problem" "$fe_deg" "$n_additional_refinements" "$delta_t" "$max_iters" "$tol";;
                                                    *) echo "Unknown simd value: $simd";;
                                                esac
                                            else
                                                # MATRIX-BASED
                                                case "$simd" in
                                                    0)
                                                        test_base_dir="$(test_base_dir_for_run "$solver" "$problem" "$fe_deg" "$n_additional_refinements" "$n_ranks" "$n_threads" "0")"
                                                        run_solver_attempt "run $i" "$test_base_dir" "$solver" "0" "$problem" "$fe_deg" "$n_additional_refinements" "$n_ranks" "$n_threads" "$delta_t" "$max_iters" "$tol" mpirun --report-bindings -n "$n_ranks" --bind-to core --map-by package:PE="$n_threads" ./matrix_based "$n_threads" "$problem" "$fe_deg" "$n_additional_refinements" "$delta_t" "$max_iters" "$tol";;
                                                    1)  continue;;
                                                    *) echo "Unknown simd value: $simd";;
                                                esac
                                            fi
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
    echo ""
done

echo "All tests completed."

# Generate summary table
echo ""
if [[ -s "$LATEST_RUN_MANIFEST" ]]; then
    echo "Generating test summary..."
    python3 "$SCRIPT_DIR/summarize_tests.py" "$TESTS_DIR" --latest --manifest "$LATEST_RUN_MANIFEST" 2>/dev/null || echo "Error: Summary generation failed"
else
    echo "No completed test runs to summarize."
fi
