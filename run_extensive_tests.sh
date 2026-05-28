#!/bin/bash

# -----------------------------------------------------------------------------
# run_extensive_tests.sh
# -----------------------------------------------------------------------------
#
# This script performs extensive testing over the various problems.
#
# Usage:
#   ./run_extensive_tests.sh [options]
#
# Options:
#   --solver <list>                     List of solvers to test (mb, mf).
#                                       Default: "all"
#   --problem <list>                    List of problems to test, space-separated or "all".
#                                       Default: "all"
#   --n_tests <int>                     Number of runs for each parameter combination.
#                                       Default: 5
#   --n_additional_refinements <list>   List of additional global mesh refinements (on top of problem default).
#                                       Default: "0 3 6"
#   --n_procs <list>                     List of numbers of processes to use.
#                                       Default: "1 2 4 8 16"
#   --n_threads <list>                  List of numbers of threads to use.
#                                       Default: "1 4 8"
#   --simd <list>                       List of values of SIMD to use.
#                                       Default: "0 1"
#   --delta_t <list>                    List of timesteps to use.
#                                       Default: 0 (i.e. problem default; this may vary across problems)
#   --max_iters <list>                  List of maximum iterations.
#                                       Default: "100"
#   --tol <list>                        List of tolerances (per time step).
#                                       Default: "1e-6 1-8 1e-10 1e-12"
#   --run-timeout-seconds <int>         Discard a single run if it lasts longer
#                                       than this many seconds. Use 0 to disable.
#                                       Default: 60
#   --use-scratch-local                 Write tests to /scratch_local and copy
#                                       the completed tests folder to
#                                       /scratch_global at the end.
#   --scratch-local-root <path>         Root used with --use-scratch-local.
#                                       Default: /scratch_local/$USER/<repo>
#   --scratch-global-root <path>        Copy destination root used with
#                                       --use-scratch-local.
#                                       Default: /scratch_global/$USER/<repo>
#   --help                              Show this help message.
#
# Output:
#   Test results are saved in the ./tests/ directory unless
#   --use-scratch-local is set.
#   A summary table is printed at the end showing aggregated statistics for each unique parameter combination.
#
# -----------------------------------------------------------------------------

# Default Parameters
SOLVER="mb mf"
PROBLEM="advanced lab_02 lab_03 parabolic transient mms"
N_TESTS=5
N_ADDITIONAL_REFINEMENTS="0 3 6"
N_PROCS="1 2"
N_THREADS="1 2 4 8"
SIMD="0 1"
DELTA_T="0"
MAX_ITERS="1000"
TOL="0.000001 0.00000001 0.0000000001 0.000000000001"
RUN_TIMEOUT_SECONDS=60
USE_SCRATCH_LOCAL=0
SCRATCH_LOCAL_ROOT=""
SCRATCH_GLOBAL_ROOT=""

# Function Definitions

show_help() {
    grep -E '^# ?' "$0" | cut -c3-
    exit 0
}

# Argument Parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver) shift; SOLVER=""; while [[ $# -gt 0 && "$1" != --* ]]; do SOLVER+="$1 "; shift; done;;
        --problem) shift; PROBLEM=""; while [[ $# -gt 0 && "$1" != --* ]]; do PROBLEM+="$1 "; shift; done;;
        --n_tests) N_TESTS="$2"; shift 2;; 
        --n_additional_refinements) shift; N_ADDITIONAL_REFINEMENTS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_ADDITIONAL_REFINEMENTS+="$1 "; shift; done;;
        --n_procs) shift; N_PROCS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_PROCS+="$1 "; shift; done;;
        --n_threads) shift; N_THREADS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_THREADS+="$1 "; shift; done;;
        --simd) shift; SIMD=""; while [[ $# -gt 0 && "$1" != --* ]]; do SIMD+="$1 "; shift; done;;
        --delta_t) shift; DELTA_T=""; while [[ $# -gt 0 && "$1" != --* ]]; do DELTA_T+="$1 "; shift; done;;
        --max_iters) shift; MAX_ITERS=""; while [[ $# -gt 0 && "$1" != --* ]]; do MAX_ITERS+="$1 "; shift; done;;
        --tol) shift; TOL=""; while [[ $# -gt 0 && "$1" != --* ]]; do TOL+="$1 "; shift; done;;
        --run-timeout-seconds) RUN_TIMEOUT_SECONDS="$2"; shift 2;;
        --use-scratch-local) USE_SCRATCH_LOCAL=1; shift;;
        --scratch-local-root) SCRATCH_LOCAL_ROOT="$2"; USE_SCRATCH_LOCAL=1; shift 2;;
        --scratch-global-root) SCRATCH_GLOBAL_ROOT="$2"; USE_SCRATCH_LOCAL=1; shift 2;;
        --help) show_help;; 
        *) 
            echo "Unknown parameter: $1"
            show_help;; 
    esac
done

if ! [[ "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Error: --run-timeout-seconds must be a non-negative integer" >&2
    exit 2
fi

if [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]] && ! command -v timeout >/dev/null 2>&1; then
    echo "Error: --run-timeout-seconds requires the timeout command" >&2
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

# Ensure the tests directory exists before truncating/creating the manifest.
mkdir -p "$TESTS_DIR"

# Start this extensive run with an empty manifest. The ':' command does
# nothing; the redirection is the useful part, truncating or creating the file.
: > "$LATEST_RUN_MANIFEST"

# Child processes inherit exported variables. The C++ output helpers read
# these paths when choosing the tests root and appending completed runs.
export MFSOLVER_TESTS_DIR="$TESTS_DIR"
export MFSOLVER_LATEST_RUN_MANIFEST="$LATEST_RUN_MANIFEST"

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
    local n_additional_refinements="$3"
    local n_procs="$4"
    local n_threads="$5"
    local simd="$6"
    local solver_dir

    if [[ "$solver" == "mf" ]]; then
        solver_dir="matrix_free"
    else
        solver_dir="matrix_based"
    fi

    printf '%s/%s/%s/%s/%s/%s/%s\n' \
        "$TESTS_DIR" \
        "$solver_dir" \
        "$problem" \
        "$n_additional_refinements" \
        "$n_procs" \
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

remove_manifest_entry() {
    # A completed run appends its test_N path to latest_run_tests.txt. If the
    # process is killed after writing that line, remove it so the summary sees
    # the timed-out run as if it had never happened.
    local discarded_dir="$1"
    local tmp_manifest

    [[ -f "$LATEST_RUN_MANIFEST" ]] || return 0

    tmp_manifest="$(mktemp "${LATEST_RUN_MANIFEST}.tmp.XXXXXX")" || return 1
    grep -Fxv "$discarded_dir" "$LATEST_RUN_MANIFEST" > "$tmp_manifest" || true
    mv "$tmp_manifest" "$LATEST_RUN_MANIFEST"
}

discard_new_test_dirs() {
    # Compare the parameter directory before and after the command. Any new
    # test_N directory belongs to the timed-out attempt and should be removed,
    # leaving possible holes such as test_0, test_2, test_4.
    local base_dir="$1"
    local before_file="$2"
    local candidate

    [[ -d "$base_dir" ]] || return 0

    while IFS= read -r candidate; do
        if ! grep -Fxq "$candidate" "$before_file"; then
            remove_manifest_entry "$candidate"
            rm -rf -- "$candidate"
        fi
    done < <(list_test_dirs "$base_dir")
}

run_solver_command() {
    # Run one physical solver attempt. Timeout exit codes are treated as
    # discarded samples, not failed experiments, so the surrounding sweep keeps
    # going and later averages only completed logs.
    local run_label="$1"
    local base_dir="$2"
    shift 2
    local before_file
    local status

    before_file="$(mktemp "${TMPDIR:-/tmp}/mfsolver-test-dirs-before.XXXXXX")" || return 1
    list_test_dirs "$base_dir" > "$before_file"

    if [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]]; then
        timeout --foreground -k 10s "${RUN_TIMEOUT_SECONDS}s" "$@"
    else
        "$@"
    fi
    status=$?

    if [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]] && [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
        discard_new_test_dirs "$base_dir" "$before_file"
        echo "Discarded $run_label: exceeded ${RUN_TIMEOUT_SECONDS}s"
        status=0
    fi

    rm -f "$before_file"
    return "$status"
}

for solver in $SOLVER; do
    echo "--- Solver: $solver ---"
    
    for problem in $PROBLEM; do
        for n_additional_refinements in $N_ADDITIONAL_REFINEMENTS; do
            for n_procs in $N_PROCS; do
                for n_threads in $N_THREADS; do
                    for simd in $SIMD; do
                        for delta_t in $DELTA_T; do
                            for max_iters in $MAX_ITERS; do
                                for tol in $TOL; do
                                    for ((i=0; i<N_TESTS; i++)); do
                                        echo "RUN $i: solver=$solver problem=$problem n_additional_refinements=$n_additional_refinements n_procs=$n_procs n_threads=$n_threads simd=$simd max_iters=$max_iters tol=$tol"

                                        if [[ "$solver" == "mf" ]]; then
                                            # MATRIX-FREE
                                            test_base_dir="$(test_base_dir_for_run "$solver" "$problem" "$n_additional_refinements" "$n_procs" "$n_threads" "$simd")"
                                            case "$simd" in
                                                0)  run_solver_command "run $i" "$test_base_dir" mpirun -n "$n_procs" ./matrix_free_no_simd "$n_threads" "$simd" "$problem" "$n_additional_refinements" "$delta_t" "$max_iters" "$tol";;
                                                1)  run_solver_command "run $i" "$test_base_dir" mpirun -n "$n_procs" ./matrix_free_simd "$n_threads" "$simd" "$problem" "$n_additional_refinements" "$delta_t" "$max_iters" "$tol";; 
                                                *) echo "Unknown simd value: $simd";;
                                            esac
                                        else
                                            # MATRIX-BASED
                                            case "$simd" in
                                                0)
                                                    test_base_dir="$(test_base_dir_for_run "$solver" "$problem" "$n_additional_refinements" "$n_procs" "$n_threads" "0")"
                                                    run_solver_command "run $i" "$test_base_dir" mpirun -n "$n_procs" ./matrix_based "$n_threads" "$problem" "$n_additional_refinements" "$delta_t" "$max_iters" "$tol";;
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
