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
#   --solvers <list>                    List of solvers to test (mb, mf).
#                                       Default: "all"
#   --problems <list>                   List of problems to test, space-separated or "all".
#                                       Default: "all"
#   --n_tests <int>                     Number of runs for each parameter combination.
#                                       Default: 5
#   --n_additional_refinements <list>   List of additional global mesh refinements (on top of problem default).
#                                       Default: "0 3 6"
#   --n_proc <list>                     List of numbers of processes to use.
#                                       Default: "1 2 4 8 16"
#   --n_threads <list>                  List of numbers of threads to use.
#                                       Default: "1 4 8"
#   --simd <list>                       List of values of SIMD to use.
#                                       Default: "0 1"
#   --max_iters <list>                  List of maximum iterations.
#                                       Default: "100"
#   --max_err <list>                    List of maximum error (per time step).
#                                       Default: "1e-6 1-8 1e-10 1e-12"
#   --help                              Show this help message.
#
# -----------------------------------------------------------------------------

# Default Parameters
SOLVERS="mb mf"
PROBLEMS="advanced lab_02 lab_03 parabolic transient"
N_TESTS=5
N_ADDITIONAL_REFINEMENTS="0 3 6"
N_PROC="1 2 4 8 16"
N_THREADS="1 4 8"
SIMD="0 1"
MAX_ITERS="100"
MAX_ERR="0.000001 0.00000001 0.0000000001 0.000000000001"

# Function Definitions

show_help() {
    grep -E '^# ?' "$0" | cut -c3-
    exit 0
}

# Argument Parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solvers) shift; SOLVERS=""; while [[ $# -gt 0 && "$1" != --* ]]; do SOLVERS+="$1 "; shift; done;;
        --problems) shift; PROBLEMS=""; while [[ $# -gt 0 && "$1" != --* ]]; do PROBLEMS+="$1 "; shift; done;;
        --n_tests) N_TESTS="$2"; shift 2;; 
        --n_additional_refinements) shift; N_ADDITIONAL_REFINEMENTS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_ADDITIONAL_REFINEMENTS+="$1 "; shift; done;;
        --n_proc) shift; N_PROC=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_PROC+="$1 "; shift; done;;
        --n_threads) shift; N_THREADS=""; while [[ $# -gt 0 && "$1" != --* ]]; do N_THREADS+="$1 "; shift; done;;
        --simd) shift; SIMD=""; while [[ $# -gt 0 && "$1" != --* ]]; do SIMD+="$1 "; shift; done;;
        --max_iters) shift; MAX_ITERS=""; while [[ $# -gt 0 && "$1" != --* ]]; do MAX_ITERS+="$1 "; shift; done;;
        --max_err) shift; MAX_ERR=""; while [[ $# -gt 0 && "$1" != --* ]]; do MAX_ERR+="$1 "; shift; done;;
        --help) show_help;; 
        *) 
            echo "Unknown parameter: $1"
            show_help;; 
    esac
done

# Run tests
echo "=== Running tests ==="

for solver in $SOLVERS; do
    echo "--- Solver: $solver ---"
    
    for problem in $PROBLEMS; do
        for n_additional_refinements in $N_ADDITIONAL_REFINEMENTS; do
            for n_proc in $N_PROC; do
                for n_threads in $N_THREADS; do
                    for simd in $SIMD; do
                        for max_iters in $MAX_ITERS; do
                            for max_err in $MAX_ERR; do
                                for ((i=0; i<N_TESTS; i++)); do
                                    echo "RUN $i: solver=$solver problem=$problem n_additional_refinements=$n_additional_refinements n_proc=$n_proc n_threads=$n_threads simd=$simd max_iters=$max_iters max_err=$max_err"

                                    if [[ "$solver" == "mf" ]]; then
                                        # MATRIX-FREE
                                        case "$simd" in
                                            0)  mpirun -n "$n_proc" ./matrix_free_no_simd "$n_threads" "$simd" "$problem" "$n_additional_refinements" "$max_iters" "$max_err";;
                                            1)  mpirun -n "$n_proc" ./matrix_free_simd "$n_threads" "$simd" "$problem" "$n_additional_refinements" "$max_iters" "$max_err";; 
                                            *) echo "Unknown simd value: $simd";;
                                        esac
                                    else
                                        # MATRIX-BASED
                                        case "$simd" in
                                            0)  mpirun -n "$n_proc" ./matrix_based "$n_threads" "$problem" "$n_additional_refinements" "$max_iters" "$max_err";;
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
    echo ""
done

echo "All tests completed."
