#!/bin/bash

set -euo pipefail

DEFAULT_CONTAINER="${DEFAULT_CONTAINER:-$HOME/amsc_mk_2025.sif}"
AVX512_CONTAINER="${AVX512_CONTAINER:-$HOME/dealii-avx512.sif}"
N_BUILD_JOBS="${N_BUILD_JOBS:-4}"
export N_BUILD_JOBS

show_help() {
    echo "Compile the three solver executables in their intended containers.

Usage:
  ./compile_locally.sh [options]

Options:
  --default_container <path>   Container used for matrix_based and
                               matrix_free_no_simd.
                               Default: \$DEFAULT_CONTAINER or ~/amsc_mk_2025.sif
  --avx512_container <path>    Container used for matrix_free_simd.
                               Default: \$AVX512_CONTAINER or ~/dealii-avx512.sif
  --n_build_jobs <int>         Parallel build jobs passed to cmake --build.
                               Default: \$N_BUILD_JOBS or 4
  --help                       Show this help message.

The compiled executables are written to the repository root:
  ./matrix_based
  ./matrix_free_no_simd
  ./matrix_free_simd"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --default_container)
            DEFAULT_CONTAINER="$2"
            shift 2
            ;;
        --avx512_container)
            AVX512_CONTAINER="$2"
            shift 2
            ;;
        --n_build_jobs)
            N_BUILD_JOBS="$2"
            export N_BUILD_JOBS
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            show_help
            ;;
    esac
done

if ! [[ "$N_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --n_build_jobs must be a positive integer" >&2
    exit 2
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "Error: compile_locally.sh requires the apptainer command" >&2
    exit 2
fi

if [[ ! -f "$DEFAULT_CONTAINER" ]]; then
    echo "Error: default container not found: $DEFAULT_CONTAINER" >&2
    exit 2
fi

if [[ ! -f "$AVX512_CONTAINER" ]]; then
    echo "Warning: AVX512 container not found at: $AVX512_CONTAINER" >&2
    echo "Skipping compilation of matrix_free_simd." >&2
    BUILD_AVX512=false
else
    BUILD_AVX512=true
fi

echo "Building matrix_based and matrix_free_no_simd in:"
echo "  $DEFAULT_CONTAINER"

APPTAINERENV_N_BUILD_JOBS=$N_BUILD_JOBS apptainer exec --cleanenv "$DEFAULT_CONTAINER" bash -lc '
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    if [[ -r /u/sw/lmod/8.5.8/init/bash ]]; then
        source /u/sw/lmod/8.5.8/init/bash
    elif [[ -r /u/sw/etc/bash.bashrc ]]; then
        source /u/sw/etc/bash.bashrc
    fi
fi

# Local non-interactive shells may have the module function but not the module
# search path. Without this, Lmod reports MODULEPATH as undefined and cannot
# find the gcc/deal.II modulefiles.
export MODULEPATH="${MODULEPATH:-/u/sw/modules}"

module load toolchains/gcc-glibc/11.2.0 2>/dev/null || module load gcc-glibc
module load dealii

cmake -S . -B build/baseline \
    -DMPI_C_COMPILER=$(command -v mpicc) \
    -DMPI_CXX_COMPILER=$(command -v mpicxx) \
    -DMFSOLVER_BUILD_MATRIX_BASED=ON \
    -DMFSOLVER_BUILD_MATRIX_FREE_NO_SIMD=ON \
    -DMFSOLVER_BUILD_MATRIX_FREE_SIMD=OFF

cmake --build build/baseline -j "$N_BUILD_JOBS" --target matrix_based matrix_free_no_simd
'

if [[ "$BUILD_AVX512" = true ]]; then
    echo "Building matrix_free_simd in:"
    echo "  $AVX512_CONTAINER"

    # The AVX512 image is self-contained and intentionally does not use modules.
    APPTAINERENV_N_BUILD_JOBS=$N_BUILD_JOBS apptainer exec --cleanenv "$AVX512_CONTAINER" bash -lc '
    set -euo pipefail

    cmake -S . -B build/avx512 \
        -DMPI_C_COMPILER=$(command -v mpicc) \
        -DMPI_CXX_COMPILER=$(command -v mpicxx) \
        -DMFSOLVER_BUILD_MATRIX_BASED=OFF \
        -DMFSOLVER_BUILD_MATRIX_FREE_NO_SIMD=OFF \
        -DMFSOLVER_BUILD_MATRIX_FREE_SIMD=ON

    cmake --build build/avx512 -j "$N_BUILD_JOBS" --target matrix_free_simd
    '
fi

echo "Build complete:"
echo "  ./matrix_based"
echo "  ./matrix_free_no_simd"
if [[ "$BUILD_AVX512" = true ]]; then
    echo "  ./matrix_free_simd"
fi

