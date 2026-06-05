#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=8
#PBS -l place=scatter
#PBS -l walltime=10:00:00
#PBS -q cpu
#PBS -N test
#PBS -j oe
#PBS -o pbs_jobs/test.pbs.out

# Always run from the directory where you submitted the job
cd "$PBS_O_WORKDIR" || exit 1

set -euo pipefail

LOG_FILE="$PWD/pbs_jobs/last_output.txt"
mkdir -p "$(dirname "$LOG_FILE")"



# Capture the whole PBS job, not only run_extensive_tests.sh. This makes early
# failures in apptainer, module loading, cmake, or make visible in last_output.
exec > >(tee "$LOG_FILE") 2>&1

echo "PBS job started on $(hostname) at $(date)"
# Checking the number of nodes, cores, hardware threads
cat $PBS_NODEFILE
echo "Working directory: $PWD"

USER_NAME="${USER:-$(id -un)}"
REPO_NAME="$(basename "$PBS_O_WORKDIR")"
SCRATCH_LOCAL_ROOT="/scratch_local/$USER_NAME/$REPO_NAME"
SCRATCH_GLOBAL_ROOT="/scratch_global/$USER_NAME/$REPO_NAME"

if [ ! -d /scratch_local ]; then
    echo "Error: /scratch_local is not available on this node" >&2
    exit 1
fi

if [ ! -d /scratch_global ]; then
    echo "Error: /scratch_global is not available on this node" >&2
    exit 1
fi

mkdir -p "$SCRATCH_LOCAL_ROOT/apptainer_tmp" "$SCRATCH_LOCAL_ROOT/apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH_LOCAL_ROOT/apptainer_tmp"
export APPTAINER_CACHEDIR="$SCRATCH_LOCAL_ROOT/apptainer_cache"

# This line needs to remain hard-coded since containers are physically in this location
CONTAINER_DIR="/work/u11172853/containers"
DEFAULT_CONTAINER="$CONTAINER_DIR/amsc_mk_2025.sif"
AVX512_CONTAINER="$CONTAINER_DIR/dealii-avx512.sif"
CONTAINER_BIND="/scratch_local,/scratch_global"

if [ ! -f "$DEFAULT_CONTAINER" ]; then
    echo "Error: default container not found: $DEFAULT_CONTAINER" >&2
    exit 1
fi

if [ ! -f "$AVX512_CONTAINER" ]; then
    echo "Error: AVX512 container not found: $AVX512_CONTAINER" >&2
    exit 1
fi

# Matrix-based and matrix-free SIMD=0 are built in the baseline container.
apptainer exec --cleanenv --bind "$CONTAINER_BIND" "$DEFAULT_CONTAINER" bash -lc '
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    source /u/sw/lmod/8.5.8/init/bash
fi
export MODULEPATH="${MODULEPATH:-/u/sw/modules}"

module load toolchains/gcc-glibc/11.2.0
module load dealii

USER_NAME=$(id -un)
REPO_NAME=$(basename "$PWD")
SCRATCH_LOCAL_ROOT="/scratch_local/$USER_NAME/$REPO_NAME"
SCRATCH_GLOBAL_ROOT="/scratch_global/$USER_NAME/$REPO_NAME"

cmake -S . -B build/baseline \
    -DMPI_C_COMPILER=$(command -v mpicc) \
    -DMPI_CXX_COMPILER=$(command -v mpicxx) \
    -DMFSOLVER_BUILD_MATRIX_BASED=ON \
    -DMFSOLVER_BUILD_MATRIX_FREE_NO_SIMD=ON \
    -DMFSOLVER_BUILD_MATRIX_FREE_SIMD=OFF
cmake --build build/baseline -j 4 --target matrix_based matrix_free_no_simd
'

# Matrix-free SIMD=1 is built in the AVX512 container because the target uses
# -march=native and should only be compiled/executed in that image. The AVX512
# image is self-contained, so this block intentionally does not load modules.
apptainer exec --cleanenv --bind "$CONTAINER_BIND" "$AVX512_CONTAINER" bash -lc '
set -euo pipefail

cmake -S . -B build/avx512 \
    -DMPI_C_COMPILER=$(command -v mpicc) \
    -DMPI_CXX_COMPILER=$(command -v mpicxx) \
    -DMFSOLVER_BUILD_MATRIX_BASED=OFF \
    -DMFSOLVER_BUILD_MATRIX_FREE_NO_SIMD=OFF \
    -DMFSOLVER_BUILD_MATRIX_FREE_SIMD=ON
cmake --build build/avx512 -j 4 --target matrix_free_simd
'

./run_extensive_tests.sh \
    --default_container "$DEFAULT_CONTAINER" \
    --avx512_container "$AVX512_CONTAINER" \
    --apptainer_bind "$CONTAINER_BIND" \
    --solver mb mf \
    --n_tests 1 \
    --fe_deg 2 \
    --tol 0.001 \
    --n_ranks 1 2 4 8 \
    --n_threads 1 2 \
    --run_timeout_seconds 90 \
    --n_additional_refinements 2 \
    --use_scratch_local \
    --scratch_local_root "$SCRATCH_LOCAL_ROOT" \
    --scratch_global_root "$SCRATCH_GLOBAL_ROOT"

echo "PBS job finished at $(date)"
