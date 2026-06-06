#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=2
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -q cpu
#PBS -N test_mf_overnight
#PBS -j oe
#PBS -o pbs_jobs/test_mf_overnight.pbs.out

# Always run from the directory where you submitted the job
cd "$PBS_O_WORKDIR" || exit 1

set -euo pipefail

LOG_FILE="$PWD/pbs_jobs/last_mf_overnight_output.txt"
mkdir -p "$(dirname "$LOG_FILE")"

# Capture the whole PBS job stdout/stderr
exec > >(tee "$LOG_FILE") 2>&1

echo "PBS overnight job started on $(hostname) at $(date)"
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

# Container directories on compute nodes
CONTAINER_DIR="/work/u11172853/containers"
DEFAULT_CONTAINER="$CONTAINER_DIR/amsc_mk_2025.sif"
AVX512_CONTAINER="$CONTAINER_DIR/dealii-avx512.sif"
CONTAINER_BIND="/scratch_local,/scratch_global"

# Build matrix_free_no_simd in the baseline container
echo "=== Building Matrix-Free Non-SIMD ==="
apptainer exec --cleanenv --bind "$CONTAINER_BIND" "$DEFAULT_CONTAINER" bash -lc '
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
    source /u/sw/lmod/8.5.8/init/bash
fi
export MODULEPATH="${MODULEPATH:-/u/sw/modules}"
module load toolchains/gcc-glibc/11.2.0
module load dealii

cmake -S . -B build/baseline \
    -DMPI_C_COMPILER=$(command -v mpicc) \
    -DMPI_CXX_COMPILER=$(command -v mpicxx) \
    -DMFSOLVER_BUILD_MATRIX_BASED=OFF \
    -DMFSOLVER_BUILD_MATRIX_FREE_NO_SIMD=ON \
    -DMFSOLVER_BUILD_MATRIX_FREE_SIMD=OFF
cmake --build build/baseline -j 4 --target matrix_free_no_simd
'

# Build matrix_free_simd in the AVX512 container
echo "=== Building Matrix-Free SIMD ==="
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

# We split the sweeps to avoid huge combinations and target scaling specifically.
# All tests write to the same scratch tests folder, and we will summarize everything at the end.

echo "=== Sweep 1: Standard Verification (All Problems, deg=2, ref=2, n_threads=1,2,4,8) ==="
./run_extensive_tests.sh \
    --default_container "$DEFAULT_CONTAINER" \
    --avx512_container "$AVX512_CONTAINER" \
    --apptainer_bind "$CONTAINER_BIND" \
    --solver mf \
    --problem "advanced lab_02 lab_03 parabolic transient mms" \
    --fe_deg 2 \
    --n_additional_refinements 2 \
    --n_ranks 1 2 \
    --n_threads 1 2 4 8 \
    --simd 0 1 \
    --n_tests 1 \
    --tol 0.001 \
    --run_timeout_seconds 120 \
    --use_scratch_local \
    --scratch_local_root "$SCRATCH_LOCAL_ROOT" \
    --scratch_global_root "$SCRATCH_GLOBAL_ROOT"

echo "=== Sweep 2: Large Problem Thread Scaling (transient, deg=2, ref=3, n_threads=1,2,4,8) ==="
# A larger refinement=3 (2.1M DoFs) makes the fine-grid calculation heavy enough to amortize TBB overhead.
./run_extensive_tests.sh \
    --default_container "$DEFAULT_CONTAINER" \
    --avx512_container "$AVX512_CONTAINER" \
    --apptainer_bind "$CONTAINER_BIND" \
    --solver mf \
    --problem "transient" \
    --fe_deg 2 \
    --n_additional_refinements 3 \
    --n_ranks 1 2 \
    --n_threads 1 2 4 8 \
    --simd 0 1 \
    --n_tests 1 \
    --tol 0.001 \
    --run_timeout_seconds 400 \
    --use_scratch_local \
    --scratch_local_root "$SCRATCH_LOCAL_ROOT" \
    --scratch_global_root "$SCRATCH_GLOBAL_ROOT"

echo "=== Sweep 3: High Degree FE Scaling (transient, deg=4, ref=1, n_threads=1,2,4,8) ==="
# Q4 finite elements have heavy local cell computations, which is perfect for demonstrating SIMD & thread scaling.
./run_extensive_tests.sh \
    --default_container "$DEFAULT_CONTAINER" \
    --avx512_container "$AVX512_CONTAINER" \
    --apptainer_bind "$CONTAINER_BIND" \
    --solver mf \
    --problem "transient" \
    --fe_deg 4 \
    --n_additional_refinements 1 \
    --n_ranks 1 2 \
    --n_threads 1 2 4 8 \
    --simd 0 1 \
    --n_tests 1 \
    --tol 0.001 \
    --run_timeout_seconds 300 \
    --use_scratch_local \
    --scratch_local_root "$SCRATCH_LOCAL_ROOT" \
    --scratch_global_root "$SCRATCH_GLOBAL_ROOT"

echo "=== Generating final consolidated overnight test summary ==="
python3 summarize_tests.py "$SCRATCH_LOCAL_ROOT/tests" --all 2>/dev/null || echo "Error: Summary generation failed"

echo "PBS overnight job finished at $(date)"
