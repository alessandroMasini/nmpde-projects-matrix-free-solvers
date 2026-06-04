#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=8
#PBS -l place=scatter
#PBS -l walltime=10:00:00
#PBS -q cpu
#PBS -N test
#PBS -j oe
#PBS -o /home/u11172853/nmpde-projects-matrix-free-solvers/pbs_jobs/test.pbs.out

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

# Run the build and test script inside the container, with cluster scratch paths visible.
apptainer exec --cleanenv --bind /scratch_local,/scratch_global /work/u11172853/containers/amsc_mk_2025.sif bash -lc '
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

cmake -U MPI_* \
    -DMPI_C_COMPILER=$(command -v mpicc) \
    -DMPI_CXX_COMPILER=$(command -v mpicxx) \
    .
make -j 4
./run_extensive_tests.sh \
    --solver mb mf \
    --n_tests 1 \
    --fe_deg 2 \
    --tol 0.001 \
    --n_ranks 1 2 4 8 \
    --n_threads 1 2 \
    --run-timeout-seconds 90 \
    --n_additional_refinements 2 \
    --use-scratch-local \
    --scratch-local-root "$SCRATCH_LOCAL_ROOT" \
    --scratch-global-root "$SCRATCH_GLOBAL_ROOT"
'

echo "PBS job finished at $(date)"
