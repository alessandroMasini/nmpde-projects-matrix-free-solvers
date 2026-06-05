#!/bin/bash
#PBS -l select=1:ncpus=1
#PBS -l place=scatter
#PBS -l walltime=00:00:30
#PBS -q cpu
#PBS -N hello_world
#PBS -j oe

# Always run from the directory where you submitted the job
# cd "$PBS_O_WORKDIR" || exit 1

# Apptainer temp/cache (helpful on clusters, especially for bigger images)
# mkdir -p "$HOME/apptainer_tmp"
# export APPTAINER_TMPDIR="$HOME/apptainer_tmp"
# export APPTAINER_CACHEDIR="$HOME/apptainer_tmp"

# (Optional) be explicit about threads if any BLAS/OpenMP is used
# export OMP_NUM_THREADS=2

# Run from the submitted repository when PBS provides it; otherwise use the
# current directory so the helper also works when launched manually.
cd "${PBS_O_WORKDIR:-$PWD}" || exit 1

echo "Hello world" | tee "$PWD/hello.txt"
