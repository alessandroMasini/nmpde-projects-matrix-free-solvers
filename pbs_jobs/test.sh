#!/bin/bash
#PBS -l select=1:ncpus=16
#PBS -l place=scatter
#PBS -l walltime=08:00:00
#PBS -q cpu
#PBS -N test
#PBS -j oe

# Always run from the directory where you submitted the job
cd "$PBS_O_WORKDIR" || exit 1

# Apptainer temp/cache (helpful on clusters, especially for bigger images)
# mkdir -p "$HOME/apptainer_tmp"
# export APPTAINER_TMPDIR="$HOME/apptainer_tmp"
# export APPTAINER_CACHEDIR="$HOME/apptainer_tmp"

# (Optional) be explicit about threads if any BLAS/OpenMP is used
# export OMP_NUM_THREADS=2

# Run your script INSIDE the container
apptainer exec --cleanenv ./amsc_mk_2025.sif ./run_extensive_tests.sh | tee /home/u11172853/src/outputs/de_cmaes_scale.txt