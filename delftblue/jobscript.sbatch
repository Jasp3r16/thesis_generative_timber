#!/bin/bash
# DelftBlue SLURM job: c23_run_ga_batch.py
# Current config in the script: scenario "A", N_RUNS=3 (seeds 42/43/44),
# CMAES_GENERATIONS=500, popsize 30  ->  3 runs x 15,000 evals.
#
# Submit from the repo root on DelftBlue:
#     sbatch delftblue/jobscript.sbatch
#
# !! Before first submit, set --account to YOUR allocation (run `sacctmgr show
#    assoc user=$USER format=account` to list the ones you can charge to).

#SBATCH --job-name=ga_batch_A
#SBATCH --partition=compute
#SBATCH --account=ACCOUNT_PLACEHOLDER
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/ga_batch_%j.out
#SBATCH --error=logs/ga_batch_%j.err

set -euo pipefail

module load 2024r1
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate timber-ga

# Run from the directory the job was submitted from (repo root expected).
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# The GA evaluates candidates sequentially; extra cores only help the BLAS /
# CBC / torch intra-op threads. Cap them to the allocation to avoid oversubscribe.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "Python: $(which python)"

# -u = unbuffered, so the per-generation prints stream into the .out log live.
python -u c23_run_ga_batch.py

echo "Done — $(date)"
