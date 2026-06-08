#!/bin/bash
# DelftBlue SLURM ARRAY job: c23_run_ga_batch.py — one seed per task.
#   task 0 -> seed 42 -> RUN1
#   task 1 -> seed 43 -> RUN2
#   task 2 -> seed 44 -> RUN3
# Each task runs ONE 500-gen run (15,000 evals, ~4-6 h on CPU), all three in
# parallel. The one-time normalization bounds are seeded from BASE_SEED inside
# the script, so they are identical across tasks (runs stay comparable).
#
# Submit from the repo root on DelftBlue:
#     sbatch delftblue/jobscript.sbatch
#
# !! Before first submit, set --account to YOUR allocation:
#    sacctmgr show assoc user=$USER format=account

#SBATCH --job-name=ga_A
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-2
#SBATCH --output=logs/ga_A_%A_%a.out
#SBATCH --error=logs/ga_A_%A_%a.err

set -euo pipefail

module load 2024r1
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate timber-ga

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# GA evaluates candidates sequentially; extra cores only help BLAS / CBC / torch
# intra-op threads. Cap them to the allocation to avoid oversubscription.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Drive the run via the env-var hook in c23_run_ga_batch.py.
export GA_SCENARIO=A
export GA_GENERATIONS=500
export GA_RUN_INDEX="${SLURM_ARRAY_TASK_ID}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "Scenario=${GA_SCENARIO}  gens=${GA_GENERATIONS}  run_index=${GA_RUN_INDEX} (seed $((42 + SLURM_ARRAY_TASK_ID)))"

# -u = unbuffered so per-generation prints stream live into the .out log.
python -u c23_run_ga_batch.py

echo "Done task ${SLURM_ARRAY_TASK_ID} — $(date)"
