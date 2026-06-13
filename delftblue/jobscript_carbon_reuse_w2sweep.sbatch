#!/bin/bash
# §6.4.4 follow-up: carbon-avoided reward (v2) across LOW omega_2 {0.5, 1, 2}.
# The first batch (omega_2 = 3, 6) made v2 worse than v1 on BOTH cost and reuse —
# possibly because omega_2 was over-boosted (the carbon-avoided reward turned out
# stronger than estimated). This sweep checks the low end: if v2 just relaxes back
# toward v1 as omega_2 falls (and never beats it), that confirms the raw-counterfactual
# reward cannot improve on volume-weighting and needs a cost-anchored reformulation.
#
# Pair these with the existing v1 (omega_2=1) and v2 (omega_2=3,6) runs for the full
# omega_2 map {0.5, 1, 2 | 3, 6}.
#
# Config: scenario A, cap=10, 250 generations, seed-matched (42/43/44), all v2.
# Array 0-8 = 3 omega_2 values x 3 seeds.
#   w2 index = TASK / 3 -> OMEGA2[idx] ;  seed index = TASK % 3 -> GA_RUN_INDEX
#
#   sbatch delftblue/jobscript_carbon_reuse_w2sweep.sbatch

#SBATCH --job-name=ga_Acrw2
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-8
#SBATCH --output=logs/ga_Acrw2_%A_%a.out
#SBATCH --error=logs/ga_Acrw2_%A_%a.err

set -euo pipefail

module load 2024r1
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate timber-ga

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OMEGA2=(0.5      1.0     2.0)
LABELS=(v2carbW0p5 v2carbW1 v2carbW2)
WIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

export GA_SCENARIO=A
export GA_GENERATIONS=250
export GA_RUN_INDEX="${SIDX}"
export GA_REUSE_REWARD=carbon_avoided
export GA_OMEGA_2="${OMEGA2[$WIDX]}"
export GA_RUN_LABEL="${LABELS[$WIDX]}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "CRW2  carbon_avoided  omega_2=${GA_OMEGA_2}  seed=$((42 + SIDX))  (cap=10, 250 gen)"

python -u c23_run_ga_batch.py

echo "Done CRW2 task ${SLURM_ARRAY_TASK_ID} (omega_2=${GA_OMEGA_2}, seed $((42+SIDX))) — $(date)"
