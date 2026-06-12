#!/bin/bash
# Carbon-per-unit-reuse fitness term (thesis §6.4.4) — v1 vs v2 head-to-head.
# Tests whether rewarding CARBON AVOIDED by reuse (instead of reuse VOLUME) resolves
# the cost paradox: same reuse intent, but the optimiser is no longer pushed toward
# heavy, carbon-expensive reclaimed pieces.
#
#   v1 "volume"        : F = w1*cost_norm - w2*reuse_norm        (thesis default)
#   v2 "carbon_avoided": F = w1*cost_norm - w2*(avoided/C_max)   (§6.4.4 add-on)
#   Both modes always report reuse_fraction AND carbon_avoided, so the pair is
#   directly comparable. v1 here == the thesis baseline (control).
#
#   SCALE NOTE: from the §6.4.2 data, avoided ≈ (all-new − Stock A) ≈ 68 kg, and
#   avoided/C_max ≈ 0.17 vs v1 reuse_norm ≈ 1.0 — the v2 reward is ~6x weaker at
#   equal w2. So v2 is run at BOOSTED w2 (3 and 6) to match reward strength;
#   otherwise v2 would just collapse reuse toward cost-min and tell us nothing.
#
# Config: scenario A, cap=10, 250 generations, seed-matched (42/43/44).
# Array 0-8 = 3 conditions x 3 seeds:
#   cond 0 : v1 volume,         w2 = 1.0   (thesis baseline / control)
#   cond 1 : v2 carbon_avoided, w2 = 3.0
#   cond 2 : v2 carbon_avoided, w2 = 6.0
#   cond index = TASK / 3   ;   seed index = TASK % 3
#
#   sbatch delftblue/jobscript_carbon_reuse.sbatch
#
# Exports self-tag ..._RUN{n}_{label}; run_config records reuse_reward_mode + weights.

#SBATCH --job-name=ga_Acreuse
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-8
#SBATCH --output=logs/ga_Acreuse_%A_%a.out
#SBATCH --error=logs/ga_Acreuse_%A_%a.err

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

MODES=(volume        carbon_avoided carbon_avoided)
OMEGA2=(1.0          3.0            6.0)
LABELS=(v1vol        v2carbW3       v2carbW6)
CIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

export GA_SCENARIO=A
export GA_GENERATIONS=250
export GA_RUN_INDEX="${SIDX}"
export GA_REUSE_REWARD="${MODES[$CIDX]}"
export GA_OMEGA_2="${OMEGA2[$CIDX]}"
export GA_RUN_LABEL="${LABELS[$CIDX]}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "CREUSE  reuse_reward=${GA_REUSE_REWARD}  omega_2=${GA_OMEGA_2}  seed=$((42 + SIDX))  (cap=10, 250 gen)"

python -u c23_run_ga_batch.py

echo "Done carbon-reuse task ${SLURM_ARRAY_TASK_ID} (${GA_RUN_LABEL}, seed $((42+SIDX))) — $(date)"
