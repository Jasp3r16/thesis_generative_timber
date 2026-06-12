#!/bin/bash
# All-NEW timber cost at the 2025 vs 2026 A1-A3 factor (for the A4 Slide-5 number).
# Gives the "Stock A is X% lower than all-new" figure at the 2026 factor, by pairing
# this all-new cost with the Stock A sweep (§6.4.2): advantage = (allnew - StockA)/allnew.
#
#   Scenario "new" (max_uses=10, the thesis New condition; no reclaimed stock).
#   factor 0.250 (2025) doubles as a gate — should reproduce thesis New ≈ 195.7.
#   factor 0.112 (2026) is the number we actually need.
#
# Array 0-5 = 2 factors x 3 seeds.
#   factor index = TASK / 3 -> FACTORS[idx] ;  seed index = TASK % 3 -> GA_RUN_INDEX
#
#   sbatch delftblue/jobscript_allnew_tduk.sbatch

#SBATCH --job-name=ga_NEWsweep
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-5
#SBATCH --output=logs/ga_NEWsweep_%A_%a.out
#SBATCH --error=logs/ga_NEWsweep_%A_%a.err

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

FACTORS=(0.250 0.112)
FIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

export GA_SCENARIO=new
export GA_GENERATIONS=250
export GA_RUN_INDEX="${SIDX}"
export GA_A1A3_PER_KG="${FACTORS[$FIDX]}"
export GA_RUN_LABEL="NEW"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "NEWSWEEP  A1A3_PER_KG=${GA_A1A3_PER_KG}  seed=$((42 + SIDX))  scenario=new (cap=10, 250 gen)"

python -u c23_run_ga_batch.py

echo "Done all-new task ${SLURM_ARRAY_TASK_ID} (factor ${GA_A1A3_PER_KG}, seed $((42+SIDX))) — $(date)"
