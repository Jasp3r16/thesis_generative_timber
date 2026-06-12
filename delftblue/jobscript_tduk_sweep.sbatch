#!/bin/bash
# TDUK A1-A3 embodied-carbon SWEEP (thesis §6.4.2).
# Sweeps the new-softwood A1-A3 factor from the 2025 value to the 2026 value and
# measures how Stock A cost / reuse respond as new timber's embodied carbon falls.
#
#   The factor is swept in PER-MASS units (kg CO2e/kg) via GA_A1A3_PER_KG, which
#   directly overrides IMPACT_FACTOR_A1_A3 (default 0.25). This makes the 0.250
#   point reproduce the thesis Stock A baseline EXACTLY — no density conversion.
#     2025 (thesis) : 0.250 kg/kg   <- = IMPACT_FACTOR_A1_A3, reproduces baseline
#     2026 (TDUK)   : 0.112 kg/kg   <- = 47 kg/m3 / 420 (new-timber mean_density)
#
# Config: scenario A, cap=10 (matches thesis), 250 generations, seed-matched.
# Array 0-14 = 5 factor points x 3 seeds (42/43/44).
#   factor index = TASK / 3   ->  FACTORS[idx]
#   seed   index = TASK % 3   ->  GA_RUN_INDEX (seed 42+idx)
#
#   sbatch delftblue/jobscript_tduk_sweep.sbatch
#
# Exports self-tag as ..._RUN{n}_TDUK{factor}_... (e.g. RUN1_TDUK0p250, RUN1_TDUK0p112)
# and run_config.json records a1a3_per_kg_override — no collisions, fully traceable.

#SBATCH --job-name=ga_Asweep
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-14
#SBATCH --output=logs/ga_Asweep_%A_%a.out
#SBATCH --error=logs/ga_Asweep_%A_%a.err

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

# 2025 -> 2026, evenly spaced (kg CO2e/kg). FACTORS[0]=0.250 reproduces thesis.
FACTORS=(0.250 0.216 0.181 0.147 0.112)
FIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

export GA_SCENARIO=A
export GA_GENERATIONS=250
export GA_RUN_INDEX="${SIDX}"
export GA_A1A3_PER_KG="${FACTORS[$FIDX]}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "SWEEP  A1A3_PER_KG=${GA_A1A3_PER_KG} kg/kg  seed=$((42 + SIDX))  (cap=10, 250 gen)"

python -u c23_run_ga_batch.py

echo "Done sweep task ${SLURM_ARRAY_TASK_ID} (factor ${GA_A1A3_PER_KG}, seed $((42+SIDX))) — $(date)"
