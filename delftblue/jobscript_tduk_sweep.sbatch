#!/bin/bash
# TDUK A1-A3 embodied-carbon SWEEP (thesis §6.4.2).
# Sweeps the new-softwood A1-A3 factor from the 2025 value to the 2026 value and
# measures how Stock A cost / reuse respond as new timber's embodied carbon falls.
#
#   NS density is a constant 350 kg/m3 in this stock, so a per-VOLUME factor of
#   X kg CO2e/m3 equals a per-MASS factor of X/350 kg/kg EXACTLY. Hence:
#     2025 (thesis) : 87.5 kg/m3  ==  0.25 kg/kg   <- reproduces the thesis baseline
#     2026 (TDUK)   : 47.0 kg/m3  ==  0.134 kg/kg
#   The 87.5 point is therefore a built-in sanity check against thesis Stock A.
#
# Config: scenario A, cap=10 (matches thesis), 250 generations, seed-matched.
# Array 0-14 = 5 factor points x 3 seeds (42/43/44).
#   factor index = TASK / 3   ->  FACTORS[idx]
#   seed   index = TASK % 3   ->  GA_RUN_INDEX (seed 42+idx)
#
#   sbatch delftblue/jobscript_tduk_sweep.sbatch
#
# Exports self-tag as ..._RUN{n}_TDUK{factor}_... (e.g. RUN1_TDUK87p5, RUN1_TDUK47p0)
# and run_config.json records a1a3_per_m3_override — no collisions, fully traceable.

#SBATCH --job-name=ga_Asweep
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
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

# 2025 -> 2026, evenly spaced (kg CO2e/m3). FACTORS[0]=87.5 reproduces thesis.
FACTORS=(87.5 77.4 67.3 57.2 47.0)
FIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

export GA_SCENARIO=A
export GA_GENERATIONS=250
export GA_RUN_INDEX="${SIDX}"
export GA_A1A3_PER_M3="${FACTORS[$FIDX]}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "SWEEP  A1A3_PER_M3=${GA_A1A3_PER_M3} kg/m3  seed=$((42 + SIDX))  (cap=10, 250 gen)"

python -u c23_run_ga_batch.py

echo "Done sweep task ${SLURM_ARRAY_TASK_ID} (factor ${GA_A1A3_PER_M3}, seed $((42+SIDX))) — $(date)"
