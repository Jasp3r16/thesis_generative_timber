#!/bin/bash
# Cost–reuse objective-weight SWEEP (thesis §6.4.3).
# Traces the cost vs reuse trade-off the scalar fitness can reach, by sweeping the
# weight ratio omega_1 (cost) : omega_2 (reuse). Fitness = w1*cost_norm - w2*reuse_norm
# + w4*struct (minimised), so a higher w1 favours low cost, a higher w2 favours reuse.
#
#   omega_2 is held at 1.0; omega_1 is swept. The (1.0, 1.0) point reproduces the
#   thesis Stock A baseline (built-in sanity check).
#   2025/structural weighting (w_structural 2.0->0.8) and cap=10 are unchanged.
#
# Config: scenario A, cap=10, 250 generations, seed-matched (42/43/44).
# Array 0-14 = 5 weight points x 3 seeds.
#   weight index = TASK / 3   ->  OMEGA1[idx]   (omega_2 = 1.0)
#   seed   index = TASK % 3   ->  GA_RUN_INDEX
#
#   sbatch delftblue/jobscript_weight_sweep.sbatch
#
# Exports self-tag ..._RUN{n}_W{omega1}_... (e.g. RUN1_W0p25) and run_config.json
# records the full fitness_weights — non-dominated front extracted across all 15.

#SBATCH --job-name=ga_Awsweep
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-14
#SBATCH --output=logs/ga_Awsweep_%A_%a.out
#SBATCH --error=logs/ga_Awsweep_%A_%a.err

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

# cost-favouring  <----------------------------->  reuse-favouring
# omega_1:        4.0   2.0   1.0(thesis)   0.5   0.25     (omega_2 = 1.0)
OMEGA1=(4.0 2.0 1.0 0.5 0.25)
WIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

export GA_SCENARIO=A
export GA_GENERATIONS=250
export GA_RUN_INDEX="${SIDX}"
export GA_OMEGA_1="${OMEGA1[$WIDX]}"
export GA_OMEGA_2=1.0
export GA_RUN_LABEL="W${OMEGA1[$WIDX]}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "WSWEEP  omega_1=${GA_OMEGA_1}  omega_2=${GA_OMEGA_2}  seed=$((42 + SIDX))  (cap=10, 250 gen)"

python -u c23_run_ga_batch.py

echo "Done weight-sweep task ${SLURM_ARRAY_TASK_ID} (omega_1=${GA_OMEGA_1}, seed $((42+SIDX))) — $(date)"
