#!/bin/bash
# DelftBlue SLURM ARRAY job: c23_run_ga_batch.py at 1000 generations (thesis §6.4.1
# convergence FOLLOW-UP). Identical setup to jobscript.sbatch (Scenario A, cap=10,
# 10 seeds 42..51, one seed per task) EXCEPT the budget is doubled to 1000 gens.
#   task i -> seed (42 + i) -> RUN(i+1).  Array 0-9 = seeds 42..51.
#
# WHY: the 500-gen batch (jobscript.sbatch) showed best fitness still landing at the
# FINAL generation in every seed (~18-21% of the gain arriving after gen 250), so 500
# was "defensible but not clearly converged". 1000 gens tests whether the search has a
# long shallow tail: compare the in-window gain over [500, 1000] against the total gain.
#   - negligible [500,1000] gain  -> 500 was effectively the floor (validates §6.4.1).
#   - meaningful  [500,1000] gain -> even 500 is premature (slow-tail finding).
#
# Exports self-tag as ..._RUN{n}_g1000_... via GA_RUN_LABEL, so they sit ALONGSIDE the
# unlabeled 500-gen RUN{n} exports (same scenario/cap/seeds) instead of overwriting them.
# NOTE: do NOT assume a 1000-gen run is a bit-exact prefix of the matching 500-gen run —
# the MILP solver is mildly non-deterministic (~+/-2-3 kg same-seed). Treat the two
# batches as seed-paired, and measure the [500,1000] gain WITHIN each 1000-gen run.
#
# Each task runs ONE 1000-gen run (30,000 evals, ~6-10 h on CPU), all in parallel.
# Walltime raised to 20 h (2x the 12 h used for 500 gen) for margin on the slowest seed;
# still well under the qos=normal 24 h cap.
#
# Submit from the repo root on DelftBlue:
#     sbatch delftblue/jobscript_conv1000.sbatch

#SBATCH --job-name=ga_Ag1000
#SBATCH --partition=compute
#SBATCH --account=education-abe-msc-a
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-9
#SBATCH --output=logs/ga_Ag1000_%A_%a.out
#SBATCH --error=logs/ga_Ag1000_%A_%a.err

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
# cap stays at the config default (new_stock_max_uses=10) to match §6.4.1 / the thesis.
export GA_SCENARIO=A
export GA_GENERATIONS=1000
export GA_RUN_INDEX="${SLURM_ARRAY_TASK_ID}"
export GA_RUN_LABEL=g1000

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "Scenario=${GA_SCENARIO}  gens=${GA_GENERATIONS}  run_index=${GA_RUN_INDEX} (seed $((42 + SLURM_ARRAY_TASK_ID)))  label=${GA_RUN_LABEL}"

# -u = unbuffered so per-generation prints stream live into the .out log.
python -u c23_run_ga_batch.py

echo "Done task ${SLURM_ARRAY_TASK_ID} — $(date)"
