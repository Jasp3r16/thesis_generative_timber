#!/bin/bash
#SBATCH --job-name=c21_sweep_p2
#SBATCH --output=logs/c21_p2_%A_%a.out
#SBATCH --error=logs/c21_p2_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-24

set -euo pipefail

# Update these for your DelftBlue project/account setup if needed.
# SBATCH --account=<your_account>

REPO_DIR="${SLURM_SUBMIT_DIR}"
GRID_FILE="${REPO_DIR}/workflows/delftblue_hyperparameter_grid_phase2.txt"
NB_FILE="${REPO_DIR}/c21_surrogate_model_training.ipynb"
RUN_DIR="${REPO_DIR}/workflows/delftblue_runs_phase2"
mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}"

if [[ ! -f "${GRID_FILE}" ]]; then
  echo "Grid file not found: ${GRID_FILE}" >&2
  exit 1
fi

line=$(grep '^RUN2_' "${GRID_FILE}" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [[ -z "${line}" ]]; then
  echo "No run config for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

run_id=$(echo "${line}" | awk '{print $1}')
lr=$(echo "${line}" | sed -n 's/.*lr=\([^ ]*\).*/\1/p')
epochs=$(echo "${line}" | sed -n 's/.*epochs=\([^ ]*\).*/\1/p')
batch_size=$(echo "${line}" | sed -n 's/.*batch_size=\([^ ]*\).*/\1/p')
hidden_dim=$(echo "${line}" | sed -n 's/.*hidden_dim=\([^ ]*\).*/\1/p')
weight_decay=$(echo "${line}" | sed -n 's/.*weight_decay=\([^ ]*\).*/\1/p')
train_split_ratio=$(echo "${line}" | sed -n 's/.*train_split_ratio=\([^ ]*\).*/\1/p')
random_seed=$(echo "${line}" | sed -n 's/.*random_seed=\([^ ]*\).*/\1/p')

export C21_RUN_ID="${run_id}"
export C21_LEARNING_RATE="${lr}"
export C21_EPOCHS="${epochs}"
export C21_BATCH_SIZE="${batch_size}"
export C21_HIDDEN_DIM="${hidden_dim}"
export C21_WEIGHT_DECAY="${weight_decay}"
export C21_TRAIN_SPLIT_RATIO="${train_split_ratio}"
export C21_RANDOM_SEED="${random_seed}"
export C21_USE_PRETRAINED="false"

echo "Running ${run_id} with lr=${lr}, epochs=${epochs}, batch_size=${batch_size}, hidden_dim=${hidden_dim}, weight_decay=${weight_decay}"

cd "${REPO_DIR}"

python -m jupyter nbconvert \
  --to notebook \
  --execute "${NB_FILE}" \
  --output "c21_surrogate_model_training_${run_id}.ipynb" \
  --output-dir "${RUN_DIR}" \
  --ExecutePreprocessor.timeout=-1

echo "Finished ${run_id}"