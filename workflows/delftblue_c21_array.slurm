#!/bin/bash
# DelftBlue job policy: account=education-abe-msc-a, partition=gpu, walltime=08:00:00
#SBATCH --job-name=c21_sweep
#SBATCH --output=logs/c21_%j.out
#SBATCH --error=logs/c21_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-abe-msc-a

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
GRID_FILE="${REPO_DIR}/workflows/delftblue_hyperparameter_grid.txt"
RUN_DIR="${REPO_DIR}/workflows/delftblue_runs"
mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}"

# DelftBlue runtime modules
module purge
module load 2024r1
module load python
module try-load cuda/12.2.0
module try-load cuda
module try-load cudnn/8.9.5-cuda-12.2
module try-load cudnn
echo "Loaded modules:" 
module list 2>&1 || true

# Default venv location; can be overridden before sbatch
DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"
if [[ ! -f "${DELFTBLUE_VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${DELFTBLUE_VENV}" >&2
  echo "Create it first with workflows/delftblue_phase1_setup.sh" >&2
  exit 1
fi
source "${DELFTBLUE_VENV}/bin/activate"

if [[ ! -f "${GRID_FILE}" ]]; then
  echo "Grid file not found: ${GRID_FILE}" >&2
  exit 1
fi

TASK_INDEX="${C21_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
  echo "Missing run index. Set C21_TASK_INDEX (or submit as SLURM array)." >&2
  exit 1
fi

line=$(grep '^RUN_' "${GRID_FILE}" | sed -n "${TASK_INDEX}p")
if [[ -z "${line}" ]]; then
  echo "No run config for index ${TASK_INDEX}" >&2
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
export C21_TASK_INDEX="${TASK_INDEX}"

# DelftBlue data storage path (config.py will use this)
export DELFTBLUE_DATA_BASE="/scratch/${USER}"

echo "Running index=${TASK_INDEX}, run_id=${run_id} with lr=${lr}, epochs=${epochs}, batch_size=${batch_size}, hidden_dim=${hidden_dim}, weight_decay=${weight_decay}"

cd "${REPO_DIR}"

python workflows/c21_slurm_train.py

echo "Finished ${run_id}"