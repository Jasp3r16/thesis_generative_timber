#!/bin/bash
# DelftBlue CPU smoke test for c21 pipeline while GPU entitlement is pending
#SBATCH --job-name=c21_cpu_smoke
#SBATCH --output=delfblue_logs/smoke/c21_cpu_smoke_%j.out
#SBATCH --error=delfblue_logs/smoke/c21_cpu_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=education-abe-msc-a

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${REPO_DIR}/delfblue_logs" "${REPO_DIR}/delfblue_logs/smoke"

module purge
module load 2024r1
module load python
module list 2>&1 || true
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
echo "WHICH_PYTHON=$(command -v python)"
python --version

DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"
if [[ ! -f "${DELFTBLUE_VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${DELFTBLUE_VENV}" >&2
  exit 1
fi
source "${DELFTBLUE_VENV}/bin/activate"
echo "VENV_PYTHON=$(command -v python)"
python --version

export DELFTBLUE_DATA_BASE="${DELFTBLUE_DATA_BASE:-/scratch/${USER}}"
export C21_RUN_ID="cpu_smoke"
export C21_TASK_INDEX="1"
export C21_USE_PRETRAINED="false"
export C21_USE_TRAINING_TIME_LIMIT="true"
export C21_TRAINING_TIME_LIMIT_SECONDS="300"
export C21_EPOCHS="2"
export C21_BATCH_SIZE="8"
export C21_HIDDEN_DIM="64"
export C21_LEARNING_RATE="0.0005"
export C21_WEIGHT_DECAY="0.0"
export C21_TRAIN_SPLIT_RATIO="0.8"
export C21_RANDOM_SEED="42"

cd "${REPO_DIR}"
python workflows/c21_slurm_train.py
