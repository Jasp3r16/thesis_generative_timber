#!/bin/bash
set -euo pipefail

# Usage:
#   bash workflows/delftblue_phase1_setup.sh
# Optional overrides:
#   DELFTBLUE_VENV=/scratch/$USER/venvs/thesis_gnn
#   REPO_DIR=/path/to/thesis_generative_timber

REPO_DIR="${REPO_DIR:-$PWD}"
DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"

module purge
module load 2024r1
module load python
module try-load cuda/12.2.0
module try-load cuda
module try-load cudnn/8.9.5-cuda-12.2
module try-load cudnn
echo "Loaded modules:"
module list 2>&1 || true

mkdir -p "$(dirname "${DELFTBLUE_VENV}")"
python -m venv "${DELFTBLUE_VENV}"
source "${DELFTBLUE_VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${REPO_DIR}/requirements.txt"

python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
PY

echo "Setup complete"
echo "Repo: ${REPO_DIR}"
echo "Venv: ${DELFTBLUE_VENV}"
echo "Submit phase 1 with: sbatch workflows/delftblue_c21_array.slurm"
