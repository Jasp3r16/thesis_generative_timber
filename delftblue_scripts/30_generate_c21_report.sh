#!/bin/bash
set -euo pipefail

# Generate aggregated hyperparameter report after a sweep.
# Usage:
#   bash delftblue_scripts/30_generate_c21_report.sh
# Optional overrides:
#   DELFTBLUE_VENV=/scratch/$USER/venvs/thesis_gnn
#   DELFTBLUE_DATA_BASE=/scratch/$USER

module purge
module load 2024r1
module load python

DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"
if [[ ! -f "${DELFTBLUE_VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${DELFTBLUE_VENV}" >&2
  exit 1
fi
source "${DELFTBLUE_VENV}/bin/activate"

export DELFTBLUE_DATA_BASE="${DELFTBLUE_DATA_BASE:-/scratch/${USER}}"

python workflows/c21_hyperparameter_report.py
