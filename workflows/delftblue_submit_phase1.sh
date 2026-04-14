#!/bin/bash
set -euo pipefail

# Submit phase 1 as individual jobs (no array), compatible with strict assoc limits.
# Usage:
#   bash workflows/delftblue_submit_phase1.sh

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GRID_FILE="${REPO_DIR}/workflows/delftblue_hyperparameter_grid.txt"

if [[ ! -f "${GRID_FILE}" ]]; then
  echo "Grid file not found: ${GRID_FILE}" >&2
  exit 1
fi

count=$(grep -c '^RUN_' "${GRID_FILE}")
if [[ "${count}" -eq 0 ]]; then
  echo "No RUN_ entries found in ${GRID_FILE}" >&2
  exit 1
fi

echo "Submitting ${count} phase-1 jobs (1 GPU each)..."
for i in $(seq 1 "${count}"); do
  jid=$(sbatch --export=ALL,C21_TASK_INDEX="${i}" workflows/delftblue_c21_array.slurm | awk '{print $4}')
  echo "Submitted index=${i} job_id=${jid}"
done

echo "Done. Monitor with: squeue -u ${USER}"
