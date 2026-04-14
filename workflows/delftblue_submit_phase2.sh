#!/bin/bash
set -euo pipefail

# Submit phase 2 as individual jobs (no array), compatible with strict assoc limits.
# Usage:
#   bash workflows/delftblue_submit_phase2.sh

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GRID_FILE="${REPO_DIR}/workflows/delftblue_hyperparameter_grid_phase2.txt"

mkdir -p "${REPO_DIR}/logs"

if [[ ! -f "${GRID_FILE}" ]]; then
  echo "Grid file not found: ${GRID_FILE}" >&2
  exit 1
fi

count=$(grep -c '^RUN2_' "${GRID_FILE}")
if [[ "${count}" -eq 0 ]]; then
  echo "No RUN2_ entries found in ${GRID_FILE}" >&2
  exit 1
fi

echo "Submitting ${count} phase-2 jobs (1 GPU each)..."
for i in $(seq 1 "${count}"); do
  jid=$(sbatch --export=ALL,C21_TASK_INDEX="${i}" workflows/delftblue_c21_array_phase2.slurm | awk '{print $4}')
  echo "Submitted index=${i} job_id=${jid}"
done

echo "Done. Monitor with: squeue -u ${USER}"
