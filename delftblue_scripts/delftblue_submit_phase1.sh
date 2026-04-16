#!/bin/bash
set -euo pipefail

# Submit phase 1 as individual jobs (no array), compatible with strict assoc limits.
# Usage:
#   bash workflows/delftblue_submit_phase1.sh

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GRID_FILE="${REPO_DIR}/workflows/delftblue_hyperparameter_grid.txt"
SLURM_FILE="${REPO_DIR}/workflows/delftblue_c21_array.slurm"
DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"
GH_DATA_DIR="${DELFTBLUE_DATA_BASE:-/scratch/${USER}}/data/01_grasshopper_data"

mkdir -p "${REPO_DIR}/delfblue_logs"
mkdir -p "${REPO_DIR}/delfblue_logs/phase1" "${REPO_DIR}/delfblue_logs/reports"

if [[ ! -f "${GRID_FILE}" ]]; then
  echo "Grid file not found: ${GRID_FILE}" >&2
  exit 1
fi

if [[ ! -f "${SLURM_FILE}" ]]; then
  echo "SLURM file not found: ${SLURM_FILE}" >&2
  exit 1
fi

if [[ ! -f "${DELFTBLUE_VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${DELFTBLUE_VENV}" >&2
  echo "Run: bash workflows/delftblue_phase1_setup.sh" >&2
  exit 1
fi

if [[ ! -d "${GH_DATA_DIR}" ]]; then
  echo "Missing Grasshopper data directory: ${GH_DATA_DIR}" >&2
  exit 1
fi

if grep -q $'\r' "${SLURM_FILE}"; then
  echo "Detected DOS line breaks in ${SLURM_FILE}. Fixing..."
  sed -i 's/\r$//' "${SLURM_FILE}"
fi

count=$(grep -c '^RUN_' "${GRID_FILE}")
if [[ "${count}" -eq 0 ]]; then
  echo "No RUN_ entries found in ${GRID_FILE}" >&2
  exit 1
fi

first_line=$(grep '^RUN_' "${GRID_FILE}" | sed -n '1p')
node_csv=$(echo "${first_line}" | sed -n 's/.*node_csv=\([^ ]*\).*/\1/p')
edge_csv=$(echo "${first_line}" | sed -n 's/.*edge_csv=\([^ ]*\).*/\1/p')
global_csv=$(echo "${first_line}" | sed -n 's/.*global_csv=\([^ ]*\).*/\1/p')

# Backward-compatible defaults if the grid line does not define explicit dataset columns.
node_csv="${node_csv:-v4_node_C12_S9999_D20260409.csv}"
edge_csv="${edge_csv:-v4_edge_C12_S9999_D20260409.csv}"
global_csv="${global_csv:-v4_global_C4_S9999_D20260409.csv}"

for csv in "${node_csv}" "${edge_csv}" "${global_csv}"; do
  if [[ ! -f "${GH_DATA_DIR}/${csv}" ]]; then
    echo "Missing dataset file: ${GH_DATA_DIR}/${csv}" >&2
    exit 1
  fi
done

echo "Preflight OK"
echo "- Grid entries: ${count}"
echo "- Venv: ${DELFTBLUE_VENV}"
echo "- Data: ${GH_DATA_DIR}"

echo "Submitting ${count} phase-1 jobs (1 GPU each)..."
job_ids=()
for i in $(seq 1 "${count}"); do
  jid=$(sbatch \
    --output="${REPO_DIR}/delfblue_logs/phase1/c21_phase1_run_${i}_%j.out" \
    --error="${REPO_DIR}/delfblue_logs/phase1/c21_phase1_run_${i}_%j.err" \
    --export=ALL,C21_TASK_INDEX="${i}" \
    "${SLURM_FILE}" | awk '{print $4}')
  echo "Submitted index=${i} job_id=${jid}"
  job_ids+=("${jid}")
done

echo "Done. Monitor with: squeue -u ${USER}"

# Submit report job to run automatically after all phase-1 jobs complete.
dep_ids=$(IFS=:; echo "${job_ids[*]}")
report_job_id=$(sbatch \
  --dependency="afterany:${dep_ids}" \
  --job-name="c21_phase1_report" \
  --output="${REPO_DIR}/delfblue_logs/reports/c21_phase1_report_%j.out" \
  --error="${REPO_DIR}/delfblue_logs/reports/c21_phase1_report_%j.err" \
  --wrap="cd '${REPO_DIR}' && bash workflows/delftblue_generate_c21_report.sh" \
  | awk '{print $4}')

echo "Queued report job_id=${report_job_id} (runs after all submitted jobs finish)."
