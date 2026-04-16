#!/bin/bash
set -euo pipefail

# Submit phase 1 as one array submission.
# Usage:
#   bash delftblue_scripts/10_submit_phase1.sh

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GRID_FILE="${REPO_DIR}/delftblue_scripts/delftblue_hyperparameter_grid.txt"
SLURM_FILE="${REPO_DIR}/delftblue_scripts/11_phase1_worker.slurm"
DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"
GH_DATA_DIR="${DELFTBLUE_DATA_BASE:-/scratch/${USER}}/data/01_grasshopper_data"
REPORT_PARTITION="${REPORT_PARTITION:-compute}"
REPORT_TIME="${REPORT_TIME:-00:30:00}"
REPORT_CPUS="${REPORT_CPUS:-2}"
REPORT_MEM_PER_CPU="${REPORT_MEM_PER_CPU:-2G}"
REPORT_ACCOUNT="${REPORT_ACCOUNT:-education-abe-msc-a}"
ARRAY_MAX_CONCURRENT="${ARRAY_MAX_CONCURRENT:-2}"

mkdir -p "${REPO_DIR}/delftblue_logs"
mkdir -p "${REPO_DIR}/delftblue_logs/phase1" "${REPO_DIR}/delftblue_logs/reports"

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
  echo "Run: bash delftblue_scripts/01_setup_environment.sh" >&2
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

array_spec="1-${count}"
if [[ -n "${ARRAY_MAX_CONCURRENT}" ]]; then
  array_spec="${array_spec}%${ARRAY_MAX_CONCURRENT}"
fi

echo "Submitting one phase-1 array job for ${count} runs (array=${array_spec})..."
sweep_job_id=$(sbatch \
  --array="${array_spec}" \
  --output="${REPO_DIR}/delftblue_logs/phase1/c21_phase1_%A_%a.out" \
  --error="${REPO_DIR}/delftblue_logs/phase1/c21_phase1_%A_%a.err" \
  --export=ALL \
  "${SLURM_FILE}" | awk '{print $4}')
echo "Submitted phase-1 array job_id=${sweep_job_id}"

echo "Done. Monitor with: squeue -u ${USER}"

# Submit report job to run automatically after the phase-1 array completes.
report_job_id=$(sbatch \
  --dependency="afterany:${sweep_job_id}" \
  --job-name="c21_phase1_report" \
  --partition="${REPORT_PARTITION}" \
  --ntasks=1 \
  --cpus-per-task="${REPORT_CPUS}" \
  --mem-per-cpu="${REPORT_MEM_PER_CPU}" \
  --time="${REPORT_TIME}" \
  --account="${REPORT_ACCOUNT}" \
  --output="${REPO_DIR}/delftblue_logs/reports/c21_phase1_report_%j.out" \
  --error="${REPO_DIR}/delftblue_logs/reports/c21_phase1_report_%j.err" \
  --wrap="cd '${REPO_DIR}' && bash delftblue_scripts/30_generate_c21_report.sh" \
  | awk '{print $4}')

echo "Queued report job_id=${report_job_id} (runs after all submitted jobs finish)."
