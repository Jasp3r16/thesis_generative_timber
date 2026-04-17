#!/bin/bash
set -euo pipefail

# Submit the CPU smoke test job.
# Usage:
#   bash delftblue_scripts/02_submit_cpu_smoke.sh

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SLURM_FILE="${REPO_DIR}/delftblue_scripts/02_cpu_smoke_worker.slurm"
DELFTBLUE_VENV="${DELFTBLUE_VENV:-/scratch/${USER}/venvs/thesis_gnn}"
GH_DATA_DIR="${DELFTBLUE_DATA_BASE:-/scratch/${USER}}/data/01_grasshopper_data"
SMOKE_PARTITION="${SMOKE_PARTITION:-compute}"
SMOKE_TIME="${SMOKE_TIME:-00:20:00}"
SMOKE_CPUS="${SMOKE_CPUS:-2}"
SMOKE_MEM_PER_CPU="${SMOKE_MEM_PER_CPU:-2G}"
SMOKE_ACCOUNT="${SMOKE_ACCOUNT:-education-abe-msc-a}"

mkdir -p "${REPO_DIR}/delftblue_logs" "${REPO_DIR}/delftblue_logs/smoke"

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

echo "Preflight OK"
echo "- Venv: ${DELFTBLUE_VENV}"
echo "- Data: ${GH_DATA_DIR}"
echo "- Worker: ${SLURM_FILE}"

jid=$(sbatch \
	--partition="${SMOKE_PARTITION}" \
	--ntasks=1 \
	--cpus-per-task="${SMOKE_CPUS}" \
	--mem-per-cpu="${SMOKE_MEM_PER_CPU}" \
	--time="${SMOKE_TIME}" \
	--account="${SMOKE_ACCOUNT}" \
	--output="${REPO_DIR}/delftblue_logs/smoke/c21_cpu_smoke_%j.out" \
	--error="${REPO_DIR}/delftblue_logs/smoke/c21_cpu_smoke_%j.err" \
	--export=ALL \
	"${SLURM_FILE}" | awk '{print $4}')
echo "Submitted CPU smoke job_id=${jid}"
echo "Monitor: squeue -u ${USER}"
