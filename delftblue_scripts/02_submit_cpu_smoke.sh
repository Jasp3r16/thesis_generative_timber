#!/bin/bash
set -euo pipefail

# Submit the CPU smoke test job.
# Usage:
#   bash delftblue_scripts/02_submit_cpu_smoke.sh

mkdir -p delftblue_logs

jid=$(sbatch delftblue_scripts/02_cpu_smoke_worker.slurm | awk '{print $4}')
echo "Submitted CPU smoke job_id=${jid}"
echo "Monitor: squeue -u ${USER}"
