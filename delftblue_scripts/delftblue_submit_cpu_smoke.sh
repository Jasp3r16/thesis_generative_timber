#!/bin/bash
set -euo pipefail

# Submit the CPU smoke test job.
# Usage:
#   bash delftblue_scripts/delftblue_submit_cpu_smoke.sh

mkdir -p delfblue_logs

jid=$(sbatch delftblue_scripts/delftblue_cpu_smoke.slurm | awk '{print $4}')
echo "Submitted CPU smoke job_id=${jid}"
echo "Monitor: squeue -u ${USER}"
