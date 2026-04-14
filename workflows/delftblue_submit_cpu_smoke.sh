#!/bin/bash
set -euo pipefail

# Submit the CPU smoke test job.
# Usage:
#   bash workflows/delftblue_submit_cpu_smoke.sh

mkdir -p logs

jid=$(sbatch workflows/delftblue_cpu_smoke.slurm | awk '{print $4}')
echo "Submitted CPU smoke job_id=${jid}"
echo "Monitor: squeue -u ${USER}"
