# DelftBlue Script Run Order

Use these scripts in order from the repository root.

1. Environment setup (run once, or after dependency changes)
   - `bash delftblue_scripts/01_setup_environment.sh`

2. Optional smoke test (quick CPU validation)
   - `bash delftblue_scripts/02_submit_cpu_smoke.sh`
   - Worker used: `02_cpu_smoke_worker.slurm`

3. Phase 1 full sweep (GPU)
   - `bash delftblue_scripts/10_submit_phase1.sh`
   - Worker used: `11_phase1_worker.slurm`

4. Phase 2 full sweep (GPU)
   - `bash delftblue_scripts/20_submit_phase2.sh`
   - Worker used: `21_phase2_worker.slurm`

5. Report generation (usually auto-submitted by phase submit scripts)
   - `bash delftblue_scripts/30_generate_c21_report.sh`

## Notes

- Phase submit scripts submit one SLURM array job per phase and auto-queue the report script after it finishes.
- Default array concurrency is hard-coded to `2` in phase submit scripts (education-account safe).
- Optional override, for example:
   - `ARRAY_MAX_CONCURRENT=1 bash delftblue_scripts/10_submit_phase1.sh`
- Monitor queue with: `squeue -u $USER`
- Grid files are still:
  - `delftblue_hyperparameter_grid.txt` (phase 1)
  - `delftblue_hyperparameter_grid_phase2.txt` (phase 2)
