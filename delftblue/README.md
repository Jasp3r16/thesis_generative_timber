# Running `c23_run_ga_batch.py` on DelftBlue

Self-contained instructions for running the GA batch (scenario A, 500
generations) on TU Delft's DelftBlue cluster. The code runs **unmodified** — on
the cluster `config.py` finds no OneDrive and falls back to repo-relative data
paths, which `upload.sh` populates.

| Item | Value |
|------|-------|
| Login node | `login.delftblue.tudelft.nl` (needs TU Delft network / **eduVPN**) |
| Partition | `compute` (CPU; GNN is single-graph inference, MILP dominates) |
| Job type | **SLURM array** `0-2` — one seed per task (42/43/44 -> RUN1/2/3), all parallel |
| Per-task config | scenario A, `CMAES_GENERATIONS=500`, popsize 30 = 15,000 evals (~4-6 h CPU) |
| Data uploaded | `complete_timber_A.csv` (60 KB), `search_space_5x3.json` (in repo), GNN bundle (~4.4 MB) |

## 0. Prerequisites
- Be on **eduVPN** (or the wired TU Delft network). The login node is not reachable from the open internet.
- A valid SLURM **account/allocation**. List yours on the cluster with:
  `sacctmgr show assoc user=$USER format=account` — put one into `jobscript.sbatch` (`--account=`).

## 1. SSH access (one-time)
A dedicated key was generated locally at `~/.ssh/id_ed25519_delftblue`.
Add this block to `~/.ssh/config` (replace `NETID`):

```
Host delftblue
    HostName login.delftblue.tudelft.nl
    User NETID
    IdentityFile ~/.ssh/id_ed25519_delftblue
```

Install the public key on DelftBlue (run once, in your own terminal — it will
prompt for your **NetID password**):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_delftblue.pub NETID@login.delftblue.tudelft.nl
# If ssh-copy-id is unavailable (plain Windows OpenSSH), do it manually:
#   type ~/.ssh/id_ed25519_delftblue.pub  ->  append to ~/.ssh/authorized_keys on DelftBlue
```

Verify: `ssh delftblue` should log in without a password prompt.

## 2. Upload code + data (local, Git Bash)
```bash
bash delftblue/upload.sh
```

## 3. Build the environment (cluster, login node, one-time)
```bash
ssh delftblue
cd thesis_generative_timber
bash delftblue/setup_env.sh
```

## 4. Submit
```bash
# still on the cluster, in the repo root
nano delftblue/jobscript.sbatch   # set --account=... first
sbatch delftblue/jobscript.sbatch # launches array tasks 0,1,2 (seeds 42/43/44)
```

## 5. Monitor
```bash
squeue -u $USER                      # all three array tasks
tail -f logs/ga_A_<ARRAYJOBID>_0.out # live per-generation progress for task 0 (RUN1)
sacct -j <ARRAYJOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem  # after finish
```
`<ARRAYJOBID>` is the base job id; per-task logs are `..._0`, `..._1`, `..._2`.

## 6. Retrieve results (local, Git Bash)
GA exports land under `60_Research_Exports/03_ga_data/` on the cluster. The pull
runs on the local (rsync-less) machine, so use `scp -r`:
```bash
scp -r "delftblue:thesis_generative_timber/60_Research_Exports/03_ga_data" \
  "/c/Users/Jasper/OneDrive/06 Building Technology TU/2.2 - 2.4/60_Research_Exports/"
```

## Timing & walltime

Measured per-eval cost (real pipeline, scenario A):

| Environment | Per-eval | 500-gen run (15,000 evals) |
|-------------|----------|----------------------------|
| Local laptop **GPU** (RTX A1000) | ~0.72 s (from past 250-gen logs) | ~3 h |
| Local **CPU** (benchmark, all full-pipeline evals) | ~1.37 s | ~5.7 h (upper bound) |
| DelftBlue `compute` (CPU) | expected ~1.0-1.4 s | ~4-6 h |

DelftBlue's `compute` partition has **no GPU**, so per-eval is ~2x the GPU
laptop number — a 500-gen run is ~4-6 h, not ~3 h. The array gives each seed its
own 12 h budget (2x+ margin) and runs all three in parallel, so wall-clock to all
results is ~4-6 h regardless. `CMAES_STAGNATION=30` may end runs earlier.

A GPU partition would cut a run back toward ~3 h but needs a GPU allocation,
longer queue waits, and a CUDA torch build — not worth it for a 120-edge graph
where the MILP solve (CPU-only) is a large share of per-eval cost.

The env-var hook in `c23_run_ga_batch.py` (`GA_SCENARIO`, `GA_GENERATIONS`,
`GA_RUN_INDEX`) is what lets each array task run one seed; it is a no-op when the
vars are unset, so local runs are unchanged. Verified locally: task index 2 ->
seed 44 -> `RUN3` export, with normalization bounds identical across tasks.
