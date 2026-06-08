# Running `c23_run_ga_batch.py` on DelftBlue

Self-contained instructions for running the GA batch (scenario A, 500
generations) on TU Delft's DelftBlue cluster. The code runs **unmodified** — on
the cluster `config.py` finds no OneDrive and falls back to repo-relative data
paths, which `upload.sh` populates.

| Item | Value |
|------|-------|
| Login node | `login.delftblue.tudelft.nl` (needs TU Delft network / **eduVPN**) |
| Partition | `compute` (CPU; GNN is single-graph inference, MILP dominates) |
| Job config | scenario A, `N_RUNS=3` (seeds 42/43/44), `CMAES_GENERATIONS=500`, popsize 30 |
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
sbatch delftblue/jobscript.sbatch
```

## 5. Monitor
```bash
squeue -u $USER                 # queue / running state
tail -f logs/ga_batch_<JOBID>.out   # live per-generation progress
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem   # after it finishes
```

## 6. Retrieve results (local, Git Bash)
GA exports land under `60_Research_Exports/03_ga_data/` on the cluster.
```bash
rsync -avz "delftblue:thesis_generative_timber/60_Research_Exports/03_ga_data/" \
  "/c/Users/Jasper/OneDrive/06 Building Technology TU/2.2 - 2.4/60_Research_Exports/03_ga_data/"
```

## Walltime note
The `.sbatch` requests **24 h**. That covers 3 sequential 500-gen runs only if a
single run stays under ~8 h; `CMAES_STAGNATION=30` often ends runs early, but
worst case (no early stop) could exceed 24 h. If runs are slow, switch to **one
seed per job** via a SLURM array (see "Array variant" below) so each seed gets
its own 24 h budget and they run in parallel.

### Array variant (one seed per job)
Submit three independent jobs (seeds 42/43/44) instead of one sequential job —
faster wall-clock and each fits the 24 h limit comfortably. This needs the small
env-var hook in the run script; ask before enabling, as it changes `c23_run_ga_batch.py`.
