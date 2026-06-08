#!/bin/bash
# Run LOCALLY (Git Bash on the Windows machine), NOT on DelftBlue.
# Pushes the code + the three data inputs to the cluster, placing the data in
# the repo-relative locations config.py falls back to when OneDrive is absent.
# Uses tar+scp (Git-for-Windows ships no rsync).
#
# Prereq: working `ssh delftblue` (key installed, on TU Delft network / eduVPN).
#     bash delftblue/upload.sh
set -euo pipefail

REMOTE=delftblue                       # ssh host alias from ~/.ssh/config
DEST=thesis_generative_timber          # remote dir under $HOME

REPO="/c/Users/Jasper/Documents/PyRepo/thesis_generative_timber"
ONEDRIVE="/c/Users/Jasper/OneDrive/06 Building Technology TU/2.2 - 2.4"
PREFIX="ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863"

TMPTAR="/tmp/ga_code_$$.tar.gz"

echo "1/4  Packing code (excluding caches / data dirs)..."
tar -czf "$TMPTAR" -C "$REPO" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    --exclude='60_Research_Exports' --exclude='30_Data_Inventory' \
    .

echo "2/4  Uploading + extracting code -> $REMOTE:$DEST/"
ssh "$REMOTE" "mkdir -p '$DEST'"
scp "$TMPTAR" "$REMOTE:$DEST/_code.tar.gz"
ssh "$REMOTE" "cd '$DEST' && tar -xzf _code.tar.gz && rm -f _code.tar.gz"
rm -f "$TMPTAR"

echo "3/4  Uploading stock CSV (scenario A)..."
ssh "$REMOTE" "mkdir -p '$DEST/30_Data_Inventory/03_timber_data'"
scp "$ONEDRIVE/30_Data_Inventory/03_timber_data/complete_timber_A.csv" \
    "$REMOTE:$DEST/30_Data_Inventory/03_timber_data/"

echo "4/4  Uploading GNN bundle..."
ssh "$REMOTE" "mkdir -p '$DEST/60_Research_Exports/01_surrogate_models'"
scp -r "$ONEDRIVE/60_Research_Exports/01_surrogate_models/$PREFIX" \
    "$REMOTE:$DEST/60_Research_Exports/01_surrogate_models/"

echo
echo "Upload complete. Next on the cluster:"
echo "    ssh $REMOTE"
echo "    cd $DEST && bash delftblue/setup_env.sh   # once"
echo "    sacctmgr show assoc user=\$USER format=account   # find your --account"
echo "    nano delftblue/jobscript.sbatch            # set --account, then:"
echo "    sbatch delftblue/jobscript.sbatch"
