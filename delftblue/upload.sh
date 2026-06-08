#!/bin/bash
# Run LOCALLY (Git Bash on the Windows machine), NOT on DelftBlue.
# Pushes the code + the three data inputs to the cluster, placing the data in
# the repo-relative locations config.py falls back to when OneDrive is absent.
#
# Prereq: working `ssh delftblue` (key installed, on TU Delft network / eduVPN).
#     bash delftblue/upload.sh
set -euo pipefail

REMOTE=delftblue                       # ssh host alias from ~/.ssh/config
DEST=thesis_generative_timber          # remote dir under $HOME

REPO="/c/Users/Jasper/Documents/PyRepo/thesis_generative_timber"
ONEDRIVE="/c/Users/Jasper/OneDrive/06 Building Technology TU/2.2 - 2.4"
PREFIX="ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863"

echo "1/3  Uploading code -> $REMOTE:$DEST/"
rsync -avz \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
    --exclude '60_Research_Exports' --exclude '30_Data_Inventory' \
    "$REPO/" "$REMOTE:$DEST/"

echo "2/3  Uploading stock CSV (scenario A)"
ssh "$REMOTE" "mkdir -p '$DEST/30_Data_Inventory/03_timber_data'"
rsync -avz \
    "$ONEDRIVE/30_Data_Inventory/03_timber_data/complete_timber_A.csv" \
    "$REMOTE:$DEST/30_Data_Inventory/03_timber_data/"

echo "3/3  Uploading GNN bundle"
ssh "$REMOTE" "mkdir -p '$DEST/60_Research_Exports/01_surrogate_models/$PREFIX'"
rsync -avz \
    "$ONEDRIVE/60_Research_Exports/01_surrogate_models/$PREFIX/" \
    "$REMOTE:$DEST/60_Research_Exports/01_surrogate_models/$PREFIX/"

echo
echo "Upload complete. Next on the cluster:"
echo "    ssh $REMOTE"
echo "    cd $DEST && bash delftblue/setup_env.sh   # once"
echo "    sbatch delftblue/jobscript.sbatch"
