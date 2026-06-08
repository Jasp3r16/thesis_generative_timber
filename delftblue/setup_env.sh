#!/bin/bash
# One-time environment setup on DelftBlue.
# Run on a LOGIN node (it only installs packages; no heavy compute):
#     bash delftblue/setup_env.sh
#
# Creates a conda env "timber-ga" with CPU-only PyTorch + torch-geometric and
# the rest of the project dependencies. The compute partition has no GPU, so a
# CPU torch build is intentional (smaller, no CUDA runtime needed).
set -euo pipefail

module load 2024r1
module load miniconda3

# Make `conda activate` work inside a non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=timber-ga

if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "Env '${ENV_NAME}' already exists — skipping create. Re-run after 'conda env remove -n ${ENV_NAME}' to rebuild."
else
    conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# CPU-only PyTorch first, so torch-geometric resolves against it.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric

# Remaining deps (>= versions from requirements.txt; CPU torch already pinned).
python -m pip install \
    "numpy>=2.4" "pandas>=3.0" "scipy>=1.17" "scikit-learn>=1.8" \
    "matplotlib>=3.10" "seaborn>=0.13" "PuLP>=3.3" "cma>=4.4"

echo
echo "Environment '${ENV_NAME}' ready. Verifying core imports..."
python - <<'PY'
import torch, torch_geometric, numpy, pandas, pulp, cma, sklearn
print("torch            ", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("torch_geometric  ", torch_geometric.__version__)
print("numpy            ", numpy.__version__)
print("pandas           ", pandas.__version__)
print("pulp             ", pulp.__version__)
print("cma              ", cma.__version__)
print("OK")
PY
