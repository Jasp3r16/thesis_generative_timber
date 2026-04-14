"""
DelftBlue-specific configuration override.
This is imported in config.py when running on DelftBlue.

Set DELFTBLUE_DATA_BASE environment variable to your user's scratch directory.
Example: export DELFTBLUE_DATA_BASE=/scratch/username
"""

from pathlib import Path
import sys
import os

# ==========================================
# 1. LOCAL CODE REPOSITORY (same on DelftBlue)
# ==========================================
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
WORKFLOWS_PATH = REPO_ROOT / "workflows"
DATA_IO_PATH = REPO_ROOT / "02_data_io"

# Make sure Python can always resolve imports from the src folder.
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
if str(WORKFLOWS_PATH) not in sys.path:
    sys.path.append(str(WORKFLOWS_PATH))
if str(DATA_IO_PATH) not in sys.path:
    sys.path.append(str(DATA_IO_PATH))

# ==========================================
# 2. DELFTBLUE SCRATCH STORAGE
# ==========================================
# Read from environment variable set by SLURM script
DELFTBLUE_DATA_BASE = os.getenv("DELFTBLUE_DATA_BASE")
if not DELFTBLUE_DATA_BASE:
    # Fallback: assume standard DelftBlue scratch path for user
    import getpass
    username = getpass.getuser()
    DELFTBLUE_DATA_BASE = f"/scratch/{username}"

DELFTBLUE_DATA_BASE = Path(DELFTBLUE_DATA_BASE)

# Specific folders in scratch storage
DATA_PATH = DELFTBLUE_DATA_BASE / "data"
EXPORT_PATH = DELFTBLUE_DATA_BASE / "results"

# Subfolders inside the data directory
GH_DATA_PATH = DATA_PATH / "01_grasshopper_data"
RAW_DATA_PATH = DATA_PATH / "02_raw_data"
TIMBER_STOCK_PATH = DATA_PATH / "03_timber_data"

# Subfolders inside the research exports
SM_EXPORT_PATH = EXPORT_PATH / "01_surrogate_models"
SM_DATA_PATH = EXPORT_PATH / "02_surrogate_model_data"

# ==========================================
# 3. INITIALIZATION
# ==========================================
# Create directories if they don't exist
EXPORT_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)
SM_EXPORT_PATH.mkdir(parents=True, exist_ok=True)
SM_DATA_PATH.mkdir(parents=True, exist_ok=True)

print("System loaded successfully (DelftBlue mode).\n")
print(f"Code is running locally from: {REPO_ROOT.name}")
print(f"Data connected to DelftBlue scratch: {DELFTBLUE_DATA_BASE}\n")
