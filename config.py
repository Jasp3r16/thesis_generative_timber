from pathlib import Path
import sys
import os

# ==========================================
# 0. ENVIRONMENT DETECTION (DelftBlue vs Local)
# ==========================================
# Check if running on DelftBlue by looking for SLURM environment variable
IS_DELFTBLUE = "SLURM_ARRAY_TASK_ID" in os.environ or os.getenv("DELFTBLUE_DATA_BASE")

if IS_DELFTBLUE:
    # Import DelftBlue-specific configuration
    from config_delftblue import (
        REPO_ROOT, SRC_PATH, WORKFLOWS_PATH, DATA_IO_PATH,
        DATA_PATH, EXPORT_PATH, GH_DATA_PATH, RAW_DATA_PATH, 
        TIMBER_STOCK_PATH, SM_EXPORT_PATH, SM_DATA_PATH
    )
    print("✓ Loaded DelftBlue configuration\n")

else:
    # ==========================================
    # 1. LOCAL CODE REPOSITORY (PyRepo on C drive)
    # ==========================================
    # This resolves the repository folder where this script lives.
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
    # 2. CLOUD DATA STORAGE (OneDrive)
    # ==========================================
    # Path.home() automatically resolves C:\Users\YourName on any machine.
    ONEDRIVE_ROOT = Path.home() / "OneDrive" / "06 Building Technology TU" / "2.2 - 2.4"

    # Specific folders in OneDrive.
    DATA_PATH = ONEDRIVE_ROOT / "30_Data_Inventory"
    EXPORT_PATH = ONEDRIVE_ROOT / "60_Research_Exports"

    # Subfolders inside the data inventory.
    GH_DATA_PATH = DATA_PATH / "01_grasshopper_data"
    RAW_DATA_PATH = DATA_PATH / "02_raw_data"
    TIMBER_STOCK_PATH = DATA_PATH / "03_timber_data"

    # Subfolders inside the research exports.
    SM_EXPORT_PATH = EXPORT_PATH / "01_surrogate_models"
    SM_DATA_PATH = EXPORT_PATH / "02_surrogate_model_data"
    GA_DATA_PATH = EXPORT_PATH / "03_ga_data"

    # ==========================================
    # 3. INITIALIZATION
    # ==========================================
    # Create the export folder automatically on OneDrive if it does not exist yet.
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    GA_DATA_PATH.mkdir(parents=True, exist_ok=True)

    print("System loaded successfully.\n")
    print(f"Code is running locally from: {REPO_ROOT.name}")
    print(f"Data connected to OneDrive: {ONEDRIVE_ROOT.name}\n")
    
print(f"GH data directory: {GH_DATA_PATH}")
print(f"Raw data directory: {RAW_DATA_PATH}")
print(f"Export directory: {EXPORT_PATH}\n")