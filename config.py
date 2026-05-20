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
    GEOM_DATA_PATH = DATA_PATH / "01_geometry_data"
    GH_DATA_PATH = DATA_PATH / "02_grasshopper_data"
    TIMBER_STOCK_PATH = DATA_PATH / "03_timber_data"

    # Subfolders inside the research exports.
    SM_EXPORT_PATH = EXPORT_PATH / "01_surrogate_models"
    SM_DATA_PATH = EXPORT_PATH / "02_surrogate_model_data"
    GA_DATA_PATH = EXPORT_PATH / "03_ga_data"

    print(f"Config System loaded successfully, Code running locally from {REPO_ROOT.name} and Data is connected to OneDrive {ONEDRIVE_ROOT.name}.\n")
    
# ==========================================
# 4. VISUALIZATION THEME (Centralized Color Palette)
# ==========================================
# Color scheme for all plots and visualizations
PLOT_COLORS = {
    "primary": "#61788C",       # Dark blue (convergence best, primary lines)
    "secondary": "#9CA5A6",     # Sage/light blue-green (mean, secondary)
    "accent": "#F2994B",        # Orange (warnings, thresholds)
    "danger": "#D9653B",        # Red-orange (worst, failures)
    "neutral": "#D7D9D9",       # Light gray (background, grids)
    "black": "#000000",         # Black (text, axes)
    "white": "#FFFFFF",         # White (background)
    "NS": "#61788C",
    "RS": "#F2994B",
    "extra_colors": {
    "deep_navy": "#2F3E4F",
    "muted_teal": "#4F8A8B",
    "soft_sage_green": "#A8B89A",
    "dusty_plum": "#7B667D"}

}

PLOT_STYLE = {
    "figsize_small": (8, 5),
    "figsize_medium": (12, 7),
    "figsize_large": (16, 10),
    "dpi": 100,
    "grid_alpha": 0.3,
    "line_width": 2.0,
    "marker_size": 5,
}