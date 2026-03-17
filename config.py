from pathlib import Path
import sys

# Dit vindt de map '50_Repository' waar dit bestand in staat
REPO_ROOT = Path(__file__).resolve().parent

# De gedeelde hoofdmap 'Thesis_Project' (één niveau omhoog)
PROJECT_ROOT = REPO_ROOT.parent

# De specifieke mappen met jouw namen
DATA_DIR = PROJECT_ROOT / "30_Data_Inventory"
EXPORT_DIR = PROJECT_ROOT / "60_Research_Exports"
SRC_DIR = REPO_ROOT / "src"  # Nieuwe toevoeging!

# Submappen binnen de data-inventory
GH_DATA_DIR = DATA_DIR / "02_grasshopper_data"
RAW_DATA_DIR = DATA_DIR / "03_raw_data"

# Handig: Maak de export-map automatisch aan als deze nog niet bestaat
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

print(f"Systeem geladen. Project root: {PROJECT_ROOT.name}")