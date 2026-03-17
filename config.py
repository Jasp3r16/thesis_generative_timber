from pathlib import Path
import sys

# Dit vindt de map '50_Repository' waar dit bestand in staat
REPO_ROOT = Path(__file__).resolve().parent

# De gedeelde hoofdmap 'Thesis_Project' (één niveau omhoog)
PROJECT_ROOT = REPO_ROOT.parent

# De specifieke mappen met jouw namen
DATA_PATH = PROJECT_ROOT / "30_Data_Inventory"
EXPORT_PATH = PROJECT_ROOT / "60_Research_Exports"
SRC_PATH = REPO_ROOT / "src"  # Nieuwe toevoeging!

# Submappen binnen de data-inventory
GH_DATA_PATH = DATA_PATH / "01_grasshopper_data"
RAW_DATA_PATH = DATA_PATH / "02_raw_data"

# Handig: Maak de export-map automatisch aan als deze nog niet bestaat
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

print(f"Systeem geladen. Project root: {PROJECT_ROOT.name}")

'''
print(f"Data directory: {DATA_PATH}")
print(f"GH data directory: {GH_DATA_PATH}")
print(f"Raw data directory: {RAW_DATA_PATH}")
'''