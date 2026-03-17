from pathlib import Path
import sys

# ==========================================
# 1. LOKALE CODE REPOSITORY (PyRepo op C-schijf)
# ==========================================
# Dit vindt de map '50_Repository' waar dit script in staat
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"

# Zorg dat Python de 'src' map altijd kan vinden voor je imports
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


# ==========================================
# 2. CLOUD DATA OPSLAG (OneDrive)
# ==========================================
# Path.home() pakt automatisch C:\Users\JouwNaam, ongeacht welke laptop je gebruikt!
ONEDRIVE_ROOT = Path.home() / "OneDrive" / "06 Building Technology TU" / "2.2 - 2.4"

# De specifieke mappen op je OneDrive
DATA_PATH = ONEDRIVE_ROOT / "30_Data_Inventory"
EXPORT_PATH = ONEDRIVE_ROOT / "60_Research_Exports"

# Submappen binnen de data-inventory
GH_DATA_PATH = DATA_PATH / "01_grasshopper_data"
RAW_DATA_PATH = DATA_PATH / "02_raw_data"


# ==========================================
# 3. INITIALISATIE
# ==========================================
# Handig: Maak de export-map automatisch aan op OneDrive als deze nog niet bestaat
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

print(f"✅ Systeem succesvol geladen.")
print(f"📂 Code draait lokaal vanuit: {REPO_ROOT.name}")
print(f"☁️ Data gekoppeld aan OneDrive: {ONEDRIVE_ROOT.name}")

'''
# Optionele debug prints voor als je paden wilt checken
print(f"Data directory: {DATA_PATH}")
print(f"GH data directory: {GH_DATA_PATH}")
print(f"Raw data directory: {RAW_DATA_PATH}")
'''
