from pathlib import Path

# Dit vindt de map '50_Repository' waar dit bestand in staat
REPO_ROOT = Path(__file__).resolve().parent

# De gedeelde hoofdmap 'Thesis_Project' (één niveau omhoog)
PROJECT_ROOT = REPO_ROOT.parent

# De specifieke mappen met jouw namen
DATA_DIR = PROJECT_ROOT / "30_Data & Inventory"
EXPORT_DIR = PROJECT_ROOT / "60_Research Exports"

# Handig: Maak de export-map automatisch aan als deze nog niet bestaat
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Systeem geladen. Project root: {PROJECT_ROOT.name}")