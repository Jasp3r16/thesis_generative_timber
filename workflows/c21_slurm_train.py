#!/usr/bin/env python
"""
c21_slurm_train.py - SLURM wrapper for c21 training
Executes c21_train.main() in batch mode (no Jupyter overhead).
Used by DelftBlue SLURM scripts as: python c21_slurm_train.py
"""

import sys
import os

# Add repo root to path so imports work from workflows/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.c21_train import main

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error:\n{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
