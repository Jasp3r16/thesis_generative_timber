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
from workflows.c21_slurm_evaluate import export_slurm_evaluation

if __name__ == "__main__":
    try:
        results = main()

        export_eval = os.getenv("C21_EXPORT_EVAL", "true").lower() == "true"
        eval_strict = os.getenv("C21_EVAL_STRICT", "false").lower() == "true"

        if export_eval:
            try:
                export_slurm_evaluation(results)
            except Exception as eval_exc:
                print(f"\n⚠ SLURM post-evaluation export failed:\n{eval_exc}", file=sys.stderr)
                if eval_strict:
                    raise

        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed with error:\n{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
