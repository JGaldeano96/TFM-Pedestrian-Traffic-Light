"""
generate_yolo_test_metrics.py

Evalúa todos los modelos YOLO entrenados sobre los datasets
independientes de test diurno y nocturno.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# ROOT
# ============================================================

ROOT_DIR = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT_DIR),
)


# ============================================================
# IMPORTS
# ============================================================

from scripts.utils.test_functions import (
    generate_yolo_test_metrics,
)


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = (
    ROOT_DIR
    / "results"
    / "yolo"
)


TEST_DATASET_ROOT = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "yolo_test"
)


OUTPUT_CSV = (
    ROOT_DIR
    / "results"
    / "yolo"
    / "test"
    / "yolo_test_metrics.csv"
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_yolo_test_metrics(
        results_root=RESULTS_ROOT,
        test_dataset_root=TEST_DATASET_ROOT,
        output_csv=OUTPUT_CSV,
    )