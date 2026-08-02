"""
generate_yolo_dataset.py

Genera un dataset YOLO a partir de una exportación de Label Studio.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.utils.dataset_functions import generate_yolo_dataset


# ============================================================
# DATASET VERSION
# ============================================================

DATASET_VERSION = "v2"


# ============================================================
# RUTAS
# ============================================================

JSON_PATH = (
    ROOT_DIR
    / "data"
    / "annotation_exports"
    / f"dataset_{DATASET_VERSION}.json"
)

SPLIT_PATH = (
    ROOT_DIR
    / "data"
    / "splits"
    / f"split_{DATASET_VERSION}.json"
)

YOLO_DATASET_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "yolo"
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_yolo_dataset(
        root_dir=ROOT_DIR,
        json_path=JSON_PATH,
        split_path=SPLIT_PATH,
        yolo_dataset_dir=YOLO_DATASET_DIR,
    )