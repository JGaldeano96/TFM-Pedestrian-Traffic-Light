"""
generate_classifier_dataset.py

Genera un dataset de clasificación a partir de una exportación de
Label Studio.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.utils.dataset_functions import (
    generate_classifier_dataset,
)


# ============================================================
# DATASET VERSION
# ============================================================

DATASET_VERSION = "v1"


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 128


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

CLASSIFIER_DATASET_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "classifier"
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_classifier_dataset(
        root_dir=ROOT_DIR,
        json_path=JSON_PATH,
        split_path=SPLIT_PATH,
        classifier_dataset_dir=CLASSIFIER_DATASET_DIR,
        image_size=IMAGE_SIZE,
    )