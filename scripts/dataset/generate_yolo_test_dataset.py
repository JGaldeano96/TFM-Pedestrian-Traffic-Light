"""
generate_yolo_test_dataset.py

Genera los datasets YOLO independientes de testing
a partir de las exportaciones de Label Studio.

Los datasets se mantienen separados en:

    data/datasets/yolo_test/diurn/
    data/datasets/yolo_test/nocturn/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_DIR))

from scripts.utils.dataset_functions import (
    generate_yolo_test_dataset,
)


# ============================================================
# RUTAS
# ============================================================

ANNOTATION_EXPORTS_DIR = (
    ROOT_DIR
    / "data"
    / "annotation_exports"
)

YOLO_TEST_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "yolo_test"
)


# ============================================================
# TEST DATASETS
# ============================================================

TEST_DATASETS = {

    "diurn": (
        ANNOTATION_EXPORTS_DIR
        / "test_diurn.json"
    ),

    "nocturn": (
        ANNOTATION_EXPORTS_DIR
        / "test_nocturn.json"
    ),

}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for dataset_name, json_path in TEST_DATASETS.items():

        generate_yolo_test_dataset(
            root_dir=ROOT_DIR,
            json_path=json_path,
            output_dir=(
                YOLO_TEST_DIR
                / dataset_name
            ),
        )