"""
create_classifier_test_dataset.py

Genera datasets independientes de testing para el clasificador
a partir de exportaciones de Label Studio.

Los datasets se mantienen separados por condiciones de iluminación:

data/datasets/classifier_test/
├── diurn/
│   ├── Red/
│   └── Green/
└── nocturn/
    ├── Red/
    └── Green/

Los datasets de test son completamente independientes de los
datasets de entrenamiento y validación.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# ROOT DIRECTORY
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_DIR))


from scripts.utils.dataset_functions import (
    generate_classifier_test_dataset,
)


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 128


# ============================================================
# RUTAS
# ============================================================

ANNOTATION_EXPORTS_DIR = (
    ROOT_DIR
    / "data"
    / "annotation_exports"
)


CLASSIFIER_TEST_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "classifier_test"
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

        generate_classifier_test_dataset(
            root_dir=ROOT_DIR,
            json_path=json_path,
            output_dir=(
                CLASSIFIER_TEST_DIR
                / dataset_name
            ),
            image_size=IMAGE_SIZE,
        )