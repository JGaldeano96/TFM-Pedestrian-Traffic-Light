"""
generate_CNN_test_metrics.py

Evalúa los modelos CNN entrenados con dataset_v1 y dataset_v2
sobre los datasets independientes de test.

Convención de clases:

    0 -> Red
    1 -> Green

La salida sigmoid de los modelos representa:

    P(Green)

Regla de decisión:

    probability >= threshold -> Green
    probability < threshold  -> Red
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# ROOT
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.insert(
    0,
    str(ROOT_DIR),
)


# ============================================================
# IMPORTS
# ============================================================

from scripts.utils.test_functions import (
    generate_classifier_test_metrics,
)


# ============================================================
# CLASS MAPPING
# ============================================================

# image_dataset_from_directory() asigna:
#
#     0 -> Green
#     1 -> Red
#
# Las funciones de test invierten estas etiquetas para utilizar
# nuestra convención:
#
#     0 -> Red
#     1 -> Green

CLASS_NAMES = [
    "Red",
    "Green",
]


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = (
    ROOT_DIR
    / "results"
    / "classifier"
)


TEST_DATASET_ROOT = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "classifier_test"
)


OUTPUT_DIR = (
    ROOT_DIR
    / "results"
    / "classifier"
    / "test"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (
    128,
    128,
)

BATCH_SIZE = 32


# ============================================================
# MODEL THRESHOLDS
# ============================================================

# Threshold definitivo utilizado para la evaluación de cada
# modelo.
#
# Estos valores han sido seleccionados a partir del análisis
# realizado sobre la distribución de P(Green), ROC, Youden y
# comportamiento de FP/FN.

MODEL_THRESHOLDS = {
    "dataset_v1": 0.047,
    "dataset_v2": 0.800,
}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "CNN CLASSIFIER TEST"
    )

    print("=" * 70)

    print()

    # ========================================================
    # CLASS MAPPING
    # ========================================================

    print(
        "Class mapping:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {index} -> {class_name}"
        )

    print()

    # ========================================================
    # MODEL OUTPUT
    # ========================================================

    print(
        "Model output:"
    )

    print(
        "  sigmoid -> P(Green)"
    )

    print()

    # ========================================================
    # DECISION RULE
    # ========================================================

    print(
        "Decision rule:"
    )

    print(
        "  probability >= threshold -> Green"
    )

    print(
        "  probability < threshold  -> Red"
    )

    print()

    # ========================================================
    # MODEL THRESHOLDS
    # ========================================================

    print(
        "Model thresholds:"
    )

    for dataset_version, threshold in MODEL_THRESHOLDS.items():

        print(
            f"  {dataset_version} -> "
            f"{threshold:.3f}"
        )

    print()

    # ========================================================
    # GENERATE METRICS
    # ========================================================

    generate_classifier_test_metrics(
        results_root=RESULTS_ROOT,
        test_dataset_root=TEST_DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        thresholds={
            "dataset_v1": 0.047,
            "dataset_v2": 0.8,
        },
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    print()

    print(
        "Test evaluation completed successfully."
    )