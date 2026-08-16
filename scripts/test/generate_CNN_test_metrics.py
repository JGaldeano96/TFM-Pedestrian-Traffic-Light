"""
generate_CNN_test_metrics.py

Evalúa los modelos CNN entrenados con dataset_v1 y dataset_v2
sobre los datasets independientes de test diurno y nocturno.

Convención de clases utilizada en TODO el proyecto:

    0 -> Red
    1 -> Green

La salida sigmoid de los modelos representa:

    P(Green)

Por tanto:

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

# IMPORTANTE:
#
# image_dataset_from_directory() ordena las carpetas
# alfabéticamente:
#
#     0 -> Green
#     1 -> Red
#
# Sin embargo, los modelos han sido entrenados utilizando:
#
#     0 -> Red
#     1 -> Green
#
# Las funciones de test_functions.py se encargan de invertir
# las etiquetas del dataset de test antes de calcular
# las métricas.

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
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "CNN CLASSIFIER TEST"
    )

    print("=" * 70)

    print()

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

    print(
        "Model output:"
    )

    print(
        "  sigmoid -> P(Green)"
    )

    print()

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
    # GENERATE METRICS
    # ========================================================

    generate_classifier_test_metrics(
        results_root=RESULTS_ROOT,
        test_dataset_root=TEST_DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    print()

    print(
        "Test evaluation completed successfully."
    )