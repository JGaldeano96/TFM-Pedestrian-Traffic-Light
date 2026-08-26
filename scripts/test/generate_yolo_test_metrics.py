# =============================================================================
# DECLARACIÓN SOBRE EL USO DE INTELIGENCIA ARTIFICIAL GENERATIVA
# =============================================================================
#
# El desarrollo de este código ha contado con la asistencia de herramientas de
# Inteligencia Artificial Generativa como apoyo en tareas de implementación,
# revisión y mejora del código.

# La arquitectura de la solución, las decisiones técnicas y metodológicas, la
# selección y configuración de modelos, el diseño de los experimentos, las
# métricas de evaluación y la interpretación de los resultados han sido
# definidos por el autor.

# # Todo el código incorporado al proyecto ha sido revisado, comprendido,
# adaptado, integrado y probado por el autor antes de su utilización.
#
# La IA generativa ha sido empleada únicamente como herramienta de apoyo, sin
# sustituir el criterio técnico ni la responsabilidad del autor sobre el
# desarrollo y validación del proyecto.
# =============================================================================


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