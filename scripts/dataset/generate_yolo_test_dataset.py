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