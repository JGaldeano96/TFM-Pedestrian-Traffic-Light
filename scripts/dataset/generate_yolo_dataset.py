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