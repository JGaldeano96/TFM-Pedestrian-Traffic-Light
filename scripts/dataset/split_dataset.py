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
split_dataset.py

Divide el dataset en TRAIN y VAL utilizando vídeos completos como
unidad de partición.

La estrategia consiste en generar múltiples particiones aleatorias
(Random Search) y seleccionar aquella que mejor equilibra:

- La proporción de objetos (70/30 por defecto).
- La distribución de las clases Red y Green.

No copia imágenes.
No genera etiquetas YOLO.
No escribe archivos.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.utils.dataset_functions import (
    load_dataset,
    split_dataset,
    summarize_split,
)

JSON_PATH = (
    ROOT_DIR
    / "data"
    / "annotation_exports"
    / "dataset_v1.json"
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = load_dataset(JSON_PATH)

    split = split_dataset(
        df=df,
        train_size=0.70,
        iterations=10000,
        random_state=42,
    )

    summarize_split(
        df=df,
        split=split,
    )

    # ----------------------------------------------------------
    # Comprobar que ningún frame aparece en ambos conjuntos
    # ----------------------------------------------------------

    train_images = set(
        df[
            df["video_id"].isin(split["train"])
        ]["image_path"]
    )

    val_images = set(
        df[
            df["video_id"].isin(split["val"])
        ]["image_path"]
    )

    duplicated = train_images & val_images

    if duplicated:

        raise RuntimeError(
            f"Se han encontrado {len(duplicated)} imágenes duplicadas entre TRAIN y VAL."
        )

    print("\n✓ No existe data leakage entre TRAIN y VAL.")
