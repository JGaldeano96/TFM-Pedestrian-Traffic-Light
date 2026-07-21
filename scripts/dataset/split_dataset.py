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

from pathlib import Path

from label_studio_parser import load_dataset

from scripts.utils.dataset_functions import (
    split_dataset,
    summarize_split,
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    JSON_PATH = Path(
        "/home/jgaldeano/tfm/data/annotation_exports/dataset_v1.json"
    )

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