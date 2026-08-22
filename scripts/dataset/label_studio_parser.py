"""
label_studio_parser.py

Lee una exportación JSON de Label Studio y la convierte en un DataFrame.

Este script únicamente ejecuta las funciones definidas en
scripts/utils/dataset_functions.py para facilitar el mantenimiento
del proyecto.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.utils.dataset_functions import load_dataset


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

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(df.head())

    print()

    print(df.info())

    print()

    print("=" * 70)
    print("CLASSES")
    print("=" * 70)

    print(df["state"].value_counts())

    print()

    print("=" * 70)
    print("OBJECTS PER VIDEO")
    print("=" * 70)

    print(df.groupby("video_id").size())

    print()

    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)

    print(df.describe(include="all"))
