# ============================================================
# CNN - PREDICCIONES SOBRE VALIDACIÓN
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_VERSION = "v2"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
THRESHOLD = 0.5


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "results"
    / "classifier"
    / f"dataset_{DATASET_VERSION}"
    / f"classifier_{DATASET_VERSION}"
    / "best.keras"
)

VAL_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "classifier"
    / "val"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "results"
    / "classifier"
    / f"dataset_{DATASET_VERSION}"
    / f"classifier_{DATASET_VERSION}"
    / f"predictions_dataset_{DATASET_VERSION}.csv"
)


# ============================================================
# CARGAR MODELO
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# CARGAR VALIDACIÓN
# ============================================================
#
# Keras crea:
#
#     0 -> Green
#     1 -> Red
#
# Nosotros queremos:
#
#     0 -> Red
#     1 -> Green
#
# Por eso invertimos las etiquetas.
# ============================================================

dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# PREDICCIONES
# ============================================================

rows = []

for images, labels in dataset:

    probabilities = model.predict(
        images,
        verbose=0,
    ).ravel()

    labels = 1 - labels.numpy()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    for i in range(len(labels)):

        true_label = int(labels[i])
        predicted_label = int(predictions[i])
        probability = float(probabilities[i])

        if true_label == 0 and predicted_label == 1:
            error_type = "FP"

        elif true_label == 1 and predicted_label == 0:
            error_type = "FN"

        else:
            error_type = "correct"

        rows.append({
            "filename": Path(
                dataset.file_paths[len(rows)]
            ).name,

            "filepath": dataset.file_paths[
                len(rows)
            ],

            "true_label": true_label,

            "true_class": (
                "Red"
                if true_label == 0
                else "Green"
            ),

            "predicted_label": predicted_label,

            "predicted_class": (
                "Red"
                if predicted_label == 0
                else "Green"
            ),

            "probability_green": probability,

            "threshold": THRESHOLD,

            "error_type": error_type,
        })


# ============================================================
# GUARDAR CSV
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print(
    f"\nPredicciones guardadas en:\n{OUTPUT_PATH}"
)

print(
    f"\nImágenes evaluadas: {len(df)}"
)

print(
    f"\nModelo utilizado:\n{MODEL_PATH}"
)