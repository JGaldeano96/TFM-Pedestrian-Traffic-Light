"""
label_studio_parser.py

Utilidades para leer una exportación JSON de Label Studio y convertirla
en un DataFrame de pandas.

Cada fila del DataFrame representa un único semáforo anotado.
"""

from pathlib import Path
import json
import warnings

import pandas as pd


def load_dataset(json_path: str | Path) -> pd.DataFrame:
    """
    Lee un JSON exportado por Label Studio y devuelve un DataFrame.

    Parameters
    ----------
    json_path : str | Path
        Ruta al fichero JSON exportado.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por objeto anotado.
    """

    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    rows = []

    for task in tasks:
        rows.extend(_parse_task(task))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Tipado eficiente
    df["class_id"] = df["class_id"].astype("int8")

    df["original_width"] = df["original_width"].astype("int16")
    df["original_height"] = df["original_height"].astype("int16")

    float_columns = [
        "x_center",
        "y_center",
        "width",
        "height",
        "area",
    ]

    for column in float_columns:
        df[column] = df[column].astype("float32")

    df["video_id"] = pd.Categorical(df["video_id"])

    df["state"] = pd.Categorical(
        df["state"],
        categories=["Red", "Green", "Off"],
    )

    # Ordenar para tener siempre el mismo orden
    df.sort_values(
        by=["video_id", "filename", "object_id"],
        inplace=True,
    )

    df.reset_index(drop=True, inplace=True)

    return df


def _parse_task(task: dict) -> list[dict]:
    """
    Convierte una tarea de Label Studio en una lista de objetos.
    """

    image_path = task["data"]["image"]

    # ----------------------------------------------------------
    # Extraer vídeo y nombre del fichero
    # ----------------------------------------------------------

    relative_path = image_path.split("?d=")[-1]

    full_path = Path("/home/jgaldeano/tfm") / relative_path

    video_id = full_path.parent.name
    filename = full_path.name

    if not full_path.exists():
        warnings.warn(f"No existe la imagen: {full_path}")

    # ----------------------------------------------------------
    # Imagen sin anotaciones
    # ----------------------------------------------------------

    if not task["annotations"]:
        return []

    annotation = task["annotations"][0]

    if not annotation["result"]:
        return []

    rectangles = {}
    states = {}

    # ----------------------------------------------------------
    # Primera pasada:
    # separar bounding boxes y estados
    # ----------------------------------------------------------

    for result in annotation["result"]:

        object_id = result["id"]

        if result["type"] == "rectanglelabels":

            rectangles[object_id] = result

        elif result["type"] == "choices":

            states[object_id] = result["value"]["choices"][0]

    rows = []

    # ----------------------------------------------------------
    # Segunda pasada:
    # unir bounding box + estado
    # ----------------------------------------------------------

    for object_id, rectangle in rectangles.items():

        if object_id not in states:

            warnings.warn(
                f"El objeto '{object_id}' no tiene estado asociado."
            )

            continue

        value = rectangle["value"]

        # --------------------------------------
        # Conversión Label Studio -> YOLO
        # --------------------------------------

        x = value["x"] / 100
        y = value["y"] / 100

        w = value["width"] / 100
        h = value["height"] / 100

        x_center = x + w / 2
        y_center = y + h / 2

        rows.append(
            {
                "class_id": 0,
                "class_name": "pedestrian_traffic_light",

                "object_id": object_id,

                "image_path": full_path,

                "video_id": video_id,
                "filename": filename,

                "original_width": rectangle["original_width"],
                "original_height": rectangle["original_height"],

                "x_center": x_center,
                "y_center": y_center,

                "width": w,
                "height": h,

                "area": w * h,

                "state": states[object_id],
            }
        )

    # ----------------------------------------------------------
    # Estados sin bounding box
    # ----------------------------------------------------------

    orphan_states = set(states.keys()) - set(rectangles.keys())

    for object_id in orphan_states:

        warnings.warn(
            f"El estado '{object_id}' no tiene bounding box."
        )

    return rows


if __name__ == "__main__":

    JSON_PATH = (
        "/home/jgaldeano/tfm/data/annotation_exports/dataset_v1.json"
    )

    df = load_dataset(JSON_PATH)

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(df.head())

    print()

    print(df.info())

    print()

    print("=" * 70)
    print("CLASES")
    print("=" * 70)

    print(df["state"].value_counts())

    print()

    print("=" * 70)
    print("OBJETOS POR VÍDEO")
    print("=" * 70)

    print(df.groupby("video_id").size())

    print()

    print("=" * 70)
    print("ESTADÍSTICAS")
    print("=" * 70)

    print(df.describe(include="all"))