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

from pathlib import Path

import numpy as np
import pandas as pd

from label_studio_parser import load_dataset


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    iterations: int = 10000,
    random_state: int = 42,
) -> dict[str, list[str]]:

    if not 0 < train_size < 1:
        raise ValueError(
            "train_size debe estar comprendido entre 0 y 1."
        )

    rng = np.random.default_rng(random_state)

    video_stats = _compute_video_statistics(df)

    videos = video_stats["video_id"].tolist()

    lookup = (
        video_stats
        .set_index("video_id")
        .to_dict("index")
    )

    total_videos = len(videos)

    n_train = round(total_videos * train_size)

    total_objects = len(df)

    target_train_objects = total_objects * train_size

    global_distribution = (
        df["state"]
        .value_counts(normalize=True)
        .reindex(
            ["Red", "Green"],
            fill_value=0,
        )
    )

    best_cost = np.inf
    best_split = None

    for _ in range(iterations):

        train_videos = rng.choice(
            videos,
            size=n_train,
            replace=False,
        )

        train_set = set(train_videos)

        test_videos = [
            video
            for video in videos
            if video not in train_set
        ]

        cost = _compute_cost(
            train_videos=train_videos,
            test_videos=test_videos,
            lookup=lookup,
            target_train_objects=target_train_objects,
            total_objects=total_objects,
            global_distribution=global_distribution,
        )

        if cost < best_cost:

            best_cost = cost

            best_split = {
                "train": sorted(train_videos),
                "val": sorted(test_videos),
            }

    _validate_split(
        best_split,
        total_videos,
    )

    return best_split


def _compute_video_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for video_id, group in df.groupby(
        "video_id",
        observed=True,
    ):

        rows.append(
            {
                "video_id": video_id,

                "images":
                    group["filename"].nunique(),

                "objects":
                    len(group),

                "Red":
                    (group["state"] == "Red").sum(),

                "Green":
                    (group["state"] == "Green").sum(),
            }
        )

    return pd.DataFrame(rows)


def _compute_cost(
    train_videos,
    test_videos,
    lookup,
    target_train_objects,
    total_objects,
    global_distribution,
):

    def subset_statistics(videos):

        objects = sum(
            lookup[v]["objects"]
            for v in videos
        )

        red = sum(
            lookup[v]["Red"]
            for v in videos
        )

        green = sum(
            lookup[v]["Green"]
            for v in videos
        )

        if objects == 0:

            red_ratio = 0
            green_ratio = 0

        else:

            red_ratio = red / objects
            green_ratio = green / objects

        return (
            objects,
            red_ratio,
            green_ratio,
        )

    (
        train_objects,
        train_red,
        train_green,
    ) = subset_statistics(train_videos)

    (
        test_objects,
        test_red,
        test_green,
    ) = subset_statistics(test_videos)

    object_error = (
        abs(
            train_objects
            - target_train_objects
        )
        / target_train_objects
    )

    class_error = (
        abs(train_red - global_distribution["Red"])
        + abs(test_red - global_distribution["Red"])
        + abs(train_green - global_distribution["Green"])
        + abs(test_green - global_distribution["Green"])
    )

    return (
        2.0 * object_error
        + class_error
    )
    
def summarize_split(
    df: pd.DataFrame,
    split: dict[str, list[str]],
) -> None:

    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    total_videos = df["video_id"].nunique()
    total_images = df["filename"].nunique()
    total_objects = len(df)

    global_distribution = (
        df["state"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(
            ["Red", "Green", "Off"],
            fill_value=0,
        )
    )

    print("\nGLOBAL")
    print("-" * 70)
    print(f"Videos : {total_videos}")
    print(f"Images : {total_images}")
    print(f"Objects: {total_objects}")

    print("\nClasses")

    for state, pct in global_distribution.items():
        print(f"  {state:<5}: {pct:6.2f}%")

    for subset in ("train", "val"):

        subset_df = df[
            df["video_id"].isin(split[subset])
        ]

        print("\n" + "=" * 70)
        print(subset.upper())
        print("=" * 70)

        print(f"Videos : {len(split[subset])}")
        print(f"Images : {subset_df['filename'].nunique()}")
        print(f"Objects: {len(subset_df)}")

        print(
            f"Object ratio: "
            f"{100 * len(subset_df) / total_objects:.2f}%"
        )

        distribution = (
            subset_df["state"]
            .value_counts(normalize=True)
            .mul(100)
            .reindex(
                ["Red", "Green", "Off"],
                fill_value=0,
            )
        )

        print("\nClasses")

        for state, pct in distribution.items():
            print(f"  {state:<5}: {pct:6.2f}%")


def _validate_split(
    split: dict[str, list[str]],
    total_videos: int,
) -> None:

    train = set(split["train"])
    val = set(split["val"])

    duplicated = train & val

    if duplicated:
        raise ValueError(
            "Hay vídeos repetidos entre TRAIN y VAL."
        )

    assigned = len(train | val)

    if assigned != total_videos:
        raise ValueError(
            f"Se asignaron {assigned} vídeos "
            f"pero existen {total_videos}."
        )


if __name__ == "__main__":

    JSON_PATH = Path(
        "/home/jgaldeano/tfm/data/annotation_exports/dataset_v1.json"
    )

    df = load_dataset(JSON_PATH)

    split = split_dataset(
        df,
        train_size=0.70,
        iterations=10000,
        random_state=42,
    )

    summarize_split(
        df,
        split,
    )

    # ----------------------------------------------------------
    # Comprobar que ningún frame aparece en ambos conjuntos
    # ----------------------------------------------------------

    train_images = set(
        df[df["video_id"].isin(split["train"])]["image_path"]
    )

    val_images = set(
        df[df["video_id"].isin(split["val"])]["image_path"]
    )

    duplicated = train_images & val_images

    if duplicated:
        raise RuntimeError(
            f"Se han encontrado {len(duplicated)} imágenes duplicadas entre TRAIN y VAL."
        )

    print("\n✓ No existe data leakage entre TRAIN y VAL.")