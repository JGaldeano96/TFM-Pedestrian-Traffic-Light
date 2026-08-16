from __future__ import annotations

import csv
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# EVALUATE YOLO MODEL
# ============================================================

def evaluate_yolo_model(
    model_path: Path,
    test_dataset_dir: Path,
    imgsz: int,
) -> dict:
    """
    Evalúa un modelo YOLO sobre un dataset de test independiente.

    Parameters
    ----------
    model_path : Path
        Ruta al fichero best.pt.

    test_dataset_dir : Path
        Directorio del dataset de test.
        Debe contener:
            images/
            labels/
            dataset.yaml

    imgsz : int
        Tamaño de imagen utilizado durante la evaluación.

    Returns
    -------
    dict
        Métricas obtenidas durante la evaluación.
    """

    model = YOLO(
        str(model_path)
    )

    results = model.val(
        data=str(
            test_dataset_dir / "dataset.yaml"
        ),
        imgsz=imgsz,
        split="test",
        verbose=False,
    )

    metrics = {
        "precision": float(
            results.box.mp
        ),

        "recall": float(
            results.box.mr
        ),

        "mAP50": float(
            results.box.map50
        ),

        "mAP50-95": float(
            results.box.map
        ),
    }

    return metrics


# ============================================================
# GENERATE YOLO TEST METRICS
# ============================================================

def generate_yolo_test_metrics(
    results_root: Path,
    test_dataset_root: Path,
    output_csv: Path,
) -> None:
    """
    Recorre todos los datasets y modelos YOLO disponibles y
    genera un CSV con las métricas obtenidas sobre los datasets
    independientes de test.

    Estructura esperada:

        results/
        └── yolo/
            ├── dataset_v1/
            │   ├── 640/
            │   │   ├── yolo26n/
            │   │   │   └── weights/
            │   │   │       └── best.pt
            │   │   └── yolo26s/
            │   │       └── weights/
            │   │           └── best.pt
            │   ├── 800/
            │   ├── 960/
            │   └── 1088/
            │
            └── dataset_v2/
                └── ...

    Datasets independientes:

        data/
        └── datasets/
            └── yolo_test/
                ├── diurn/
                └── nocturn/

    Parameters
    ----------
    results_root : Path
        Directorio que contiene dataset_v1, dataset_v2, etc.

    test_dataset_root : Path
        Directorio que contiene los datasets diurn y nocturn.

    output_csv : Path
        Ruta donde se guardará el CSV.
    """

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 70)
    print("YOLO INDEPENDENT TEST EVALUATION")
    print("=" * 70)

    # ========================================================
    # Buscar versiones de dataset
    # ========================================================

    dataset_dirs = sorted(
        path
        for path in results_root.glob("dataset_*")
        if path.is_dir()
    )

    print(
        f"\nDatasets found: {len(dataset_dirs)}"
    )

    if not dataset_dirs:

        print(
            "\nWARNING: No se han encontrado datasets."
        )

        return

    # ========================================================
    # Recorrer datasets
    # ========================================================

    for dataset_dir in dataset_dirs:

        dataset_version = dataset_dir.name

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"DATASET: {dataset_version}"
        )

        print(
            f"{'=' * 70}"
        )

        # ----------------------------------------------------
        # Buscar tamaños de imagen
        # ----------------------------------------------------

        image_size_dirs = sorted(
            (
                path
                for path in dataset_dir.iterdir()
                if (
                    path.is_dir()
                    and path.name.isdigit()
                )
            ),
            key=lambda path: int(path.name),
        )

        # ----------------------------------------------------
        # Recorrer tamaños
        # ----------------------------------------------------

        for image_size_dir in image_size_dirs:

            imgsz = int(
                image_size_dir.name
            )

            print(
                f"\nImage size: {imgsz}"
            )

            # ------------------------------------------------
            # Buscar modelos
            # ------------------------------------------------

            model_dirs = sorted(
                path
                for path in image_size_dir.iterdir()
                if path.is_dir()
            )

            for model_dir in model_dirs:

                model_name = model_dir.name

                model_path = (
                    model_dir
                    / "weights"
                    / "best.pt"
                )

                if not model_path.exists():

                    print(
                        "\nWARNING: "
                        f"No existe {model_path}"
                    )

                    continue

                print(
                    "\n" + "-" * 60
                )

                print(
                    f"Model: {model_name}"
                )

                print(
                    f"Path : {model_path}"
                )

                # ==========================================
                # Evaluar test diurno y nocturno
                # ==========================================

                for test_type in (
                    "diurn",
                    "nocturn",
                ):

                    test_dir = (
                        test_dataset_root
                        / test_type
                    )

                    dataset_yaml = (
                        test_dir
                        / "dataset.yaml"
                    )

                    if not dataset_yaml.exists():

                        print(
                            f"\nWARNING: "
                            f"No existe {dataset_yaml}"
                        )

                        continue

                    print(
                        f"\nTest dataset: {test_type}"
                    )

                    try:

                        metrics = evaluate_yolo_model(
                            model_path=model_path,
                            test_dataset_dir=test_dir,
                            imgsz=imgsz,
                        )

                    except Exception as exc:

                        print(
                            "\nERROR evaluando "
                            f"{model_name} "
                            f"en {test_type}:"
                        )

                        print(
                            exc
                        )

                        continue

                    # ======================================
                    # Guardar resultados
                    # ======================================

                    row = {

                        "dataset": dataset_version,

                        "image_size": imgsz,

                        "model": model_name,

                        "test_type": test_type,

                        "model_path": str(
                            model_path
                        ),

                        "precision": metrics[
                            "precision"
                        ],

                        "recall": metrics[
                            "recall"
                        ],

                        "mAP50": metrics[
                            "mAP50"
                        ],

                        "mAP50-95": metrics[
                            "mAP50-95"
                        ],
                    }

                    rows.append(
                        row
                    )

                    # ======================================
                    # Mostrar métricas
                    # ======================================

                    print(
                        f"  Precision : "
                        f"{metrics['precision']:.4f}"
                    )

                    print(
                        f"  Recall    : "
                        f"{metrics['recall']:.4f}"
                    )

                    print(
                        f"  mAP50     : "
                        f"{metrics['mAP50']:.4f}"
                    )

                    print(
                        f"  mAP50-95  : "
                        f"{metrics['mAP50-95']:.4f}"
                    )

    # ========================================================
    # Comprobar resultados
    # ========================================================

    if not rows:

        print(
            "\nNo se han obtenido métricas."
        )

        return

    # ========================================================
    # Guardar CSV
    # ========================================================

    fieldnames = [

        "dataset",

        "image_size",

        "model",

        "test_type",

        "model_path",

        "precision",

        "recall",

        "mAP50",

        "mAP50-95",
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Resumen
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST METRICS GENERATED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nEvaluations: {len(rows)}"
    )

    print(
        "\nCSV:"
    )

    print(
        output_csv
    )