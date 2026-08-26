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



from __future__ import annotations

import csv
from pathlib import Path


import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)


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
    
    from ultralytics import YOLO

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
    
    
    
    
    
    
    
    
    
    
    
# ============================================================
# CNN CLASSIFIER TEST
# ============================================================


def _load_classifier_test_dataset(
    dataset_dir: Path,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Carga un dataset independiente de test para el clasificador CNN.

    Estructura esperada:

        dataset_dir/
        ├── Green/
        └── Red/

    image_dataset_from_directory() asigna automáticamente:

        0 -> Green
        1 -> Red

    No se modifican las etiquetas dentro del Dataset.

    La conversión a nuestra convención:

        0 -> Red
        1 -> Green

    se realiza posteriormente al obtener las predicciones.
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataset


# ============================================================
# OBTENER PREDICCIONES
# ============================================================

def _get_classifier_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Obtiene las etiquetas reales y las probabilidades producidas
    por el modelo.

    El dataset de Keras utiliza:

        0 -> Green
        1 -> Red

    Nuestra convención de evaluación utiliza:

        0 -> Red
        1 -> Green

    Por tanto:

        y_true = 1 - original_label

    La salida sigmoid del modelo representa:

        P(Green)
    """

    y_true = []

    y_scores = []

    # --------------------------------------------------------
    # Recorrer dataset
    # --------------------------------------------------------

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        # ----------------------------------------------------
        # Invertir etiquetas
        #
        # Keras:
        #   0 -> Green
        #   1 -> Red
        #
        # Evaluación:
        #   0 -> Red
        #   1 -> Green
        # ----------------------------------------------------

        labels = 1 - labels

        y_true.extend(
            labels.numpy()
        )

        y_scores.extend(
            predictions.ravel()
        )

    # --------------------------------------------------------
    # Convertir a NumPy
    # --------------------------------------------------------

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_scores = np.asarray(
        y_scores,
        dtype=float,
    )

    # --------------------------------------------------------
    # Rutas
    # --------------------------------------------------------

    file_paths = list(
        dataset.file_paths
    )

    return (
        y_true,
        y_scores,
        file_paths,
    )


# ============================================================
# MÉTRICAS DE CLASIFICACIÓN
# ============================================================

def _calculate_classifier_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Calcula las principales métricas de clasificación binaria.

    Convención:

        0 = Red
        1 = Green

    y_scores:

        P(Green)

    Regla:

        P(Green) >= threshold -> Green
        P(Green) < threshold  -> Red
    """

    y_pred = (
        y_scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    false_green_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {

        "threshold": threshold,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "specificity": specificity,

        "false_positive_rate": (
            false_positive_rate
        ),

        "false_negative_rate": (
            false_negative_rate
        ),

        "false_green_rate": (
            false_green_rate
        ),

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp,
    }


# ============================================================
# GUARDAR PREDICCIONES INDIVIDUALES
# ============================================================

def _save_classifier_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    file_paths: list[str],
    output_path: Path,
    threshold: float,
) -> None:
    """
    Guarda las predicciones individuales realizadas sobre
    el conjunto de test.

    Convención:

        0 = Red
        1 = Green

    probability_green:

        P(Green)

    error_type:

        FP     -> Red real / Green predicho
        FN     -> Green real / Red predicho
        correct
    """

    y_pred = (
        y_scores >= threshold
    ).astype(int)

    rows = []

    for i in range(
        len(y_true)
    ):

        true_label = int(
            y_true[i]
        )

        predicted_label = int(
            y_pred[i]
        )

        probability_green = float(
            y_scores[i]
        )

        if (
            true_label == 0
            and predicted_label == 1
        ):

            error_type = "FP"

        elif (
            true_label == 1
            and predicted_label == 0
        ):

            error_type = "FN"

        else:

            error_type = "correct"

        true_class = (
            "Red"
            if true_label == 0
            else "Green"
        )

        predicted_class = (
            "Red"
            if predicted_label == 0
            else "Green"
        )

        rows.append({

            "filename": Path(
                file_paths[i]
            ).name,

            "filepath": file_paths[i],

            "true_label": true_label,

            "true_class": true_class,

            "predicted_label": predicted_label,

            "predicted_class": predicted_class,

            "probability_green": (
                probability_green
            ),

            "threshold": threshold,

            "error_type": error_type,
        })

    predictions_df = pd.DataFrame(
        rows
    )

    predictions_df.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# ROC / AUC / YOUDEN
# ============================================================

def _calculate_roc_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict:
    """
    Calcula ROC-AUC y el threshold obtenido mediante
    el índice de Youden.

    Convención:

        0 = Red
        1 = Green

    y_scores representa:

        P(Green)

    Youden:

        J = TPR - FPR

    IMPORTANTE:
    El threshold de Youden se calcula únicamente como
    información descriptiva.

    NO se utiliza para realizar las predicciones finales.
    """

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_scores,
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = auc(
        fpr,
        tpr,
    )

    # --------------------------------------------------------
    # Eliminar thresholds infinitos
    # --------------------------------------------------------

    valid = np.isfinite(
        thresholds
    )

    valid_fpr = fpr[
        valid
    ]

    valid_tpr = tpr[
        valid
    ]

    valid_thresholds = thresholds[
        valid
    ]

    # --------------------------------------------------------
    # Youden
    #
    # J = TPR - FPR
    # --------------------------------------------------------

    youden_scores = (
        valid_tpr
        - valid_fpr
    )

    youden_index = np.argmax(
        youden_scores
    )

    youden_threshold = float(
        valid_thresholds[
            youden_index
        ]
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    return {
        "roc_auc": float(
            roc_auc
        ),

        "youden_threshold": (
            youden_threshold
        ),
    }



# ============================================================
# MÉTRICAS POR THRESHOLD
# ============================================================

def _generate_classifier_threshold_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Calcula las métricas principales para thresholds
    comprendidos entre 0.00 y 1.00 con pasos de 0.01.

    Convención:

        0 = Red
        1 = Green

    La salida del modelo representa:

        P(Green)

    Este análisis es independiente del threshold operativo
    definido manualmente para cada modelo.
    """

    thresholds = np.round(
        np.arange(
            0.00,
            1.01,
            0.01,
        ),
        2,
    )

    results = []

    for threshold in thresholds:

        metrics = _calculate_classifier_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=float(threshold),
        )

        results.append(
            metrics
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# GENERAR MÉTRICAS DEL TEST
# ============================================================

def generate_classifier_test_metrics(
    results_root: Path,
    test_dataset_root: Path,
    output_dir: Path,
    thresholds: dict[str, float],
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
) -> None:
    """
    Evalúa los clasificadores CNN entrenados con dataset_v1
    y dataset_v2 sobre conjuntos independientes de test
    diurno y nocturno.

    Thresholds utilizados para la evaluación principal:

        dataset_v1 -> threshold definido manualmente
        dataset_v2 -> threshold definido manualmente

    ROC-AUC y Youden se calculan únicamente como métricas
    descriptivas del conjunto de test.

    El threshold de Youden NO sustituye al threshold
    establecido para cada modelo.
    """

    print("=" * 70)

    print(
        "GENERATING CNN CLASSIFIER TEST METRICS"
    )

    print("=" * 70)

    print(
        "\nClass mapping:"
        "\n  0 -> Red"
        "\n  1 -> Green"
    )

    print(
        "\nModel output:"
        "\n  sigmoid -> P(Green)"
    )

    print(
        "\nEvaluation thresholds:"
    )

    for dataset_version, threshold in thresholds.items():

        print(
            f"  {dataset_version} -> "
            f"{threshold:.4f}"
        )

    print(
        "\nDecision rule:"
        "\n  probability >= threshold -> Green"
        "\n  probability < threshold  -> Red"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # MODELOS
    # ========================================================

    models = {

        "dataset_v1": (
            results_root
            / "dataset_v1"
            / "classifier_v1"
            / "best.keras"
        ),

        "dataset_v2": (
            results_root
            / "dataset_v2"
            / "classifier_v2"
            / "best.keras"
        ),
    }

    # ========================================================
    # VALIDAR THRESHOLDS
    # ========================================================

    for dataset_version in models:

        if dataset_version not in thresholds:

            raise ValueError(
                f"No se ha definido un threshold para "
                f"{dataset_version}"
            )

        threshold = thresholds[
            dataset_version
        ]

        if not 0 <= threshold <= 1:

            raise ValueError(
                f"El threshold de {dataset_version} "
                f"debe estar entre 0 y 1. "
                f"Valor recibido: {threshold}"
            )

    # ========================================================
    # TEST DATASETS
    # ========================================================

    test_types = {

        "diurn": (
            test_dataset_root
            / "diurn"
        ),

        "nocturn": (
            test_dataset_root
            / "nocturn"
        ),
    }

    # ========================================================
    # CHECK MODELS
    # ========================================================

    print(
        "\nChecking models..."
    )

    for dataset_version, model_path in models.items():

        if not model_path.exists():

            raise FileNotFoundError(
                f"No existe el modelo:\n"
                f"{model_path}"
            )

        print(
            f"{dataset_version}: "
            f"{model_path}"
        )

    # ========================================================
    # CHECK DATASETS
    # ========================================================

    print(
        "\nChecking test datasets..."
    )

    for test_type, dataset_dir in test_types.items():

        if not dataset_dir.exists():

            raise FileNotFoundError(
                f"No existe el dataset de test:\n"
                f"{dataset_dir}"
            )

        print(
            f"{test_type}: "
            f"{dataset_dir}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    metrics_results = []

    threshold_results = []

    # ========================================================
    # EVALUATION
    # ========================================================

    for dataset_version, model_path in models.items():

        print(
            "\n" + "-" * 70
        )

        print(
            f"Loading model: "
            f"{dataset_version}"
        )

        model = tf.keras.models.load_model(
            model_path
        )

        # ----------------------------------------------------
        # Threshold manual del modelo
        # ----------------------------------------------------

        test_threshold = thresholds[
            dataset_version
        ]

        print(
            f"Evaluation threshold: "
            f"{test_threshold:.4f}"
        )

        # ====================================================
        # TEST DIURNO / NOCTURNO
        # ====================================================

        for test_type, dataset_dir in test_types.items():

            print(
                "\n" + "-" * 70
            )

            print(
                f"Testing:"
                f"\n  Model      : {dataset_version}"
                f"\n  Test type  : {test_type}"
                f"\n  Dataset    : {dataset_dir}"
                f"\n  Threshold  : {test_threshold:.4f}"
            )

            # =================================================
            # DATASET
            # =================================================

            dataset = _load_classifier_test_dataset(
                dataset_dir=dataset_dir,
                image_size=image_size,
                batch_size=batch_size,
            )

            print(
                f"Found "
                f"{len(dataset.file_paths)} "
                f"files belonging to "
                f"{len(dataset.class_names)} "
                f"classes."
            )

            print(
                f"  Classes    : "
                f"{dataset.class_names}"
            )

            # =================================================
            # PREDICTIONS
            # =================================================

            (
                y_true,
                y_scores,
                file_paths,
            ) = _get_classifier_predictions(
                model=model,
                dataset=dataset,
            )

            # =================================================
            # BASIC SANITY CHECK
            # =================================================

            print(
                f"\n  Real Red   : "
                f"{np.sum(y_true == 0)}"
            )

            print(
                f"  Real Green : "
                f"{np.sum(y_true == 1)}"
            )

            print(
                f"  Mean P(Green): "
                f"{np.mean(y_scores):.4f}"
            )

            print(
                f"  Min P(Green): "
                f"{np.min(y_scores):.4f}"
            )

            print(
                f"  Max P(Green): "
                f"{np.max(y_scores):.4f}"
            )

            # =================================================
            # ROC / AUC / YOUDEN
            # =================================================

            roc_metrics = _calculate_roc_metrics(
                y_true=y_true,
                y_scores=y_scores,
            )

            youden_threshold = (
                roc_metrics[
                    "youden_threshold"
                ]
            )

            # =================================================
            # MÉTRICAS PRINCIPALES
            #
            # IMPORTANTE:
            #
            # Aquí se utiliza el threshold MANUAL definido
            # para cada modelo.
            # =================================================

            metrics = _calculate_classifier_metrics(
                y_true=y_true,
                y_scores=y_scores,
                threshold=test_threshold,
            )

            # -------------------------------------------------
            # ROC-AUC + YOUDEN
            # -------------------------------------------------

            metrics.update({

                "roc_auc": (
                    roc_metrics[
                        "roc_auc"
                    ]
                ),

                "youden_threshold": (
                    youden_threshold
                ),
            })

            # -------------------------------------------------
            # Metadata
            # -------------------------------------------------

            metrics.update({

                "dataset": dataset_version,

                "test_type": test_type,

                "model_path": str(
                    model_path
                ),

                "num_images": len(
                    y_true
                ),
            })

            metrics_results.append(
                metrics
            )

            # =================================================
            # THRESHOLD ANALYSIS
            #
            # Mantiene todos los thresholds entre 0.00 y 1.00
            # para poder estudiar el comportamiento del modelo.
            #
            # NO modifica el threshold utilizado por el modelo.
            # =================================================

            threshold_df = (
                _generate_classifier_threshold_metrics(
                    y_true=y_true,
                    y_scores=y_scores,
                )
            )

            threshold_df.insert(
                0,
                "dataset",
                dataset_version,
            )

            threshold_df.insert(
                1,
                "test_type",
                test_type,
            )

            threshold_results.append(
                threshold_df
            )

            # =================================================
            # INDIVIDUAL PREDICTIONS
            #
            # Utiliza el threshold MANUAL del modelo.
            # =================================================

            predictions_path = (
                output_dir
                / (
                    f"predictions_"
                    f"{dataset_version}_"
                    f"{test_type}.csv"
                )
            )

            _save_classifier_predictions(
                y_true=y_true,
                y_scores=y_scores,
                file_paths=file_paths,
                output_path=predictions_path,
                threshold=test_threshold,
            )

            # =================================================
            # PRINT RESULTS
            # =================================================

            print(
                "\n  Evaluation:"
            )

            print(
                f"  Threshold  : "
                f"{test_threshold:.4f}"
            )

            print(
                f"  Youden thr : "
                f"{youden_threshold:.4f}"
            )

            print(
                f"\n  Accuracy   : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"  Precision  : "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"  Recall     : "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"  F1         : "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"  Specificity: "
                f"{metrics['specificity']:.4f}"
            )

            print(
                f"  ROC-AUC    : "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"  TN         : "
                f"{metrics['TN']}"
            )

            print(
                f"  FP         : "
                f"{metrics['FP']}"
            )

            print(
                f"  FN         : "
                f"{metrics['FN']}"
            )

            print(
                f"  TP         : "
                f"{metrics['TP']}"
            )

    # ========================================================
    # GUARDAR MÉTRICAS PRINCIPALES
    # ========================================================

    metrics_df = pd.DataFrame(
        metrics_results
    )

    metrics_path = (
        output_dir
        / "cnn_test_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # ========================================================
    # GUARDAR THRESHOLD METRICS
    # ========================================================

    threshold_df = pd.concat(
        threshold_results,
        ignore_index=True,
    )

    threshold_path = (
        output_dir
        / "cnn_test_threshold_metrics.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CNN TEST METRICS GENERATED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMetrics:"
        f"\n{metrics_path}"
    )

    print(
        f"\nThreshold metrics:"
        f"\n{threshold_path}"
    )

    print(
        f"\nPredictions:"
        f"\n{output_dir}"
    )

    print(
        "\nEvaluation thresholds:"
    )

    for dataset_version, threshold in thresholds.items():

        print(
            f"  {dataset_version}: "
            f"{threshold:.4f}"
        )

    print(
        "=" * 70
    )

    print(
        "\nTest evaluation completed successfully."
    )