# ============================================================
# LIBRERÍAS
# ============================================================

from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tf2onnx
import onnx
import random

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)

# ============================================================
# SEMILLAS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# GPU
# ============================================================

print("GPU:", tf.config.list_physical_devices("GPU"))


# ============================================================
# RUTAS Y CONFIGURACIÓN

# Esto garantiza que el script sea reutilizable en futuras
# versiones del dataset
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_VERSION = "v2"
MODEL_NAME = "classifier_v2"

DATASET_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "classifier"
)

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "classifier"
    / f"dataset_{DATASET_VERSION}"
    / MODEL_NAME
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURACIÓN DEL DATASET
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# CREACIÓN DE LOS DATASETS
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# INFORMACIÓN DEL DATASET
# ============================================================

print(f"Clases: {train_ds.class_names}")
print(f"Número de clases: {len(train_ds.class_names)}")


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        mode="horizontal",
    ),

    tf.keras.layers.RandomRotation(
        factor=0.03,
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05,
    ),

    tf.keras.layers.RandomZoom(
        height_factor=(-0.15, 0.10),
        width_factor=(-0.15, 0.10),
    ),

    tf.keras.layers.RandomContrast(
        factor=0.20,
    ),

])

# ============================================================
# ============================================================
# MODELO CNN
# ============================================================
# ============================================================


# ============================================================
# INPUT
# ============================================================

inputs = tf.keras.Input(
    shape=(*IMAGE_SIZE, 3),
)

x = data_augmentation(inputs)

x = tf.keras.layers.Rescaling(
    1 / 255,
)(x)


# ============================================================
# BLOQUE 1
# ============================================================

x = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_1",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)


x = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_2",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)

x = tf.keras.layers.MaxPooling2D(
    pool_size=2,
)(x)


# ============================================================
# BLOQUE 2
# ============================================================

x = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_3",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)


x = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_4",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)

x = tf.keras.layers.MaxPooling2D(
    pool_size=2,
)(x)


# ============================================================
# BLOQUE 3
# ============================================================

x = tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_5",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)


x = tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_6",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)

x = tf.keras.layers.MaxPooling2D(
    pool_size=2,
)(x)


# ============================================================
# BLOQUE 4
# ============================================================

x = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_7",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)


x = tf.keras.layers.Conv2D(
    filters=256,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name="conv_8",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("silu")(x)


# ============================================================
# CLASIFICADOR
# ============================================================

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30,
)(x)

x = tf.keras.layers.Dense(
    units=256,
    activation="silu",
)(x)

x = tf.keras.layers.Dropout(
    0.20,
)(x)

x = tf.keras.layers.Dense(
    units=64,
    activation="silu",
)(x)

outputs = tf.keras.layers.Dense(
    units=1,
    activation="sigmoid",
)(x)


# ============================================================
# MODELO
# ============================================================

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
)

model.summary()




# ============================================================
# COMPILER
# ============================================================

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=3e-4,
    weight_decay=1e-4,
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=RESULTS_DIR / "best.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

# ============================================================
# FIT
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=75,
    verbose=2,
    callbacks=[
        lr_scheduler,
        early_stopping,
        checkpoint,
    ],
)

# ============================================================
# ============================================================
# RESULTADOS
# ============================================================
# ============================================================


# ============================================================
# GUARDAR RESULTADOS 
# ============================================================

def plot_history(history: tf.keras.callbacks.History) -> None:
    """Plot del Loss y Accuracy por epoca en train y val"""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    axes[0].plot(
        history.history["loss"],
        label="train",
    )
    axes[0].plot(
        history.history["val_loss"],
        label="val",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(
        history.history["accuracy"],
        label="train",
    )
    axes[1].plot(
        history.history["val_accuracy"],
        label="val",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()

    # Guardar figura
    output_path = RESULTS_DIR / "training_history.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    
# ============================================================
# FP Y FN SEGÚN THRESHOLD
# ============================================================

def save_fp_fn_by_threshold(
    model: tf.keras.Model,
    dataset: tf.data.Dataset
):
    """
    Calcula FP y FN para thresholds entre 0.00 y 1.00
    con incrementos de 0.01 y guarda los resultados
    en CSV y PNG.
    """

    # ========================================================
    # Obtener etiquetas reales y probabilidades
    # ========================================================

    y_true = []
    y_score = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(labels.numpy())
        y_score.extend(predictions.ravel())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # ========================================================
    # Thresholds de 0.00 a 1.00
    # ========================================================

    thresholds = np.round(
        np.arange(0.00, 1.01, 0.01),
        2,
    )

    results = []

    # ========================================================
    # Calcular FP y FN
    # ========================================================

    for threshold in thresholds:

        y_pred = (
            y_score >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append({
            "threshold": threshold,
            "FP": fp,
            "FN": fn,
        })

    # ========================================================
    # DataFrame
    # ========================================================

    results_df = pd.DataFrame(results)

    # ========================================================
    # Guardar CSV
    # ========================================================

    csv_path = (
        RESULTS_DIR
        / "fp_fn_by_threshold.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # Crear gráfico
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        results_df["threshold"],
        results_df["FP"],
        label="False Positives (FP)",
    )

    ax.plot(
        results_df["threshold"],
        results_df["FN"],
        label="False Negatives (FN)",
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")

    ax.set_title(
        "False Positives and False Negatives by Threshold"
    )

    ax.set_xticks(
        np.arange(0.00, 1.01, 0.05)
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # ========================================================
    # Guardar gráfico
    # ========================================================

    plot_path = (
        RESULTS_DIR
        / "fp_fn_by_threshold.png"
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Información
    # ========================================================

    print(
        f"FP/FN por threshold guardado en: {csv_path}"
    )

    print(
        f"Gráfico FP/FN guardado en: {plot_path}"
    )
    


# ========================================================
# Sacar imágenes
# ========================================================

plot_history(history)

save_fp_fn_by_threshold(
    model=model,
    dataset=val_ds,
)



# ============================================================
# CURVA ROC Y PUNTO YOUDEN
# ============================================================

def roc_youden_plot(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[float, float]:
    """
    Calcula la curva ROC, el AUC y el threshold óptimo
    mediante el índice de Youden.

    Guarda la curva ROC en RESULTS_DIR.
    """

    # ========================================================
    # Obtener etiquetas reales y probabilidades
    # ========================================================

    y_true = []
    y_scores = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(
            labels.numpy()
        )

        y_scores.extend(
            predictions.ravel()
        )

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # ========================================================
    # Calcular ROC
    # ========================================================

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_scores,
    )

    # ========================================================
    # Calcular AUC
    # ========================================================

    roc_auc = auc(
        fpr,
        tpr,
    )

    # ========================================================
    # Índice de Youden
    # ========================================================

    youden_scores = tpr - fpr

    youden_index = np.argmax(
        youden_scores
    )

    best_threshold = thresholds[
        youden_index
    ]

    best_fpr = fpr[
        youden_index
    ]

    best_tpr = tpr[
        youden_index
    ]

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
    )

    # Punto de Youden
    ax.scatter(
        best_fpr,
        best_tpr,
        s=120,
        zorder=3,
    )

    ax.text(
        best_fpr,
        best_tpr,
        f"Youden\nthr={best_threshold:.3f}",
        fontsize=10,
        verticalalignment="bottom",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    # ========================================================
    # Guardar imagen
    # ========================================================

    output_path = (
        RESULTS_DIR
        / "roc_youden.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Resultados
    # ========================================================

    print()
    print("ROC / Youden:")
    print(f"AUC: {roc_auc:.4f}")
    print(
        f"Best threshold (Youden): "
        f"{best_threshold:.4f}"
    )
    print(
        f"ROC curve guardada en: "
        f"{output_path}"
    )

    return best_threshold, roc_auc


best_threshold, roc_auc = roc_youden_plot(
    model=model,
    dataset=val_ds,
)

# ========================================================
# Errores
# ========================================================


def save_classification_errors(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold = best_threshold,
) -> None:
    """
    Guarda las imágenes clasificadas incorrectamente por el modelo.

    FP:
        Clase real = 0
        Predicción = 1

    FN:
        Clase real = 1
        Predicción = 0
    """

    # ========================================================
    # Crear directorio para los errores
    # ========================================================

    errors_dir = RESULTS_DIR / "errores"

    fp_dir = errors_dir / "FP"
    fn_dir = errors_dir / "FN"

    fp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fn_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Contador de errores
    # ========================================================

    fp_count = 0
    fn_count = 0

    # ========================================================
    # Recorrer validation dataset
    # ========================================================

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0,
        ).ravel()

        y_true = labels.numpy()

        y_pred = (
            predictions >= threshold
        ).astype(int)

        # ====================================================
        # Buscar errores
        # ====================================================

        for i in range(len(images)):

            true_label = int(y_true[i])
            predicted_label = int(y_pred[i])
            probability = float(predictions[i])

            # ------------------------------------------------
            # FP
            # ------------------------------------------------

            if true_label == 0 and predicted_label == 1:

                filename = (
                    f"FP_{fp_count:04d}"
                    f"_real_{true_label}"
                    f"_pred_{predicted_label}"
                    f"_prob_{probability:.3f}.png"
                )

                output_path = fp_dir / filename

                image = images[i].numpy()

                # Como el dataset contiene imágenes RGB
                # con valores 0-255, las convertimos a uint8.
                image = np.clip(
                    image,
                    0,
                    255,
                ).astype(np.uint8)

                plt.imsave(
                    output_path,
                    image,
                )

                fp_count += 1

            # ------------------------------------------------
            # FN
            # ------------------------------------------------

            elif true_label == 1 and predicted_label == 0:

                filename = (
                    f"FN_{fn_count:04d}"
                    f"_real_{true_label}"
                    f"_pred_{predicted_label}"
                    f"_prob_{probability:.3f}.png"
                )

                output_path = fn_dir / filename

                image = images[i].numpy()

                image = np.clip(
                    image,
                    0,
                    255,
                ).astype(np.uint8)

                plt.imsave(
                    output_path,
                    image,
                )

                fn_count += 1

    # ========================================================
    # Resumen
    # ========================================================

    print()
    print("Errores de clasificación:")
    print(f"  False Positives (FP): {fp_count}")
    print(f"  False Negatives (FN): {fn_count}")
    print(f"  Threshold: {threshold}")
    print(f"  Directorio: {errors_dir}")
    

save_classification_errors(
    model=model,
    dataset=val_ds,
)

# ============================================================
# MATRIZ DE CONFUSIÓN CON THRESHOLD DE YOUDEN
# ============================================================

def save_confusion_matrix(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold: float,
) -> None:
    """
    Calcula y guarda la matriz de confusión utilizando
    el threshold óptimo obtenido mediante Youden.
    """

    y_true = []
    y_scores = []

    # --------------------------------------------------------
    # Obtener etiquetas y probabilidades
    # --------------------------------------------------------

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(
            labels.numpy()
        )

        y_scores.extend(
            predictions.ravel()
        )

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # --------------------------------------------------------
    # Aplicar threshold
    # --------------------------------------------------------

    y_pred = (
        y_scores >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    # --------------------------------------------------------
    # Nombres de las clases
    # --------------------------------------------------------

    class_names = dataset.class_names

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        f"Confusion Matrix "
        f"(Threshold = {threshold:.3f})"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    output_path = (
        RESULTS_DIR
        / "confusion_matrix_youden.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Información
    # --------------------------------------------------------

    print(
        f"Matriz de confusión guardada en: "
        f"{output_path}"
    )

    print(
        f"Threshold utilizado: {threshold:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)
    
save_confusion_matrix(
    model=model,
    dataset=val_ds,
    threshold=best_threshold,
)



# ============================================================
# EXPORTAR MODELO A ONNX
# ============================================================

def export_model_to_onnx(
    model: tf.keras.Model,
) -> None:
    """
    Exporta el modelo Keras a formato ONNX.
    """

    import tf2onnx

    output_path = (
        RESULTS_DIR
        / "classifier.onnx"
    )

    input_signature = [
        tf.TensorSpec(
            shape=(None, *IMAGE_SIZE, 3),
            dtype=tf.float32,
            name="input",
        )
    ]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=18,
        output_path=str(output_path),
    )

    print(
        f"Modelo ONNX guardado en: "
        f"{output_path}"
    )
    
export_model_to_onnx(model)