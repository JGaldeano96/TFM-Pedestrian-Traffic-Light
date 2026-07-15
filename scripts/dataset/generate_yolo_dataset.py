"""
generate_yolo_dataset.py

Genera automáticamente un dataset compatible con Ultralytics YOLO a
partir de una exportación de Label Studio.

El script:

1. Lee el JSON de Label Studio.
2. Construye el DataFrame mediante label_studio_parser.py.
3. Carga un split existente o genera uno nuevo.
4. Guarda el split en data/splits/.
5. Limpia el dataset anterior.
6. Copia las imágenes a train/ y val/.
7. Genera los ficheros TXT de YOLO.
8. Crea dataset.yaml.
9. Muestra un resumen final.

El dataset generado es completamente reproducible gracias al fichero
split_vX.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from label_studio_parser import load_dataset
from split_dataset import split_dataset


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path("/home/jgaldeano/tfm")

JSON_PATH = (
    ROOT_DIR
    / "data"
    / "annotation_exports"
    / "dataset_v1.json"
)

SPLIT_NAME = "split_v1.json"

SPLIT_PATH = (
    ROOT_DIR
    / "data"
    / "splits"
    / SPLIT_NAME
)

YOLO_DATASET_DIR = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "yolo"
)

IMAGES_DIR = YOLO_DATASET_DIR / "images"
LABELS_DIR = YOLO_DATASET_DIR / "labels"

TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VAL_IMAGES_DIR = IMAGES_DIR / "val"

TRAIN_LABELS_DIR = LABELS_DIR / "train"
VAL_LABELS_DIR = LABELS_DIR / "val"

DATASET_YAML = YOLO_DATASET_DIR / "dataset.yaml"


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def generate_yolo_dataset() -> None:
    """
    Genera un dataset YOLO completo.
    """

    print("=" * 70)
    print("GENERATING YOLO DATASET")
    print("=" * 70)


    # --------------------------------------------------------
    # Leer anotaciones
    # --------------------------------------------------------

    print("\nLoading annotations...")

    df = load_dataset(JSON_PATH)

    print(f"Objects : {len(df)}")
    print(f"Images  : {df['filename'].nunique()}")
    print(f"Videos  : {df['video_id'].nunique()}")


    # --------------------------------------------------------
    # Cargar o generar split
    # --------------------------------------------------------

    if SPLIT_PATH.exists():

        print(f"\nLoading split: {SPLIT_NAME}")

        with open(
            SPLIT_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            split = json.load(f)

    else:

        print("\nSplit not found.")
        print("Generating new split...")

        split = split_dataset(
            df=df,
            train_size=0.70,
            iterations=10000,
            random_state=42,
        )

        SPLIT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            SPLIT_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                split,
                f,
                indent=4,
                sort_keys=True,
            )

        print(
            f"Split saved to:\n{SPLIT_PATH}"
        )


    # --------------------------------------------------------
    # Preparar directorios del dataset
    # --------------------------------------------------------

    print("\nCleaning previous dataset...")

    _clean_directory(TRAIN_IMAGES_DIR)
    _clean_directory(VAL_IMAGES_DIR)

    _clean_directory(TRAIN_LABELS_DIR)
    _clean_directory(VAL_LABELS_DIR)


    # --------------------------------------------------------
    # Leer todas las tareas del JSON
    #
    # Importante:
    # El DataFrame únicamente contiene imágenes con anotaciones.
    # Aquí necesitamos recorrer TODAS las imágenes exportadas por
    # Label Studio para copiar también las imágenes negativas
    # (sin semáforos), ya que YOLO las utiliza durante el
    # entrenamiento mediante un TXT vacío.
    # --------------------------------------------------------

    print("\nLoading Label Studio tasks...")

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        tasks = json.load(f)

    print(f"Tasks: {len(tasks)}")


    # --------------------------------------------------------
    # Copiar imágenes y registrar el subconjunto
    # --------------------------------------------------------

    train_videos = set(split["train"])
    val_videos = set(split["val"])


    copied_images = {
        "train": set(),
        "val": set(),
    }


    image_split = {}


    print("\nCopying images...")


    for task in tasks:

        image_path = task["data"]["image"]

        relative_path = image_path.split("?d=")[-1]

        source_image = ROOT_DIR / relative_path


        if not source_image.exists():

            raise FileNotFoundError(
                f"No existe la imagen: {source_image}"
            )


        video_id = source_image.parent.name
        filename = source_image.name


        # ----------------------------------------------------
        # Determinar subconjunto según vídeo
        # ----------------------------------------------------

        if video_id in train_videos:

            subset = "train"
            destination = TRAIN_IMAGES_DIR / filename


        elif video_id in val_videos:

            subset = "val"
            destination = VAL_IMAGES_DIR / filename


        else:

            raise ValueError(
                f"El vídeo '{video_id}' no aparece en el split."
            )


        image_split[filename] = {
            "subset": subset,
            "image_path": destination,
        }


        # ----------------------------------------------------
        # Evitar copiar imágenes duplicadas
        # ----------------------------------------------------

        if filename not in copied_images[subset]:

            shutil.copy2(
                source_image,
                destination,
            )

            copied_images[subset].add(filename)


    print(
        f"Train images : {len(copied_images['train'])}"
    )

    print(
        f"Val images   : {len(copied_images['val'])}"
    )


    # --------------------------------------------------------
    # Comprobación de integridad
    # --------------------------------------------------------

    duplicated = (
        copied_images["train"]
        &
        copied_images["val"]
    )


    if duplicated:

        raise RuntimeError(
            "Existen imágenes presentes en TRAIN y VAL."
        )


    print("Image copy completed.")
    
    # --------------------------------------------------------
    # Generar los ficheros TXT de YOLO
    # --------------------------------------------------------

    print("\nGenerating YOLO labels...")


    grouped = df.groupby(
        "filename",
        observed=True,
    )


    generated_labels = 0


    # --------------------------------------------------------
    # Imágenes con anotaciones
    # --------------------------------------------------------

    for filename, annotations in grouped:


        if filename not in image_split:

            raise RuntimeError(
                f"La imagen {filename} tiene anotaciones pero "
                "no aparece en el split."
            )


        subset = image_split[filename]["subset"]


        if subset == "train":

            label_path = (
                TRAIN_LABELS_DIR
                /
                Path(filename).with_suffix(".txt").name
            )


        else:

            label_path = (
                VAL_LABELS_DIR
                /
                Path(filename).with_suffix(".txt").name
            )


        with open(
            label_path,
            "w",
            encoding="utf-8",
        ) as f:


            for _, row in annotations.iterrows():


                # ------------------------------------------------
                # Conversión de clases
                #
                # Actualmente:
                # Off -> Red
                #
                # Clase única:
                # 0 = pedestrian_traffic_light
                # ------------------------------------------------

                class_id = 0


                f.write(
                    f"{class_id} "
                    f"{row['x_center']:.6f} "
                    f"{row['y_center']:.6f} "
                    f"{row['width']:.6f} "
                    f"{row['height']:.6f}\n"
                )


        generated_labels += 1



    # --------------------------------------------------------
    # Crear TXT vacíos para imágenes negativas
    # --------------------------------------------------------
    #
    # YOLO necesita que todas las imágenes tengan un fichero
    # .txt asociado. Las imágenes sin objetos llevan un txt vacío.
    # --------------------------------------------------------

    for filename, info in image_split.items():


        if filename in grouped.groups:

            continue


        if info["subset"] == "train":

            label_path = (
                TRAIN_LABELS_DIR
                /
                Path(filename).with_suffix(".txt").name
            )


        else:

            label_path = (
                VAL_LABELS_DIR
                /
                Path(filename).with_suffix(".txt").name
            )


        label_path.touch()

        generated_labels += 1



    print(
        f"Label files : {generated_labels}"
    )



    # --------------------------------------------------------
    # Validación básica del dataset
    # --------------------------------------------------------

    _validate_dataset(
        train_images=copied_images["train"],
        val_images=copied_images["val"],
        train_labels_dir=TRAIN_LABELS_DIR,
        val_labels_dir=VAL_LABELS_DIR,
    )



    # --------------------------------------------------------
    # Crear dataset.yaml
    # --------------------------------------------------------

    print("\nCreating dataset.yaml...")


    _create_dataset_yaml()



    # --------------------------------------------------------
    # Resumen final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("YOLO DATASET GENERATED")
    print("=" * 70)


    print(
        f"Train images : {len(copied_images['train'])}"
    )

    print(
        f"Val images   : {len(copied_images['val'])}"
    )

    print(
        f"Label files  : {generated_labels}"
    )


    print("\nSplit file:")
    print(SPLIT_PATH)


    print("\nDataset:")
    print(YOLO_DATASET_DIR)
    
# ============================================================
# FUNCIONES AUXILIARES
# ============================================================


def _clean_directory(directory: Path) -> None:
    """
    Elimina todos los archivos de un directorio.

    Si el directorio no existe, lo crea.
    """

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in directory.iterdir():

        if file.is_file():

            file.unlink()

        elif file.is_dir():

            shutil.rmtree(file)



def _create_dataset_yaml() -> None:
    """
    Crea el fichero dataset.yaml compatible con Ultralytics YOLO.
    """

    dataset = {

        "path": str(YOLO_DATASET_DIR),

        "train": "images/train",

        "val": "images/val",

        "names": {
            0: "pedestrian_traffic_light",
        },
    }


    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            dataset,
            f,
            sort_keys=False,
            allow_unicode=True,
        )



def _validate_dataset(
    train_images: set[str],
    val_images: set[str],
    train_labels_dir: Path,
    val_labels_dir: Path,
) -> None:
    """
    Comprueba la integridad básica del dataset YOLO generado.
    """


    print("\nValidating dataset...")


    # --------------------------------------------------------
    # No puede haber imágenes repetidas entre train y val
    # --------------------------------------------------------

    duplicated = train_images & val_images


    if duplicated:

        raise RuntimeError(
            f"Imágenes duplicadas entre train y val: {duplicated}"
        )



    # --------------------------------------------------------
    # Cada imagen debe tener un txt asociado
    # --------------------------------------------------------

    train_image_stems = {
        Path(img).stem
        for img in train_images
    }


    train_label_stems = {
        label.stem
        for label in train_labels_dir.glob("*.txt")
    }


    missing_train_labels = (
        train_image_stems
        -
        train_label_stems
    )


    if missing_train_labels:

        raise RuntimeError(
            "Imágenes TRAIN sin etiqueta:\n"
            f"{missing_train_labels}"
        )



    val_image_stems = {
        Path(img).stem
        for img in val_images
    }


    val_label_stems = {
        label.stem
        for label in val_labels_dir.glob("*.txt")
    }


    missing_val_labels = (
        val_image_stems
        -
        val_label_stems
    )


    if missing_val_labels:

        raise RuntimeError(
            "Imágenes VAL sin etiqueta:\n"
            f"{missing_val_labels}"
        )



    print("Dataset validation passed.")




# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    generate_yolo_dataset()