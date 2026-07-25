from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import json
import shutil
import yaml
import cv2


################################################
# SPLIT_DATASET.PY
################################################




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

    video_stats = compute_video_statistics(df)

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

        cost = compute_cost(
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

    validate_split(
        best_split,
        total_videos,
    )

    return best_split


def compute_video_statistics(
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
                "images": group["filename"].nunique(),
                "objects": len(group),
                "Red": (group["state"] == "Red").sum(),
                "Green": (group["state"] == "Green").sum(),
            }
        )

    return pd.DataFrame(rows)


def compute_cost(
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


def validate_split(
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
        
        
        
        
        
        
################################################
# LABEL_STUDIO_PARSER.PY
################################################





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






################################################
# GENERATE_YOLO_DATASET.PY
################################################

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
            
            
def _create_dataset_yaml(
    dataset_yaml: Path,
    yolo_dataset_dir: Path,
) -> None:
    """
    Crea el fichero dataset.yaml compatible con Ultralytics YOLO.
    """

    dataset = {

        "path": str(yolo_dataset_dir),

        "train": "images/train",

        "val": "images/val",

        "names": {
            0: "pedestrian_traffic_light",
        },
    }

    with open(
        dataset_yaml,
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

    duplicated = train_images & val_images

    if duplicated:

        raise RuntimeError(
            f"Imágenes duplicadas entre train y val: {duplicated}"
        )

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
    

def generate_yolo_dataset(
    root_dir: Path,
    json_path: Path,
    split_path: Path,
    yolo_dataset_dir: Path,
) -> None:
    """
    Genera un dataset YOLO completo a partir de una exportación
    de Label Studio.
    """

    images_dir = yolo_dataset_dir / "images"
    labels_dir = yolo_dataset_dir / "labels"

    train_images_dir = images_dir / "train"
    val_images_dir = images_dir / "val"

    train_labels_dir = labels_dir / "train"
    val_labels_dir = labels_dir / "val"

    dataset_yaml = (
        yolo_dataset_dir
        / "dataset.yaml"
    )

    print("=" * 70)
    print("GENERATING YOLO DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Leer anotaciones
    # --------------------------------------------------------

    print("\nLoading annotations...")

    df = load_dataset(json_path)

    print(f"Objects : {len(df)}")
    print(f"Images  : {df['filename'].nunique()}")
    print(f"Videos  : {df['video_id'].nunique()}")

    # --------------------------------------------------------
    # Cargar o generar split
    # --------------------------------------------------------

    if split_path.exists():

        print(f"\nLoading split: {split_path.name}")

        with open(
            split_path,
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

        split_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            split_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                split,
                f,
                indent=4,
                sort_keys=True,
            )

        print(f"Split saved to:\n{split_path}")

    # --------------------------------------------------------
    # Preparar directorios
    # --------------------------------------------------------

    print("\nCleaning previous dataset...")

    _clean_directory(train_images_dir)
    _clean_directory(val_images_dir)

    _clean_directory(train_labels_dir)
    _clean_directory(val_labels_dir)

    # --------------------------------------------------------
    # Leer tareas de Label Studio
    # --------------------------------------------------------

    print("\nLoading Label Studio tasks...")

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        tasks = json.load(f)

    print(f"Tasks: {len(tasks)}")

    # --------------------------------------------------------
    # Copiar imágenes
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

        source_image = root_dir / relative_path

        if not source_image.exists():

            raise FileNotFoundError(
                f"No existe la imagen: {source_image}"
            )

        video_id = source_image.parent.name
        filename = source_image.name

        if video_id in train_videos:

            subset = "train"
            destination = train_images_dir / filename

        elif video_id in val_videos:

            subset = "val"
            destination = val_images_dir / filename

        else:

            raise ValueError(
                f"El vídeo '{video_id}' no aparece en el split."
            )

        image_split[filename] = {
            "subset": subset,
            "image_path": destination,
        }

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
    # Generar etiquetas YOLO
    # --------------------------------------------------------

    print("\nGenerating YOLO labels...")

    grouped = df.groupby(
        "filename",
        observed=True,
    )

    generated_labels = 0

    for filename, annotations in grouped:

        if filename not in image_split:

            raise RuntimeError(
                f"La imagen {filename} tiene anotaciones "
                "pero no aparece en el split."
            )

        subset = image_split[filename]["subset"]

        if subset == "train":

            label_path = (
                train_labels_dir
                / Path(filename).with_suffix(".txt").name
            )

        else:

            label_path = (
                val_labels_dir
                / Path(filename).with_suffix(".txt").name
            )

        with open(
            label_path,
            "w",
            encoding="utf-8",
        ) as f:

            for _, row in annotations.iterrows():

                f.write(
                    f"0 "
                    f"{row['x_center']:.6f} "
                    f"{row['y_center']:.6f} "
                    f"{row['width']:.6f} "
                    f"{row['height']:.6f}\n"
                )

        generated_labels += 1

    # --------------------------------------------------------
    # Crear TXT vacíos
    # --------------------------------------------------------

    for filename, info in image_split.items():

        if filename in grouped.groups:
            continue

        if info["subset"] == "train":

            label_path = (
                train_labels_dir
                / Path(filename).with_suffix(".txt").name
            )

        else:

            label_path = (
                val_labels_dir
                / Path(filename).with_suffix(".txt").name
            )

        label_path.touch()

        generated_labels += 1

    print(
        f"Label files : {generated_labels}"
    )

    # --------------------------------------------------------
    # Validar dataset
    # --------------------------------------------------------

    _validate_dataset(
        train_images=copied_images["train"],
        val_images=copied_images["val"],
        train_labels_dir=train_labels_dir,
        val_labels_dir=val_labels_dir,
    )

    # --------------------------------------------------------
    # Crear dataset.yaml
    # --------------------------------------------------------

    print("\nCreating dataset.yaml...")

    _create_dataset_yaml(
        dataset_yaml=dataset_yaml,
        yolo_dataset_dir=yolo_dataset_dir,
    )

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
    print(split_path)

    print("\nDataset:")
    print(yolo_dataset_dir)
    
    
    
    
    
    
    
    
    
# ============================================================
# CLASIFICADOR
# ============================================================






# ============================================================
# CLASIFICADOR - FUNCIONES AUXILIARES
# ============================================================

from pathlib import Path

import cv2
import numpy as np


def _create_classifier_directories(
    classifier_dataset_dir: Path,
) -> dict[str, Path]:
    """
    Crea la estructura del dataset de clasificación.

    classifier/
    ├── train/
    │   ├── Red/
    │   └── Green/
    └── val/
        ├── Red/
        └── Green/

    Si el directorio ya existe se elimina su contenido.

    Returns
    -------
    dict[str, Path]
        Diccionario con todas las rutas creadas.
    """

    _clean_directory(classifier_dataset_dir)

    directories = {}

    for subset in ("train", "val"):

        for state in ("Red", "Green"):

            path = (
                classifier_dataset_dir
                / subset
                / state
            )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            directories[(subset, state)] = path

    return directories


def _crop_bbox(
    image: np.ndarray,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> np.ndarray:
    """
    Recorta una bounding box en formato YOLO.

    Parameters
    ----------
    image : np.ndarray
        Imagen original.

    x_center, y_center : float
        Coordenadas normalizadas del centro.

    width, height : float
        Tamaño normalizado de la bounding box.

    Returns
    -------
    np.ndarray
        Recorte correspondiente al objeto.

    Raises
    ------
    ValueError
        Si el recorte resulta inválido.
    """

    image_height, image_width = image.shape[:2]

    x1 = int((x_center - width / 2) * image_width)
    y1 = int((y_center - height / 2) * image_height)

    x2 = int((x_center + width / 2) * image_width)
    y2 = int((y_center + height / 2) * image_height)

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    if x2 <= x1 or y2 <= y1:

        raise ValueError(
            "Bounding box inválida."
        )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:

        raise ValueError(
            "El recorte está vacío."
        )

    return crop


def _save_classifier_crop(
    crop: np.ndarray,
    output_dir: Path,
    filename: str,
    object_id: str,
) -> None:
    """
    Guarda un recorte del clasificador.

    El nombre generado será:

        IMG_000123_abcd1234.jpg
    """

    output_path = (
        output_dir
        / f"{Path(filename).stem}_{object_id}.jpg"
    )

    success = cv2.imwrite(
        str(output_path),
        crop,
    )

    if not success:

        raise IOError(
            f"No se pudo guardar {output_path}"
        )


def _map_classifier_label(
    state: str,
) -> str:
    """
    Convierte las etiquetas originales a las utilizadas
    por el clasificador.

    Red   -> Red
    Green -> Green
    Off   -> Red
    """

    mapping = {
        "Red": "Red",
        "Green": "Green",
        "Off": "Red",
    }

    try:

        return mapping[state]

    except KeyError as exc:

        raise ValueError(
            f"Estado desconocido: {state}"
        ) from exc


def _validate_classifier_dataset(
    classifier_dataset_dir: Path,
) -> None:
    """
    Comprueba la integridad del dataset generado.
    """

    print("\nValidating classifier dataset...")

    for subset in ("train", "val"):

        for state in ("Red", "Green"):

            directory = (
                classifier_dataset_dir
                / subset
                / state
            )

            if not directory.exists():

                raise RuntimeError(
                    f"No existe {directory}"
                )

            images = list(
                directory.glob("*.jpg")
            )

            if len(images) == 0:

                raise RuntimeError(
                    f"La carpeta {directory} está vacía."
                )

            for image_path in images:

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:

                    raise RuntimeError(
                        f"Imagen corrupta: {image_path}"
                    )

    print("Classifier dataset validation passed.")
    
    
    
    
    
    
def generate_classifier_dataset(
    root_dir: Path,
    json_path: Path,
    split_path: Path,
    classifier_dataset_dir: Path,
    image_size: int,
) -> None:
    """
    Genera un dataset para el clasificador de estados del semáforo.

    Cada imagen generada corresponde al recorte de un único
    semáforo anotado.

    Actualmente:

        Off -> Red
    """

    print("=" * 70)
    print("GENERATING CLASSIFIER DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Leer anotaciones
    # --------------------------------------------------------

    print("\nLoading annotations...")

    df = load_dataset(json_path)

    print(f"Objects : {len(df)}")
    print(f"Images  : {df['filename'].nunique()}")
    print(f"Videos  : {df['video_id'].nunique()}")

    # --------------------------------------------------------
    # Leer split
    # --------------------------------------------------------

    if not split_path.exists():

        raise FileNotFoundError(
            f"No existe el split:\n{split_path}"
        )

    print(f"\nLoading split: {split_path.name}")

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(f)

    train_videos = set(split["train"])
    val_videos = set(split["val"])

    # --------------------------------------------------------
    # Preparar directorios
    # --------------------------------------------------------

    print("\nPreparing directories...")

    directories = _create_classifier_directories(
        classifier_dataset_dir
    )

    # --------------------------------------------------------
    # Generar recortes
    # --------------------------------------------------------

    print("\nGenerating crops...")

    generated = 0

    for _, row in df.iterrows():

        image = cv2.imread(
            str(row["image_path"])
        )

        if image is None:

            raise RuntimeError(
                f"No se pudo leer:\n{row['image_path']}"
            )

        crop = _crop_bbox(
            image=image,
            x_center=row["x_center"],
            y_center=row["y_center"],
            width=row["width"],
            height=row["height"],
        )

        # ----------------------------------------------------
        # Redimensionar el recorte
        # ----------------------------------------------------

        crop = cv2.resize(
            crop,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

        label = _map_classifier_label(
            row["state"]
        )

        if row["video_id"] in train_videos:

            subset = "train"

        elif row["video_id"] in val_videos:

            subset = "val"

        else:

            raise RuntimeError(
                f"El vídeo '{row['video_id']}' "
                "no aparece en el split."
            )

        output_dir = directories[
            (subset, label)
        ]

        _save_classifier_crop(
            crop=crop,
            output_dir=output_dir,
            filename=row["filename"],
            object_id=row["object_id"],
        )

        generated += 1

    print(f"Crops generated : {generated}")

    # --------------------------------------------------------
    # Validación
    # --------------------------------------------------------

    _validate_classifier_dataset(
        classifier_dataset_dir
    )

    # --------------------------------------------------------
    # Resumen final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFIER DATASET GENERATED")
    print("=" * 70)

    train_red = len(
        list(
            (classifier_dataset_dir / "train" / "Red").glob("*.jpg")
        )
    )

    train_green = len(
        list(
            (classifier_dataset_dir / "train" / "Green").glob("*.jpg")
        )
    )

    val_red = len(
        list(
            (classifier_dataset_dir / "val" / "Red").glob("*.jpg")
        )
    )

    val_green = len(
        list(
            (classifier_dataset_dir / "val" / "Green").glob("*.jpg")
        )
    )

    print("\nTRAIN")
    print(f"Red   : {train_red}")
    print(f"Green : {train_green}")

    print("\nVAL")
    print(f"Red   : {val_red}")
    print(f"Green : {val_green}")

    print(f"\nImage size : {image_size}x{image_size}")
    print(f"Total crops : {generated}")

    print("\nDataset:")
    print(classifier_dataset_dir)