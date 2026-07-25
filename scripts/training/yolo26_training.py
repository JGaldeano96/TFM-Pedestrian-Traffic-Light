from pathlib import Path

from ultralytics import YOLO

# ============================================================
# RUTA RAÍZ DEL PROYECTO

# Esto hace que la ruta de partida sea la carpeta del proyecto 
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# ============================================================
# PARAMETRIZACIÓN DEL ENTRENAMIENTO

# Esto ayuda a que el script sea parametrizable y no sea 
# necesario tener uno por cada versión del dataset y tamaño
# de imagen
# ============================================================

DATASET_VERSION = "v1"

MODEL_NAME = "yolo26n" # ["yolo26n", "yolo26s"]
MODEL_WEIGHTS = "yolo26n.pt"

IMAGE_SIZE = 640 # [640, 800, 960, 1088]

# ============================================================
# RUTAS
# ============================================================

DATASET_YAML = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "yolo"
    / "dataset.yaml"
)

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "yolo"
    / f"dataset_{DATASET_VERSION}"
    / str(IMAGE_SIZE)
)

# ============================================================
# ENTRENAMIENTO
# ============================================================

model = YOLO(MODEL_WEIGHTS)

results = model.train(
    data=str(DATASET_YAML),

    epochs=75,
    imgsz=IMAGE_SIZE,

    batch=8,
    patience=20,

    optimizer="auto",
    cache=False,
    multi_scale=False,

    device=0,
    workers=2,

    seed=0,

    project=str(RESULTS_DIR),
    name=MODEL_NAME,

    # Data augmentation
    fliplr=0.5,
    degrees=3,
    translate=0.1,
)