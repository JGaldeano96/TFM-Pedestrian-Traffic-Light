from pathlib import Path

from ultralytics import YOLO

# ============================================================
# RUTA RAÍZ DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# ============================================================
# PARAMETRIZACIÓN DEL ENTRENAMIENTO
# ============================================================

DATASET_VERSION = "v2"

MODEL_NAME = "yolo26s"  # ["yolo26n", "yolo26s"]
MODEL_WEIGHTS = "yolo26s.pt"

IMAGE_SIZE = 1088  # [640, 800, 960, 1088]

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
# PESOS DEL MODELO
# ============================================================

if DATASET_VERSION == "v1":

    WEIGHTS = MODEL_WEIGHTS

else:

    previous_version = f"v{int(DATASET_VERSION[1:]) - 1}"

    WEIGHTS = (
        ROOT_DIR
        / "results"
        / "yolo"
        / f"dataset_{previous_version}"
        / str(IMAGE_SIZE)
        / MODEL_NAME
        / "weights"
        / "best.pt"
    )

# ============================================================
# ENTRENAMIENTO
# ============================================================

model = YOLO(str(WEIGHTS))

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