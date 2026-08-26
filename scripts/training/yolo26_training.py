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