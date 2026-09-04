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



"""Configuración centralizada de modelos y parámetros de la demo."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
YOLO_MODELS_DIR = MODELS_DIR / "yolo" / "dataset_v2"
CLASSIFIER_MODEL_PATH = (
    MODELS_DIR / "classifier" / "dataset_v2" / "classifier_v2.onnx"
)

YOLO_ARCHITECTURES = ("yolo26n", "yolo26s")
YOLO_IMAGE_SIZES = (640, 800, 960, 1088)
DEFAULT_YOLO_ARCHITECTURE = "yolo26s"
DEFAULT_YOLO_IMAGE_SIZE = 1088

DEFAULT_YOLO_CONFIDENCE = 0.25
DEFAULT_CLASSIFIER_THRESHOLD = 0.50


@dataclass(frozen=True)
class YoloVariant:
    """Describe una variante entrenada de YOLO para dataset V2."""

    architecture: str
    image_size: int
    path: Path

    @property
    def label(self) -> str:
        """Devuelve un nombre legible para la interfaz."""

        return f"{self.architecture.upper()} · {self.image_size} px · dataset V2"


def get_yolo_variant(architecture: str, image_size: int) -> YoloVariant:
    """Obtiene la ruta centralizada de una combinación YOLO V2."""

    if architecture not in YOLO_ARCHITECTURES:
        raise ValueError(f"Arquitectura YOLO no soportada: {architecture}")
    if image_size not in YOLO_IMAGE_SIZES:
        raise ValueError(f"Resolución YOLO no soportada: {image_size}")

    filename = f"{architecture}_{image_size}_dataset_v2_best.pt"
    return YoloVariant(
        architecture=architecture,
        image_size=image_size,
        path=YOLO_MODELS_DIR / filename,
    )


def all_yolo_variants() -> tuple[YoloVariant, ...]:
    """Lista las ocho combinaciones YOLO disponibles para dataset V2."""

    return tuple(
        get_yolo_variant(architecture, image_size)
        for architecture in YOLO_ARCHITECTURES
        for image_size in YOLO_IMAGE_SIZES
    )
