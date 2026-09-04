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


"""Preprocesamiento, clasificación y anotación de fotogramas."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from tfm_demo.model_loading import ClassifierSpec


class InferenceError(RuntimeError):
    """Indica una entrada inválida o una salida inesperada durante inferencia."""


@dataclass(frozen=True)
class VisualAdjustments:
    """Ajustes visuales aplicados antes de los dos modelos."""

    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0


@dataclass(frozen=True)
class DetectionPrediction:
    """Resultado conjunto de YOLO y del clasificador para una detección."""

    bbox: tuple[int, int, int, int]
    yolo_confidence: float
    green_probability: float
    state: str


@dataclass(frozen=True)
class SelectedDetection:
    """Única detección elegida como la más cercana del fotograma."""

    bbox: tuple[int, int, int, int]
    yolo_confidence: float


def apply_visual_adjustments(
    frame_bgr: np.ndarray,
    adjustments: VisualAdjustments,
) -> np.ndarray:
    """Aplica contraste, brillo y saturación a un fotograma BGR."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise InferenceError("El fotograma debe ser una imagen BGR de tres canales.")

    if (
        adjustments.brightness == 0
        and np.isclose(adjustments.contrast, 1.0)
        and np.isclose(adjustments.saturation, 1.0)
    ):
        return frame_bgr

    adjusted = frame_bgr.astype(np.float32)
    adjusted = (
        (adjusted - 127.5) * float(adjustments.contrast)
        + 127.5
        + float(adjustments.brightness)
    )
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    if np.isclose(adjustments.saturation, 1.0):
        return adjusted

    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= float(adjustments.saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def clip_bbox(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Recorta una caja XYXY a los límites de imagen o devuelve ``None``."""

    if len(bbox) != 4:
        raise InferenceError(f"Una bounding box debe tener 4 valores: {bbox!r}")

    x1 = max(0, min(frame_width, int(np.floor(float(bbox[0])))))
    y1 = max(0, min(frame_height, int(np.floor(float(bbox[1])))))
    x2 = max(0, min(frame_width, int(np.ceil(float(bbox[2])))))
    y2 = max(0, min(frame_height, int(np.ceil(float(bbox[3])))))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def select_largest_valid_detection(
    raw_bboxes: Sequence[Sequence[float]],
    raw_confidences: Sequence[float],
    frame_width: int,
    frame_height: int,
    minimum_confidence: float,
) -> SelectedDetection | None:
    """Selecciona la caja válida de mayor área tras aplicar el umbral YOLO.

    El área se calcula después de recortar la caja a los límites del fotograma.
    En caso de empate se elige la detección de mayor confianza.
    """

    if len(raw_bboxes) != len(raw_confidences):
        raise InferenceError("YOLO devolvió distinto número de cajas y confianzas.")

    selected: SelectedDetection | None = None
    selected_key = (-1, -1.0)
    for raw_bbox, raw_confidence in zip(raw_bboxes, raw_confidences):
        confidence = float(raw_confidence)
        if confidence < minimum_confidence:
            continue
        bbox = clip_bbox(raw_bbox, frame_width, frame_height)
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        candidate_key = (area, confidence)
        if candidate_key > selected_key:
            selected = SelectedDetection(
                bbox=bbox,
                yolo_confidence=confidence,
            )
            selected_key = candidate_key
    return selected


def prepare_classifier_batch(
    crops_bgr: Sequence[np.ndarray],
    spec: ClassifierSpec,
) -> np.ndarray:
    """Reproduce la entrada de entrenamiento: RGB, tamaño ONNX y 0–255.

    La capa ``Rescaling(1 / 255)`` forma parte del grafo exportado; por ello
    esta función no divide los píxeles de nuevo.
    """

    if not crops_bgr:
        raise InferenceError("No se recibieron recortes para el clasificador.")

    prepared: list[np.ndarray] = []
    for index, crop in enumerate(crops_bgr):
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            raise InferenceError(f"El recorte {index} está vacío o no es BGR válido.")

        resized_bgr = cv2.resize(
            crop,
            (spec.width, spec.height),
            interpolation=cv2.INTER_AREA,
        )
        resized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        prepared.append(resized.astype(spec.dtype, copy=False))

    batch = np.stack(prepared, axis=0)
    if spec.layout == "NCHW":
        batch = np.transpose(batch, (0, 3, 1, 2))
    return np.ascontiguousarray(batch)


def predict_green_probabilities(
    session: ort.InferenceSession,
    spec: ClassifierSpec,
    crops_bgr: Sequence[np.ndarray],
) -> np.ndarray:
    """Obtiene una P(Green) por recorte y valida forma, finitud y rango."""

    batch = prepare_classifier_batch(crops_bgr, spec)
    try:
        raw_output = session.run(
            [spec.output_name],
            {spec.input_name: batch},
        )[0]
    except Exception as exc:
        raise InferenceError(f"Falló la inferencia ONNX: {exc}") from exc

    probabilities = np.asarray(raw_output, dtype=np.float32)
    batch_size = len(crops_bgr)
    if probabilities.size != batch_size:
        raise InferenceError(
            "La salida ONNX no contiene una probabilidad por recorte: "
            f"entrada={batch_size}, salida={probabilities.shape}."
        )

    probabilities = probabilities.reshape(batch_size)
    if not np.all(np.isfinite(probabilities)):
        raise InferenceError("La salida ONNX contiene NaN o infinito.")
    if np.any(probabilities < -1e-5) or np.any(probabilities > 1.00001):
        raise InferenceError(
            "La salida ONNX no parece ser P(Green): hay valores fuera de [0, 1]."
        )
    return np.clip(probabilities, 0.0, 1.0)


def annotate_frame(
    frame_bgr: np.ndarray,
    predictions: Sequence[DetectionPrediction],
) -> np.ndarray:
    """Dibuja cajas y probabilidades legibles sobre un fotograma BGR."""

    annotated = frame_bgr.copy()
    scale = max(0.45, min(frame_bgr.shape[:2]) / 1100.0)
    thickness = max(2, int(round(scale * 3)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for prediction in predictions:
        color = (52, 199, 89) if prediction.state == "Green" else (45, 45, 230)
        x1, y1, x2, y2 = prediction.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        first_line = (
            f"{prediction.state}  |  YOLO {prediction.yolo_confidence:.2f}"
        )
        second_line = f"P(Green) {prediction.green_probability:.2f}"

        line_height = max(17, int(round(25 * scale)))
        text_scale = max(0.42, 0.58 * scale)
        text_sizes = [
            cv2.getTextSize(line, font, text_scale, thickness=1)[0]
            for line in (first_line, second_line)
        ]
        label_width = max(size[0] for size in text_sizes) + 12
        label_height = line_height * 2 + 8
        label_top = y1 - label_height if y1 >= label_height else y1
        label_bottom = min(frame_bgr.shape[0], label_top + label_height)
        label_right = min(frame_bgr.shape[1], x1 + label_width)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            first_line,
            (x1 + 6, label_top + line_height),
            font,
            text_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            second_line,
            (x1 + 6, label_top + line_height * 2),
            font,
            text_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
