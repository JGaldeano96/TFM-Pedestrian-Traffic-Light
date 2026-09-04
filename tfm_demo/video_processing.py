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



"""Procesamiento de vídeo para la tubería YOLO → clasificador ONNX."""

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import onnxruntime as ort
import torch

from tfm_demo.inference import (
    DetectionPrediction,
    VisualAdjustments,
    annotate_frame,
    apply_visual_adjustments,
    predict_green_probabilities,
    select_largest_valid_detection,
)
from tfm_demo.model_loading import ClassifierSpec


class VideoProcessingError(RuntimeError):
    """Expone al usuario un fallo comprensible durante el procesamiento."""


@dataclass(frozen=True)
class VideoProcessingOptions:
    """Parámetros inmutables de una ejecución completa."""

    yolo_confidence: float
    classifier_threshold: float
    yolo_image_size: int
    yolo_device: str
    visual_adjustments: VisualAdjustments
    preview_every_frames: int = 5
    keep_audio: bool = False
    maximum_output_size: tuple[int, int] | None = (1280, 720)


@dataclass(frozen=True)
class TimingStatistics:
    """Distribución de latencias de una etapa de inferencia."""

    executions: int
    total_ms: float
    minimum_ms: float | None
    mean_ms: float | None
    median_ms: float | None
    maximum_ms: float | None


@dataclass(frozen=True)
class VideoProcessingResult:
    """Resumen cuantitativo y técnico del vídeo generado."""

    output_path: Path
    frames: int
    detections: int
    red_predictions: int
    green_predictions: int
    elapsed_seconds: float
    detector_timing: TimingStatistics
    classifier_timing: TimingStatistics
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_fps: float
    output_fps: float
    codec: str
    audio_included: bool = False
    warning: str | None = None


ProgressCallback = Callable[[int, int], None]
PreviewCallback = Callable[[np.ndarray, np.ndarray, np.ndarray, int], None]


def summarize_timings(durations_ms: Sequence[float]) -> TimingStatistics:
    """Calcula estadísticas descriptivas de una secuencia de latencias en ms."""

    if not durations_ms:
        return TimingStatistics(
            executions=0,
            total_ms=0.0,
            minimum_ms=None,
            mean_ms=None,
            median_ms=None,
            maximum_ms=None,
        )

    values = np.asarray(durations_ms, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Las duraciones deben ser valores finitos y no negativos.")
    return TimingStatistics(
        executions=int(values.size),
        total_ms=float(values.sum()),
        minimum_ms=float(values.min()),
        mean_ms=float(values.mean()),
        median_ms=float(np.median(values)),
        maximum_ms=float(values.max()),
    )


def _synchronize_yolo_device(device: str) -> None:
    """Sincroniza CUDA para que la medición de YOLO incluya el trabajo de GPU."""

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def calculate_output_dimensions(
    source_width: int,
    source_height: int,
    maximum_output_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Encaja una resolución en un límite, conservando proporción y sin ampliar."""

    if source_width <= 0 or source_height <= 0:
        raise VideoProcessingError("La resolución de origen debe ser positiva.")
    if maximum_output_size is None:
        return source_width, source_height

    maximum_width, maximum_height = maximum_output_size
    if maximum_width <= 0 or maximum_height <= 0:
        raise VideoProcessingError("Los límites de resolución deben ser positivos.")

    scale = min(
        1.0,
        maximum_width / source_width,
        maximum_height / source_height,
    )
    output_width = max(2, int(round(source_width * scale)))
    output_height = max(2, int(round(source_height * scale)))

    # H.264 con yuv420p requiere dimensiones pares.
    output_width -= output_width % 2
    output_height -= output_height % 2
    return output_width, output_height


def has_mp4_signature(path: Path) -> bool:
    """Comprueba el contenedor mediante la caja ISO BMFF ``ftyp``."""

    try:
        with path.open("rb") as video_file:
            header = video_file.read(64)
    except OSError:
        return False
    return len(header) >= 12 and b"ftyp" in header[4:64]


def _open_video(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise VideoProcessingError(f"No existe el vídeo de entrada: {path}")
    if path.suffix.lower() != ".mp4" or not has_mp4_signature(path):
        raise VideoProcessingError(
            "El archivo no parece ser un MP4 válido (contenedor ISO BMFF)."
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(
            "OpenCV no pudo abrir el MP4. Comprueba que el archivo no esté dañado."
        )
    return capture


def _encode_browser_mp4(
    intermediate_path: Path,
    source_path: Path,
    output_path: Path,
    keep_audio: bool,
) -> tuple[str, str | None]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copyfile(intermediate_path, output_path)
        return (
            "MP4V (OpenCV)",
            "FFmpeg no está disponible: se entrega MP4V, que algunos navegadores "
            "pueden no reproducir.",
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(intermediate_path),
    ]
    if keep_audio:
        command.extend([
            "-i",
            str(source_path),
        ])

    command.extend(["-map", "0:v:0"])
    if keep_audio:
        command.extend(["-map", "1:a:0?"])
    else:
        command.append("-an")

    command.extend([
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ])
    if keep_audio:
        command.extend([
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
        ])
    command.append(str(output_path))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size:
        return "H.264 (FFmpeg/libx264)", None

    shutil.copyfile(intermediate_path, output_path)
    detail = completed.stderr.strip().splitlines()
    reason = detail[-1] if detail else "error desconocido"
    return (
        "MP4V (OpenCV)",
        "Falló la conversión H.264; se entrega el MP4V de OpenCV. "
        f"Detalle: {reason}",
    )


def _has_audio_stream(path: Path, assume_when_unavailable: bool) -> bool:
    """Comprueba si el MP4 final contiene al menos una pista de audio."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return assume_when_unavailable
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def process_video(
    input_path: Path,
    output_path: Path,
    detector: object,
    classifier_session: ort.InferenceSession,
    classifier_spec: ClassifierSpec,
    options: VideoProcessingOptions,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
) -> VideoProcessingResult:
    """Procesa un MP4 completo y genera una salida anotada reproducible."""

    started_at = time.perf_counter()
    capture = _open_video(input_path)
    frame_count_hint = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))

    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise VideoProcessingError("El vídeo declara una resolución inválida.")
    fps_warning: str | None = None
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0
        fps_warning = "El vídeo no declara FPS válidos; se utilizaron 25 FPS."

    output_fps = source_fps
    output_width, output_height = calculate_output_dimensions(
        source_width,
        source_height,
        options.maximum_output_size,
    )

    intermediate_path = output_path.with_name(f"{output_path.stem}_opencv.mp4")
    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError(
            "OpenCV no pudo crear el vídeo MP4 de salida con el códec mp4v."
        )

    frames = detections = red_predictions = green_predictions = 0
    detector_durations_ms: list[float] = []
    classifier_durations_ms: list[float] = []

    try:
        while True:
            readable, original_frame = capture.read()
            if not readable:
                break

            frame_index = frames
            transformed_frame = apply_visual_adjustments(
                original_frame,
                options.visual_adjustments,
            )

            valid_bboxes: list[tuple[int, int, int, int]] = []
            yolo_confidences: list[float] = []
            crops: list[np.ndarray] = []
            selected_detection = None

            _synchronize_yolo_device(options.yolo_device)
            detector_started = time.perf_counter()
            try:
                yolo_results = detector.predict(
                    source=transformed_frame,
                    conf=options.yolo_confidence,
                    imgsz=options.yolo_image_size,
                    device=options.yolo_device,
                    verbose=False,
                )
                result = yolo_results[0] if yolo_results else None
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes):
                    raw_bboxes = boxes.xyxy.detach().cpu().numpy()
                    raw_confidences = boxes.conf.detach().cpu().numpy()
                    selected_detection = select_largest_valid_detection(
                        raw_bboxes=raw_bboxes,
                        raw_confidences=raw_confidences,
                        frame_width=source_width,
                        frame_height=source_height,
                        minimum_confidence=options.yolo_confidence,
                    )
                _synchronize_yolo_device(options.yolo_device)
            except Exception as exc:
                raise VideoProcessingError(
                    f"YOLO falló en el fotograma {frame_index + 1}: {exc}"
                ) from exc
            detector_durations_ms.append(
                (time.perf_counter() - detector_started) * 1000.0
            )

            if selected_detection is not None:
                x1, y1, x2, y2 = selected_detection.bbox
                crop = transformed_frame[y1:y2, x1:x2]
                if crop.size:
                    valid_bboxes.append(selected_detection.bbox)
                    yolo_confidences.append(selected_detection.yolo_confidence)
                    crops.append(crop)

            if crops:
                classifier_started = time.perf_counter()
                try:
                    raw_probabilities = predict_green_probabilities(
                        classifier_session,
                        classifier_spec,
                        crops,
                    )
                except Exception as exc:
                    raise VideoProcessingError(
                        f"El clasificador ONNX falló en el fotograma "
                        f"{frame_index + 1}: {exc}"
                    ) from exc
                classifier_durations_ms.append(
                    (time.perf_counter() - classifier_started) * 1000.0
                )
            else:
                raw_probabilities = np.empty(0, dtype=np.float32)

            frame_predictions: list[DetectionPrediction] = []
            for bbox, yolo_confidence, green_probability in zip(
                valid_bboxes,
                yolo_confidences,
                raw_probabilities,
            ):
                state = (
                    "Green"
                    if green_probability >= options.classifier_threshold
                    else "Red"
                )
                if state == "Green":
                    green_predictions += 1
                else:
                    red_predictions += 1
                frame_predictions.append(
                    DetectionPrediction(
                        bbox=bbox,
                        yolo_confidence=yolo_confidence,
                        green_probability=float(green_probability),
                        state=state,
                    )
                )

            detections += len(frame_predictions)
            annotated_frame = annotate_frame(transformed_frame, frame_predictions)
            if (output_width, output_height) == (source_width, source_height):
                output_frame = annotated_frame
            else:
                output_frame = cv2.resize(
                    annotated_frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            writer.write(output_frame)
            frames += 1

            preview_interval = max(1, options.preview_every_frames)
            progress_interval = max(1, preview_interval // 2)
            if progress_callback is not None and (
                frames == 1
                or frames % progress_interval == 0
                or (frame_count_hint and frames == frame_count_hint)
            ):
                progress_callback(frames, frame_count_hint)
            if preview_callback is not None and (
                frame_index % preview_interval == 0
                or (frame_count_hint and frames == frame_count_hint)
            ):
                preview_callback(
                    original_frame,
                    transformed_frame,
                    annotated_frame,
                    frames,
                )
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        intermediate_path.unlink(missing_ok=True)
        raise VideoProcessingError("El MP4 no contiene ningún fotograma legible.")
    if not intermediate_path.is_file() or intermediate_path.stat().st_size == 0:
        raise VideoProcessingError("La codificación del vídeo intermedio falló.")

    try:
        codec, encoding_warning = _encode_browser_mp4(
            intermediate_path,
            input_path,
            output_path,
            options.keep_audio,
        )
    finally:
        intermediate_path.unlink(missing_ok=True)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VideoProcessingError("No se generó un vídeo final reproducible.")

    audio_included = _has_audio_stream(
        output_path,
        assume_when_unavailable=options.keep_audio,
    )
    audio_warning = None
    if options.keep_audio and not audio_included:
        audio_warning = (
            "Se solicitó mantener el audio, pero la entrada o la salida no contiene "
            "una pista de audio utilizable."
        )

    elapsed_seconds = time.perf_counter() - started_at
    warning_parts = [
        warning
        for warning in (fps_warning, encoding_warning, audio_warning)
        if warning
    ]
    return VideoProcessingResult(
        output_path=output_path,
        frames=frames,
        detections=detections,
        red_predictions=red_predictions,
        green_predictions=green_predictions,
        elapsed_seconds=elapsed_seconds,
        detector_timing=summarize_timings(detector_durations_ms),
        classifier_timing=summarize_timings(classifier_durations_ms),
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
        source_fps=source_fps,
        output_fps=output_fps,
        codec=codec,
        audio_included=audio_included,
        warning=" ".join(warning_parts) or None,
    )
