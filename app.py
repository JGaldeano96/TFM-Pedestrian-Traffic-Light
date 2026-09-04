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



"""Punto de entrada de la demostración Streamlit del TFM."""

import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from tfm_demo.config import (
    CLASSIFIER_MODEL_PATH,
    DEFAULT_CLASSIFIER_THRESHOLD,
    DEFAULT_YOLO_ARCHITECTURE,
    DEFAULT_YOLO_CONFIDENCE,
    DEFAULT_YOLO_IMAGE_SIZE,
    YOLO_ARCHITECTURES,
    YOLO_IMAGE_SIZES,
    get_yolo_variant,
)
from tfm_demo.inference import VisualAdjustments
from tfm_demo.model_loading import (
    ModelConfigurationError,
    load_classifier_model,
    load_yolo_model,
    select_yolo_device,
)
from tfm_demo.video_processing import (
    VideoProcessingError,
    VideoProcessingOptions,
    VideoProcessingResult,
    has_mp4_signature,
    process_video,
)


VISUAL_DEFAULTS = {
    "brightness": 0,
    "contrast": 1.0,
    "saturation": 1.0,
}

OUTPUT_SIZE_PRESETS: dict[str, tuple[int, int] | None] = {
    "HD adaptable · máximo 1280×720": (1280, 720),
    "Full HD · máximo 1920×1080": (1920, 1080),
    "Compacto · máximo 854×480": (854, 480),
    "Resolución original": None,
}

PLAYER_WIDTHS: dict[str, int | str] = {
    "Compacto · 360 px": 360,
    "Mediano · 480 px": 480,
    "Grande · 720 px": 720,
    "Ocupar ancho disponible": "stretch",
}


def _reset_visual_controls() -> None:
    """Restaura únicamente los ajustes que alteran la imagen."""

    for key, value in VISUAL_DEFAULTS.items():
        st.session_state[key] = value


def _clear_previous_result() -> None:
    """Evita mostrar la salida de un vídeo subido anteriormente."""

    st.session_state.pop("processed_video", None)


def _resize_for_preview(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Limita el coste de transferir previsualizaciones de alta resolución."""

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame.copy()
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _render_responsive_video(
    data: bytes,
    player_width: int | str,
    container_key: str,
) -> None:
    """Muestra un vídeo dentro de un contenedor que respeta el tamaño elegido."""

    with st.container(
        key=container_key,
        width=player_width,
        horizontal_alignment="left",
    ):
        st.video(data, format="video/mp4", width="stretch")


def _safe_download_name(uploaded_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded_name).stem).strip("_")
    return f"{stem or 'video'}_procesado.mp4"


def _render_summary(result: dict[str, object]) -> None:
    summary = result["summary"]
    if not isinstance(summary, VideoProcessingResult):
        return

    st.subheader("Resumen de la ejecución")
    first_row = st.columns(4)
    first_row[0].metric("Fotogramas", f"{summary.frames:,}".replace(",", "."))
    first_row[1].metric(
        "Detecciones cercanas",
        f"{summary.detections:,}".replace(",", "."),
    )
    first_row[2].metric("🔴 Red", summary.red_predictions)
    first_row[3].metric("🟢 Green", summary.green_predictions)
    processing_fps = (
        summary.frames / summary.elapsed_seconds
        if summary.elapsed_seconds > 0
        else 0.0
    )
    performance_row = st.columns(2)
    performance_row[0].metric(
        "Tiempo total de procesamiento",
        f"{summary.elapsed_seconds:.1f} s",
    )
    performance_row[1].metric(
        "Fotogramas procesados por segundo",
        f"{processing_fps:.2f} FPS",
        help=(
            "Rendimiento global: incluye transformación, YOLO, clasificador, "
            "anotación y codificación del vídeo."
        ),
    )

    st.subheader("Latencia por modelo")
    timing_rows = []
    for stage, timing in (
        ("Detector YOLO", summary.detector_timing),
        ("Clasificador ONNX", summary.classifier_timing),
    ):
        timing_rows.append(
            {
                "Etapa": stage,
                "Ejecuciones": timing.executions,
                "Mínimo (ms)": (
                    f"{timing.minimum_ms:.2f}"
                    if timing.minimum_ms is not None
                    else "—"
                ),
                "Promedio (ms)": (
                    f"{timing.mean_ms:.2f}" if timing.mean_ms is not None else "—"
                ),
                "Mediana (ms)": (
                    f"{timing.median_ms:.2f}"
                    if timing.median_ms is not None
                    else "—"
                ),
                "Máximo (ms)": (
                    f"{timing.maximum_ms:.2f}"
                    if timing.maximum_ms is not None
                    else "—"
                ),
            }
        )
    st.table(timing_rows)
    st.caption(
        "YOLO se mide una vez por fotograma. El clasificador solo se mide en "
        "fotogramas donde YOLO encuentra una caja válida para recortar."
    )
    if summary.classifier_timing.executions:
        detector_total = summary.detector_timing.total_ms / 1000.0
        classifier_total = summary.classifier_timing.total_ms / 1000.0
        bottleneck = (
            "Detector YOLO"
            if detector_total >= classifier_total
            else "Clasificador ONNX"
        )
        st.info(
            f"Mayor coste acumulado entre los modelos: **{bottleneck}**. "
            f"YOLO: {detector_total:.2f} s · "
            f"clasificador: {classifier_total:.2f} s."
        )
    else:
        st.info(
            "El clasificador no se ejecutó porque YOLO no produjo ninguna "
            "detección válida."
        )
    st.caption(
        f"{summary.source_width}×{summary.source_height} → "
        f"{summary.output_width}×{summary.output_height} · "
        f"{summary.source_fps:.2f} FPS de entrada "
        f"→ {summary.output_fps:.2f} FPS de salida · "
        f"audio {'incluido' if summary.audio_included else 'eliminado'} · "
        f"modelo {result['model_label']} · umbral clasificador "
        f"{float(result['classifier_threshold']):.2f}"
    )
    if summary.warning:
        st.warning(summary.warning)


def run_app() -> None:
    """Construye la interfaz y coordina una ejecución de la demo."""

    st.set_page_config(
        page_title="TFM · Detección de semáforos",
        page_icon="🚦",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(155deg, #07111f 0%, #101c2f 52%, #07111f 100%); }
        [data-testid="stMetric"] { background: rgba(25, 39, 61, .72); border: 1px solid #2b3c55;
            padding: .8rem 1rem; border-radius: .8rem; }
        .pipeline { color: #a9b8ca; font-size: 1.05rem; margin-top: -.45rem; }
        .model-card { border: 1px solid #2b3c55; border-radius: .8rem; padding: .75rem 1rem;
            background: rgba(20, 34, 54, .72); }
        .st-key-input_video_player [data-testid="stVideo"],
        .st-key-processed_video_player [data-testid="stVideo"] {
            width: auto !important;
            max-width: 100% !important;
            max-height: 70vh !important;
            object-fit: contain;
        }
        .st-key-processing_preview img {
            width: auto !important;
            max-width: 100% !important;
            max-height: 70vh !important;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🚦 Semáforos peatonales en vídeo")
    st.markdown(
        '<p class="pipeline">Fotograma → ajustes visuales → YOLO V2 → caja más cercana → '
        'clasificador ONNX V2 → Red / Green</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuración")
        st.subheader("Detector YOLO · dataset V2")
        architecture = st.selectbox(
            "Arquitectura",
            YOLO_ARCHITECTURES,
            index=YOLO_ARCHITECTURES.index(DEFAULT_YOLO_ARCHITECTURE),
            format_func=lambda value: value.upper(),
        )
        image_size = st.select_slider(
            "Resolución de inferencia",
            options=YOLO_IMAGE_SIZES,
            value=DEFAULT_YOLO_IMAGE_SIZE,
            format_func=lambda value: f"{value} px",
        )
        yolo_confidence = st.slider(
            "Confianza mínima YOLO",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_YOLO_CONFIDENCE,
            step=0.05,
        )
        classifier_threshold = st.slider(
            "Umbral P(Green)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CLASSIFIER_THRESHOLD,
            step=0.05,
            help="P(Green) < umbral: Red; P(Green) ≥ umbral: Green.",
        )

        st.divider()
        visual_header, reset_column = st.columns([2, 1])
        visual_header.subheader("Imagen")
        reset_column.button(
            "Restablecer",
            on_click=_reset_visual_controls,
            use_container_width=True,
        )
        brightness = st.slider(
            "Brillo",
            min_value=-100,
            max_value=100,
            value=VISUAL_DEFAULTS["brightness"],
            step=5,
            key="brightness",
        )
        contrast = st.slider(
            "Contraste",
            min_value=0.50,
            max_value=1.50,
            value=VISUAL_DEFAULTS["contrast"],
            step=0.05,
            key="contrast",
        )
        saturation = st.slider(
            "Saturación",
            min_value=0.0,
            max_value=2.0,
            value=VISUAL_DEFAULTS["saturation"],
            step=0.05,
            key="saturation",
        )
        st.caption(
            f"Brillo {brightness:+d} · Contraste {contrast:.2f}× · "
            f"Saturación {saturation:.2f}×"
        )

        st.divider()
        show_live_preview = st.toggle(
            "Mostrar previsualización durante el procesamiento",
            value=False,
            help=(
                "Muestra fotogramas sueltos al ritmo de la inferencia, no a los "
                "FPS del vídeo. Desactivarla reduce trabajo de la interfaz."
            ),
        )
        preview_every_frames = st.select_slider(
            "Frecuencia de la previsualización",
            options=(1, 2, 5, 10, 20, 30),
            value=10,
            format_func=lambda value: (
                "Cada fotograma"
                if value == 1
                else f"Cada {value} fotogramas"
            ),
            help=(
                "Solo afecta a la imagen mostrada mientras se procesa. "
                "El vídeo final siempre contiene todos los fotogramas."
            ),
            disabled=not show_live_preview,
        )
        keep_audio = st.toggle(
            "Mantener audio en el vídeo procesado",
            value=False,
            key="keep_audio",
            on_change=_clear_previous_result,
            help=(
                "Desactivado: elimina completamente la pista de audio del MP4. "
                "Activado: conserva el audio de entrada."
            ),
        )
        st.caption(
            "El MP4 final conserva todos los fotogramas y se genera siempre "
            "a velocidad normal (1×)."
        )
        output_size_label = st.selectbox(
            "Resolución del vídeo final",
            tuple(OUTPUT_SIZE_PRESETS),
            index=0,
            help=(
                "Encaja el vídeo dentro del límite seleccionado manteniendo "
                "la proporción. Nunca recorta ni amplía el original."
            ),
        )
        player_width_label = st.selectbox(
            "Tamaño del reproductor",
            tuple(PLAYER_WIDTHS),
            index=0,
            help=(
                "Limita el ancho elegido y la altura al 70 % de la pantalla. "
                "Solo cambia la vista de Streamlit, no el archivo descargado."
            ),
        )
        maximum_output_size = OUTPUT_SIZE_PRESETS[output_size_label]
        player_width = PLAYER_WIDTHS[player_width_label]

    selected_variant = get_yolo_variant(architecture, image_size)
    yolo_device = select_yolo_device()
    model_columns = st.columns(2)
    try:
        with st.spinner("Cargando los modelos de la demo…"):
            detector = load_yolo_model(str(selected_variant.path))
            classifier_session, classifier_spec = load_classifier_model(
                str(CLASSIFIER_MODEL_PATH),
                prefer_cuda=yolo_device.uses_gpu,
            )
    except (FileNotFoundError, ModelConfigurationError, RuntimeError) as exc:
        st.error(f"No se pudieron preparar los modelos: {exc}")
        st.info(
            "Comprueba las rutas centralizadas en `tfm_demo/config.py` y la "
            "estructura documentada bajo `models/`."
        )
        st.stop()

    if yolo_device.uses_gpu:
        model_columns[0].success(
            f"YOLO cargado · {selected_variant.label} · {yolo_device.label}"
        )
    else:
        model_columns[0].warning(
            f"YOLO cargado · {selected_variant.label} · {yolo_device.label}"
        )
    model_columns[0].caption(
        str(selected_variant.path.relative_to(selected_variant.path.parents[3]))
    )
    provider = (
        classifier_spec.providers[0]
        if classifier_spec.providers
        else "desconocido"
    )
    if provider == "CUDAExecutionProvider":
        model_columns[1].success(f"Clasificador ONNX cargado · {provider}")
    else:
        model_columns[1].warning(
            f"Clasificador ONNX cargado · {provider} · fallback sin GPU"
        )
    model_columns[1].caption(classifier_spec.description)

    st.subheader("Vídeo de entrada")
    uploaded_video = st.file_uploader(
        "Selecciona un archivo MP4",
        type=("mp4",),
        accept_multiple_files=False,
        on_change=_clear_previous_result,
        key="video_upload",
        help="La resolución, la proporción y los FPS se conservan en la salida.",
    )
    if uploaded_video is not None:
        st.caption(
            f"{uploaded_video.name} · {uploaded_video.size / (1024 * 1024):.1f} MB"
        )
        with st.expander("Reproducir vídeo original"):
            _render_responsive_video(
                uploaded_video.getvalue(),
                player_width,
                "input_video_player",
            )

    start_processing = st.button(
        "▶ Procesar vídeo",
        type="primary",
        disabled=uploaded_video is None,
        use_container_width=True,
    )

    if start_processing and uploaded_video is not None:
        st.session_state.pop("processed_video", None)
        progress_bar = st.progress(0.0, text="Preparando vídeo…")
        frame_status = st.empty()
        if show_live_preview:
            preview_container = st.container(
                key="processing_preview",
                width=player_width,
                horizontal_alignment="left",
            )
            preview_placeholder = preview_container.empty()
        else:
            preview_placeholder = None

        def update_progress(current_frame: int, total_frames: int) -> None:
            if total_frames > 0:
                ratio = min(1.0, current_frame / total_frames)
                text = f"Fotograma {current_frame:,} de {total_frames:,}"
            else:
                ratio = 0.0
                text = f"Fotograma {current_frame:,}"
            progress_bar.progress(ratio, text=text.replace(",", "."))
            frame_status.caption("Inferencia en curso · no cierres esta pestaña")

        def update_preview(
            _original: np.ndarray,
            _transformed: np.ndarray,
            annotated: np.ndarray,
            frame_number: int,
        ) -> None:
            display_frame = _resize_for_preview(annotated)
            if preview_placeholder is None:
                return
            preview_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                caption=(
                    f"Entrada transformada y anotada · fotograma {frame_number:,}"
                ).replace(",", "."),
                width="stretch",
            )

        try:
            with tempfile.TemporaryDirectory(prefix="tfm_streamlit_") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                input_path = temp_dir / "uploaded_video.mp4"
                output_path = temp_dir / "processed_video.mp4"
                input_path.write_bytes(uploaded_video.getvalue())
                if not has_mp4_signature(input_path):
                    raise VideoProcessingError(
                        "El archivo subido tiene extensión .mp4 pero su contenedor no es MP4 válido."
                    )

                summary = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    detector=detector,
                    classifier_session=classifier_session,
                    classifier_spec=classifier_spec,
                    options=VideoProcessingOptions(
                        yolo_confidence=yolo_confidence,
                        classifier_threshold=classifier_threshold,
                        yolo_image_size=image_size,
                        yolo_device=yolo_device.argument,
                        visual_adjustments=VisualAdjustments(
                            brightness=brightness,
                            contrast=contrast,
                            saturation=saturation,
                        ),
                        preview_every_frames=preview_every_frames,
                        keep_audio=keep_audio,
                        maximum_output_size=maximum_output_size,
                    ),
                    progress_callback=update_progress,
                    preview_callback=(update_preview if show_live_preview else None),
                )
                st.session_state["processed_video"] = {
                    "bytes": output_path.read_bytes(),
                    "download_name": _safe_download_name(uploaded_video.name),
                    "summary": summary,
                    "model_label": selected_variant.label,
                    "classifier_threshold": classifier_threshold,
                }
            progress_bar.progress(1.0, text="Procesamiento completado")
            frame_status.success("Vídeo codificado y listo para reproducir.")
            if preview_placeholder is not None:
                preview_placeholder.empty()
        except (VideoProcessingError, OSError, ValueError) as exc:
            progress_bar.empty()
            frame_status.empty()
            if preview_placeholder is not None:
                preview_placeholder.empty()
            st.error(f"No se pudo procesar el vídeo: {exc}")

    processed_result = st.session_state.get("processed_video")
    if isinstance(processed_result, dict):
        st.divider()
        st.subheader("Vídeo procesado")
        video_bytes = processed_result["bytes"]
        st.download_button(
            "⬇ Descargar vídeo procesado",
            data=video_bytes,
            file_name=str(processed_result["download_name"]),
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )
        summary = processed_result.get("summary")
        if isinstance(summary, VideoProcessingResult):
            expected_duration = summary.frames / summary.output_fps
            st.caption(
                f"MP4 generado a {summary.output_fps:.2f} FPS · "
                f"duración aproximada {expected_duration:.2f} s · "
                "velocidad normal 1×"
            )
        _render_responsive_video(
            video_bytes,
            player_width,
            "processed_video_player",
        )
        _render_summary(processed_result)


if __name__ == "__main__":
    run_app()
