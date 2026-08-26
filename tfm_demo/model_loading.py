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



"""Carga y validación de los modelos YOLO y ONNX."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from ultralytics import YOLO


class ModelConfigurationError(RuntimeError):
    """Indica que un modelo no existe o expone una interfaz inesperada."""


@dataclass(frozen=True)
class ClassifierSpec:
    """Contrato de entrada y salida observado en el clasificador ONNX."""

    input_name: str
    output_name: str
    input_shape: tuple[Any, ...]
    output_shape: tuple[Any, ...]
    layout: Literal["NHWC", "NCHW"]
    height: int
    width: int
    dtype: np.dtype
    providers: tuple[str, ...]

    @property
    def description(self) -> str:
        """Resume la interfaz del modelo para mostrarla en la aplicación."""

        return (
            f"{self.layout} · RGB {self.width}×{self.height} · "
            f"{self.dtype.name} · salida P(Green)"
        )


@dataclass(frozen=True)
class YoloDevice:
    """Dispositivo de inferencia elegido para Ultralytics."""

    argument: str
    label: str
    uses_gpu: bool


def select_yolo_device() -> YoloDevice:
    """Prioriza la primera GPU CUDA y usa CPU solo si no está disponible."""

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return YoloDevice(
            argument="cuda:0",
            label=f"CUDA:0 · {torch.cuda.get_device_name(0)}",
            uses_gpu=True,
        )
    return YoloDevice(
        argument="cpu",
        label="CPU · CUDA no disponible",
        uses_gpu=False,
    )


def _fixed_dimension(value: Any, name: str) -> int:
    """Convierte una dimensión estática positiva o genera un error claro."""

    if isinstance(value, int) and value > 0:
        return value
    raise ModelConfigurationError(
        f"La dimensión {name} del clasificador debe ser estática; se recibió {value!r}."
    )


def _infer_classifier_spec(session: ort.InferenceSession) -> ClassifierSpec:
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise ModelConfigurationError(
            f"El clasificador debe tener una sola entrada; se encontraron {len(inputs)}."
        )
    if len(outputs) != 1:
        raise ModelConfigurationError(
            f"El clasificador debe tener una sola salida; se encontraron {len(outputs)}."
        )

    model_input = inputs[0]
    model_output = outputs[0]
    input_shape = tuple(model_input.shape)
    output_shape = tuple(model_output.shape)

    if len(input_shape) != 4:
        raise ModelConfigurationError(
            f"Se esperaba una entrada de imagen 4D; el ONNX declara {input_shape}."
        )

    if input_shape[-1] == 3:
        layout: Literal["NHWC", "NCHW"] = "NHWC"
        height = _fixed_dimension(input_shape[1], "alto")
        width = _fixed_dimension(input_shape[2], "ancho")
    elif input_shape[1] == 3:
        layout = "NCHW"
        height = _fixed_dimension(input_shape[2], "alto")
        width = _fixed_dimension(input_shape[3], "ancho")
    else:
        raise ModelConfigurationError(
            "No se pudo identificar el eje RGB en la entrada ONNX "
            f"{input_shape}; se esperaba NHWC o NCHW con 3 canales."
        )

    dtype_by_onnx_type = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
    }
    try:
        dtype = dtype_by_onnx_type[model_input.type]
    except KeyError as exc:
        raise ModelConfigurationError(
            f"Tipo de entrada ONNX no soportado: {model_input.type}."
        ) from exc

    if len(output_shape) > 2 or (
        len(output_shape) == 2
        and isinstance(output_shape[-1], int)
        and output_shape[-1] != 1
    ):
        raise ModelConfigurationError(
            "La salida debe ser escalar, [N] o [N, 1]; "
            f"el ONNX declara {output_shape}."
        )

    return ClassifierSpec(
        input_name=model_input.name,
        output_name=model_output.name,
        input_shape=input_shape,
        output_shape=output_shape,
        layout=layout,
        height=height,
        width=width,
        dtype=dtype,
        providers=tuple(session.get_providers()),
    )


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str) -> YOLO:
    """Carga una instancia YOLO una sola vez por ruta de pesos."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentran los pesos YOLO: {path}")
    return YOLO(str(path))


@st.cache_resource(show_spinner=False)
def load_classifier_model(
    model_path: str,
    prefer_cuda: bool,
) -> tuple[ort.InferenceSession, ClassifierSpec]:
    """Carga, optimiza y valida una sesión ONNX Runtime cacheada."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el clasificador ONNX: {path}")

    available_providers = ort.get_available_providers()
    provider_order = (
        ("CUDAExecutionProvider", "CPUExecutionProvider")
        if prefer_cuda
        else ("CPUExecutionProvider",)
    )
    preferred_providers = [
        provider for provider in provider_order if provider in available_providers
    ]
    if not preferred_providers:
        raise ModelConfigurationError(
            "ONNX Runtime no ofrece un proveedor de ejecución CUDA o CPU."
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=preferred_providers,
        )
    except Exception as cuda_exc:
        if "CUDAExecutionProvider" not in preferred_providers:
            raise ModelConfigurationError(
                f"ONNX Runtime no pudo cargar {path}: {cuda_exc}"
            ) from cuda_exc
        try:
            session = ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as cpu_exc:
            raise ModelConfigurationError(
                "ONNX Runtime no pudo cargar el clasificador ni con CUDA ni con "
                f"CPU: CUDA={cuda_exc}; CPU={cpu_exc}"
            ) from cpu_exc

    return session, _infer_classifier_spec(session)
