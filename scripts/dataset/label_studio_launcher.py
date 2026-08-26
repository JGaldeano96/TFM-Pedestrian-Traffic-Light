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


import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Permitir servir archivos locales
os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"

# Directorio raíz al que Label Studio podrá acceder
os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(ROOT_DIR)

subprocess.run(["label-studio"], check=True)
