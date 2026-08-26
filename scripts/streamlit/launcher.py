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


"""Lanza la demostración Streamlit desde el intérprete Python activo."""

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
APP_PATH = ROOT_DIR / "app.py"


def main() -> int:
    """Ejecuta ``streamlit run app.py`` desde la raíz del repositorio."""

    if not APP_PATH.is_file():
        print(f"No se encuentra la aplicación Streamlit: {APP_PATH}", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        *sys.argv[1:],
    ]
    try:
        completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    except KeyboardInterrupt:
        return 130
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
