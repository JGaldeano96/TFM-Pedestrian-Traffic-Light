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



"""
print_project_tree.py

Imprime el árbol del proyecto ignorando archivos grandes y carpetas
que no aportan información sobre la estructura.
"""

from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".ipynb_checkpoints",
}

EXCLUDED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".pt",
    ".onnx",
    ".keras",
    ".h5",
    ".tflite",
    ".zip",
    ".gz",
    ".tar",
    ".txt",
    ".jpg:Zone.Identifier"

}


def print_tree(directory: Path, prefix: str = "") -> None:
    entries = []

    for entry in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):

        if entry.name.startswith("."):
            continue

        if entry.is_dir() and entry.name in EXCLUDED_DIRS:
            continue

        if entry.is_file() and entry.suffix.lower() in EXCLUDED_EXTENSIONS:
            continue

        entries.append(entry)

    total = len(entries)

    for index, entry in enumerate(entries):
        connector = "└── " if index == total - 1 else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if index == total - 1 else "│   "
            print_tree(entry, prefix + extension)


if __name__ == "__main__":
    root = Path.cwd()

    print(root.name)
    print_tree(root)