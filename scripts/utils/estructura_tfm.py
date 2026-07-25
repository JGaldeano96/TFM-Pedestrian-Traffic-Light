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