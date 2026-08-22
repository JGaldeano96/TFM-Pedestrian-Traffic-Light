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
