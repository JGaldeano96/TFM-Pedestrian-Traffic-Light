import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Permitir servir archivos locales
os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"

# Directorio raíz al que Label Studio podrá acceder
os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(ROOT_DIR)

subprocess.run(["label-studio"], check=True)
