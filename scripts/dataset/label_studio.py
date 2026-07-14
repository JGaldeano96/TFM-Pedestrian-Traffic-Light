import os
import subprocess
from pathlib import Path

# Permitir servir archivos locales
os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"

# Directorio raíz al que Label Studio podrá acceder
os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = "/home/jgaldeano/tfm"

label_studio = Path.home() / "envs" / "labelstudio" / "bin" / "label-studio"

subprocess.run([str(label_studio)], check=True)