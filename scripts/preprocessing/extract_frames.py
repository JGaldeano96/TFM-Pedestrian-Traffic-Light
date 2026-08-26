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
Extrae fotogramas de los vídeos procesados para la creación del dataset.

El script recorre todos los vídeos MP4 del directorio de entrada y extrae
fotogramas a una frecuencia configurable (FPS) mediante FFmpeg.

Cada vídeo genera una carpeta con sus correspondientes imágenes en formato
JPG. Si un vídeo ya ha sido procesado, se omite.

Al finalizar, se muestra un resumen con el número de vídeos procesados,
omitidos, errores y el tiempo total de ejecución.
"""

from pathlib import Path
import subprocess
import time


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT_DIR
    / "data"
    / "source_data"
    / "processed_data"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "source_data"
    / "frames"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 3


# ============================================================
# VÍDEOS
# ============================================================

videos = sorted(INPUT_DIR.glob("*.mp4"))

total = len(videos)

processed = 0
skipped = 0
errors = 0

start = time.time()

print("=" * 60)
print("FRAME EXTRACTION")
print("=" * 60)
print(f"Vídeos encontrados: {total}\n")


# ============================================================
# PROCESAMIENTO
# ============================================================

for index, video in enumerate(videos, start=1):

    video_code = video.stem

    frame_folder = OUTPUT_DIR / video_code

    print(f"[{index}/{total}] {video.name}")

    # --------------------------------------------------------
    # COMPROBAR SI YA HA SIDO PROCESADO
    # --------------------------------------------------------

    if frame_folder.exists():

        skipped += 1

        print("   ✓ Ya procesado -> Skip\n")

        continue

    # --------------------------------------------------------
    # CREAR CARPETA DE FRAMES
    # --------------------------------------------------------

    frame_folder.mkdir(parents=True)

    output_pattern = (
        frame_folder
        / f"{video_code}_frame%06d.jpg"
    )

    # --------------------------------------------------------
    # EXTRAER FRAMES
    # --------------------------------------------------------

    try:

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video),
                "-vf",
                f"fps={FPS}",
                "-q:v",
                "2",
                str(output_pattern),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
            ],
            check=True,
        )

        num_frames = len(
            list(frame_folder.glob("*.jpg"))
        )

        print(
            f"   ✓ Frames extraídos: {num_frames}\n"
        )

        processed += 1

    except Exception as e:

        errors += 1

        print(f"   ✗ ERROR: {e}\n")


# ============================================================
# RESUMEN
# ============================================================

elapsed = time.time() - start

minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print("=" * 60)
print("RESUMEN")
print("=" * 60)

print(f"Vídeos encontrados : {total}")
print(f"Procesados         : {processed}")
print(f"Saltados           : {skipped}")
print(f"Errores            : {errors}")
print(f"Tiempo total       : {minutes} min {seconds} s")

print("=" * 60)