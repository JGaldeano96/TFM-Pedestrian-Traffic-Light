"""
Preprocesa los vídeos del proyecto para unificar su formato en MP4.

El script recorre un directorio de entrada buscando archivos .mov y .mp4.
Los vídeos MOV se convierten a MP4 mediante FFmpeg (H.264/AAC), mientras
que los archivos MP4 se copian directamente sin recodificación.

Si el vídeo ya existe en el directorio de salida, se omite su procesamiento.

Durante la ejecución se muestra el progreso de la conversión y, al finalizar,
se presenta un resumen con el número de vídeos convertidos, copiados, omitidos,
errores y el tiempo total de procesamiento.
"""

from pathlib import Path
import subprocess
import shutil
import time


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT_DIR
    / "data"
    / "source_data"
    / "raw_data"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "source_data"
    / "processed_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCIONES
# ============================================================

def get_video_duration(video_path: Path) -> float:
    """
    Devuelve la duración del vídeo en segundos utilizando ffprobe.
    """

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def convert_mov_to_mp4(
    input_video: Path,
    output_video: Path,
) -> None:
    """
    Convierte un vídeo MOV a MP4 mostrando el porcentaje de progreso.
    """

    duration = get_video_duration(input_video)

    command = [
        "ffmpeg",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(output_video),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        universal_newlines=True,
    )

    last_percent = -1

    for line in process.stdout:

        line = line.strip()

        if line.startswith("out_time_ms="):

            milliseconds = int(line.split("=")[1])

            seconds = milliseconds / 1_000_000

            percent = min(
                100,
                int((seconds / duration) * 100),
            )

            if percent != last_percent:

                print(
                    f"\r   Conversión: {percent:3d} %",
                    end="",
                )

                last_percent = percent

    process.wait()

    print("\r   Conversión: 100 %")


# ============================================================
# BÚSQUEDA DE VÍDEOS
# ============================================================

video_extensions = {".mov", ".mp4"}

videos = sorted(
    [
        file
        for file in INPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in video_extensions
    ]
)

total = len(videos)

converted = 0
copied = 0
skipped = 0
errors = 0

start_time = time.time()

print("=" * 60)
print("VIDEO PREPROCESSING")
print("=" * 60)
print(f"Vídeos encontrados: {total}")
print()


# ============================================================
# PROCESAMIENTO
# ============================================================

for index, input_video in enumerate(videos, start=1):

    output_video = OUTPUT_DIR / f"{input_video.stem}.mp4"

    print(f"[{index}/{total}] {input_video.name}")

    # --------------------------------------------------------
    # COMPROBAR SI YA EXISTE
    # --------------------------------------------------------

    if output_video.exists():

        skipped += 1

        print("   ✓ Ya existe -> Skip\n")

        continue

    # --------------------------------------------------------
    # PROCESAR VÍDEO
    # --------------------------------------------------------

    try:

        if input_video.suffix.lower() == ".mp4":

            print("   Copiando MP4...")

            shutil.copy2(
                input_video,
                output_video,
            )

            copied += 1

            print("   ✓ Copiado\n")

        else:

            print("   Convirtiendo MOV -> MP4...")

            convert_mov_to_mp4(
                input_video,
                output_video,
            )

            converted += 1

            print("   ✓ Conversión finalizada\n")

    except Exception as e:

        errors += 1

        print(f"   ✗ ERROR: {e}\n")


# ============================================================
# RESUMEN
# ============================================================

elapsed = time.time() - start_time

minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print("=" * 60)
print("RESUMEN")
print("=" * 60)

print(f"Vídeos encontrados : {total}")
print(f"Convertidos        : {converted}")
print(f"Copiados           : {copied}")
print(f"Saltados           : {skipped}")
print(f"Errores            : {errors}")
print(f"Tiempo total       : {minutes} min {seconds} s")

print("=" * 60)