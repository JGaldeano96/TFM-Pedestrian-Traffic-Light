from pathlib import Path
import subprocess
import time

# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_DIR = Path("/home/jgaldeano/tfm/data/source_data/processed_data")
OUTPUT_DIR = Path("/home/jgaldeano/tfm/data/source_data/frames")

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

    if frame_folder.exists():

        skipped += 1

        print("   ✓ Ya procesado -> Skip\n")

        continue

    frame_folder.mkdir(parents=True)

    output_pattern = frame_folder / f"{video_code}_frame%06d.jpg"

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

        num_frames = len(list(frame_folder.glob("*.jpg")))

        print(f"   ✓ Frames extraídos: {num_frames}\n")

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