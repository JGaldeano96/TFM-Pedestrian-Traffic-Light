"""
Cuenta el número de fotogramas extraídos para cada vídeo.

El script recorre las carpetas de fotogramas generadas durante el proceso de extracción,
cuenta las imágenes almacenadas en cada una y muestra un resumen con el número de vídeos
procesados, el total de imágenes y la media de fotogramas por vídeo.
"""


from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

FRAMES_DIR = (
    ROOT_DIR
    / "data"
    / "source_data"
    / "frames"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ============================================================
# CONTADOR
# ============================================================

total_images = 0
total_videos = 0

print("=" * 60)
print("FRAME COUNTER")
print("=" * 60)

for folder in sorted(FRAMES_DIR.iterdir()):

    if not folder.is_dir():
        continue

    num_images = sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"{folder.name}: {num_images} imágenes")

    total_images += num_images
    total_videos += 1

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(f"Vídeos procesados : {total_videos}")
print(f"Total de imágenes : {total_images}")

if total_videos > 0:
    print(f"Media por vídeo   : {total_images / total_videos:.2f}")
    
    print("=" * 60)
