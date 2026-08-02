from pathlib import Path
import json
from collections import Counter, defaultdict
from statistics import mean, median, stdev

# Ruta al JSON exportado por Label Studio
JSON_PATH = Path("/home/jgaldeano/tfm/data/annotation_exports/dataset_v2.json")


def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_images = len(dataset)
    images_without_annotations = 0
    total_objects = 0

    # Contadores
    class_counter = Counter()
    objects_per_image = Counter()
    video_counter = Counter()
    video_class_counter = defaultdict(Counter)
    resolutions = Counter()

    # Estadísticas de las bounding boxes
    widths = []
    heights = []
    areas = []

    for task in dataset:

        image_path = task["data"]["image"]

        # Extraer el identificador del vídeo
        parts = image_path.split("/")
        video_id = parts[-2] if len(parts) >= 2 else "unknown"

        video_counter[video_id] += 1

        # Imágenes sin anotaciones
        if not task["annotations"]:
            images_without_annotations += 1
            continue

        annotation = task["annotations"][0]

        if not annotation["result"]:
            images_without_annotations += 1
            continue

        image_objects = 0

        for result in annotation["result"]:

            if result["type"] == "rectanglelabels":

                w = result["value"]["width"]
                h = result["value"]["height"]

                widths.append(w)
                heights.append(h)
                areas.append(w * h)

                resolutions[
                    (
                        result["original_width"],
                        result["original_height"]
                    )
                ] += 1

                image_objects += 1

            elif result["type"] == "choices":

                state = result["value"]["choices"][0]

                class_counter[state] += 1
                video_class_counter[video_id][state] += 1
                total_objects += 1

        objects_per_image[image_objects] += 1

    # ==========================================================
    # RESUMEN
    # ==========================================================

    print_section("RESUMEN DEL DATASET")

    print(f"Imágenes totales          : {total_images}")
    print(f"Imágenes anotadas         : {total_images-images_without_annotations}")
    print(f"Imágenes sin anotaciones  : {images_without_annotations}")
    print(f"Objetos etiquetados       : {total_objects}")

    # ==========================================================
    # CLASES
    # ==========================================================

    print_section("CLASES")

    for cls, count in sorted(class_counter.items()):
        porcentaje = count / total_objects * 100
        print(f"{cls:<10}: {count:5d} ({porcentaje:6.2f} %)")

    # ==========================================================
    # OBJETOS POR IMAGEN
    # ==========================================================

    print_section("OBJETOS POR IMAGEN")

    for n_obj, n_images in sorted(objects_per_image.items()):
        porcentaje = n_images / total_images * 100
        print(f"{n_obj} objeto(s): {n_images:4d} imágenes ({porcentaje:5.2f} %)")

    # ==========================================================
    # BOUNDING BOXES
    # ==========================================================

    print_section("ESTADÍSTICAS DE BOUNDING BOXES (%)")

    print("ANCHURA")
    print(f"  Media      : {mean(widths):.2f}")
    print(f"  Mediana    : {median(widths):.2f}")
    print(f"  Mínima     : {min(widths):.2f}")
    print(f"  Máxima     : {max(widths):.2f}")
    print(f"  Desv. típ. : {stdev(widths):.2f}")

    print()

    print("ALTURA")
    print(f"  Media      : {mean(heights):.2f}")
    print(f"  Mediana    : {median(heights):.2f}")
    print(f"  Mínima     : {min(heights):.2f}")
    print(f"  Máxima     : {max(heights):.2f}")
    print(f"  Desv. típ. : {stdev(heights):.2f}")

    print()

    print("ÁREA")
    print(f"  Media      : {mean(areas):.2f}")
    print(f"  Mediana    : {median(areas):.2f}")
    print(f"  Mínima     : {min(areas):.2f}")
    print(f"  Máxima     : {max(areas):.2f}")
    print(f"  Desv. típ. : {stdev(areas):.2f}")

    # ==========================================================
    # RESOLUCIONES
    # ==========================================================

    print_section("RESOLUCIONES")

    for resolution, count in resolutions.items():
        print(f"{resolution[0]}x{resolution[1]} -> {count} bounding boxes")

    # ==========================================================
    # IMÁGENES POR VÍDEO
    # ==========================================================

    print_section("IMÁGENES POR VÍDEO")

    for video, n in sorted(video_counter.items()):
        print(f"{video}: {n} imágenes")

    # ==========================================================
    # OBJETOS POR VÍDEO
    # ==========================================================

    print_section("OBJETOS POR VÍDEO")

    for video in sorted(video_counter):

        total_video = sum(video_class_counter[video].values())

        print(
            f"{video}: "
            f"{total_video:4d} objetos "
            f"(Red={video_class_counter[video]['Red']}, "
            f"Green={video_class_counter[video]['Green']}, "
            f"Off={video_class_counter[video]['Off']})"
        )

    # ==========================================================
    # ESTADÍSTICAS GENERALES
    # ==========================================================

    print_section("ESTADÍSTICAS GENERALES")

    print(
        f"Media de objetos por imagen anotada: "
        f"{total_objects/(total_images-images_without_annotations):.2f}"
    )


if __name__ == "__main__":
    main()