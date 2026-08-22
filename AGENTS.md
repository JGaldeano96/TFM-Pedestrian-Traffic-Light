# Repository Guidelines

## Estructura del proyecto

El repositorio implementa una tubería de visión artificial en dos etapas: detección con YOLO y clasificación binaria con una CNN. Mantén cada cambio en su área:

- `scripts/dataset/`: generación, división y análisis de datasets.
- `scripts/preprocessing/`: conversión de vídeo y extracción de fotogramas.
- `scripts/training/`: entrenamiento y métricas de validación.
- `scripts/test/` y `scripts/utils/`: evaluación independiente y utilidades compartidas.
- `notebooks/`: exploración reproducible; el código reutilizable debe trasladarse a `scripts/`.
- `data/`: fuentes, anotaciones, particiones y datasets generados.
- `models/`, `results/`, `runs/` y `report/`: pesos, métricas y artefactos; evita versionar salidas grandes o regenerables.

## Instalación y comandos habituales

El entorno objetivo es Python 3.10 sobre Linux/WSL2, con GPU opcional.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python scripts/dataset/generate_yolo_dataset.py
python scripts/training/yolo26_training.py
python scripts/training/CNN_training.py
python scripts/test/generate_yolo_test_metrics.py
python scripts/test/generate_CNN_test_metrics.py
```

`requirements.txt` está actualmente vacío: documenta y fija allí cualquier dependencia nueva antes de esperar una instalación reproducible. Ejecuta los comandos desde la raíz, pues las rutas se resuelven respecto al repositorio.

## Estilo y convenciones

Usa Python con sangría de cuatro espacios, imports agrupados, `pathlib.Path` para rutas y docstrings en funciones públicas. Sigue `snake_case` para módulos, funciones y variables, `UPPER_SNAKE_CASE` para configuración (`ROOT_DIR`, `IMAGE_SIZE`) y nombres descriptivos para versiones (`dataset_v2`, `split_v2.json`). Conserva la convención crítica del clasificador: `Red = 0`, `Green = 1` y la salida representa `P(Green)`. No hay formateador configurado; limita cambios de formato ajenos a la tarea.

## Pruebas y validación

No existe una suite automatizada con `pytest`. Antes de enviar cambios, ejecuta el script afectado sobre una muestra y los generadores de métricas pertinentes. Comprueba datasets diurnos y nocturnos, ausencia de fuga entre `train`, `val` y `test`, y rutas de salida bajo `results/`. Nombra futuros tests como `test_<comportamiento>.py`.

## Commits y pull requests

El historial usa mensajes descriptivos en español, normalmente en infinitivo o imperativo (`Añadir...`, `Corregir...`, `Modificar...`). Haz commits acotados y explica cambios de datos o modelos. Cada PR debe incluir objetivo, comandos ejecutados, datasets/versiones y métricas comparativas; enlaza la incidencia y adjunta gráficas o capturas cuando cambien resultados visuales. No incluyas datos sensibles, vídeos originales ni pesos grandes sin aprobación explícita.
