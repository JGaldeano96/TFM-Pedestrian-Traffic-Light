# Pedestrian Traffic Light Detection and Classification

Deep Learning system for the **detection and classification of pedestrian traffic lights** in urban environments, designed as a technological aid for people with visual impairment.

The project combines an object detector based on **YOLO** with a dedicated **CNN classifier** to identify the state of each detected pedestrian traffic light.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
  - [Stage 1 — Object Detection](#stage-1--object-detection)
  - [Stage 2 — State Classification](#stage-2--state-classification)
- [Models](#models)
  - [YOLO Detector](#yolo-detector)
  - [Traffic Light Classifier](#traffic-light-classifier)
- [Data Augmentation](#data-augmentation)
- [Dataset](#dataset)
- [Dataset Generation Pipeline](#dataset-generation-pipeline)
- [YOLO Dataset](#yolo-dataset)
- [Classifier Dataset](#classifier-dataset)
- [Independent Test Sets](#independent-test-sets)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Training](#training)
- [Evaluation](#evaluation)
- [Threshold Analysis](#threshold-analysis)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Installation](#installation)
- [Streamlit Demo](#streamlit-demo)
- [Dataset Generation](#dataset-generation)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [End-to-End Evaluation](#end-to-end-evaluation)
- [Future Work](#future-work)
- [Ethical and Safety Considerations](#ethical-and-safety-considerations)
- [Technologies](#technologies)
- [Author](#author)
- [License](#license)

---

# Overview

Pedestrian traffic lights are a critical element when navigating urban environments. For people with visual impairment, determining whether a pedestrian crossing is currently safe can be difficult or impossible without additional assistance.

This project explores a computer vision system capable of:

1. Detecting pedestrian traffic lights in images.
2. Localizing each traffic light using bounding boxes.
3. Extracting the detected traffic light from the image.
4. Classifying its current state.
5. Providing the information required by a future assistive application.

The system is designed around a **two-stage architecture**:

**Input Image / Video**

↓

**YOLO Detector**

↓

**Bounding Box**

↓

**Traffic Light Crop**

↓

**CNN Classifier**

↓

**Red / Green**

The detector and classifier are trained independently, allowing each component to be optimized for its specific task.

---

# Objectives

The main objective of the project is to develop and evaluate a computer vision pipeline capable of recognizing pedestrian traffic lights and determining their state.

The specific objectives are:

- Develop a dataset containing pedestrian traffic lights captured from a first-person perspective.
- Annotate pedestrian traffic lights using bounding boxes.
- Train an object detection model using YOLO.
- Train a convolutional neural network for traffic light state classification.
- Evaluate the influence of image resolution on object detection.
- Analyze classification errors using confusion matrices and different decision thresholds.
- Analyze model interpretability using **Grad-CAM++**.
- Evaluate the final models on independent test datasets.
- Prepare the models for potential deployment on mobile or embedded devices using **TensorFlow Lite**.

---

# System Architecture

The complete system consists of two independent Deep Learning models.

## Stage 1 — Object Detection

The first model receives the complete image:

**Image**

↓

**YOLO**

↓

**Bounding Box + Confidence + Class**

The detector identifies pedestrian traffic lights regardless of their current state.

The detection task therefore has a single class:

**`pedestrian_traffic_light`**

---

## Stage 2 — State Classification

The detected bounding box is then cropped from the original image.

**YOLO Bounding Box**

↓

**Crop**

↓

**Resize to 128 × 128**

↓

**CNN**

↓

**Red / Green**

The classifier is a binary classifier.

The original annotation states are mapped as follows:

| Original state | Classifier class |
|---|---|
| `Red` | `Red` |
| `Green` | `Green` |
| `Off` | `Red` |

The `Off` state is intentionally grouped with `Red` for the current binary classification formulation.

---

# Models

## YOLO Detector

The object detector is based on the YOLO architecture and is trained using the Ultralytics implementation.

Different model sizes and input resolutions are evaluated.

Experiments include:

- YOLO Nano
- YOLO Small
- 640 × 640
- 800 × 800
- 960 × 960
- 1088 × 1088

The final detector is intended to provide a good balance between:

- Detection accuracy
- Inference speed
- Computational cost
- Potential mobile deployment

The detector predicts a single class:

**`pedestrian_traffic_light`**

---

## Traffic Light Classifier

The classifier is implemented using TensorFlow/Keras.

The input resolution is:

**128 × 128 × 3**

The architecture is based on a custom convolutional neural network containing:

- Image rescaling
- Convolutional layers
- SiLU activation
- Max pooling
- Global Average Pooling
- Fully connected layers
- Sigmoid output

The output represents the probability of the positive class:

**P(Green)**

The final classification is determined using a configurable decision threshold.

For example:

**P(Green) ≥ 0.5 → Green**

**P(Green) < 0.5 → Red**

However, the threshold is not necessarily fixed at `0.5`. Threshold analysis is performed to study the trade-off between false positives and false negatives.

---

# Data Augmentation

Data augmentation is applied to the classifier during training to improve generalization.

The transformations are designed to reproduce realistic variations that may occur when capturing pedestrian traffic lights.

Examples include:

- Small rotations
- Horizontal translations
- Small zoom variations
- Contrast variations
- Brightness variations

Large rotations or vertical flips are deliberately avoided.

A pedestrian traffic light will not realistically appear upside down in the target application, so transformations that introduce physically implausible samples are avoided.

---

# Dataset

The dataset was created using images captured from a pedestrian's point of view in urban environments.

The annotation process was performed using **Label Studio**.

Each annotated traffic light contains:

- Image
- Bounding box
- Traffic light state
- Video identifier
- Object identifier

The annotation states are:

- `Red`
- `Green`
- `Off`

---

# Dataset Generation Pipeline

The datasets are not manually copied into the final training structure.

Instead, Python scripts generate the datasets automatically from the Label Studio exports.

The general pipeline is:

**Raw Images**

↓

**Label Studio**

↓

**JSON Export**

↓

**Dataset Processing**

↓

**YOLO Dataset + Classifier Dataset**

This makes the dataset generation process reproducible.

---

# YOLO Dataset

The YOLO training dataset is generated from:

`data/annotation_exports/dataset_vX.json`

and the corresponding video-level split:

`data/splits/split_vX.json`

The generated dataset follows the standard YOLO structure:

`data/datasets/yolo/`

- `images/`
  - `train/`
  - `val/`
- `labels/`
  - `train/`
  - `val/`
- `dataset.yaml`

The split is performed at **video level**, rather than at individual image level.

This is important because consecutive frames from the same video are highly correlated. Splitting individual frames could therefore introduce data leakage between training and validation sets.

---

# Classifier Dataset

The classifier dataset is generated by:

1. Reading the Label Studio annotations.
2. Loading the original image.
3. Extracting each annotated bounding box.
4. Cropping the traffic light.
5. Resizing the crop to `128 × 128`.
6. Mapping the original state to the classifier class.
7. Assigning the crop to train or validation according to the video-level split.

The resulting structure is:

`data/datasets/classifier/`

- `train/`
  - `Red/`
  - `Green/`
- `val/`
  - `Red/`
  - `Green/`

---

# Independent Test Sets

A completely independent test set is maintained separately from the training and validation data.

Two independent test datasets are used:

`data/annotation_exports/`

- `test_diurn.json`
- `test_nocturn.json`

They represent two different environmental conditions:

- **Diurnal** — daytime conditions
- **Nocturnal** — nighttime conditions

The datasets are intentionally kept separate.

The corresponding YOLO test datasets are generated under:

`data/datasets/yolo_test/`

- `diurn/`
- `nocturn/`

The classifier will also use independent daytime and nighttime test datasets.

These datasets are not used during training or model selection.

This allows the final evaluation to provide a more realistic estimate of how well the models generalize to previously unseen data.

---

# Data Leakage Prevention

A major consideration in this project is preventing data leakage.

Images extracted from the same video are highly similar. Therefore, randomly splitting individual frames could result in nearly identical images appearing in both training and validation sets.

To avoid this, the dataset split is performed using the **video identifier**.

For example:

**Video 000001**

→ frame 001  
→ frame 002  
→ frame 003  

→ **TRAIN**

while:

**Video 000002**

→ frame 001  
→ frame 002  
→ frame 003  

→ **VALIDATION**

No frames from the same video should appear in both subsets.

The independent test datasets are kept completely separate from the training and validation pipeline.

---

# Training

## YOLO

YOLO models are trained using the Ultralytics framework.

Typical training parameters include:

- Epochs
- Image size
- Batch size
- Patience
- Automatic optimizer selection
- Data augmentation
- Multi-scale training configuration

Different input resolutions and model sizes are evaluated to determine the best accuracy/efficiency trade-off.

---

## Classifier

The classifier is trained using TensorFlow/Keras.

The training process uses:

- AdamW optimizer
- Binary Cross-Entropy loss
- ReduceLROnPlateau
- EarlyStopping
- ModelCheckpoint
- Data augmentation

Example optimizer configuration:

`AdamW(learning_rate=3e-4)`

The learning rate is dynamically reduced when validation performance stops improving.

The training process uses `ReduceLROnPlateau` to improve convergence and allow the model to refine its weights with progressively smaller learning rates.

Early stopping is used to prevent unnecessary training.

The best model checkpoint is saved according to validation performance.

---

# Evaluation

The evaluation is not limited to accuracy.

## Object Detection

The YOLO models are evaluated using standard object detection metrics, including:

- Precision
- Recall
- mAP@50
- mAP@50:95

Different input resolutions and model sizes are compared.

---

## Classification

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- False Positives
- False Negatives
- Decision threshold analysis

Particular attention is given to the distinction between false positives and false negatives.

### False Positive

A real **Red** traffic light is classified as **Green**.

**Real: Red**

**Predicted: Green**

This is particularly important because it could incorrectly indicate that crossing is safe.

### False Negative

A real **Green** traffic light is classified as **Red**.

**Real: Green**

**Predicted: Red**

This is conservative from a safety perspective but can reduce the usefulness of the system.

---

# Threshold Analysis

The classifier outputs:

**P(Green)**

Instead of assuming that `0.5` is always the optimal threshold, different thresholds are evaluated.

For example:

- `0.10`
- `0.20`
- `0.30`
- `0.40`
- `0.50`
- `0.60`
- `0.70`
- `0.80`
- `0.90`

The evolution of false positives and false negatives is analyzed across the complete `[0, 1]` interval.

This makes it possible to select a threshold according to the requirements of the final application.

The threshold can therefore be selected according to the desired balance between:

**False Green predictions**

and

**False Red predictions**

---

# Explainability

The classifier is also analyzed using **Grad-CAM++**.

Grad-CAM++ is used to visualize which regions of the input image contribute most strongly to the model's prediction.

This is particularly useful for analyzing incorrect predictions.

The analysis can help determine whether the classifier is actually focusing on the traffic light or exploiting unintended visual cues such as:

- Background
- Buildings
- Sky
- Poles
- Reflections
- Image borders
- Other objects

Grad-CAM++ is especially valuable when investigating false positives and false negatives.

---

# Project Structure

The main project structure is:

`tfm/`

- `data/`
  - `annotation/`
  - `annotation_exports/`
    - `dataset_v1.json`
    - `dataset_v2.json`
    - `test_diurn.json`
    - `test_nocturn.json`
  - `datasets/`
    - `yolo/`
    - `yolo_test/`
      - `diurn/`
      - `nocturn/`
    - `classifier/`
  - `splits/`
    - `split_v1.json`
    - `split_v2.json`

- `scripts/`
  - `dataset/`
    - `generate_yolo_dataset.py`
    - `generate_yolo_test_dataset.py`
    - `generate_classifier_dataset.py`
    - `generate_classifier_test_dataset.py`
  - `utils/`
    - `dataset_functions.py`
  - `streamlit/`
    - `launcher.py`

- `results/`
  - `classifier/`
  - `yolo/`

- `models/`
  - `classifier/dataset_v2/classifier_v2.onnx`
  - `yolo/dataset_v2/`

- `tfm_demo/`
  - `config.py`
  - `model_loading.py`
  - `inference.py`
  - `video_processing.py`

- `app.py`

- `notebooks/`

- `requirements.txt`

- `README.md`

---

# Environment

The project uses Linux/WSL2 for the Deep Learning environment.

The main technologies include:

- Python 3.10
- TensorFlow
- Keras
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Label Studio
- tf-keras-vis
- CUDA
- WSL2

GPU acceleration is used during model training when available.

The main GPU used during development is:

**NVIDIA GeForce RTX 3070 Laptop GPU**

---

# Installation

Clone the repository:

`git clone <repository-url>`

Move into the project directory:

`cd tfm`

Create the required Python environments according to the project requirements.

For example:

`python3 -m venv env`

Activate the environment:

`source env/bin/activate`

Install the required dependencies:

`pip install -r requirements.txt`

The project can use separate environments for different stages of the pipeline when required.

---

# Streamlit Demo

The repository includes a functional video demo for the thesis presentation. It runs the complete pipeline without TensorFlow:

**MP4 → visual adjustments → YOLO dataset V2 → largest valid box → ONNX Runtime classifier V2 → annotated MP4**

## Demo installation

The target is Python 3.10 in the existing YOLO environment. From the repository root:

```bash
source /home/jgaldeano/envs/yolo/bin/activate
pip uninstall -y onnxruntime
pip install -r requirements.txt
```

`requirements.txt` contains only the direct runtime dependencies of the demo. It deliberately does not include TensorFlow or Keras; classification is performed exclusively with the CUDA build of ONNX Runtime. The CPU package is removed first to prevent both ONNX Runtime distributions from sharing the same Python module.

Both stages prefer the first NVIDIA GPU. YOLO receives `device="cuda:0"` explicitly and the classifier requests `CUDAExecutionProvider`. If CUDA is unavailable or blocked by WSL/Windows, the application remains usable with CPU and shows that fallback prominently instead of reporting GPU execution incorrectly.

## Expected model layout

Model paths are centralized in `tfm_demo/config.py`. The application expects only dataset V2 artifacts:

```text
models/
├── classifier/
│   └── dataset_v2/
│       └── classifier_v2.onnx
└── yolo/
    └── dataset_v2/
        ├── yolo26n_640_dataset_v2_best.pt
        ├── yolo26n_800_dataset_v2_best.pt
        ├── yolo26n_960_dataset_v2_best.pt
        ├── yolo26n_1088_dataset_v2_best.pt
        ├── yolo26s_640_dataset_v2_best.pt
        ├── yolo26s_800_dataset_v2_best.pt
        ├── yolo26s_960_dataset_v2_best.pt
        └── yolo26s_1088_dataset_v2_best.pt
```

These files are safe copies of the corresponding `best.pt` and `classifier.onnx` artifacts under `results/`; the originals are not moved. Model binaries remain excluded by `.gitignore`, so they must be provisioned locally when the repository is cloned on another machine.

## Starting the application

Run the following exact command from the repository root:

```bash
source /home/jgaldeano/envs/yolo/bin/activate
streamlit run app.py
```

The repository also provides a launcher suitable for the IDE **Run** action:

```bash
source /home/jgaldeano/envs/yolo/bin/activate
python scripts/streamlit/launcher.py
```

Any additional Streamlit arguments are forwarded, for example `python scripts/streamlit/launcher.py --server.port 8502`.

Upload an MP4, select the YOLO architecture and trained input resolution, choose the thresholds and visual adjustments, and press **Procesar vídeo**. The application shows the transformed and annotated input during processing, plays the final result and exposes an MP4 download together with detection and performance metrics.

## Controls and decision convention

- YOLO architecture: YOLO26n or YOLO26s, always trained on dataset V2.
- YOLO resolution: 640, 800, 960 or 1088 pixels; each option loads its own V2 `best.pt`.
- Closest traffic light: after clipping boxes to the image, only the valid box with the largest area is cropped, classified and drawn. Confidence breaks an exact area tie.
- Brightness: -100 to +100.
- Contrast: 0.50× to 1.50×.
- Saturation: 0.00× to 2.00×.
- YOLO confidence threshold and classifier decision threshold.
- Final playback speed: 0.50× to 2.00×. At the default 1.00× every frame is retained and the original FPS are preserved, so a 60 FPS recording remains approximately 60 FPS.
- Optional live preview, disabled by default to reduce interface overhead. It shows individual frames at inference speed, not real-time playback. Its frequency can be set to every 1, 2, 5, 10, 20 or 30 processed frames and never changes the generated video.
- Output audio: disabled by default for presentation-ready downloads. It can be enabled to preserve the source audio, synchronized with the selected playback speed.
- Output resolution: original, Full HD, HD or compact. The image is fitted inside the selected limit without cropping, distortion or upscaling; HD is the default for easier display on laptops and projectors.
- Player size: compact (360 px), medium (480 px), large (720 px) or the full available width. Its height is also capped at 70% of the browser window so portrait videos fit on screen. This only changes the Streamlit layout, not the downloaded MP4.
- Optional temporal EMA compares only the current closest box with the previous frame by IoU. It has no identities and is not a tracking system. Labels always display raw `P(Green)` and, when enabled, the separate EMA value used for the decision.

The mandatory binary convention is:

- `Red = 0`
- `Green = 1`
- classifier output = `P(Green)`
- `P(Green) < threshold` → `Red`
- `P(Green) >= threshold` → `Green`

The exported ONNX declares a float32 NHWC input `[N, 128, 128, 3]`. Crops are converted from OpenCV BGR to RGB and sent in the original 0–255 range because the trained `Rescaling(1/255)` layer is embedded in the ONNX graph. The output `green_probability [N, 1]` is checked as one finite probability per crop.

## FFmpeg and video compatibility

OpenCV first writes an intermediate MP4 while preserving aspect ratio. The selected resolution preset only downsizes: for example, a vertical 1080×1920 source fitted to HD becomes 404×720. At 1.00× it preserves the original FPS and duration. When another playback speed is selected, every frame is still written and the output FPS are multiplied by that factor. When FFmpeg with `libx264` is available, the final file is converted to H.264/yuv420p with web fast-start metadata. Audio is omitted by default; when explicitly enabled, the source track is encoded as AAC and synchronized with the selected speed.

Verify FFmpeg with:

```bash
ffmpeg -version
```

If FFmpeg is absent or H.264 conversion fails, the app returns the OpenCV MP4V file and displays a warning. MP4V playback depends on the browser.

## Known demo limitations

- Processing is offline, not real time, and high-resolution YOLO variants can be slow without a GPU.
- The live preview cannot run at 60 FPS unless the complete inference pipeline itself reaches 60 FPS. It is only a progress visualization; playback speed is evaluated in the final video player.
- Temporal stabilization only compares consecutive largest boxes by IoU; it does not track identities. Fast camera movement can reset the EMA.
- There is no concept of unique traffic lights. Counts represent frames in which a closest valid detection was selected and classified.
- Uploaded and generated videos are kept in memory by the Streamlit session; very long files require substantial RAM.
- The prototype is for academic demonstration and must not be used as a safety-critical crossing aid.

---

# Dataset Generation

## Generate YOLO Training Dataset

The YOLO dataset can be generated using:

`python scripts/dataset/generate_yolo_dataset.py`

This reads the Label Studio export and generates:

`data/datasets/yolo/`

---

## Generate Independent YOLO Test Dataset

The independent daytime and nighttime test datasets can be generated using:

`python scripts/dataset/generate_yolo_test_dataset.py`

The resulting structure is:

`data/datasets/yolo_test/`

- `diurn/`
- `nocturn/`

---

## Generate Classifier Training Dataset

The classifier dataset can be generated using:

`python scripts/dataset/generate_classifier_dataset.py`

The generated structure is:

`data/datasets/classifier/`

- `train/`
  - `Red/`
  - `Green/`
- `val/`
  - `Red/`
  - `Green/`

---

## Generate Independent Classifier Test Dataset

The independent classifier datasets are generated from the dedicated Label Studio exports:

- `test_diurn.json`
- `test_nocturn.json`

The generated datasets maintain the same separation between daytime and nighttime conditions.

---

# Reproducibility

The project attempts to maintain reproducibility through:

- Fixed random seeds
- Versioned datasets
- Versioned train/validation splits
- Automated dataset generation
- Explicit model configurations
- Saved model checkpoints
- Independent test datasets

The main random seed used during development is:

`SEED = 42`

Dataset versions are explicitly defined, for example:

`DATASET_VERSION = "v2"`

This allows different dataset versions to be regenerated and compared.

---

# Results

The project evaluates the complete pipeline at multiple levels.

## Detection

YOLO experiments compare:

**Model size × Input resolution × Detection performance × Inference cost**

The goal is not simply to maximize mAP, but to find an appropriate model for the eventual assistive application.

---

## Classification

The classifier is evaluated independently from YOLO using ground-truth crops and, where appropriate, using detector-generated crops.

This distinction allows the two sources of error to be studied separately:

**Detection error + Classification error = End-to-end system error**

This is important because a classification error does not necessarily mean that the CNN itself is responsible. An incorrect or poorly localized bounding box produced by YOLO can also affect the final classification.

---

# End-to-End Evaluation

The final system can be evaluated as:

**Input image**

↓

**YOLO**

↓

**Traffic light detection**

↓

**Bounding box crop**

↓

**CNN Classifier**

↓

**Red / Green**

An incorrect final prediction can originate from several different sources:

1. The detector fails to detect the traffic light.
2. The detector produces an inaccurate bounding box.
3. The classifier incorrectly classifies a correct crop.
4. The decision threshold produces an inappropriate classification.

The independent test datasets allow the complete pipeline to be evaluated under previously unseen conditions.

---

# Future Work

Potential future developments include:

- End-to-end real-time inference.
- Object tracking between consecutive frames.
- Temporal smoothing of predictions.
- Confidence-based decision logic.
- Integration with a mobile application.
- Conversion of the classifier to TensorFlow Lite.
- Quantization and model optimization.
- Optimization of YOLO for edge/mobile inference.
- Audio feedback for users with visual impairment.
- Evaluation in additional environmental conditions.
- Larger and more diverse independent test datasets.

A particularly relevant future improvement is temporal consistency.

Instead of making an independent decision for every frame:

**Frame 1 → Green**

**Frame 2 → Green**

**Frame 3 → Red**

**Frame 4 → Green**

**Frame 5 → Green**

the system could use temporal information to reduce unstable predictions.

---

# Ethical and Safety Considerations

This project is an experimental research prototype.

The predictions of the system should **not be considered a reliable replacement for human perception or established mobility assistance systems**.

In particular, an incorrect `Green` prediction could potentially lead to unsafe behavior.

For this reason, the evaluation places particular emphasis on:

- False Green predictions
- False Red predictions
- Decision threshold selection
- Independent testing
- Nighttime performance
- Generalization to unseen environments

---

# Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | CNN classifier |
| PyTorch | YOLO training backend |
| Ultralytics YOLO | Object detection |
| OpenCV | Image processing |
| NumPy | Numerical computation |
| Pandas | Dataset processing |
| Matplotlib | Visualization |
| Label Studio | Image annotation |
| tf-keras-vis | Grad-CAM++ explainability |
| CUDA | GPU acceleration |
| WSL2 | Linux development environment |
| TensorFlow Lite | Planned mobile deployment |

---

# Author

**Javier Galdeano**

Data Scientist

Master's Degree in Data Science, Big Data and Artificial Intelligence

This project is developed as part of a Master's Thesis focused on the application of Deep Learning and Computer Vision to assistive technology for people with visual impairment.

---

# License

This project is intended primarily for academic and research purposes.

If a specific open-source license is added to the repository, the terms of that license will apply to the source code and associated materials.
