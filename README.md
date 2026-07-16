# Human Photo Classifier

A desktop application that automatically selects and copies **only the photos that contain people**, powered by YOLO.
The person-detection logic is fully decoupled from the GUI, so it can be reused unchanged across CLI, batch jobs, server APIs, tests, and more.

![Human Photo Classifier GUI](assets/screenshot.png)

## Download

Pre-built Windows executables (GUI and CLI) are attached to each
[GitHub Release](../../releases). No Python installation is required — download,
unzip, and run `HumanPhotoClassifier.exe`. Detection models are downloaded
automatically on first run (internet required once).

To run from source instead, see [Installation](#installation).

## Features

- **Layered architecture**: GUI (input/display), filters (detection), and services (files) are independent packages.
- **Swappable detection engine**: built on the `ImagePersonFilter` abstract interface.
- **Load models once**: models are not rebuilt for each image.
- **Asynchronous processing**: detection runs on a separate thread, so the UI never freezes.
- **Cancellable jobs**: processing can be stopped safely at any time.
- **Corrupt-image handling**: fully corrupt files are excluded; partially damaged person photos are sorted into `others/damaged`.
- **Optional face and gender recognition**: uses InsightFace to detect the presence of a face and classify gender, then organizes results into category folders automatically.
- **Modern dark-theme UI** (optimized for Full HD).

## Classification Folder Rules

When the classification options are enabled, subfolders are created automatically inside the output folder.

| Face recognition | Gender recognition | Subfolders created |
|:--:|:--:|--|
| ✗ | ✗ | None (person photos only, original structure preserved) |
| ✓ | ✗ | `with_face` / `no_face` / `unclassified` / `others` |
| ✗ | ✓ | `male` / `female` / `unclassified` / `others` |
| ✓ | ✓ | `with_face/male` · `with_face/female` · `no_face` · `unclassified` · `others` |

- **Face and gender classification apply to person photos only.**
- **Non-person photos are copied to `others`.** (When the classification options are off, only person photos are extracted and nothing else is copied.)
- **Person photos that are partially damaged are sorted into `others/damaged`.** Images too corrupt to open at all are excluded and not copied.
- Gender can only be determined when a face is present, so enabling gender recognition also enables face recognition.
- When multiple faces are present, the photo is classified by the gender of the **largest (main subject) face**.
- Person photos whose face analysis fails are sent to `unclassified`.

### GPU Acceleration

- **Person detection (YOLO)**: `cpu` / `cuda` selectable in the app's device setting.
- **Face and gender analysis (InsightFace)**: **uses the GPU automatically** when `onnxruntime-gpu` is available, and falls back to CPU otherwise (logged).
- The CUDA 12 / cuDNN 9 runtime DLLs required for GPU are **reused from the bundled PyTorch (cu124) install**, so no separate CUDA Toolkit is needed.

## Project Structure

```text
humanClassifier/
├── main.py                     # GUI entry point
├── cli.py                      # Command-line entry point
├── core/                       # GUI-independent orchestration
│   └── pipeline.py             # Shared scan → detect → classify → copy pipeline
├── gui/                        # PyQt5 UI layer (no detection logic)
│   ├── main_window.py          # Folder selection, settings, progress, log, start/stop
│   ├── detection_worker.py     # QThread wrapper around core.pipeline
│   └── theme.py                # Dark-theme color palette + QSS
├── filters/                    # Detection layer (no GUI dependency)
│   ├── base.py                 # ImagePersonFilter abstract interface
│   ├── models.py               # PersonDetection / PersonFilterResult / FaceInfo …
│   ├── person_filter.py        # YoloPersonFilter + dedicated exceptions
│   ├── face_analyzer.py        # InsightFaceAnalyzer (face detection + gender)
│   └── image_integrity.py      # Image integrity checks (corrupt/partial damage)
├── services/                   # File-handling layer
│   ├── image_scanner.py        # Collect images by extension / subfolder
│   ├── file_copy_service.py    # Structure-preserving, dedup-safe copy (category folders)
│   └── categorizer.py          # Map face/gender → category subfolder path
├── assets/                     # App icon
├── tests/                      # pytest unit tests (no real inference)
├── requirements.txt
├── run.bat                     # Launch the GUI
├── cli.bat                     # Launch the CLI
└── test.bat                    # Run the tests
```

## Layer Responsibilities

| Layer | Responsible for | Does not do |
|-------|-----------------|-------------|
| `gui` | Folder selection, settings input, progress/log display, start/stop | Interpreting YOLO results |
| `filters` | Model loading, inference, person decision, result conversion, exception mapping | File copying, GUI signals |
| `services` | Image collection, folder creation, structure-preserving copy, deduplication | Detection |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. Select the **source folder** and the **output folder**.
2. Configure the model, confidence, image size, and device (CPU/GPU).
3. Click **Start**; only photos containing people are copied to the output folder.

> On first run, the YOLO and InsightFace models are downloaded automatically.

## Command-Line Interface (CLI)

The same pipeline is available without the GUI:

```bash
python cli.py SOURCE DESTINATION [options]

# Examples
python cli.py ./photos ./sorted
python cli.py ./photos ./sorted --device cuda --face --gender
python cli.py ./photos ./sorted --model yolo11s.pt --confidence 0.15 --quiet
```

On Windows you can also use `cli.bat SOURCE DESTINATION [options]`.

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | YOLO model file | `yolo11n.pt` |
| `--confidence` | Detection confidence (0–1) | `0.20` |
| `--image-size` | Inference image size | `960` |
| `--device` | `cpu`, `cuda`, or `cuda:0` | `cpu` |
| `--no-recursive` | Do not scan subfolders | off |
| `--face` | Sort person photos into `with_face` / `no_face` | off |
| `--gender` | Sort person photos by gender (`male` / `female`) | off |
| `-q`, `--quiet` | Show a progress bar instead of per-image lines | off |

Press `Ctrl+C` to stop gracefully; a summary is still printed. The exit code is `0` on success and `1` on a fatal error.

## Using the Filter Module Standalone

The filter can be reused without the GUI.

```python
from filters import YoloPersonFilter

person_filter = YoloPersonFilter(
    model_name="yolo11n.pt",
    confidence=0.20,
    image_size=960,
    device="cpu",
)

# Single image
if person_filter.contains_person("photo.jpg"):
    print("A person is present.")

# Detailed result
result = person_filter.analyze("photo.jpg")
for det in result.detections:
    print(det.confidence, det.bounding_box)

# Batch (supports a cancel callback, no PyQt signals required)
for r in person_filter.analyze_many(["a.jpg", "b.jpg"]):
    print(r.image_path, r.contains_person)
```

## Exception Hierarchy

```text
PersonFilterError
├── ModelLoadError      # Model file/library problem → abort the whole job
└── ImageAnalysisError  # Corrupt image / inference failure → skip that image only
```

Path and configuration errors are handled with the standard `FileNotFoundError` and `ValueError`.

## Tests

Tests run without downloading real models or performing inference (a fake `ultralytics` is injected).

```bash
python -m pytest
```

## Tuning Detection Accuracy

| Goal | Settings |
|------|----------|
| Default | `confidence=0.20`, `imgsz=960`, `yolo11n.pt` |
| Higher recall | `confidence=0.10–0.20`, `imgsz=1280+`, `yolo11s/m.pt` |

A low confidence threshold can increase false positives. If you need very high accuracy even for photos showing only part of a body, consider pose/segmentation models, fine-tuning, or a second-pass verification ensemble. Because the interface is decoupled, you can swap the engine by implementing `ImagePersonFilter`.

## License

Released under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see
[LICENSE](LICENSE). This is required because the project depends on
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and
[PyQt5](https://pypi.org/project/PyQt5/) (GPL-3.0).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for all dependency
licenses. **Note:** the InsightFace face/gender models are licensed for
non-commercial research use only, and the project as a whole is not suitable
for proprietary/commercial distribution as-is.
