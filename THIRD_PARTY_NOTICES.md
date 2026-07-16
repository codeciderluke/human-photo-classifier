# Third-Party Notices

Human Photo Classifier is licensed under **AGPL-3.0** (see [LICENSE](LICENSE)).
It uses the third-party components listed below, each under its own license.

## Why AGPL-3.0

This project depends on **Ultralytics YOLO**, which is licensed under
**AGPL-3.0**. AGPL-3.0 is a strong copyleft license: any distributed or
network-served software that includes it must also be released under
AGPL-3.0 with source available. To comply, this project as a whole is
released under AGPL-3.0. (For a non-copyleft/commercial use, Ultralytics
offers a separate Enterprise License.)

The GUI uses **PyQt5**, which is licensed under **GPL-3.0**. AGPL-3.0 is
compatible with GPL-3.0, so the combined work is distributed under AGPL-3.0.

## Dependency licenses

| Component | License | Notes |
|-----------|---------|-------|
| Ultralytics YOLO (`ultralytics`) | AGPL-3.0 | Drives the AGPL-3.0 choice for this project |
| PyQt5 | GPL-3.0 | GUI framework (commercial license also available from Riverbank) |
| InsightFace (`insightface`) code | MIT | See model note below |
| PyTorch (`torch`) | BSD-3-Clause | |
| ONNX Runtime (`onnxruntime`) | MIT | |
| OpenCV (`opencv-python`) | Apache-2.0 | |
| Pillow | HPND | |
| NumPy | BSD-3-Clause | |
| SciPy | BSD-3-Clause | |

## Model weights (downloaded at first run, not bundled)

- **YOLO weights** (e.g. `yolo11n.pt`) are provided by Ultralytics under
  **AGPL-3.0**.
- **InsightFace models** (e.g. `buffalo_l`) are provided by InsightFace for
  **non-commercial / academic research use only**. Do **not** use this
  application's face/gender features for commercial purposes with these
  models. For commercial use, replace them with appropriately licensed models.

## Commercial use

Because of the AGPL-3.0 (Ultralytics), GPL-3.0 (PyQt5), and the
non-commercial InsightFace model terms, this project **is not suitable for
proprietary or commercial distribution as-is**. To make it commercial-ready
you would need to obtain an Ultralytics Enterprise License, replace or
commercially license PyQt5 (e.g. move to PySide6 / LGPL), and swap in
commercially licensed face/gender models.

This file is informational and is not legal advice.
