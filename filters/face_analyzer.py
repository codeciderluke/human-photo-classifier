"""Face detection and gender estimation filter.

Uses InsightFace (``buffalo_l``) to detect faces in an image and estimate each
face's gender/age. Like the person filter (``person_filter``), it does not
depend on the GUI, and the model is loaded only once at instance creation.

Through the ``ImageFaceAnalyzer`` abstract interface, the face engine can later
be swapped (OpenCV DNN, MediaPipe, external API, etc.).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .models import GENDER_FEMALE, GENDER_MALE, FaceAnalysisResult, FaceInfo
from .person_filter import CorruptImageError

logger = logging.getLogger(__name__)


class FaceAnalysisError(RuntimeError):
    """Top-level exception for errors during face analysis."""


class FaceModelLoadError(FaceAnalysisError):
    """Face/gender model failed to load."""


class ImageFaceAnalyzer(ABC):
    """Abstract interface for face detection and gender estimation."""

    @abstractmethod
    def analyze(self, image_path: str | Path) -> FaceAnalysisResult:
        """Analyze an image and return the face/gender result."""
        raise NotImplementedError

    def has_face(self, image_path: str | Path) -> bool:
        """Return whether the image contains at least one face."""
        return self.analyze(image_path).has_face


class InsightFaceAnalyzer(ImageFaceAnalyzer):
    """InsightFace-based face detection + gender estimation."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str | int = "cpu",
        det_size: int = 640,
        min_confidence: float = 0.50,
    ) -> None:
        """Create the analyzer and load the InsightFace model.

        Args:
            model_name: InsightFace model pack name.
            device: Inference device: ``"cpu"`` or ``"cuda"``/``"cuda:0"``/``0``.
            det_size: Face detection input size (one side of the square, in pixels).
            min_confidence: Face detections below this value are ignored.

        Raises:
            ValueError: A setting is out of the valid range.
            FaceModelLoadError: The model failed to load.
        """
        if det_size <= 0:
            raise ValueError("det_size must be positive.")
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be greater than 0 and at most 1.")

        self.model_name = model_name
        self.device = device
        self.min_confidence = min_confidence
        ctx_id = self._device_to_ctx(device)

        # When GPU is requested, make sure onnxruntime can find the CUDA/cuDNN DLLs.
        if ctx_id >= 0:
            self._prepare_cuda_dll_search()

        # If GPU was requested but onnxruntime lacks CUDA support, silently fall back to CPU.
        if ctx_id >= 0 and not self._cuda_available():
            logger.warning(
                "onnxruntime has no CUDA support; running the face model on CPU. "
                "Install onnxruntime-gpu if you want GPU acceleration."
            )
            ctx_id = -1

        # Import the heavy dependency only when used.
        from insightface.app import FaceAnalysis

        # Specify the execution providers to avoid unnecessary CUDA init errors.
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )

        logger.info("Loading face model: %s (ctx_id=%d)", model_name, ctx_id)
        try:
            self._app = FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "genderage"],
                providers=providers,
            )
            self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        except Exception as exc:  # noqa: BLE001
            raise FaceModelLoadError(
                f"Failed to load face model: {model_name}"
            ) from exc
        logger.info("Face model loaded: %s", model_name)

    @staticmethod
    def _cuda_available() -> bool:
        """Return whether onnxruntime can use the CUDA execution provider."""
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _prepare_cuda_dll_search() -> None:
        """Prepare the search path so onnxruntime can find the CUDA/cuDNN DLLs.

        onnxruntime-gpu needs ``cudart64_12.dll``, ``cudnn64_9.dll``, etc.; even
        without a system CUDA toolkit, the same-version DLLs bundled by PyTorch
        can be reused. ``import torch`` registers its own lib folder in the DLL
        search path on Windows, so we leverage that and register it explicitly too.
        """
        try:
            import os

            import torch

            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib) and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
                logger.debug("Added CUDA DLL search path: %s", torch_lib)
        except Exception:  # noqa: BLE001
            logger.debug("Skipping CUDA DLL search path setup.", exc_info=True)

    @staticmethod
    def _device_to_ctx(device: str | int) -> int:
        """Convert a device string to an InsightFace ctx_id (CPU=-1, GPU=0)."""
        text = str(device).strip().lower()
        if text == "cpu":
            return -1
        # "cuda", "cuda:0", "0", etc. map to GPU 0.
        if ":" in text:
            try:
                return int(text.split(":", 1)[1])
            except ValueError:
                return 0
        if text.isdigit():
            return int(text)
        return 0

    def analyze(self, image_path: str | Path) -> FaceAnalysisResult:
        """Analyze a single image and return the face/gender result.

        Raises:
            FileNotFoundError: The image file does not exist.
            ValueError: The path is not a file.
            CorruptImageError: The image cannot be read.
            FaceAnalysisError: An error occurred during analysis.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file path: {path}")

        image = self._read_image(path)

        try:
            faces = self._app.get(image)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Face analysis failed: %s", path)
            raise FaceAnalysisError(f"Face analysis failed: {path}") from exc

        infos = self._to_face_infos(faces)
        logger.debug("%d face(s) detected: %s", len(infos), path)
        return FaceAnalysisResult(
            image_path=path,
            has_face=bool(infos),
            faces=tuple(infos),
        )

    @staticmethod
    def _read_image(path: Path):
        """Read an image as a BGR array, handling non-ASCII paths and HEIC.

        ``cv2.imread`` cannot handle non-ASCII paths on Windows, so this uses
        ``numpy.fromfile`` + ``cv2.imdecode``. For formats OpenCV cannot decode
        (e.g. HEIC), it falls back to Pillow.
        """
        import cv2
        import numpy as np

        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except OSError as exc:
            raise CorruptImageError(f"Cannot read image: {path}") from exc

        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image

        # OpenCV could not decode it (e.g. HEIC); try Pillow.
        try:
            from PIL import Image

            with Image.open(path) as pil_image:
                rgb = np.asarray(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            raise CorruptImageError(
                f"Corrupt or unsupported image: {path}"
            ) from exc

    def _to_face_infos(self, faces) -> list[FaceInfo]:
        """Convert InsightFace results into an engine-independent ``FaceInfo`` list."""
        infos: list[FaceInfo] = []
        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self.min_confidence:
                continue

            box = [float(v) for v in face.bbox]
            age = getattr(face, "age", None)
            infos.append(
                FaceInfo(
                    confidence=score,
                    bounding_box=(box[0], box[1], box[2], box[3]),
                    gender=self._gender_of(face),
                    age=int(age) if age is not None else None,
                )
            )
        return infos

    @staticmethod
    def _gender_of(face) -> str | None:
        """Extract the gender from an InsightFace face object."""
        sex = getattr(face, "sex", None)
        if isinstance(sex, str):
            if sex.upper() == "M":
                return GENDER_MALE
            if sex.upper() == "F":
                return GENDER_FEMALE

        gender = getattr(face, "gender", None)
        if gender is None:
            return None
        return GENDER_MALE if int(gender) == 1 else GENDER_FEMALE
