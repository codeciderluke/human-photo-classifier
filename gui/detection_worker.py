"""Background detection worker.

Thin ``QThread`` wrapper around the shared ``core.pipeline`` so the GUI thread
never blocks. Pipeline callbacks are forwarded as Qt signals.
"""

from __future__ import annotations

import logging

from PyQt5.QtCore import QThread, pyqtSignal

from core import ClassifyConfig, ClassifySummary, PipelineError, run_classification

logger = logging.getLogger(__name__)

# Kept as aliases for existing GUI imports.
DetectionConfig = ClassifyConfig
DetectionSummary = ClassifySummary


class DetectionWorker(QThread):
    """Runs the classification pipeline on a background thread.

    Signals:
        progress: (current index, total count).
        log: message to show the user.
        finished_summary: ClassifySummary when the job ends.
        failed: fatal error message that aborted the job.
    """

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished_summary = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: ClassifyConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request a stop; applied before the next image is processed."""
        self._stop_requested = True
        self.log.emit("Stop requested. Will halt after the current image...")

    def run(self) -> None:
        """Worker thread entry point."""
        try:
            summary = run_classification(
                self._config,
                on_progress=self.progress.emit,
                on_log=self.log.emit,
                should_stop=lambda: self._stop_requested,
            )
        except PipelineError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - report, never crash the thread
            logger.exception("Unexpected error in worker")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_summary.emit(summary)
