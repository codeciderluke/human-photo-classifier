"""Main application window.

Handles only folder selection, settings input, progress display, log display,
and start/stop. It does not interpret YOLO result objects directly; actual
detection is delegated to ``DetectionWorker``.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .detection_worker import DetectionConfig, DetectionSummary, DetectionWorker
from .theme import build_stylesheet


class MainWindow(QWidget):
    """Main window of the Person Photo Extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: DetectionWorker | None = None

        self.setWindowTitle("Human Photo Classifier")
        self.setObjectName("root")
        self._apply_icon()
        # Default size optimized for a Full HD (1920x1080) screen
        self.setMinimumSize(1100, 720)
        self.resize(1440, 900)
        self.setStyleSheet(build_stylesheet())

        self._build_ui()
        self._center_on_screen()

    def _apply_icon(self) -> None:
        """Set the window icon if the asset is available."""
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _center_on_screen(self) -> None:
        """Center the window on the current screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 32, 40, 32)
        outer.setSpacing(24)

        outer.addLayout(self._build_header())

        # Two-column body: left = settings/actions, right = log
        body = QHBoxLayout()
        body.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(24)
        left.addWidget(self._build_settings_card())
        left.addWidget(self._build_action_card())
        left.addStretch(1)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        body.addWidget(left_wrap, stretch=5)
        body.addWidget(self._build_log_card(), stretch=4)

        outer.addLayout(body, stretch=1)

        footer = QLabel("Designed by Codecider Lab")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        outer.addWidget(footer)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Human Photo Classifier")
        title.setObjectName("title")

        subtitle = QLabel(
            "Automatically selects photos that contain people, using YOLO."
        )
        subtitle.setObjectName("subtitle")

        header.addWidget(title)
        header.addWidget(subtitle)
        return header

    def _card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        return card, layout

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _build_settings_card(self) -> QFrame:
        card, layout = self._card()

        # --- Folder selection ---
        layout.addWidget(self._section_label("FOLDERS"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select a source folder")
        self.source_edit.setReadOnly(True)
        source_btn = QPushButton("Browse")
        source_btn.clicked.connect(self._pick_source)

        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Select the destination folder")
        self.dest_edit.setReadOnly(True)
        dest_btn = QPushButton("Browse")
        dest_btn.clicked.connect(self._pick_destination)

        grid.addWidget(self._field_label("Source"), 0, 0)
        grid.addWidget(self.source_edit, 0, 1)
        grid.addWidget(source_btn, 0, 2)
        grid.addWidget(self._field_label("Destination"), 1, 0)
        grid.addWidget(self.dest_edit, 1, 1)
        grid.addWidget(dest_btn, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        # --- Detection settings ---
        layout.addSpacing(4)
        layout.addWidget(self._section_label("DETECTION"))

        opts = QGridLayout()
        opts.setHorizontalSpacing(14)
        opts.setVerticalSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"]
        )

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.00)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.20)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(64)
        self.imgsz_spin.setValue(960)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "cuda:0"])

        opts.addWidget(self._field_label("Model"), 0, 0)
        opts.addWidget(self.model_combo, 0, 1)
        opts.addWidget(self._field_label("Confidence"), 0, 2)
        opts.addWidget(self.conf_spin, 0, 3)
        opts.addWidget(self._field_label("Image size"), 1, 0)
        opts.addWidget(self.imgsz_spin, 1, 1)
        opts.addWidget(self._field_label("Device"), 1, 2)
        opts.addWidget(self.device_combo, 1, 3)
        opts.setColumnStretch(1, 1)
        opts.setColumnStretch(3, 1)
        layout.addLayout(opts)

        self.recursive_check = QCheckBox("Search subfolders")
        self.recursive_check.setChecked(True)
        layout.addWidget(self.recursive_check)

        # --- Classification options ---
        layout.addSpacing(4)
        layout.addWidget(self._section_label("CLASSIFICATION"))

        self.face_check = QCheckBox(
            "Face detection (sort into with_face / no_face)"
        )
        self.face_check.setChecked(False)

        self.gender_check = QCheckBox(
            "Gender recognition (sort into male / female)"
        )
        self.gender_check.setChecked(False)
        # Gender recognition requires face detection.
        self.gender_check.toggled.connect(self._on_gender_toggled)

        layout.addWidget(self.face_check)
        layout.addWidget(self.gender_check)

        return card

    def _on_gender_toggled(self, checked: bool) -> None:
        """Enabling gender also enables face (gender requires face)."""
        if checked:
            self.face_check.setChecked(True)

    def _build_action_card(self) -> QFrame:
        card, layout = self._card()

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch(1)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("subtitle")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        buttons.addWidget(self.status_label)

        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        return card

    def _build_log_card(self) -> QFrame:
        card, layout = self._card()
        layout.addWidget(self._section_label("LOG"))

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("log")
        self.log_view.setReadOnly(True)
        # Cap history so very large batches don't grow memory without bound.
        self.log_view.setMaximumBlockCount(5000)
        layout.addWidget(self.log_view, stretch=1)

        return card

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    # -------------------------------------------------------------- actions

    def _pick_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a source folder")
        if folder:
            self.source_edit.setText(folder)

    def _pick_destination(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select the destination folder"
        )
        if folder:
            self.dest_edit.setText(folder)

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _start(self) -> None:
        source = self.source_edit.text().strip()
        destination = self.dest_edit.text().strip()

        if not source:
            self._warn("Please select a source folder.")
            return
        if not destination:
            self._warn("Please select a destination folder.")
            return
        if source == destination:
            self._warn("Source and destination folders must be different.")
            return

        config = DetectionConfig(
            source_folder=source,
            destination_folder=destination,
            model_name=self.model_combo.currentText().strip(),
            confidence=self.conf_spin.value(),
            image_size=self.imgsz_spin.value(),
            device=self.device_combo.currentText().strip(),
            recursive=self.recursive_check.isChecked(),
            detect_face=self.face_check.isChecked(),
            detect_gender=self.gender_check.isChecked(),
        )

        self._worker = DetectionWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._append_log)
        self._worker.finished_summary.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_thread_done)

        self._set_running(True)
        self.log_view.clear()
        self.progress.setValue(0)
        self.status_label.setText("Analyzing...")
        self._worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.stop_btn.setEnabled(False)

    # --------------------------------------------------------- worker slots

    def _on_progress(self, current: int, total: int) -> None:
        percent = int(current / total * 100) if total else 0
        self.progress.setValue(percent)
        self.status_label.setText(f"{current} / {total}  ·  {percent}%")

    def _on_finished(self, summary: DetectionSummary) -> None:
        state = "Stopped" if summary.stopped else "Done"
        self.status_label.setText(state)
        if not summary.stopped:
            self.progress.setValue(100)
        self._append_log(
            f"-- {state}: total {summary.total} · "
            f"person {summary.with_person} · others {summary.others} · "
            f"damaged->others {summary.damaged} · copied {summary.copied} · "
            f"corrupt skipped {summary.corrupt} · failed {summary.failed}"
        )
        # Show per-category counts
        if summary.category_counts:
            for category, count in sorted(summary.category_counts.items()):
                self._append_log(f"    · {category}: {count}")

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Error")
        self._append_log(f"[Error] {message}")
        self._warn(message, title="Stopped")

    def _on_thread_done(self) -> None:
        self._set_running(False)
        self._worker = None

    # ---------------------------------------------------------------- utils

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for widget in (
            self.source_edit,
            self.dest_edit,
            self.model_combo,
            self.conf_spin,
            self.imgsz_spin,
            self.device_combo,
            self.recursive_check,
            self.face_check,
            self.gender_check,
        ):
            widget.setEnabled(not running)

    def _warn(self, message: str, title: str = "Notice") -> None:
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Safely clean up a running worker when the window closes."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        event.accept()
