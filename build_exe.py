"""Build Windows executables for the GUI and CLI with PyInstaller.

Run from an environment that has the app dependencies + pyinstaller installed
(a CPU-only virtual environment is recommended for a smaller, portable build):

    python build_exe.py            # build both
    python build_exe.py gui        # GUI only
    python build_exe.py cli        # CLI only

Output goes to dist/HumanPhotoClassifier/ and dist/hpc-cli/.
Model weights are downloaded at first run, so they are not bundled.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Heavy packages whose data files / submodules must be collected wholesale.
COLLECT = ["ultralytics", "insightface", "onnxruntime", "cv2", "torch", "torchvision"]

# Submodules PyInstaller's static analysis can miss.
HIDDEN = ["scipy", "skimage", "sklearn", "pandas"]


def _common_args() -> list[str]:
    args: list[str] = ["--noconfirm", "--clean"]
    for pkg in COLLECT:
        args += ["--collect-all", pkg]
    for mod in HIDDEN:
        args += ["--collect-submodules", mod]
    return args


def _run(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", *args]
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_gui() -> None:
    _run([
        *_common_args(),
        "--windowed",
        "--name", "HumanPhotoClassifier",
        "--icon", "assets/icon.ico",
        "--add-data", "assets;assets",
        "main.py",
    ])


def build_cli() -> None:
    _run([
        *_common_args(),
        "--console",
        "--name", "hpc-cli",
        "--icon", "assets/icon.ico",
        "cli.py",
    ])


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target in ("gui", "both"):
        build_gui()
    if target in ("cli", "both"):
        build_cli()
    if target not in ("gui", "cli", "both"):
        print("usage: python build_exe.py [gui|cli|both]")
        return 2
    print("\nDone. See the dist/ folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
