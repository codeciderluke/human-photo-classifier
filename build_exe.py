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

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Heavy packages whose data files / submodules must be collected wholesale.
COLLECT = ["ultralytics", "insightface", "onnxruntime", "cv2", "torch",
           "torchvision", "pi_heif"]

# Submodules PyInstaller's static analysis can miss.
HIDDEN = ["scipy", "skimage", "sklearn", "pandas"]

# Unused heavy modules that break or bloat the build (onnx reference evaluator).
EXCLUDE = ["onnx.reference"]


def _extra_data() -> list[str]:
    """Data files insightface resolves via a path that collect-all misses."""
    import insightface

    objects = Path(insightface.__file__).parent / "data" / "objects"
    if objects.is_dir():
        # insightface looks for these at <bundle>/objects/ when frozen.
        return ["--add-data", f"{objects}{os.pathsep}objects"]
    return []


def _common_args() -> list[str]:
    args: list[str] = ["--noconfirm", "--clean"]
    for pkg in COLLECT:
        args += ["--collect-all", pkg]
    for mod in HIDDEN:
        args += ["--collect-submodules", mod]
    for mod in EXCLUDE:
        args += ["--exclude-module", mod]
    args += _extra_data()
    return args


def _run(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", *args]
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _fix_vc_runtime(app_name: str) -> None:
    """Overwrite bundled VC++ runtime DLLs with the system (newer) version.

    PyQt5 bundles an old VC++ runtime (14.26) under Qt5/bin. It loads before
    torch's required newer runtime and breaks torch's DLL init (c10.dll
    WinError 1114). Replace every copy with the system's.
    """
    import shutil

    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    internal = ROOT / "dist" / app_name / "_internal"
    if not internal.is_dir():
        return
    names = ("MSVCP140.dll", "MSVCP140_1.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")
    for name in names:
        src = system32 / name
        if not src.exists():
            continue
        for dst in internal.rglob(name):
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass


def build_gui() -> None:
    # Windowed (no console window). Native libs load correctly once the stale
    # PyQt5 VC++ runtime is replaced (see _fix_vc_runtime) and std streams are
    # repaired at startup (see main.py).
    _run([
        *_common_args(),
        "--windowed",
        "--name", "HumanPhotoClassifier",
        "--icon", "assets/icon.ico",
        "--add-data", "assets;assets",
        "main.py",
    ])
    _fix_vc_runtime("HumanPhotoClassifier")


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
