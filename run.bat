@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  Human Classifier - launcher
rem  - First run: auto-create .venv and install dependencies
rem  - Next runs: launch the app directly
rem  (ASCII-only messages to stay codepage-safe on Windows)
rem ============================================================

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

rem --- Check Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

rem --- Create venv only if missing ---
if not exist "%VENV_PY%" (
    echo [SETUP] Creating virtual environment...
    rem Reuse system torch/CUDA via system-site-packages
    python -m venv --system-site-packages "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo [SETUP] Upgrading pip...
    "%VENV_PY%" -m pip install --upgrade pip

    echo [SETUP] Installing dependencies...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [SETUP] Environment ready.
)

rem --- Launch app ---
echo [RUN] Starting Human Classifier...
"%VENV_PY%" main.py
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [WARN] Program exited with code %EXIT_CODE%.
    pause
)

endlocal
exit /b %EXIT_CODE%
