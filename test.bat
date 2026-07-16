@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment not found. Run run.bat once first.
    pause
    exit /b 1
)

echo [TEST] Running unit tests...
"%VENV_PY%" -m pytest
set "EXIT_CODE=%errorlevel%"

pause
endlocal
exit /b %EXIT_CODE%
