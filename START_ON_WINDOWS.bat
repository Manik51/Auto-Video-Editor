@echo off
title AutoVideoEditor AI Launcher
color 0b
echo ====================================================
echo       AutoVideoEditor AI - Instant Launcher
echo ====================================================
echo.
cd /d "%~dp0"

echo [1/2] Checking Python environment...
python -c "import gradio, cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing core dependencies...
    python -m pip install gradio opencv-python soundfile scipy numpy Pillow --quiet
)

echo.
echo [2/2] Launching AutoVideoEditor Web Interface...
echo Opening browser at http://localhost:7860
start "" "http://localhost:7860"
python ui/gradio_app.py

pause
