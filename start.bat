@echo off
title Gate Entry System

:: ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Make sure Python is installed and in PATH.
    pause
    exit /b 1
)

:: ── Check model ───────────────────────────────────────────────────────────────
if not exist "models\yolov8n_plate.onnx" (
    echo [INFO] ONNX model not found. Downloading now...
    python download_model.py
    if errorlevel 1 (
        echo [ERROR] Model download failed. Check your internet connection.
        pause
        exit /b 1
    )
)

:: ── Check dependencies ────────────────────────────────────────────────────────
echo [INFO] Checking dependencies...
python -c "import fastapi, onnxruntime, cv2, pytesseract" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

:: ── Start server ──────────────────────────────────────────────────────────────
echo.
echo  ================================
echo   Gate Entry System
echo   http://localhost:8000
echo  ================================
echo.
echo  Opening browser in 7 seconds...
echo  Press Ctrl+C to stop the server.
echo.

:: Open browser after short delay (server needs a moment to start)
ping -n 8 127.0.0.1 >nul 2>&1
start http://localhost:8000

:: Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000

pause