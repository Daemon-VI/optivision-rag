@echo off
REM ============================================================
REM  OptiVision RAG - demo launcher for the Stage-I presentation
REM
REM  Double-click this file, or run it from a terminal.
REM  It starts the Gradio app and opens the browser.
REM
REM  The ColSmol checkpoint is already cached on this machine, so
REM  the first compression takes ~30 s (encoding), not a download.
REM ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   ERROR: .venv not found in %CD%
    echo   Create it with:  python -m venv .venv
    echo.
    pause
    exit /b 1
)

REM Bound the Hugging Face metadata check so a flaky network cannot
REM stall startup; the model itself is served from the local cache.
set HF_HUB_ETAG_TIMEOUT=10
set HF_HUB_DOWNLOAD_TIMEOUT=30
set TOKENIZERS_PARALLELISM=false
set PYTHONUNBUFFERED=1

echo.
echo   Starting OptiVision RAG demo...
echo   The browser opens automatically. Press Ctrl+C here to stop.
echo.

start "" http://127.0.0.1:7860
".venv\Scripts\python.exe" app.py

echo.
echo   Demo stopped.
pause
