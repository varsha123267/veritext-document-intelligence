@echo off
title Stacks - Document Intelligence and RAG
cd /d "%~dp0"

echo ================================================
echo   Stacks - Document Intelligence and RAG
echo ================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or 3.11 from python.org,
  echo tick "Add Python to PATH" during install, then double-click this file again.
  pause
  exit /b 1
)

if not exist venv (
  echo [1/4] Creating virtual environment...
  python -m venv venv
)

call venv\Scripts\activate.bat

echo [2/4] Installing dependencies. The first time takes a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installing dependencies failed. Check your internet connection and try again.
  pause
  exit /b 1
)

if not exist .env (
  copy .env.example .env >nul
  echo.
  echo [3/4] Notepad will open now. Replace  your-key-here  with your Gemini API key,
  echo       press Ctrl+S to save, then close Notepad.
  notepad .env
) else (
  echo [3/4] .env already exists - using your saved key.
)

echo.
echo [4/4] Starting the server and opening the app.
echo       The first start takes 20-60 seconds while the search model loads.
echo       The dot at the top of the page turns green by itself when it is ready.
echo       KEEP THIS WINDOW OPEN while you use the app. Close it to stop.
echo.
start "" "%~dp0frontend\index.html"
uvicorn app.main:app --port 8000
pause
