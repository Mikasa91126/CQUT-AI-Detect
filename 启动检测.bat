@echo off
chcp 65001 >nul
title CQUT-AI-Detect - AI text detection launcher
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem Prefer the project virtualenv when it exists, otherwise use system python.
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m aidetect.launch %*
  goto :done
)

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python not found. Install Python 3.10+ and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

python -m aidetect.launch %*
if errorlevel 1 (
  echo.
  echo Something went wrong. Try:  python -m aidetect doctor
  echo (If models are missing:      python -m aidetect fetch all^)
  echo.
  pause
  exit /b 1
)

:done
