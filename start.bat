@echo off
rem SPDX-FileCopyrightText: 2026 Karol Kosciolek
rem SPDX-License-Identifier: GPL-3.0-only
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] The environment is missing. Run install.bat first.
    pause
    exit /b 1
)
if exist "venv\torch-constraints.txt" set "PIP_CONSTRAINT=%CD%\venv\torch-constraints.txt"
set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_DISABLE_TELEMETRY=1"
venv\Scripts\python.exe -m app.launcher
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Contsy Studio could not start. Check the message above.
    echo Run install.bat to repair missing dependencies. Engine details: comfyui.log
    pause
)
exit /b %APP_EXIT%
