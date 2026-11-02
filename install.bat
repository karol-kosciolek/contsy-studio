@echo off
rem SPDX-FileCopyrightText: 2026 Karol Kosciolek
rem SPDX-License-Identifier: GPL-3.0-only
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
echo Contsy Studio - Windows installation
echo.
set "PYTHON_EXE="
if not defined PYTHON_EXE for /f "delims=" %%p in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
if not defined PYTHON_EXE for /f "delims=" %%p in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python312\python.exe"
if not defined PYTHON_EXE if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
if not defined PYTHON_EXE (
    where winget >nul 2>nul
    if errorlevel 1 goto no_winget
    echo Installing Python 3.12 from the Windows package manager...
    winget install --id Python.Python.3.12 --exact --source winget --accept-source-agreements --accept-package-agreements
    if errorlevel 1 goto failed
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)
if not exist "%PYTHON_EXE%" if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python312\python.exe"
if not exist "%PYTHON_EXE%" goto failed
"%PYTHON_EXE%" -c "import sys; assert sys.version_info[:2] in ((3,11),(3,12)) and sys.maxsize>2**32, 'Python 3.11 or 3.12 64-bit is required'"
if errorlevel 1 goto failed
where git >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if errorlevel 1 goto no_winget
    winget install --id Git.Git --exact --source winget --accept-source-agreements --accept-package-agreements
    if errorlevel 1 goto failed
)
set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
where ffmpeg >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if errorlevel 1 goto no_winget
    winget install --id Gyan.FFmpeg --exact --source winget --accept-source-agreements --accept-package-agreements
    if errorlevel 1 goto failed
)
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%p"
set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
"%PYTHON_EXE%" -m app.setup_windows --prepare-venv
if errorlevel 1 goto failed
venv\Scripts\python.exe -m app.setup_windows --install
if errorlevel 1 goto failed
rem Asks Lite or Full and downloads every model; "install.bat lite|full|skip" answers in advance
venv\Scripts\python.exe -m app.setup_windows --models %1
if errorlevel 1 goto models_failed
echo.
echo Contsy Studio is installed. Double-click start.bat to open it.
pause
exit /b 0
:models_failed
echo.
echo [ERROR] The model download stopped. Run install.bat again to continue: finished files are kept.
pause
exit /b 1
:no_winget
echo [ERROR] Windows Package Manager is missing. Install App Installer from Microsoft Store,
echo or install Python, Git and FFmpeg using the links below, then retry.
echo App Installer: https://apps.microsoft.com/detail/9NBLGGH4NNS1
goto failed
:failed
echo.
echo [ERROR] Installation stopped. See the error above and run install.bat again.
echo Python: https://www.python.org/downloads/windows/
echo Git: https://git-scm.com/download/win
echo FFmpeg: https://ffmpeg.org/download.html
echo Existing user data and models have been preserved.
pause
exit /b 1
