@echo off
cd /d "%~dp0"

echo ============================================================
echo VOD-to-Shorts Clipper - Installation
echo ============================================================
echo.

set PYTHON_INSTALLED=0
set FFMPEG_INSTALLED=0

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [MISSING] Python not found. Installing Python 3.12...
    winget install Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    if %errorlevel% equ 0 (
        echo [OK] Python installed
        set PYTHON_INSTALLED=1
    ) else (
        echo ERROR: Failed to install Python.
        echo Download manually from: https://www.python.org/downloads/
        pause
        exit /b 1
    )
) else (
    echo [OK] Python already installed
    python --version
)

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [MISSING] FFmpeg not found. Installing FFmpeg...
    winget install Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if %errorlevel% equ 0 (
        echo [OK] FFmpeg installed
        set FFMPEG_INSTALLED=1
    ) else (
        echo ERROR: Failed to install FFmpeg.
        echo Download manually from: https://ffmpeg.org/download.html
        pause
        exit /b 1
    )
) else (
    echo [OK] FFmpeg already installed
)

if "%FFMPEG_INSTALLED%"=="1" (
    if exist "C:\Program Files\Gyan\FFmpeg\bin\ffmpeg.exe" (
        set "PATH=C:\Program Files\Gyan\FFmpeg\bin;%PATH%"
    )
)

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo FFmpeg was just installed. Please CLOSE this window and run install.bat again.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Creating virtual environment...
echo ============================================================

if exist "venv" (
    echo [OK] Virtual environment already exists
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo INSTALLATION COMPLETE!
echo ============================================================
echo.
echo Run 'run.bat' to start the clipper.
echo.
pause
