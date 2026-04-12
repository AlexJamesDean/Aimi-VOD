@echo off
setlocal enabledelayedexpansion

:: Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

:: Check if FFmpeg is available
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg not found in PATH.
    echo Please close and reopen this terminal, or add FFmpeg to PATH.
    echo.
)

:: Run the clipper with any arguments passed to this script
python vod_clipper.py %*

if !errorlevel! neq 0 (
    echo.
    echo ERROR: vod_clipper.py failed with error code !errorlevel!
    pause
    exit /b !errorlevel!
)
