@echo off
echo ========================================
echo Windows GPU Worker Service Launcher
echo ========================================

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Starting GPU Worker Service...
echo.
echo IMPORTANT: Make sure Windows Firewall allows connections on port 8765
echo If you get connection errors from Docker, you may need to:
echo 1. Allow port 8765 through Windows Firewall
echo 2. Check that host.docker.internal resolves correctly
echo.

REM Run the Python launcher
python launch.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo GPU Worker Service failed to start.
    echo Check the error messages above.
    pause
)
