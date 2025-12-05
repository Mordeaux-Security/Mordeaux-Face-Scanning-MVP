@echo off
REM Simple crawler runner with log capture (Windows batch)
REM Usage: run-crawler.bat [sites.txt] or run-crawler.bat url1 url2 ...

setlocal

REM Change to script directory
cd /d "%~dp0"

REM Default log file
set LOG_FILE=debugamd.txt

REM Check for virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found. Run: python -m venv .venv
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=src

REM Build command
set CMD=python -m diabetes_crawler.main

REM Check arguments
if "%1"=="" (
    REM No arguments, use default sites.txt
    if exist "sites.txt" (
        set CMD=%CMD% --sites-file sites.txt
    ) else (
        echo Error: No sites provided and sites.txt not found
        exit /b 1
    )
) else (
    REM Check if first arg is a file
    if exist "%1" (
        set CMD=%CMD% --sites-file %1
    ) else (
        REM Treat as URLs
        set CMD=%CMD% --sites %*
    )
)

echo Running: %CMD%
echo Logging to: %LOG_FILE%
echo.

REM Run command and capture output (Windows doesn't have tee, so use redirection)
%CMD% > "%LOG_FILE%" 2>&1
type "%LOG_FILE%"


