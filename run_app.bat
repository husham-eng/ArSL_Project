@echo off
REM ---------------------------------------------------------------
REM  ArSL Assistant launcher - double-click to start the application
REM  (no terminal commands, no PowerShell execution-policy changes).
REM  Python is looked up in this order:
REM    1) .venv\Scripts\python.exe inside this folder (made by setup_windows.bat)
REM    2) the path written on the first line of local_python_path.txt
REM    3) "python" on the system PATH
REM ---------------------------------------------------------------
setlocal
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "local_python_path.txt" set /p PY=<"local_python_path.txt"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ArSL] Python was not found.
    echo Run setup_windows.bat first, or write the full path of your python.exe
    echo on the first line of local_python_path.txt in this folder.
    pause
    exit /b 1
)
echo [ArSL] Starting with: %PY%
"%PY%" app\practice_session.py
if errorlevel 1 (
    echo.
    echo [ArSL] The application stopped with an error - see the messages above.
    pause
)
