@echo off
REM ============================================================
REM build_exe.bat
REM Freezes app\practice_session.py and every dependency it loads
REM at runtime (src\grayscale, src\letter_to_word, src\admin,
REM src\profile, the trained checkpoint, hand_signs images, and
REM conversation_scenarios.json) into a single standalone folder
REM under dist\ArSL_Assistant\ that runs on a machine with no
REM Python installed at all.
REM
REM Run this from the PROJECT ROOT (the folder containing app\,
REM src\, and grayscale_simplecnn_best.pt), on the SAME machine
REM you already trained/tested on -- torch and every other
REM dependency must already be pip-installed here.
REM
REM   cd C:\path\to\arsl_assistant_repo
REM   packaging\build_exe.bat
REM ============================================================
setlocal

if not exist "grayscale_simplecnn_best.pt" (
    echo [error] grayscale_simplecnn_best.pt not found in the current folder.
    echo Run this script from the project root, e.g.:
    echo     cd C:\Users\DELL\Downloads\arsl_assistant_repo
    echo     packaging\build_exe.bat
    pause
    exit /b 1
)

echo === Installing PyInstaller and its extra hooks (one-time) ===
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 goto :error

echo === Cleaning any previous build (avoids stale/partial artifacts) ===
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo.
echo === Building standalone app (this can take several minutes -- torch is large) ===
python -m PyInstaller ^
  --name "ArSL_Assistant" ^
  --onedir ^
  --windowed ^
  --noconfirm ^
  --paths "src\grayscale" ^
  --paths "src\letter_to_word" ^
  --paths "src\admin" ^
  --paths "src\profile" ^
  --add-data "grayscale_simplecnn_best.pt;." ^
  --add-data "app\conversation_scenarios.json;app" ^
  --add-data "app\hand_signs;app\hand_signs" ^
  --collect-all torch ^
  --collect-all torchvision ^
  --collect-all customtkinter ^
  --collect-all pyttsx3 ^
  --collect-all sounddevice ^
  --hidden-import "PIL._tkinter_finder" ^
  app\practice_session.py
if errorlevel 1 goto :error

echo.
echo === Done ===
echo The standalone app is in: dist\ArSL_Assistant\
echo Try it now by double-clicking: dist\ArSL_Assistant\ArSL_Assistant.exe
echo.
echo Next step: open packaging\installer.iss in Inno Setup to build the
echo one-click installer (see packaging\README_PACKAGING_ar.md).
pause
exit /b 0

:error
echo.
echo [error] The build failed -- scroll up to see the actual Python/PyInstaller
echo error message above. Common fixes:
echo   - "ModuleNotFoundError: No module named X" during the built app's first
echo     run -^> re-run this script adding: --hidden-import X
echo   - "operator torchvision::nms does not exist" when running the built
echo     .exe -^> already handled by --collect-all torch/torchvision above;
echo     if it still happens, delete build\ and dist\ manually and re-run
echo     this script for a fully clean rebuild.
echo   - Antivirus flagging/deleting files under build\ or dist\ during the
echo     build -^> add an exception for this project folder and retry.
pause
exit /b 1
