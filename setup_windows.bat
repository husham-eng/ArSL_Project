@echo off
REM ---------------------------------------------------------------
REM  One-time setup for a new Windows machine:
REM   - creates a private Python environment in .venv
REM   - installs the required libraries
REM   - downloads the model files if they are missing
REM  Requires Python 3 on PATH (https://www.python.org/downloads/,
REM  tick "Add python.exe to PATH" during installation) and internet.
REM ---------------------------------------------------------------
setlocal
cd /d "%~dp0"
set "REL=https://github.com/husham-eng/ArSL_Project/releases/download/v1.0.5"
set "HAND=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

where python >nul 2>nul || (echo Python not found on PATH. Install Python 3 first. & pause & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating .venv ...
    python -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
set "PY=.venv\Scripts\python.exe"

echo [2/3] Installing libraries (this can take several minutes) ...
"%PY%" -m pip install --upgrade pip
if exist requirements.txt "%PY%" -m pip install -r requirements.txt
if exist app\requirements_app.txt "%PY%" -m pip install -r app\requirements_app.txt
"%PY%" -m pip install torch torchvision opencv-python customtkinter pillow numpy pyttsx3 arabic-reshaper python-bidi gTTS sounddevice SpeechRecognition mediapipe matplotlib
if errorlevel 1 (echo Library installation failed - see messages above. & pause & exit /b 1)

echo [3/3] Checking model files ...
if not exist app\models mkdir app\models
if not exist grayscale_simplecnn_best.pt curl -L -f -o grayscale_simplecnn_best.pt "%REL%/grayscale_simplecnn_best.pt"
if not exist color_sign_resnet18_best.pt curl -L -f -o color_sign_resnet18_best.pt "%REL%/color_sign_resnet18_best.pt"
if not exist app\models\hand_landmarker.task curl -L -f -o app\models\hand_landmarker.task "%HAND%"

set "MISSING="
for %%F in (grayscale_simplecnn_best.pt color_sign_resnet18_best.pt app\models\hand_landmarker.task) do (
    if exist "%%F" if %%~zF==0 del "%%F"
)
if not exist grayscale_simplecnn_best.pt (set "MISSING=1" & echo   MISSING: grayscale_simplecnn_best.pt)
if not exist color_sign_resnet18_best.pt (set "MISSING=1" & echo   MISSING: color_sign_resnet18_best.pt)
if not exist app\models\hand_landmarker.task (set "MISSING=1" & echo   MISSING: app\models\hand_landmarker.task)
if defined MISSING (
    echo Some model files could not be downloaded. Download them manually - see docs\INSTALL.md.
) else (
    echo All model files are present.
)
echo.
echo Setup finished. Start the application with run_app.bat
echo (or run make_shortcut.bat once to put a shortcut on the Desktop).
pause
