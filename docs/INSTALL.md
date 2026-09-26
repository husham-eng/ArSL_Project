# Installing and running the ArSL Assistant (Windows)

This guide is for anyone who wants to run the practice application on their
own Windows computer. No programming knowledge is needed.

## What you need

- Windows 10 or 11 and an internet connection (for installation, speech output and free-mode speech recognition).
- A camera. The system was developed with an Intel RealSense D435, but any webcam that gives a normal color image works.
- **Python 3** from https://www.python.org/downloads/ — during installation, tick **"Add python.exe to PATH"**. The project was tested with Python 3.14.
- About 3 GB of free disk space (the deep-learning libraries are large).

## Step 1 — Download the project

Either click **Code → Download ZIP** on https://github.com/husham-eng/ArSL_Project and unzip it,
or, if you use Git: `git clone https://github.com/husham-eng/ArSL_Project.git`

Put the folder somewhere with a **plain English path** (for example `C:\ArSL_Project`).
Paths containing Arabic or other non-Latin characters can break some libraries.

## Step 2 — One-time setup

Double-click **`setup_windows.bat`** in the project folder. It:

1. creates a private Python environment (`.venv`) inside the folder,
2. installs all required libraries (this takes several minutes),
3. downloads the three model files if they are missing:
   - `grayscale_simplecnn_best.pt` and `color_sign_resnet18_best.pt` (from the project's GitHub release),
   - `app\models\hand_landmarker.task` (Google MediaPipe's official hand-tracking model).

If a model file cannot be downloaded, the script says which one. Download it manually from
https://github.com/husham-eng/ArSL_Project/releases (release assets) and place it as follows:

| File | Location |
|---|---|
| `grayscale_simplecnn_best.pt` | project folder (next to `run_app.bat`) |
| `color_sign_resnet18_best.pt` | project folder (next to `run_app.bat`) |
| `hand_landmarker.task` | `app\models\` |

## Step 3 — Start the application

- Double-click **`run_app.bat`**, or
- double-click **`make_shortcut.bat`** once to create an **"ArSL Assistant"** icon on your Desktop, then use that icon from now on.

A small console window opens (minimized when started from the shortcut). Keep it open:
it shows diagnostic messages and is useful if something goes wrong.

### Already have your own Python environment?

Instead of running `setup_windows.bat`, write the full path of that environment's
`python.exe` on the first line of a text file named **`local_python_path.txt`** in the
project folder (no quotes), for example:

```
C:\Users\me\envs\arsl\Scripts\python.exe
```

`run_app.bat` will then use it.

## Checking that everything works

After the window opens:

- The status line should say **"Model ready (grayscale + color fusion) ✅"**. If it says *grayscale only*, `color_sign_resnet18_best.pt` is missing.
- The console should show `[info] Real hand tracking active`. If not, `hand_landmarker.task` or the `mediapipe` library is missing, and the guide box will stay fixed in the middle of the image instead of following the hand.
- Press **Start camera**: the guide box should follow your hand.

## Interface language

Use the **🌐 Language** menu at the top of the window to switch between Arabic,
English and Russian. The application restarts in the chosen language, and the
choice is remembered. Only the interface changes: the sign letters, words,
practice scenarios and spoken output remain Arabic by design.

## Common problems

| Problem | Solution |
|---|---|
| "Python was not found" | Install Python 3 with "Add python.exe to PATH" ticked, then run `setup_windows.bat` again. |
| Black camera image | Press **Other camera**; some cameras (like the D435) expose several streams. |
| Warning "gray/IR stream" next to the camera number | Harmless with a white background, as long as your hand appears in its normal color. |
| No sound | Arabic speech uses Google's online text-to-speech; check the internet connection. |
| The window is taller than the screen | Maximize the window. |

See `docs/USER_GUIDE_AR.md` for how to use the application.
