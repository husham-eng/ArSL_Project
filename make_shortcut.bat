@echo off
REM Creates an "ArSL Assistant" shortcut on the Desktop that runs run_app.bat
REM (console window starts minimized; its messages help with troubleshooting).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\ArSL Assistant.lnk'); $s.TargetPath='%~dp0run_app.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Description='Arabic Sign Language Assistant'; $s.Save()"
if errorlevel 1 (echo Could not create the shortcut.) else (echo Shortcut "ArSL Assistant" created on the Desktop.)
pause
