; ============================================================
; installer.iss
; Inno Setup script that packages the frozen app (produced by
; build_exe.bat in dist\ArSL_Assistant\) into a normal Windows
; installer: a single ArSL_Assistant_Setup.exe that end users
; double-click, click Next a few times, and get a Start Menu +
; optional Desktop shortcut. No terminal, no Python, no pip --
; everything the app needs (including the trained model and
; every Python dependency) is already inside dist\ArSL_Assistant\.
;
; Requires Inno Setup (free): https://jrsoftware.org/isdl.php
; Build with: open this file in the Inno Setup IDE, then
; Build > Compile (or press Ctrl+F9).
; ============================================================

#define MyAppName "ArSL Assistant"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "ArSL Assistant Project"
#define MyAppExeName "ArSL_Assistant.exe"

[Setup]
AppId={{B6C1A6C4-3F1E-4C2A-9C2B-ARSL-ASSISTANT}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; The frozen app (torch included) is large -- LZMA2 keeps the
; installer file itself reasonably small despite that.
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=installer_output
OutputBaseFilename=ArSL_Assistant_Setup
; Uncomment and point at a .ico file if you have one:
; SetupIconFile=app_icon.ico
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Pulls in EVERYTHING PyInstaller produced -- the .exe, every
; bundled .dll/.pyd, the checkpoint, hand_signs images, and
; conversation_scenarios.json -- since build_exe.bat already
; placed them all under dist\ArSL_Assistant\.
Source: "..\dist\ArSL_Assistant\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
