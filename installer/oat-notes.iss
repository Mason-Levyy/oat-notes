#define AppName "Oat Notes"
#define AppVersion "0.1.0"
#define AppPublisher "Mason Levy"
#define AppExeName "oat-notes.exe"
#ifndef AppSource
  #define AppSource "..\dist\oat-notes"
#endif

[Setup]
AppId={{A819DC58-4182-45E9-9618-CA604ACBE8B5}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\Oat Notes
DefaultGroupName=Oat Notes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=OatNotes-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
SetupIconFile=..\assets\oat-notes.ico
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Dirs]
Name: "{userdocs}\Oat Notes"

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Oat Notes"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{userdocs}\Oat Notes"; IconFilename: "{app}\{#AppExeName}"
Name: "{autodesktop}\Oat Notes"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{userdocs}\Oat Notes"; IconFilename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch Oat Notes"; Flags: nowait postinstall skipifsilent
