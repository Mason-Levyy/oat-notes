param(
    [switch]$SkipSync,
    [switch]$SkipTests,
    [switch]$SkipAppBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$appSource = Join-Path $env:LOCALAPPDATA "oat-notes-build\dist\oat-notes"

if (-not $SkipAppBuild) {
    $buildArgs = @()
    if ($SkipSync) { $buildArgs += "-SkipSync" }
    if ($SkipTests) { $buildArgs += "-SkipTests" }
    & "$PSScriptRoot\build_app.ps1" @buildArgs
}

$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if ($iscc) {
    $isccPath = $iscc.Source
}
else {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    $isccPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $isccPath) {
    throw "Inno Setup 6 is required. Install it, then rerun this script."
}

if (-not (Test-Path (Join-Path $appSource "oat-notes.exe"))) {
    throw "Built application not found at $appSource"
}

& $isccPath "/DAppSource=$appSource" "$root\installer\oat-notes.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed" }
Write-Host "Installer built: $root\dist\installer\OatNotes-Setup.exe"
