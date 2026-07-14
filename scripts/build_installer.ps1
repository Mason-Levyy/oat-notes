param(
    [switch]$SkipSync,
    [switch]$SkipTests,
    [switch]$SkipAppBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stageRoot = Join-Path $env:LOCALAPPDATA "oat-notes-build"
$appSource = Join-Path $stageRoot "dist\oat-notes"

if (-not $SkipAppBuild) {
    # Do not rebuild over the normal runnable folder: Windows locks native
    # DLLs while that copy of oat-notes is open. The installer gets a private
    # staging folder so upgrades can be built while the app is running.
    $installerDistPath = Join-Path $stageRoot "installer-dist"
    $appSource = Join-Path $installerDistPath "oat-notes"
    $buildParams = @{ DistPath = $installerDistPath }
    if ($SkipSync) { $buildParams.SkipSync = $true }
    if ($SkipTests) { $buildParams.SkipTests = $true }
    & "$PSScriptRoot\build_app.ps1" @buildParams
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
