param(
    [switch]$SkipSync,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stageRoot = Join-Path $env:LOCALAPPDATA "oat-notes-build"
$workPath = Join-Path $stageRoot "build"
$distPath = Join-Path $stageRoot "dist"

Push-Location $root
try {
    if (-not $SkipSync) {
        uv sync --extra openvino
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
    }
    if (-not $SkipTests) {
        uv run --no-sync pytest
        if ($LASTEXITCODE -ne 0) { throw "tests failed" }
    }
    uv run --no-sync pyinstaller `
        --workpath $workPath `
        --distpath $distPath `
        --noconfirm `
        oat-notes.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
    Write-Host "Application built: $distPath\oat-notes\oat-notes.exe"
}
finally {
    Pop-Location
}
