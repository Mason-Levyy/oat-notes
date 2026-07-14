param(
    [switch]$SkipSync,
    [switch]$SkipTests,
    [string]$DistPath
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stageRoot = Join-Path $env:LOCALAPPDATA "oat-notes-build"
$buildEnvironment = Join-Path $stageRoot ".venv"
$workPath = Join-Path $stageRoot "build"
$resolvedDistPath = if ($DistPath) { $DistPath } else { Join-Path $stageRoot "dist" }
$previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$env:UV_PROJECT_ENVIRONMENT = $buildEnvironment

Push-Location $root
try {
    if (-not $SkipSync) {
        uv sync --frozen --link-mode copy --extra openvino
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
    }
    if (-not $SkipTests) {
        uv run --no-sync pytest
        if ($LASTEXITCODE -ne 0) { throw "tests failed" }
    }
    uv run --no-sync pyinstaller `
        --workpath $workPath `
        --distpath $resolvedDistPath `
        --noconfirm `
        oat-notes.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
    Write-Host "Application built: $resolvedDistPath\oat-notes\oat-notes.exe"
}
finally {
    Pop-Location
    $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
}
