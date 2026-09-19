$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

uv run ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Advisory only: mypy is not a gate yet. The number should only go down.
$mypy = uvx mypy src/oat_notes 2>&1 | Select-Object -Last 1
Write-Host "mypy: $mypy"
exit 0
