$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

uv run ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pytest
exit $LASTEXITCODE
