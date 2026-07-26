import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_sherpa_wrapper_and_native_core_are_both_pinned():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert "sherpa-onnx==1.13.4" in dependencies
    assert "sherpa-onnx-core==1.13.4" in dependencies


def test_app_build_uses_isolated_frozen_uv_environment():
    script = (ROOT / "scripts" / "build_app.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $stageRoot ".venv"' in script
    assert "$env:UV_PROJECT_ENVIRONMENT = $buildEnvironment" in script
    assert "uv sync --frozen --link-mode copy --extra openvino" in script


def test_installer_build_uses_unlocked_private_app_staging():
    app_script = (ROOT / "scripts" / "build_app.ps1").read_text(encoding="utf-8")
    installer_script = (ROOT / "scripts" / "build_installer.ps1").read_text(
        encoding="utf-8"
    )
    assert "[string]$DistPath" in app_script
    assert 'Join-Path $stageRoot "installer-dist"' in installer_script
    assert "$buildParams = @{ DistPath = $installerDistPath }" in installer_script
    assert '& "$PSScriptRoot\\build_app.ps1" @buildParams' in installer_script


def test_version_is_consistent_across_the_build():
    import re

    import oat_notes

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    installer = (ROOT / "installer" / "oat-notes.iss").read_text(encoding="utf-8")
    declared = re.search(r'#define AppVersion "([^"]+)"', installer).group(1)

    assert project["project"]["version"] == oat_notes.__version__ == declared


def test_overlay_toolkit_is_bundled():
    spec = (ROOT / "oat-notes.spec").read_text(encoding="utf-8")
    assert "'tkinter'" in spec


def test_startup_shortcut_launches_in_the_background():
    installer = (ROOT / "installer" / "oat-notes.iss").read_text(encoding="utf-8")
    assert 'Name: "startupicon"' in installer
    assert '{userstartup}\\Oat Notes' in installer
    assert 'Parameters: "--background"' in installer
