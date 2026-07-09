# Builds dist\oat-notes.exe (console app; web UI opens on bare launch).
# The OpenVINO backend is excluded to keep the exe lean - run from source
# (uv run oat-notes --backend openvino) for NPU transcription.
uv run pyinstaller `
    --onefile `
    --name oat-notes `
    --collect-data faster_whisper `
    --collect-data oat_notes `
    --collect-all ctranslate2 `
    --exclude-module openvino `
    --exclude-module openvino_genai `
    --exclude-module openvino_tokenizers `
    --exclude-module openvino_telemetry `
    --noconfirm `
    scripts\launcher.py
