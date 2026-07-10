# Builds dist\oat-notes.exe (console app; web UI opens on the NPU on bare launch).
uv run --no-sync pyinstaller `
    --onefile `
    --name oat-notes `
    --collect-data faster_whisper `
    --collect-data oat_notes `
    --collect-all ctranslate2 `
    --collect-all openvino `
    --collect-all openvino_genai `
    --collect-all openvino_tokenizers `
    --noconfirm `
    scripts\launcher.py
