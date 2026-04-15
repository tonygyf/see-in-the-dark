Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "d:\typer\cursor project\see in the dark"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual env not found: .venv"
}

.\.venv\Scripts\python.exe src\train_laptop_starter.py --config configs\laptop_4060_realdata_ctw1500_high_intensity.yaml
