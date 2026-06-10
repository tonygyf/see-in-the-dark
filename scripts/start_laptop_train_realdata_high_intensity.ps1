Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual env not found: .venv"
}

.\.venv\Scripts\python.exe src\train_laptop_starter.py --config configs\laptop_4060_realdata_ctw1500_high_intensity.yaml
