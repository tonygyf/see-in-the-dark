param(
    [switch]$SkipDownload,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual env not found: .venv"
}

$args = @("scripts\prepare_ctw1500_test.py", "--root", $projectRoot)
if ($SkipDownload) {
    $args += "--skip-download"
}
if ($Force) {
    $args += "--force"
}

& $pythonExe @args
