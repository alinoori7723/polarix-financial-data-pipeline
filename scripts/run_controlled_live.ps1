[CmdletBinding()]
param(
    [string]$RepoPath = (Split-Path -Parent $PSScriptRoot),
    [string]$ReportsDir = (Join-Path (Split-Path -Parent $PSScriptRoot) ".polarix\reports"),
    [string]$VenvActivate = ""
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " Polarix Controlled Live Run -- Supervisor"
Write-Host "============================================================"
Write-Host "WARNING: Run from interactive RDP session only."
Write-Host "WARNING: Do not log off Windows while MT5 logger is running."
Write-Host "WARNING: Disconnect RDP only if leaving the session."
Write-Host "------------------------------------------------------------"

Set-Location -Path $RepoPath

if (-not $VenvActivate) {
    $VenvActivate = Join-Path $RepoPath ".venv\Scripts\Activate.ps1"
}
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtualenv activate script not found: $VenvActivate"
    exit 2
}
. $VenvActivate
$PolarixSrc = Join-Path $RepoPath "src"
if ($env:PYTHONPATH) {
    if ($env:PYTHONPATH -notmatch [regex]::Escape($PolarixSrc)) {
        $env:PYTHONPATH = "$PolarixSrc;$env:PYTHONPATH"
    }
} else {
    $env:PYTHONPATH = $PolarixSrc
}

if (-not (Test-Path $ReportsDir)) {
    New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
}

$stamp = (Get-Date -Format "yyyyMMddTHHmmssZ")
$consoleLog = Join-Path $ReportsDir "controlled_live_${stamp}.log"

Write-Host "Console log: $consoleLog"
Write-Host "PYTHONPATH:  $env:PYTHONPATH"
Write-Host "Launch command: python -u scripts\run_controlled_live.py"
Write-Host "------------------------------------------------------------"
$python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
& $python -u (Join-Path $RepoPath "scripts\run_controlled_live.py") 2>&1 |
    Tee-Object -FilePath $consoleLog

$rc = $LASTEXITCODE
Write-Host "------------------------------------------------------------"
Write-Host "Supervisor exit code: $rc"
exit $rc
