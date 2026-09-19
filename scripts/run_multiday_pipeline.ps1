Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = 'Stop'

Write-Host "Polarix multi-day pipeline"
Write-Host "WARNING: This script reads/writes local Parquet and may call Databento only if explicitly allowed."
Write-Host "WARNING: It does not trade."
Write-Host "WARNING: Do not run broad multi-day downloads without checking disk and API budget."

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
& $Python -u (Join-Path $RepoRoot "scripts\run_multiday_pipeline.py") @Args
exit $LASTEXITCODE
