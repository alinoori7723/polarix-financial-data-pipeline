[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$TaskName,
    [Parameter(Mandatory = $true)] [datetime]$StartTime,
    [string]$RepoPath = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " Polarix One-Time Scheduled Task Installer"
Write-Host "============================================================"
Write-Host "WARNING: MT5 requires an interactive Windows desktop session."
Write-Host "WARNING: This task will run ONLY when the user is logged on."
Write-Host "WARNING: For the first live run, prefer manual launch via:"
Write-Host "           scripts\run_controlled_live.ps1"
Write-Host "------------------------------------------------------------"

$script = Join-Path $RepoPath "scripts\run_controlled_live.ps1"
if (-not (Test-Path $script)) {
    Write-Error "Launch script not found: $script"
    exit 2
}
$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = (Get-Command powershell -ErrorAction Stop).Source }

$action = New-ScheduledTaskAction -Execute $pwsh `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Once -At $StartTime
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable:$false `
    -Hidden:$false `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null
Write-Host ""
Write-Host "Created scheduled task:"
Get-ScheduledTask -TaskName $TaskName | Format-List `
    TaskName, State, Triggers, Principal, Actions, Settings

Write-Host ""
Write-Host "Reminder: this is a ONE-TIME task. The first live run must still"
Write-Host "be launched manually from an interactive RDP session using"
Write-Host "scripts\run_controlled_live.ps1 unless explicitly approved."
exit 0
