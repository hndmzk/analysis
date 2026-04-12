<#
.SYNOPSIS
    Register (or update) the weekly audit task in Windows Task Scheduler.
    Must be run as Administrator.

.PARAMETER DayOfWeek
    Day to run. Default: Saturday

.PARAMETER Time
    Time to run (HH:mm). Default: 06:00

.PARAMETER TaskName
    Scheduler task name. Default: MarketPredictionAgent-WeeklyAudit

.PARAMETER Unregister
    Remove the task instead of creating it.

.EXAMPLE
    # Register (run as Admin):
    .\scripts\register_scheduled_task.ps1

    # Change to Sunday 08:00:
    .\scripts\register_scheduled_task.ps1 -DayOfWeek Sunday -Time "08:00"

    # Remove:
    .\scripts\register_scheduled_task.ps1 -Unregister
#>

param(
    [string]$DayOfWeek = "Saturday",
    [string]$Time = "06:00",
    [string]$TaskName = "MarketPredictionAgent-WeeklyAudit",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$TaskName' unregistered."
    exit 0
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$ScriptPath = Join-Path $ProjectRoot "scripts\scheduled_audit.ps1"

if (-not (Test-Path $ScriptPath)) {
    Write-Error "scheduled_audit.ps1 not found at $ScriptPath"
    exit 1
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$ScriptPath`"" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek $DayOfWeek `
    -At $Time

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Limited

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Set-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal | Out-Null
    Write-Host "Task '$TaskName' updated: $DayOfWeek at $Time"
}
else {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Weekly full_light public audit for market-prediction-agent" | Out-Null
    Write-Host "Task '$TaskName' registered: $DayOfWeek at $Time"
}

Write-Host ""
Write-Host "Details:"
Write-Host "  Script : $ScriptPath"
Write-Host "  WorkDir: $ProjectRoot"
Write-Host "  Schedule: Every $DayOfWeek at $Time"
Write-Host "  Logs   : $ProjectRoot\storage\logs\scheduled_audits\"
Write-Host ""
Write-Host "To test immediately:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
