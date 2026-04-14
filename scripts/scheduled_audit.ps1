<#
.SYNOPSIS
    Scheduled wrapper for run_public_audit_suite.py (full_light profile).
    Designed to be invoked by Windows Task Scheduler.

.DESCRIPTION
    Runs the full_light audit with live data, logs stdout/stderr to a
    timestamped file under storage/logs/scheduled_audits/, and writes a
    one-line status to latest_run.json for easy monitoring.

.PARAMETER Profile
    Audit profile preset. Default: full_light

.PARAMETER SourceMode
    Data source mode. Default: live

.PARAMETER TickerSets
    Optional ticker-set override passed to run_public_audit_suite.py.

.PARAMETER TaskLabel
    Optional label used to separate log and latest status filenames for
    multiple scheduled audit tasks.

.EXAMPLE
    .\scripts\scheduled_audit.ps1
    .\scripts\scheduled_audit.ps1 -Profile fast -SourceMode dummy
    .\scripts\scheduled_audit.ps1 -Profile fast -SourceMode live -TickerSets "AAPL,MSFT,NVDA,AMZN,META" -TaskLabel "individual_stocks_fast"
#>

param(
    [string]$Profile = "full_light",
    [string]$SourceMode = "live",
    [string]$TickerSets = "",
    [string]$TaskLabel = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$LogDir = Join-Path $ProjectRoot "storage\logs\scheduled_audits"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SafeTaskLabel = $TaskLabel -replace "[^A-Za-z0-9_.-]", "_"
$FilePrefix = if ($SafeTaskLabel) { "$Timestamp-$SafeTaskLabel" } else { $Timestamp }
$StatusFileName = if ($SafeTaskLabel) { "latest_run_$SafeTaskLabel.json" } else { "latest_run.json" }
$LogFile = Join-Path $LogDir "$FilePrefix.log"
$StdoutFile = Join-Path $LogDir "$FilePrefix.stdout.tmp"
$StderrFile = Join-Path $LogDir "$FilePrefix.stderr.tmp"
$StatusFile = Join-Path $LogDir $StatusFileName

# Keep scheduled runs predictable on small Windows hosts. OpenBLAS can otherwise
# over-allocate worker threads and fail late in the audit.
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "1" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "1" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "1" }
$env:NUMEXPR_NUM_THREADS = if ($env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS } else { "1" }
$env:LOKY_MAX_CPU_COUNT = if ($env:LOKY_MAX_CPU_COUNT) { $env:LOKY_MAX_CPU_COUNT } else { "2" }

function Invoke-AuditNotification {
    param(
        [string]$ProjectRoot,
        [string]$StatusFile,
        [string]$LogFile
    )

    try {
        $NotifyScript = Join-Path $ProjectRoot "scripts\notify_audit_status.py"
        $UvNotifyCommand = (Get-Command uv -ErrorAction Stop).Source
        "=== NOTIFICATION ===" | Out-File -FilePath $LogFile -Encoding utf8 -Append
        & $UvNotifyCommand run python $NotifyScript --status-file $StatusFile 2>&1 |
            Out-File -FilePath $LogFile -Encoding utf8 -Append
        if ($LASTEXITCODE -ne 0) {
            "Notification hook exited with code $LASTEXITCODE" | Out-File -FilePath $LogFile -Encoding utf8 -Append
        }
    }
    catch {
        "Notification hook failed: $($_.Exception.ToString())" | Out-File -FilePath $LogFile -Encoding utf8 -Append
    }
}

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$StartTime = Get-Date

try {
    $AuditScript = Join-Path $ProjectRoot "scripts\run_public_audit_suite.py"
    $UvCommand = (Get-Command uv -ErrorAction Stop).Source
    $ArgumentList = "run python `"$AuditScript`" --profile `"$Profile`" --source-mode `"$SourceMode`""
    if ($TickerSets) {
        $ArgumentList = "$ArgumentList --ticker-sets `"$TickerSets`""
    }

    Push-Location $ProjectRoot
    try {
        $Process = Start-Process `
            -FilePath $UvCommand `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $ProjectRoot `
            -NoNewWindow `
            -PassThru `
            -Wait `
            -RedirectStandardOutput $StdoutFile `
            -RedirectStandardError $StderrFile
    }
    finally {
        Pop-Location
    }

    $ExitCode = $Process.ExitCode
    $EndTime = Get-Date
    $Duration = ($EndTime - $StartTime).TotalSeconds

    $StdoutText = ""
    "=== STDOUT ===" | Out-File -FilePath $LogFile -Encoding utf8
    if (Test-Path $StdoutFile) {
        $StdoutText = Get-Content -Path $StdoutFile -Raw
        Get-Content -Path $StdoutFile | Out-File -FilePath $LogFile -Encoding utf8 -Append
    }
    "=== STDERR ===" | Out-File -FilePath $LogFile -Encoding utf8 -Append
    if (Test-Path $StderrFile) {
        Get-Content -Path $StderrFile | Out-File -FilePath $LogFile -Encoding utf8 -Append
    }
    Remove-Item -Path $StdoutFile, $StderrFile -ErrorAction SilentlyContinue

    $status = @{
        timestamp    = $Timestamp
        profile      = $Profile
        source_mode  = $SourceMode
        ticker_sets  = $TickerSets
        task_label   = $SafeTaskLabel
        exit_code    = $ExitCode
        duration_sec = [math]::Round($Duration, 1)
        log_file     = $LogFile
        success      = ($ExitCode -eq 0)
    }

    if ($StdoutText) {
        try {
            $SuiteOutput = $StdoutText | ConvertFrom-Json -ErrorAction Stop
            if ($SuiteOutput.suite_path) {
                $status["suite_path"] = [string]$SuiteOutput.suite_path
            }
        }
        catch {
            "Could not parse audit stdout JSON for suite_path: $($_.Exception.Message)" |
                Out-File -FilePath $LogFile -Encoding utf8 -Append
        }
    }

    if ($ExitCode -ne 0) {
        $status["error"] = ((Get-Content -Path $LogFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n")
    }

    $status | ConvertTo-Json -Depth 2 | Out-File -FilePath $StatusFile -Encoding utf8
    Invoke-AuditNotification -ProjectRoot $ProjectRoot -StatusFile $StatusFile -LogFile $LogFile

    if ($ExitCode -ne 0) {
        Write-Host "Audit exited with code $ExitCode. See $LogFile"
        exit $ExitCode
    }
    else {
        Write-Host "Audit completed in ${Duration}s. Log: $LogFile"
    }
}
catch {
    $EndTime = Get-Date
    $Duration = ($EndTime - $StartTime).TotalSeconds

    $errorMsg = $_.Exception.ToString()
    "ERROR: $errorMsg" | Out-File -FilePath $LogFile -Encoding utf8 -Append

    $status = @{
        timestamp    = $Timestamp
        profile      = $Profile
        source_mode  = $SourceMode
        ticker_sets  = $TickerSets
        task_label   = $SafeTaskLabel
        exit_code    = 1
        duration_sec = [math]::Round($Duration, 1)
        log_file     = $LogFile
        success      = $false
        error        = $errorMsg
    } | ConvertTo-Json -Depth 2

    $status | Out-File -FilePath $StatusFile -Encoding utf8
    Invoke-AuditNotification -ProjectRoot $ProjectRoot -StatusFile $StatusFile -LogFile $LogFile
    exit 1
}
