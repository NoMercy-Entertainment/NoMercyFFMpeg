# Structured result recorder for tests/tests.ps1. Emits the identical schema to
# lib/report.sh so the checklist bot has one parser, not two.
#
# The pretty console output stays as it was for humans. Machines read this JSON
# instead — scraping the tick and cross glyphs would make the merge gate depend
# on emoji surviving every terminal, code page and log collector on the way to
# the bot.

$script:REPORT_PLATFORM = ''
$script:REPORT_ENTRIES = [System.Collections.Generic.List[object]]::new()

function Report-Init {
    param([Parameter(Mandatory)][string]$Platform)
    $script:REPORT_PLATFORM = $Platform
    $script:REPORT_ENTRIES = [System.Collections.Generic.List[object]]::new()
}

function Report-Add {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][ValidateSet('passed', 'failed', 'skipped')][string]$Status,
        [int]$DurationSeconds = 0,
        [string]$Reason = $null
    )
    $script:REPORT_ENTRIES.Add([ordered]@{
            name       = $Name
            status     = $Status
            duration_s = $DurationSeconds
            reason     = [string]::IsNullOrWhiteSpace($Reason) ? $null : $Reason
        })
}

# ffmpeg prints its own version on the first line of -version; read it from the
# binary under test rather than trusting whatever the caller was told it is.
function Get-ReportFfmpegVersion {
    param([string]$Binary)
    if (-not (Test-Path $Binary)) { return 'unknown' }
    try {
        $line = (& $Binary -hide_banner -version 2>$null | Select-Object -First 1)
        $parts = "$line" -split '\s+'
        if ($parts.Count -ge 3) { return $parts[2] }
        return 'unknown'
    }
    catch { return 'unknown' }
}

function Report-Write {
    param(
        [Parameter(Mandatory)][string]$Path,
        [string]$Binary
    )
    $counts = @{ passed = 0; failed = 0; skipped = 0 }
    foreach ($entry in $script:REPORT_ENTRIES) { $counts[$entry.status]++ }

    $document = [ordered]@{
        schema         = 1
        platform       = $script:REPORT_PLATFORM
        ffmpeg_version = (Get-ReportFfmpegVersion -Binary $Binary)
        host           = [ordered]@{
            os       = 'Windows'
            arch     = $env:PROCESSOR_ARCHITECTURE
            hostname = [System.Net.Dns]::GetHostName()
            kernel   = [System.Environment]::OSVersion.Version.ToString()
        }
        capabilities   = [ordered]@{
            nvenc        = (Hw-Capability -Feature 'nvenc')
            amf          = (Hw-Capability -Feature 'amf')
            vpl          = (Hw-Capability -Feature 'vpl')
            videotoolbox = (Hw-Capability -Feature 'videotoolbox')
        }
        totals         = [ordered]@{
            passed  = $counts.passed
            failed  = $counts.failed
            skipped = $counts.skipped
            total   = $script:REPORT_ENTRIES.Count
        }
        tests          = @($script:REPORT_ENTRIES)
    }

    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $document | ConvertTo-Json -Depth 6 | Set-Content -Path $Path -Encoding utf8
}
