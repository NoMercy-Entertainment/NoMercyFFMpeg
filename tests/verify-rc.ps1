# Verifies one Release Candidate archive on the machine that can actually run
# it, and writes a verdict the checklist bot can act on. Windows counterpart of
# verify-rc.sh — same arguments, same verdict schema.
#
# The integrity check is not decoration. The bot ticks a merge-blocking box
# based on this file, so the verdict has to say which bytes were tested — a
# report that cannot name its own artifact's digest proves nothing.
param(
    [Parameter(Mandatory)][string]$Archive,
    [string]$Manifest = '',
    [Parameter(Mandatory)][string]$Platform,
    [Parameter(Mandatory)][string]$WorkDir,
    [Parameter(Mandatory)][string]$Json,
    [string]$Commit = '',
    [string]$Tag = ''
)

$ErrorActionPreference = 'Stop'
$HERE = Split-Path -Parent $PSCommandPath

function Write-FailVerdict {
    param([string]$Stage, [string]$Detail)
    $document = [ordered]@{
        schema        = 1
        platform      = $Platform
        tag           = $Tag
        commit        = $Commit
        verdict       = 'fail'
        failed_stage  = $Stage
        detail        = $Detail
        integrity     = [ordered]@{ verified = $false }
        report        = $null
    }
    $directory = Split-Path -Parent $Json
    if ($directory -and -not (Test-Path $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $document | ConvertTo-Json -Depth 6 | Set-Content -Path $Json -Encoding utf8
    Write-Host "❌ ${Stage}: ${Detail}"
    exit 1
}

Write-Host "─── Verifying $(Split-Path -Leaf $Archive) for ${Platform} ───"

if (-not (Test-Path $Archive)) { Write-FailVerdict 'download' "archive not found: $Archive" }

$assetName = Split-Path -Leaf $Archive
$actualSha = (Get-FileHash -Path $Archive -Algorithm SHA256).Hash.ToLower()

# The manifest is the release's own record of what was published. Checking the
# archive against it catches a truncated download and a swapped asset alike.
$expectedSha = ''
if ($Manifest -and (Test-Path $Manifest)) {
    $manifestDocument = Get-Content -Raw -Path $Manifest | ConvertFrom-Json
    $entry = $manifestDocument.assets | Where-Object { $_.name -eq $assetName }
    if (-not $entry) { Write-FailVerdict 'integrity' "manifest.json has no entry for $assetName" }
    $expectedSha = "$($entry.sha256)".ToLower()
    if ($expectedSha -ne $actualSha) {
        Write-FailVerdict 'integrity' "sha256 mismatch — manifest $expectedSha, downloaded $actualSha"
    }
    Write-Host "✅ sha256 matches manifest: $actualSha"
}
else {
    Write-Host '::warning::no manifest supplied — recording the digest without cross-checking it'
}

if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

try {
    if ($assetName -like '*.zip') { Expand-Archive -Path $Archive -DestinationPath $WorkDir -Force }
    elseif ($assetName -like '*.tar.gz') { tar -xzf $Archive -C $WorkDir }
    else { Write-FailVerdict 'extract' "unsupported archive type: $assetName" }
}
catch { Write-FailVerdict 'extract' $_.Exception.Message }

if (-not (Test-Path "$WorkDir\ffmpeg.exe")) {
    Write-FailVerdict 'extract' "no ffmpeg.exe inside $assetName"
}

# A build for another OS or CPU can load and then die in ways that look like a
# codec failure. Refuse to report on a binary this host cannot execute at all.
& "$WorkDir\ffmpeg.exe" -hide_banner -version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-FailVerdict 'exec' "the $Platform binary does not execute on this host"
}

$reportPath = Join-Path $WorkDir 'report.json'
& "$HERE\tests.ps1" -Workspace $WorkDir -Platform $Platform -JsonReport $reportPath
$suiteExit = $LASTEXITCODE

if (-not (Test-Path $reportPath)) {
    Write-FailVerdict 'suite' "tests.ps1 produced no report (exit $suiteExit)"
}

$document = [ordered]@{
    schema     = 1
    platform   = $Platform
    tag        = $Tag
    commit     = $Commit
    verdict    = ($suiteExit -eq 0) ? 'pass' : 'fail'
    suite_exit = $suiteExit
    integrity  = [ordered]@{
        verified = [bool]$expectedSha
        asset    = $assetName
        sha256   = $actualSha
    }
    # Windows targets always run natively here; the field exists so the bot has
    # one shape to read across every platform.
    execution  = [ordered]@{
        native      = $true
        translation = $null
        host_arch   = $env:PROCESSOR_ARCHITECTURE
    }
    report     = (Get-Content -Raw -Path $reportPath | ConvertFrom-Json)
}

$directory = Split-Path -Parent $Json
if ($directory -and -not (Test-Path $directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$document | ConvertTo-Json -Depth 8 | Set-Content -Path $Json -Encoding utf8

Write-Host "📄 Verdict ($($document.verdict)) written to $Json"
exit $suiteExit
