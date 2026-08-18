# Minimal CI smoke gate for Windows: assert ffmpeg.exe/ffprobe.exe run, report
# the expected version, and exit cleanly. NOT the full codec suite (tests.ps1).
#
# Cross-exec note (mirrors the same handling in smoke.sh): windows-aarch64 is
# built for ARM64, and the windows-latest runner is x64. Windows-on-ARM can run
# x64 binaries through emulation, but not the reverse, so an ARM64 PE cannot be
# executed here at all. For those platforms we never execute — we validate the
# PE header (MZ/PE signatures + Machine field) instead, which also catches a
# mislabeled or truncated artifact. Native platforms are always executed, so a
# broken native binary can never slip through.
param(
    [Parameter(Mandatory = $true)][string]$Workspace,
    [Parameter(Mandatory = $true)][string]$ExpectedVersion,
    [string]$Platform = ''
)
$ErrorActionPreference = 'Stop'

function Fail($msg) { Write-Host "❌ $msg"; exit 1 }
function Note($msg) { Write-Host "ℹ️  $msg" }

# Platforms whose binary is built for a different CPU arch than this runner and
# therefore cannot be executed on it. Maps platform to the expected PE Machine
# value. Keep in sync with the build matrix if another Windows arch is added.
function Get-CrossExecMachine($platform) {
    switch ($platform) {
        'windows-aarch64' { return 0xAA64 }  # IMAGE_FILE_MACHINE_ARM64
        default           { return $null }
    }
}

# Reads the COFF Machine field: e_lfanew is a 4-byte LE offset at 0x3C, the
# "PE\0\0" signature sits there, and Machine is the 2 bytes immediately after.
function Get-PeMachine($path) {
    $fs = [System.IO.File]::OpenRead($path)
    try {
        $br = New-Object System.IO.BinaryReader($fs)
        if ($fs.Length -lt 0x40) { return $null }
        $fs.Position = 0
        if ($br.ReadUInt16() -ne 0x5A4D) { return $null }   # 'MZ'
        $fs.Position = 0x3C
        $peOff = $br.ReadInt32()
        if ($peOff -le 0 -or ($peOff + 6) -gt $fs.Length) { return $null }
        $fs.Position = $peOff
        if ($br.ReadUInt32() -ne 0x00004550) { return $null }  # 'PE\0\0'
        return $br.ReadUInt16()
    } finally { $fs.Close() }
}

function Assert-PeMachine($bin, $expected) {
    $got = Get-PeMachine $bin
    if ($null -eq $got) { Fail "$(Split-Path $bin -Leaf): not a valid PE image" }
    if ($got -ne $expected) {
        Fail ("{0}: PE machine=0x{1:X4}, expected 0x{2:X4}" -f (Split-Path $bin -Leaf), $got, $expected)
    }
}

$ffmpeg  = Join-Path $Workspace 'ffmpeg.exe'
$ffprobe = Join-Path $Workspace 'ffprobe.exe'

if (-not (Test-Path $ffmpeg))  { Fail "ffmpeg.exe not found at $ffmpeg" }
if (-not (Test-Path $ffprobe)) { Fail "ffprobe.exe not found at $ffprobe" }
if ((Get-Item $ffmpeg).Length  -eq 0) { Fail 'ffmpeg.exe is empty' }
if ((Get-Item $ffprobe).Length -eq 0) { Fail 'ffprobe.exe is empty' }

# Cross-exec platforms: never execute — validate PE headers and stop here.
$machine = Get-CrossExecMachine $Platform
if ($null -ne $machine) {
    Assert-PeMachine $ffmpeg  $machine
    Assert-PeMachine $ffprobe $machine
    Note ("Cross-exec ({0}): binaries cannot run on this x64 runner — validated PE headers instead (machine=0x{1:X4})." -f $Platform, $machine)
    Write-Host '✅ Smoke (presence + PE header) passed for cross-exec binaries.'
    exit 0
}

function Assert-Version($bin, $banner) {
    $out = & $bin -version 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { Write-Host $out; Fail "$bin -version exited $LASTEXITCODE" }
    if ($out -notmatch [regex]::Escape($banner)) { Write-Host $out; Fail "missing '$banner' banner" }
    if ($out -notmatch [regex]::Escape($ExpectedVersion)) { Write-Host $out; Fail "expected version $ExpectedVersion not found" }
    Write-Host "✅ $(Split-Path $bin -Leaf) reports version $ExpectedVersion and exits 0"
}

Assert-Version $ffmpeg  'ffmpeg version'
Assert-Version $ffprobe 'ffprobe version'
Write-Host '✅ Smoke test passed.'
exit 0
