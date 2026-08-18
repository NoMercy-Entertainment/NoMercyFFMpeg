# Hardware-capability probe shared by tests/tests.ps1 and the automated RC
# verifier. Mirrors lib/capabilities.sh — same feature names, same answers.
#
# Answers "can this host actually exercise it?" without invoking ffmpeg. Asking
# ffmpeg would make the build its own oracle: a broken encoder would look
# identical to absent hardware and skip itself green.
#
# Hw-Capability returns "available" or "absent:<reason>".

function Get-HwVideoControllers {
    try { return @(Get-CimInstance Win32_VideoController -ErrorAction Stop) }
    catch { return @() }
}

function Test-HwVendorGpu {
    param([string]$Pattern)
    $gpus = Get-HwVideoControllers
    return [bool]($gpus | Where-Object {
            $_.Name -match $Pattern -or $_.Caption -match $Pattern -or $_.AdapterCompatibility -match $Pattern
        })
}

function Hw-CapabilityNvenc {
    # nvidia-smi is definitive — it only succeeds when the driver is loaded and
    # a device answers. WMI is the fallback, and it under-reports on headless
    # and RDP sessions where the adapter does not enumerate.
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $listing = & $smi.Source -L 2>$null
        if ($LASTEXITCODE -eq 0 -and ($listing -join "`n") -match 'GPU 0:') { return 'available' }
        return 'absent:nvidia-smi reports no usable GPU'
    }
    if (Test-HwVendorGpu 'NVIDIA') { return 'available' }
    return 'absent:no NVIDIA display adapter reported by Windows'
}

function Hw-CapabilityAmf {
    if (-not (Test-HwVendorGpu 'AMD|Radeon')) {
        return 'absent:no AMD display adapter reported by Windows'
    }
    if (-not (Test-Path "$env:SystemRoot\System32\amfrt64.dll")) {
        return 'absent:AMD GPU present but amfrt64.dll is missing'
    }
    return 'available'
}

function Hw-CapabilityVpl {
    if (-not (Test-HwVendorGpu 'Intel')) {
        return 'absent:no Intel display adapter reported by Windows'
    }
    $dlls = @(
        "$env:SystemRoot\System32\libvpl.dll",
        "$env:SystemRoot\System32\libmfxhw64.dll",
        "$env:SystemRoot\System32\libvplintel_preview64.dll"
    )
    if (-not ($dlls | Where-Object { Test-Path $_ })) {
        return 'absent:Intel GPU present but no oneVPL/Media SDK runtime is installed'
    }
    return 'available'
}

function Hw-Capability {
    param([Parameter(Mandatory)][string]$Feature)
    switch ($Feature) {
        'nvenc' { return Hw-CapabilityNvenc }
        'amf' { return Hw-CapabilityAmf }
        'vpl' { return Hw-CapabilityVpl }
        'videotoolbox' { return 'absent:VideoToolbox is macOS-only' }
        default { return "absent:unknown capability '$Feature'" }
    }
}

function Hw-CapabilityReason {
    param([Parameter(Mandatory)][string]$Feature)
    $answer = Hw-Capability -Feature $Feature
    if ($answer -eq 'available') { return $null }
    return ($answer -replace '^absent:', '')
}
