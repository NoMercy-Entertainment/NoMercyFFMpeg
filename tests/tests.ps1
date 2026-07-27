# Full codec + hardware suite for a nomercy-ffmpeg build.
#
# Hardware-accelerated encoders are gated on the host actually having the
# hardware (see lib/capabilities.ps1). A missing GPU is a skip with a recorded
# reason, never a failure — and never a silent pass either.
param (
    [string]$Workspace = $(throw "Workspace path is required"),
    [string]$Platform = 'windows-x86_64',
    [string]$JsonReport = ''
)

$HERE = Split-Path -Parent $PSCommandPath
. "$HERE/lib/capabilities.ps1"
. "$HERE/lib/report.ps1"

Report-Init -Platform $Platform

$TOTAL_WIDTH_TEXT = 54
$script:TOTAL_TESTS = 0
$script:PASSED_TESTS = 0
$script:SKIPPED_TESTS = 0
$script:FAILED_TESTS = 0

$TestRoot = "$Workspace\sample_files"
$SampleVideo = "$TestRoot\sample.mp4"
$SampleAudio = "$TestRoot\sample.wav"
$SampleImage = "$TestRoot\sample.png"
$SampleSubs = "$TestRoot\sample.ass"
$AssSubPath = $SampleSubs -replace '\\','/'
$AssSubPath = $AssSubPath -replace ':','\\:'

Remove-Item -Recurse -Force -Path $TestRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $TestRoot -ErrorAction SilentlyContinue | Out-Null

# Counts the call sites at the bottom of this file so the progress counter can
# read "[3/26]". Anchored at line start so the definitions above never count.
function get_test_runs_count {
    $lines = Get-Content -Path $PSCommandPath
    return @($lines | Where-Object { $_ -match '^(run_test|run_hw_test) "' }).Count
}

$TOTAL_RUNS = get_test_runs_count

function generate_samples {
    $Total_Count = 0
    $Current_Count = 0
    $Start_Time = $null
    $End_Time = $null

    # Count needed samples
    if (-Not (Test-Path $SampleVideo)) { $Total_Count++ }
    if (-Not (Test-Path $SampleAudio)) { $Total_Count++ }
    if (-Not (Test-Path $SampleImage)) { $Total_Count++ }
    if (-Not (Test-Path $SampleSubs)) { $Total_Count++ }

    # Generate samples
    if (-Not (Test-Path $SampleVideo)) {
        $Start_Time = Get-Date
        $Current_Count++
        & "$Workspace\ffmpeg.exe" -hide_banner -y -f lavfi -i "testsrc=duration=10:size=1280x720:rate=30" -c:v libx264 -crf 23 "$SampleVideo" 2>&1 | Out-Null
        $elapsed = (New-TimeSpan -Start $Start_Time -End (Get-Date)).TotalSeconds.ToString('0')
        text_with_padding "     $ICON_PASS Sample video" "[${elapsed}s]"
    }

    if (-Not (Test-Path $SampleAudio)) {
        $Start_Time = Get-Date
        $Current_Count++
        & "$Workspace\ffmpeg.exe" -hide_banner -y -f lavfi -i "sine=frequency=1000:duration=10" -c:a pcm_s16le "$SampleAudio" 2>&1 | Out-Null
        $elapsed = (New-TimeSpan -Start $Start_Time -End (Get-Date)).TotalSeconds.ToString('0')
        text_with_padding "     $ICON_PASS Sample audio" "[${elapsed}s]"
    }

    if (-Not (Test-Path $SampleImage)) {
        $Start_Time = Get-Date
        $Current_Count++
        & "$Workspace\ffmpeg.exe" -hide_banner -y -f lavfi -i "testsrc=duration=1:size=640x480:rate=1" -frames:v 1 "$SampleImage" 2>&1 | Out-Null
        $elapsed = (New-TimeSpan -Start $Start_Time -End (Get-Date)).TotalSeconds.ToString('0')
        text_with_padding "     $ICON_PASS Sample image" "[${elapsed}s]"
    }

    if (-Not (Test-Path $SampleSubs)) {
        $Current_Count++
        $assContent = @(
            '[Script Info]'
            'Title: Test Subtitle'
            'ScriptType: v4.00+'
            ''
            '[V4+ Styles]'
            'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding'
            'Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,0'
            ''
            '[Events]'
            'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text'
            'Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,Test subtitle'
        ) -join "`n"
        [System.IO.File]::WriteAllText($SampleSubs, $assContent)  

        $elapsed = (New-TimeSpan -Start $Start_Time -End (Get-Date)).TotalSeconds.ToString('0')
        text_with_padding "     $ICON_PASS Sample subtitles" "[${elapsed}s]"
    }

    if ($Current_Count -gt 0) {
        Write-Host ([string]::new('-', $TOTAL_WIDTH_TEXT))
    }
}

function text_with_padding {
    param (
        $text_before,
        $text_after,
        $extra_padding = 0
    )
    $text_length = $text_before.Length + $text_after.Length
    $padding = $TOTAL_WIDTH_TEXT - $text_length - $extra_padding
    Write-Host -NoNewline $text_before
    Write-Host -NoNewline (" " * $padding)
    Write-Host $text_after
}

function check_command {
    if (-Not (Test-Path "$Workspace\ffmpeg.exe")) {
        Write-Host "❌ FFmpeg executable not found in current directory"
        exit 1
    }
}

function run_test {
    param (
        $name,
        $command,
        $expected_output
    )

    $script:TOTAL_TESTS++
    $name = $name.ToUpper()
    text_with_padding "🧪 Testing ${name}" "[$script:TOTAL_TESTS/$TOTAL_RUNS]"
    $Start_Time = Get-Date
    $test_output = Invoke-Expression "$Workspace\ffmpeg.exe $command 2>&1" | Out-String
    $exit_code = $LASTEXITCODE
    $duration = [int](New-TimeSpan -Start $Start_Time -End (Get-Date)).TotalSeconds

    if ( $exit_code -eq 0 -and $test_output -cmatch $expected_output ) {
        text_with_padding "✅ ${name} test passed" "[ ${duration}s ]" 1
        $script:PASSED_TESTS++
        Report-Add -Name $name -Status passed -DurationSeconds $duration
    }
    else {
        text_with_padding "❌ ${name} test failed" "[ ${duration}s ]" 1
        $script:FAILED_TESTS++
        # Print the diagnosis. A red run that swallows its own output costs
        # another full round trip on whichever machine happens to own it.
        Write-Host "   exit=${exit_code}, expected to match: ${expected_output}"
        $tail = ($test_output -split "`r?`n" | Where-Object { $_ } | Select-Object -Last 12)
        $tail | ForEach-Object { Write-Host "   | $_" }
        $short = (($test_output -split "`r?`n" | Where-Object { $_ } | Select-Object -Last 3) -join ' ')
        Report-Add -Name $name -Status failed -DurationSeconds $duration -Reason "exit ${exit_code}; ${short}"
    }
}

# Same contract as run_test, but the encoder only runs when the host owns the
# hardware. Absent hardware skips WITH the reason recorded, so a skip can be
# audited later instead of being indistinguishable from a pass.
function run_hw_test {
    param (
        $name,
        $capability,
        $command,
        $expected_output
    )

    $reason = Hw-CapabilityReason -Feature $capability
    if (-not $reason) {
        run_test $name $command $expected_output
        return
    }

    $script:TOTAL_TESTS++
    $name = $name.ToUpper()
    text_with_padding "🧪 Testing ${name}" "[$script:TOTAL_TESTS/$TOTAL_RUNS]"
    text_with_padding "➖ ${name} was skipped" "[ 0s ]" 1
    Write-Host "   $reason"
    $script:SKIPPED_TESTS++
    Report-Add -Name $name -Status skipped -DurationSeconds 0 -Reason $reason
}

# Main execution
Write-Host ([string]::new('-', $TOTAL_WIDTH_TEXT))
Write-Host '  NoMercy FFmpeg Test Suite'
Write-Host ([string]::new('-', $TOTAL_WIDTH_TEXT))

check_command
generate_samples

# Basic tests
run_test "version" "-version" "ffmpeg version"
run_test "libx264" "-y -i $SampleVideo -c:v libx264 $TestRoot\test_h264.mp4" "x264"
run_test "libx265" "-y -i $SampleVideo -c:v libx265 $TestRoot\test_h265.mp4" "x265"
run_test "libvpx" "-y -i $SampleVideo -c:v libvpx-vp9 -frames:v 1 $TestRoot\test_vp9.webm" "vp9"
run_test "libaom" "-y -i $SampleVideo -c:v libaom-av1 -frames:v 1 $TestRoot\test_av1.mkv" "av1"
run_test "libtheora" "-y -i $SampleVideo -c:v libtheora -frames:v 1 $TestRoot\test_theora.ogv" "theora"
run_test "libfdk_aac" "-y -i $SampleAudio -c:a libfdk_aac $TestRoot\test_aac.m4a" "aac"
run_test "libopus" "-y -i $SampleAudio -c:a libopus $TestRoot\test_opus.opus" "opus"
run_test "libmp3lame" "-y -i $SampleAudio -c:a libmp3lame $TestRoot\test_mp3.mp3" "mp3"
run_test "libwebp" "-y -i $SampleImage -c:v libwebp -f webp $TestRoot\test_webp.webp" "webp"
run_test "libopenjpeg" "-y -i $SampleImage -c:v libopenjpeg $TestRoot\test_jp2.jp2" "openjpeg"
run_test "libass" "-y -i '${SampleVideo}' -vf ass='${AssSubPath}' '${TestRoot}/test_ass.mp4'" "ass"
run_test "auto_mkdir" "-y -f lavfi -i `"testsrc=duration=1:size=320x240:rate=1`" -frames:v 1 $TestRoot\subdir_test\nested\output.png" "output.png"

# Hardware acceleration — each runs only where the hardware exists.
run_hw_test "NVENC" nvenc "-y -i $SampleVideo -c:v h264_nvenc $TestRoot\test_nvenc.mp4" "nvenc"
run_hw_test "VPL" vpl "-y -i $SampleVideo -c:v h264_vpl $TestRoot\test_vpl.mp4" "vpl"
run_hw_test "AMF" amf "-y -i $SampleVideo -c:v h264_amf $TestRoot\test_amf.mp4" "amf"
run_hw_test "VIDEOTOOLBOX" videotoolbox "-y -i $SampleVideo -c:v h264_videotoolbox $TestRoot\test_vt.mp4" "videotoolbox"

# Additional format tests
run_test "libbluray" "-hide_banner -protocols | findstr bluray" "bluray"
run_test "libdvdread" "-hide_banner -version | findstr dvdread" "dvdread"
run_test "libcdio" "-hide_banner -version | findstr cdio" "cdio"
run_test "libfribidi" "-hide_banner -version | findstr fribidi" "fribidi"
run_test "libsrt" "-hide_banner -version | findstr srt" "srt"
run_test "libxml2" "-hide_banner -version | findstr xml" "xml"

# AV1 codec tests
run_test "libdav1d" "-hide_banner -decoders" "dav1d"
run_test "librav1e" "-hide_banner -encoders" "rav1e"

# OCR subtitle encoder
run_test "ocr_subtitle" "-hide_banner -encoders" "ocr_subtitle"

# Print summary
Write-Host ([string]::new('-', $TOTAL_WIDTH_TEXT))
text_with_padding "📊 Summary:" ""
text_with_padding "Passed tests:" "$script:PASSED_TESTS"
text_with_padding "Skipped tests:" "$script:SKIPPED_TESTS"
text_with_padding "Failed tests:" "$script:FAILED_TESTS"
text_with_padding "Total tests:" "$script:TOTAL_TESTS"
Write-Host ([string]::new('-', $TOTAL_WIDTH_TEXT))
Write-Host ""

if ($JsonReport) {
    Report-Write -Path $JsonReport -Binary "$Workspace\ffmpeg.exe"
    Write-Host "📄 Report written to $JsonReport"
}

Remove-Item -Recurse -Force -Path $TestRoot -ErrorAction SilentlyContinue

# Exit with failure if any tests failed
if ($FAILED_TESTS -gt 0) {
    exit $FAILED_TESTS
}
exit 0
