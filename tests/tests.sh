#!/bin/bash
# Full codec + hardware suite for a nomercy-ffmpeg build.
#
# Usage: tests.sh <workspace> [--platform <name>] [--json <report path>]
#
# Hardware-accelerated encoders are gated on the host actually having the
# hardware (see lib/capabilities.sh). A missing GPU is a skip with a recorded
# reason, never a failure — and never a silent pass either.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/capabilities.sh
source "${HERE}/lib/capabilities.sh"
# shellcheck source=lib/report.sh
source "${HERE}/lib/report.sh"

Workspace="$1"
shift 2>/dev/null || true

Platform=""
ReportPath=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--platform)
		Platform="$2"
		shift 2
		;;
	--json)
		ReportPath="$2"
		shift 2
		;;
	*)
		echo "Error: unknown argument '$1'." >&2
		exit 1
		;;
	esac
done

if [[ -z "$Workspace" ]]; then
	echo "Error: Workspace path is required." >&2
	exit 1
fi

if [[ -L "$Workspace" ]]; then
	echo "Error: Workspace path cannot be a symbolic link." >&2
	exit 1
fi

[[ -z "$Platform" ]] && Platform="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
report_init "$Platform"

TOTAL_WIDTH_TEXT=54
TOTAL_TESTS=0
PASSED_TESTS=0
SKIPPED_TESTS=0
FAILED_TESTS=0

TestRoot="${Workspace}/sample_files"
SampleVideo="${TestRoot}/sample.mp4"
SampleAudio="${TestRoot}/sample.wav"
SampleImage="${TestRoot}/sample.png"
SampleSubs="${TestRoot}/sample.ass"
SampleSvg="${TestRoot}/sample.svg"

rm -rf "${TestRoot}"
mkdir -p "${TestRoot}"

# Counts the call sites at the bottom of this file so the progress counter can
# read "[3/25]". Anchored at line start so the definitions above never count.
get_test_runs_count() {
	grep -cE '^(run_test|run_hw_test) "' "${BASH_SOURCE[0]}"
}

TOTAL_RUNS=$(get_test_runs_count)

generate_samples() {
	local Total_Count=0
	local Current_Count=0
	local Start_Time End_Time

	# Count needed samples
	[[ ! -f "$SampleVideo" ]] && ((Total_Count++))
	[[ ! -f "$SampleAudio" ]] && ((Total_Count++))
	[[ ! -f "$SampleImage" ]] && ((Total_Count++))
	[[ ! -f "$SampleSubs" ]] && ((Total_Count++))
	[[ ! -f "$SampleSvg" ]] && ((Total_Count++))

	# Generate samples
	if [[ ! -f "$SampleVideo" ]]; then
		Start_Time=$(date +%s)
		Current_Count=$((Current_Count + 1))
		text_with_padding "📹 Generating sample video" "[$Current_Count/$Total_Count]" 1
		ffmpeg_command="-hide_banner -y -f lavfi -i \"testsrc=duration=10:size=1280x720:rate=30\" -c:v libx264 -crf 23 \"$SampleVideo\""
		output=$(eval "${Workspace}/ffmpeg $ffmpeg_command" 2>&1)
		exit_code=$?
		End_Time=$(date +%s)
		if [[ $exit_code -ne 0 ]]; then
			text_with_padding "❌ Error generating sample video" "[$((End_Time - Start_Time))s]" 1
			echo "$output"
			exit 1
		fi
		text_with_padding "✅ Sample video generated" "[$((End_Time - Start_Time))s]" 1
	fi

	if [[ ! -f "$SampleAudio" ]]; then
		Start_Time=$(date +%s)
		Current_Count=$((Current_Count + 1))
		text_with_padding "🔊 Generating sample audio" "[$Current_Count/$Total_Count]" 1
		ffmpeg_command="-hide_banner -y -f lavfi -i \"sine=frequency=1000:duration=10\" -c:a pcm_s16le \"$SampleAudio\""
		output=$(eval "${Workspace}/ffmpeg $ffmpeg_command" 2>&1)
		exit_code=$?
		End_Time=$(date +%s)
		if [[ $exit_code -ne 0 ]]; then
			text_with_padding "❌ Error generating sample audio" "[$((End_Time - Start_Time))s]" 1
			echo "$output"
			exit 1
		fi
		text_with_padding "✅ Sample audio generated" "[$((End_Time - Start_Time))s]" 1
	fi

	if [[ ! -f "$SampleImage" ]]; then
		Start_Time=$(date +%s)
		Current_Count=$((Current_Count + 1))
		text_with_padding "🖼️ Generating sample image" "[$Current_Count/$Total_Count]"
		ffmpeg_command="-hide_banner -y -f lavfi -i \"testsrc=duration=1:size=640x480:rate=1\" -frames:v 1 \"$SampleImage\""
		output=$(eval "${Workspace}/ffmpeg $ffmpeg_command" 2>&1)
		exit_code=$?
		End_Time=$(date +%s)
		if [[ $exit_code -ne 0 ]]; then
			text_with_padding "❌ Error generating sample image" "[$((End_Time - Start_Time))s]" 1
			echo "$output"
			exit 1
		fi
		text_with_padding "✅ Sample image generated" "[$((End_Time - Start_Time))s]" 1
	fi

	if [[ ! -f "$SampleSubs" ]]; then
		{
			echo "[Script Info]"
			echo "Title: Test Subtitle"
			echo "ScriptType: v4.00+"
			echo ""
			echo "[V4+ Styles]"
			echo "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
			echo "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,0"
			echo ""
			echo "[Events]"
			echo "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
			echo "Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,Test subtitle"
		} >$SampleSubs
		End_Time=$(date +%s)
		text_with_padding "✅ Sample subtitles generated" "[$((End_Time - Start_Time))s]" 1
	fi

	# A hand-written SVG: decoding it needs the librsvg decoder, so the librsvg
	# test below can only pass when librsvg was actually compiled in.
	if [[ ! -f "$SampleSvg" ]]; then
		Start_Time=$(date +%s)
		Current_Count=$((Current_Count + 1))
		printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="#3355ff"/></svg>' >"$SampleSvg"
		End_Time=$(date +%s)
		text_with_padding "✅ Sample SVG generated" "[$((End_Time - Start_Time))s]" 1
	fi

	if [ $Total_Count -gt 0 ]; then
		printf "%${TOTAL_WIDTH_TEXT}s\n" | tr ' ' '-' # Print a horizontal line
	fi
}

text_with_padding() {
	local text_before=$1
	local text_after=$2
	local extra_padding=${3:-0}
	local text_length=$((${#text_before} + ${#text_after}))
	local padding=$((TOTAL_WIDTH_TEXT - text_length - extra_padding))
	printf "%s%*s%s\n" "$text_before" "$padding" " " "$text_after"
}

check_command() {
	if [[ ! -f ${Workspace}/ffmpeg ]]; then
		printf "%s\n" "❌ FFmpeg executable not found in current directory"
		exit 1
	fi
	if [[ ! -x ${Workspace}/ffmpeg ]]; then
		chmod +x ${Workspace}/ffmpeg
	fi
}

run_test() {
	local name=$1
	local command=$2
	local expected_output=$3
	local test_output
	local exit_code
	local duration

	TOTAL_TESTS=$((TOTAL_TESTS + 1))
	name=$(echo $name | tr '[:lower:]' '[:upper:]')

	text_with_padding "🧪 Testing ${name}" "[${TOTAL_TESTS}/${TOTAL_RUNS}]" 1
	Start_Time=$(date +%s)

	test_output=$(eval "${Workspace}/ffmpeg $command" 2>&1)
	exit_code=$?
	End_Time=$(date +%s)
	duration=$((End_Time - Start_Time))

	if [[ $exit_code -eq 0 ]] && echo "$test_output" | grep -q "$expected_output"; then
		text_with_padding "✅ ${name} test passed" "[${duration}s]" 1
		PASSED_TESTS=$((PASSED_TESTS + 1))
		report_add "${name}" passed "${duration}"
	else
		text_with_padding "❌ ${name} test failed" "[${duration}s]" 1
		FAILED_TESTS=$((FAILED_TESTS + 1))
		# Print the diagnosis. A red run that swallows its own output costs
		# another full round trip on whichever machine happens to own it.
		echo "   exit=${exit_code}, expected to match: ${expected_output}"
		echo "$test_output" | tail -12 | sed 's/^/   | /'
		report_add "${name}" failed "${duration}" \
			"exit ${exit_code}; $(echo "$test_output" | tail -3 | tr '\n' ' ')"
	fi
}

# Same contract as run_test, but the encoder only runs when the host owns the
# hardware. Absent hardware skips WITH the reason recorded, so a skip can be
# audited later instead of being indistinguishable from a pass.
run_hw_test() {
	local name=$1
	local capability=$2
	local command=$3
	local expected_output=$4
	local reason

	if reason=$(hw_capability_reason "$capability"); then
		run_test "$name" "$command" "$expected_output"
		return
	fi

	TOTAL_TESTS=$((TOTAL_TESTS + 1))
	name=$(echo $name | tr '[:lower:]' '[:upper:]')
	text_with_padding "🧪 Testing ${name}" "[${TOTAL_TESTS}/${TOTAL_RUNS}]" 1
	text_with_padding "➖ ${name} was skipped" "[0s]" 1
	echo "   ${reason}"
	SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
	report_add "${name}" skipped 0 "${reason}"
}

# Main execution
printf "%${TOTAL_WIDTH_TEXT}s\n" | tr ' ' '-' # Print a horizontal line
printf "%s\n" "        _   _       __  __                      "
printf "%s\n" "       | \ | | ___ |  \/  | ___ _ __ ___ _   _  "
printf "%s\n" "       |  \| |/ _ \| |\/| |/ _ \ '__/ __| | | | "
printf "%s\n" "       | |\  | (_) | |  | |  __/ | | (__| |_| | "
printf "%s\n" "       |_| \_|\___/|_|  |_|\___|_|  \___|\__, | "
printf "%s\n" "         _____ _____ __  __ ____  _____ _|___/  "
printf "%s\n" "        |  ___|  ___|  \/  |  _ \| ____/ ___|   "
printf "%s\n" "        | |_  | |_  | |\/| | |_) |  _|| |  _    "
printf "%s\n" "        |  _| |  _| | |  | |  __/| |__| |_| |   "
printf "%s\n" "        |_|   |_|   |_|  |_|_|   |_____\____|   "
printf "%s\n" ""
printf "%${TOTAL_WIDTH_TEXT}s\n" | tr ' ' '-' # Print a horizontal line

check_command
generate_samples

# Basic tests
run_test "version" "-version" "ffmpeg version"
run_test "libx264" "-y -i ${SampleVideo} -c:v libx264 ${TestRoot}/test_h264.mp4" "x264"
run_test "libx265" "-y -i ${SampleVideo} -c:v libx265 ${TestRoot}/test_h265.mp4" "x265"
run_test "libvpx" "-y -i ${SampleVideo} -c:v libvpx-vp9 -frames:v 1 ${TestRoot}/test_vp9.webm" "vp9"
run_test "libaom" "-y -i ${SampleVideo} -c:v libaom-av1 -frames:v 1 ${TestRoot}/test_av1.mkv" "av1"
run_test "libtheora" "-y -i ${SampleVideo} -c:v libtheora -frames:v 1 ${TestRoot}/test_theora.ogv" "theora"
run_test "libfdk_aac" "-y -i ${SampleAudio} -c:a libfdk_aac ${TestRoot}/test_aac.m4a" "aac"
run_test "libopus" "-y -i ${SampleAudio} -c:a libopus ${TestRoot}/test_opus.opus" "opus"
run_test "libmp3lame" "-y -i ${SampleAudio} -c:a libmp3lame ${TestRoot}/test_mp3.mp3" "mp3"
run_test "libwebp" "-y -i ${SampleImage} -c:v libwebp -f webp ${TestRoot}/test_webp.webp" "webp"
run_test "libopenjpeg" "-y -i ${SampleImage} -c:v libopenjpeg ${TestRoot}/test_jp2.jp2" "openjpeg"
run_test "librsvg" "-y -i ${SampleSvg} -frames:v 1 ${TestRoot}/test_svg.png" "svg"
run_test "libass" "-y -i ${SampleVideo} -vf \"ass=${SampleSubs}\" ${TestRoot}/test_ass.mp4" "ass"
run_test "auto_mkdir" "-y -f lavfi -i \"testsrc=duration=1:size=320x240:rate=1\" -frames:v 1 ${TestRoot}/subdir_test/nested/output.png" "output.png"

# Hardware acceleration — each runs only where the hardware exists.
run_hw_test "NVENC" nvenc "-y -i ${SampleVideo} -c:v h264_nvenc ${TestRoot}/test_nvenc.mp4" "nvenc"
run_hw_test "VPL" vpl "-y -i ${SampleVideo} -c:v h264_vpl ${TestRoot}/test_vpl.mp4" "vpl"
run_hw_test "AMF" amf "-y -i ${SampleVideo} -c:v h264_amf ${TestRoot}/test_amf.mp4" "amf"
run_hw_test "VIDEOTOOLBOX" videotoolbox "-y -i ${SampleVideo} -c:v h264_videotoolbox ${TestRoot}/test_vt.mp4" "videotoolbox"

# Additional format tests
run_test "libbluray" "-hide_banner -protocols | grep bluray" "bluray"
run_test "libdvdread" "-hide_banner -version | grep dvdread" "dvdread"
run_test "libcdio" "-hide_banner -version | grep cdio" "cdio"
run_test "libfribidi" "-hide_banner -version | grep fribidi" "fribidi"
run_test "libsrt" "-hide_banner -version | grep srt" "srt"
run_test "libxml2" "-hide_banner -version | grep xml" "xml"

# AV1 codec tests
run_test "libdav1d" "-hide_banner -decoders" "dav1d"
run_test "librav1e" "-hide_banner -encoders" "rav1e"

# Stemsplit filter
run_test "stemsplit" "-hide_banner -filters | grep stemsplit" "stemsplit"

# OCR subtitle encoder
run_test "ocr_subtitle" "-hide_banner -encoders" "ocr_subtitle"

# Print summary
printf "%${TOTAL_WIDTH_TEXT}s\n" | tr ' ' '-' # Print a horizontal line
text_with_padding "📊 Summary:" ""
text_with_padding "Passed tests:" "${PASSED_TESTS}"
text_with_padding "Skipped tests:" "${SKIPPED_TESTS}"
text_with_padding "Failed tests:" "${FAILED_TESTS}"
text_with_padding "Total tests:" "${TOTAL_TESTS}"
printf "%${TOTAL_WIDTH_TEXT}s\n" | tr ' ' '-' # Print a horizontal line
echo ""

if [[ -n "$ReportPath" ]]; then
	report_write "$ReportPath" "${Workspace}/ffmpeg"
	echo "📄 Report written to ${ReportPath}"
fi

rm -rf "${TestRoot}"

# Exit with failure if any tests failed
if [ "${FAILED_TESTS}" -gt 0 ]; then
	exit $FAILED_TESTS
fi
exit 0
