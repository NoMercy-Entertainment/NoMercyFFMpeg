#!/usr/bin/env bash
# Structured result recorder for tests/tests.sh.
#
# The pretty console output stays exactly as it was for humans. Machines read
# this JSON instead — scraping ✅/❌ lines would make the merge gate depend on
# emoji surviving every terminal, locale and log collector between here and the
# checklist bot.
#
# Usage:
#   report_init linux-x86_64
#   report_add libx264 passed 3
#   report_add AMF skipped 0 "no AMD display adapter on the PCI bus"
#   report_write /path/to/report.json

REPORT_PLATFORM=""
REPORT_ENTRIES=()

_json_escape() {
	local s="$1"
	s="${s//\\/\\\\}"
	s="${s//\"/\\\"}"
	s="${s//$'\t'/\\t}"
	s="${s//$'\r'/}"
	s="${s//$'\n'/\\n}"
	printf '%s' "$s"
}

report_init() {
	REPORT_PLATFORM="$1"
	REPORT_ENTRIES=()
}

report_add() {
	local name="$1" status="$2" duration="${3:-0}" reason="${4:-}"
	local entry

	entry=$(printf '{"name":"%s","status":"%s","duration_s":%s,"reason":' \
		"$(_json_escape "$name")" "$(_json_escape "$status")" "${duration}")
	if [[ -z "$reason" ]]; then
		entry+='null}'
	else
		entry+=$(printf '"%s"}' "$(_json_escape "$reason")")
	fi
	REPORT_ENTRIES+=("$entry")
}

_report_count() {
	local want="$1" n=0 entry
	for entry in "${REPORT_ENTRIES[@]:-}"; do
		[[ "$entry" == *"\"status\":\"${want}\""* ]] && n=$((n + 1))
	done
	printf '%s' "$n"
}

# ffmpeg prints its own version on the first line of -version; read it from the
# binary under test rather than trusting whatever the caller was told it is.
_report_ffmpeg_version() {
	local binary="$1"
	[[ -x "$binary" ]] || {
		printf 'unknown'
		return
	}
	"$binary" -hide_banner -version 2>/dev/null | head -1 | awk '{print $3}' || printf 'unknown'
}

report_write() {
	local path="$1" binary="${2:-}"
	local tests_json="" entry sep=""

	for entry in "${REPORT_ENTRIES[@]:-}"; do
		[[ -z "$entry" ]] && continue
		tests_json+="${sep}${entry}"
		sep=","
	done

	mkdir -p "$(dirname "$path")"
	cat >"$path" <<EOF
{
  "schema": 1,
  "platform": "$(_json_escape "${REPORT_PLATFORM}")",
  "ffmpeg_version": "$(_json_escape "$(_report_ffmpeg_version "${binary}")")",
  "host": {
    "os": "$(_json_escape "$(uname -s)")",
    "arch": "$(_json_escape "$(uname -m)")",
    "hostname": "$(_json_escape "$(uname -n)")",
    "kernel": "$(_json_escape "$(uname -r)")"
  },
  "capabilities": {
    "nvenc": "$(_json_escape "$(hw_capability nvenc)")",
    "amf": "$(_json_escape "$(hw_capability amf)")",
    "vpl": "$(_json_escape "$(hw_capability vpl)")",
    "videotoolbox": "$(_json_escape "$(hw_capability videotoolbox)")"
  },
  "totals": {
    "passed": $(_report_count passed),
    "failed": $(_report_count failed),
    "skipped": $(_report_count skipped),
    "total": ${#REPORT_ENTRIES[@]}
  },
  "tests": [${tests_json}]
}
EOF
}
