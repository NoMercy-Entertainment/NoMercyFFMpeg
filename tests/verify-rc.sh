#!/usr/bin/env bash
# Verifies one Release Candidate archive on the machine that can actually run
# it, and writes a verdict the checklist bot can act on.
#
# Usage:
#   verify-rc.sh --archive <file> --manifest <manifest.json> --platform <name> \
#                --workdir <dir> --json <verdict.json> [--commit <sha>] [--tag <tag>]
#
# The integrity check is not decoration. The bot ticks a merge-blocking box
# based on this file, so the verdict has to say which bytes were tested — a
# report that cannot name its own artifact's digest proves nothing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

Archive=""
Manifest=""
Platform=""
WorkDir=""
JsonOut=""
Commit=""
Tag=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--archive)
		Archive="$2"
		shift 2
		;;
	--manifest)
		Manifest="$2"
		shift 2
		;;
	--platform)
		Platform="$2"
		shift 2
		;;
	--workdir)
		WorkDir="$2"
		shift 2
		;;
	--json)
		JsonOut="$2"
		shift 2
		;;
	--commit)
		Commit="$2"
		shift 2
		;;
	--tag)
		Tag="$2"
		shift 2
		;;
	*)
		echo "Error: unknown argument '$1'." >&2
		exit 2
		;;
	esac
done

# Spelled out rather than looped over ${!name} + ${name,,} — macOS still ships
# bash 3.2, where lowercase expansion is a syntax error.
[[ -n "$Archive" ]] || { echo "Error: --archive is required." >&2; exit 2; }
[[ -n "$Platform" ]] || { echo "Error: --platform is required." >&2; exit 2; }
[[ -n "$WorkDir" ]] || { echo "Error: --workdir is required." >&2; exit 2; }
[[ -n "$JsonOut" ]] || { echo "Error: --json is required." >&2; exit 2; }

# sha256 has a different tool name on every platform this fleet covers.
sha256_of() {
	local file="$1"
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$file" | awk '{print $1}'
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 "$file" | awk '{print $1}'
	elif command -v sha256 >/dev/null 2>&1; then
		sha256 -q "$file"
	else
		echo "Error: no sha256 tool found on this host." >&2
		return 1
	fi
}

fail_verdict() {
	local stage="$1" detail="$2"
	mkdir -p "$(dirname "$JsonOut")"
	cat >"$JsonOut" <<EOF
{
  "schema": 1,
  "platform": "${Platform}",
  "tag": "${Tag}",
  "commit": "${Commit}",
  "verdict": "fail",
  "failed_stage": "${stage}",
  "detail": "${detail//\"/\'}",
  "integrity": { "verified": false },
  "report": null
}
EOF
	echo "❌ ${stage}: ${detail}" >&2
	exit 1
}

echo "─── Verifying $(basename "$Archive") for ${Platform} ───"

[[ -f "$Archive" ]] || fail_verdict download "archive not found: ${Archive}"

AssetName="$(basename "$Archive")"
ActualSha="$(sha256_of "$Archive")" || fail_verdict integrity "no sha256 tool available"

# The manifest is the release's own record of what was published. Checking the
# archive against it catches a truncated download and a swapped asset alike.
ExpectedSha=""
if [[ -n "$Manifest" && -f "$Manifest" ]]; then
	ExpectedSha=$(
		tr -d '\n\t ' <"$Manifest" |
			grep -o "\"name\":\"${AssetName}\",\"sha256\":\"[0-9a-f]*\"" |
			grep -o '[0-9a-f]\{64\}' | head -1
	)
	[[ -n "$ExpectedSha" ]] || fail_verdict integrity "manifest.json has no entry for ${AssetName}"
	[[ "$ExpectedSha" == "$ActualSha" ]] ||
		fail_verdict integrity "sha256 mismatch — manifest ${ExpectedSha}, downloaded ${ActualSha}"
	echo "✅ sha256 matches manifest: ${ActualSha}"
else
	echo "::warning::no manifest supplied — recording the digest without cross-checking it"
fi

rm -rf "$WorkDir"
mkdir -p "$WorkDir"
case "$AssetName" in
*.tar.gz) tar -xzf "$Archive" -C "$WorkDir" || fail_verdict extract "tar failed" ;;
*.zip) unzip -oq "$Archive" -d "$WorkDir" || fail_verdict extract "unzip failed" ;;
*) fail_verdict extract "unsupported archive type: ${AssetName}" ;;
esac

[[ -f "${WorkDir}/ffmpeg" ]] || fail_verdict extract "no ffmpeg binary inside ${AssetName}"
chmod +x "${WorkDir}/ffmpeg" "${WorkDir}/ffprobe" 2>/dev/null || true

# A build for another OS or CPU can load and then die in ways that look like a
# codec failure. Refuse to report on a binary this host cannot execute at all.
if ! "${WorkDir}/ffmpeg" -hide_banner -version >/dev/null 2>&1; then
	fail_verdict exec "the ${Platform} binary does not execute on $(uname -s)/$(uname -m)"
fi

# Record whether the binary ran natively. darwin-x86_64 on an Apple Silicon Mac
# is a Rosetta 2 translation, and "tested on real hardware" has to say so —
# otherwise the evidence reads as an x86 Mac nobody owns.
TargetArch="${Platform##*-}"
HostArch="$(uname -m)"
case "$HostArch" in
amd64) HostArch="x86_64" ;;
aarch64) HostArch="arm64" ;;
esac
case "$TargetArch" in
aarch64) TargetArch="arm64" ;;
esac

Native="true"
Translation="null"
if [[ "$TargetArch" != "$HostArch" ]]; then
	Native="false"
	if [[ "$(uname -s)" == "Darwin" ]]; then
		Translation='"rosetta2"'
	else
		Translation='"unknown"'
	fi
	echo "ℹ️  ${Platform} is running translated on ${HostArch}"
fi

ReportPath="${WorkDir}/report.json"
bash "${HERE}/tests.sh" "$WorkDir" --platform "$Platform" --json "$ReportPath"
SuiteExit=$?

[[ -f "$ReportPath" ]] || fail_verdict suite "tests.sh produced no report (exit ${SuiteExit})"

Verdict="fail"
[[ "$SuiteExit" -eq 0 ]] && Verdict="pass"

mkdir -p "$(dirname "$JsonOut")"
{
	printf '{\n'
	printf '  "schema": 1,\n'
	printf '  "platform": "%s",\n' "$Platform"
	printf '  "tag": "%s",\n' "$Tag"
	printf '  "commit": "%s",\n' "$Commit"
	printf '  "verdict": "%s",\n' "$Verdict"
	printf '  "suite_exit": %s,\n' "$SuiteExit"
	printf '  "integrity": {\n'
	printf '    "verified": %s,\n' "$([[ -n "$ExpectedSha" ]] && echo true || echo false)"
	printf '    "asset": "%s",\n' "$AssetName"
	printf '    "sha256": "%s"\n' "$ActualSha"
	printf '  },\n'
	printf '  "execution": {\n'
	printf '    "native": %s,\n' "$Native"
	printf '    "translation": %s,\n' "$Translation"
	printf '    "host_arch": "%s"\n' "$HostArch"
	printf '  },\n'
	printf '  "report": '
	cat "$ReportPath"
	printf '\n}\n'
} >"$JsonOut"

echo "📄 Verdict (${Verdict}) written to ${JsonOut}"
exit "$SuiteExit"
