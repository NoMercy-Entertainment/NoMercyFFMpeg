#!/usr/bin/env bash
# Hardware-capability probe shared by tests/tests.sh and the automated RC
# verifier. Answers one question per feature — "can this host actually exercise
# it?" — without ever invoking ffmpeg.
#
# Asking ffmpeg would make the build its own oracle: a broken encoder would look
# identical to absent hardware and skip itself green. Every probe here reads the
# machine instead.
#
# Usage:  hw_capability nvenc   ->  prints "available" or "absent:<reason>"
#
# Detection is dependency-free on purpose. lspci is absent from minimal
# containers, and the previous suite treated "lspci missing" as "no GPU", so
# sysfs is read directly where possible.

HW_VENDOR_NVIDIA="0x10de"
HW_VENDOR_AMD="0x1002"
HW_VENDOR_INTEL="0x8086"

# Class 0x03xxxx is "display controller" in the PCI spec.
_hw_has_pci_display_vendor() {
	local want="$1"
	local dev vendor class

	case "$(uname -s)" in
	Linux)
		for dev in /sys/bus/pci/devices/*; do
			[[ -r "${dev}/vendor" && -r "${dev}/class" ]] || continue
			vendor=$(<"${dev}/vendor")
			class=$(<"${dev}/class")
			[[ "${vendor}" == "${want}" && "${class}" == 0x03* ]] && return 0
		done
		return 1
		;;
	FreeBSD)
		pciconf -lv 2>/dev/null | grep -qi "vendor=${want#0x}" && return 0
		return 1
		;;
	*)
		return 1
		;;
	esac
}

# Plain glob expansion rather than compgen -G, so this stays valid on the bash
# 3.2 that macOS still ships.
_hw_find_lib() {
	local pattern="$1"
	local dir candidate
	for dir in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu \
		/usr/local/lib /opt/amdgpu-pro/lib/x86_64-linux-gnu; do
		[[ -d "$dir" ]] || continue
		for candidate in ${dir}/${pattern}; do
			[[ -e "$candidate" ]] && return 0
		done
	done
	ldconfig -p 2>/dev/null | grep -q "${pattern%%\**}" && return 0
	return 1
}

_hw_capability_nvenc() {
	case "$(uname -s)" in
	Darwin)
		echo "absent:macOS has no NVIDIA driver stack"
		return
		;;
	esac

	# nvidia-smi is definitive — it only succeeds when the driver is loaded AND
	# a device answers. Fall back to the bus when the tool is not installed.
	if command -v nvidia-smi >/dev/null 2>&1; then
		if nvidia-smi -L 2>/dev/null | grep -q "^GPU 0:"; then
			echo "available"
		else
			echo "absent:nvidia-smi reports no usable GPU"
		fi
		return
	fi

	if _hw_has_pci_display_vendor "${HW_VENDOR_NVIDIA}"; then
		if _hw_find_lib "libnvidia-encode.so*"; then
			echo "available"
		else
			echo "absent:NVIDIA GPU present but libnvidia-encode is missing"
		fi
		return
	fi

	echo "absent:no NVIDIA display adapter on the PCI bus"
}

_hw_capability_amf() {
	case "$(uname -s)" in
	Darwin | FreeBSD)
		echo "absent:AMF is not shipped for $(uname -s)"
		return
		;;
	esac

	if ! _hw_has_pci_display_vendor "${HW_VENDOR_AMD}"; then
		echo "absent:no AMD display adapter on the PCI bus"
		return
	fi
	if ! _hw_find_lib "libamfrt64.so*"; then
		echo "absent:AMD GPU present but the AMF runtime is missing"
		return
	fi
	echo "available"
}

_hw_capability_vpl() {
	case "$(uname -s)" in
	Darwin | FreeBSD)
		echo "absent:oneVPL is not shipped for $(uname -s)"
		return
		;;
	esac

	if ! _hw_has_pci_display_vendor "${HW_VENDOR_INTEL}"; then
		echo "absent:no Intel display adapter on the PCI bus"
		return
	fi
	if ! _hw_find_lib "libvpl.so*" && ! _hw_find_lib "libmfx.so*"; then
		echo "absent:Intel GPU present but neither libvpl nor libmfx is installed"
		return
	fi
	echo "available"
}

_hw_capability_videotoolbox() {
	if [[ "$(uname -s)" == "Darwin" ]]; then
		echo "available"
	else
		echo "absent:VideoToolbox is macOS-only"
	fi
}

hw_capability() {
	case "$1" in
	nvenc) _hw_capability_nvenc ;;
	amf) _hw_capability_amf ;;
	vpl) _hw_capability_vpl ;;
	videotoolbox) _hw_capability_videotoolbox ;;
	*)
		echo "absent:unknown capability '$1'"
		return 1
		;;
	esac
}

hw_capability_available() {
	[[ "$(hw_capability "$1")" == "available" ]]
}

# Succeeds when the feature is usable. Otherwise prints why not and fails, so
# callers can write `if reason=$(hw_capability_reason nvenc); then`.
hw_capability_reason() {
	local answer
	answer="$(hw_capability "$1")"
	if [[ "$answer" == "available" ]]; then
		return 0
	fi
	printf '%s' "${answer#absent:}"
	return 1
}

# Running this file directly prints the whole capability table — the fleet
# health check and anyone debugging a runner use it that way.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	for _feature in nvenc amf vpl videotoolbox; do
		printf '%-14s %s\n' "${_feature}" "$(hw_capability "${_feature}")"
	done
fi
