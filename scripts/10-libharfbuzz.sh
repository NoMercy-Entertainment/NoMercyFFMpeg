#!/bin/bash

cd /build/harfbuzz

if [[ "${TARGET_OS}" == "darwin" ]]; then
	mkdir -p /build/harfbuzz/src/unicode
	cp -r /usr/include/unicode/* /build/harfbuzz/src/unicode
fi

# Configure HarfBuzz with CoreText support for Darwin
HARFBUZZ_EXTRA_FLAGS=""
if [[ "${TARGET_OS}" == "darwin" ]]; then
	# HarfBuzz 14.x added a GPU feature (default enabled) whose util/meson.build
	# requires an Objective-C++ (objcpp) compiler on darwin — which our Meson cross
	# file doesn't define, so configure aborts. The GPU demo tool isn't needed for
	# libharfbuzz, so disable it on darwin to drop the objcpp requirement.
	#
	# NOTE: do NOT inject objcpp into the shared /build/cross_file.txt here — later
	# steps (e.g. librsvg's glib/cairo) copy that file and add objcpp themselves,
	# which would then double-define it ("objcpp ... already exists"). Disabling the
	# gpu feature is sufficient and self-contained.
	HARFBUZZ_EXTRA_FLAGS="-Dcoretext=enabled -Dgpu=disabled"
fi

if [[ "${TARGET_OS}" == "windows" && "${ARCH}" == "aarch64" ]]; then
	# This target builds with llvm-mingw (clang). src/hb.hh promotes 35 warnings
	# to hard errors via "#pragma GCC diagnostic error", and clang warns about
	# things GCC does not -- -Wunused-template fires throughout hb-algs.hh, so
	# upstream's own headers fail to compile. The code is fine; only the warning
	# set differs.
	#
	# It has to be patched in the source: a diagnostic pragma overrides the
	# command line, so neither -Wno-unused-template nor meson's -Dwerror=false
	# has any effect (the compile line never carries -Werror to begin with).
	# Downgrade error -> warning so they stay visible without being fatal.
	sed -i 's|^#pragma GCC diagnostic error|#pragma GCC diagnostic warning|' src/hb.hh
	grep -q '^#pragma GCC diagnostic error' src/hb.hh && { log -a "harfbuzz: hb.hh pragma downgrade failed"; exit 1; }
fi

meson build --prefix=${PREFIX} --buildtype=release -Ddefault_library=static \
	${HARFBUZZ_EXTRA_FLAGS} \
	--cross-file="/build/cross_file.txt" | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
	exit 1
fi

ninja -j$(nproc) -C build && ninja -C build install
rm -rf /build/harfbuzz

add_enable "--enable-libharfbuzz"

exit 0
