#!/bin/bash

cd /build/opus

OPUS_EXTRA_FLAGS=""
if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # opus enables ARM asm with runtime CPU detection, but has no detection
    # backend for Windows-on-ARM, so celt/arm/armcpu.c fails to compile with
    #   "Configured to use ARM asm but no CPU detection method available for
    #    your platform. Reconfigure with --disable-rtcd (or send patch)"
    # Taking upstream's own advice: drop runtime dispatch and use the ARM asm
    # selected at compile time. NEON is baseline on aarch64, so nothing is lost.
    OPUS_EXTRA_FLAGS="--disable-rtcd"
fi

./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --disable-extra-programs \
    ${OPUS_EXTRA_FLAGS} --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared --disable-extra-programs \
    ${OPUS_EXTRA_FLAGS} --host=${CROSS_PREFIX%-} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

make -j$(nproc) && make install
rm -rf /build/opus

add_enable "--enable-libopus"

exit 0
