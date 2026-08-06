#!/bin/bash

if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # aec*.c pick an MSVC inline-asm implementation of aec_get_shift() behind
    #     #if SYS_WINDOWS && !ARCH_X86_64
    # which was written when "Windows but not x86_64" could only mean 32-bit
    # x86. On Windows-on-ARM it is also true, so an aarch64 build selects
    #     __asm { bsr eax, v }
    # i.e. x86 assembly, in a syntax clang does not accept at all:
    #     error: expected 'volatile', 'inline', 'goto', or '('
    # Narrow the condition to what it actually means so aarch64 takes the
    # portable C branch that already exists in the #else.
    # NB: these sources are CRLF, so the guard line ends "...ARCH_X86_64\r".
    # Do not anchor with $ -- it will not match, and the replacement silently
    # does nothing. Leaving the tail unanchored keeps the CR in place.
    for f in aec.c aec_ctx.c aec_rdo.c aec_vrdo.c aec_fastrdo.c; do
        sed -i 's|^#if SYS_WINDOWS && !ARCH_X86_64|#if SYS_WINDOWS \&\& !ARCH_X86_64 \&\& !defined(__aarch64__)|' \
            "/build/libxavs2/source/encoder/${f}"
    done
    # Assert on what the patch ADDS, not on the absence of the old pattern:
    # an absence check passes vacuously if the match never happened.
    for f in aec.c aec_ctx.c aec_rdo.c aec_vrdo.c aec_fastrdo.c; do
        grep -q 'defined(__aarch64__)' "/build/libxavs2/source/encoder/${f}" || {
            log "xavs2: aec x86-asm guard patch did not apply to ${f}"
            exit 1
        }
    done

    # Same hard error FreeBSD hits below: clang rejects the encoder's
    # thread-entry function pointer mismatch by default.
    export CFLAGS="${CFLAGS} -Wno-incompatible-function-pointer-types"
fi

if [[ ${TARGET_OS} == "freebsd" ]]; then
    # pthread_{set,get}affinity_np are declared in <pthread_np.h> on FreeBSD
    sed -i '0,/^#include/s//#include <pthread.h>\n#include <pthread_np.h>\n&/' /build/libxavs2/source/common/threadpool.c
    # clang makes the encoder's thread-entry function pointer mismatch a hard
    # error by default; configure --extra-cflags does not reach these objects
    export CFLAGS="${CFLAGS} -Wno-incompatible-function-pointer-types"
fi

if [[ ${TARGET_OS} == "darwin" ]]; then
    cp -r /build/libxavs2/build/linux /build/libxavs2/build/darwin
    cd /build/libxavs2/build/darwin

    if [[ ${ARCH} == "arm64" ]]; then
        CROSS_PREFIX="aarch64-apple-darwin24.1-"
    fi

    ./configure --prefix=${PREFIX} \
        --disable-cli --enable-static --enable-pic --disable-avs --disable-swscale --disable-lavf --disable-ffms --disable-gpac --disable-lsmash --extra-asflags="-w-macro-params-legacy" \
        --extra-cflags="-Wno-dev -Wno-typedef-redefinition -Wno-unused-but-set-variable -Wno-tautological-compare -Wno-format -Wno-incompatible-function-pointer-types" \
        --host=${CROSS_PREFIX%-} | log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
    fi
else
    CONF_FLAGS=""

    if [[ ${TARGET_OS} == "windows" ]]; then
        cp -r /build/libxavs2/build/linux /build/libxavs2/build/windows
        cd /build/libxavs2/build/windows
        if [[ ${ARCH} == "aarch64" ]]; then
            CONF_FLAGS="--disable-asm"
        fi
    else
        if [[ ${ARCH} == "aarch64" ]]; then
            CONF_FLAGS="--disable-asm"
        fi
        cd /build/libxavs2/build/linux
    fi

    ./configure --prefix=${PREFIX} \
        --disable-cli --enable-static --enable-pic --disable-avs --disable-swscale --disable-lavf --disable-ffms --disable-gpac --disable-lsmash --extra-asflags="-w-macro-params-legacy" \
        ${CONF_FLAGS} --host=${CROSS_PREFIX%-} \
        --cross-prefix=${CROSS_PREFIX} | log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
    fi
fi

make -j$(nproc) && make install | log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

rm -rf /build/libxavs2

add_enable "--enable-libxavs2"

exit 0
