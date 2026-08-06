#!/bin/bash

LIBRAV1E_TARGET=""
if [[ "${ARCH}" == "aarch64" && "${TARGET_OS}" == "linux" ]]; then
    LIBRAV1E_TARGET="--target=${ARCH}-unknown-linux-gnu"
elif [[ "${ARCH}" == "x86_64" && "${TARGET_OS}" == "windows" ]]; then
    LIBRAV1E_TARGET="--target=${ARCH}-pc-windows-gnu"
    export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=${CC}
elif [[ "${ARCH}" == "aarch64" && "${TARGET_OS}" == "windows" ]]; then
    # gnullvm, not msvc: the toolchain is llvm-mingw, so there is no MSVC linker
    # here. gnullvm is the Rust target that pairs with llvm-mingw, and it links
    # through the same aarch64-w64-mingw32 driver everything else uses.
    LIBRAV1E_TARGET="--target=${ARCH}-pc-windows-gnullvm"
    export CARGO_TARGET_AARCH64_PC_WINDOWS_GNULLVM_LINKER=${CC}
elif [[ "${ARCH}" == "arm64" && "${TARGET_OS}" == "darwin" ]]; then
    LIBRAV1E_TARGET="--target=aarch64-apple-darwin"
elif [[ "${ARCH}" == "x86_64" && "${TARGET_OS}" == "darwin" ]]; then
    LIBRAV1E_TARGET="--target=${ARCH}-apple-darwin"
elif [[ "${ARCH}" == "x86_64" && "${TARGET_OS}" == "freebsd" ]]; then
    LIBRAV1E_TARGET="--target=${ARCH}-unknown-freebsd"
    export CARGO_TARGET_X86_64_UNKNOWN_FREEBSD_LINKER=${CC}
fi

OLD_LDFLAGS="${LDFLAGS}"
OLD_RUSTFLAGS="${RUSTFLAGS}"
# Set RUSTFLAGS for Darwin to suppress duplicate symbol warnings
if [[ "${TARGET_OS}" == "darwin" ]]; then
	export LDFLAGS="${LDFLAGS} -Wl,-dead_strip_dylibs -Wl,-multiply_defined,suppress"
    export RUSTFLAGS="${RUSTFLAGS} -C link-arg=-Wl,-dead_strip_dylibs -C link-arg=-Wl,-multiply_defined,suppress -C panic=abort -C lto=thin -C opt-level=3"
fi

cd /build/librav1e

if [[ "${TARGET_OS}" == "windows" ]] \
    || [[ "${TARGET_OS}" == "linux" && "${ARCH}" == "aarch64" ]] \
    || [[ "${TARGET_OS}" == "darwin" ]] \
    || [[ "${TARGET_OS}" == "freebsd" ]]; then
    sed -i 's/,[[:space:]]*"git_version"//' /build/librav1e/Cargo.toml
fi

cargo cinstall -j$(nproc) -v ${LIBRAV1E_TARGET} --prefix=${PREFIX} --library-type=staticlib --crt-static --release 2>&1 | log -a
if [ ${PIPESTATUS[0]} -ne 0 ]; then
	LDFLAGS="${OLD_LDFLAGS}"
	RUSTFLAGS="${OLD_RUSTFLAGS}"
    exit 1
fi

if [[ "${ARCH}" == "x86_64" && "${TARGET_OS}" == "linux" ]]; then
    sed -i 's/-lgcc_s//' ${PREFIX}/lib/x86_64-linux-gnu/pkgconfig/rav1e.pc
    cp ${PREFIX}/lib/x86_64-linux-gnu/pkgconfig/rav1e.pc ${PREFIX}/lib/pkgconfig/rav1e.pc
else
    sed -i 's/-lgcc_s//' ${PREFIX}/lib/pkgconfig/rav1e.pc
fi

# Rust std for the windows-gnu target pulls in these system libs; without them
# FFmpeg's static pkg-config link test against rav1e fails on Windows.
if [[ "${TARGET_OS}" == "windows" ]]; then
    if ! grep -q "Libs.private:" ${PREFIX}/lib/pkgconfig/rav1e.pc; then
        echo "Libs.private: -lws2_32 -lbcrypt -luserenv -lntdll -lkernel32 -ladvapi32" >>${PREFIX}/lib/pkgconfig/rav1e.pc
    fi
fi

# Rust std for the freebsd target pulls in these system libs (all present as
# static archives in the FreeBSD base system); without them FFmpeg's static
# pkg-config link test against rav1e fails.
if [[ "${TARGET_OS}" == "freebsd" ]]; then
    if ! grep -q "Libs.private:" ${PREFIX}/lib/pkgconfig/rav1e.pc; then
        echo "Libs.private: -lexecinfo -lpthread -lm -lrt -lutil" >>${PREFIX}/lib/pkgconfig/rav1e.pc
    fi
fi
LDFLAGS="${OLD_LDFLAGS}"
RUSTFLAGS="${OLD_RUSTFLAGS}"

rm -rf /build/librav1e

add_enable "--enable-librav1e"

exit 0
