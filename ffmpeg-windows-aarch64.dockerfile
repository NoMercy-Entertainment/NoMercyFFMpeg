# Create a Windows ffmpeg build
# CI pins this to the base built in the same run (BASE_TAG=<commit sha>);
# local/compose builds use the default "latest". See .github/workflows.
ARG BASE_TAG=latest
FROM nomercyentertainment/ffmpeg-base:${BASE_TAG} AS windows

LABEL maintainer="Phillippe Pelzer"
LABEL version="1.0.1"
LABEL description="FFmpeg for Windows arm64"

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Update and install dependencies
RUN echo "------------------------------------------------------" \
    && echo "        _   _       __  __                      " \
    && echo "       | \ | | ___ |  \/  | ___ _ __ ___ _   _  " \
    && echo "       |  \| |/ _ \| |\/| |/ _ \ '__/ __| | | | " \
    && echo "       | |\  | (_) | |  | |  __/ | | (__| |_| | " \
    && echo "       |_| \_|\___/|_|  |_|\___|_|  \___|\__, | " \
    && echo "         _____ _____ __  __ ____  _____ _|___/  " \
    && echo "        |  ___|  ___|  \/  |  _ \| ____/ ___|   " \
    && echo "        | |_  | |_  | |\/| | |_) |  _|| |  _    " \
    && echo "        |  _| |  _| | |  | |  __/| |__| |_| |   " \
    && echo "        |_|   |_|   |_|  |_|_|   |_____\____|   " \
    && echo "" \
    && echo "------------------------------------------------------" \
    && echo "📦 Start FFmpeg for Windows arm64 build" \
    && echo "------------------------------------------------------" \
    && echo "🔧 Start downloading and installing dependencies" \
    && echo "------------------------------------------------------"\
    && echo "🔄 Checking for updates" \
    && apt-get update >/dev/null 2>&1 \
    && echo "✅ Updating completed successfully" \
    && echo "------------------------------------------------------" \
    && echo "🔧 Installing dependencies" \
    && apt-get install -y --no-install-recommends \
    mingw-w64 libgit2-dev zip openjdk-11-jdk ant >/dev/null 2>&1 \
    && apt-get autoremove -y >/dev/null 2>&1 && apt-get autoclean -y >/dev/null 2>&1 && apt-get clean -y >/dev/null 2>&1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && echo "✅ Installations completed successfully" \
    && echo "------------------------------------------------------"

ENV PREFIX=/ffmpeg_build/windows

# Windows-on-ARM toolchain: llvm-mingw (clang/LLD), not GCC.
#
# This used to build GCC from Windows-on-ARM-Experiments/mingw-woarm64-build.
# That toolchain is research-grade and produced two hard blockers:
#   1. ICE in the SEH backend on -fstack-protector-strong
#      (gcc/config/mingw/winnt.cc:1492 seh_pattern_emit), so every library had
#      to be built without stack-protector hardening.
#   2. Its own CRT header does not parse in some translation units --
#      _mingw.h:634 leaves __MINGW_FASTFAIL_INLINE unexpanded, which broke
#      fribidi (and therefore libass, which requires it).
#
# llvm-mingw is the toolchain production Windows-on-ARM builds actually use.
# It ships prebuilt (~85MB, no source build), provides the full
# aarch64-w64-mingw32-* tool set including gcc/g++ wrappers so CROSS_PREFIX and
# the shared scripts work unchanged, and supports stack-protector properly.
# UCRT rather than MSVCRT: Windows-on-ARM is UCRT-only.
#
# Its C++ runtime is libc++, and it ships no libstdc++ at all. Many component
# scripts write "Libs.private: -lstdc++" into their .pc files (x265, libpng,
# giflib, lept, tesseract, spirv-cross, libplacebo, ...), which is correct for
# every GCC platform. On this target -lstdc++ resolves to nothing, so any
# pkg-config link test using those .pc files fails -- surfacing misleadingly as
# FFmpeg's "ERROR: x265 not found using pkg-config" even though x265 built and
# installed fine. Aliasing libstdc++.a to libc++.a below makes those flags
# resolve to this toolchain's real C++ runtime, which is what they mean, and
# keeps every shared script untouched.
ARG LLVM_MINGW_VERSION=20260728
ENV LLVM_MINGW_DIR=/opt/llvm-mingw
RUN echo "------------------------------------------------------" \
    && echo "🔧 Start downloading llvm-mingw ${LLVM_MINGW_VERSION} (Windows-on-ARM toolchain)" \
    && TARBALL="llvm-mingw-${LLVM_MINGW_VERSION}-ucrt-ubuntu-22.04-x86_64.tar.xz" \
    && curl -fsSL --retry 5 --retry-delay 5 -o /tmp/llvm-mingw.tar.xz \
        "https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_MINGW_VERSION}/${TARBALL}" \
    && mkdir -p ${LLVM_MINGW_DIR} \
    && tar -xJf /tmp/llvm-mingw.tar.xz -C ${LLVM_MINGW_DIR} --strip-components=1 \
    && rm -f /tmp/llvm-mingw.tar.xz \
    && echo "🔧 Verifying the toolchain" \
    && ${LLVM_MINGW_DIR}/bin/aarch64-w64-mingw32-gcc --version | head -1 \
    && printf '#include <stdlib.h>\nint main(void){return 0;}\n' > /tmp/probe.c \
    && ${LLVM_MINGW_DIR}/bin/aarch64-w64-mingw32-gcc -O2 -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
        -c /tmp/probe.c -o /tmp/probe.o \
    && rm -f /tmp/probe.c /tmp/probe.o \
    && echo "🔧 Aliasing libstdc++ -> libc++" \
    && for d in ${LLVM_MINGW_DIR}/aarch64-w64-mingw32/lib; do \
        [ -f "$d/libc++.a" ] || { echo "libc++.a not found in $d"; exit 1; }; \
        ln -sf libc++.a "$d/libstdc++.a"; \
    done \
    && printf 'int main(void){return 0;}\n' > /tmp/probe.cpp \
    && ${LLVM_MINGW_DIR}/bin/aarch64-w64-mingw32-g++ /tmp/probe.cpp -lstdc++ -o /tmp/probe.exe \
    && rm -f /tmp/probe.cpp /tmp/probe.exe \
    && echo "✅ llvm-mingw installed; stack-protector and -lstdc++ both link" \
    && echo "------------------------------------------------------"

# Install Rust and Cargo
RUN echo "------------------------------------------------------" \
    && echo "🔄 Start installing Rust and Cargo" \
    && rustup target add aarch64-pc-windows-gnullvm >/dev/null 2>&1 \
    && cargo install cargo-c >/dev/null 2>&1 \
    && echo "✅ Installations completed successfully" \
    && echo "------------------------------------------------------"

RUN cd /build

# Set environment variables for building ffmpeg
ENV TARGET_OS=windows
ENV PREFIX=/ffmpeg_build/windows
ENV ARCH=aarch64
ENV CROSS_PREFIX=${ARCH}-w64-mingw32-
ENV CC=${CROSS_PREFIX}gcc
ENV CXX=${CROSS_PREFIX}g++
ENV LD=${CROSS_PREFIX}ld
# llvm-mingw ships ar/ranlib/nm directly; the gcc-* LTO wrappers are GCC-only
# and do not exist here, so use the plain tools.
ENV AR=${CROSS_PREFIX}ar
ENV RANLIB=${CROSS_PREFIX}ranlib
ENV STRIP=${CROSS_PREFIX}strip
ENV NM=${CROSS_PREFIX}nm
ENV WINDRES=${CROSS_PREFIX}windres
ENV DLLTOOL=${CROSS_PREFIX}dlltool
ENV STAGE_CFLAGS="-fno-semantic-interposition" 
ENV STAGE_CXXFLAGS="-fno-semantic-interposition"
ENV PKG_CONFIG=pkg-config
ENV PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig
ENV PATH="${PREFIX}/bin:${LLVM_MINGW_DIR}/bin:${PATH}"
# -fstack-protector-strong is back, matching every other platform: the GCC ICE
# that forced it out was a mingw-woarm64-build defect and clang handles it (the
# toolchain install step above proves it compiles before we get this far).
# The old -I/-L into ${PREFIX}/aarch64-w64-mingw32 are gone with that toolchain;
# llvm-mingw carries its own sysroot under ${LLVM_MINGW_DIR}.
# -static-libgcc/-static-libstdc++ are deliberately NOT in CFLAGS/CXXFLAGS here,
# only in LDFLAGS. They are link-time flags: GCC ignores them silently when
# compiling, but clang emits -Wunused-command-line-argument, and many configure
# probes deliberately run with -Werror to make warnings fail the test. That
# turns a harmless flag into a wrong answer:
#   * libgcrypt configure.ac:1398 sets -Werror before its '__thread' probe, so
#     detection said "no", HAVE_GCC_STORAGE_CLASS__THREAD went undefined, and
#     fips.c failed with "use of undeclared identifier 'the_tc'".
#   * meson's "usable header" check reported stdatomic.h unusable, surfacing as
#     libvmaf's "Atomics not supported".
# Linking still gets them via LDFLAGS.
ENV CFLAGS="-I${PREFIX}/include -O2 -pipe -D_FORTIFY_SOURCE=2 -fstack-protector-strong"
ENV CXXFLAGS="-I${PREFIX}/include -O2 -pipe -D_FORTIFY_SOURCE=2 -fstack-protector-strong"
ENV LDFLAGS="-static-libgcc -static-libstdc++ -L${PREFIX}/lib -O2 -pipe -fstack-protector-strong"

# Create the build directory
RUN mkdir -p ${PREFIX}

# Create Meson cross file for Windows
RUN echo "[binaries]" > /build/cross_file.txt && \
    echo "c = '${CC}'" >> /build/cross_file.txt && \
    echo "cpp = '${CXX}'" >> /build/cross_file.txt && \
    echo "ld = '${LD}'" >> /build/cross_file.txt && \
    echo "ar = '${AR}'" >> /build/cross_file.txt && \
    echo "ranlib = '${RANLIB}'" >> /build/cross_file.txt && \
    echo "strip = '${STRIP}'" >> /build/cross_file.txt && \
    echo "nm = '${NM}'" >> /build/cross_file.txt && \
    echo "windres = '${WINDRES}'" >> /build/cross_file.txt && \
    echo "dlltool = '${DLLTOOL}'" >> /build/cross_file.txt && \
    echo "pkgconfig = '${PKG_CONFIG}'" >> /build/cross_file.txt && \
    echo "pkg-config = '${PKG_CONFIG}'" >> /build/cross_file.txt && \
    echo "" >> /build/cross_file.txt && \
    echo "[host_machine]" >> /build/cross_file.txt && \
    echo "system = 'windows'" >> /build/cross_file.txt && \
    echo "cpu_family = '${ARCH}'" >> /build/cross_file.txt && \
    echo "cpu = '${ARCH}'" >> /build/cross_file.txt && \
    echo "endian = 'little'" >> /build/cross_file.txt && \
    echo "" >> /build/cross_file.txt && \
    echo "[properties]" >> /build/cross_file.txt && \
    echo "c_args = ['-I${PREFIX}/include', '-O2', '-pipe', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong']" >> /build/cross_file.txt && \
    echo "cpp_args = ['-I${PREFIX}/include', '-O2', '-pipe', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong']" >> /build/cross_file.txt && \
    echo "c_link_args = ['-static-libgcc', '-static-libstdc++', '-L${PREFIX}/lib', '-O2', '-pipe', '-fstack-protector-strong']" >> /build/cross_file.txt && \
    echo "cpp_link_args = ['-static-libgcc', '-static-libstdc++', '-L${PREFIX}/lib', '-O2', '-pipe', '-fstack-protector-strong']" >> /build/cross_file.txt

ENV CMAKE_COMMON_ARG="-DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_RC_COMPILER=${WINDRES} -DENABLE_SHARED=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release"

# Create the build directory
RUN mkdir -p ${PREFIX}

ENV FFMPEG_ENABLES="" \
    FFMPEG_CFLAGS="" \
    FFMPEG_LDFLAGS="" \
    FFMPEG_EXTRA_LIBFLAGS=""

# Copy the build scripts
COPY ./scripts /scripts

# Convert Windows line endings to Unix line endings
RUN find /scripts -type f -name "*.sh" -exec sed -i 's/\r$//' {} +

# Initialize the build
RUN touch /build/enable.txt /build/cflags.txt /build/ldflags.txt /build/extra_libflags.txt \
    && chmod +x /scripts/init/init.sh \
    && /scripts/init/init.sh \
    || (echo "❌ FFmpeg build failed" ; exit 1)

# Copy the dev scripts
COPY ./dev /test

# Convert Windows line endings to Unix line endings
RUN find /test -type f -name "*.sh" -exec sed -i 's/\r$//' {} +

# Run the dev scripts to build dependencies
RUN chmod +x /test/init/dev.sh \
    && /test/init/dev.sh \
    || (echo "❌ FFmpeg build failed" ; exit 1)

# ffmpeg
RUN FFMPEG_ENABLES=$(cat /build/enable.txt) export FFMPEG_ENABLES \
    && CFLAGS="${CFLAGS} $(cat /build/cflags.txt)" export CFLAGS \
    && LDFLAGS="${LDFLAGS} $(cat /build/ldflags.txt)" export LDFLAGS \
    && FFMPEG_EXTRA_LIBFLAGS="-lpthread -lm $(cat /build/extra_libflags.txt)" export FFMPEG_EXTRA_LIBFLAGS \
    && echo "------------------------------------------------------" \
    && echo "🚧 Start building FFmpeg" \
    && echo "------------------------------------------------------" \
    && cd /build/ffmpeg \
    && echo "⚙️ Configure FFmpeg                              [1/2]" \
    && ./configure --pkg-config-flags=--static \
    --arch=${ARCH} \
    --target-os=mingw32 \
    --cross-prefix=${CROSS_PREFIX} \
    --pkg-config=pkg-config \
    --prefix=${PREFIX} \
    --enable-cross-compile \
    --disable-shared \
    --enable-ffplay \
    --enable-static \
    --enable-gpl \
    --enable-version3 \
    --enable-nonfree \
    ${FFMPEG_ENABLES} \
    --enable-filter=all \
    --enable-runtime-cpudetect \
    --extra-version="NoMercy-MediaServer" \
    --extra-cflags="-static -static-libgcc -static-libstdc++" \
    --extra-ldflags="-static -static-libgcc -static-libstdc++" \
    --extra-libs="${FFMPEG_EXTRA_LIBFLAGS}" >/ffmpeg_build.log 2>&1 \
    || (cat "/ffmpeg_build.log" ; echo "--- ffbuild/config.log (tail) ---" ; tail -80 ffbuild/config.log 2>/dev/null ; echo "❌ FFmpeg build failed" ; false) \
    && echo "🛠️ Building FFmpeg                               [2/2]" \
    && make -j$(nproc) >/ffmpeg_build.log 2>&1 || (cat "/ffmpeg_build.log" ; echo "❌ FFmpeg build failed" ; exit 1) && make install >/dev/null 2>&1 \
    && rm -rf /build/ffmpeg \
    && echo "------------------------------------------------------" \
    && echo "✅ FFmpeg was built successfully" \
    && echo "------------------------------------------------------" 

RUN chmod +x /scripts/init/package.sh && /scripts/init/package.sh

FROM alpine:latest AS final

COPY --from=windows /output/ffmpeg-9.0-windows-aarch64.zip /build/ffmpeg-9.0-windows-aarch64.zip

CMD ["cp", "/build/ffmpeg-9.0-windows-aarch64.zip", "/output"]
