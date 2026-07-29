# Create a Freebsd ffmpeg build
# CI pins this to the base built in the same run (BASE_TAG=<commit sha>);
# local/compose builds use the default "latest". See .github/workflows.
ARG BASE_TAG=latest
FROM nomercyentertainment/ffmpeg-base:${BASE_TAG} AS freebsd

LABEL maintainer="Phillippe Pelzer"
LABEL version="1.0.1"
LABEL description="FFmpeg for Freebsd x86_64"

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
    && echo "📦 Start FFmpeg for Freebsd x86_64 build" \
    && echo "------------------------------------------------------" \
    && echo "🔧 Start downloading and installing dependencies" \
    && echo "------------------------------------------------------"\
    && echo "🔄 Checking for updates" \
    && apt-get update >/dev/null 2>&1 \
    && echo "✅ Updating completed successfully" \
    && echo "------------------------------------------------------" \
    && echo "🔧 Installing dependencies" \
    && apt-get install -y --no-install-recommends \
    g++ gcc clang lld llvm xz-utils openjdk-11-jdk ant >/dev/null 2>&1 \
    && apt-get upgrade -y >/dev/null 2>&1 && apt-get autoremove -y >/dev/null 2>&1 && apt-get autoclean -y >/dev/null 2>&1 && apt-get clean -y >/dev/null 2>&1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && echo "✅ Installations completed successfully" \
    && echo "------------------------------------------------------"

# Set environment variables for building ffmpeg
ENV TARGET_OS=freebsd
ENV PREFIX=/ffmpeg_build/freebsd
ENV ARCH=x86_64
ENV FREEBSD_VERSION=14.3
ENV SYSROOT=/opt/freebsd-sysroot
ENV CROSS_PREFIX=${ARCH}-unknown-freebsd14-
ENV CC=${CROSS_PREFIX}gcc
ENV CXX=${CROSS_PREFIX}g++
ENV LD=${CROSS_PREFIX}ld
ENV AR=${CROSS_PREFIX}gcc-ar
ENV RANLIB=${CROSS_PREFIX}gcc-ranlib
ENV STRIP=${CROSS_PREFIX}strip
ENV NM=${CROSS_PREFIX}gcc-nm
# ENV WINDRES=${CROSS_PREFIX}windres
# ENV DLLTOOL=${CROSS_PREFIX}dlltool
ENV STAGE_CFLAGS="-fvisibility=hidden -fno-semantic-interposition"
ENV STAGE_CXXFLAGS="-fvisibility=hidden -fno-semantic-interposition"
ENV PKG_CONFIG=pkg-config
ENV PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig
ENV PATH="${PREFIX}/bin:${PATH}"
ENV CFLAGS="-I${PREFIX}/include -O2 -pipe -fPIC -DPIC -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fstack-clash-protection -pthread"
ENV CXXFLAGS="-I${PREFIX}/include -O2 -pipe -fPIC -DPIC -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fstack-clash-protection -pthread"
ENV LDFLAGS="-L${PREFIX}/lib -O2 -pipe -fstack-protector-strong -fstack-clash-protection -Wl,-z,relro,-z,now -pthread -lm"

# Set up the FreeBSD sysroot and clang-based cross-toolchain
# (there is no apt cross-gcc for FreeBSD; clang + lld + the FreeBSD base
# system headers/libs provide the ${CROSS_PREFIX}* tools instead)
RUN echo "------------------------------------------------------" \
    && echo "🔧 Start setting up FreeBSD sysroot and toolchain" \
    && mkdir -p ${SYSROOT} \
    && wget -O /tmp/base.txz https://download.freebsd.org/releases/amd64/${FREEBSD_VERSION}-RELEASE/base.txz >/dev/null 2>&1 \
    && tar -xJf /tmp/base.txz -C ${SYSROOT} ./lib ./usr/lib ./usr/include ./usr/libdata >/dev/null 2>&1 \
    && rm -f /tmp/base.txz \
    # -Qunused-arguments: -fuse-ld=lld is unused in compile-only invocations and
    # meson probes with -Werror=unused-command-line-argument, failing every check
    && printf '#!/bin/sh\nexec clang --target=%s --sysroot=%s -fuse-ld=lld -Qunused-arguments "$@"\n' "${CROSS_PREFIX%-}" "${SYSROOT}" > /usr/local/bin/${CROSS_PREFIX}gcc \
    && printf '#!/bin/sh\nexec clang++ --target=%s --sysroot=%s -stdlib=libc++ -fuse-ld=lld -Qunused-arguments "$@"\n' "${CROSS_PREFIX%-}" "${SYSROOT}" > /usr/local/bin/${CROSS_PREFIX}g++ \
    && chmod +x /usr/local/bin/${CROSS_PREFIX}gcc /usr/local/bin/${CROSS_PREFIX}g++ \
    && ln -s /usr/bin/ld.lld /usr/local/bin/${CROSS_PREFIX}ld \
    && ln -s /usr/bin/llvm-ar /usr/local/bin/${CROSS_PREFIX}ar \
    && ln -s /usr/bin/llvm-ar /usr/local/bin/${CROSS_PREFIX}gcc-ar \
    && ln -s /usr/bin/llvm-ranlib /usr/local/bin/${CROSS_PREFIX}ranlib \
    && ln -s /usr/bin/llvm-ranlib /usr/local/bin/${CROSS_PREFIX}gcc-ranlib \
    && ln -s /usr/bin/llvm-nm /usr/local/bin/${CROSS_PREFIX}nm \
    && ln -s /usr/bin/llvm-nm /usr/local/bin/${CROSS_PREFIX}gcc-nm \
    && ln -s /usr/bin/llvm-strip /usr/local/bin/${CROSS_PREFIX}strip \
    && ln -s /usr/bin/llvm-objdump /usr/local/bin/${CROSS_PREFIX}objdump \
    && ln -s /usr/bin/llvm-strings /usr/local/bin/${CROSS_PREFIX}strings \
    && ln -s /usr/bin/llvm-size /usr/local/bin/${CROSS_PREFIX}size \
    && ln -s /usr/bin/llvm-readelf /usr/local/bin/${CROSS_PREFIX}readelf \
    && ln -s /usr/bin/llvm-objcopy /usr/local/bin/${CROSS_PREFIX}objcopy \
    # FreeBSD has no libstdc++ (libc++ + libcxxrt) and no libdl (dlopen is in
    # libc); the build scripts write -lstdc++/-ldl into .pc files, so shim them
    && printf 'INPUT(-lc++ -lcxxrt)\n' > ${SYSROOT}/usr/lib/libstdc++.a \
    && llvm-ar rc ${SYSROOT}/usr/lib/libdl.a \
    # libgcc_s is shared-only on FreeBSD; static equivalents are gcc + gcc_eh.
    # cmake-generated .pc files embed -lgcc_s from clang's implicit link libs
    && printf 'INPUT(-lgcc -lgcc_eh)\n' > ${SYSROOT}/usr/lib/libgcc_s.a \
    # no libatomic on FreeBSD (atomics live in compiler-rt); empty shim
    && llvm-ar rc ${SYSROOT}/usr/lib/libatomic.a \
    && ${CROSS_PREFIX}gcc --version >/dev/null 2>&1 \
    # Rust std for the FreeBSD target (librav1e is built with cargo-c)
    && rustup target add x86_64-unknown-freebsd >/dev/null 2>&1 \
    && echo "✅ FreeBSD sysroot and toolchain set up successfully" \
    && echo "------------------------------------------------------"

# Create Meson cross file for Freebsd
RUN echo "[binaries]" > /build/cross_file.txt && \
    echo "c = '${CC}'" >> /build/cross_file.txt && \
    echo "cpp = '${CXX}'" >> /build/cross_file.txt && \
    echo "ld = '${LD}'" >> /build/cross_file.txt && \
    echo "ar = '${AR}'" >> /build/cross_file.txt && \
    echo "ranlib = '${RANLIB}'" >> /build/cross_file.txt && \
    echo "strip = '${STRIP}'" >> /build/cross_file.txt && \
    echo "pkgconfig = '${PKG_CONFIG}'" >> /build/cross_file.txt && \
    echo "pkg-config = '${PKG_CONFIG}'" >> /build/cross_file.txt && \
    echo "" >> /build/cross_file.txt && \
    echo "[host_machine]" >> /build/cross_file.txt && \
    echo "system = 'freebsd'" >> /build/cross_file.txt && \
    echo "cpu_family = '${ARCH}'" >> /build/cross_file.txt && \
    echo "cpu = '${ARCH}'" >> /build/cross_file.txt && \
    echo "endian = 'little'" >> /build/cross_file.txt

ENV CMAKE_COMMON_ARG="-DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_SYSTEM_NAME=FreeBSD -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DENABLE_SHARED=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release"

# Create the build directory. Meson installs .pc files to libdata/pkgconfig
# for FreeBSD hosts; link it to lib/pkgconfig so pkg-config discovery and the
# scripts' .pc fixups keep working unchanged.
RUN mkdir -p ${PREFIX}/lib/pkgconfig ${PREFIX}/libdata \
    && ln -s ../lib/pkgconfig ${PREFIX}/libdata/pkgconfig

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
    # -lmd: FreeBSD's liblzma.a needs libmd's SHA256 at static link time
    && FFMPEG_EXTRA_LIBFLAGS="-lpthread -lm -lmd $(cat /build/extra_libflags.txt)" export FFMPEG_EXTRA_LIBFLAGS \
    && echo "------------------------------------------------------" \
    && echo "🚧 Start building FFmpeg" \
    && echo "------------------------------------------------------" \
    && cd /build/ffmpeg \
    && echo "⚙️ Configure FFmpeg                              [1/2]" \
    && ./configure --pkg-config-flags=--static \
    --arch=${ARCH} \
    --target-os=${TARGET_OS} \
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
    --extra-cflags="-static" \
    --extra-ldflags="-static" \
    --extra-libs="${FFMPEG_EXTRA_LIBFLAGS}" >/ffmpeg_build.log 2>&1 \
    || (cat "/ffmpeg_build.log" ; cat "/build/ffmpeg/ffbuild/config.log" 2>/dev/null ; echo "❌ FFmpeg build failed" ; false) \
    && echo "🛠️ Building FFmpeg                               [2/2]" \
    && make -j$(nproc) >/ffmpeg_build.log 2>&1 || (cat "/ffmpeg_build.log" ; cat "/build/ffmpeg/ffbuild/config.log" ; echo "❌ FFmpeg build failed" ; exit 1) && make install >/dev/null 2>&1 \
    && rm -rf /build/ffmpeg \
    && echo "------------------------------------------------------" \
    && echo "✅ FFmpeg was built successfully" \
    && echo "------------------------------------------------------" 

RUN chmod +x /scripts/init/package.sh && /scripts/init/package.sh

FROM alpine:latest AS final

COPY --from=freebsd /output/ffmpeg-8.1.2-freebsd-x86_64.tar.gz /build/ffmpeg-8.1.2-freebsd-x86_64.tar.gz

CMD ["cp", "/build/ffmpeg-8.1.2-freebsd-x86_64.tar.gz", "/output"]
