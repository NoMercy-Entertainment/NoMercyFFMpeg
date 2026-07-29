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

RUN echo "------------------------------------------------------" \
    && echo "🔧 Start downloading Windows-on-ARM" \
    && git clone https://github.com/Windows-on-ARM-Experiments/mingw-woarm64-build.git >/dev/null 2>&1 \
    && echo "✅ Windows-on-ARM source code downloaded successfully" \
    && cd mingw-woarm64-build \
    && find . -type f -exec sed -i 's|sudo ||g' {} + \
    && echo "🔧 Start building Windows-on-ARM" \
    && TOOLCHAIN_PATH=${PREFIX}/aarch64-w64-mingw32 ./build.sh >/dev/null 2>&1 \
    && echo "✅ Windows-on-ARM installed successfully" \
    && echo "------------------------------------------------------"

# Install Rust and Cargo
RUN echo "------------------------------------------------------" \
    && echo "🔄 Start installing Rust and Cargo" \
    && rustup target add aarch64-pc-windows-msvc >/dev/null 2>&1 \
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
ENV AR=${CROSS_PREFIX}gcc-ar
ENV RANLIB=${CROSS_PREFIX}gcc-ranlib
ENV STRIP=${CROSS_PREFIX}strip
ENV NM=${CROSS_PREFIX}gcc-nm
ENV WINDRES=${CROSS_PREFIX}windres
ENV DLLTOOL=${CROSS_PREFIX}dlltool
ENV STAGE_CFLAGS="-fno-semantic-interposition" 
ENV STAGE_CXXFLAGS="-fno-semantic-interposition"
ENV PKG_CONFIG=pkg-config
ENV PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig
ENV PATH="${PREFIX}/bin:${PREFIX}/aarch64-w64-mingw32/bin:${PATH}"
ENV CFLAGS="-static-libgcc -static-libstdc++ -I${PREFIX}/aarch64-w64-mingw32/include -I${PREFIX}/include -O2 -pipe -D_FORTIFY_SOURCE=2 -fstack-protector-strong"
ENV CXXFLAGS="-static-libgcc -static-libstdc++ -I${PREFIX}/aarch64-w64-mingw32/include -I${PREFIX}/include -O2 -pipe -D_FORTIFY_SOURCE=2 -fstack-protector-strong"
ENV LDFLAGS="-static-libgcc -static-libstdc++ -L${PREFIX}/aarch64-w64-mingw32/lib -L${PREFIX}/lib -O2 -pipe -fstack-protector-strong"

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
    echo "c_args = ['-static-libgcc', '-static-libstdc++', '-I${PREFIX}/aarch64-w64-mingw32/include', '-I${PREFIX}/include', '-O2', '-pipe', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong']" >> /build/cross_file.txt && \
    echo "cpp_args = ['-static-libgcc', '-static-libstdc++', '-I${PREFIX}/aarch64-w64-mingw32/include', '-I${PREFIX}/include', '-O2', '-pipe', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong']" >> /build/cross_file.txt && \
    echo "c_link_args = ['-static-libgcc', '-static-libstdc++', '-L${PREFIX}/aarch64-w64-mingw32/lib', '-L${PREFIX}/lib', '-O2', '-pipe', '-fstack-protector-strong']" >> /build/cross_file.txt && \
    echo "cpp_link_args = ['-static-libgcc', '-static-libstdc++', '-L${PREFIX}/aarch64-w64-mingw32/lib', '-L${PREFIX}/lib', '-O2', '-pipe', '-fstack-protector-strong']" >> /build/cross_file.txt

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
    || (cat "/ffmpeg_build.log" ; echo "❌ FFmpeg build failed" ; false) \
    && echo "🛠️ Building FFmpeg                               [2/2]" \
    && make -j$(nproc) >/ffmpeg_build.log 2>&1 || (cat "/ffmpeg_build.log" ; echo "❌ FFmpeg build failed" ; exit 1) && make install >/dev/null 2>&1 \
    && rm -rf /build/ffmpeg \
    && echo "------------------------------------------------------" \
    && echo "✅ FFmpeg was built successfully" \
    && echo "------------------------------------------------------" 

RUN chmod +x /scripts/init/package.sh && /scripts/init/package.sh

FROM alpine:latest AS final

COPY --from=windows /output/ffmpeg-8.1.2-windows-aarch64.zip /build/ffmpeg-8.1.2-windows-aarch64.zip

CMD ["cp", "/build/ffmpeg-8.1.2-windows-aarch64.zip", "/output"]
