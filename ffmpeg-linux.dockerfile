# Create a Linux ffmpeg build
FROM stoney-ffmpeg-base AS linux

LABEL maintainer="Phillippe Pelzer"
LABEL version="1.0"
LABEL description="FFmpeg for Linux"

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Set environment variables for building ffmpeg
ENV PREFIX=/ffmpeg_build/linux
ENV ARCH=x86_64
ENV CROSS_PREFIX=${ARCH}-linux-gnu-
ENV CC=${CROSS_PREFIX}gcc
ENV CXX=${CROSS_PREFIX}g++
ENV LD=${CROSS_PREFIX}ld
ENV AR=${CROSS_PREFIX}gcc-ar
ENV RANLIB=${CROSS_PREFIX}gcc-ranlib
ENV STRIP=${CROSS_PREFIX}strip
# ENV WINDRES=${CROSS_PREFIX}windres
ENV NM=${CROSS_PREFIX}gcc-nm
# ENV DLLTOOL=${CROSS_PREFIX}dlltool
ENV STAGE_CFLAGS="-fvisibility=hidden -fno-semantic-interposition"
ENV STAGE_CXXFLAGS="-fvisibility=hidden -fno-semantic-interposition"
ENV PKG_CONFIG=pkg-config
ENV PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig
ENV PATH="${PREFIX}/bin:${PATH}"
ENV CFLAGS="-static-libgcc -static-libstdc++ -I${PREFIX}/include -O2 -pipe -fPIC -DPIC -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fstack-clash-protection -pthread"
ENV CXXFLAGS="-static-libgcc -static-libstdc++ -I${PREFIX}/include -O2 -pipe -fPIC -DPIC -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fstack-clash-protection -pthread" 
ENV LDFLAGS="-static-libgcc -static-libstdc++ -L${PREFIX}/lib -O2 -pipe -fstack-protector-strong -fstack-clash-protection -Wl,-z,relro,-z,now -pthread -lm"

# Create the build directory
RUN mkdir -p ${PREFIX}

# Create Meson cross file for Linux
RUN echo "[binaries]" > cross_file.txt && \
    echo "c = '${CC}'" >> cross_file.txt && \
    echo "cpp = '${CXX}'" >> cross_file.txt && \
    echo "ar = '${AR}'" >> cross_file.txt && \
    echo "strip = '${STRIP}'" >> cross_file.txt && \
    echo "pkgconfig = '${PKG_CONFIG}'" >> cross_file.txt && \
    echo "" >> cross_file.txt && \
    echo "[host_machine]" >> cross_file.txt && \
    echo "system = 'linux'" >> cross_file.txt && \
    echo "cpu_family = '${ARCH}'" >> cross_file.txt && \
    echo "cpu = '${ARCH}'" >> cross_file.txt && \
    echo "endian = 'little'" >> cross_file.txt

ENV CMAKE_COMMON_ARG="-DCMAKE_INSTALL_PREFIX=${PREFIX} -DENABLE_SHARED=OFF -DBUILD_SHARED_LIBS=OFF"

# iconv
WORKDIR /build/iconv
RUN ./configure --prefix=${PREFIX} --enable-extra-encodings --enable-static --disable-shared --with-pic \
    --host=${CROSS_PREFIX%-} \
    && make -j$(nproc) && make install

# libxml2
WORKDIR /build/libxml2
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --without-python --disable-maintainer-mode \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared --without-python --disable-maintainer-mode \
    && make -j$(nproc) && make install

# zlib
WORKDIR /build/zlib
RUN ./configure --prefix=${PREFIX} --static \
    && make -j$(nproc) && make install

# fftw3
WORKDIR /build/fftw3
RUN ./bootstrap.sh --prefix=${PREFIX} --enable-static --disable-shared --enable-maintainer-mode --disable-fortran \
    --disable-doc --with-our-malloc --enable-threads --with-combined-threads --with-incoming-stack-boundary=2 \
    --enable-sse2 --enable-avx --enable-avx2 \
    --host=${CROSS_PREFIX%-} \
    && make -j$(nproc) && make install

# libfreetype
WORKDIR /build/freetype
RUN ./configure --prefix=${PREFIX} --enable-static --disable-shared \
    && make -j$(nproc) && make install

# fribidi
WORKDIR /build/fribidi
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --disable-bin --disable-docs --disable-tests \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared --disable-bin --disable-docs --disable-tests \
    && make -j$(nproc) && make install

# fontconfig
WORKDIR /build/fontconfig
RUN ./autogen.sh --prefix=${PREFIX} --disable-docs --enable-iconv --enable-libxml2 --enable-static --disable-shared --sysconfdir=/etc --localstatedir=/var \
    && ./configure --prefix=${PREFIX} --disable-docs --enable-iconv --enable-libxml2 --enable-static --disable-shared --sysconfdir=/etc --localstatedir=/var \
    && make -j$(nproc) && make install

# harfbuzz
WORKDIR /build/harfbuzz
RUN meson build --prefix=${PREFIX} --buildtype=release -Ddefault_library=static \
    --cross-file=../cross_file.txt \
    && ninja -C build && ninja -C build install

# avisynth
WORKDIR /build/avisynth
RUN mkdir -p build && cd build \
    && cmake -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DHEADERS_ONLY=ON \
    && make -j$(nproc) && make VersionGen install

# chromaprint
WORKDIR /build/chromaprint
RUN mkdir -p build && cd build \
    && cmake -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TOOLS=OFF \
    -DBUILD_TESTS=OFF \
    -DFFT_LIB=fftw3 \
    && make -j$(nproc) && make install \
    && echo "Libs.private: -lfftw3 -lstdc++" >> ${PREFIX}/lib/pkgconfig/libchromaprint.pc \
    && echo "Cflags.private: -DCHROMAPRINT_NODLL" >> ${PREFIX}/lib/pkgconfig/libchromaprint.pc

# libass
WORKDIR /build/libass
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    && make -j$(nproc) && make install

# mp3lame
WORKDIR /build/lame
RUN ./configure --prefix=${PREFIX} --enable-static --disable-shared --enable-nasm --disable-gtktest --disable-cpml --disable-frontend --disable-decoder \
    && make -j$(nproc) && make install

# libvpx
WORKDIR /build/libvpx
RUN ./configure --prefix=${PREFIX} --enable-vp9-highbitdepth --enable-static --enable-pic \
    --disable-shared --disable-examples --disable-tools --disable-docs --disable-unit-tests \
    && make -j$(nproc) && make install

# x264
WORKDIR /build/x264
RUN ./configure --prefix=${PREFIX} --disable-cli --enable-static --enable-pic --disable-shared --disable-lavf --disable-swscale \
    && make -j$(nproc) && make install

# x265
RUN mkdir -p /build/x265/build/linux
# build x265 12bit
WORKDIR /build/x265/build/linux
RUN rm -rf 8bit 10bit 12bit && mkdir -p 8bit 10bit 12bit
RUN cd 12bit && cmake ${CMAKE_COMMON_ARG} -DHIGH_BIT_DEPTH=ON -DENABLE_HDR10_PLUS=ON -DEXPORT_C_API=OFF -DENABLE_CLI=OFF -DMAIN12=ON -S ../../../source -B . \
    && make -j$(nproc)

# build x265 10bit
WORKDIR /build/x265/build/linux
RUN cd 10bit && cmake ${CMAKE_COMMON_ARG} -DHIGH_BIT_DEPTH=ON -DENABLE_HDR10_PLUS=ON -DEXPORT_C_API=OFF -DENABLE_CLI=OFF -S ../../../source -B . \
    && make -j$(nproc)

# build x265 8bit
WORKDIR /build/x265/build/linux
RUN cd 8bit && mv ../12bit/libx265.a ./libx265_main12.a && mv ../10bit/libx265.a ./libx265_main10.a \
    && cmake ${CMAKE_COMMON_ARG} -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -S ../../../source -B . \
    && make -j$(nproc)

# install x265
WORKDIR /build/x265/build/linux/8bit
RUN mv libx265.a libx265_main.a \
    && { \
    echo "CREATE libx265.a"; \
    echo "ADDLIB libx265_main.a"; \
    echo "ADDLIB libx265_main10.a"; \
    echo "ADDLIB libx265_main12.a"; \
    echo "SAVE"; \
    echo "END"; \
    } | ar -M \
    && make -j$(nproc) && make install \
    && echo "Libs.private: -lstdc++" >> "${PREFIX}/lib/pkgconfig/x265.pc"

# xvid
WORKDIR /build/xvidcore/build/generic
RUN CFLAGS="${CFLAGS} -fstrength-reduce -ffast-math" \
    ./configure --enable-static --disable-shared \
    --prefix=${PREFIX} \
    --libdir=${PREFIX}/lib \
    --host=${CROSS_PREFIX%-} \ 
    CC=${CC} \
    CXX=${CXX} \
    && make -j$(nproc) && make install

# fdk-aac
WORKDIR /build/fdk-aac
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared \
    && make -j$(nproc) && make install

# opus
WORKDIR /build/opus
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --disable-extra-programs \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared --disable-extra-programs \
    && make -j$(nproc) && make install

# libwebp
WORKDIR /build/libwebp
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pic --enable-libwebpmux --disable-libwebpextras --disable-libwebpdemux --disable-sdl --disable-gl --disable-png --disable-jpeg --disable-tiff --disable-gif \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pic --enable-libwebpmux --disable-libwebpextras --disable-libwebpdemux --disable-sdl --disable-gl --disable-png --disable-jpeg --disable-tiff --disable-gif \
    && make -j$(nproc) && make install

# openjpeg
WORKDIR /build/openjpeg
RUN mkdir build && cd build \
    && cmake -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_PKGCONFIG_FILES=ON \
    -DBUILD_CODEC=OFF \
    -DWITH_ASTYLE=OFF \
    -DBUILD_TESTING=OFF \
    && make -j$(nproc) && make install

# zimg
WORKDIR /build/zimg
RUN ./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    && ./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    && make -j$(nproc) && make install

# ffnvcodec
WORKDIR /build/ffnvcodec
RUN make PREFIX=${PREFIX} install

# ffmpeg
WORKDIR /build/ffmpeg
RUN ./configure --pkg-config-flags=--static \
    --arch=${ARCH} \
    --target-os=linux \
    --cross-prefix=${CROSS_PREFIX} \
    --pkg-config=pkg-config \
    --prefix=${PREFIX} \
    --enable-cross-compile \
    --disable-shared \
    --enable-static \
    --enable-gpl \
    --enable-version3 \
    --enable-nonfree \
    --enable-iconv \
    --enable-libxml2 \
    --enable-libfreetype \
    --enable-libfribidi \
    --enable-fontconfig \
    --enable-avisynth \
    --enable-chromaprint \
    --enable-libass \
    --enable-libmp3lame \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libxvid \
    --enable-libfdk-aac \
    --enable-libopus \
    --enable-libwebp \
    --enable-libopenjpeg \
    --enable-libzimg \
    --enable-ffnvcodec \
    # --enable-cuda-llvm \
    --enable-runtime-cpudetect \
    --extra-version="NoMercy-MediaServer" \
    --extra-cflags="-static -static-libgcc -static-libstdc++ -I/${PREFIX}/include" \
    --extra-ldflags="-static -static-libgcc -static-libstdc++ -L/${PREFIX}/lib" \
    --extra-libs="-lpthread -lm" \
    || (cat ffbuild/config.log ; false) && \
    make -j$(nproc) && make install

RUN mkdir -p /ffmpeg/linux

RUN cp ${PREFIX}/bin/ffmpeg /ffmpeg/linux
RUN cp ${PREFIX}/bin/ffprobe /ffmpeg/linux

RUN tar -czf /ffmpeg-linux-7.1.tar.gz -C /ffmpeg/linux .
# cleanup

ADD start-linux.sh /start-linux.sh
RUN chmod 755 /start-linux.sh

# Set the entrypoint
CMD ["/start-linux.sh"]