#!/bin/bash

if [ ! -d ${PREFIX}/lib/pkgconfig ]; then
    mkdir -p ${PREFIX}/lib/pkgconfig
fi

# #region libpng
cd /build/libpng

LIBPNG_EXTRA_FLAGS=""
if [[ ${TARGET_OS} == "freebsd" ]]; then
    # the pngvalid test program uses feenableexcept, which FreeBSD's fenv.h
    # hides under the _POSIX_SOURCE that pngpriv.h defines — skip the tests
    LIBPNG_EXTRA_FLAGS="--disable-tests --disable-tools"
fi

./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pkgconfigdir=${PREFIX}/lib/pkgconfig \
    ${LIBPNG_EXTRA_FLAGS} --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pkgconfigdir=${PREFIX}/lib/pkgconfig \
    CPPFLAGS="${CPPFLAGS} -I${PREFIX}/include" \
    LDFLAGS="${LDFLAGS} -L${PREFIX}/lib -lz" \
    ${LIBPNG_EXTRA_FLAGS} --host=${CROSS_PREFIX%-} | log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Failed to build libpng config"
    exit 1
fi

make clean
make -j$(nproc)
make install

if [ ! -f ${PREFIX}/lib/libpng.a ]; then
    log "Failed to build libpng a "
    exit 1
fi
if [ ! -f ${PREFIX}/include/png.h ]; then
    log "Failed to build libpng h "
    exit 1
fi
if [ ! -f ${PREFIX}/include/pngconf.h ]; then
    log "Failed to build libpng c"
    exit 1
fi
if [ ! -f ${PREFIX}/include/pnglibconf.h ]; then
    log "Failed to build libpng ch"
    exit 1
fi
if [ ! -f ${PREFIX}/lib/pkgconfig/libpng.pc ]; then
    log "Failed to build libpng p"
    exit 1
fi
if [[ ${TARGET_OS} != "linux" ]]; then
    echo "Libs.private: -lstdc++ -lz" >>${PREFIX}/lib/pkgconfig/libpng.pc
else
    echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/libpng.pc
fi
cd /build
rm -rf /build/libpng
if pkg-config --modversion libpng >/dev/null 2>&1; then
    log "libpng is installed."
else
    log "libpng is missing!"
    exit 1 # Optional: Exit script if libpng is not found
fi
# #endregion

#region libgif
cd /build/giflib

if [[ ${TARGET_OS} != "windows" ]]; then
    apt-get update
    apt-get install -y --no-install-recommends imagemagick

    if [[ ${TARGET_OS} == "darwin" ]]; then
        sed -i 's/-Wl,-soname/-Wl,-install_name/g' Makefile
    fi

    make PREFIX=${PREFIX} || (
        log "Error: giflib make failed."
        exit 1
    )

    make PREFIX=${PREFIX} install || (
        log "Error: giflib install failed."
        exit 1
    )
else
    if [[ ${ARCH} == "aarch64" ]]; then
        # giflib's Makefile links the shared libgif.so with -soname, an ELF-only
        # flag. GNU ld only warns about it, so windows-x86_64 builds fine, but
        # this target links with lld, which hard-errors:
        #   lld: error: unknown argument: -soname
        # make then dies before producing libgif.a. Only the static library and
        # the header are consumed here, so build just the static target and skip
        # `make install`, whose shared targets hit the same error. The
        # gif_lib.h / libgif.a copies below install exactly what is needed.
        make libgif.a || (
            log "Error: giflib make failed."
            exit 1
        )
    else
        make || (
            log "Error: giflib make failed."
            exit 1
        )
        make install || (
            log "Error: giflib install failed."
            exit 1
        )
    fi
    if [ ! -f ${PREFIX}/include/gif_lib.h ]; then
        if [ -f gif_lib.h ]; then
            cp gif_lib.h ${PREFIX}/include/gif_lib.h
        else
            log "Failed to build giflib 1"
            exit 1
        fi
    fi
    if [ ! -f ${PREFIX}/lib/libgif.a ]; then
        if [ -f libgif.a ]; then
            cp libgif.a ${PREFIX}/lib/libgif.a
        else
            log "Failed to build giflib 2"
            exit 1
        fi
    fi
fi

if [ ! -f ${PREFIX}/lib/pkgconfig/giflib.pc ]; then
    if [ -f giflib.pc ]; then
        cp giflib.pc ${PREFIX}/lib/pkgconfig/giflib.pc
        sed -i "s|prefix=.*|prefix=${PREFIX}|g" ${PREFIX}/lib/pkgconfig/giflib.pc
    else
        {
            echo "prefix=${PREFIX}"
            echo "exec_prefix=\${prefix}"
            echo "libdir=\${exec_prefix}/lib"
            echo "includedir=\${prefix}/include"
            echo ""
            echo "Name: giflib"
            echo "Description: GIF library"
            echo "Version: 5.2.2"
            echo "Libs: -L\${libdir} -lgif"
            echo "Cflags: -I\${includedir}"
        } >${PREFIX}/lib/pkgconfig/giflib.pc
    fi
fi
if [ ! -f ${PREFIX}/lib/pkgconfig/giflib.pc ]; then
    log "Failed to build giflib 3"
    exit 1
fi
if [[ ${TARGET_OS} != "linux" ]]; then
    echo "Libs.private: -lstdc++ -lz" >>${PREFIX}/lib/pkgconfig/giflib.pc
else
    echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/giflib.pc
fi
cd /build
rm -rf /build/giflib
#endregion

#region libtiff
cd /build/libtiff
./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pkgconfigdir=${PREFIX}/lib/pkgconfig \
    --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pkgconfigdir=${PREFIX}/lib/pkgconfig \
    --host=${CROSS_PREFIX%-} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Failed to build libtiff"
    exit 1
fi
make -j$(nproc) && make install
if [ ! -f ${PREFIX}/lib/pkgconfig/libtiff-4.pc ]; then
    log "Failed to build libtiff"
    exit 1
fi
if [[ ${TARGET_OS} != "linux" ]]; then
    echo "Libs.private: -lstdc++ -lz" >>${PREFIX}/lib/pkgconfig/libtiff-4.pc
else
    echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/libtiff-4.pc
fi
cd /build
rm -rf /build/libtiff
#endregion

#region libwebp
cd /build/libwebp
./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    --enable-libwebpmux --enable-libwebpextras --enable-libwebpdemux --enable-libwebpdecoder \
    --disable-sdl --disable-gl --enable-gif --enable-jpeg --enable-tiff \
    --with-pngincludedir=${PREFIX}/include --with-pnglibdir=${PREFIX}/lib --enable-png \
    LDFLAGS="${LDFLAGS} -L${PREFIX}/lib" \
    CPPFLAGS="${CPPFLAGS} -I${PREFIX}/include" \
    CFLAGS="${CFLAGS} -I${PREFIX}/include" \
    LIBS="-lpng16 -lz" \
    PNG_INCLUDES="I${PREFIX}/include" \
    PNG_LIBS="${PREFIX}/lib" \
    --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    --enable-libwebpmux --enable-libwebpextras --enable-libwebpdemux --enable-libwebpdecoder \
    --disable-sdl --disable-gl --enable-gif --enable-jpeg --enable-tiff \
    --with-pngincludedir=${PREFIX}/include --with-pnglibdir=${PREFIX}/lib --enable-png \
    LDFLAGS="${LDFLAGS} -L${PREFIX}/lib" \
    CPPFLAGS="${CPPFLAGS} -I${PREFIX}/include" \
    CFLAGS="${CFLAGS} -I${PREFIX}/include" \
    LIBS="-lpng16 -lz" \
    PNG_INCLUDES="${PREFIX}/include" \
    PNG_LIBS="${PREFIX}/lib" \
    --host=${CROSS_PREFIX%-} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Failed to build libwebp"
    exit 1
fi

make -j$(nproc) && make install
rm -rf /build/libwebp

cp ${PREFIX}/lib/pkgconfig/libsharpyuv.pc ${PREFIX}/lib/pkgconfig/sharpyuv.pc
#endregion

add_enable "--enable-libwebp"

exit 0
