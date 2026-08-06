#!/bin/bash

# There used to be a windows/aarch64 skip here, justified as "fribidi is not
# supported on Windows ARM64 it wants to use libdrm". fribidi has no libdrm
# dependency; the real failure was the aarch64-w64-mingw32 GCC crashing on
# -fstack-protector-strong (see ffmpeg-windows-aarch64.dockerfile). With that
# flag dropped fribidi cross-compiles normally, and libass needs it -- skipping
# it only moved the failure to "checking for fribidi >= 0.19.1... no".

cd /build/fribidi
./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --disable-bin --disable-docs --disable-tests \
    --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared --disable-bin --disable-docs --disable-tests \
    --host=${CROSS_PREFIX%-} | log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

make -j$(nproc) && make install

rm -rf /build/fribidi

add_enable "--enable-libfribidi"

exit 0
