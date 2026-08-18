#!/bin/bash

if [[ ${TARGET_OS} == "darwin" || ${TARGET_OS} == "freebsd" ]]; then
    # QSV needs libva + an Intel media driver; libva is Linux-only here
    exit 255
fi

if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # QSV is Intel-only and there is no Intel GPU behind a Windows-on-ARM
    # device; oneVPL itself assumes an x86 target.
    exit 255
fi

cd /build/libvpl
mkdir -p build && cd build
cmake -GNinja -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DCMAKE_INSTALL_BINDIR=${PREFIX}/bin -DCMAKE_INSTALL_LIBDIR=${PREFIX}/lib \
    -DBUILD_DISPATCHER=ON -DBUILD_DEV=ON \
    -DBUILD_PREVIEW=OFF -DBUILD_TOOLS=OFF -DBUILD_TOOLS_ONEVPL_EXPERIMENTAL=OFF -DINSTALL_EXAMPLE_CODE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

ninja -j$(nproc) && ninja install
rm -rf /build/libvpl ${PREFIX}/{etc,share}

# libvpl is C++ but upstream leaves vpl.pc's C++ runtime empty (CXX_LIB is never set),
# so add -lstdc++ for FFmpeg's static pkg-config link test against the C++ dispatcher.
sed -i 's/^Libs.private:/Libs.private: -lstdc++/' ${PREFIX}/lib/pkgconfig/vpl.pc

add_enable "--enable-libvpl"

exit 0
