#!/bin/bash

cd /build/zimg

if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # Two portability bugs that only this target trips:
    #
    # 1. common/arm/cpuinfo_arm.cpp includes <Windows.h> with a capital W.
    #    MSVC's filesystem is case-insensitive so upstream never noticed;
    #    mingw ships lowercase windows.h, so it is "file not found". Only ARM
    #    builds compile this file, which is why windows-x86_64 is unaffected.
    sed -i 's|#include <Windows.h>|#include <windows.h>|' src/zimg/common/arm/cpuinfo_arm.cpp
    grep -q '#include <windows.h>' src/zimg/common/arm/cpuinfo_arm.cpp || { log -a "zimg: cpuinfo_arm.cpp windows.h patch did not apply"; exit 1; }

    # 2. api/zimg.cpp uses std::exception_ptr, std::current_exception and
    #    std::rethrow_exception without including <exception>. libstdc++ pulls
    #    it in transitively via <memory>/<string>; libc++ does not, giving
    #    "no type named 'exception_ptr' in namespace 'std'".
    sed -i '0,/^#include <cmath>/s||#include <exception>\n#include <cmath>|' src/zimg/api/zimg.cpp
    grep -q '^#include <exception>' src/zimg/api/zimg.cpp || { log -a "zimg: zimg.cpp <exception> patch did not apply"; exit 1; }
fi

./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared --with-pic \
    --host=${CROSS_PREFIX%-} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

make -j$(nproc) && make install
rm -rf /build/zimg

add_enable "--enable-libzimg"

exit 0
