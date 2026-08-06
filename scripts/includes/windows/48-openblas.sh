#!/bin/bash
if [[ ${TARGET_OS} != "windows" ]]; then
    exit 255
fi

if [[ ${ARCH} == "aarch64" ]]; then
    # OpenBLAS is skipped on Windows-on-ARM. With DYNAMIC_ARCH=ON its CMake
    # enumerates the x86 kernel family regardless of TARGET ("Targeting the
    # ATOM architecture", kernel_CORE2, cpuid.S, ...), which cannot assemble
    # for aarch64. Setting TARGET=ARMV8 does not change that list.
    #
    # This costs nothing but BLAS acceleration inside the whisper filter:
    # 48-whisper.sh enables BLAS only "if [[ -f ${PREFIX}/lib/libopenblas.a ]]",
    # and OpenBLAS is a windows-only extra to begin with -- linux, darwin and
    # freebsd already build whisper without it. Transcription still works, it
    # just uses ggml's built-in kernels.
    exit 255
fi

rm -f /ffmpeg_build.log
touch /ffmpeg_build.log

git clone --branch v0.3.34 https://github.com/OpenMathLib/OpenBLAS /build/OpenBLAS
cd /build/OpenBLAS

mkdir build && cd build
cmake -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DBINARY=64 \
    -DBUILD_DEPRECATED=OFF \
    -DBUILD_LAPACK_DEPRECATED=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_WITHOUT_LAPACK=YES \
    -DCMAKE_MT=mt \
    -DCROSS=ON \
    -DDYNAMIC_ARCH=ON \
    -DHOSTCC=gcc \
    -DNUM_THREADS=64 \
    -DTARGET=NEHALEM \
    -DUTEST_CHECK=OFF \
    -DVERBOSE=ON | log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Error: OpenBLAS configure failed"
    exit 1
fi

cmake --build . -j$(nproc) --config Release | log -a
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Error: OpenBLAS build failed"
    exit 1
fi

cmake --install . --config Release | log -a
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Error: OpenBLAS install failed"
    exit 1
else 
    echo "OpenBLAS installed successfully" > /ffmpeg_build.log
fi

cd /build
rm -rf /build/OpenBLAS

exit 0