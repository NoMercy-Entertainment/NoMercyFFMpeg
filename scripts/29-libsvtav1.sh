#!/bin/bash

EXTRA_CONFIGURE_ARGS=""
if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # SVT-AV1 builds optional aarch64 extension kernels (dotprod, i8mm, SVE)
    # and is supposed to compile each set with its own -march, e.g.
    # check_both_flags_add(-march=armv8.2-a+dotprod). That probe does not take
    # under this cross toolchain, so the sources are compiled without the
    # feature while their intrinsics require it:
    #   error: always_inline function 'vdotq_u32' requires target feature
    #   'dotprod', but would be inlined into a function compiled without it
    #
    # Disable the optional sets and keep baseline NEON (ENABLE_NEON stays on),
    # which is mandatory on aarch64. Raising -march globally instead would lift
    # the hardware floor for every Windows-on-ARM device, which is not worth a
    # marginal encoder speedup. SVE/SVE2 are off regardless: no shipping
    # Windows-on-ARM part implements them.
    EXTRA_CONFIGURE_ARGS="-DENABLE_NEON_DOTPROD=OFF -DENABLE_NEON_I8MM=OFF -DENABLE_SVE=OFF -DENABLE_SVE2=OFF"
elif [[ ${TARGET_OS} == "darwin" && ${ARCH} == "x86_64" ]]; then
    EXTRA_CONFIGURE_ARGS="-DCPUINFO_ARCHITECTURE=${ARCH} \
                          -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -Wno-dev \
                          -DUSE_EXTERNAL_CPUINFO=OFF"
fi

cd /build/libsvtav1
mkdir -p build && cd build
cmake -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DBUILD_APPS=OFF -DBUILD_EXAMPLES=OFF -DENABLE_AVX512=ON ${EXTRA_CONFIGURE_ARGS} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

make -j$(nproc) && make install
rm -rf /build/libsvtav1

add_enable "--enable-libsvtav1"

exit 0
