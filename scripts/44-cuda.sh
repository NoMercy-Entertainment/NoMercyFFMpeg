#!/bin/bash

if [[ ${TARGET_OS} == "darwin" || ${TARGET_OS} == "freebsd" ]]; then
    # No CUDA on FreeBSD, and the host CUDA libs are Linux/glibc binaries
    exit 255
fi

if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # NVIDIA ships no Windows-on-ARM driver, so there is nothing to talk to,
    # and the copies below are x86-64 Linux binaries regardless.
    exit 255
fi

cp -R /usr/local/cuda/include/* ${PREFIX}/include
cp -R /usr/local/cuda/lib64/* ${PREFIX}/lib

# remove all copied *.so files
rm -f ${PREFIX}/lib/*.so ${PREFIX}/lib/*.so.*

# sed -i '/enabled libnpp/,+2c\
# enabled libnpp            \&\& { check_lib libnpp npp.h nppGetLibVersion -lnppc_static -lnppial_static -lnppicc_static -lnppidei_static -lnppif_static -lnppig_static -lnppim_static -lnppist_static -lnppisu_static -lnppitc_static ||\
#                                check_lib libnpp npp.h nppGetLibVersion -lnppig -lnppicc -lnppc -lnppidei -lnppif ||\
#                                check_lib libnpp npp.h nppGetLibVersion -lnppi -lnppif -lnppc -lnppidei ||\
#                                die "ERROR: libnpp not found"; }' /build/ffmpeg/configure

# add_enable "--enable-cuda --enable-cuda-nvcc --enable-cuvid --enable-libnpp"
add_enable "--enable-cuda --enable-cuda-nvcc --enable-cuvid"

exit 0
