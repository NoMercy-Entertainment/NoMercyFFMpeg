#!/bin/bash

cd /build/libvmaf

if [[ "${TARGET_OS}" == "windows" && "${ARCH}" == "aarch64" ]]; then
    # This target uses llvm-mingw, whose C++ library is libc++. The bundled
    # libsvm (src/svm.cpp) declares its own global `swap` template. libc++
    # calls unqualified swap() inside __split_buffer so ADL can find a better
    # overload; svm_node lives in the global namespace, so libsvm's swap and
    # std::swap are both candidates and every std::vector<svm_node> operation
    # fails with "call to 'swap' is ambiguous". libstdc++ does not do this,
    # which is why the GCC platforms never saw it.
    # std::swap is behaviourally identical to the helper being removed.
    sed -i 's|^template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }|#include <utility>\nusing std::swap;|' libvmaf/src/svm.cpp
    grep -q '^using std::swap;' libvmaf/src/svm.cpp || { log -a "libvmaf: svm.cpp swap patch did not apply"; exit 1; }
fi

mkdir build && cd build
meson --prefix=${PREFIX} \
    --buildtype=release --default-library=static -Dbuilt_in_models=true -Denable_tests=false -Denable_docs=false -Denable_avx512=true -Denable_float=true \
    --cross-file="/build/cross_file.txt" ../libvmaf | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

ninja -j$(nproc) && ninja install
sed -i 's/Libs.private:/Libs.private: -lstdc++/; t; $ a Libs.private: -lstdc++' ${PREFIX}/lib/pkgconfig/libvmaf.pc
rm -rf /build/libvmaf

add_enable "--enable-libvmaf"

exit 0
