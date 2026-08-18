#!/bin/bash

if [[ ${TARGET_OS} == "darwin" ]]; then
    exit 255
fi

cd /build/vulkan-headers
find . -type f -name '*' -exec sed -i 's/-lshaderc_shared/-lshaderc_combined/' {} +
mkdir build && cd build
cmake -GNinja -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DBUILD_TESTING=OFF | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

ninja -j$(nproc) && ninja install
rm -rf /build/vulkan-headers
echo "prefix=${PREFIX}" >${PREFIX}/lib/pkgconfig/vulkan.pc
echo "exec_prefix=\${prefix}" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "libdir=\${prefix}/lib" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "includedir=\${prefix}/include" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "Name: Vulkan" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "Description: Vulkan Headers" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "Version: 1.4.307" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "Libs: -L\${libdir} -lvulkan" >>${PREFIX}/lib/pkgconfig/vulkan.pc
echo "Cflags: -I\${includedir}" >>${PREFIX}/lib/pkgconfig/vulkan.pc

# shaderc
cd /build/shaderc
./utils/git-sync-deps
find . -type f -name '*' -exec sed -i 's/-lshaderc_shared/-lshaderc_combined/' {} +
mkdir -p build && cd build
cmake -GNinja -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
    -DENABLE_EXCEPTIONS=ON -DENABLE_CTEST=OFF -DENABLE_GLSLANG_BINARIES=OFF -DSPIRV_SKIP_EXECUTABLES=ON \
    -DSPIRV_TOOLS_BUILD_STATIC=ON -DBUILD_SHARED_LIBS=OFF \
    -DSHADERC_SKIP_TESTS=ON -DSHADERC_ENABLE_SHARED_CRT=ON | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

ninja -j$(nproc) && ninja install
cp libshaderc_util/libshaderc_util.a ${PREFIX}/lib
echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/shaderc.pc
echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/shaderc_static.pc
echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/shaderc_combined.pc
rm -rf /build/shaderc
# spirv-cross
cd /build/spirv-cross
export VER_MAJ="$(grep 'set(spirv-cross-abi-major' CMakeLists.txt | sed -re 's/.* ([0-9]+)\)/\1/')"
export VER_MIN="$(grep 'set(spirv-cross-abi-minor' CMakeLists.txt | sed -re 's/.* ([0-9]+)\)/\1/')"
export VER_PCH="$(grep 'set(spirv-cross-abi-patch' CMakeLists.txt | sed -re 's/.* ([0-9]+)\)/\1/')"
export VER_FULL="$VER_MAJ.$VER_MIN.$VER_PCH"
find . -type f -name '*' -exec sed -i 's/-lshaderc_shared/-lshaderc_combined/' {} +
mkdir -p build && cd build
cmake -S .. -B . \
    ${CMAKE_COMMON_ARG} \
    -DSPIRV_CROSS_SHARED=OFF -DSPIRV_CROSS_STATIC=ON -DSPIRV_CROSS_CLI=OFF -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_FORCE_PIC=ON -DSPIRV_CROSS_ENABLE_CPP=OFF | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

make -j$(nproc) && make install
echo "prefix=${PREFIX}" >${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "exec_prefix=\${prefix}" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "libdir=\${prefix}/lib" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "sharedlibdir=\${prefix}/lib" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "includedir=\${prefix}/include/spirv_cross" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "Name: spirv-cross-c-shared" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "Description: C API for SPIRV-Cross" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "Version: ${VER_FULL}" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "Requires:" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "Libs: -L\${libdir} -L\${sharedlibdir} -lspirv-cross-c -lspirv-cross-glsl -lspirv-cross-hlsl -lspirv-cross-reflect -lspirv-cross-msl -lspirv-cross-util -lspirv-cross-core -lstdc++" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
echo "Cflags: -I\${includedir}" >>${PREFIX}/lib/pkgconfig/spirv-cross.pc
cp ${PREFIX}/lib/pkgconfig/spirv-cross.pc ${PREFIX}/lib/pkgconfig/spirv-cross-c-shared.pc
rm -rf /build/spirv-cross

# libplacebo
cd /build/libplacebo
git submodule update --init --recursive
sed -i 's/DPL_EXPORT/DPL_STATIC/' src/meson.build
find . -type f -name '*' -exec sed -i 's/-lshaderc_shared/-lshaderc_combined/' {} +
mkdir -p build && cd build
meson --prefix=${PREFIX} \
    --buildtype=release \
    --default-library=static \
    -Dvulkan=enabled \
    -Dvk-proc-addr=disabled \
    -Dvulkan-registry=${PREFIX}/share/vulkan/registry/vk.xml \
    -Dshaderc=enabled \
    -Dglslang=disabled \
    -Ddemos=false \
    -Dtests=false \
    -Dbench=false \
    -Dfuzz=false \
    --cross-file="/build/cross_file.txt" .. | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    exit 1
fi

ninja -j$(nproc) && ninja install
echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/libplacebo.pc
sed -i 's/-lshaderc_shared/-lshaderc_combined/' ${PREFIX}/lib/pkgconfig/libplacebo.pc
rm -rf /build/libplacebo

# FFmpeg 9.0 dropped --enable-libshaderc (and --enable-libglslang): it no longer
# links a SPIR-V compiler library, it invokes a host glslc/glslangValidator
# binary at build time to compile the Vulkan shaders. Passing the old flag is a
# hard "Unknown option" configure failure. The host compiler comes from the
# glslang-tools package installed in ffmpeg-base.dockerfile; without it
# configure silently disables spirv_compiler and every *_vulkan filter with it.
# shaderc is still built above because libplacebo links it (-Dshaderc=enabled).
add_enable "--enable-vulkan --enable-libplacebo"

exit 0
