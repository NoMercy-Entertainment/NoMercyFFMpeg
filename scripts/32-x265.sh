#!/bin/bash

CMAKE_X265_ARG="${CMAKE_COMMON_ARG} -DENABLE_ALPHA=ON -DCMAKE_ASM_NASM_FLAGS=-w-macro-params-legacy"

if [[ "${TARGET_OS}" == "darwin" ]]; then
    CMAKE_X265_ARG="-DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=OFF -DENABLE_ALPHA=ON -DCMAKE_ASM_NASM_FLAGS=-w-macro-params-legacy"
fi

cd /build/x265

# build x265 12bit
if [[ "${TARGET_OS}" == "windows" ]]; then
    rm -rf build/windows/12bit build/windows/10bit build/windows/8bit
    mkdir -p build/windows/12bit build/windows/10bit build/windows/8bit
    cd build/windows/12bit
elif [[ "${TARGET_OS}" == "darwin" ]]; then
    rm -rf build/windows/12bit build/windows/10bit build/windows/8bit
    mkdir -p build/windows/12bit build/windows/10bit build/windows/8bit
    cd build/windows/12bit
else
    rm -rf build/linux/12bit build/linux/10bit build/linux/8bit
    mkdir -p build/linux/12bit build/linux/10bit build/linux/8bit
    cd build/linux/12bit
fi

cmake ${CMAKE_X265_ARG} -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_CLI=OFF -DMAIN12=ON -S ../../../source -B .

make -j$(nproc)

# build x265 10bit
cd ../10bit
cmake ${CMAKE_X265_ARG} -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_CLI=OFF -S ../../../source -B .

make -j$(nproc)

# build x265 8bit
cd ../8bit
mv ../12bit/libx265.a ./libx265_main12.a && mv ../10bit/libx265.a ./libx265_main10.a
cmake ${CMAKE_X265_ARG} -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -S ../../../source -B .

make -j$(nproc)

# install x265
mv libx265.a libx265_main.a

# Member count of the 8-bit-only archive, to prove the merge below actually
# added the other two. Existence proves nothing: an unmerged libx265.a is a
# perfectly good archive that simply lacks x265_10bit:: and x265_12bit::, and
# the first thing that notices is ffmpeg's configure, 25 minutes later, saying
# "x265 not found using pkg-config" — which sends you looking at pkg-config.
main_members=$(${AR} t libx265_main.a 2>/dev/null | wc -l)

if [[ "${TARGET_OS}" == "darwin" ]]; then
    ${CROSS_PREFIX}libtool -static -o libx265.a libx265_main.a libx265_main10.a libx265_main12.a
    ${RANLIB} libx265.a
else
    {
        echo "CREATE libx265.a"
        echo "ADDLIB libx265_main.a"
        echo "ADDLIB libx265_main10.a"
        echo "ADDLIB libx265_main12.a"
        echo "SAVE"
        echo "END"
    } | ${AR} -M

    if [ ! -f libx265.a ]; then
        echo "Error: ${AR} failed to create libx265.a" >/ffmpeg_build.log
        exit 1
    fi
fi

merged_members=$(${AR} t libx265.a 2>/dev/null | wc -l)
if [ "${merged_members}" -le "${main_members}" ]; then
    {
        echo "Error: the x265 multilib merge did not take."
        echo "libx265_main.a has ${main_members} members, merged libx265.a has ${merged_members}."
        echo "A merged archive must contain all three, or ffmpeg fails much later"
        echo "with 'undefined symbol: x265_10bit::x265_api_get_*'."
    } >/ffmpeg_build.log
    exit 1
fi

make install

# Put the merged archive in place AFTER install. `make install` depends on the
# `all` target, so it can relink libx265.a from the 8-bit objects and overwrite
# the merge that just happened — whether it does comes down to timestamps,
# which is why this fails intermittently rather than every time.
cp -f libx265.a ${PREFIX}/lib/libx265.a
${RANLIB} ${PREFIX}/lib/libx265.a 2>/dev/null || true

rm -rf /build/x265

if [ ! -f ${PREFIX}/lib/libx265.a ]; then
    echo "Error: libx265.a is missing from lib" >/ffmpeg_build.log
    exit 1
fi

installed_members=$(${AR} t ${PREFIX}/lib/libx265.a 2>/dev/null | wc -l)
if [ "${installed_members}" -ne "${merged_members}" ]; then
    {
        echo "Error: the installed libx265.a is not the merged one."
        echo "Merged archive has ${merged_members} members, installed has ${installed_members}."
    } >/ffmpeg_build.log
    exit 1
fi

if [[ ${TARGET_OS} != "darwin" ]]; then
    if [ ! -f ${PREFIX}/lib/pkgconfig/x265.pc ]; then
        echo "Error: x265.pc is missing" >/ffmpeg_build.log
        exit 1
    else
        echo "Libs.private: -lstdc++" >>${PREFIX}/lib/pkgconfig/x265.pc
    fi
fi

add_enable "--enable-libx265"

exit 0
