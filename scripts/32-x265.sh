#!/bin/bash

# x265 is built three times (12-, 10- and 8-bit) and the three archives are
# merged into one libx265.a. None of that used to be checked: not one cmake,
# make or mv exit status, and the single check at the end only caught a merge
# that produced no file at all.
#
# CI run 30399011649 (freebsd-x86_64) installed a libx265.a without the 10/12-bit
# halves and reported x265 as built successfully. The failure surfaced ~25
# minutes later as ffmpeg's
#   ERROR: x265 not found using pkg-config
# which points at pkg-config rather than at x265; the real cause was further
# down its log, "undefined symbol: x265_10bit::x265_api_get_*". The same job had
# passed hours earlier with this script and x265 pinned at Release_4.1, and the
# trigger has not been reproduced.
#
# So the checks below are deliberately about the artifact rather than about any
# one suspected cause: every step reports its own failure, and the merged and
# installed archives are proven to contain all three halves. Note that member
# counts alone are not enough -- three 8-bit archives merge to a perfectly
# plausible 3x74 members -- hence the symbol check.
die() {
    echo "Error: $*" >/ffmpeg_build.log
    exit 1
}

# Member count of an archive; 0 when it is missing or unreadable.
archive_members() {
    ${AR} t "$1" 2>/dev/null | wc -l
}

# Count symbols of a namespace that the archive DEFINES. Undefined references
# do not count: the 8-bit half references x265_10bit::x265_api_get_* without
# defining it, so a plain grep matches even when the 10-bit half is absent --
# which is the exact failure this is here to detect.
archive_symbols() {
    ${NM} "$1" 2>/dev/null | awk -v ns="$2" '$0 ~ ns && $(NF-1) != "U" { n++ } END { print n+0 }'
}

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

cmake ${CMAKE_X265_ARG} -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_CLI=OFF -DMAIN12=ON -S ../../../source -B . \
    || die "x265 12-bit configure failed"

make -j$(nproc) || die "x265 12-bit build failed"

# build x265 10bit
cd ../10bit
cmake ${CMAKE_X265_ARG} -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_CLI=OFF -S ../../../source -B . \
    || die "x265 10-bit configure failed"

make -j$(nproc) || die "x265 10-bit build failed"

# build x265 8bit
cd ../8bit
mv ../12bit/libx265.a ./libx265_main12.a || die "x265 12-bit build produced no libx265.a"
mv ../10bit/libx265.a ./libx265_main10.a || die "x265 10-bit build produced no libx265.a"
cmake ${CMAKE_X265_ARG} -DEXTRA_LIB="x265_main10.a;x265_main12.a" -DEXTRA_LINK_FLAGS=-L. -DLINKED_10BIT=ON -DLINKED_12BIT=ON -S ../../../source -B . \
    || die "x265 8-bit configure failed"

make -j$(nproc) || die "x265 8-bit build failed"

# install x265
mv libx265.a libx265_main.a || die "x265 8-bit build produced no libx265.a"

# Member counts of the three halves, so the merge below can be proven rather
# than assumed. An empty or truncated input merges without complaint.
members_main=$(archive_members libx265_main.a)
members_main10=$(archive_members libx265_main10.a)
members_main12=$(archive_members libx265_main12.a)
for bits in main main10 main12; do
    eval "count=\${members_${bits}}"
    [ "${count}" -gt 0 ] || die "libx265_${bits}.a is empty or unreadable"
done
members_expected=$((members_main + members_main10 + members_main12))
if [[ "${TARGET_OS}" == "darwin" ]]; then
    ${CROSS_PREFIX}libtool -static -o libx265.a libx265_main.a libx265_main10.a libx265_main12.a \
        || die "libtool failed to merge the x265 archives"
else
    {
        echo "CREATE libx265.a"
        echo "ADDLIB libx265_main.a"
        echo "ADDLIB libx265_main10.a"
        echo "ADDLIB libx265_main12.a"
        echo "SAVE"
        echo "END"
    } | ${AR} -M || die "${AR} failed to merge the x265 archives"

    if [ ! -f libx265.a ]; then
        die "${AR} failed to create libx265.a"
    fi
fi

# Rebuild the symbol index on every platform, not just darwin. Linkers resolve
# archive members through that index; members that are present but unindexed
# are invisible and fail exactly like members that are absent.
${RANLIB} libx265.a || die "${RANLIB} failed on the merged libx265.a"

members_merged=$(archive_members libx265.a)
if [[ "${TARGET_OS}" == "darwin" ]]; then
    # libtool is not member-preserving the way `ar -M` is: merging three 52-member
    # archives yields 154, not 156, so an exact total is the wrong thing to assert
    # here. Only require that the merge grew past the 8-bit half; the symbol check
    # below is what actually proves the other two went in.
    if [ "${members_merged}" -le "${members_main}" ]; then
        die "the x265 multilib merge did not take: libx265_main.a has ${members_main}" \
            "members, merged libx265.a has ${members_merged}"
    fi
elif [ "${members_merged}" -ne "${members_expected}" ]; then
    die "the x265 multilib merge did not take: expected ${members_expected} members" \
        "(${members_main} + ${members_main10} + ${members_main12}), merged libx265.a has ${members_merged}"
fi

# The merge exists to satisfy exactly these two namespaces. Check them by name
# rather than trusting the member count, because that is what ffmpeg links
# against and what the failure reports as undefined.
for ns in x265_10bit x265_12bit; do
    found=$(archive_symbols libx265.a "${ns}")
    [ "${found}" -gt 0 ] || die "merged libx265.a exposes no ${ns}:: symbols"
done

make install || die "x265 install failed"
rm -rf /build/x265

if [ ! -f ${PREFIX}/lib/libx265.a ]; then
    echo "Error: libx265.a is missing from lib" >/ffmpeg_build.log
    exit 1
fi

# The archive that ffmpeg links against is the installed one, so verify that,
# not just the one left in the build tree.
members_installed=$(archive_members ${PREFIX}/lib/libx265.a)
if [ "${members_installed}" -ne "${members_merged}" ]; then
    die "the installed libx265.a is not the merged one:" \
        "merged has ${members_merged} members, installed has ${members_installed}"
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
