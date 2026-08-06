#!/bin/bash

if [[ ${TARGET_OS} == "darwin" || ${TARGET_OS} == "freebsd" ]]; then
    # AMD provides no AMF runtime for FreeBSD
    exit 255
fi

if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # Windows-on-ARM devices carry Qualcomm Adreno or Nvidia/ARM parts, never
    # an AMD GPU, so the AMF runtime this would bind to does not exist there.
    exit 255
fi

cd /build/amf

# FFmpeg 8's libavfilter/vsrc_amf.c (built as C) transitively includes these
# AMF component headers, but the headers ship an unguarded
#   extern "C"
#   { ... }
# block. Wrap each one in #ifdef __cplusplus so C compilation succeeds —
# the declarations reference C++ namespace types (amf::AMFContext*) and were
# never C-callable to begin with.
for h in amf/public/include/components/Ambisonic2SRenderer.h \
         amf/public/include/components/AudioCapture.h \
         amf/public/include/components/ChromaKey.h \
         amf/public/include/components/DisplayCapture.h \
         amf/public/include/components/VideoCapture.h \
         amf/public/include/components/ZCamLiveStream.h; do
    awk '
        /^extern "C"$/ && !in_block { print "#ifdef __cplusplus"; print; in_block = 1; next }
        in_block && /^}$/ { print; print "#endif"; in_block = 0; next }
        { print }
    ' "$h" > "$h.new" && mv "$h.new" "$h"
done

mv amf/public/include ${PREFIX}/include/AMF
rm -rf /build/amf

add_enable "--enable-amf"

exit 0
