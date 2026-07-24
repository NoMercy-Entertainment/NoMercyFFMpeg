#!/bin/bash

#/**************************************/#
#/*  NoMercy Entertainment - dvdread:  */#
#/*  protocol wrapper over libdvdread  */#
#/**************************************/#

# Restores the dvdread:// URI form (issue #27) as a thin byte-stream protocol
# over libdvdread, alongside the dvdvideo demuxer. Requires libdvdread from
# 17-libdvdread.sh; skip when it is not installed.
if [ ! -f ${PREFIX}/lib/pkgconfig/libdvdread.pc ]; then
    exit 255
fi

# Copy the protocol source into the FFmpeg source tree
cp /scripts/includes/dvdread_proto.c /build/ffmpeg/libavformat/dvdread_proto.c

# 1. Register the protocol extern declaration in protocols.c
#    (PROTOCOL_LIST is computed by configure's find_things_extern from this
#    file, so no further registration is needed.)
log "Step 1: Adding extern declaration to protocols.c"

if ! grep -q "ff_dvdread_protocol" /build/ffmpeg/libavformat/protocols.c; then
    sed -i 's/^extern const URLProtocol ff_bluray_protocol;$/&\nextern const URLProtocol ff_dvdread_protocol;/' /build/ffmpeg/libavformat/protocols.c
    log "  Added extern declaration"
else
    log "  Extern declaration already exists"
fi

if grep -q "ff_dvdread_protocol" /build/ffmpeg/libavformat/protocols.c; then
    log "  Verified in protocols.c"
else
    log "  ERROR: Verification failed!"
    exit 1
fi

# 2. Add the protocol object to the Makefile, next to the bluray protocol.
#    Anchored loosely on the start of the OBJS-$(CONFIG_BLURAY_PROTOCOL) line
#    so alignment drift does not break the match.
log "Step 2: Adding to Makefile"

if ! grep -q "dvdread_proto.o" /build/ffmpeg/libavformat/Makefile; then
    sed -i '/^OBJS-\$(CONFIG_BLURAY_PROTOCOL)/a OBJS-$(CONFIG_DVDREAD_PROTOCOL)           += dvdread_proto.o' /build/ffmpeg/libavformat/Makefile
    log "  Added to Makefile"
else
    log "  Makefile entry already exists"
fi

if grep -q "dvdread_proto.o" /build/ffmpeg/libavformat/Makefile; then
    log "  Verified in Makefile"
else
    log "  ERROR: Verification failed!"
    exit 1
fi

# 3. configure — tie the protocol to libdvdread, grouped with the other
#    external-library *_protocol_deps lines. The protocol then turns on
#    automatically because the build already passes --enable-libdvdread
#    (added by 17-libdvdread.sh); no add_enable is needed here.
log "Step 3: Adding protocol dependency to configure"

if ! grep -q "dvdread_protocol_deps" /build/ffmpeg/configure; then
    sed -i 's/^bluray_protocol_deps="libbluray"$/&\ndvdread_protocol_deps="libdvdread"/' /build/ffmpeg/configure
    log "  Added dependency line"
else
    log "  Dependency line already exists"
fi

if grep -q "dvdread_protocol_deps" /build/ffmpeg/configure; then
    log "  Verified in configure"
else
    log "  ERROR: Verification failed!"
    exit 1
fi

exit 0
