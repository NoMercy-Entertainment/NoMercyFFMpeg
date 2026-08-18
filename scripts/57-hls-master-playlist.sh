#!/bin/bash

#/**************************************/#
#/*  NoMercy Entertainment — HLS       */#
#/*  master playlist spec compliance   */#
#/**************************************/#

# Replace the upstream HLS muxer sources with patched versions that complete
# the master playlist for Apple HLS spec compliance:
#   - FRAME-RATE and VIDEO-RANGE (SDR/PQ/HLG) on #EXT-X-STREAM-INF
#   - aname: var_stream_map key for a custom audio rendition NAME
#   - autoselect:/sforced:/characteristics: var_stream_map keys for
#     AUTOSELECT, FORCED and CHARACTERISTICS on EXT-X-MEDIA renditions
#     (AUTOSELECT=YES is implied for DEFAULT/FORCED renditions)
#   - #EXT-X-INDEPENDENT-SEGMENTS in the master playlist when
#     -hls_flags independent_segments is set
#   - CLOSED-CAPTIONS=NONE on variants when no cc_stream_map is given
#   - FORCED=YES/NO always written on subtitle renditions
#   - agroup: names used verbatim (no hardcoded "group_" prefix)
#   - FRAME-RATE always with three decimals (24.000)
#   - auto-set the hvc1 tag on HEVC streams instead of silently
#     dropping the CODECS attribute
#   - HEVC codec string from hvcC extradata (stream copies from MP4/MOV)
#   - loud warning when a CODECS attribute cannot be built
# See: https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/issues/12

FFDIR="/build/ffmpeg/libavformat"

for f in hlsenc.c hlsplaylist.c hlsplaylist.h codecstring.c; do
    if [ ! -f "/scripts/includes/${f}" ]; then
        log "  ERROR: /scripts/includes/${f} not found"
        exit 1
    fi
    cp "/scripts/includes/${f}" "${FFDIR}/${f}"
    log "  Applied ${f}"
done

# ff_hls_write_audio_rendition() gained aname, autoselect and characteristics
# parameters; update the single call site in the DASH muxer (NULL/-1 keep the
# upstream behaviour there). The nb_channels replacement is scoped to the
# ff_hls_write_audio_rendition call because dashenc.c has a second,
# unrelated ch_layout.nb_channels); line.
if ! grep -q "playlist_file, NULL, NULL, i, is_default," "${FFDIR}/dashenc.c"; then
    sed -i 's|playlist_file, NULL, i, is_default,|playlist_file, NULL, NULL, i, is_default,|' "${FFDIR}/dashenc.c"
fi
if ! grep -q "ch_layout.nb_channels, -1, NULL);" "${FFDIR}/dashenc.c"; then
    sed -i '/ff_hls_write_audio_rendition(/,+2 s|ch_layout\.nb_channels);|ch_layout.nb_channels, -1, NULL);|' "${FFDIR}/dashenc.c"
fi

if grep -q "playlist_file, NULL, NULL, i, is_default," "${FFDIR}/dashenc.c" &&
   grep -q "ch_layout.nb_channels, -1, NULL);" "${FFDIR}/dashenc.c"; then
    log "  Updated ff_hls_write_audio_rendition call site in dashenc.c"
else
    log "  ERROR: failed to update ff_hls_write_audio_rendition call site in dashenc.c"
    exit 1
fi

echo "HLS master playlist patch applied successfully" > /ffmpeg_build.log

exit 0
