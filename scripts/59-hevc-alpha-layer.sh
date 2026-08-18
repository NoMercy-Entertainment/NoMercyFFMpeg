#!/bin/bash

#/**************************************/#
#/*  NoMercy Entertainment - HEVC      */#
#/*  alpha (auxiliary) layer decoding  */#
#/**************************************/#

# Restores alpha-channel decoding for HEVC-with-alpha video produced by Apple
# VideoToolbox (iPhone/iPad recordings, Keynote/Motion exports, HEVC stickers).
#
# Such files carry ONE hvc1 track holding a two-layer HEVC bitstream:
#   nuh_layer_id = 0 -> colour, nuh_layer_id = 1 -> alpha
# declared through a VPS extension with scalability_mask = AUXILIARY, AuxId = 1.
#
# Early VideoToolbox writes a non-standard VPS extension. Stock 8.1.2 failed to
# parse it, fell back to nb_layers = 1, logged "Ignoring unsupported VPS
# extension" and silently dropped the alpha layer -- decoding to an opaque
# yuv420p frame, with alphaextract failing on "Requested planes not available."
#
# Three patches are applied, all needed for the alpha layer to come out right:
#
#   1. hevc-alpha-videotoolbox.patch     upstream eedf8f0165fe (2026-04-01)
#      Keeps the alpha layer instead of discarding the broken VPS extension.
#      SHIPS IN 9.0 -- step 1 detects the marker and skips.
#
#   2. decode-get-format-first-sw.patch  upstream 3befae81f1dc (2026-03-27)
#      avcodec_default_get_format() picks the FIRST software format instead of
#      the last, so the alpha format the hevc decoder offers first actually
#      wins. Without this, ffprobe and every libavcodec API consumer still
#      decode the base layer only; ffmpeg CLI was unaffected because it
#      installs its own front-to-back get_format callback.
#      SHIPS IN 9.0 -- step 3 detects the marker and skips.
#
#   3. hevc-alpha-stream-pixfmt.patch    NoMercy-only, optional
#      Reports yuva420p at stream level. Probing never decodes a slice, so
#      without this ffprobe -show_streams shows the base-layer yuv420p while
#      the frames decode as yuva420p. Delete the file to drop this one.
#      STILL REQUIRED on 9.0 -- upstream export_stream_params() is unchanged.
#
# Both upstream commits landed on master after release/8.1 branched and were
# never backported, so no 8.1.x point release carries them. FFmpeg 9.0 does.
# The patch files are kept so this script still works against 8.1.x; each step
# is marker-guarded, so applying them on 9.0 is a no-op rather than a conflict.
#
# See: https://trac.ffmpeg.org/ticket/7965

# Lives in /scripts/includes (tracked) rather than /scripts/patches, which is
# gitignored wholesale for the AACS keydb and so never reaches CI.
PATCH="/scripts/includes/hevc-alpha-videotoolbox.patch"
TARGET="/build/ffmpeg/libavcodec/hevc/ps.c"

if [ ! -f "${PATCH}" ]; then
    log "  ERROR: ${PATCH} not found"
    exit 1
fi

if [ ! -f "${TARGET}" ]; then
    log "  ERROR: ${TARGET} not found"
    exit 1
fi

# Each step below is guarded independently -- do not early-exit when one is
# already applied, or a re-run would silently skip the later patches.
if grep -q "Broken VPS extension, treating as alpha video" "${TARGET}"; then
    log "Step 1: VPS patch already present, skipping"
else
    log "Step 1: Applying HEVC alpha layer patch to libavcodec/hevc/ps.c"

    # Dry run first so a context drift after an ffmpeg_version bump fails loudly
    # here instead of half-applying and breaking the build later.
    if ! patch -p1 --dry-run --forward -d /build/ffmpeg <"${PATCH}" >/dev/null 2>&1; then
        log "  ERROR: patch does not apply cleanly against FFmpeg ${ffmpeg_version}"
        log "         rebase scripts/includes/hevc-alpha-videotoolbox.patch"
        exit 1
    fi

    patch -p1 --forward -d /build/ffmpeg <"${PATCH}" | log -a
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "  ERROR: patch failed to apply"
        exit 1
    fi
fi

log "Step 2: Verifying patched source"

# All three hunks must be in place: the two AVERROR_PATCHWELCOME conversions in
# decode_vps_ext() and the alpha-preserving fallback in ff_hevc_decode_nal_vps().
if ! grep -q "Unsupported num_output_layer_sets" "${TARGET}"; then
    log "  ERROR: hunk 1 (num_output_layer_sets) missing"
    exit 1
fi

if ! grep -q "Broken VPS extension, treating as alpha video" "${TARGET}"; then
    log "  ERROR: hunk 3 (alpha fallback) missing"
    exit 1
fi

if ! grep -q "poc_lsb_not_present |= 1 << 1" "${TARGET}"; then
    log "  ERROR: hunk 3 (poc_lsb_not_present workaround) missing"
    exit 1
fi

log "  Verified all hunks in ps.c"

# Second upstream patch, required for the alpha layer to be decoded at all by
# anything using libavcodec's default get_format callback. The hevc decoder
# offers the alpha format first and the base-layer format last; stock 8.1.2's
# avcodec_default_get_format() returned the LAST software entry, so it always
# landed on the base layer. ffmpeg CLI installs its own front-to-back callback,
# which is why the CLI produced alpha while ffprobe did not. Fixed upstream in
# 9.0; the guard below skips this step there.
GETFMT_PATCH="/scripts/includes/decode-get-format-first-sw.patch"
GETFMT_TARGET="/build/ffmpeg/libavcodec/decode.c"

if [ ! -f "${GETFMT_PATCH}" ]; then
    log "  ERROR: ${GETFMT_PATCH} not found"
    exit 1
fi

if grep -q "Choose the first software format" "${GETFMT_TARGET}"; then
    log "Step 3: get_format patch already present, skipping"
else
    log "Step 3: Applying get_format patch to libavcodec/decode.c"

    if ! patch -p1 --dry-run --forward -d /build/ffmpeg <"${GETFMT_PATCH}" >/dev/null 2>&1; then
        log "  ERROR: patch does not apply cleanly against FFmpeg ${ffmpeg_version}"
        log "         rebase scripts/includes/decode-get-format-first-sw.patch"
        exit 1
    fi

    patch -p1 --forward -d /build/ffmpeg <"${GETFMT_PATCH}" | log -a
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "  ERROR: get_format patch failed to apply"
        exit 1
    fi

    if ! grep -q "Choose the first software format" "${GETFMT_TARGET}"; then
        log "  ERROR: get_format patch verification failed"
        exit 1
    fi

    log "  Verified get_format patch in decode.c"
fi

# Optional follow-up patch (NoMercy-only, not upstream): make ffprobe report
# yuva420p at stream level instead of the base-layer yuv420p. Delete
# hevc-alpha-stream-pixfmt.patch to fall back to upstream behaviour -- the
# alpha layer still decodes correctly without it, only the advertised stream
# format differs.
PIXFMT_PATCH="/scripts/includes/hevc-alpha-stream-pixfmt.patch"
PIXFMT_TARGET="/build/ffmpeg/libavcodec/hevc/hevcdec.c"

if [ ! -f "${PIXFMT_PATCH}" ]; then
    log "Step 4: stream pix_fmt patch not present, skipping (optional)"
elif grep -q "map_to_alpha_format(s, sps->pix_fmt);" "${PIXFMT_TARGET}" &&
     grep -q "forward-declared so export_stream_params" "${PIXFMT_TARGET}"; then
    log "Step 4: stream pix_fmt patch already present, skipping"
else
    log "Step 4: Applying stream pix_fmt patch to libavcodec/hevc/hevcdec.c"

    if ! patch -p1 --dry-run --forward -d /build/ffmpeg <"${PIXFMT_PATCH}" >/dev/null 2>&1; then
        log "  ERROR: patch does not apply cleanly against FFmpeg ${ffmpeg_version}"
        log "         rebase scripts/includes/hevc-alpha-stream-pixfmt.patch,"
        log "         or delete it to fall back to upstream behaviour"
        exit 1
    fi

    patch -p1 --forward -d /build/ffmpeg <"${PIXFMT_PATCH}" | log -a
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "  ERROR: stream pix_fmt patch failed to apply"
        exit 1
    fi

    if ! grep -q "forward-declared so export_stream_params" "${PIXFMT_TARGET}"; then
        log "  ERROR: forward declaration missing from hevcdec.c"
        exit 1
    fi

    log "  Verified stream pix_fmt patch in hevcdec.c"
fi

# No configure flag needed: multi-layer HEVC lives in the built-in hevc decoder,
# and the auxiliary-layer scaffolding (d3220ed8181d, d367016d3cca) already ships
# in 8.1.2 -- these patches only unblock the VideoToolbox bitstreams.

echo "HEVC alpha layer patches applied successfully" >/ffmpeg_build.log

exit 0
