/*
 * Copyright (c) 2026 Phillippe Pelzer
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/*
 * stemsplit: separate music into vocal and accompaniment stems using the
 * Deezer Spleeter 2stems network, run on ggml (the same library statically
 * linked into this binary for the whisper filter, via whisper.pc).
 *
 * This is the Task 4 skeleton: the filter registers, negotiates 44.1 kHz
 * stereo planar float, and passes audio through unchanged. No STFT, no
 * ggml graph, no model loading happens here - later tasks build all of
 * that on top of this scaffold.
 */

#include <limits.h>
#include <math.h>
#include <string.h>

#include <ggml.h>

#include "libavutil/opt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/samplefmt.h"
#include "libavutil/mem.h"
#include "libavutil/tx.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/audio.h"
#include "libavfilter/filters.h"
#include "formats.h"

#define SS_SAMPLE_RATE 44100
#define SS_FRAME_LENGTH 4096
#define SS_FRAME_STEP   1024
#define SS_BINS         (SS_FRAME_LENGTH / 2 + 1)   /* 2049 */
#define SS_T            512
#define SS_F            1024
#define SS_CHANNELS     2
#define SS_EPSILON      1e-10f
#define SS_WINDOW_COMPENSATION (2.0f / 3.0f)

enum { SS_STEM_VOCALS, SS_STEM_ACCOMPANIMENT, SS_STEM_ALL };
enum { SS_HB_PASSTHROUGH, SS_HB_ZEROS, SS_HB_AVERAGE };

typedef struct StemSplitContext {
    const AVClass *class;
    char    *model_path;
    int      stem;
    int      highband;
    int64_t  overlap;
    int      nb_threads;
    char    *dump_dir;
    char    *debug_input_path;
    int      passthrough_dsp;

    /* STFT / inverse STFT (Task 5): a 4096-point av_tx real DFT pair at hop
     * 1024, with a periodic Hann window applied on both analysis and
     * synthesis. window/in_buf/ola_buf/spec are scratch owned by the
     * context so ss_analyze()/ss_synthesize() don't allocate per call. */
    AVTXContext    *tx_fwd, *tx_inv;
    av_tx_fn        tx_fwd_fn, tx_inv_fn;
    float          *window;   /* SS_FRAME_LENGTH, periodic Hann */
    float          *in_buf;   /* SS_FRAME_LENGTH scratch: windowed real input
                                * for analysis, real ifft output for synthesis */
    float          *ola_buf;  /* SS_CHANNELS * SS_FRAME_LENGTH overlap-add
                                * accumulator, channel c at ola_buf + c*SS_FRAME_LENGTH */
    AVComplexFloat *spec;     /* SS_BINS scratch: mutable copy of the spectrum
                                * ss_synthesize() hands to the inverse transform,
                                * which overwrites its input */

    /* Minimal frame-hop driver used by `passthrough_dsp` to exercise
     * ss_analyze()/ss_synthesize() end to end for the round-trip test. Task
     * 9's real T=512 segmenter (ss_process_segment) replaces this; plain
     * hop-by-hop overlap-add doesn't need to know about network segments,
     * so this is deliberately not organised around them. */
    float   *pt_window[SS_CHANNELS]; /* sliding SS_FRAME_LENGTH analysis window */
    float   *pt_stage[SS_CHANNELS];  /* raw samples waiting to complete the next hop */
    AVComplexFloat *pt_spec;         /* SS_BINS scratch: this hop's analyzed spectrum.
                                       * Heap-allocated (av_malloc is SIMD-aligned) rather
                                       * than a local, since av_tx requires aligned buffers
                                       * unless AV_TX_UNALIGNED was set at init. */
    int      pt_stage_fill;          /* valid samples in pt_stage, 0..SS_FRAME_STEP */
    int64_t  pt_crop_remaining;      /* output samples still to discard (the lead-in) */
    int64_t  pt_total_in;            /* real input samples received so far */
    int64_t  pt_emitted;             /* real (post-crop) output samples emitted so far */

    int      eof;
    int64_t  next_pts;
} StemSplitContext;

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE };
    AVChannelLayout chlayouts[] = { AV_CHANNEL_LAYOUT_STEREO, { 0 } };
    int sample_rates[] = { SS_SAMPLE_RATE, -1 };
    int ret;

    if ((ret = ff_set_common_formats_from_list2(ctx, cfg_in, cfg_out, sample_fmts)) < 0)
        return ret;
    if ((ret = ff_set_common_channel_layouts_from_list2(ctx, cfg_in, cfg_out, chlayouts)) < 0)
        return ret;
    return ff_set_common_samplerates_from_list2(ctx, cfg_in, cfg_out, sample_rates);
}

/* ---- STFT / inverse STFT ------------------------------------------------
 *
 * 4096-point av_tx real DFT pair (AV_TX_FLOAT_RDFT), hop 1024, periodic Hann
 * window applied on both analysis and synthesis. Verified against
 * libavutil/tx.h and af_afftdn.c's usage: the forward real-to-complex
 * transform of N real samples produces exactly N/2+1 = SS_BINS complex
 * bins, packed contiguously with no special Nyquist/DC packing; the inverse
 * complex-to-real transform is unnormalized unless given scale 1/N, and it
 * always overwrites its input array, which is why ss_synthesize() works on
 * a private scratch copy (s->spec) rather than the caller's spectrum.
 */

static int ss_dsp_init(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    float scale = 1.0f;
    int ret;

    s->window  = av_malloc_array(SS_FRAME_LENGTH, sizeof(*s->window));
    s->in_buf  = av_malloc_array(SS_FRAME_LENGTH, sizeof(*s->in_buf));
    s->spec    = av_malloc_array(SS_BINS, sizeof(*s->spec));
    s->ola_buf = av_calloc(SS_CHANNELS * SS_FRAME_LENGTH, sizeof(*s->ola_buf));
    if (!s->window || !s->in_buf || !s->spec || !s->ola_buf)
        return AVERROR(ENOMEM);

    /* Periodic Hann, matching tf.signal.hann_window(periodic=True): divide
     * by N, not N - 1. The non-periodic form is the classic off-by-one
     * mistake here and shows up as a small but nonzero round-trip error. */
    for (int n = 0; n < SS_FRAME_LENGTH; n++)
        s->window[n] = 0.5f - 0.5f * cosf(2.0f * M_PI * n / SS_FRAME_LENGTH);

    ret = av_tx_init(&s->tx_fwd, &s->tx_fwd_fn, AV_TX_FLOAT_RDFT, 0,
                      SS_FRAME_LENGTH, &scale, 0);
    if (ret < 0)
        return ret;

    scale = 1.0f / SS_FRAME_LENGTH;
    return av_tx_init(&s->tx_inv, &s->tx_inv_fn, AV_TX_FLOAT_RDFT, 1,
                       SS_FRAME_LENGTH, &scale, 0);
}

static void ss_dsp_free(StemSplitContext *s)
{
    av_tx_uninit(&s->tx_fwd);
    av_tx_uninit(&s->tx_inv);
    av_freep(&s->window);
    av_freep(&s->in_buf);
    av_freep(&s->spec);
    av_freep(&s->ola_buf);
}

/* `in` is SS_FRAME_LENGTH samples of one channel; `out` receives SS_BINS
 * complex bins. */
static void ss_analyze(StemSplitContext *s, const float *in, AVComplexFloat *out)
{
    for (int n = 0; n < SS_FRAME_LENGTH; n++)
        s->in_buf[n] = s->window[n] * in[n];

    s->tx_fwd_fn(s->tx_fwd, out, s->in_buf, sizeof(float));
}

/* `in` is SS_BINS complex bins; windows the inverse transform's real output
 * a second time, scales by the Hann/75%-overlap reconstruction gain, and
 * overlap-adds into `ola` starting at `offset`. */
static void ss_synthesize(StemSplitContext *s, const AVComplexFloat *in, float *ola, int offset)
{
    /* The inverse RDFT overwrites its input (libavutil/tx.h), so hand it a
     * scratch copy rather than the caller's (possibly shared) spectrum. */
    memcpy(s->spec, in, SS_BINS * sizeof(*s->spec));
    s->tx_inv_fn(s->tx_inv, s->in_buf, s->spec, sizeof(*s->spec));

    for (int n = 0; n < SS_FRAME_LENGTH; n++)
        ola[offset + n] += s->window[n] * s->in_buf[n] * SS_WINDOW_COMPENSATION;
}

/* ---- Minimal round-trip driver for `passthrough_dsp` -------------------
 *
 * Feeds SS_FRAME_LENGTH zeros ahead of the first real sample (the lead-in),
 * runs ss_analyze()/ss_synthesize() back to back with no mask at hop
 * SS_FRAME_STEP, and crops the first SS_FRAME_LENGTH output samples so the
 * round trip lines up with the original signal. Task 9's real segmenter
 * replaces all of this; it exists only so Task 5 has something an ordinary
 * ffmpeg invocation can measure.
 */

/* Processes one hop's worth of newly staged samples (already sitting in
 * s->pt_stage, full for the normal path, zero-padded by the caller for the
 * EOF flush): emits the now-final front hop of ola_buf, then slides both
 * the analysis window and ola_buf forward and folds in the new frame.
 * `is_flush` caps emission at s->pt_total_in so tail zero-padding never
 * leaks into the output. */
static int ss_pt_process_hop(AVFilterContext *ctx, int is_flush)
{
    StemSplitContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int skip = 0, to_emit;
    int ret;

    /* Fold the newly staged hop into the sliding analysis window. */
    for (int c = 0; c < SS_CHANNELS; c++) {
        memmove(s->pt_window[c], s->pt_window[c] + SS_FRAME_STEP,
                (SS_FRAME_LENGTH - SS_FRAME_STEP) * sizeof(float));
        memcpy(s->pt_window[c] + SS_FRAME_LENGTH - SS_FRAME_STEP, s->pt_stage[c],
               SS_FRAME_STEP * sizeof(float));
    }

    /* The front hop of ola_buf can no longer be touched by any future frame
     * (the next frame's window starts one hop later): pull it out before
     * shifting ola_buf to make room for the new frame. */
    if (s->pt_crop_remaining > 0) {
        skip = (int) FFMIN(s->pt_crop_remaining, SS_FRAME_STEP);
        s->pt_crop_remaining -= skip;
    }
    to_emit = SS_FRAME_STEP - skip;
    if (is_flush)
        to_emit = (int) FFMIN(to_emit, FFMAX(s->pt_total_in - s->pt_emitted, 0));

    if (to_emit > 0) {
        AVFrame *out = ff_get_audio_buffer(outlink, to_emit);
        if (!out)
            return AVERROR(ENOMEM);
        for (int c = 0; c < SS_CHANNELS; c++)
            memcpy(out->extended_data[c], s->ola_buf + c * SS_FRAME_LENGTH + skip,
                   to_emit * sizeof(float));

        out->pts = s->next_pts;
        if (s->next_pts != AV_NOPTS_VALUE)
            s->next_pts += av_rescale_q(to_emit, (AVRational) { 1, outlink->sample_rate },
                                         outlink->time_base);
        s->pt_emitted += to_emit;

        ret = ff_filter_frame(outlink, out);
        if (ret < 0)
            return ret;
    }

    /* Shift ola_buf by one hop and add the new frame's contribution into
     * the freshly-vacated tail. */
    for (int c = 0; c < SS_CHANNELS; c++) {
        float *ola = s->ola_buf + c * SS_FRAME_LENGTH;

        memmove(ola, ola + SS_FRAME_STEP, (SS_FRAME_LENGTH - SS_FRAME_STEP) * sizeof(float));
        memset(ola + SS_FRAME_LENGTH - SS_FRAME_STEP, 0, SS_FRAME_STEP * sizeof(float));

        ss_analyze(s, s->pt_window[c], s->pt_spec);
        ss_synthesize(s, s->pt_spec, ola, 0);
    }

    return 0;
}

static int ss_passthrough_push(AVFilterContext *ctx, AVFrame *frame)
{
    StemSplitContext *s = ctx->priv;
    int offset = 0;

    s->pt_total_in += frame->nb_samples;

    while (offset < frame->nb_samples) {
        int n = FFMIN(frame->nb_samples - offset, SS_FRAME_STEP - s->pt_stage_fill);

        for (int c = 0; c < SS_CHANNELS; c++)
            memcpy(s->pt_stage[c] + s->pt_stage_fill,
                   (const float *) frame->extended_data[c] + offset,
                   n * sizeof(float));
        s->pt_stage_fill += n;
        offset += n;

        if (s->pt_stage_fill == SS_FRAME_STEP) {
            int ret = ss_pt_process_hop(ctx, 0);
            if (ret < 0)
                return ret;
            s->pt_stage_fill = 0;
        }
    }

    return 0;
}

static int ss_passthrough_flush(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;

    /* Keep zero-padding and sliding the window (exactly the pad_end tail
     * behaviour of the design's forward pipeline) until every real sample
     * has been overlap-added and emitted. */
    for (;;) {
        int ret;

        for (int c = 0; c < SS_CHANNELS; c++)
            memset(s->pt_stage[c] + s->pt_stage_fill, 0,
                   (SS_FRAME_STEP - s->pt_stage_fill) * sizeof(float));

        ret = ss_pt_process_hop(ctx, 1);
        if (ret < 0)
            return ret;
        s->pt_stage_fill = 0;

        if (s->pt_emitted >= s->pt_total_in)
            break;
    }

    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    int ret;

    if (!s->model_path || !s->model_path[0]) {
        av_log(ctx, AV_LOG_ERROR, "No model path specified. Use the 'model' option.\n");
        return AVERROR(EINVAL);
    }

    s->next_pts = AV_NOPTS_VALUE;

    ret = ss_dsp_init(ctx);
    if (ret < 0)
        return ret;

    for (int c = 0; c < SS_CHANNELS; c++) {
        s->pt_window[c] = av_calloc(SS_FRAME_LENGTH, sizeof(*s->pt_window[c]));
        s->pt_stage[c]  = av_calloc(SS_FRAME_STEP, sizeof(*s->pt_stage[c]));
        if (!s->pt_window[c] || !s->pt_stage[c])
            return AVERROR(ENOMEM);
    }
    s->pt_spec = av_malloc_array(SS_BINS, sizeof(*s->pt_spec));
    if (!s->pt_spec)
        return AVERROR(ENOMEM);
    s->pt_crop_remaining = SS_FRAME_LENGTH;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;

    ss_dsp_free(s);
    for (int c = 0; c < SS_CHANNELS; c++) {
        av_freep(&s->pt_window[c]);
        av_freep(&s->pt_stage[c]);
    }
    av_freep(&s->pt_spec);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    StemSplitContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    if (s->passthrough_dsp) {
        int ret;

        if (s->next_pts == AV_NOPTS_VALUE)
            s->next_pts = frame->pts != AV_NOPTS_VALUE ? frame->pts : 0;

        ret = ss_passthrough_push(ctx, frame);
        av_frame_free(&frame);
        return ret;
    }

    /* Skeleton: forward the frame unchanged. No STFT, no separation. */
    s->next_pts = frame->pts + av_rescale_q(frame->nb_samples,
                                            (AVRational) { 1, inlink->sample_rate },
                                            inlink->time_base);
    return ff_filter_frame(outlink, frame);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    StemSplitContext *s = ctx->priv;
    int64_t pts;
    int status;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (!s->eof && ff_inlink_queued_frames(inlink)) {
        AVFrame *frame = NULL;
        int ret;

        ret = ff_inlink_consume_frame(inlink, &frame);
        if (ret < 0)
            return ret;
        if (ret > 0)
            return filter_frame(inlink, frame);
    }

    if (!s->eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        s->eof = status == AVERROR_EOF;
        if (s->eof && s->passthrough_dsp) {
            int ret = ss_passthrough_flush(ctx);
            if (ret < 0)
                return ret;
        }
    }

    if (s->eof) {
        ff_outlink_set_status(outlink, AVERROR_EOF, s->next_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

#define OFFSET(x) offsetof(StemSplitContext, x)
#define FLAGS (AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

static const AVOption stemsplit_options[] = {
    { "model", "Path to the stemsplit GGUF model file", OFFSET(model_path), AV_OPT_TYPE_STRING, .flags = FLAGS },
    { "stem", "Stem to output", OFFSET(stem), AV_OPT_TYPE_INT, { .i64 = SS_STEM_ALL }, 0, SS_STEM_ALL, FLAGS, .unit = "stem" },
        { "vocals", "vocal stem only", 0, AV_OPT_TYPE_CONST, { .i64 = SS_STEM_VOCALS }, 0, 0, FLAGS, .unit = "stem" },
        { "accompaniment", "accompaniment stem only", 0, AV_OPT_TYPE_CONST, { .i64 = SS_STEM_ACCOMPANIMENT }, 0, 0, FLAGS, .unit = "stem" },
        { "all", "both stems, one per output pad", 0, AV_OPT_TYPE_CONST, { .i64 = SS_STEM_ALL }, 0, 0, FLAGS, .unit = "stem" },
    { "highband", "how to reconstruct frequencies above the network's band", OFFSET(highband), AV_OPT_TYPE_INT, { .i64 = SS_HB_PASSTHROUGH }, 0, SS_HB_AVERAGE, FLAGS, .unit = "highband" },
        { "passthrough", "copy the input high band unchanged", 0, AV_OPT_TYPE_CONST, { .i64 = SS_HB_PASSTHROUGH }, 0, 0, FLAGS, .unit = "highband" },
        { "zeros", "silence the high band", 0, AV_OPT_TYPE_CONST, { .i64 = SS_HB_ZEROS }, 0, 0, FLAGS, .unit = "highband" },
        { "average", "split the high band evenly between stems", 0, AV_OPT_TYPE_CONST, { .i64 = SS_HB_AVERAGE }, 0, 0, FLAGS, .unit = "highband" },
    { "overlap", "segment overlap as a duration, crossfaded on output", OFFSET(overlap), AV_OPT_TYPE_DURATION, { .i64 = 0 }, 0, 60000000, FLAGS },
    { "threads", "number of ggml threads", OFFSET(nb_threads), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, FLAGS },
    { "dump", "Internal: directory to dump intermediate tensors to for parity testing; empty disables", OFFSET(dump_dir), AV_OPT_TYPE_STRING, { .str = "" }, .flags = FLAGS },
    { "debug_input", "Internal: raw [C][T][F] float32 spectrogram to inject as the "
                     "network input, bypassing the STFT",
      OFFSET(debug_input_path), AV_OPT_TYPE_STRING, {.str = ""}, .flags = FLAGS },
    { "passthrough_dsp", "Internal: run STFT analysis and synthesis with no mask, "
                         "for round-trip testing",
      OFFSET(passthrough_dsp), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(stemsplit);

const FFFilter ff_af_stemsplit = {
    .p.name        = "stemsplit",
    .p.description = NULL_IF_CONFIG_SMALL("Separate music into vocal and accompaniment stems."),
    .p.priv_class  = &stemsplit_class,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .priv_size     = sizeof(StemSplitContext),
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
