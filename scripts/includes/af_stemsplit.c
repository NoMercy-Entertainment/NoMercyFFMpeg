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

static av_cold int init(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;

    if (!s->model_path || !s->model_path[0]) {
        av_log(ctx, AV_LOG_ERROR, "No model path specified. Use the 'model' option.\n");
        return AVERROR(EINVAL);
    }

    s->next_pts = AV_NOPTS_VALUE;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    /* Nothing allocated yet in this skeleton: no model is loaded, no STFT
     * state exists. Later tasks free their own resources here. */
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    StemSplitContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

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

    if (!s->eof && ff_inlink_acknowledge_status(inlink, &status, &pts))
        s->eof = status == AVERROR_EOF;

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
