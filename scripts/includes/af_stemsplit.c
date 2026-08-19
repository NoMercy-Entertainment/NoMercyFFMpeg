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
 * The filter registers, negotiates 44.1 kHz stereo planar float, and can
 * round-trip audio through STFT analysis/synthesis (passthrough_dsp) or
 * load and validate the model's 100 GGUF tensors (ss_model_load). The
 * encoder half of the network is built and runs on ggml (ss_build_encoder);
 * the decoder, the mask and the real per-segment driver are later tasks. For
 * now the encoder is reachable only through the internal debug_input option,
 * which injects a spectrogram straight from disk for the per-layer parity
 * test of design section 9.2.
 */

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>

#include "libavutil/avassert.h"
#include "libavutil/file_open.h"
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

/* ---- Model tensors (Task 6) ---------------------------------------------
 *
 * One SSNet per instrument: six encoder blocks (conv1..conv6), six decoder
 * blocks (up1..up6), each carrying weight/bias/BatchNorm-affine, plus a
 * BatchNorm-less output convolution. This mirrors
 * tools/spleeter-gguf/convert.py's EXPECTED table exactly -- that table is
 * the authoritative tensor contract, not this comment.
 *
 * Ruling 9: every conv[].bn_a/bn_b and up[].bn_a/bn_b is loaded and shape
 * validated here, including vocals/accompaniment.conv6's, even though the
 * real Spleeter graph never applies conv6's BatchNorm (up1 consumes raw
 * conv6 -- design section 4.3.1(b)). That is a Task 7 compute-graph
 * decision, not a loading-time one: dropping the tensor here would change
 * the file's tensor count and break the converter's verify() contract.
 */
typedef struct SSLayer {
    struct ggml_tensor *w, *b, *bn_a, *bn_b;
} SSLayer;

typedef struct SSNet {
    SSLayer conv[6];
    SSLayer up[6];
    SSLayer out;
} SSNet;

#define SS_MODEL_ARCH     "spleeter-unet-v1"
#define SS_NB_INSTRUMENTS 2
#define SS_KERNEL         5

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

    /* Model loading (Task 6): 100 GGUF tensors resolved by name into two
     * SSNet structures and validated by dtype and shape. Loading only --
     * no compute graph is built from these until Tasks 7/8. */
    struct ggml_context        *gguf_ctx;    /* owns all tensor metadata and data;
                                               * created by gguf_init_from_file with
                                               * no_alloc=false, freed in ss_model_free */
    struct ggml_backend        *backend;     /* CPU backend, set up here for the
                                               * compute-graph tasks to reuse */
    struct ggml_backend_buffer *weights_buf; /* unused for now: no_alloc=false already
                                               * backs every weight tensor with plain
                                               * host memory inside gguf_ctx, so there is
                                               * nothing for a backend buffer to own on
                                               * the CPU-only backend this project ships.
                                               * Kept so the field exists if a future
                                               * backend needs it (ss_model_free frees it
                                               * if ever non-NULL). */
    SSNet    nets[SS_NB_INSTRUMENTS];
    int      nb_instruments;
    char    *instrument_names[SS_NB_INSTRUMENTS];

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

/* ---- Model loading (Task 6) ----------------------------------------------
 *
 * Resolves the model's 100 GGUF tensors by name into two SSNet structures
 * (index 0 / 1 = the first / second instrument named by the model's
 * "stemsplit.instruments" metadata), validating every tensor's dtype and
 * ne[] against the shapes implied by tools/spleeter-gguf/convert.py's
 * EXPECTED table. That table -- not this file -- is the authoritative
 * tensor contract; the shapes below are its ggml-side mirror, confirmed
 * against the real converted model with a throwaway probe before writing
 * this validation (GGUF stores a tensor's ggml `ne[]` as its numpy shape
 * reversed, so EXPECTED's python (5,5,ic,oc) becomes ne=[oc,ic,5,5]).
 *
 * This only loads and validates weights; no compute graph is built here.
 */

/* (in_channels, out_channels) per encoder block conv1..conv6. */
static const int ss_conv_io[6][2] = {
    { 2,  16}, { 16,  32}, { 32,  64}, { 64, 128}, {128, 256}, {256, 512}
};
/* (in_channels, out_channels) per decoder block up1..up6. */
static const int ss_up_io[6][2] = {
    {512, 256}, {512, 128}, {256,  64}, {128,  32}, { 64,  16}, { 32,   1}
};

static void cb_log(enum ggml_log_level level, const char *text, void *user_data)
{
    AVFilterContext *ctx = user_data;
    int av_log_level = AV_LOG_DEBUG;
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        av_log_level = AV_LOG_ERROR;
        break;
    case GGML_LOG_LEVEL_WARN:
        av_log_level = AV_LOG_WARNING;
        break;
    }
    av_log(ctx, av_log_level, "%s", text);
}

/* Looks up one tensor by name and validates its dtype and shape, naming the
 * offending tensor in every failure message (design section 10). Returns
 * NULL on any mismatch; the caller turns that into AVERROR(EINVAL). */
static struct ggml_tensor *ss_resolve_tensor(AVFilterContext *ctx,
                                              struct ggml_context *gctx,
                                              const char *name,
                                              const int64_t ne[GGML_MAX_DIMS],
                                              enum ggml_type type)
{
    struct ggml_tensor *t = ggml_get_tensor(gctx, name);
    int i;

    if (!t) {
        av_log(ctx, AV_LOG_ERROR,
               "Model is missing required tensor '%s'. The file may be "
               "truncated or built for a different architecture.\n", name);
        return NULL;
    }
    if (t->type != type) {
        av_log(ctx, AV_LOG_ERROR,
               "Tensor '%s' has the wrong data type: expected %s, got %s.\n",
               name, ggml_type_name(type), ggml_type_name(t->type));
        return NULL;
    }
    for (i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] != ne[i]) {
            av_log(ctx, AV_LOG_ERROR,
                   "Tensor '%s' has the wrong shape: expected "
                   "[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "], got "
                   "[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "].\n",
                   name, ne[0], ne[1], ne[2], ne[3],
                   t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            return NULL;
        }
    }
    return t;
}

/* Resolves one conv/up/out block's weight, bias and (if has_bn) BatchNorm
 * affine tensors for one instrument. */
static int ss_resolve_layer(AVFilterContext *ctx, struct ggml_context *gctx,
                             const char *inst, const char *layer,
                             const int64_t w_ne[GGML_MAX_DIMS], int out_ch,
                             int has_bn, SSLayer *out)
{
    char name[80];
    const int64_t b_ne[GGML_MAX_DIMS] = { out_ch, 1, 1, 1 };

    snprintf(name, sizeof(name), "%s.%s.weight", inst, layer);
    out->w = ss_resolve_tensor(ctx, gctx, name, w_ne, GGML_TYPE_F16);
    if (!out->w)
        return AVERROR(EINVAL);

    snprintf(name, sizeof(name), "%s.%s.bias", inst, layer);
    out->b = ss_resolve_tensor(ctx, gctx, name, b_ne, GGML_TYPE_F32);
    if (!out->b)
        return AVERROR(EINVAL);

    if (has_bn) {
        snprintf(name, sizeof(name), "%s.%s.bn_a", inst, layer);
        out->bn_a = ss_resolve_tensor(ctx, gctx, name, b_ne, GGML_TYPE_F32);
        if (!out->bn_a)
            return AVERROR(EINVAL);

        snprintf(name, sizeof(name), "%s.%s.bn_b", inst, layer);
        out->bn_b = ss_resolve_tensor(ctx, gctx, name, b_ne, GGML_TYPE_F32);
        if (!out->bn_b)
            return AVERROR(EINVAL);
    } else {
        out->bn_a = out->bn_b = NULL;
    }

    return 0;
}

/* ---- Kernel repacking: GGUF axis order -> ggml's convolution axis order --
 *
 * gguf-py writes a tensor's numpy shape *reversed* into the file, so
 * convert.py's (kw, kh, ic, oc) array arrives here as ne = [oc, ic, kh, kw]:
 * ggml's ne[0], the fastest-varying axis, is the numpy array's *last* axis.
 * ggml's convolution ops want the exact opposite, ne = [kw, kh, ic, oc], kw
 * fastest. Probed against the pinned whisper.cpp 1.9.1 ggml before this was
 * written: a kernel with ne = [KW, KH, IC, OC] convolved against data with
 * ne = [W, H, IC, N] reproduces a hand-computed reference exactly.
 * (ggml.h's "// a: [OC, IC, KH, KW]" comments say the same thing written in
 * reversed, numpy-style order -- they are not a second convention.)
 *
 * The correction is a full reversal of all four axes. It is done once, here,
 * rather than per graph build, and it is applied uniformly to every kernel --
 * conv*, up* and out -- because it corrects the container's axis convention,
 * not any one layer's use of it. Leaving up* and out in the on-disk order would
 * hand the decoder a half-converted model. After the reversal:
 *
 *   conv*.weight  [oc,ic,kh,kw] -> [kw,kh,ic,oc]   ggml_conv_2d_direct wants
 *                                                  ne[2]=IC, ne[3]=OC
 *   up*.weight    [ic,oc,kh,kw] -> [kw,kh,oc,ic]   ggml_conv_transpose_2d_p0
 *                                                  wants ne[2]=OC, ne[3]=IC
 *   out.weight    [2,1,4,4]     -> [4,4,1,2]       the output conv, same
 *                                                  [KW,KH,IC,OC] convention
 *
 * conv_weight() in tools/spleeter-gguf/convert.py already swaps TensorFlow's
 * kh/kw before writing, which is what makes the plain reversal land kh and kw
 * the right way round here; the two halves only make sense together.
 *
 * The tensor is rewritten in place. That is deliberate and it is why
 * ss_resolve_tensor()'s shape validation runs first, against the on-disk
 * order: the loader's contract is with the file, this is the adaptation layer
 * that sits after it. Nothing else reads these tensors in between.
 */
static int ss_repack_kernel(struct ggml_tensor *t)
{
    const int64_t n0 = t->ne[0], n1 = t->ne[1], n2 = t->ne[2], n3 = t->ne[3];
    const size_t nb = ggml_nbytes(t);
    uint16_t *src = t->data;
    uint16_t *tmp;
    int64_t i0, i1, i2, i3;

    /* Every .weight is F16 and contiguous; ss_resolve_layer() has already
     * rejected the model otherwise. The permutation only moves 16-bit units
     * around, so it needs no float arithmetic and cannot lose precision. */
    av_assert0(t->type == GGML_TYPE_F16);
    av_assert0(ggml_is_contiguous(t));

    tmp = av_malloc(nb);
    if (!tmp)
        return AVERROR(ENOMEM);

    for (i3 = 0; i3 < n3; i3++)
        for (i2 = 0; i2 < n2; i2++)
            for (i1 = 0; i1 < n1; i1++)
                for (i0 = 0; i0 < n0; i0++)
                    tmp[i3 + n3 * (i2 + n2 * (i1 + n1 * i0))] =
                        src[i0 + n0 * (i1 + n1 * (i2 + n2 * i3))];

    memcpy(src, tmp, nb);
    av_free(tmp);

    t->ne[0] = n3;
    t->ne[1] = n2;
    t->ne[2] = n1;
    t->ne[3] = n0;
    t->nb[0] = ggml_type_size(t->type);
    t->nb[1] = t->nb[0] * t->ne[0];
    t->nb[2] = t->nb[1] * t->ne[1];
    t->nb[3] = t->nb[2] * t->ne[2];

    return 0;
}

/* Repacks all 26 kernels (13 layers x 2 instruments) of a loaded model. */
static int ss_repack_all_kernels(StemSplitContext *s)
{
    int k, n, ret;

    for (k = 0; k < s->nb_instruments; k++) {
        for (n = 0; n < 6; n++) {
            if ((ret = ss_repack_kernel(s->nets[k].conv[n].w)) < 0)
                return ret;
            if ((ret = ss_repack_kernel(s->nets[k].up[n].w)) < 0)
                return ret;
        }
        if ((ret = ss_repack_kernel(s->nets[k].out.w)) < 0)
            return ret;
    }

    return 0;
}

/* Joins s->instrument_names into "a, b" for log/error messages. */
static void ss_join_instruments(const StemSplitContext *s, char *buf, size_t buf_size)
{
    size_t off = 0;
    int i;

    buf[0] = 0;
    for (i = 0; i < s->nb_instruments; i++) {
        int len = snprintf(buf + off, buf_size - off, "%s%s",
                            i ? ", " : "", s->instrument_names[i]);
        if (len > 0)
            off += FFMIN((size_t) len, buf_size - off - 1);
    }
}

static void ss_model_free(StemSplitContext *s)
{
    int i;

    if (s->backend)
        ggml_backend_free(s->backend);
    s->backend = NULL;

    if (s->weights_buf)
        ggml_backend_buffer_free(s->weights_buf);
    s->weights_buf = NULL;

    if (s->gguf_ctx)
        ggml_free(s->gguf_ctx);
    s->gguf_ctx = NULL;

    for (i = 0; i < SS_NB_INSTRUMENTS; i++)
        av_freep(&s->instrument_names[i]);
    s->nb_instruments = 0;

    /* Tensor pointers inside s->nets live in gguf_ctx, already freed above;
     * clear them so a stray use-after-free reads NULL, not garbage. */
    memset(s->nets, 0, sizeof(s->nets));
}

static int ss_model_load(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    struct gguf_init_params gp = { .no_alloc = false, .ctx = &s->gguf_ctx };
    struct gguf_context *gguf = NULL;
    int64_t key;
    size_t n_inst;
    const char *arch;
    char avail[SS_NB_INSTRUMENTS * 32];
    int k, n, i;
    int ret;

    /* Defensive idempotency: no known FFmpeg path calls ss_model_load()
     * twice on a live context, but if one ever does, s->backend, s->gguf_ctx
     * and s->instrument_names[] would each be silently overwritten and
     * leaked rather than freed. Make "a load always starts from a clean
     * slate" an explicit invariant instead of an implicit assumption. */
    ss_model_free(s);

    /* Bridge ggml's own log output (including gguf_init_from_file's
     * internal warnings, e.g. "invalid magic characters") to av_log at
     * matching severity, before anything below can trigger it. */
    ggml_log_set(cb_log, ctx);

    if (!s->model_path || !*s->model_path) {
        av_log(ctx, AV_LOG_ERROR,
               "No model specified. Use the 'model' option to point at a "
               "stemsplit .gguf file.\n");
        return AVERROR(EINVAL);
    }

    gguf = gguf_init_from_file(s->model_path, gp);
    if (!gguf) {
        av_log(ctx, AV_LOG_ERROR,
               "Could not read model '%s': not a valid GGUF file, or the "
               "file is missing or unreadable. Download it from the "
               "releases page of this repository "
               "(spleeter-2stems-f16.gguf, "
               "https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/releases).\n",
               s->model_path);
        return AVERROR(EIO);
    }

    /* ---- architecture tag: refuse to guess ---- */
    key = gguf_find_key(gguf, "stemsplit.arch");
    if (key < 0) {
        av_log(ctx, AV_LOG_ERROR,
               "Model '%s' has no 'stemsplit.arch' tag; expected '%s'. "
               "Refusing to guess.\n", s->model_path, SS_MODEL_ARCH);
        ret = AVERROR(EINVAL);
        goto done;
    }
    if (gguf_get_kv_type(gguf, key) != GGUF_TYPE_STRING) {
        av_log(ctx, AV_LOG_ERROR,
               "Model '%s' has a malformed 'stemsplit.arch' tag (not a "
               "string); expected '%s'. Refusing to guess.\n",
               s->model_path, SS_MODEL_ARCH);
        ret = AVERROR(EINVAL);
        goto done;
    }
    arch = gguf_get_val_str(gguf, key);
    if (strcmp(arch, SS_MODEL_ARCH)) {
        av_log(ctx, AV_LOG_ERROR,
               "Model '%s' has unrecognised architecture '%s'; expected "
               "'%s'. Refusing to guess.\n", s->model_path, arch, SS_MODEL_ARCH);
        ret = AVERROR(EINVAL);
        goto done;
    }

    /* ---- instrument list ---- */
    key = gguf_find_key(gguf, "stemsplit.instruments");
    if (key < 0 || gguf_get_kv_type(gguf, key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf, key) != GGUF_TYPE_STRING) {
        av_log(ctx, AV_LOG_ERROR,
               "Model '%s' is missing a valid 'stemsplit.instruments' "
               "string array in its metadata.\n", s->model_path);
        ret = AVERROR(EINVAL);
        goto done;
    }
    n_inst = gguf_get_arr_n(gguf, key);
    if (n_inst != SS_NB_INSTRUMENTS) {
        av_log(ctx, AV_LOG_ERROR,
               "Model '%s' declares %zu instrument(s); architecture '%s' "
               "requires exactly %d. Refusing to guess.\n",
               s->model_path, n_inst, SS_MODEL_ARCH, SS_NB_INSTRUMENTS);
        ret = AVERROR(EINVAL);
        goto done;
    }
    for (i = 0; i < SS_NB_INSTRUMENTS; i++) {
        s->instrument_names[i] = av_strdup(gguf_get_arr_str(gguf, key, i));
        if (!s->instrument_names[i]) {
            ret = AVERROR(ENOMEM);
            goto done;
        }
    }
    s->nb_instruments = SS_NB_INSTRUMENTS;

    /* ---- 100 tensors: conv1..conv6, up1..up6, out, per instrument ---- */
    for (k = 0; k < SS_NB_INSTRUMENTS; k++) {
        const char *inst = s->instrument_names[k];

        for (n = 0; n < 6; n++) {
            char layer[16];
            const int64_t w_ne[GGML_MAX_DIMS] = {
                ss_conv_io[n][1], ss_conv_io[n][0], SS_KERNEL, SS_KERNEL
            };
            snprintf(layer, sizeof(layer), "conv%d", n + 1);
            ret = ss_resolve_layer(ctx, s->gguf_ctx, inst, layer, w_ne,
                                    ss_conv_io[n][1], 1, &s->nets[k].conv[n]);
            if (ret < 0)
                goto done;
        }

        for (n = 0; n < 6; n++) {
            char layer[16];
            const int64_t w_ne[GGML_MAX_DIMS] = {
                ss_up_io[n][0], ss_up_io[n][1], SS_KERNEL, SS_KERNEL
            };
            snprintf(layer, sizeof(layer), "up%d", n + 1);
            ret = ss_resolve_layer(ctx, s->gguf_ctx, inst, layer, w_ne,
                                    ss_up_io[n][1], 1, &s->nets[k].up[n]);
            if (ret < 0)
                goto done;
        }

        {
            const int64_t w_ne[GGML_MAX_DIMS] = { 2, 1, 4, 4 };
            ret = ss_resolve_layer(ctx, s->gguf_ctx, inst, "out", w_ne, 2, 0,
                                    &s->nets[k].out);
            if (ret < 0)
                goto done;
        }
    }

    /* ---- GGUF axis order -> ggml convolution axis order ---- */
    ret = ss_repack_all_kernels(s);
    if (ret < 0)
        goto done;

    /* ---- the requested stem must actually be in this model ---- */
    if (s->stem != SS_STEM_ALL) {
        const char *want = s->stem == SS_STEM_VOCALS ? "vocals" : "accompaniment";
        int found = 0;

        for (i = 0; i < s->nb_instruments; i++) {
            if (!strcmp(s->instrument_names[i], want)) {
                found = 1;
                break;
            }
        }
        if (!found) {
            ss_join_instruments(s, avail, sizeof(avail));
            av_log(ctx, AV_LOG_ERROR,
                   "Requested stem '%s' is not provided by model '%s'. "
                   "Available stems: %s.\n", want, s->model_path, avail);
            ret = AVERROR(EINVAL);
            goto done;
        }
    }

    /* ---- backend: CPU only (design non-goal: no GPU backends) ---- */
    s->backend = ggml_backend_cpu_init();
    if (!s->backend) {
        av_log(ctx, AV_LOG_ERROR,
               "Could not initialize the ggml CPU backend.\n");
        ret = AVERROR(ENOMEM);
        goto done;
    }

    ss_join_instruments(s, avail, sizeof(avail));
    av_log(ctx, AV_LOG_INFO,
           "stemsplit: model '%s' loaded (100 tensors); instruments: %s.\n",
           s->model_path, avail);
    ret = 0;

done:
    gguf_free(gguf);
    return ret;
}

/* ---- Encoder graph (Task 7) ---------------------------------------------
 *
 * Six blocks of Conv2D 5x5 stride 2 "SAME" + bias, then BatchNorm and
 * LeakyReLU(0.2) -- except the sixth, which stops after the bias (see
 * ss_build_encoder). Each block has *two* outputs that both matter, and they
 * are not interchangeable; ss_encoder_block's contract spells that out.
 *
 * Tensor layout throughout is ggml's ne = [F, T, C, N], frequency fastest,
 * which is also the byte layout the fixtures and ss_dump_tensor use.
 */

/* Graph-metadata budget for one instrument's encoder. Each block builds about
 * ten tensors (pad, kernel cast, conv, bias reshape + add, two BatchNorm
 * reshapes + mul + add, leaky_relu) -- roughly 60 in total -- so this is a
 * comfortable margin that still fails loudly rather than silently if the graph
 * grows unexpectedly. */
#define SS_GRAPH_TENSORS 512

/* TensorFlow's "SAME" padding for kernel 5 / stride 2 is asymmetric: for an
 * input of N (even), the output is N/2 and the total padding is
 * (N/2 - 1) * 2 + 5 - N == 3, split as floor(3/2) = 1 before and 2 after, on
 * each spatial axis. ggml's convolution ops take a single p0/p1 per axis and
 * so can only pad symmetrically, so the asymmetry is materialised here with
 * ggml_pad_ext -- which zero-pads each of the four dimensions independently on
 * the low and high side -- and the convolution then runs with p0 = p1 = 0.
 * Size check: (N + 1 + 2 - 5)/2 + 1 == N/2.
 *
 * Symmetric pad 2 produces the correct output *shape* but shifts every window
 * by one input sample, which sounds plausible and is wrong. That single
 * failure is what this task's per-layer parity test exists to catch. Do not
 * "simplify" this.
 *
 * With ne = [F, T, C, N] both ne[0] and ne[1] are spatial; ne[2] (channels)
 * and ne[3] (batch) are left alone.
 */
static struct ggml_tensor *ss_pad_tf(struct ggml_context *g,
                                     struct ggml_tensor *x,
                                     int before, int after)
{
    return ggml_pad_ext(g, x, before, after, before, after, 0, 0, 0, 0);
}

/* The converter reduces every BatchNorm to a per-channel affine
 * y = a*x + b (design 4.3.1, "Note on activation ordering"), so applying it
 * is a broadcast multiply and add. bn_a/bn_b are 1-D [C]; reshaping them to
 * [1, 1, C, 1] is what makes ggml repeat them across frequency and time. */
static struct ggml_tensor *ss_bn(struct ggml_context *g,
                                 struct ggml_tensor *x,
                                 const SSLayer *l)
{
    struct ggml_tensor *a = ggml_reshape_4d(g, l->bn_a, 1, 1, l->bn_a->ne[0], 1);
    struct ggml_tensor *b = ggml_reshape_4d(g, l->bn_b, 1, 1, l->bn_b->ne[0], 1);

    return ggml_add(g, ggml_mul(g, x, a), b);
}

/* pad(1, 2) -> conv 5x5 stride 2 pad 0 -> bias. This is exactly Keras'
 * Conv2D(..., padding="same") output: Keras folds the bias into the layer's
 * result, so everything Spleeter calls "conv_n" includes it.
 *
 * Ruling 16: this uses ggml_conv_2d_direct with the kernel cast to F32, not
 * ggml_conv_2d. ggml_conv_2d lowers to im2col with dst_type = GGML_TYPE_F16,
 * so it rounds the *activations* to half precision as well as using the
 * model's F16 weights. That second rounding is a property of the op, not of
 * the artifact, and it is what stopped per-layer parity: measured against the
 * reference, ggml_conv_2d gives 0/12 layers at rtol 1e-3 while this form gives
 * 12/12. It is also 11% faster and 5 MiB lighter, because conv_2d_direct never
 * materialises the ~26 MiB F16 im2col buffer -- here the accurate path is also
 * the cheap one.
 *
 * The model on disk is untouched: it is still F16 (spleeter-2stems-f16.gguf,
 * 39,319,552 bytes) and ggml_cast is a runtime graph node. Do not "optimise"
 * this back to ggml_conv_2d; design section 9.2's parity test will catch it,
 * which is the point of that test. */
static struct ggml_tensor *ss_conv_bias(struct ggml_context *g,
                                        struct ggml_tensor *x,
                                        const SSLayer *l)
{
    struct ggml_tensor *bias = ggml_reshape_4d(g, l->b, 1, 1, l->b->ne[0], 1);
    struct ggml_tensor *w    = ggml_cast(g, l->w, GGML_TYPE_F32);

    x = ss_pad_tf(g, x, 1, 2);
    x = ggml_conv_2d_direct(g, w, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                            /*d0*/ 1, /*d1*/ 1);

    return ggml_add(g, x, bias);
}

/* One encoder block, with two outputs.
 *
 * Returns the activated tensor -- LeakyReLU(0.2)(BatchNorm(conv)) -- which
 * feeds the next encoder block. *raw_out receives the post-bias,
 * pre-BatchNorm convolution output, which feeds the decoder's skip
 * connection and is what the parity fixtures record.
 *
 * Both are needed. Upstream Spleeter concatenates the RAW tensors into the
 * decoder -- merge1 = [conv5, drop1] ... merge5 = [conv1, batch11] -- and
 * never the activated rel1..rel5 (design section 4.3.1(a)). A block that
 * returned only the activated tensor could not wire the skips correctly,
 * and that is a separation-quality bug, not just a failing fixture.
 *
 * "Raw" means after the bias add. Taking the tap before it would be wrong by
 * a per-channel constant -- exactly the kind of error that survives casual
 * listening.
 */
static struct ggml_tensor *ss_encoder_block(struct ggml_context *g,
                                            struct ggml_tensor *x,
                                            const SSLayer *l,
                                            struct ggml_tensor **raw_out)
{
    struct ggml_tensor *raw = ss_conv_bias(g, x, l);

    *raw_out = raw;

    return ggml_leaky_relu(g, ss_bn(g, raw, l), 0.2f, false);
}

/* Builds all six encoder blocks, filling raw[0..5] with conv1..conv6's raw
 * (post-bias, pre-BatchNorm) outputs -- the tensors the decoder's skips
 * consume and the parity fixtures record.
 *
 * conv6 is special and stops after the bias. Upstream computes batch6 and a
 * LeakyReLU of it and then discards both; up1 consumes raw conv6 (design
 * section 4.3.1(b)). So conv6's BatchNorm is not merely unused here, it is
 * never built: its bn_a/bn_b are still loaded and shape-validated by
 * ss_model_load(), because they must exist in the file, and they must never
 * be applied.
 */
static void ss_build_encoder(struct ggml_context *g,
                             struct ggml_tensor *input,
                             const SSNet *net,
                             struct ggml_tensor *raw[6])
{
    struct ggml_tensor *x = input;
    int n;

    av_assert0(input->ne[0] == SS_F);
    av_assert0(input->ne[1] == SS_T);
    av_assert0(input->ne[2] == SS_CHANNELS);
    av_assert0(input->ne[3] == 1);

    for (n = 0; n < 6; n++) {
        av_assert0(x->ne[2] == ss_conv_io[n][0]);

        if (n == 5)
            raw[n] = ss_conv_bias(g, x, &net->conv[n]);
        else
            x = ss_encoder_block(g, x, &net->conv[n], &raw[n]);

        /* Assert the shape before trusting the values: a shape assertion
         * that fires is worth ten minutes of debugging a parity mismatch.
         * Each block halves both spatial axes; the channel count comes from
         * the model's own (in, out) table rather than a second copy of it.
         * Expected, in [F, T, C] order: [512,256,16] [256,128,32]
         * [128,64,64] [64,32,128] [32,16,256] [16,8,512]. */
        av_assert0(raw[n]->ne[0] == SS_F >> (n + 1));
        av_assert0(raw[n]->ne[1] == SS_T >> (n + 1));
        av_assert0(raw[n]->ne[2] == ss_conv_io[n][1]);
        av_assert0(raw[n]->ne[3] == 1);
    }
}

/* Writes one tensor to <dump_dir>/<name>.f32 as raw float32 in ggml's native
 * order: ne = [F, T, C] with F fastest, i.e. a [C][T][F] byte layout. That is
 * exactly what tools/spleeter-gguf/compare.py reads back with
 * np.fromfile(...).reshape(C, T, F), and what dump_reference.py wrote. A
 * no-op when the internal `dump` option is unset.
 *
 * Returns an error rather than the void the brief specified, so a failed
 * write surfaces as a filter error here instead of as a puzzling "missing
 * dump file" from the comparator much later.
 */
static int ss_dump_tensor(AVFilterContext *ctx, const char *name,
                          struct ggml_tensor *t)
{
    StemSplitContext *s = ctx->priv;
    const size_t nb = ggml_nbytes(t);
    char path[1024];
    void *buf;
    FILE *f;
    size_t written;

    if (!s->dump_dir || !*s->dump_dir)
        return 0;

    av_assert0(t->type == GGML_TYPE_F32);
    av_assert0(ggml_is_contiguous(t));

    buf = av_malloc(nb);
    if (!buf)
        return AVERROR(ENOMEM);
    ggml_backend_tensor_get(t, buf, 0, nb);

    snprintf(path, sizeof(path), "%s/%s.f32", s->dump_dir, name);
    f = avpriv_fopen_utf8(path, "wb");
    if (!f) {
        av_log(ctx, AV_LOG_ERROR,
               "Could not open '%s' for writing; does the 'dump' directory "
               "exist?\n", path);
        av_free(buf);
        return AVERROR(EIO);
    }

    written = fwrite(buf, 1, nb, f);
    fclose(f);
    av_free(buf);

    if (written != nb) {
        av_log(ctx, AV_LOG_ERROR, "Short write to '%s'.\n", path);
        return AVERROR(EIO);
    }

    av_log(ctx, AV_LOG_VERBOSE,
           "stemsplit: dumped %s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] "
           "to %s\n", name, t->ne[0], t->ne[1], t->ne[2], path);

    return 0;
}

/* ---- Internal parity path: `debug_input` --------------------------------
 *
 * Design section 9.2 compares the network layer by layer against a Python
 * reference. Those fixtures were computed by
 * tools/spleeter-gguf/dump_reference.py from a fixed *synthetic* magnitude
 * spectrogram, not from audio, so the comparison has to inject that exact
 * tensor: feeding real audio through the STFT would hand the network a
 * completely different input and fail for a trivial reason. The STFT is
 * covered separately by Task 5's round-trip test, which is the point -- each
 * half is tested in isolation.
 *
 * The file is [C=2][T=512][F=1024] float32, frequency contiguous, which is
 * byte-for-byte ggml's ne = [F, T, C, 1], so it is read straight into the
 * input tensor with no reordering.
 */
static int ss_read_debug_input(AVFilterContext *ctx, float **out)
{
    StemSplitContext *s = ctx->priv;
    const size_t want = (size_t) SS_CHANNELS * SS_T * SS_F * sizeof(float);
    float *buf;
    FILE *f;
    long size;
    int ret = 0;

    *out = NULL;

    f = avpriv_fopen_utf8(s->debug_input_path, "rb");
    if (!f) {
        av_log(ctx, AV_LOG_ERROR, "Could not open debug_input file '%s'.\n",
               s->debug_input_path);
        return AVERROR(EIO);
    }

    if (fseek(f, 0, SEEK_END) < 0 || (size = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Could not size debug_input file '%s'.\n",
               s->debug_input_path);
        fclose(f);
        return AVERROR(EIO);
    }

    if ((uint64_t) size != (uint64_t) want) {
        av_log(ctx, AV_LOG_ERROR,
               "debug_input file '%s' is %ld bytes; expected %zu "
               "([C=%d][T=%d][F=%d] float32).\n",
               s->debug_input_path, size, want, SS_CHANNELS, SS_T, SS_F);
        fclose(f);
        return AVERROR(EINVAL);
    }

    buf = av_malloc(want);
    if (!buf) {
        fclose(f);
        return AVERROR(ENOMEM);
    }

    if (fread(buf, 1, want, f) != want) {
        av_log(ctx, AV_LOG_ERROR, "Short read from debug_input file '%s'.\n",
               s->debug_input_path);
        av_freep(&buf);
        ret = AVERROR(EIO);
    }

    fclose(f);
    *out = buf;

    return ret;
}

/* Builds, allocates, runs and dumps one instrument's encoder on `data`. */
static int ss_run_encoder_once(AVFilterContext *ctx, int k, const float *data)
{
    StemSplitContext *s = ctx->priv;
    const size_t input_bytes = (size_t) SS_CHANNELS * SS_T * SS_F * sizeof(float);
    struct ggml_init_params ip = {
        .mem_size   = ggml_tensor_overhead() * SS_GRAPH_TENSORS +
                      ggml_graph_overhead(),
        .mem_buffer = NULL,
        /* Metadata only: ggml_gallocr owns the activation memory. */
        .no_alloc   = true,
    };
    struct ggml_context *g;
    struct ggml_gallocr *galloc = NULL;
    struct ggml_cgraph *gf;
    struct ggml_tensor *input, *raw[6];
    char name[80];
    int n, ret = 0;

    g = ggml_init(ip);
    if (!g)
        return AVERROR(ENOMEM);

    input = ggml_new_tensor_4d(g, GGML_TYPE_F32, SS_F, SS_T, SS_CHANNELS, 1);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    /* Also flagged as an output: ggml_gallocr never recycles the memory of a
     * tensor flagged output, so the injected values survive the compute and
     * can be read back afterwards. That is what makes the dumped
     * <instrument>.input.f32 a genuine record of what the graph consumed
     * rather than a copy of what we meant to give it. */
    ggml_set_output(input);

    ss_build_encoder(g, input, &s->nets[k], raw);

    gf = ggml_new_graph(g);
    for (n = 0; n < 6; n++) {
        ggml_set_output(raw[n]);
        ggml_build_forward_expand(gf, raw[n]);
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s->backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        av_log(ctx, AV_LOG_ERROR,
               "Could not allocate the stemsplit compute graph.\n");
        ret = AVERROR(ENOMEM);
        goto done;
    }

    /* Only now that gallocr has backed the graph does the input tensor have
     * a buffer to be written into. */
    ggml_backend_tensor_set(input, data, 0, input_bytes);

    ggml_backend_cpu_set_n_threads(s->backend, s->nb_threads > 0 ? s->nb_threads
                                   : FFMAX(1, ff_filter_get_nb_threads(ctx)));

    if (ggml_backend_graph_compute(s->backend, gf) != GGML_STATUS_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "stemsplit: graph computation failed.\n");
        ret = AVERROR_EXTERNAL;
        goto done;
    }

    snprintf(name, sizeof(name), "%s.input", s->instrument_names[k]);
    ret = ss_dump_tensor(ctx, name, input);

    for (n = 0; ret >= 0 && n < 6; n++) {
        snprintf(name, sizeof(name), "%s.conv%d", s->instrument_names[k], n + 1);
        ret = ss_dump_tensor(ctx, name, raw[n]);
    }

done:
    ggml_gallocr_free(galloc);
    ggml_free(g);

    return ret;
}

/* Runs every instrument's encoder once on the injected spectrogram. */
static int ss_run_debug_input(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    float *data = NULL;
    int k, ret;

    ret = ss_read_debug_input(ctx, &data);
    if (ret < 0)
        return ret;

    for (k = 0; k < s->nb_instruments; k++) {
        ret = ss_run_encoder_once(ctx, k, data);
        if (ret < 0)
            break;
        av_log(ctx, AV_LOG_INFO,
               "stemsplit: debug_input encoder pass complete for '%s'.\n",
               s->instrument_names[k]);
    }

    av_freep(&data);

    return ret;
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

    /* Ruling 2: model loading is skippable so passthrough_dsp's round-trip
     * test (Task 5) keeps working with no model at all. ss_model_load()
     * does its own "no model given" check for the real (non-passthrough)
     * path -- see design section 10, row 1. */
    if (!s->passthrough_dsp) {
        ret = ss_model_load(ctx);
        if (ret < 0)
            return ret;
    }

    /* Internal parity path (design section 9.2): when `debug_input` names a
     * raw spectrogram, run the network on it once here and dump the taps the
     * fixtures record, bypassing the STFT entirely. */
    if (!s->passthrough_dsp && s->debug_input_path && *s->debug_input_path) {
        ret = ss_run_debug_input(ctx);
        if (ret < 0)
            return ret;
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

    ss_model_free(s);

    /* ggml_log_set() is process-global, not per-context: ss_model_load()
     * registered cb_log with THIS ctx as its user_data. If we leave that
     * registration in place, any ggml call anywhere in the process after
     * this AVFilterContext is freed -- including from an unrelated whisper
     * filter instance still running in the same filtergraph/process, which
     * this media server routinely does for subtitles -- invokes cb_log with
     * a dangling AVFilterContext* and writes through it via av_log(): a
     * use-after-free. Restore ggml's default sink instead. If another ggml
     * user (e.g. whisper) is still live, its messages just fall back to
     * ggml's default logging rather than being routed through freed memory
     * -- degraded logging beats a crash. Do not remove this. */
    ggml_log_set(NULL, NULL);

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
