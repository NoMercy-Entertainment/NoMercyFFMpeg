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
 * The filter negotiates 44.1 kHz stereo planar float, loads and validates the
 * model's 100 GGUF tensors (ss_model_load), builds the two U-Nets into a
 * single ggml graph once (ss_graph_build) and reuses it for every segment.
 * The segment driver (ss_push_samples / ss_process_segment) runs a 4096-point
 * STFT at hop 1024, hands each 512-frame segment's magnitude to the networks,
 * turns their estimated magnitudes into ratio masks (ss_build_masks), extends
 * those masks over the band the network does not model (`highband`), applies
 * them to the ORIGINAL complex mixture spectrum -- the mixture's phase is
 * reused unchanged -- and resynthesises one stem per output pad.
 *
 * `debug_input` remains an internal parity hook: it injects a spectrogram
 * straight from disk, runs the networks on it once and dumps every layer tap
 * for the per-layer comparison of design section 9.2. It bypasses the audio
 * driver entirely.
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

/* Stride between consecutive spectra in the segment ring, in complex floats.
 *
 * av_tx writes its output with aligned SIMD stores unless AV_TX_UNALIGNED was
 * requested at init, so every spectrum it is handed must start on a 64-byte
 * boundary. SS_BINS is 2049 -- odd -- so packing spectra back to back would
 * put every second one 8 bytes off and fault on the first vmovaps. Rounding
 * the stride up to a multiple of 8 complex floats makes each spectrum start
 * 16448 bytes (257 * 64) apart, and av_calloc already returns a 64-byte
 * aligned base. The padding bins are never read. */
#define SS_SPEC_STRIDE  FFALIGN(SS_BINS, 8)

enum { SS_STEM_VOCALS, SS_STEM_ACCOMPANIMENT, SS_STEM_ALL };
enum { SS_HB_PASSTHROUGH, SS_HB_ZEROS, SS_HB_AVERAGE };

/* Index into a per-segment mask: [C][T][SS_BINS] float, bin fastest. The
 * network's own tensors are [C][T][SS_F] instead -- the model only predicts
 * the lower SS_F of the SS_BINS bins -- so the two layouts are deliberately
 * not interchangeable and get separate accessors. */
static inline size_t ss_idx(int c, int t, int f)
{
    return ((size_t) c * SS_T + t) * SS_BINS + f;
}

/* Index into a network input/output tensor: [C][T][SS_F] float. */
static inline size_t ss_nidx(int c, int t, int f)
{
    return ((size_t) c * SS_T + t) * SS_F + f;
}

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

    /* STFT / inverse STFT (Task 5): a 4096-point av_tx real DFT pair at hop
     * 1024, with a periodic Hann window applied on both analysis and
     * synthesis. window/in_buf/spec are scratch owned by the context so
     * ss_analyze()/ss_synthesize() don't allocate per call.
     *
     * There is deliberately no overlap-add accumulator here. Synthesis writes
     * into a caller-supplied accumulator instead -- s->ola[pad][channel], one
     * per output pad, owned by the segment driver -- because the number of
     * accumulators depends on how many output pads `stem` produced, which the
     * DSP layer knows nothing about. */
    AVTXContext    *tx_fwd, *tx_inv;
    av_tx_fn        tx_fwd_fn, tx_inv_fn;
    float          *window;   /* SS_FRAME_LENGTH, periodic Hann */
    float          *in_buf;   /* SS_FRAME_LENGTH scratch: windowed real input
                                * for analysis, real ifft output for synthesis */
    AVComplexFloat *spec;     /* SS_BINS scratch: mutable copy of the spectrum
                                * ss_synthesize() hands to the inverse transform,
                                * which overwrites its input */

    /* Model loading (Task 6): 100 GGUF tensors resolved by name into two
     * SSNet structures and validated by dtype and shape. */
    struct ggml_context        *gguf_ctx;    /* owns all tensor metadata and data;
                                               * created by gguf_init_from_file with
                                               * no_alloc=false, freed in ss_model_free */
    struct ggml_backend        *backend;     /* CPU backend */
    SSNet    nets[SS_NB_INSTRUMENTS];
    int      nb_instruments;
    char    *instrument_names[SS_NB_INSTRUMENTS];

    /* Compute graph (design 6.3: "built once and reused"). Both instruments'
     * networks live in ONE graph sharing ONE input tensor, so ggml_gallocr's
     * liveness analysis can hand the second network the memory the first has
     * finished with -- two separate graphs would keep both networks'
     * activations resident at once. Built lazily on the first inference and
     * kept for the whole filter's life; see ss_graph_build. */
    struct ggml_context *graph_ctx;
    struct ggml_cgraph  *graph;
    struct ggml_gallocr *galloc;
    struct ggml_tensor  *g_input;
    struct ggml_tensor  *g_est[SS_NB_INSTRUMENTS];   /* estimated magnitude S_i */
    struct ggml_tensor  *g_mask[SS_NB_INSTRUMENTS];  /* the raw sigmoid, a fixture tap */
    struct ggml_tensor  *g_raw[SS_NB_INSTRUMENTS][6];
    struct ggml_tensor  *g_up[SS_NB_INSTRUMENTS][6];

    /* ---- Segment driver (Task 9) ----------------------------------------
     *
     * Analysis slides an SS_FRAME_LENGTH window over the input in SS_FRAME_STEP
     * hops, starting from an all-zero window: that zero window IS the design's
     * 4096-sample lead-in (design 4.2 step 1), and crop_remaining discards the
     * matching 4096 output samples again at step 9.
     *
     * Frame spectra land in a ring of exactly SS_T slots. Segment k covers
     * global frames [k*hop_frames, k*hop_frames + SS_T), which is the whole
     * ring, and the frames it retires -- [k*hop_frames, (k+1)*hop_frames) --
     * are precisely the slots the next segment's new frames overwrite, so the
     * ring never needs to be larger than one segment.
     */
    int      overlap_frames;  /* `overlap` converted to frames, clamped [0, SS_T-1] */
    int      hop_frames;      /* SS_T - overlap_frames: frames retired per segment */
    int      nb_outputs;      /* 2 for stem=all, else 1 */
    int      out_inst[SS_NB_INSTRUMENTS];  /* output pad -> instrument index */

    float   *an_window[SS_CHANNELS]; /* SS_FRAME_LENGTH sliding analysis window */
    float   *an_stage[SS_CHANNELS];  /* SS_FRAME_STEP staging for a partial hop */
    int      an_stage_fill;
    AVComplexFloat *spec_ring;       /* [SS_T][SS_CHANNELS][SS_SPEC_STRIDE] mixture STFT */
    float   *mag;                    /* [C][T][F] network input, one segment */
    float   *est[SS_NB_INSTRUMENTS]; /* [C][T][F] estimated magnitudes */
    float   *mask[SS_NB_INSTRUMENTS];/* [C][T][SS_BINS] this segment's ratio masks */

    /* Only allocated when overlap != 0. Overlapping segments disagree about
     * the mask for the frames they share, so each segment's masks are summed
     * into a ring with a per-frame taper (seg_win) and divided by the summed
     * weight (w_acc) when the frame retires. At overlap=0 every weight is 1
     * and exactly one segment covers each frame, so this whole path is skipped
     * and the result is bit-identical to the unblended mask. */
    float   *mask_acc[SS_NB_INSTRUMENTS];
    float   *w_acc;                  /* [SS_T] summed taper per ring slot */
    float   *seg_win;                /* [SS_T] per-frame taper within a segment */
    int64_t  acc_filled;             /* global frame index accumulation has reached */

    AVComplexFloat *work_spec;              /* SS_BINS scratch: masked spectrum */
    float   *ola[SS_NB_INSTRUMENTS][SS_CHANNELS]; /* SS_FRAME_LENGTH per output pad */

    int64_t  frames_in;      /* frames analysed into the ring */
    int64_t  frames_done;    /* frames retired: masked, synthesised, emitted */
    int64_t  next_segment;   /* index of the next segment to run */
    int64_t  crop_remaining; /* lead-in output samples still to discard */
    int64_t  total_in;       /* real input samples received */
    int64_t  emitted;        /* real (post-crop) output samples emitted */
    int      started;        /* the zero lead-in frame has been pushed */
    int      flushing;       /* inside the EOF drain: cap output at total_in */

    int      eof;
    int      status;
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
    if (!s->window || !s->in_buf || !s->spec)
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
 * overlap-adds the whole SS_FRAME_LENGTH result into `ola`. The caller slides
 * `ola` by one hop after each frame (ss_emit_frames), so synthesis always
 * starts at its origin. */
static void ss_synthesize(StemSplitContext *s, const AVComplexFloat *in, float *ola)
{
    /* The inverse RDFT overwrites its input (libavutil/tx.h), so hand it a
     * scratch copy rather than the caller's (possibly shared) spectrum. */
    memcpy(s->spec, in, SS_BINS * sizeof(*s->spec));
    s->tx_inv_fn(s->tx_inv, s->in_buf, s->spec, sizeof(*s->spec));

    for (int n = 0; n < SS_FRAME_LENGTH; n++)
        ola[n] += s->window[n] * s->in_buf[n] * SS_WINDOW_COMPENSATION;
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

/* Defined with the compute graph further down; declared here because
 * ss_model_load() must tear the graph down before it frees the weights the
 * graph's tensors point into. */
static void ss_graph_free(StemSplitContext *s);

/* Instrument names come out of the model file and are used to build
 * filesystem paths (ss_dump_tensor), filtergraph pad labels
 * (ss_append_outputs) and tensor names (ss_resolve_layer), so they are
 * untrusted input reaching three interfaces that all attach meaning to
 * punctuation. Restrict them to [A-Za-z0-9_-], 1..31 characters: long enough
 * for any plausible stem name, short enough that SS_NB_INSTRUMENTS of them
 * always fit in ss_join_instruments' buffer, and narrow enough that '/', '\'
 * and ".." can never appear. A model naming a stem "../../../etc/x" would
 * otherwise have ss_dump_tensor write outside the `dump` directory. */
static int ss_valid_instrument_name(const char *name)
{
    size_t len = strlen(name);
    size_t i;

    if (len < 1 || len > 31)
        return 0;
    for (i = 0; i < len; i++) {
        const char c = name[i];

        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return 0;
    }

    return 1;
}

static int ss_model_load(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    struct gguf_init_params gp = { .no_alloc = false, .ctx = &s->gguf_ctx };
    struct gguf_context *gguf = NULL;
    int64_t key;
    size_t n_inst;
    const char *arch;
    /* 34 per instrument: 31 name characters (ss_valid_instrument_name's cap),
     * ", " and the terminator, so a validated instrument list never has to be
     * truncated into an error message. */
    char avail[SS_NB_INSTRUMENTS * 34];
    int k, n, i;
    int ret;

    /* Defensive idempotency: no known FFmpeg path calls ss_model_load()
     * twice on a live context, but if one ever does, s->backend, s->gguf_ctx
     * and s->instrument_names[] would each be silently overwritten and
     * leaked rather than freed.
     *
     * The graph goes first, and the order is load-bearing rather than
     * cosmetic: every tensor in s->graph_ctx (g_input, g_raw, g_up, g_est,
     * g_mask) points at weight data owned by s->gguf_ctx, which ss_model_free
     * releases. Freeing only the model would leave a live graph of dangling
     * weight pointers, and the next ss_infer() would compute through freed
     * memory instead of failing. This is uninit()'s ordering, for the same
     * reason. Together the two calls make "a load always starts from a clean
     * slate, with nothing left pointing into the old one" true rather than
     * merely intended. */
    ss_graph_free(s);
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
    for (i = 0; i < SS_NB_INSTRUMENTS; i++) {
        if (!ss_valid_instrument_name(s->instrument_names[i])) {
            av_log(ctx, AV_LOG_ERROR,
                   "Model '%s' names instrument %d '%s'. Instrument names "
                   "become output pad labels and dump file names, so they "
                   "are restricted to 1 to 31 characters of "
                   "[A-Za-z0-9_-].\n",
                   s->model_path, i, s->instrument_names[i]);
            ret = AVERROR(EINVAL);
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

/* Graph-metadata budget for the full forward pass of BOTH instruments,
 * which now share a single graph. An encoder block builds about ten tensors
 * (pad, kernel cast, conv, bias reshape + add, two BatchNorm reshapes + mul
 * + add, leaky_relu) and a decoder block about twelve (kernel cast,
 * transposed conv, crop view + cont, bias reshape + add, relu, two BatchNorm
 * reshapes + mul + add, concat), with six more for the output layer --
 * roughly 140 per instrument, so about 280 in total. The margin is
 * deliberate; this is a tripwire that fails loudly rather than a target. */
#define SS_GRAPH_TENSORS 1024

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

/* ---- Decoder graph (Task 8) ---------------------------------------------
 *
 * Six blocks of Conv2DTranspose 5x5 stride 2 "SAME" + bias, then ReLU,
 * BatchNorm and a concatenation with the matching *raw* encoder tensor,
 * followed by the dilated output convolution and the sigmoid mask.
 *
 * Two asymmetries with the encoder are real and must not be tidied away:
 *
 *  - The decoder activates BEFORE normalising (convT -> ReLU -> BatchNorm),
 *    where the encoder normalises first (conv -> BatchNorm -> LeakyReLU).
 *    That is why the converter cannot fold BatchNorm into the convolution and
 *    exports every one of them as a standalone per-channel affine instead
 *    (design 4.3.1, "Note on activation ordering").
 *  - Where the encoder pads before convolving, the decoder crops after: see
 *    ss_crop2d.
 *
 * Layout is ggml's ne = [F, T, C, N] throughout, as in the encoder.
 */

/* The decoder's half of TensorFlow's asymmetric "SAME".
 *
 * ggml_conv_transpose_2d_p0 does no padding at all, so a stride-2 kernel-5
 * transpose of an N-wide input emits (N - 1) * 2 + 5 == 2N + 3 columns.
 * TensorFlow's Conv2DTranspose with padding="same" emits 2N. The three extra
 * columns are exactly the ones the forward convolution's padding introduced,
 * and they come off the same sides: the forward pass padded 1 before and 2
 * after (ss_pad_tf), so the transpose drops 1 from the front and 2 from the
 * back, on both spatial axes. That lands the result back on the very grid the
 * encoder sampled, which is what makes the skip concatenation meaningful.
 *
 * Cropping symmetrically -- or 2 from the front and 1 from the back -- gives
 * the same output *shape* and shifts the whole feature map by one sample,
 * the decoder's equivalent of the encoder's symmetric-padding trap. Design
 * section 9.2's per-layer parity test is what catches it.
 *
 * The view is not contiguous (it keeps the parent's row stride), and the next
 * convolution requires contiguous input, hence the ggml_cont. `off` is applied
 * to ne[0] and ne[1] only; channels and batch are taken whole.
 */
static struct ggml_tensor *ss_crop2d(struct ggml_context *g,
                                     struct ggml_tensor *x,
                                     int off, int n0, int n1)
{
    struct ggml_tensor *v;

    av_assert0(off >= 0 && n0 > 0 && n1 > 0);
    av_assert0(off + n0 <= x->ne[0] && off + n1 <= x->ne[1]);

    v = ggml_view_4d(g, x, n0, n1, x->ne[2], x->ne[3],
                     x->nb[1], x->nb[2], x->nb[3],
                     (size_t) off * x->nb[0] + (size_t) off * x->nb[1]);

    return ggml_cont(g, v);
}

/* One decoder block.
 *
 * Returns the tensor that feeds the next block -- the concatenation of the
 * skip tensor and this block's output, or just this block's output when
 * `skip` is NULL (up6, which has no matching encoder layer). *tap_out
 * receives the pre-concatenation tensor, which is what design section 9.2's
 * fixtures record for up1..up6; the extra out-parameter mirrors
 * ss_encoder_block, which needs one for the same reason.
 *
 * Concat order is [skip, x] -- the encoder tensor FIRST. Upstream Spleeter
 * writes Concatenate(axis=-1)([conv5, drop1]) and so on, and axis=-1 is the
 * channel axis, which is ggml's ne[2] here. Reversing the two operands would
 * silently permute the next convolution's input channels: no shape changes,
 * no error is raised, and the output is wrong everywhere.
 *
 * Ruling 16's F32 kernel cast applies here too, and it matters more than it
 * did in the encoder. Handed an F16 kernel, ggml_conv_transpose_2d_p0 rounds
 * the input activations to F16 as well and lands about 2e-4 relative away
 * from a double-precision reference; handed an F32 kernel it computes in F32
 * throughout and lands at 6.5e-8. Both figures were measured against the
 * pinned ggml before this code was written, along with a check that its
 * work-buffer sizing is type-aware (4245760 bytes for an F32 kernel where an
 * F16 one gets 2123008), so the F32 path is a real path and not an overflow
 * waiting to happen. Six of these layers compound, so do not revert the cast.
 */
static struct ggml_tensor *ss_decoder_block(struct ggml_context *g,
                                            struct ggml_tensor *x,
                                            struct ggml_tensor *skip,
                                            const SSLayer *l,
                                            struct ggml_tensor **tap_out)
{
    struct ggml_tensor *bias = ggml_reshape_4d(g, l->b, 1, 1, l->b->ne[0], 1);
    struct ggml_tensor *w    = ggml_cast(g, l->w, GGML_TYPE_F32);

    x = ggml_conv_transpose_2d_p0(g, w, x, /*stride*/ 2);
    x = ss_crop2d(g, x, /*off*/ 1, (int) (x->ne[0] - 3), (int) (x->ne[1] - 3));
    x = ggml_add(g, x, bias);
    x = ggml_relu(g, x);        /* the decoder activates BEFORE normalising */
    x = ss_bn(g, x, l);

    *tap_out = x;

    return skip ? ggml_concat(g, skip, x, /*dim*/ 2) : x;
}

/* Builds the six decoder blocks and the output layer.
 *
 * up[0..5] receive each block's pre-concatenation output, the tensors the
 * parity fixtures record. *mask_out receives the sigmoid output of the final
 * convolution -- also a fixture tap, named "out" -- and *est_out receives
 * that mask multiplied by the network input, which is this instrument's
 * estimated magnitude spectrogram.
 *
 * up1 is fed raw conv6, not an activated tensor: upstream computes conv6's
 * BatchNorm and LeakyReLU and discards both (design 4.3.1(b)), which is why
 * ss_build_encoder stops the sixth block after its bias.
 *
 * The output convolution is kernel 4 with dilation 2, so its effective width
 * is (4 - 1) * 2 + 1 == 7 and "SAME" needs 6 columns of padding split evenly,
 * 3 each side. That one is symmetric, so ggml's own p0/p1 expresses it and no
 * ss_pad_tf/ss_crop2d dance is needed. Size check: 1024 + 6 - 6 == 1024.
 */
static void ss_build_decoder(struct ggml_context *g,
                             struct ggml_tensor *input,
                             struct ggml_tensor *const raw[6],
                             const SSNet *net,
                             struct ggml_tensor *up[6],
                             struct ggml_tensor **mask_out,
                             struct ggml_tensor **est_out)
{
    /* up1..up5 concatenate with raw conv5..conv1; up6 has no skip. */
    static const int skip_for[6] = { 4, 3, 2, 1, 0, -1 };
    struct ggml_tensor *x = raw[5];
    struct ggml_tensor *bias, *w;
    int n;

    for (n = 0; n < 6; n++) {
        av_assert0(x->ne[2] == ss_up_io[n][0]);

        x = ss_decoder_block(g, x,
                             skip_for[n] >= 0 ? raw[skip_for[n]] : NULL,
                             &net->up[n], &up[n]);

        /* Each block doubles both spatial axes, so up_n is the mirror of
         * conv_{6-n}. Expected, in [F, T, C] order: [32,16,256] [64,32,128]
         * [128,64,64] [256,128,32] [512,256,16] [1024,512,1]. Asserted before
         * the values are trusted, exactly as in ss_build_encoder. */
        av_assert0(up[n]->ne[0] == SS_F >> (5 - n));
        av_assert0(up[n]->ne[1] == SS_T >> (5 - n));
        av_assert0(up[n]->ne[2] == ss_up_io[n][1]);
        av_assert0(up[n]->ne[3] == 1);
    }

    av_assert0(x == up[5] && x->ne[2] == 1);

    bias = ggml_reshape_4d(g, net->out.b, 1, 1, net->out.b->ne[0], 1);
    w    = ggml_cast(g, net->out.w, GGML_TYPE_F32);

    x = ggml_conv_2d_direct(g, w, x, /*s0*/ 1, /*s1*/ 1, /*p0*/ 3, /*p1*/ 3,
                            /*d0*/ 2, /*d1*/ 2);
    x = ggml_add(g, x, bias);
    x = ggml_sigmoid(g, x);

    av_assert0(x->ne[0] == SS_F);
    av_assert0(x->ne[1] == SS_T);
    av_assert0(x->ne[2] == SS_CHANNELS);
    av_assert0(x->ne[3] == 1);

    *mask_out = x;

    /* The mask times the input is this instrument's estimated magnitude, and
     * that is where this task stops. Turning the per-instrument estimates
     * into the ratio mask that actually gets applied to the mixture spectrum
     * needs every instrument at once and belongs to the segment driver
     * (design section 4.2); nothing here may anticipate it. */
    *est_out = ggml_mul(g, x, input);
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

/* ---- Inference driver ---------------------------------------------------
 *
 * The production entry point into the network. Design section 6.3 says the
 * graph is "built once and reused"; ss_graph_build does that, and ss_infer
 * only writes a new segment's magnitude into the shared input tensor and
 * recomputes.
 *
 * It is built lazily, on the first inference, rather than in init(). FFmpeg
 * constructs a filtergraph twice, so init() runs twice per command line and a
 * graph built there would be allocated twice -- once for a context that never
 * sees a sample. "Once" is the property design 6.3 asks for; init() is only
 * where it expected to get it.
 *
 * Both instruments share ONE ggml graph and ONE input tensor. They are
 * independent subgraphs expanded back to back, so ggml_gallocr can hand the
 * second network the activation memory the first has finished with; two
 * separate graphs would need two buffers sized for the peak, both resident.
 */

static void ss_graph_free(StemSplitContext *s)
{
    if (s->galloc)
        ggml_gallocr_free(s->galloc);
    s->galloc = NULL;

    if (s->graph_ctx)
        ggml_free(s->graph_ctx);
    s->graph_ctx = NULL;

    /* All tensors below lived in graph_ctx, freed above: NULL them so a stray
     * use-after-free reads NULL rather than garbage. */
    s->graph   = NULL;
    s->g_input = NULL;
    memset(s->g_est,  0, sizeof(s->g_est));
    memset(s->g_mask, 0, sizeof(s->g_mask));
    memset(s->g_raw,  0, sizeof(s->g_raw));
    memset(s->g_up,   0, sizeof(s->g_up));
}

static int ss_graph_build(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    /* Design section 9.2's parity taps are pinned as graph outputs so
     * ggml_gallocr cannot recycle them before they are dumped. That pinning
     * costs about 43 MiB per instrument, so it is done only when `dump` is
     * actually set -- a production run keeps just the two estimates alive. */
    const int taps = s->dump_dir && *s->dump_dir;
    struct ggml_init_params ip = {
        .mem_size   = ggml_tensor_overhead() * SS_GRAPH_TENSORS +
                      ggml_graph_overhead(),
        .mem_buffer = NULL,
        /* Metadata only: ggml_gallocr owns the activation memory. */
        .no_alloc   = true,
    };
    struct ggml_context *g;
    struct ggml_cgraph *gf;
    int k, n;

    av_assert0(!s->graph_ctx);

    g = ggml_init(ip);
    if (!g)
        return AVERROR(ENOMEM);
    s->graph_ctx = g;

    s->g_input = ggml_new_tensor_4d(g, GGML_TYPE_F32, SS_F, SS_T, SS_CHANNELS, 1);
    ggml_set_name(s->g_input, "input");
    ggml_set_input(s->g_input);
    if (taps)
        /* ggml_gallocr never recycles a tensor flagged output, so the injected
         * values survive the compute and the dumped <instrument>.input.f32 is
         * a genuine record of what the graph consumed rather than a copy of
         * what we meant to give it. */
        ggml_set_output(s->g_input);

    gf = ggml_new_graph(g);
    if (!gf) {
        ss_graph_free(s);
        return AVERROR(ENOMEM);
    }

    for (k = 0; k < s->nb_instruments; k++) {
        ss_build_encoder(g, s->g_input, &s->nets[k], s->g_raw[k]);
        ss_build_decoder(g, s->g_input, s->g_raw[k], &s->nets[k], s->g_up[k],
                         &s->g_mask[k], &s->g_est[k]);

        if (taps) {
            for (n = 0; n < 6; n++) {
                ggml_set_output(s->g_raw[k][n]);
                ggml_build_forward_expand(gf, s->g_raw[k][n]);
            }
            for (n = 0; n < 6; n++) {
                ggml_set_output(s->g_up[k][n]);
                ggml_build_forward_expand(gf, s->g_up[k][n]);
            }
            ggml_set_output(s->g_mask[k]);
            ggml_build_forward_expand(gf, s->g_mask[k]);
        }

        /* The estimated magnitude is what the ratio-mask formula consumes, so
         * it is pinned on every path, dumping or not. Expanding it pulls in
         * that instrument's entire network. */
        ggml_set_output(s->g_est[k]);
        ggml_build_forward_expand(gf, s->g_est[k]);
    }

    s->graph = gf;

    s->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s->backend));
    if (!s->galloc || !ggml_gallocr_alloc_graph(s->galloc, gf)) {
        av_log(ctx, AV_LOG_ERROR,
               "Could not allocate the stemsplit compute graph.\n");
        ss_graph_free(s);
        return AVERROR(ENOMEM);
    }

    av_log(ctx, AV_LOG_VERBOSE,
           "stemsplit: compute graph built once, %d node(s) for %d "
           "instrument(s), %.1f MiB of activations.\n",
           ggml_graph_n_nodes(gf), s->nb_instruments,
           ggml_gallocr_get_buffer_size(s->galloc, 0) / (1024.0 * 1024.0));

    return 0;
}

/* Dumps every parity tap of the last compute. A no-op unless `dump` is set,
 * in which case ss_graph_build has pinned the tensors this reads. */
static int ss_dump_taps(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    char name[80];
    int k, n, ret = 0;

    if (!s->dump_dir || !*s->dump_dir)
        return 0;

    for (k = 0; ret >= 0 && k < s->nb_instruments; k++) {
        snprintf(name, sizeof(name), "%s.input", s->instrument_names[k]);
        ret = ss_dump_tensor(ctx, name, s->g_input);

        for (n = 0; ret >= 0 && n < 6; n++) {
            snprintf(name, sizeof(name), "%s.conv%d", s->instrument_names[k], n + 1);
            ret = ss_dump_tensor(ctx, name, s->g_raw[k][n]);
        }
        for (n = 0; ret >= 0 && n < 6; n++) {
            snprintf(name, sizeof(name), "%s.up%d", s->instrument_names[k], n + 1);
            ret = ss_dump_tensor(ctx, name, s->g_up[k][n]);
        }
        if (ret >= 0) {
            /* "out" is the sigmoid mask, matching dump_reference.py's tap: it
             * records `out` and multiplies separately into an untapped tensor.
             * The estimated magnitude goes out through ss_infer's `out`
             * parameter, not through the fixture named "out". */
            snprintf(name, sizeof(name), "%s.out", s->instrument_names[k]);
            ret = ss_dump_tensor(ctx, name, s->g_mask[k]);
        }
        if (ret >= 0) {
            /* Not a fixture -- compare.py iterates fixture keys and never
             * looks for this -- but it makes the mask multiply checkable from
             * outside the filter: est must equal out * input, element for
             * element. */
            snprintf(name, sizeof(name), "%s.est", s->instrument_names[k]);
            ret = ss_dump_tensor(ctx, name, s->g_est[k]);
        }
    }

    return ret;
}

/* Runs both networks over one segment's magnitude spectrogram.
 *
 * `mag` is [C=2][T=512][F=1024] float32, frequency contiguous -- byte for byte
 * ggml's ne = [F, T, C, 1]. out[i], when non-NULL, receives instrument i's
 * ESTIMATED MAGNITUDE S_i = sigmoid_mask * mag in the same layout (Ruling 17),
 * not the bare sigmoid. The caller owns both buffers.
 */
static int ss_infer(AVFilterContext *ctx, const float *mag,
                    float *out[SS_NB_INSTRUMENTS])
{
    StemSplitContext *s = ctx->priv;
    const size_t bytes = (size_t) SS_CHANNELS * SS_T * SS_F * sizeof(float);
    int k, ret;

    if (!s->graph) {
        ret = ss_graph_build(ctx);
        if (ret < 0)
            return ret;
    }

    ggml_backend_tensor_set(s->g_input, mag, 0, bytes);

    ggml_backend_cpu_set_n_threads(s->backend, s->nb_threads > 0 ? s->nb_threads
                                   : FFMAX(1, ff_filter_get_nb_threads(ctx)));

    if (ggml_backend_graph_compute(s->backend, s->graph) != GGML_STATUS_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "stemsplit: graph computation failed.\n");
        return AVERROR_EXTERNAL;
    }

    for (k = 0; k < s->nb_instruments; k++) {
        if (!out || !out[k])
            continue;
        av_assert0(ggml_nbytes(s->g_est[k]) == bytes);
        ggml_backend_tensor_get(s->g_est[k], out[k], 0, bytes);
    }

    return ss_dump_taps(ctx);
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

/* Runs the whole network once per instrument on the injected spectrogram.
 *
 * The estimated-magnitude buffers are allocated and thrown away here. They
 * are not wasted: they are what makes ss_infer's output path -- the thing the
 * segment driver will actually consume -- run under the parity test rather
 * than sitting untested until a later task wires it up.
 */
static int ss_run_debug_input(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    const size_t bytes = (size_t) SS_CHANNELS * SS_T * SS_F * sizeof(float);
    float *out[SS_NB_INSTRUMENTS] = { NULL };
    float *data = NULL;
    int k, ret;

    ret = ss_read_debug_input(ctx, &data);
    if (ret < 0)
        return ret;

    for (k = 0; k < s->nb_instruments; k++) {
        out[k] = av_malloc(bytes);
        if (!out[k]) {
            ret = AVERROR(ENOMEM);
            goto done;
        }
    }

    ret = ss_infer(ctx, data, out);
    if (ret >= 0)
        av_log(ctx, AV_LOG_INFO,
               "stemsplit: debug_input forward pass complete for %d "
               "instrument(s).\n", s->nb_instruments);

done:
    for (k = 0; k < SS_NB_INSTRUMENTS; k++)
        av_freep(&out[k]);
    av_freep(&data);

    return ret;
}

/* ---- Ratio masks and high-band extension --------------------------------
 *
 * Design section 4.2 step 5, reproduced exactly:
 *
 *     output_sum = sum_j (S_j ^ 2) + EPSILON
 *     mask_i     = (S_i ^ 2 + EPSILON / N) / output_sum
 *
 * EPSILON appears TWICE with different scaling -- whole in the denominator,
 * divided by N in the numerator. That is upstream Spleeter's formula, not a
 * typo, and it is what makes the masks sum to exactly 1 when every estimate is
 * zero (silence) instead of 0/0. Do not "simplify" the two terms into one.
 *
 * The separation exponent is 2, so S_i^2 is written out rather than reached
 * through powf: a fixed exponent of 2 is the released 2stems configuration.
 *
 * Above bin SS_F (11 025 Hz) the network predicts nothing at all, so the mask
 * there is a policy decision rather than a computation -- design section 7.
 */
static void ss_build_masks(StemSplitContext *s, const float *est[SS_NB_INSTRUMENTS],
                           float *mask[SS_NB_INSTRUMENTS])
{
    const int n = s->nb_instruments;
    int c, t, f, i;

    for (c = 0; c < SS_CHANNELS; c++)
        for (t = 0; t < SS_T; t++) {
            for (f = 0; f < SS_F; f++) {
                const size_t k = ss_nidx(c, t, f);
                float sum = SS_EPSILON;

                for (i = 0; i < n; i++)
                    sum += est[i][k] * est[i][k];
                for (i = 0; i < n; i++)
                    mask[i][ss_idx(c, t, f)] =
                        (est[i][k] * est[i][k] + SS_EPSILON / n) / sum;
            }

            /* ---- the band the network does not model ---- */
            switch (s->highband) {
            case SS_HB_ZEROS:
                /* Upstream Spleeter ("mask_extension": "zeros"): silent above
                 * 11 kHz in every stem. Kept so the parity tests have
                 * something bit-comparable to compare against. */
                for (i = 0; i < n; i++)
                    memset(mask[i] + ss_idx(c, t, SS_F), 0,
                           (SS_BINS - SS_F) * sizeof(float));
                break;

            case SS_HB_AVERAGE:
                /* Spleeter's other documented mode: this frame's mask averaged
                 * over the modelled bins, tiled across the rest. */
                for (i = 0; i < n; i++) {
                    const float *m = mask[i] + ss_idx(c, t, 0);
                    float *hi = mask[i] + ss_idx(c, t, SS_F);
                    double acc = 0.0;
                    float v;

                    for (f = 0; f < SS_F; f++)
                        acc += m[f];
                    v = (float) (acc / SS_F);
                    for (f = 0; f < SS_BINS - SS_F; f++)
                        hi[f] = v;
                }
                break;

            case SS_HB_PASSTHROUGH:
            default:
                /* The default, and an explicit product decision: the untouched
                 * high band goes to exactly one stem -- the LAST instrument in
                 * the model's list, which for 2stems is accompaniment -- and is
                 * zeroed in the others. Cymbals and air survive in a karaoke
                 * track, which upstream's "zeros" throws away. */
                for (i = 0; i < n; i++) {
                    float *hi = mask[i] + ss_idx(c, t, SS_F);
                    const float v = i == n - 1 ? 1.0f : 0.0f;

                    for (f = 0; f < SS_BINS - SS_F; f++)
                        hi[f] = v;
                }
                break;
            }
        }
}

/* ---- Segment driver -----------------------------------------------------
 *
 * Design section 4.2 steps 1 to 3 and 7 to 9, streamed.
 *
 * Analysis slides a 4096-sample window over the input in 1024-sample hops.
 * The window starts all-zero and the first frame is analysed before any real
 * sample arrives: that zero window IS step 1's 4096-sample lead-in, and
 * crop_remaining discards the matching 4096 output samples in step 9. Framing
 * a signal that has 4096 zeros glued to its front is exactly what a zeroed
 * sliding window does, so the lead-in costs one frame, not a copy of the
 * whole waveform.
 *
 * Frame spectra go into a ring of exactly SS_T slots. Segment k covers global
 * frames [k*hop, k*hop + SS_T) -- the whole ring -- and retires frames
 * [k*hop, (k+1)*hop), whose slots are precisely the ones segment k+1's new
 * frames overwrite. So the ring is never short and never has to grow.
 *
 * A retired frame's mask multiplies the ORIGINAL COMPLEX mixture spectrum
 * stored in that ring slot (step 7): the mixture's phase is reused unchanged
 * and never reconstructed.
 */

static int ss_emit_frames(AVFilterContext *ctx, int nb_frames);

/* Runs one segment: gathers its magnitude out of the ring, infers, builds the
 * ratio masks, blends them with any overlapping neighbour, and retires the
 * frames the next segment will overwrite. */
static int ss_process_segment(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    const int64_t base = s->next_segment * s->hop_frames;
    const float *est_p[SS_NB_INSTRUMENTS];
    int c, j, f, i, ret;

    /* ss_build_masks() writes s->mask indexed by SEGMENT-RELATIVE frame j,
     * while ss_emit_frames() reads it back indexed by RING SLOT. Those are the
     * same index only because at overlap=0 hop_frames == SS_T, which makes
     * base a multiple of SS_T and so slot == (base + j) % SS_T == j. With
     * overlap the two disagree, and the mask_acc ring below -- which IS
     * slot-indexed -- is what ss_emit_frames reads instead. Assert the
     * condition rather than trust it: if a future change ever lets hop_frames
     * differ from SS_T at overlap 0, the mask would be silently read from the
     * wrong frame, which sounds like smearing and raises no error. */
    av_assert0(s->overlap_frames > 0 || base % SS_T == 0);

    for (c = 0; c < SS_CHANNELS; c++)
        for (j = 0; j < SS_T; j++) {
            const AVComplexFloat *row = s->spec_ring +
                ((size_t) ((base + j) % SS_T) * SS_CHANNELS + c) * SS_SPEC_STRIDE;
            float *dst = s->mag + ss_nidx(c, j, 0);

            /* Raw magnitude, no normalisation -- design 4.2 step 3. */
            for (f = 0; f < SS_F; f++)
                dst[f] = hypotf(row[f].re, row[f].im);
        }

    for (i = 0; i < SS_NB_INSTRUMENTS; i++)
        est_p[i] = s->est[i];

    ret = ss_infer(ctx, s->mag, s->est);
    if (ret < 0)
        return ret;

    ss_build_masks(s, est_p, s->mask);

    if (s->overlap_frames > 0) {
        int64_t t;

        /* Slots this segment is the first to touch start from zero; the rest
         * already carry the previous segment's weighted contribution. */
        for (t = FFMAX(base, s->acc_filled); t < base + SS_T; t++) {
            const int slot = (int) (t % SS_T);

            s->w_acc[slot] = 0.0f;
            for (i = 0; i < s->nb_instruments; i++)
                for (c = 0; c < SS_CHANNELS; c++)
                    memset(s->mask_acc[i] + ss_idx(c, slot, 0), 0,
                           SS_BINS * sizeof(float));
        }
        s->acc_filled = base + SS_T;

        for (j = 0; j < SS_T; j++) {
            const int slot = (int) ((base + j) % SS_T);
            const float v = s->seg_win[j];

            s->w_acc[slot] += v;
            for (i = 0; i < s->nb_instruments; i++)
                for (c = 0; c < SS_CHANNELS; c++) {
                    const float *src = s->mask[i] + ss_idx(c, j, 0);
                    float *dst = s->mask_acc[i] + ss_idx(c, slot, 0);

                    for (f = 0; f < SS_BINS; f++)
                        dst[f] += v * src[f];
                }
        }
    }

    s->next_segment++;

    return ss_emit_frames(ctx, (int) (base + s->hop_frames - s->frames_done));
}

/* Retires nb_frames frames: masks each one's mixture spectrum, resynthesises
 * it into every output pad's overlap-add accumulator, and pushes the samples
 * that no future frame can still touch.
 *
 * A sample s takes contributions from every frame t with
 * t*STEP <= s < t*STEP + LENGTH, so once frame t has been added, everything
 * below (t+1)*STEP is final: exactly one hop per frame, which is why the
 * accumulator only needs to be one frame long.
 *
 * ONE AVFrame IS EMITTED PER RETIRED STFT FRAME -- SS_FRAME_STEP samples --
 * not one concatenated buffer per segment. Concatenating produced 520,192-
 * and 524,288-sample frames, and FLAC rejects those outright ("invalid block
 * size"; its ceiling is 65,535). Capping the size has to happen here because
 * libavfilter offers a filter no way to declare or discover a downstream
 * preference: FLAC, ALAC, WavPack, TTA and Monkey's Audio all declare
 * AV_CODEC_CAP_VARIABLE_FRAME_SIZE, so ffmpeg does not auto-reframe them;
 * av_buffersink_set_frame_size is called by the CLI only for encoders with a
 * fixed frame size; and min_samples/max_samples govern a filter's INPUT, not
 * its output. The loop below always produced exactly one hop per iteration --
 * that granularity was simply being concatenated away.
 *
 * The chunk boundaries land where they should. The lead-in crop is
 * SS_FRAME_LENGTH == 4 * SS_FRAME_STEP, a whole number of hops, so it skips
 * whole frames and never splits one. The only short frame is the last, where
 * s->flushing truncates the zero-padded tail back to the true sample count --
 * exactly where a short frame belongs.
 */
static int ss_emit_frames(AVFilterContext *ctx, int nb_frames)
{
    StemSplitContext *s = ctx->priv;
    AVFrame *out[SS_NB_INSTRUMENTS] = { NULL };
    int j, c, fi, b, ret = 0;

    if (nb_frames <= 0)
        return 0;

    /* out[] and s->ola[] are sized by instrument count; one pad per instrument
     * is the maximum ss_append_outputs can produce. */
    av_assert0(ctx->nb_outputs <= SS_NB_INSTRUMENTS);

    for (fi = 0; fi < nb_frames; fi++) {
        const int slot = (int) (s->frames_done % SS_T);
        const float inv_w = s->overlap_frames > 0 ? 1.0f / s->w_acc[slot] : 1.0f;
        int src_off = 0, n = SS_FRAME_STEP;

        for (j = 0; j < ctx->nb_outputs; j++) {
            const int inst = s->out_inst[j];
            /* At overlap=0 this reads s->mask by RING SLOT even though
             * ss_build_masks wrote it by SEGMENT-RELATIVE frame. The two
             * indices coincide only because hop_frames == SS_T there, which
             * makes every segment start on a multiple of SS_T -- the invariant
             * ss_process_segment asserts. With overlap they differ, and the
             * mask_acc ring, which is slot-indexed by construction, carries
             * the blended masks instead. */
            const float *m_base = s->overlap_frames > 0 ? s->mask_acc[inst]
                                                         : s->mask[inst];

            for (c = 0; c < SS_CHANNELS; c++) {
                const AVComplexFloat *spec = s->spec_ring +
                    ((size_t) slot * SS_CHANNELS + c) * SS_SPEC_STRIDE;
                const float *m = m_base + ss_idx(c, slot, 0);

                for (b = 0; b < SS_BINS; b++) {
                    const float gain = m[b] * inv_w;

                    s->work_spec[b].re = spec[b].re * gain;
                    s->work_spec[b].im = spec[b].im * gain;
                }
                ss_synthesize(s, s->work_spec, s->ola[j][c]);
            }
        }

        /* Step 9's crop: drop the lead-in the analysis window prepended. */
        if (s->crop_remaining > 0) {
            const int skip = (int) FFMIN(s->crop_remaining, (int64_t) n);

            s->crop_remaining -= skip;
            src_off += skip;
            n       -= skip;
        }

        /* At EOF the last segment was zero-padded to a full SS_T frames; the
         * pad must not reach the output, so the true sample count caps it. */
        if (s->flushing)
            n = (int) FFMIN((int64_t) n, FFMAX(s->total_in - s->emitted, 0));

        if (n > 0) {
            const int64_t pts = s->next_pts;
            const int64_t dur = av_rescale_q(n, (AVRational) { 1, SS_SAMPLE_RATE },
                                             ctx->outputs[0]->time_base);

            for (j = 0; j < ctx->nb_outputs; j++) {
                if (ff_outlink_get_status(ctx->outputs[j]))
                    continue;   /* downstream is gone; keep processing, drop output */
                out[j] = ff_get_audio_buffer(ctx->outputs[j], n);
                if (!out[j]) {
                    ret = AVERROR(ENOMEM);
                    goto fail;
                }
                for (c = 0; c < SS_CHANNELS; c++)
                    memcpy(out[j]->extended_data[c], s->ola[j][c] + src_off,
                           n * sizeof(float));
                out[j]->pts      = pts;
                out[j]->duration = dur;
            }

            if (s->next_pts != AV_NOPTS_VALUE)
                s->next_pts += dur;
            s->emitted += n;
        }

        /* Slide each accumulator by one hop and zero the tail that exposes,
         * ready for the next frame's overlap-add. */
        for (j = 0; j < ctx->nb_outputs; j++)
            for (c = 0; c < SS_CHANNELS; c++) {
                float *o = s->ola[j][c];

                memmove(o, o + SS_FRAME_STEP,
                        (SS_FRAME_LENGTH - SS_FRAME_STEP) * sizeof(float));
                memset(o + SS_FRAME_LENGTH - SS_FRAME_STEP, 0,
                       SS_FRAME_STEP * sizeof(float));
            }

        s->frames_done++;

        for (j = 0; j < ctx->nb_outputs; j++) {
            AVFrame *frame = out[j];

            if (!frame)
                continue;
            out[j] = NULL;
            ret = ff_filter_frame(ctx->outputs[j], frame);
            if (ret < 0)
                goto fail;
        }
    }

    return 0;

fail:
    for (j = 0; j < SS_NB_INSTRUMENTS; j++)
        av_frame_free(&out[j]);
    return ret;
}

/* Analyses the current window into the ring and runs a segment if that
 * completed one. */
static int ss_push_frame(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    const int slot = (int) (s->frames_in % SS_T);
    int c;

    for (c = 0; c < SS_CHANNELS; c++)
        ss_analyze(s, s->an_window[c],
                   s->spec_ring + ((size_t) slot * SS_CHANNELS + c) * SS_SPEC_STRIDE);
    s->frames_in++;

    /* Run before the next frame can overwrite a slot this segment still
     * needs: segment k's last frame arriving is exactly the moment its first
     * frame's slot becomes the next segment's target. */
    if (s->frames_in >= s->next_segment * s->hop_frames + SS_T)
        return ss_process_segment(ctx);

    return 0;
}

/* Slides the analysis window by one hop, folding in whatever is staged. */
static int ss_push_hop(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    int c;

    for (c = 0; c < SS_CHANNELS; c++) {
        memmove(s->an_window[c], s->an_window[c] + SS_FRAME_STEP,
                (SS_FRAME_LENGTH - SS_FRAME_STEP) * sizeof(float));
        memcpy(s->an_window[c] + SS_FRAME_LENGTH - SS_FRAME_STEP,
               s->an_stage[c], SS_FRAME_STEP * sizeof(float));
    }
    s->an_stage_fill = 0;

    return ss_push_frame(ctx);
}

/* Emits the lead-in frame: the all-zero analysis window, which is frame 0 of
 * the zero-prepended signal. Skipping it would shift every segment one frame
 * against the reference for no gain -- it costs one FFT of silence. */
static int ss_driver_start(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;

    if (s->started)
        return 0;
    s->started = 1;

    return ss_push_frame(ctx);
}

static int ss_push_samples(AVFilterContext *ctx, const AVFrame *frame)
{
    StemSplitContext *s = ctx->priv;
    int offset = 0, ret;

    ret = ss_driver_start(ctx);
    if (ret < 0)
        return ret;

    s->total_in += frame->nb_samples;

    while (offset < frame->nb_samples) {
        const int n = FFMIN(frame->nb_samples - offset,
                            SS_FRAME_STEP - s->an_stage_fill);
        int c;

        for (c = 0; c < SS_CHANNELS; c++)
            memcpy(s->an_stage[c] + s->an_stage_fill,
                   (const float *) frame->extended_data[c] + offset,
                   n * sizeof(float));
        s->an_stage_fill += n;
        offset += n;

        if (s->an_stage_fill == SS_FRAME_STEP) {
            ret = ss_push_hop(ctx);
            if (ret < 0)
                return ret;
        }
    }

    return 0;
}

/* EOF drain. Keeps feeding zeros -- pad_end for the tail frames, and the time
 * axis zero-padded to a whole segment for the network -- until every real
 * input sample has come back out. ss_emit_frames caps the output at
 * total_in while s->flushing is set, so the padding never reaches the file. */
static int ss_flush(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    int ret;

    ret = ss_driver_start(ctx);
    if (ret < 0)
        return ret;

    s->flushing = 1;

    while (s->emitted < s->total_in) {
        int c;

        for (c = 0; c < SS_CHANNELS; c++)
            memset(s->an_stage[c] + s->an_stage_fill, 0,
                   (SS_FRAME_STEP - s->an_stage_fill) * sizeof(float));

        ret = ss_push_hop(ctx);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static void ss_driver_free(StemSplitContext *s)
{
    int c, i;

    for (c = 0; c < SS_CHANNELS; c++) {
        av_freep(&s->an_window[c]);
        av_freep(&s->an_stage[c]);
    }
    for (i = 0; i < SS_NB_INSTRUMENTS; i++) {
        av_freep(&s->est[i]);
        av_freep(&s->mask[i]);
        av_freep(&s->mask_acc[i]);
        for (c = 0; c < SS_CHANNELS; c++)
            av_freep(&s->ola[i][c]);
    }
    av_freep(&s->spec_ring);
    av_freep(&s->mag);
    av_freep(&s->w_acc);
    av_freep(&s->seg_win);
    av_freep(&s->work_spec);
}

static int ss_driver_init(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    const size_t net_floats = (size_t) SS_CHANNELS * SS_T * SS_F;
    const size_t mask_floats = (size_t) SS_CHANNELS * SS_T * SS_BINS;
    int c, i, j;

    /* Ruling 11: `overlap` is a duration, because a user should not have to
     * know that a segment is 512 STFT frames of 1024 samples to ask for a
     * second and a half of crossfade. The conversion lives here, where the
     * arithmetic is. */
    const int64_t want_frames = av_rescale(s->overlap, SS_SAMPLE_RATE,
                                           AV_TIME_BASE) / SS_FRAME_STEP;

    s->overlap_frames = (int) av_clip64(want_frames, 0, SS_T - 1);
    s->hop_frames = SS_T - s->overlap_frames;

    /* The option accepts up to 60 s, but a segment is only SS_T frames long,
     * so anything past SS_T-1 frames (~11.86 s) is silently the same request.
     * Say so rather than let a user believe overlap=30 did something. */
    if (want_frames > s->overlap_frames)
        av_log(ctx, AV_LOG_WARNING,
               "stemsplit: overlap of %.3f s exceeds the %.3f s maximum "
               "(a segment is %d frames); clamping to %.3f s.\n",
               s->overlap / (double) AV_TIME_BASE,
               (double) (SS_T - 1) * SS_FRAME_STEP / SS_SAMPLE_RATE, SS_T,
               (double) s->overlap_frames * SS_FRAME_STEP / SS_SAMPLE_RATE);

    for (c = 0; c < SS_CHANNELS; c++) {
        s->an_window[c] = av_calloc(SS_FRAME_LENGTH, sizeof(float));
        s->an_stage[c]  = av_calloc(SS_FRAME_STEP, sizeof(float));
        if (!s->an_window[c] || !s->an_stage[c])
            return AVERROR(ENOMEM);
    }

    s->spec_ring = av_calloc((size_t) SS_T * SS_CHANNELS * SS_SPEC_STRIDE,
                             sizeof(*s->spec_ring));
    s->mag       = av_calloc(net_floats, sizeof(float));
    s->work_spec = av_calloc(SS_BINS, sizeof(*s->work_spec));
    s->seg_win   = av_calloc(SS_T, sizeof(float));
    if (!s->spec_ring || !s->mag || !s->work_spec || !s->seg_win)
        return AVERROR(ENOMEM);

    for (i = 0; i < s->nb_instruments; i++) {
        s->est[i]  = av_calloc(net_floats, sizeof(float));
        s->mask[i] = av_calloc(mask_floats, sizeof(float));
        if (!s->est[i] || !s->mask[i])
            return AVERROR(ENOMEM);
    }

    for (j = 0; j < s->nb_outputs; j++)
        for (c = 0; c < SS_CHANNELS; c++) {
            s->ola[j][c] = av_calloc(SS_FRAME_LENGTH, sizeof(float));
            if (!s->ola[j][c])
                return AVERROR(ENOMEM);
        }

    /* Per-frame taper inside a segment. Zero overlap means a flat window and
     * no accumulator at all, which is both the default and the reference
     * behaviour. With overlap, the taper rises over the first overlap_frames
     * and falls over the last, and the sum of tapers normalises it -- so the
     * very first and very last segments, which have no neighbour to blend
     * with, come back out at unit gain automatically.
     *
     * The weights are EQUAL-GAIN (they sum to 1 after normalisation), not
     * equal-power. Both segments carry the same mixture phase -- they differ
     * only in a real, non-negative mask -- so their outputs add coherently and
     * sqrt weights would put a +3 dB hump in the middle of every crossfade.
     * Equal power is the right rule for uncorrelated material; this is the
     * opposite case. */
    for (j = 0; j < SS_T; j++) {
        float v = 1.0f;

        if (s->overlap_frames > 0) {
            const float d = s->overlap_frames + 1.0f;

            if (j < s->overlap_frames)
                v = (j + 1) / d;
            if (j >= SS_T - s->overlap_frames)
                v = FFMIN(v, (SS_T - j) / d);
        }
        s->seg_win[j] = v;
    }

    if (s->overlap_frames > 0) {
        s->w_acc = av_calloc(SS_T, sizeof(float));
        if (!s->w_acc)
            return AVERROR(ENOMEM);
        for (i = 0; i < s->nb_instruments; i++) {
            s->mask_acc[i] = av_calloc(mask_floats, sizeof(float));
            if (!s->mask_acc[i])
                return AVERROR(ENOMEM);
        }
    }

    s->crop_remaining = SS_FRAME_LENGTH;

    /* The clamp is [0, SS_T-1] per Ruling 11, so a user can ask for an overlap
     * of nearly a whole segment -- and pay for it, because the cost is exactly
     * SS_T / hop_frames forward passes per segment of audio. Say so rather
     * than let it look like a hang. */
    if (s->hop_frames * 4 < SS_T)
        av_log(ctx, AV_LOG_WARNING,
               "stemsplit: overlap advances only %d of %d frames per segment, "
               "so the network runs %.0fx as often as at overlap=0. Expect a "
               "proportional slowdown.\n",
               s->hop_frames, SS_T, SS_T / (double) s->hop_frames);

    av_log(ctx, AV_LOG_VERBOSE,
           "stemsplit: segment %d frames (%.3f s), advance %d frames, "
           "overlap %d frames, %d output pad(s).\n",
           SS_T, (double) SS_T * SS_FRAME_STEP / SS_SAMPLE_RATE,
           s->hop_frames, s->overlap_frames, s->nb_outputs);

    return 0;
}

/* ---- Filter plumbing ----------------------------------------------------- */

/* Index of `name` in the model's instrument list, or a negative error. The
 * caller has already been validated by ss_model_load, so a miss here is an
 * internal inconsistency rather than a user error. */
static int ss_find_instrument(const StemSplitContext *s, const char *name)
{
    int i;

    for (i = 0; i < s->nb_instruments; i++)
        if (!strcmp(s->instrument_names[i], name))
            return i;

    return AVERROR(EINVAL);
}

/* stem=all exposes one output pad per instrument, named after the model's own
 * instrument list, which is the af_channelsplit pattern and the reason the
 * filter carries AVFILTER_FLAG_DYNAMIC_OUTPUTS. stem=<name> exposes a single
 * pad called "default": the ffmpeg CLI treats an unconnected filter output as
 * an error, so the single-pad form is what makes the common karaoke case work
 * with a plain -af instead of -filter_complex. */
static int ss_append_outputs(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    int i, ret;

    if (s->stem == SS_STEM_ALL) {
        for (i = 0; i < s->nb_instruments; i++) {
            AVFilterPad pad = { .type = AVMEDIA_TYPE_AUDIO };

            pad.name = av_strdup(s->instrument_names[i]);
            if (!pad.name)
                return AVERROR(ENOMEM);
            /* _free_name so the strdup is released even if the append fails. */
            ret = ff_append_outpad_free_name(ctx, &pad);
            if (ret < 0)
                return ret;
            s->out_inst[i] = i;
        }
        s->nb_outputs = s->nb_instruments;
    } else {
        AVFilterPad pad = { .name = "default", .type = AVMEDIA_TYPE_AUDIO };
        const char *want = s->stem == SS_STEM_VOCALS ? "vocals" : "accompaniment";

        ret = ss_find_instrument(s, want);
        if (ret < 0)
            return ret;
        s->out_inst[0] = ret;
        s->nb_outputs = 1;

        ret = ff_append_outpad(ctx, &pad);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    int ret;

    s->next_pts = AV_NOPTS_VALUE;

    ret = ss_model_load(ctx);
    if (ret < 0)
        return ret;

    ret = ss_append_outputs(ctx);
    if (ret < 0)
        return ret;

    ret = ss_dsp_init(ctx);
    if (ret < 0)
        return ret;

    /* Internal parity path (design section 9.2): when `debug_input` names a
     * raw spectrogram, run the networks on it once here and dump the taps the
     * fixtures record, bypassing the STFT and the segment driver entirely.
     * The driver's buffers are not allocated on this path and audio is passed
     * through untouched -- this hook exists to measure the network, not to
     * separate anything. */
    if (s->debug_input_path && *s->debug_input_path)
        return ss_run_debug_input(ctx);

    return ss_driver_init(ctx);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;

    /* Before ss_model_free: every graph tensor's weights live in gguf_ctx. */
    ss_graph_free(s);
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
    ss_driver_free(s);
}

static int ss_filter_frame(AVFilterContext *ctx, AVFrame *frame)
{
    StemSplitContext *s = ctx->priv;
    int ret;

    /* s->next_pts is advanced in the OUTPUT link's time base (ss_emit_frames),
     * so seed it in that base too. Both are 1/44100 today because
     * query_formats pins the sample rate, but that is a coincidence of the
     * current format negotiation rather than something this line should
     * assume. */
    if (s->next_pts == AV_NOPTS_VALUE)
        s->next_pts = frame->pts != AV_NOPTS_VALUE
                    ? av_rescale_q(frame->pts, ctx->inputs[0]->time_base,
                                   ctx->outputs[0]->time_base)
                    : 0;

    /* Parity hook: the network has already been measured against the injected
     * spectrogram in init(); the audio itself is of no interest. */
    if (s->debug_input_path && *s->debug_input_path)
        return ff_filter_frame(ctx->outputs[0], frame);

    ret = ss_push_samples(ctx, frame);
    av_frame_free(&frame);

    return ret;
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    StemSplitContext *s = ctx->priv;
    AVFrame *in = NULL;
    int64_t pts;
    int i, status, ret, nb_eofs = 0;

    /* Every output gone means nobody is listening: tell the input to stop
     * rather than keep paying for 11.9 s segments nothing will read. */
    for (i = 0; i < ctx->nb_outputs; i++)
        nb_eofs += ff_outlink_get_status(ctx->outputs[i]) == AVERROR_EOF;
    if (nb_eofs == ctx->nb_outputs) {
        ff_inlink_set_status(inlink, AVERROR_EOF);
        return 0;
    }

    if (!s->eof) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0)
            return ss_filter_frame(ctx, in);
    }

    if (!s->eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        s->eof    = 1;
        s->status = status;
        if (status == AVERROR_EOF && !(s->debug_input_path && *s->debug_input_path)) {
            ret = ss_flush(ctx);
            if (ret < 0)
                return ret;
        }
    }

    if (s->eof) {
        for (i = 0; i < ctx->nb_outputs; i++) {
            if (ff_outlink_get_status(ctx->outputs[i]))
                continue;
            ff_outlink_set_status(ctx->outputs[i], s->status, s->next_pts);
        }
        return 0;
    }

    FF_FILTER_FORWARD_WANTED_ANY(ctx, inlink);

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
        { "passthrough", "route the input high band into the last stem", 0, AV_OPT_TYPE_CONST, { .i64 = SS_HB_PASSTHROUGH }, 0, 0, FLAGS, .unit = "highband" },
        { "zeros", "silence the high band in every stem (upstream Spleeter)", 0, AV_OPT_TYPE_CONST, { .i64 = SS_HB_ZEROS }, 0, 0, FLAGS, .unit = "highband" },
        { "average", "tile each frame's mean mask across the high band", 0, AV_OPT_TYPE_CONST, { .i64 = SS_HB_AVERAGE }, 0, 0, FLAGS, .unit = "highband" },
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
    /* Output pads are appended in init(); there is no FILTER_OUTPUTS. */
    .p.flags       = AVFILTER_FLAG_DYNAMIC_OUTPUTS,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .priv_size     = sizeof(StemSplitContext),
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
