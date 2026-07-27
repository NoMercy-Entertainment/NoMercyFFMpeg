/*
 * Copyright (c) 2025
 *
 * Key and chord detection filter for FFmpeg.
 * Detects the musical key and the chord progression of audio.
 *
 * Method:
 *   - Sliding Hann window, real FFT (libavutil/tx).
 *   - Spectral peak picking with parabolic interpolation.
 *   - Tuning estimation from a cents-deviation histogram of the peaks.
 *   - Harmonic pitch class profile (HPCP-style): every peak contributes to
 *     the pitch classes of its candidate fundamentals f/1..f/4 with weights
 *     0.6^(h-1) (Gomez 2006, Oudre 2009).
 *   - Chords: 24 binary major/minor triad templates, cosine similarity,
 *     low-pass filtered scores (Oudre 2009) + minimum-duration segments.
 *   - Key: Pearson correlation of the accumulated chroma against rotated
 *     key profiles (Krumhansl-Kessler, Temperley-Kostka-Payne, Sha'ath,
 *     EDMA), the classic Krumhansl-Schmuckler approach. Each window enters
 *     the accumulated chroma weighted by how tonal it is, so percussion and
 *     other unpitched material cannot bias the result.
 *   - Tonal centre: profile correlation separates a key from its relative
 *     major/minor by hundredths, because the two share a pitch collection,
 *     so a near tie there is close to a coin toss. When that happens the
 *     committed chord progression breaks the tie: which chords occurred,
 *     for how long, and whether any of them fall outside a candidate key's
 *     collection. Confident correlations are left alone.
 *
 * Output (per frame metadata, and log events on change):
 *   lavfi.keydetect.key            e.g. "C", "F#m"
 *   lavfi.keydetect.key_confidence Pearson r of the best key
 *   lavfi.keydetect.key_alt        runner-up key
 *   lavfi.keydetect.key_alt_confidence  Pearson r of the runner-up
 *   lavfi.keydetect.key_margin     r of the best minus r of the runner-up.
 *                                  A better reliability signal than r on
 *                                  its own: relative-key mistakes are near
 *                                  ties, not low scoring answers.
 *   lavfi.keydetect.key_candidates "Cm:0.885,D#:0.879,Gm:0.812"
 *   lavfi.keydetect.key_reranked   1 if the chord progression overrode the
 *                                  chroma correlation
 *   lavfi.keydetect.chord          current stable chord, "N" = no chord
 *   lavfi.keydetect.chords         chord progression so far, e.g. "C-Am-F-G"
 *
 * This file is intended for integration into FFmpeg's libavfilter.
 * Compile within an FFmpeg source tree (verified against FFmpeg 8.1.2).
 */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/mathematics.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/tx.h"

#include "audio.h"
#include "avfilter.h"
#include "filters.h"

#define NUM_CHORD_TEMPLATES 24
#define NUM_KEYS            24    /* 12 major then 12 minor */
#define NUM_CANDIDATES      3     /* key candidates reported */
#define PROGRESSION_MAX     64
#define TUNING_BINS         100   /* 1-cent resolution over [-50, +50) */
#define NUM_HARMONICS       4     /* harmonic folding depth              */
#define HARMONIC_DECAY      0.6f  /* weight of h-th harmonic = decay^(h-1) */
#define PEAK_FLOOR_REL      0.01f /* keep peaks >= 1% of frame max (-40 dB) */
#define FREQ_MIN            55.0  /* A1: lowest peak frequency used */
#define FREQ_MAX            2000.0/* above this it is almost all harmonics */
#define FUND_MIN            25.0  /* lowest folded fundamental accepted */
#define SMOOTH_MAX_HOPS     64
#define PROG_STR_SIZE       512

enum KeyProfile {
    PROFILE_KRUMHANSL,
    PROFILE_TEMPERLEY,
    PROFILE_SHAATH,
    PROFILE_EDMA,
    NB_PROFILES
};

static const char *const pitch_names[12] = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};

/* Key profiles, index 0 = tonic. Values verified against music21,
 * Essentia (key.cpp) and libKeyFinder (constants.cpp). */
static const double key_profiles[NB_PROFILES][2][12] = {
    [PROFILE_KRUMHANSL] = {
        { 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88 },
        { 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17 },
    },
    [PROFILE_TEMPERLEY] = { /* Temperley-Kostka-Payne 2005 */
        { 0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400 },
        { 0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330 },
    },
    [PROFILE_SHAATH] = { /* Sha'ath 2011, libKeyFinder precision values */
        { 7.23900502618145, 3.50351166725159, 3.58445177536649, 2.84511816478676,
          5.81898892118550, 4.55865057415321, 2.44778850545507, 6.99473192146830,
          3.39106613673505, 4.55614256655143, 4.07392666663524, 4.45932757378887 },
        { 7.00255045060284, 3.14360279015997, 4.35904319714963, 5.40418120718934,
          3.67234420879306, 4.08971184917798, 3.90791435991554, 6.19960288562316,
          3.63424625625277, 2.87241191079876, 5.35467999794543, 3.83242038595048 },
    },
    [PROFILE_EDMA] = { /* Faraldo 2016, electronic/pop */
        { 1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39 },
        { 1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32 },
    },
};

/* Harmonic function weights used to score a committed chord against a
 * candidate key: [key mode][chord quality][interval from key root].
 *
 * Positive entries are the diatonic degrees, weighted by how strongly each
 * one implies its key (the tonic triad most, then the dominant, and so on).
 * OUT marks chords outside the key's collection, which is the decisive
 * signal: a B major chord rules out F# minor, an E major chord rules out
 * C major. BOR is the mild penalty for borrowings that are idiomatic in pop
 * without indicating a key change (Mixolydian bVII in major, Picardy third
 * in minor) - not evidence for the key, but not disqualifying either. */
#define FW_OUT (-0.60f)
#define FW_BOR (-0.10f)
static const float func_weight[2][2][12] = {
    [0] = { /* major key */
        /* major chord */ { 1.00f, FW_OUT, FW_OUT, FW_BOR, FW_OUT, 0.70f,
                            FW_OUT, 0.85f, FW_BOR, FW_OUT, FW_BOR, FW_OUT },
        /* minor chord */ { FW_OUT, FW_OUT, 0.50f, FW_OUT, 0.40f, FW_BOR,
                            FW_OUT, FW_OUT, FW_OUT, 0.55f, FW_OUT, FW_OUT },
    },
    [1] = { /* minor key */
        /* major chord */ { FW_BOR, FW_OUT, FW_OUT, 0.60f, FW_OUT, 0.35f,
                            FW_OUT, 0.85f, 0.60f, FW_OUT, 0.65f, FW_OUT },
        /* minor chord */ { 1.00f, FW_OUT, 0.35f, FW_OUT, FW_OUT, 0.65f,
                            FW_OUT, 0.60f, FW_OUT, FW_OUT, FW_BOR, FW_OUT },
    },
};

/* Loop based pop tends to start and end its cycle on the tonic. Weak cues
 * on their own, which is why they only ever break a correlation tie. */
#define TONIC_FIRST_BONUS 0.15f
#define TONIC_LAST_BONUS  0.20f
/* Below this much committed chord evidence the progression is too thin to
 * override the chroma. Measured as an absolute amount of stable chord time,
 * not as a fraction of the input: a song with long unpitched stretches can
 * still state its harmony perfectly clearly in the parts that have chords,
 * and that evidence is worth just as much. Chromatic junk is rejected
 * separately, by requiring the winning progression to actually be coherent
 * with some key. */
#define MIN_DISTINCT_CHORDS 3
#define MIN_STABLE_MS       5000

/* Binary triad templates: 12 major then 12 minor, root order C..B.
 * Major: root, +4, +7. Minor: root, +3, +7. */
static const uint8_t chord_templates[NUM_CHORD_TEMPLATES][12] = {
    { 1,0,0,0,1,0,0,1,0,0,0,0 }, /* C   */
    { 0,1,0,0,0,1,0,0,1,0,0,0 }, /* C#  */
    { 0,0,1,0,0,0,1,0,0,1,0,0 }, /* D   */
    { 0,0,0,1,0,0,0,1,0,0,1,0 }, /* D#  */
    { 0,0,0,0,1,0,0,0,1,0,0,1 }, /* E   */
    { 1,0,0,0,0,1,0,0,0,1,0,0 }, /* F   */
    { 0,1,0,0,0,0,1,0,0,0,1,0 }, /* F#  */
    { 0,0,1,0,0,0,0,1,0,0,0,1 }, /* G   */
    { 1,0,0,1,0,0,0,0,1,0,0,0 }, /* G#  */
    { 0,1,0,0,1,0,0,0,0,1,0,0 }, /* A   */
    { 0,0,1,0,0,1,0,0,0,0,1,0 }, /* A#  */
    { 0,0,0,1,0,0,1,0,0,0,0,1 }, /* B   */
    { 1,0,0,1,0,0,0,1,0,0,0,0 }, /* Cm  */
    { 0,1,0,0,1,0,0,0,1,0,0,0 }, /* C#m */
    { 0,0,1,0,0,1,0,0,0,1,0,0 }, /* Dm  */
    { 0,0,0,1,0,0,1,0,0,0,1,0 }, /* D#m */
    { 0,0,0,0,1,0,0,1,0,0,0,1 }, /* Em  */
    { 1,0,0,0,0,1,0,0,1,0,0,0 }, /* Fm  */
    { 0,1,0,0,0,0,1,0,0,1,0,0 }, /* F#m */
    { 0,0,1,0,0,0,0,1,0,0,1,0 }, /* Gm  */
    { 0,0,0,1,0,0,0,0,1,0,0,1 }, /* G#m */
    { 1,0,0,0,1,0,0,0,0,1,0,0 }, /* Am  */
    { 0,1,0,0,0,1,0,0,0,0,1,0 }, /* A#m */
    { 0,0,1,0,0,0,1,0,0,0,0,1 }, /* Bm  */
};

typedef struct KeyDetectContext {
    const AVClass *class;

    /* Options */
    int window_ms;
    int hop_ms;
    int profile;
    double tuning_hz;
    int detect_tuning;
    float chord_threshold;
    double silence_db;
    int smooth_ms;
    int min_chord_ms;
    float compression;
    float salience;
    int rerank;
    double rerank_margin;
    double rerank_gap;

    /* Configuration derived */
    int sample_rate;
    int win_len;
    int fft_len;
    int hop_len;
    int smooth_hops;
    int min_dur_hops;
    double silence_rms2;
    /* Key profile statistics for Pearson correlation */
    double prof_mean[2], prof_sd[2];

    /* FFT */
    AVTXContext *tx_ctx;
    av_tx_fn tx_fn;
    float *fft_in;            /* [fft_len] zero padded            */
    AVComplexFloat *fft_out;  /* [fft_len / 2 + 1]                */
    float *win_func;          /* Hann table [win_len]             */
    float *mag;               /* magnitude spectrum [fft_len/2+1] */

    /* Input ring buffer of downmixed mono */
    float *ring;              /* [win_len] */
    int64_t total_in;         /* total mono samples received       */
    int64_t next_win_end;     /* absolute sample pos ending window */

    /* Tuning */
    double tune_hist[TUNING_BINS];
    double tune_offset;       /* estimated deviation in semitones  */

    /* Key */
    double key_chroma[12];    /* accumulated chroma of whole input */
    int key_index;            /* 0..23 (root + 12*minor), -1 unset */
    double key_confidence;
    double key_r[NUM_KEYS];   /* correlation of every candidate     */
    int key_rank[NUM_CANDIDATES]; /* best candidates, -1 unset      */
    double key_margin;        /* r of the best minus the runner up  */

    /* Chords */
    float *score_lp;          /* [smooth_hops][24] score history   */
    int score_pos;
    int score_fill;
    int seg_label;            /* current smoothed label, -1 = N    */
    int seg_len;              /* hops the label persisted          */
    int stable_label;         /* last label that met min duration  */
    int last_committed;       /* last chord added to progression   */
    int progression[PROGRESSION_MAX];
    int progression_len;

    /* Chord evidence for the key decision. Accumulated over the whole
     * input, unlike progression[] which only keeps the newest entries. */
    double chord_dur[NUM_CHORD_TEMPLATES]; /* hops each chord was held */
    int chord_first;          /* first stable chord, -1 unset      */
    int chord_last;           /* most recent stable chord          */
    int64_t stable_hops;      /* hops spent on a stable chord      */
    int64_t total_hops;       /* hops analysed                     */
    int reranked;             /* key decided by chord evidence     */
} KeyDetectContext;

#define OFFSET(x) offsetof(KeyDetectContext, x)
#define FLAGS (AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

static const AVOption keydetect_options[] = {
    { "window_ms", "analysis window length in ms", OFFSET(window_ms), AV_OPT_TYPE_INT, { .i64 = 190 }, 50, 500, FLAGS },
    { "hop_ms", "analysis hop length in ms", OFFSET(hop_ms), AV_OPT_TYPE_INT, { .i64 = 50 }, 10, 500, FLAGS },
    { "profile", "key profile", OFFSET(profile), AV_OPT_TYPE_INT, { .i64 = PROFILE_SHAATH }, 0, NB_PROFILES - 1, FLAGS, .unit = "profile" },
        { "krumhansl", "Krumhansl-Kessler (1982)", 0, AV_OPT_TYPE_CONST, { .i64 = PROFILE_KRUMHANSL }, 0, 0, FLAGS, .unit = "profile" },
        { "temperley", "Temperley-Kostka-Payne (2005)", 0, AV_OPT_TYPE_CONST, { .i64 = PROFILE_TEMPERLEY }, 0, 0, FLAGS, .unit = "profile" },
        { "shaath", "Sha'ath (2011), general audio", 0, AV_OPT_TYPE_CONST, { .i64 = PROFILE_SHAATH }, 0, 0, FLAGS, .unit = "profile" },
        { "edma", "Faraldo (2016), electronic/pop", 0, AV_OPT_TYPE_CONST, { .i64 = PROFILE_EDMA }, 0, 0, FLAGS, .unit = "profile" },
    { "tuning_hz", "reference A4 frequency in Hz", OFFSET(tuning_hz), AV_OPT_TYPE_DOUBLE, { .dbl = 440.0 }, 400, 480, FLAGS },
    { "detect_tuning", "estimate tuning deviation from the signal", OFFSET(detect_tuning), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "threshold", "minimum chord template similarity", OFFSET(chord_threshold), AV_OPT_TYPE_FLOAT, { .dbl = 0.55 }, 0, 1, FLAGS },
    { "silence_threshold", "silence RMS threshold in dB", OFFSET(silence_db), AV_OPT_TYPE_DOUBLE, { .dbl = -60.0 }, -120, 0, FLAGS },
    { "smooth_ms", "chord score smoothing length in ms", OFFSET(smooth_ms), AV_OPT_TYPE_INT, { .i64 = 1000 }, 100, 5000, FLAGS },
    { "min_chord_ms", "minimum chord duration in ms", OFFSET(min_chord_ms), AV_OPT_TYPE_INT, { .i64 = 400 }, 100, 5000, FLAGS },
    { "compression", "chroma log compression factor", OFFSET(compression), AV_OPT_TYPE_FLOAT, { .dbl = 100.0 }, 1, 10000, FLAGS },
    { "salience", "weight the key chroma by how tonal each window is", OFFSET(salience), AV_OPT_TYPE_FLOAT, { .dbl = 1.0 }, 0, 1, FLAGS },
    { "rerank", "break near ties in the key correlation using the chord progression", OFFSET(rerank), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "rerank_margin", "consider keys within this correlation distance of the leader", OFFSET(rerank_margin), AV_OPT_TYPE_DOUBLE, { .dbl = 0.06 }, 0, 1, FLAGS },
    { "rerank_gap", "chord score the challenger must win by", OFFSET(rerank_gap), AV_OPT_TYPE_DOUBLE, { .dbl = 0.05 }, 0, 2, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(keydetect);

static void chord_name(int idx, char *out, size_t out_size)
{
    if (idx < 0)
        snprintf(out, out_size, "N");
    else if (idx < 12)
        snprintf(out, out_size, "%s", pitch_names[idx]);
    else
        snprintf(out, out_size, "%sm", pitch_names[idx - 12]);
}

static void key_name(int idx, char *out, size_t out_size)
{
    if (idx < 0)
        out[0] = '\0';
    else
        snprintf(out, out_size, "%s%s", pitch_names[idx % 12], idx >= 12 ? "m" : "");
}

/* Oldest-to-newest progression string, at most max_items newest entries. */
static void progression_string(KeyDetectContext *s, char *out, size_t out_size,
                               int max_items)
{
    int n     = FFMIN(s->progression_len, max_items);
    int start = s->progression_len - n;
    size_t used = 0;

    out[0] = '\0';
    for (int i = start; i < s->progression_len && used < out_size - 1; i++) {
        char one[8];
        int len;
        chord_name(s->progression[i], one, sizeof(one));
        len = snprintf(out + used, out_size - used, "%s%s",
                       used ? "-" : "", one);
        if (len < 0 || (size_t)len >= out_size - used)
            break;
        used += len;
    }
}

static av_cold void uninit(AVFilterContext *ctx)
{
    KeyDetectContext *s = ctx->priv;

    av_tx_uninit(&s->tx_ctx);
    av_freep(&s->fft_in);
    av_freep(&s->fft_out);
    av_freep(&s->win_func);
    av_freep(&s->mag);
    av_freep(&s->ring);
    av_freep(&s->score_lp);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    KeyDetectContext *s = ctx->priv;
    float scale = 1.f;
    int ret;

    /* config_props may run more than once: release previous state */
    uninit(ctx);

    s->sample_rate = inlink->sample_rate;
    s->win_len = FFMAX(128, (int)((int64_t)s->sample_rate * s->window_ms / 1000));
    s->fft_len = 1 << av_ceil_log2(s->win_len);
    s->hop_len = FFMAX(1, (int)((int64_t)s->sample_rate * s->hop_ms / 1000));
    s->smooth_hops  = av_clip(s->smooth_ms / s->hop_ms, 1, SMOOTH_MAX_HOPS);
    s->min_dur_hops = FFMAX(1, s->min_chord_ms / s->hop_ms);
    s->silence_rms2 = pow(10.0, s->silence_db / 10.0);

    ret = av_tx_init(&s->tx_ctx, &s->tx_fn, AV_TX_FLOAT_RDFT, 0,
                     s->fft_len, &scale, 0);
    if (ret < 0)
        return ret;

    s->fft_in   = av_calloc(s->fft_len, sizeof(*s->fft_in));
    s->fft_out  = av_calloc(s->fft_len / 2 + 2, sizeof(*s->fft_out));
    s->win_func = av_malloc_array(s->win_len, sizeof(*s->win_func));
    s->mag      = av_malloc_array(s->fft_len / 2 + 1, sizeof(*s->mag));
    s->ring     = av_calloc(s->win_len, sizeof(*s->ring));
    s->score_lp = av_calloc((size_t)s->smooth_hops * NUM_CHORD_TEMPLATES,
                            sizeof(*s->score_lp));
    if (!s->fft_in || !s->fft_out || !s->win_func || !s->mag || !s->ring ||
        !s->score_lp)
        return AVERROR(ENOMEM);

    for (int i = 0; i < s->win_len; i++)
        s->win_func[i] = 0.5f - 0.5f * cosf(2.0 * M_PI * i / (s->win_len - 1));

    /* Rotation-invariant profile statistics for Pearson correlation */
    for (int mode = 0; mode < 2; mode++) {
        const double *p = key_profiles[s->profile][mode];
        double sum = 0.0, sq = 0.0;
        for (int i = 0; i < 12; i++) {
            sum += p[i];
            sq  += p[i] * p[i];
        }
        s->prof_mean[mode] = sum / 12.0;
        s->prof_sd[mode]   = sqrt(fmax(sq / 12.0 - s->prof_mean[mode] * s->prof_mean[mode], 1e-12));
    }

    s->total_in       = 0;
    s->next_win_end   = s->win_len;
    s->tune_offset    = 0.0;
    s->key_index      = -1;
    s->key_confidence = 0.0;
    s->key_margin     = 0.0;
    for (int i = 0; i < NUM_CANDIDATES; i++)
        s->key_rank[i] = -1;
    memset(s->key_r, 0, sizeof(s->key_r));
    s->score_pos      = 0;
    s->score_fill     = 0;
    s->seg_label      = INT_MIN;
    s->seg_len        = 0;
    s->stable_label   = INT_MIN;
    s->last_committed = INT_MIN;
    s->progression_len = 0;
    s->chord_first    = -1;
    s->chord_last     = -1;
    s->stable_hops    = 0;
    s->total_hops     = 0;
    s->reranked       = 0;
    memset(s->chord_dur, 0, sizeof(s->chord_dur));
    memset(s->tune_hist, 0, sizeof(s->tune_hist));
    memset(s->key_chroma, 0, sizeof(s->key_chroma));

    return 0;
}

/* Update the tuning estimate from the cents-deviation histogram. */
static void update_tuning(KeyDetectContext *s)
{
    int best = TUNING_BINS / 2;
    double total = 0.0;

    if (!s->detect_tuning)
        return;
    for (int i = 0; i < TUNING_BINS; i++) {
        total += s->tune_hist[i];
        if (s->tune_hist[i] > s->tune_hist[best])
            best = i;
    }
    if (total <= 0.0)
        return;
    /* bin center in cents relative to the equal tempered grid */
    s->tune_offset = (best - TUNING_BINS / 2 + 0.5) / 100.0;
}

/* Analyze one window ending at absolute sample position s->next_win_end.
 * Returns the current chord label (>= 0 chord index, -1 no chord). */
static int process_window(KeyDetectContext *s)
{
    float chroma[12] = { 0 };
    float scores[NUM_CHORD_TEMPLATES];
    float max_mag = 0.0f, vmax = 0.0f, chroma_norm2 = 0.0f, raw_best = 0.0f;
    double rms2 = 0.0;
    int64_t start = s->next_win_end - s->win_len;
    int k_lo, k_hi, silent;

    for (int i = 0; i < s->win_len; i++) {
        float v = s->ring[(start + i) % s->win_len];
        rms2 += (double)v * v;
        s->fft_in[i] = v * s->win_func[i];
    }
    memset(s->fft_in + s->win_len, 0,
           (s->fft_len - s->win_len) * sizeof(*s->fft_in));
    rms2 /= s->win_len;
    silent = rms2 < s->silence_rms2;

    if (!silent) {
        s->tx_fn(s->tx_ctx, s->fft_out, s->fft_in, sizeof(float));

        k_lo = FFMAX(2, (int)(FREQ_MIN * s->fft_len / s->sample_rate));
        k_hi = FFMIN(s->fft_len / 2 - 1,
                     (int)(FREQ_MAX * s->fft_len / s->sample_rate));

        for (int k = k_lo - 1; k <= k_hi + 1; k++) {
            float re = s->fft_out[k].re, im = s->fft_out[k].im;
            s->mag[k] = sqrtf(re * re + im * im);
            if (k >= k_lo && k <= k_hi && s->mag[k] > max_mag)
                max_mag = s->mag[k];
        }

        update_tuning(s);

        /* Spectral peaks -> tuning histogram + harmonic-folded chroma */
        for (int k = k_lo; k <= k_hi; k++) {
            float m = s->mag[k];
            double a, b, c, delta, freq, midi, dev;

            if (m < max_mag * PEAK_FLOOR_REL ||
                m <= s->mag[k - 1] || m < s->mag[k + 1])
                continue;

            /* Parabolic interpolation on log magnitude */
            a = log(s->mag[k - 1] + 1e-20);
            b = log(m + 1e-20);
            c = log(s->mag[k + 1] + 1e-20);
            delta = 0.5 * (a - c) / (a - 2.0 * b + c + 1e-20);
            delta = av_clipd(delta, -0.5, 0.5);
            freq = (k + delta) * (double)s->sample_rate / s->fft_len;

            /* Deviation from the equal tempered grid, in semitones */
            midi = 69.0 + 12.0 * log2(freq / s->tuning_hz);
            dev  = midi - floor(midi + 0.5);
            s->tune_hist[av_clip((int)floor((dev + 0.5) * TUNING_BINS),
                                 0, TUNING_BINS - 1)] += m;

            /* Fold the peak onto candidate fundamentals f/1 .. f/4 */
            for (int h = 1; h <= NUM_HARMONICS; h++) {
                double fund = freq / h;
                double fm, d;
                float w;
                int n, pc;

                if (fund < FUND_MIN)
                    break;
                fm = 69.0 + 12.0 * log2(fund / s->tuning_hz) - s->tune_offset;
                n  = (int)floor(fm + 0.5);
                d  = fm - n;                    /* [-0.5, 0.5] */
                pc = ((n % 12) + 12) % 12;
                w  = cosf(M_PI * d);            /* cos^2 sub-semitone window */
                chroma[pc] += m * (w * w) * powf(HARMONIC_DECAY, h - 1);
            }
        }

        /* Normalize to unit max first, then compress. Compressing the raw
         * magnitudes and normalizing afterwards makes the chroma shape depend
         * on the absolute input level: log1p(g*x)/log1p(g*xmax) tends to 1 for
         * every bin as the gain g grows, so loud masters yield a flatter
         * chroma than quiet ones and can decide a different key for the same
         * recording. Dividing by the max first is scale free, and
         * log1p(gamma*c)/log1p(gamma) keeps the result in [0, 1]
         * (Mueller/Ewert log compression). */
        for (int i = 0; i < 12; i++)
            if (chroma[i] > vmax)
                vmax = chroma[i];
        if (vmax > 0.0f) {
            const float lden = log1pf(s->compression);
            for (int i = 0; i < 12; i++) {
                chroma[i] = log1pf(s->compression * (chroma[i] / vmax)) / lden;
                chroma_norm2 += chroma[i] * chroma[i];
            }
        }
    }

    /* Chord template scores (cosine similarity, template norm = sqrt(3)) */
    for (int t = 0; t < NUM_CHORD_TEMPLATES; t++) {
        float dot = 0.0f;
        for (int i = 0; i < 12; i++)
            if (chord_templates[t][i])
                dot += chroma[i];
        scores[t] = chroma_norm2 > 1e-12f
                  ? dot / (sqrtf(chroma_norm2) * sqrtf(3.0f)) : 0.0f;
        if (scores[t] > raw_best)
            raw_best = scores[t];
    }

    /* Accumulate the key chroma weighted by how tonal this window is.
     * Unpitched content is not harmless: the spectrum is sampled on a
     * linear frequency grid while pitch classes are logarithmic, so noise
     * lands unevenly across the twelve bins in a fixed pattern. Averaging
     * every window equally therefore adds a systematically biased pedestal
     * that the Pearson correlation cannot cancel (it removes a constant
     * offset, not a shaped one), and enough of it decides the wrong key.
     *
     * The best raw chord similarity measures exactly the property wanted:
     * a flat chroma scores 3/(sqrt(12)*sqrt(3)) = 0.5 against any triad
     * template, a clean triad approaches 1. Rescaling that to [0, 1] gives
     * a tonalness weight for free, and the salience option controls how
     * much of it is applied so the old uniform behaviour stays reachable. */
    if (!silent && chroma_norm2 > 1e-12f) {
        float w = av_clipf(2.0f * raw_best - 1.0f, 0.0f, 1.0f);
        float weight = (1.0f - s->salience) + s->salience * w;
        for (int i = 0; i < 12; i++)
            s->key_chroma[i] += chroma[i] * weight;
    }

    /* Low-pass filter the scores over the last smooth_hops windows */
    memcpy(s->score_lp + (size_t)s->score_pos * NUM_CHORD_TEMPLATES,
           scores, sizeof(scores));
    s->score_pos = (s->score_pos + 1) % s->smooth_hops;
    if (s->score_fill < s->smooth_hops)
        s->score_fill++;

    {
        float best_sim = -1.0f;
        int best = -1;
        for (int t = 0; t < NUM_CHORD_TEMPLATES; t++) {
            float sum = 0.0f;
            for (int j = 0; j < s->score_fill; j++)
                sum += s->score_lp[(size_t)j * NUM_CHORD_TEMPLATES + t];
            sum /= s->score_fill;
            if (sum > best_sim) {
                best_sim = sum;
                best = t;
            }
        }
        return best_sim >= s->chord_threshold ? best : -1;
    }
}

/* Krumhansl-Schmuckler key finding: Pearson correlation of the accumulated
 * chroma against all 24 rotations of the selected profile pair. Keeps the
 * full correlation vector and the best NUM_CANDIDATES candidates, so that
 * callers can see how close the runner up was. */
static void compute_key(KeyDetectContext *s)
{
    double cm = 0.0, csd = 0.0;
    int rank[NUM_CANDIDATES];

    for (int i = 0; i < 12; i++)
        cm += s->key_chroma[i];
    cm /= 12.0;
    for (int i = 0; i < 12; i++)
        csd += (s->key_chroma[i] - cm) * (s->key_chroma[i] - cm);
    csd = sqrt(csd / 12.0);
    if (csd < 1e-9)
        return;

    for (int mode = 0; mode < 2; mode++) {
        const double *p = key_profiles[s->profile][mode];
        for (int root = 0; root < 12; root++) {
            double dot = 0.0;
            for (int i = 0; i < 12; i++)
                dot += s->key_chroma[i] * p[(i - root + 12) % 12];
            s->key_r[root + 12 * mode] =
                (dot / 12.0 - cm * s->prof_mean[mode]) /
                (csd * s->prof_sd[mode]);
        }
    }

    for (int i = 0; i < NUM_CANDIDATES; i++)
        rank[i] = -1;
    for (int k = 0; k < NUM_KEYS; k++) {
        for (int i = 0; i < NUM_CANDIDATES; i++) {
            if (rank[i] < 0 || s->key_r[k] > s->key_r[rank[i]]) {
                for (int j = NUM_CANDIDATES - 1; j > i; j--)
                    rank[j] = rank[j - 1];
                rank[i] = k;
                break;
            }
        }
    }

    memcpy(s->key_rank, rank, sizeof(rank));
    s->key_index      = rank[0];
    s->key_confidence = s->key_r[rank[0]];
    s->key_margin     = rank[1] >= 0 ? s->key_r[rank[0]] - s->key_r[rank[1]]
                                     : 0.0;
}

/* Duration weighted harmonic-function score of the committed chord
 * progression against one candidate key. The averaged chroma cannot see
 * which chords occurred or how long they lasted, but the segment tracker
 * already measured exactly that; this turns it into key evidence.
 * Range is roughly [-0.6, 1.35]. */
static double chord_key_score(KeyDetectContext *s, int key)
{
    const int kr = key % 12, km = key / 12;
    const int tonic = km ? kr + 12 : kr;
    double score = 0.0;

    if (s->stable_hops <= 0)
        return 0.0;

    for (int c = 0; c < NUM_CHORD_TEMPLATES; c++) {
        int iv;
        if (s->chord_dur[c] <= 0.0)
            continue;
        iv = ((c % 12) - kr + 12) % 12;
        score += (s->chord_dur[c] / s->stable_hops) *
                 func_weight[km][c / 12][iv];
    }
    if (s->chord_first == tonic)
        score += TONIC_FIRST_BONUS;
    if (s->chord_last == tonic)
        score += TONIC_LAST_BONUS;

    return score;
}

/* Is there enough committed chord evidence to be worth trusting? Thin,
 * chromatic output means the material has no chords to find rather than
 * that the detector failed, and must not override the chroma. */
static int chord_evidence_usable(KeyDetectContext *s)
{
    int distinct = 0;

    if (s->stable_hops * (int64_t)s->hop_ms < MIN_STABLE_MS)
        return 0;
    for (int c = 0; c < NUM_CHORD_TEMPLATES; c++)
        if (s->chord_dur[c] > 0.0)
            distinct++;
    return distinct >= MIN_DISTINCT_CHORDS;
}

/* Break a near tie in the chroma correlation using the chord progression.
 * Relative major/minor pairs share a pitch collection, so the correlation
 * separates them by hundredths and the winner is close to arbitrary; the
 * chord sequence is what actually carries the tonal centre. Only candidates
 * within rerank_margin of the leader are considered, and the challenger has
 * to beat the leader's chord score by a clear gap, so confident correlation
 * results are never disturbed. */
static void rerank_key(KeyDetectContext *s)
{
    double lead_score, best_score;
    int lead, best;

    s->reranked = 0;
    if (!s->rerank || s->key_rank[0] < 0 || !chord_evidence_usable(s))
        return;

    lead       = s->key_rank[0];
    lead_score = chord_key_score(s, lead);
    best       = lead;
    best_score = lead_score;

    for (int k = 0; k < NUM_KEYS; k++) {
        double sc;
        if (k == lead || s->key_r[k] < s->key_r[lead] - s->rerank_margin)
            continue;
        sc = chord_key_score(s, k);
        if (sc > best_score) {
            best_score = sc;
            best = k;
        }
    }

    /* best_score <= 0 means no candidate key explains the progression, i.e.
     * the committed chords are chromatic rather than harmonic (a pitched
     * riff or 808 line tracked faithfully because it is all the harmonic
     * content there is). Say nothing rather than override the chroma. */
    if (best == lead || best_score <= 0.0 ||
        best_score - lead_score < s->rerank_gap)
        return;

    /* Keep the displaced correlation winner as the reported alternative.
     * key_margin deliberately keeps meaning "how far apart the two leading
     * correlation candidates were", which is the ambiguity measure worth
     * thresholding on, so it is not rewritten here. */
    s->key_rank[1] = lead;
    s->key_rank[0] = best;
    s->key_index      = best;
    s->key_confidence = s->key_r[best];
    s->reranked       = 1;
}

/* "Cm:0.885,D#:0.879,Gm:0.812" over the reported candidates. */
static void candidates_string(KeyDetectContext *s, char *out, size_t out_size)
{
    size_t used = 0;

    out[0] = '\0';
    for (int i = 0; i < NUM_CANDIDATES && s->key_rank[i] >= 0; i++) {
        char one[8];
        int len;
        key_name(s->key_rank[i], one, sizeof(one));
        len = snprintf(out + used, out_size - used, "%s%s:%.3f",
                       used ? "," : "", one, s->key_r[s->key_rank[i]]);
        if (len < 0 || (size_t)len >= out_size - used)
            break;
        used += len;
    }
}

/* Segment tracking: a label must persist for min_dur_hops before it becomes
 * the stable chord and is committed to the progression. */
static void update_segments(AVFilterContext *ctx, int label, double t)
{
    KeyDetectContext *s = ctx->priv;

    if (label == s->seg_label) {
        s->seg_len++;
    } else {
        s->seg_label = label;
        s->seg_len = 1;
    }

    if (s->seg_len == s->min_dur_hops) {
        s->stable_label = s->seg_label;
        if (s->seg_label >= 0 && s->seg_label != s->last_committed) {
            char name[8];
            if (s->progression_len == PROGRESSION_MAX) {
                memmove(s->progression, s->progression + 1,
                        (PROGRESSION_MAX - 1) * sizeof(*s->progression));
                s->progression_len--;
            }
            s->progression[s->progression_len++] = s->seg_label;
            s->last_committed = s->seg_label;
            chord_name(s->seg_label, name, sizeof(name));
            av_log(ctx, AV_LOG_INFO, "t=%.2f chord=%s\n", t, name);
        }
    }

    /* Duration of every stable chord, for the key decision. */
    s->total_hops++;
    if (s->stable_label >= 0) {
        s->chord_dur[s->stable_label] += 1.0;
        s->stable_hops++;
        if (s->chord_first < 0)
            s->chord_first = s->stable_label;
        s->chord_last = s->stable_label;
    }
}

static void set_frame_metadata(KeyDetectContext *s, AVFrame *frame)
{
    char buf[PROG_STR_SIZE];

    if (s->key_index >= 0) {
        key_name(s->key_index, buf, sizeof(buf));
        av_dict_set(&frame->metadata, "lavfi.keydetect.key", buf, 0);
        snprintf(buf, sizeof(buf), "%.3f", s->key_confidence);
        av_dict_set(&frame->metadata, "lavfi.keydetect.key_confidence", buf, 0);
        snprintf(buf, sizeof(buf), "%.3f", s->key_margin);
        av_dict_set(&frame->metadata, "lavfi.keydetect.key_margin", buf, 0);
        if (s->key_rank[1] >= 0) {
            key_name(s->key_rank[1], buf, sizeof(buf));
            av_dict_set(&frame->metadata, "lavfi.keydetect.key_alt", buf, 0);
            snprintf(buf, sizeof(buf), "%.3f", s->key_r[s->key_rank[1]]);
            av_dict_set(&frame->metadata,
                        "lavfi.keydetect.key_alt_confidence", buf, 0);
        }
        candidates_string(s, buf, sizeof(buf));
        av_dict_set(&frame->metadata, "lavfi.keydetect.key_candidates", buf, 0);
        av_dict_set(&frame->metadata, "lavfi.keydetect.key_reranked",
                    s->reranked ? "1" : "0", 0);
    }
    if (s->stable_label != INT_MIN) {
        chord_name(s->stable_label, buf, sizeof(buf));
        av_dict_set(&frame->metadata, "lavfi.keydetect.chord", buf, 0);
    }
    if (s->progression_len > 0) {
        progression_string(s, buf, sizeof(buf), 16);
        av_dict_set(&frame->metadata, "lavfi.keydetect.chords", buf, 0);
    }
}

static void emit_final(AVFilterContext *ctx)
{
    KeyDetectContext *s = ctx->priv;
    char buf[PROG_STR_SIZE];

    if (s->key_index >= 0) {
        key_name(s->key_index, buf, sizeof(buf));
        av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.key=%s\n", buf);
        fprintf(stderr, "\nlavfi.keydetect.key=%s\n", buf);
        snprintf(buf, sizeof(buf), "%.3f", s->key_confidence);
        av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.key_confidence=%s\n", buf);
        fprintf(stderr, "lavfi.keydetect.key_confidence=%s\n", buf);
        if (s->key_rank[1] >= 0) {
            key_name(s->key_rank[1], buf, sizeof(buf));
            av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.key_alt=%s\n", buf);
            fprintf(stderr, "lavfi.keydetect.key_alt=%s\n", buf);
            snprintf(buf, sizeof(buf), "%.3f", s->key_r[s->key_rank[1]]);
            av_log(ctx, AV_LOG_INFO,
                   "lavfi.keydetect.key_alt_confidence=%s\n", buf);
            fprintf(stderr, "lavfi.keydetect.key_alt_confidence=%s\n", buf);
        }
        snprintf(buf, sizeof(buf), "%.3f", s->key_margin);
        av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.key_margin=%s\n", buf);
        fprintf(stderr, "lavfi.keydetect.key_margin=%s\n", buf);
        candidates_string(s, buf, sizeof(buf));
        av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.key_candidates=%s\n", buf);
        fprintf(stderr, "lavfi.keydetect.key_candidates=%s\n", buf);
        av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.key_reranked=%d\n",
               s->reranked);
        fprintf(stderr, "lavfi.keydetect.key_reranked=%d\n", s->reranked);
    }
    if (s->progression_len > 0) {
        progression_string(s, buf, sizeof(buf), PROGRESSION_MAX);
        av_log(ctx, AV_LOG_INFO, "lavfi.keydetect.chords=%s\n", buf);
        fprintf(stderr, "lavfi.keydetect.chords=%s\n", buf);
    }
}

static int filter_frame(AVFilterContext *ctx, AVFrame *frame)
{
    KeyDetectContext *s = ctx->priv;
    const int nb_channels = frame->ch_layout.nb_channels;
    const int nb_samples  = frame->nb_samples;
    int key_before = s->key_index;

    for (int i = 0; i < nb_samples; i++) {
        float v = 0.0f;
        for (int ch = 0; ch < nb_channels; ch++)
            v += ((const float *)frame->extended_data[ch])[i];
        v /= nb_channels;

        s->ring[s->total_in % s->win_len] = v;
        s->total_in++;

        if (s->total_in >= s->next_win_end) {
            double t = (double)s->next_win_end / s->sample_rate;
            int label = process_window(s);
            update_segments(ctx, label, t);
            compute_key(s);
            rerank_key(s);
            s->next_win_end += s->hop_len;
        }
    }

    if (s->key_index >= 0 && s->key_index != key_before) {
        char buf[8];
        key_name(s->key_index, buf, sizeof(buf));
        av_log(ctx, AV_LOG_INFO, "t=%.2f key=%s (r=%.2f)\n",
               (double)s->total_in / s->sample_rate, buf, s->key_confidence);
    }

    set_frame_metadata(s, frame);
    return ff_filter_frame(ctx->outputs[0], frame);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink  = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *frame = NULL;
    int ret, status;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = ff_inlink_consume_frame(inlink, &frame);
    if (ret < 0)
        return ret;
    if (ret > 0)
        return filter_frame(ctx, frame);

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF)
            emit_final(ctx);
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);
    return FFERROR_NOT_READY;
}

static const AVFilterPad keydetect_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = config_input,
    },
};

const FFFilter ff_af_keydetect = {
    .p.name        = "keydetect",
    .p.description = NULL_IF_CONFIG_SMALL("Detect musical key and chord progression from audio."),
    .p.priv_class  = &keydetect_class,
    .p.flags       = AVFILTER_FLAG_METADATA_ONLY,
    .priv_size     = sizeof(KeyDetectContext),
    .uninit        = uninit,
    .activate      = activate,
    FILTER_INPUTS(keydetect_inputs),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_SAMPLEFMTS(AV_SAMPLE_FMT_FLTP),
};
