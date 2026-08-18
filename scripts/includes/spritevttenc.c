/*
 * Sprite sheet + WebVTT muxer
 * Copyright (c) 2026 NoMercy Entertainment
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Sprite sheet muxer with WebVTT timeline
 *
 * Produces a tiled sprite sheet image (.png or .webp) and a companion .vtt
 * file mapping timestamps to sprite regions using W3C Media Fragments #xywh=.
 *
 *   ffmpeg -i input.mp4 -vf "fps=1/5,scale=160:90" -f spritevtt out.webp
 *   # Produces: out.webp + out.vtt
 */

#include <math.h>

#include "avformat.h"
#include "internal.h"
#include "mux.h"
#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"

typedef struct SpriteVTTContext {
    AVClass *class;

    /* Options */
    int sprite_columns;    /* 0 = auto (square grid) */
    char *vtt_filename;    /* override VTT filename */
    int relative_path;     /* use relative path in VTT cues (default 1) */

    /* Frame buffer */
    uint8_t **frame_data;  /* array of raw frame data buffers */
    int *frame_sizes;      /* array of frame data sizes */
    int64_t *frame_pts;    /* array of presentation timestamps */
    int frame_count;       /* number of buffered frames */
    int frame_capacity;    /* allocated capacity */

    /* Frame dimensions (from stream codecpar) */
    int frame_w;
    int frame_h;
    enum AVPixelFormat pix_fmt;
    enum AVColorRange color_range;
    int frame_data_size;   /* total bytes per frame */
} SpriteVTTContext;

#define OFFSET(x) offsetof(SpriteVTTContext, x)
#define E AV_OPT_FLAG_ENCODING_PARAM

static const AVOption spritevtt_options[] = {
    { "sprite_columns", "Number of columns in sprite grid (0=auto square)",
      OFFSET(sprite_columns), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1000, E },
    { "vtt_filename", "Override companion VTT filename",
      OFFSET(vtt_filename), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "relative_path", "Use relative filename in VTT cues",
      OFFSET(relative_path), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, E },
    { NULL },
};

static const AVClass spritevtt_class = {
    .class_name = "spritevtt muxer",
    .item_name  = av_default_item_name,
    .option     = spritevtt_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

/**
 * Pick the color range to paint empty grid cells with.
 *
 * Neither the swscale context built in the trailer (created without range
 * hints) nor the image encoders are told the stream's color range — both
 * infer it from the pixel format alone — so the fill has to be derived the
 * same way, otherwise the "black" written here does not decode as black.
 * Formats that are full range by definition therefore win over whatever the
 * stream claims; for everything else an explicit signal is honored and
 * limited range is the fallback.
 */
static enum AVColorRange spritevtt_fill_range(enum AVPixelFormat pix_fmt,
                                              enum AVColorRange stream_range)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);

    switch (pix_fmt) {
    /* Full range by name; matches swscale's handle_jpeg() */
    case AV_PIX_FMT_YUVJ411P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUVJ440P:
    case AV_PIX_FMT_YUVJ444P:
        return AVCOL_RANGE_JPEG;
    default:
        break;
    }

    /* Grayscale, with or without alpha, is full range there too */
    if (desc && desc->nb_components <= 2 &&
        !(desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL)))
        return AVCOL_RANGE_JPEG;

    if (stream_range == AVCOL_RANGE_JPEG || stream_range == AVCOL_RANGE_MPEG)
        return stream_range;

    return AVCOL_RANGE_MPEG;
}

static void format_vtt_time(char *buf, size_t buf_size, int64_t ms)
{
    int h  = (int)(ms / 3600000);
    int m  = (int)((ms % 3600000) / 60000);
    int s  = (int)((ms % 60000) / 1000);
    int ml = (int)(ms % 1000);
    snprintf(buf, buf_size, "%02d:%02d:%02d.%03d", h, m, s, ml);
}

static av_cold int spritevtt_init(AVFormatContext *s)
{
    SpriteVTTContext *ctx = s->priv_data;
    AVStream *st;

    if (s->nb_streams != 1) {
        av_log(s, AV_LOG_ERROR,
               "spritevtt muxer requires exactly 1 video stream, got %d\n",
               s->nb_streams);
        return AVERROR(EINVAL);
    }

    st = s->streams[0];

    if (st->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
        av_log(s, AV_LOG_ERROR,
               "spritevtt muxer requires a video stream\n");
        return AVERROR(EINVAL);
    }

    ctx->frame_w     = st->codecpar->width;
    ctx->frame_h     = st->codecpar->height;
    ctx->pix_fmt     = (enum AVPixelFormat)st->codecpar->format;
    ctx->color_range = st->codecpar->color_range;

    if (ctx->frame_w <= 0 || ctx->frame_h <= 0) {
        av_log(s, AV_LOG_ERROR,
               "Invalid frame dimensions %dx%d\n",
               ctx->frame_w, ctx->frame_h);
        return AVERROR(EINVAL);
    }

    if (ctx->pix_fmt == AV_PIX_FMT_NONE) {
        av_log(s, AV_LOG_ERROR, "Unknown pixel format\n");
        return AVERROR(EINVAL);
    }

    ctx->frame_data_size = av_image_get_buffer_size(ctx->pix_fmt,
                                                     ctx->frame_w,
                                                     ctx->frame_h, 1);
    if (ctx->frame_data_size <= 0) {
        av_log(s, AV_LOG_ERROR,
               "Could not determine frame buffer size for %s %dx%d\n",
               av_get_pix_fmt_name(ctx->pix_fmt),
               ctx->frame_w, ctx->frame_h);
        return AVERROR(EINVAL);
    }

    /* Initial allocation */
    ctx->frame_capacity = 256;
    ctx->frame_data = av_calloc(ctx->frame_capacity, sizeof(*ctx->frame_data));
    ctx->frame_sizes = av_calloc(ctx->frame_capacity, sizeof(*ctx->frame_sizes));
    ctx->frame_pts  = av_calloc(ctx->frame_capacity, sizeof(*ctx->frame_pts));
    if (!ctx->frame_data || !ctx->frame_sizes || !ctx->frame_pts)
        return AVERROR(ENOMEM);

    ctx->frame_count = 0;

    avpriv_set_pts_info(st, 64, 1, 1000);

    return 0;
}

static int spritevtt_write_header(AVFormatContext *s)
{
    return 0;
}

static int spritevtt_write_packet(AVFormatContext *s, AVPacket *pkt)
{
    SpriteVTTContext *ctx = s->priv_data;

    if (!pkt->data || pkt->size <= 0)
        return 0;

    /* Grow arrays if at capacity */
    if (ctx->frame_count >= ctx->frame_capacity) {
        int new_cap = ctx->frame_capacity * 2;
        uint8_t **new_data;
        int *new_sizes;
        int64_t *new_pts;

        new_data = av_realloc_array(ctx->frame_data, new_cap,
                                    sizeof(*ctx->frame_data));
        if (!new_data)
            return AVERROR(ENOMEM);
        ctx->frame_data = new_data;

        new_sizes = av_realloc_array(ctx->frame_sizes, new_cap,
                                     sizeof(*ctx->frame_sizes));
        if (!new_sizes)
            return AVERROR(ENOMEM);
        ctx->frame_sizes = new_sizes;

        new_pts = av_realloc_array(ctx->frame_pts, new_cap,
                                   sizeof(*ctx->frame_pts));
        if (!new_pts)
            return AVERROR(ENOMEM);
        ctx->frame_pts = new_pts;

        ctx->frame_capacity = new_cap;
    }

    ctx->frame_data[ctx->frame_count] = av_memdup(pkt->data, pkt->size);
    if (!ctx->frame_data[ctx->frame_count])
        return AVERROR(ENOMEM);

    ctx->frame_sizes[ctx->frame_count] = pkt->size;
    ctx->frame_pts[ctx->frame_count]   = pkt->pts;
    ctx->frame_count++;

    return 0;
}

/**
 * Blit a single raw frame buffer onto the canvas at (dst_x, dst_y).
 * Handles both planar and packed pixel formats correctly.
 */
static int blit_frame_to_canvas(SpriteVTTContext *ctx, AVFrame *canvas,
                                const uint8_t *frame_buf, int frame_buf_size,
                                int dst_x, int dst_y)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(ctx->pix_fmt);
    uint8_t *src_data[4] = { NULL };
    int src_linesize[4] = { 0 };
    int ret, p;

    if (!desc)
        return AVERROR(EINVAL);

    /* Fill source plane pointers from the flat buffer */
    ret = av_image_fill_arrays(src_data, src_linesize,
                               frame_buf, ctx->pix_fmt,
                               ctx->frame_w, ctx->frame_h, 1);
    if (ret < 0)
        return ret;

    /* Copy each plane */
    for (p = 0; p < 4 && src_data[p]; p++) {
        int plane_w = ctx->frame_w;
        int plane_h = ctx->frame_h;
        int plane_dst_x = dst_x;
        int plane_dst_y = dst_y;
        int bytes_per_pixel;
        int y;

        /* Chroma planes have subsampled dimensions */
        if (p == 1 || p == 2) {
            plane_w = AV_CEIL_RSHIFT(ctx->frame_w, desc->log2_chroma_w);
            plane_h = AV_CEIL_RSHIFT(ctx->frame_h, desc->log2_chroma_h);
            plane_dst_x = AV_CEIL_RSHIFT(dst_x, desc->log2_chroma_w);
            plane_dst_y = AV_CEIL_RSHIFT(dst_y, desc->log2_chroma_h);
        }

        /* Calculate bytes per pixel for this plane */
        if (desc->nb_components == 1) {
            /* Grayscale: 1 byte per pixel */
            bytes_per_pixel = 1;
        } else if (p == 0) {
            /* Luma plane or packed first plane */
            bytes_per_pixel = (desc->comp[0].step > 0) ? desc->comp[0].step : 1;
        } else {
            /* Chroma planes */
            bytes_per_pixel = (desc->comp[p].step > 0) ? desc->comp[p].step : 1;
        }

        for (y = 0; y < plane_h; y++) {
            uint8_t *dst = canvas->data[p]
                         + (plane_dst_y + y) * canvas->linesize[p]
                         + plane_dst_x * bytes_per_pixel;
            const uint8_t *src = src_data[p]
                               + y * src_linesize[p];
            memcpy(dst, src, plane_w * bytes_per_pixel);
        }
    }

    return 0;
}

static int spritevtt_write_trailer(AVFormatContext *s)
{
    SpriteVTTContext *ctx = s->priv_data;
    int n = ctx->frame_count;
    int cols, rows, grid_w, grid_h;
    const char *ext;
    int is_webp;
    AVFrame *canvas = NULL;
    AVFrame *converted = NULL;
    const AVCodec *enc = NULL;
    AVCodecContext *enc_ctx = NULL;
    AVPacket *out_pkt = NULL;
    struct SwsContext *sws = NULL;
    AVIOContext *vtt_pb = NULL;
    char *vtt_path = NULL;
    const enum AVPixelFormat *sup_fmts = NULL;
    int nb_sup_fmts = 0;
    int ret = 0;
    int fmt_ok, i, p;

    if (n == 0) {
        av_log(s, AV_LOG_WARNING,
               "No frames received, writing empty VTT\n");
        goto write_vtt;
    }

    /* Calculate grid dimensions */
    cols = ctx->sprite_columns > 0
         ? ctx->sprite_columns
         : (int)ceil(sqrt((double)n));
    rows = (n + cols - 1) / cols;
    grid_w = cols * ctx->frame_w;
    grid_h = rows * ctx->frame_h;

    /* Validate dimensions against format limits */
    ext = strrchr(s->url, '.');
    is_webp = ext && !av_strcasecmp(ext, ".webp");

    if (is_webp && (grid_w > 16383 || grid_h > 16383)) {
        av_log(s, AV_LOG_ERROR,
               "Sprite sheet dimensions %dx%d exceed WebP maximum (16383x16383). "
               "Reduce frame count (larger fps interval), use smaller thumbnails, "
               "or use PNG output.\n",
               grid_w, grid_h);
        return AVERROR(EINVAL);
    }

    /* Allocate canvas */
    canvas = av_frame_alloc();
    if (!canvas)
        return AVERROR(ENOMEM);

    canvas->format = ctx->pix_fmt;
    canvas->width  = grid_w;
    canvas->height = grid_h;

    ret = av_frame_get_buffer(canvas, 0);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR,
               "Failed to allocate canvas %dx%d\n", grid_w, grid_h);
        goto cleanup;
    }

    /* Paint the whole canvas black up front; the blit below overwrites the
     * cells that have a frame, and whatever is left over is what the player
     * shows after the last thumbnail. Zeroing the planes is not enough: zero
     * is neutral only for RGB, while zeroed YUV chroma is saturated green. */
    {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(ctx->pix_fmt);
        ptrdiff_t canvas_linesizes[4];

        for (p = 0; p < 4; p++)
            canvas_linesizes[p] = canvas->linesize[p];

        ret = av_image_fill_black(canvas->data, canvas_linesizes, ctx->pix_fmt,
                                  spritevtt_fill_range(ctx->pix_fmt,
                                                       ctx->color_range),
                                  grid_w, grid_h);
        if (ret < 0) {
            /* Exotic format av_image_fill_black() cannot express black in.
             * Fall back to zero-fill — wrong color for YUV, but the buffer
             * from av_frame_get_buffer() is uninitialized otherwise. */
            av_log(s, AV_LOG_WARNING,
                   "Could not fill canvas with black for pixel format %s (%s); "
                   "empty grid cells may show a color cast\n",
                   av_get_pix_fmt_name(ctx->pix_fmt), av_err2str(ret));
            for (p = 0; p < 4 && canvas->data[p]; p++) {
                int plane_h = grid_h;
                /* Only chroma planes (1, 2) are subsampled; luma (0) and alpha (3) are full-size */
                if (p == 1 || p == 2)
                    plane_h = AV_CEIL_RSHIFT(grid_h, desc->log2_chroma_h);
                memset(canvas->data[p], 0, (size_t)canvas->linesize[p] * plane_h);
            }
            ret = 0;
        }
    }

    /* Blit each buffered frame into its grid position */
    for (i = 0; i < n; i++) {
        int col = i % cols;
        int row = i / cols;
        int dst_x = col * ctx->frame_w;
        int dst_y = row * ctx->frame_h;

        ret = blit_frame_to_canvas(ctx, canvas,
                                   ctx->frame_data[i],
                                   ctx->frame_sizes[i],
                                   dst_x, dst_y);
        if (ret < 0) {
            av_log(s, AV_LOG_ERROR,
                   "Failed to blit frame %d to canvas\n", i);
            goto cleanup;
        }
    }

    /* Find image encoder */
    {
        enum AVCodecID codec_id = is_webp ? AV_CODEC_ID_WEBP : AV_CODEC_ID_PNG;
        enc = avcodec_find_encoder(codec_id);
        if (!enc) {
            av_log(s, AV_LOG_ERROR,
                   "Could not find %s encoder\n",
                   is_webp ? "WebP" : "PNG");
            ret = AVERROR_ENCODER_NOT_FOUND;
            goto cleanup;
        }
    }

    enc_ctx = avcodec_alloc_context3(enc);
    if (!enc_ctx) {
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    enc_ctx->width     = grid_w;
    enc_ctx->height    = grid_h;
    enc_ctx->time_base = (AVRational){ 1, 1 };

    /* Check if canvas pixel format is supported by the encoder.
     * AVCodec.pix_fmts was removed in FFmpeg 9; avcodec_get_supported_config()
     * is the replacement. A NULL list means every pixel format is accepted. */
    fmt_ok = 0;
    ret = avcodec_get_supported_config(NULL, enc, AV_CODEC_CONFIG_PIX_FORMAT, 0,
                                       (const void **)&sup_fmts, &nb_sup_fmts);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR,
               "Could not query supported pixel formats for %s encoder\n",
               enc->name);
        goto cleanup;
    }

    if (!sup_fmts) {
        fmt_ok = 1;
    } else {
        for (p = 0; p < nb_sup_fmts; p++) {
            if (sup_fmts[p] == ctx->pix_fmt) {
                fmt_ok = 1;
                break;
            }
        }
    }

    if (fmt_ok) {
        enc_ctx->pix_fmt = ctx->pix_fmt;
    } else {
        /* Use first supported format and convert with swscale */
        if (nb_sup_fmts <= 0 || sup_fmts[0] == AV_PIX_FMT_NONE) {
            av_log(s, AV_LOG_ERROR,
                   "Encoder has no supported pixel formats\n");
            ret = AVERROR(EINVAL);
            goto cleanup;
        }
        enc_ctx->pix_fmt = sup_fmts[0];

        av_log(s, AV_LOG_INFO,
               "Converting pixel format %s -> %s for %s encoder\n",
               av_get_pix_fmt_name(ctx->pix_fmt),
               av_get_pix_fmt_name(enc_ctx->pix_fmt),
               enc->name);

        sws = sws_getContext(grid_w, grid_h, ctx->pix_fmt,
                             grid_w, grid_h, enc_ctx->pix_fmt,
                             SWS_BILINEAR, NULL, NULL, NULL);
        if (!sws) {
            av_log(s, AV_LOG_ERROR,
                   "Could not create swscale context\n");
            ret = AVERROR(ENOMEM);
            goto cleanup;
        }

        converted = av_frame_alloc();
        if (!converted) {
            ret = AVERROR(ENOMEM);
            goto cleanup;
        }
        converted->format = enc_ctx->pix_fmt;
        converted->width  = grid_w;
        converted->height = grid_h;

        ret = av_frame_get_buffer(converted, 0);
        if (ret < 0) {
            av_log(s, AV_LOG_ERROR,
                   "Failed to allocate converted frame\n");
            goto cleanup;
        }

        sws_scale(sws, (const uint8_t *const *)canvas->data,
                  canvas->linesize, 0, grid_h,
                  converted->data, converted->linesize);
    }

    ret = avcodec_open2(enc_ctx, enc, NULL);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR,
               "Failed to open %s encoder: %s\n",
               enc->name, av_err2str(ret));
        goto cleanup;
    }

    out_pkt = av_packet_alloc();
    if (!out_pkt) {
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    ret = avcodec_send_frame(enc_ctx, converted ? converted : canvas);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR,
               "Error sending frame to encoder: %s\n",
               av_err2str(ret));
        goto cleanup;
    }

    /* Flush encoder — signal end of stream so buffered encoders
     * (libwebp_anim) produce output. Without this, receive_packet
     * returns EAGAIN because the encoder is waiting for more frames. */
    avcodec_send_frame(enc_ctx, NULL);

    /* Drain all encoded packets */
    while (1) {
        ret = avcodec_receive_packet(enc_ctx, out_pkt);
        if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
            break;
        if (ret < 0) {
            av_log(s, AV_LOG_ERROR,
                   "Error receiving packet from encoder: %s\n",
                   av_err2str(ret));
            goto cleanup;
        }
        avio_write(s->pb, out_pkt->data, out_pkt->size);
        av_packet_unref(out_pkt);
    }
    avio_flush(s->pb);
    ret = 0;

write_vtt:
    /* Determine VTT filename */
    if (ctx->vtt_filename) {
        /* Build path: extract directory from s->url, append vtt_filename */
        const char *last_sep = strrchr(s->url, '/');
        const char *last_bsep;

        if (!last_sep)
            last_sep = strrchr(s->url, '\\');

        /* Check for backslash after last forward slash */
        last_bsep = strrchr(s->url, '\\');
        if (last_bsep && (!last_sep || last_bsep > last_sep))
            last_sep = last_bsep;

        if (last_sep) {
            int dir_len = (int)(last_sep - s->url + 1);
            vtt_path = av_asprintf("%.*s%s",
                                   dir_len, s->url, ctx->vtt_filename);
        } else {
            vtt_path = av_strdup(ctx->vtt_filename);
        }
    } else {
        /* Replace output extension with .vtt */
        const char *dot = strrchr(s->url, '.');
        if (dot) {
            int base_len = (int)(dot - s->url);
            vtt_path = av_asprintf("%.*s.vtt", base_len, s->url);
        } else {
            vtt_path = av_asprintf("%s.vtt", s->url);
        }
    }

    if (!vtt_path) {
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    ret = avio_open(&vtt_pb, vtt_path, AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR,
               "Failed to open VTT file '%s': %s\n",
               vtt_path, av_err2str(ret));
        goto cleanup;
    }

    avio_printf(vtt_pb, "WEBVTT\n\n");

    if (n > 0) {
        /* Determine image filename for VTT cues */
        const char *img_name;

        if (ctx->relative_path) {
            const char *sep = strrchr(s->url, '/');
            const char *bsep = strrchr(s->url, '\\');
            if (bsep && (!sep || bsep > sep))
                sep = bsep;
            img_name = sep ? sep + 1 : s->url;
        } else {
            img_name = s->url;
        }

        /* Write cues */
        for (i = 0; i < n; i++) {
            int col = i % cols;
            int row = i / cols;
            int x = col * ctx->frame_w;
            int y = row * ctx->frame_h;
            int64_t start_ms = ctx->frame_pts[i];
            int64_t end_ms;
            char start_str[16], end_str[16];

            if (i + 1 < n)
                end_ms = ctx->frame_pts[i + 1];
            else if (n >= 2)
                end_ms = start_ms + (ctx->frame_pts[n - 1] - ctx->frame_pts[n - 2]);
            else
                end_ms = start_ms + 5000; /* fallback: 5s for single frame */

            format_vtt_time(start_str, sizeof(start_str), start_ms);
            format_vtt_time(end_str,   sizeof(end_str),   end_ms);

            avio_printf(vtt_pb, "%s --> %s\n%s#xywh=%d,%d,%d,%d\n\n",
                        start_str, end_str, img_name,
                        x, y, ctx->frame_w, ctx->frame_h);
        }
    }

    avio_flush(vtt_pb);
    avio_closep(&vtt_pb);

    ret = 0;

cleanup:
    av_packet_free(&out_pkt);
    avcodec_free_context(&enc_ctx);
    if (sws)
        sws_freeContext(sws);
    av_frame_free(&converted);
    av_frame_free(&canvas);
    av_freep(&vtt_path);

    return ret;
}

static void spritevtt_deinit(AVFormatContext *s)
{
    SpriteVTTContext *ctx = s->priv_data;
    int i;

    if (ctx->frame_data) {
        for (i = 0; i < ctx->frame_count; i++)
            av_freep(&ctx->frame_data[i]);
        av_freep(&ctx->frame_data);
    }

    av_freep(&ctx->frame_sizes);
    av_freep(&ctx->frame_pts);

    ctx->frame_count    = 0;
    ctx->frame_capacity = 0;
}

const FFOutputFormat ff_spritevtt_muxer = {
    .p.name           = "spritevtt",
    .p.long_name      = NULL_IF_CONFIG_SMALL("Sprite sheet with WebVTT timeline"),
    .p.extensions     = "spritevtt",
    .p.video_codec    = AV_CODEC_ID_RAWVIDEO,
    .p.audio_codec    = AV_CODEC_ID_NONE,
    .p.subtitle_codec = AV_CODEC_ID_NONE,
    .p.flags          = AVFMT_VARIABLE_FPS | AVFMT_TS_NONSTRICT,
    .p.priv_class     = &spritevtt_class,
    .flags_internal   = FF_OFMT_FLAG_MAX_ONE_OF_EACH,
    .priv_data_size   = sizeof(SpriteVTTContext),
    .init             = spritevtt_init,
    .write_header     = spritevtt_write_header,
    .write_packet     = spritevtt_write_packet,
    .write_trailer    = spritevtt_write_trailer,
    .deinit           = spritevtt_deinit,
};
