/*
 * DVD-Video (libdvdread) protocol
 *
 * Exposes the title-set VOBs of a DVD-Video structure (disc device, ISO
 * image or VIDEO_TS directory) as a single seekable MPEG-PS byte stream,
 * mirroring the shape of the bluray: protocol. Fine-grained title/chapter
 * handling remains the job of the dvdvideo demuxer; this is a thin
 * byte-level wrapper so that <scheme>://<device> URI callers keep working.
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

#include <inttypes.h>
#include <string.h>

#include <dvdread/dvd_reader.h>
#include <dvdread/ifo_read.h>
#include <dvdread/ifo_types.h>

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "avformat.h"
#include "url.h"

#define DVDREAD_PROTO_PREFIX "dvdread:"
#define DVD_BLOCK_SIZE       DVD_VIDEO_LB_LEN /* 2048 */
#define DVD_MAX_TITLE_SETS   99

typedef struct DVDReadContext {
    const AVClass *class;

    dvd_reader_t  *dvd;
    dvd_file_t    *file;

    /** user option: logical title number, 0 = auto (main feature) */
    int            title;

    uint32_t       blocks;   /**< size of the opened title-set VOBs in blocks */
    int64_t        pos;      /**< current byte position */

    int            cache_lb; /**< block held in cache, -1 = none */
    uint8_t        cache[DVD_BLOCK_SIZE];
} DVDReadContext;

#define OFFSET(x) offsetof(DVDReadContext, x)
static const AVOption options[] = {
    { "title", "logical title number (0 = main feature)",
      OFFSET(title), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 99, AV_OPT_FLAG_DECODING_PARAM },
    { NULL }
};

static const AVClass dvdread_context_class = {
    .class_name = "dvdread",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

/**
 * Map the requested logical title to its title set (VTS) via the VMG IFO,
 * or, when no title was requested, pick the title set with the largest
 * title VOBs (the main feature on virtually every disc).
 *
 * Returns the VTS number (>= 1) or a negative AVERROR code.
 */
static int dvdread_resolve_vts(URLContext *h)
{
    DVDReadContext *dc  = h->priv_data;
    ifo_handle_t   *vmg = ifoOpen(dc->dvd, 0);
    int             vts = 0;

    if (dc->title > 0) {
        if (!vmg) {
            av_log(h, AV_LOG_WARNING,
                   "cannot open VMG IFO; assuming title %d maps to title set %d\n",
                   dc->title, dc->title);
            return dc->title;
        }
        if (!vmg->tt_srpt || dc->title > vmg->tt_srpt->nr_of_srpts) {
            av_log(h, AV_LOG_ERROR, "title %d not present on disc (%d titles)\n",
                   dc->title, vmg->tt_srpt ? vmg->tt_srpt->nr_of_srpts : 0);
            ifoClose(vmg);
            return AVERROR(EINVAL);
        }
        vts = vmg->tt_srpt->title[dc->title - 1].title_set_nr;
    } else {
        int      nb_vts = vmg ? vmg->vmgi_mat->vmg_nr_of_title_sets
                              : DVD_MAX_TITLE_SETS;
        ssize_t  best   = -1;
        int      i;

        for (i = 1; i <= nb_vts; i++) {
            dvd_file_t *f = DVDOpenFile(dc->dvd, i, DVD_READ_TITLE_VOBS);
            ssize_t     size;

            if (!f) {
                if (!vmg)
                    break; /* title sets are numbered contiguously */
                continue;
            }
            size = DVDFileSize(f);
            DVDCloseFile(f);
            if (size > best) {
                best = size;
                vts  = i;
            }
        }
    }

    if (vmg)
        ifoClose(vmg);

    if (vts < 1) {
        av_log(h, AV_LOG_ERROR, "no usable title set found\n");
        return AVERROR(EIO);
    }
    return vts;
}

static int dvdread_close(URLContext *h)
{
    DVDReadContext *dc = h->priv_data;

    if (dc->file)
        DVDCloseFile(dc->file);
    dc->file = NULL;
    if (dc->dvd)
        DVDClose(dc->dvd);
    dc->dvd = NULL;

    return 0;
}

static int dvdread_open(URLContext *h, const char *path, int flags)
{
    DVDReadContext *dc = h->priv_data;
    const char     *diskname;
    ssize_t         blocks;
    int             vts;

    if (flags & AVIO_FLAG_WRITE)
        return AVERROR(ENOSYS);

    if (!av_strstart(path, DVDREAD_PROTO_PREFIX, &diskname))
        return AVERROR(EINVAL);
    /* accept dvdread:PATH, dvdread://PATH and dvdread:///ABSOLUTE/PATH */
    if (!strncmp(diskname, "//", 2))
        diskname += 2;
    if (!*diskname) {
        av_log(h, AV_LOG_ERROR, "missing disc path\n");
        return AVERROR(EINVAL);
    }

    dc->dvd = DVDOpen(diskname);
    if (!dc->dvd) {
        av_log(h, AV_LOG_ERROR, "unable to open DVD structure at '%s'\n", diskname);
        return AVERROR(EIO);
    }

    vts = dvdread_resolve_vts(h);
    if (vts < 0) {
        dvdread_close(h);
        return vts;
    }

    dc->file = DVDOpenFile(dc->dvd, vts, DVD_READ_TITLE_VOBS);
    if (!dc->file) {
        av_log(h, AV_LOG_ERROR, "unable to open title VOBs of title set %d\n", vts);
        dvdread_close(h);
        return AVERROR(EIO);
    }

    blocks = DVDFileSize(dc->file);
    if (blocks <= 0) {
        av_log(h, AV_LOG_ERROR, "title set %d has no title VOB data\n", vts);
        dvdread_close(h);
        return AVERROR(EIO);
    }

    dc->blocks   = blocks;
    dc->pos      = 0;
    dc->cache_lb = -1;

    av_log(h, AV_LOG_INFO, "reading title set %d (%" PRId64 " bytes)\n",
           vts, (int64_t)blocks * DVD_BLOCK_SIZE);

    return 0;
}

static int dvdread_read(URLContext *h, unsigned char *buf, int size)
{
    DVDReadContext *dc    = h->priv_data;
    int64_t         total = (int64_t)dc->blocks * DVD_BLOCK_SIZE;
    int             done  = 0;

    if (dc->pos >= total)
        return AVERROR_EOF;
    size = FFMIN(size, total - dc->pos);

    while (done < size) {
        int lb  = dc->pos / DVD_BLOCK_SIZE;
        int off = dc->pos % DVD_BLOCK_SIZE;

        if (!off && size - done >= DVD_BLOCK_SIZE) {
            /* aligned: read whole blocks straight into the caller's buffer */
            int     nb_blocks = FFMIN((size - done) / DVD_BLOCK_SIZE,
                                      (int)(dc->blocks - lb));
            ssize_t ret       = DVDReadBlocks(dc->file, lb, nb_blocks, buf + done);

            if (ret <= 0)
                return done ? done : AVERROR(EIO);
            done    += ret * DVD_BLOCK_SIZE;
            dc->pos += ret * DVD_BLOCK_SIZE;
        } else {
            /* unaligned head or tail: serve from the single-block cache */
            int n;

            if (dc->cache_lb != lb) {
                if (DVDReadBlocks(dc->file, lb, 1, dc->cache) != 1)
                    return done ? done : AVERROR(EIO);
                dc->cache_lb = lb;
            }
            n = FFMIN(size - done, DVD_BLOCK_SIZE - off);
            memcpy(buf + done, dc->cache + off, n);
            done    += n;
            dc->pos += n;
        }
    }

    return done;
}

static int64_t dvdread_seek(URLContext *h, int64_t pos, int whence)
{
    DVDReadContext *dc    = h->priv_data;
    int64_t         total = (int64_t)dc->blocks * DVD_BLOCK_SIZE;

    switch (whence) {
    case SEEK_SET:
        break;
    case SEEK_CUR:
        pos += dc->pos;
        break;
    case SEEK_END:
        pos += total;
        break;
    case AVSEEK_SIZE:
        return total;
    default:
        return AVERROR(EINVAL);
    }

    if (pos < 0 || pos > total)
        return AVERROR(EINVAL);

    dc->pos = pos;
    return pos;
}

const URLProtocol ff_dvdread_protocol = {
    .name            = "dvdread",
    .url_open        = dvdread_open,
    .url_read        = dvdread_read,
    .url_seek        = dvdread_seek,
    .url_close       = dvdread_close,
    .priv_data_size  = sizeof(DVDReadContext),
    .priv_data_class = &dvdread_context_class,
};
