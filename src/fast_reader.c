#include "fast_reader.h"
#include "stage_prof.h"

/* zlib-ng (native zng_* API) for the streaming plain-gzip inflate path: its
 * SIMD inflate is ~2x zlib's, and it keeps the streaming inflate() model that a
 * FASTQ-scale reader needs (the decompressed stream is far larger than RAM, so a
 * one-shot decompressor like libdeflate cannot be used here). The native API is
 * used (not --zlib-compat) so these symbols never collide with the system zlib
 * that htslib and the legacy gzFile reader still use. libdeflate stays on the
 * bounded, independent BGZF blocks below. */
#include <zlib-ng.h>
#include <libdeflate.h>

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FR_CIN_CAP   (1u << 18)   /* 256 KiB compressed/raw input buffer      */
#define FR_BOUT_CAP  (1u << 16)   /* 64 KiB: max BGZF block uncompressed size */
#define FR_HDR_PEEK  18           /* bytes needed to reach BGZF BSIZE         */

struct fast_reader {
    int          fd;
    fr_format_t  fmt;

    /* Compressed/raw input buffer; [cin_pos, cin_len) is valid, unconsumed. */
    unsigned char *cin;
    size_t         cin_len, cin_pos, cin_cap;
    int            eof_in;

    /* gzip */
    zng_stream zs;
    int      zs_active;
    int      gz_member_open;   /* a gzip member is mid-decode (set on Z_OK, cleared at Z_STREAM_END) */

    /* bgzf */
    struct libdeflate_decompressor *ld;
    unsigned char *bout;                 /* one decompressed block */
    size_t         bout_len, bout_pos, bout_cap;
    int            done;                 /* bgzf stream exhausted */
};

/* --------------------------------------------------------------------- */

fr_format_t fr_detect(const unsigned char *h, size_t n)
{
    if (n < 2) return FR_PLAIN;                      /* empty / tiny -> plain */
    if (h[0] == 0x1f && h[1] == 0x8b) {              /* gzip family */
        if (n >= FR_HDR_PEEK && h[2] == 0x08 && (h[3] & 0x04)) {  /* deflate + FEXTRA */
            unsigned xlen = (unsigned)h[10] | ((unsigned)h[11] << 8);
            /* BGZF: BC subfield (htslib always writes it first, at offset 12) */
            if (xlen >= 6 && h[12] == 'B' && h[13] == 'C') return FR_BGZF;
        }
        return FR_GZIP;
    }
    if (h[0] == 0x42 && h[1] == 0x5a) return FR_UNSUPPORTED;        /* bzip2 "BZ" */
    if (n >= 4 && h[0] == 0x28 && h[1] == 0xb5 && h[2] == 0x2f && h[3] == 0xfd)
        return FR_UNSUPPORTED;                                     /* zstd */
    if (n >= 6 && h[0] == 0xfd && h[1] == 0x37 && h[2] == 0x7a && h[3] == 0x58 && h[4] == 0x5a)
        return FR_UNSUPPORTED;                                     /* xz */
    return FR_PLAIN;
}

/* Refill the input buffer from fd, preserving unconsumed bytes at the front.
 * Returns 0 on success, -1 on read error. */
static int fr_fill(fast_reader_t *fr)
{
    if (fr->cin_pos > 0) {
        size_t keep = fr->cin_len - fr->cin_pos;
        if (keep > 0) memmove(fr->cin, fr->cin + fr->cin_pos, keep);
        fr->cin_len = keep;
        fr->cin_pos = 0;
    }
    double _t0 = sp_enabled() ? sp_wall() : 0.0;
    while (fr->cin_len < fr->cin_cap && !fr->eof_in) {
        ssize_t r = read(fr->fd, fr->cin + fr->cin_len, fr->cin_cap - fr->cin_len);
        if (r < 0) { if (errno == EINTR) continue; if (sp_enabled()) sp_read_add(SP_READ_DISKWAIT, sp_wall()-_t0); return -1; }
        if (r == 0) { fr->eof_in = 1; break; }
        fr->cin_len += (size_t)r;
        if (sp_enabled()) sp_read_bytes((long)r, 0);   /* fd (on-disk) bytes */
    }
    if (sp_enabled()) sp_read_add(SP_READ_DISKWAIT, sp_wall() - _t0);
    return 0;
}

/* --------------------------------------------------------------------- */

fast_reader_t *fast_reader_dopen(int fd, const char **err)
{
    /* Ownership of fd transfers to the reader at open time (see fast_reader.h),
     * so every error path must close it -- otherwise non-legacy callers leak the
     * descriptor when open fails. */
#define FR_DOPEN_FAIL(msg) do { if (err) *err = (msg); if (fd >= 0) close(fd); return NULL; } while (0)
    unsigned char hdr[FR_HDR_PEEK];
    size_t got = 0;
    while (got < sizeof hdr) {
        ssize_t r = read(fd, hdr + got, sizeof hdr - got);
        if (r < 0) { if (errno == EINTR) continue; FR_DOPEN_FAIL("read failed"); }
        if (r == 0) break;
        got += (size_t)r;
    }

    fr_format_t fmt = fr_detect(hdr, got);
    if (fmt == FR_UNSUPPORTED) {
        FR_DOPEN_FAIL("unsupported compression format; bwa-mem3 supports plain, gzip, and BGZF");
    }

    fast_reader_t *fr = (fast_reader_t *)calloc(1, sizeof *fr);
    if (!fr) FR_DOPEN_FAIL("out of memory");
    fr->fd = fd;
    fr->fmt = fmt;
    fr->cin_cap = FR_CIN_CAP;
    fr->cin = (unsigned char *)malloc(fr->cin_cap);
    if (!fr->cin) { free(fr); FR_DOPEN_FAIL("out of memory"); }

    /* Prime the sniffed header bytes so nothing is lost on non-seekable fds. */
    memcpy(fr->cin, hdr, got);
    fr->cin_len = got;
    fr->cin_pos = 0;

    if (fmt == FR_GZIP) {
        fr->zs.zalloc = Z_NULL; fr->zs.zfree = Z_NULL; fr->zs.opaque = Z_NULL;
        fr->zs.next_in = Z_NULL; fr->zs.avail_in = 0;
        if (zng_inflateInit2(&fr->zs, 15 + 16) != Z_OK) {   /* 15+16 = gzip wrapper */
            free(fr->cin); free(fr); FR_DOPEN_FAIL("inflateInit2 failed");
        }
        fr->zs_active = 1;
    } else if (fmt == FR_BGZF) {
        fr->ld = libdeflate_alloc_decompressor();
        fr->bout_cap = FR_BOUT_CAP;
        fr->bout = (unsigned char *)malloc(fr->bout_cap);
        if (!fr->ld || !fr->bout) {
            if (fr->ld) libdeflate_free_decompressor(fr->ld);
            free(fr->bout); free(fr->cin); free(fr);
            FR_DOPEN_FAIL("out of memory");
        }
    }
#undef FR_DOPEN_FAIL
    return fr;
}

/* --------------------------------------------------------------------- */

static int fr_read_plain(fast_reader_t *fr, unsigned char *buf, int len)
{
    int out = 0;
    if (fr->cin_pos < fr->cin_len) {              /* serve primed/buffered bytes */
        size_t avail = fr->cin_len - fr->cin_pos;
        int take = avail < (size_t)len ? (int)avail : len;
        memcpy(buf, fr->cin + fr->cin_pos, (size_t)take);
        fr->cin_pos += (size_t)take;
        out += take;
    }
    double _t0 = sp_enabled() ? sp_wall() : 0.0;
    while (out < len && !fr->eof_in) {
        ssize_t r = read(fr->fd, buf + out, len - out);
        if (r < 0) { if (errno == EINTR) continue; if (sp_enabled()) sp_read_add(SP_READ_DISKWAIT, sp_wall()-_t0); return -1; }
        if (r == 0) { fr->eof_in = 1; break; }
        out += (int)r;
        if (sp_enabled()) sp_read_bytes((long)r, 0);   /* fd (on-disk) bytes */
    }
    if (sp_enabled()) sp_read_add(SP_READ_DISKWAIT, sp_wall() - _t0);
    return out;
}

static int fr_read_gzip(fast_reader_t *fr, unsigned char *buf, int len)
{
    fr->zs.next_out = buf;
    fr->zs.avail_out = (uInt)len;
    while (fr->zs.avail_out > 0) {
        if (fr->zs.avail_in == 0) {
            if (fr_fill(fr) < 0) return -1;
            if (fr->cin_len - fr->cin_pos == 0) {           /* no more input */
                if (fr->gz_member_open) return -1;          /* member truncated mid-stream */
                break;
            }
            fr->zs.next_in = fr->cin + fr->cin_pos;
            fr->zs.avail_in = (uInt)(fr->cin_len - fr->cin_pos);
        }
        double _td = sp_enabled() ? sp_wall() : 0.0;
        int ret = zng_inflate(&fr->zs, Z_NO_FLUSH);
        if (sp_enabled()) sp_read_add(SP_READ_DECOMPRESS, sp_wall() - _td);
        fr->cin_pos = fr->cin_len - fr->zs.avail_in;       /* track consumed */
        if (ret == Z_STREAM_END) {                          /* member done */
            fr->gz_member_open = 0;
            if (zng_inflateReset(&fr->zs) != Z_OK) return -1;
            if (fr->zs.avail_in == 0 && fr->eof_in) break;  /* nothing follows */
            continue;                                       /* next member */
        }
        if (ret == Z_BUF_ERROR) {                           /* needs more input */
            if (fr->eof_in && fr->zs.avail_in == 0) {
                if (fr->gz_member_open) return -1;              /* truncated: EOF mid-member */
                break;
            }
            continue;
        }
        if (ret != Z_OK) return -1;                         /* hard error */
        fr->gz_member_open = 1;                              /* Z_OK: member mid-decode */
    }
    return len - (int)fr->zs.avail_out;
}

/* Decode the next BGZF block into fr->bout, skipping empty (EOF-marker) blocks
 * and stopping at clean EOF. Returns 0 on success/EOF (fr->done set on EOF),
 * -1 on a decode/format error. */
static int fr_next_bgzf_block(fast_reader_t *fr)
{
    for (;;) {
        fr->bout_len = fr->bout_pos = 0;
        if (fr_fill(fr) < 0) return -1;
        size_t avail = fr->cin_len - fr->cin_pos;
        if (avail == 0) { fr->done = 1; return 0; }         /* clean EOF */
        if (avail < FR_HDR_PEEK) return -1;                  /* truncated header */

        const unsigned char *p = fr->cin + fr->cin_pos;
        if (!(p[0] == 0x1f && p[1] == 0x8b && p[2] == 0x08 && (p[3] & 0x04))) return -1;

        unsigned xlen = (unsigned)p[10] | ((unsigned)p[11] << 8);
        unsigned end = 12 + xlen;
        /* xlen is attacker-controlled (up to 65535); the extra field must lie
         * entirely within the bytes actually read, or the scan below could
         * over-read past `avail` into uninitialized / out-of-bounds memory. */
        if (end > avail) return -1;                          /* extra field overruns input */
        unsigned bsize = 0; int found = 0;
        unsigned off = 12;
        while (off + 4 <= end) {                             /* scan extra for BC */
            unsigned slen = (unsigned)p[off + 2] | ((unsigned)p[off + 3] << 8);
            /* The BC subfield carries a 2-byte payload at p[off+4..off+5];
             * confirm those bytes are within the extra field before reading. */
            if (p[off] == 'B' && p[off + 1] == 'C' && slen == 2 && off + 6 <= end) {
                bsize = (unsigned)p[off + 4] | ((unsigned)p[off + 5] << 8);
                found = 1; break;
            }
            off += 4 + slen;
        }
        if (!found) return -1;

        size_t block_len = (size_t)bsize + 1;
        if (block_len < (size_t)(12 + xlen + 8)) return -1;  /* malformed */
        if (avail < block_len) return -1;                    /* truncated block */

        size_t cdata_off = 12 + xlen;
        size_t cdata_len = block_len - cdata_off - 8;
        uint32_t crc = (uint32_t)p[block_len - 8]        | ((uint32_t)p[block_len - 7] << 8) |
                       ((uint32_t)p[block_len - 6] << 16) | ((uint32_t)p[block_len - 5] << 24);
        uint32_t isize = (uint32_t)p[block_len - 4]        | ((uint32_t)p[block_len - 3] << 8) |
                         ((uint32_t)p[block_len - 2] << 16) | ((uint32_t)p[block_len - 1] << 24);

        fr->cin_pos += block_len;                            /* consume the block */

        if (isize == 0) continue;                            /* EOF marker / empty block */
        if (isize > fr->bout_cap) return -1;                 /* >64 KiB: not BGZF */

        size_t actual = 0;
        double _td = sp_enabled() ? sp_wall() : 0.0;
        enum libdeflate_result r =
            libdeflate_deflate_decompress(fr->ld, p + cdata_off, cdata_len,
                                          fr->bout, isize, &actual);
        if (sp_enabled()) sp_read_add(SP_READ_DECOMPRESS, sp_wall() - _td);
        if (r != LIBDEFLATE_SUCCESS || actual != isize) return -1;
        if (libdeflate_crc32(0, fr->bout, isize) != crc) return -1;

        fr->bout_len = isize;
        fr->bout_pos = 0;
        if (sp_enabled()) sp_read_bytes(0, 1);   /* one decoded BGZF block */
        return 0;
    }
}

static int fr_read_bgzf(fast_reader_t *fr, unsigned char *buf, int len)
{
    int out = 0;
    while (out < len) {
        if (fr->bout_pos < fr->bout_len) {
            size_t avail = fr->bout_len - fr->bout_pos;
            int take = avail < (size_t)(len - out) ? (int)avail : (len - out);
            memcpy(buf + out, fr->bout + fr->bout_pos, (size_t)take);
            fr->bout_pos += (size_t)take;
            out += take;
            continue;
        }
        if (fr->done) break;
        if (fr_next_bgzf_block(fr) < 0) return -1;
        if (fr->bout_len == 0 && fr->done) break;
    }
    return out;
}

int fast_reader_read(fast_reader_t *fr, void *buf, int len)
{
    if (len <= 0) return 0;
    switch (fr->fmt) {
        case FR_PLAIN: return fr_read_plain(fr, (unsigned char *)buf, len);
        case FR_GZIP:  return fr_read_gzip(fr, (unsigned char *)buf, len);
        case FR_BGZF:  return fr_read_bgzf(fr, (unsigned char *)buf, len);
        default:       return -1;
    }
}

fr_format_t fast_reader_format(const fast_reader_t *fr) { return fr->fmt; }

void fast_reader_close(fast_reader_t *fr)
{
    if (!fr) return;
    if (fr->zs_active) zng_inflateEnd(&fr->zs);
    if (fr->ld) libdeflate_free_decompressor(fr->ld);
    free(fr->bout);
    free(fr->cin);
    if (fr->fd >= 0) close(fr->fd);
    free(fr);
}
