/* Standalone correctness self-test for fast_reader.
 *
 * Generates plain, gzip, multi-member gzip, and BGZF encodings of a known
 * payload (programmatically — no committed fixtures), reads each back through
 * fast_reader, and asserts the decoded bytes equal the original. Also checks
 * fr_detect classification and the unsupported-codec path.
 *
 * Build (standalone, no bwa-mem3 link needed):
 *   cc -I src -I/opt/homebrew/include test/fast_reader_selftest.c src/fast_reader.c \
 *      -L/opt/homebrew/lib -lz -ldeflate -o fast_reader_selftest
 */
#include "fast_reader.h"
#include "gz_test_util.h"   /* gz_member(), GZ_TRUNCATE_TAIL_DROP */

#include <zlib.h>
#include <libdeflate.h>

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* fast_reader.c is instrumented with stage_prof hooks, but they are compiled
 * out by default (no -DSTAGE_PROF): stage_prof.h then provides no-op inlines, so
 * fast_reader.c is self-contained and this standalone test needs no sp_* stubs. */

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (msg)); g_fail = 1; } \
    else         { fprintf(stderr, "ok:   %s\n", (msg)); } \
} while (0)

/* Allocate or abort the self-test: every buffer is dereferenced immediately, so
 * an unchecked malloc would crash on OOM instead of failing controllably. */
#define XMALLOC(ptr, n) do { (ptr) = malloc((n)); assert((ptr) != NULL); } while (0)

/* Build a multi-record FASTQ-ish payload large enough to span buffer refills. */
static unsigned char *make_payload(size_t *out_n)
{
    size_t cap = 4u << 20;          /* 4 MiB */
    unsigned char *buf;
    XMALLOC(buf, cap);
    size_t n = 0;
    for (int i = 0; n + 256 < cap; i++) {
        n += (size_t)snprintf((char *)buf + n, cap - n,
            "@read%d/1 some comment %d\n"
            "ACGTACGTNNNNacgtACGT%d\n"
            "+\n"
            "IIIIIIIIIIIIIIIIIIII\n", i, i, i);
    }
    *out_n = n;
    return buf;
}

static void write_file(const char *path, const unsigned char *p, size_t n)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    assert(fd >= 0);
    size_t off = 0;
    while (off < n) { ssize_t w = write(fd, p + off, n - off); assert(w > 0); off += (size_t)w; }
    close(fd);
}

/* One BGZF block (n <= 65536). Returns block length written to out. */
static size_t bgzf_block(struct libdeflate_compressor *c,
                         const unsigned char *in, size_t n, unsigned char *out)
{
    size_t bound = libdeflate_deflate_compress_bound(c, n);
    unsigned char *cbuf;
    XMALLOC(cbuf, bound);
    size_t clen = libdeflate_deflate_compress(c, in, n, cbuf, bound);
    assert(clen > 0 || n == 0);
    if (clen == 0) {                 /* empty input: emit an empty stored block */
        cbuf[0] = 0x03; cbuf[1] = 0x00; clen = 2;
    }
    size_t block_len = 12 + 6 + clen + 8;
    uint16_t bsize = (uint16_t)(block_len - 1);
    unsigned char *p = out;
    p[0]=0x1f; p[1]=0x8b; p[2]=0x08; p[3]=0x04;
    p[4]=p[5]=p[6]=p[7]=0; p[8]=0; p[9]=0xff;
    p[10]=6; p[11]=0;                          /* XLEN = 6 */
    p[12]='B'; p[13]='C'; p[14]=2; p[15]=0;
    p[16]=bsize & 0xff; p[17]=(bsize >> 8) & 0xff;
    memcpy(p + 18, cbuf, clen);
    uint32_t crc = libdeflate_crc32(0, in, n);
    uint32_t isize = (uint32_t)n;
    unsigned char *t = p + 18 + clen;
    t[0]=crc&0xff; t[1]=(crc>>8)&0xff; t[2]=(crc>>16)&0xff; t[3]=(crc>>24)&0xff;
    t[4]=isize&0xff; t[5]=(isize>>8)&0xff; t[6]=(isize>>16)&0xff; t[7]=(isize>>24)&0xff;
    free(cbuf);
    return block_len;
}

/* Read a whole file through fast_reader and compare to expected. */
static int roundtrip(const char *path, const unsigned char *expect, size_t n,
                     fr_format_t expect_fmt)
{
    int fd = open(path, O_RDONLY); assert(fd >= 0);
    const char *err = NULL;
    fast_reader_t *fr = fast_reader_dopen(fd, &err);
    if (!fr) { fprintf(stderr, "  dopen failed: %s\n", err ? err : "?"); return 0; }
    if (fast_reader_format(fr) != expect_fmt) { fprintf(stderr, "  wrong format\n"); fast_reader_close(fr); return 0; }
    /* Each fast_reader_read may write up to 65536 bytes at got + total before
     * the overrun guard runs, so the buffer must hold n plus one full chunk
     * of slack past the expected length (otherwise a final over-length read
     * heap-overflows before the guard can catch it). */
    unsigned char *got;
    XMALLOC(got, n + 65536);
    size_t total = 0; int ok = 1;
    for (;;) {
        int r = fast_reader_read(fr, got + total, 65536);
        if (r < 0) { fprintf(stderr, "  read error\n"); ok = 0; break; }
        if (r == 0) break;
        total += (size_t)r;
        if (total > n + 65536) { fprintf(stderr, "  overrun\n"); ok = 0; break; }
    }
    if (ok && (total != n || memcmp(got, expect, n) != 0)) {
        fprintf(stderr, "  mismatch: got %zu vs %zu bytes\n", total, n);
        ok = 0;
    }
    free(got);
    fast_reader_close(fr);
    return ok;
}

/* Read a whole file through fast_reader and report whether a decode error
 * (fast_reader_read < 0) was observed before clean EOF. This is the contract a
 * truncated compressed stream must honor: fast_reader.h promises "-1 on a decode
 * error", never a short read that looks like clean EOF. */
static int reports_read_error(const char *path)
{
    int fd = open(path, O_RDONLY); assert(fd >= 0);
    const char *err = NULL;
    fast_reader_t *fr = fast_reader_dopen(fd, &err);
    if (!fr) return 0;   /* header intact: failure must surface at read time (dopen already closed fd) */
    unsigned char buf[65536];
    int saw_error = 0;
    for (;;) {
        int r = fast_reader_read(fr, buf, sizeof buf);
        if (r < 0) { saw_error = 1; break; }
        if (r == 0) break;               /* clean EOF */
    }
    fast_reader_close(fr);
    return saw_error;
}

int main(void)
{
    /* --- fr_detect --- */
    unsigned char gz[18]  = {0x1f,0x8b,0x08,0x00};
    unsigned char bg[18]  = {0x1f,0x8b,0x08,0x04,0,0,0,0,0,0xff,6,0,'B','C',2,0,0x12,0};
    unsigned char bz[2]   = {0x42,0x5a};
    unsigned char zs4[4]  = {0x28,0xb5,0x2f,0xfd};
    CHECK(fr_detect((unsigned char*)"",0) == FR_PLAIN,        "fr_detect: empty -> plain");
    CHECK(fr_detect((unsigned char*)"@SEQ",4) == FR_PLAIN,    "fr_detect: text -> plain");
    CHECK(fr_detect(gz,18) == FR_GZIP,                        "fr_detect: gzip");
    CHECK(fr_detect(bg,18) == FR_BGZF,                        "fr_detect: bgzf");
    CHECK(fr_detect(bz,2) == FR_UNSUPPORTED,                  "fr_detect: bzip2 -> unsupported");
    CHECK(fr_detect(zs4,4) == FR_UNSUPPORTED,                 "fr_detect: zstd -> unsupported");
    /* Documented boundary: the BGZF-vs-gzip distinction needs the full 18 bytes,
     * so a short buffer that starts with the gzip magic still classifies GZIP;
     * a lone 0x1f (1 byte, below the n>=2 magic check) is PLAIN. */
    unsigned char gz2[2] = {0x1f,0x8b};
    unsigned char one[1] = {0x1f};
    CHECK(fr_detect(gz2,2) == FR_GZIP,                        "fr_detect: 2-byte gzip magic -> gzip");
    CHECK(fr_detect(one,1) == FR_PLAIN,                       "fr_detect: 1-byte 0x1f -> plain");

    size_t n; unsigned char *pay = make_payload(&n);
    fprintf(stderr, "payload: %zu bytes\n", n);

    /* --- plain --- */
    write_file("/tmp/fr_plain.fq", pay, n);
    CHECK(roundtrip("/tmp/fr_plain.fq", pay, n, FR_PLAIN), "roundtrip: plain");

    /* --- single-member gzip --- */
    {
        size_t cap = n + (n >> 1) + 4096;
        unsigned char *gzb;
        XMALLOC(gzb, cap);
        size_t gl = gz_member(pay, n, gzb, cap);
        write_file("/tmp/fr_one.fq.gz", gzb, gl);
        CHECK(roundtrip("/tmp/fr_one.fq.gz", pay, n, FR_GZIP), "roundtrip: single-member gzip");

        /* Truncated single-member gzip: dropping the tail (final deflate block
         * bytes + the 8-byte CRC32/ISIZE trailer) means the member never reaches
         * Z_STREAM_END. fast_reader_read must surface -1, not a short clean read
         * that the FASTQ layer above would mistake for end-of-file. */
        assert(gl > 64);
        write_file("/tmp/fr_trunc.fq.gz", gzb, gl - GZ_TRUNCATE_TAIL_DROP);
        CHECK(reports_read_error("/tmp/fr_trunc.fq.gz"),
              "truncated gzip -> fast_reader_read returns -1");
        free(gzb);
    }

    /* --- BGZF truncated INSIDE a block: fr_next_bgzf_block's `avail < block_len`
     * check must surface -1. NOTE this only covers mid-block truncation; a BGZF
     * stream truncated at a whole-block boundary (missing the trailing empty
     * EOF-marker block) is currently accepted as clean EOF, since the codec does
     * not require the marker -- a separate, known gap tracked outside this
     * change, not exercised here. --- */
    {
        struct libdeflate_compressor *c = libdeflate_alloc_compressor(6);
        size_t cap = n + (n >> 1) + 65536;
        unsigned char *bgz;
        XMALLOC(bgz, cap);
        size_t chunk = n > 60000 ? 60000 : n;
        size_t bl = bgzf_block(c, pay, chunk, bgz);
        assert(bl > 16);
        write_file("/tmp/fr_trunc.bgzf.gz", bgz, bl - 8);   /* cut inside the block */
        CHECK(reports_read_error("/tmp/fr_trunc.bgzf.gz"),
              "mid-block-truncated bgzf -> fast_reader_read returns -1");
        libdeflate_free_compressor(c);
        free(bgz);
    }

    /* --- multi-member gzip (cat a.gz b.gz) --- */
    {
        size_t half = n / 2;
        size_t cap = n + (n >> 1) + 8192;
        unsigned char *a, *b;
        XMALLOC(a, cap);
        XMALLOC(b, cap);
        size_t la = gz_member(pay, half, a, cap);
        size_t lb = gz_member(pay + half, n - half, b, cap);
        unsigned char *cat;
        XMALLOC(cat, la + lb);
        memcpy(cat, a, la); memcpy(cat + la, b, lb);
        write_file("/tmp/fr_multi.fq.gz", cat, la + lb);
        CHECK(roundtrip("/tmp/fr_multi.fq.gz", pay, n, FR_GZIP), "roundtrip: multi-member gzip");
        free(a); free(b); free(cat);
    }

    /* --- BGZF (multiple blocks + EOF marker) --- */
    {
        struct libdeflate_compressor *c = libdeflate_alloc_compressor(6);
        size_t cap = n + (n >> 1) + 65536;
        unsigned char *bgz;
        XMALLOC(bgz, cap);
        size_t off = 0, src = 0;
        while (src < n) {
            size_t chunk = n - src; if (chunk > 60000) chunk = 60000;   /* < 64 KiB */
            off += bgzf_block(c, pay + src, chunk, bgz + off);
            src += chunk;
        }
        off += bgzf_block(c, NULL, 0, bgz + off);                       /* EOF marker */
        write_file("/tmp/fr_bgzf.fq.gz", bgz, off);
        CHECK(roundtrip("/tmp/fr_bgzf.fq.gz", pay, n, FR_BGZF), "roundtrip: bgzf");
        libdeflate_free_compressor(c);
        free(bgz);
    }

    /* --- unsupported codec through fast_reader_dopen: reject, set err, close
     * fd. fr_detect classification is unit-tested above; this drives the dopen
     * plumbing around it (the FR_DOPEN_FAIL path). --- */
    {
        unsigned char bz2[8] = {0x42,0x5a,0x68,0x39,0x31,0x41,0x59,0x26};   /* "BZh91AY&" */
        write_file("/tmp/fr_unsupported.bz2", bz2, sizeof bz2);
        int ufd = open("/tmp/fr_unsupported.bz2", O_RDONLY); assert(ufd >= 0);
        const char *uerr = NULL;
        fast_reader_t *ufr = fast_reader_dopen(ufd, &uerr);
        CHECK(ufr == NULL && uerr != NULL, "dopen: unsupported codec -> NULL + err set");
        if (ufr) fast_reader_close(ufr);   /* only if the reject regressed */
    }

    free(pay);
    fprintf(stderr, g_fail ? "\nSELFTEST FAILED\n" : "\nSELFTEST PASSED\n");
    return g_fail;
}
