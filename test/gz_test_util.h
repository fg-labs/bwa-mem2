/* Shared test helper: compress a buffer into one complete gzip member via zlib.
 * Used by fast_reader_selftest.c and bseq_read_truncated_gzip_test.c so the
 * (non-trivial) deflate setup lives in one place. Header-only static inline:
 * each standalone test TU is its own binary, so there is no ODR concern. */
#ifndef GZ_TEST_UTIL_H
#define GZ_TEST_UTIL_H

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <zlib.h>

/* One complete gzip member (15+16 window = gzip wrapper). Aborts (assert) if the
 * whole input did not deflate into `cap`. Returns the compressed length. */
static inline size_t gz_member(const unsigned char *in, size_t n,
                               unsigned char *out, size_t cap)
{
    z_stream zs;
    memset(&zs, 0, sizeof zs);
    deflateInit2(&zs, 6, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
    zs.next_in = (Bytef *)in; zs.avail_in = (uInt)n;
    zs.next_out = out; zs.avail_out = (uInt)cap;
    int r = deflate(&zs, Z_FINISH); assert(r == Z_STREAM_END);
    size_t len = cap - zs.avail_out;
    deflateEnd(&zs);
    return len;
}

/* Bytes to drop from a complete gzip member to truncate it mid-stream: enough to
 * remove the 8-byte CRC32/ISIZE trailer and part of the final deflate block so
 * inflate never reaches Z_STREAM_END. */
#define GZ_TRUNCATE_TAIL_DROP 40

#endif /* GZ_TEST_UTIL_H */
