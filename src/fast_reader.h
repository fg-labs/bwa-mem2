/* fast_reader: content-detecting, gzread-compatible FASTQ byte source.
 *
 * Sniffs the input codec by magic bytes and serves decompressed bytes via a
 * gzread-style API, so the existing kseq parser can sit on top unchanged:
 *
 *   - plain  : raw passthrough
 *   - gzip   : zlib streaming inflate, spanning concatenated members
 *   - bgzf   : libdeflate, one block at a time (the parallel-ready seam)
 *
 * libdeflate's decompress API is one-shot (no streaming), so it is used only
 * for the bounded <=64 KB BGZF blocks; vanilla gzip stays on zlib's streaming
 * inflate. See docs/design for the rationale and the per-arch measurements.
 */
#ifndef FAST_READER_H
#define FAST_READER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FR_PLAIN       =  0,
    FR_GZIP        =  1,
    FR_BGZF        =  2,
    FR_UNSUPPORTED = -1  /* recognized but unsupported: bzip2 / xz / zstd */
} fr_format_t;

/* Classify an input by its leading bytes. Up to 18 bytes are needed to
 * distinguish BGZF from plain gzip (the BC extra subfield carrying BSIZE);
 * fewer bytes is fine. Input that does not begin with a known magic classifies
 * as FR_PLAIN, but a short buffer that starts with the gzip magic (1f 8b) yet
 * is shorter than 18 bytes classifies as FR_GZIP (the BGZF distinction is what
 * needs the full 18 bytes), and empty/1-byte input is FR_PLAIN. */
fr_format_t fr_detect(const unsigned char *hdr, size_t n);

typedef struct fast_reader fast_reader_t;

/* Open a reader over an already-open file descriptor. Takes ownership of fd
 * (closed by fast_reader_close). Sniffs the codec and primes the consumed
 * header bytes internally, so non-seekable inputs (pipes/stdin) work without
 * seeking. On error returns NULL and, if err != NULL, sets *err to a static
 * message (e.g. an unsupported codec). */
fast_reader_t *fast_reader_dopen(int fd, const char **err);

/* gzread-compatible: read up to len decompressed bytes into buf. Returns the
 * number of bytes produced (0 at clean EOF) or -1 on a decode error.
 * Transparently spans concatenated gzip members and BGZF blocks. */
int fast_reader_read(fast_reader_t *fr, void *buf, int len);

fr_format_t fast_reader_format(const fast_reader_t *fr);

/* Close the reader and the underlying fd, and free all buffers. */
void fast_reader_close(fast_reader_t *fr);

#ifdef __cplusplus
}
#endif

#endif /* FAST_READER_H */
