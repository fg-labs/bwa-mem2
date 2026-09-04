/* SPDX-License-Identifier: MIT */

/* test/pwrite_all_status_test.cpp — contract test for pwrite_all_status()
 * (src/io_utils.h), the non-fatal write-the-whole-buffer core shared by the
 * fatal pwrite_all() wrapper and the OpenMP-region caller in index_prelude.cpp.
 *
 * The distinguishing behavior — a status return instead of err_fatal()->exit()
 * — is the reason the function exists, so it is what this pins:
 *   - a good write returns 0 and lands the exact bytes at the offset;
 *   - a write to a bad fd returns a positive errno (not a crash);
 *   - the multi-chunk path (small max_chunk) still writes the whole buffer and
 *     bumps the chunk counter more than once.
 *
 * No committed fixture; the buffer is generated and written to a temp file.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include "io_utils.h"

static int fails = 0;

int main(void)
{
    char tmpl[] = "/tmp/pwrite_status_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) { perror("mkstemp"); return 1; }

    /* Good write: whole buffer at a non-zero offset, status 0, bytes land. */
    const char msg[] = "hello pwrite_all_status";
    int rc = pwrite_all_status(fd, msg, sizeof msg, 8);
    if (rc != 0) { fprintf(stderr, "FAIL: good write returned %d (expected 0)\n", rc); ++fails; }
    char back[sizeof msg] = {0};
    if (pread(fd, back, sizeof msg, 8) != (ssize_t)sizeof msg || memcmp(back, msg, sizeof msg) != 0) {
        fprintf(stderr, "FAIL: bytes at offset 8 do not match what was written\n"); ++fails;
    }

    /* Multi-chunk: force the clamp with a tiny max_chunk; whole buffer still
     * lands and the chunk counter shows it was split. */
    unsigned char big[1000];
    for (size_t i = 0; i < sizeof big; ++i) big[i] = (unsigned char)(i & 0xff);
    rc = pwrite_all_status(fd, big, sizeof big, 0, /*max_chunk=*/64);
    if (rc != 0) { fprintf(stderr, "FAIL: multi-chunk write returned %d (expected 0)\n", rc); ++fails; }
    if (pwrite_all_chunk_count() < 2) {
        fprintf(stderr, "FAIL: multi-chunk write did not split (chunk_count=%lu)\n",
                pwrite_all_chunk_count()); ++fails;
    }
    unsigned char rb[1000] = {0};
    if (pread(fd, rb, sizeof big, 0) != (ssize_t)sizeof big || memcmp(rb, big, sizeof big) != 0) {
        fprintf(stderr, "FAIL: multi-chunk bytes do not match\n"); ++fails;
    }

    close(fd);

    /* Bad fd: a positive errno is returned rather than a crash / exit. */
    rc = pwrite_all_status(fd, msg, sizeof msg, 0);   /* fd is now closed -> EBADF */
    if (rc <= 0) { fprintf(stderr, "FAIL: write to closed fd returned %d (expected positive errno)\n", rc); ++fails; }

    unlink(tmpl);
    if (fails) { fprintf(stderr, "pwrite_all_status_test: %d failure(s)\n", fails); return 1; }
    printf("PASS: pwrite_all_status_test\n");
    return 0;
}
