/* Regression: bseq_read_fast must abort loudly, not return a short clean batch
 * followed by clean EOF, when a gzip input is truncated mid-member.
 *
 * This exercises the full chain end to end:
 *   fast_reader (fr_read_gzip returns -1 once a member is left mid-decode at
 *   source EOF) -> fr_fastq (maps a codec error to -2) -> bseq_read_fast
 *   (err_fatal on a -2). A silent EOF anywhere in that chain drops every read
 *   after the corruption point and exits 0 -- exactly the failure this guards.
 *
 * Two shapes, since bseq_read_fast has separate err_fatal call sites per mate:
 *   - SE: a single truncated input hits the 1st-input decode-error path.
 *   - PE: a full 1st input paired with a truncated 2nd input hits the
 *     2nd-input decode-error path (g1 == 1 while g2 == -2).
 * Each is a forked child that must die by SIGABRT (err_fatal, stubbed to
 * abort()) rather than reach a clean n == 0 EOF.
 *
 * Build (standalone -- only the reader TUs, no bwa-mem3 link):
 *   cc -O2 -I src test/bseq_read_truncated_gzip_test.c \
 *      src/fr_fastq.c src/fast_reader.c src/fast_reader_bseq.c \
 *      -lz -ldeflate -o bseq_read_truncated_gzip_test
 */
#include "bwa.h"
#include "fast_reader.h"
#include "fast_reader_bseq.h"
#include "gz_test_util.h"   /* gz_member(), GZ_TRUNCATE_TAIL_DROP */
#include "read_arena.h"

#include <assert.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* stage_prof + err_fatal stubs so this links without the bwa-mem3 object graph.
 * err_fatal is stubbed to abort() so the expected death is an observable
 * SIGABRT rather than a plain exit. */
int    sp_enabled(void) { return 0; }
double sp_wall(void) { return 0.0; }
void   sp_read_add(int which, double s) { (void)which; (void)s; }
void   sp_read_bytes(long a, long b) { (void)a; (void)b; }
void   sp_read_get(double *d, double *c, double *p)
{ if (d) *d = 0; if (c) *c = 0; if (p) *p = 0; }
void   err_fatal(const char *h, const char *fmt, ...)
{ (void)h; (void)fmt; fprintf(stderr, "err_fatal\n"); abort(); }

/* Write `len` bytes of `buf` to a fresh temp file; returns a heap path the
 * caller must unlink()+free(). */
static char *write_temp(const unsigned char *buf, size_t len)
{
    char *path = strdup("/tmp/bseq_trunc_XXXXXX");
    int fd = mkstemp(path);
    assert(fd >= 0);
    size_t off = 0;
    while (off < len) { ssize_t w = write(fd, buf + off, len - off); assert(w > 0); off += (size_t)w; }
    close(fd);
    return path;
}

/* Run bseq_read_fast to completion (or abort) over the given input file(s) in a
 * forked child, and return 1 iff the child died by SIGABRT (the err_fatal path).
 * A clean n == 0 EOF, or any other exit, is a failure. */
static int child_aborts(const char *p1, const char *p2)
{
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        int fd1 = open(p1, O_RDONLY);
        int fd2 = p2 ? open(p2, O_RDONLY) : -1;
        if (fd1 < 0 || (p2 && fd2 < 0)) _exit(3);
        const char *err = NULL;
        fast_reader_t *fr1 = fast_reader_dopen(fd1, &err);
        fast_reader_t *fr2 = p2 ? fast_reader_dopen(fd2, &err) : NULL;
        if (!fr1 || (p2 && !fr2)) _exit(2);   /* headers intact: dopen must succeed */
        void *ks1 = fast_kseq_init(fr1);
        void *ks2 = p2 ? fast_kseq_init(fr2) : NULL;
        read_arena_t *arena = NULL;
        for (;;) {
            int nn = 0; int64_t s = 0;
            bseq1_t *seqs = bseq_read_fast(1 << 16, &nn, ks1, ks2, &s, &arena, 0);
            if (nn == 0) { free(seqs); _exit(0); }   /* clean EOF WITHOUT aborting == the bug */
            free(seqs);                              /* comment==NULL (copy_comment=0); arena carried */
        }
    }
    int st = 0;
    assert(waitpid(pid, &st, 0) == pid);
    if (WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT) return 1;
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0)
        fprintf(stderr, "  reached clean EOF instead of aborting\n");
    else if (WIFEXITED(st))
        fprintf(stderr, "  child exited %d (test setup, not the death path)\n", WEXITSTATUS(st));
    else
        fprintf(stderr, "  child died by signal %d (not SIGABRT)\n", WIFSIGNALED(st) ? WTERMSIG(st) : -1);
    return 0;
}

/* Build a valid multi-record FASTQ payload (heap; caller frees). */
static unsigned char *make_fastq(size_t *out_n)
{
    size_t cap = 1u << 20;
    unsigned char *fq = (unsigned char *)malloc(cap);
    assert(fq != NULL);
    size_t n = 0;
    for (int i = 0; n + 128 < cap; i++)
        n += (size_t)snprintf((char *)fq + n, cap - n,
                              "@r%d\nACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIII\n", i);
    *out_n = n;
    return fq;
}

/* gzip `fq` into a fresh heap buffer; returns it and sets *out_len. */
static unsigned char *gzip_all(const unsigned char *fq, size_t n, size_t *out_len)
{
    size_t gcap = n + (n >> 1) + 4096;
    unsigned char *gz = (unsigned char *)malloc(gcap);
    assert(gz != NULL);
    size_t gl = gz_member(fq, n, gz, gcap);
    assert(gl > 64);
    *out_len = gl;
    return gz;
}

int main(void)
{
    int fails = 0;
    size_t n; unsigned char *fq = make_fastq(&n);
    size_t gl; unsigned char *gz = gzip_all(fq, n, &gl);

    /* SE: single truncated input -> 1st-input decode-error err_fatal. */
    {
        char *p = write_temp(gz, gl - GZ_TRUNCATE_TAIL_DROP);
        int ok = child_aborts(p, NULL);
        fprintf(stderr, "%s: SE truncated gzip -> abort\n", ok ? "PASS" : "FAIL");
        if (!ok) fails = 1;
        unlink(p); free(p);
    }

    /* PE: full 1st input + truncated 2nd input -> 2nd-input decode-error
     * err_fatal (g1 == 1 while g2 == -2). */
    {
        char *p1 = write_temp(gz, gl);                              /* full, valid */
        char *p2 = write_temp(gz, gl - GZ_TRUNCATE_TAIL_DROP);      /* truncated 2nd mate */
        int ok = child_aborts(p1, p2);
        fprintf(stderr, "%s: PE truncated 2nd input -> abort\n", ok ? "PASS" : "FAIL");
        if (!ok) fails = 1;
        unlink(p1); free(p1);
        unlink(p2); free(p2);
    }

    free(fq); free(gz);
    return fails;
}
