/* Coverage for bseq_read_fast's comment handling (the `-C` / copy_comment gate).
 *
 * fr_fastq_diff_test already proves the PARSER extracts comments byte-identically
 * to kseq, but it never drives the ADAPTER (bseq_read_fast / fr_rec_to_bseq1),
 * whose contract is subtler: bseq1_t.comment is populated only when the caller
 * asked for it AND the record actually had one —
 *     s->comment = (copy_comment && r->comment_l) ? dup(...) : NULL
 * (fast_reader_bseq.c). This test pins that gate across both the copy_comment
 * values, present/absent comments, and single- vs paired-end.
 *
 * Build (standalone — only the reader TUs, no bwa-mem3 link):
 *   cc -I src test/bseq_read_comment_copy_test.c \
 *      src/fast_reader_bseq.c src/fr_fastq.c src/fast_reader.c \
 *      -lz -ldeflate -o bseq_read_comment_copy_test
 */
#include "bwa.h"
#include "fast_reader.h"
#include "fast_reader_bseq.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* stage_prof + err_fatal stubs so this links without the bwa-mem3 object graph. */
int    sp_enabled(void) { return 0; }
double sp_wall(void) { return 0.0; }
void   sp_read_add(int which, double seconds) { (void)which; (void)seconds; }
void   sp_read_bytes(long fd_bytes, long bgzf_blocks) { (void)fd_bytes; (void)bgzf_blocks; }
void   sp_read_get(double *disk, double *decomp, double *parse)
{ if (disk) *disk = 0; if (decomp) *decomp = 0; if (parse) *parse = 0; }
void   err_fatal(const char *header, const char *fmt, ...)
{ (void)header; (void)fmt; fprintf(stderr, "err_fatal\n"); abort(); }

static int g_fail = 0;

/* Two records: r0 carries a comment, r1 has none (bare name). */
static const char *PAYLOAD =
    "@r0 the comment\nACGT\n+\nIIII\n"
    "@r1\nTTGG\n+\nFFFF\n";

static char *write_payload(void)
{
    char *path = strdup("/tmp/bseq_comment_XXXXXX");
    int fd = mkstemp(path);
    assert(fd >= 0);
    size_t len = strlen(PAYLOAD), off = 0;
    while (off < len) { ssize_t w = write(fd, PAYLOAD + off, len - off); assert(w > 0); off += (size_t)w; }
    close(fd);
    return path;
}

/* Read the whole (tiny) payload as one batch and check r0/r1's comment against
 * expectation. `expect_r0` is the string r0's comment must equal, or NULL if it
 * must be absent. r1 never has a comment, so it must always be NULL. */
static void check(const char *label, int copy_comment, int paired, const char *expect_r0)
{
    char *p1 = write_payload();
    char *p2 = paired ? write_payload() : NULL;
    int fd1 = open(p1, O_RDONLY);
    int fd2 = paired ? open(p2, O_RDONLY) : -1;
    assert(fd1 >= 0 && (!paired || fd2 >= 0));

    const char *err = NULL;
    fast_reader_t *fr1 = fast_reader_dopen(fd1, &err);
    fast_reader_t *fr2 = paired ? fast_reader_dopen(fd2, &err) : NULL;
    void *ks1 = fast_kseq_init(fr1);
    void *ks2 = paired ? fast_kseq_init(fr2) : NULL;

    int n = 0;
    int64_t size = 0;
    read_arena_t *arena = NULL;
    bseq1_t *seqs = bseq_read_fast(1 << 20, &n, ks1, ks2, &size, &arena, copy_comment);

    /* SE: 2 records in file order (r0, r1). PE: 4 records, each iteration
     * emitting mate1 then mate2 of the same record — seqs[0],seqs[1] are both
     * files' r0 (comment), seqs[2],seqs[3] both files' r1 (none). So the source
     * record index is i/2 for PE and i for SE; only record 0 carries a comment. */
    int ok = (n == (paired ? 4 : 2));
    if (!ok) fprintf(stderr, "    %s: record count %d (expected %d)\n", label, n, paired ? 4 : 2);
    for (int i = 0; i < n && ok; i++) {
        int rec_index = paired ? i / 2 : i;
        const char *want = (rec_index == 0) ? expect_r0 : NULL;   /* r0 has a comment, r1 doesn't */
        const char *got = seqs[i].comment;
        int match = (want == NULL) ? (got == NULL) : (got != NULL && strcmp(got, want) == 0);
        if (!match) {
            ok = 0;
            fprintf(stderr, "    %s: rec %d comment='%s' (expected '%s')\n",
                    label, i, got ? got : "(null)", want ? want : "(null)");
        }
    }
    fprintf(stderr, "%s: %s\n", ok ? "ok  " : "FAIL", label);
    if (!ok) g_fail = 1;

    for (int i = 0; i < n; i++) free(seqs[i].comment);
    free(seqs);
    if (arena) read_arena_destroy(arena);
    fast_kseq_destroy(ks1);
    if (ks2) fast_kseq_destroy(ks2);
    fast_reader_close(fr1);
    if (fr2) fast_reader_close(fr2);
    unlink(p1); free(p1);
    if (p2) { unlink(p2); free(p2); }
}

/* Drive the readno-suffix trim (fr_trim_readno_len) through the adapter, which
 * fr_fastq_diff_test never does (the parser doesn't trim). Boundaries: a bare
 * trailing "/<digit>" is stripped ("a/1" -> "a"), but a name too short to have
 * anything before the suffix is kept ("/1", length 2, l>2 guard fails), and a
 * "/<non-digit>" is kept ("r/x"). */
static void check_trim(void)
{
    static const char *P =
        "@a/1\nAC\n+\nII\n"
        "@/1\nGT\n+\nFF\n"
        "@r/x\nTT\n+\nGG\n";
    static const char *EXPECT[3] = { "a", "/1", "r/x" };

    char *path = strdup("/tmp/bseq_trim_XXXXXX");
    int fd = mkstemp(path); assert(fd >= 0);
    size_t len = strlen(P), off = 0;
    while (off < len) { ssize_t w = write(fd, P + off, len - off); assert(w > 0); off += (size_t)w; }
    close(fd);

    int rfd = open(path, O_RDONLY); assert(rfd >= 0);
    const char *err = NULL;
    fast_reader_t *fr = fast_reader_dopen(rfd, &err);
    void *ks = fast_kseq_init(fr);
    int n = 0; int64_t size = 0;
    read_arena_t *arena = NULL;
    bseq1_t *seqs = bseq_read_fast(1 << 20, &n, ks, NULL, &size, &arena, 0);

    int ok = (n == 3);
    if (!ok) fprintf(stderr, "    trim: record count %d (expected 3)\n", n);
    for (int i = 0; i < n && ok; i++) {
        if (strcmp(seqs[i].name, EXPECT[i]) != 0) {
            ok = 0;
            fprintf(stderr, "    trim: rec %d name='%s' (expected '%s')\n",
                    i, seqs[i].name, EXPECT[i]);
        }
    }
    fprintf(stderr, "%s: readno-suffix trim boundaries\n", ok ? "ok  " : "FAIL");
    if (!ok) g_fail = 1;

    for (int i = 0; i < n; i++) free(seqs[i].comment);
    free(seqs);
    if (arena) read_arena_destroy(arena);
    fast_kseq_destroy(ks);
    fast_reader_close(fr);
    unlink(path); free(path);
}

int main(void)
{
    /* copy_comment=1 → present comment copied through; absent stays NULL. */
    check("SE copy_comment=1", 1, 0, "the comment");
    check("PE copy_comment=1", 1, 1, "the comment");
    /* copy_comment=0 → comment dropped to NULL even when the record had one. */
    check("SE copy_comment=0", 0, 0, NULL);
    check("PE copy_comment=0", 0, 1, NULL);

    check_trim();

    fprintf(stderr, "\n%s\n", g_fail ? "COMMENT-COPY TEST FAILED" : "ALL COMMENT-COPY CASES OK");
    return g_fail ? 1 : 0;
}
