/* test/klib_alloc_fail_test.c -- the vendored klib headers kbtree.h, kseq.h
 * and ksort.h must abort loudly, not NULL-deref, when a backing allocation
 * fails.
 *
 * Each of these headers allocates (a B-tree node, a stream buffer, a merge
 * scratch, a grown line/sequence buffer) and writes through the result on the
 * next line. Historically the NULL check after each allocation was a plain
 * assert(); this build never defines NDEBUG, so that was live in practice, but
 * a guard whose absence is a NULL write has to hold by construction rather
 * than by build flag, and the headers are vendored (they include nothing from
 * the surrounding project), so each now carries its own self-contained
 * out-of-memory abort in the shape kvec.h established. This test forces one
 * allocation to fail per case and asserts that contract: SIGABRT carrying an
 * out-of-memory diagnostic; never a segfault, never silent continuation.
 *
 * The failure injection is FAIL-ONCE, not sticky: each armed flag trips the
 * next matching allocation and immediately disarms. A sticky flag would let a
 * *downstream* guard on the same path fire the identical SIGABRT+OOM after the
 * intended guard was removed, so the test would stay green against the very
 * bug it exists to catch. With fail-once, deleting the guard under test makes
 * that one allocation return NULL with nothing to catch it, and the child dies
 * some other way (SIGSEGV, or a clean exit) -- which this test reports as a
 * failure, because it requires SIGABRT specifically.
 *
 * calloc/malloc and realloc are armed by separate flags so a case can target a
 * grow-path realloc guard without the setup calloc/malloc tripping first.
 *
 * Cases cover the reachable guard classes across all three headers: a calloc
 * guard (kbtree kb_init, kbtree split-path node, kseq kseq_init), a malloc
 * guard (ksort merge scratch), and a realloc guard (kseq line-buffer growth in
 * ks_getuntil2). Guards in klib functions no bwa-mem3 path exercises
 * (kb_destroy/kb_reset/traverse iterators) are out of scope.
 *
 * Standalone (header-only; links no bwa-mem3 objects) and forked, like
 * kvec_alloc_fail_test: the abort is contained so the parent can inspect how
 * the child died, and the signal is the discriminator.
 *
 * The allocator seam: test_calloc()/test_malloc()/test_realloc() are defined
 * before the `#define calloc`/`malloc`/`realloc` lines, so their own libc calls
 * bind to the real allocator; the headers, included after, route through the
 * seam. No interposition and no linker --wrap, so it is portable across the
 * macOS and Linux CI rows.
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* Fail-once injection. Armed immediately before the allocation that must fail;
 * the seam disarms it as it returns NULL, so exactly one allocation of that
 * kind fails and every other takes the real path. calloc/malloc share a flag;
 * realloc has its own so a grow-path case skips the setup allocations. */
static int g_fail_alloc = 0;
static int g_fail_realloc = 0;

static void *test_calloc(size_t n, size_t sz)
{
    if (g_fail_alloc) { g_fail_alloc = 0; return NULL; }
    return calloc(n, sz);
}

static void *test_malloc(size_t n)
{
    if (g_fail_alloc) { g_fail_alloc = 0; return NULL; }
    return malloc(n);
}

static void *test_realloc(void *p, size_t n)
{
    if (g_fail_realloc) { g_fail_realloc = 0; return NULL; }
    return realloc(p, n);
}

#define calloc(n, sz) test_calloc((n), (sz))
#define malloc(n) test_malloc((n))
#define realloc(p, n) test_realloc((p), (n))

#include "kbtree.h"
#include "kseq.h"
#include "ksort.h"

KBTREE_INIT(kbint, int, kb_generic_cmp)
KSORT_INIT_GENERIC(int)

/* A synthetic FASTA record served to kseq one buffer-fill at a time. The
 * sequence line is long enough (well past the 256-byte initial seq buffer) to
 * force the ks_getuntil2 line-buffer growth realloc. Set by the realloc case. */
static const char *g_stream = NULL;
static size_t g_stream_off = 0;

static int fake_read(int fd, void *buf, unsigned len)
{
    (void)fd;
    if (g_stream == NULL) return 0;
    size_t remaining = strlen(g_stream) - g_stream_off;
    size_t n = remaining < (size_t)len ? remaining : (size_t)len;
    memcpy(buf, g_stream + g_stream_off, n);
    g_stream_off += n;
    return (int)n;
}
KSEQ_INIT(int, fake_read)

enum klib_case {
    CASE_KBTREE_INIT,
    CASE_KBTREE_SPLIT,
    CASE_KSEQ_INIT,
    CASE_KSORT_MERGE,
    CASE_KSEQ_REALLOC,
};

/* Drive one header until its armed allocation fails. On the guarded headers
 * this aborts and never returns; without the guard it NULL-derefs (or, for a
 * store the compiler forwards past the NULL, falls through to _exit). */
static void child_force_oom(int wfd, enum klib_case which)
{
    /* Route this child's stderr into the parent's pipe so the parent can read
     * the diagnostic the abort path prints. */
    dup2(wfd, STDERR_FILENO);

    switch (which) {
    case CASE_KBTREE_INIT: {
        g_fail_alloc = 1;
        kbtree_t(kbint) *t = kb_init(kbint, KB_DEFAULT_SIZE); /* calloc -> NULL -> abort */
        (void)t;
        break;
    }
    case CASE_KBTREE_SPLIT: {
        /* Fill the root leaf to capacity on the real allocator, then arm the
         * failure so the split on the next insert (a node calloc, the first
         * allocation after the fill) sees NULL. Node capacity is 2t-1 with t
         * derived from KB_DEFAULT_SIZE and the key size, exactly as kb_init
         * computes it. */
        kbtree_t(kbint) *t = kb_init(kbint, KB_DEFAULT_SIZE);
        if (t == NULL) _exit(3);
        int cap = 2 * t->t - 1;
        for (int k = 0; k < cap; ++k) kb_put(kbint, t, k);
        g_fail_alloc = 1;
        kb_put(kbint, t, cap);                                /* split calloc -> NULL -> abort */
        break;
    }
    case CASE_KSEQ_INIT: {
        g_fail_alloc = 1;
        kseq_t *s = kseq_init(0);                             /* calloc -> NULL -> abort */
        (void)s;
        break;
    }
    case CASE_KSORT_MERGE: {
        int a[8] = { 5, 3, 7, 1, 8, 2, 6, 4 };
        g_fail_alloc = 1;
        ks_mergesort(int, 8, a, NULL);                        /* malloc -> NULL -> abort */
        break;
    }
    case CASE_KSEQ_REALLOC: {
        /* A record whose sequence line far exceeds the 256-byte initial seq
         * buffer, so reading it grows the line buffer via realloc in
         * ks_getuntil2. Arm only the realloc flag: kseq_init/ks_init and the
         * initial seq malloc succeed, and the growth realloc is the first (and
         * only) realloc, so it takes the failure. */
        static char rec[420];
        rec[0] = '>'; rec[1] = 's'; rec[2] = '\n';
        memset(rec + 3, 'A', 400);
        rec[403] = '\n'; rec[404] = '\0';
        g_stream = rec;
        g_stream_off = 0;
        kseq_t *s = kseq_init(0);
        g_fail_realloc = 1;
        kseq_read(s);                                         /* line-grow realloc -> NULL -> abort */
        (void)s;
        break;
    }
    }

    /* Reached only if the armed allocation did NOT abort. The exit code is
     * immaterial: the parent's discriminator is a SIGABRT carrying the OOM
     * diagnostic, which only the guarded header produces. */
    _exit(0);
}

/* Fork a child that drives `which` until the armed allocation fails, and
 * assert it terminated the guarded way: SIGABRT carrying the OOM diagnostic.
 * Returns 0 on pass, 1 on failure (with a reason on stderr). */
static int run_case(const char *name, enum klib_case which)
{
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return 1; }

    /* Flush before forking so a prior case's PASS line is not re-emitted by
     * the child on abort. */
    fflush(NULL);

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        close(pipefd[0]);
        child_force_oom(pipefd[1], which);
        _exit(0);              /* unreachable on both guarded and unguarded code */
    }
    close(pipefd[1]);

    /* Drain the child's stderr so a full pipe can never deadlock it. */
    char buf[512];
    size_t got = 0;
    ssize_t r;
    while (got < sizeof(buf) - 1 &&
           (r = read(pipefd[0], buf + got, sizeof(buf) - 1 - got)) > 0) {
        got += (size_t)r;
    }
    buf[got] = '\0';
    close(pipefd[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return 1; }

    if (!WIFSIGNALED(status)) {
        fprintf(stderr, "FAIL[%s]: child exited normally (status %d); the OOM "
                        "path did not abort -- allocation guard missing\n",
                name, WEXITSTATUS(status));
        return 1;
    }
    if (WTERMSIG(status) != SIGABRT) {
        fprintf(stderr, "FAIL[%s]: child died from signal %d, expected SIGABRT "
                        "(%d). SIGSEGV here means the header NULL-dereferenced "
                        "instead of guarding the failed allocation.\n",
                name, WTERMSIG(status), SIGABRT);
        return 1;
    }
    if (strstr(buf, "out of memory") == NULL) {
        fprintf(stderr, "FAIL[%s]: child aborted (SIGABRT) but without the "
                        "expected out-of-memory diagnostic; its stderr was:\n%s\n",
                name, buf);
        return 1;
    }

    printf("PASS[%s]: aborts with an OOM diagnostic when the allocation fails\n", name);
    return 0;
}

int main(void)
{
    /* Prove the allocator seam is in effect before relying on it: with each
     * flag armed, the matching interposed allocator must hand back NULL, and
     * fail-once must leave the next call unaffected. */
    g_fail_alloc = 1;
    void *probe_c = calloc(1, 8);
    void *probe_c2 = calloc(1, 8);   /* fail-once: this one must succeed */
    g_fail_alloc = 1;
    void *probe_m = malloc(8);
    g_fail_realloc = 1;
    void *probe_r = realloc(NULL, 8);
    if (probe_c != NULL || probe_m != NULL || probe_r != NULL || probe_c2 == NULL) {
        fprintf(stderr, "FAIL: allocator seam not in effect as expected "
                        "(armed alloc returned non-NULL, or fail-once did not "
                        "disarm)\n");
        free(probe_c); free(probe_c2); free(probe_m); free(probe_r);
        return 1;
    }
    free(probe_c2);
    g_fail_alloc = 0;
    g_fail_realloc = 0;

    int rc = 0;
    rc |= run_case("kbtree kb_init",      CASE_KBTREE_INIT);
    rc |= run_case("kbtree kb_put split", CASE_KBTREE_SPLIT);
    rc |= run_case("kseq kseq_init",      CASE_KSEQ_INIT);
    rc |= run_case("ksort ks_mergesort",  CASE_KSORT_MERGE);
    rc |= run_case("kseq line realloc",   CASE_KSEQ_REALLOC);
    return rc;
}
