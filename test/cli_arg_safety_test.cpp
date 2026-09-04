/* SPDX-License-Identifier: MIT */

/* test/cli_arg_safety_test.cpp — memory-safety / correctness contract tests for
 * two CLI-input helpers in src/bwa.cpp that the byte-identity gate never
 * exercises (it runs plain `mem ref r1 r2` with no -R/-H and via the fast
 * reader):
 *
 *   - bwa_escape (reached through the public bwa_insert_header): a trailing
 *     backslash in -R/-H text must not walk past the NUL terminator and copy /
 *     write out of bounds. Asserted against hardcoded expected strings (an
 *     independent oracle -- NOT a comparison against another call of the same
 *     helper). The out-of-bounds read is inside libbwa, so it is caught
 *     reliably only when libbwa itself is built with ASan: CI runs this binary
 *     plain on every matrix row (the oracle still catches a wrong-length copy)
 *     AND re-runs it against the ASan build on the canonical row.
 *
 *   - bseq_classify(0, NULL, ...): the `-p` interleaved path can hand it an
 *     empty batch (input ending on a cohort-slice boundary); it must not
 *     dereference seqs[0]. Asserted to return empty, not crash.
 *
 * No committed fixture; inputs are string literals.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bwa.h"

static int fails = 0;

/* bwa_escape runs inside bwa_insert_header on the freshly duplicated line, so
 * the returned header is the escaped form of `line`. */
static void check_escape(const char *line, const char *expected)
{
    char *h = bwa_insert_header(line, NULL);
    if (h == NULL || strcmp(h, expected) != 0) {
        fprintf(stderr, "FAIL: bwa_insert_header(\"%s\"): got \"%s\", expected \"%s\"\n",
                line, h ? h : "(null)", expected);
        ++fails;
    }
    free(h);
}

int main(void)
{
    /* Trailing backslash: the escape is incomplete, so it is dropped and the
     * scan stops at the NUL rather than reading past it. "\\t" is a real
     * escape (-> TAB); the final "\\" is the lone trailing backslash. */
    check_escape("@PG\\tID:x\\", "@PG\tID:x");

    /* A lone backslash as the entire escapable tail of the line. */
    check_escape("@CO\\", "@CO");

    /* Unknown escape (\x) is dropped (unchanged behavior); the point here is
     * that an unknown escape followed by end-of-string is also in bounds. */
    check_escape("@PG\\tID:\\x", "@PG\tID:");

    /* Recognized escapes still work and nothing trails them. */
    check_escape("@PG\\tA\\nB\\rC\\\\D", "@PG\tA\nB\rC\\D");

    /* bseq_classify must tolerate an empty batch (the -p cohort-boundary case)
     * instead of dereferencing seqs[0] of a NULL array. */
    {
        int m[2] = {-1, -1};
        bseq1_t *sep[2] = {(bseq1_t *)0x1, (bseq1_t *)0x1};
        bseq_classify(0, NULL, m, sep);
        if (m[0] != 0 || m[1] != 0 || sep[0] != NULL || sep[1] != NULL) {
            fprintf(stderr, "FAIL: bseq_classify(0, NULL): m={%d,%d} sep={%p,%p}\n",
                    m[0], m[1], (void *)sep[0], (void *)sep[1]);
            ++fails;
        }
    }

    if (fails) {
        fprintf(stderr, "cli_arg_safety_test: %d failure(s)\n", fails);
        return 1;
    }
    printf("PASS: cli_arg_safety_test\n");
    return 0;
}
