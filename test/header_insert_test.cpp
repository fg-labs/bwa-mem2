// Regression test for bwa_insert_header_file.
//
// bwa_insert_header_file reads all @-prefixed lines from a file, assembles
// them into a single buffer, and calls bwa_insert_header once instead of
// once per line — turning the -H ingestion path from O(n^2) into O(n).
// This test proves the batched path produces byte-identical output to the
// per-line baseline across the cases the old code supported.
//
// Usage:
//   header_insert_test          # runs all cases, exits 0 on success

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "bwa.h"

static char *per_line_baseline(const std::vector<std::string> &lines, char *hdr)
{
    // Mirrors the pre-patch fastmap.cpp loop: strip trailing '\n' from each
    // line then call bwa_insert_header. Any line not starting with '@' is
    // silently skipped by bwa_insert_header itself.
    for (const auto &raw : lines) {
        std::string line = raw;
        if (!line.empty() && line.back() == '\n') line.pop_back();
        hdr = bwa_insert_header(line.c_str(), hdr);
    }
    return hdr;
}

static std::string write_tmp(const std::vector<std::string> &lines)
{
    char tmpl[] = "/tmp/bwa_hdr_test_XXXXXX";
    int fd = mkstemp(tmpl);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "w");
    assert(fp != nullptr);
    for (const auto &l : lines) fputs(l.c_str(), fp);
    fclose(fp);
    return std::string(tmpl);
}

static void run_case(const char *name,
                     const std::vector<std::string> &lines,
                     const char *seed_hdr,
                     const char *expected_literal = nullptr)
{
    std::string path = write_tmp(lines);

    // Baseline: walk lines through per-line bwa_insert_header.
    char *expected = seed_hdr ? strdup(seed_hdr) : nullptr;
    expected = per_line_baseline(lines, expected);

    // Batched path: open file and hand it to bwa_insert_header_file.
    char *actual = seed_hdr ? strdup(seed_hdr) : nullptr;
    FILE *fp = fopen(path.c_str(), "r");
    assert(fp != nullptr);
    actual = bwa_insert_header_file(fp, actual);
    fclose(fp);
    unlink(path.c_str());

    // Independent oracle: when the caller supplies the exact expected output,
    // assert the batched path against that literal, not only against the
    // per-line baseline. Both paths run bwa_escape, so a self-consistency check
    // alone can pass against a shared escaping bug (e.g. lone-backslash
    // handling); the literal pins the true result.
    if (expected_literal != nullptr) {
        if (actual == nullptr || strcmp(actual, expected_literal) != 0) {
            fprintf(stderr, "FAIL: %s: literal oracle mismatch\n  expected: %s\n  actual:   %s\n",
                    name, expected_literal, actual ? actual : "(null)");
            exit(1);
        }
    }

    // Both null is a valid outcome (empty / no-@ file, no seed).
    if (expected == nullptr && actual == nullptr) {
        fprintf(stderr, "OK:   %s (both null)\n", name);
        return;
    }
    if (expected == nullptr || actual == nullptr) {
        fprintf(stderr, "FAIL: %s: null mismatch (expected=%p actual=%p)\n",
                name, (void *)expected, (void *)actual);
        exit(1);
    }
    if (strcmp(expected, actual) != 0) {
        fprintf(stderr, "FAIL: %s\n  expected: %s\n  actual:   %s\n",
                name, expected, actual);
        exit(1);
    }
    fprintf(stderr, "OK:   %s\n", name);
    free(expected);
    free(actual);
}

int main(int, char **)
{
    // Case 1: multiple @SQ lines, no prior header.
    run_case("multi-SQ",
             {"@HD\tVN:1.6\tSO:coordinate\n",
              "@SQ\tSN:chr1\tLN:1000\n",
              "@SQ\tSN:chr2\tLN:2000\n",
              "@SQ\tSN:chrM\tLN:16569\n"},
             nullptr);

    // Case 2: seed hdr non-null (simulates a prior -H @RG line), then read
    // file. Batched path must produce the same concatenation as per-line.
    run_case("seeded",
             {"@SQ\tSN:chr1\tLN:1000\n",
              "@SQ\tSN:chr2\tLN:2000\n"},
             "@RG\tID:foo\tSM:bar");

    // Case 3: non-@ lines interleaved — baseline skips them (bwa_insert_header
    // early-returns), batched must skip them too.
    run_case("mixed-non-at",
             {"@HD\tVN:1.6\n",
              "not a header line\n",
              "@SQ\tSN:chr1\tLN:1000\n",
              "# comment\n",
              "@SQ\tSN:chr2\tLN:2000\n"},
             nullptr);

    // Case 4: escape sequences — bwa_insert_header calls bwa_escape, which
    // translates \\t, \\n, \\r, \\\\. Running it once on the concatenation
    // must match running it per line.
    run_case("escapes",
             {"@CO\tfield1\\tfield2\\nwith\\\\backslash\n",
              "@SQ\tSN:chr1\tLN:1000\n"},
             nullptr);

    // Case 4b: a retained line ending in a lone trailing backslash, followed
    // by another @ line. bwa_escape must not be able to treat the '\n'
    // separator the batched path inserts between two retained lines as the
    // second half of an escape sequence -- that would both mistranslate/drop
    // the separator AND merge two SAM header records into one line. The
    // per-line baseline never has this failure mode: it escapes and joins one
    // line at a time, so bwa_escape never sees a separator it didn't itself
    // just insert as the previous call's boundary.
    // Independent oracle: bwa_escape drops a lone trailing backslash, so the
    // retained @CO line loses its final '\' and the two records join with a
    // single '\n' -- pinned literally so a shared escaping bug can't hide behind
    // the baseline (which runs the same bwa_escape).
    run_case("trailing-backslash-then-at-line",
             {"@CO\tfield\\\n",
              "@SQ\tSN:chr1\tLN:1000\n"},
             nullptr,
             "@CO\tfield\n@SQ\tSN:chr1\tLN:1000");

    // Case 5: empty file — calloc(1, 0) edge case. Must leave hdr unchanged
    // (null in / null out).
    run_case("empty-file", {}, nullptr);

    // Case 6: empty file but seed hdr non-null — must return the seed string
    // unchanged.
    run_case("empty-file-seeded", {}, "@RG\tID:only");

    // Case 7: file with only non-@ lines — baseline produces hdr unchanged,
    // batched must too.
    run_case("no-at-lines",
             {"not a header\n",
              "also not a header\n"},
             "@RG\tID:seed");

    // Case 8: single @-line without trailing newline on last line — SAM
    // tools commonly emit a trailing newline, but verify we don't blow up
    // if the last line lacks one.
    run_case("no-trailing-newline",
             {"@HD\tVN:1.6\n",
              "@SQ\tSN:chr1\tLN:1000"},
             nullptr);

    // The fgets budget is sizeof(chunk)-1; a line filling the buffer exactly is AT
    // the budget, not over it, and must be accepted. Such a line fills `chunk`
    // without a trailing newline, but feof() is not yet set, so the naive
    // fill-and-reject check used to reject a line that was exactly at the limit.
    const size_t kBudget = 0x10000 - 1;  // must match `char chunk[0x10000]` in bwa.cpp

    // Case 8b: an @-line whose content is EXACTLY the budget with no trailing
    // newline at end-of-file. Baseline (per-line bwa_insert_header) accepts it;
    // the batched path must too rather than rejecting a line at the limit.
    {
        std::string at_budget = "@CO\t";
        at_budget.append(kBudget - at_budget.size(), 'x');  // total length == kBudget
        assert(at_budget.size() == kBudget);
        run_case("at-budget-no-newline", {at_budget}, nullptr);
    }

    // Case 8c: the same at-budget line but properly terminated by a newline (and
    // followed by end-of-file). fgets still returns the content chunk full and
    // newline-less, deferring the '\n' to the next read -- which carries no further
    // line content, so the line is accepted.
    {
        std::string at_budget = "@CO\t";
        at_budget.append(kBudget - at_budget.size(), 'x');
        assert(at_budget.size() == kBudget);
        run_case("at-budget-with-newline", {at_budget + "\n"}, nullptr);
    }

    // Case 9: a single @-line longer than the 64 KiB fgets budget. The pre-
    // patch loop asserted on buf[i-1] == '\n' and aborted; the batched path
    // must also fail loudly rather than silently truncate. We fork because
    // bwa_insert_header_file calls err_fatal -> exit(EXIT_FAILURE).
    {
        std::string huge_line = "@CO\t";
        huge_line.append(70000, 'x');
        huge_line.push_back('\n');
        std::string path = write_tmp({huge_line});
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            // Silence the expected stderr from err_fatal so the test output
            // stays clean.
            FILE *devnull = freopen("/dev/null", "w", stderr);
            (void) devnull;
            FILE *fp = fopen(path.c_str(), "r");
            assert(fp != nullptr);
            char *out = bwa_insert_header_file(fp, nullptr);
            // Should not return — err_fatal must exit before we get here.
            (void) out;
            fclose(fp);
            _exit(0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        unlink(path.c_str());
        if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE) {
            fprintf(stderr,
                    "FAIL: oversize-line: expected exit(EXIT_FAILURE), got "
                    "exited=%d status=%d signaled=%d signal=%d\n",
                    WIFEXITED(status), WEXITSTATUS(status),
                    WIFSIGNALED(status), WTERMSIG(status));
            exit(1);
        }
        fprintf(stderr, "OK:   oversize-line\n");
    }

    // Case 9b: an @-line exactly ONE byte over the budget with no trailing
    // newline. This pins the boundary: the first chunk fills the buffer, the next
    // read yields a single further content byte, and the batched path must reject.
    // We fork because bwa_insert_header_file calls err_fatal -> exit(EXIT_FAILURE).
    {
        std::string over = "@CO\t";
        over.append((0x10000 - 1) + 1 - over.size(), 'x');  // total length == kBudget + 1
        assert(over.size() == (size_t)(0x10000 - 1) + 1);
        std::string path = write_tmp({over});
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            FILE *devnull = freopen("/dev/null", "w", stderr);
            (void) devnull;
            FILE *fp = fopen(path.c_str(), "r");
            assert(fp != nullptr);
            char *out = bwa_insert_header_file(fp, nullptr);
            (void) out;  // Should not return -- err_fatal must exit first.
            fclose(fp);
            _exit(0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        unlink(path.c_str());
        if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE) {
            fprintf(stderr,
                    "FAIL: over-budget-by-one: expected exit(EXIT_FAILURE), got "
                    "exited=%d status=%d signaled=%d signal=%d\n",
                    WIFEXITED(status), WEXITSTATUS(status),
                    WIFSIGNALED(status), WTERMSIG(status));
            exit(1);
        }
        fprintf(stderr, "OK:   over-budget-by-one\n");
    }

    // Case 9c: a NON-@ line far longer than the budget. Non-header lines are
    // dropped regardless of length, so this must NOT abort -- only a kept
    // (@-prefixed) line over budget is fatal (Case 9/9b). With a following @-line
    // the dropped over-budget line leaves only the @SQ record. This runs inline
    // (no fork): a regression that re-fatals on the non-@ line would exit(1) here.
    {
        std::string long_non_header;
        long_non_header.append(70000, 'x');  // no '@' prefix, well over the 64 KiB budget
        long_non_header.push_back('\n');
        run_case("long-non-header-dropped",
                 {long_non_header, "@SQ\tSN:chr1\tLN:1000\n"},
                 nullptr,
                 "@SQ\tSN:chr1\tLN:1000");
    }

    // Case 10: a NON-SEEKABLE stream (pipe). ftell() returns -1 on a pipe, so
    // the pre-fix fseek/ftell sizing silently discarded the whole -H file. The
    // read-side of a pipe is not seekable, so this exercises the line-by-line
    // path. Independent oracle: the assembled, escaped header is hardcoded, not
    // compared against another bwa_insert_header_file call.
    {
        int pfd[2];
        assert(pipe(pfd) == 0);
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            close(pfd[0]);
            const char *payload = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n";
            ssize_t w = write(pfd[1], payload, strlen(payload));
            (void) w;
            close(pfd[1]);
            _exit(0);
        }
        close(pfd[1]);
        FILE *fp = fdopen(pfd[0], "r");
        assert(fp != nullptr);
        char *out = bwa_insert_header_file(fp, nullptr);
        fclose(fp);
        int status = 0;
        waitpid(pid, &status, 0);
        const char *expected = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000";
        if (out == nullptr || strcmp(out, expected) != 0) {
            fprintf(stderr, "FAIL: non-seekable pipe: got \"%s\", expected \"%s\"\n",
                    out ? out : "(null)", expected);
            free(out);
            exit(1);
        }
        free(out);
        fprintf(stderr, "OK:   non-seekable pipe\n");
    }

    fprintf(stderr, "ALL HEADER INSERT TESTS PASSED\n");
    return 0;
}
