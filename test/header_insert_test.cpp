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
                     const char *seed_hdr)
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
    run_case("trailing-backslash-then-at-line",
             {"@CO\tfield\\\n",
              "@SQ\tSN:chr1\tLN:1000\n"},
             nullptr);

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
