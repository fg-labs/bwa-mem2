// test/unit/test_kswv_capacity_guard.cpp
//
// The kswv rescue wrappers pack each lane's reference/query into a per-thread
// SoA buffer sized maxRefLen*SIMD_WIDTH / maxQerLen*SIMD_WIDTH, where
// maxRefLen/maxQerLen are the constructor arguments plus a fixed +16 headroom
// (kswv ctor, kswv.cpp). A pair whose len1 >= maxRefLen (or whose padded query
// quantum >= maxQerLen) would write past that buffer, so every rescue wrapper
// guards the gather loop with xassert(sp.len1 < maxRefLen) /
// xassert(sp.len2 < maxQerLen) / xassert(quant[j] < maxQerLen). xassert stays
// live under a hypothetical -DNDEBUG build, unlike the plain assert() these
// replaced/joined -- so the guard must ABORT, not silently overrun, when a
// pair exceeds capacity.
//
// This exercises that contract on whatever tier the host builds (the guard is
// present in all six rescue wrappers -- NEON/AVX2/AVX-512BW x 8-bit/16-bit).
// It constructs a kswv with a deliberately tiny capacity and a pair far larger
// than it, drives getScores16, and requires the process to die on SIGABRT
// rather than run into the pack. A positive control with adequate capacity
// must instead complete cleanly, proving the death is the guard and not the
// harness setup.
//
// Forked like test/unit/test_xassert_macro.cpp: the child restores the default
// SIGABRT disposition and silences stderr/core dumps, so the parent can read
// how it terminated without the child re-entering the doctest runner.

#include <csignal>
#include <cstdint>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "doctest/doctest.h"

#include "ksw_runner.h"    // DEFAULT_GAP_OPEN / DEFAULT_GAP_EXTEND
#include "kswv_runner.h"   // BWA_TESTS_HAVE_KSWV
#include "scoring.h"
#include "seqpair.h"
#include "seqpair_batch.h"

#if BWA_TESTS_HAVE_KSWV

#include "kswv.h"

namespace {

// Run `body` in a forked child with stderr and core dumps suppressed; return
// its wait status. A body that runs to completion calls _exit(0); the guard
// firing ends the child on SIGABRT (via _err_fatal_simple_core -> abort()).
int run_in_child(void (*body)()) {
    const pid_t pid = fork();
    if (pid == 0) {
        signal(SIGABRT, SIG_DFL);
        struct rlimit no_core = {0, 0};
        setrlimit(RLIMIT_CORE, &no_core);
        (void)freopen("/dev/null", "w", stderr);
        body();
        _exit(0);
    }
    REQUIRE(pid > 0);
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    return status;
}

// A single ref/query pair, both `len` bases long (2-bit encoded, repeating
// A/C/G/T so the content is well-formed for the kernel in the passing case).
bwa_tests::TestPair long_pair(int len) {
    bwa_tests::TestPair p;
    p.ref.resize(len);
    p.qry.resize(len);
    for (int i = 0; i < len; ++i) {
        p.ref[i] = static_cast<uint8_t>(i & 3);
        p.qry[i] = static_cast<uint8_t>((i + 1) & 3);
    }
    p.tag = "capacity-guard";
    return p;
}

// Drive getScores16 (phase 0 runs the gather loop that carries the guard) over
// one `pairLen`-base pair through a kswv constructed with capacity `cap`. With
// cap << pairLen the guard must fire; with cap >= pairLen it must not.
void run_one(int cap, int pairLen) {
    const bwa_tests::ScoringMatrix mat = bwa_tests::build_scoring_matrix(1, 4, 1);
    const std::vector<bwa_tests::TestPair> pairs = {long_pair(pairLen)};
    bwa_tests::BatchBuffers bb(pairs, /*xtra_flags=*/0);
    kswv w(bwa_tests::DEFAULT_GAP_OPEN, bwa_tests::DEFAULT_GAP_EXTEND,
           bwa_tests::DEFAULT_GAP_OPEN, bwa_tests::DEFAULT_GAP_EXTEND,
           mat[0], mat[1], /*numThreads=*/1, /*maxRefLen=*/cap, /*maxQerLen=*/cap);
    w.getScores16(bb.pairs(), bb.ref_buf(), bb.qer_buf(), bb.aln(), bb.n(),
                  /*numThreads=*/1, /*phase=*/0);
}

// cap 8 -> internal capacity 24 (8 + 16 headroom); a 64-base pair blows past it.
void trip_guard() { run_one(/*cap=*/8, /*pairLen=*/64); }

// cap 64 -> internal capacity 80; the same 64-base pair fits with room to spare.
void within_capacity() { run_one(/*cap=*/64, /*pairLen=*/64); }

}  // namespace

TEST_CASE("kswv rescue wrapper aborts when a pair exceeds SoA capacity") {
    const int status = run_in_child(trip_guard);
    const bool exited_cleanly = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    CHECK_FALSE(exited_cleanly);  // must not have reached the pack / _exit(0)
    if (WIFSIGNALED(status)) CHECK(WTERMSIG(status) == SIGABRT);
}

TEST_CASE("kswv rescue wrapper runs when the pair fits within SoA capacity") {
    const int status = run_in_child(within_capacity);
    REQUIRE(WIFEXITED(status));  // WEXITSTATUS is only defined once this holds
    CHECK(WEXITSTATUS(status) == 0);
}

#endif  // BWA_TESTS_HAVE_KSWV
