/* Parity gate for the prefetch SA-lookup pipeline vs the scalar compressed lookup.
 *
 * get_sa_entries_prefetch() drives call_one_step() in a software-pipelined batch;
 * get_sa_entry_compressed() is the straight-line scalar walk the rest of the code
 * already trusts. For every suffix-array row both must return the SAME reference
 * coordinate.
 *
 * The one row where they used to disagree is the sentinel ($) row: the scalar walk
 * returns the accumulated LF `offset` when it lands on the sentinel, but
 * call_one_step() dropped it (set sa_entry = 0), so any suffix whose LF walk
 * reaches the sentinel before hitting a sampled position was reported up to
 * `offset` bases too far left -- i.e. hits within the first few bases of the
 * concatenated reference. This sweeps EVERY row, so it catches that row on any
 * index (phix included) and is byte-identical everywhere else.
 *
 * A single SMEM spanning the whole BWT interval [0, reference_seq_len) with
 * max_occ == reference_seq_len makes the prefetch enumerator stage every row at
 * step 1, so coordArray[i] is the prefetch pipeline's coordinate for row i.
 *
 * Contract: coordArray[i] == get_sa_entry_compressed(i) for all i.
 *   Fails (RED) on the pre-fix build for the sentinel-reaching rows; passes (GREEN)
 *   after. All accessors are public members of FMI_search. A third witness runs the
 *   same sweep with drop_sentinel_offset = 1 (what --compat=bwa-mem2 selects) and
 *   requires it to differ from the oracle on exactly the sentinel-reaching rows, so
 *   the bwa-mem2 arm is pinned to bwa-mem2's behaviour rather than silently fixed.
 *
 * On the oracle: get_sa_entry_compressed is a separate, straight-line implementation
 * of the same lookup (not a rearrangement of call_one_step), and it does NOT share
 * the bug -- it already returns `offset` at the sentinel -- so the cross-path parity
 * genuinely diverges pre-fix rather than being a self-consistency check that passes
 * against the bug. This mirrors the repo's other parity gates (bwtseed/smem lockstep).
 * As a second, implementation-independent witness the test also calls call_one_step
 * directly on the sentinel row and asserts it returns the accumulated offset -- that
 * bakes the fails-before condition in so a fixture change cannot silently disarm it.
 *
 * Usage: sa_lookup_sentinel_parity_test <bwa-mem3 index prefix>   (e.g. fixtures/phix.fa)
 */
#include "FMI_search.h"
#include "bwa.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <bwa-mem3 index prefix>\n", argv[0]);
        return 2;
    }

    std::unique_ptr<FMI_search> fmi(new FMI_search(argv[1]));
    fmi->load_index();

    const int64_t ref_len = fmi->reference_seq_len;
    fprintf(stderr, "sa_lookup_sentinel_parity_test: reference_seq_len=%lld sentinel_index=%lld\n",
            (long long)ref_len, (long long)fmi->sentinel_index);
    if (ref_len <= 0) { fprintf(stderr, "FAIL: empty index\n"); return 1; }

    // Witness 1 (implementation-independent, fixture-robust): drive the fixed branch
    // directly. At the sentinel ($) row call_one_step hits its b == 4 branch; with a
    // nonzero accumulated offset it must return that offset (the fix), not 0 (the bug).
    // The branch is only reachable when the sentinel row is not itself SA-sampled
    // (otherwise call_one_step takes the direct-read fast path and never sees b == 4),
    // so a sampled sentinel is a loud failure -- pick a fixture whose sentinel row is
    // non-sampled -- never a silent skip that would let the whole gate go vacuous.
    if ((fmi->sentinel_index & SA_COMPX_MASK) == 0) {
        fprintf(stderr, "FAIL: sentinel_index %lld is SA-sampled, so call_one_step's "
                "b==4 offset branch is unreachable here; use a fixture whose sentinel row "
                "is non-sampled\n", (long long)fmi->sentinel_index);
        return 1;
    }
    {
        const int64_t seed_offset = 7;
        int64_t sa = -1, off = seed_offset;
        const int64_t quit = fmi->call_one_step(fmi->sentinel_index, sa, off);
        if (quit != 1 || sa != seed_offset) {
            fprintf(stderr, "FAIL: sentinel-offset branch: call_one_step(sentinel) returned "
                    "quit=%lld sa=%lld, want 1/%lld (the fix returns the accumulated offset)\n",
                    (long long)quit, (long long)sa, (long long)seed_offset);
            return 1;
        }
        // The --compat=bwa-mem2 arm (drop_sentinel_offset = 1) must reproduce
        // bwa-mem2's coordinate on the same row: 0, the walk dropped.
        sa = -1; off = seed_offset;
        const int64_t quit2 = fmi->call_one_step(fmi->sentinel_index, sa, off, /*drop*/ 1);
        if (quit2 != 1 || sa != 0) {
            fprintf(stderr, "FAIL: sentinel-offset branch under drop_sentinel_offset=1 returned "
                    "quit=%lld sa=%lld, want 1/0 (bwa-mem2's dropped walk)\n",
                    (long long)quit2, (long long)sa);
            return 1;
        }
    }

    // Witness 2 (whole-pipeline parity): sweep every row through get_sa_entries_prefetch
    // and compare against the scalar get_sa_entry_compressed oracle.
    // One SMEM covering every row; a max_occ >= s forces step 1 (enumerate all rows).
    // Use INT32_MAX rather than a narrowed (int32_t)ref_len: it exceeds ref_len for
    // any fixture up to a 2-gigarow index, so step stays 1 without a narrowing cast.
    SMEM smem = {};
    smem.rid = 0; smem.m = 0; smem.n = 0; smem.l = 0; smem.k = 0; smem.s = ref_len;

    std::vector<int64_t> coordArray((size_t)ref_len, -1);
    int64_t coordCount = 0;   // get_sa_entries_prefetch accumulates the count into [0]
    int64_t id_ = 0;

    fmi->get_sa_entries_prefetch(&smem, coordArray.data(), &coordCount, /*count*/ 1,
                                 /*max_occ*/ INT32_MAX, /*tid*/ 0, id_);

    if (coordCount != ref_len) {
        fprintf(stderr, "FAIL: staged %lld rows, expected %lld\n",
                (long long)coordCount, (long long)ref_len);
        return 1;
    }

    int64_t mismatches = 0;
    for (int64_t i = 0; i < ref_len; i++) {
        const int64_t want = fmi->get_sa_entry_compressed(i);
        if (coordArray[i] != want) {
            if (mismatches < 5)
                fprintf(stderr, "  row %lld: prefetch=%lld compressed=%lld\n",
                        (long long)i, (long long)coordArray[i], (long long)want);
            mismatches++;
        }
    }

    if (mismatches != 0) {
        fprintf(stderr, "FAIL: %lld/%lld rows disagree with get_sa_entry_compressed\n",
                (long long)mismatches, (long long)ref_len);
        return 1;
    }

    // Witness 3 (the --compat=bwa-mem2 arm): the same sweep with the walk dropped at
    // the sentinel must differ from the oracle on EXACTLY the sentinel-reaching rows
    // (reported as 0 where the oracle is the walk offset), and on at least one row,
    // so the fixture is known to exercise the branch through the whole pipeline.
    std::vector<int64_t> coordArrayDrop((size_t)ref_len, -1);
    int64_t coordCountDrop = 0, idDrop = 0;
    fmi->get_sa_entries_prefetch(&smem, coordArrayDrop.data(), &coordCountDrop, /*count*/ 1,
                                 /*max_occ*/ INT32_MAX, /*tid*/ 0, idDrop, /*drop*/ 1);
    if (coordCountDrop != ref_len) {
        fprintf(stderr, "FAIL: drop arm staged %lld rows, expected %lld\n",
                (long long)coordCountDrop, (long long)ref_len);
        return 1;
    }
    int64_t dropped_rows = 0, bad_drop_rows = 0;
    for (int64_t i = 0; i < ref_len; i++) {
        if (coordArrayDrop[i] == coordArray[i]) continue;
        // A differing row must be a sentinel-reaching row: bwa-mem2 says 0, the
        // correct coordinate is a positive walk offset. Anything else is a bug in
        // the arm, not bwa-mem2's behaviour.
        if (coordArrayDrop[i] == 0 && coordArray[i] > 0) dropped_rows++;
        else {
            if (bad_drop_rows < 5)
                fprintf(stderr, "  row %lld: drop arm=%lld fixed=%lld (not a sentinel drop)\n",
                        (long long)i, (long long)coordArrayDrop[i], (long long)coordArray[i]);
            bad_drop_rows++;
        }
    }
    if (bad_drop_rows != 0 || dropped_rows == 0) {
        fprintf(stderr, "FAIL: drop arm: %lld sentinel drops, %lld other differences "
                "(want >= 1 and 0)\n", (long long)dropped_rows, (long long)bad_drop_rows);
        return 1;
    }

    fprintf(stderr, "PASS: sentinel-offset branch returns the accumulated offset, all "
            "%lld rows byte-identical (get_sa_entries_prefetch == get_sa_entry_compressed), "
            "and the --compat=bwa-mem2 arm drops exactly the %lld sentinel-reaching rows\n",
            (long long)ref_len, (long long)dropped_rows);
    return 0;
}
