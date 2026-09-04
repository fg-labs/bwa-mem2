// test/unit/test_bandedswa_band_clamp.cpp
//
// Regression test for the 8-bit banded-SW per-lane band clamp.
//
// scalarBandedSWA narrows the band before the DP loop ("adjust $w if it is too large"):
//
//   max_ins = max(1, (qlen*max_sc + end_bonus - o_ins)/e_ins + 1)
//   max_del = max(1, (qlen*max_sc + end_bonus - o_del)/e_del + 1)
//   w       = min(w, max_ins, max_del)
//
// where max_sc is the largest scoring-matrix entry. It bounds how far a gap can profitably
// run before the end bonus can no longer repay it.
//
// The vector wrappers used to compute this with `_mm_add_epi8(qlen, end_bonus - o)` stored
// into a `uint8_t`. Whenever `qlen*max_sc + end_bonus - o` is NEGATIVE the int8 add wraps,
// the byte reads back near 255, the clamp evaluates to ~256 and effectively disappears -- so
// the kernel ran the full band `w` where the scalar ran a band as narrow as 1. It then
// explored cells the scalar never visits and returned different `gscore`/`gtle`. `score`,
// `tle`, `qle` and `max_off` happened to agree, which is why the existing gates missed it.
// The pre-scaled `qlen*max_sc` reach lives in the `qlen` SoA already, so `-A > 1` (max_sc
// > 1) is covered by the wide-arithmetic clamp without re-multiplying.
//
// The wrap condition is exactly `qlen*max_sc + end_bonus < o`, so it is NOT exotic: at bwa's
// DEFAULT -O 6 it fires for any short extension once the end bonus is small (1 + 0 - 6 < 0).
//
// Comparison follows the kernel's documented query-end contract (see the gtle CONTRACT
// comment in bandedSWA.cpp): the local-alignment fields must match unconditionally, while
// gscore/gtle are only compared when either side reports gscore > 0 -- which is exactly when
// bwa-mem consumes them. Every pair is inside bsw8_envelope_ok, i.e. reachable in production.
//
// Runs on every CI matrix row that defines a vector kernel, so the x86 SSE4.1 (128-bit) and
// AVX2 (256-bit) wrappers are covered as well as NEON. Deterministic (fixed RNG seed).
#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"
#include "bandedSWA.h"

#if HAVE_BSW_VECTOR_8_16

namespace {

struct Out { int score, tle, gtle, qle, gscore, max_off; };

/// One (scoring parameters) trial. Returns the number of vector-vs-scalar mismatches over the
/// pairs the CURRENT 8-bit routing envelope admits, and reports how many were checked so a
/// vacuous trial cannot masquerade as a pass.
long trial(int a, int b, int o, int e, int zdrop, int w, int end_bonus,
           int minq, int maxq, long n, long *checked)
{
    const int maxStep = a > 1 ? a : 1;
    const int STRIDE  = 400;

    int8_t mat[25];
    { int k = 0;
      for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? (int8_t)a : (int8_t)-b; mat[k++] = -1; }
      for (int j = 0; j < 5; ++j) mat[k++] = -1; }

    BandedPairWiseSW bsw(o, e, o, e, zdrop, end_bonus, mat, (int8_t)a, (int8_t)b, 1);

    std::mt19937_64 rng(0xEC7C0DE4ull);
    std::vector<uint8_t> ref((size_t)STRIDE * n, 0), qer((size_t)STRIDE * n, 0);
    std::vector<SeqPair> pairs(n);
    std::vector<Out> oracle(n);
    std::uniform_int_distribution<int> lenD(minq, maxq), hD(1, zdrop + 1), unit(3, 12);

    for (long c = 0; c < n; ++c) {
        int aa = lenD(rng), bb = lenD(rng);
        int len1 = aa > bb ? aa : bb, len2 = aa > bb ? bb : aa;   /* len1 >= len2 */
        int h0 = hD(rng);
        uint8_t *s1 = &ref[(size_t)c * STRIDE];
        uint8_t *s2 = &qer[(size_t)c * STRIDE];
        int u = unit(rng); uint8_t ub[16];
        for (int i = 0; i < u; i++) ub[i] = (uint8_t)(rng() % 4);
        for (int i = 0; i < len1; i++) s1[i] = ub[i % u];
        int ti = 0;
        for (int i = 0; i < len2; i++) {
            uint8_t base = (ti < len1) ? s1[ti] : (uint8_t)(rng() % 4);
            if ((int)(rng() % 100) < 5) base = (uint8_t)(rng() % 4);
            s2[i] = base;
            if ((rng() % 40) == 0) { if (rng() & 1) ti += 2; } else ti++;
        }
        SeqPair &p = pairs[c];
        p.id = c; p.len1 = len1; p.len2 = len2; p.h0 = h0;
        p.idr = (int)((size_t)c * STRIDE); p.idq = (int)((size_t)c * STRIDE);
        p.seqid = c; p.regid = c;
        p.score = p.tle = p.gtle = p.qle = p.gscore = p.max_off = -1;
        Out &O = oracle[c];
        O.score = bsw.scalarBandedSWA(len2, s2, len1, s1, w, h0,
                                      &O.qle, &O.tle, &O.gtle, &O.gscore, &O.max_off);
    }

    /* Mirror of bwamem.cpp bsw8_envelope_ok: only pairs it admits may reach getScores8. */
    std::vector<long> idx;
    idx.reserve(n);
    for (long c = 0; c < n; ++c) {
        const SeqPair &p = pairs[c];
        int shorter = p.len1 < p.len2 ? p.len1 : p.len2;
        if (p.len1 < MAX_SEQ_LEN8 && p.len2 < MAX_SEQ_LEN8 && p.len1 >= p.len2 &&
            w <= 124 && zdrop + maxStep <= 253 && p.h0 <= zdrop + 1 &&
            (p.h0 + shorter * maxStep) < 255 - maxStep) {
            idx.push_back(c);
        }
    }
    const long adm = (long)idx.size();
    *checked = adm;
    if (adm == 0) return 0;

    const long round_adm = ((adm + SIMD_WIDTH8 - 1) / SIMD_WIDTH8) * SIMD_WIDTH8;
    std::vector<SeqPair> ap(round_adm);
    for (long k = 0; k < adm; ++k) ap[k] = pairs[idx[k]];
    bsw.getScores8(ap.data(), ref.data(), qer.data(), (int32_t)adm, 1, w);

    long diffs = 0;
    for (long k = 0; k < adm; ++k) {
        const SeqPair &g = ap[k];
        const Out &O = oracle[idx[k]];
        /* Comparison follows the kernel's documented query-end contract (see the
         * gtle CONTRACT comment in bandedSWA.cpp, and bandedswa_zdrop_eweight_test):
         * the local-alignment fields must match unconditionally, while gscore/gtle are
         * only observable when a to-end alignment actually exists. bwa-mem branches on
         * `gscore <= 0`, so a vector gscore of 0 against a scalar -1 selects the same
         * branch and cannot reach a SAM record; the vector's row loop stops at
         * mlenw = min(qlen+myband, tlen) while the scalar runs to its dynamic m==0
         * break, so it legitimately misses trailing all-zero query-end rows. Anything
         * with gscore > 0 on either side IS observable and must match exactly. */
        const bool local_differs = g.score != O.score || g.tle != O.tle ||
                                   g.qle != O.qle || g.max_off != O.max_off;
        const bool toend_observable = g.gscore > 0 || O.gscore > 0;
        const bool toend_differs = toend_observable &&
                                   (g.gscore != O.gscore || g.gtle != O.gtle);
        if (local_differs || toend_differs) {
            if (diffs < 5) {
                MESSAGE("  len1=" << g.len1 << " len2=" << g.len2 << " h0=" << g.h0
                        << " | vec " << g.score << "/" << g.tle << "/" << g.gtle << "/"
                        << g.qle << "/" << g.gscore << "/" << g.max_off
                        << " | scalar " << O.score << "/" << O.tle << "/" << O.gtle << "/"
                        << O.qle << "/" << O.gscore << "/" << O.max_off);
            }
            diffs++;
        }
    }
    return diffs;
}

/// 16-bit analogue of `trial`. The 16-bit wrappers had the SAME wrap bug the 8-bit
/// wrappers were fixed for, but it was never ported: the per-lane band clamp added
/// `qlen*max_sc + (end_bonus - o)` with a 16-bit modular add and read the sum back
/// through `uint16_t`, so a NEGATIVE reach wrapped to ~65525, the clamp evaluated to a
/// huge band and disappeared -- the kernel then ran the full band `w` where the scalar
/// ran a band as narrow as 1, exploring off-diagonal cells the scalar never visits.
///
/// The clamp reach depends only on the QUERY length `len2` (`qlen[l] == len2*max_sc`),
/// not on the target, so a short query under a large gap-open
/// (`len2*max_sc + end_bonus < o`) triggers the wrap regardless of `len1`. This stays
/// in getScores16's byte-identity domain by keeping `len1 >= len2` (the same shape the
/// getScores16 longread parity test uses); the query-end contract for `len2 > len1` is
/// separate and not exercised here. Every pair is routed to getScores16 directly.
long trial16(int a, int b, int o, int e, int zdrop, int w, int end_bonus,
             int minq, int maxq, long n, long *checked)
{
    const int STRIDE = 400;

    int8_t mat[25];
    { int k = 0;
      for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? (int8_t)a : (int8_t)-b; mat[k++] = -1; }
      for (int j = 0; j < 5; ++j) mat[k++] = -1; }

    BandedPairWiseSW bsw(o, e, o, e, zdrop, end_bonus, mat, (int8_t)a, (int8_t)b, 1);

    std::mt19937_64 rng(0x16B0C0DEull);
    std::vector<uint8_t> ref((size_t)STRIDE * n, 0), qer((size_t)STRIDE * n, 0);
    std::vector<SeqPair> pairs(n);
    std::vector<Out> oracle(n);
    std::uniform_int_distribution<int> lenD(minq, maxq), hD(1, zdrop + 1), unit(3, 12);

    for (long c = 0; c < n; ++c) {
        int aa = lenD(rng), bb = lenD(rng);
        int len1 = aa > bb ? aa : bb, len2 = aa > bb ? bb : aa;   /* len1 >= len2 (in-domain) */
        int h0 = hD(rng);
        uint8_t *s1 = &ref[(size_t)c * STRIDE];   /* target (ref, len1) */
        uint8_t *s2 = &qer[(size_t)c * STRIDE];   /* query  (read, len2) */
        int u = unit(rng); uint8_t ub[16];
        for (int i = 0; i < u; i++) ub[i] = (uint8_t)(rng() % 4);
        for (int i = 0; i < len1; i++) s1[i] = ub[i % u];
        int ti = 0;
        for (int i = 0; i < len2; i++) {
            uint8_t base = (ti < len1) ? s1[ti] : (uint8_t)(rng() % 4);
            if ((int)(rng() % 100) < 5) base = (uint8_t)(rng() % 4);
            s2[i] = base;
            if ((rng() % 20) == 0) { if (rng() & 1) ti += 2; } else ti++;   /* sparse indels */
        }
        SeqPair &p = pairs[c];
        p.id = c; p.len1 = len1; p.len2 = len2; p.h0 = h0;
        p.idr = (int)((size_t)c * STRIDE); p.idq = (int)((size_t)c * STRIDE);
        p.seqid = c; p.regid = c;
        p.score = p.tle = p.gtle = p.qle = p.gscore = p.max_off = -1;
        Out &O = oracle[c];
        O.score = bsw.scalarBandedSWA(len2, s2, len1, s1, w, h0,
                                      &O.qle, &O.tle, &O.gtle, &O.gscore, &O.max_off);
    }

    *checked = n;
    const long round_n = ((n + SIMD_WIDTH16 - 1) / SIMD_WIDTH16) * SIMD_WIDTH16;
    std::vector<SeqPair> ap(round_n);   /* padding lanes value-initialized: len2 == 0 */
    for (long k = 0; k < n; ++k) ap[k] = pairs[k];
    bsw.getScores16(ap.data(), ref.data(), qer.data(), (int32_t)n, 1, w);

    long diffs = 0;
    for (long k = 0; k < n; ++k) {
        const SeqPair &g = ap[k];
        const Out &O = oracle[k];
        /* Same query-end contract as the 8-bit trial: local fields unconditional,
         * gscore/gtle only when a to-end alignment is observable on either side. */
        const bool local_differs = g.score != O.score || g.tle != O.tle ||
                                   g.qle != O.qle || g.max_off != O.max_off;
        const bool toend_observable = g.gscore > 0 || O.gscore > 0;
        const bool toend_differs = toend_observable &&
                                   (g.gscore != O.gscore || g.gtle != O.gtle);
        if (local_differs || toend_differs) {
            if (diffs < 5) {
                MESSAGE("  len1=" << g.len1 << " len2=" << g.len2 << " h0=" << g.h0
                        << " | vec " << g.score << "/" << g.tle << "/" << g.gtle << "/"
                        << g.qle << "/" << g.gscore << "/" << g.max_off
                        << " | scalar " << O.score << "/" << O.tle << "/" << O.gtle << "/"
                        << O.qle << "/" << O.gscore << "/" << O.max_off);
            }
            diffs++;
        }
    }
    return diffs;
}

/// Long-read fill-overflow case for the 16-bit tier -- the *reachable* half of #468.
///
/// This is a DIFFERENT failure from the short-query wrap the `trial16` cases above
/// exercise. There the clamp's own add wrapped a NEGATIVE reach; here the wrapper's
/// SoA fill `len2 * max_sc` overflows the narrow `uint16_t` slot on its own for a
/// long query at `-A >= 3` (`len2 * max_sc > 65535` -- e.g. len2 = 21846 at -A3 ->
/// 65538, which truncates to 2 in a uint16_t). The truncated reach clamps the vector
/// band to ~2, so an off-diagonal indel wider than that is missed, while the wide
/// `int32_t` slot (post-fix) keeps the full band `w`. `-A2` cannot reach this: the
/// ref-window guard caps `len2 <= 32767`, so `len2 * 2 <= 65534` never overflows.
///
/// The actual alignment score stays inside int16 -- a short high-scoring motif in an
/// otherwise-mismatching long read -- so the pair is a valid 16-bit-tier input even
/// though `len2 * max_sc` is not. The optimum matches motif A on the diagonal, deletes
/// `del` ref bases, then matches motif B off-diagonal by `del`; reaching B needs band
/// >= del. Fails on the pre-fix (narrow-slot) build (vector band ~2 < del, misses B)
/// and passes after (vector band == scalar band == w). Every lane carries the identical
/// constructed pair (deterministic), so a whole batch is checked.
long trial16_longread_fill_overflow(int a, int b, int o, int e, int zdrop, int w,
                                    int end_bonus, int len2, int del, long *checked)
{
    const int len1   = len2 + del;                 /* ref carries the extra `del` bases */
    const int STRIDE = len1 + 8;
    const int M      = 30;                          /* motif length (A and B) */

    int8_t mat[25];
    { int k = 0;
      for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? (int8_t)a : (int8_t)-b; mat[k++] = -1; }
      for (int j = 0; j < 5; ++j) mat[k++] = -1; }

    BandedPairWiseSW bsw(o, e, o, e, zdrop, end_bonus, mat, (int8_t)a, (int8_t)b, 1);

    const long n = SIMD_WIDTH16;                    /* exactly one batch, no padding */
    std::mt19937_64 rng(0x106EAD16ull);
    std::vector<uint8_t> ref((size_t)STRIDE * n, 0), qer((size_t)STRIDE * n, 0);

    /* Build ONE golden pair in lane 0, then replicate it across every lane.
     *   ref (target, len1): [A:M][del bases][B:M][random tail]
     *   qer (query,  len2): [A:M]          [B:M][random tail]
     * A and B are exact matches; the deletion bases live only in ref; the tail is
     * independent random, so it cannot beat the motif and zdrop terminates through it. */
    uint8_t A[M], B[M];
    for (int i = 0; i < M; i++) A[i] = (uint8_t)(rng() % 4);
    for (int i = 0; i < M; i++) B[i] = (uint8_t)(rng() % 4);

    uint8_t *rr = ref.data(), *qq = qer.data();    /* lane 0 */
    for (int i = 0; i < len1; i++) rr[i] = (uint8_t)(rng() % 4);
    for (int i = 0; i < len2; i++) qq[i] = (uint8_t)(rng() % 4);
    for (int i = 0; i < M; i++) { rr[i] = A[i]; qq[i] = A[i]; }              /* A on-diagonal at 0 */
    for (int i = 0; i < M; i++) { rr[M + del + i] = B[i]; qq[M + i] = B[i]; } /* B off-diagonal by del */

    for (long c = 1; c < n; ++c) {                 /* replicate lane 0 */
        std::copy(rr, rr + len1, ref.data() + (size_t)c * STRIDE);
        std::copy(qq, qq + len2, qer.data() + (size_t)c * STRIDE);
    }

    Out oracle;
    oracle.score = bsw.scalarBandedSWA(len2, qq, len1, rr, w, /*h0*/ 1,
                                       &oracle.qle, &oracle.tle, &oracle.gtle,
                                       &oracle.gscore, &oracle.max_off);

    std::vector<SeqPair> pairs(n);
    for (long c = 0; c < n; ++c) {
        SeqPair &p = pairs[c];
        p.id = c; p.len1 = len1; p.len2 = len2; p.h0 = 1;
        p.idr = (int)((size_t)c * STRIDE); p.idq = (int)((size_t)c * STRIDE);
        p.seqid = c; p.regid = c;
        p.score = p.tle = p.gtle = p.qle = p.gscore = p.max_off = -1;
    }
    *checked = n;
    bsw.getScores16(pairs.data(), ref.data(), qer.data(), (int32_t)n, 1, w);

    long diffs = 0;
    for (long c = 0; c < n; ++c) {
        const SeqPair &g = pairs[c];
        const bool local_differs = g.score != oracle.score || g.tle != oracle.tle ||
                                   g.qle != oracle.qle || g.max_off != oracle.max_off;
        const bool toend_observable = g.gscore > 0 || oracle.gscore > 0;
        const bool toend_differs = toend_observable &&
                                   (g.gscore != oracle.gscore || g.gtle != oracle.gtle);
        if (local_differs || toend_differs) {
            if (diffs < 3)
                MESSAGE("  lane=" << c << " len1=" << g.len1 << " len2=" << g.len2
                        << " | vec " << g.score << "/" << g.tle << "/" << g.qle << "/"
                        << g.max_off << " | scalar " << oracle.score << "/" << oracle.tle
                        << "/" << oracle.qle << "/" << oracle.max_off);
            diffs++;
        }
    }
    return diffs;
}

} // namespace

TEST_CASE("bandedSWA 8-bit per-lane band clamp matches the scalar reference"
          * doctest::test_suite("unit/bandedswa")) {
    // Scoring sets straddling the wrap boundary in both directions. The `wrap`
    // sets are the ones that reproduced the bug: they make
    // qlen*max_sc + end_bonus - o negative for the shortest generated query.
    struct Case { int a, o, e, b, end_bonus, minq, maxq; const char *label; };
    const Case cases[] = {
        {1,  6, 1, 4,  5, 10, 120, "bwa defaults"},
        {1, 16, 1, 4,  5, 10, 120, "-O16 (wrap)"},
        {1, 16, 1, 9,  5, 10, 120, "-B9 -O16 (wrap)"},
        {1, 16, 3, 9,  5, 10, 120, "-B9 -O16 -E3 (wrap)"},
        {1,  6, 1, 4,  0,  1, 120, "-O6 -L0 short query (wrap, DEFAULT -O)"},
        {1,  6, 1, 4,  0,  3, 120, "-O6 -L0 minq=3 (wrap, DEFAULT -O)"},
        {1, 16, 1, 4, 20, 10, 120, "-O16 large end bonus (no wrap)"},
        {1, 16, 1, 4,  5, 15, 120, "-O16 longer queries (no wrap)"},
        {2,  6, 1, 8,  5, 10, 120, "-A2 (max_sc factor)"},
        {5,  6, 1, 20, 5, 10,  60, "-A5 (max_sc factor)"},
    };

    for (const Case &c : cases) {
        SUBCASE(c.label) {
            long checked = 0;
            const long diffs = trial(c.a, c.b, c.o, c.e, 100, 100, c.end_bonus,
                                     c.minq, c.maxq, 8000, &checked);
            // A trial that admitted nothing would pass vacuously.
            CHECK(checked > 0);
            CHECK(diffs == 0);
        }
    }
}

TEST_CASE("bandedSWA 16-bit per-lane band clamp matches the scalar reference"
          * doctest::test_suite("unit/bandedswa")) {
    // The 16-bit wrappers had the same wrap the 8-bit fix cured, but it was never
    // ported. The `wrap` sets make qlen*max_sc + end_bonus - o negative for the short
    // len2 > len1 pairs that reach the 16-bit tier, so the OLD 16-bit clamp disabled
    // itself and getScores16 diverged from the scalar oracle. On the DEFAULT (non-wrap)
    // set the old and new 16-bit outputs are byte-identical -- the fix only changes the
    // wrapping path -- so this case must pass on the fixed build and FAIL on the old one.
    struct Case { int a, o, e, b, end_bonus, minq, maxq; const char *label; };
    const Case cases[] = {
        {1,  6, 1, 4,  5, 40, 400, "bwa defaults (no wrap)"},
        {1, 16, 1, 4,  5,  4,  14, "-O16 short query (wrap)"},
        {1, 16, 1, 9,  5,  4,  14, "-B9 -O16 short query (wrap)"},
        {1, 16, 3, 9,  5,  4,  14, "-B9 -O16 -E3 short query (wrap)"},
        {1,  6, 1, 4,  0,  4,  10, "-O6 -L0 short query (wrap, DEFAULT -O)"},
        {2, 16, 1, 8,  5,  4,  10, "-A2 -O16 short query (wrap, max_sc factor)"},
    };

    for (const Case &c : cases) {
        SUBCASE(c.label) {
            long checked = 0;
            const long diffs = trial16(c.a, c.b, c.o, c.e, 100, 100, c.end_bonus,
                                       c.minq, c.maxq, 8000, &checked);
            CHECK(checked > 0);
            CHECK(diffs == 0);
        }
    }
}

TEST_CASE("bandedSWA 16-bit band clamp: long-read fill overflow (the reachable half of #468)"
          * doctest::test_suite("unit/bandedswa")) {
    // The wrapper's `len2 * max_sc` SoA fill overflows the narrow uint16_t slot for a
    // long query at -A >= 3. The old narrow slot truncated the band-clamp reach, so the
    // vector explored a band ~2 wide where the scalar reference used the full band w and
    // captured an off-diagonal deletion. Each case's `del` exceeds the truncated band but
    // is <= w, so it must FAIL on the pre-fix (narrow-slot) build and PASS after -- unlike
    // the short-query cases above, this is the only half of the fix that is reachable in
    // the production routing envelope.
    struct Case { int a, b, o, e, w, end_bonus, len2, del; const char *label; };
    const Case cases[] = {
        // len2*a overflows uint16_t and truncates to a tiny reach:
        {3, 8, 6, 1, 100, 5, 21846, 10, "-A3 len2=21846 (65538 -> 2), 10bp deletion"},
        {4, 9, 6, 1, 100, 5, 16384, 12, "-A4 len2=16384 (65536 -> 0), 12bp deletion"},
    };

    for (const Case &c : cases) {
        SUBCASE(c.label) {
            long checked = 0;
            const long diffs = trial16_longread_fill_overflow(
                c.a, c.b, c.o, c.e, /*zdrop*/ 100, c.w, c.end_bonus, c.len2, c.del, &checked);
            CHECK(checked > 0);
            CHECK(diffs == 0);
        }
    }
}

#endif // HAVE_BSW_VECTOR_8_16
