// test/unit/test_bandedswa_zdrop_gate.cpp
//
// Regression tests for two vector banded-SW z-drop defects, both differential
// against the scalarBandedSWA oracle:
//   1. the z-drop gate at `-d 0` (z-drop disabled) -- see below;
//   2. the 16-bit z-drop drift term losing its gap-extend weight at `-E > 1`
//      (the ZSCORE16 macro; a deterministic off-diagonal peak/deletion/recovery
//      pair), in the third TEST_CASE at the bottom of the file.
//
// --- (1) z-drop gate at `-d 0` (z-drop disabled) ---
//
// scalarBandedSWA gates its z-drop early-exit on `zdrop > 0` (bandedSWA.cpp, the
// `else if (zdrop > 0)` branch), so with z-drop disabled it never truncates. The
// vector kernels (8-bit and 16-bit, every tier) applied the z-drop test
// UNCONDITIONALLY, so at `zdrop == 0` they compared the running drop against a
// zero threshold and killed a lane as soon as its score fell one point below the
// row max -- truncating alignments the scalar reference runs to completion. The
// two then disagreed on the query-end fields (gscore/gtle) and, once the vector
// stops early, on the local fields too.
//
// The fix gates every vector z-drop mask update on `zdrop > 0`, matching the
// scalar. On the DEFAULT `zdrop = 100` the gate is a no-op (the branch is taken),
// so this changes nothing on the shipping path; it only restores agreement at
// `zdrop == 0`. This case must therefore PASS on the fixed build and FAIL on the
// old one.
//
// Each pair is a matching prefix followed by a divergent tail, so the alignment
// score peaks early and then falls -- exactly the shape that makes an
// unconditional z-drop fire at `zdrop == 0`. Deterministic (fixed RNG seed).
#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"
#include "bandedSWA.h"

#if HAVE_BSW_VECTOR_8_16

namespace {

struct Out { int score, tle, gtle, qle, gscore, max_off; };

/// Compare getScores<width> to scalarBandedSWA at zdrop == 0 over n peak-then-drop
/// pairs. Returns the mismatch count (per the documented query-end contract) and
/// reports how many pairs were checked so a vacuous trial cannot pass.
long zdrop_trial(int width, int a, int b, int o, int e, int end_bonus, int w,
                 int minlen, int maxlen, long n, long *checked)
{
    const int STRIDE = maxlen + 256;
    const int zdrop = 0;   // z-drop DISABLED -- the case under test

    int8_t mat[25];
    { int k = 0;
      for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? (int8_t)a : (int8_t)-b; mat[k++] = -1; }
      for (int j = 0; j < 5; ++j) mat[k++] = -1; }

    BandedPairWiseSW bsw(o, e, o, e, zdrop, end_bonus, mat, (int8_t)a, (int8_t)b, 1);

    std::mt19937_64 rng(width == 8 ? 0x2D809A7Eull : 0x2D8016A7ull);
    std::vector<uint8_t> ref((size_t)STRIDE * n, 0), qer((size_t)STRIDE * n, 0);
    std::vector<SeqPair> pairs(((n + 63) / 64) * 64);   // SIMD-width round-up + slack
    std::vector<Out> oracle(n);
    std::uniform_int_distribution<int> lenD(minlen, maxlen);

    for (long c = 0; c < n; ++c) {
        int la = lenD(rng), lb = lenD(rng);
        int len1 = la > lb ? la : lb, len2 = la > lb ? lb : la;   /* len1 >= len2 */
        uint8_t *s1 = &ref[(size_t)c * STRIDE];   /* target (ref) */
        uint8_t *s2 = &qer[(size_t)c * STRIDE];   /* query  (read) */
        for (int i = 0; i < len1; ++i) s1[i] = (uint8_t)(rng() % 4);
        /* Full-length match with a short mismatch VALLEY near the middle. Without
         * z-drop the optimal alignment crosses the valley and runs to the read end
         * (high score, gscore set); an unconditional z-drop at zdrop==0 fires at the
         * valley dip and dies there, reporting only the pre-valley peak -- so score,
         * tle, qle, gscore and gtle all diverge. The suffix (len - valley) matches
         * outscore the 2-base valley (2a matches lost to 2b), so the global max is
         * genuinely past the valley. */
        for (int i = 0; i < len2; ++i) s2[i] = (i < len1) ? s1[i] : (uint8_t)(rng() % 4);
        int mid = len2 / 2;
        for (int d = 0; d < 2; ++d) {
            int j = mid + d;
            if (j < len2) { uint8_t rb = (j < len1) ? s1[j] : s2[j]; s2[j] = (uint8_t)((rb + 1) % 4); }
        }
        int h0 = a;   /* small seed */
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
    if (width == 8) bsw.getScores8(pairs.data(), ref.data(), qer.data(), (int32_t)n, 1, w);
    else            bsw.getScores16(pairs.data(), ref.data(), qer.data(), (int32_t)n, 1, w);

    long diffs = 0;
    for (long c = 0; c < n; ++c) {
        const SeqPair &g = pairs[c];
        const Out &O = oracle[c];
        /* Same query-end contract as the other bandedSWA parity tests: local
         * fields unconditional; gscore/gtle only when a to-end alignment is
         * observable on either side. */
        const bool local_differs = g.score != O.score || g.tle != O.tle ||
                                   g.qle != O.qle || g.max_off != O.max_off;
        const bool toend_observable = g.gscore > 0 || O.gscore > 0;
        const bool toend_differs = toend_observable &&
                                   (g.gscore != O.gscore || g.gtle != O.gtle);
        if (local_differs || toend_differs) {
            if (diffs < 5) {
                MESSAGE("  w" << width << " len1=" << g.len1 << " len2=" << g.len2
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

/// Non-default gap-extend (-E > 1): the z-drop drift term must be weighted by the
/// gap-extend penalty. scalarBandedSWA fires z-drop on
///   (max - m) - |drift| * e_{del|ins} > zdrop
/// (the `* e_del` / `* e_ins` factors in its else-if(zdrop>0) block). The 16-bit
/// vector ZSCORE16 macro computed `insdel` but dropped the `* e`, testing
/// (max - m) - |drift|. At the default -E 1 the factor is 1, so the bug is invisible
/// there (and the diagonal peak-then-valley trial above has drift == 0, so it cannot
/// see it either).
///
/// This constructs a DETERMINISTIC pair whose optimum is forced OFF the diagonal with
/// a large residual drop right where the z-drop is evaluated, so the pre-fix
/// (unweighted) test crosses zdrop while the correct (weighted) test does not:
///
///   ref   = prefix[P] . gap[D] (all 'T') . suffix[L]      (len1 = P + D + L)
///   query = prefix[P] .                     suffix[L]      (len2 = P + L)
///
///   * prefix is identical in both, so the running max is the seed peak at (P,P).
///   * suffix S[k] = k % 3 (bases A/C/G); the gap bases are all base 3 ('T'). The
///     query's suffix matches the ref's suffix only D cells off the diagonal (the
///     deletion). ON the diagonal, query S[k] meets a gap 'T' (k<D) or S[k-D]; with
///     D % 3 != 0, S[k-D] != S[k] for every k, so the on-diagonal path mismatches
///     every base, decays by b per row, and dies (m==0), leaving the off-diagonal
///     recovered path as the row max.
///   * that off-diagonal row max sits |drift| = D cells off the diagonal and, in the
///     transition rows, far enough below the seed peak that (max-m) - D exceeds zdrop
///     but (max-m) - D*e does not: the pre-fix kernel z-drops and truncates at the
///     seed peak, while the scalar and fixed kernel run on and the recovering suffix
///     (L*a beats the gap cost o + D*e) exceeds the peak, so the emitted score/tle
///     differ.
///
/// FAILS pre-fix, PASSES after. Every lane carries the identical constructed pair, so
/// it is deterministic and routing-independent (getScores16 is driven directly). The
/// 8-bit tier is not exercised: it already weights the drift (wide-int32 z-drop in
/// the smithWaterman*_8 kernels), and its routing envelope caps scores well below a
/// recovery this long.
long zdrop_gapdrift_trial16(int a, int b, int o, int e, int end_bonus, int w,
                            int zdrop, int P, int D, int L, long *checked)
{
    const int len1 = P + D + L;    /* ref: prefix + deleted gap + suffix */
    const int len2 = P + L;        /* query: prefix + suffix (gap skipped) */
    const int STRIDE = len1 + 64;

    int8_t mat[25];
    { int k = 0;
      for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? (int8_t)a : (int8_t)-b; mat[k++] = -1; }
      for (int j = 0; j < 5; ++j) mat[k++] = -1; }

    BandedPairWiseSW bsw(o, e, o, e, zdrop, end_bonus, mat, (int8_t)a, (int8_t)b, 1);

    const long n = SIMD_WIDTH16;   /* one full batch, all lanes identical */
    std::vector<uint8_t> ref((size_t)STRIDE * n, 0), qer((size_t)STRIDE * n, 0);

    uint8_t *rr = ref.data(), *qq = qer.data();    /* lane 0 */
    for (int k = 0; k < P; ++k) { uint8_t base = (uint8_t)((k * 7 + 1) & 3); rr[k] = base; qq[k] = base; }
    for (int k = 0; k < D; ++k) rr[P + k] = 3;                        /* gap bases: all 'T' */
    for (int k = 0; k < L; ++k) { uint8_t s = (uint8_t)(k % 3);       /* suffix S[k] = k%3 in {A,C,G} */
        rr[P + D + k] = s; qq[P + k] = s; }

    for (long c = 1; c < n; ++c) {                 /* replicate lane 0 into every lane */
        std::copy(rr, rr + len1, ref.data() + (size_t)c * STRIDE);
        std::copy(qq, qq + len2, qer.data() + (size_t)c * STRIDE);
    }

    Out oracle;
    oracle.score = bsw.scalarBandedSWA(len2, qq, len1, rr, w, /*h0*/ a,
                                       &oracle.qle, &oracle.tle, &oracle.gtle,
                                       &oracle.gscore, &oracle.max_off);

    std::vector<SeqPair> pairs(n);
    for (long c = 0; c < n; ++c) {
        SeqPair &p = pairs[c];
        p.id = c; p.len1 = len1; p.len2 = len2; p.h0 = a;
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
                MESSAGE("  e=" << e << " zdrop=" << zdrop << " P=" << P << " D=" << D
                        << " L=" << L << " | vec " << g.score << "/" << g.tle << "/" << g.qle
                        << "/" << g.gtle << "/" << g.gscore << "/" << g.max_off
                        << " | scalar " << oracle.score << "/" << oracle.tle << "/" << oracle.qle
                        << "/" << oracle.gtle << "/" << oracle.gscore << "/" << oracle.max_off);
            diffs++;
        }
    }
    return diffs;
}

} // namespace

TEST_CASE("bandedSWA vector z-drop is disabled at zdrop==0 (vector == scalar)"
          * doctest::test_suite("unit/bandedswa-zdrop")) {
    long checked = 0;
    // Small deterministic corpus (256 pairs): every pair is a peak-then-valley
    // shape that the pre-fix unconditional z-drop truncates, so ~all pairs
    // diverge on the old kernel -- a few hundred is ample to reproduce the
    // failure while staying inside the 100 ms unit-test budget. The 8-bit and
    // 16-bit widths test the same invariant with different lengths, so each is a
    // SUBCASE of the one case.
    SUBCASE("getScores8") {
        const long diffs = zdrop_trial(8, 1, 4, 6, 1, 5, 100, 60, 200, 256, &checked);
        CHECK(checked > 0);
        CHECK(diffs == 0);
    }
    SUBCASE("getScores16") {
        const long diffs = zdrop_trial(16, 1, 4, 6, 1, 5, 100, 120, 400, 256, &checked);
        CHECK(checked > 0);
        CHECK(diffs == 0);
    }
}

TEST_CASE("bandedSWA 16-bit z-drop weights the drift by gap-extend at -E > 1 (vector == scalar)"
          * doctest::test_suite("unit/bandedswa-zdrop")) {
    // At non-default -E the z-drop residual-drop term is (max-m) - |drift|*e; the
    // pre-fix ZSCORE16 dropped the *e factor, so it z-drops at the wrong threshold on
    // OFF-DIAGONAL drops (drift > 0). Sweep noisy pairs (frequent mismatches to drop
    // the score + frequent short indels to drive the alignment off-diagonal) so the
    // divergent regime is hit statistically, and compare getScores16 to the scalar
    // oracle. FAILS pre-fix, PASSES after. e = 1 is a no-op (covered by the golden
    // gate + the diagonal trial above).
    long checked = 0;
    struct Case { int a, b, o, e, zdrop, P, D, L; const char *label; };
    const Case cases[] = {
        {2, 6, 6, 3, 20, 40, 10, 120, "-A2 -E3 D=10 zdrop=20"},
        {2, 6, 6, 3, 30, 40, 20, 140, "-A2 -E3 D=20 zdrop=30"},
        {3, 6, 6, 3, 20, 40, 15, 140, "-A3 -E3 D=15 zdrop=20"},
    };
    for (const Case &c : cases) {
        SUBCASE(c.label) {
            const long diffs = zdrop_gapdrift_trial16(c.a, c.b, c.o, c.e, /*end_bonus*/ 5,
                                                      /*w*/ 100, c.zdrop, c.P, c.D, c.L, &checked);
            CHECK(checked > 0);
            CHECK(diffs == 0);
        }
    }
}

#else  // !HAVE_BSW_VECTOR_8_16

// Vector 8/16-bit kernels are not built for this tier (scalar-only build), so the
// getScores<width>-vs-scalar parity above has nothing to exercise. Register the
// case with a doctest::skip() decorator so a scalar-only build reports it as an
// explicit SKIP -- not a passing test -- and cannot silently claim "no failures"
// for a gate that never ran. (getScores<width> uses the compile-time baseline
// kernel, so each CI matrix row -- NEON, AVX2, ... -- exercises its own tier; a
// tier is covered by building the suite for it.)
TEST_CASE("bandedSWA vector z-drop gate: SKIPPED (vector 8/16-bit kernels not built)"
          * doctest::test_suite("unit/bandedswa-zdrop")
          * doctest::skip()) {
    // Body is never run under doctest::skip(); the case exists only so the
    // unsupported build state is visible as a skip in the test report.
}

#endif // HAVE_BSW_VECTOR_8_16
