// test/unit/test_bandedswa_asym.cpp
//
// γ (D3 asymmetric BS scoring) gate. The SIMD banded SW kernels
// (getScores8 / getScores16) must score an arbitrary, possibly *asymmetric*
// substitution matrix identically to the scalar oracle (scalarBandedSWA, which
// already consumes the full mat[25]). The generic-matrix seam landed (commit
// e62a253): the SIMD path now consumes the full mat[25] rather than a symmetric
// XOR-LUT, so on a bisulfite OT matrix (ref-C x read-T freed) it matches the
// scalar oracle on converted reads.
//
// Expected state (now GREEN): both the symmetric control and the asymmetric
// cases PASS. Convention: target-major, mat[ref*5 + read].

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "bandedSWA.h"
#include "simd_dispatch.h"

#if HAVE_BSW_VECTOR_8_16

namespace {

// Standard symmetric nt4 matrix (diagonal = +a, off-diagonal = -b, N = ambig).
void build_sym_mat(int8_t mat[25], int a, int b, int ambig) {
    int k = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? (int8_t)a : (int8_t)(-b);
        mat[k++] = (int8_t)ambig;
    }
    for (int j = 0; j < 5; ++j) mat[k++] = (int8_t)ambig;
}

// OT bisulfite matrix: symmetric, then free ref-C x read-T (a C->T conversion is
// not a mismatch). A,C,G,T = 0,1,2,3; target-major mat[ref*5 + read].
void build_ot_mat(int8_t mat[25], int a, int b, int ambig) {
    build_sym_mat(mat, a, b, ambig);
    mat[1 * 5 + 3] = (int8_t)a;   // ref C (1), read T (3) -> match (allowed conversion)
}

struct Out { int score, tle, gtle, qle, gscore, max_off; };

// Generate n converted-read pairs (read = C->T projection of the ref prefix,
// plus sparse real mismatches) and compare getScores<width> to scalarBandedSWA
// under the given matrix. Returns the number of pairs whose SIMD result record
// differs from the scalar oracle on ANY field (score, tle, gtle, qle, gscore,
// max_off) -- an ambiguous-base regression can perturb a secondary field while
// leaving score intact, so score alone is not a sufficient parity gate.
int score_mismatches(int width, const int8_t mat[25], int a, int b,
                     int n, int maxlen, unsigned long seed, int n_pct = 0) {
    const int ambig = -1, o = 6, e = 1, end_bonus = 5, w = 100, zdrop = 100;
    const int STRIDE = maxlen + 256;
    BandedPairWiseSW bsw(o, e, o, e, zdrop, end_bonus, const_cast<int8_t*>(mat), a, b, 1);

    std::vector<uint8_t> ref((size_t)STRIDE * n, 0), qer((size_t)STRIDE * n, 0);
    std::vector<SeqPair> pairs(((n + 63) / 64) * 64);   // SIMD-width round-up + slack
    std::vector<Out> oracle(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> lenD(40, maxlen);

    for (int c = 0; c < n; ++c) {
        int la = lenD(rng), lb = lenD(rng);
        int len1 = la > lb ? la : lb, len2 = la > lb ? lb : la;
        uint8_t *s1 = &ref[(size_t)c * STRIDE];   // target (ref), original bases
        uint8_t *s2 = &qer[(size_t)c * STRIDE];   // query (read)
        for (int i = 0; i < len1; ++i) s1[i] = (uint8_t)(rng() % 4);
        for (int i = 0; i < len2; ++i) {
            uint8_t rb = s1[i];
            uint8_t base = (rb == 1) ? 3 : rb;                 // C->T conversion in the read
            if ((int)(rng() % 100) < 3) base = (uint8_t)(rng() % 4);  // 3% real mismatch
            s2[i] = base;
        }
        // Ambiguous-base (N = nt4 code 4) injection: exercises the query-side
        // 4->8 remap in neon_soa_pack16 (target N stays 4). The scalar oracle
        // scores N through the same mat[25] (row/col 4), so both paths must agree.
        if (n_pct > 0) {
            for (int i = 0; i < len1; ++i)
                if ((int)(rng() % 100) < n_pct) s1[i] = 4;     // target N
            for (int i = 0; i < len2; ++i)
                if ((int)(rng() % 100) < n_pct) s2[i] = 4;     // query N -> code 8 in the kernel
        }
        SeqPair &sp = pairs[c];
        sp.id = c; sp.len1 = len1; sp.len2 = len2; sp.h0 = 40;
        sp.idr = (int)((size_t)c * STRIDE); sp.idq = (int)((size_t)c * STRIDE);
        sp.seqid = c; sp.regid = c;
        sp.score = sp.tle = sp.gtle = sp.qle = sp.gscore = sp.max_off = -1;
        Out &O = oracle[c];
        O.score = bsw.scalarBandedSWA(len2, s2, len1, s1, w, sp.h0,
                                      &O.qle, &O.tle, &O.gtle, &O.gscore, &O.max_off);
    }
    if (width == 8) bsw.getScores8(pairs.data(), ref.data(), qer.data(), (int32_t)n, 1, w);
    else            bsw.getScores16(pairs.data(), ref.data(), qer.data(), (int32_t)n, 1, w);

    int diff = 0;
    for (int c = 0; c < n; ++c) {
        const SeqPair &p = pairs[c];
        const Out &o = oracle[c];
        if (p.score != o.score || p.tle != o.tle || p.gtle != o.gtle ||
            p.qle != o.qle || p.gscore != o.gscore || p.max_off != o.max_off)
            ++diff;
    }
    return diff;
}

} // namespace

// These parity cases deliberately run getScores8/16 on whatever tier the host
// (or BWAMEM3_FORCE_TIER) selects, NOT only ARM NEON: exercising every tier is
// what lets this suite catch a tier-specific kernel divergence -- e.g. a 256-bit
// AVX2 gap-open sign bug that leaked a negative F into the band's zero-scan and
// perturbed gtle, invisible to a NEON-only gate. The NEON path itself is covered
// by the arm64 CI matrix rows. This sentinel makes the active tier visible in the
// test log and asserts a real vector kernel (not the scalar fallback) is running,
// so a misconfigured runner that quietly dropped to scalar fails loudly here.
TEST_CASE("bandedSWA parity gate exposes and asserts the active SIMD tier"
          * doctest::test_suite("unit/bandedswa-asym")) {
    bwamem3_simd_init();
    const int tier = bwamem3_simd_tier();
    MESSAGE("bandedSWA parity gate active SIMD tier: " << std::string(bwamem3_simd_tier_name(tier)));
    CHECK(tier != BWAMEM3_TIER_NONE);
}

TEST_CASE("bandedSWA symmetric control: getScores8 == scalar (sanity)"
          * doctest::test_suite("unit/bandedswa-asym")) {
    int8_t mat[25]; build_sym_mat(mat, 1, 4, -1);
    CHECK(score_mismatches(8, mat, 1, 4, 4000, 200, 1) == 0);
}

TEST_CASE("bandedSWA getScores8 honors an asymmetric (OT bisulfite) matrix"
          * doctest::test_suite("unit/bandedswa-asym")) {
    int8_t mat[25]; build_ot_mat(mat, 1, 4, -1);
    // GREEN since the generic-matrix seam (e62a253): the SIMD path consumes the
    // full mat[25] and rewards the freed ref-C x read-T cell like the scalar oracle.
    CHECK(score_mismatches(8, mat, 1, 4, 4000, 200, 2) == 0);
}

TEST_CASE("bandedSWA getScores16 honors an asymmetric (OT bisulfite) matrix"
          * doctest::test_suite("unit/bandedswa-asym")) {
    int8_t mat[25]; build_ot_mat(mat, 1, 4, -1);
    CHECK(score_mismatches(16, mat, 1, 4, 4000, 400, 3) == 0);
}

// Ambiguous-base coverage for the packed 8-bit path. Every other 8-bit
// extension parity case builds sequences from rng()%4, so the query-side N->8
// remap in neon_soa_pack16 (and the target-side "N stays 4" fill) went
// untested. With ~10% N in both target and query, getScores8 must still match
// the scalar oracle, which scores N through the same mat[25] (row/col 4).
// 512 pairs (32 full SIMD batches of 16 lanes) rather than 4000: the scalar
// oracle runs once per pair, so a large corpus can blow the unit-test time
// budget under the sanitizer CI configs. The fixed seed makes the corpus
// deterministic and reproducible, and at 10% N over the generated lengths every
// batch carries N at target and query boundary and interior positions.
TEST_CASE("bandedSWA getScores8 matches scalar with ambiguous (N) bases"
          * doctest::test_suite("unit/bandedswa-asym")) {
    int8_t mat[25]; build_sym_mat(mat, 1, 4, -1);
    CHECK(score_mismatches(8, mat, 1, 4, 512, 200, 4, /*n_pct=*/10) == 0);
}

TEST_CASE("bandedSWA getScores16 matches scalar with ambiguous (N) bases (control)"
          * doctest::test_suite("unit/bandedswa-asym")) {
    int8_t mat[25]; build_sym_mat(mat, 1, 4, -1);
    CHECK(score_mismatches(16, mat, 1, 4, 512, 400, 5, /*n_pct=*/10) == 0);
}

#endif // HAVE_BSW_VECTOR_8_16
