// test/unit/test_kswv_correctness.cpp
//
// Self-consistency test for kswv::getScores8 versus scalar ksw_align2.
// Ported from test/kswv_selftest.cpp, now using the bwa_tests framework.
//
// Locks in bit-identical batched-SIMD output for 10,000 random pairs plus
// curated edge cases. Runs on every CI matrix row (SSE4.1, AVX2, AVX2
// clang, AVX2 no-mimalloc, multi-arch, ARM64 Linux, macOS ARM64).

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "kswr_cmp.h"
#include "ksw_runner.h"
#include "kswv_runner.h"
#include "scoring.h"
#include "seqpair_gen.h"

#if BWA_TESTS_HAVE_KSWV

namespace {

// Edge-case pair builder. Deterministic given seed.
std::vector<bwa_tests::TestPair> build_edge_cases(std::mt19937 &rng) {
    using bwa_tests::gen_exact_match_pair;
    using bwa_tests::gen_all_mismatch_pair;
    using bwa_tests::gen_homopolymer_pair;
    using bwa_tests::gen_sub_cluster_pair;
    using bwa_tests::gen_with_n_bases_pair;
    using bwa_tests::gen_random_pair;

    std::vector<bwa_tests::TestPair> pairs;
    pairs.push_back(gen_exact_match_pair(50));
    pairs.push_back(gen_exact_match_pair(100));
    pairs.push_back(gen_all_mismatch_pair(50));
    pairs.push_back(gen_all_mismatch_pair(100));
    pairs.push_back(gen_homopolymer_pair(50, 0));
    pairs.push_back(gen_homopolymer_pair(50, 1));
    pairs.push_back(gen_homopolymer_pair(50, 2));
    pairs.push_back(gen_homopolymer_pair(50, 3));
    pairs.push_back(gen_sub_cluster_pair(rng, 100, 150, 40, 3));
    pairs.push_back(gen_sub_cluster_pair(rng, 100, 150, 40, 10));
    pairs.push_back(gen_with_n_bases_pair(rng, 100, 150, 5));
    pairs.push_back(gen_with_n_bases_pair(rng, 100, 150, 20));
    pairs.push_back(gen_random_pair(rng, 20, 40));
    pairs.push_back(gen_random_pair(rng, 10, 30));
    return pairs;
}

std::vector<bwa_tests::TestPair> build_bulk_random(std::mt19937 &rng, int n) {
    std::vector<bwa_tests::TestPair> pairs;
    pairs.reserve(n);
    std::uniform_int_distribution<int> qlen_d(50, 128);
    std::uniform_int_distribution<int> rlen_d(100, 250);
    for (int i = 0; i < n; i++) {
        pairs.push_back(bwa_tests::gen_random_pair(rng, qlen_d(rng), rlen_d(rng)));
    }
    return pairs;
}

} // namespace

TEST_CASE("kswv::getScores8 matches scalar ksw_align2 on 10k random + curated edge pairs"
          * doctest::test_suite("unit/kswv")) {

    auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);

    std::mt19937 rng(42);
    auto pairs = build_edge_cases(rng);
    auto bulk  = build_bulk_random(rng, 10000);
    pairs.insert(pairs.end(), bulk.begin(), bulk.end());

    std::vector<kswr_t> scalar_aln;
    scalar_aln.reserve(pairs.size());
    for (const auto &p : pairs) {
        scalar_aln.push_back(bwa_tests::run_scalar_ksw(p, mat));
    }

    auto batched = bwa_tests::run_kswv_batch(pairs, mat);
    REQUIRE(batched.size() == pairs.size());

    int score_mism = 0, coord_mism = 0, score2_mism = 0;
    for (size_t i = 0; i < pairs.size(); i++) {
        CAPTURE(i);
        CAPTURE(pairs[i].tag);
        CAPTURE(scalar_aln[i].score);
        CAPTURE(batched[i].score);
        const bool score_ok  = bwa_tests::kswr_score_eq(scalar_aln[i], batched[i]);
        const bool coord_ok  = bwa_tests::kswr_coords_eq(scalar_aln[i], batched[i]);
        const bool score2_ok = bwa_tests::kswr_score2_eq(scalar_aln[i], batched[i]);
        // Only emit per-pair CHECKs on mismatch — at 10k pairs the doctest
        // log would otherwise carry ~30k passing assertions per run, which
        // dwarfs the actual signal on a regression.
        if (!score_ok)  { ++score_mism;  CHECK(score_ok); }
        if (!coord_ok)  { ++coord_mism;  CHECK(coord_ok); }
        if (!score2_ok) { ++score2_mism; CHECK(score2_ok); }
    }

    MESSAGE("kswv vs scalar: score_mism=" << score_mism
            << " coord_mism=" << coord_mism
            << " score2_mism=" << score2_mism
            << " over " << pairs.size() << " pairs");
    // Aggregate gates so a regression with no per-pair CHECK still fails
    // the test (e.g. if all 10k pairs happen to mismatch in coords only,
    // those per-pair CHECKs cover it; this catches counting drift too).
    CHECK(score_mism == 0);
    CHECK(coord_mism == 0);
    CHECK(score2_mism == 0);
}

TEST_CASE("kswv::getScores16 matches scalar ksw_align2 on 10k random + curated edge pairs"
          * doctest::test_suite("unit/kswv")) {

    auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);

    std::mt19937 rng(42);
    auto pairs = build_edge_cases(rng);
    auto bulk  = build_bulk_random(rng, 10000);
    pairs.insert(pairs.end(), bulk.begin(), bulk.end());

    std::vector<kswr_t> scalar_aln;
    scalar_aln.reserve(pairs.size());
    for (const auto &p : pairs) {
        scalar_aln.push_back(bwa_tests::run_scalar_ksw(p, mat));
    }

    // use16 = true drives the 16-bit kernel (kswv256_16 / kswv512_16 /
    // kswv_neon_16) regardless of the production l_ms*a routing.
    auto batched = bwa_tests::run_kswv_batch(pairs, mat,
                                             bwa_tests::DEFAULT_GAP_OPEN,
                                             bwa_tests::DEFAULT_GAP_EXTEND,
                                             0, /*use16=*/true);
    REQUIRE(batched.size() == pairs.size());

    int score_mism = 0, coord_mism = 0, score2_mism = 0;
    for (size_t i = 0; i < pairs.size(); i++) {
        CAPTURE(i);
        CAPTURE(pairs[i].tag);
        CAPTURE(scalar_aln[i].score);
        CAPTURE(batched[i].score);
        const bool score_ok  = bwa_tests::kswr_score_eq(scalar_aln[i], batched[i]);
        const bool coord_ok  = bwa_tests::kswr_coords_eq(scalar_aln[i], batched[i]);
        const bool score2_ok = bwa_tests::kswr_score2_eq(scalar_aln[i], batched[i]);
        if (!score_ok)  { ++score_mism;  CHECK(score_ok); }
        if (!coord_ok)  { ++coord_mism;  CHECK(coord_ok); }
        if (!score2_ok) { ++score2_mism; CHECK(score2_ok); }
    }

    MESSAGE("kswv16 vs scalar: score_mism=" << score_mism
            << " coord_mism=" << coord_mism
            << " score2_mism=" << score2_mism
            << " over " << pairs.size() << " pairs");
    CHECK(score_mism == 0);
    CHECK(coord_mism == 0);
    CHECK(score2_mism == 0);
}

TEST_CASE("kswv::getScores16 matches scalar on high scores that overflow the 8-bit kernel"
          * doctest::test_suite("unit/kswv")) {
    // match=14 with long, near-exact pairs pushes alignment scores well
    // past 255, where the 8-bit kernel saturates. The 16-bit kernel must
    // still reproduce scalar ksw_align2 exactly. match=14 also makes
    // min_seed_len*match = 19*14 = 266 >= 250, so default_xtra_flags drops
    // KSW_XBYTE for BOTH the scalar reference and the batch — i.e. both run
    // the word path, exactly the case production routes to getScores16.
    auto mat = bwa_tests::build_scoring_matrix(14, 8, 1);

    std::mt19937 rng(1234);
    std::vector<bwa_tests::TestPair> pairs;
    std::uniform_int_distribution<int> qlen_d(80, 128);
    std::uniform_int_distribution<int> rlen_d(150, 250);
    for (int i = 0; i < 5000; i++) {
        pairs.push_back(bwa_tests::gen_sub_cluster_pair(
            rng, qlen_d(rng), rlen_d(rng), 40, 2));
    }

    std::vector<kswr_t> scalar_aln;
    scalar_aln.reserve(pairs.size());
    for (const auto &p : pairs) {
        scalar_aln.push_back(bwa_tests::run_scalar_ksw(p, mat));
    }

    auto batched = bwa_tests::run_kswv_batch(pairs, mat,
                                             bwa_tests::DEFAULT_GAP_OPEN,
                                             bwa_tests::DEFAULT_GAP_EXTEND,
                                             0, /*use16=*/true);
    REQUIRE(batched.size() == pairs.size());

    int score_mism = 0, coord_mism = 0, score2_mism = 0, over255 = 0;
    for (size_t i = 0; i < pairs.size(); i++) {
        CAPTURE(i);
        CAPTURE(pairs[i].tag);
        CAPTURE(scalar_aln[i].score);
        CAPTURE(batched[i].score);
        if (scalar_aln[i].score > 255) ++over255;
        const bool score_ok  = bwa_tests::kswr_score_eq(scalar_aln[i], batched[i]);
        const bool coord_ok  = bwa_tests::kswr_coords_eq(scalar_aln[i], batched[i]);
        const bool score2_ok = bwa_tests::kswr_score2_eq(scalar_aln[i], batched[i]);
        if (!score_ok)  { ++score_mism;  CHECK(score_ok); }
        if (!coord_ok)  { ++coord_mism;  CHECK(coord_ok); }
        if (!score2_ok) { ++score2_mism; CHECK(score2_ok); }
    }

    MESSAGE("kswv16 high-score: over255=" << over255
            << " score_mism=" << score_mism
            << " coord_mism=" << coord_mism
            << " score2_mism=" << score2_mism
            << " over " << pairs.size() << " pairs");
    // Guard the test's own premise: at least some pairs must exceed the
    // 8-bit ceiling, else this wouldn't be testing the 16-bit range.
    CHECK(over255 > 0);
    CHECK(score_mism == 0);
    CHECK(coord_mism == 0);
    CHECK(score2_mism == 0);
}

TEST_CASE("kswv handles every curated edge case identically to scalar"
          * doctest::test_suite("unit/kswv")) {

    auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);
    std::mt19937 rng(42);

    // Helper lambda: run scalar+batched on a single pair and check all
    // three comparators (score, coords, score2). Mirrors the bulk test so a
    // regression in any of those dimensions is pinpointed by SUBCASE name.
    auto check_pair_parity = [&](const bwa_tests::TestPair &p) {
        auto s = bwa_tests::run_scalar_ksw(p, mat);
        auto b = bwa_tests::run_kswv_batch({p}, mat);
        CHECK(bwa_tests::kswr_score_eq(s, b[0]));
        CHECK(bwa_tests::kswr_coords_eq(s, b[0]));
        CHECK(bwa_tests::kswr_score2_eq(s, b[0]));
    };

    SUBCASE("exact match len 50")   { check_pair_parity(bwa_tests::gen_exact_match_pair(50)); }
    SUBCASE("exact match len 100")  { check_pair_parity(bwa_tests::gen_exact_match_pair(100)); }
    SUBCASE("all mismatch len 50")  { check_pair_parity(bwa_tests::gen_all_mismatch_pair(50)); }
    SUBCASE("homopolymer A")        { check_pair_parity(bwa_tests::gen_homopolymer_pair(50, 0)); }
    SUBCASE("sub cluster len 10")   { check_pair_parity(bwa_tests::gen_sub_cluster_pair(rng, 100, 150, 40, 10)); }
    SUBCASE("20% N bases")          { check_pair_parity(bwa_tests::gen_with_n_bases_pair(rng, 100, 150, 20)); }
}

// The two-row (KSWV_NEON_U8_CELL_PAIR) and one-row (KSWV_NEON_U8_CELL) macros
// restate the same u8-rescue recurrence, so a change applied to one but not the
// other would diverge silently. rescue_rowpair_enabled() reads BWA3_RESCUE_ROWPAIR
// on every call (the NEON u8 kernel only), so this drives one batch through
// getScores8 with pairing off then on. Two gates:
//   * strict all-field A/B (one-row == two-row) catches drift between the macros;
//   * an INDEPENDENT scalar oracle on the one-row arm — the shared KSWV_U8_EPILOGUE
//     means a bug there shifts BOTH arms together, so the A/B alone would stay
//     green; the existing scalar-oracle cases run only the default (pairing-on)
//     mode, leaving the one-row path without a reference otherwise.
TEST_CASE("kswv u8 rescue: BWA3_RESCUE_ROWPAIR off == on, and one-row matches scalar"
          * doctest::test_suite("unit/kswv")) {

#if !defined(__ARM_NEON) && !defined(__aarch64__)
    // BWA3_RESCUE_ROWPAIR gates only the NEON u8 rescue kernel; on x86 tiers
    // run_kswv_batch dispatches to kswv256/512_u8 where both arms run identical
    // code, so this case would pass without exercising the row-pair logic.
    MESSAGE("skipped: BWA3_RESCUE_ROWPAIR affects the NEON u8 rescue kernel only");
    return;
#else
    auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);

    std::mt19937 rng(1234);
    auto pairs = build_edge_cases(rng);
    // A few hundred bulk pairs is plenty: macro drift surfaces on the first
    // affected cell, and each batch sweeps the full DP twice, so keep well under
    // the ~100 ms per-case budget (the scalar-oracle cases above use 10k).
    auto bulk  = build_bulk_random(rng, 300);
    pairs.insert(pairs.end(), bulk.begin(), bulk.end());

    // Independent oracle: scalar ksw_align2 for every pair.
    std::vector<kswr_t> scalar_aln;
    scalar_aln.reserve(pairs.size());
    for (const auto &p : pairs) scalar_aln.push_back(bwa_tests::run_scalar_ksw(p, mat));

    // Save/restore the caller's env so doctest run order can't leak the override
    // into a sibling case that assumes the default (pairing on).
    const char *saved = getenv("BWA3_RESCUE_ROWPAIR");
    const std::string saved_val = saved ? saved : "";
    const bool had_saved = saved != nullptr;

    setenv("BWA3_RESCUE_ROWPAIR", "0", 1);
    auto one_row = bwa_tests::run_kswv_batch(pairs, mat);
    setenv("BWA3_RESCUE_ROWPAIR", "1", 1);
    auto two_row = bwa_tests::run_kswv_batch(pairs, mat);

    if (had_saved) setenv("BWA3_RESCUE_ROWPAIR", saved_val.c_str(), 1);
    else           unsetenv("BWA3_RESCUE_ROWPAIR");

    REQUIRE(one_row.size() == pairs.size());
    REQUIRE(two_row.size() == pairs.size());

    int drift = 0, oracle_mism = 0;
    for (size_t i = 0; i < pairs.size(); i++) {
        const kswr_t &a = one_row[i];
        const kswr_t &b = two_row[i];
        // (1) Strict all-field A/B: rowpair is the same kernel row-blocked, so the
        // two arms must match exactly -- not the scalar-vs-batched tolerant compare.
        const bool eq = a.score == b.score && a.te == b.te && a.qe == b.qe
                     && a.score2 == b.score2 && a.te2 == b.te2
                     && a.tb == b.tb && a.qb == b.qb;
        if (!eq) {
            ++drift;
            CAPTURE(i); CAPTURE(pairs[i].tag);
            CAPTURE(a.score); CAPTURE(b.score);
            CAPTURE(a.te); CAPTURE(b.te); CAPTURE(a.qe); CAPTURE(b.qe);
            CHECK(eq);
        }
        // (2) Independent oracle on the one-row arm: a bug in the shared epilogue
        // shifts both arms together, so check one_row against scalar ksw_align2
        // (tolerant compare — scalar vs batched has known-benign discrepancies).
        const bool o_score  = bwa_tests::kswr_score_eq(scalar_aln[i], a);
        const bool o_coord  = bwa_tests::kswr_coords_eq(scalar_aln[i], a);
        const bool o_score2 = bwa_tests::kswr_score2_eq(scalar_aln[i], a);
        if (!(o_score && o_coord && o_score2)) {
            ++oracle_mism;
            CAPTURE(i); CAPTURE(pairs[i].tag);
            CAPTURE(scalar_aln[i].score); CAPTURE(a.score);
            CHECK(o_score); CHECK(o_coord); CHECK(o_score2);
        }
    }
    MESSAGE("rowpair off-vs-on drift=" << drift << ", one-row vs scalar mism="
            << oracle_mism << " over " << pairs.size() << " pairs");
    CHECK(drift == 0);
    CHECK(oracle_mism == 0);
#endif
}

// The two-row sweep recovers the query-end column either inline (a per-cell
// strict-greater blend of a carried column counter) or lazily (row i's H written
// back into the dead diagonal slot H0[j], both rows' running maxima checkpointed
// per QE_BLK, and the shared KSWV_U8_EPILOGUE rescan run for both rows).
// rescue_lazyqe_enabled() reads BWA3_RESCUE_LAZYQE on every call, so this drives
// one batch through the paired kernel with the lazy path off then on. Both arms
// run the same recurrence, so the A/B is strict all-field; the inline arm is
// additionally checked against the scalar oracle so a shared-epilogue bug cannot
// hide by shifting both arms together. Pairing is forced ON for both arms
// (BWA3_RESCUE_ROWPAIR=1) so the lazy toggle is the only difference.
TEST_CASE("kswv u8 rescue: BWA3_RESCUE_LAZYQE off == on in the two-row sweep, and inline matches scalar"
          * doctest::test_suite("unit/kswv")) {

#if !defined(__ARM_NEON) && !defined(__aarch64__)
    MESSAGE("skipped: BWA3_RESCUE_LAZYQE affects the NEON u8 rescue kernel only");
    return;
#else
    auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);

    std::mt19937 rng(4321);
    auto pairs = build_edge_cases(rng);
    auto bulk  = build_bulk_random(rng, 300);
    pairs.insert(pairs.end(), bulk.begin(), bulk.end());

    std::vector<kswr_t> scalar_aln;
    scalar_aln.reserve(pairs.size());
    for (const auto &p : pairs) scalar_aln.push_back(bwa_tests::run_scalar_ksw(p, mat));

    // Save/restore both env vars so doctest run order cannot leak an override
    // into a sibling case that assumes the defaults (pairing on, lazy on).
    const char *saved_rp = getenv("BWA3_RESCUE_ROWPAIR");
    const std::string saved_rp_val = saved_rp ? saved_rp : "";
    const bool had_rp = saved_rp != nullptr;
    const char *saved_lq = getenv("BWA3_RESCUE_LAZYQE");
    const std::string saved_lq_val = saved_lq ? saved_lq : "";
    const bool had_lq = saved_lq != nullptr;

    setenv("BWA3_RESCUE_ROWPAIR", "1", 1);
    setenv("BWA3_RESCUE_LAZYQE", "0", 1);
    auto inline_qe = bwa_tests::run_kswv_batch(pairs, mat);
    setenv("BWA3_RESCUE_LAZYQE", "1", 1);
    auto lazy_qe = bwa_tests::run_kswv_batch(pairs, mat);

    if (had_rp) setenv("BWA3_RESCUE_ROWPAIR", saved_rp_val.c_str(), 1);
    else        unsetenv("BWA3_RESCUE_ROWPAIR");
    if (had_lq) setenv("BWA3_RESCUE_LAZYQE", saved_lq_val.c_str(), 1);
    else        unsetenv("BWA3_RESCUE_LAZYQE");

    REQUIRE(inline_qe.size() == pairs.size());
    REQUIRE(lazy_qe.size() == pairs.size());

    int drift = 0, oracle_mism = 0;
    for (size_t i = 0; i < pairs.size(); i++) {
        const kswr_t &a = inline_qe[i];
        const kswr_t &b = lazy_qe[i];
        const bool eq = a.score == b.score && a.te == b.te && a.qe == b.qe
                     && a.score2 == b.score2 && a.te2 == b.te2
                     && a.tb == b.tb && a.qb == b.qb;
        if (!eq) {
            ++drift;
            CAPTURE(i); CAPTURE(pairs[i].tag);
            CAPTURE(a.score); CAPTURE(b.score);
            CAPTURE(a.te); CAPTURE(b.te); CAPTURE(a.qe); CAPTURE(b.qe);
            CAPTURE(a.tb); CAPTURE(b.tb); CAPTURE(a.qb); CAPTURE(b.qb);
            CHECK(eq);
        }
        const bool o_score  = bwa_tests::kswr_score_eq(scalar_aln[i], a);
        const bool o_coord  = bwa_tests::kswr_coords_eq(scalar_aln[i], a);
        const bool o_score2 = bwa_tests::kswr_score2_eq(scalar_aln[i], a);
        if (!(o_score && o_coord && o_score2)) {
            ++oracle_mism;
            CAPTURE(i); CAPTURE(pairs[i].tag);
            CAPTURE(scalar_aln[i].score); CAPTURE(a.score);
            CHECK(o_score); CHECK(o_coord); CHECK(o_score2);
        }
    }
    MESSAGE("lazyqe off-vs-on drift=" << drift << ", inline vs scalar mism="
            << oracle_mism << " over " << pairs.size() << " pairs");
    CHECK(drift == 0);
    CHECK(oracle_mism == 0);
#endif
}

#endif // BWA_TESTS_HAVE_KSWV
