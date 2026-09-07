// test/unit/test_ksw_global2_wave.cpp
//
// Differential byte-identity test for the anti-diagonal (wavefront) SIMD
// ksw_global2 kernel (src/ksw_global2_wave.h) versus the scalar reference.
//
// The public ksw_global2 dispatcher routes a given (input, w) to exactly ONE
// path — the wavefront kernel when a CIGAR is requested, m==5, and
// w >= KSW_WAVE_WMIN; the scalar path otherwise — so a differential test needs
// a second, unconditional entry to the scalar oracle: ksw_global2_scalar_ref
// (a thin test hook in ksw.cpp). For each input we run both and assert the
// score, n_cigar, and every CIGAR op are identical.
//
// Tier behaviour: on a SIMD tier (AVX-512/AVX2/NEON) a wide enough w drives the
// wavefront kernel, so this is a real kernel-vs-scalar gate. On a scalar tier
// (sse41/sse42/avx, or any arch where the header defines no kernel) ksw_global2
// is itself scalar, so the comparison is scalar-vs-scalar and trivially passes.
// Full-tier coverage therefore comes from the CI matrix building this per tier,
// exactly as test_kswv_correctness.cpp does.
//
// The proof of byte-identity lives in the LEMMA 1-10 comment atop
// src/ksw_global2_wave.h; this test is its empirical gate.

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "ksw.h"          // ksw_global2, ksw_global2_scalar_ref, ksw_g2_wave_exec_count (extern "C")
#include "scoring.h"      // build_scoring_matrix
#include "seqpair.h"      // TestPair
#include "seqpair_gen.h"  // deterministic generators
#include "simd_dispatch.h"  // bwamem3_simd_init, bwamem3_simd_tier, BWAMEM3_TIER_*

namespace {

using bwa_tests::TestPair;

// One ksw_global2 result: score, the declared op count, and the emitted CIGAR
// (op|len words). n_cigar is retained and compared directly — the copied vector
// length is only a proxy that a (n_cigar>0, cigar==nullptr) result would slip
// past as an empty CIGAR.
struct G2Result {
    int                   score;
    int                   n_cigar;
    std::vector<uint32_t> cigar;
};

// ksw_global2 / ksw_global2_scalar_ref have C language linkage (ksw.h wraps
// them in extern "C"), which is part of the function type in C++17 — so the
// pointer typedef must be declared extern "C" too, or the assignment is
// ill-formed.
extern "C" {
using Ksw2Fn = int (*)(int, const uint8_t *, int, const uint8_t *, int,
                       const int8_t *, int, int, int, int, int, int *, uint32_t **);
}

// Run one ksw_global2-shaped entry point on a pair at band half-width w.
// query := pair.qry, target := pair.ref (matching ksw_global2's parameter order).
G2Result run_g2(Ksw2Fn fn, const TestPair &p, const int8_t *mat, int w,
                int o_del, int e_del, int o_ins, int e_ins) {
    int       n_cigar = 0;
    uint32_t *cigar   = nullptr;
    const int score = fn(static_cast<int>(p.qry.size()), p.qry.data(),
                         static_cast<int>(p.ref.size()), p.ref.data(),
                         5, mat, o_del, e_del, o_ins, e_ins, w, &n_cigar, &cigar);
    G2Result r;
    r.score = score;
    r.n_cigar = n_cigar;
    // Contract: a positive op count must come with a non-null buffer, and a null
    // buffer must report zero ops. Assert it before copying so a broken
    // (n_cigar>0, cigar==nullptr) result fails loudly instead of masquerading as
    // an empty CIGAR.
    REQUIRE((n_cigar > 0) == (cigar != nullptr));
    if (n_cigar > 0) r.cigar.assign(cigar, cigar + n_cigar);
    free(cigar);
    return r;
}

// The feasibility floor every production caller enforces (bwa.cpp): a band
// narrower than this cannot connect the two ends and is out of scope for the
// byte-identity claim, so every w we test is clamped up to it.
int band_floor(const TestPair &p) {
    const int d = std::abs(static_cast<int>(p.ref.size()) - static_cast<int>(p.qry.size()));
    return d + 3;
}

// Curated edge cases. The regimes the proof itself flags as delicate:
// N/ambiguous bases (score-table indices 20-24), bands right at the per-arch
// KSW_WAVE_WMIN crossover vs wide enough to force the interior fast-path chunk,
// |tlen-qlen| near the band floor, and small/degenerate targets.
std::vector<TestPair> build_edge_cases(std::mt19937 &rng) {
    using namespace bwa_tests;
    std::vector<TestPair> pairs;
    pairs.push_back(gen_exact_match_pair(50));
    pairs.push_back(gen_exact_match_pair(100));
    pairs.push_back(gen_all_mismatch_pair(50));
    pairs.push_back(gen_homopolymer_pair(60, 0));
    pairs.push_back(gen_homopolymer_pair(60, 3));
    pairs.push_back(gen_sub_cluster_pair(rng, 100, 150, 40, 3));
    pairs.push_back(gen_sub_cluster_pair(rng, 120, 130, 40, 10));   // |tlen-qlen| small
    pairs.push_back(gen_with_n_bases_pair(rng, 100, 150, 5));       // ambiguous bases
    pairs.push_back(gen_with_n_bases_pair(rng, 100, 150, 20));
    pairs.push_back(gen_tandem_repeat_pair(rng, 100, 150));
    // Degenerate / small targets (exercise the boundary and tlen-small corners).
    pairs.push_back(TestPair{{0}, {0}, "len1-match"});
    pairs.push_back(TestPair{{0, 1, 2, 3}, {0, 1, 2, 3}, "len4-eq"});
    pairs.push_back(TestPair{{}, {0, 1, 2}, "qlen0-tlen3"});        // empty query
    pairs.push_back(TestPair{{0, 1, 2}, {}, "tlen0-qlen3"});        // empty target
    pairs.push_back(TestPair{{0, 1, 2}, {0, 1, 2, 3, 0}, "tlt-qgt"});
    return pairs;
}

std::vector<TestPair> build_bulk_random(std::mt19937 &rng, int n) {
    std::vector<TestPair> pairs;
    pairs.reserve(n);
    std::uniform_int_distribution<int> qlen_d(30, 200);
    std::uniform_int_distribution<int> rlen_d(30, 260);
    for (int i = 0; i < n; i++)
        pairs.push_back(bwa_tests::gen_random_pair(rng, qlen_d(rng), rlen_d(rng)));
    return pairs;
}

// Band widths to test per pair, each clamped up to the pair's feasibility floor.
// Spans below/at/above the per-arch crossovers (int16 10/10/16, int32 16/26/52
// across AVX-512/AVX2/NEON) plus wide bands that force the interior fast-path
// chunk, so both the boundary-only and the interior code paths are exercised
// regardless of the compiled tier. 9/10 straddle the AVX2 / AVX-512 int16
// crossover exactly (KSW_WAVE16_WMIN = 10 on those tiers; NEON's is 16, covered
// by the 16 entry) and 32/33 the AVX2 tier's former value (33), so byte-identity
// -- and, via run_parity's routing gate, which kernel ran -- is asserted right
// at the current boundary and at the old one the crossover was lowered from.
const int kBandTargets[] = {3, 9, 10, 16, 20, 26, 32, 33, 64, 200};

// One (o_del,e_del,o_ins,e_ins) tuple per index; asymmetric gaps included so the
// del/ins direction bytes are exercised independently.
struct GapSet { int o_del, e_del, o_ins, e_ins; };
const GapSet kGapSets[] = {{5, 2, 5, 2}, {6, 1, 6, 1}, {4, 3, 7, 1}};

bool cigar_eq(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b) {
    return a == b;
}

// Whether the RUNNING tier has a wavefront ksw_global2 kernel. On a scalar tier
// (sse41/sse42/avx) ksw_global2 is itself scalar, so run_parity compares
// scalar-vs-scalar and proves nothing about the SIMD path — the exec-count gate
// below is therefore asserted only on these tiers.
bool tier_has_wavefront() {
    // bwamem3_simd_tier() reports BWAMEM3_TIER_NONE until the dispatcher is
    // initialised. On x86 the dispatched exec-count hooks initialise it as a side
    // effect, but on NEON (single tier, hooks called directly) nothing does, so
    // without this call the probe read "no wavefront" and every routing gate below
    // was skipped on arm64. Idempotent (std::call_once inside).
    bwamem3_simd_init();
    const int tier = bwamem3_simd_tier();
    return tier == BWAMEM3_TIER_AVX2 || tier == BWAMEM3_TIER_AVX512BW ||
           tier == BWAMEM3_TIER_NEON;
}

// Differential sweep over the (gap-set x band-width) grid. The gap and width
// dimensions are doctest SUBCASEs so a mismatch — or a REQUIRE abort inside
// run_g2 — isolates the single (gap, w) variant that broke instead of taking
// down the whole grid, and each variant is reported independently. doctest
// re-enters this body once per leaf, so the accumulators below scope naturally
// to one (gap, w) variant. `expect_int16` is set by callers whose pairs are all
// int16-safe, adding a gate that the narrow int16 kernel actually ran rather
// than every pair falling back to the int32 wave; `expect_int16_rejected` by
// callers whose pairs all fail the int16 range gate, asserting the int16 kernel
// never ran (the fallback really is the int32 wave or scalar).
void run_parity(const std::vector<TestPair> &pairs, const int8_t *mat,
                bool expect_int16 = false, bool expect_int16_rejected = false) {
    for (const GapSet &g : kGapSets) {
        const std::string gname = "gap[" + std::to_string(g.o_del) + "," +
            std::to_string(g.e_del) + "," + std::to_string(g.o_ins) + "," +
            std::to_string(g.e_ins) + "]";
        SUBCASE(gname.c_str()) {
            for (int wt : kBandTargets) {
                const std::string wname = "w=" + std::to_string(wt);
                SUBCASE(wname.c_str()) {
                    int score_mism = 0, ncigar_mism = 0, cigar_mism = 0;
                    long compared = 0;
                    const unsigned long wave_before   = ksw_g2_wave_exec_count();
                    const unsigned long wave16_before = ksw_g2_wave16_exec_count();
                    for (const auto &p : pairs) {
                        const int floor = band_floor(p);
                        const int w = wt > floor ? wt : floor;
                        const G2Result got = run_g2(ksw_global2, p, mat, w,
                                                    g.o_del, g.e_del, g.o_ins, g.e_ins);
                        const G2Result ref = run_g2(ksw_global2_scalar_ref, p, mat, w,
                                                    g.o_del, g.e_del, g.o_ins, g.e_ins);
                        ++compared;
                        // Only emit per-case CHECKs on mismatch — each leaf is
                        // hundreds of comparisons; passing assertions would bury
                        // the signal on a regression. Aggregate gates below still
                        // fail even if a per-case CHECK is somehow skipped.
                        const bool s_ok = got.score == ref.score;
                        const bool n_ok = got.n_cigar == ref.n_cigar;
                        const bool c_ok = cigar_eq(got.cigar, ref.cigar);
                        if (!s_ok || !n_ok || !c_ok) {
                            CAPTURE(p.tag);
                            CAPTURE(w);
                            CAPTURE(g.o_del); CAPTURE(g.e_del); CAPTURE(g.o_ins); CAPTURE(g.e_ins);
                            CAPTURE(got.score); CAPTURE(ref.score);
                            CAPTURE(got.n_cigar); CAPTURE(ref.n_cigar);
                            CAPTURE(got.cigar.size()); CAPTURE(ref.cigar.size());
                        }
                        if (!s_ok) { ++score_mism;  CHECK(s_ok); }
                        if (!n_ok) { ++ncigar_mism; CHECK(n_ok); }
                        if (!c_ok) { ++cigar_mism;  CHECK(c_ok); }
                    }
                    MESSAGE("wavefront ksw_global2 vs scalar @ " << gname << " " << wname
                            << ": score_mism=" << score_mism << " n_cigar_mism=" << ncigar_mism
                            << " cigar_mism=" << cigar_mism << " over " << compared << " pairs");
                    CHECK(score_mism == 0);
                    CHECK(ncigar_mism == 0);
                    CHECK(cigar_mism == 0);
                    // Routing gate against the tier's OWN compiled crossovers (the
                    // test hooks return KSW_WAVE_WMIN / KSW_WAVE16_WMIN for the
                    // active tier, 0 where it has no such kernel), so the gate is
                    // exact on every tier instead of a hard-coded wide width:
                    //  - wt >= the int32 crossover: some wavefront kernel must have
                    //    run for every pair (int16 where the range fits, else
                    //    int32) -- an unmoved counter means the run silently
                    //    degenerated to scalar-vs-scalar;
                    //  - int16-safe pairs at wt >= the int16 crossover: the narrow
                    //    int16 kernel itself must have run -- this is what pins the
                    //    lowered AVX2 crossover (10): a constant that drifted back
                    //    up would leave w in [10, old) scalar and fail here;
                    //  - int16-rejected pairs: the int16 kernel must never run.
                    // Every pair's actual w is >= wt (the floor only raises it), so
                    // a gate keyed on wt holds for the whole set. On a scalar tier
                    // both crossovers read 0 and no gate applies.
                    if (tier_has_wavefront()) {
                        const int wmin32 = ksw_g2_wave_wmin();
                        const int wmin16 = ksw_g2_wave16_wmin();
                        REQUIRE(wmin32 > 0);
                        REQUIRE(wmin16 > 0);
                        if (wt >= wmin32)
                            CHECK(ksw_g2_wave_exec_count() > wave_before);
                        if (expect_int16 && wt >= wmin16) {
                            CHECK(ksw_g2_wave_exec_count() > wave_before);
                            CHECK(ksw_g2_wave16_exec_count() > wave16_before);
                        }
                        if (expect_int16_rejected)
                            CHECK(ksw_g2_wave16_exec_count() == wave16_before);
                    }
                }
            }
        }
    }
}

}  // namespace

TEST_CASE("wavefront ksw_global2 is byte-identical to scalar on random + edge pairs"
          * doctest::test_suite("unit/ksw_global2")) {
    const auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);
    std::mt19937 rng(42);
    auto pairs = build_edge_cases(rng);
    // A representative random sample (not an exhaustive fuzz — that lives in the
    // standalone kernel gate) so the committed test stays inside the unit-test
    // time budget while still crossing every per-arch width threshold via
    // kBandTargets. The edge cases carry the delicate corners.
    auto bulk  = build_bulk_random(rng, 80);
    pairs.insert(pairs.end(), bulk.begin(), bulk.end());
    run_parity(pairs, mat.data());
}

// Short queries + a small mismatch penalty (b=1): the whole score span fits
// int16, so the narrow-width int16 tier is exercised on the dispatcher's fast
// paths. Byte-identity is checked against the scalar oracle exactly as above, so
// whichever tier the dispatcher picks must match.
TEST_CASE("wavefront ksw_global2 byte-identical on short/low-penalty pairs (int16 tier)"
          * doctest::test_suite("unit/ksw_global2")) {
    const auto mat = bwa_tests::build_scoring_matrix(1, 1, 1);   // b=1 keeps the score span small
    std::mt19937 rng(1234);
    std::vector<bwa_tests::TestPair> pairs;
    std::uniform_int_distribution<int> qd(20, 90), rd(20, 95);   // 2x76bp-scale short reads
    for (int i = 0; i < 200; i++) pairs.push_back(bwa_tests::gen_random_pair(rng, qd(rng), rd(rng)));
    pairs.push_back(bwa_tests::gen_with_n_bases_pair(rng, 76, 80, 6));   // N bases in the int16 regime
    pairs.push_back(bwa_tests::gen_exact_match_pair(76));
    // Equal-length pairs (band_floor == 3) so the exact crossover widths in
    // kBandTargets (9/10/32/33) run un-clamped in the int16-SAFE range too,
    // mirroring the int16-fallback case below: the random pairs above have large
    // |tlen-qlen|, so their floor clamps those narrow widths away. all_mismatch
    // and homopolymer keep qlen == tlen (unlike gen_exact_match_pair, whose ref
    // is 2x the query), so the routing gate pins the narrow int16 kernel right at
    // the lowered AVX2 crossover and at the old boundary it was lowered from.
    pairs.push_back(bwa_tests::gen_all_mismatch_pair(76));
    pairs.push_back(bwa_tests::gen_homopolymer_pair(76, 0));
    // These pairs are all short/low-penalty, so their score range provably fits
    // int16 — assert the narrow int16 kernel actually ran, not just some wave.
    run_parity(pairs, mat.data(), /*expect_int16=*/true);
}

// Long queries push H_upper = qlen*A + A past the int16 gate's ceiling
// (32767 - SLACK = 32703), so ksw_g2_wave16_safe returns false and the
// dispatcher must fall back to the int32 wave (w >= its WMIN) or scalar. This is
// the ONLY case in the suite that exercises the int32 wavefront kernel: every
// wavefront tier now has KSW_WAVE16_WMIN <= KSW_WAVE_WMIN (AVX2 10/26, AVX-512
// 10/16, NEON 16/52), so the int16 tier otherwise always wins the
// safe-and-wide-enough race. Confirms the cascade
// stays byte-identical when the int16 tier is rejected. (The score matrix is
// int8_t, so |value| <= 127; with the max match score A=127, qlen>256 is what
// takes H_upper over the ceiling — hence the >256bp query lengths here.)
TEST_CASE("wavefront ksw_global2 byte-identical when int16 falls back to int32/scalar"
          * doctest::test_suite("unit/ksw_global2")) {
    // A=127 (max int8 match): qlen>=270 gives H_upper >= 270*127+127 = 34417 > 32703,
    // so the int16 gate rejects every pair below (scores stay well inside int32).
    const auto mat = bwa_tests::build_scoring_matrix(127, 127, 127);
    std::mt19937 rng(9);
    std::vector<bwa_tests::TestPair> pairs;
    std::uniform_int_distribution<int> qd(270, 360), rd(270, 370);
    for (int i = 0; i < 25; i++) pairs.push_back(bwa_tests::gen_random_pair(rng, qd(rng), rd(rng)));
    // Equal-length pairs (band_floor == 3) so the exact crossover widths in
    // kBandTargets (9/10/32/33) run without being clamped up in the int16-
    // fallback range too. The random pairs above have large |tlen-qlen|, so
    // their floor clamps those narrow widths away; these prove byte-identity at
    // the boundaries on the int32-wave/scalar cascade the int16 gate falls back
    // to for these long queries. all_mismatch and homopolymer keep qlen == tlen
    // (unlike gen_exact_match_pair, whose ref is 2x the query); both stay >= 270
    // so H_upper still clears the int16 gate's ceiling.
    pairs.push_back(bwa_tests::gen_all_mismatch_pair(300));
    pairs.push_back(bwa_tests::gen_homopolymer_pair(300, 0));
    run_parity(pairs, mat.data(), /*expect_int16=*/false, /*expect_int16_rejected=*/true);
}

// Windowed high-water decay of the direction-byte store zr (see
// KswWaveScratch::ensure()). One wide-band spike grows zr to a large high-water;
// a subsequent flood of narrow calls must release it back toward the floor. This
// proves the retention policy actually fires — a grow-only buffer would keep the
// spike's tens-of-MB footprint pinned per thread for the process lifetime — and
// that every call stays byte-identical to the scalar oracle across the shrink. On
// a scalar tier zr is never used (ksw_global2 is itself scalar), so the capacity
// assertions are gated on a wavefront runtime tier.
TEST_CASE("wavefront ksw_global2 zr scratch decays after a wide-band spike"
          * doctest::test_suite("unit/ksw_global2")) {
    const auto mat = bwa_tests::build_scoring_matrix(1, 4, 1);
    std::mt19937 rng(4242);

    // Run one pair through both the dispatcher and the scalar oracle at band w and
    // assert byte-identity, so the shrink can never quietly corrupt output.
    auto check_parity = [&](const bwa_tests::TestPair &p, int w) {
        const G2Result got = run_g2(ksw_global2, p, mat.data(), w, 5, 2, 5, 2);
        const G2Result ref = run_g2(ksw_global2_scalar_ref, p, mat.data(), w, 5, 2, 5, 2);
        CHECK(got.score == ref.score);
        CHECK(got.n_cigar == ref.n_cigar);
        CHECK(cigar_eq(got.cigar, ref.cigar));
    };

    // Wide-band spike: qlen=tlen=2400, w=2400 => zneed ~ 11 MiB (band*(tlen+qlen)),
    // well above the 4 MiB shrink floor, so a later all-narrow window must release it.
    const bwa_tests::TestPair spike = bwa_tests::gen_random_pair(rng, 2400, 2400);
    check_parity(spike, 2400);
    const unsigned long cap_after_spike = ksw_g2_wave_zr_capacity();

    // Flood of narrow calls (zneed ~ a few KB). 200 spans multiple decay windows
    // regardless of the window phase left by earlier tests, so the spike's
    // footprint is guaranteed to flush out of the window max and be released.
    for (int i = 0; i < 200; ++i)
        check_parity(bwa_tests::gen_random_pair(rng, 50, 50), 64);
    const unsigned long cap_after_flood = ksw_g2_wave_zr_capacity();

    MESSAGE("zr capacity: after spike=" << cap_after_spike
            << " after narrow flood=" << cap_after_flood);
    if (tier_has_wavefront()) {
        // The spike grew zr past the floor; the narrow flood must have released it
        // (grow-only would leave cap_after_flood == cap_after_spike).
        CHECK(cap_after_spike > (4u << 20));
        CHECK(cap_after_flood < cap_after_spike);
    }
}
