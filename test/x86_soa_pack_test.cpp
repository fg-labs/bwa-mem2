// x86_soa_pack / x86_soa_pack_u16 byte-identity vs the scalar SoA fills they replaced.
//
// The AVX2 / AVX-512BW 8-bit and 16-bit batch wrappers (kswv mate rescue,
// bandedSWA extension) used to build their SoA input with one strided byte or
// halfword store per base; src/x86_soa_pack.h builds the same layouts with a
// tiled register transpose (x86_soa_pack for the 8-bit kernels, x86_soa_pack_u16
// for the int16 kernels). The contract is "byte-for-byte what the scalar fill wrote",
// including every pad row -- and the whole-aligner oracles cannot check that:
// the tier-parity script compares two x86 tiers that both run this header, and
// a SAM comparison is blind to pad bytes chosen so they never change an
// alignment. So this test is the in-tree oracle: for randomized lane groups
// it computes the scalar reference fill and asserts memcmp equality over the
// FULL buffer, canary rows beyond nrows included.
//
// Built at -march=native, so the AVX2 widths (8-bit W == 32, 16-bit W == 16)
// always run on an AVX2 host and the AVX-512BW widths (8-bit W == 64, 16-bit
// W == 32) run when the host (and therefore the build) has AVX-512BW; each
// AVX-512BW case records a skip otherwise. On a host without AVX2 (arm64, or an
// x86 build below the AVX2 tier) the header is empty and the single case below
// records the skip.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "x86_soa_pack.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__AVX2__)

namespace {

// The per-lane contract the wrappers' scalar loops implemented: real bases
// (remapped when asked) for k < len, padA on [len, padStart), padB from
// padStart on, for rows [0, nrows).
void scalar_fill(uint8_t *soa, int W, const std::vector<std::vector<uint8_t>> &seq,
                 const std::vector<int> &len, const std::vector<int> &padStart, int nrows,
                 uint8_t padA, uint8_t padB, bool remap, uint8_t remapFrom, uint8_t remapTo) {
    for (int j = 0; j < W; j++) {
        for (int k = 0; k < nrows; k++) {
            uint8_t v;
            if (k < len[j]) {
                v = seq[j][k];
                if (remap && v == remapFrom) v = remapTo;
            } else {
                v = (k < padStart[j]) ? padA : padB;
            }
            soa[(size_t)k * W + j] = v;
        }
    }
}

struct Params {
    const char *name;
    uint8_t padA, padB;
    bool remap;
    uint8_t remapFrom, remapTo;
    bool padStartIsLen;  // reference-side call: padStart == len
    bool quantum16;      // rescue query: padStart = len rounded up to 16
};

// The four wrapper calls (kswv.cpp / bandedSWA.cpp) plus two adversarial
// parameter sets: a pad byte equal to remapFrom (pad bytes must not be
// remapped) and remapFrom equal to remapTo with a distinct padA/padB pair.
const Params kParams[] = {
    {"rescue reference (0xFF pad, no remap)", 0xFF, 0xFF, false, 4, 4, true, false},
    {"rescue query (DUMMY5 to quantum, 0xFF after, N -> 8)", 5, 0xFF, true, 4, 8, false, true},
    {"extension reference (DUMMY1 pad, N stays 4)", 99, 99, false, 4, 8, true, false},
    {"extension query (DUMMY2 pad, N -> 8)", 100, 100, true, 4, 8, true, false},
    {"pad byte equal to remapFrom", 4, 4, true, 4, 8, false, false},
    {"padA below padStart, arbitrary padStart", 7, 200, true, 1, 250, false, false},
};

// Pick a lane length that lands on the interesting spots: 0, one short of a
// tile, a whole tile, one past, the last row, and anything in between.
int pick_len(std::mt19937 &rng, int nrows) {
    switch (rng() % 8) {
        case 0: return 0;
        case 1: return nrows - 1;
        case 2: return std::min(nrows - 1, 16 * (int)(1 + rng() % 4));
        case 3: return std::min(nrows - 1, 16 * (int)(1 + rng() % 4) - 1);
        case 4: return std::min(nrows - 1, 16 * (int)(1 + rng() % 4) + 1);
        default: return (int)(rng() % nrows);
    }
}

template <int W>
void run_trials(const Params &p, int trials, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int t = 0; t < trials; t++) {
        // Row counts on both sides of the tile size, including exact multiples
        // and one past (the kernels pass maxLen + 1).
        int nrows;
        switch (t % 4) {
            case 0: nrows = 16 * (int)(1 + rng() % 12); break;
            case 1: nrows = 16 * (int)(1 + rng() % 12) + 1; break;
            case 2: nrows = 1 + (int)(rng() % 20); break;
            default: nrows = 1 + (int)(rng() % 400); break;
        }
        std::vector<std::vector<uint8_t>> seq(W);
        std::vector<int> len(W), padStart(W);
        std::vector<const uint8_t *> seqp(W);
        const bool wideBytes = (t % 5 == 4);  // every byte value, not just bases
        for (int j = 0; j < W; j++) {
            len[j] = pick_len(rng, nrows);
            if (p.padStartIsLen) padStart[j] = len[j];
            else if (p.quantum16) padStart[j] = ((len[j] + 15) / 16) * 16;
            else padStart[j] = len[j] + (int)(rng() % 40);  // may exceed nrows
            // Exactly len bytes so an over-read is out of bounds (visible
            // under ASan), and a null pointer for an empty lane.
            seq[j].resize(len[j]);
            for (int k = 0; k < len[j]; k++) {
                uint8_t b = wideBytes ? (uint8_t)(rng() % 256) : (uint8_t)(rng() % 5);
                if (!wideBytes && rng() % 7 == 0) b = p.remapFrom;  // plenty of remap hits
                if (!wideBytes && rng() % 11 == 0) b = p.padA;      // pad values as real bases
                seq[j][k] = b;
            }
            seqp[j] = len[j] ? seq[j].data() : nullptr;
        }
        // Two canary rows past nrows: the pack must write rows [0, nrows) only.
        const size_t bytes = (size_t)(nrows + 2) * W;
        void *gotp = nullptr, *wantp = nullptr;
        REQUIRE(posix_memalign(&gotp, 64, bytes) == 0);   // the kernels' buffers are 64-byte aligned
        REQUIRE(posix_memalign(&wantp, 64, bytes) == 0);
        uint8_t *got = (uint8_t *)gotp, *want = (uint8_t *)wantp;
        memset(got, 0xA5, bytes);
        memset(want, 0xA5, bytes);
        scalar_fill(want, W, seq, len, padStart, nrows, p.padA, p.padB, p.remap, p.remapFrom, p.remapTo);
        x86_soa_pack<W>(got, seqp.data(), len.data(), padStart.data(), nrows, p.padA, p.padB,
                        p.remap, p.remapFrom, p.remapTo);
        const bool same = memcmp(got, want, bytes) == 0;
        if (!same) {
            size_t i = 0;
            while (i < bytes && got[i] == want[i]) i++;
            const int row = (int)(i / W), lane = (int)(i % W);
            const int g = got[i], w = want[i];
            free(got);   // FAIL unwinds the test case, so release the buffers first
            free(want);
            FAIL(p.name << ": W=" << W << " trial " << t << " nrows=" << nrows
                        << " first mismatch row " << row << " lane " << lane
                        << " (len " << len[lane] << ", padStart " << padStart[lane]
                        << "): got " << g << " want " << w);
        }
        free(got);
        free(want);
    }
}

}  // namespace

TEST_CASE("x86_soa_pack<32> matches the scalar SoA fill byte for byte") {
    uint32_t seed = 0x5eed0032u;
    for (const Params &p : kParams) {
        CAPTURE(p.name);
        run_trials<32>(p, 300, seed++);
    }
}

#if defined(__AVX512BW__)
TEST_CASE("x86_soa_pack<64> matches the scalar SoA fill byte for byte") {
    uint32_t seed = 0x5eed0064u;
    for (const Params &p : kParams) {
        CAPTURE(p.name);
        run_trials<64>(p, 300, seed++);
    }
}
#else
TEST_CASE("x86_soa_pack<64>: skipped, this build has no AVX-512BW") {
    MESSAGE("host/build lacks AVX-512BW; the 64-lane path is not instantiable here");
    CHECK(true);
}
#endif

// ---- 16-bit lanes: same oracle for x86_soa_pack_u16 (int16 kernels' SoA).
namespace {

void scalar_fill_u16(uint16_t *soa, int W, const std::vector<std::vector<uint8_t>> &seq,
                     const std::vector<int> &len, const std::vector<int> &padStart, int nrows,
                     uint16_t padA, uint16_t padB, uint16_t remapFrom, uint16_t remapTo) {
    for (int j = 0; j < W; j++) {
        for (int k = 0; k < nrows; k++) {
            uint16_t v;
            if (k < len[j]) {
                v = seq[j][k];
                if (v == remapFrom) v = remapTo;
            } else {
                v = (k < padStart[j]) ? padA : padB;
            }
            soa[(size_t)k * W + j] = v;
        }
    }
}

struct Params16 {
    const char *name;
    uint16_t padA, padB, remapFrom, remapTo;
    bool padStartIsLen;
    bool quantum8;  // rescue query: padStart = len rounded up to 8
};

// The four 16-bit wrapper calls (rescue ref/query, extension ref/query) plus
// a pad byte equal to remapFrom.
const Params16 kParams16[] = {
    {"rescue16 reference (0xFFFF pad, N -> 15)", 0xFFFF, 0xFFFF, 4, 15, true, false},
    {"rescue16 query (DUMMY3 to quantum, 0xFFFF after, N -> 16)", 26, 0xFFFF, 4, 16, false, true},
    {"extension16 reference (DUMMY1 pad, N -> 0xFFFF)", 99, 99, 4, 0xFFFF, true, false},
    {"extension16 query (DUMMY2 pad, N -> 0xFFFF)", 100, 100, 4, 0xFFFF, true, false},
    {"pad value equal to remapFrom", 4, 4, 4, 15, false, false},
};

template <int W>
void run_trials_u16(const Params16 &p, int trials, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int t = 0; t < trials; t++) {
        int nrows;
        switch (t % 4) {
            case 0: nrows = 8 * (int)(1 + rng() % 24); break;
            case 1: nrows = 8 * (int)(1 + rng() % 24) + 1; break;
            case 2: nrows = 1 + (int)(rng() % 12); break;
            default: nrows = 1 + (int)(rng() % 400); break;
        }
        std::vector<std::vector<uint8_t>> seq(W);
        std::vector<int> len(W), padStart(W);
        std::vector<const uint8_t *> seqp(W);
        const bool wideBytes = (t % 5 == 4);
        for (int j = 0; j < W; j++) {
            switch (rng() % 6) {
                case 0: len[j] = 0; break;
                case 1: len[j] = nrows - 1; break;
                case 2: len[j] = std::min(nrows - 1, 8 * (int)(1 + rng() % 6)); break;
                case 3: len[j] = std::min(nrows - 1, 8 * (int)(1 + rng() % 6) + 1); break;
                default: len[j] = (int)(rng() % nrows); break;
            }
            if (p.padStartIsLen) padStart[j] = len[j];
            else if (p.quantum8) padStart[j] = ((len[j] + 7) / 8) * 8;
            else padStart[j] = len[j] + (int)(rng() % 20);
            seq[j].resize(len[j]);
            for (int k = 0; k < len[j]; k++) {
                uint8_t b = wideBytes ? (uint8_t)(rng() % 256) : (uint8_t)(rng() % 5);
                if (!wideBytes && rng() % 7 == 0) b = (uint8_t)p.remapFrom;
                seq[j][k] = b;
            }
            seqp[j] = len[j] ? seq[j].data() : nullptr;
        }
        const size_t bytes = (size_t)(nrows + 2) * W * sizeof(uint16_t);
        void *gotp = nullptr, *wantp = nullptr;
        REQUIRE(posix_memalign(&gotp, 64, bytes) == 0);
        REQUIRE(posix_memalign(&wantp, 64, bytes) == 0);
        uint16_t *got = (uint16_t *)gotp, *want = (uint16_t *)wantp;
        memset(got, 0xA5, bytes);
        memset(want, 0xA5, bytes);
        scalar_fill_u16(want, W, seq, len, padStart, nrows, p.padA, p.padB, p.remapFrom, p.remapTo);
        x86_soa_pack_u16<W>(got, seqp.data(), len.data(), padStart.data(), nrows, p.padA, p.padB,
                            p.remapFrom, p.remapTo);
        const bool same = memcmp(got, want, bytes) == 0;
        if (!same) {
            size_t i = 0;
            const size_t n = bytes / sizeof(uint16_t);
            while (i < n && got[i] == want[i]) i++;
            const int row = (int)(i / W), lane = (int)(i % W);
            const int g = got[i], w = want[i];
            free(got);   // FAIL unwinds the test case, so release the buffers first
            free(want);
            FAIL(p.name << ": W=" << W << " trial " << t << " nrows=" << nrows
                        << " first mismatch row " << row << " lane " << lane
                        << " (len " << len[lane] << ", padStart " << padStart[lane]
                        << "): got " << g << " want " << w);
        }
        free(got);
        free(want);
    }
}

}  // namespace

TEST_CASE("x86_soa_pack_u16<16> matches the scalar int16 SoA fill") {
    uint32_t seed = 0x5eed1016u;
    for (const Params16 &p : kParams16) {
        CAPTURE(p.name);
        run_trials_u16<16>(p, 300, seed++);
    }
}

#if defined(__AVX512BW__)
TEST_CASE("x86_soa_pack_u16<32> matches the scalar int16 SoA fill") {
    uint32_t seed = 0x5eed1032u;
    for (const Params16 &p : kParams16) {
        CAPTURE(p.name);
        run_trials_u16<32>(p, 300, seed++);
    }
}
#else
TEST_CASE("x86_soa_pack_u16<32>: skipped, this build has no AVX-512BW") {
    MESSAGE("host/build lacks AVX-512BW; the 32-lane int16 path is not instantiable here");
    CHECK(true);
}
#endif

#else  // !__AVX2__

TEST_CASE("x86_soa_pack: skipped, header requires an AVX2 x86 build") {
    MESSAGE("no AVX2 in this build; x86_soa_pack.h is empty here");
    CHECK(true);
}

#endif  // __AVX2__
