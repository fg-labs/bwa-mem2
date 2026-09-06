// x86_soa_pack byte-identity vs the scalar SoA fill it replaced.
//
// The AVX2 / AVX-512BW 8-bit batch wrappers (kswv mate rescue, bandedSWA
// extension) used to build their SoA input with one strided byte store per
// base; src/x86_soa_pack.h builds the same layout with a tiled register
// transpose. Its contract is "byte-for-byte what the scalar fill wrote",
// including every pad row -- and the whole-aligner oracles cannot check that:
// the tier-parity script compares two x86 tiers that both run this header, and
// a SAM comparison is blind to pad bytes chosen so they never change an
// alignment. So this test is the in-tree oracle: for randomized lane groups
// it computes the scalar reference fill and asserts memcmp equality over the
// FULL buffer, canary rows beyond nrows included.
//
// Built at -march=native, so W == 32 always runs on an AVX2 host and W == 64
// runs when the host (and therefore the build) has AVX-512BW. On a host
// without AVX2 (arm64, or an x86 build below the AVX2 tier) the header is
// empty and the single case below records the skip.

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
            FAIL(p.name << ": W=" << W << " trial " << t << " nrows=" << nrows
                        << " first mismatch row " << row << " lane " << lane
                        << " (len " << len[lane] << ", padStart " << padStart[lane]
                        << "): got " << (int)got[i] << " want " << (int)want[i]);
        }
        CHECK(same);
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

#else  // !__AVX2__

TEST_CASE("x86_soa_pack: skipped, header requires an AVX2 x86 build") {
    MESSAGE("no AVX2 in this build; x86_soa_pack.h is empty here");
    CHECK(true);
}

#endif  // __AVX2__
