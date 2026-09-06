// test/unit/test_sam_encode.cpp
//
// SEQ/QUAL encoder gate: sam_encode_seq_fwd / sam_encode_seq_rev /
// sam_encode_qual_rev must match a byte-wise scalar reference for every
// length (the vector loop, the overlapped final vector, and the short-input
// scalar path all have to agree), must clamp codes >= 5 to 'N', must never
// write outside [dst, dst + n), and must never read outside [src, src + n).
// The last two are checked with guard bytes and with inputs placed flush
// against inaccessible pages, so an out-of-range access faults instead of
// passing silently.
//
// The test binds whichever implementation the host compiles (NEON on arm64,
// SSSE3 on x86, scalar elsewhere), so each CI row checks its own tier against
// the same reference.

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "doctest/doctest.h"
#include "sam_encode.h"

namespace {

void ref_seq_fwd(char *dst, const uint8_t *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = "ACGTN"[src[i] <= 4 ? src[i] : 4];
}
void ref_seq_rev(char *dst, const uint8_t *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = "TGCAN"[src[n - 1 - i] <= 4 ? src[n - 1 - i] : 4];
}
void ref_qual_rev(char *dst, const char *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[n - 1 - i];
}

// Random 0..7 codes: 0..4 are ACGTN, 5..7 exercise the clamp.
std::vector<uint8_t> random_codes(std::mt19937 &rng, int n) {
    std::vector<uint8_t> v((size_t) n);
    for (auto &c : v) c = (uint8_t) (rng() % 8);
    return v;
}
std::vector<char> random_quals(std::mt19937 &rng, int n) {
    std::vector<char> v((size_t) n);
    for (auto &c : v) c = (char) ('!' + rng() % 42);
    return v;
}

const int kGuard = 32;
const char kCanary = (char) 0xEE;

// Run one encoder on a guarded destination and compare with the reference.
template <class Fn, class Ref, class T>
bool check_one(Fn fn, Ref ref, const std::vector<T> &src, int n) {
    std::vector<char> got((size_t) n + 2 * kGuard, kCanary), want(got);
    fn(got.data() + kGuard, src.data(), n);
    ref(want.data() + kGuard, src.data(), n);
    return got == want;   // payload identical AND both guard regions untouched
}

// Lengths that cover every path: empty, sub-vector, exact multiples, one
// past a multiple (odd tails), and typical read lengths.
std::vector<int> lengths() {
    std::vector<int> v;
    for (int n = 0; n <= 70; ++n) v.push_back(n);
    for (int n : {76, 100, 101, 127, 128, 129, 150, 151, 250, 255, 256, 257, 1000, 1001}) v.push_back(n);
    return v;
}

}  // namespace

TEST_CASE("sam_encode: fwd/rev SEQ and rev QUAL match the scalar reference at every length"
          * doctest::test_suite("unit/sam_encode")) {
    // One SUBCASE per encoder so doctest reports and --subcase-filters each
    // independently (per-length subcases are not used: doctest re-enters the
    // case per subcase path and does not compose with a loop of same-named
    // subcases -- the lengths, which include the 16-byte-tail boundaries
    // 16/17/31/32/33, are swept inside each encoder's subcase instead). Whichever
    // tier is compiled (SAM_FAST_IMPL) is the one under test; CI builds this file
    // for every tier, so all tiers are compared against the scalar reference.
    std::mt19937 rng(20260906);
    SUBCASE("seq_fwd") {
        for (int n : lengths())
            for (int rep = 0; rep < 4; ++rep)
                CHECK_MESSAGE(check_one(sam_encode_seq_fwd, ref_seq_fwd, random_codes(rng, n), n),
                              "seq_fwd n=" << n);
    }
    SUBCASE("seq_rev") {
        for (int n : lengths())
            for (int rep = 0; rep < 4; ++rep)
                CHECK_MESSAGE(check_one(sam_encode_seq_rev, ref_seq_rev, random_codes(rng, n), n),
                              "seq_rev n=" << n);
    }
    SUBCASE("qual_rev") {
        for (int n : lengths())
            for (int rep = 0; rep < 4; ++rep)
                CHECK_MESSAGE(check_one(sam_encode_qual_rev, ref_qual_rev, random_quals(rng, n), n),
                              "qual_rev n=" << n);
    }
}

TEST_CASE("sam_encode: codes >= 5 clamp to N in both directions"
          * doctest::test_suite("unit/sam_encode")) {
    std::vector<uint8_t> codes = {0, 1, 2, 3, 4, 5, 6, 7, 255, 4, 3, 2, 1, 0, 9, 8, 16, 17, 100};
    const int n = (int) codes.size();
    std::string fwd((size_t) n, '\0'), rev((size_t) n, '\0');
    sam_encode_seq_fwd(&fwd[0], codes.data(), n);
    sam_encode_seq_rev(&rev[0], codes.data(), n);
    CHECK(fwd == "ACGTNNNNNNTGCANNNNN");
    CHECK(rev == "NNNNNTGCANNNNNNACGT");
}

// Three pages: the outer two are made inaccessible, the middle one holds the
// buffer. `at_end` places the buffer so its last byte is the last byte of the
// accessible page (an over-read/over-write past the end faults); otherwise its
// first byte is the first byte of the page (an access before the start faults).
struct FencedPage {
    char *base = nullptr; size_t page = 0;
    FencedPage() {
        page = (size_t) sysconf(_SC_PAGESIZE);
        base = (char *) mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        REQUIRE(base != MAP_FAILED);
        REQUIRE(mprotect(base, page, PROT_NONE) == 0);
        REQUIRE(mprotect(base + 2 * page, page, PROT_NONE) == 0);
    }
    ~FencedPage() { munmap(base, 3 * page); }
    char *place(int n, bool at_end) { return at_end ? base + 2 * page - n : base + page; }
};

TEST_CASE("sam_encode: no access outside [0, n) on either buffer"
          * doctest::test_suite("unit/sam_encode")) {
    std::mt19937 rng(7);
    FencedPage src_page, dst_page;
    // One SUBCASE per source/destination placement (the boundary configurations)
    // so a fault is reported against the exact placement; the lengths, including
    // the 16-byte-tail boundaries 16/17/31/32/33, sweep inside each.
    auto run = [&](bool src_end, bool dst_end) {
        for (int n : {1, 15, 16, 17, 31, 32, 33, 76, 150, 151, 250}) {
            auto codes = random_codes(rng, n);
            auto quals = random_quals(rng, n);
            std::vector<char> want((size_t) n), got((size_t) n);

            uint8_t *s = (uint8_t *) src_page.place(n, src_end);
            char *d = dst_page.place(n, dst_end);
            memcpy(s, codes.data(), (size_t) n);
            sam_encode_seq_fwd(d, s, n); ref_seq_fwd(want.data(), codes.data(), n);
            memcpy(got.data(), d, (size_t) n);
            CHECK_MESSAGE(got == want, "seq_fwd fenced n=" << n);
            sam_encode_seq_rev(d, s, n); ref_seq_rev(want.data(), codes.data(), n);
            memcpy(got.data(), d, (size_t) n);
            CHECK_MESSAGE(got == want, "seq_rev fenced n=" << n);

            char *q = src_page.place(n, src_end);
            memcpy(q, quals.data(), (size_t) n);
            sam_encode_qual_rev(d, q, n); ref_qual_rev(want.data(), quals.data(), n);
            memcpy(got.data(), d, (size_t) n);
            CHECK_MESSAGE(got == want, "qual_rev fenced n=" << n);
        }
    };
    SUBCASE("src at start, dst at start") { run(false, false); }
    SUBCASE("src at start, dst at end")   { run(false, true);  }
    SUBCASE("src at end, dst at start")   { run(true,  false); }
    SUBCASE("src at end, dst at end")     { run(true,  true);  }
}
