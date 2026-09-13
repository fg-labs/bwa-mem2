// test/unit/test_pair64_sort.cpp -- pdqsort_128 reproduces ks_introsort_128.
//
// mem_pair (src/bwamem_pair.cpp) sorts two pair64_t arrays whose (x, y) keys
// are pairwise distinct by construction: `v` packs the region index and read
// number into y, `u` packs the (k, i) candidate pair into y. On distinct keys
// the sorted order under pair64_lt is unique, so any correct sort produces the
// same array, and pdqsort_128 can stand in for the klib introsort. This pins
// that claim against the introsort itself, on arrays shaped like mem_pair's
// (many equal x -- same position, different regions -- and distinct y), and
// asserts the distinctness the argument rests on. (Duplicate pair64_t keys are
// byte-identical, so a permutation difference among them could not be observed
// anyway; the uniqueness argument, not the algorithm, carries the parity.)
//
// Fixtures are generated in-memory; no test data files are read.
#include <cstdint>
#include <cstring>
#include <vector>
#include "doctest/doctest.h"
#include "utils.h"

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 11; }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)); }
};

// n keys in mem_pair's `v` shape: x = position drawn from `x_pool` slots (so
// equal x is common), y = score << 32 | i << 2 | strand << 1 | r with the index
// i distinct per element, so (x, y) is always distinct.
std::vector<pair64_t> mem_pair_like(Rng &rng, int n, int x_pool) {
    std::vector<pair64_t> v;
    for (int i = 0; i < n; ++i) {
        pair64_t k;
        k.x = static_cast<uint64_t>(rng.in(0, x_pool - 1)) * 1000;
        k.y = static_cast<uint64_t>(rng.in(20, 300)) << 32 |
              static_cast<uint64_t>(i) << 2 | static_cast<uint64_t>(rng.in(0, 1)) << 1 |
              static_cast<uint64_t>(rng.in(0, 1));
        v.push_back(k);
    }
    return v;
}

// n keys in mem_pair's `u` shape: y = k << 32 | i over distinct (k, i) pairs,
// x = q << 32 | 32-bit hash with q drawn from few values so x ties are common.
std::vector<pair64_t> mem_pair_u_like(Rng &rng, int n, int q_pool) {
    std::vector<pair64_t> v;
    for (int t = 0; t < n; ++t) {
        pair64_t u;
        // Distinct (k, i): walk a grid so no pair repeats.
        const uint64_t k = static_cast<uint64_t>(t / 7), i = static_cast<uint64_t>(t % 7);
        u.y = k << 32 | i;
        u.x = static_cast<uint64_t>(rng.in(0, q_pool - 1)) << 32 | (rng.next() & 0xffffffffULL);
        v.push_back(u);
    }
    return v;
}

bool same(const std::vector<pair64_t> &a, const std::vector<pair64_t> &b) {
    return a.size() == b.size() &&
           (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(pair64_t)) == 0);
}

}  // namespace

TEST_CASE("pdqsort_128 equals ks_introsort_128 on distinct (x, y) keys"
          * doctest::test_suite("unit/pair64_sort")) {
    // Each variant is its own SUBCASE so doctest reports and --subcase-filters
    // them independently; doctest re-enters the body per subcase, so each holds
    // its own Rng with a distinct seed.
    SUBCASE("v-shape keys: heavy x ties (realistic case), y always distinct") {
        Rng rng(0x5eed1234ULL);
        for (int t = 0; t < 300; ++t) {
            const int n = 1 + t;
            std::vector<pair64_t> in = mem_pair_like(rng, n, 1 + t / 10);
            std::vector<pair64_t> p(in), k(in);
            pdqsort_128(p.size(), p.data());
            ks_introsort_128(k.size(), k.data());
            CHECK(same(p, k));
            // Sorted, and every adjacent pair strictly ordered (distinct keys).
            for (size_t i = 1; i < k.size(); ++i)
                CHECK((k[i - 1].x < k[i].x || (k[i - 1].x == k[i].x && k[i - 1].y < k[i].y)));
        }
    }
    // The `u` shape (candidate pairs): equal pairing scores are common, the
    // low-32 hash rarely collides, and (k, i) is distinct.
    SUBCASE("u-shape candidate pairs: common equal scores, rare hash collisions") {
        Rng rng(0x5eed5678ULL);
        for (int t = 0; t < 200; ++t) {
            const int n = 1 + t;
            std::vector<pair64_t> in = mem_pair_u_like(rng, n, 1 + t / 20);
            std::vector<pair64_t> p(in), k(in);
            pdqsort_128(p.size(), p.data());
            ks_introsort_128(k.size(), k.data());
            CHECK(same(p, k));
            for (size_t i = 1; i < k.size(); ++i)
                CHECK((k[i - 1].x < k[i].x || (k[i - 1].x == k[i].x && k[i - 1].y < k[i].y)));
        }
    }
}

// mem_pair reaches the pairing sort with zero candidates when a thread's first
// call (or a released scratch buffer) leaves the buffer null and n == 0. The
// call site guards `if (v.n > 1)` so PDQSORT_INIT never forms `a + n` on a null
// pointer; this pins the empty/single-element sort as a no-op that leaves the
// (possibly one) element in place. Buffers are reserved so data() is non-null.
TEST_CASE("pdqsort_128 is a no-op on empty and single-element inputs"
          * doctest::test_suite("unit/pair64_sort")) {
    std::vector<pair64_t> empty;
    empty.reserve(1);  // non-null data() for the n == 0 no-op
    pdqsort_128(empty.size(), empty.data());
    CHECK(empty.empty());

    std::vector<pair64_t> one(1);
    one[0].x = 42;
    one[0].y = 7;
    pdqsort_128(one.size(), one.data());
    CHECK(one[0].x == 42u);
    CHECK(one[0].y == 7u);
}
