// test/unit/test_ksort_permutation.cpp -- pins klib ks_introsort's PERMUTATION.
//
// Several sorts in bwa-mem3 run klib's unstable introsort under a comparator
// that is only a partial order (the default dedup `re` sort, the chain filter's
// weight sort, the per-read SMEM sort). On tied keys the output order is
// whatever the algorithm's exact sequence of compares and swaps leaves, and
// bwa-mem2's SAM output is defined by that permutation. So ks_introsort's
// behaviour on ties is a contract, not an implementation detail: any change
// to ksort.h that alters the compare/swap sequence changes output on ties.
//
// This test fixes that contract as a checksum of the tie order the current
// algorithm produces on deterministic, tie-heavy fixtures at sizes that cover
// the n == 2 fast path, the "one partition then insertion sort" sizes (n <= 16),
// and the recursive sizes. Two element sizes are pinned -- an 8-byte key/id
// struct instantiated here and the 112-byte mem_alnreg_t through the dedup
// `re` oracle -- so a change in how records are moved is caught at both.
//
// The constants were captured from the algorithm as it stood before any
// permutation-preserving micro-optimisation (hold-and-shift insertion pass);
// a legitimate change to the algorithm must update them AND the equivalence
// notes, since it changes SAM output on tied inputs.
#include <cstdint>
#include <cstring>
#include <vector>
#include "doctest/doctest.h"
#include "bwamem.h"
#include "ksort.h"

extern void bwamem3_dedup_sort_by_re_exact(int n, mem_alnreg_t *a);   // ks_introsort(mem_ars2_m2)

namespace {

// Pinned checksums (see the header comment), captured from the algorithm
// before the hold-and-shift insertion pass was introduced.
const uint64_t KEYID_DENSE_PIN   = 4231154945247265514ULL;
const uint64_t KEYID_SPARSE_PIN  = 14599179933308614186ULL;
const uint64_t ALNREG_DENSE_PIN  = 15252340977868174694ULL;
const uint64_t ALNREG_SPARSE_PIN = 10019861062961686046ULL;

struct KeyId { uint32_t key, id; };
#define keyid_lt(a, b) ((a).key < (b).key)
KSORT_INIT(kid, KeyId, keyid_lt)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 11; }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)); }
};

const int SIZES[] = {2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 20, 24, 33, 40, 64, 100, 129, 257, 600, 1500};
const int N_SIZES = static_cast<int>(sizeof(SIZES) / sizeof(SIZES[0]));

inline uint64_t fnv(uint64_t h, uint64_t v) {
    for (int b = 0; b < 8; ++b) { h ^= (v >> (8 * b)) & 0xff; h *= 1099511628211ULL; }
    return h;
}

// Checksum of the id order after sorting tie-heavy KeyId arrays: keys drawn
// from a pool of `pool` values (so ties are dense), ids = input position.
uint64_t keyid_checksum(int pool, uint64_t seed) {
    uint64_t h = 1469598103934665603ULL;
    Rng rng(seed);
    for (int si = 0; si < N_SIZES; ++si) {
        const int n = SIZES[si];
        for (int rep = 0; rep < 5; ++rep) {
            std::vector<KeyId> v(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) { v[static_cast<size_t>(i)].key = static_cast<uint32_t>(rng.in(0, pool - 1)); v[static_cast<size_t>(i)].id = static_cast<uint32_t>(i); }
            ks_introsort(kid, static_cast<size_t>(n), v.data());
            for (int i = 1; i < n; ++i) REQUIRE(v[static_cast<size_t>(i - 1)].key <= v[static_cast<size_t>(i)].key);  // sorted
            for (int i = 0; i < n; ++i) h = fnv(h, v[static_cast<size_t>(i)].id);
        }
    }
    return h;
}

// Same for mem_alnreg_t under the `re`-only comparator; seedcov = input position.
uint64_t alnreg_checksum(int pool, uint64_t seed) {
    uint64_t h = 1469598103934665603ULL;
    Rng rng(seed);
    for (int si = 0; si < N_SIZES; ++si) {
        const int n = SIZES[si];
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<mem_alnreg_t> v(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                mem_alnreg_t &r = v[static_cast<size_t>(i)];
                memset(&r, 0, sizeof(r));
                r.re = 1000 + 37 * rng.in(0, pool - 1);
                r.rb = r.re - rng.in(20, 150);
                r.seedcov = i;
            }
            bwamem3_dedup_sort_by_re_exact(n, v.data());
            for (int i = 1; i < n; ++i) REQUIRE(v[static_cast<size_t>(i - 1)].re <= v[static_cast<size_t>(i)].re);
            for (int i = 0; i < n; ++i) h = fnv(h, static_cast<uint64_t>(v[static_cast<size_t>(i)].seedcov));
        }
    }
    return h;
}

}  // namespace

TEST_CASE("ks_introsort's tie permutation is pinned (8-byte elements)"
          * doctest::test_suite("unit/ksort_permutation")) {
    const uint64_t dense  = keyid_checksum(3, 0x7d0ULL);
    const uint64_t sparse = keyid_checksum(40, 0x7d1ULL);
    INFO("keyid dense=" << dense << " sparse=" << sparse);
    CHECK(dense  == KEYID_DENSE_PIN);
    CHECK(sparse == KEYID_SPARSE_PIN);
}

TEST_CASE("ks_introsort's tie permutation is pinned (112-byte mem_alnreg_t)"
          * doctest::test_suite("unit/ksort_permutation")) {
    const uint64_t dense  = alnreg_checksum(3, 0x7d2ULL);
    const uint64_t sparse = alnreg_checksum(40, 0x7d3ULL);
    INFO("alnreg dense=" << dense << " sparse=" << sparse);
    CHECK(dense  == ALNREG_DENSE_PIN);
    CHECK(sparse == ALNREG_SPARSE_PIN);
}
