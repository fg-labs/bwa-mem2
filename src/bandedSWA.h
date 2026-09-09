/*************************************************************************************
                           The MIT License

   BWA-MEM2  (Sequence alignment using Burrows-Wheeler Transform),
   Copyright (C) 2019  Intel Corporation, Heng Li.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

Authors: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>;
*****************************************************************************************/

#ifndef SCALAR_BANDEDSWA_HPP
#define SCALAR_BANDEDSWA_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "macro.h"

/* SIMD compatibility layer for ARM/x86 */
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
    /* ARM/Apple Silicon - use sse2neon for SSE translation */
    #include "simd_compat.h"
#elif (__AVX512BW__ || __AVX2__)
    #include <immintrin.h>
#else
    #include <smmintrin.h>  // for SSE4.1
    #ifndef __mmask8
    #define __mmask8 uint8_t
    #endif
    #ifndef __mmask16
    #define __mmask16 uint16_t
    #endif
#endif

#define MAX_SEQ_LEN_REF 256
#define MAX_SEQ_LEN_QER 128
#define MAX_SEQ_LEN_EXT 256
#define MAX_NUM_PAIRS 10000000
#define MAX_NUM_PAIRS_ALLOC 20000

/* Whether any vector banded-SW variant (getScores8/getScores16 +
 * smithWatermanBatchWrapper8/16) is declared and defined. ARM gets the
 * SSE2/NEON path via sse2neon. x86 gets the SSE2 path only with SSSE3
 * (the SSE2/NEON kernels use _mm_shuffle_epi8/PSHUFB), the AVX2 path with
 * AVX2, or the AVX512 path with AVX512BW. The Makefile passes -mssse3 on
 * every x86 arch target, so this is true in every CI build today — but
 * any caller-side dispatch must consult this macro rather than hand-rolled
 * subset checks (e.g. `!__SSE2__`) so a hypothetical SSE2-only-no-SSSE3
 * build still picks scalarBandedSWAWrapper instead of link-failing. */
#if (defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)) || \
    (__AVX512BW__) || (__AVX2__) || ((__SSE2__) && (__SSSE3__))
#define HAVE_BSW_VECTOR_8_16 1
#else
#define HAVE_BSW_VECTOR_8_16 0
#endif

// used in BSW and SAM-SW
#define DEFAULT_AMBIG -1


// SIMD_WIDTH in bits
// ARM64/NEON (128-bit vectors)
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
#ifndef SIMD_WIDTH8
#define SIMD_WIDTH8 16    // 128-bit / 8-bit = 16 elements
#endif
#ifndef SIMD_WIDTH16
#define SIMD_WIDTH16 8    // 128-bit / 16-bit = 8 elements
#endif

// AVX2
#elif ((!__AVX512BW__) & (__AVX2__))
#define SIMD_WIDTH8 32
#define SIMD_WIDTH16 16

// AVX512
#elif __AVX512BW__
#define SIMD_WIDTH8 64
#define SIMD_WIDTH16 32

// SSE2
#elif ((!__AVX512BW__) & (!__AVX2__) & (__SSE2__))
#define SIMD_WIDTH8 16
#define SIMD_WIDTH16 8

// Scalar
#else
#define SIMD_WIDTH8 1
#define SIMD_WIDTH16 1
#endif

#define MAX_LINE_LEN 256
#define MAX_SEQ_LEN8 1088
#define MAX_SEQ_LEN16 32768
#define MATRIX_MIN_CUTOFF 0
#define LOW_INIT_VALUE -128
#define SORT_BLOCK_SIZE 16384
#define min_(x, y) ((x)>(y)?(y):(x))
#define max_(x, y) ((x)>(y)?(x):(y))

// True when `mat`'s ACGT 4x4 submatrix is NOT the plain symmetric
// match/mismatch matrix implied by (w_match, w_mismatch) — i.e. the cheap
// symmetric XOR LUT cannot represent it and the generic (target-major) LUT
// path is required. The default aligner matrix is symmetric, so this returns
// false on the hot path and the kernels keep their original fast prepass;
// it returns true only for an asymmetric matrix (bisulfite OT/OB), which pays
// the heavier generic prepass. N cells are handled separately, so only the
// ACGT submatrix is inspected. w_mismatch is the kernel's stored (negated)
// penalty, matching how mat off-diagonals are encoded.
static inline bool bsw_generic_matrix(const int8_t *mat, int8_t w_match, int8_t w_mismatch)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int8_t expect = (i == j) ? w_match : w_mismatch;
            if (mat[i * 5 + j] != expect) return true;
        }
    return false;
}

// Dev/validation hook: when BWAMEM3_FORCE_GENMAT is set in the environment, the
// SW kernels take the generic-matrix prepass even for a symmetric matrix. This
// lets the standalone kernel bench (which uses the default symmetric matrix)
// (a) measure the generic path's throughput and (b) prove it is byte-identical
// to the symmetric path on symmetric inputs (amat == pmat there). Read once;
// zero cost on the production path when unset (magic-static init, no syscall in
// the hot loop). NEVER affects output for a symmetric matrix — only which
// prepass macro runs.
static inline bool bsw_force_generic_matrix()
{
    static const bool forced = (getenv("BWAMEM3_FORCE_GENMAT") != NULL);
    return forced;
}

// Rank-1 fast-path descriptor for the generic-matrix seam. Bisulfite frees
// exactly ONE ordered off-diagonal cell per strand to a match (OT: ref-C/read-T;
// OB: ref-G/read-A). When the matrix is the plain symmetric matrix plus a single
// such freed-to-match cell, the kernel can skip the byte-shuffle LUT entirely
// and just extend the match condition (SBT_PREPASS16_RANK1) — no TBL, no
// sign-extend, near parity with symmetric. Any other deviation (>=2 freed cells,
// a changed diagonal, or a freed value != w_match) falls back to the general LUT
// path. `rank1` is the only field bandedSWA's kernel branches on; (ref,read) name
// the freed cell. When `forced` is set on an otherwise-symmetric matrix we
// synthesize the no-op cell (0,0) so the bench exercises and times the rank-1 path
// while staying byte-identical to symmetric.
//
// `single` generalizes the structural test to "exactly ONE off-diagonal cell
// deviates from the canonical mismatch, to ANY value, with no diagonal change".
// `value` carries that freed cell's value. `rank1` is exactly `single && value ==
// w_match`, so it is byte-identical to the pre-generalization predicate and every
// bandedSWA caller (which reads only `.rank1`/`.ref`/`.read`) is unaffected. The
// kswv rescue kernel reads `single`/`value` to express NEUTRAL scoring (one cell
// freed to 0), which is single but not rank1.
struct BswFreedCell { bool rank1; bool single; int8_t ref; int8_t read; int8_t value; };
static inline BswFreedCell bsw_freed_cell(const int8_t *mat, int8_t w_match,
                                          int8_t w_mismatch, bool forced)
{
    int n_freed = 0;
    int8_t fr_ref = 0, fr_read = 0, fr_val = w_match;
    bool ok = true;   // no diagonal change (an unexpressible deviation)
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int8_t expect = (i == j) ? w_match : w_mismatch;
            if (mat[i * 5 + j] == expect) continue;
            if (i != j) {                                // off-diagonal freed (any value)
                n_freed++; fr_ref = (int8_t)i; fr_read = (int8_t)j;
                fr_val = mat[i * 5 + j];
            } else {
                ok = false;                              // diagonal change: not expressible
            }
        }
    BswFreedCell c;
    c.single = ok && (n_freed == 1 || (forced && n_freed == 0));
    c.rank1  = c.single && fr_val == w_match;   // preserves the old predicate exactly
    c.ref    = fr_ref;
    c.read   = fr_read;
    c.value  = fr_val;                          // == w_match for the forced no-op
    return c;
}

// COLLAPSED bisulfite (the `--meth` default) frees the conversion cell AND its
// mirror, so C/T (and G/A) are mutually interchangeable: two off-diagonal cells
// (i,j) and (j,i), both to a match, with everything else canonical. This is the
// rank-1 case plus its transpose; the kernel handles it by freeing BOTH ordered
// cells (the rank-1 case is the degenerate (i,j)==(j,i) where the mirror equals
// the primary). `supported` is true ONLY for an exact symmetric mirror pair; a
// non-mirror two-cell matrix (e.g. OT+OB combined), >2 freed cells, a changed
// diagonal, or a non-match freed value all leave it false → scalar fallback.
// (refA,readA) is the primary cell, (refB,readB) its mirror.
struct BswFreedPair { bool supported; int8_t refA, readA, refB, readB; };
static inline BswFreedPair bsw_freed_pair(const int8_t *mat, int8_t w_match,
                                          int8_t w_mismatch)
{
    int n_freed = 0;
    int8_t r[2] = {0, 0}, c[2] = {0, 0};
    bool ok = true;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int8_t expect = (i == j) ? w_match : w_mismatch;
            if (mat[i * 5 + j] == expect) continue;
            if (i != j && mat[i * 5 + j] == w_match) {   // off-diagonal freed to match
                if (n_freed < 2) { r[n_freed] = (int8_t)i; c[n_freed] = (int8_t)j; }
                n_freed++;
            } else {
                ok = false;                              // diagonal change / non-match freed
            }
        }
    BswFreedPair p;
    p.supported = false;
    p.refA = p.readA = p.refB = p.readB = 0;
    if (ok && n_freed == 2 && r[0] == c[1] && c[0] == r[1]) {  // exact (i,j)/(j,i) mirror
        p.supported = true;
        p.refA = r[0]; p.readA = c[0];
        p.refB = r[1]; p.readB = c[1];
    }
    return p;
}

typedef struct dnaSeqPair
{
    int32_t idr, idq, id;
    int32_t len1, len2;
    int32_t h0;
    int seqid, regid;
    int32_t score, tle, gtle, qle;
    int32_t gscore, max_off;
    // PR 26c.1: per-pair SW band upper bound derived from the ungapped
    // score. 0 means "use default opt->w"; any positive value is an
    // upper bound on useful band offset for this pair. Default-initialized
    // here so stack-local SeqPair instances that don't reach every
    // construction-site assignment (e.g. RIGHT-only paths that skip the
    // LEFT tight_band assignment, or seeding paths that don't run
    // ungapped_analyze) start at the safe sentinel rather than indeterminate.
    int32_t tight_band = 0;
    // --adaptive-band: per-pair band implied by the chain's seed diagonal spread
    // (capped at opt->w). Retry-floor for adaptive banding; 0 when off/no indel.
    int32_t chain_band = 0;
    // Q3 instrumentation: would-be ungapped extension score (full diagonal
    // walk, mirrors the HIT-path walk semantics). Computed at LEFT queue
    // time for non-HIT pairs; carried through SW retry-collect; read at
    // commit. -1 means undefined (e.g., zero-length).
    int32_t ugp_walk_score = -1;
    // Set by the construction-time RIGHT ungapped attempt (when
    // a->score != -1 made fp_h0 known) so the post-left-SW pass doesn't
    // double-count UGP_R_ATTEMPT / UGP_R_TIGHT / UGP_R_HIT.
    uint8_t ugp_r_attempted = 0;
    // issue 173 / Task 5 (--meth batched mate rescue): per-pair bisulfite
    // hypothesis tag for the batched kswv mate-rescue partition. 1 = OT
    // (mat_ot, frees C->T), 0 = OB (mat_ob, frees G->A); -1 = non-meth
    // (default; the kernels never read this field, so it does not perturb
    // the non-meth getScores8/16 results — proven by the unchanged goldens).
    int8_t meth_hyp = -1;
}SeqPair;


typedef struct dnaOutScore
{
    int32_t score, tle, gtle, qle;
    int32_t gscore, max_off;
} OutScore;

typedef struct {
    int32_t h, e;
} eh_t;


#include "kernel_dispatch.h"  /* must come before BandedPairWiseSW class so per-tier compiles see the renamed symbol */

#include <memory>

/* Abstract interface for the banded Smith-Waterman batch kernel.
 * Concrete implementations are per-tier mangled subclasses (BandedPairWiseSW
 * compiled with KERNEL_VARIANT=_sse41/_sse42/_avx/_avx2/_avx512bw on x86,
 * unsuffixed on arm64). Construct via make_banded_pair_wise_sw().
 *
 * getScores8/getScores16 are declared unconditionally here even though the
 * concrete class only defines them when HAVE_BSW_VECTOR_8_16 holds (which
 * currently means at least SSSE3 on x86 or NEON on arm64 — every tier we
 * ship). A future build that disables vector kernels (HAVE_BSW_VECTOR_8_16=0)
 * would fail to compile because the `final` concrete class would lack
 * overrides for these pure-virtual methods. If that build is ever needed,
 * either tighten the guards on these decls to match the cpp side, or
 * provide scalar wrappers. */
class IBandedPairWiseSW {
public:
    virtual ~IBandedPairWiseSW() = default;

    virtual int scalarBandedSWA(int qlen, const uint8_t *query, int tlen,
                                const uint8_t *target, int32_t w,
                                int h0, int *_qle, int *_tle,
                                int *_gtle, int *_gscore,
                                int *_max_off) = 0;

    virtual void scalarBandedSWAWrapper(SeqPair *seqPairArray,
                                        uint8_t *seqBufRef,
                                        uint8_t *seqBufQer,
                                        int numPairs,
                                        int nthreads,
                                        int32_t w) = 0;

    /* Batched banded Smith-Waterman over `numPairs` SeqPairs.
     *
     * PADDING-LANE CONTRACT (must be honored by every caller):
     * the kernel rounds the batch up to a whole number of SIMD lanes
     * (SIMD_WIDTH8 for getScores8, SIMD_WIDTH16 for getScores16 — tier
     * dependent, up to 64) and *writes* the trailing padding lanes
     * pairArray[numPairs .. roundup(numPairs, SIMD_WIDTH)) itself, setting
     * their id and zeroing len1/len2/idr/idq/h0 on every tier (idr/idq handling
     * is described below). The caller MUST therefore allocate at
     * least roundup(numPairs, SIMD_WIDTH) SeqPair slots, NOT just numPairs, or those
     * writes (and the SoA gather that reads len1/len2 back) run off the end of
     * the array. A caller that does not know the active tier should round up to
     * the max lane count (64). The pipeline over-allocates and is safe; a
     * direct caller that sized the array to exactly numPairs crashed on
     * AVX2/AVX-512 — see the test helper BatchBuffers
     * (test/framework/seqpair_batch.h), which allocates numPairs + SIMD_WIDTH8
     * SeqPair slots and so satisfies this rule.
     *
     * Every tier's per-lane compute loop iterates over the padded tail (its
     * inner j-loop reaches i+j == roundNumPairs-1) and forms seqBufRef+idr /
     * seqBufQer+idq for each padded lane, though it never dereferences them
     * (padded len1==len2==0, so the copy loops run zero iterations). A padding
     * loop therefore keeps idr/idq in a defined state (zeroed) so that
     * pointer is seqBuf+0 (in-bounds) rather than seqBuf+<indeterminate>. Tiers
     * whose prefetch is bounded to < roundNumPairs also read padded idr/idq to
     * form a prefetch address, which the same zeroing keeps in-bounds. No tier
     * ever reads past the rounded-up region, so no +PFD slack beyond
     * roundup(numPairs, SIMD_WIDTH) is required of the caller:
     *
     *   - The 128-bit 8-bit implementation (SSE2/NEON getScores8) zeroes padded
     *     idr/idq (like the tiers below) and additionally bounds the prefetch of
     *     pairArray[i+j+PFD] to < numPairs -- a pure locality choice, since a
     *     padded successor is non-existent and there is no useful line to
     *     prefetch; the numPairs bound loses no real prefetch.
     *
     *   - Every getScores16 (and the 256-bit/512-bit 8-bit getScores8) bounds the
     *     prefetch to < roundNumPairs, so it reads padded lanes' idr/idq to form
     *     a prefetch address; its padding loop zeroes idr/idq so that read, and
     *     the per-lane compute pointer, both land at seqBuf offset 0.
     *
     * The per-lane SoA seed loop likewise reads back each lane's h0 (the tail of
     * a batch indexes padding lanes, since the inner loop spans a full SIMD_WIDTH);
     * the padding loop zeroes h0 above for the same reason. A padded lane's result
     * is never consumed -- every tier sets its len1 = 0 (so its DP body does no
     * work) and the caller reads results back only for pairArray[0 .. numPairs).
     * Padded lanes do still take part in the batch's cross-lane reductions (e.g.
     * the all-lanes-done exit test), so the zeroing is not purely cosmetic; it is
     * whole-aligner output that is byte-identical with vs. without it (validated
     * across all SIMD tiers), and the zeroing keeps the padded-lane read well-
     * defined for MemorySanitizer besides. */
    virtual void getScores8(SeqPair *pairArray,
                            uint8_t *seqBufRef,
                            uint8_t *seqBufQer,
                            int32_t numPairs,
                            uint16_t numThreads,
                            int32_t w) = 0;

    /* See getScores8 for the padding-lane / prefetch contract. getScores16
     * always takes the < roundNumPairs prefetch branch described there (with
     * SIMD_WIDTH16 lanes) and zeroes padded idr/idq accordingly. */
    virtual void getScores16(SeqPair *pairArray,
                             uint8_t *seqBufRef,
                             uint8_t *seqBufQer,
                             int32_t numPairs,
                             uint16_t numThreads,
                             int32_t w) = 0;

    /* SW_cells is a public field on BandedPairWiseSW today; expose as a getter
     * on the interface so non-virtual access through the unique_ptr works. */
    virtual uint64_t sw_cells() const = 0;

    /* Enable getScores{8,16}'s sub-slice overshoot guard on this object. Off by
     * default: only sub-slice callers (the --meth OT/OB kernels) need it, and
     * for whole-array callers the guard is pure overhead. Set once at setup,
     * before any (possibly multi-threaded) getScores call. */
    virtual void set_guard_overshoot(bool on) = 0;
};

/* Factory: returns a per-tier concrete BandedPairWiseSW. Construction
 * arguments mirror the BandedPairWiseSW ctor exactly. */
std::unique_ptr<IBandedPairWiseSW> make_banded_pair_wise_sw(
    int o_del, int e_del, int o_ins, int e_ins, int zdrop,
    int end_bonus, const int8_t *mat,
    int8_t w_match, int8_t w_mismatch, int numThreads);

/* Per-tier factory function forward declarations.
 * Defined in bandedSWA.<tier>.o (each kernel TU compile).
 * Called from simd_dispatch.cpp to construct the right per-tier concrete
 * class without exposing the class layout to the dispatcher TU (which
 * previously caused an ODR size mismatch / heap corruption). */
#if defined(__x86_64__) || defined(__i386__)
extern "C" IBandedPairWiseSW *make_bsw_kernel_sse41(int, int, int, int, int, int,
                                                    const int8_t*, int8_t, int8_t, int);
extern "C" IBandedPairWiseSW *make_bsw_kernel_sse42(int, int, int, int, int, int,
                                                    const int8_t*, int8_t, int8_t, int);
extern "C" IBandedPairWiseSW *make_bsw_kernel_avx(int, int, int, int, int, int,
                                                  const int8_t*, int8_t, int8_t, int);
extern "C" IBandedPairWiseSW *make_bsw_kernel_avx2(int, int, int, int, int, int,
                                                   const int8_t*, int8_t, int8_t, int);
extern "C" IBandedPairWiseSW *make_bsw_kernel_avx512bw(int, int, int, int, int, int,
                                                       const int8_t*, int8_t, int8_t, int);
#endif


class BandedPairWiseSW final : public IBandedPairWiseSW {
    
public:
    uint64_t SW_cells;

    BandedPairWiseSW(const int o_del, const int e_del, const int o_ins,
                     const int e_ins, const int zdrop,
                     const int end_bonus, const int8_t *mat_,
                     const int8_t w_match, const int8_t w_mismatch, int numThreads);
    ~BandedPairWiseSW();

    // Owns dp_slab_ via raw pointer; copying or moving would alias the
    // allocation and double-free on destruction. Disable both.
    BandedPairWiseSW(const BandedPairWiseSW&)            = delete;
    BandedPairWiseSW& operator=(const BandedPairWiseSW&) = delete;
    BandedPairWiseSW(BandedPairWiseSW&&)                 = delete;
    BandedPairWiseSW& operator=(BandedPairWiseSW&&)      = delete;

    uint64_t sw_cells() const override { return SW_cells; }

    // When true, getScores{8,16} save/restore the padding-lane overshoot past
    // numPairs (needed only by sub-slice callers -- the --meth OT/OB kernels).
    // Read-only during scoring; set once at setup via set_guard_overshoot.
    bool guard_overshoot_ = false;
    void set_guard_overshoot(bool on) override { guard_overshoot_ = on; }

    // Scalar code section
    int scalarBandedSWA(int qlen, const uint8_t *query, int tlen,
                        const uint8_t *target, int32_t w,
                        int h0, int *_qle, int *_tle,
                        int *_gtle, int *_gscore,
                        int *_max_off) override;

    void scalarBandedSWAWrapper(SeqPair *seqPairArray,
                                uint8_t *seqBufRef,
                                uint8_t *seqBufQer,
                                int numPairs,
                                int nthreads,
                                int32_t w) override;

#if (defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)) || ((!__AVX512BW__) && (!__AVX2__) && (__SSE2__) && (__SSSE3__))
    // On ARM: use SSE2 path via sse2neon translation
    // On x86 SSE2: native SSE2 implementation (requires SSSE3 for
    // _mm_shuffle_epi8 / PSHUFB used by SBT_PREPASS8_LUT). The declaration
    // guard here MUST stay in lockstep with the matching definition guard
    // in bandedSWA.cpp — the `#if ((!__AVX512BW__) && (!__AVX2__) &&
    // (__SSE2__) && (__SSSE3__))` block that opens the SSE2/NEON section
    // (search for "SSE2 code"); otherwise the SSE2-only build will link-fail.
    // 8 bit vector code section
    void getScores8(SeqPair *pairArray,
                    uint8_t *seqBufRef,
                    uint8_t *seqBufQer,
                    int32_t numPairs,
                    uint16_t numThreads,
                    int32_t w) override;

    void smithWatermanBatchWrapper8(SeqPair *pairArray,
                                   uint8_t *seqBufRef,
                                   uint8_t *seqBufQer,
                                   int32_t numPairs,
                                   uint16_t numThreads,
                                   int32_t w);

    void smithWaterman128_8(uint8_t seq1SoA[],
                            uint8_t seq2SoA[],
                            int nrow,
                            int ncol,
                            SeqPair *p,
                            uint8_t h0[],
                            uint16_t tid,
                            int32_t numPairs,
                            int zdrop,
                            int32_t w,
                            uint8_t myband[]);
    // 16 bit vector code section
    void getScores16(SeqPair *pairArray,
                     uint8_t *seqBufRef,
                     uint8_t *seqBufQer,
                     int32_t numPairs,
                     uint16_t numThreads,
                     int32_t w) override;

    void smithWatermanBatchWrapper16(SeqPair *pairArray,
                                     uint8_t *seqBufRef,
                                     uint8_t *seqBufQer,
                                     int32_t numPairs,
                                     uint16_t numThreads,
                                     int32_t w);

    void smithWaterman128_16(uint16_t seq1SoA[],
                             uint16_t seq2SoA[],
                             uint16_t nrow,
                             uint16_t ncol,
                             SeqPair *p,
                             uint16_t h0[],
                             uint16_t tid,
                             int32_t numPairs,
                             int zdrop,
                             int32_t w,
                             uint16_t qlen[],
                             uint16_t myband[]);
    
#endif  // ARM/SSE2

#if !defined(__ARM_NEON) && !defined(__aarch64__) && !defined(APPLE_SILICON)
#if ((!__AVX512BW__) & (__AVX2__))
    // AVX256 is not updated for banding and separate ins/del in the inner loop.
    // 8 bit vector code section
    void getScores8(SeqPair *pairArray,
                    uint8_t *seqBufRef,
                    uint8_t *seqBufQer,
                    int32_t numPairs,
                    uint16_t numThreads,
                    int32_t w) override;

    void smithWatermanBatchWrapper8(SeqPair *pairArray,
                                   uint8_t *seqBufRef,
                                   uint8_t *seqBufQer,
                                   int32_t numPairs,
                                   uint16_t numThreads,
                                   int32_t w);

    void smithWaterman256_8(uint8_t seq1SoA[],
                            uint8_t seq2SoA[],
                            int nrow,
                            int ncol,
                            SeqPair *p,
                            uint8_t h0[],
                            uint16_t tid,
                            int32_t numPairs,
                            int zdrop,
                            int32_t w,
                            uint8_t myband[]);
    // 16 bit vector code section
    void getScores16(SeqPair *pairArray,
                     uint8_t *seqBufRef,
                     uint8_t *seqBufQer,
                     int32_t numPairs,
                     uint16_t numThreads,
                     int32_t w) override;

    void smithWatermanBatchWrapper16(SeqPair *pairArray,
                                     uint8_t *seqBufRef,
                                     uint8_t *seqBufQer,
                                     int32_t numPairs,
                                     uint16_t numThreads,
                                     int32_t w);

    void smithWaterman256_16(uint16_t seq1SoA[],
                             uint16_t seq2SoA[],
                             uint16_t nrow,
                             uint16_t ncol,
                             SeqPair *p,
                             uint16_t h0[],
                             uint16_t tid,
                             int32_t numPairs,
                             int zdrop,
                             int32_t w,
                             uint16_t qlen[],
                             uint16_t myband[]);
    
#endif  //avx2
#endif  // !ARM guard for AVX2

#if !defined(__ARM_NEON) && !defined(__aarch64__) && !defined(APPLE_SILICON)
#if __AVX512BW__
    // 8 bit vector code section
    void getScores8(SeqPair *pairArray,
                    uint8_t *seqBufRef,
                    uint8_t *seqBufQer,
                    int32_t numPairs,
                    uint16_t numThreads,
                    int32_t w) override;

    void smithWatermanBatchWrapper8(SeqPair *pairArray,
                                   uint8_t *seqBufRef,
                                   uint8_t *seqBufQer,
                                   int32_t numPairs,
                                   uint16_t numThreads,
                                   int32_t w);

    void smithWaterman512_8(uint8_t seq1SoA[],
                            uint8_t seq2SoA[],
                            int nrow,
                            int ncol,
                            SeqPair *p,
                            uint8_t h0[],
                            uint16_t tid,
                            int32_t numPairs,
                            int zdrop,
                            int32_t w,
                            uint8_t myband[]);

    // 16 bit vector code section
    void getScores16(SeqPair *pairArray,
                     uint8_t *seqBufRef,
                     uint8_t *seqBufQer,
                     int32_t numPairs,
                     uint16_t numThreads,
                     int32_t w) override;

    void smithWatermanBatchWrapper16(SeqPair *pairArray,
                                     uint8_t *seqBufRef,
                                     uint8_t *seqBufQer,
                                     int32_t numPairs,
                                     uint16_t numThreads,
                                     int32_t w);
    
    void smithWaterman512_16(uint16_t seq1SoA[],
                             uint16_t seq2SoA[],
                             uint16_t nrow,
                             uint16_t ncol,
                             SeqPair *p,
                             uint16_t h0[],
                             uint16_t tid,
                             int32_t numPairs,
                             int zdrop,
                             int32_t w,
                             uint16_t qlen[],
                             uint16_t myband[]);
#endif  // __AVX512BW__
#endif  // !ARM guard for AVX512

    int64_t getTicks();
    
private:
    int m;
    int end_bonus, zdrop;
    int o_del, o_ins, e_del, e_ins;
    const int8_t *mat;

    int8_t w_match;
    int8_t w_mismatch;
    int8_t w_open;
    int8_t w_extend;
    int8_t w_ambig;

    // Scoring-matrix-derived state. bsw_generic_matrix / bsw_freed_cell /
    // build_pmat16 / build_amat16 are pure functions of mat / w_match /
    // w_mismatch / w_ambig -- all set once in the constructor and never mutated
    // -- yet each vector kernel rebuilt them from scratch on every SIMD-batch
    // call. Compute them once at construction (bsw_build_mat_cache) and have the
    // kernels read these members; the per-call result is bit-identical since the
    // inputs are call-invariant. Filled in the ctor (before any kernel runs), so
    // safe against the concurrent kernels that share one instance by tid.
    //
    // Invariant (structural, not defensively guarded): every constructor must
    // call bsw_build_mat_cache() before the instance is used. Holds because this
    // class is `final` with a single constructor and deleted copy/move, so no
    // instance can reach a kernel with the cache unfilled. A future second ctor
    // MUST call bsw_build_mat_cache() too -- default member initializers are
    // deliberately omitted (a zeroed cache would not be a safe fallback: the
    // kernels would read all-zero pmat/amat LUT bytes and mis-score, not degrade
    // to the symmetric fast path).
    //
    // Alignment: bsw_pmat_bytes_ / bsw_amat_bytes_ are read with the *aligned*
    // _mm_load_si128 at every kernel site. The aligned(16) attribute makes each a
    // 16-aligned member and forces alignof(BandedPairWiseSW) >= 16, so the bytes
    // are 16-aligned for *any* allocation of the object (heap, stack, or embedded)
    // -- the aligned load is always valid, not merely for the current new-only
    // instantiation. bsw_pmat16_lut_ is a 64-byte value but is intentionally kept
    // aligned(16) and read with the *unaligned* _mm512_loadu_si512: aligning it to
    // 64 would force alignof(class)==64 (an over-aligned-new dependency) for a LUT
    // load that runs once per SIMD batch, not in the DP inner loop -- not worth it.
    bool         bsw_gen_mat_;              // bsw_generic_matrix(...) || bsw_force_generic_matrix()
    BswFreedCell bsw_fc_;
    int8_t       bsw_pmat_bytes_[16]  __attribute__((aligned(16)));  // build_pmat16 (aligned _mm_load_si128)
    int8_t       bsw_amat_bytes_[16]  __attribute__((aligned(16)));  // build_amat16 (aligned _mm_load_si128)
    int16_t      bsw_pmat16_lut_[32]  __attribute__((aligned(16)));  // build_pmat16_lut (AVX-512 16b, loadu)
    void         bsw_build_mat_cache();

    int8_t *F8_;
    int8_t *H8_, *H8__;

    int16_t *F16_;
    int16_t *H16_, *H16__;

    // Single 64-byte-aligned slab backing F8_/H8_/H8__/F16_/H16_/H16__
    // and the per-thread SBT pre-pass scratch (sbt8_/sbt16_). The eight
    // member pointers above/below are views into this slab; the destructor
    // frees only `dp_slab_`. sbt8_ / sbt16_ moved off the stack — the
    // 16-bit variant (MAX_SEQ_LEN16 * SIMD_WIDTH16 * 2 bytes) is multi-MB
    // and was a stack-overflow risk on small-stack threads.
    int8_t  *sbt8_;
    int16_t *sbt16_;
    void *dp_slab_;

    int64_t sort1Ticks;
    int64_t setupTicks;
    int64_t swTicks;
    int64_t sort2Ticks;
};


#define DP  4
#define DP1 5
#define DP2 6
#define DP3 7

#endif
