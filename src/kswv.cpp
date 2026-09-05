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

Authors: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>
*****************************************************************************************/

#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include "kernel_dispatch.h"
#include "kswv.h"
#include "limits.h"
#include "utils.h"  /* xassert: release-active invariant guard */

/* Column blocking for the u8 kernels' post-row query-end recovery. The row's
 * running max is checkpointed every QE_BLK columns, so the recovery scans only
 * the block(s) holding the max instead of the whole row. Measured on a
 * 5M-read WGS slice at 64 lanes: 68.1 % of the rows that need a query end
 * resolve inside a single block and 89.6 % within two (mean 1.50), cutting
 * the scan from 160 columns to ~34 -- 4.7x. QE_BLK=16 is the flat part of
 * that curve; 8 is a wash and 32 is
 * worse, because halving the block-index sweep doubles the block being scanned.
 *
 * QE_MAXBLK bounds the checkpoint array. In the 8-bit kernels ncol cannot exceed
 * 256 -- the u8 admission bound (matesw_use_u8) caps the real query at len2 <=
 * 250, whose padded quantum8 is at most 256 -- so 256/QE_BLK blocks plus one for
 * a partial tail is always enough. */
#define QE_BLK    16
#define QE_MAXBLK (256 / QE_BLK + 1)

/* ARM/NEON support */
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
#include "neon_utils.h"
#include "simd_compat.h"
#endif


// ------------------------------------------------------------------------------------
// MACROs for vector code
#if MAINY
uint64_t prof[10][112];
#else
extern uint64_t prof[10][112];
#endif

#define AMBIG_ 4  // ambiguous base
// for 16 bit
#define DUMMY1_ 4
#define DUMMY2_ 5
#define DUMMY3 26
#define AMBR16 15
#define AMBQ16 16
// for 8-bit
#define DUMMY8 8
#define DUMMY5 5
#define AMBRQ 0xFF
#define AMBR 4
#define AMBQ 8

/* Query-padding contract, shared by the batch wrappers and the kernels.
 *
 * A lane's query occupies columns [0, len2); the wrapper pads [len2, quantum)
 * with the ambiguous-query code (DUMMY5 in the 8-bit kernels, DUMMY3 in the
 * 16-bit ones) and every column from the lane's quantum out to the group's
 * widest quantum with the high-bit sentinel (0xFF / 0xFFFF). The
 * kernels must therefore zero the DP diagonal term on any cell whose query
 * byte has bit 7 set, and can derive the FIRST such column for a whole group
 * as min(query_quantum(len2)) over its lanes.
 *
 * Both halves read the quantum from here so the fill and the mask can never
 * disagree: the lane count is the 8-bit SSE lane width (16) for the 8-bit
 * kernels and the 16-bit one (8) for the 16-bit kernels, matching bwa-mem2. */
static inline int query_quantum8(int len2)  { return ((len2 + 16 - 1) / 16) * 16; }
static inline int query_quantum16(int len2) { return ((len2 + 8 - 1) / 8) * 8; }

/* Why the two query pads differ, since only one of them is inert:
 *   [len2, quantum)      DUMMY5 -> sbt = shift, i.e. m11 = h00. This
 *                        reproduces ksw_qinit's profile-0 padding out to
 *                        slen*p (ksw.cpp:103) -- scalar ksw computes these
 *                        columns and scans them when picking qe (ksw.cpp:221),
 *                        so they are part of the answer and MUST be computed.
 *                        query_quantum8/16 IS ksw's slen*p.
 *   [quantum, maxLen2]   0xFF/0xFFFF. Pure SoA batching slack with no scalar
 *                        counterpart, so it must contribute NOTHING. Padding
 *                        it with DUMMY5 instead would carry the diagonal
 *                        undamped down-right across rows and can become a
 *                        later row's maximum -- clearing minsc and inventing a
 *                        second-best score. The high bit is what lets the
 *                        kernels zero m11 there instead.
 * A lane's own inert region therefore starts at query_quantum, and a whole
 * group's starts at the min over its lanes. */

/* First column of a group at which ANY lane can hold the DUMMY5 query pad:
 *
 *     min{ len2 : query_quantum8(len2) > len2 }   over the group's lanes
 *
 * A lane writes DUMMY5 only on [len2, quantum), so a lane whose len2 is already
 * a multiple of 16 contributes no such column at all and is skipped. Below the
 * minimum, the per-cell "is this the query's DUMMY5 tail padding?" test cannot
 * fire on any lane, so the 8-bit kernels elide it there. This is not a claim
 * about what that test COMPUTES -- it is that no input below the returned
 * column can select it, so the bytes written are identical either way.
 *
 * `cap` clamps the result to the range the caller actually subdivides: the NEON
 * and AVX-512BW kernels pass jsplit, because a group can have a jdummy above
 * jsplit (one lane padding from a long len2 while another has a short
 * exact-multiple len2) and splitting the expensive range as well would cost
 * more loop bodies than it saves. The AVX2 kernel has no jsplit range and
 * passes ncol, which the initial value already satisfies.
 *
 * `width` is the caller's lane count rather than SIMD_WIDTH8 directly so the
 * three kernels share one definition of a derivation the DUMMY5 elision's
 * correctness rests on -- three hand-maintained copies is how one tier comes to
 * drift from the others. */
static inline int compute_jdummy(const SeqPair *p, int width, int ncol, int cap)
{
    int jdummy = ncol;
    for (int l = 0; l < width; l++) {
        const int len2_l = p[l].len2;
        if (query_quantum8(len2_l) > len2_l && len2_l < jdummy) jdummy = len2_l;
    }
    return jdummy < cap ? jdummy : cap;
}

/* "No freed cell active on this lane" markers for the freed-cell (--meth)
 * blend, one per kernel width. Each must be a value a query byte can never
 * take, so the blend provably never fires on a lane with no active freed cell:
 *   8-bit   s2 holds 0-3, DUMMY5(5), AMBQ(8), or the 0xFF pad  -> 0xFE
 *   16-bit  s2 holds 0-3, AMBQ16(16), DUMMY3(26), or the 0xFFFF pad -> 0x7FFF
 * Both deliberately avoid the pad value itself. Aliasing it would make the
 * blend fire on every padded column, leaving correctness dependent on the
 * boundary mask zeroing m11 there -- two mechanisms propping each other up.
 * Spelled as #define to match the sentinel block above (DUMMY5/AMBQ/...). */
#define FREED_INACTIVE8  0xFE
#define FREED_INACTIVE16 0x7FFF


// -----------------------------------------------------------------------------------
#if __AVX512BW__

/* BND is the boundary mask: bit 7 set in the reference base (rows past len1) or
 * the query base (columns past this lane's quantum) marks a padding cell whose
 * diagonal term must be zeroed. Callers pass the hoisted per-row
 * rowboundary512 below jsplit, where no query byte can carry the high bit, and
 * the full per-cell movepi8_mask(or(s1,s2)) from jsplit on. */
#define MAIN_SAM_CODE8_OPT(s1, s2, h00, h11, e11, f11, f21, max512, sft512, BND, \
                           NEED_DUMMY)                                  \
    {                                                                   \
        __m512i sbt11, xor11;                                           \
        xor11 = _mm512_xor_si512(s1, s2);                               \
        sbt11 = _mm512_shuffle_epi8(permSft512, xor11);                 \
        /* DUMMY5 tail test; provably unselectable below jdummy, so the \
         * cmp+blend pair is elided there. See the jdummy derivation in \
         * the u8 kernel. */                                            \
        if (NEED_DUMMY) {                                               \
            __mmask64 cmpq = _mm512_cmpeq_epu8_mask(s2, five512);       \
            sbt11 = _mm512_mask_blend_epi8(cmpq, sbt11, sft512);        \
        }                                                               \
        if (HasFreed) {                                                 \
            /* One compare + one blend against the per-row active_frread512   \
             * target (built once per row); FREED_INACTIVE8 lanes never match \
             * any s2 value. Fewer ops than the two-blend form and, unlike the\
             * mask-OR collapse, no serializing kor on the blend's input.     \
             * The blend target is freedval512 (fr_val biased by +shift), so  \
             * NEUTRAL's zero-scored conversion cell uses the same path. */   \
            __mmask64 freed = _mm512_cmpeq_epi8_mask(s2, active_frread512); \
            sbt11 = _mm512_mask_blend_epi8(freed, sbt11, freedval512);  \
        }                                                               \
        __mmask64 cmp = (BND);                                          \
        __m512i m11 = _mm512_adds_epu8(h00, sbt11);                     \
        m11 = _mm512_mask_blend_epi8(cmp, m11, zero512);                \
        m11 = _mm512_subs_epu8(m11, sft512);                            \
        h11 = _mm512_max_epu8(m11, e11);                                \
        h11 = _mm512_max_epu8(h11, f11);                                \
        /* Row max only; the query-end column is recovered after the row \
         * from the QE_BLK checkpoints. See kswv_neon_u8_impl. */        \
        imax512 = _mm512_max_epu8(imax512, h11);                        \
        __m512i gapE512 = _mm512_subs_epu8(h11, oe_ins512);             \
        e11 = _mm512_subs_epu8(e11, e_ins512);                          \
        e11 = _mm512_max_epu8(gapE512, e11);                            \
        __m512i gapD512 = _mm512_subs_epu8(h11, oe_del512);             \
        f21 = _mm512_subs_epu8(f11, e_del512);                          \
        f21 = _mm512_max_epu8(gapD512, f21);                            \
    }

#define MAIN_SAM_CODE16_OPT(s1, s2, h00, h11, e11, f11, f21, max512)    \
    {                                                                   \
        __m512i sbt11, xor11, or11;                                     \
        xor11 = _mm512_xor_si512(s1, s2);                               \
        sbt11 = _mm512_permutexvar_epi16(xor11, perm512);               \
        if (HasFreed) {                                                 \
            /* One compare + one blend against the per-row active_frread512   \
             * target (built once per row); FREED_INACTIVE16 lanes never      \
             * match a real u16 s2 (0-3/AMBQ16=16/DUMMY3=26/0xFFFF pad). */   \
            __mmask32 freed = _mm512_cmpeq_epi16_mask(s2, active_frread512); \
            sbt11 = _mm512_mask_blend_epi16(freed, sbt11, freedval512); \
        }                                                               \
        __m512i m11 = _mm512_add_epi16(h00, sbt11);                     \
        or11 =  _mm512_or_si512(s1, s2);                                \
        __mmask64 cmp = _mm512_movepi8_mask(or11);                      \
        m11 = _mm512_mask_blend_epi8(cmp, m11, zero512);                \
        h11 = _mm512_max_epi16(m11, e11);                               \
        h11 = _mm512_max_epi16(h11, f11);                               \
        h11 = _mm512_max_epi16(h11, zero512);                           \
        __mmask32 cmp0 = _mm512_cmpgt_epi16_mask(h11, imax512);         \
        imax512 = _mm512_max_epi16(imax512, h11);                       \
        iqe512 = _mm512_mask_blend_epi16(cmp0, iqe512, l512);           \
        __m512i gapE512 = _mm512_sub_epi16(h11, oe_ins512);             \
        e11 = _mm512_sub_epi16(e11, e_ins512);                          \
        e11 = _mm512_max_epi16(gapE512, e11);                           \
        __m512i gapD512 = _mm512_sub_epi16(h11, oe_del512);             \
        f21 = _mm512_sub_epi16(f11, e_del512);                          \
        f21 = _mm512_max_epi16(gapD512, f21);                           \
    }

#endif

// constructor
kswv::kswv(const int o_del, const int e_del, const int o_ins,
           const int e_ins, const int8_t w_match, const int8_t w_mismatch,
           int numThreads, int32_t maxRefLen = MAX_SEQ_LEN_REF_SAM,
           int32_t maxQerLen = MAX_SEQ_LEN_QER_SAM)
{

    this->m = 5;
    this->o_del = o_del;
    this->o_ins = o_ins;
    this->e_del = e_del;
    this->e_ins = e_ins;
    
    this->w_match    = w_match;
    this->w_mismatch = w_mismatch;
    this->w_open     = o_del;  // redundant, used in vector code.
    this->w_extend   = e_del;  // redundant, used in vector code.
    this->w_ambig    = DEFAULT_AMBIG;
    this->g_qmax = max_(w_match, w_mismatch);
    this->g_qmax = max_(this->g_qmax, w_ambig);

    this->maxRefLen = maxRefLen + 16;
    this->maxQerLen = maxQerLen + 16;
    
    this->swTicks = 0;
    setupTicks = 0;
    sort1Ticks = 0;
    swTicks = 0;
    sort2Ticks = 0;

    F16     = (int16_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 64);
    H16_0   = (int16_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 64);
    H16_1   = (int16_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 64);
    rowMax16 = (int16_t *)_mm_malloc(this->maxRefLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 64);

    F8 = (uint8_t*) F16;
    H8_0 = (uint8_t*) H16_0;
    H8_1 = (uint8_t*) H16_1;
    rowMax8 = (uint8_t*) rowMax16;

    /* Column-index broadcast table for the u8 kernels' post-row query-end
     * rescan; see kswv.h. Shared by all threads and never written after
     * construction. */
    colIdx8 = (uint8_t *)_mm_malloc((size_t) this->maxQerLen * SIMD_WIDTH8
                                    * sizeof(uint8_t), 64);
    for (int32_t j = 0; j < this->maxQerLen; ++j)
        for (int k = 0; k < SIMD_WIDTH8; ++k)
            colIdx8[(size_t) j * SIMD_WIDTH8 + k] = (uint8_t) j;
}

// Mat-aware constructor (issue 173). Delegates to the 9-arg ctor (identical
// buffer allocation / field init), then inspects the supplied 5x5 scoring
// matrix for the asymmetric (bisulfite OT/OB) freed cell.
//
//   nullptr / symmetric matrix ⇒ the existing symmetric XOR-LUT kernel runs
//     unchanged (has_freed = needs_scalar = false).
//   exactly one off-diagonal cell freed to ANY value in [w_mismatch, w_match]
//     (genomic OT/OB frees it to +a, neutral to 0) ⇒ has_freed, fr_ref/fr_read
//     name it and fr_val carries the value; the kernel blends the cell to fr_val.
//   an exact mirrored freed pair (collapsed --meth) ⇒ has_freed, both ordered
//     cells named; the kernel applies both blends (the single-cell case is the
//     degenerate one where the mirror equals the primary).
//   any other asymmetric matrix (non-mirror multi-cell free, changed diagonal,
//     freed value outside [w_mismatch, w_match]) ⇒ needs_scalar; the caller
//     routes those pairs to ksw_align2.
//   On SSE41/SSE42/AVX tiers (no HasFreed kernel override) ⇒ needs_scalar for
//     ANY freed-cell matrix, so the caller falls back to ksw_align2 rather than
//     running a kernel that would drop the bisulfite override.
//
// Sign convention: w_match is +a, w_mismatch is -b (negative), and mat25's
// off-diagonals are likewise the negated penalty. All three are sign-consistent
// with each other, so they are passed straight through to the Task-1 detectors
// without re-negation.
kswv::kswv(const int o_del, const int e_del, const int o_ins,
           const int e_ins, const int8_t w_match, const int8_t w_mismatch,
           int numThreads, int32_t maxRefLen, int32_t maxQerLen,
           const int8_t *mat25)
    : kswv(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
           numThreads, maxRefLen, maxQerLen)
{
    if (mat25 != nullptr) {
        // Only the NEON, AVX2, and AVX-512BW kernel bodies implement the
        // HasFreed override; the SSE41/SSE42/AVX tiers ship getScores8/16 as
        // unreachable exit() stubs (see the SSE-only fallback below). Detect
        // the freed cell(s) only where the kernel can actually apply them; on a
        // freed-cell-less tier, leave needs_scalar = true so the caller routes
        // the pair to the scalar ksw_align2 fallback instead of running a
        // kernel that would silently drop the bisulfite override.
        //
        // This compile-time test IS the runtime-tier test, and is not a
        // baseline-vs-host mismatch waiting to happen. kswv.cpp is compiled
        // ONCE PER TIER (KERNEL_SRCS x {sse41,sse42,avx,avx2,avx512bw} in the
        // Makefile), each copy under that tier's -m flags with the class name
        // mangled to kswv_<tier> by kernel_dispatch.h. simd_dispatch.cpp's
        // make_kswv() switches on the runtime g_tier (incl. BWAMEM3_FORCE_TIER)
        // to call make_kswv_kernel_<tier>_mat, which constructs THIS ctor out
        // of the <tier> TU — so the macros below are exactly the ones that
        // selected the kernel body compiled alongside it. Forcing sse41 on an
        // AVX2 host runs kswv_sse41::kswv from kswv.sse41.o, where __AVX2__ is
        // undefined and needs_scalar is correctly set. Keep this predicate in
        // sync with the #if/#elif ladder that guards the kernel bodies below.
#if defined(__AVX512BW__) || defined(__AVX2__) \
        || defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
        constexpr bool freed_simd_supported = true;
#else
        constexpr bool freed_simd_supported = false;
#endif
        // Fold `forced` (BWAMEM3_FORCE_GENMAT) into the outer gate so the
        // documented symmetric no-op path is exercised even for a matrix
        // bsw_generic_matrix() would otherwise treat as symmetric.
        const bool forced  = bsw_force_generic_matrix();
        const bool generic = bsw_generic_matrix(mat25, w_match, w_mismatch) || forced;
        if (generic) {
            BswFreedCell fc = bsw_freed_cell(mat25, w_match, w_mismatch, forced);
            if (fc.single) {
                // ONE off-diagonal cell freed. GENOMIC frees it to a match
                // (fc.value == w_match); NEUTRAL frees it to 0. Either way the
                // mirror == primary, so the kernel's second blend is idempotent.
                // The kernel blends the freed cell to fr_val (see fr_val member);
                // fr_val == w_match reproduces the pre-generalization match blend.
                if (!freed_simd_supported) { needs_scalar = true; return; }
                // bsw_freed_cell accepts ANY freed value, but the 8-bit kernel
                // blends fr_val + shift into the biased u8 domain, which only
                // represents [w_mismatch, w_match]. mem_opt_fill_meth_mat never
                // produces anything outside it (+a or 0), so this cannot fire
                // today — but a freed value below/above the domain would wrap the
                // biased byte and SILENTLY mis-score, so route it to the scalar
                // ksw_align2 fallback (always correct) rather than trust it.
                if (fc.value < w_mismatch || fc.value > w_match) {
                    needs_scalar = true; return;
                }
                has_freed = true;
                fr_ref  = fr_ref2  = fc.ref;
                fr_read = fr_read2 = fc.read;
                fr_val  = fc.value;
            } else {
                // COLLAPSED (the --meth default) frees the conversion cell AND
                // its mirror to a match; the kernel frees both ordered cells.
                BswFreedPair fp = bsw_freed_pair(mat25, w_match, w_mismatch);
                if (fp.supported) {
                    if (!freed_simd_supported) { needs_scalar = true; return; }
                    has_freed = true;
                    fr_ref  = fp.refA;  fr_read  = fp.readA;
                    fr_ref2 = fp.refB;  fr_read2 = fp.readB;
                    fr_val  = w_match;   // collapsed frees both cells to a match
                } else {
                    needs_scalar = true;
                }
            }
        }
    }
}

// destructor 
kswv::~kswv() {
    _mm_free(F16); _mm_free(H16_0); _mm_free(H16_1);
    _mm_free(rowMax16);
    _mm_free(colIdx8);
}


/* u8-tier saturation guard, shared by every u8 kernel (NEON/AVX2/AVX-512).
 * Defined ahead of the per-arch kernel-body ladder so all three tiers see it.
 *
 * The u8 tier is byte-identical to the 16-bit tier only while the max attainable
 * biased score cannot saturate the uint8_t running value. Bound it on the
 * group's REAL max query length -- max over the lanes of p[i].len2 (padding
 * lanes carry len2 == 0, and every column past a lane's quantum is inert and
 * adds no score) -- NOT the padded column count `ncol`. query_quantum8 rounds a
 * 241..249 bp mate up to 256, so a guard on `ncol` false-trips (256*1 + shift >=
 * 256) on inputs the u8 admission bound in mem_matesw already proved safe; that
 * over-conservative guard aborted lightly-trimmed 2x250 mates on release builds.
 *
 * The routing contract (see matesw_use_u8) admits a pair only when
 * len2*w_match + shift <= 254, which also implies no biased saturation. A
 * violation here is silent score/XS/MAPQ corruption, not a crash, so it is an
 * xassert (release-active). */
static inline void kswv_u8_saturation_guard(const SeqPair *p, int8_t w_match, uint8_t shift) {
    int maxq = 0;
    for (int i = 0; i < SIMD_WIDTH8; ++i) if (p[i].len2 > maxq) maxq = p[i].len2;
    xassert((int64_t)maxq * (int64_t)w_match + (int64_t)shift <= 254,
            "u8 rescue tier: max score + bias can saturate; USQADD and biased "
            "forms would diverge -- the 16-bit tier must handle this score range");
}

/*******************************************************************************
 * ARM/NEON Implementation
 * Native NEON for 128-bit vectors (16 x 8-bit or 8 x 16-bit elements)
 * This provides optimized SIMD on Apple Silicon where sse2neon can't help
 * (AVX-512 has no sse2neon translation)
 ******************************************************************************/
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)

void kswv::getScores8(SeqPair *pairArray,
                      uint8_t *seqBufRef,
                      uint8_t *seqBufQer,
                      kswr_t* aln,
                      int32_t numPairs,
                      uint16_t numThreads,
                      int phase)
{
    kswvBatchWrapper8(pairArray, seqBufRef, seqBufQer, aln,
                      numPairs, numThreads, phase);
}

void kswv::kswvBatchWrapper8(SeqPair *pairArray,
                             uint8_t *seqBufRef,
                             uint8_t *seqBufQer,
                             kswr_t* aln,
                             int32_t numPairs,
                             uint16_t numThreads,
                             int phase)
{
    uint8_t *seq1SoA = NULL;
    seq1SoA = (uint8_t *)_mm_malloc(this->maxRefLen * SIMD_WIDTH8 * numThreads * sizeof(uint8_t), 128);

    uint8_t *seq2SoA = NULL;
    seq2SoA = (uint8_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH8 * numThreads * sizeof(uint8_t), 128);

    assert(seq1SoA != NULL);
    assert(seq2SoA != NULL);

    int32_t ii;
    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH8 - 1) / SIMD_WIDTH8) * SIMD_WIDTH8;

    for (ii = numPairs; ii < roundNumPairs; ii++)
    {
        pairArray[ii].regid = ii;
        pairArray[ii].id = ii;
        pairArray[ii].len1 = 0;
        pairArray[ii].len2 = 0;
        // The kernel's per-lane loop reads idr/idq/h0 for every lane up to
        // SIMD_WIDTH, so it reads these padded lanes too; zero them (as id/len1/
        // len2 above) to keep those reads well-defined. Padded lanes join the SIMD
        // batch (and its cross-lane reductions), but the caller reads results back
        // only for real lanes and whole-aligner output is byte-identical (validated).
        pairArray[ii].idr = 0;
        pairArray[ii].idq = 0;
        pairArray[ii].h0 = 0;
    }

    {
        int32_t i;
        uint16_t tid = 0;   // forcing single thread
        uint8_t *mySeq1SoA = seq1SoA + tid * this->maxRefLen * SIMD_WIDTH8;
        uint8_t *mySeq2SoA = seq2SoA + tid * this->maxQerLen * SIMD_WIDTH8;
        uint8_t *seq1;
        uint8_t *seq2;

        int nstart = 0, nend = numPairs;

        for (i = nstart; i < nend; i += SIMD_WIDTH8)
        {
            int32_t j, k;
            int maxLen1 = 0;
            int maxLen2 = 0;

            for (j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq1 = seqBufRef + (int64_t)sp.id * this->maxRefLen;
#else
                seq1 = seqBufRef + sp.idr;
#endif
                for (k = 0; k < sp.len1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH8 + j] = (seq1[k] == AMBIG_ ? AMBR : seq1[k]);
                }
                if (maxLen1 < sp.len1) maxLen1 = sp.len1;
            }
            for (j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
                for (k = sp.len1; k <= maxLen1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH8 + j] = 0xFF;
                }
            }

            for (j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq2 = seqBufQer + (int64_t)sp.id * this->maxQerLen;
#else
                seq2 = seqBufQer + sp.idq;
#endif
                int quanta = query_quantum8(sp.len2);
                for (k = 0; k < sp.len2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH8 + j] = (seq2[k] == AMBIG_ ? AMBQ : seq2[k]);
                }

                for (k = sp.len2; k < quanta; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH8 + j] = DUMMY5;
                }
                if (maxLen2 < quanta) maxLen2 = quanta;
            }

            for (j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
                int quanta = query_quantum8(sp.len2);
                for (k = quanta; k <= maxLen2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH8 + j] = 0xFF;
                }
            }

            kswv_neon_u8(mySeq1SoA, mySeq2SoA,
                         maxLen1, maxLen2,
                         pairArray + i,
                         aln, i,
                         tid,
                         numPairs,
                         phase);
        }
    }

    _mm_free(seq1SoA);
    _mm_free(seq2SoA);

    return;
}

/* USQADD: apply the substitution score with a single
 * vsqaddq_u8 (AArch64 USQADD, unsigned-saturating accumulate of a SIGNED addend)
 * on a signed-delta score table, instead of the biased vqaddq_u8 + de-biasing
 * vqsubq_u8 pair. One op fewer, and it removes the de-bias from the DIAGONAL
 * loop-carried chain (h00 -> m11 -> h11), shortening the critical path of this
 * latency-bound kernel. Byte-identical: usqadd(h00, delta) == clamp(h00+delta,
 * 0, 255), the exact value the biased pair produces in the non-saturating range
 * the assert below guarantees; the H domain is unbiased in BOTH forms (the bias
 * cancels within the cell), so every downstream H comparison is unchanged.
 * Default ON; BWA3_RESCUE_USQADD=0 restores the biased form. Monomorphised as a
 * template bool (folds like HasFreed), never a per-cell branch. */
/* Read a BWA3_RESCUE_* on/off toggle (default ON; a leading "0" disables). Not
 * cached: the env is read on every call so a unit test can flip the selected
 * path in-process (the parity test relies on this) and a mid-run override takes
 * effect. The cost is one getenv per rescue batch -- never per cell -- so it is
 * negligible; the runtime bool still folds to a monomorphised template. */
static bool rescue_env_on(const char *var)
{
    const char *e = getenv(var);
    return !e || e[0] != '0';
}

static bool rescue_usqadd_enabled() { return rescue_env_on("BWA3_RESCUE_USQADD"); }

/* Two-target-row blocking of the u8 rescue kernel. Processing rows i and i+1 in
 * one column sweep lets the pair share the query-column load, the vertical-carry
 * load (F[j+1]) and the H/F stores -- row i+1's diagonal is row i's previous-column
 * H and its vertical carry is what row i just produced, both handed over in
 * registers -- so a pair costs five memory ops for two cells instead of ten, and
 * the two independent H chains give the out-of-order engine something to
 * interleave (the point for this latency-bound kernel). Byte-identical: the
 * per-cell arithmetic is untouched and the row epilogues still run in row order,
 * so a freeze at row i still suppresses row i+1.
 *
 * Default ON; BWA3_RESCUE_ROWPAIR=0 restores the one-row-at-a-time sweep.
 * Monomorphised as a template bool, never a per-cell branch. */
static bool rescue_rowpair_enabled() { return rescue_env_on("BWA3_RESCUE_ROWPAIR"); }

/* Query-end recovery in the two-row sweep. Row i+1's H is stored (H1) and can be
 * rescanned after the row exactly as the one-row body does; row i's H only ever
 * reached row i+1 through a register, so the paired sweep originally tracked the
 * argmax column INLINE for both rows: a strict-greater compare, a blend of a
 * carried column counter, and the counter's increment -- five of the loop's ~31
 * vector ALU ops, in a loop that is issue-bound rather than latency-bound (two
 * independent E chains of two ops each against ~8 cycles of issue on four SIMD
 * pipes). Those are the same three per-cell ops the one-row body dropped for
 * -2.6% whole-aligner CPU on NEON when it went lazy.
 *
 * LazyQE stores row i's H too, in place: after the pair loads its diagonal
 * H0[j], that slot is dead for the rest of the sweep (the next column reads
 * H0[j+1]), so row i's H for column j is written back to H0[j] -- one store on
 * the idle store pipes instead of five ALU ops -- and the post-row rescan reads
 * row i from H0[j2] where it reads row i+1 from H1[j2+1]. The buffer is fully
 * rewritten by the next row after the swap, and H0[0] (the column -1 boundary,
 * which must read 0) is re-zeroed before the swap. Row maxima are checkpointed
 * per QE_BLK columns for both rows so the rescan stays block-bounded, exactly as
 * in the one-row body. Byte-identical: both forms compute
 * min{ j : H[j] == rowmax } over the same stored values. Default ON;
 * BWA3_RESCUE_LAZYQE=0 restores the inline argmax. */
static bool rescue_lazyqe_enabled() { return rescue_env_on("BWA3_RESCUE_LAZYQE"); }

/* Thin dispatcher: route to the HasFreed × USQADD × RowPair × LazyQE template
 * instantiation (see the per-flag notes at the dispatch site below). The
 * <false> HasFreed path dead-code-eliminates every freed-cell override →
 * byte-identical to the pre-issue-173 kernel for symmetric (non-meth) matrices. */
int kswv::kswv_neon_u8(uint8_t seq1SoA[],
                       uint8_t seq2SoA[],
                       int16_t nrow,
                       int16_t ncol,
                       SeqPair *p,
                       kswr_t *aln,
                       int po_ind,
                       uint16_t tid,
                       int32_t numPairs,
                       int phase)
{
    const bool usq = rescue_usqadd_enabled();
    const bool pair = rescue_rowpair_enabled();
    const bool lazy = pair && rescue_lazyqe_enabled();

    /* Route to the HasFreed x USQADD x RowPair x LazyQE instantiation. Each flag
     * folds a per-cell branch out of the monomorphised body; the runtime dispatch
     * is a once-per-call decision, not a per-cell one. LazyQE only exists for the
     * two-row sweep, so the one-row instantiations are always <..., false, false>
     * (12 bodies, not 16). */
#define KSWV_U8_DISPATCH(HF, UQ, RP, LQ)                                        \
    kswv_neon_u8_impl<HF, UQ, RP, LQ>(seq1SoA, seq2SoA, nrow, ncol, p, aln,     \
                                      po_ind, tid, numPairs, phase)
#define KSWV_U8_DISPATCH_PAIR(HF, UQ)                                           \
    (pair ? (lazy ? KSWV_U8_DISPATCH(HF, UQ, true, true)                        \
                  : KSWV_U8_DISPATCH(HF, UQ, true, false))                      \
          : KSWV_U8_DISPATCH(HF, UQ, false, false))
    if (has_freed) {
        if (usq)  return KSWV_U8_DISPATCH_PAIR(true, true);
        else      return KSWV_U8_DISPATCH_PAIR(true, false);
    }
    if (usq)  return KSWV_U8_DISPATCH_PAIR(false, true);
    else      return KSWV_U8_DISPATCH_PAIR(false, false);
#undef KSWV_U8_DISPATCH_PAIR
#undef KSWV_U8_DISPATCH
}

template<bool HasFreed, bool USQADD, bool RowPair, bool LazyQE>
int kswv::kswv_neon_u8_impl(uint8_t seq1SoA[],
                            uint8_t seq2SoA[],
                            int16_t nrow,
                            int16_t ncol,
                            SeqPair *p,
                            kswr_t *aln,
                            int po_ind,
                            uint16_t tid,
                            int32_t numPairs,
                            int phase)
{
    int m_b, n_b;
    uint8_t minsc[SIMD_WIDTH8] __attribute__((aligned(128))) = {0};
    uint8_t endsc[SIMD_WIDTH8] __attribute__((aligned(128))) = {0};
    uint64_t *b;

    uint8x16_t zero_vec = vdupq_n_u8(0);

    /* Per-row running-max checkpoints, one vector per QE_BLK columns. Rewritten
     * every row, so no clearing is needed; a few hundred bytes of stack, hot in
     * L1 beside H0/H1/F. */
    alignas(64) uint8_t blockMax[QE_MAXBLK * SIMD_WIDTH8];
    /* Second checkpoint array for row i+1 of the two-row sweep under LazyQE;
     * blockMax serves row i there and the one-row body otherwise. */
    alignas(64) uint8_t blockMax1[QE_MAXBLK * SIMD_WIDTH8];
    (void) blockMax1;

    m_b = n_b = 0; b = 0;

    int8_t temp[SIMD_WIDTH8] __attribute((aligned(128))) = {0};

    uint8_t shift = 127, mdiff = 0;
    mdiff = max_(this->w_match, (int8_t)this->w_mismatch);
    mdiff = max_(mdiff, (int8_t)this->w_ambig);
    shift = min_(this->w_match, (int8_t)this->w_mismatch);
    shift = min_((int8_t)shift, this->w_ambig);

    shift = 256 - (uint8_t)shift;
    mdiff += shift;

    /* Build scoring table */
    temp[0] = this->w_match;
    temp[1] = temp[2] = temp[3] = this->w_mismatch;
    temp[4] = temp[5] = temp[6] = temp[7] = this->w_ambig;
    temp[8] = temp[9] = temp[10] = temp[11] = this->w_ambig;
    temp[12] = this->w_ambig;

    /* USQADD: the table holds the SIGNED deltas themselves and there is no bias
     * (the single vsqaddq_u8 is clamp(h + delta, 0, 255) directly, the exact
     * value the biased add+de-bias pair yields whenever the unbiased result is
     * in [0,255] -- which holds for every live cell of the u8 tier, selected
     * only when the max attainable score fits u8). The biased form instead adds
     * `shift` to every entry here and removes it per cell. Byte-identity is
     * gated per-field by the golden check. */
    if (!USQADD) {
        for (int i = 0; i < 16; i++)
            temp[i] += shift;
    }

    /* u8-tier saturation guard. Bounds on the group's REAL max query length,
     * not the padded `ncol` (which rounds 241..249 up to 256 and false-trips on
     * safe, admitted mates). See kswv_u8_saturation_guard. */
    kswv_u8_saturation_guard(p, this->w_match, shift);

    uint8x16_t permSft = vld1q_u8((const uint8_t*)temp);
    uint8x16_t sft_vec = vdupq_n_u8(shift);

    uint16_t minsc_msk_a = 0x0000, endsc_msk_a = 0x0000;
    int val = 0;
    for (int i = 0; i < SIMD_WIDTH8; i++)
    {
        int xtra = p[i].h0;
        val = (xtra & KSW_XSUBO) ? xtra & 0xffff : 0x10000;
        if (val <= 255) {
            minsc[i] = val;
            minsc_msk_a |= (0x1 << i);
        }
        val = (xtra & KSW_XSTOP) ? xtra & 0xffff : 0x10000;
        if (val <= 255) {
            endsc[i] = val;
            endsc_msk_a |= (0x1 << i);
        }
    }

    uint8x16_t minsc_vec = vld1q_u8(minsc);
    uint8x16_t endsc_vec = vld1q_u8(endsc);

    uint8x16_t e_del_vec = vdupq_n_u8(this->e_del);
    uint8x16_t oe_del_vec = vdupq_n_u8(this->o_del + this->e_del);
    uint8x16_t e_ins_vec = vdupq_n_u8(this->e_ins);
    uint8x16_t oe_ins_vec = vdupq_n_u8(this->o_ins + this->e_ins);
    uint8x16_t five_vec = vdupq_n_u8(DUMMY5);
    /* Per-cell increment for the two-row sweep's carried argmax column counter. */
    const uint8x16_t one_vec = vdupq_n_u8(1);
    /* Padding sentinel test: bit 7 set marks a reference row past len1 or a
     * query column past this lane's quantum. */
    const uint8x16_t highbit_vec = vdupq_n_u8(0x80);
    uint8x16_t gmax_vec = zero_vec;
    // 8-bit SW processes 16 pairs per SIMD batch, but te must be tracked in
    // 16-bit precision (ref positions > 255 are possible). Use two
    // int16x8 vectors — _lo for pairs 0-7, _hi for pairs 8-15 — so every
    // pair has its own te slot. The earlier single int16x8 te_vec plus
    // hacky "duplicate on store" only tracked pairs 0-7 correctly; pairs
    // 8-15 got copies of the wrong pair's te.
    int16x8_t te_vec_lo = vdupq_n_s16(-1);
    int16x8_t te_vec_hi = vdupq_n_s16(-1);
    uint8x16_t cmax_vec = vdupq_n_u8(255);
    // "Frozen" mask: tracks which pairs have hit their KSW_XSTOP target
    // in a prior row. Once frozen, a pair's gmax/te/qe must not be updated
    // — otherwise later rows (which may contain untouched ref bytes when
    // the caller only partially reversed the buffer) can add spurious
    // matches, pushing phase-1's score above phase-0's and breaking the
    // write-back gate `aln[ind].score == score[l]` at line ~538.
    uint8x16_t frozen_vec = zero_vec;

    /* Per-lane mask of "this lane has a real KSW_XSTOP target set" (derived
     * from endsc_msk_a). endsc[i] defaults to 0 for lanes without a target,
     * which would otherwise make vcgeq_u8(gmax, endsc_vec) trivially true
     * and cause premature freezing. */
    uint8_t _has_endsc_bytes[SIMD_WIDTH8];
    for (int _i = 0; _i < SIMD_WIDTH8; _i++) {
        _has_endsc_bytes[_i] = (endsc_msk_a & (1 << _i)) ? 0xFF : 0x00;
    }
    uint8x16_t has_endsc_vec = vld1q_u8(_has_endsc_bytes);

    uint16_t exit0 = 0xFFFF;

    tid = 0;
    uint8_t *H0 = H8_0 + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *H1 = H8_1 + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *F = F8 + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *rowMax = rowMax8 + tid * SIMD_WIDTH8 * this->maxRefLen;

    /* Per-strip warm-up prefetches. F and H1 are read+written every row across
     * all columns (loop-resident, fit in L1), and seq2SoA is swept every row,
     * so they want a keep hint (locality 1-2), not NTA. seq1SoA is read once
     * per row (one vector at seq1SoA[i]) — genuinely streaming, so locality 0.
     * Args: (addr, rw=0 read, locality 0 lowest .. 3 highest). */
    __builtin_prefetch(F + SIMD_WIDTH8, 0, 1);
    __builtin_prefetch(seq2SoA, 0, 2);
    __builtin_prefetch(seq1SoA, 0, 0);
    __builtin_prefetch(H1 + SIMD_WIDTH8, 0, 1);

    /* Initialize arrays */
    for (int i = 0; i <= ncol; i++)
    {
        vst1q_u8(H0 + i * SIMD_WIDTH8, zero_vec);
        vst1q_u8(F + i * SIMD_WIDTH8, zero_vec);
    }

    uint8x16_t max_vec = zero_vec, imax_vec, pimax_vec = zero_vec;
    uint16_t mask16 = 0x0000;
    uint16_t minsc_msk = 0x0000;

    uint8x16_t qe_vec = vdupq_n_u8(0);
    vst1q_u8(H0, zero_vec);
    vst1q_u8(H1, zero_vec);

    /* Boundary-mask fast-path bounds, both derived from the group's lanes:
     *   jsplit  - first column any lane pads with the 0xFF query sentinel, i.e.
     *             min(query_quantum8(len2)). Below it no query byte can carry
     *             bit 7, so the mask reduces to the per-row high_bit(s1).
     *   minLen1 - shortest reference window. Below it no lane pads its
     *             reference either, so the mask is provably all-zero and the
     *             blend is skipped outright.
     * Together they split the inner loop into three ranges (see below), of
     * which a uniform-length group only ever runs the cheapest. That is the
     * common case in PHASE 0 -- production ref windows and read lengths cluster
     * tightly -- but not in phase 1, where the caller re-enters with
     * len2 = qe + 1 per pair (bwamem_pair.cpp:875): those groups are ragged
     * whatever the input lengths were, so jsplit < ncol and the per-cell range
     * covers most of the row. The measured phase-0 parity therefore does not
     * carry over to phase 1.
     *
     * That cost is now MEASURED, not just predicted: a standalone SW-kernel
     * benchmark drives phase 1 and times it separately (its rescue mode prints
     * a phase-1 table alongside phase 0), and on quiet cloud hosts the
     * per-cell range costs phase-1 8-bit throughput -7.2% at qlen 100 and
     * -4.3% at qlen 150 on avx512bw, and -2.1% / -1.4% on Graviton4 NEON.
     * Phase 0 is at parity on every tier, as
     * this fast path intends.
     *
     * Two cross-checks fall out of that, and both hold, which is what makes the
     * attribution to THIS range rather than something ambient:
     *   - avx2 measures ~0 (-0.8% / +0.5%, i.e. noise). It must: avx2 never had
     *     the hoist to restore, so the fix did not change its code at all (see
     *     the register-pressure note in the avx2 kernel below).
     *   - the cost vanishes at qlen 250 (+0.5% avx512bw, -0.0% NEON), where the
     *     router picks the 16-bit kernel, which this hoist does not touch.
     *
     * The lever, if that ever matters: sort a phase-1 batch by len2 so each lane
     * group's quanta agree and jsplit returns to ncol. bwamem.cpp:3462 already
     * does precisely this for the extend path (sortPairsLen with key_by_max,
     * "so ncol/minq are tight"), and it is byte-identical there because results
     * scatter back by regid. Not done here because the end-to-end payoff is
     * projected below the +/-0.5% noise floor: kswv is ~31% of mem CPU and phase 1
     * is a fraction of that, and EXT-13 -- the same technique -- was predicted at
     * 1-3% and measured -0.19%.
     * Dummy tail lanes carry len1 == len2 == 0, pinning both to 0 for
     * a partial final group: no fast path, still correct. */
    int minLen1 = p[0].len1, jsplit = ncol;
    for (int l = 0; l < SIMD_WIDTH8; l++) {
        if (p[l].len1 < minLen1) minLen1 = p[l].len1;
        const int quanta = query_quantum8(p[l].len2);
        if (quanta < jsplit) jsplit = quanta;
    }

    /* Same unswitch one alphabet symbol over. Every cell pays a vceqq+vbslq to
     * ask "is this column the query's DUMMY5 tail padding?", and per the fill
     * contract above that is a provable no-op below compute_jdummy's column --
     * nothing else in the alphabet is 5 (real bases are 0-3, ambiguous is
     * AMBQ=8, the tail is 0xFF). See compute_jdummy for the derivation and for
     * why the result is clamped to jsplit.
     *
     * Production pays this on 150 bp queries padded to 160, so 150 of every 160
     * columns take the cheap body. Ragged phase-1 groups (len2 = qe + 1 per
     * pair) push jdummy down and get correspondingly less. */
    const int jdummy = compute_jdummy(p, SIMD_WIDTH8, ncol, jsplit);

    /* Per-row epilogue, shared by the one-row and two-row sweeps. Threads the
     * cross-row state (pimax_vec/mask16 lagged row-max store, gmax/te/qe, the
     * frozen mask and exit0) and MUST run once per row in ascending row order,
     * so freezing at row i suppresses row i+1 -- which is exactly what lets the
     * paired sweep call it twice back to back and stay byte-identical to the
     * one-row loop. EPI_INLINE_QE selects how the query-end column is found:
     *   false -- the one-row path's lazy rescan of the stored H for this row,
     *            restricted to the QE_BLK block(s) that can hold the row max;
     *   true  -- a column value already carried per-cell by the paired sweep
     *            (its row i never stores H, so there is nothing to rescan).
     * Both compute min{ j : H1[j+1] == imax }, so the two are byte-identical. */
#define KSWV_U8_EPILOGUE(EPI_ROW, EPI_IMAX, EPI_INLINE_QE, EPI_COL, EPI_BLOCKMAX, EPI_HBASE) \
    {                                                                           \
        const int __row = (EPI_ROW);                                            \
        const uint8x16_t __imax = (EPI_IMAX);                                   \
        uint8_t *const __bmax = (EPI_BLOCKMAX);                                 \
        const uint8_t *const __hbase = (EPI_HBASE);                             \
        int16x8_t i_vec = vdupq_n_s16(__row);                                   \
        /* Close the final (possibly partial) block for the lazy scan; the      \
         * inline-argmax path neither fills the checkpoints nor scans them. */  \
        if (!(EPI_INLINE_QE))                                                    \
            vst1q_u8(__bmax + ((ncol - 1) / QE_BLK) * SIMD_WIDTH8, __imax);      \
        /* Block I - row max tracking (lagged one row via pimax_vec). */        \
        if (__row > 0) {                                                        \
            uint8x16_t cmp_gt = vcgtq_u8(__imax, pimax_vec);                     \
            uint16_t msk16 = neon_movemask_u8(cmp_gt);                           \
            msk16 |= mask16;                                                     \
            pimax_vec = vbslq_u8(cmp_gt, zero_vec, pimax_vec);                   \
            vst1q_u8(rowMax + (__row - 1) * SIMD_WIDTH8, pimax_vec);             \
            mask16 = ~msk16;                                                     \
        }                                                                       \
        pimax_vec = __imax;                                                      \
        /* Check minsc threshold */                                             \
        uint8x16_t cmp_ge = vcgeq_u8(__imax, minsc_vec);                         \
        minsc_msk = neon_movemask_u8(cmp_ge);                                    \
        minsc_msk &= minsc_msk_a;                                                \
        /* Block II: gmax, te */                                               \
        uint8x16_t cmp0 = vcgtq_u8(__imax, gmax_vec);                            \
        uint16_t cmp0_msk = neon_movemask_u8(cmp0);                              \
        cmp0_msk &= exit0;                                                       \
        uint16x8_t cmp_lo_16 = vreinterpretq_u16_u8(vzip1q_u8(cmp0, cmp0));      \
        uint16x8_t cmp_hi_16 = vreinterpretq_u16_u8(vzip2q_u8(cmp0, cmp0));      \
        uint16x8_t frozen_lo_16 = vreinterpretq_u16_u8(vzip1q_u8(frozen_vec, frozen_vec)); \
        uint16x8_t frozen_hi_16 = vreinterpretq_u16_u8(vzip2q_u8(frozen_vec, frozen_vec)); \
        cmp_lo_16 = vbicq_u16(cmp_lo_16, frozen_lo_16);                          \
        cmp_hi_16 = vbicq_u16(cmp_hi_16, frozen_hi_16);                          \
        uint8x16_t cmp0_active = vbicq_u8(cmp0, frozen_vec);                     \
        uint8x16_t new_gmax = vmaxq_u8(gmax_vec, __imax);                        \
        gmax_vec = vbslq_u8(frozen_vec, gmax_vec, new_gmax);                     \
        te_vec_lo = vbslq_s16(cmp_lo_16, i_vec, te_vec_lo);                      \
        te_vec_hi = vbslq_s16(cmp_hi_16, i_vec, te_vec_hi);                      \
        if (!(EPI_INLINE_QE)) {                                                  \
            if (vmaxvq_u8(cmp0_active)) {                                        \
                uint8x16_t iqe_vec = vdupq_n_u8(0xFF);                           \
                uint8x16_t foundBlk = zero_vec;                                  \
                const uint8_t *colIdx = this->colIdx8;                           \
                const int nblocks = (ncol + QE_BLK - 1) / QE_BLK;                \
                for (int b = 0; b < nblocks; b++) {                             \
                    uint8x16_t reached = vceqq_u8(vld1q_u8(__bmax + b * SIMD_WIDTH8), __imax); \
                    uint8x16_t newly = vandq_u8(vbicq_u8(reached, foundBlk), cmp0_active); \
                    foundBlk = vorrq_u8(foundBlk, reached);                      \
                    if (vmaxvq_u8(newly)) {                                      \
                        const int j0 = b * QE_BLK;                               \
                        const int j1 = (j0 + QE_BLK < ncol) ? (j0 + QE_BLK) : ncol; \
                        uint8x16_t got = zero_vec;                               \
                        for (int j2 = j0; j2 < j1; j2++) {                       \
                            uint8x16_t eq = vandq_u8(                            \
                                vceqq_u8(vld1q_u8(__hbase + j2 * SIMD_WIDTH8), __imax), newly); \
                            uint8x16_t first = vbicq_u8(eq, got);                \
                            iqe_vec = vbslq_u8(first, vld1q_u8(colIdx + j2 * SIMD_WIDTH8), iqe_vec); \
                            got = vorrq_u8(got, eq);                             \
                        }                                                       \
                    }                                                           \
                    if (!vmaxvq_u8(vbicq_u8(cmp0_active, foundBlk))) break;      \
                }                                                               \
                qe_vec = vbslq_u8(cmp0_active, iqe_vec, qe_vec);                 \
            }                                                                   \
        } else {                                                                \
            qe_vec = vbslq_u8(cmp0_active, (EPI_COL), qe_vec);                   \
        }                                                                       \
        /* Check end score threshold */                                        \
        uint8x16_t cmp_end = vcgeq_u8(gmax_vec, endsc_vec);                      \
        uint16_t cmp_end_msk = neon_movemask_u8(cmp_end);                        \
        cmp_end_msk &= endsc_msk_a;                                              \
        uint8x16_t just_hit = vandq_u8(cmp_end, has_endsc_vec);                  \
        frozen_vec = vorrq_u8(frozen_vec, just_hit);                             \
        uint8x16_t left_vec = vqaddq_u8(gmax_vec, sft_vec);                      \
        uint8x16_t cmp2 = vcgeq_u8(left_vec, cmax_vec);                          \
        uint16_t cmp2_msk = neon_movemask_u8(cmp2);                              \
        exit0 = (~(cmp_end_msk | cmp2_msk)) & exit0;                             \
    }

    /* Two-target-row cell: rows i and i+1 at column j in one body. Row i's
     * diagonal is H0[j] (= H(i-1,j-1)); row i+1's diagonal is row i's H at the
     * PREVIOUS column, carried in d1, and its vertical carry is row i's f21_0,
     * handed over in a register. Only row i+1's H and F reach memory (row i's are
     * consumed entirely by row i+1), so the pair costs five memory ops for two
     * cells. Arithmetic is cell-for-cell identical to KSWV_NEON_U8_CELL. The
     * query-end column is recovered one of two ways (LazyQE, see
     * rescue_lazyqe_enabled): lazily, by writing row i's H back into the dead
     * diagonal slot H0[j] and checkpointing both rows' running maxima per
     * QE_BLK for a post-row rescan; or inline, tracking the argmax column per
     * row (j_v, strict-greater blend). BND0/BND1 are the per-row boundary masks;
     * APPLY_BND/NEED_DUMMY fold at compile time like the one-row macro.
     * Persistent per-sweep regs: imax0/1, e11_0/1, d1 (+ col0/1, j_v inline). */
#define KSWV_NEON_U8_CELL_PAIR(APPLY_BND, BND0, BND1, NEED_DUMMY)                \
        {                                                                       \
            uint8x16_t s2 = vld1q_u8(seq2SoA + j * SIMD_WIDTH8);                 \
            uint8x16_t f11 = vld1q_u8(F + (j + 1) * SIMD_WIDTH8);                \
            uint8x16_t h00 = vld1q_u8(H0 + j * SIMD_WIDTH8);                     \
            uint8x16_t cmpq;                                                     \
            if (NEED_DUMMY) cmpq = vceqq_u8(s2, five_vec);                       \
            /* ---- Row i (diagonal H0[j]) ---- */                              \
            uint8x16_t sbt0 = vqtbl1q_u8(permSft, veorq_u8(s1_0, s2));           \
            if (NEED_DUMMY)                                                      \
                sbt0 = vbslq_u8(cmpq, USQADD ? zero_vec : sft_vec, sbt0);        \
            if (HasFreed)                                                        \
                sbt0 = vbslq_u8(vceqq_u8(s2, active_frread_0), freedval_vec, sbt0); \
            uint8x16_t m11_0;                                                    \
            if (USQADD) {                                                        \
                m11_0 = NEON_SQADD_U8(h00, sbt0);                                \
                if (APPLY_BND) m11_0 = vbslq_u8((BND0), zero_vec, m11_0);        \
            } else {                                                            \
                m11_0 = vqaddq_u8(h00, sbt0);                                    \
                if (APPLY_BND) m11_0 = vbslq_u8((BND0), zero_vec, m11_0);        \
                m11_0 = vqsubq_u8(m11_0, sft_vec);                              \
            }                                                                   \
            uint8x16_t hme0 = vmaxq_u8(m11_0, e11_0);                            \
            uint8x16_t h11_0 = vmaxq_u8(hme0, f11);                              \
            /* LazyQE: row i's H for column j lands in the diagonal slot just     \
             * consumed above (H0[j]); the rescan reads it back at index j. */  \
            if (LazyQE) vst1q_u8(H0 + j * SIMD_WIDTH8, h11_0);                   \
            else col0 = vbslq_u8(vcgtq_u8(h11_0, imax0), j_v, col0);             \
            imax0 = vmaxq_u8(imax0, h11_0);                                      \
            uint8x16_t mf0 = vmaxq_u8(m11_0, f11);                               \
            e11_0 = vmaxq_u8(vqsubq_u8(mf0, oe_ins_vec),                         \
                             vqsubq_u8(e11_0, e_ins_vec));                       \
            uint8x16_t f21_0 = vmaxq_u8(vqsubq_u8(hme0, oe_del_vec),             \
                                        vqsubq_u8(f11, e_del_vec));              \
            /* ---- Row i+1 (diagonal d1 = row i's prev-col H; vcarry f21_0) --- */ \
            uint8x16_t sbt1 = vqtbl1q_u8(permSft, veorq_u8(s1_1, s2));           \
            if (NEED_DUMMY)                                                      \
                sbt1 = vbslq_u8(cmpq, USQADD ? zero_vec : sft_vec, sbt1);        \
            if (HasFreed)                                                        \
                sbt1 = vbslq_u8(vceqq_u8(s2, active_frread_1), freedval_vec, sbt1); \
            uint8x16_t m11_1;                                                    \
            if (USQADD) {                                                        \
                m11_1 = NEON_SQADD_U8(d1, sbt1);                                 \
                if (APPLY_BND) m11_1 = vbslq_u8((BND1), zero_vec, m11_1);        \
            } else {                                                            \
                m11_1 = vqaddq_u8(d1, sbt1);                                     \
                if (APPLY_BND) m11_1 = vbslq_u8((BND1), zero_vec, m11_1);        \
                m11_1 = vqsubq_u8(m11_1, sft_vec);                              \
            }                                                                   \
            uint8x16_t hme1 = vmaxq_u8(m11_1, e11_1);                            \
            uint8x16_t h11_1 = vmaxq_u8(hme1, f21_0);                            \
            if (!LazyQE) col1 = vbslq_u8(vcgtq_u8(h11_1, imax1), j_v, col1);     \
            imax1 = vmaxq_u8(imax1, h11_1);                                      \
            uint8x16_t mf1 = vmaxq_u8(m11_1, f21_0);                             \
            e11_1 = vmaxq_u8(vqsubq_u8(mf1, oe_ins_vec),                         \
                             vqsubq_u8(e11_1, e_ins_vec));                       \
            uint8x16_t f21_1 = vmaxq_u8(vqsubq_u8(hme1, oe_del_vec),             \
                                        vqsubq_u8(f21_0, e_del_vec));            \
            vst1q_u8(H1 + (j + 1) * SIMD_WIDTH8, h11_1);                         \
            vst1q_u8(F + (j + 1) * SIMD_WIDTH8, f21_1);                          \
            d1 = h11_0;                                                          \
            if (LazyQE) {                                                        \
                /* Row-max checkpoints for both rows, once per QE_BLK columns,  \
                 * same cadence as the one-row body. */                          \
                if (j == qeNext) {                                               \
                    vst1q_u8(blockMax  + qeBlk * SIMD_WIDTH8, imax0);            \
                    vst1q_u8(blockMax1 + qeBlk * SIMD_WIDTH8, imax1);            \
                    qeBlk++; qeNext += QE_BLK;                                   \
                }                                                                \
            } else {                                                             \
                j_v = vaddq_u8(j_v, one_vec);                                    \
            }                                                                    \
        }

    int i = 0, limit = nrow;
    while (i < nrow)
    {
        /* ---- Two-target-row fast path ---------------------------------------
         * Taken when a full pair remains AND both rows lie on the same side of
         * minLen1, so one uniform column-range body serves the pair. The single
         * pair that straddles minLen1 (i < minLen1 <= i+1) fails the guard and
         * drops to the one-row body below, as does the final odd row and every
         * row when RowPair is compiled out. */
        if (RowPair && i + 1 < nrow && ((i + 1 < minLen1) || (i >= minLen1))) {
            uint8x16_t s1_0 = vld1q_u8(seq1SoA + (i + 0) * SIMD_WIDTH8);
            uint8x16_t s1_1 = vld1q_u8(seq1SoA + (i + 1) * SIMD_WIDTH8);
            uint8x16_t imax0 = zero_vec, imax1 = zero_vec;
            /* Inline-argmax state (only touched when !LazyQE). */
            uint8x16_t col0 = zero_vec, col1 = zero_vec;
            uint8x16_t e11_0 = zero_vec, e11_1 = zero_vec;
            uint8x16_t d1 = zero_vec, j_v = zero_vec;
            /* Row-max checkpoint cursor for the LazyQE rescan (both rows). */
            int qeNext = QE_BLK - 1, qeBlk = 0;
            (void) qeNext; (void) qeBlk;

            /* Freed-cell override built once per row for BOTH rows of the pair
             * (s1 differs between them); see the one-row body below for the
             * fold into a single per-lane active_frread. */
            uint8x16_t freedval_vec, active_frread_0, active_frread_1;
            if (HasFreed) {
                freedval_vec = vdupq_n_u8((uint8_t)(fr_val + (USQADD ? 0 : (int)shift)));
                const uint8x16_t frv  = vdupq_n_u8((uint8_t)fr_ref);
                const uint8x16_t frv2 = vdupq_n_u8((uint8_t)fr_ref2);
                const uint8x16_t frd  = vdupq_n_u8((uint8_t)fr_read);
                const uint8x16_t frd2 = vdupq_n_u8((uint8_t)fr_read2);
                const uint8x16_t fin  = vdupq_n_u8(FREED_INACTIVE8);
                active_frread_0 = vbslq_u8(vceqq_u8(s1_0, frv2), frd2, fin);
                active_frread_0 = vbslq_u8(vceqq_u8(s1_0, frv),  frd, active_frread_0);
                active_frread_1 = vbslq_u8(vceqq_u8(s1_1, frv2), frd2, fin);
                active_frread_1 = vbslq_u8(vceqq_u8(s1_1, frv),  frd, active_frread_1);
            }

            int j = 0;
            if (i + 1 < minLen1) {
                for (; j < jdummy; j++)
                    KSWV_NEON_U8_CELL_PAIR(false, zero_vec, zero_vec, false)
                for (; j < jsplit; j++)
                    KSWV_NEON_U8_CELL_PAIR(false, zero_vec, zero_vec, true)
            } else {
                const uint8x16_t rb0 = vtstq_u8(s1_0, highbit_vec);
                const uint8x16_t rb1 = vtstq_u8(s1_1, highbit_vec);
                for (; j < jdummy; j++)
                    KSWV_NEON_U8_CELL_PAIR(true, rb0, rb1, false)
                for (; j < jsplit; j++)
                    KSWV_NEON_U8_CELL_PAIR(true, rb0, rb1, true)
            }
            for (; j < ncol; j++)
                KSWV_NEON_U8_CELL_PAIR(true,
                                       vtstq_u8(vorrq_u8(s1_0, s2), highbit_vec),
                                       vtstq_u8(vorrq_u8(s1_1, s2), highbit_vec), true)

            /* Row epilogues in row order: a freeze at row i must suppress row
             * i+1. LazyQE rescans row i from the in-place H0 (column j at index
             * j) and row i+1 from H1 (column j at index j+1); otherwise both use
             * the inline argmax column carried by the sweep. */
            if (LazyQE) {
                KSWV_U8_EPILOGUE(i, imax0, false, zero_vec, blockMax, H0)
                if (exit0 == 0) { limit = i; i += 1; break; }
                KSWV_U8_EPILOGUE(i + 1, imax1, false, zero_vec, blockMax1, H1 + SIMD_WIDTH8)
                if (exit0 == 0) { limit = i + 1; i += 2; break; }
                /* Restore the column -1 boundary the in-place store clobbered
                 * (H0[0] held row i's column 0); after the swap this buffer is
                 * H1, then H0 again two rows on, when its [0] must read 0. */
                vst1q_u8(H0, zero_vec);
            } else {
                KSWV_U8_EPILOGUE(i, imax0, true, col0, blockMax, H1 + SIMD_WIDTH8)
                if (exit0 == 0) { limit = i; i += 1; break; }
                KSWV_U8_EPILOGUE(i + 1, imax1, true, col1, blockMax, H1 + SIMD_WIDTH8)
                if (exit0 == 0) { limit = i + 1; i += 2; break; }
            }
            (void) col0; (void) col1; (void) j_v;

            uint8_t *S = H1; H1 = H0; H0 = S;
            i += 2;
            continue;
        }

        uint8x16_t e11 = zero_vec;
        uint8x16_t h00, h11, h10, s1;
        int j;

        s1 = vld1q_u8(seq1SoA + (i + 0) * SIMD_WIDTH8);
        h10 = zero_vec;
        imax_vec = zero_vec;

        /* Freed-cell override (issue 173, bisulfite OT/OB + TAPS neutral). The
         * freed cell (s1==fr_ref, s2==fr_read) scores fr_val. s1 is
         * loop-invariant across the inner j loop, so hoist the per-row
         * comparison here; per-cell only the s2 compare + blend remains.
         * freedval_vec is fr_val biased by +shift into the u8 domain (same bias
         * temp[] carries): fr_val==w_match (GENOMIC/COLLAPSED) equals temp[0]
         * exactly, NEUTRAL (fr_val==0) scores the conversion as 0 after unbiasing.
         * When !HasFreed, all of this and the per-cell block below compile out
         * (identical codegen to today). */
        /* Free a SYMMETRIC PAIR of cells: (fr_ref,fr_read) and (fr_ref2,fr_read2).
         * GENOMIC/NEUTRAL set the mirror == primary; COLLAPSED sets it to the
         * transpose so C/T (or G/A) are interchangeable.
         *
         * Fold the pair into ONE per-lane target base, active_frread: a lane whose
         * ref is fr_ref frees against fr_read, a lane whose ref is fr_ref2 frees
         * against fr_read2, and any other lane gets FREED_INACTIVE8. The two
         * ref-side comparisons are mutually exclusive (s1 can't equal both fr_ref
         * and fr_ref2), so at most one wins per lane. s2 holds real bases 0-3, the
         * ambig/pad codes DUMMY5(5)/AMBQ(8), or the 0xFF query pad — never
         * FREED_INACTIVE8 — so on a sentinel lane the single per-cell compare is
         * unconditionally false. This
         * turns the per-cell freed path into one vceqq + one vbslq (down from two
         * compares, two ands, an orr, and a blend). Both cells of the pair score
         * the same fr_val, so one freedval_vec covers the fold. */
        uint8x16_t freedval_vec, active_frread;
        if (HasFreed) {
            /* USQADD: signed delta fr_val (no +shift bias). */
            freedval_vec = vdupq_n_u8((uint8_t)(fr_val + (USQADD ? 0 : (int)shift)));
            uint8x16_t rowfreed  = vceqq_u8(s1, vdupq_n_u8((uint8_t)fr_ref));
            uint8x16_t rowfreed2 = vceqq_u8(s1, vdupq_n_u8((uint8_t)fr_ref2));
            active_frread = vbslq_u8(rowfreed2, vdupq_n_u8((uint8_t)fr_read2),
                                     vdupq_n_u8(FREED_INACTIVE8));
            active_frread = vbslq_u8(rowfreed,  vdupq_n_u8((uint8_t)fr_read),
                                     active_frread);
        }

        /* One DP cell. BND is the boundary mask -- bit 7 set in the reference
         * base (rows past len1) or the query base (columns past this lane's
         * quantum) -- and APPLY_BND gates whether it is applied at all. Both
         * are compile-time constants at each call site, so `if (APPLY_BND)`
         * folds exactly like the `if (HasFreed)` below it and no per-cell
         * branch survives. Passing them in lets each column range pay only for
         * the half that can actually fire: nothing below both minLen1 and
         * jsplit, the hoisted per-row reference mask below jsplit, and the full
         * per-cell form from jsplit on. */
#define KSWV_NEON_U8_CELL(APPLY_BND, BND, NEED_DUMMY)                           \
        {                                                                       \
            uint8x16_t f11, s2, f21;                                            \
            h00 = vld1q_u8(H0 + j * SIMD_WIDTH8);                               \
            s2 = vld1q_u8(seq2SoA + j * SIMD_WIDTH8);                           \
            f11 = vld1q_u8(F + (j + 1) * SIMD_WIDTH8);                          \
                                                                                \
            /* Core Smith-Waterman computation using NEON */                    \
            uint8x16_t xor_val = veorq_u8(s1, s2);                              \
            uint8x16_t sbt = vqtbl1q_u8(permSft, xor_val);                      \
                                                                                \
            /* Check for ambiguous query base (DUMMY5). Elided below jdummy,   \
             * where no lane can hold one -- see the derivation above. */       \
            if (NEED_DUMMY) {                                                   \
                uint8x16_t cmpq = vceqq_u8(s2, five_vec);                       \
                sbt = vbslq_u8(cmpq, USQADD ? zero_vec : sft_vec, sbt);         \
            }                                                                   \
                                                                                \
            /* Apply the freed-cell override: force the biased freed score      \
             * (fr_val + shift) where this column's read base equals the row's  \
             * active freed base (built once per row in active_frread above).   \
             * One compare + one blend; FREED_INACTIVE8 lanes never match any  \
             * s2 value (0-3/5/8 or the 0xFF pad), so they blend through        \
             * unchanged. */                                                    \
            if (HasFreed) {                                                     \
                sbt = vbslq_u8(vceqq_u8(s2, active_frread), freedval_vec, sbt); \
            }                                                                   \
                                                                                \
            /* Diagonal: USQADD applies the signed delta and clamps in one op,   \
             * dropping the de-bias vqsubq_u8 from the h00->m11->h11 chain. The   \
             * BND blend to zero still runs after (boundary cells -> 0 in both).  \
             * The biased form: add, blend, then de-bias. */                      \
            uint8x16_t m11;                                                       \
            if (USQADD) {                                                         \
                m11 = NEON_SQADD_U8(h00, sbt);                                    \
                if (APPLY_BND) m11 = vbslq_u8((BND), zero_vec, m11);             \
            } else {                                                             \
                m11 = vqaddq_u8(h00, sbt);                                       \
                if (APPLY_BND) m11 = vbslq_u8((BND), zero_vec, m11);             \
                m11 = vqsubq_u8(m11, sft_vec);                                   \
            }                                                                    \
                                                                                \
            /* hme = max(m11, e11); reused verbatim for `me` (gap-D) below,     \
             * where the original code recomputed the identical max(m11,e11) -- \
             * e11 is not modified between the two, so this is byte-identical   \
             * and drops one vmaxq per cell. */                                 \
            uint8x16_t hme = vmaxq_u8(m11, e11);                                \
            h11 = vmaxq_u8(hme, f11);                                           \
                                                                                \
            /* Row max only. The COLUMN at which it sits (the query end) is    \
             * recovered after the row, and only on the rows that actually     \
             * advance a lane's global max -- see the rescan below. */         \
            imax_vec = vmaxq_u8(imax_vec, h11);                                 \
                                                                                \
            /* Reassociate BOTH gap recurrences off h11 (Daily-scan algebra).   \
             * e uses max(m11,f11), f uses max(m11,e11) — each drops the term   \
             * dominated by its own extension arg (O>=0, saturating u8). me must\
             * read the OLD e11, so compute it before e11 is overwritten. */    \
            uint8x16_t mf = vmaxq_u8(m11, f11);                                 \
            uint8x16_t me = hme;  /* == vmaxq_u8(m11, e11), e11 unchanged */    \
            uint8x16_t gapE = vqsubq_u8(mf, oe_ins_vec);                        \
            e11 = vqsubq_u8(e11, e_ins_vec);                                    \
            e11 = vmaxq_u8(gapE, e11);                                          \
                                                                                \
            /* Gap extension for F (reassociated) */                            \
            uint8x16_t gapD = vqsubq_u8(me, oe_del_vec);                        \
            f21 = vqsubq_u8(f11, e_del_vec);                                    \
            f21 = vmaxq_u8(gapD, f21);                                          \
                                                                                \
            vst1q_u8(H1 + (j + 1) * SIMD_WIDTH8, h11);                          \
            vst1q_u8(F + (j + 1) * SIMD_WIDTH8, f21);                           \
            /* Checkpoint the running row max once per QE_BLK columns, so the   \
             * post-row query-end recovery can find the block holding the max   \
             * without rescanning the whole row. One vector store per QE_BLK    \
             * cells, on a branch taken 1-in-QE_BLK with a fixed stride -- the  \
             * predictor sees a regular pattern, and the four unswitched column \
             * loops stay intact. */                                            \
            if (j == qeNext) {                                                  \
                vst1q_u8(blockMax + qeBlk * SIMD_WIDTH8, imax_vec);             \
                qeBlk++; qeNext += QE_BLK;                                      \
            }                                                                   \
        }

        int qeNext = QE_BLK - 1, qeBlk = 0;
        j = 0;
        if (i < minLen1) {
            /* No lane has begun reference padding and no column below jsplit
             * carries query padding: the mask is provably all-zero here. */
            for (; j < jdummy; j++)
                KSWV_NEON_U8_CELL(false, zero_vec, false)
            for (; j < jsplit; j++)
                KSWV_NEON_U8_CELL(false, zero_vec, true)
        } else {
            /* Reference half only; loop-invariant across j. */
            const uint8x16_t rowboundary = vtstq_u8(s1, highbit_vec);
            for (; j < jdummy; j++)
                KSWV_NEON_U8_CELL(true, rowboundary, false)
            for (; j < jsplit; j++)
                KSWV_NEON_U8_CELL(true, rowboundary, true)
        }
        for (; j < ncol; j++)
            KSWV_NEON_U8_CELL(true, vtstq_u8(vorrq_u8(s1, s2), highbit_vec), true)
#undef KSWV_NEON_U8_CELL

        /* One-row epilogue: recover qe lazily from the stored H (EPI_INLINE_QE
         * = false). Byte-identical to the pre-refactor inline body. */
        KSWV_U8_EPILOGUE(i, imax_vec, false, zero_vec, blockMax, H1 + SIMD_WIDTH8)
        if (exit0 == 0)
        {
            limit = i;
            i += 1;
            break;
        }

        uint8_t *S = H1; H1 = H0; H0 = S;
        i += 1;
    }
#undef KSWV_NEON_U8_CELL_PAIR
#undef KSWV_U8_EPILOGUE

    /* Store final row max. Guard on i > 0: when every pair in the batch
     * has len1 == 0, nrow == 0, the DP loop never runs, and `i - 1`
     * underflows into the allocation prefix. Upstream PR 289 / issue 38. */
    if (i > 0) {
        vst1q_u8(rowMax + (i - 1) * SIMD_WIDTH8, pimax_vec);
    }

    /* Extract results */
    uint8_t score[SIMD_WIDTH8] __attribute((aligned(128)));
    int16_t te1[SIMD_WIDTH8] __attribute((aligned(128)));
    uint8_t qe[SIMD_WIDTH8] __attribute((aligned(128)));

    vst1q_u8(score, gmax_vec);
    vst1q_s16(te1, te_vec_lo);       // pairs 0-7
    vst1q_s16(te1 + 8, te_vec_hi);   // pairs 8-15
    vst1q_u8(qe, qe_vec);

    int live = 0;
    for (int l = 0; l < SIMD_WIDTH8 && (po_ind + l) < numPairs; l++) {
        int ind = po_ind + l;
        int16_t *te = te1;
#if !MAINY
        ind = p[l].regid;
        if (phase) {
            if (aln[ind].score == score[l]) {
                aln[ind].tb = aln[ind].te - te[l];
                aln[ind].qb = aln[ind].qe - qe[l];
            }
        } else {
            aln[ind].score = score[l] + shift < 255 ? score[l] : 255;
            aln[ind].te = te[l];
            aln[ind].qe = qe[l];
            if (aln[ind].score != 255) {
                qe[l] = 1;
                live++;
            }
            else qe[l] = 0;
        }
#else
        aln[ind].score = score[l] + shift < 255 ? score[l] : 255;
        aln[ind].te = te[l];
        aln[ind].qe = qe[l];
        if (aln[ind].score != 255) {
            qe[l] = 1;
            live++;
        }
        else qe[l] = 0;
#endif
    }

#if !MAINY
    if (phase) return 1;
#endif

    if (live == 0) return 1;

    /* Score2 and te2 computation.
     *
     * Per-lane scalar emulation of ksw_u8's b[] build + score2 scan
     * (src/ksw.cpp:201). scalar ksw_u8 collapses consecutive rows >=
     * minsc into a single b[] entry anchored at the max-score row; a
     * plateau straddling the primary region [te - val, te + val] is
     * excluded entirely when its anchor lies inside. A dense SIMD scan
     * treats every row independently and pulls out-of-region rows of a
     * boundary-straddling plateau in as false suboptimals, inflating `sub`.
     * Emulate scalar exactly, per lane, on the precomputed
     * rowMax. Cost is O(SIMD_WIDTH8 * limit) scalar ops per batch,
     * negligible next to the SIMD DP that built rowMax. */
    int qmax = this->g_qmax;
    int16_t low[SIMD_WIDTH8]  __attribute__((aligned(64)));
    int16_t high[SIMD_WIDTH8] __attribute__((aligned(64)));
    for (int j = 0; j < SIMD_WIDTH8; j++) {
        int val = (score[j] + qmax - 1) / qmax;
        low[j]  = te1[j] - val;
        high[j] = te1[j] + val;
    }

    /* rowMax entries produced: i == limit on normal completion, and
     * i == limit + 1 after early exit (limit = i++ stores old i). Using i
     * directly gives a single expression that covers both exit paths and
     * includes the terminating row stored at rowMax[limit]. */
    const int processed_rows = i;

    for (int l = 0; l < SIMD_WIDTH8 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        /* Match scalar ksw_u8: when KSW_XSUBO is absent, minsc = 0x10000
         * so the b[] list never starts and score2/te2 stay at -1.
         * minsc_msk_a tracks lanes with KSW_XSUBO set; lanes not in the
         * mask have minsc[l] zero-initialized and must skip the scan. */
        if (!qe[l] || !(minsc_msk_a & (1u << l))) {
            aln[ind].score2 = -1;
            aln[ind].te2    = -1;
            continue;
        }

        int len1_l  = (int)p[l].len1;
        int low_l   = (int)low[l];
        int high_l  = (int)high[l];
        int minsc_l = (int)minsc[l];
        int score2  = -1;
        int te2     = -1;
        /* b_pos = -2 sentinel: (-2 + 1) != any real row, forces an
         * "append" on the first qualifying row. */
        int b_score = -1;
        int b_pos   = -2;

        int nrows = processed_rows < len1_l ? processed_rows : len1_l;
        for (int i2 = 0; i2 < nrows; i2++) {
            int imax = (int)rowMax[i2 * SIMD_WIDTH8 + l];
            if (imax < minsc_l) continue;

            if (b_pos + 1 != i2) {
                /* APPEND: flush the outgoing b[] entry to score2 first. */
                if (b_pos >= 0 &&
                    (b_pos < low_l || b_pos > high_l) &&
                    b_score > score2) {
                    score2 = b_score;
                    te2    = b_pos;
                }
                b_score = imax;
                b_pos   = i2;
            } else if (b_score < imax) {
                /* UPDATE: strict greater extends the run's anchor. Equal
                 * imax leaves b_pos in place (matches scalar). */
                b_score = imax;
                b_pos   = i2;
            }
        }

        /* Flush trailing b[] entry. */
        if (b_pos >= 0 &&
            (b_pos < low_l || b_pos > high_l) &&
            b_score > score2) {
            score2 = b_score;
            te2    = b_pos;
        }

        aln[ind].score2 = score2;
        aln[ind].te2    = te2;
    }

    return 1;
}

/* 16-bit NEON implementation */
void kswv::getScores16(SeqPair *pairArray,
                       uint8_t *seqBufRef,
                       uint8_t *seqBufQer,
                       kswr_t* aln,
                       int32_t numPairs,
                       uint16_t numThreads,
                       int phase)
{
    kswvBatchWrapper16(pairArray, seqBufRef, seqBufQer, aln,
                       numPairs, numThreads, phase);
}

void kswv::kswvBatchWrapper16(SeqPair *pairArray,
                              uint8_t *seqBufRef,
                              uint8_t *seqBufQer,
                              kswr_t* aln,
                              int32_t numPairs,
                              uint16_t numThreads,
                              int phase)
{
    int16_t *seq1SoA = NULL;
    seq1SoA = (int16_t *)_mm_malloc(this->maxRefLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 128);

    int16_t *seq2SoA = NULL;
    seq2SoA = (int16_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 128);

    assert(seq1SoA != NULL);
    assert(seq2SoA != NULL);

    int32_t ii;
    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH16 - 1) / SIMD_WIDTH16) * SIMD_WIDTH16;

    for (ii = numPairs; ii < roundNumPairs; ii++)
    {
        pairArray[ii].regid = ii;
        pairArray[ii].id = ii;
        pairArray[ii].len1 = 0;
        pairArray[ii].len2 = 0;
        // The kernel's per-lane loop reads idr/idq/h0 for every lane up to
        // SIMD_WIDTH, so it reads these padded lanes too; zero them (as id/len1/
        // len2 above) to keep those reads well-defined. Padded lanes join the SIMD
        // batch (and its cross-lane reductions), but the caller reads results back
        // only for real lanes and whole-aligner output is byte-identical (validated).
        pairArray[ii].idr = 0;
        pairArray[ii].idq = 0;
        pairArray[ii].h0 = 0;
    }

    {
        int32_t i;
        uint16_t tid = 0;
        int16_t *mySeq1SoA = seq1SoA + tid * this->maxRefLen * SIMD_WIDTH16;
        int16_t *mySeq2SoA = seq2SoA + tid * this->maxQerLen * SIMD_WIDTH16;
        uint8_t *seq1;
        uint8_t *seq2;

        int nstart = 0, nend = numPairs;

        for (i = nstart; i < nend; i += SIMD_WIDTH16)
        {
            int32_t j, k;
            int maxLen1 = 0;
            int maxLen2 = 0;

            for (j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq1 = seqBufRef + (int64_t)sp.id * this->maxRefLen;
#else
                seq1 = seqBufRef + sp.idr;
#endif
                for (k = 0; k < sp.len1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH16 + j] = (seq1[k] == AMBIG_ ? AMBR16 : seq1[k]);
                }
                if (maxLen1 < sp.len1) maxLen1 = sp.len1;
            }
            for (j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
                for (k = sp.len1; k <= maxLen1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH16 + j] = 0xFFFF;
                }
            }

            for (j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq2 = seqBufQer + (int64_t)sp.id * this->maxQerLen;
#else
                seq2 = seqBufQer + sp.idq;
#endif
                int quanta = query_quantum16(sp.len2);
                for (k = 0; k < sp.len2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH16 + j] = (seq2[k] == AMBIG_ ? AMBQ16 : seq2[k]);
                }

                for (k = sp.len2; k < quanta; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH16 + j] = DUMMY3;
                }
                if (maxLen2 < quanta) maxLen2 = quanta;
            }

            for (j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
                int quanta = query_quantum16(sp.len2);
                for (k = quanta; k <= maxLen2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH16 + j] = 0xFFFF;
                }
            }

            kswv_neon_16(mySeq1SoA, mySeq2SoA,
                         maxLen1, maxLen2,
                         pairArray + i,
                         aln, i,
                         tid,
                         numPairs,
                         phase);
        }
    }

    _mm_free(seq1SoA);
    _mm_free(seq2SoA);
    return;
}

/* Thin dispatcher: see kswv_neon_u8. */
int kswv::kswv_neon_16(int16_t seq1SoA[],
                       int16_t seq2SoA[],
                       int16_t nrow,
                       int16_t ncol,
                       SeqPair *p,
                       kswr_t *aln,
                       int po_ind,
                       uint16_t tid,
                       int32_t numPairs,
                       int phase)
{
    return has_freed
        ? kswv_neon_16_impl<true>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                  po_ind, tid, numPairs, phase)
        : kswv_neon_16_impl<false>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                   po_ind, tid, numPairs, phase);
}

template<bool HasFreed>
int kswv::kswv_neon_16_impl(int16_t seq1SoA[],
                            int16_t seq2SoA[],
                            int16_t nrow,
                            int16_t ncol,
                            SeqPair *p,
                            kswr_t *aln,
                            int po_ind,
                            uint16_t tid,
                            int32_t numPairs,
                            int phase)
{
    /* NEON 16-bit mate-rescue kernel.
     *
     * Structural port of kswv_neon_u8: real rowMax writes each row,
     * per-lane freeze once a lane hits its KSW_XSTOP target (gated by
     * has_endsc_vec so lanes without a target aren't frozen prematurely),
     * per-row frozen_bits collapsed from frozen_vec for the batched early
     * exit, and a scalar b[]-emulation score2 scan over rowMax matching
     * scalar ksw_i16 exactly. The
     * pre-rewrite kernel never wrote rowMax, never computed score2
     * (hardcoded to -1), and global-broke the DP as soon as any lane's
     * gmax >= endsc_vec (trivially true for mate-rescue pairs whose
     * endsc_vec defaults to 0 since KSW_XSTOP is unset) — which made
     * `score`/`te`/`qe` almost garbage too. Replaced with proper SIMD
     * semantics matching the 8-bit NEON kernel.
     *
     * 16-bit specifics vs 8-bit:
     *   - 8 lanes (SIMD_WIDTH16), single te_vec (no _lo/_hi split)
     *   - int16 arithmetic; no shift offset, clamp intermediate H to 0
     *     via vmaxq_s16 (scalar ksw_i16 does the same)
     *   - No u8 overflow check; int16 has plenty of headroom for SW scores
     */
    int16_t minsc[SIMD_WIDTH16] __attribute((aligned(128))) = {0};
    int16_t endsc[SIMD_WIDTH16] __attribute((aligned(128))) = {0};

    int16x8_t zero_vec = vdupq_n_s16(0);
    int16x8_t one_vec  = vdupq_n_s16(1);

    /* Scoring table: 32-byte int8 table indexed by (s1 ^ s2) in [0..31].
     *
     * IMPORTANT: The table must be byte-wide (not int16-wide) AND must
     * cover all 32 possible xor values. The pre-rewrite kernel used an
     * 8-entry int16 table (16 bytes) and did `vqtbl1q_s8(perm, xor_val)`
     * + reinterpret to int16 — treating each int16 xor value's low+high
     * bytes as two byte indices, producing sbt[i] = (w_match << 8) |
     * perm[xor_low], inflating match scores by 256. That bug was latent
     * while a 1M-read smoke slice never exercised the 16-bit path
     * (l_ms*opt->a < 250 routes to 8-bit), but `-A 2` makes l_ms*2 = 300
     * >= 250 and surfaces it.
     *
     * The 16-bit kernel's SoA uses wider ambig / tail-padding constants
     * than 8-bit (AMBR16=15, AMBQ16=16, DUMMY3=26 vs AMBR=4, AMBQ=8,
     * DUMMY5=5), so xor values can reach 31. A 32-byte table built via
     * vqtbl2_s8 handles all of them:
     *   [0]     -> w_match       (base == base; also AMBR^AMBR, AMBQ^AMBQ,
     *                             but those pairings don't occur)
     *   [1..3]  -> w_mismatch    (base != base)
     *   [21,24..27] -> 0         (anything ^ DUMMY3 = query tail padding,
     *                             must contribute 0 like scalar would)
     *   other   -> w_ambig       (any xor involving AMBR16 or AMBQ16)
     * Boundary (high bit of s1|s2) is handled separately by zeroing m11. */
    int8_t temp8[32] __attribute((aligned(16)));
    for (int i = 0; i < 32; i++) temp8[i] = this->w_ambig;
    temp8[0] = this->w_match;
    temp8[1] = temp8[2] = temp8[3] = this->w_mismatch;
    /* Query-tail DUMMY3 (=26): s1 ∈ {0..3, 15} xor 26 = {21, 24..27}.
     * Those indices must contribute 0 to match scalar's tail semantics. */
    temp8[21] = 0;
    temp8[24] = temp8[25] = temp8[26] = temp8[27] = 0;
    int8x16x2_t perm_vec;
    perm_vec.val[0] = vld1q_s8(temp8);
    perm_vec.val[1] = vld1q_s8(temp8 + 16);

    /* Per-lane minsc / endsc + msk bitmasks. Same convention as the 8-bit
     * kernel: minsc_msk_a bit set iff lane has a real KSW_XSUBO target that
     * fits in s16; same for endsc_msk_a / KSW_XSTOP. */
    uint8_t minsc_msk_a = 0x00, endsc_msk_a = 0x00;
    for (int i = 0; i < SIMD_WIDTH16; i++) {
        int xtra = p[i].h0;
        int val = (xtra & KSW_XSUBO) ? xtra & 0xffff : 0x10000;
        if (val <= SHRT_MAX) { minsc[i] = (int16_t)val; minsc_msk_a |= (0x1 << i); }
        val = (xtra & KSW_XSTOP) ? xtra & 0xffff : 0x10000;
        if (val <= SHRT_MAX) { endsc[i] = (int16_t)val; endsc_msk_a |= (0x1 << i); }
    }

    int16x8_t minsc_vec = vld1q_s16(minsc);
    int16x8_t endsc_vec = vld1q_s16(endsc);

    int16x8_t e_del_vec  = vdupq_n_s16(this->e_del);
    int16x8_t oe_del_vec = vdupq_n_s16(this->o_del + this->e_del);
    int16x8_t e_ins_vec  = vdupq_n_s16(this->e_ins);
    int16x8_t oe_ins_vec = vdupq_n_s16(this->o_ins + this->e_ins);

    int16x8_t gmax_vec = zero_vec;
    int16x8_t te_vec   = vdupq_n_s16(-1);
    int16x8_t qe_vec   = zero_vec;

    /* Per-lane freeze once a pair hits its KSW_XSTOP target. Gated by
     * has_endsc_vec so lanes with endsc[i] == 0 (no XSTOP) never freeze. */
    int16x8_t frozen_vec = zero_vec;
    int16_t _has_endsc_arr[SIMD_WIDTH16] __attribute((aligned(128)));
    for (int i = 0; i < SIMD_WIDTH16; i++)
        _has_endsc_arr[i] = (endsc_msk_a & (1 << i)) ? (int16_t)0xFFFF : (int16_t)0x0000;
    int16x8_t has_endsc_vec = vld1q_s16(_has_endsc_arr);

    tid = 0;
    int16_t *H0     = H16_0    + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *H1     = H16_1    + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *F      = F16      + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *rowMax = rowMax16 + tid * SIMD_WIDTH16 * this->maxRefLen;

    /* Per-strip warm-up prefetches. F and H1 are loop-resident (read+written
     * every row, fit in L1) and seq2SoA is swept every row, so they want a keep
     * hint (locality 1-2), not NTA. seq1SoA is read once per row — streaming, so
     * locality 0. Args: (addr, rw=0 read, locality 0 lowest .. 3 highest). */
    __builtin_prefetch(F + SIMD_WIDTH16, 0, 1);
    __builtin_prefetch(seq2SoA, 0, 2);
    __builtin_prefetch(seq1SoA, 0, 0);
    __builtin_prefetch(H1 + SIMD_WIDTH16, 0, 1);

    for (int i = 0; i <= ncol; i++) {
        vst1q_s16(H0   + i * SIMD_WIDTH16, zero_vec);
        vst1q_s16(F    + i * SIMD_WIDTH16, zero_vec);
    }
    vst1q_s16(H0, zero_vec);
    vst1q_s16(H1, zero_vec);

    int16x8_t imax_vec, pimax_vec = zero_vec;

    int i, limit = nrow;
    for (i = 0; i < nrow; i++) {
        int16x8_t e11 = zero_vec;
        int16x8_t s1  = vld1q_s16(seq1SoA + i * SIMD_WIDTH16);
        imax_vec = zero_vec;
        int16x8_t iqe_vec = vdupq_n_s16(-1);
        int16x8_t l_vec   = zero_vec;
        int16x8_t i_vec   = vdupq_n_s16((int16_t)i);

        /* Rank-1 freed-cell override (issue 173 + TAPS neutral), folded into ONE
         * per-row target base active_frread16 — the u16 analog of
         * kswv_neon_u8_impl's active_frread. A lane whose ref is fr_ref frees
         * against fr_read, a lane whose ref is fr_ref2 frees against fr_read2 (the
         * two ref-side compares are mutually exclusive), and every other lane gets
         * a sentinel no s2 ever equals. u16 s2 ∈ {0-3, AMBQ16=16, DUMMY3=26,
         * 0xFFFF pad}, so FREED_INACTIVE16 is safe here; 0xFFFF, the pad, would
         * NOT be — the u8 path picks FREED_INACTIVE8 over its own 0xFF pad on
         * the same reasoning. The 16-bit kernel has NO shift bias (m11 = h00 +
         * sbt directly), so the freed value is just fr_val: fr_val == w_match
         * (GENOMIC/COLLAPSED) equals temp8[0], NEUTRAL (fr_val==0) scores the
         * conversion as 0. Sign-extended to int16. When !HasFreed, this and the
         * per-cell block below compile out. */
        int16x8_t freedval_vec16, active_frread16;
        if (HasFreed) {
            freedval_vec16 = vdupq_n_s16((int16_t)fr_val);
            uint16x8_t rowfreed16   = vceqq_s16(s1, vdupq_n_s16((int16_t)fr_ref));
            uint16x8_t rowfreed2_16 = vceqq_s16(s1, vdupq_n_s16((int16_t)fr_ref2));
            active_frread16 = vbslq_s16(rowfreed2_16, vdupq_n_s16((int16_t)fr_read2),
                                        vdupq_n_s16((int16_t)FREED_INACTIVE16));
            active_frread16 = vbslq_s16(rowfreed16,  vdupq_n_s16((int16_t)fr_read),
                                        active_frread16);
        }

        for (int j = 0; j < ncol; j++) {
            int16x8_t h00 = vld1q_s16(H0 + j * SIMD_WIDTH16);
            int16x8_t s2  = vld1q_s16(seq2SoA + j * SIMD_WIDTH16);
            int16x8_t f11 = vld1q_s16(F + (j + 1) * SIMD_WIDTH16);

            int16x8_t xor_val = veorq_s16(s1, s2);
            /* 8-lane int16 table lookup. Narrow xor (int16x8, values in
             * [0..31]) to 8 u8 indices (low byte of each int16; high byte
             * is always 0 for these xor values). Look up in the 32-byte
             * int8 score table via vqtbl2, sign-extend int8x8 back to
             * int16x8. */
            uint8x8_t  idx8 = vmovn_u16(vreinterpretq_u16_s16(xor_val));
            int8x8_t   sbt8 = vqtbl2_s8(perm_vec, idx8);
            int16x8_t  sbt  = vmovl_s8(sbt8);

            /* Rank-1 freed-cell override (issue 173 + TAPS neutral): force the
             * freed score where this column's read base equals the row's active
             * freed base (built once per row in active_frread16 above). One compare
             * + one blend; FREED_INACTIVE16 lanes never match a real s2 so they
             * pass through. */
            if (HasFreed) {
                sbt = vbslq_s16(vceqq_s16(s2, active_frread16), freedval_vec16, sbt);
            }

            /* Boundary: high bit set in (s1 | s2) indicates padding. */
            int16x8_t or_val = vorrq_s16(s1, s2);
            uint16x8_t high_bit = vshrq_n_u16(vreinterpretq_u16_s16(or_val), 15);
            uint16x8_t is_boundary = vceqq_u16(high_bit, vdupq_n_u16(1));

            int16x8_t m11 = vaddq_s16(h00, sbt);
            m11 = vbslq_s16(is_boundary, zero_vec, m11);

            int16x8_t h11 = vmaxq_s16(m11, e11);
            h11 = vmaxq_s16(h11, f11);
            h11 = vmaxq_s16(h11, zero_vec);

            uint16x8_t cmp0 = vcgtq_s16(h11, imax_vec);
            imax_vec = vmaxq_s16(imax_vec, h11);
            iqe_vec  = vbslq_s16(cmp0, l_vec, iqe_vec);

            /* Reassociate both gap recurrences off h11 (variant C: h11 is
             * clamped to 0 above, so fold that 0 into mf/me to preserve the
             * floor). me reads OLD e11 — compute before the e-update. */
            int16x8_t mf = vmaxq_s16(vmaxq_s16(m11, f11), zero_vec);
            int16x8_t me = vmaxq_s16(vmaxq_s16(m11, e11), zero_vec);
            int16x8_t gapE = vsubq_s16(mf, oe_ins_vec);
            e11 = vsubq_s16(e11, e_ins_vec);
            e11 = vmaxq_s16(gapE, e11);

            int16x8_t gapD = vsubq_s16(me, oe_del_vec);
            int16x8_t f21  = vsubq_s16(f11, e_del_vec);
            f21 = vmaxq_s16(gapD, f21);

            vst1q_s16(H1 + (j + 1) * SIMD_WIDTH16, h11);
            vst1q_s16(F  + (j + 1) * SIMD_WIDTH16, f21);
            l_vec = vaddq_s16(l_vec, one_vec);
        }

        /* Block I: write prior-row's pimax to rowMax. No intermediate
         * masking — frozen lanes' imax after the freeze row is still
         * produced by the DP but the score2 scan filters by per-lane
         * primary region / minsc / qe, which matches scalar. */
        if (i > 0) {
            vst1q_s16(rowMax + (i - 1) * SIMD_WIDTH16, pimax_vec);
        }
        pimax_vec = imax_vec;

        /* Block II: gmax / te / qe update with per-lane freeze mask. */
        uint16x8_t cmp0 = vcgtq_s16(imax_vec, gmax_vec);
        uint16x8_t frozen_u16 = vreinterpretq_u16_s16(frozen_vec);
        uint16x8_t cmp0_active = vbicq_u16(cmp0, frozen_u16);

        int16x8_t new_gmax = vmaxq_s16(gmax_vec, imax_vec);
        gmax_vec = vbslq_s16(frozen_u16, gmax_vec, new_gmax);

        te_vec = vbslq_s16(cmp0_active, i_vec, te_vec);
        qe_vec = vbslq_s16(cmp0_active, iqe_vec, qe_vec);

        /* Freeze newly endsc-qualifying lanes (has_endsc gate prevents
         * trivial freeze for lanes without a KSW_XSTOP target). */
        uint16x8_t cmp_end   = vcgeq_s16(gmax_vec, endsc_vec);
        uint16x8_t just_hit  = vandq_u16(cmp_end, vreinterpretq_u16_s16(has_endsc_vec));
        frozen_vec = vorrq_s16(frozen_vec, vreinterpretq_s16_u16(just_hit));

        /* Collapse frozen_vec (lanes are 0x0000 or 0xFFFF) to an 8-bit
         * mask: bit l set iff lane l has hit its KSW_XSTOP target. Uses
         * the existing _mm_movemask_epi16 helper (extracts each u16
         * MSB), avoiding the scalar store+loop round-trip. */
        uint8_t frozen_bits = (uint8_t)_mm_movemask_epi16(vreinterpretq_m128i_s16(frozen_vec));

        /* Early exit only when every lane that *could* freeze has frozen
         * — i.e. all KSW_XSTOP-carrying lanes are done. Non-XSTOP lanes
         * (endsc_msk_a bit clear) never contribute to frozen_bits, so the
         * batched break matches scalar ksw_i16 (no batched global exit). */
        if (endsc_msk_a != 0 && (frozen_bits & endsc_msk_a) == endsc_msk_a) {
            limit = i++;
            break;
        }

        int16_t *S = H1; H1 = H0; H0 = S;
    }

    /* Store final row's pimax. Guard on i > 0 for all-padding batches
     * (nrow == 0 → i stays at 0 → underflow). See issue 38 / PR 289. */
    if (i > 0) {
        vst1q_s16(rowMax + (i - 1) * SIMD_WIDTH16, pimax_vec);
    }

    /* Extract primary results. */
    int16_t score[SIMD_WIDTH16] __attribute((aligned(128)));
    int16_t te1[SIMD_WIDTH16]   __attribute((aligned(128)));
    int16_t qe[SIMD_WIDTH16]    __attribute((aligned(128)));
    vst1q_s16(score, gmax_vec);
    vst1q_s16(te1,   te_vec);
    vst1q_s16(qe,    qe_vec);

    int live = 0;
    for (int l = 0; l < SIMD_WIDTH16 && (po_ind + l) < numPairs; l++) {
        int ind = po_ind + l;
#if !MAINY
        ind = p[l].regid;
        if (phase) {
            if (aln[ind].score == score[l]) {
                aln[ind].tb = aln[ind].te - te1[l];
                aln[ind].qb = aln[ind].qe - qe[l];
            }
        } else {
            aln[ind].score = score[l];
            aln[ind].te    = te1[l];
            aln[ind].qe    = qe[l];
            if (score[l] > 0) { qe[l] = 1; live++; } else qe[l] = 0;
        }
#else
        aln[ind].score = score[l];
        aln[ind].te    = te1[l];
        aln[ind].qe    = qe[l];
        if (score[l] > 0) { qe[l] = 1; live++; } else qe[l] = 0;
#endif
    }

#if !MAINY
    if (phase) return 1;
#endif
    if (live == 0) return 1;

    /* Score2 / te2 via per-lane scalar b[]-emulation over the now-
     * populated rowMax. Identical semantics to the 8-bit kernel's
     * score2 scan; differs only in int16_t types and SIMD_WIDTH16 stride. */
    int qmax = this->g_qmax;
    int16_t low[SIMD_WIDTH16]  __attribute((aligned(128)));
    int16_t high[SIMD_WIDTH16] __attribute((aligned(128)));
    for (int j = 0; j < SIMD_WIDTH16; j++) {
        int val = (score[j] + qmax - 1) / qmax;
        low[j]  = te1[j] - val;
        high[j] = te1[j] + val;
    }

    /* rowMax entries produced: i == nrow on normal completion, and
     * i == limit + 1 after early exit via KSW_XSTOP (limit = i++ stores
     * the old i). Using i directly covers both paths and includes the
     * terminating row stored at rowMax[i-1]. */
    const int processed_rows = i;

    for (int l = 0; l < SIMD_WIDTH16 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        /* Match scalar ksw_i16: when KSW_XSUBO is absent, minsc = 0x10000
         * so the b[] list never starts and score2/te2 stay at -1.
         * minsc_msk_a tracks lanes with KSW_XSUBO set; lanes not in the
         * mask have minsc[l] zero-initialized and must skip the scan.
         *
         * The !qe[l] liveness check is equivalent to kswv512_16's
         * te[l] < 0: the extraction block above overwrites qe[l] with
         * 1 (score > 0) or 0 as a liveness flag before this scan runs,
         * so !qe[l] cleanly marks lanes with no primary alignment. Do
         * not rewrite to te[l] < 0 without also reverting that
         * liveness-flag overwrite. */
        if (!qe[l] || !(minsc_msk_a & (1u << l))) {
            aln[ind].score2 = -1;
            aln[ind].te2    = -1;
            continue;
        }

        int len1_l  = (int)p[l].len1;
        int low_l   = (int)low[l];
        int high_l  = (int)high[l];
        int minsc_l = (int)minsc[l];
        int score2  = -1;
        int te2     = -1;
        int b_score = -1;
        int b_pos   = -2;

        int nrows = processed_rows < len1_l ? processed_rows : len1_l;
        for (int i2 = 0; i2 < nrows; i2++) {
            int imax = (int)rowMax[i2 * SIMD_WIDTH16 + l];
            if (imax < minsc_l) continue;

            if (b_pos + 1 != i2) {
                if (b_pos >= 0 &&
                    (b_pos < low_l || b_pos > high_l) &&
                    b_score > score2) {
                    score2 = b_score;
                    te2    = b_pos;
                }
                b_score = imax;
                b_pos   = i2;
            } else if (b_score < imax) {
                b_score = imax;
                b_pos   = i2;
            }
        }

        if (b_pos >= 0 &&
            (b_pos < low_l || b_pos > high_l) &&
            b_score > score2) {
            score2 = b_score;
            te2    = b_pos;
        }

        aln[ind].score2 = score2;
        aln[ind].te2    = te2;
    }

    return 1;
}

#elif ((!__AVX512BW__) & (__AVX2__))
/* AVX2 kswv kernel — 256-bit vectors, 32 u8 lanes per batch.
 *
 * Direct port of the corrected NEON kernel (kswv_neon_u8 above) with all
 * four bug fixes pre-applied (te split, freeze mask, per-lane score2
 * exclusion, minsc filter in score2 scan).
 *
 * Intrinsic translation notes:
 *   - No unsigned 8-bit compare in AVX2 → sign-flip trick:
 *       cmpgt_u8(a, b) = cmpgt_i8(a ^ 0x80, b ^ 0x80)
 *   - No 8-bit shift → isolate high bit via sign compare against zero.
 *   - 16-byte lookup table → broadcast to 256 bits then shuffle
 *     (shuffle_epi8 is 128-bit-lane; broadcast makes both lanes
 *     identical so any 0..15 index lands on the correct byte).
 *   - te uses SIMD_WIDTH8=32 lanes worth of int16 → two __m256i
 *     (te_vec_lo = pairs 0..15, te_vec_hi = pairs 16..31).
 */

#include <immintrin.h>

static inline __m256i avx2_cmpgt_u8(__m256i a, __m256i b)
{
    const __m256i sign = _mm256_set1_epi8((char)0x80);
    return _mm256_cmpgt_epi8(_mm256_xor_si256(a, sign),
                             _mm256_xor_si256(b, sign));
}

static inline __m256i avx2_cmpge_u8(__m256i a, __m256i b)
{
    /* a >= b  <=>  b <= a  <=>  max(a,b) == a */
    return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
}

/* NEON-style select: pick src where mask byte is 0xFF, else dst.
 * _mm256_blendv_epi8(dst, src, mask) uses the SIGN bit of each mask
 * byte: 0xFF (sign=1) picks src, 0x00 picks dst. Matches NEON's
 * vbslq_u8(mask, src, dst). */
static inline __m256i avx2_blendv_u8(__m256i mask, __m256i src, __m256i dst)
{
    return _mm256_blendv_epi8(dst, src, mask);
}

/* Widen the low/high 128 bits of a u8x32 into u16x16, zero-extended. */
static inline __m256i avx2_widen_u8_lo(__m256i v)
{
    return _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 0));
}
static inline __m256i avx2_widen_u8_hi(__m256i v)
{
    return _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1));
}

/* Signed 16-bit compares are provided directly by AVX2 (_mm256_cmpgt_epi16,
 * _mm256_cmpeq_epi16). No cmplt / cmpge; synthesize. */
static inline __m256i avx2_cmpgt_s16(__m256i a, __m256i b)
{
    return _mm256_cmpgt_epi16(a, b);
}
static inline __m256i avx2_cmpge_s16(__m256i a, __m256i b)
{
    /* a >= b  <=>  max(a,b) == a (signed). */
    return _mm256_cmpeq_epi16(_mm256_max_epi16(a, b), a);
}

/* Thin dispatcher: route to the HasFreed template instantiation. The <false>
 * path dead-code-eliminates every freed-cell override → byte-identical to the
 * pre-issue-173 kernel for symmetric (non-meth) matrices. Mirrors kswv_neon_u8. */
int kswv::kswv256_u8(uint8_t seq1SoA[],
                     uint8_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    return has_freed
        ? kswv256_u8_impl<true>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                po_ind, tid, numPairs, phase)
        : kswv256_u8_impl<false>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                 po_ind, tid, numPairs, phase);
}

template<bool HasFreed>
int kswv::kswv256_u8_impl(uint8_t seq1SoA[],
                     uint8_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    uint8_t minsc[SIMD_WIDTH8] __attribute__((aligned(64))) = {0};
    uint8_t endsc[SIMD_WIDTH8] __attribute__((aligned(64))) = {0};

    const __m256i zero_vec = _mm256_setzero_si256();

    /* Per-row running-max checkpoints; see kswv_neon_u8_impl. */
    alignas(64) uint8_t blockMax[QE_MAXBLK * SIMD_WIDTH8];

    /* Score lookup table (16 bytes). Indexed by (s1 ^ s2). */
    int8_t temp[SIMD_WIDTH8] __attribute__((aligned(64))) = {0};

    uint8_t shift;
    shift = min_(this->w_match, (int8_t)this->w_mismatch);
    shift = min_((int8_t)shift, this->w_ambig);
    shift = 256 - (uint8_t)shift;

    temp[0] = this->w_match;
    temp[1] = temp[2] = temp[3] = this->w_mismatch;
    temp[4] = temp[5] = temp[6] = temp[7] = this->w_ambig;
    temp[8] = temp[9] = temp[10] = temp[11] = this->w_ambig;
    temp[12] = this->w_ambig;
    for (int i = 0; i < 16; i++) temp[i] += shift;

    /* Broadcast 16-byte table across both 128-bit lanes of a 256-bit
     * vector so _mm256_shuffle_epi8 (128-bit-lane) still hits the right
     * entry for any 0..15 index. */
    __m128i permSft_128 = _mm_loadu_si128((const __m128i*)temp);
    __m256i permSft     = _mm256_broadcastsi128_si256(permSft_128);
    __m256i sft_vec     = _mm256_set1_epi8((char)shift);

    /* u8-tier saturation guard (see kswv_u8_saturation_guard). The u8 admission
     * bound keeps every admitted pair safe; this catches a violating param set
     * before it silently corrupts scores. */
    kswv_u8_saturation_guard(p, this->w_match, shift);

    uint32_t minsc_msk_a = 0, endsc_msk_a = 0;
    for (int i = 0; i < SIMD_WIDTH8; i++) {
        int xtra = p[i].h0;
        int val  = (xtra & KSW_XSUBO) ? (xtra & 0xffff) : 0x10000;
        if (val <= 255) { minsc[i] = (uint8_t)val; minsc_msk_a |= (1u << i); }
        val = (xtra & KSW_XSTOP) ? (xtra & 0xffff) : 0x10000;
        if (val <= 255) { endsc[i] = (uint8_t)val; endsc_msk_a |= (1u << i); }
    }

    const __m256i minsc_vec  = _mm256_loadu_si256((const __m256i*)minsc);
    const __m256i endsc_vec  = _mm256_loadu_si256((const __m256i*)endsc);
    const __m256i e_del_vec  = _mm256_set1_epi8((char)this->e_del);
    const __m256i oe_del_vec = _mm256_set1_epi8((char)(this->o_del + this->e_del));
    const __m256i e_ins_vec  = _mm256_set1_epi8((char)this->e_ins);
    const __m256i oe_ins_vec = _mm256_set1_epi8((char)(this->o_ins + this->e_ins));
    const __m256i five_vec   = _mm256_set1_epi8((char)DUMMY5);

    /* First column any lane can hold the DUMMY5 query pad. Below it the
     * cmpeq+blendv pair that tests for it cannot fire, so it is elided --
     * same derivation and byte-identity argument as the NEON kernel.
     * Unlike the boundary-mask hoist noted in the loop below, this REMOVES a
     * live vector (five_vec is dead in the cheap body) rather than pinning
     * one, so it does not provoke the AVX2 register-file spill that made the
     * boundary hoist a regression on this tier.
     *
     * ncol as the cap, not jsplit: this kernel has no jsplit range to subdivide,
     * so the cap is a no-op here (compute_jdummy starts at ncol and only ever
     * lowers it). */
    const int jdummy_a2 = compute_jdummy(p, SIMD_WIDTH8, ncol, ncol);
    const __m256i cmax_vec   = _mm256_set1_epi8((char)255);

    __m256i gmax_vec   = zero_vec;
    /* Fix 1: split te into two int16 vectors covering 32 pairs. */
    __m256i te_vec_lo  = _mm256_set1_epi16(-1);  /* pairs 0..15 */
    __m256i te_vec_hi  = _mm256_set1_epi16(-1);  /* pairs 16..31 */
    /* Fix 2: per-lane freeze once pair hits KSW_XSTOP. */
    __m256i frozen_vec = zero_vec;

    /* Fix 2 helper: only freeze lanes that actually set an endsc target.
     * endsc_vec defaults to 0 for lanes without a target (compares
     * trivially true with vcgeq_u8) — has_endsc_vec masks that out. */
    uint8_t _has_endsc_bytes[SIMD_WIDTH8];
    for (int i = 0; i < SIMD_WIDTH8; i++)
        _has_endsc_bytes[i] = (endsc_msk_a & (1u << i)) ? 0xFF : 0x00;
    __m256i has_endsc_vec = _mm256_loadu_si256((const __m256i*)_has_endsc_bytes);

    uint32_t exit0 = 0xFFFFFFFFu;

    tid = 0;
    uint8_t *H0     = H8_0    + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *H1     = H8_1    + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *F      = F8      + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *rowMax = rowMax8 + tid * SIMD_WIDTH8 * this->maxRefLen;

    // Per-strip L1 prefetches mirroring the AVX-512 8-bit and 16-bit kernels.
    // The 8-bit AVX2 path was missing them; without these the inner loop
    // stalls on L1 misses for F/H1/seq for the first several rows.
    _mm_prefetch((const char*) (F + SIMD_WIDTH8), _MM_HINT_T0);
    _mm_prefetch((const char*) seq2SoA, _MM_HINT_T1);
    _mm_prefetch((const char*) seq1SoA, _MM_HINT_NTA);
    _mm_prefetch((const char*) (H1 + SIMD_WIDTH8), _MM_HINT_T0);

    for (int i = 0; i <= ncol; i++) {
        _mm256_storeu_si256((__m256i*)(H0   + i * SIMD_WIDTH8), zero_vec);
        _mm256_storeu_si256((__m256i*)(F    + i * SIMD_WIDTH8), zero_vec);
    }
    _mm256_storeu_si256((__m256i*)H0, zero_vec);
    _mm256_storeu_si256((__m256i*)H1, zero_vec);

    __m256i pimax_vec = zero_vec;
    uint32_t mask32   = 0;

    __m256i imax_vec;
    __m256i qe_vec = zero_vec;

    int i, limit = nrow;
    for (i = 0; i < nrow; i++) {
        __m256i e11 = zero_vec;
        __m256i s1  = _mm256_loadu_si256((const __m256i*)(seq1SoA + i * SIMD_WIDTH8));
        imax_vec    = zero_vec;
        __m256i i_vec_s16 = _mm256_set1_epi16((int16_t)i);

        /* Freed-cell override (issue 173, bisulfite OT/OB + TAPS neutral). The
         * freed cell (s1==fr_ref, s2==fr_read) scores fr_val. s1 is
         * loop-invariant across the inner j loop, so hoist the per-row
         * comparison here; per-cell only the s2 compare + blend remains.
         * freedval256 is fr_val biased by +shift into the u8 domain (same bias
         * temp[] carries): fr_val==w_match (GENOMIC/COLLAPSED) equals temp[0]
         * exactly, NEUTRAL (fr_val==0) scores the conversion as 0 after unbiasing.
         * When !HasFreed, all of this and the per-cell block below compile out
         * (identical codegen to today). */
        /* Free a SYMMETRIC PAIR of cells (collapsed bisulfite); GENOMIC/NEUTRAL set
         * the mirror == primary so the second blend is idempotent. Both cells score
         * the same fr_val, so one freedval256 covers the folded target below. */
        __m256i freedval256, active_frread;
        if (HasFreed) {
            freedval256 = _mm256_set1_epi8((char)(fr_val + shift));
            __m256i rowfreed  = _mm256_cmpeq_epi8(s1, _mm256_set1_epi8((char)fr_ref));
            __m256i rowfreed2 = _mm256_cmpeq_epi8(s1, _mm256_set1_epi8((char)fr_ref2));
            /* One per-lane freed target: fr_read where ref==fr_ref, fr_read2 where
             * ref==fr_ref2, else FREED_INACTIVE8 (no s2 value — 0-3/5/8 or the
             * 0xFF pad — ever equals it). The two ref tests are mutually exclusive.
             * Collapses the per-cell freed path to one cmpeq + one blendv (see
             * NEON variant). */
            active_frread = avx2_blendv_u8(rowfreed2, _mm256_set1_epi8((char)fr_read2),
                                           _mm256_set1_epi8((char)FREED_INACTIVE8));
            active_frread = avx2_blendv_u8(rowfreed,  _mm256_set1_epi8((char)fr_read),
                                           active_frread);
        }

#define KSWV_AVX2_U8_CELL(NEED_DUMMY) \
{ \
            __m256i h00 = _mm256_loadu_si256((const __m256i*)(H0 + j * SIMD_WIDTH8)); \
            __m256i s2  = _mm256_loadu_si256((const __m256i*)(seq2SoA + j * SIMD_WIDTH8)); \
            __m256i f11 = _mm256_loadu_si256((const __m256i*)(F + (j + 1) * SIMD_WIDTH8)); \
 \
            __m256i xor_val = _mm256_xor_si256(s1, s2); \
            __m256i sbt     = _mm256_shuffle_epi8(permSft, xor_val); \
 \
            if (NEED_DUMMY) { \
                __m256i cmpq = _mm256_cmpeq_epi8(s2, five_vec); \
                sbt = avx2_blendv_u8(cmpq, sft_vec, sbt); \
            } \
 \
            /* Apply the freed-cell override: force the biased freed score \
             * (fr_val + shift) where this column's read base equals the row's \
             * active freed base (active_frread, built once per row above). One \
             * cmpeq + one blendv; FREED_INACTIVE8 lanes never match any s2 value \
             * (0-3/5/8 or the 0xFF pad) so they pass through. */ \
            if (HasFreed) { \
                sbt = _mm256_blendv_epi8(sbt, freedval256, \
                                         _mm256_cmpeq_epi8(s2, active_frread)); \
            } \
 \
            /* High bit of (s1 | s2) indicates boundary (padding 0xFF). \
             * Sign compare against zero gives 0xFF wherever high bit is \
             * set in the u8 byte. NB: the jsplit-bounded hoist that NEON and \
             * AVX-512 use below their group's first padded query column \
             * regresses here — AVX2's 16-register file spills once the mask is \
             * loop-invariant (however computed), so this tier keeps the \
             * per-cell s1|s2 form over the whole row. Never hoist the s2 half \
             * away on any tier: query columns past a lane's quantum DO carry \
             * bit 7 (see the query-padding contract at the top of this file). */ \
            __m256i or_val      = _mm256_or_si256(s1, s2); \
            __m256i is_boundary = _mm256_cmpgt_epi8(zero_vec, or_val); \
 \
            __m256i m11 = _mm256_adds_epu8(h00, sbt); \
            m11 = avx2_blendv_u8(is_boundary, zero_vec, m11); \
            m11 = _mm256_subs_epu8(m11, sft_vec); \
 \
            __m256i h11 = _mm256_max_epu8(m11, e11); \
            h11 = _mm256_max_epu8(h11, f11); \
 \
            /* Row max only; the query-end column is recovered after the row. \
             * See the rescan below kswv_neon_u8_impl's equivalent. */ \
            imax_vec = _mm256_max_epu8(imax_vec, h11); \
 \
            __m256i gapE = _mm256_subs_epu8(h11, oe_ins_vec); \
            e11 = _mm256_subs_epu8(e11, e_ins_vec); \
            e11 = _mm256_max_epu8(gapE, e11); \
 \
            __m256i gapD = _mm256_subs_epu8(h11, oe_del_vec); \
            __m256i f21  = _mm256_subs_epu8(f11, e_del_vec); \
            f21 = _mm256_max_epu8(gapD, f21); \
 \
            _mm256_storeu_si256((__m256i*)(H1 + (j + 1) * SIMD_WIDTH8), h11); \
            _mm256_storeu_si256((__m256i*)(F  + (j + 1) * SIMD_WIDTH8), f21); \
            if (j == qeNext) { \
                _mm256_store_si256((__m256i*)(blockMax + qeBlk * SIMD_WIDTH8), imax_vec); \
                qeBlk++; qeNext += QE_BLK; \
            } \
}
        int qeNext = QE_BLK - 1, qeBlk = 0;
        int j = 0;
        for (; j < jdummy_a2; j++) KSWV_AVX2_U8_CELL(false)
        for (; j < ncol; j++)      KSWV_AVX2_U8_CELL(true)
#undef KSWV_AVX2_U8_CELL

        /* Close the final (possibly partial) block; see kswv_neon_u8_impl. */
        _mm256_store_si256((__m256i*)(blockMax + ((ncol - 1) / QE_BLK) * SIMD_WIDTH8),
                           imax_vec);

        /* Block I - rowMax tracking. Same shape as NEON. */
        if (i > 0) {
            __m256i cmp_gt = avx2_cmpgt_u8(imax_vec, pimax_vec);
            uint32_t msk32 = (uint32_t)_mm256_movemask_epi8(cmp_gt);
            msk32 |= mask32;
            pimax_vec = avx2_blendv_u8(cmp_gt, zero_vec, pimax_vec);
            _mm256_storeu_si256((__m256i*)(rowMax + (i - 1) * SIMD_WIDTH8), pimax_vec);
            mask32 = ~msk32;
        }
        pimax_vec = imax_vec;

        /* Block II: gmax, te with freeze mask (fix 2) */
        __m256i cmp0 = avx2_cmpgt_u8(imax_vec, gmax_vec);
        uint32_t cmp0_msk = (uint32_t)_mm256_movemask_epi8(cmp0);
        cmp0_msk &= exit0;

        /* 16-bit mirrors of cmp0 for te update (fix 1 split) */
        __m256i cmp_lo_16 = avx2_cmpgt_s16(
            avx2_widen_u8_lo(imax_vec), avx2_widen_u8_lo(gmax_vec));
        __m256i cmp_hi_16 = avx2_cmpgt_s16(
            avx2_widen_u8_hi(imax_vec), avx2_widen_u8_hi(gmax_vec));

        /* 16-bit mirrors of frozen_vec to mask te updates. */
        __m256i frozen_lo_16 = avx2_cmpgt_s16(
            avx2_widen_u8_lo(frozen_vec), _mm256_setzero_si256());
        __m256i frozen_hi_16 = avx2_cmpgt_s16(
            avx2_widen_u8_hi(frozen_vec), _mm256_setzero_si256());

        cmp_lo_16 = _mm256_andnot_si256(frozen_lo_16, cmp_lo_16);
        cmp_hi_16 = _mm256_andnot_si256(frozen_hi_16, cmp_hi_16);
        __m256i cmp0_active = _mm256_andnot_si256(frozen_vec, cmp0);

        __m256i new_gmax = _mm256_max_epu8(gmax_vec, imax_vec);
        gmax_vec = avx2_blendv_u8(frozen_vec, gmax_vec, new_gmax);

        te_vec_lo = avx2_blendv_u8(cmp_lo_16, i_vec_s16, te_vec_lo);
        te_vec_hi = avx2_blendv_u8(cmp_hi_16, i_vec_s16, te_vec_hi);

        /* Deferred query-end recovery; see the rescan in kswv_neon_u8_impl for
         * the full derivation of why this is identical to the per-cell form. */
        const int active_msk = _mm256_movemask_epi8(cmp0_active);
        if (active_msk)
        {
            __m256i iqe_vec = _mm256_set1_epi8((char)0xFF);
            __m256i foundBlk = zero_vec;
            const uint8_t *colIdx = this->colIdx8;
            const int nblocks = (ncol + QE_BLK - 1) / QE_BLK;
            for (int b = 0; b < nblocks; b++) {
                __m256i reached = _mm256_cmpeq_epi8(
                    _mm256_load_si256((const __m256i*)(blockMax + b * SIMD_WIDTH8)), imax_vec);
                __m256i newly = _mm256_and_si256(
                    _mm256_andnot_si256(foundBlk, reached), cmp0_active);
                foundBlk = _mm256_or_si256(foundBlk, reached);
                if (_mm256_movemask_epi8(newly)) {
                    const int j0 = b * QE_BLK;
                    const int j1 = (j0 + QE_BLK < ncol) ? (j0 + QE_BLK) : ncol;
                    __m256i got = zero_vec;
                    for (int j2 = j0; j2 < j1; j2++) {
                        __m256i eq = _mm256_and_si256(
                            _mm256_cmpeq_epi8(
                                _mm256_loadu_si256((const __m256i*)(H1 + (j2 + 1) * SIMD_WIDTH8)),
                                imax_vec),
                            newly);
                        iqe_vec = avx2_blendv_u8(
                            _mm256_andnot_si256(got, eq),
                            _mm256_loadu_si256((const __m256i*)(colIdx + j2 * SIMD_WIDTH8)),
                            iqe_vec);
                        got = _mm256_or_si256(got, eq);
                    }
                }
                if ((_mm256_movemask_epi8(foundBlk) & active_msk) == active_msk) break;
            }
            qe_vec = avx2_blendv_u8(cmp0_active, iqe_vec, qe_vec);
        }

        /* End-score check + freeze update. */
        __m256i cmp_end = avx2_cmpge_u8(gmax_vec, endsc_vec);
        uint32_t cmp_end_msk = (uint32_t)_mm256_movemask_epi8(cmp_end);
        cmp_end_msk &= endsc_msk_a;

        __m256i just_hit = _mm256_and_si256(cmp_end, has_endsc_vec);
        frozen_vec = _mm256_or_si256(frozen_vec, just_hit);

        /* Overflow check. */
        __m256i left_vec = _mm256_adds_epu8(gmax_vec, sft_vec);
        __m256i cmp2     = avx2_cmpge_u8(left_vec, cmax_vec);
        uint32_t cmp2_msk = (uint32_t)_mm256_movemask_epi8(cmp2);

        exit0 = (~(cmp_end_msk | cmp2_msk)) & exit0;
        if (exit0 == 0) { limit = i++; break; }

        uint8_t *S = H1; H1 = H0; H0 = S;
    }

    /* Guard on i > 0: when every pair in the batch has len1 == 0, nrow
     * is 0, the DP loop never runs, and `i - 1` underflows into the
     * rowMax allocation prefix. See issue 38 / upstream PR 289. */
    if (i > 0) {
        _mm256_storeu_si256((__m256i*)(rowMax + (i - 1) * SIMD_WIDTH8), pimax_vec);
    }

    /* Extract results. */
    uint8_t score[SIMD_WIDTH8] __attribute__((aligned(64)));
    int16_t te1[SIMD_WIDTH8]    __attribute__((aligned(64)));
    uint8_t qe[SIMD_WIDTH8]     __attribute__((aligned(64)));

    _mm256_storeu_si256((__m256i*)score, gmax_vec);
    _mm256_storeu_si256((__m256i*)te1, te_vec_lo);        /* pairs 0..15 */
    _mm256_storeu_si256((__m256i*)(te1 + 16), te_vec_hi); /* pairs 16..31 */
    _mm256_storeu_si256((__m256i*)qe, qe_vec);

    int live = 0;
    for (int l = 0; l < SIMD_WIDTH8 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        int16_t *te = te1;
        if (phase) {
            if (aln[ind].score == score[l]) {
                aln[ind].tb = aln[ind].te - te[l];
                aln[ind].qb = aln[ind].qe - qe[l];
            }
        } else {
            aln[ind].score = score[l] + shift < 255 ? score[l] : 255;
            aln[ind].te = te[l];
            aln[ind].qe = qe[l];
            if (aln[ind].score != 255) { qe[l] = 1; live++; }
            else qe[l] = 0;
        }
    }

    if (phase) return 1;
    if (live == 0) return 1;

    /* Score2 and te2 computation.
     *
     * The SIMD dense-rowMax approach (max2 over every row outside the
     * primary region) diverges from scalar ksw_u8 on plateaus: scalar
     * collapses consecutive rows >= minsc into a single b[] entry
     * anchored at the max-score row, so a plateau straddling the primary
     * region boundary (anchor inside, later rows outside) is excluded
     * entirely. Dense rowMax would otherwise pull partial-plateau rows
     * in and inflate `sub`. Emulate scalar exactly, per lane, on the
     * precomputed rowMax. Cost is O(SIMD_WIDTH8 * limit) scalar ops per
     * batch, negligible next to the SIMD DP. */
    int qmax = this->g_qmax;
    int16_t low[SIMD_WIDTH8]  __attribute__((aligned(64)));
    int16_t high[SIMD_WIDTH8] __attribute__((aligned(64)));
    for (int j = 0; j < SIMD_WIDTH8; j++) {
        int val = (score[j] + qmax - 1) / qmax;
        low[j]  = te1[j] - val;
        high[j] = te1[j] + val;
    }

    /* rowMax entries produced: i == limit on normal completion, and
     * i == limit + 1 after early exit (limit = i++ stores old i). Using i
     * directly gives a single expression that covers both exit paths and
     * includes the terminating row stored at rowMax[limit]. */
    const int processed_rows = i;

    for (int l = 0; l < SIMD_WIDTH8 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        /* Match scalar ksw_u8: when KSW_XSUBO is absent, minsc = 0x10000
         * so the b[] list never starts and score2/te2 stay at -1.
         * minsc_msk_a tracks lanes with KSW_XSUBO set; lanes not in the
         * mask have minsc[l] zero-initialized and must skip the scan. */
        if (!qe[l] || !(minsc_msk_a & (1u << l))) {
            aln[ind].score2 = -1;
            aln[ind].te2    = -1;
            continue;
        }

        int len1_l  = (int)p[l].len1;
        int low_l   = (int)low[l];
        int high_l  = (int)high[l];
        int minsc_l = (int)minsc[l];
        int score2  = -1;
        int te2     = -1;
        /* b_pos = -2 sentinel: (-2 + 1) != any real row index, forces
         * an "append" on first qualifying row. */
        int b_score = -1;
        int b_pos   = -2;

        int nrows = processed_rows < len1_l ? processed_rows : len1_l;
        for (int i2 = 0; i2 < nrows; i2++) {
            int imax = (int)rowMax[i2 * SIMD_WIDTH8 + l];
            if (imax < minsc_l) continue;

            if (b_pos + 1 != i2) {
                /* APPEND: flush the outgoing b[] entry to score2 first. */
                if (b_pos >= 0 &&
                    (b_pos < low_l || b_pos > high_l) &&
                    b_score > score2) {
                    score2 = b_score;
                    te2    = b_pos;
                }
                b_score = imax;
                b_pos   = i2;
            } else if (b_score < imax) {
                /* UPDATE: strict greater extends the run's anchor. Equal
                 * imax leaves b_pos in place (matches scalar). */
                b_score = imax;
                b_pos   = i2;
            }
        }

        /* Flush trailing b[] entry. */
        if (b_pos >= 0 &&
            (b_pos < low_l || b_pos > high_l) &&
            b_score > score2) {
            score2 = b_score;
            te2    = b_pos;
        }

        aln[ind].score2 = score2;
        aln[ind].te2    = te2;
    }

    return 1;
}

void kswv::kswvBatchWrapper8_avx2(SeqPair *pairArray,
                                  uint8_t *seqBufRef,
                                  uint8_t *seqBufQer,
                                  kswr_t* aln,
                                  int32_t numPairs,
                                  uint16_t numThreads,
                                  int phase)
{
    uint8_t *seq1SoA = (uint8_t*)_mm_malloc(
        (size_t)this->maxRefLen * SIMD_WIDTH8 * numThreads * sizeof(uint8_t), 128);
    uint8_t *seq2SoA = (uint8_t*)_mm_malloc(
        (size_t)this->maxQerLen * SIMD_WIDTH8 * numThreads * sizeof(uint8_t), 128);
    assert(seq1SoA && seq2SoA);

    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH8 - 1) / SIMD_WIDTH8) * SIMD_WIDTH8;
    for (int32_t ii = numPairs; ii < roundNumPairs; ii++) {
        pairArray[ii].regid = ii;
        pairArray[ii].id    = ii;
        pairArray[ii].len1  = 0;
        pairArray[ii].len2  = 0;
        // The kernel's per-lane loop reads idr/idq/h0 for every lane up to
        // SIMD_WIDTH, so it reads these padded lanes too; zero them (as id/len1/
        // len2 above) to keep those reads well-defined. Padded lanes join the SIMD
        // batch (and its cross-lane reductions), but the caller reads results back
        // only for real lanes and whole-aligner output is byte-identical (validated).
        pairArray[ii].idr = 0;
        pairArray[ii].idq = 0;
        pairArray[ii].h0 = 0;
    }

    uint16_t tid = 0;
    uint8_t *mySeq1SoA = seq1SoA + tid * this->maxRefLen * SIMD_WIDTH8;
    uint8_t *mySeq2SoA = seq2SoA + tid * this->maxQerLen * SIMD_WIDTH8;

    for (int32_t i = 0; i < numPairs; i += SIMD_WIDTH8) {
        int maxLen1 = 0, maxLen2 = 0;

        for (int j = 0; j < SIMD_WIDTH8; j++) {
            SeqPair sp = pairArray[i + j];
            uint8_t *seq1 = seqBufRef + sp.idr;
            for (int k = 0; k < sp.len1; k++)
                mySeq1SoA[k * SIMD_WIDTH8 + j] = (seq1[k] == AMBIG_ ? AMBR : seq1[k]);
            if (maxLen1 < sp.len1) maxLen1 = sp.len1;
        }
        for (int j = 0; j < SIMD_WIDTH8; j++) {
            SeqPair sp = pairArray[i + j];
            for (int k = sp.len1; k <= maxLen1; k++)
                mySeq1SoA[k * SIMD_WIDTH8 + j] = 0xFF;
        }

        for (int j = 0; j < SIMD_WIDTH8; j++) {
            SeqPair sp = pairArray[i + j];
            uint8_t *seq2 = seqBufQer + sp.idq;
            int quanta = query_quantum8(sp.len2);
            for (int k = 0; k < sp.len2; k++)
                mySeq2SoA[k * SIMD_WIDTH8 + j] = (seq2[k] == AMBIG_ ? AMBQ : seq2[k]);
            for (int k = sp.len2; k < quanta; k++)
                mySeq2SoA[k * SIMD_WIDTH8 + j] = DUMMY5;
            if (maxLen2 < quanta) maxLen2 = quanta;
        }
        for (int j = 0; j < SIMD_WIDTH8; j++) {
            SeqPair sp = pairArray[i + j];
            int quanta = query_quantum8(sp.len2);
            for (int k = quanta; k <= maxLen2; k++)
                mySeq2SoA[k * SIMD_WIDTH8 + j] = 0xFF;
        }

        kswv256_u8(mySeq1SoA, mySeq2SoA,
                   (int16_t)maxLen1, (int16_t)maxLen2,
                   pairArray + i, aln, i, tid,
                   numPairs, phase);
    }

    _mm_free(seq1SoA);
    _mm_free(seq2SoA);
}

void kswv::getScores8(SeqPair *pairArray,
                      uint8_t *seqBufRef,
                      uint8_t *seqBufQer,
                      kswr_t *aln,
                      int32_t numPairs,
                      uint16_t numThreads,
                      int phase)
{
    kswvBatchWrapper8_avx2(pairArray, seqBufRef, seqBufQer, aln,
                           numPairs, numThreads, phase);
}

/* AVX2 16-bit kswv kernel — 256-bit vectors, 16 int16 lanes per batch.
 *
 * Direct port of the corrected NEON kernel (kswv_neon_16 above). On AVX2,
 * SIMD_WIDTH16 == 16, so all 16 int16 lanes fit in a single __m256i —
 * unlike the 8-bit AVX2 kernel there is no _lo/_hi te split. The NEON
 * template is the mask-free idiom of choice here: where AVX-512's
 * kswv512_16 expresses the per-lane freeze/early-exit with __mmask32
 * registers, AVX2 has no mask registers, so the NEON full-lane-mask
 * (0xFFFF/0x0000) formulation maps over cleanly with blendv/andnot.
 *
 * Intrinsic translation notes (vs NEON kswv_neon_16):
 *   - 32-entry int8 score table → two 16-byte halves, each broadcast to
 *     256 bits, two _mm256_shuffle_epi8, blended by (xor >= 16). The low
 *     byte of each int16 lane carries the looked-up int8 score; it is
 *     sign-extended to int16 via slli(8)+srai(8).
 *   - is_boundary (high bit of s1|s2) → _mm256_srai_epi16(or_val, 15),
 *     which yields 0xFFFF wherever the MSB is set, 0x0000 otherwise.
 *   - vbslq_s16(mask, a, b) → avx2_blendv_u8(mask, a, b) (byte-granular
 *     blend is exact because every mask lane is uniform 0xFFFF / 0x0000).
 *   - vbicq_u16(a, b) → _mm256_andnot_si256(b, a).
 *   - frozen_bits: extract the two 128-bit halves, _mm_packs_epi16 them
 *     to bytes (0xFFFF→0xFF, 0x0000→0x00), then _mm_movemask_epi8 — bit l
 *     == lane l. NB: the 128-bit packs on extracted halves is required;
 *     _mm256_packs_epi16 would lane-cross and scramble lane→bit order.
 */
/* Thin dispatcher: see kswv256_u8 above. */
int kswv::kswv256_16(int16_t seq1SoA[],
                     int16_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    return has_freed
        ? kswv256_16_impl<true>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                po_ind, tid, numPairs, phase)
        : kswv256_16_impl<false>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                 po_ind, tid, numPairs, phase);
}

template<bool HasFreed>
int kswv::kswv256_16_impl(int16_t seq1SoA[],
                     int16_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    int16_t minsc[SIMD_WIDTH16] __attribute__((aligned(64))) = {0};
    int16_t endsc[SIMD_WIDTH16] __attribute__((aligned(64))) = {0};

    const __m256i zero_vec = _mm256_setzero_si256();
    const __m256i one_vec  = _mm256_set1_epi16(1);

    /* 32-entry int8 score table indexed by (s1 ^ s2) in [0..31]. Built
     * identically to the NEON kernel, including the query-tail DUMMY3
     * (=26) entries that must contribute 0. */
    int8_t temp8[32] __attribute__((aligned(16)));
    for (int i = 0; i < 32; i++) temp8[i] = this->w_ambig;
    temp8[0] = this->w_match;
    temp8[1] = temp8[2] = temp8[3] = this->w_mismatch;
    temp8[21] = 0;
    temp8[24] = temp8[25] = temp8[26] = temp8[27] = 0;

    __m256i permLo = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i*)temp8));
    __m256i permHi = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i*)(temp8 + 16)));
    const __m256i fifteen_vec = _mm256_set1_epi16(15);

    uint32_t minsc_msk_a = 0, endsc_msk_a = 0;
    for (int i = 0; i < SIMD_WIDTH16; i++) {
        int xtra = p[i].h0;
        int val  = (xtra & KSW_XSUBO) ? (xtra & 0xffff) : 0x10000;
        if (val <= SHRT_MAX) { minsc[i] = (int16_t)val; minsc_msk_a |= (1u << i); }
        val = (xtra & KSW_XSTOP) ? (xtra & 0xffff) : 0x10000;
        if (val <= SHRT_MAX) { endsc[i] = (int16_t)val; endsc_msk_a |= (1u << i); }
    }

    const __m256i endsc_vec  = _mm256_loadu_si256((const __m256i*)endsc);
    const __m256i e_del_vec  = _mm256_set1_epi16((int16_t)this->e_del);
    const __m256i oe_del_vec = _mm256_set1_epi16((int16_t)(this->o_del + this->e_del));
    const __m256i e_ins_vec  = _mm256_set1_epi16((int16_t)this->e_ins);
    const __m256i oe_ins_vec = _mm256_set1_epi16((int16_t)(this->o_ins + this->e_ins));

    __m256i gmax_vec   = zero_vec;
    __m256i te_vec     = _mm256_set1_epi16(-1);
    __m256i qe_vec     = zero_vec;
    __m256i frozen_vec = zero_vec;

    /* Only freeze lanes that actually set an endsc (KSW_XSTOP) target. */
    int16_t _has_endsc_arr[SIMD_WIDTH16] __attribute__((aligned(64)));
    for (int i = 0; i < SIMD_WIDTH16; i++)
        _has_endsc_arr[i] = (endsc_msk_a & (1u << i)) ? (int16_t)0xFFFF : (int16_t)0x0000;
    __m256i has_endsc_vec = _mm256_loadu_si256((const __m256i*)_has_endsc_arr);

    tid = 0;
    int16_t *H0     = H16_0    + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *H1     = H16_1    + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *F      = F16      + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *rowMax = rowMax16 + tid * SIMD_WIDTH16 * this->maxRefLen;

    /* Per-strip L1 prefetches, mirroring the AVX-512 and NEON 16-bit
     * kernels. F/H1/seq2SoA are loop-resident (T0/T1); seq1SoA is read
     * once per row (NTA). */
    _mm_prefetch((const char*) (F + SIMD_WIDTH16), _MM_HINT_T0);
    _mm_prefetch((const char*) seq2SoA, _MM_HINT_T0);
    _mm_prefetch((const char*) seq1SoA, _MM_HINT_NTA);
    _mm_prefetch((const char*) (H1 + SIMD_WIDTH16), _MM_HINT_T1);

    for (int i = 0; i <= ncol; i++) {
        _mm256_storeu_si256((__m256i*)(H0   + i * SIMD_WIDTH16), zero_vec);
        _mm256_storeu_si256((__m256i*)(F    + i * SIMD_WIDTH16), zero_vec);
    }
    _mm256_storeu_si256((__m256i*)H0, zero_vec);
    _mm256_storeu_si256((__m256i*)H1, zero_vec);

    __m256i pimax_vec = zero_vec;

    __m256i imax_vec;
    int i, limit = nrow;
    for (i = 0; i < nrow; i++) {
        __m256i e11 = zero_vec;
        __m256i s1  = _mm256_loadu_si256((const __m256i*)(seq1SoA + i * SIMD_WIDTH16));
        imax_vec    = zero_vec;
        __m256i iqe_vec = _mm256_set1_epi16(-1);
        __m256i l_vec   = zero_vec;
        __m256i i_vec   = _mm256_set1_epi16((int16_t)i);

        /* Rank-1 freed-cell override (issue 173 + TAPS neutral), folded into ONE
         * per-row target base active_frread — the u16 analog of kswv256_u8_impl's
         * active_frread. A lane whose ref is fr_ref frees against fr_read, a lane
         * whose ref is fr_ref2 frees against fr_read2 (the two ref-side compares
         * are mutually exclusive), and every other lane gets a sentinel no real s2
         * ever equals. u16 s2 ∈ {0-3, AMBQ16=16, DUMMY3=26, 0xFFFF pad}, so FREED_INACTIVE16
         * is safe here. cmpeq_epi16 produces whole-int16 masks, so the byte-wise
         * blendv selects whole lanes correctly. The 16-bit kernel has NO shift bias
         * (m11 = h00 + sbt directly), so the freed value is just fr_val: fr_val ==
         * w_match (GENOMIC/COLLAPSED) equals temp8[0], NEUTRAL (fr_val==0) scores
         * the conversion as 0. !HasFreed → this and the per-cell block compile
         * out. */
        __m256i freedval256, active_frread;
        if (HasFreed) {
            freedval256 = _mm256_set1_epi16((int16_t)fr_val);
            __m256i rowfreed  = _mm256_cmpeq_epi16(s1, _mm256_set1_epi16((int16_t)fr_ref));
            __m256i rowfreed2 = _mm256_cmpeq_epi16(s1, _mm256_set1_epi16((int16_t)fr_ref2));
            active_frread = _mm256_blendv_epi8(_mm256_set1_epi16((int16_t)FREED_INACTIVE16),
                                               _mm256_set1_epi16((int16_t)fr_read2), rowfreed2);
            active_frread = _mm256_blendv_epi8(active_frread,
                                               _mm256_set1_epi16((int16_t)fr_read), rowfreed);
        }

        for (int j = 0; j < ncol; j++) {
            __m256i h00 = _mm256_loadu_si256((const __m256i*)(H0 + j * SIMD_WIDTH16));
            __m256i s2  = _mm256_loadu_si256((const __m256i*)(seq2SoA + j * SIMD_WIDTH16));
            __m256i f11 = _mm256_loadu_si256((const __m256i*)(F + (j + 1) * SIMD_WIDTH16));

            /* 32-entry table lookup (see header comment). The low byte of
             * each int16 lane is the score; sign-extend it to int16. */
            __m256i xor_val = _mm256_xor_si256(s1, s2);
            __m256i sbt_lo  = _mm256_shuffle_epi8(permLo, xor_val);
            __m256i sbt_hi  = _mm256_shuffle_epi8(permHi, xor_val);
            __m256i hi_sel  = _mm256_cmpgt_epi16(xor_val, fifteen_vec);
            __m256i sbt_b   = _mm256_blendv_epi8(sbt_lo, sbt_hi, hi_sel);
            __m256i sbt     = _mm256_srai_epi16(_mm256_slli_epi16(sbt_b, 8), 8);

            /* Rank-1 freed-cell override (issue 173 + TAPS neutral): force the
             * freed score where this column's read base equals the row's active
             * freed base (built once per row in active_frread above). One cmpeq +
             * one blendv; FREED_INACTIVE16 lanes never match a real s2 so they
             * pass through. cmpeq_epi16 gives whole-int16 masks, so the byte-wise
             * blendv is lane-correct. */
            if (HasFreed) {
                sbt = _mm256_blendv_epi8(sbt, freedval256, _mm256_cmpeq_epi16(s2, active_frread));
            }

            /* Boundary: high bit set in (s1 | s2) marks padding (0xFFFF). */
            __m256i or_val      = _mm256_or_si256(s1, s2);
            __m256i is_boundary = _mm256_srai_epi16(or_val, 15);

            __m256i m11 = _mm256_add_epi16(h00, sbt);
            /* Zero m11 on padding lanes. is_boundary is full-width (0x0000/
             * 0xFFFF per lane), so andnot(is_boundary, m11) is byte-identical to
             * blendv(m11, 0, is_boundary) but cheaper here: on this 16-bit kernel
             * andnot lets clang-19 pick a tighter register allocation, measured
             * at +8-10% on Intel Sapphire Rapids and neutral on AMD Zen3, avx2
             * tier (issue #380). NB: the 8-bit twin at the top of this file keeps
             * blendv on purpose — there andnot perturbs regalloc the other way
             * and regresses ~2-3% on both vendors. */
            m11 = _mm256_andnot_si256(is_boundary, m11);

            __m256i h11 = _mm256_max_epi16(m11, e11);
            h11 = _mm256_max_epi16(h11, f11);
            h11 = _mm256_max_epi16(h11, zero_vec);

            __m256i cmp0 = avx2_cmpgt_s16(h11, imax_vec);
            imax_vec = _mm256_max_epi16(imax_vec, h11);
            iqe_vec  = avx2_blendv_u8(cmp0, l_vec, iqe_vec);

            __m256i gapE = _mm256_sub_epi16(h11, oe_ins_vec);
            e11 = _mm256_sub_epi16(e11, e_ins_vec);
            e11 = _mm256_max_epi16(gapE, e11);

            __m256i gapD = _mm256_sub_epi16(h11, oe_del_vec);
            __m256i f21  = _mm256_sub_epi16(f11, e_del_vec);
            f21 = _mm256_max_epi16(gapD, f21);

            _mm256_storeu_si256((__m256i*)(H1 + (j + 1) * SIMD_WIDTH16), h11);
            _mm256_storeu_si256((__m256i*)(F  + (j + 1) * SIMD_WIDTH16), f21);
            l_vec = _mm256_add_epi16(l_vec, one_vec);
        }

        /* Block I: write prior row's pimax to rowMax (plain store; the
         * score2 scan filters per lane, matching the NEON 16-bit kernel). */
        if (i > 0) {
            _mm256_storeu_si256((__m256i*)(rowMax + (i - 1) * SIMD_WIDTH16), pimax_vec);
        }
        pimax_vec = imax_vec;

        /* Block II: gmax / te / qe update with per-lane freeze mask. */
        __m256i cmp0        = avx2_cmpgt_s16(imax_vec, gmax_vec);
        __m256i cmp0_active = _mm256_andnot_si256(frozen_vec, cmp0);

        __m256i new_gmax = _mm256_max_epi16(gmax_vec, imax_vec);
        gmax_vec = avx2_blendv_u8(frozen_vec, gmax_vec, new_gmax);

        te_vec = avx2_blendv_u8(cmp0_active, i_vec, te_vec);
        qe_vec = avx2_blendv_u8(cmp0_active, iqe_vec, qe_vec);

        /* Freeze newly endsc-qualifying lanes (has_endsc gate). */
        __m256i cmp_end  = avx2_cmpge_s16(gmax_vec, endsc_vec);
        __m256i just_hit = _mm256_and_si256(cmp_end, has_endsc_vec);
        frozen_vec = _mm256_or_si256(frozen_vec, just_hit);

        /* Collapse frozen_vec (lanes 0x0000/0xFFFF) to a 16-bit mask:
         * bit l set iff lane l hit its KSW_XSTOP target. 128-bit packs on
         * the extracted halves keeps bit l == lane l (a 256-bit packs
         * would lane-cross). */
        __m128i fr_lo = _mm256_extracti128_si256(frozen_vec, 0);
        __m128i fr_hi = _mm256_extracti128_si256(frozen_vec, 1);
        uint16_t frozen_bits =
            (uint16_t)_mm_movemask_epi8(_mm_packs_epi16(fr_lo, fr_hi));

        /* Early exit only when every KSW_XSTOP-carrying lane is done.
         * Non-XSTOP lanes never contribute to frozen_bits, so this matches
         * scalar ksw_i16 (no batched global exit). */
        if (endsc_msk_a != 0 && (frozen_bits & endsc_msk_a) == endsc_msk_a) {
            limit = i++;
            break;
        }

        int16_t *S = H1; H1 = H0; H0 = S;
    }

    /* Store final row's pimax. Guard on i > 0 for all-padding batches. */
    if (i > 0) {
        _mm256_storeu_si256((__m256i*)(rowMax + (i - 1) * SIMD_WIDTH16), pimax_vec);
    }

    /* Extract primary results. */
    int16_t score[SIMD_WIDTH16] __attribute__((aligned(64)));
    int16_t te1[SIMD_WIDTH16]   __attribute__((aligned(64)));
    int16_t qe[SIMD_WIDTH16]    __attribute__((aligned(64)));
    _mm256_storeu_si256((__m256i*)score, gmax_vec);
    _mm256_storeu_si256((__m256i*)te1,   te_vec);
    _mm256_storeu_si256((__m256i*)qe,    qe_vec);

    int live = 0;
    for (int l = 0; l < SIMD_WIDTH16 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        if (phase) {
            if (aln[ind].score == score[l]) {
                aln[ind].tb = aln[ind].te - te1[l];
                aln[ind].qb = aln[ind].qe - qe[l];
            }
        } else {
            aln[ind].score = score[l];
            aln[ind].te    = te1[l];
            aln[ind].qe    = qe[l];
            if (score[l] > 0) { qe[l] = 1; live++; } else qe[l] = 0;
        }
    }

    if (phase) return 1;
    if (live == 0) return 1;

    /* Score2 / te2 via per-lane scalar b[]-emulation over rowMax. Identical
     * semantics to the NEON 16-bit kernel; pure scalar, SIMD_WIDTH16 stride. */
    int qmax = this->g_qmax;
    int16_t low[SIMD_WIDTH16]  __attribute__((aligned(64)));
    int16_t high[SIMD_WIDTH16] __attribute__((aligned(64)));
    for (int j = 0; j < SIMD_WIDTH16; j++) {
        int val = (score[j] + qmax - 1) / qmax;
        low[j]  = te1[j] - val;
        high[j] = te1[j] + val;
    }

    const int processed_rows = i;

    for (int l = 0; l < SIMD_WIDTH16 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        if (!qe[l] || !(minsc_msk_a & (1u << l))) {
            aln[ind].score2 = -1;
            aln[ind].te2    = -1;
            continue;
        }

        int len1_l  = (int)p[l].len1;
        int low_l   = (int)low[l];
        int high_l  = (int)high[l];
        int minsc_l = (int)minsc[l];
        int score2  = -1;
        int te2     = -1;
        int b_score = -1;
        int b_pos   = -2;

        int nrows = processed_rows < len1_l ? processed_rows : len1_l;
        for (int i2 = 0; i2 < nrows; i2++) {
            int imax = (int)rowMax[i2 * SIMD_WIDTH16 + l];
            if (imax < minsc_l) continue;

            if (b_pos + 1 != i2) {
                if (b_pos >= 0 &&
                    (b_pos < low_l || b_pos > high_l) &&
                    b_score > score2) {
                    score2 = b_score;
                    te2    = b_pos;
                }
                b_score = imax;
                b_pos   = i2;
            } else if (b_score < imax) {
                b_score = imax;
                b_pos   = i2;
            }
        }

        if (b_pos >= 0 &&
            (b_pos < low_l || b_pos > high_l) &&
            b_score > score2) {
            score2 = b_score;
            te2    = b_pos;
        }

        aln[ind].score2 = score2;
        aln[ind].te2    = te2;
    }

    return 1;
}

void kswv::kswvBatchWrapper16_avx2(SeqPair *pairArray,
                                   uint8_t *seqBufRef,
                                   uint8_t *seqBufQer,
                                   kswr_t* aln,
                                   int32_t numPairs,
                                   uint16_t numThreads,
                                   int phase)
{
    int16_t *seq1SoA = (int16_t *)_mm_malloc(
        (size_t)this->maxRefLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 128);
    int16_t *seq2SoA = (int16_t *)_mm_malloc(
        (size_t)this->maxQerLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 128);
    assert(seq1SoA != NULL);
    assert(seq2SoA != NULL);

    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH16 - 1) / SIMD_WIDTH16) * SIMD_WIDTH16;
    for (int32_t ii = numPairs; ii < roundNumPairs; ii++) {
        pairArray[ii].regid = ii;
        pairArray[ii].id    = ii;
        pairArray[ii].len1  = 0;
        pairArray[ii].len2  = 0;
        // The kernel's per-lane loop reads idr/idq/h0 for every lane up to
        // SIMD_WIDTH, so it reads these padded lanes too; zero them (as id/len1/
        // len2 above) to keep those reads well-defined. Padded lanes join the SIMD
        // batch (and its cross-lane reductions), but the caller reads results back
        // only for real lanes and whole-aligner output is byte-identical (validated).
        pairArray[ii].idr = 0;
        pairArray[ii].idq = 0;
        pairArray[ii].h0 = 0;
    }

    uint16_t tid = 0;
    int16_t *mySeq1SoA = seq1SoA + tid * this->maxRefLen * SIMD_WIDTH16;
    int16_t *mySeq2SoA = seq2SoA + tid * this->maxQerLen * SIMD_WIDTH16;

    for (int32_t i = 0; i < numPairs; i += SIMD_WIDTH16) {
        int maxLen1 = 0, maxLen2 = 0;

        for (int j = 0; j < SIMD_WIDTH16; j++) {
            SeqPair sp = pairArray[i + j];
            uint8_t *seq1 = seqBufRef + sp.idr;
            for (int k = 0; k < sp.len1; k++)
                mySeq1SoA[k * SIMD_WIDTH16 + j] = (seq1[k] == AMBIG_ ? AMBR16 : seq1[k]);
            if (maxLen1 < sp.len1) maxLen1 = sp.len1;
        }
        for (int j = 0; j < SIMD_WIDTH16; j++) {
            SeqPair sp = pairArray[i + j];
            for (int k = sp.len1; k <= maxLen1; k++)
                mySeq1SoA[k * SIMD_WIDTH16 + j] = 0xFFFF;
        }

        for (int j = 0; j < SIMD_WIDTH16; j++) {
            SeqPair sp = pairArray[i + j];
            uint8_t *seq2 = seqBufQer + sp.idq;
            int quanta = query_quantum16(sp.len2);
            for (int k = 0; k < sp.len2; k++)
                mySeq2SoA[k * SIMD_WIDTH16 + j] = (seq2[k] == AMBIG_ ? AMBQ16 : seq2[k]);
            for (int k = sp.len2; k < quanta; k++)
                mySeq2SoA[k * SIMD_WIDTH16 + j] = DUMMY3;
            if (maxLen2 < quanta) maxLen2 = quanta;
        }
        for (int j = 0; j < SIMD_WIDTH16; j++) {
            SeqPair sp = pairArray[i + j];
            int quanta = query_quantum16(sp.len2);
            for (int k = quanta; k <= maxLen2; k++)
                mySeq2SoA[k * SIMD_WIDTH16 + j] = 0xFFFF;
        }

        kswv256_16(mySeq1SoA, mySeq2SoA,
                   (int16_t)maxLen1, (int16_t)maxLen2,
                   pairArray + i, aln, i, tid,
                   numPairs, phase);
    }

    _mm_free(seq1SoA);
    _mm_free(seq2SoA);
}

void kswv::getScores16(SeqPair *pairArray,
                       uint8_t *seqBufRef,
                       uint8_t *seqBufQer,
                       kswr_t *aln,
                       int32_t numPairs,
                       uint16_t numThreads,
                       int phase)
{
    kswvBatchWrapper16_avx2(pairArray, seqBufRef, seqBufQer, aln,
                            numPairs, numThreads, phase);
}

#elif __AVX512BW__
/* AVX-512 Implementation for x86 */
void kswv::getScores8(SeqPair *pairArray,
                      uint8_t *seqBufRef,
                      uint8_t *seqBufQer,
                      kswr_t* aln,                    
                      int32_t numPairs,
                      uint16_t numThreads,
                      int phase)
{
    kswvBatchWrapper8(pairArray, seqBufRef, seqBufQer, aln,
                      numPairs, numThreads, phase);
}

#define PFD_ 2
void kswv::kswvBatchWrapper8(SeqPair *pairArray,
                             uint8_t *seqBufRef,
                             uint8_t *seqBufQer,
                             kswr_t* aln,
                             int32_t numPairs,
                             uint16_t numThreads,
                             int phase)
{
    int64_t st1, st2, st3, st4, st5;
#if RDT
    st1 = __rdtsc();
#endif
    uint8_t *seq1SoA = NULL;
    seq1SoA = (uint8_t *)_mm_malloc(this->maxRefLen * SIMD_WIDTH8 * numThreads * sizeof(uint8_t), 64);
    
    uint8_t *seq2SoA = NULL;
    seq2SoA = (uint8_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH8 * numThreads * sizeof(uint8_t), 64);
    
    assert(seq1SoA != NULL);
    assert(seq2SoA != NULL);

    int32_t ii;
    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH8 - 1) / SIMD_WIDTH8 ) * SIMD_WIDTH8;
    // assert(roundNumPairs < BATCH_SIZE * SEEDS_PER_READ);
    for(ii = numPairs; ii < roundNumPairs; ii++)
    {
        pairArray[ii].regid = ii;
        pairArray[ii].id = ii;
        pairArray[ii].len1 = 0;
        pairArray[ii].len2 = 0;
        // The kernel's per-lane loop reads idr/idq/h0 for every lane up to
        // SIMD_WIDTH, so it reads these padded lanes too; zero them (as id/len1/
        // len2 above) to keep those reads well-defined. Padded lanes join the SIMD
        // batch (and its cross-lane reductions), but the caller reads results back
        // only for real lanes and whole-aligner output is byte-identical (validated).
        pairArray[ii].idr = 0;
        pairArray[ii].idq = 0;
        pairArray[ii].h0 = 0;
    }

#if RDT
    st2 = __rdtsc();
#endif
    
#if SORT_PAIRS     // disbaled in bwa-mem2 (only used in separate benchmark sw code)
    {
        // Sort the sequences according to decreasing order of lengths
        SeqPair *tempArray = (SeqPair *)_mm_malloc(SORT_BLOCK_SIZE * numThreads *
                                                   sizeof(SeqPair), 64);
        int16_t *hist = (int16_t *)_mm_malloc((this->maxQerLen + 32) * numThreads *
                                              sizeof(int16_t), 64);
        
        #pragma omp parallel num_threads(numThreads)
        {
            int32_t tid = omp_get_thread_num();
            SeqPair *myTempArray = tempArray + tid * SORT_BLOCK_SIZE;
            int16_t *myHist = hist + tid * (this->maxQerLen + 32);
            
            #pragma omp for
            for(ii = 0; ii < roundNumPairs; ii+=SORT_BLOCK_SIZE)
            {
                int32_t first, last;
                first = ii;
                last  = ii + SORT_BLOCK_SIZE;
                if(last > roundNumPairs) last = roundNumPairs;
                sortPairsLen(pairArray + first, last - first, myTempArray, myHist);
            }
        }
        _mm_free(hist);
    }
#endif

#if RDT
    st3 = __rdtsc();
#endif
    
    //#pragma omp parallel num_threads(numThreads)
    {
        int32_t i;
        // uint16_t tid = omp_get_thread_num();
        uint16_t tid = 0;   // forcing single thread
        uint8_t *mySeq1SoA = seq1SoA + tid * this->maxRefLen * SIMD_WIDTH8;
        uint8_t *mySeq2SoA = seq2SoA + tid * this->maxQerLen * SIMD_WIDTH8;
        uint8_t *seq1;
        uint8_t *seq2;
                
        int nstart = 0, nend = numPairs;
        
        // #pragma omp for schedule(dynamic, 128)
        for(i = nstart; i < nend; i+=SIMD_WIDTH8)
        {
            int32_t j, k;
            int maxLen1 = 0;
            int maxLen2 = 0;

            for(j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY               
                seq1 = seqBufRef + (int64_t)sp.id * this->maxRefLen;
#else
                seq1 = seqBufRef + sp.idr;
#endif
                for(k = 0; k < sp.len1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH8 + j] = (seq1[k] == AMBIG_ ? AMBR:seq1[k]);
                }
                if(maxLen1 < sp.len1) maxLen1 = sp.len1;
            }
            for(j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
                for(k = sp.len1; k <= maxLen1; k++) //removed "="
                {
                    mySeq1SoA[k * SIMD_WIDTH8 + j] = 0xFF;
                }
            }
            
            for(j = 0; j < SIMD_WIDTH8; j++)
            {               
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq2 = seqBufQer + (int64_t)sp.id * this->maxQerLen;
#else
                seq2 = seqBufQer + sp.idq;
#endif
                // assert(sp.len2 < this->maxQerLen);
                int quanta = query_quantum8(sp.len2);
                for(k = 0; k < sp.len2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH8 + j] = (seq2[k]==AMBIG_? AMBQ:seq2[k]);
                }

                for(k = sp.len2; k < quanta; k++) {
                    mySeq2SoA[k * SIMD_WIDTH8 + j] = DUMMY5;  // SSE quanta
                }
                if(maxLen2 < (quanta)) maxLen2 = quanta;
            }
            
            for(j = 0; j < SIMD_WIDTH8; j++)
            {
                SeqPair sp = pairArray[i + j];
                int quanta = query_quantum8(sp.len2);
                for(k = quanta; k <= maxLen2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH8 + j] = 0xFF;
                }
            }

            kswv512_u8(mySeq1SoA, mySeq2SoA,
                       maxLen1, maxLen2,
                       pairArray + i,
                       aln, i,
                       tid,
                       numPairs,
                       phase);
        }
    }

#if RDT 
    st4 = __rdtsc();
#endif
    
#if SORT_PAIRS     // disbaled in bwa-mem2 (only used in separate benchmark sw code)
    {
        // Sort the sequences according to increasing order of id
        #pragma omp parallel num_threads(numThreads)
        {
            int32_t tid = omp_get_thread_num();
            SeqPair *myTempArray = tempArray + tid * SORT_BLOCK_SIZE;
            
            #pragma omp for
            for(ii = 0; ii < roundNumPairs; ii+=SORT_BLOCK_SIZE)
            {
                int32_t first, last;
                first = ii;
                last  = ii + SORT_BLOCK_SIZE;
                if(last > roundNumPairs) last = roundNumPairs;
                sortPairsId(pairArray + first, first, last - first, myTempArray);
            }
        }
        _mm_free(tempArray);
    }
#endif

#if RDT
    st5 = __rdtsc();
    setupTicks = st2 - st1;
    sort1Ticks = st3 - st2;
    swTicks = st4 - st3;
    sort2Ticks = st5 - st4;
#endif
    
    // free mem
    _mm_free(seq1SoA);
    _mm_free(seq2SoA);
    
    return;
}

/* Thin dispatcher: see kswv256_u8 / kswv_neon_u8. */
int kswv::kswv512_u8(uint8_t seq1SoA[],
                     uint8_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    return has_freed
        ? kswv512_u8_impl<true>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                po_ind, tid, numPairs, phase)
        : kswv512_u8_impl<false>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                 po_ind, tid, numPairs, phase);
}

template<bool HasFreed>
int kswv::kswv512_u8_impl(uint8_t seq1SoA[],
                     uint8_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    int m_b, n_b;
    uint8_t minsc[SIMD_WIDTH8] __attribute__((aligned(64))) = {0};
    uint8_t endsc[SIMD_WIDTH8] __attribute__((aligned(64))) = {0};
    uint64_t *b;

    __m512i zero512 = _mm512_setzero_si512();
    __m512i one512  = _mm512_set1_epi8(1);

    /* Per-row running-max checkpoints; see kswv_neon_u8_impl. */
    alignas(64) uint8_t blockMax[QE_MAXBLK * SIMD_WIDTH8];
    
    m_b = n_b = 0; b = 0;

    int8_t temp[SIMD_WIDTH8] __attribute((aligned(64))) = {0};

    uint8_t shift = 127, mdiff = 0, qmax_;
    mdiff = max_(this->w_match, (int8_t) this->w_mismatch);
    mdiff = max_(mdiff, (int8_t) this->w_ambig);
    shift = min_(this->w_match, (int8_t) this->w_mismatch);
    shift = min_((int8_t) shift, this->w_ambig);

    qmax_ = mdiff;
    shift = 256 - (uint8_t) shift;
    mdiff += shift;
    
    temp[0] = this->w_match;                                   // states: 1. matches
    temp[1] = temp[2] = temp[3] =  this->w_mismatch;           // 2. mis-matches
    temp[4] = temp[5] = temp[6] = temp[7] =  this->w_ambig;    // 3. beyond boundary
    temp[8] = temp[9] = temp[10] = temp[11] = this->w_ambig;   // 4. 0 - sse2 region
    temp[12] = this->w_ambig;                                  // 5. ambig

    for (int i=0; i<16; i++) // for shuffle_epi8
        temp[i] += shift;

    int pos = 0;
    for (int i=16; i<SIMD_WIDTH8; i++) {
        temp[i] = temp[pos++];
        if (pos % 16 == 0) pos = 0;
    }
    
    __m512i permSft512 = _mm512_load_si512(temp);
    __m512i sft512 = _mm512_set1_epi8(shift);

    /* u8-tier saturation guard (see kswv_u8_saturation_guard). The u8 admission
     * bound keeps every admitted pair safe; this catches a violating param set
     * before it silently corrupts scores. */
    kswv_u8_saturation_guard(p, this->w_match, shift);
    __m512i cmax512 = _mm512_set1_epi8(255);
    
    // __m512i minsc512, endsc512;
    __mmask64 minsc_msk_a = 0x0000, endsc_msk_a = 0x0000;
    int val = 0;
    for (int i=0; i<SIMD_WIDTH8; i++)
    {
        int xtra = p[i].h0;
        val = (xtra & KSW_XSUBO)? xtra & 0xffff : 0x10000;
        if (val <= 255) {
            minsc[i] = val;
            minsc_msk_a |= (0x1L << i);
        }
        // msc_mask;
        val = (xtra & KSW_XSTOP)? xtra & 0xffff : 0x10000;
        if (val <= 255) {
            endsc[i] = val;
            endsc_msk_a |= (0x1L << i);
        }
    }

    __m512i minsc512 = _mm512_load_si512((__m512i*) minsc);
    __m512i endsc512 = _mm512_load_si512((__m512i*) endsc);
       
    __m512i mismatch512 = _mm512_set1_epi8(this->w_mismatch + shift);
    __m512i e_del512    = _mm512_set1_epi8(this->e_del);
    __m512i oe_del512   = _mm512_set1_epi8(this->o_del + this->e_del);
    __m512i e_ins512    = _mm512_set1_epi8(this->e_ins);
    __m512i oe_ins512   = _mm512_set1_epi8(this->o_ins + this->e_ins);
    __m512i five512     = _mm512_set1_epi8(DUMMY5); // ambig mapping element
    __m512i gmax512     = zero512; // exit1 = zero512;
    __m512i te512       = _mm512_set1_epi16(-1);  // changed to -1
    __m512i te512_      = _mm512_set1_epi16(-1);  // changed to -1
    
    __mmask64 exit0 = 0xFFFFFFFFFFFFFFFF;

    tid = 0;  // no threading for now !!
    uint8_t *H0     = H8_0 + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *H1     = H8_1 + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *F      = F8 + tid * SIMD_WIDTH8 * this->maxQerLen;
    uint8_t *rowMax = rowMax8 + tid * SIMD_WIDTH8 * this->maxRefLen;

    // Mirror kswv512_16's per-strip prefetches (see kswv.cpp ~line 2547).
    // The 8-bit kernel was overlooked when those prefetches were added in
    // PR #58; without them the inner loop stalls on L1 misses for F/H1/seq
    // for the first several rows. Identical hint shape; load addresses
    // scaled to SIMD_WIDTH8 (64 bytes per row).
    _mm_prefetch((const char*) (F + SIMD_WIDTH8), _MM_HINT_T0);
    _mm_prefetch((const char*) seq2SoA, _MM_HINT_T1);
    _mm_prefetch((const char*) seq1SoA, _MM_HINT_NTA);
    _mm_prefetch((const char*) (H1 + SIMD_WIDTH8), _MM_HINT_T0);

    for (int i=0; i <=ncol; i++)
    {
        _mm512_store_si512((__m512*) (H0 + i * SIMD_WIDTH8), zero512);
        _mm512_store_si512((__m512*) (F + i * SIMD_WIDTH8), zero512);
    }

    __m512i max512 = zero512, imax512, pimax512 = zero512;
    __mmask64 mask512 = 0x0000;
    __mmask64 minsc_msk = 0x0000;

    __m512i qe512 = _mm512_set1_epi8(0);
    _mm512_store_si512((__m512i *)(H0), zero512);
    _mm512_store_si512((__m512i *)(H1), zero512);

    /* First column any lane pads with the 0xFF query sentinel, i.e.
     * min(query_quantum8(len2)) over the group -- see the derivation in
     * kswv_neon_u8_impl. Below it the boundary mask reduces to the per-row
     * high_bit(s1); at and above it the full per-cell form is required. A
     * uniform-length group gets jsplit == ncol and never runs the per-cell
     * form -- common in phase 0, but see the phase-1 caveat on the NEON
     * derivation: len2 = qe + 1 makes phase-1 groups ragged regardless of input
     * lengths, so they do run it. (No minLen1 range here: unlike NEON's vbsl,
     * the AVX-512 blend is a masked move that costs the same whether or not the
     * mask is empty, so splitting the reference half out as well buys
     * nothing.) */
    int jsplit = ncol;
    for (int l = 0; l < SIMD_WIDTH8; l++) {
        const int quanta = query_quantum8(p[l].len2);
        if (quanta < jsplit) jsplit = quanta;
    }

    /* First column any lane can hold the DUMMY5 query pad, so the cmp+blend
     * for it can be elided below this. Same derivation and same byte-identity
     * argument as the NEON kernel above; clamped to jsplit because it only
     * subdivides the [0, jsplit) range. */
    const int jdummy = compute_jdummy(p, SIMD_WIDTH8, ncol, jsplit);

    int i, limit = nrow;
    for (i=0; i < nrow; i++)
    {
        __m512i e11 = zero512;
        __m512i h00, h11, h10, s1;
        __m512i i512 = _mm512_set1_epi16(i);
        int j ;
        
        s1 = _mm512_load_si512((__m512i *)(seq1SoA + (i + 0) * SIMD_WIDTH8));
        h10 = zero512;
        imax512 = zero512;

        /* Freed-cell override (issue 173, bisulfite OT/OB + TAPS neutral). s1 is
         * loop-invariant across the inner j loop, so hoist the per-row fr_ref
         * comparisons here; MAIN_SAM_CODE8_OPT consumes active_frread512/
         * freedval512 per cell when HasFreed. freedval512 is the freed cell's
         * score (fr_val) biased by +shift into the u8 domain; fr_val==w_match
         * (GENOMIC/COLLAPSED) reproduces the biased match entry temp[0] exactly,
         * while NEUTRAL (fr_val==0) scores the conversion as 0 after unbiasing.
         * When !HasFreed, this and the per-cell block in the macro compile out
         * (identical codegen to today). Mirrors bandedSWA's blessed forms. */
        /* Free a SYMMETRIC PAIR of cells (collapsed bisulfite); GENOMIC/NEUTRAL set
         * the mirror == primary so the second blend is idempotent. Both cells score
         * the same fr_val, so one freedval512 covers the folded target below. */
        __m512i freedval512, active_frread512;
        if (HasFreed) {
            freedval512 = _mm512_set1_epi8((char)(fr_val + shift));
            __mmask64 rowfreed512  = _mm512_cmpeq_epi8_mask(s1, _mm512_set1_epi8((char)fr_ref));
            __mmask64 rowfreed2_512= _mm512_cmpeq_epi8_mask(s1, _mm512_set1_epi8((char)fr_ref2));
            /* One per-lane freed target: fr_read where ref==fr_ref, fr_read2 where
             * ref==fr_ref2, else FREED_INACTIVE8 (no s2 value — 0-3/5/8 or the
             * 0xFF pad — ever equals it). The two ref tests are mutually
             * exclusive; the macro then does a single per-cell cmpeq + blend. */
            active_frread512 = _mm512_mask_blend_epi8(rowfreed2_512,
                                   _mm512_set1_epi8((char)FREED_INACTIVE8), _mm512_set1_epi8((char)fr_read2));
            active_frread512 = _mm512_mask_blend_epi8(rowfreed512,
                                   active_frread512, _mm512_set1_epi8((char)fr_read));
        }

        /* Reference-side half of the boundary mask; loop-invariant across j. */
        const __mmask64 rowboundary512 = _mm512_movepi8_mask(s1);

#define KSWV_AVX512_U8_CELL(BND, NEED_DUMMY)                                        \
        {                                                                           \
            __m512i f11, s2, f21;                                                   \
            h00 = _mm512_load_si512((__m512i *)(H0 + j * SIMD_WIDTH8));             \
            s2  = _mm512_load_si512((__m512i *)(seq2SoA + (j) * SIMD_WIDTH8));      \
            f11 = _mm512_load_si512((__m512i *)(F + (j+1) * SIMD_WIDTH8));          \
                                                                                    \
            MAIN_SAM_CODE8_OPT(s1, s2, h00, h11, e11, f11, f21, max512, sft512,     \
                               (BND), (NEED_DUMMY));                                \
                                                                                    \
            _mm512_store_si512((__m512i *)(H1 + (j + 1) * SIMD_WIDTH8), h11);       \
            _mm512_store_si512((__m512i *)(F + (j + 1)* SIMD_WIDTH8), f21);         \
            if (j == qeNext) {                                                      \
                _mm512_store_si512((__m512i *)(blockMax + qeBlk * SIMD_WIDTH8),      \
                                   imax512);                                        \
                qeBlk++; qeNext += QE_BLK;                                          \
            }                                                                       \
        }

        int qeNext = QE_BLK - 1, qeBlk = 0;
        /* Below jsplit the two forms are equal, so the meth (HasFreed)
         * instantiation keeps the per-cell one: hoisting there pins a third
         * live k-reg against the two freed masks and measured slightly slower.
         * Above jsplit, OR the already-computed s1 half with the s2 half as
         * masks rather than re-ORing the vectors -- a kord instead of a vpord,
         * off the contended vector ports. */
        for (j = 0; j < jdummy; j++)
            KSWV_AVX512_U8_CELL(HasFreed
                ? _mm512_movepi8_mask(_mm512_or_si512(s1, s2))
                : rowboundary512, false)
        for (; j < jsplit; j++)
            KSWV_AVX512_U8_CELL(HasFreed
                ? _mm512_movepi8_mask(_mm512_or_si512(s1, s2))
                : rowboundary512, true)
        for (; j < ncol; j++)
            KSWV_AVX512_U8_CELL(rowboundary512 | _mm512_movepi8_mask(s2), true)
#undef KSWV_AVX512_U8_CELL

        /* Close the final (possibly partial) block; see kswv_neon_u8_impl. */
        _mm512_store_si512((__m512i *)(blockMax + ((ncol - 1) / QE_BLK) * SIMD_WIDTH8),
                           imax512);

        // Block I
        if (i > 0)
        {
            __mmask64 msk64 = _mm512_cmpgt_epu8_mask(imax512, pimax512);
            msk64 |= mask512;
            pimax512 = _mm512_mask_blend_epi8(msk64, pimax512, zero512);
            pimax512 = _mm512_mask_blend_epi8(minsc_msk, zero512, pimax512);
            pimax512 = _mm512_mask_blend_epi8(exit0, zero512, pimax512);
            
            _mm512_store_si512((__m512i *) (rowMax + (i-1)*SIMD_WIDTH8), pimax512);
            mask512 = ~msk64;
        }
        pimax512 = imax512;
        minsc_msk = _mm512_cmpge_epu8_mask(imax512, minsc512);
        minsc_msk &= minsc_msk_a;

        // Block II: gmax, te
        __mmask64 cmp0 = _mm512_cmpgt_epu8_mask(imax512, gmax512);
        cmp0 &= exit0;
        gmax512 = _mm512_mask_blend_epi8(cmp0, gmax512, imax512);
        te512 = _mm512_mask_blend_epi16((__mmask32)cmp0, te512,  i512);
        te512_ = _mm512_mask_blend_epi16((__mmask32) (cmp0 >> SIMD_WIDTH16), te512_,  i512);
        /* Deferred query-end recovery over the QE_BLK checkpoints; see
         * kswv_neon_u8_impl for the derivation. `found` and the per-block
         * masks are k-regs here, so a block that holds no active lane's max
         * costs one compare, one kandn and one kor. */
        if (cmp0)
        {
            __m512i iqe512 = _mm512_set1_epi8(-1);
            __mmask64 foundBlk = 0;
            const uint8_t *colIdx = this->colIdx8;
            const int nblocks = (ncol + QE_BLK - 1) / QE_BLK;
            for (int b = 0; b < nblocks; b++) {
                __mmask64 reached = _mm512_cmpeq_epu8_mask(
                    _mm512_load_si512((__m512i *)(blockMax + b * SIMD_WIDTH8)), imax512);
                __mmask64 newly = reached & ~foundBlk & cmp0;
                foundBlk |= reached;
                if (newly) {
                    const int j0 = b * QE_BLK;
                    const int j1 = (j0 + QE_BLK < ncol) ? (j0 + QE_BLK) : ncol;
                    __mmask64 got = 0;
                    for (int j2 = j0; j2 < j1; j2++) {
                        __mmask64 eq = _mm512_cmpeq_epu8_mask(
                            _mm512_load_si512((__m512i *)(H1 + (j2 + 1) * SIMD_WIDTH8)),
                            imax512) & newly;
                        iqe512 = _mm512_mask_blend_epi8(
                            eq & ~got, iqe512,
                            _mm512_loadu_si512((const __m512i *)(colIdx + j2 * SIMD_WIDTH8)));
                        got |= eq;
                    }
                }
                if ((foundBlk & cmp0) == cmp0) break;
            }
            qe512 = _mm512_mask_blend_epi8(cmp0, qe512, iqe512);
        }
        
        cmp0 = _mm512_cmpge_epu8_mask(gmax512, endsc512);
        cmp0 &= endsc_msk_a;
        
        __m512i left512 = _mm512_adds_epu8(gmax512, sft512);
        __mmask64 cmp2 = _mm512_cmpge_epu8_mask(left512, cmax512);

        exit0 = (~(cmp0 | cmp2)) & exit0;
        if (exit0 == 0)
        {
            limit = i++;
            break;
        }       

        uint8_t *S = H1; H1 = H0; H0 = S;
        i512 = _mm512_add_epi16(i512, one512);
    } // for nrow

    pimax512 = _mm512_mask_blend_epi8(mask512, pimax512, zero512);
    pimax512 = _mm512_mask_blend_epi8(minsc_msk, zero512, pimax512);
    pimax512 = _mm512_mask_blend_epi8(exit0, zero512, pimax512);
    /* Guard on i > 0: all-len1==0 batches give nrow == 0 and would
     * otherwise underflow into the rowMax allocation. issue 38 / PR 289. */
    if (i > 0) {
        _mm512_store_si512((__m512i *) (rowMax + (i-1) * SIMD_WIDTH8), pimax512);
    }

    /******************* DP loop over *****************************/
    /**************** Partial output setting **********************/
    uint8_t score[SIMD_WIDTH8] __attribute((aligned(64)));
    int16_t te1[SIMD_WIDTH8] __attribute((aligned(64)));
    uint8_t qe[SIMD_WIDTH8] __attribute((aligned(64)));
    /* low[]/high[] are declared inside the score2 block below. */

    _mm512_store_si512((__m512i *) score, gmax512);
    _mm512_store_si512((__m512i *) te1, te512);
    _mm512_store_si512((__m512i *) (te1 + SIMD_WIDTH16), te512_);
    _mm512_store_si512((__m512i *) qe, qe512);

    int live = 0;
    for (int l=0; l<SIMD_WIDTH8 && (po_ind + l) < numPairs; l++) {
        int ind = po_ind + l;
        int16_t *te;
        if (i < SIMD_WIDTH16) te = te1;
        else te = te1;
#if !MAINY
        ind = p[l].regid;    // index of corr. aln
        if (phase) {
            if (aln[ind].score == score[l]) {
                aln[ind].tb = aln[ind].te - te[l];
                aln[ind].qb = aln[ind].qe - qe[l];
            }
        } else {
            aln[ind].score = score[l] + shift < 255? score[l] : 255;
            aln[ind].te = te[l];
            aln[ind].qe = qe[l];
            if (aln[ind].score != 255) {
                qe[l] = 1;
                live ++;                
            }
            else qe[l] = 0;
        }
#else
        aln[ind].score = score[l] + shift < 255? score[l] : 255;
        aln[ind].te = te[l];
        aln[ind].qe = qe[l];
        if (aln[ind].score != 255) {
            qe[l] = 1;
            live ++;                
        }
        else qe[l] = 0;     
#endif
    }
    
#if !MAINY
    if (phase) return 1;
#endif

    if (live == 0) return 1;

    /* Score2 and te2 computation.
     *
     * Per-lane scalar emulation of ksw_u8's b[] build + score2 scan
     * (src/ksw.cpp:201). scalar ksw_u8 collapses consecutive rows >=
     * minsc into a single b[] entry anchored at the max-score row; a
     * plateau straddling the primary region [te - val, te + val] is
     * excluded entirely when its anchor lies inside. A dense SIMD scan
     * treats every row independently and pulls out-of-region rows of a
     * boundary-straddling plateau in as false suboptimals, inflating `sub`.
     * Emulate scalar exactly, per lane, on the precomputed
     * rowMax. Cost is O(SIMD_WIDTH8 * limit) scalar ops per batch,
     * negligible next to the SIMD DP that built rowMax. */
    int qmax = this->g_qmax;
    int16_t low[SIMD_WIDTH8]  __attribute__((aligned(64)));
    int16_t high[SIMD_WIDTH8] __attribute__((aligned(64)));
    for (int j = 0; j < SIMD_WIDTH8; j++) {
        int val = (score[j] + qmax - 1) / qmax;
        low[j]  = te1[j] - val;
        high[j] = te1[j] + val;
    }

    /* rowMax entries produced: i == limit on normal completion, and
     * i == limit + 1 after early exit (limit = i++ stores old i). Using i
     * directly gives a single expression that covers both exit paths and
     * includes the terminating row stored at rowMax[limit]. */
    const int processed_rows = i;

    for (int l = 0; l < SIMD_WIDTH8 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        /* Match scalar ksw_u8: when KSW_XSUBO is absent, minsc = 0x10000
         * so the b[] list never starts and score2/te2 stay at -1.
         * minsc_msk_a tracks lanes with KSW_XSUBO set; lanes not in the
         * mask have minsc[l] zero-initialized and must skip the scan. */
        if (!qe[l] || !(minsc_msk_a & ((__mmask64)1 << l))) {
            aln[ind].score2 = -1;
            aln[ind].te2    = -1;
            continue;
        }

        int len1_l  = (int)p[l].len1;
        int low_l   = (int)low[l];
        int high_l  = (int)high[l];
        int minsc_l = (int)minsc[l];
        int score2  = -1;
        int te2     = -1;
        /* b_pos = -2 sentinel: (-2 + 1) != any real row, forces an
         * "append" on the first qualifying row. */
        int b_score = -1;
        int b_pos   = -2;

        int nrows = processed_rows < len1_l ? processed_rows : len1_l;
        for (int i2 = 0; i2 < nrows; i2++) {
            int imax = (int)rowMax[i2 * SIMD_WIDTH8 + l];
            if (imax < minsc_l) continue;

            if (b_pos + 1 != i2) {
                /* APPEND: flush the outgoing b[] entry to score2 first. */
                if (b_pos >= 0 &&
                    (b_pos < low_l || b_pos > high_l) &&
                    b_score > score2) {
                    score2 = b_score;
                    te2    = b_pos;
                }
                b_score = imax;
                b_pos   = i2;
            } else if (b_score < imax) {
                /* UPDATE: strict greater extends the run's anchor. Equal
                 * imax leaves b_pos in place (matches scalar). */
                b_score = imax;
                b_pos   = i2;
            }
        }

        /* Flush trailing b[] entry. */
        if (b_pos >= 0 &&
            (b_pos < low_l || b_pos > high_l) &&
            b_score > score2) {
            score2 = b_score;
            te2    = b_pos;
        }

        aln[ind].score2 = score2;
        aln[ind].te2    = te2;
    }

    return 1;   
}

/*********************************** Vectorized Code 16 bit *****************************/
/// 16 bit lanes
void kswv::getScores16(SeqPair *pairArray,
                       uint8_t *seqBufRef,
                       uint8_t *seqBufQer,
                       kswr_t* aln,
                       int32_t numPairs,
                       uint16_t numThreads,
                       int phase)
{
    kswvBatchWrapper16(pairArray, seqBufRef, seqBufQer, aln,
                       numPairs, numThreads, phase);
}

void kswv::kswvBatchWrapper16(SeqPair *pairArray,
                              uint8_t *seqBufRef,
                              uint8_t *seqBufQer,
                              kswr_t* aln,
                              int32_t numPairs,                           
                              uint16_t numThreads,
                              int phase)
{
    int64_t st1, st2, st3, st4, st5;
#if RDT
    st1 = __rdtsc();
#endif
    
    int16_t *seq1SoA = NULL;
    seq1SoA = (int16_t *)_mm_malloc(this->maxRefLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 64);
    
    int16_t *seq2SoA = NULL;
    seq2SoA = (int16_t *)_mm_malloc(this->maxQerLen * SIMD_WIDTH16 * numThreads * sizeof(int16_t), 64);

    assert(seq1SoA != NULL);
    assert(seq2SoA != NULL);    
    
    int32_t ii;
    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH16 - 1) / SIMD_WIDTH16 ) * SIMD_WIDTH16;
    // assert(roundNumPairs < BATCH_SIZE * SEEDS_PER_READ);
    for(ii = numPairs; ii < roundNumPairs; ii++)
    {
        pairArray[ii].regid = ii;
        pairArray[ii].id = ii;
        pairArray[ii].len1 = 0;
        pairArray[ii].len2 = 0;
        // The kernel's per-lane loop reads idr/idq/h0 for every lane up to
        // SIMD_WIDTH, so it reads these padded lanes too; zero them (as id/len1/
        // len2 above) to keep those reads well-defined. Padded lanes join the SIMD
        // batch (and its cross-lane reductions), but the caller reads results back
        // only for real lanes and whole-aligner output is byte-identical (validated).
        pairArray[ii].idr = 0;
        pairArray[ii].idq = 0;
        pairArray[ii].h0 = 0;
    }

#if RDT
    st2 = __rdtsc();
#endif
    
#if SORT_PAIRS     // disbaled in bwa-mem2 (only used in separate benchmark sw code)
    {
        // Sort the sequences according to decreasing order of lengths
        SeqPair *tempArray = (SeqPair *)_mm_malloc(SORT_BLOCK_SIZE * numThreads *
                                                   sizeof(SeqPair), 64);
        int16_t *hist = (int16_t *)_mm_malloc((this->maxQerLen + 32) * numThreads *
                                              sizeof(int16_t), 64);
        
        #pragma omp parallel num_threads(numThreads)
        {
            int32_t tid = omp_get_thread_num();
            SeqPair *myTempArray = tempArray + tid * SORT_BLOCK_SIZE;
            int16_t *myHist = hist + tid * (this->maxQerLen + 32);
            
            #pragma omp for
            for(ii = 0; ii < roundNumPairs; ii+=SORT_BLOCK_SIZE)
            {
                int32_t first, last;
                first = ii;
                last  = ii + SORT_BLOCK_SIZE;
                if(last > roundNumPairs) last = roundNumPairs;
                sortPairsLen(pairArray + first, last - first, myTempArray, myHist);
            }
        }
        _mm_free(hist);
    }
#endif

#if RDT
    st3 = __rdtsc();
#endif
    
// #pragma omp parallel num_threads(numThreads)
    {
        int32_t i;
        // uint16_t tid = omp_get_thread_num();
        uint16_t tid = 0;  // no threading here.
        int16_t *mySeq1SoA = NULL;
        mySeq1SoA = seq1SoA + tid * this->maxRefLen * SIMD_WIDTH16;
        assert(mySeq1SoA != NULL);
            
        int16_t *mySeq2SoA = NULL;
        mySeq2SoA = seq2SoA + tid * this->maxQerLen * SIMD_WIDTH16;
        assert(mySeq1SoA != NULL);
        
        uint8_t *seq1;
        uint8_t *seq2;
        
        int nstart = 0, nend = numPairs;


//#pragma omp for schedule(dynamic, 128)
        for(i = nstart; i < nend; i+=SIMD_WIDTH16)
        {
            int32_t j, k;
            int maxLen1 = 0;
            int maxLen2 = 0;

            for(j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq1 = seqBufRef + (int64_t)sp.id * this->maxRefLen;
#else
                seq1 = seqBufRef + sp.idr;
#endif
                assert(sp.len1 < this->maxRefLen);
                for(k = 0; k < sp.len1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH16 + j] = (seq1[k] == AMBIG_ ? AMBR16:seq1[k]);
                }
                if(maxLen1 < sp.len1) maxLen1 = sp.len1;
            }
            for(j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
                for(k = sp.len1; k <= maxLen1; k++) //removed "="
                {
                    // mySeq1SoA[k * SIMD_WIDTH16 + j] = DUMMY1_;
                    mySeq1SoA[k * SIMD_WIDTH16 + j] = 0xFFFF;
                }
            }

            for(j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
#if MAINY
                seq2 = seqBufQer + (int64_t)sp.id * this->maxQerLen;
#else
                seq2 = seqBufQer + sp.idq;
#endif
                assert(sp.len2 <= this->maxQerLen);


                // int quanta = 8 - sp.len2 % 8;  // based on SSE-16 bit lane
                int quanta = query_quantum16(sp.len2);
                assert(sp.len2 < this->maxQerLen);
                for(k = 0; k < sp.len2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH16 + j] = (seq2[k]==AMBIG_? AMBQ16:seq2[k]);
                }
                
                assert(quanta < this->maxQerLen); 
                for(k = sp.len2; k < quanta; k++) {
                    mySeq2SoA[k * SIMD_WIDTH16 + j] = DUMMY3;
                }
                if(maxLen2 < (quanta)) maxLen2 = quanta;
            }
            
            for(j = 0; j < SIMD_WIDTH16; j++)
            {
                SeqPair sp = pairArray[i + j];
                int quanta = query_quantum16(sp.len2);
                for(k = quanta; k <= maxLen2; k++)
                {
                    // mySeq2SoA[k * SIMD_WIDTH16 + j] = DUMMY2_;
                    mySeq2SoA[k * SIMD_WIDTH16 + j] = 0xFFFF;
                }
            }

            kswv512_16(mySeq1SoA, mySeq2SoA,
                       maxLen1, maxLen2,
                       pairArray + i,
                       aln, i,
                       tid,
                       numPairs,
                       phase);
        }
    }

#if RDT
    st4 = __rdtsc();
#endif
    
#if SORT_PAIRS     // disbaled in bwa-mem2 (only used in separate benchmark sw code)
    {
        // Sort the sequences according to increasing order of id
        #pragma omp parallel num_threads(numThreads)
        {
            int32_t tid = omp_get_thread_num();
            SeqPair *myTempArray = tempArray + tid * SORT_BLOCK_SIZE;
            
            #pragma omp for
            for(ii = 0; ii < roundNumPairs; ii+=SORT_BLOCK_SIZE)
            {
                int32_t first, last;
                first = ii;
                last  = ii + SORT_BLOCK_SIZE;
                if(last > roundNumPairs) last = roundNumPairs;
                sortPairsId(pairArray + first, first, last - first, myTempArray);
            }
        }
        _mm_free(tempArray);
    }
#endif

#if RDT
    st5 = __rdtsc();
    setupTicks = st2 - st1;
    sort1Ticks = st3 - st2;
    swTicks = st4 - st3;
    sort2Ticks = st5 - st4;
#endif
    // free mem
    _mm_free(seq1SoA);
    _mm_free(seq2SoA);
    return; 
}

/* Thin dispatcher: see kswv512_u8 above. */
int kswv::kswv512_16(int16_t seq1SoA[],
                     int16_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    return has_freed
        ? kswv512_16_impl<true>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                po_ind, tid, numPairs, phase)
        : kswv512_16_impl<false>(seq1SoA, seq2SoA, nrow, ncol, p, aln,
                                 po_ind, tid, numPairs, phase);
}

template<bool HasFreed>
int kswv::kswv512_16_impl(int16_t seq1SoA[],
                     int16_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t *aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase)
{
    int m_b, n_b;
    int16_t minsc[SIMD_WIDTH16] = {0}, endsc[SIMD_WIDTH16] = {0};
    uint64_t *b;
    int limit = nrow;
    
    __m512i zero512 = _mm512_setzero_si512();
    __m512i one512  = _mm512_set1_epi16(1);
    __m512i minus1  = _mm512_set1_epi16(-1);

    int16_t temp[SIMD_WIDTH16] __attribute((aligned(64))) = {0};
    int16_t temp1[SIMD_WIDTH16] __attribute((aligned(64)));
    int16_t temp2[SIMD_WIDTH16] __attribute((aligned(64)));

    // query profile, we use xor operations on the strings
    // temp[0] = this->w_match;
    // temp[1] = temp[2] = temp[3] =  this->w_mismatch;
    // temp[4] = temp[5] = temp[6] = temp[7] =  this->w_mismatch;
    // temp[16] = temp[17] = temp[18] = temp[19] = this->w_ambig;
    // temp[31] =  this->w_ambig;
    // temp[12] = temp[13] = temp[14] = temp[15] =  this->w_ambig;
    // temp[10] = temp[20] = temp[30] = this->w_mismatch;
    
    temp[0] = this->w_match;    // matching
    temp[1]  = temp[2]  = temp[3]  =  this->w_mismatch;  // mis-matching    
    temp[12] = temp[13] = temp[14] = temp[15] =  this->w_ambig;
    temp[16] = temp[17] = temp[18] = temp[19] = this->w_ambig;
    temp[31] = this->w_ambig;
       
    __m512i perm512 = _mm512_load_si512(temp);
        
    m_b = n_b = 0; b = 0;

    __mmask32 minsc_msk_a = 0x0000, endsc_msk_a = 0x0000;
    int val = 0;
    for (int i=0; i<SIMD_WIDTH16; i++) {
        int xtra = p[i].h0;
        val = (xtra & KSW_XSUBO)? xtra & 0xffff : 0x10000;
        if (val <= SHRT_MAX) {
            minsc[i] = val;
            //if (val < SHRT_MIN)
            //  minsc[i] = -1;
            minsc_msk_a |= (0x1 << i);
        }
        // msc_mask;
        val = (xtra & KSW_XSTOP)? xtra & 0xffff : 0x10000;
        if (val <= SHRT_MAX) {
            endsc[i] = val;
            //if (val < SHRT_MIN)
            //  endsc[i] = -1;
            endsc_msk_a |= (0x1 << i);
        }
    }

    __m512i minsc512 = _mm512_load_si512((__m512i*) minsc);
    __m512i endsc512 = _mm512_load_si512((__m512i*) endsc);
    
    __m512i e_del512    = _mm512_set1_epi16(this->e_del);
    __m512i oe_del512   = _mm512_set1_epi16(this->o_del + this->e_del);
    __m512i e_ins512    = _mm512_set1_epi16(this->e_ins);
    __m512i oe_ins512   = _mm512_set1_epi16(this->o_ins + this->e_ins);
    __m512i gmax512     = zero512; // exit1 = zero512;
    // __m512i te512       = zero512;  // change to -1
    __m512i te512       = _mm512_set1_epi16(-1);
    __mmask32 exit0 = 0xFFFFFFFF;

    
    tid = 0;  // no threading here.
    int16_t *H0     = H16_0 + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *H1     = H16_1 + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *F      = F16 + tid * SIMD_WIDTH16 * this->maxQerLen;
    int16_t *rowMax = rowMax16 + tid * SIMD_WIDTH16 * this->maxRefLen;
    
    _mm_prefetch((const char*) (F + SIMD_WIDTH16), _MM_HINT_T0);
    _mm_prefetch((const char*) seq2SoA, _MM_HINT_T1);
    _mm_prefetch((const char*) seq1SoA, _MM_HINT_NTA);
    _mm_prefetch((const char*) (H1 + SIMD_WIDTH16), _MM_HINT_T0);

    for (int i=ncol; i >= 0; i--) {
        _mm512_store_si512((__m512*) (H0 + i * SIMD_WIDTH16), zero512);
        _mm512_store_si512((__m512*) (F + i * SIMD_WIDTH16), zero512);
    }

    __m512i max512 = zero512, imax512, pimax512 = zero512;
    __mmask32 mask512 = 0x0000;
    __mmask32 minsc_msk = 0x0000;

    __m512i qe512 = _mm512_set1_epi16(0);
    _mm512_store_si512((__m512i *)(H0), zero512);
    _mm512_store_si512((__m512i *)(H1), zero512);
    __m512i i512 = zero512;
    
    int i;
    for (i=0; i < nrow; i++)
    {
        __m512i e11 = zero512;
        __m512i h00, h11, h10, s1;
        int j;
        
        s1 = _mm512_load_si512((__m512i *)(seq1SoA + (i + 0) * SIMD_WIDTH16));
        h10 = zero512;
        imax512 = zero512;
        __m512i iqe512 = _mm512_set1_epi16(-1);

        /* Rank-1 freed-cell override (issue 173 + TAPS neutral), folded into ONE
         * per-row target base active_frread512 — the u16 analog of
         * kswv512_u8_impl's active_frread (consumed by MAIN_SAM_CODE16_OPT).
         * fr_read where ref==fr_ref, fr_read2 where ref==fr_ref2 (mutually
         * exclusive), else a sentinel no real s2 equals: u16 s2 ∈ {0-3, AMBQ16=16,
         * DUMMY3=26, 0xFFFF pad}, so FREED_INACTIVE16 is safe. The 16-bit kernel has NO shift
         * bias (m11 = h00 + sbt directly), so the freed value is just fr_val:
         * fr_val == w_match (GENOMIC/COLLAPSED) equals temp[0], while NEUTRAL
         * (fr_val==0) scores the conversion as 0. !HasFreed → this and the per-cell
         * block in the macro compile out. */
        __m512i freedval512, active_frread512;
        if (HasFreed) {
            freedval512 = _mm512_set1_epi16((int16_t)fr_val);
            __mmask32 rowfreed512  = _mm512_cmpeq_epi16_mask(s1, _mm512_set1_epi16((int16_t)fr_ref));
            __mmask32 rowfreed2_512= _mm512_cmpeq_epi16_mask(s1, _mm512_set1_epi16((int16_t)fr_ref2));
            active_frread512 = _mm512_mask_blend_epi16(rowfreed2_512,
                                   _mm512_set1_epi16((int16_t)FREED_INACTIVE16), _mm512_set1_epi16((int16_t)fr_read2));
            active_frread512 = _mm512_mask_blend_epi16(rowfreed512,
                                   active_frread512, _mm512_set1_epi16((int16_t)fr_read));
        }

        __m512i l512 = zero512;
        for (j=0; j<ncol; j++)
        {
            __m512i f11, s2, f21;
            h00 = _mm512_load_si512((__m512i *)(H0 + j * SIMD_WIDTH16));
            s2  = _mm512_load_si512((__m512i *)(seq2SoA + (j) * SIMD_WIDTH16));
            f11 = _mm512_load_si512((__m512i *)(F + (j+1) * SIMD_WIDTH16));

            MAIN_SAM_CODE16_OPT(s1, s2, h00, h11, e11, f11, f21, max512);

            _mm512_store_si512((__m512i *)(H1 + (j+1) * SIMD_WIDTH16), h11);
            _mm512_store_si512((__m512i *)(F + (j+1) * SIMD_WIDTH16), f21);
            l512 = _mm512_add_epi16(l512, one512);
            // prof[DP2][0] += 22;
            
        }   /* Inner DP loop */
        
        // Block I
        if (i > 0) {
            __mmask32 msk32 = _mm512_cmpgt_epi16_mask(imax512, pimax512);
            msk32 |= mask512;
            // msk32 &= ~minsc_msk;
            pimax512 = _mm512_mask_blend_epi16(msk32, pimax512, minus1);            
            pimax512 = _mm512_mask_blend_epi16(minsc_msk, minus1, pimax512);
            //pimax512 = _mm512_mask_mov_epi16(minus1, minsc_msk, pimax512);

            // new
            // pimax512 = _mm512_mask_blend_epi16(exit0, zero512, pimax512);
            pimax512 = _mm512_mask_blend_epi16(exit0, minus1, pimax512);
            
            _mm512_store_si512((__m512i *) (rowMax + (i-1)*SIMD_WIDTH16), pimax512);
            mask512 = ~msk32;
        }
        pimax512 = imax512;
        minsc_msk = _mm512_cmpge_epi16_mask(imax512, minsc512);
        minsc_msk &= minsc_msk_a;
        
        // Block II: gmax, te
        __mmask32 cmp0 = _mm512_cmpgt_epi16_mask(imax512, gmax512);
        cmp0 &= exit0;
        // gmax512 = _mm512_max_epi16(gmax512, imax512);
        gmax512 = _mm512_mask_blend_epi16(cmp0, gmax512, imax512);
        // te512 = _mm512_mask_mov_epi16(te512, cmp0, i512);
        te512 = _mm512_mask_blend_epi16(cmp0, te512,  i512);
        qe512 = _mm512_mask_blend_epi16(cmp0, qe512,  iqe512);
        
        cmp0 = _mm512_cmpge_epi16_mask(gmax512, endsc512);
        cmp0 &= endsc_msk_a;
        
        exit0 = (~cmp0) & exit0;
        if (exit0 == 0) {
            limit = i++;
            break;
        }
        
        int16_t *S = H1; H1 = H0; H0 = S;
        i512 = _mm512_add_epi16(i512, one512);
    } // for nrow
    
    pimax512 = _mm512_mask_blend_epi16(mask512, pimax512, minus1);
    pimax512 = _mm512_mask_blend_epi16(minsc_msk, minus1, pimax512);
    pimax512 = _mm512_mask_blend_epi16(exit0, minus1, pimax512);
    /* Guard on i > 0: all-len1==0 batches give nrow == 0 and would
     * otherwise underflow into the rowMax allocation. issue 38 / PR 289. */
    if (i > 0) {
        _mm512_store_si512((__m512i *) (rowMax + (i-1) * SIMD_WIDTH16), pimax512);
    }
    // __m512i max512_ = max512;

    /******************* DP loop over *****************************/
    /*************** Partial output setting ***************/
    int16_t score[SIMD_WIDTH16] __attribute((aligned(64)));
    int16_t te[SIMD_WIDTH16] __attribute((aligned(64)));
    int16_t qe[SIMD_WIDTH16] __attribute((aligned(64)));
    int16_t low[SIMD_WIDTH16] __attribute((aligned(64)));
    int16_t high[SIMD_WIDTH16] __attribute((aligned(64)));  
    _mm512_store_si512((__m512i *) score, gmax512); 
    _mm512_store_si512((__m512i *) te, te512);
    _mm512_store_si512((__m512i *) qe, qe512);

    for (int l=0; l<SIMD_WIDTH16 && (po_ind + l) < numPairs; l++) {
        int ind = po_ind + l;
#if !MAINY
        ind = p[l].regid;    // index of corr. aln
        if (phase) {
            if (aln[ind].score == score[l]) {
                aln[ind].tb = aln[ind].te - te[l];
                aln[ind].qb = aln[ind].qe - qe[l];
            }
        } else {
            aln[ind].score = score[l];
            aln[ind].te = te[l];
            aln[ind].qe = qe[l];
        }
#else
        aln[ind].score = score[l];
        aln[ind].te = te[l];
        aln[ind].qe = qe[l];
#endif
    }
    
#if !MAINY
    if (phase) return 1;
#endif

    /*************** Score2 and te2 — per-lane scalar emulation *******************
     *
     * Same b[]-consolidation fix as the 8-bit kernels (see kswv256_u8 for
     * full rationale). Additionally fixes three pre-existing bugs the
     * pre-fix 16-bit kernel carried but 8-bit did not:
     *
     *   (a) aggregate maxl/minh bounds — the same fix-3 issue PR #21
     *       addressed on the AVX-512BW 8-bit kernel; this 16-bit one
     *       never got it. A lane with a tight primary region would see
     *       rows inside its own [low, high] count as score2 candidates
     *       because the outer loop used the batch-wide max(low) /
     *       min(high) instead of per-lane bounds.
     *   (b) no minsc filter — scalar ksw_i16 records b[] entries only
     *       when imax >= minsc; pre-fix 16-bit had no such gate,
     *       leaking sub-minsc plateau scores into max2.
     *   (c) no qe mask — lanes without a valid primary (qe == 0) could
     *       contribute spurious rmax values since their rowMax was
     *       never zeroed.
     *
     * All three fall out of the per-lane scalar loop naturally. */
    int qmax = this->g_qmax;
    for (int j = 0; j < SIMD_WIDTH16; j++) {
        int val = (score[j] + qmax - 1) / qmax;
        low[j]  = te[j] - val;
        high[j] = te[j] + val;
    }

    /* rowMax entries produced: i == limit on normal completion, and
     * i == limit + 1 after early exit (limit = i++ stores old i). Using i
     * directly gives a single expression that covers both exit paths and
     * includes the terminating row stored at rowMax[limit]. */
    const int processed_rows = i;

    for (int l = 0; l < SIMD_WIDTH16 && (po_ind + l) < numPairs; l++) {
        int ind = p[l].regid;
        /* Match scalar ksw_i16: when KSW_XSUBO is absent, minsc = 0x10000
         * (> SHRT_MAX) so b[] never starts and score2/te2 stay at -1.
         * minsc_msk_a tracks lanes with KSW_XSUBO set; lanes not in the
         * mask have minsc[l] zero-initialized and must skip the scan.
         *
         * Inactive-lane check is te[l] < 0, not !qe[l]: unlike the 8-bit
         * kernels, 16-bit qe[l] is the real query-end coordinate and
         * qe == 0 is a valid endpoint (best cell at query column 0).
         * te[l] is initialized to -1 and only overwritten when a new
         * gmax is recorded, so te[l] < 0 cleanly marks lanes with no
         * primary alignment. */
        if (te[l] < 0 || !(minsc_msk_a & ((__mmask32)1 << l))) {
            aln[ind].score2 = -1;
            aln[ind].te2    = -1;
            continue;
        }

        int len1_l  = (int)p[l].len1;
        int low_l   = (int)low[l];
        int high_l  = (int)high[l];
        int minsc_l = (int)minsc[l];
        int score2  = -1;
        int te2     = -1;
        /* b_pos = -2 sentinel: (-2 + 1) != any real row, forces "append"
         * on the first qualifying row. */
        int b_score = -1;
        int b_pos   = -2;

        int nrows = processed_rows < len1_l ? processed_rows : len1_l;
        for (int i2 = 0; i2 < nrows; i2++) {
            int imax = (int)rowMax[i2 * SIMD_WIDTH16 + l];
            if (imax < minsc_l) continue;

            if (b_pos + 1 != i2) {
                /* APPEND: flush outgoing b[] entry to score2 first. */
                if (b_pos >= 0 &&
                    (b_pos < low_l || b_pos > high_l) &&
                    b_score > score2) {
                    score2 = b_score;
                    te2    = b_pos;
                }
                b_score = imax;
                b_pos   = i2;
            } else if (b_score < imax) {
                /* UPDATE: strict greater extends the run's anchor. */
                b_score = imax;
                b_pos   = i2;
            }
        }

        /* Flush trailing b[] entry. */
        if (b_pos >= 0 &&
            (b_pos < low_l || b_pos > high_l) &&
            b_score > score2) {
            score2 = b_score;
            te2    = b_pos;
        }

        aln[ind].score2 = score2;
        aln[ind].te2    = te2;
    }

    return 1;
}

#elif ((!__AVX512BW__) && (!__AVX2__) && (defined(__SSE2__) || defined(__SSSE3__) || defined(__SSE4_1__)))
/* SSE-only fallback stubs (sse41/sse42/avx tiers).
 *
 * The batched kswv kernel requires AVX2 or AVX-512BW. These methods are
 * unreachable in the shipping single-binary build for two independent reasons:
 *   - Non-kernel TUs (incl. src/bwamem_pair.cpp, where the only call site
 *     mem_sam_pe_batch() lives) compile at BASELINE_ARCH, which defaults to
 *     avx2 (Makefile). A binary built with -mavx2 cannot run on a sub-AVX2
 *     CPU, so the runtime dispatcher never selects an SSE-tier kswv there.
 *   - If BASELINE_ARCH is overridden to sse41, then __AVX2__ is undefined in
 *     those TUs and BWAMEM_BATCHED_MATESW=0 (see macro.h:79-88), so the entire
 *     batched mate-rescue path is excluded at compile time and mate rescue runs
 *     through the scalar ksw_align2 path instead.
 *
 * The stubs exist only so the Ikswv vtable resolves at link time on every
 * x86 tier (sse41/sse42/avx all need getScores8/getScores16 bodies even
 * though the dispatcher never calls them). As a second line of defense the
 * mat-aware ctor (above) reports needsScalar() == true on these freed-cell-less
 * tiers, so make_kswv()'s caller routes any asymmetric (meth) pair to the
 * scalar fallback before it could ever reach these stubs. The exit() here is
 * the last-resort guard if a future change opens a runtime call site anyway.
 */
void kswv::getScores8(SeqPair * /*pairArray*/,
                      uint8_t * /*seqBufRef*/,
                      uint8_t * /*seqBufQer*/,
                      kswr_t*   /*aln*/,
                      int32_t   /*numPairs*/,
                      uint16_t  /*numThreads*/,
                      int       /*phase*/)
{
    fprintf(stderr,
            "[E::%s] batched kswv kernel not available at this SIMD tier (SSE only). "
            "Recompile with AVX2+ or use the scalar mate-SW path.\n",
            __func__);
    exit(EXIT_FAILURE);
}

void kswv::getScores16(SeqPair * /*pairArray*/,
                       uint8_t * /*seqBufRef*/,
                       uint8_t * /*seqBufQer*/,
                       kswr_t*   /*aln*/,
                       int32_t   /*numPairs*/,
                       uint16_t  /*numThreads*/,
                       int       /*phase*/)
{
    fprintf(stderr,
            "[E::%s] batched kswv kernel not available at this SIMD tier (SSE only). "
            "Recompile with AVX2+ or use the scalar mate-SW path.\n",
            __func__);
    exit(EXIT_FAILURE);
}

#endif // AVX512BW / SSE-only fallback



/**************************************Scalar code***************************************/
/* This is the original SW code from bwa-mem. We are keeping both, 8-bit and 16-bit 
   implementations, here for benchmarking purpose. 
   The interface call to the code is very simple and similar to the one we used above.
   By default the code is disabled.
 */

#if ORIGINAL_SW_SAM
void kswv::bwa_fill_scmat(int8_t mat[25]) {
    int a = this->w_match;
    int b = this->w_mismatch;
    int ambig = this->w_ambig;
    
    int i, j, k;
    for (i = k = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j)
            mat[k++] = i == j? a : b;
        mat[k++] = ambig; // ambiguous base
    }
    for (j = 0; j < 5; ++j) mat[k++] = ambig;
}

/**
 * Initialize the query data structure
 *
 * @param size   Number of bytes used to store a score; valid valures are 1 or 2
 * @param qlen   Length of the query sequence
 * @param query  Query sequence
 * @param m      Size of the alphabet
 * @param mat    Scoring matrix in a one-dimension array
 *
 * @return       Query data structure
 */
kswq_t* kswv::ksw_qinit(int size, int qlen, uint8_t *query, int m, const int8_t *mat)
{
    kswq_t *q;
    int slen, a, tmp, p;

    size = size > 1? 2 : 1;
    p = 8 * (3 - size); // # values per __m128i
    slen = (qlen + p - 1) / p; // segmented length
    q = (kswq_t*)malloc(sizeof(kswq_t) + 256 + 16 * slen * (m + 4)); // a single block of memory
    q->qp = (__m128i*)(((size_t)q + sizeof(kswq_t) + 15) >> 4 << 4); // align memory
    q->H0 = q->qp + slen * m;
    q->H1 = q->H0 + slen;
    q->E  = q->H1 + slen;
    q->Hmax = q->E + slen;
    q->slen = slen; q->qlen = qlen; q->size = size;
    // compute shift
    tmp = m * m;
    for (a = 0, q->shift = 127, q->mdiff = 0; a < tmp; ++a) { // find the minimum and maximum score
        if (mat[a] < (int8_t)q->shift) q->shift = mat[a];
        if (mat[a] > (int8_t)q->mdiff) q->mdiff = mat[a];
    }
    
    q->max = q->mdiff;
    q->shift = 256 - q->shift; // NB: q->shift is uint8_t
    q->mdiff += q->shift; // this is the difference between the min and max scores

    // An example: p=8, qlen=19, slen=3 and segmentation:
    //  {{0,3,6,9,12,15,18,-1},{1,4,7,10,13,16,-1,-1},{2,5,8,11,14,17,-1,-1}}
    if (size == 1) {
        int8_t *t = (int8_t*)q->qp;
        for (a = 0; a < m; ++a) {
            int i, k, nlen = slen * p;
            const int8_t *ma = mat + a * m;
            for (i = 0; i < slen; ++i)
                for (k = i; k < nlen; k += slen) // p iterations
                    *t++ = (k >= qlen? 0 : ma[query[k]]) + q->shift;
        }
    } else {
        int16_t *t = (int16_t*)q->qp;
        for (a = 0; a < m; ++a) {
            int i, k, nlen = slen * p;
            const int8_t *ma = mat + a * m;
            for (i = 0; i < slen; ++i)
                for (k = i; k < nlen; k += slen) // p iterations
                    *t++ = (k >= qlen? 0 : ma[query[k]]);
        }
    }
    return q;
}

kswr_t kswv::kswvScalar_u8(kswq_t *q, int tlen, const uint8_t *target,
                          int _o_del, int _e_del, int _o_ins, int _e_ins,
                          int xtra) // the first gap costs -(_o+_e)
{
    int slen, i, m_b, n_b, te = -1, gmax = 0, minsc, endsc;
    uint64_t *b;
    __m128i zero, oe_del, e_del, oe_ins, e_ins, shift, *H0, *H1, *E, *Hmax;
    kswr_t r;

#define __max_16(ret, xx) do { \
        (xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 8)); \
        (xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 4)); \
        (xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 2)); \
        (xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 1)); \
        (ret) = _mm_extract_epi16((xx), 0) & 0x00ff;        \
    } while (0)
    
    // initialization
    r = g_defr;
    minsc = (xtra & KSW_XSUBO)? xtra & 0xffff : 0x10000;
    endsc = (xtra & KSW_XSTOP)? xtra & 0xffff : 0x10000;
    m_b = n_b = 0; b = 0;   zero = _mm_set1_epi32(0);
    oe_del = _mm_set1_epi8(_o_del + _e_del);
    e_del = _mm_set1_epi8(_e_del);
    oe_ins = _mm_set1_epi8(_o_ins + _e_ins);
    e_ins = _mm_set1_epi8(_e_ins);
    shift = _mm_set1_epi8(q->shift);
    H0 = q->H0; H1 = q->H1; E = q->E; Hmax = q->Hmax;
    slen = q->slen;
    for (i = 0; i < slen; ++i) {
        _mm_store_si128(E + i, zero);
        _mm_store_si128(H0 + i, zero);
        _mm_store_si128(Hmax + i, zero);
    }
    // the core loop
    for (i = 0; i < tlen; ++i) {
        int j, k, cmp, imax;
        __m128i e, h, t, f = zero, max = zero, *S = q->qp + target[i] * slen; // s is the 1st score vector
        h = _mm_load_si128(H0 + slen - 1); // h={2,5,8,11,14,17,-1,-1} in the above example
        h = _mm_slli_si128(h, 1); // h=H(i-1,-1); << instead of >> because x64 is little-endian
        for (j = 0; LIKELY(j < slen); ++j) {
            /* SW cells are computed in the following order:
             *   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
             *   E(i+1,j) = max{H(i,j)-q, E(i,j)-r}
             *   F(i,j+1) = max{H(i,j)-q, F(i,j)-r}
             */
            // compute H'(i,j); note that at the beginning, h=H'(i-1,j-1)
            h = _mm_adds_epu8(h, _mm_load_si128(S + j));
            h = _mm_subs_epu8(h, shift); // h=H'(i-1,j-1)+S(i,j)
            e = _mm_load_si128(E + j); // e=E'(i,j)
            h = _mm_max_epu8(h, e);
            h = _mm_max_epu8(h, f); // h=H'(i,j)
            max = _mm_max_epu8(max, h); // set max
            _mm_store_si128(H1 + j, h); // save to H'(i,j)
            // now compute E'(i+1,j)
            e = _mm_subs_epu8(e, e_del); // e=E'(i,j) - e_del
            t = _mm_subs_epu8(h, oe_del); // h=H'(i,j) - o_del - e_del
            e = _mm_max_epu8(e, t); // e=E'(i+1,j)
            _mm_store_si128(E + j, e); // save to E'(i+1,j)
            // now compute F'(i,j+1)
            f = _mm_subs_epu8(f, e_ins);
            t = _mm_subs_epu8(h, oe_ins); // h=H'(i,j) - o_ins - e_ins
            f = _mm_max_epu8(f, t);
            // get H'(i-1,j) and prepare for the next j
            h = _mm_load_si128(H0 + j); // h=H'(i-1,j)
        }
        // NB: we do not need to set E(i,j) as we disallow adjecent insertion and then deletion
        for (k = 0; LIKELY(k < 16); ++k) { // this block mimics SWPS3; NB: H(i,j) updated in the lazy-F loop cannot exceed max
            f = _mm_slli_si128(f, 1);
            for (j = 0; LIKELY(j < slen); ++j) {
                h = _mm_load_si128(H1 + j);
                h = _mm_max_epu8(h, f); // h=H'(i,j)
                _mm_store_si128(H1 + j, h);
                h = _mm_subs_epu8(h, oe_ins);
                f = _mm_subs_epu8(f, e_ins);
                cmp = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_subs_epu8(f, h), zero));
                if (UNLIKELY(cmp == 0xffff)) goto end_loop16;
            }
        }
end_loop16:
        //int k;for (k=0;k<16;++k)printf("%d ", ((uint8_t*)&max)[k]);printf("\n");
        __max_16(imax, max); // imax is the maximum number in max
        if (imax >= minsc) { // write the b array; this condition adds branching unfornately
            if (n_b == 0 || (int32_t)b[n_b-1] + 1 != i) { // then append
                if (n_b == m_b) {
                    m_b = m_b? m_b<<1 : 8;
                    b = (uint64_t*) realloc (b, 8 * m_b);
                }
                b[n_b++] = (uint64_t)imax<<32 | i;
            } else if ((int)(b[n_b-1]>>32) < imax) b[n_b-1] = (uint64_t)imax<<32 | i; // modify the last
        }
        if (imax > gmax) {
            gmax = imax; te = i; // te is the end position on the target
            for (j = 0; LIKELY(j < slen); ++j) // keep the H1 vector
                _mm_store_si128(Hmax + j, _mm_load_si128(H1 + j));
            if (gmax + q->shift >= 255 || gmax >= endsc) break;
        }
        S = H1; H1 = H0; H0 = S; // swap H0 and H1
    }
    r.score = gmax + q->shift < 255? gmax : 255;
    r.te = te;
    if (r.score != 255) { // get a->qe, the end of query match; find the 2nd best score
        int max = -1, tmp, low, high, qlen = slen * 16;
        uint8_t *t = (uint8_t*) Hmax;
        for (i = 0; i < qlen; ++i, ++t)
            if ((int)*t > max) max = *t, r.qe = i / 16 + i % 16 * slen;
            else if ((int)*t == max && (tmp = i / 16 + i % 16 * slen) < r.qe) r.qe = tmp; 
        if (b) {
            assert(q->max != 0);
            i = (r.score + q->max - 1) / q->max;
            low = te - i; high = te + i;
            for (i = 0; i < n_b; ++i) {
                int e = (int32_t)b[i];
                if ((e < low || e > high) && (int)(b[i]>>32) > r.score2)
                    r.score2 = b[i]>>32, r.te2 = e;
            }
        }
    }
    
#if MAXI
    fprintf(stderr, "score: %d, te: %d, qe: %d, score2: %d, te2: %d\n",
            r.score, r.te, r.qe, r.score2, r.te2);
#endif
    
    free(b);
    return r;
}

kswr_t kswv::kswvScalar_i16(kswq_t *q, int tlen, const uint8_t *target,
                            int _o_del, int _e_del, int _o_ins, int _e_ins,
                            int xtra) // the first gap costs -(_o+_e)
{
    int slen, i, m_b, n_b, te = -1, gmax = 0, minsc, endsc;
    uint64_t *b;
    __m128i zero, oe_del, e_del, oe_ins, e_ins, *H0, *H1, *E, *Hmax;
    kswr_t r;
#define SIMD16 8

#define __max_8(ret, xx) do { \
        (xx) = _mm_max_epi16((xx), _mm_srli_si128((xx), 8)); \
        (xx) = _mm_max_epi16((xx), _mm_srli_si128((xx), 4)); \
        (xx) = _mm_max_epi16((xx), _mm_srli_si128((xx), 2)); \
        (ret) = _mm_extract_epi16((xx), 0); \
    } while (0)

    // initialization
    r = g_defr;
    minsc = (xtra&KSW_XSUBO)? xtra&0xffff : 0x10000;
    endsc = (xtra&KSW_XSTOP)? xtra&0xffff : 0x10000;
    m_b = n_b = 0; b = 0;
    zero = _mm_set1_epi32(0);
    oe_del = _mm_set1_epi16(_o_del + _e_del);
    e_del = _mm_set1_epi16(_e_del);
    oe_ins = _mm_set1_epi16(_o_ins + _e_ins);
    e_ins = _mm_set1_epi16(_e_ins);
    H0 = q->H0; H1 = q->H1; E = q->E; Hmax = q->Hmax;
    slen = q->slen;
    for (i = 0; i < slen; ++i) {
        _mm_store_si128(E + i, zero);
        _mm_store_si128(H0 + i, zero);
        _mm_store_si128(Hmax + i, zero);
    }
    // the core loop
    for (i = 0; i < tlen; ++i) {
        int j, k, imax;
        __m128i e, t, h, f = zero, max = zero, *S = q->qp + target[i] * slen; // s is the 1st score vector
        h = _mm_load_si128(H0 + slen - 1); // h={2,5,8,11,14,17,-1,-1} in the above example
        h = _mm_slli_si128(h, 2);
        for (j = 0; LIKELY(j < slen); ++j) {

            h = _mm_adds_epi16(h, *S++);
            e = _mm_load_si128(E + j);
            h = _mm_max_epi16(h, e);
            h = _mm_max_epi16(h, f);
            max = _mm_max_epi16(max, h);
            _mm_store_si128(H1 + j, h);
            e = _mm_subs_epu16(e, e_del);
            t = _mm_subs_epu16(h, oe_del);
            e = _mm_max_epi16(e, t);
            _mm_store_si128(E + j, e);
            f = _mm_subs_epu16(f, e_ins);
            t = _mm_subs_epu16(h, oe_ins);
            f = _mm_max_epi16(f, t);
            h = _mm_load_si128(H0 + j);
        }
        
        for (k = 0; LIKELY(k < 16); ++k) {
            f = _mm_slli_si128(f, 2);
            for (j = 0; LIKELY(j < slen); ++j) {
                h = _mm_load_si128(H1 + j);
                h = _mm_max_epi16(h, f);
                _mm_store_si128(H1 + j, h);
                h = _mm_subs_epu16(h, oe_ins);
                f = _mm_subs_epu16(f, e_ins);
                if(UNLIKELY(!_mm_movemask_epi8(_mm_cmpgt_epi16(f, h)))) goto end_loop8;
            }
        }
        
end_loop8:
        __max_8(imax, max);
        if (imax >= minsc) {
            if (n_b == 0 || (int32_t)b[n_b-1] + 1 != i) {
                if (n_b == m_b) {
                    m_b = m_b? m_b<<1 : 8;
                    b = (uint64_t*)realloc(b, 8 * m_b);
                }
                b[n_b++] = (uint64_t)imax<<32 | i;
            } else if ((int)(b[n_b-1]>>32) < imax) b[n_b-1] = (uint64_t)imax<<32 | i; // modify the last
        }
        if (imax > gmax) {
            gmax = imax; te = i;
            for (j = 0; LIKELY(j < slen); ++j)
                _mm_store_si128(Hmax + j, _mm_load_si128(H1 + j));
            if (gmax >= endsc) break;
        }
        S = H1; H1 = H0; H0 = S;
    }
    
    r.score = gmax; r.te = te;
    {
        int max = -1, tmp, low, high, qlen = slen * 8;
        uint16_t *t = (uint16_t*)Hmax;
        for (i = 0, r.qe = -1; i < qlen; ++i, ++t)
            if ((int)*t > max) max = *t, r.qe = i / 8 + i % 8 * slen;
            else if ((int)*t == max && (tmp = i / 8 + i % 8 * slen) < r.qe) r.qe = tmp; 
        if (b) {
            assert(q->max != 0);
            i = (r.score + q->max - 1) / q->max;
            low = te - i; high = te + i;
            for (i = 0; i < n_b; ++i) {
                int e = (int32_t)b[i];
                if ((e < low || e > high) && (int)(b[i]>>32) > r.score2)
                    r.score2 = b[i]>>32, r.te2 = e;
            }
        }
    }

    free(b);
    return r;
}

// -------------------------------------------------------------
// kswc scalar, wrapper function, the interface.
//-------------------------------------------------------------
void kswv::kswvScalarWrapper(SeqPair *seqPairArray,
                             uint8_t *seqBufRef,
                             uint8_t *seqBufQer,
                             kswr_t* aln,
                             int numPairs,
                             int nthreads,
                             bool sw, int tid) {
    
    int8_t mat[25];
    bwa_fill_scmat(mat);

    int st = 0, ed = numPairs;    
    for (int i = st; i < ed; i++)
    {
        SeqPair *p = seqPairArray + i;
        kswr_t *myaln = aln + p->regid;
            
        uint8_t *target = seqBufRef + p->idr;
        uint8_t *query = seqBufQer + p->idq;

        int tlen = p->len1;
        int qlen = p->len2;
        int xtra = p->h0;

        kswq_t *q;
        kswr_t ks;
            
        int vw = (xtra & KSW_XBYTE)? 1 : 2;
        if (sw == 0) {
            assert(vw == 1);
            q = ksw_qinit((xtra & KSW_XBYTE)? 1 : 2, qlen, query, this->m, mat);
            ks = kswvScalar_u8(q, tlen, target,
                               o_del, e_del,
                               o_ins, e_ins,
                               xtra, tid, i);
            free(q);
        } else {
            assert(vw == 2);
            q = ksw_qinit(2, qlen, query, this->m, mat);
            ks = kswvScalar_i16(q, tlen, target,
                                o_del, e_del,
                                o_ins, e_ins,
                                xtra);
            free(q);
        }

        myaln->score = ks.score;
        myaln->tb = ks.tb;
        myaln->te = ks.te;
        myaln->qb = ks.qb;
        myaln->qe = ks.qe;
        myaln->score2 = ks.score2;
        myaln->te2 = ks.te2;
    }

}
#endif

/************************** standalone benchmark code *************************************/
/** This is a standalone code for the benchmarking purpose 
    By default it is disabled.
**/

#if MAINY
#define DEFAULT_MATCH 1
#define DEFAULT_MISMATCH -4
#define DEFAULT_OPEN 6
#define DEFAULT_EXTEND 1
#define DEFAULT_AMBIG -1

// #define MAX_NUM_PAIRS 1000
// #define MATRIX_MIN_CUTOFF -100000000
// #define LOW_INIT_VALUE (INT32_MIN/2)
// #define AMBIG 52

int spot = 42419;
double freq = 2.3*1e9;
int32_t w_match, w_mismatch, w_open, w_extend, w_ambig;
uint64_t SW_cells;
char *pairFileName;
FILE *pairFile;
int8_t h0 = 0;
double clock_freq;
//uint64_t prof[10][112], data, SW_cells2;


void parseCmdLine(int argc, char *argv[])
{
    int i;
    w_match = DEFAULT_MATCH;
    w_mismatch = DEFAULT_MISMATCH;
    w_open = DEFAULT_OPEN;
    w_extend = DEFAULT_EXTEND;
    w_ambig = DEFAULT_AMBIG;
    
    int pairFlag = 0;
    for(i = 1; i < argc; i+=2)
    {
        if(strcmp(argv[i], "-match") == 0)
        {
            w_match = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-mismatch") == 0) //penalty, +ve number
        {
            w_mismatch = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-ambig") == 0)
        {
            w_ambig = atoi(argv[i + 1]);
        }

        if(strcmp(argv[i], "-gapo") == 0)
        {
            w_open = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-gape") == 0)
        {
            w_extend = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-pairs") == 0)
        {
            pairFileName = argv[i + 1];
            pairFlag = 1;
        }
        if(strcmp(argv[i], "-h0") == 0)
        {
            h0 = atoi(argv[i + 1]);
        }
    }
    if(pairFlag == 0)
    {
        printf("ERROR! pairFileName not specified.\n");
        exit(EXIT_FAILURE);
    }
}

int loadPairs(SeqPair *seqPairArray, uint8_t *seqBufRef, uint8_t* seqBufQer, FILE *pairFile)
{
    static int32_t cnt = 0;
    int32_t numPairs = 0;
    while(numPairs < MAX_NUM_PAIRS_ALLOC)
    {
        int32_t xtra = 0;
        char temp[10];
        fgets(temp, 10, pairFile);
        sscanf(temp, "%x", &xtra);
        //printf("xtra: %d, %x, %s\n", xtra, xtra, temp);

        //if(!fgets((char *)(seqBuf + numPairs * 2 * MAX_SEQ_LEN), MAX_SEQ_LEN, pairFile))
        if(!fgets((char *)(seqBufRef + numPairs * this->maxRefLen), this->maxRefLen, pairFile))
        {
            break;
        }
        //if(!fgets((char *)(seqBuf + (numPairs * 2 + 1) * MAX_SEQ_LEN), MAX_SEQ_LEN, pairFile))
        if(!fgets((char *)(seqBufQer + numPairs * this->maxQerLen), this->maxQerLen, pairFile)) 
        {
            printf("ERROR! Odd number of sequences in %s\n", pairFileName);
            break;
        }

        SeqPair sp;
        sp.id = numPairs;
        // sp.seq1 = seqBuf + numPairs * 2 * MAX_SEQ_LEN;
        // sp.seq2 = seqBuf + (numPairs * 2 + 1) * MAX_SEQ_LEN;
        sp.len1 = strnlen((char *)(seqBufRef + numPairs * this->maxRefLen), this->maxRefLen) - 1;
        sp.len2 = strnlen((char *)(seqBufQer + numPairs * this->maxQerLen), this->maxQerLen) - 1;
        sp.h0 = xtra;
        // sp.score = 0;

        //uint8_t *seq1 = seqBuf + numPairs * 2 * MAX_SEQ_LEN;
        //uint8_t *seq2 = seqBuf + (numPairs * 2 + 1) * MAX_SEQ_LEN;
        uint8_t *seq1 = seqBufRef + numPairs * this->maxRefLen;
        uint8_t *seq2 = seqBufQer + numPairs * this->maxQerLen;

        for (int l=0; l<sp.len1; l++)
            seq1[l] -= 48;
        for (int l=0; l<sp.len2; l++)
            seq2[l] -= 48;

        seqPairArray[numPairs] = sp;
        numPairs++;
        // SW_cells += (sp.len1 * sp.len2);
    }
    // fclose(pairFile);
    return numPairs;
}

// profiling stats
uint64_t find_stats(uint64_t *val, int nt, double &min, double &max, double &avg)
{
    min = 1e10;
    max = 0;
    avg = 0;
    for (int i=0; i<nt; i++) {
        avg += val[i];
        if (max < val[i]) max = val[i];
        if (min > val[i]) min = val[i];     
    }
    avg /= nt;
}

int main(int argc, char *argv[])
{

#ifdef VTUNE_ANALYSIS
    printf("Vtune analysis enabled....\n");
    __itt_pause();
#endif
    FILE *fsam = fopen("fsam.txt", "w");    
    kswv *mate;

    parseCmdLine(argc, argv);

    SeqPair *seqPairArray = (SeqPair *)_mm_malloc((MAX_NUM_PAIRS + SIMD_WIDTH8) * sizeof(SeqPair), 64);
    uint8_t *seqBufRef = NULL, *seqBufQer = NULL;
    seqBufRef = (uint8_t *)_mm_malloc((this->maxRefLen * MAX_NUM_PAIRS + MAX_LINE_LEN)
                                      * sizeof(int8_t), 64);
    seqBufQer = (uint8_t *)_mm_malloc((this->maxQerLen * MAX_NUM_PAIRS + MAX_LINE_LEN)
                                      * sizeof(int8_t), 64);

    kswr_t *aln = NULL;
    aln = (kswr_t *) _mm_malloc ((MAX_NUM_PAIRS + SIMD_WIDTH8) * sizeof(kswr_t), 64);
    
    if (seqBufRef == NULL || seqBufQer == NULL || aln == NULL)
    {
        printf("Memory not allocated\nExiting...\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Memory allocated: %0.2lf MB\n",
               ((int64_t)(this->maxRefLen + this->maxQerLen ) * MAX_NUM_PAIRS + MAX_LINE_LEN)/1e6);
    }
    uint64_t tim = __rdtsc(), readTim = 0;
    
    int32_t numThreads = 1;
#pragma omp parallel
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        if(tid == (nt - 1))
        {
            numThreads = nt;
        }
    }
    numThreads =1 ;
    //printf("Done reading input file!!, numPairs: %d, nt: %d\n",
    //     numPairs, numThreads);

#if SORT_PAIRS     // disbaled in bwa-mem2 (only used in separate benchmark sw code)
    printf("\tSorting is enabled !!\n");
#endif

#if MAXI
    printf("\tResults printing (on console) is enabled.....\n");
#endif

    tim = __rdtsc();
    sleep(1);
    freq = __rdtsc() - tim;

    int numPairs = 0, totNumPairs = 0;
    FILE *pairFile = fopen(pairFileName, "r");
    if(pairFile == NULL)
    {
        fprintf(stderr, "Could not open file: %s\n", pairFileName);
        exit(EXIT_FAILURE);
    }

    // BandedPairWiseSW *pwsw = new BandedPairWiseSW(w_match, w_mismatch, w_open,
    //                                            w_extend, w_ambig, end_bonus);
    kswv *pwsw = new kswv(w_open, w_extend, w_open, w_extend, w_match, w_mismatch, numThreads);
    

    int64_t myTicks = 0;
    printf("Processor freq: %0.2lf MHz\n", freq/1e6);
    printf("Executing int8 code....\n");
#if VEC
    printf("Executing AVX512 vectorized code!!\n");
    
    while (1) {
        uint64_t tim = __rdtsc();

        //printf("Loading current batch of pairs..\n");
        numPairs = loadPairs(seqPairArray, seqBufRef, seqBufQer, pairFile);
        totNumPairs += numPairs;
#if STAT
        printf("Loading done, numPairs: %d\n", numPairs);      
        if (totNumPairs > spot) spot -= totNumPairs - numPairs;
        else continue;
#endif
        readTim += __rdtsc() - tim;
        if (numPairs == 0) break;

        tim = __rdtsc();
        int phase = 0;
        pwsw->getScores8(seqPairArray, seqBufRef, seqBufQer, aln, numPairs, numThreads, phase);
        // pwsw->getScores16(seqPairArray, seqBufRef, seqBufQer, aln, numPairs, numThreads, phase);
        myTicks += __rdtsc() - tim;

#if STAT
        for (int l=0; l<10; l++)
        {
            // SeqPair r = seqPairArray[l];
            kswr_t r = aln[l];
            fprintf(stderr, "%d %d %d %d %d %d %d\n", r.score, r.tb, r.te, r.qb, r.qe, r.score2, r.te2);
        }       
        break;
#endif
        
#if MAXI
        // printf("Execution complete!!, writing output!\n");
        //for (int l=0; l<numPairs; l++)
        //{
        //  // SeqPair r = seqPairArray[l];
        //  kswr_t r = aln[l];
        //  fprintf(stderr, "%d %d %d %d %d %d %d\n", r.score, r.tb, r.te, r.qb, r.qe, r.score2, r.te2);
        //}
        // printf("Vector code: Writing output completed!!!\n\n");
        // printf("Vector code -- Wrote output to the file\n");
#endif
    }
    
    // int64_t myTicks = 0;//pwsw->getTicks();
    printf("Read time  = %0.2lf\n", readTim/freq);
    printf("Overall SW cycles = %ld, %0.2lf\n", myTicks, myTicks*1.0/freq);
    printf("SW cells(T)  = %ld\n", SW_cells);
    printf("SW cells(||)  = %ld\n", SW_cells2);
    printf("SW GCUPS  = %lf\n", SW_cells * freq / myTicks);
    
    {
        printf("More stats:\n");
        // double freq = 2.3*1e9;
        double min, max, avg;
        find_stats(prof[1], numThreads, min, max, avg);
        printf("Time in pre-processing: %0.2lf (%0.2lf, %0.2lf)\n",
               avg*1.0/freq, min*1.0/freq, max*1.0/freq);
        find_stats(prof[0], numThreads, min, max, avg);
        printf("Time spent in smithWaterman(): %0.2lf (%0.2lf, %0.2lf)\n",
               avg*1.0/freq, min*1.0/freq, max*1.0/freq);
        printf("\nTotal cells computed: %ld, %0.2lf\n",
               prof[2][0], prof[2][0]*1.0/(totNumPairs));
        printf("\tTotal useful cells computed: %ld, %0.2lf\n",
               prof[3][0], prof[3][0]*1.0/(totNumPairs));
        // printf("Total bp read from memory: %ld\n", data);
        printf("Computations after exit: %ld\n", prof[4][0]);
        printf("Cumulative seq1/2 len: %ld, %ld\n", prof[4][1], prof[4][2]);
    
#if 1
        printf("\nDebugging info:\n");
        printf("Time taken for DP loop: %0.2lf\n", prof[DP][0]*1.0/freq);
        printf("Time taken for DP loop upper part: %0.2lf\n", prof[DP3][0]*1.0/freq);   
        printf("Time taken for DP inner loop: %0.2lf\n", prof[DP1][0]*1.0/freq);
        printf("Time taken for DP loop lower part: %ld\n", prof[DP2][0]);
#endif
    }

    
#else
    printf("Executing scalar code!!\n");
    while (1) {
        uint64_t tim = __rdtsc();
        numPairs = loadPairs(seqPairArray, seqBufRef, seqBufQer, pairFile);
        totNumPairs += numPairs;
#if STAT
        printf("Loading done, numPairs: %d\n", numPairs);
        if (totNumPairs > spot) spot -= totNumPairs - numPairs;
        else continue;
#endif
        readTim += __rdtsc() - tim;
        if (numPairs == 0) break;

        tim = __rdtsc();
        pwsw->kswvScalarWrapper(seqPairArray,
                               seqBufRef,
                               seqBufQer,
                               aln,
                               numPairs,
                               numThreads);
        myTicks += __rdtsc() - tim;
#if STAT
        for (int l=0; l<10; l++)
        {
            // SeqPair r = seqPairArray[l];
            kswr_t r = aln[l];
            fprintf(stderr, "%d %d %d %d %d %d %d\n", r.score, r.tb, r.te, r.qb, r.qe, r.score2, r.te2);
        }       
        break;
#endif
        
    } // while
    
    printf("Read time  = %0.2lf\n", readTim/freq);
    printf("Overall SW cycles = %ld, %0.2lf\n", myTicks, myTicks*1.0/freq);
    printf("Time taken for DP loop lower part: %ld \n", prof[DP2][0]);

#if 1
        printf("\nDebugging info:\n");
        printf("Time taken for DP loop: %0.2lf\n", prof[DP][0]*1.0/freq);
        printf("Time taken for DP loop upper part: %0.2lf\n", prof[DP3][0]*1.0/freq);   
        printf("Time taken for DP inner loop: %0.2lf\n", prof[DP1][0]*1.0/freq);
        printf("Time taken for DP loop lower part: %ld\n", prof[DP2][0]);
#endif

#endif


#ifdef VTUNE_ANALYSIS
    printf("Vtune analysis enabled....\n");
#endif
        
    // free memory
    _mm_free(seqPairArray);
    _mm_free(seqBufRef);
    _mm_free(seqBufQer);
    _mm_free(aln);
    
    fclose(pairFile);
    fclose(fsam);
    return 1;
}

#endif  // MAINY

/* Per-tier factory function. Compiled into each KERNEL_VARIANT build of
 * this TU; the symbol is mangled by kernel_dispatch.h to
 * make_kswv_kernel_<tier>. On arm64 (no KERNEL_VARIANT) this is the
 * unmangled make_kswv_kernel.
 *
 * Returns Ikswv* (not unique_ptr) because extern "C" disallows non-POD
 * return types. The dispatcher in simd_dispatch.cpp wraps it into
 * unique_ptr at the call site. */
extern "C" Ikswv *make_kswv_kernel(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen)
{
    return new kswv(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                    numThreads, maxRefLen, maxQerLen);
}

/* Mat-aware 10-arg per-tier factory (issue 173). Mangled by
 * kernel_dispatch.h to make_kswv_kernel_<tier>_mat (unmangled
 * make_kswv_kernel_mat on arm64). Constructs via the mat-aware ctor so the
 * freed cell is detected; mat25 == nullptr reproduces the 9-arg behavior. */
extern "C" Ikswv *make_kswv_kernel_mat(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen,
    const int8_t *mat25)
{
    return new kswv(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                    numThreads, maxRefLen, maxQerLen, mat25);
}
