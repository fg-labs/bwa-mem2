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

#ifndef _KSWV_H_
#define  _KSWV_H_

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "macro.h"

#if !MAINY
#include "ksw.h"
#include "bandedSWA.h"
#else
    #if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
        #include "simd_compat.h"
    #else
        #include <immintrin.h>
    #endif
#endif

/* LIKELY/UNLIKELY come from macro.h (included above). */

#define MAX_SEQ_LEN_REF_SAM 2048
#define MAX_SEQ_LEN_QER_SAM 512

#if MAINY
#define KSW_XBYTE  0x10000
#define KSW_XSTOP  0x20000
#define KSW_XSUBO  0x40000
#define KSW_XSTART 0x80000

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif


#define MAX_SEQ_LEN_EXT 256

/* SIMD width definitions based on architecture */
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
    /* ARM/Apple Silicon - 128-bit NEON vectors */
    #ifndef SIMD_WIDTH8
    #define SIMD_WIDTH8 16   /* 128-bit / 8-bit = 16 elements */
    #endif
    #ifndef SIMD_WIDTH16
    #define SIMD_WIDTH16 8   /* 128-bit / 16-bit = 8 elements */
    #endif
#elif __AVX512BW__
    #define SIMD_WIDTH8 64
    #define SIMD_WIDTH16 32
#elif __AVX2__
    /* AVX2 - 256-bit vectors. Stub kernel for now (phase 1); the real
     * vector kernel lands on the c6i iteration in phase 2. */
    #define SIMD_WIDTH8 32
    #define SIMD_WIDTH16 16
#endif

#define max(x, y) ((x)>(y)?(x):(y))
#define min(x, y) ((x)>(y)?(y):(x))

#define MAX_NUM_PAIRS 1000000
#define MAX_NUM_PAIRS_ALLOC 20000

#define DEFAULT_AMBIG -1

typedef struct dnaSeqPair
{
	int32_t idr, idq, id;
	int32_t len1, len2;
	int32_t h0;
	int seqid, regid;
	int score; // best score
	int te, qe; // target end and query end
	int score2, te2; // second best score and ending position on the target
	int tb, qb; // target start and query start
}SeqPair;

typedef struct {
	int qlen, slen;
	uint8_t shift, mdiff, max, size;
	__m128i *qp, *H0, *H1, *E, *Hmax;
} kswq_t;

typedef struct {
	int score; // best score
	int te, qe; // target end and query end
	int score2, te2; // second best score and ending position on the target
	int tb, qb; // target start and query start
} kswr_t;


const kswr_t g_defr = { 0, -1, -1, -1, -1, -1, -1 };

#define __max_16(ret, xx) do { \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 8)); \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 4)); \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 2)); \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 1)); \
    	(ret) = _mm_extract_epi16((xx), 0) & 0x00ff; \
	} while (0)

#define DP  6
#define DP1 7
#define DP2 8
#define DP3 9

#endif

#include "kernel_dispatch.h"  /* must come before kswv class so per-tier compiles see the renamed symbol */

#include <memory>

/* Abstract interface for the vectorized KSW batch kernel.
 * Concrete implementations are per-tier mangled subclasses (kswv compiled
 * with KERNEL_VARIANT=_sse41/_sse42/_avx/_avx2/_avx512bw on x86, unsuffixed
 * on arm64). Construct via make_kswv().
 *
 * getScores8/getScores16 are declared unconditionally here even though the
 * concrete class only defines them on builds with vector support. Every
 * tier we ship satisfies this; if a future build disables vector kernels,
 * these decls would need conditional guards to match the cpp side. */
class Ikswv {
public:
    virtual ~Ikswv() = default;

    virtual void getScores8(SeqPair *pairArray,
                            uint8_t *seqBufRef,
                            uint8_t *seqBufQer,
                            kswr_t* aln,
                            int32_t numPairs,
                            uint16_t numThreads,
                            int phase) = 0;

    virtual void getScores16(SeqPair *pairArray,
                             uint8_t *seqBufRef,
                             uint8_t *seqBufQer,
                             kswr_t* aln,
                             int32_t numPairs,
                             uint16_t numThreads,
                             int phase) = 0;

    /* True when the construction matrix is asymmetric in a way THIS concrete
     * tier cannot express, so the caller must route those pairs to the scalar
     * fallback (ksw_align2). Two reasons it can be true:
     *   - the matrix shape is unsupported: a non-mirror multi-cell free, a
     *     changed diagonal, or a freed value outside [w_mismatch, w_match]
     *     (outside that domain the 8-bit kernel's biased-u8 blend would wrap);
     *     or
     *   - the running tier lacks the freed-cell kernel override. Only the NEON,
     *     AVX2, and AVX-512BW kernels implement it; an SSE41/SSE42/AVX kswv
     *     reports true for any freed-cell matrix.
     * A symmetric matrix, a single freed cell scored to ANY in-domain value
     * (bisulfite genomic OT/OB frees it to +a, neutral to 0), or an exact
     * mirrored freed pair (collapsed --meth) returns false on a tier that
     * implements the override. Declared on the abstract interface so the
     * dispatcher never depends on the concrete kswv layout. */
    virtual bool needsScalar() const = 0;

};

/* Factory: returns a per-tier concrete kswv. Construction args mirror the
 * kswv ctor exactly. */
std::unique_ptr<Ikswv> make_kswv(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen);

/* Mat-aware factory overload (issue 173). `mat25` is the 5x5 scoring matrix
 * the caller will use for SW; when non-null and asymmetric, the ctor detects
 * the freed cell (bisulfite OT/OB) and the value it is freed to, or flags
 * needsScalar() for any richer asymmetry. `mat25 == nullptr` (or a symmetric
 * matrix) reproduces the 9-arg behavior exactly. The sign convention matches
 * make_kswv's existing callers: w_match is +a, w_mismatch is -b (negative),
 * and mat25's off-diagonals are likewise the negated penalty — passed straight
 * through to the Task-1 detectors without re-negation. */
std::unique_ptr<Ikswv> make_kswv(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen,
    const int8_t *mat25);

/* Per-tier factory function forward declarations.
 * Defined in kswv.<tier>.o (each kernel TU compile).
 * Called from simd_dispatch.cpp to construct the right per-tier concrete
 * class without exposing the class layout to the dispatcher TU (which
 * previously caused an ODR size mismatch / heap corruption). */
#if defined(__x86_64__) || defined(__i386__)
extern "C" Ikswv *make_kswv_kernel_sse41(int, int, int, int, int8_t, int8_t,
                                         int, int32_t, int32_t);
extern "C" Ikswv *make_kswv_kernel_sse42(int, int, int, int, int8_t, int8_t,
                                         int, int32_t, int32_t);
extern "C" Ikswv *make_kswv_kernel_avx(int, int, int, int, int8_t, int8_t,
                                       int, int32_t, int32_t);
extern "C" Ikswv *make_kswv_kernel_avx2(int, int, int, int, int8_t, int8_t,
                                        int, int32_t, int32_t);
extern "C" Ikswv *make_kswv_kernel_avx512bw(int, int, int, int, int8_t, int8_t,
                                            int, int32_t, int32_t);

/* Mat-aware 10-arg per-tier overloads (issue 173). Distinct C symbols (the
 * `_mat` suffix) since C linkage has no overloading; the trailing arg is the
 * 5x5 scoring matrix forwarded to the ctor for freed-cell detection. */
extern "C" Ikswv *make_kswv_kernel_sse41_mat(int, int, int, int, int8_t, int8_t,
                                             int, int32_t, int32_t, const int8_t *);
extern "C" Ikswv *make_kswv_kernel_sse42_mat(int, int, int, int, int8_t, int8_t,
                                             int, int32_t, int32_t, const int8_t *);
extern "C" Ikswv *make_kswv_kernel_avx_mat(int, int, int, int, int8_t, int8_t,
                                           int, int32_t, int32_t, const int8_t *);
extern "C" Ikswv *make_kswv_kernel_avx2_mat(int, int, int, int, int8_t, int8_t,
                                            int, int32_t, int32_t, const int8_t *);
extern "C" Ikswv *make_kswv_kernel_avx512bw_mat(int, int, int, int, int8_t, int8_t,
                                                int, int32_t, int32_t, const int8_t *);
#endif


class kswv final : public Ikswv {
public:

	kswv(const int o_del, const int e_del, const int o_ins,
		 const int e_ins, const int8_t w_match, const int8_t w_mismatch,
		 int numThreads, int32_t maxRefLen, int32_t maxQerLen);

	/* Mat-aware ctor (issue 173): same as above, plus a 5x5 scoring matrix
	 * used to detect the freed cell(s) and the value they are freed to.
	 * `mat25 == nullptr` reproduces the
	 * 9-arg ctor exactly. Delegates to the 9-arg ctor, then runs detection. */
	kswv(const int o_del, const int e_del, const int o_ins,
		 const int e_ins, const int8_t w_match, const int8_t w_mismatch,
		 int numThreads, int32_t maxRefLen, int32_t maxQerLen,
		 const int8_t *mat25);

	~kswv() override;

	/* See Ikswv::needsScalar. Set in the mat-aware ctor; false for the 9-arg
	 * ctor (symmetric / nullptr matrix). */
	bool needsScalar() const override { return needs_scalar; }

	// kswv owns heap buffers (rowMax8/16, F/H/E vectors, etc.) freed in the
	// destructor. Allowing copy/move would alias those allocations and make
	// double-free / use-after-free trivial to introduce. Mirrors the same
	// guard on BandedPairWiseSW.
	kswv(const kswv&)            = delete;
	kswv& operator=(const kswv&) = delete;
	kswv(kswv&&)                 = delete;
	kswv& operator=(kswv&&)      = delete;

	void getScores8(SeqPair *pairArray,
					uint8_t *seqBufRef,
					uint8_t *seqBufQer,
					kswr_t* aln,
					int32_t numPairs,
					uint16_t numThreads,
					int phase) override;

	void getScores16(SeqPair *pairArray,
					 uint8_t *seqBufRef,
					 uint8_t *seqBufQer,
					 kswr_t* aln,
					 int32_t numPairs,
					 uint16_t numThreads,
					 int phase) override;

	void kswvScalarWrapper(SeqPair *seqPairArray,
                           uint8_t *seqBufRef,
                           uint8_t *seqBufQer,
                           kswr_t* aln,
                           int numPairs,
                           int nthreads,
                           bool sw, int tid);

	kswq_t* ksw_qinit(int size, int qlen, uint8_t *query, int m, const int8_t *mat);
	
private:
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
	/* ARM/NEON implementations (native NEON for hot paths) */
	void kswvBatchWrapper8(SeqPair *pairArray,
						   uint8_t *seqBufRef,
						   uint8_t *seqBufQer,
						   kswr_t* aln,
						   int32_t numPairs,
						   uint16_t numThreads,
						   int phase);

	/* Thin dispatcher: selects the HasFreed template instantiation based on
	 * the freed-cell flag. The <false> instantiation dead-code-
	 * eliminates every freed-cell override block, giving codegen identical to
	 * the pre-issue-173 kernel for the non-meth path. */
	int kswv_neon_u8(uint8_t seq1SoA[],
				     uint8_t seq2SoA[],
				     int16_t nrow,
				     int16_t ncol,
				     SeqPair *p,
				     kswr_t *aln,
				     int po_ind,
				     uint16_t tid,
				     int32_t numPairs,
				     int phase);

	/* Templated u8 kernel body. When HasFreed, applies the freed-cell
	 * (fr_ref x fr_read -> fr_val) override per cell; otherwise the override
	 * blocks compile out entirely. LazyQE selects, for the two-row sweep only,
	 * post-row query-end recovery from the stored H (true) over the per-cell
	 * argmax blend (false); it is meaningless when RowPair is false. */
	template<bool HasFreed, bool USQADD, bool RowPair, bool LazyQE>
	int kswv_neon_u8_impl(uint8_t seq1SoA[],
					      uint8_t seq2SoA[],
					      int16_t nrow,
					      int16_t ncol,
					      SeqPair *p,
					      kswr_t *aln,
					      int po_ind,
					      uint16_t tid,
					      int32_t numPairs,
					      int phase);

	void kswvBatchWrapper16(SeqPair *pairArray,
							uint8_t *seqBufRef,
							uint8_t *seqBufQer,
							kswr_t* aln,
							int32_t numPairs,
							uint16_t numThreads,
							int phase);

	/* Thin dispatcher: see kswv_neon_u8 above. */
	int kswv_neon_16(int16_t seq1SoA[],
                     int16_t seq2SoA[],
                     int16_t nrow,
                     int16_t ncol,
                     SeqPair *p,
                     kswr_t* aln,
                     int po_ind,
                     uint16_t tid,
                     int32_t numPairs,
                     int phase);

	/* Templated i16 kernel body; see kswv_neon_u8_impl. */
	/* Templated 16-bit kernel body. RowPair / LazyQE mirror the u8 kernel's
	 * two-row sweep and its lazy query-end recovery (LazyQE is meaningless
	 * when RowPair is false; the one-row body is always lazy). */
	template<bool HasFreed, bool RowPair, bool LazyQE>
	int kswv_neon_16_impl(int16_t seq1SoA[],
                          int16_t seq2SoA[],
                          int16_t nrow,
                          int16_t ncol,
                          SeqPair *p,
                          kswr_t* aln,
                          int po_ind,
                          uint16_t tid,
                          int32_t numPairs,
                          int phase);

#elif ((!__AVX512BW__) & (__AVX2__))
	/* AVX2 (256-bit, 32-lane u8) batched mate-rescue SW kernel.
	 * Modeled on the corrected NEON kernel (not AVX-512, which has a
	 * pre-existing coord/score2 bug class that the NEON port uncovered
	 * and fixed; see PR 18). All four NEON bug fixes are pre-applied:
	 *  (1) te tracked in two half-width int16 vectors (_lo/_hi), since
	 *      32 u8 lanes need 32 int16 slots but a __m256i int16 holds 16.
	 *  (2) per-lane freeze mask once a pair hits KSW_XSTOP.
	 *  (3) score2 scan with per-lane len1/low/high/qe exclusion.
	 *  (4) minsc filter on rowMax in the score2 scan. */
	void kswvBatchWrapper8_avx2(SeqPair *pairArray,
								uint8_t *seqBufRef,
								uint8_t *seqBufQer,
								kswr_t* aln,
								int32_t numPairs,
								uint16_t numThreads,
								int phase);

	/* Thin dispatcher: selects the HasFreed template instantiation based on
	 * the freed-cell flag. The <false> instantiation dead-code-
	 * eliminates every freed-cell override block, giving codegen identical to
	 * the pre-issue-173 kernel for the non-meth path. Mirrors kswv_neon_u8. */
	int kswv256_u8(uint8_t seq1SoA[],
				   uint8_t seq2SoA[],
				   int16_t nrow,
				   int16_t ncol,
				   SeqPair *p,
				   kswr_t *aln,
				   int po_ind,
				   uint16_t tid,
				   int32_t numPairs,
				   int phase);

	/* Templated u8 kernel body; see kswv_neon_u8_impl. */
	template<bool HasFreed>
	int kswv256_u8_impl(uint8_t seq1SoA[],
				   uint8_t seq2SoA[],
				   int16_t nrow,
				   int16_t ncol,
				   SeqPair *p,
				   kswr_t *aln,
				   int po_ind,
				   uint16_t tid,
				   int32_t numPairs,
				   int phase);

	void kswvBatchWrapper16_avx2(SeqPair *pairArray,
								 uint8_t *seqBufRef,
								 uint8_t *seqBufQer,
								 kswr_t* aln,
								 int32_t numPairs,
								 uint16_t numThreads,
								 int phase);

	/* Thin dispatcher: see kswv256_u8 above. */
	int kswv256_16(int16_t seq1SoA[],
				   int16_t seq2SoA[],
				   int16_t nrow,
				   int16_t ncol,
				   SeqPair *p,
				   kswr_t *aln,
				   int po_ind,
				   uint16_t tid,
				   int32_t numPairs,
				   int phase);

	/* Templated i16 kernel body; see kswv_neon_u8_impl. */
	template<bool HasFreed>
	int kswv256_16_impl(int16_t seq1SoA[],
				   int16_t seq2SoA[],
				   int16_t nrow,
				   int16_t ncol,
				   SeqPair *p,
				   kswr_t *aln,
				   int po_ind,
				   uint16_t tid,
				   int32_t numPairs,
				   int phase);

#elif __AVX512BW__
	void kswvBatchWrapper8(SeqPair *pairArray,
						   uint8_t *seqBufRef,
						   uint8_t *seqBufQer,
						   kswr_t* aln,
						   int32_t numPairs,
						   uint16_t numThreads,
						   int phase);

	/* Thin dispatcher: selects the HasFreed template instantiation based on
	 * the freed-cell flag. See kswv256_u8 / kswv_neon_u8. */
	int kswv512_u8(uint8_t seq1SoA[],
				   uint8_t seq2SoA[],
				   int16_t nrow,
				   int16_t ncol,
				   SeqPair *p,
				   kswr_t *aln,
				   int po_ind,
				   uint16_t tid,
				   int32_t numPairs,
				   int phase);

	/* Templated u8 kernel body; see kswv_neon_u8_impl. */
	template<bool HasFreed>
	int kswv512_u8_impl(uint8_t seq1SoA[],
				   uint8_t seq2SoA[],
				   int16_t nrow,
				   int16_t ncol,
				   SeqPair *p,
				   kswr_t *aln,
				   int po_ind,
				   uint16_t tid,
				   int32_t numPairs,
				   int phase);

	void kswvBatchWrapper16(SeqPair *pairArray,
							uint8_t *seqBufRef,
							uint8_t *seqBufQer,
							kswr_t* aln,
							int32_t numPairs,
							uint16_t numThreads,
							int phase);

	/* Thin dispatcher: see kswv512_u8 above. */
	int kswv512_16(int16_t seq1SoA[],
                   int16_t seq2SoA[],
                   int16_t nrow,
                   int16_t ncol,
                   SeqPair *p,
                   kswr_t* aln,
                   int po_ind,
                   uint16_t tid,
                   int32_t numPairs,
                   int phase);

	/* Templated i16 kernel body; see kswv_neon_u8_impl. */
	template<bool HasFreed>
	int kswv512_16_impl(int16_t seq1SoA[],
                   int16_t seq2SoA[],
                   int16_t nrow,
                   int16_t ncol,
                   SeqPair *p,
                   kswr_t* aln,
                   int po_ind,
                   uint16_t tid,
                   int32_t numPairs,
                   int phase);
#endif
	
	kswr_t kswvScalar_u8(kswq_t *q, int tlen, const uint8_t *target,
						int _o_del, int _e_del, int _o_ins, int _e_ins,
						int xtra);  // the first gap costs -(_o+_e)
	
	kswr_t kswvScalar_i16(kswq_t *q, int tlen, const uint8_t *target,
						  int _o_del, int _e_del, int _o_ins, int _e_ins,
						  int xtra); // the first gap costs -(_o+_e)
	
	void bwa_fill_scmat(int8_t mat[25]);
		
	int m;
	int o_del, o_ins, e_del, e_ins;
	// const int8_t *mat;

	/* Freed-cell descriptor (issue 173), populated by the mat-aware ctor.
	 * `has_freed` ⇒ off-diagonal cells were freed from their symmetric value
	 * (bisulfite). The kernel frees a SYMMETRIC PAIR of cells: (fr_ref x
	 * fr_read) and (fr_ref2 x fr_read2). GENOMIC and NEUTRAL free ONE cell
	 * (to +a and to 0 respectively), so the mirror
	 * equals the primary (fr_ref2==fr_ref, fr_read2==fr_read) and the second
	 * blend is idempotent. COLLAPSED (the --meth default) frees the conversion
	 * cell AND its mirror, so (fr_ref2,fr_read2) = (fr_read,fr_ref).
	 * `fr_val` is the VALUE the freed cell(s) score to: w_match for GENOMIC and
	 * COLLAPSED (freed to a match), or any other constant for a single-cell free
	 * that is not a match (NEUTRAL frees the conversion cell to 0). The kernel
	 * blends the freed cell(s) to fr_val (biased by +shift in the 8-bit domain),
	 * so fr_val==w_match reproduces the pre-generalization match blend exactly.
	 * `needs_scalar` ⇒ the matrix is asymmetric in a way the kernel cannot
	 * express (non-mirror multi-cell, changed diagonal, …) and the caller must
	 * fall back to scalar. The 9-arg ctor leaves all false. ABI note: these live
	 * here (not in the dispatcher TU) and are safe to add because every TU
	 * includes this same kswv.h. */
	int8_t fr_ref = 0, fr_read = 0;
	int8_t fr_ref2 = 0, fr_read2 = 0;
	int8_t fr_val = 0;   /* set to w_match in the mat-aware ctor unless a single
	                        non-match cell was freed (NEUTRAL); default 0 is unused
	                        when has_freed is false. */
	bool has_freed = false;
	bool needs_scalar = false;

	int8_t w_match;
	int8_t w_mismatch;
	int8_t w_open;
	int8_t w_extend;
	int8_t w_ambig;
	uint8_t *F8;
	uint8_t *H8_0, *H8_1;
	uint8_t *rowMax8;

	/* Column index broadcast across all lanes: colIdx8[j*W + k] == (uint8_t) j.
	 * All three u8 kernels -- NEON, AVX2 and AVX-512BW -- read it when they
	 * recover the query end after the row: the scan needs the scanned column
	 * index in every lane, and one load is cheaper there than a broadcast or a
	 * running vector add.
	 *
	 * maxQerLen * SIMD_WIDTH8 bytes -- 8 KB at 16 lanes, 33 KB at 64 -- but a
	 * row only touches the block(s) the QE_BLK checkpoints select, so the live
	 * working set is a fraction of that and stays L1-resident alongside H0/H1/F.
	 *
	 * Only the low byte matters: ncol cannot exceed 255 in the 8-bit kernels,
	 * capped by the width guard (l_ms * a < 250). That is the same bound
	 * QE_MAXBLK sizes the checkpoint array against -- see the QE_BLK comment in
	 * kswv.cpp. */
	uint8_t *colIdx8;

	/* 16-bit twins for the NEON 16-bit kernel's lazy query-end recovery:
	 * colIdx16[j*W16 + k] == (int16_t) j, and qeBlk16 holds TWO rows' worth of
	 * per-QE_BLK running-max checkpoints (row i at qeBlk16, row i+1 at
	 * qeBlk16 + qeBlk16Stride) for the two-row sweep. Both are sized from
	 * maxQerLen in the constructor: unlike the u8 kernels the 16-bit tier has
	 * no 256-column admission bound, so a fixed stack array would not do. */
	int16_t *colIdx16;
	int16_t *qeBlk16;
	int32_t  qeBlk16Stride;

	int16_t *F16;
	int16_t *H16_0, *H16_1;
	int16_t *rowMax16;
	int32_t maxRefLen, maxQerLen;
	
	int g_qmax;
	int64_t sort1Ticks;
	int64_t setupTicks;
	int64_t swTicks;
	int64_t sort2Ticks;
};

#endif
