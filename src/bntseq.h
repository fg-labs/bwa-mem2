/* The MIT License

   Copyright (c) 2008 Genome Research Ltd (GRL).

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
*/

/* Contact: Heng Li <hli@jimmy.harvard.edu> */

#ifndef BWT_BNTSEQ_H
#define BWT_BNTSEQ_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <zlib.h>

#ifndef BWA_UBYTE
#define BWA_UBYTE
typedef uint8_t ubyte_t;
#endif

typedef struct {
	int64_t offset;
	int32_t len;
	int32_t n_ambs;
	uint32_t gi;
	int32_t is_alt;
	char *name, *anno;
} bntann1_t;

typedef struct {
	int64_t offset;
	int32_t len;
	char amb;
} bntamb1_t;

/* SAM-A3: bucket width (in bp, as a power-of-two shift) for the position->rid
 * acceleration table. One int32 table entry per 2^SHIFT bp of the packed
 * forward genome, plus one trailing bracket entry. At 2^14 (16 KiB) the hg38
 * (~3.1 Gbp) table is ~189k entries (~0.75 MB); the per-query search is bounded
 * by log2 of the number of contig starts inside a single bucket window (see
 * bns_pos2rid / bns_build_pos2rid). */
#define BNS_POS2RID_SHIFT 14

typedef struct {
	int64_t l_pac;
	int32_t n_seqs;
	uint32_t seed;
	bntann1_t *anns; // n_seqs elements
	int32_t n_holes;
	bntamb1_t *ambs; // n_holes elements
	FILE *fp_pac;
	/* SAM-A3: coarse position->rid bucket table, or NULL when not built.
	 * pos2rid_bucket[p >> BNS_POS2RID_SHIFT] holds the largest rid whose
	 * anns[rid].offset <= (bucket start); that entry and the next one bracket
	 * the exact rid, which bns_pos2rid resolves with a binary search over the
	 * bracket.
	 *
	 * Derived from anns[].offset and NOT serialized. Both ends uphold that:
	 * every writer that copies this struct wholesale detaches the pointer
	 * first so the image carries NULL (bwa_shm_pack_into), and
	 * every restore path that memcpy's it back (shm attach, mem-blob restore)
	 * resets this to NULL and rebuilds via bns_build_pos2rid before the first
	 * lookup. The reader-side reset is deliberately unconditional rather than
	 * relying on the writer: an image staged by an older or out-of-tree writer
	 * may still carry a heap address from that process, which is meaningless
	 * (and unfreeable) here. */
	int32_t *pos2rid_bucket;
} bntseq_t;

extern unsigned char nst_nt4_table[256];

/* Re-encode one sequence byte to its nt4 code. A value already in [0, limit)
 * passes through (limit is 4 for the plain path, 5 for callers that already hold
 * N=4); anything else is mapped through nst_nt4_table. The parameter is
 * unsigned char, so a signed `char` argument converts value-mod-256 at the call
 * boundary -- a high-bit byte (>= 0x80) can never sneak past the guard as a
 * negative value and be stored verbatim / index the table out of bounds. */
static inline unsigned char nst_nt4_decode(unsigned char c, unsigned char limit) {
    return c < limit ? c : nst_nt4_table[c];
}

/* 2-bit pac primitives — published so callers in other TUs (e.g. the
 * meth XM:Z slice path) can decode bases inline without going through
 * the malloc'ing bns_get_seq path. `pac` is the standard 2-bit packed
 * buffer (4 bases/byte, top-first); `l` is the global bp offset.
 *
 * Arguments must be side-effect-free: `l` is evaluated 3x in _get_pac
 * and 3x in _set_pac. The macro bodies are fully parenthesized so that
 * arithmetic on the result (e.g. `_get_pac(pac, i) + 1`) sees the
 * intended `& 3` masking instead of `& (3 + 1)`. */
#ifndef _get_pac
#define _get_pac(pac, l) ((((pac)[(l) >> 2]) >> ((~(l) & 3) << 1)) & 3)
#endif
#ifndef _set_pac
#define _set_pac(pac, l, c) ((pac)[(l) >> 2] |= (uint8_t)((c) << ((~(l) & 3) << 1)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

	void bns_dump(const bntseq_t *bns, const char *prefix);
	bntseq_t *bns_restore(const char *prefix);
	bntseq_t *bns_restore_core(const char *ann_filename, const char* amb_filename, const char* pac_filename);
	void bns_destroy(bntseq_t *bns);
	int64_t bns_fasta2bntseq(gzFile fp_fa, const char *prefix, int for_only);
	int bns_pos2rid(const bntseq_t *bns, int64_t pos_f);
	/* SAM-A3: build the coarse position->rid bucket table (see bntseq_t).
	 * Idempotent and safe on NULL: a no-op if bns is NULL, the table is
	 * already built, or l_pac/n_seqs are non-positive (bns_pos2rid then
	 * transparently falls back to the exact binary search). */
	void bns_build_pos2rid(bntseq_t *bns);
	int bns_cnt_ambi(const bntseq_t *bns, int64_t pos_f, int len, int *ref_id);
	/* Iterate over the bns->ambs intervals overlapping [pos_f, pos_f + len),
	 * in sorted (offset) order. `fn` receives the half-open interval
	 * [amb_st, amb_en) clipped to [pos_f, pos_f + len) plus the caller
	 * context. Iteration stops when fn returns nonzero. Returns the last
	 * fn return value (0 if no intervals visited). */
	typedef int (*bns_amb_visit_fn)(int64_t amb_st, int64_t amb_en, void *ctx);
	int bns_iter_ambi(const bntseq_t *bns, int64_t pos_f, int len,
	                  bns_amb_visit_fn fn, void *ctx);
	/* Decode forward/reverse-complement bases [beg, end) of `pac` into
	 * caller-provided `dst` (capacity must be >= max(end - beg, 0)).
	 * `dst` is filled with 2-bit-decoded bases (0..3 = A/C/G/T); ambiguous
	 * (N) positions are NOT marked here — callers that care consult
	 * bns->ambs separately (see bns_iter_ambi). On bridging the
	 * forward/reverse boundary `*len_out` is set to 0 and dst is
	 * untouched. Otherwise `*len_out == end - beg` (after end-clamp to
	 * `2 * l_pac` and beg-clamp to 0). */
	void bns_get_seq_into(int64_t l_pac, const uint8_t *pac,
	                      int64_t beg, int64_t end,
	                      uint8_t *dst, int64_t *len_out);
	void bns_fetch_seq_into(const bntseq_t *bns, const uint8_t *pac,
	                        int64_t *beg, int64_t mid, int64_t *end, int *rid,
	                        uint8_t *dst, int64_t *len_out);
	// Zero-copy v2 variants used by mem_chain2aln_across_reads_V2 and the
	// mem_matesw_batch_* path. Callers must NOT free the returned pointer.
	//
	// Two lifetime regimes, selected by `ref_string`:
	//   - ref_string != NULL: returns a pointer into the pre-unpacked `.0123`
	//     reference materialized at startup. Stable for the run; any number of
	//     windows may be held live at once.
	//   - ref_string == NULL (pac-fetch): the `.0123` was not loaded, so the
	//     window is reconstructed on demand from `pac` into a PER-THREAD scratch
	//     buffer. SINGLE LIVE WINDOW — the returned pointer aliases scratch that
	//     this thread's NEXT v2 call overwrites, so each caller must consume (or
	//     copy) the window before fetching again on the same thread. Do NOT cache
	//     multiple pac-fetch windows across calls.
	// The seqb scratch arg is currently unused by both v2 variants (kept for
	// signature parity with their original definition); pass any buffer or NULL.
	uint8_t *bns_get_seq_v2(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end,
	                        int64_t *len, uint8_t *ref_string, uint8_t *seqb);
	uint8_t *bns_fetch_seq_v2(const bntseq_t *bns, const uint8_t *pac,
	                          int64_t *beg, int64_t mid, int64_t *end, int *rid,
	                          uint8_t *ref_string, uint8_t *seqb);
	int bns_intv2rid(const bntseq_t *bns, int64_t rb, int64_t re);

#ifdef __cplusplus
}
#endif

static inline int64_t bns_depos(const bntseq_t *bns, int64_t pos, int *is_rev)
{
	return (*is_rev = (pos >= bns->l_pac))? (bns->l_pac<<1) - 1 - pos : pos;
}

#endif
