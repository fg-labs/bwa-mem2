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
         Heng Li <hli@jimmy.harvard.edu>
*****************************************************************************************/

#include <limits.h>
#include "bwa.h"
#include "bwamem.h"
#include "bntseq.h"
#include "kstring.h"


/***********************
 *** Extra functions ***
 ***********************/
#if 0
mem_alnreg_v mem_align1(const mem_opt_t *opt, const bwt_t *bwt, const bntseq_t *bns, const uint8_t *pac, int l_seq, const char *seq_)
{ // the difference from mem_align1_core() is that this routine: 1) calls mem_mark_primary_se(); 2) does not modify the input sequence
	extern mem_alnreg_v mem_align1_core(const mem_opt_t *opt, const bwt_t *bwt, const bntseq_t *bns, const uint8_t *pac, int l_seq, char *seq, void *buf);
	extern void mem_mark_primary_se(const mem_opt_t *opt, int n, mem_alnreg_t *a, int64_t id);
	mem_alnreg_v ar;
	char *seq;
	seq = (char*) malloc(l_seq);
	memcpy(seq, seq_, l_seq); // makes a copy of seq_
	ar = mem_align1_core(opt, bwt, bns, pac, l_seq, seq, 0);
	mem_mark_primary_se(opt, ar.n, ar.a, lrand48());
	free(seq);
	return ar;
}
#endif

static inline int get_pri_idx(double XA_drop_ratio, const mem_alnreg_t *a, int i)
{
	int k = a[i].secondary_all;
	if (k >= 0 && a[i].score >= a[k].score * XA_drop_ratio) return k;
	return -1;
}

// Okay, returning strings is bad, but this has happened a lot elsewhere. If I have time, I need serious code cleanup.
// When out_hn is non-NULL, it is set to a newly-allocated int[a->n] of per-primary
// hit counts (cnt[r]) so callers can emit the HN:i tag; caller owns the buffer.
char **mem_gen_alt(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_alnreg_v *a, int l_query, const char *query, int **out_hn, const char *meth_orig_query) // ONLY work after mem_mark_primary_se()
{
	int i, k, r, *cnt, tot;
	kstring_t *aln = 0, str = {0,0,0};
	char **XA = 0, *has_alt;
	/* mem_gen_alt runs once per read (twice per pair) and, for the common
	 * unique-mapper case, exits early at `tot == 0` after only allocating these
	 * scratch arrays. Keep the small ones off the heap: has_alt is a per-read
	 * transient sized to a->n (typically a handful of alnregs), so a stack
	 * buffer with a heap fallback for the rare large-a->n read avoids a
	 * calloc/free round-trip on every call. */
	enum { HAS_ALT_STACK_N = 256 };
	char has_alt_stack[HAS_ALT_STACK_N];

	cnt = (int *) calloc(a->n, sizeof(int));
    xassert(cnt != NULL || a->n == 0, "out of memory: cnt");  /* calloc(0) may return NULL; cnt unused when a->n==0 */
	has_alt = a->n <= HAS_ALT_STACK_N ? has_alt_stack : (char *) malloc(a->n);
    xassert(has_alt != NULL, "out of memory: has_alt");
	memset(has_alt, 0, a->n);
	for (i = 0, tot = 0; i < a->n; ++i) {
		r = get_pri_idx(opt->XA_drop_ratio, a->a, i);
		if (r >= 0) {
			++cnt[r], ++tot;
			if (a->a[i].is_alt) has_alt[r] = 1;
		}
	}
	// Publish per-primary hit counts for HN:i emission before the XA_hits cap
	// can truncate the XA string; HN should reflect what *would* have been in
	// XA, not what was actually emitted. cnt already holds exactly these counts
	// and is not needed after this function, so hand it to the caller directly
	// rather than allocating and copying a second array (caller owns/frees it).
	if (out_hn) *out_hn = cnt;
	if (tot == 0) goto end_gen_alt;
	aln = (kstring_t*) calloc(a->n, sizeof(kstring_t));
    xassert(aln != NULL, "out of memory: aln");
	for (i = 0; i < a->n; ++i) {
		mem_aln_t t;
		if ((r = get_pri_idx(opt->XA_drop_ratio, a->a, i)) < 0) continue;
		if (cnt[r] > opt->max_XA_hits_alt || (!has_alt[r] && cnt[r] > opt->max_XA_hits)) continue;
		/* Pass meth_orig_query so each XA sub-entry regenerates under the SAME
		 * NM/MD policy as the primary record. Omitting it silently falls back to
		 * the literal predicate, so one record would report matrix-derived NM:i:
		 * on the primary and conversion-counting NM inside its own XA:Z. */
		t = mem_reg2aln(opt, bns, pac, l_query, query, &a->a[i], meth_orig_query);
		str.l = 0;
		kputs(bns->anns[t.rid].name, &str);
		kputc(',', &str); kputc("+-"[t.is_rev], &str); kputl(t.pos + 1, &str);
		kputc(',', &str);
		//{ //deb block
		//	char *tt = strstr(str.s, "-161728"); // delete,debug
		//	if (tt) {
		//		printf("Before %s, ncigar: %d\n", str.s, t.n_cigar);
		//		for (k = 0; k < t.n_cigar; ++k) printf("%d\n", t.cigar[k]>>4);
		//	}
		//}
		for (k = 0; k < t.n_cigar; ++k) {
			kputw(t.cigar[k]>>4, &str);
			kputc("MIDSHN"[t.cigar[k]&0xf], &str);
		}
		kputc(',', &str); kputw(t.NM, &str);
		if (opt->flag & MEM_F_XB) {
			kputc(',', &str); kputw(t.score, &str);
			kputc(',', &str); kputw(t.mapq, &str);
		}
		kputc(';', &str);
		free(t.cigar);
		kputsn(str.s, str.l, &aln[r]);
	}
	XA = (char**) calloc(a->n, sizeof(char*));
    xassert(XA != NULL, "out of memory: XA");
	for (k = 0; k < a->n; ++k)
		XA[k] = aln[k].s;

end_gen_alt:
	if (has_alt != has_alt_stack) free(has_alt);
	if (!out_hn) free(cnt); // else cnt was handed to *out_hn; caller frees it
	free(aln); free(str.s);
	return XA;
}
