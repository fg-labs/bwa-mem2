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

   Modified Copyright (C) 2020 Intel Corporation, Heng Li.
   Contacts: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>;
   Heng Li <hli@jimmy.harvard.edu> 
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
/* SIMD 2-bit->byte unpack fast path in bns_get_seq_into. Written once in SSE2/SSSE3;
 * on ARM it runs via sse2neon (as the rest of the aligner's SSE does), natively on x86.
 * Requires SSSE3 (pshufb) for the reverse strand; the x86 baseline is -mssse3, and
 * sse2neon provides it. Any build below SSSE3 falls back to the scalar LUT (no regression). */
#if defined(__aarch64__) || defined(__ARM_NEON)
#include "simd_compat.h"
#define BNS_SIMD_UNPACK 1
#elif defined(__SSSE3__)
#include <immintrin.h>
#define BNS_SIMD_UNPACK 1
#endif
#include "bntseq.h"
#include "utils.h"
#include "macro.h"

#include "kseq.h"
KSEQ_DECLARE(gzFile)

#include "khash.h"
KHASH_MAP_INIT_STR(str, int)

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

extern uint64_t tprof[LIM_R][LIM_C];

/* Build "<prefix><suffix>" into `out` (sized `outsz`), aborting via err_fatal
 * if the result would exceed `outsz`. Used by the bns_* file-path helpers
 * below; replaces the prior strcpy_s/strcat_s calls. */
static void bns_build_path(char *out, size_t outsz, const char *prefix, const char *suffix)
{
	int n = snprintf(out, outsz, "%s%s", prefix, suffix);
	if (n < 0 || (size_t)n >= outsz)
		err_fatal(__func__, "path too long for prefix '%s' (suffix '%s')", prefix, suffix);
}

unsigned char nst_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

void bns_dump(const bntseq_t *bns, const char *prefix)
{
	char str[PATH_MAX];
	FILE *fp;
	int i;
	{ // dump .ann
		bns_build_path(str, sizeof(str), prefix, ".ann");
		fp = xopen(str, "w");
		err_fprintf(fp, "%lld %d %u\n", (long long)bns->l_pac, bns->n_seqs, bns->seed);
		for (i = 0; i != bns->n_seqs; ++i) {
			bntann1_t *p = bns->anns + i;
			err_fprintf(fp, "%d %s", p->gi, p->name);
			if (p->anno[0]) err_fprintf(fp, " %s\n", p->anno);
			else err_fprintf(fp, "\n");
			err_fprintf(fp, "%lld %d %d\n", (long long)p->offset, p->len, p->n_ambs);
		}
		err_fflush(fp);
		err_fclose(fp);
	}
	{ // dump .amb
		bns_build_path(str, sizeof(str), prefix, ".amb");
		fp = xopen(str, "w");
		err_fprintf(fp, "%lld %d %u\n", (long long)bns->l_pac, bns->n_seqs, bns->n_holes);
		for (i = 0; i != bns->n_holes; ++i) {
			bntamb1_t *p = bns->ambs + i;
			err_fprintf(fp, "%lld %d %c\n", (long long)p->offset, p->len, p->amb);
		}
		err_fflush(fp);
		err_fclose(fp);
	}
}

bntseq_t *bns_restore_core(const char *ann_filename, const char* amb_filename, const char* pac_filename)
{
	char str[8193];
	FILE *fp;
	const char *fname;
	bntseq_t *bns;
	long long xx;
	int i;
	int scanres;
	bns = (bntseq_t*)calloc(1, sizeof(bntseq_t));
	xassert(bns != NULL, "out of memory: bns");
	{ // read .ann
		fp = xopen(fname = ann_filename, "r");
		scanres = fscanf(fp, "%lld%d%u", &xx, &bns->n_seqs, &bns->seed);
		assert(bns->n_seqs >= 0 && bns->n_seqs <= INT_MAX);
		if (scanres != 3) goto badread;
		bns->l_pac = xx;
		bns->anns = (bntann1_t*)calloc(bns->n_seqs, sizeof(bntann1_t));
        // A zero-sequence .ann is degenerate but not an allocation failure:
        // calloc(0, ...) is free to return NULL, so only a non-empty request
        // coming back NULL means we are out of memory.
        xassert(bns->n_seqs == 0 || bns->anns != NULL, "out of memory: bns->anns");
		for (i = 0; i < bns->n_seqs; ++i) {
			bntann1_t *p = bns->anns + i;
			char *q = str;
			int c;
			// read gi and sequence name
			scanres = fscanf(fp, "%u%8192s", &p->gi, str);
			if (scanres != 2) goto badread;
			p->name = strdup(str);
			// read fasta comments 
			while (q - str < sizeof(str) - 1 && (c = fgetc(fp)) != '\n' && c != EOF) *q++ = c;
			while (c != '\n' && c != EOF) c = fgetc(fp);
			if (c == EOF) {
				scanres = EOF;
				goto badread;
			}
			*q = 0;
			assert(strlen(str) < 8192);
			if (q - str > 1 && strcmp(str, " (null)") != 0) p->anno = strdup(str + 1); // skip leading space
			else p->anno = strdup("");
			// read the rest
			scanres = fscanf(fp, "%lld%d%d", &xx, &p->len, &p->n_ambs);
			if (scanres != 3) goto badread;
			p->offset = xx;
		}
		err_fclose(fp);
	}
	{ // read .amb
		int64_t l_pac;
		int32_t n_seqs;
		fp = xopen(fname = amb_filename, "r");
		scanres = fscanf(fp, "%lld%d%d", &xx, &n_seqs, &bns->n_holes);
		assert(bns->n_holes >= 0 && bns->n_holes <= INT_MAX);
		if (scanres != 3) goto badread;
		l_pac = xx;
		xassert(l_pac == bns->l_pac && n_seqs == bns->n_seqs, "inconsistent .ann and .amb files.");
        if(bns->n_holes){
            bns->ambs = (bntamb1_t*)calloc(bns->n_holes, sizeof(bntamb1_t));
            xassert(bns->ambs != NULL, "out of memory: bns->ambs");
        }
        else{
            bns->ambs = 0;
        }
		for (i = 0; i < bns->n_holes; ++i) {
			bntamb1_t *p = bns->ambs + i;
			scanres = fscanf(fp, "%lld%d%8192s", &xx, &p->len, str);
			if (scanres != 3) goto badread;
			p->offset = xx;
			p->amb = str[0];
		}
		err_fclose(fp);
	}
	{ // open .pac
		bns->fp_pac = xopen(pac_filename, "rb");
	}
	bns_build_pos2rid(bns); // SAM-A3: build the position->rid acceleration table once
	return bns;

 badread:
	if (EOF == scanres) {
		err_fatal(__func__, "Error reading %s : %s\n", fname, ferror(fp) ? strerror(errno) : "Unexpected end of file");
	}
	err_fatal(__func__, "Parse error reading %s\n", fname);
}

bntseq_t *bns_restore(const char *prefix)
{  
	char ann_filename[PATH_MAX], amb_filename[PATH_MAX], pac_filename[PATH_MAX], alt_filename[PATH_MAX];
	FILE *fp;
	bntseq_t *bns;
	bns_build_path(ann_filename, sizeof(ann_filename), prefix, ".ann");
	bns_build_path(amb_filename, sizeof(amb_filename), prefix, ".amb");
	bns_build_path(pac_filename, sizeof(pac_filename), prefix, ".pac");
	bns = bns_restore_core(ann_filename, amb_filename, pac_filename);
	if (bns == 0) return 0;
	bns_build_path(alt_filename, sizeof(alt_filename), prefix, ".alt");
	if ((fp = fopen(alt_filename, "r")) != 0) { // read .alt file if present
		char str[1024];
		khash_t(str) *h;
		int c, i, absent;
		khint_t k;
		h = kh_init(str);
        xassert(h != NULL, "out of memory: bns name hash");
		for (i = 0; i < bns->n_seqs; ++i) {
			k = kh_put(str, h, bns->anns[i].name, &absent);
			kh_val(h, k) = i;
		}
		i = 0;
		int truncated = 0;
		while ((c = fgetc(fp)) != EOF) {
			if (c == '\t' || c == '\n' || c == '\r') {
				str[i] = 0;
				if (!truncated && str[0] != '@') {
					k = kh_get(str, h, str);
					if (k != kh_end(h))
						bns->anns[kh_val(h, k)].is_alt = 1;
				}
				while (c != '\n' && c != EOF) c = fgetc(fp);
				i = 0;
				truncated = 0;
			} else if (i + 1 < (int)sizeof(str)) {
				str[i++] = c;
			} else {
				// Name exceeds sizeof(str)-1; drop remaining chars to avoid
				// overflowing the stack buffer, and skip the hash lookup so
				// a truncated prefix can't accidentally match a different
				// contig that shares those leading bytes.
				truncated = 1;
			}
		}
		kh_destroy(str, h);
		fclose(fp);
	}
	return bns;
}

void bns_destroy(bntseq_t *bns)
{
	if (bns == 0) return;
	else {
		int i;
		if (bns->fp_pac) err_fclose(bns->fp_pac);
		free(bns->pos2rid_bucket); // SAM-A3: free(NULL) is a no-op when never built
		free(bns->ambs);
		for (i = 0; i < bns->n_seqs; ++i) {
			free(bns->anns[i].name);
			free(bns->anns[i].anno);
		}
		free(bns->anns);
		free(bns);
	}
}

/* _get_pac and _set_pac now published in bntseq.h so meth_orig_ref.cpp
 * (and any other in-tree caller that needs inline 2-bit decode) can use
 * them without re-defining. */

static uint8_t *add1(const kseq_t *seq, bntseq_t *bns, uint8_t *pac, int64_t *m_pac, int *m_seqs, int *m_holes, bntamb1_t **q)
{
	bntann1_t *p;
	int i, lasts;
	if (bns->n_seqs == *m_seqs) {
		*m_seqs <<= 1;
		/* Grow via a temp so a failed realloc neither leaks the old block nor
		 * leaves the live pointer NULL (deref'd below). */
		bntann1_t *tmp = (bntann1_t*)realloc(bns->anns, *m_seqs * sizeof(bntann1_t));
		xassert(tmp != NULL, "out of memory: bns->anns");
		bns->anns = tmp;
	}
	p = bns->anns + bns->n_seqs;
	p->name = strdup((char*)seq->name.s);
	p->anno = seq->comment.l > 0? strdup((char*)seq->comment.s) : strdup("(null)");
	p->gi = 0; p->len = seq->seq.l;
	p->offset = (bns->n_seqs == 0)? 0 : (p-1)->offset + (p-1)->len;
	p->n_ambs = 0;
	for (i = lasts = 0; i < seq->seq.l; ++i) {
		int c = nst_nt4_table[(unsigned char)seq->seq.s[i]];
		if (c >= 4) { // N
			if (lasts == seq->seq.s[i]) { // contiguous N
				++(*q)->len;
			} else {
				if (bns->n_holes == *m_holes) {
					(*m_holes) <<= 1;
					/* temp + check: a failed realloc-onto-self would leak the old
					 * block and NULL bns->ambs, then *q below derefs it. */
					bntamb1_t *tmp = (bntamb1_t*)realloc(bns->ambs, (*m_holes) * sizeof(bntamb1_t));
					xassert(tmp != NULL, "out of memory: bns->ambs");
					bns->ambs = tmp;
				}
				*q = bns->ambs + bns->n_holes;
				(*q)->len = 1;
				(*q)->offset = p->offset + i;
				(*q)->amb = seq->seq.s[i];
				++p->n_ambs;
				++bns->n_holes;
			}
		}
		lasts = seq->seq.s[i];
		{ // fill buffer
			if (c >= 4) c = lrand48()&3;
			if (bns->l_pac == *m_pac) { // double the pac size
				*m_pac <<= 1;
				/* temp + check: without it a failed realloc leaks the old pac and
				 * the memset below derefs the NULL it left behind. */
				uint8_t *tmp = (uint8_t*) realloc(pac, *m_pac/4);
				if (tmp == NULL) { perror("Reallocation of pac failed"); exit(EXIT_FAILURE); }
				pac = tmp;
				memset(pac + bns->l_pac/4, 0, (*m_pac - bns->l_pac)/4);
			}
			_set_pac(pac, bns->l_pac, c);
			++bns->l_pac;
		}
	}
	++bns->n_seqs;
	return pac;
}

int64_t bns_fasta2bntseq(gzFile fp_fa, const char *prefix, int for_only)
{
	extern void seq_reverse(int len, ubyte_t *seq, int is_comp); // in bwaseqio.c
	kseq_t *seq;
	char name[PATH_MAX];
	bntseq_t *bns;
	uint8_t *pac = 0;
	int32_t m_seqs, m_holes;
	int64_t ret = -1, m_pac, l;
	bntamb1_t *q;
	FILE *fp;

	// initialization
	seq = kseq_init(fp_fa);
	bns = (bntseq_t*)calloc(1, sizeof(bntseq_t));
    xassert(bns != NULL, "out of memory: bns");
	bns->seed = 11; // fixed seed for random generator
	srand48(bns->seed);
	m_seqs = m_holes = 8; m_pac = 0x10000;
	bns->anns = (bntann1_t*)calloc(m_seqs, sizeof(bntann1_t));
    xassert(bns->anns != NULL, "out of memory: bns->anns");
	bns->ambs = (bntamb1_t*)calloc(m_holes, sizeof(bntamb1_t));
    xassert(bns->ambs != NULL, "out of memory: bns->ambs");
	pac = (uint8_t*) calloc(m_pac/4, 1);
	if (pac == NULL) { perror("Allocation of pac failed"); exit(EXIT_FAILURE); }
	q = bns->ambs;
	bns_build_path(name, sizeof(name), prefix, ".pac");
	fp = xopen(name, "wb");
	// read sequences
	while (kseq_read(seq) >= 0) pac = add1(seq, bns, pac, &m_pac, &m_seqs, &m_holes, &q);
	if (!for_only) { // add the reverse complemented sequence
		m_pac = (bns->l_pac * 2 + 3) / 4 * 4;
		/* temp + check so a failed grow doesn't leak the old pac before aborting. */
		uint8_t *tmp = (uint8_t*) realloc(pac, m_pac/4);
		if (tmp == NULL) { perror("Reallocation of pac failed"); exit(EXIT_FAILURE); }
		pac = tmp;
		memset(pac + (bns->l_pac+3)/4, 0, (m_pac - (bns->l_pac+3)/4*4) / 4);
		for (l = bns->l_pac - 1; l >= 0; --l, ++bns->l_pac)
			_set_pac(pac, bns->l_pac, 3-_get_pac(pac, l));
	}
	ret = bns->l_pac;
	{ // finalize .pac file
		ubyte_t ct;
		err_fwrite(pac, 1, (bns->l_pac>>2) + ((bns->l_pac&3) == 0? 0 : 1), fp);
		// the following codes make the pac file size always (l_pac/4+1+1)
		if (bns->l_pac % 4 == 0) {
			ct = 0;
			err_fwrite(&ct, 1, 1, fp);
		}
		ct = bns->l_pac % 4;
		err_fwrite(&ct, 1, 1, fp);
		// close .pac file
		err_fflush(fp);
		err_fclose(fp);
	}
	bns_dump(bns, prefix);
	bns_destroy(bns);
	kseq_destroy(seq);
	free(pac);
	return ret;
}

int bwa_fa2pac(int argc, char *argv[])
{
	int c, for_only = 0;
	gzFile fp;
	while ((c = getopt(argc, argv, "f")) >= 0) {
		switch (c) {
			case 'f': for_only = 1; break;
		}
	}
	if (argc == optind) {
		fprintf(stderr, "Usage: bwa fa2pac [-f] <in.fasta> [<out.prefix>]\n");
		return 1;
	}
	fp = xzopen(argv[optind], "r");
	bns_fasta2bntseq(fp, (optind+1 < argc)? argv[optind+1] : argv[optind], for_only);
	err_gzclose(fp);
	return 0;
}

/* SAM-A3: build the coarse position->rid bucket table. bucket[b] is the
 * largest rid whose anns[rid].offset <= (b << BNS_POS2RID_SHIFT), i.e. the
 * contig that covers the first position of bucket b. Because anns[].offset is
 * monotonically increasing, a single linear merge over contigs fills the whole
 * table in O(n_buckets + n_seqs).
 *
 * The table carries one extra trailing entry past the last in-range bucket so
 * that bns_pos2rid can always read bucket[b + 1]; that pair brackets the answer
 * and is what bounds the lookup (see bns_pos2rid). */
void bns_build_pos2rid(bntseq_t *bns)
{
	int64_t n_buckets, b;
	int rid;
	if (bns == NULL || bns->pos2rid_bucket != NULL) return; // NULL or already built
	if (bns->l_pac <= 0 || bns->n_seqs <= 0) return;        // leave NULL: fall back to binary search
	n_buckets = (bns->l_pac >> BNS_POS2RID_SHIFT) + 1;
	bns->pos2rid_bucket = (int32_t*)calloc((size_t)n_buckets + 1, sizeof(int32_t));
	if (bns->pos2rid_bucket == NULL) return;                // OOM: fall back to binary search
	rid = 0;
	for (b = 0; b <= n_buckets; ++b) { // <=: fill the trailing bracket entry too
		int64_t pos = b << BNS_POS2RID_SHIFT;
		while (rid + 1 < bns->n_seqs && bns->anns[rid + 1].offset <= pos) ++rid;
		bns->pos2rid_bucket[b] = rid;
	}
}

int bns_pos2rid(const bntseq_t *bns, int64_t pos_f)
{
	if (pos_f >= bns->l_pac) return -1;
	if (bns->pos2rid_bucket != NULL) { // SAM-A3: O(1) bucket bracket + bounded search
		int lo, hi;
		// Negative pos_f clamps to bucket 0, matching the binary-search branch
		// below (which returns rid 0 for any pos_f < anns[0].offset).
		int64_t b = pos_f < 0? 0 : (pos_f >> BNS_POS2RID_SHIFT);
		// bucket[b] is the last contig starting at or before this bucket's
		// first position, so anns[bucket[b]].offset <= pos_f; bucket[b+1] is
		// the last one starting at or before the NEXT bucket's first position,
		// which pos_f is below. So the answer is bracketed by [lo, hi], and
		// hi - lo is just the number of contig starts inside this bucket
		// window. The common case (no contig starts inside the window) has
		// lo == hi and returns without touching anns[] at all.
		lo = bns->pos2rid_bucket[b];
		hi = bns->pos2rid_bucket[b + 1];
		// Narrow to the largest rid with anns[rid].offset <= pos_f. This is
		// the exact same predicate the binary search resolves, so the returned
		// rid is byte-identical (including on-offset boundaries and, since
		// offsets are strictly increasing, ties resolve to the largest index).
		// Searching rather than scanning keeps a reference packed with many
		// sub-bucket-width contigs (panels, transcriptomes) logarithmic in the
		// bucket's contig count instead of linear in it.
		while (lo < hi) {
			int mid = lo + ((hi - lo + 1) >> 1); // round up: mid > lo, so this terminates
			if (bns->anns[mid].offset <= pos_f) lo = mid;
			else hi = mid - 1;
		}
		return lo;
	}
	{ // fallback: exact original binary search (bns without a built table)
		int left, mid, right;
		left = 0; mid = 0; right = bns->n_seqs;
		while (left < right) { // binary search
			mid = (left + right) >> 1;
			if (pos_f >= bns->anns[mid].offset) {
				if (mid == bns->n_seqs - 1) break;
				if (pos_f < bns->anns[mid+1].offset) break; // bracketed
				left = mid + 1;
			} else right = mid;
		}
		return mid;
	}
}

int bns_intv2rid(const bntseq_t *bns, int64_t rb, int64_t re)
{
	int is_rev, rid_b, rid_e;
	if (rb < bns->l_pac && re > bns->l_pac) return -2;
	assert(rb <= re);
	rid_b = bns_pos2rid(bns, bns_depos(bns, rb, &is_rev));
	rid_e = rb < re? bns_pos2rid(bns, bns_depos(bns, re - 1, &is_rev)) : rid_b;
	return rid_b == rid_e? rid_b : -1;
}

/* Iterate ambs intervals overlapping [pos_f, pos_f + len). Sorted by
 * offset. The pre-existing bns_cnt_ambi did binary-search-then-break-on-
 * first-overlap, which under-counts when a window legitimately overlaps
 * two intervals; folding the count through this iterator fixes that as
 * a side effect. */
int bns_iter_ambi(const bntseq_t *bns, int64_t pos_f, int len,
                  bns_amb_visit_fn fn, void *ctx)
{
	int left, mid, right;
	int64_t window_end = pos_f + len;
	int rc = 0;
	if (bns == NULL || bns->n_holes == 0 || len <= 0 || fn == NULL) return 0;

	/* Lower-bound on offset+len: find the first amb whose end-position
	 * is greater than pos_f (i.e. whose interval can possibly overlap). */
	left = 0; right = bns->n_holes;
	while (left < right) {
		mid = (left + right) >> 1;
		if ((int64_t)(bns->ambs[mid].offset + bns->ambs[mid].len) <= pos_f) left = mid + 1;
		else right = mid;
	}

	/* Walk forward through overlapping ambs until we run off the end. */
	for (int i = left; i < bns->n_holes; ++i) {
		const bntamb1_t *a = &bns->ambs[i];
		if ((int64_t)a->offset >= window_end) break;
		int64_t st = (int64_t)a->offset > pos_f ? (int64_t)a->offset : pos_f;
		int64_t en = (int64_t)(a->offset + a->len) < window_end
		             ? (int64_t)(a->offset + a->len) : window_end;
		if (st >= en) continue;
		rc = fn(st, en, ctx);
		if (rc != 0) return rc;
	}
	return rc;
}

namespace {
struct cnt_ambi_ctx { int nn; };
int cnt_ambi_visit(int64_t st, int64_t en, void *ctx) {
	((cnt_ambi_ctx *)ctx)->nn += (int)(en - st);
	return 0; /* keep iterating */
}
}

int bns_cnt_ambi(const bntseq_t *bns, int64_t pos_f, int len, int *ref_id)
{
	if (ref_id) *ref_id = bns_pos2rid(bns, pos_f);
	cnt_ambi_ctx ctx = { 0 };
	bns_iter_ambi(bns, pos_f, len, cnt_ambi_visit, &ctx);
	return ctx.nn;
}

/* Byte->4-base lookup tables for the 2-bit packed `.pac`.
 *
 * A packed byte holds 4 bases, most-significant-first, matching `_get_pac`:
 * for a global position `p`, the base occupies bits `(~p & 3) << 1 .. +1`, so
 * the base at intra-byte position `j` (j == p & 3) is `(byte >> ((3-j)<<1)) & 3`.
 *
 *   fwd[byte][j] = base at intra-byte position j  (forward order, j = 0..3)
 *   rev[byte][j] = 3 - base at intra-byte position (3-j)  (reverse strand:
 *                  bases emitted high-position-first and base-complemented,
 *                  i.e. rev[byte][j] = 3 - ((byte >> (j<<1)) & 3))
 *
 * The tables are `const` and populated exactly once at first use; the
 * function-local static gives C++11 thread-safe initialization with no
 * per-call rebuild.
 */
namespace {
struct Pac2Nt4Lut {
	uint8_t fwd[256][4];
	uint8_t rev[256][4];
	Pac2Nt4Lut() {
		for (int b = 0; b < 256; ++b) {
			for (int j = 0; j < 4; ++j) {
				fwd[b][j] = (uint8_t)((b >> ((3 - j) << 1)) & 3);
				rev[b][j] = (uint8_t)(3 - ((b >> (j << 1)) & 3));
			}
		}
	}
};
const Pac2Nt4Lut &pac2nt4_lut() {
	static const Pac2Nt4Lut lut;
	return lut;
}
}  // namespace

#ifdef BNS_SIMD_UNPACK
/* Expand 16 packed bytes (64 2-bit bases) into four __m128i, then interleave to base order.
 * In-byte positions are bits [7:6],[5:4],[3:2],[1:0] (MSB-first, matching _get_pac); a
 * 16-bit shift + &3 isolates each position per byte (cross-byte bits are masked away).
 * The unpack ladder interleaves the four position vectors to base0,base1,base2,base3 per
 * byte, so o0..o3 hold bases [0..63] in ascending order. */
static inline void bns_expand4(__m128i v, __m128i *o0, __m128i *o1, __m128i *o2, __m128i *o3)
{
	const __m128i m3 = _mm_set1_epi8(3);
	__m128i b0 = _mm_and_si128(_mm_srli_epi16(v, 6), m3), b1 = _mm_and_si128(_mm_srli_epi16(v, 4), m3);
	__m128i b2 = _mm_and_si128(_mm_srli_epi16(v, 2), m3), b3 = _mm_and_si128(v, m3);
	__m128i lo01 = _mm_unpacklo_epi8(b0, b1), hi01 = _mm_unpackhi_epi8(b0, b1);
	__m128i lo23 = _mm_unpacklo_epi8(b2, b3), hi23 = _mm_unpackhi_epi8(b2, b3);
	*o0 = _mm_unpacklo_epi16(lo01, lo23); *o1 = _mm_unpackhi_epi16(lo01, lo23);
	*o2 = _mm_unpacklo_epi16(hi01, hi23); *o3 = _mm_unpackhi_epi16(hi01, hi23);
}
#endif

void bns_get_seq_into(int64_t l_pac, const uint8_t *pac,
                      int64_t beg, int64_t end,
                      uint8_t *dst, int64_t *len_out)
{
	if (end < beg) end ^= beg, beg ^= end, end ^= beg; // if end is smaller, swap
	if (end > l_pac<<1) end = l_pac<<1;
	if (beg < 0) beg = 0;
	if (beg >= l_pac || end <= l_pac) {
		int64_t k, l = 0;
		*len_out = end - beg;
		if (beg >= l_pac) { // reverse strand
			int64_t beg_f = (l_pac<<1) - 1 - end;
			int64_t end_f = (l_pac<<1) - 1 - beg;
			const uint8_t (*rev)[4] = pac2nt4_lut().rev;
			k = end_f;
			// leading partial bases until k is the last base of its byte
			for (; k > beg_f && (k & 3) != 3; --k) dst[l++] = 3 - _get_pac(pac, k);
#ifdef BNS_SIMD_UNPACK
			// SIMD reverse: forward-expand the 16-byte block [k-63, k], then reverse the byte
			// order (pshufb) and complement (^3). k is the last base of its byte, so the block
			// stays byte-aligned; byte-identical to the rev-LUT below (verified over 200k windows).
			// The load reads pac[(k>>2)-15 .. k>>2] -- the same span the scalar loop below reads;
			// the guard (k-64 >= beg_f, and beg_f >= -1 after the end<=2*l_pac clamp) keeps the low
			// index >= 0, so it never reads before pac[0].
			{
				const __m128i rmask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
				const __m128i m3 = _mm_set1_epi8(3);
				for (; k - 64 >= beg_f; k -= 64) {
					__m128i v = _mm_loadu_si128((const __m128i *)(pac + ((k >> 2) - 15))), o0, o1, o2, o3;
					bns_expand4(v, &o0, &o1, &o2, &o3);   // o0..o3 = forward bases [k-63, k] ascending
					_mm_storeu_si128((__m128i *)(dst + l),      _mm_xor_si128(_mm_shuffle_epi8(o3, rmask), m3));
					_mm_storeu_si128((__m128i *)(dst + l + 16), _mm_xor_si128(_mm_shuffle_epi8(o2, rmask), m3));
					_mm_storeu_si128((__m128i *)(dst + l + 32), _mm_xor_si128(_mm_shuffle_epi8(o1, rmask), m3));
					_mm_storeu_si128((__m128i *)(dst + l + 48), _mm_xor_si128(_mm_shuffle_epi8(o0, rmask), m3));
					l += 64;
				}
			}
#endif
			// whole packed bytes: expand 4 rev-complemented bases per indexed load
			for (; k - 4 >= beg_f; k -= 4) {
				const uint8_t *q = rev[pac[k >> 2]];
				dst[l++] = q[0]; dst[l++] = q[1]; dst[l++] = q[2]; dst[l++] = q[3];
			}
			// trailing partial bases
			for (; k > beg_f; --k) dst[l++] = 3 - _get_pac(pac, k);
		} else { // forward strand
			const uint8_t (*fwd)[4] = pac2nt4_lut().fwd;
			k = beg;
			// leading partial bases until k is byte-aligned
			for (; k < end && (k & 3) != 0; ++k) dst[l++] = _get_pac(pac, k);
#ifdef BNS_SIMD_UNPACK
			// SIMD: 16 packed bytes -> 64 bases per step; byte-identical to the LUT below
			// (verified vs it over 200k random windows), ~3x faster on ~150 bp windows.
			for (; k + 64 <= end; k += 64) {
				__m128i v = _mm_loadu_si128((const __m128i *)(pac + (k >> 2))), o0, o1, o2, o3;
				bns_expand4(v, &o0, &o1, &o2, &o3);
				_mm_storeu_si128((__m128i *)(dst + l),      o0);
				_mm_storeu_si128((__m128i *)(dst + l + 16), o1);
				_mm_storeu_si128((__m128i *)(dst + l + 32), o2);
				_mm_storeu_si128((__m128i *)(dst + l + 48), o3);
				l += 64;
			}
#endif
			// whole packed bytes: expand 4 bases per indexed load
			for (; k + 4 <= end; k += 4) {
				const uint8_t *q = fwd[pac[k >> 2]];
				dst[l++] = q[0]; dst[l++] = q[1]; dst[l++] = q[2]; dst[l++] = q[3];
			}
			// trailing partial bases
			for (; k < end; ++k) dst[l++] = _get_pac(pac, k);
		}
	} else {
		*len_out = 0; // if bridging the forward-reverse boundary, return nothing
	}
}

void bns_fetch_seq_into(const bntseq_t *bns, const uint8_t *pac,
                        int64_t *beg, int64_t mid, int64_t *end, int *rid,
                        uint8_t *dst, int64_t *len_out)
{
	int64_t far_beg, far_end;
	int is_rev;

	if (*end < *beg) *end ^= *beg, *beg ^= *end, *end ^= *beg; // if end is smaller, swap
	assert(*beg <= mid && mid < *end);

	*rid = bns_pos2rid(bns, bns_depos(bns, mid, &is_rev));
	far_beg = bns->anns[*rid].offset;
	far_end = far_beg + bns->anns[*rid].len;
	if (is_rev) { // flip to the reverse strand
		int64_t tmp = far_beg;
		far_beg = (bns->l_pac<<1) - far_end;
		far_end = (bns->l_pac<<1) - tmp;
	}
	*beg = *beg > far_beg? *beg : far_beg;
	*end = *end < far_end? *end : far_end;

	bns_get_seq_into(bns->l_pac, pac, *beg, *end, dst, len_out);

	if (*end - *beg != *len_out) {
		fprintf(stderr, "[E::%s] begin=%ld, mid=%ld, end=%ld, len=%ld, rid=%d, far_beg=%ld, far_end=%ld\n",
				__func__, (long)*beg, (long)mid, (long)*end, (long)(*len_out), *rid, (long)far_beg, (long)far_end);
	}
	assert(*end - *beg == *len_out); // assertion failure should never happen
}

// pac-fetch scratch buffer (used by bns_get_seq_v2's ref_string==NULL path).
// Thread-local so each worker reconstructs windows into its own buffer; the
// destructor frees it at thread exit (via __cxa_thread_atexit), so a worker
// thread's buffer is not leaked when the thread terminates — without it,
// LeakSanitizer flags one live allocation per worker thread.
namespace {
struct PacFetchScratch {
    uint8_t *buf = nullptr;
    int64_t  cap = 0;
    int64_t  prev_len = 0;  // length of the window currently held in buf (debug poison only)
    ~PacFetchScratch() { free(buf); }
};
}  // namespace

// Zero-copy v2 variants. Identical semantics to bns_get_seq / bns_fetch_seq
// except they return a pointer into the caller-supplied `ref_string` (the
// .0123 reference materialized at startup) rather than a malloc'd copy. The
// `seqb` parameter is retained for signature parity with an earlier draft and
// is currently unused.
uint8_t *bns_get_seq_v2(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end,
                        int64_t *len, uint8_t *ref_string, uint8_t *seqb)
{
	uint8_t *seq = 0;
	if (ref_string == NULL) {
		/* pac-fetch: the unpacked `.0123` was not loaded. Reconstruct the window
		 * by unpacking the ORIGINAL `.pac` on demand (bns_get_seq_into: forward
		 * 2-bit unpack; reverse-strand window = reverse + complement) — byte-
		 * identical to the `.0123` it replaces.
		 *
		 * CONTRACT (single live window): the returned pointer aliases a per-thread
		 * scratch that this thread's NEXT call overwrites. Every caller must
		 * consume (or copy) the window before fetching again on the same thread —
		 * exactly the zero-copy `.0123` contract. All current consumers honor it
		 * (extension consumes rseq within the chain iteration; both mate-rescue
		 * sites copy the window into seqBufRef immediately). This is a convention,
		 * NOT enforceable as an in-function assert (the function cannot observe a
		 * caller's later deref). The BWA_MEM3_DEBUG_POISON poison-fill below is a
		 * best-effort detector: it 0xFF's the prior window so a stale read trips the byte-
		 * identity / BAM-cmp golden gate rather than silently mis-scoring.
		 * seqb (the caller's scratch) is intentionally left untouched. */
		static thread_local PacFetchScratch t_pf;
		int64_t b = beg, e = end;
		if (e < b) { int64_t t = b; b = e; e = t; }
		if (e > l_pac<<1) e = l_pac<<1;
		if (b < 0) b = 0;
		int64_t need = e - b;
		if (need <= 0) { *len = 0; return 0; }
		/* A window bridging the forward/reverse boundary yields nothing (the
		 * legacy `.0123` path returns empty without allocating; bns_get_seq_into
		 * likewise sets len=0 and writes no bytes). Short-circuit BEFORE the
		 * realloc so a bad bridge query can't grow t_pf.buf toward the doubled
		 * reference (~6.4 GB on hg38) just to return an empty window. */
		if (b < l_pac && e > l_pac) { *len = 0; return 0; }
		if (need > t_pf.cap) {
			/* Grow via a temp so a failed realloc neither leaks the old buffer nor
			 * leaves the buffer NULL with cap > 0 (which would deref NULL below).
			 * Matches the perror+exit idiom used for the .pac realloc in this file. */
			uint8_t *nb = (uint8_t*)realloc(t_pf.buf, (size_t)need);
			if (nb == NULL) { perror("Reallocation of pac-fetch buffer failed"); exit(EXIT_FAILURE); }
			t_pf.buf = nb; t_pf.cap = need;
		}
		(void)seqb;
		/* Poisoning the prior window is a DEBUG aid, not production behaviour. It was
		 * gated on `#ifndef NDEBUG`, but this build never defines NDEBUG, so it ran in
		 * every shipped binary -- and it memset the whole high-water `cap`, not the
		 * `need` bytes actually about to be written. On a 5M-pair PE run that is tens of
		 * millions of ~1 KB memsets (tens of GB of stores) whose result is immediately
		 * overwritten by bns_get_seq_into below.
		 *
		 * Now opt-in via BWA_MEM3_DEBUG_POISON so it cannot silently return. Poison the
		 * FULL prior window (`t_pf.prev_len`, recorded after the last fetch), not the
		 * current `need`: the point is to catch a caller that stale-reads the previous
		 * window, which may be LONGER than this fetch -- scoping to `need` would leave
		 * the prior window's tail (bytes need..prev_len) un-poisoned and stale reads
		 * there undetected. prev_len is always <= cap by construction, so this stays in
		 * bounds. Byte-identical: production never defines the macro, so nothing runs. */
#ifdef BWA_MEM3_DEBUG_POISON
		if (t_pf.buf && t_pf.prev_len > 0) memset(t_pf.buf, 0xFF, (size_t)t_pf.prev_len); /* poison prior window */
#endif
		/* Fetch with the already-clamped [b, e) so the bytes written stay in
		 * lock-step with `need` (the buffer size). bns_get_seq_into re-derives
		 * the same clamp, so this is byte-identical to passing [beg, end). */
		bns_get_seq_into(l_pac, pac, b, e, t_pf.buf, len);
#ifdef BWA_MEM3_DEBUG_POISON
		t_pf.prev_len = *len;  /* record this window's length so the NEXT call poisons all of it */
#endif
		return (*len > 0) ? t_pf.buf : 0;
	}
	if (end < beg) end ^= beg, beg ^= end, end ^= beg; // if end is smaller, swap
	if (end > l_pac<<1) end = l_pac<<1;
	if (beg < 0) beg = 0;
	if (beg >= l_pac || end <= l_pac) {
		*len = end - beg;
		seq = ref_string + beg; // forward and reverse halves both live in ref_string
	} else *len = 0; // if bridging the forward-reverse boundary, return nothing
	return seq;
}

uint8_t *bns_fetch_seq_v2(const bntseq_t *bns, const uint8_t *pac,
                          int64_t *beg, int64_t mid, int64_t *end, int *rid,
                          uint8_t *ref_string, uint8_t *seqb)
{
	int64_t far_beg, far_end, len;
	int is_rev;
	uint8_t *seq;

	if (*end < *beg) *end ^= *beg, *beg ^= *end, *end ^= *beg; // if end is smaller, swap
	assert(*beg <= mid && mid < *end);

	*rid = bns_pos2rid(bns, bns_depos(bns, mid, &is_rev));
	far_beg = bns->anns[*rid].offset;
	far_end = far_beg + bns->anns[*rid].len;
	if (is_rev) { // flip to the reverse strand
		int64_t tmp = far_beg;
		far_beg = (bns->l_pac<<1) - far_end;
		far_end = (bns->l_pac<<1) - tmp;
	}
	*beg = *beg > far_beg? *beg : far_beg;
	*end = *end < far_end? *end : far_end;

	seq = bns_get_seq_v2(bns->l_pac, pac, *beg, *end, &len, ref_string, seqb);

	if (seq == 0 || *end - *beg != len) {
		fprintf(stderr, "[E::%s] begin=%ld, mid=%ld, end=%ld, len=%ld, seq=%p, rid=%d, far_beg=%ld, far_end=%ld\n",
				__func__, (long)*beg, (long)mid, (long)*end, (long)len, seq, *rid, (long)far_beg, (long)far_end);
	}
	assert(seq && *end - *beg == len); // assertion failure should never happen

	return seq;
}
