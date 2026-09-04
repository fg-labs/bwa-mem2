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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <assert.h>
#include "bntseq.h"
#include "bwa.h"
#include "ksw.h"
#include "utils.h"
#include "kstring.h"
#include "kvec.h"
#include "u8vec_scratch.h"
#include <string>

int bwa_verbose = 3;
char bwa_rg_id[256];
char *bwa_pg;

/************************
 * Batch FASTA/Q reader *
 ************************/

#include "kseq.h"
KSEQ_DECLARE(gzFile)

static inline void trim_readno(kstring_t *s)
{
    if (s->l > 2 && s->s[s->l-2] == '/' && isdigit(s->s[s->l-1]))
        s->l -= 2, s->s[s->l] = 0;
}

static inline void kseq2bseq1(const kseq_t *ks, bseq1_t *s, read_arena_t *arena)
{
    // bseq_read/bseq_read_orig grow `seqs` with realloc, which leaves the
    // new entries uninitialized. Zero first so sam/bams/n_bams/cap_bams
    // start well-defined — the output loop at fastmap.cpp free()s them
    // unconditionally, so garbage from realloc would otherwise be freed.
    memset(s, 0, sizeof(*s));
    /* Honor the bseq1_t.meth_base_ot -1 sentinel ("non-meth") from bwa.h: the
     * memset above would otherwise leave it 0, which the seed-chemistry filter
     * reads as OB. --meth ingest overwrites it with the read-number 0/1. */
    s->meth_base_ot = -1;
    /* PIPE-F6: name/seq/qual are carved from the per-chunk bump arena instead
     * of individually strdup'd. Byte-identical to strdup — read_arena_dup does
     * malloc-then-memcpy-then-NUL, and kseq kstrings are NUL-terminated so
     * ks->name.l/seq.l/qual.l are the exact strlen()s strdup would have copied.
     * `comment` stays heap-owned: --meth (fastmap step 0) frees and reassigns
     * it to a fresh heap string, and the non-copy_comment path frees it early,
     * so its ownership is not uniform enough to live in the arena. */
    s->name = read_arena_dup(arena, ks->name.s, ks->name.l);
    s->comment = ks->comment.l? strdup(ks->comment.s) : 0;
    // A NULL from OOM here would be read as "no comment" and silently drop the
    // comment under -C; fail loudly instead.
    if (ks->comment.l) xassert(s->comment != NULL, "out of memory: comment");
    s->seq = read_arena_dup(arena, ks->seq.s, ks->seq.l);
    s->qual = ks->qual.l? read_arena_dup(arena, ks->qual.s, ks->qual.l) : 0;
    s->l_seq = strlen(s->seq);
}

/* Customized for MPI processing */
bseq1_t *bseq_read(int64_t chunk_size, int *n_, void *ks1_, void *ks2_,
                   FILE* fpp, int len, int64_t *s, read_arena_t **arena_out)
{
    kseq_t *ks = (kseq_t*)ks1_, *ks2 = (kseq_t*)ks2_;
    int64_t size = 0, m, n, size2 = 0;
    bseq1_t *seqs;
    m = n = 0; seqs = 0;
    /* In/out: see bseq_read_fast. NULL in = fresh arena; non-NULL = keep carving
     * from the caller's, so one cohort read in several slices ends up with one
     * arena whose lifetime spans them all. */
    const int arena_created_here = (*arena_out == NULL);
    read_arena_t *arena = arena_created_here ? read_arena_create() : *arena_out;
    char buf[len];
    
    while (kseq_read(ks) >= 0)
    {
        if (ks2 && kseq_read(ks2) < 0) { // the 2nd file has fewer reads
            fprintf(stderr, "[W::%s] the 2nd file has fewer sequences.\n", __func__);
            break;
        }
        if (n >= m) {
            m = m? m<<1 : 256;
            seqs = (bseq1_t*) realloc(seqs, m * sizeof(bseq1_t));
        }
        trim_readno(&ks->name);
        kseq2bseq1(ks, &seqs[n], arena);
        seqs[n].id = n;
        {
            //kseq_t *ksd = ks;
            //kstream_t *kst = ksd->f;
#if 0
            //printf("Check D..\n%s\n%s\n%s\n%s\n",
            //     seqs[n].name, seqs[n].seq,
            //     seqs[n].comment, seqs[n].qual);
            
            if (seqs[n].name != NULL)
                size += strlen(seqs[n].name);
            //printf("%d ", strlen(seqs[n].name)+strlen(seqs[n].comment)+1);
            if (seqs[n].comment != NULL) {
                size += strlen(seqs[n].comment);
                std::string str = seqs[n].comment;
                std::size_t found = str.find("length");
                if (found != std::string::npos) {
                    size += strlen(seqs[n].comment) + strlen(seqs[n].name) + 1;
                }
            }
            else
                size += 1;
            
            if (seqs[n].qual != NULL)
                size += strlen(seqs[n].qual);

            size += 7; // non accounted chars
#else
            //kstring_t kstr;
            //printf("%d\n", ks_getuntil2(kst, KS_SEP_LINE, &kstr, 0, 0));
            err_fgets((char*) buf, len, fpp);
            size2 += strlen((char*) buf);
            // printf("First line: %d, %s\n", strlen(buf), buf);
            err_fgets((char*) buf, len, fpp);
            size2 += strlen((char*) buf);
            if (seqs[n].qual != NULL) {
                err_fgets((char*) buf, len, fpp);
                size2 += strlen((char*) buf);
                err_fgets((char*) buf, len, fpp);
                size2 += strlen((char*) buf);
            }
#endif
        }
        //size += seqs[n++].l_seq;
        size = size2;       n++;
        
        //printf("size: %d, size2: %d\n", size, size2);
        //static int cnt = 0;
        //if (cnt++ == 4)exit(0);
        
        if (ks2) {
            trim_readno(&ks2->name);
            kseq2bseq1(ks2, &seqs[n], arena);
            seqs[n].id = n;
            n++;
            // size += seqs[n++].l_seq;
        }
        //if (size >= chunk_size && (n&1) == 0) break;
        if (size >= chunk_size) {
            break;
        }
    }
    if (size == 0) { // test if the 2nd file is finished
        if (ks2 && kseq_read(ks2) >= 0)
            fprintf(stderr, "[W::%s] the 1st file has fewer sequences.\n", __func__);
    }
    /* PIPE-F6: hand the arena to the caller, or destroy it if this batch carved
     * nothing (n == 0 → seqs stays NULL and the pipeline treats it as clean EOF,
     * freeing nothing). Only an arena created here may be destroyed here; a
     * carried one still backs earlier slices of an in-flight cohort. */
    if (n == 0 && arena_created_here) { read_arena_destroy(arena); arena = NULL; }
    *arena_out = arena;
    *n_ = n;
    *s = size;
    return seqs;
}

bseq1_t *bseq_read_orig(int64_t chunk_size, int *n_, void *ks1_, void *ks2_, int64_t *s,
                        read_arena_t **arena_out)
{
    kseq_t *ks = (kseq_t*)ks1_, *ks2 = (kseq_t*)ks2_;
    int64_t size = 0, m, n;
    bseq1_t *seqs;
    m = n = 0; seqs = 0;
    /* In/out: see bseq_read_fast. NULL in = fresh arena; non-NULL = keep carving
     * from the caller's, so one cohort read in several slices ends up with one
     * arena whose lifetime spans them all. */
    const int arena_created_here = (*arena_out == NULL);
    read_arena_t *arena = arena_created_here ? read_arena_create() : *arena_out;
    int r1;
    // kseq_read returns -1 at EOF but < -1 (e.g. -2) for a truncated record or a
    // seq/qual length mismatch. Fail loudly on that, matching the default reader
    // (bseq_read_fast, fixed in the same change), or --legacy-reader silently
    // truncates a corrupt input and exits 0.
    while ((r1 = kseq_read(ks)) >= 0)
    {
        if (ks2) {
            int r2 = kseq_read(ks2);
            if (r2 < -1)
                err_fatal(__func__, "malformed FASTQ record or read/decode error in the 2nd input (record %ld)", (long)n);
            if (r2 < 0) { // -1 EOF: the 2nd file has fewer reads
                fprintf(stderr, "[W::%s] the 2nd file has fewer sequences.\n", __func__);
                break;
            }
        }
        if (n >= m) {
            bseq1_t *tmp;
            m = m? m<<1 : 256;
            tmp = (bseq1_t*) realloc(seqs, m * sizeof(bseq1_t));
            xassert(tmp != NULL, "out of memory: seqs");
            seqs = tmp;
        }
        trim_readno(&ks->name);
        kseq2bseq1(ks, &seqs[n], arena);
        seqs[n].id = n;
        //{
        //  size += strlen(seqs[n].name);
        //  size += strlen(seqs[n].comment);
        //  size += strlen(seqs[n].qual);
        //  // fprintf(stderr, "qual len: %d %d\n", strlen(seqs[n].qual), seqs[n].l_seq);
        //  size += 7; // non accounted chars
        //}
        size += seqs[n++].l_seq;

        if (ks2) {
            trim_readno(&ks2->name);
            kseq2bseq1(ks2, &seqs[n], arena);
            seqs[n].id = n;
            size += seqs[n++].l_seq;
        }
        if (size >= chunk_size && (n&1) == 0) break;
        // if (size >= chunk_size) {
        //  break;
        // }
    }
    // The loop exits on r1 < 0; distinguish EOF (-1) from a malformed 1st-input
    // record (< -1) so truncated input fails loudly instead of ending at exit 0.
    if (r1 < -1)
        err_fatal(__func__, "malformed FASTQ record or read/decode error in the 1st input (record %ld)", (long)n);
    if (size == 0) { // test if the 2nd file is finished
        if (ks2) {
            int r2 = kseq_read(ks2);
            if (r2 < -1)
                err_fatal(__func__, "malformed FASTQ record or read/decode error in the 2nd input (record %ld)", (long)n);
            if (r2 >= 0)
                fprintf(stderr, "[W::%s] the 1st file has fewer sequences.\n", __func__);
        }
    }
    /* PIPE-F6: see bseq_read above — hand off, or free the empty arena at EOF,
     * but only when this call created it. */
    if (n == 0 && arena_created_here) { read_arena_destroy(arena); arena = NULL; }
    *arena_out = arena;
    *n_ = n;
    *s = size;
    return seqs;
}

bseq1_t *bseq_read_one_fasta_file(int64_t chunk_size, int *n_, gzFile fp, int64_t *s,
                                  read_arena_t **arena_out)
{
    kseq_t *ks = kseq_init(fp);
    bseq1_t *seq = bseq_read_orig(chunk_size, n_, ks, NULL, s, arena_out);
    kseq_destroy(ks);
    return seq;
}

void bseq_classify(int n, bseq1_t *seqs, int m[2], bseq1_t *sep[2])
{
    int i, has_last;
    kvec_t(bseq1_t) a[2] = {{0,0,0}, {0,0,0}};
    // An empty batch (reachable via the -p interleaved path when input ends on
    // a cohort-slice boundary) must not fall through to kv_push(..., seqs[i-1])
    // with i == 1 on a NULL seqs.
    if (n <= 0) { m[0] = m[1] = 0; sep[0] = sep[1] = NULL; return; }
    for (i = 1, has_last = 1; i < n; ++i) {
        if (has_last) {
            if (strcmp(seqs[i].name, seqs[i-1].name) == 0) {
                kv_push(bseq1_t, a[1], seqs[i-1]);
                kv_push(bseq1_t, a[1], seqs[i]);
                has_last = 0;
            } else kv_push(bseq1_t, a[0], seqs[i-1]);
        } else has_last = 1;
    }
    if (has_last) kv_push(bseq1_t, a[0], seqs[i-1]);
    sep[0] = a[0].a, m[0] = a[0].n;
    sep[1] = a[1].a, m[1] = a[1].n;
}

/*****************
 * CIGAR related *
 *****************/

void bwa_fill_scmat(int a, int b, int8_t mat[25])
{
    int i, j, k;
    for (i = k = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j)
            mat[k++] = i == j? a : -b;
        mat[k++] = -1; // ambiguous base
    }
    for (j = 0; j < 5; ++j) mat[k++] = -1;   // DEFAULT AMBIG
}

/* Generate CIGAR when the alignment end points are known.
 *
 * `nm_from_mat` selects how the NM/MD pass classifies an aligned column:
 *   0 (default, non-meth): a column is a mismatch iff the bases differ literally
 *     (`query != rseq`) -- the historical bwa behaviour, byte-for-byte.
 *   1 (--meth): a column is a mismatch iff the SCORING MATRIX penalises it
 *     (`mat[ref*5 + query] < 0`). Under --meth `mat` is the per-hypothesis
 *     asymmetric matrix whose bisulfite-conversion cell is set to the match
 *     score (mem_opt_fill_meth_mat), so a C->T (OT) / G->A (OB) conversion is a
 *     match for NM/MD exactly as it already is for the DP. See the policy note
 *     in mem_reg2aln().
 *
 * The matrix is safe to reuse here because the NM/MD pass and the DP share BOTH
 * buffers and frame: on the reverse strand (rb >= l_pac) `query`/`rseq` are
 * reversed in place above and the caller has already flipped the OT/OB
 * hypothesis to match, so `mat` indexes the same way in both passes. */
uint32_t *bwa_gen_cigar3(const int8_t mat[25], int o_del, int e_del, int o_ins, int e_ins, int w_, int64_t l_pac, const uint8_t *pac, int l_query, uint8_t *query, int64_t rb, int64_t re, int *score, int *n_cigar, int *NM, int nm_from_mat)
{
    uint32_t *cigar = 0;
    uint8_t tmp, *rseq;
    int i;
    int64_t rlen;
    kstring_t str;
    const char *int2base;

    if (n_cigar) *n_cigar = 0;
    if (NM) *NM = -1;
    if (l_query <= 0 || rb >= re || (rb < l_pac && re > l_pac)) return 0; // reject if negative length or bridging the forward and reverse strand
    {
        static thread_local u8vec_scratch_t t_rseq;
        size_t want = (size_t)((re - rb) + 64);  // +64 keeps the "+64" headroom the old malloc had
        if (t_rseq.v.m < want) kv_resize(uint8_t, t_rseq.v, want);
        bns_get_seq_into(l_pac, pac, rb, re, t_rseq.v.a, &rlen);
        rseq = t_rseq.v.a;
    }
    if (re - rb != rlen) goto ret_gen_cigar; // possible if out of range
    if (rb >= l_pac) { // then reverse both query and rseq; this is to ensure indels to be placed at the leftmost position
        for (i = 0; i < l_query>>1; ++i)
            tmp = query[i], query[i] = query[l_query - 1 - i], query[l_query - 1 - i] = tmp;
        for (i = 0; i < rlen>>1; ++i)
            tmp = rseq[i], rseq[i] = rseq[rlen - 1 - i], rseq[rlen - 1 - i] = tmp;
    }
    if (l_query == re - rb && w_ == 0) { // no gap; no need to do DP
        // Reached routinely for provably-ungapped, equal-length alignments. mem_reg2aln()
        // derives the emission band from infer_bw() (bwamem.cpp), which returns 0 whenever
        // the score deficit is below the two-gap threshold -- i.e. no balanced indel could
        // improve the score, so the optimal alignment is gap-free. A zero band lands here
        // and emits the single <len>M CIGAR directly, with no ksw_global2 fill or traceback.
        // (An earlier FIXME here claimed this block was unreachable; that was stale.)
        if (n_cigar) {
            cigar = (uint32_t*) malloc(4);
            xassert(cigar != NULL, "out of memory: cigar");
            cigar[0] = l_query<<4 | 0;
            *n_cigar = 1;
        }
        for (i = 0, *score = 0; i < l_query; ++i)
            *score += mat[rseq[i]*5 + query[i]];
    } else {
        int w, max_gap, max_ins, max_del, min_w;
        // set the band-width
        max_ins = (int)((double)(((l_query+1)>>1) * mat[0] - o_ins) / e_ins + 1.);
        max_del = (int)((double)(((l_query+1)>>1) * mat[0] - o_del) / e_del + 1.);
        max_gap = max_ins > max_del? max_ins : max_del;
        max_gap = max_gap > 1? max_gap : 1;
        w = (max_gap + abs(rlen - l_query) + 1) >> 1;
        w = w < w_? w : w_;
        min_w = abs(rlen - l_query) + 3;
        w = w > min_w? w : min_w;
        // NW alignment
        if (bwa_verbose >= 4) {
            fprintf(stderr, "* Global bandwidth: %d\n", w);
            fprintf(stderr, "* Global ref:   "); for (i = 0; i < rlen; ++i) fputc("ACGTN"[(int)rseq[i]], stderr); fputc('\n', stderr);
            fprintf(stderr, "* Global query: "); for (i = 0; i < l_query; ++i) fputc("ACGTN"[(int)query[i]], stderr); fputc('\n', stderr);
        }
        *score = ksw_global2(l_query, query, rlen, rseq, 5, mat, o_del, e_del, o_ins, e_ins, w, n_cigar, &cigar);
    }
    if (NM && n_cigar) {// compute NM and MD
        int k, x, y, u, n_mm = 0, n_gap = 0;
        str.l = str.m = *n_cigar * 4; str.s = (char*)cigar; // append MD to CIGAR
        int2base = rb < l_pac? "ACGTN" : "TGCAN";
        for (k = 0, x = y = u = 0; k < *n_cigar; ++k) {
            int op, len;
            cigar = (uint32_t*)str.s;
            op  = cigar[k]&0xf, len = cigar[k]>>4;
            if (op == 0) { // match
                for (i = 0; i < len; ++i) {
                    /* loop-invariant select; see the nm_from_mat contract above */
                    const int is_mm = nm_from_mat
                        ? (mat[rseq[y + i] * 5 + query[x + i]] < 0)
                        : (query[x + i] != rseq[y + i]);
                    if (is_mm) {
                        kputw(u, &str);
                        kputc(int2base[rseq[y+i]], &str);
                        ++n_mm; u = 0;
                    } else ++u;
                }
                x += len; y += len;
            } else if (op == 2) { // deletion
                if (k > 0 && k < *n_cigar - 1) { // don't do the following if D is the first or the last CIGAR
                    kputw(u, &str); kputc('^', &str);
                    for (i = 0; i < len; ++i)
                        kputc(int2base[rseq[y+i]], &str);
                    u = 0; n_gap += len;
                }
                y += len;
            } else if (op == 1) x += len, n_gap += len; // insertion
        }
        kputw(u, &str); kputc(0, &str);
        *NM = n_mm + n_gap;
        cigar = (uint32_t*)str.s;
    }
    if (rb >= l_pac) // reverse back query
        for (i = 0; i < l_query>>1; ++i)
            tmp = query[i], query[i] = query[l_query - 1 - i], query[l_query - 1 - i] = tmp;

ret_gen_cigar:
    /* rseq aliases thread-local kvec scratch in this function; do not free. */
    return cigar;
}

uint32_t *bwa_gen_cigar2(const int8_t mat[25], int o_del, int e_del, int o_ins, int e_ins, int w_, int64_t l_pac, const uint8_t *pac, int l_query, uint8_t *query, int64_t rb, int64_t re, int *score, int *n_cigar, int *NM)
{
    return bwa_gen_cigar3(mat, o_del, e_del, o_ins, e_ins, w_, l_pac, pac, l_query, query, rb, re, score, n_cigar, NM, 0);
}

uint32_t *bwa_gen_cigar(const int8_t mat[25], int q, int r, int w_, int64_t l_pac, const uint8_t *pac, int l_query, uint8_t *query, int64_t rb, int64_t re, int *score, int *n_cigar, int *NM)
{
    return bwa_gen_cigar2(mat, q, r, q, r, w_, l_pac, pac, l_query, query, rb, re, score, n_cigar, NM);
}

/*********************
 * Full index reader *
 *********************/
#if 0
char *bwa_idx_infer_prefix(const char *hint)
{
    char *prefix;
    int l_hint;
    FILE *fp;
    l_hint = strlen(hint);
    prefix = (char *) malloc(l_hint + 3 + 4 + 1);
    strcpy(prefix, hint);
    strcpy(prefix + l_hint, ".64.bwt");
    if ((fp = fopen(prefix, "rb")) != 0) {
        fclose(fp);
        prefix[l_hint + 3] = 0;
        return prefix;
    } else {
        strcpy(prefix + l_hint, ".bwt");
        if ((fp = fopen(prefix, "rb")) == 0) {
            free(prefix);
            return 0;
        } else {
            fclose(fp);
            prefix[l_hint] = 0;
            return prefix;
        }
    }
}

bwt_t *bwa_idx_load_bwt(const char *hint)
{
    char *tmp, *prefix;
    bwt_t *bwt;
    prefix = bwa_idx_infer_prefix(hint);
    if (prefix == 0) {
        if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] fail to locate the index files\n", __func__);
        return 0;
    }
    tmp = (char*) calloc(strlen(prefix) + 5, 1);
    strcat(strcpy(tmp, prefix), ".bwt"); // FM-index
    bwt = bwt_restore_bwt(tmp);
    strcat(strcpy(tmp, prefix), ".sa");  // partial suffix array (SA)
    bwt_restore_sa(tmp, bwt);
    free(tmp); free(prefix);
    return bwt;
}

bwaidx_t *bwa_idx_load_from_disk(const char *hint, int which)
{
    bwaidx_t *idx;
    char *prefix;
    prefix = bwa_idx_infer_prefix(hint);
    
    if (prefix == 0) {
        if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] fail to locate the index files\n", __func__);
        return 0;
    }
    idx = (bwaidx_t*) calloc(1, sizeof(bwaidx_t));
    xassert(idx != NULL, "out of memory: idx");
    if (which & BWA_IDX_BWT) idx->bwt = bwa_idx_load_bwt(hint);
    if (which & BWA_IDX_BNS) {
        int i, c;
        idx->bns = bns_restore(prefix);
        xassert(idx->bns != NULL, "out of memory: idx->bns");
        for (i = c = 0; i < idx->bns->n_seqs; ++i)
            if (idx->bns->anns[i].is_alt) ++c;
        if (bwa_verbose >= 3)
            fprintf(stderr, "[M::%s] read %d ALT contigs\n", __func__, c);
        if (which & BWA_IDX_PAC) {
            idx->pac = (uint8_t*) calloc(idx->bns->l_pac/4+1, 1);
            xassert(idx->pac != NULL, "out of memory: idx->pac");
            err_fread_noeof(idx->pac, 1, idx->bns->l_pac/4+1, idx->bns->fp_pac); // concatenated 2-bit encoded sequence
            err_fclose(idx->bns->fp_pac);
            idx->bns->fp_pac = 0;
        }
    }
    free(prefix);
    return idx;
}

bwaidx_t *bwa_idx_load(const char *hint, int which)
{
    return bwa_idx_load_from_disk(hint, which);
}

void bwa_idx_destroy(bwaidx_t *idx)
{
    if (idx == 0) return;
    if (idx->mem == 0) {
        if (idx->bwt) bwt_destroy(idx->bwt);
        if (idx->bns) bns_destroy(idx->bns);
        if (idx->pac) free(idx->pac);
    } else {
        // SAM-A3: pos2rid_bucket is a private heap allocation (see read_index_ele.cpp).
        free(idx->bwt); free(idx->bns->pos2rid_bucket); free(idx->bns->anns); free(idx->bns);
        if (!idx->is_shm) free(idx->mem);
    }
    free(idx);
}

int bwa_mem3idx(int64_t l_mem, uint8_t *mem, bwaidx_t *idx)
{
    int64_t k = 0, x;
    int i;

    // generate idx->bwt
    x = sizeof(bwt_t); idx->bwt = (bwt_t*) malloc(x); memcpy(idx->bwt, mem + k, x); k += x;
    x = idx->bwt->bwt_size * 4; idx->bwt->bwt = (uint32_t*)(mem + k); k += x;
    x = idx->bwt->n_sa * sizeof(bwtint_t); idx->bwt->sa = (bwtint_t*)(mem + k); k += x;

    // generate idx->bns and idx->pac
    x = sizeof(bntseq_t); idx->bns = (bntseq_t*) malloc(x); memcpy(idx->bns, mem + k, x); k += x;
    x = idx->bns->n_holes * sizeof(bntamb1_t); idx->bns->ambs = (bntamb1_t*)(mem + k); k += x;
    x = idx->bns->n_seqs  * sizeof(bntann1_t); idx->bns->anns = (bntann1_t*) malloc(x); memcpy(idx->bns->anns, mem + k, x); k += x;
    for (i = 0; i < idx->bns->n_seqs; ++i) {
        idx->bns->anns[i].name = (char*)(mem + k); k += strlen(idx->bns->anns[i].name) + 1;
        idx->bns->anns[i].anno = (char*)(mem + k); k += strlen(idx->bns->anns[i].anno) + 1;
    }
    idx->pac = (uint8_t*)(mem + k); k += idx->bns->l_pac/4+1;
    assert(k == l_mem);

    // SAM-A3: bwa_idx2mem stages a NULL pos2rid_bucket, but a blob written by
    // an older or out-of-tree writer may carry that writer's heap address, so
    // drop whatever came in unconditionally and rebuild against our own anns[].
    idx->bns->pos2rid_bucket = NULL;
    bns_build_pos2rid(idx->bns);

    idx->l_mem = k; idx->mem = mem;
    return 0;
}

int bwa_idx2mem(bwaidx_t *idx)
{
    int i;
    int64_t k, x, tmp;
    uint8_t *mem;
    int32_t *pos2rid_bucket;

    // copy idx->bwt
    x = idx->bwt->bwt_size * 4;
    mem = (uint8_t*) realloc(idx->bwt->bwt, sizeof(bwt_t) + x); idx->bwt->bwt = 0;
    memmove(mem + sizeof(bwt_t), mem, x);
    memcpy(mem, idx->bwt, sizeof(bwt_t)); k = sizeof(bwt_t) + x;
    x = idx->bwt->n_sa * sizeof(bwtint_t); mem = (uint8_t*) realloc(mem, k + x); memcpy(mem + k, idx->bwt->sa, x); k += x;
    free(idx->bwt->sa);
    free(idx->bwt); idx->bwt = 0;

    // copy idx->bns
    // SAM-A3: the derived pos2rid table is not serialized (see bntseq.h), so
    // detach it BEFORE the struct memcpy below — otherwise the packed image
    // would carry this process's heap address into every consumer of the blob.
    // The detached allocation is freed once the copy is done.
    pos2rid_bucket = idx->bns->pos2rid_bucket;
    idx->bns->pos2rid_bucket = NULL;
    tmp = idx->bns->n_seqs * sizeof(bntann1_t) + idx->bns->n_holes * sizeof(bntamb1_t);
    for (i = 0; i < idx->bns->n_seqs; ++i) // compute the size of heap-allocated memory
        tmp += strlen(idx->bns->anns[i].name) + strlen(idx->bns->anns[i].anno) + 2;
    mem = (uint8_t*) realloc(mem, k + sizeof(bntseq_t) + tmp);
    x = sizeof(bntseq_t); memcpy(mem + k, idx->bns, x); k += x;
    x = idx->bns->n_holes * sizeof(bntamb1_t); memcpy(mem + k, idx->bns->ambs, x); k += x;
    free(idx->bns->ambs);
    x = idx->bns->n_seqs * sizeof(bntann1_t); memcpy(mem + k, idx->bns->anns, x); k += x;
    for (i = 0; i < idx->bns->n_seqs; ++i) {
        x = strlen(idx->bns->anns[i].name) + 1; memcpy(mem + k, idx->bns->anns[i].name, x); k += x;
        x = strlen(idx->bns->anns[i].anno) + 1; memcpy(mem + k, idx->bns->anns[i].anno, x); k += x;
        free(idx->bns->anns[i].name); free(idx->bns->anns[i].anno);
    }
    free(idx->bns->anns);

    // copy idx->pac
    x = idx->bns->l_pac/4+1;
    mem = (uint8_t*) realloc(mem, k + x);
    memcpy(mem + k, idx->pac, x); k += x;
    free(pos2rid_bucket); // SAM-A3: derived table is rebuilt on restore, not serialized
    free(idx->bns); idx->bns = 0;
    free(idx->pac); idx->pac = 0;

    return bwa_mem3idx(k, mem, idx);
}
#endif

/***********************
 * SAM header routines *
 ***********************/

// Write the first line from `lines` to `fp` and return a pointer to the
// remainder (or NULL when consumed). `lines` is newline-separated but not
// required to be newline-terminated.
static const char *hoist_line(const char *lines, FILE *fp)
{
    const char *nl = strchr(lines, '\n');
    if (nl) {
        err_fwrite(lines, 1, (size_t)(nl - lines), fp);
        err_fputc('\n', fp);
        return nl + 1;
    } else {
        err_fputs(lines, fp);
        err_fputc('\n', fp);
        return NULL;
    }
}

// Write `lines` (newline-separated, not newline-terminated) to `fp`, dropping
// every @HD record except -- when `keep_first_HD` is set -- the first one.
//
// Only ONE @HD may reach the output, and the loser can sit anywhere in either
// stream, so it cannot always be consumed off the front the way hoist_line
// handles a leading record. A winner that is not leading has to stay inline,
// where the caller put it, hence `keep_first_HD` rather than an unconditional
// drop. Empty lines are skipped, matching the BAM writer's equivalent filter
// (bam_writer.cpp) -- a blank line is not a valid header record.
static void fputs_filtering_HD(const char *lines, int keep_first_HD, FILE *fp)
{
    int seen_HD = 0;
    const char *cur = lines, *line; size_t len;
    while (bwa_hdr_next_line(&cur, &line, &len)) {
        const int is_HD = (len >= 4 && strncmp(line, "@HD\t", 4) == 0);
        if (len > 0 && (!is_HD || (keep_first_HD && !seen_HD))) {
            err_fwrite(line, 1, len, fp);
            err_fputc('\n', fp);
        }
        if (is_HD) seen_HD = 1;
    }
}

/* Iterate the records of newline-separated SAM header text.
 *
 * Start with *p at the beginning of the text. Each call sets *line/*len to the
 * next record (newline excluded), advances *p past it, and returns 1; returns 0
 * at end of text. Text with or without a trailing newline both yield exactly
 * the records present -- no phantom empty final record.
 *
 * Exists because this exact walk -- strchr('\n'), length, advance-or-stop --
 * was hand-rolled in six places across three files (fg-labs/bwa-mem3#289),
 * each subtly its own; all six now call here. Callers that care about empty
 * records must still skip them; the iterator reports what is there.
 *
 *     const char *p = hdr_text, *line; size_t len;
 *     while (bwa_hdr_next_line(&p, &line, &len)) { ... }
 */
int bwa_hdr_next_line(const char **p, const char **line, size_t *len)
{
    if (p == NULL || *p == NULL || **p == '\0') return 0;
    const char *start = *p;
    const char *eol = strchr(start, '\n');
    *line = start;
    *len  = eol ? (size_t)(eol - start) : strlen(start);
    *p    = eol ? eol + 1 : start + *len;      /* lands on the NUL when no eol */
    return 1;
}

/* Format the generated @SQ record for contig `ann` into `out` (appended, no
 * trailing newline). Returns 0, or -1 if `out` could not be grown.
 *
 * ONE definition on purpose. This record used to be built independently in
 * three writers -- snprintf+err_fputs here, sam_hdr_add_line varargs in
 * bam_writer.cpp, ksprintf in meth_bam.cpp -- and they drifted: AH:* was
 * correct only in this copy, so --bam and --meth lost ALT status for every
 * ALT-aware reference until each copy was fixed separately
 * (fg-labs/bwa-mem3#281); consolidating them so it cannot recur again is
 * fg-labs/bwa-mem3#289. The three sinks all accept SAM header TEXT, so there
 * is no reason for three spellings.
 *
 * Contents match what bwa (bwa.c:430-433) and bwa-mem2 (bwa.cpp:535-548) emit.
 * The DECISION to emit is deliberately NOT here -- each writer gates it
 * differently (-H precedence, sidecar precedence, --meth always-emit) and that
 * is genuine per-path policy.
 *
 * Uses kstring rather than a fixed buffer: the old snprintf into char[512]
 * truncated silently on a long contig name. The one cost of that is that the
 * append can now fail, and ksprintf/kputs swallow their own failures -- so the
 * worst case is reserved up front and the status returned, the same way the
 * MC:Z and SA:Z builders in bam_writer.cpp handle it. Without this, a caller
 * has no way to tell a failed append from a formatted record and would emit
 * (or fputs) a NULL or truncated one. */
int bwa_format_sq_line(kstring_t *out, const bntann1_t *ann)
{
    /* "@SQ\tSN:" + name + "\tLN:" + int + "\tAH:*" + NUL: 7 + 4 + 11 + 5 + 1,
     * rounded up. Pre-sized, the appends below cannot need to grow `out`.
     *
     * The failure test is on `out` rather than a return value because this file
     * uses bwa's kstring.h, whose ks_resize is void and leaves s->s NULL when
     * the realloc fails -- not htslib's, which returns a status. (bam_writer.cpp
     * gets htslib's and can check directly; see the MC:Z path there.) */
    const size_t need = out->l + strlen(ann->name) + 32;
    ks_resize(out, need);
    if (out->s == NULL || out->m < need) return -1;
    ksprintf(out, "@SQ\tSN:%s\tLN:%d", ann->name, ann->len);
    /* AH:* marks an alternate locus. `is_alt` comes from <prefix>.alt via
     * bwa_idx_load and has no representation in an htslib sam_hdr_t, so every
     * writer must re-apply it from bns rather than expect it to survive. */
    if (ann->is_alt) kputs("\tAH:*", out);
    return 0;
}

int bwa_format_pa_value(char *buf, int score, int alt_sc)
{
    /* "%.3f" is what both upstreams emit (bwa bwamem.c, bwa-mem2 bwamem.cpp),
     * and it is the rendering every consumer that goes through SAM text sees.
     *
     * Deliberately no failure return. A short render would make the SAM path
     * emit a valueless `pa:f:` while bwa_pa_tag_value's strtod of the same
     * buffer stored 0 in BAM -- the very disagreement this function exists to
     * remove, reintroduced on an error path. BWA_PA_TEXT_MAX bounds "%.3f" of
     * any int/int ratio, so the guard is unreachable rather than a policy. */
    xassert(alt_sc != 0, "pa:f: has no value when alt_sc is zero");
    const int n = snprintf(buf, BWA_PA_TEXT_MAX, "%.3f", (double)score / alt_sc);
    xassert(n > 0 && n < BWA_PA_TEXT_MAX, "pa:f: value did not fit its buffer");
    return n;
}

float bwa_pa_tag_value(int score, int alt_sc)
{
    /* Round-tripping the SAM token is not a roundabout way to round -- it is
     * the definition. A BAM `pa:f:` field has to hold what `samtools view -b`
     * of the SAM text would store, and htslib parses an 'f' aux field with
     * strtod and narrows to float. Reproducing that literally is the only
     * construction guaranteed to agree with the text path on every input,
     * including the halfway cases where printf's round-to-even and round()'s
     * round-away-from-zero disagree. */
    char buf[BWA_PA_TEXT_MAX];
    bwa_format_pa_value(buf, score, alt_sc);
    return (float)strtod(buf, NULL);
}

/* True if the SAM header text `s` contains a record whose type is `tag` (e.g.
 * "@HD\t"), either as the first line or following a newline. Hoisted out of
 * meth_bam.cpp, where it was static, so the BAM writers stop open-coding it.
 *
 * `tag` is any NUL-terminated prefix, not specifically a 4-character "@XX\t" --
 * hence the name, rather than the `tag4` this carried while it was static with
 * two 4-character call sites.
 *
 * Implemented on the record iterator rather than as a strstr for "\n" + tag.
 * The needle form needs a fixed buffer, and this is public now: a caller
 * passing a longer type than the buffer holds would have its tag silently
 * truncated and get a wrong answer instead of no match. Walking records has no
 * length ceiling, and it keeps this file to one scanner. */
int bwa_hdr_text_has_type(const char *s, const char *tag)
{
    if (s == NULL || tag == NULL) return 0;
    const size_t n = strlen(tag);
    if (n == 0) return 0;
    const char *cur = s, *line; size_t len;
    while (bwa_hdr_next_line(&cur, &line, &len))
        if (len >= n && strncmp(line, tag, n) == 0) return 1;
    return 0;
}

// Return 1 iff `lines` contains any @SQ header records.
static int has_SQ(const char *lines)
{
    return bwa_hdr_text_has_type(lines, "@SQ\t");
}

// Return 1 iff `lines` contains an @HD record (first line, or after a newline).
static int has_HD(const char *lines)
{
    return bwa_hdr_text_has_type(lines, "@HD\t");
}

// Count the number of @SQ header records in `lines`.
static int count_SQ(const char *lines)
{
    int n_SQ = 0;
    const char *cur = lines, *line; size_t len;
    while (bwa_hdr_next_line(&cur, &line, &len))
        if (len >= 4 && strncmp(line, "@SQ\t", 4) == 0) ++n_SQ;
    return n_SQ;
}

/* Contig name -> bns index, for the ALT/AH sidecar check below. khash emits
 * static functions, so instantiating a set here does not collide with the
 * kh_str map bntseq.cpp builds for the same kind of lookup. */
#include "khash.h"
KHASH_MAP_INIT_STR(sqname, int)

/* Scan one @SQ record for its SN value and whether it carries an AH tag.
 * `line`/`line_len` delimit the record (no trailing newline). On a match sets
 * *sn/*sn_len to the SN value (pointing into `line`) and *has_ah, and returns
 * 1; returns 0 if the record has no SN. Tokens are matched whole, so an SN:
 * lookup can never confuse chr1 with chr1_alt. */
static int sq_scan(const char *line, size_t line_len,
                   const char **sn, size_t *sn_len, int *has_ah)
{
    if (line_len < 4 || strncmp(line, "@SQ\t", 4) != 0) return 0;
    const char *end = line + line_len;
    int found = 0;
    *has_ah = 0;
    for (const char *p = line + 3; p < end; ) {
        const char *tok = p + 1;                          /* skip the '\t' */
        const char *tok_end = (const char *)memchr(tok, '\t', (size_t)(end - tok));
        if (tok_end == NULL) tok_end = end;
        const size_t tok_len = (size_t)(tok_end - tok);
        if (tok_len > 3 && strncmp(tok, "SN:", 3) == 0) {
            *sn = tok + 3; *sn_len = tok_len - 3; found = 1;
        } else if (tok_len > 3 && strncmp(tok, "AH:", 3) == 0) {
            *has_ah = 1;
        }
        p = tok_end;
    }
    return found;
}

/* Warn when the index knows about ALT contigs but the sidecar's @SQ block does
 * not carry AH for them.
 *
 * The sidecar's @SQ is authoritative by design -- it is a port of lh3/bwa#348,
 * whose author intended the block to be produced complete by an external dict
 * tool, and `samtools dict --alt <ref>.alt` (samtools/samtools#1676) exists to
 * do exactly that. So we do NOT enrich it: silently rewriting a block the user
 * owns would be a surprise, and it would raise an unanswerable question about
 * an AH the index contradicts. But losing ALT status is also not something to
 * discover downstream in bwa-postalt.js, so say it out loud with the remedy. */
void bwa_warn_sidecar_missing_AH(const bntseq_t *bns, const char *idx_hdr_lines,
                                 const char *prefix)
{
    if (bns == NULL || !has_SQ(idx_hdr_lines) || bwa_verbose < 2) return;

    /* One pass over bns to collect the ALT names, then ONE pass over the
     * sidecar. Looking each ALT contig up separately would rescan the whole
     * block per contig: on hg38 (~3.4k records, ~800 ALT contigs, most of them
     * at the end of the file) that measured ~500x more work, and it was paid
     * even when every record already carries AH and nothing is reported.
     * Keys are borrowed from bns->anns, so kh_destroy must not free them --
     * same ownership as the .alt parser in bntseq.cpp. */
    khash_t(sqname) *alt = kh_init(sqname);
    for (int i = 0; i < bns->n_seqs; ++i) {
        if (!bns->anns[i].is_alt) continue;
        int absent;
        khiter_t k = kh_put(sqname, alt, bns->anns[i].name, &absent);
        if (absent < 0) continue;        /* kh_resize failed: skip, don't write past vals */
        kh_val(alt, k) = i;              /* recover the stable name below */
    }
    if (kh_size(alt) == 0) { kh_destroy(sqname, alt); return; }

    /* Counts only ALT contigs that the sidecar actually HAS an @SQ for. An ALT
     * contig the sidecar omits entirely is a different (larger) problem -- that
     * contig gets no @SQ at all -- and "missing AH" would misdescribe it; it is
     * already reported by the @SQ-count mismatch warning in
     * bwa_print_sam_hdr2 ("N @SQ lines loaded from index; M sequences"). */
    int n_missing = 0, first_id = -1;
    const char *cur = idx_hdr_lines, *line; size_t line_len;
    /* kh_get needs a NUL-terminated key and `sn` points into the header text, so
     * each SN has to be copied out. One reusable buffer grown to the longest SN
     * seen, NOT a fixed 256-byte stack array: contig names are strdup()ed from
     * the FASTA/.ann name (bntseq.cpp) with no length bound, so a long-named ALT
     * contig CAN exist in bns. Capping the copy silently skipped exactly those
     * records, which is the case this check exists for -- an ALT contig whose
     * sidecar @SQ lacks AH got no warning. It grows a handful of times at most
     * (monotonically), so the single-pass cost the comment above is about is
     * unchanged. */
    char *name = NULL; size_t name_cap = 0;
    while (bwa_hdr_next_line(&cur, &line, &line_len)) {
        const char *sn = NULL; size_t sn_len = 0; int has_ah = 0;
        if (sq_scan(line, line_len, &sn, &sn_len, &has_ah) && !has_ah) {
            if (sn_len + 1 > name_cap) {
                char *tmp = (char *)realloc(name, sn_len + 1);
                if (tmp == NULL) {
                    /* Degrade rather than abort: this function only produces a
                     * diagnostic, so failing the run over it would be worse than
                     * losing it. Say that it was lost. */
                    if (bwa_verbose >= 2)
                        fprintf(stderr, "[W::%s] out of memory checking the "
                                "sidecar for ALT/AH; check skipped.\n", __func__);
                    free(name);
                    kh_destroy(sqname, alt);
                    return;
                }
                name = tmp; name_cap = sn_len + 1;
            }
            memcpy(name, sn, sn_len);
            name[sn_len] = '\0';
            khiter_t k = kh_get(sqname, alt, name);
            /* Report bns's copy of the name, not the scratch buffer. */
            if (k != kh_end(alt) && n_missing++ == 0) first_id = kh_val(alt, k);
        }
    }
    free(name);
    kh_destroy(sqname, alt);
    if (n_missing == 0) return;
    const char *first = bns->anns[first_id].name;
    fprintf(stderr,
            "[W::%s] the <prefix>.hdr / <baseprefix>.dict sidecar supplies @SQ "
            "without an AH tag for %d of the index's ALT contig%s (e.g. %s); "
            "ALT status will be absent from the output header. The sidecar's @SQ is "
            "authoritative and is not modified. Regenerate it with the ALT list:\n"
            "[W::%s]   samtools dict --alt %s.alt -o <sidecar> %s\n",
            __func__, n_missing, n_missing == 1 ? "" : "s", first,
            __func__, prefix ? prefix : "<ref>", prefix ? prefix : "<ref>");
}

// Emit the full SAM header, merging (in precedence order) user `hdr_line`
// (from -H), `bns_hdr` (loaded from <prefix>.hdr or <baseprefix>.dict), and
// the index's @SQ records. Precedence mirrors lh3/bwa#348:
//   @HD : user's > index's > the target's default (none, under a target that
//         emits no @HD -- see below).
//   @SQ : if user supplies any, use user's alone (index .hdr was already
//         skipped by the caller); else index's @SQ; else generated from bns.
//   Other: all remaining lines from bns_hdr, then hdr_line, then bwa_pg.
static void print_sam_hdr(const bntseq_t *bns, const char *bns_hdr,
                          const char *hdr_line, FILE *fp,
                          const compat_target_t *compat)
{
    int i, n_SQ = count_SQ(hdr_line);
    extern char *bwa_pg;
    if (compat == NULL) compat = &COMPAT_TARGET_OFF;

    // Emit exactly one @HD record — the user's (-H) if they gave any, else the
    // sidecar's, else the target's default. Whichever stream owns the winner,
    // EVERY other @HD in either stream is dropped: emitting two is a spec
    // violation, and one bwa does not have (bwa.c:412-426 counts @HD at any
    // line start before deciding). The BAM writer already filtered the losing
    // sidecar records this way, so this keeps the two paths in agreement.
    const int user_HD = has_HD(hdr_line);
    const int idx_HD  = has_HD(bns_hdr);
    int keep_user_HD = user_HD;             // does the tail below keep its @HD?
    int keep_idx_HD  = !user_HD && idx_HD;  // user's -H beats the sidecar's

    // A LEADING winner is consumed off the front of its stream and emitted up
    // front, so it precedes @SQ as the spec requires. A later one cannot be
    // hoisted without reordering the user's own records, so it stays inline and
    // is emitted with the rest of the stream after @SQ — which is where bwa
    // puts every -H record anyway.
    if (keep_user_HD && strncmp(hdr_line, "@HD\t", 4) == 0) {
        hdr_line = hoist_line(hdr_line, fp);
        keep_user_HD = 0;                   // already emitted; drop any others
    }
    if (keep_idx_HD && strncmp(bns_hdr, "@HD\t", 4) == 0) {
        bns_hdr = hoist_line(bns_hdr, fp);
        keep_idx_HD = 0;
    }
    /* @HD policy comes from the selected compat target; the evidence for each
     * row is in src/compat_target.cpp. DO NOT "fix" the missing @HD under a
     * compat target -- suppressing it is deliberate, not an oversight. */
    if (!user_HD && !idx_HD && compat->emit_hd) {
        err_fputs(compat->hd_line, fp);
        err_fputc('\n', fp);
    }

    // Generate @SQ from bns only when neither hdr_line nor bns_hdr supply any.
    if (n_SQ == 0 && !has_SQ(bns_hdr)) {
        kstring_t sq = {0, 0, NULL};
        for (i = 0; i < bns->n_seqs; ++i) {
            sq.l = 0;                      /* reuse the buffer across contigs */
            /* Fatal, like every other write failure on this path (err_fputs
             * aborts): a header missing contigs is not a usable output, and
             * this function has no way to report a partial one to its caller. */
            if (bwa_format_sq_line(&sq, &bns->anns[i]) < 0)
                err_fatal(__func__, "failed to allocate the @SQ record for contig %s",
                          bns->anns[i].name);
            err_fputs(sq.s, fp);
            err_fputc('\n', fp);
        }
        free(sq.s);
    }

    if (n_SQ != 0 && n_SQ != bns->n_seqs && bwa_verbose >= 2)
        fprintf(stderr, "[W::%s] %d @SQ lines provided with -H; %d sequences in the index. "
                "Continue anyway.\n", __func__, n_SQ, bns->n_seqs);

    if (bns_hdr) fputs_filtering_HD(bns_hdr, keep_idx_HD, fp);
    if (hdr_line) fputs_filtering_HD(hdr_line, keep_user_HD, fp);
    if (bwa_pg) err_fputs(bwa_pg, fp);
}

// Load the contents of `<prefix>.hdr` if present, else `<baseprefix>.dict`
// (where baseprefix drops a trailing ".gz" then the final dotted suffix of the
// basename, whatever it is -- e.g. foo.fa -> foo, GRCh38.p14 -> GRCh38), into
// a newly-allocated, newline-separated but not newline-terminated string.
// Returns NULL if neither file exists (or is empty). Caller owns the result
// and must free() it.
char *bwa_load_hdr_from_index(const char *prefix)
{
    if (prefix == NULL) return NULL;

    kstring_t path = { 0, 0, NULL };
    ksprintf(&path, "%s.hdr", prefix);
    FILE *fp = fopen(path.s, "r");
    if (!fp) {
        // Try <baseprefix>.dict: drop the ".hdr" we just appended, then drop
        // a trailing ".gz" (if any) and the final dotted suffix inside the
        // basename — e.g. foo.fa -> foo, foo.fasta.gz -> foo — before
        // appending ".dict".
        size_t l;
        path.l -= 4; // drop ".hdr"
        if (path.l >= 3 && strncmp(&path.s[path.l - 3], ".gz", 3) == 0)
            path.l -= 3;
        for (l = path.l; l > 0 && path.s[l - 1] != '/'; --l) {
            if (path.s[l - 1] == '.') { path.l = l - 1; break; }
        }
        ksprintf(&path, ".dict");
        fp = fopen(path.s, "r");
    }

    char *out = NULL;
    if (fp) {
        kstring_t buf = { 0, 0, NULL };
        int c;
        while ((c = getc(fp)) != EOF)
            if (c != '\r') kputc(c, &buf);
        fclose(fp);

        while (buf.l > 0 && buf.s[buf.l - 1] == '\n') --buf.l;
        if (buf.l > 0) {
            buf.s[buf.l] = '\0';
            out = buf.s; // transfer ownership
        } else {
            free(buf.s);
        }
    }

    free(path.s);
    return out;
}

void bwa_print_sam_hdr2(const bntseq_t *bns, const char *idx_hdr_lines,
                        const char *hdr_line, FILE *fp,
                        const compat_target_t *compat)
{
    // If the user's -H supplies any @SQ, ignore the index .hdr/.dict content
    // entirely — the user has taken responsibility for the @SQ block.
    const char *bns_hdr = has_SQ(hdr_line) ? NULL : idx_hdr_lines;

    if (bns_hdr) {
        int n_SQ = count_SQ(bns_hdr);
        if (n_SQ != 0 && n_SQ != bns->n_seqs && bwa_verbose >= 2)
            fprintf(stderr,
                    "[W::%s] %d @SQ lines loaded from index; %d sequences in the index. "
                    "Continue anyway.\n", __func__, n_SQ, bns->n_seqs);
    }
    print_sam_hdr(bns, bns_hdr, hdr_line, fp, compat);
}

static char *bwa_escape(char *s)
{
    char *p, *q;
    for (p = q = s; *p; ++p) {
        if (*p == '\\') {
            ++p;
            // A trailing backslash has no escape char after it: stop before the
            // for-loop's ++p walks past the NUL and copies/writes out of bounds.
            if (*p == '\0') break;
            else if (*p == 't') *q++ = '\t';
            else if (*p == 'n') *q++ = '\n';
            else if (*p == 'r') *q++ = '\r';
            else if (*p == '\\') *q++ = '\\';
            // (an unrecognized escape is dropped, unchanged behavior)
        } else *q++ = *p;
    }
    *q = '\0';
    return s;
}

char *bwa_set_rg(const char *s)
{
    char *p, *q, *r, *rg_line = 0;
    memset(bwa_rg_id, 0, 256);
    if (strstr(s, "@RG") != s) {
        if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] the read group line is not started with @RG\n", __func__);
        goto err_set_rg;
    }
    rg_line = strdup(s);
    xassert(rg_line != NULL, "out of memory: rg_line");  // bwa_escape derefs it
    bwa_escape(rg_line);
    if ((p = strstr(rg_line, "\tID:")) == 0) {
        if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] no ID at the read group line\n", __func__);
        goto err_set_rg;
    }
    p += 4;
    for (q = p; *q && *q != '\t' && *q != '\n'; ++q);
    if (q - p + 1 > 256) {
        if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] @RG:ID is longer than 255 characters\n", __func__);
        goto err_set_rg;
    }
    for (q = p, r = bwa_rg_id; *q && *q != '\t' && *q != '\n'; ++q)
        *r++ = *q;
    return rg_line;

err_set_rg:
    free(rg_line);
    return 0;
}

// Append `s` to `hdr`, separated by '\n' when `hdr` already holds a prior
// record, WITHOUT touching bwa_escape. Shared by bwa_insert_header (which
// escapes the freshly-joined segment once, immediately after joining) and
// bwa_insert_header_file (which escapes each retained line individually
// before ever reaching this join -- see the comment on that function for why
// that order matters). `*start` receives the offset of the freshly-appended
// segment within the returned buffer, mirroring the accounting bwa_escape
// needs; pass NULL when the caller has no further use for it.
static char *bwa_join_header(const char *s, char *hdr, int *start)
{
    int len = 0;
    if (hdr) {
        len = strlen(hdr);
        int len_s = strlen(s);
        char *tmp = (char*) realloc(hdr, len + len_s + 2);
        xassert(tmp != NULL, "out of memory: hdr");  // don't leak/dangle hdr on failure
        hdr = tmp;
        hdr[len++] = '\n';
        strcpy(hdr + len, s);
    } else hdr = strdup(s);
    xassert(hdr != NULL, "out of memory: hdr");
    if (start) *start = len;
    return hdr;
}

char *bwa_insert_header(const char *s, char *hdr)
{
    if (s == 0 || s[0] != '@') return hdr;
    int start = 0;
    hdr = bwa_join_header(s, hdr, &start);
    bwa_escape(hdr + start);  // bwa_escape derefs hdr; bwa_join_header already xassert'd it
    return hdr;
}

char *bwa_insert_header_file(FILE *fp, char *hdr)
{
    // Batched counterpart to bwa_insert_header: copy every @-prefixed line from
    // fp into a single buffer, then call bwa_insert_header once on the whole
    // thing. The per-line loop in fastmap.cpp was O(n^2) because
    // bwa_insert_header strlen's and realloc's hdr on every call — this makes
    // ingestion of large (~70 MB, ~1.5 M-line) headers linear.
    //
    // Reads line by line into an explicit growable buffer rather than sizing it
    // with fseek/ftell: the latter returns -1 on a non-seekable stream (a pipe,
    // /dev/stdin, or `-H <(...)` process substitution), and the old code then
    // silently discarded the entire -H file. Every realloc is xassert'd so an
    // OOM fails with a diagnostic rather than a NULL write (a raw kstring append
    // would not check). fgets reads at most sizeof(chunk)-1 bytes; a chunk that
    // fills to that bound without a newline while more input remains is a single
    // header line longer than the budget -- rejected (matching the prior
    // contract), and keyed on the filled-budget condition rather than a running
    // byte count so it no longer false-fires on a short unterminated final line.
    //
    // Semantics match the old loop: non-@ lines are dropped; kept @-lines are
    // joined by '\n' with no trailing newline.
    //
    // Each retained line is escaped in place as soon as it is complete, BEFORE
    // any later line's '\n' separator is appended after it, and the final
    // buffer is joined onto `hdr` with bwa_join_header (which does not
    // escape). Escaping the whole concatenation in one pass instead (as a
    // single bwa_insert_header call on `buf`) is NOT equivalent to per-line
    // escaping: a retained line ending in an odd number of backslashes has an
    // incomplete escape sequence at its tail, and bwa_escape doesn't know
    // about line boundaries -- it would consume the very next byte it sees as
    // the second half of that sequence. If that next byte is the '\n'
    // separator we inserted, the separator is silently dropped (or
    // mistranslated) and two SAM header records get merged into one line.
    // Escaping each line the moment it is complete -- exactly when the
    // per-line baseline would have escaped it -- means bwa_escape never sees
    // a byte beyond the line it's escaping.
    size_t cap = 0x10000, len = 0;
    char *buf = (char *) malloc(cap);
    xassert(buf != NULL, "out of memory: header buffer");
    char chunk[0x10000];
    int at_line_start = 1;    // is the next fgets result the start of a new line?
    int keep = 0;             // are we inside an @-line being kept?
    size_t line_start = 0;    // offset in buf where the current line's raw content begins
    while (fgets(chunk, sizeof chunk, fp) != NULL) {
        int clen = (int) strlen(chunk);
        int has_nl = (clen > 0 && chunk[clen - 1] == '\n');
        if (!has_nl && clen == (int)(sizeof chunk) - 1 && !feof(fp))
            err_fatal(__func__, "header line exceeds %d-byte budget", (int)(sizeof chunk) - 1);
        if (at_line_start) keep = (chunk[0] == '@');  // decide once, at the line's first chunk
        if (keep) {
            int add = has_nl ? clen - 1 : clen;         // content without the trailing '\n'
            int sep = (len > 0 && at_line_start) ? 1 : 0;  // separator before a new kept line
            if (len + (size_t)sep + (size_t)add + 1 > cap) {
                while (len + (size_t)sep + (size_t)add + 1 > cap) cap <<= 1;
                char *tmp = (char *) realloc(buf, cap);
                xassert(tmp != NULL, "out of memory: header buffer");
                buf = tmp;
            }
            if (sep) buf[len++] = '\n';
            if (at_line_start) line_start = len;  // this chunk starts a new retained line
            memcpy(buf + len, chunk, add);
            len += add;
            if (has_nl) {
                // The line is complete -- escape it now, before a later
                // line's separator can be appended right after it.
                buf[len] = '\0';
                bwa_escape(buf + line_start);
                len = line_start + strlen(buf + line_start);
            }
        }
        at_line_start = has_nl;  // a continuation chunk (no newline yet) is not a new line
    }
    if (keep && !at_line_start) {
        // The file ended without a trailing newline on the last retained
        // line, so the has_nl branch above never got to escape it.
        buf[len] = '\0';
        bwa_escape(buf + line_start);
        len = line_start + strlen(buf + line_start);
    }
    if (len == 0) { free(buf); return hdr; }
    buf[len] = '\0';
    hdr = bwa_join_header(buf, hdr, NULL);
    free(buf);
    return hdr;
}
