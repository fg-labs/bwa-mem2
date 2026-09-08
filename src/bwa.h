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

#ifndef BWA_H_
#define BWA_H_

#include <stdint.h>
#include <stdio.h>
#include <zlib.h>
#include "bntseq.h"
#include "bwt.h"
#include "macro.h"
#include "compat_target.h"
#include "kstring.h"
#include "read_arena.h"

#define BWA_IDX_BWT 0x1
#define BWA_IDX_BNS 0x2
#define BWA_IDX_PAC 0x4
#define BWA_IDX_ALL 0x7

#define BWA_CTL_SIZE 0x10000

/* Buffer size for bwa_format_pa_value(). The widest "%.3f" of an int/int
 * ratio is "-2147483648.000" (15 chars + NUL), so this cannot truncate. */
#define BWA_PA_TEXT_MAX 20

typedef struct {
	bwt_t    *bwt; // FM-index
	bntseq_t *bns; // information on the reference sequences
	uint8_t  *pac; // the actual 2-bit encoded reference sequences with 'N' converted to a random base

	int    is_shm;
	int64_t l_mem;
	uint8_t  *mem;
} bwaidx_t;


typedef struct {
	int l_seq, id;
	char *name, *comment, *seq, *qual, *sam;
	/* BAM output: per-read list of bam1_t* accumulated by worker threads
	 * (used by --bam and --meth alike). Typed as void* to keep htslib out
	 * of this header. */
	void **bams;
	int    n_bams;
	int    cap_bams;
	/* --meth (D3) only: the ORIGINAL, unconverted read bases captured at
	 * ingest *before* the in-place C->T (R1) / G->A (R2) projection
	 * overwrites `seq`. NUL-terminated ASCII, length `l_seq`. NULL for
	 * non-meth reads (no behavior change).
	 *
	 * This is a first-class copy of what is otherwise only preserved as the
	 * `YS:Z:` comment string, so the D3 extension/scoring (PR-4) and the
	 * SAM/XM writers (PR-5) can read the original bases without parsing a
	 * comment in the hot path.
	 *
	 * ORIENTATION CONTRACT (spec S1): stored in the SAME orientation/order as
	 * `seq` at ingest — i.e. original read order, exactly as it came off the
	 * sequencer. It is NOT pre-reverse-complemented. Every downstream consumer
	 * MUST apply the SAME reverse-complement to `meth_orig_seq` that it applies
	 * to `seq`/the projected read — notably the extension/scoring path and
	 * `mem_matesw` (which reverse-complements the mate before rescue), and the
	 * SAM/XM emitters. Storing `seq` projected-forward while CIGAR/MD describe
	 * the original (or vice versa) yields an internally inconsistent BAM. */
	char  *meth_orig_seq;
	/* --meth (D3) only: the read's bisulfite chemistry from read number
	 * (R1 = 1 = OT/C->T, R2 = 0 = OB/G->A; SE = R1). -1 for non-meth. Used by the
	 * seed-chemistry filter in meth_seed_to_orig to drop cross-chemistry seeds. */
	int8_t meth_base_ot;
} bseq1_t;

extern int bwa_verbose;
extern char bwa_rg_id[256];

#ifdef __cplusplus
extern "C" {
#endif
    /* On success (return != NULL) *arena_out receives the per-chunk bump arena
     * that backs the returned reads' name/seq/qual fields; the caller owns it
     * and must read_arena_destroy() it after the chunk is fully consumed. At
     * EOF (return NULL) *arena_out is set to NULL and there is nothing to free.
     * See read_arena.h and the PIPE-F6 note for the ownership contract. */
    bseq1_t *bseq_read_orig(int64_t chunk_size, int *n_, void *ks1_, void *ks2_, int64_t *s,
                            read_arena_t **arena_out);

    bseq1_t *bseq_read(int64_t chunk_size, int *n_, void *ks1_,
                       void *ks2_, FILE* fpp, int len,
                       int64_t *sz, read_arena_t **arena_out);

    bseq1_t *bseq_read_one_fasta_file(int64_t chunk_size, int *n_, gzFile fp, int64_t *s,
                                      read_arena_t **arena_out);
    
    void bseq_classify(int n, bseq1_t *seqs, int m[2], bseq1_t *sep[2]);
    
	void bwa_fill_scmat(int a, int b, int8_t mat[25]);
    
	uint32_t *bwa_gen_cigar(const int8_t mat[25], int q, int r, int w_,
							int64_t l_pac, const uint8_t *pac, int l_query,
							uint8_t *query, int64_t rb, int64_t re,
							int *score, int *n_cigar, int *NM);
	
	uint32_t *bwa_gen_cigar2(const int8_t mat[25], int o_del, int e_del,
							 int o_ins, int e_ins, int w_, int64_t l_pac,
							 const uint8_t *pac, int l_query, uint8_t *query,
							 int64_t rb, int64_t re, int *score,
							 int *n_cigar, int *NM);

	/* As bwa_gen_cigar2, plus `nm_from_mat`: 0 derives NM/MD from literal base
	 * inequality (bwa default); 1 derives them from the scoring matrix, so a
	 * cell the matrix does not penalise is a match. --meth uses 1 so bisulfite
	 * conversions are matches for NM/MD as well as for the DP. */
	uint32_t *bwa_gen_cigar3(const int8_t mat[25], int o_del, int e_del,
							 int o_ins, int e_ins, int w_, int64_t l_pac,
							 const uint8_t *pac, int l_query, uint8_t *query,
							 int64_t rb, int64_t re, int *score,
							 int *n_cigar, int *NM, int nm_from_mat);

	/* emit_unpacked_ref=0 (the default) skips writing `<prefix>.0123`. `mem`
	 * pac-fetches the original reference from `<prefix>.pac` on demand
	 * (bns_get_seq_v2), so the unpacked `.0123` (~8x the `.pac`) is never read
	 * and not worth building. Pass 1 only to emit it for an external consumer
	 * that still requires the unpacked file (e.g. bwa-mem2). `int` (not bool)
	 * and a C++-only default keep this declaration valid C: bwa.h is inside
	 * extern "C" and is transitively included by fast_reader_bseq.c.
	 *
	 * sa_compx=3 (the default) reproduces the historical SA sample rate
	 * (1/8) byte-for-byte; `bwa-mem3 index -u INT` overrides it, validated
	 * to [0,6] by the CLI before it reaches here. */
#ifdef __cplusplus
	int bwa_idx_build(const char *fa, const char *prefix, int emit_unpacked_ref = 0, int sa_compx = 3);
#else
	int bwa_idx_build(const char *fa, const char *prefix, int emit_unpacked_ref, int sa_compx);
#endif

	/* `compat` selects the output-compatibility target whose @HD policy the
	 * header follows; NULL means COMPAT_TARGET_OFF (bwa-mem3's native output).
	 * It shapes only the default @HD -- a user's -H or the index sidecar still
	 * wins, and @SQ is untouched. */
	void bwa_print_sam_hdr2(const bntseq_t *bns, const char *idx_hdr_lines,
	                        const char *hdr_line, FILE *fp,
	                        const compat_target_t *compat);
	/* Append the generated @SQ record for `ann` (SN, LN, and AH:* when the
	 * contig is ALT) to `out`, without a trailing newline. The single
	 * definition shared by the SAM-text, --bam and --meth writers. Returns 0,
	 * or -1 if `out` could not be grown -- on -1 no record was appended and
	 * `out->s` may be NULL, so callers must not emit `out`. */
	int bwa_format_sq_line(kstring_t *out, const bntann1_t *ann);
	/* Render the `pa` tag value -- the ratio of a hit's score to the score of
	 * the better overlapping ALT hit -- into `buf`, which must be at least
	 * BWA_PA_TEXT_MAX bytes. Returns the length written, excluding the NUL.
	 *
	 * The single definition shared by the SAM-text, --bam and --meth writers.
	 * It exists because those three used to render `pa` independently: the SAM
	 * path rounded to three decimals (as both upstreams do) while the two BAM
	 * paths stored the raw quotient, so a `--bam` run and a `samtools view -b`
	 * of the same run's SAM disagreed on ~90% of the records that carry the
	 * tag (fg-labs/bwa-mem3#365). `alt_sc` must be non-zero. The return is
	 * always positive: there is no short-render path, because one would make
	 * the SAM and BAM writers disagree again. */
	int bwa_format_pa_value(char *buf, int score, int alt_sc);
	/* The same value as the float32 a BAM `pa:f:` field holds, i.e. exactly
	 * what a consumer that parses the SAM token above would store. Derived
	 * from that token rather than computed independently: rounding the
	 * quotient arithmetically is NOT equivalent, because printf breaks a tie
	 * at the fourth decimal to even where round() breaks it away from zero
	 * (39/48 renders as 0.812, not 0.813). */
	float bwa_pa_tag_value(int score, int alt_sc);
	/* True if SAM header text `s` contains a record whose type is `tag`
	 * ("@HD\t", "@SQ\t", ...), as the first line or after a newline. The type
	 * may be any length; an empty one matches nothing. */
	int bwa_hdr_text_has_type(const char *s, const char *tag);
	/* Iterate newline-separated SAM header records. Start with *p at the text;
	 * sets *line/*len to the next record (no newline) and advances *p, returning
	 * 1, or returns 0 at end of text. */
	int bwa_hdr_next_line(const char **p, const char **line, size_t *len);
	char *bwa_load_hdr_from_index(const char *prefix);
	/* Warn (at bwa_verbose >= 2) when the index marks contigs ALT but the
	 * sidecar's @SQ block carries no AH for them. The sidecar is authoritative
	 * and is never modified; this only reports the gap and its remedy. */
	void bwa_warn_sidecar_missing_AH(const bntseq_t *bns,
	                                 const char *idx_hdr_lines,
	                                 const char *prefix);
	char *bwa_set_rg(const char *s);
	char *bwa_insert_header(const char *s, char *hdr);
	char *bwa_insert_header_file(FILE *fp, char *hdr);
#ifdef __cplusplus
}
#endif

#endif
