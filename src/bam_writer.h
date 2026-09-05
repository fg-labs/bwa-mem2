/* SPDX-License-Identifier: MIT */
#ifndef BWAMEM3_BAM_WRITER_H
#define BWAMEM3_BAM_WRITER_H

#include <stdint.h>
#include "bwa.h"
#include "bwamem.h"
#include "bntseq.h"

/* htslib/sam.h and bwa-mem3's kstring.h share the KSTRING_H guard, so htslib
 * headers are only included inside src/bam_writer.cpp. Callers get a
 * forward-declared bam1_t and opaque writer handle. */
struct bam1_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bam_writer_s bam_writer_t;

/* Open a BAM writer at `path` ("-" for stdout). compression_level 0 = no
 * deflate (fast, same size as SAM), 1..9 = BGZF deflate levels. @SQ is
 * built directly from `bns->anns`. `idx_hdr_lines` carries header records
 * loaded from <prefix>.hdr or <baseprefix>.dict (or NULL if neither exists,
 * or if the user's `hdr_line` supplies @SQ records — in which case the index
 * file is ignored entirely, matching the SAM text path). These are inserted
 * after the default @HD/@SQ but before user `hdr_line`, so user -H entries
 * win on any @HD/@SQ collision via htslib's header de-dup rules. `hdr_line`
 * carries user-supplied header lines (e.g., `@RG` from `-R` and `-H`
 * insertions, as accumulated by bwa_insert_header); it is inserted before
 * `@PG`. `bwa_pg` is inserted as-is. All three may be NULL. `compat` selects
 * the output-compatibility target whose @HD policy the header follows (NULL =
 * COMPAT_TARGET_OFF); it shapes only the default @HD, never @SQ. Returns NULL
 * on failure. */
bam_writer_t *bam_writer_open(const char *path, const bntseq_t *bns,
                              const char *idx_hdr_lines,
                              const char *hdr_line, const char *bwa_pg,
                              int compression_level,
                              const compat_target_t *compat,
                              int n_bgzf_threads);

/* Write one record. Returns 0 on success, -1 on error. */
int bam_writer_write(bam_writer_t *w, struct bam1_t *b);

/* Close the writer (flushes BGZF EOF marker). Returns 0 on success. */
int bam_writer_close(bam_writer_t *w);

/* bam1_t allocation wrappers — keep htslib out of callers that can't include
 * <htslib/sam.h> because of the kstring.h guard collision. */
struct bam1_t *bam_writer_alloc(void);
void           bam_writer_free(struct bam1_t *b);

/* Append `b` to `s->bams`, growing the array as needed. On OOM, frees `b`. */
void bam_writer_bseq_push(bseq1_t *s, struct bam1_t *b);

/* Convert a mem_aln_t to a bam1_t. Analogous to mem_aln2sam but emits a
 * bam1_t directly with no SAM-text intermediate. Caller owns `b`. */
int mem_aln_to_bam(struct bam1_t *b,
                   const mem_opt_t *opt, const bntseq_t *bns,
                   const bseq1_t *s, int n_alns,
                   const mem_aln_t *list, int which,
                   const mem_aln_t *m);

/* Allocate a bam1_t, fill via mem_aln_to_bam, and append to s->bams. */
int bam_writer_push_aln(bseq1_t *s,
                        const mem_opt_t *opt, const bntseq_t *bns,
                        int n_alns, const mem_aln_t *list, int which,
                        const mem_aln_t *m);

/* Append two groups of aux tags that mem_aln2sam emits unconditionally from
 * the SAM-text path (so BAM and SAM carry the same auxiliary information):
 *   1. Tags parsed from `s->comment` (FASTQ-carried SAM tokens under -C; in
 *      --meth mode this is where YS:Z/YC:Z live too).
 *   2. XR:Z from `bns->anns[rid].anno` when MEM_F_REF_HDR is set (-V).
 *
 * Factored out so both the generic bam_writer path and the meth_bam path
 * produce identical output for --bam vs --meth. `rid` is the bwa-mem3
 * internal contig index (bns-relative), not a post-remap output tid. */
/* Returns 0 on success, -1 if a tag could not be appended (htslib sets errno
 * to ENOMEM or EINVAL). On -1 the record is incomplete and must not be
 * emitted -- a partially-tagged record is the silent corruption this return
 * exists to prevent. */
int bam_writer_append_generic_aux(struct bam1_t *b,
                                  const bseq1_t *s,
                                  const mem_opt_t *opt,
                                  const bntseq_t *bns,
                                  int rid);

#ifdef __cplusplus
}
#endif

#endif
