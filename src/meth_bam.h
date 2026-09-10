/* SPDX-License-Identifier: MIT */
#ifndef BWAMEM3_METH_BAM_H
#define BWAMEM3_METH_BAM_H

#include <stdint.h>
#include "bwa.h"
#include "bwamem.h"
#include "bntseq.h"

/* htslib's <htslib/kstring.h> and bwa-mem3's src/kstring.h share the
 * `KSTRING_H` guard, so including htslib/sam.h here would make one of them
 * a no-op. htslib is therefore only included inside src/meth_bam.cpp; this
 * header forward-declares bam1_t and exposes htslib-free wrappers. */
struct bam1_t;

/* Native BAM emission for `bwa-mem3 mem --meth`. D3 (PR-5): alignments
 * already carry ORIGINAL chrom names/coords (PR-3 seed→original remap), so
 * the @SQ header and per-record RNAME/POS come straight from the original
 * (un-converted) bns — no f/r→chrom consolidation. Converts mem_aln_t
 * records to bam1_t with the Bismark methylation tags (XR/XG/XM) and
 * chimera QC. */

#ifdef __cplusplus
extern "C" {
#endif

/* --- BAM writer lifecycle ------------------------------------------- */

/* Opaque — defined in meth_bam.cpp so this header is htslib-free. */
typedef struct meth_bam_writer_s meth_bam_writer_t;

/* Global BAM writer (set once by main_mem when --meth is active). */
extern meth_bam_writer_t *g_meth_bam_writer;

/* D3 (--meth, PR-5): the ORIGINAL (un-converted) reference's 2-bit pac,
 * set once by main_mem under --meth. meth_mem_aln_to_bam needs it to build
 * XM:Z against the original reference; the matching original bns reaches the
 * conversion as the `bns` argument (already the original under --meth). NULL
 * outside --meth. */
extern const uint8_t *g_meth_orig_pac;

/* Open a meth record writer at path ("-" for stdout). `bam` selects the
 * container: non-zero = BGZF BAM (`--bam`), 0 = plain-text SAM (the default,
 * matching non-meth output). Both write the SAME bam1_t records built by
 * meth_mem_aln_to_bam — only htslib's serialization differs — so the meth
 * overlay (XM:Z/XG:Z/XR:Z, chimera QC) is identical either way.
 * `bns` is the ORIGINAL (un-converted) reference whose contigs become the @SQ
 * block directly (D3 PR-5: no f/r consolidation). `compression_level` is the
 * BGZF deflate level: 0 = uncompressed, 1..9 = deflate; ignored when `bam` is
 * 0. `hdr_line` is the assembled user
 * header text (-R read group and/or -H lines, '\n'-joined, no trailing
 * newline), or NULL — emitted verbatim after the @SQ block so a -R read
 * group lands as an @RG header, matching the default (non-meth) writer.
 * `orig_idx_hdr_lines` is the original reference's .hdr/.dict sidecar text,
 * or NULL: its @SQ identity tags (M5/UR/AS/SP) are merged into the @SQ for
 * each contig whose SN and LN both match (an LN mismatch means a
 * stale/foreign sidecar and is skipped), and its @CO/@PG/@RG records are
 * forwarded after @SQ; its @HD and @SQ lines are dropped. Returns NULL on
 * failure. */
meth_bam_writer_t *meth_bam_writer_open(const char *path_or_dash,
                                        const bntseq_t *bns,
                                        const char *bwa_pg,
                                        const char *meth_pg_cl,
                                        const char *hdr_line,
                                        const char *orig_idx_hdr_lines,
                                        int bam,
                                        int compression_level,
                                        int n_bgzf_threads);

/* Write one bam1_t. Returns 0 on success, -1 on error. */
int meth_bam_writer_write(meth_bam_writer_t *w, struct bam1_t *b);

/* Close the writer, flushing the BGZF EOF marker when the container is BAM.
 * Frees internal hdr and htsFile. Returns 0 on success, -1 on error. */
int meth_bam_writer_close(meth_bam_writer_t *w);

/* --- mem_aln_t -> bam1_t --------------------------------------------- */

/* Convert one alignment to a bam1_t with meth transforms applied:
 *   - D3 (PR-5): RNAME/POS come straight from p->rid/p->pos (already in
 *     ORIGINAL-reference coordinates after the PR-3 seed→original remap);
 *     NO f/r→chrom rewrite. `bns`/`pac` are the ORIGINAL (un-converted)
 *     reference handles (mem_aln_bns/mem_aln_pac under --meth).
 *   - XG:Z genome-strand tag sourced from the winning hypothesis
 *     (p->meth_hypothesis: 1=OT→CT/top, 0=OB→GA/bottom), NOT f/r contig dir.
 *   - XM:Z built against the original bns/pac with strand from the hypothesis.
 *   - chimera QC (only when opt->meth_chimera_qc is set): if longest M/=/X run < 44%
 *     of l_seq, set 0x200, clear 0x2, cap mapq at 1
 *   - opt->meth_set_as_failed forces 0x200 on the matching strand ('f'=OT, 'r'=OB)
 * Caller owns `b`. Returns 0 on success, -1 on error. */
int meth_mem_aln_to_bam(struct bam1_t *b,
                        const mem_opt_t *opt, const bntseq_t *bns,
                        const uint8_t *pac,
                        const bseq1_t *s, int n_alns,
                        const mem_aln_t *list, int which,
                        const mem_aln_t *m_);

/* If any record in `group` has 0x200 set, propagate it to the rest and
 * clear 0x2 on those. */
void meth_bam_group_propagate_qcfail(struct bam1_t **group, int n);

#ifdef __cplusplus
}
#endif

#endif
