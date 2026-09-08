/* SPDX-License-Identifier: MIT */

/* htslib headers must come before any bwa-mem3 header that pulls in
 * bwa-mem3's kstring.h (they share the KSTRING_H include guard). */
#include "htslib/sam.h"
#include "htslib/kstring.h"

#include "meth_bam.h"
#include "bam_rec_scratch.h"
#include "bam_writer.h"
#include "cigar_util.h"
#include "meth_xm.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Private writer struct — opaque to callers via meth_bam.h. */
struct meth_bam_writer_s {
    htsFile   *fp;
    sam_hdr_t *hdr;
};

/* Declared in src/bwa.h (without extern "C"); match that linkage here. */
extern char bwa_rg_id[256];

meth_bam_writer_t *g_meth_bam_writer = NULL;

/* ------------------------------------------------------------------- */
/* BAM writer                                                           */
/* ------------------------------------------------------------------- */

/* Replace embedded tabs in a string with spaces (BAM CL tag fields can't
 * contain literal tabs — safer to just sanitize). Caller's buffer. */
static void sanitize_cl(char *s)
{
    for (; *s; ++s) if (*s == '\t') *s = ' ';
}

/* True if the tab-delimited @SQ `line` (length `line_len`) carries a token
 * exactly equal to `key``val` (whole token, not a prefix) — e.g. key="SN:"
 * val="<name>", or key="LN:" val="<decimal>". */
static int sq_line_has_kv(const char *line, size_t line_len,
                          const char *key, size_t key_len,
                          const char *val, size_t val_len)
{
    const char *end = line + line_len;
    const char *p = line + 3;                 /* "@SQ" -> first '\t' */
    while (p < end) {
        const char *tok = p + 1;              /* skip the '\t' */
        const char *tok_end = (const char *)memchr(tok, '\t', (size_t)(end - tok));
        if (tok_end == NULL) tok_end = end;
        if ((size_t)(tok_end - tok) == key_len + val_len &&
            strncmp(tok, key, key_len) == 0 &&
            strncmp(tok + key_len, val, val_len) == 0)
            return 1;
        p = tok_end;
    }
    return 0;
}

/* Append to `out` the reference-identity tags (M5/UR/AS/SP) found on the @SQ
 * line of `hdr_text` whose SN equals `sn`, each as "\t<TAG>:<value>". Used to
 * enrich the --meth @SQ with identity tags from the *original* (pre-c2t)
 * reference's .hdr/.dict sidecar, matched by SN. Enrichment is gated on the
 * sidecar line's LN also equalling `expected_len` (the contig length from the
 * original bns): a SN match with a differing LN means a stale or foreign sidecar
 * whose M5/UR would misidentify the sequence, so nothing is appended in that
 * case. Appends nothing if `hdr_text` is NULL/empty or has no matching @SQ.
 * SN is unique in a valid header, so the first match wins. */
/* Returns 0 on success (including "nothing to append"), -1 if a kstring append
 * failed. The caller must not add the @SQ line on -1: `out` would be a
 * truncated record, and emitting a header line missing half its identity tags
 * is worse than failing the open. */
static int meth_append_sq_extra_tags(const char *hdr_text, const char *sn,
                                     int64_t expected_len, kstring_t *out)
{
    if (hdr_text == NULL || hdr_text[0] == '\0' || sn == NULL) return 0;
    static const char *const WANT[] = { "M5:", "UR:", "AS:", "SP:" };
    const int n_want = (int)(sizeof(WANT) / sizeof(WANT[0]));
    const size_t sn_len = strlen(sn);
    char ln_buf[32];
    int ln_len = snprintf(ln_buf, sizeof(ln_buf), "%lld", (long long)expected_len);
    if (ln_len <= 0 || (size_t)ln_len >= sizeof(ln_buf)) return 0;

    const char *cur = hdr_text, *line; size_t line_len;
    while (bwa_hdr_next_line(&cur, &line, &line_len)) {
        if (line_len >= 4 && strncmp(line, "@SQ\t", 4) == 0 &&
            sq_line_has_kv(line, line_len, "SN:", 3, sn, sn_len)) {
            /* SN matched: only enrich when LN agrees, else the sidecar is
             * stale/foreign and its identity tags can't be trusted. */
            if (!sq_line_has_kv(line, line_len, "LN:", 3, ln_buf, (size_t)ln_len))
                return 0;
            const char *end = line + line_len;
            const char *p = line + 3;
            while (p < end) {
                const char *tok = p + 1;
                const char *tok_end = (const char *)memchr(tok, '\t', (size_t)(end - tok));
                if (tok_end == NULL) tok_end = end;
                for (int i = 0; i < n_want; ++i) {
                    if ((size_t)(tok_end - tok) > 3 && strncmp(tok, WANT[i], 3) == 0) {
                        if (kputc('\t', out) < 0 ||
                            kputsn(tok, (int)(tok_end - tok), out) < 0)
                            return -1;
                        break;
                    }
                }
                p = tok_end;
            }
            return 0;                         /* SN is unique; first match wins */
        }
    }
    return 0;
}

/* Append to `out` every record line of `hdr_text` that is NOT @HD or @SQ
 * (i.e. @CO/@PG/@RG and any others), each terminated by '\n'. @SQ is skipped
 * because the @SQ block built from the original reference is authoritative
 * (its identity tags are merged separately by meth_append_sq_extra_tags); @HD
 * is skipped because the
 * writer emits its own.
 *
 * Returns 0 on success, -1 if a kstring append failed -- the same contract as
 * meth_append_sq_extra_tags above, and for the same reason: on -1 `out` holds a
 * truncated record set, so the caller must not pass it to sam_hdr_add_lines. */
static int meth_append_passthrough_records(const char *hdr_text, kstring_t *out)
{
    if (hdr_text == NULL) return 0;
    const char *cur = hdr_text, *line; size_t line_len;
    while (bwa_hdr_next_line(&cur, &line, &line_len)) {
        int skip = (line_len >= 4 && (strncmp(line, "@HD\t", 4) == 0 ||
                                      strncmp(line, "@SQ\t", 4) == 0));
        if (!skip && line_len > 0) {
            if (kputsn(line, (int)line_len, out) < 0 || kputc('\n', out) < 0)
                return -1;
        }
    }
    return 0;
}

meth_bam_writer_t *meth_bam_writer_open(const char *path_or_dash,
                                        const bntseq_t *bns,
                                        const char *bwa_pg,
                                        const char *meth_pg_cl,
                                        const char *hdr_line,
                                        const char *orig_idx_hdr_lines,
                                        int bam,
                                        int compression_level)
{
    if (path_or_dash == NULL || bns == NULL) return NULL;
    meth_bam_writer_t *w = (meth_bam_writer_t *)calloc(1, sizeof(*w));
    if (w == NULL) return NULL;

    if (compression_level < 0) compression_level = 0;
    if (compression_level > 9) compression_level = 9;
    /* The container is the ONLY difference between the two output formats: the
     * records below are the same bam1_t either way, and htslib serializes them
     * to BGZF ("wb<level>") or to text ("w"). Keeping one construction path is
     * what makes `--meth` and `--meth --bam` byte-identical after decode. */
    char mode[8];
    if (bam) snprintf(mode, sizeof(mode), "wb%d", compression_level);
    else     snprintf(mode, sizeof(mode), "w");
    w->fp = hts_open(path_or_dash, mode);
    if (w->fp == NULL) { free(w); return NULL; }

    w->hdr = sam_hdr_init();
    if (w->hdr == NULL) { hts_close(w->fp); free(w); return NULL; }

    /* @HD — skip the default when the user's -H already supplies one, else
     * htslib would write two @HD lines (it does not de-dup @HD).
     *
     * This writer does not consult a compat target, unlike bwa.cpp and
     * bam_writer.cpp: --compat with --meth is a hard error (rejected in
     * main_mem's option validation), so only COMPAT_TARGET_OFF can reach here,
     * and OFF's hd_line IS BWAMEM3_DEFAULT_HD_LINE. A future target that
     * relaxes the --meth exclusion must plumb compat->emit_hd/hd_line through
     * here too. */
    if (!bwa_hdr_text_has_type(hdr_line, "@HD\t") &&
        sam_hdr_add_lines(w->hdr, BWAMEM3_DEFAULT_HD_LINE, 0) < 0) goto fail;

    /* @SQ directly from the ORIGINAL (un-converted) bns contigs (D3 PR-5: no
     * f/r consolidation — alignments already carry original rids). Each @SQ is
     * enriched by SN with the original reference's identity tags (M5/UR/AS/SP)
     * from its .hdr/.dict sidecar when available; the extra tags are appended
     * verbatim, gated on a matching SN+LN.
     *
     * AH:* on ALT contigs is appended from the ORIGINAL bns, matching both
     * upstreams' generated @SQ (bwa/bwa.c:432, bwa-mem2/src/bwa.cpp:538) and
     * our SAM text path; this block was written without it and
     * dropped ALT status for every ALT-aware reference. It cannot collide with
     * the sidecar enrichment: meth_append_sq_extra_tags copies only
     * M5/UR/AS/SP, never AH. Appended BEFORE those tags so the line reads
     * SN, LN, AH, then identity tags -- the order bwa emits. */
    {
        kstring_t sq = {0, 0, NULL};
        for (int i = 0; i < bns->n_seqs; ++i) {
            sq.l = 0;                    /* reuse the buffer across contigs */
            if (bwa_format_sq_line(&sq, &bns->anns[i]) < 0) { free(sq.s); goto fail; }
            if (meth_append_sq_extra_tags(orig_idx_hdr_lines, bns->anns[i].name,
                                          bns->anns[i].len, &sq) < 0) {
                free(sq.s);
                goto fail;
            }
            /* No `sq.s != NULL` guard: that check existed because the old
             * inline ksprintf could leave sq.s NULL with no way to say so.
             * bwa_format_sq_line reports that as -1 above, and a failed
             * enrichment append is reported as -1 too -- so by here sq holds a
             * complete record. bam_writer.cpp's copy of this loop never had it. */
            int rc = sam_hdr_add_lines(w->hdr, sq.s, sq.l);
            if (rc < 0) { free(sq.s); goto fail; }
        }
        free(sq.s);
    }

    /* Original reference's non-@HD/@SQ sidecar records (@CO/@PG/@RG),
     * forwarded after @SQ and before the user header — mirroring the index
     * tier of bam_writer_open's precedence (index > default, below user). Its
     * @SQ is consumed into the enriched @SQ block above and its @HD
     * is the writer's own, so both are dropped here. `orig_idx_hdr_lines` is
     * loaded from the *original* (pre-c2t) reference prefix by main_mem; the
     * c2t index's own sidecar (which describes the doubled f/r converted
     * contigs) is never consulted. */
    if (orig_idx_hdr_lines != NULL && orig_idx_hdr_lines[0] != '\0') {
        kstring_t pt = {0, 0, NULL};
        if (meth_append_passthrough_records(orig_idx_hdr_lines, &pt) < 0) {
            free(pt.s);
            goto fail;
        }
        int rc = (pt.l > 0) ? sam_hdr_add_lines(w->hdr, pt.s, pt.l) : 0;
        free(pt.s);
        if (rc < 0) goto fail;
    }

    /* User header text (-R read group, -H lines). Emitted after @SQ and the
     * forwarded sidecar records, before the @PG lines, matching the default
     * writer's precedence (user > index > default) so a -R read group becomes
     * an @RG header that the per-record RG:Z tags (stamped from bwa_rg_id
     * below) actually reference. The @SQ block above is authoritative
     * for --meth, so a user -H that re-supplies @SQ for an already-emitted
     * contig will make htslib reject the add and the open fails loudly rather
     * than emitting duplicate references. */
    if (hdr_line != NULL && hdr_line[0] != '\0' &&
        sam_hdr_add_lines(w->hdr, hdr_line, 0) < 0) goto fail;

    /* Original bwa-mem3 @PG */
    if (bwa_pg != NULL && bwa_pg[0] != '\0') {
        if (sam_hdr_add_lines(w->hdr, bwa_pg, 0) < 0) goto fail;
    }

    /* bwa-mem3-meth @PG — grow as needed so very long CLs aren't truncated. */
    {
        const char *cl_in = (meth_pg_cl && meth_pg_cl[0]) ? meth_pg_cl
                                                          : "bwa-mem3 mem --meth";
        char *cl_copy = strdup(cl_in);
        if (cl_copy == NULL) goto fail;
        sanitize_cl(cl_copy);
        kstring_t pg = {0, 0, NULL};
        ksprintf(&pg, "@PG\tID:bwa-mem3-meth\tPN:bwa-mem3-meth\tVN:%s-meth\tCL:%s\n",
                 PACKAGE_VERSION, cl_copy);
        free(cl_copy);
        int rc = (pg.s != NULL) ? sam_hdr_add_lines(w->hdr, pg.s, 0) : -1;
        free(pg.s);
        if (rc < 0) goto fail;
    }

    if (sam_hdr_write(w->fp, w->hdr) < 0) goto fail;
    return w;

fail:
    sam_hdr_destroy(w->hdr);
    hts_close(w->fp);
    free(w);
    return NULL;
}

int meth_bam_writer_write(meth_bam_writer_t *w, bam1_t *b)
{
    if (w == NULL || b == NULL) return -1;
    return sam_write1(w->fp, w->hdr, b);
}

int meth_bam_writer_close(meth_bam_writer_t *w)
{
    if (w == NULL) return 0;
    int rc = 0;
    if (w->hdr) { sam_hdr_destroy(w->hdr); w->hdr = NULL; }
    if (w->fp) {
        if (hts_close(w->fp) < 0) rc = -1;
        w->fp = NULL;
    }
    free(w);
    return rc;
}

/* ------------------------------------------------------------------- */
/* mem_aln_t -> bam1_t                                                  */
/* ------------------------------------------------------------------- */

/* bwameth.py's chimera heuristic: flag 0x200 + cap mapq when the longest
 * alignment run covers < MIN_LONGEST_M_PCT% of the read. */
static constexpr int MIN_LONGEST_M_PCT = 44;

/* Contig name for an SA:Z hit, "*" when the rid names no contig. The SA:Z
 * block sizes its buffer in one loop and fills it in another; both must agree
 * on the name — and therefore on its length — or the buffer is under-sized, so
 * they share this resolver rather than each repeating the bounds check. */
static inline const char *meth_sa_ref_name(const bntseq_t *bns, int32_t rid)
{
    const char *name = (rid >= 0 && rid < bns->n_seqs) ? bns->anns[rid].name : NULL;
    return name != NULL ? name : "*";
}

int meth_mem_aln_to_bam(bam1_t *b,
                        const mem_opt_t *opt, const bntseq_t *bns,
                        const uint8_t *pac,
                        const bseq1_t *s, int n_alns,
                        const mem_aln_t *list, int which,
                        const mem_aln_t *m_)
{
    if (b == NULL || opt == NULL || s == NULL || list == NULL
            || bns == NULL || pac == NULL) return -1;

    /* Reused per-record scratch, shared with the non-meth writer. */
    static thread_local bwamem3::BamRecScratch bs;

    /* Local copies so flag/rid can be mutated without touching the caller's. */
    mem_aln_t p = list[which];
    mem_aln_t m;
    const mem_aln_t *mp = NULL;
    if (m_ != NULL) { m = *m_; mp = &m; }

    p.flag |= mp ? 0x1 : 0;
    p.flag |= p.rid < 0 ? 0x4 : 0;
    p.flag |= mp && mp->rid < 0 ? 0x8 : 0;
    if (p.rid < 0 && mp && mp->rid >= 0) {
        p.rid = mp->rid; p.pos = mp->pos; p.is_rev = mp->is_rev; p.n_cigar = 0;
    }
    if (mp && mp->rid < 0 && p.rid >= 0) {
        m.rid = p.rid; m.pos = p.pos; m.is_rev = p.is_rev; m.n_cigar = 0;
    }
    p.flag |= p.is_rev ? 0x10 : 0;
    p.flag |= mp && mp->is_rev ? 0x20 : 0;

    /* Fold bwa-mem3's high-bit supp flag (0x10000) down into BAM's 0x100. */
    uint16_t flag16 = (uint16_t)((p.flag & 0xffff) | (p.flag & 0x10000 ? 0x100 : 0));

    /* D3 (PR-5): RNAME/POS are already in ORIGINAL-reference coordinates after
     * the PR-3 seed→original remap, so RNAME = p.rid directly (no f/r→chrom
     * map). The genome-strand call ('f'=OT/top, 'r'=OB/bottom) is sourced from
     * the winning hypothesis (p.meth_hypothesis: 1=OT, 0=OB), NOT from a contig
     * name prefix. `direction` is retained as the legacy 'f'/'r' char only so
     * the XG:Z text and the opt->meth_set_as_failed strand filter keep their
     * existing encoding. */
    int32_t tid = (p.rid >= 0 && p.rid < bns->n_seqs) ? p.rid : -1;
    int32_t mtid = (mp && mp->rid >= 0 && mp->rid < bns->n_seqs) ? mp->rid : -1;
    /* The `meth_hypothesis >= 0` term is DEFENSIVE, not a reachable branch: a
     * mapped record under --meth always carries 0 or 1. The value reaches `p`
     * through mem_reg2aln, which defaults it to -1 and then copies the region's
     * hypothesis over that default. A region (mem_alnreg_t) gets its hypothesis
     * in exactly three places, all of which supply 0 or 1 under --meth:
     *   - mem_chain2aln_across_reads_V2 (bwamem.cpp), copying the chain's
     *     hypothesis (itself set in chain_add_one_seed) -- chain seeds that
     *     cannot be remapped to an OT/OB hypothesis are dropped rather than
     *     carried as -1;
     *   - mem_matesw and mem_matesw_batch_post (bwamem_pair.cpp), the two
     *     mate-rescue paths.
     * The term is kept so an unmapped or non-meth record, which does keep the
     * -1 default, cannot index off ((-1)&1)==1 and be labelled top-strand. Do
     * not read it as evidence that -1 reaches a mapped record. */
    char direction = 0;
    if (tid >= 0 && p.meth_hypothesis >= 0)
        direction = (p.meth_hypothesis & 1) ? 'f' : 'r';

    /* Remap primary CIGAR: bwa-mem3 ops -> BAM ops, + soft->hard for supp */
    uint32_t *bam_cigar = NULL;
    size_t    bam_n_cigar = 0;
    if (p.n_cigar > 0) {
        bam_cigar = bs.ensure_cigar((size_t)p.n_cigar);
        if (bam_cigar == NULL) return -1;
        for (int i = 0; i < p.n_cigar; ++i) {
            int op  = p.cigar[i] & 0xf;
            int len = p.cigar[i] >> 4;
            if (!(opt->flag & MEM_F_SOFTCLIP) && !p.is_alt && (op == 3 || op == 4))
                op = which ? 4 : 3;              /* hard clip for supp */
            uint32_t bam_op = (op >= 0 && op < 5) ? BAM_OP_FROM_MEM[op] : 0;
            bam_cigar[i] = ((uint32_t)len << 4) | bam_op;
        }
        bam_n_cigar = (size_t)p.n_cigar;
    }

    /* TLEN is defined only when both mates are placed on the same contig, which
     * is what SAM means by RNAME==RNEXT. Since D3 PR-5, tid/mtid ARE p.rid /
     * mp->rid (original rids straight from the seed→original remap), so this is
     * a plain same-chromosome test. Pre-D3 it was not: output ran on
     * consolidated contigs, and mates rescued onto opposite projected strands
     * (f* vs r*) of one real chromosome had different internal rids but the same
     * output tid — hence the gate on tid/mtid rather than the raw rids. */
    hts_pos_t tlen = 0;
    if (mp && tid >= 0 && mtid >= 0 && tid == mtid
        && p.n_cigar > 0 && m.n_cigar > 0) {
        int64_t p_rlen = cigar_ref_len_mem(p.cigar, p.n_cigar);
        int64_t m_rlen = cigar_ref_len_mem(m.cigar, m.n_cigar);
        int64_t p0 = p.pos + (p.is_rev ? p_rlen - 1 : 0);
        int64_t p1 = m.pos + (m.is_rev ? m_rlen - 1 : 0);
        tlen = -(p0 - p1 + (p0 > p1 ? 1 : p0 < p1 ? -1 : 0));
    }

    /* Compute SEQ/QUAL range with supp soft-clip trim */
    int qb = 0, qe = s->l_seq;
    if (p.n_cigar && which && !(opt->flag & MEM_F_SOFTCLIP) && !p.is_alt) {
        if (!p.is_rev) {
            int c0 = p.cigar[0] & 0xf;
            int cN = p.cigar[p.n_cigar-1] & 0xf;
            if (c0 == 3 || c0 == 4) qb += p.cigar[0] >> 4;
            if (cN == 3 || cN == 4) qe -= p.cigar[p.n_cigar-1] >> 4;
        } else {
            int c0 = p.cigar[0] & 0xf;
            int cN = p.cigar[p.n_cigar-1] & 0xf;
            if (c0 == 3 || c0 == 4) qe -= p.cigar[0] >> 4;
            if (cN == 3 || cN == 4) qb += p.cigar[p.n_cigar-1] >> 4;
        }
    }

    /* Restore pre-c2t bases so MethylDackel sees real C/Ts. CodeRabbit: prefer
     * the first-class meth_orig_seq field (native-alphabet output no longer
     * depends on comment preservation); fall back to the YS:Z comment only when
     * meth_orig_seq is unavailable. FASTQ ingest builds comments as
     * "YS:Z:<l_seq bytes>\tYC:Z:XX" starting at offset 0 — rely on that. */
    const char *orig_seq = s->meth_orig_seq;
    if (orig_seq == NULL && s->comment && s->l_seq > 0
        && s->comment[0] == 'Y' && s->comment[1] == 'S' && s->comment[2] == ':') {
        orig_seq = s->comment + 5;
    }

    int emit_seq = !(p.flag & 0x100);
    size_t l_emit = 0;
    char *seq_text = NULL;
    char *qual_bin = NULL;
    if (emit_seq && qe > qb) {
        l_emit = (size_t)(qe - qb);
        seq_text = bs.ensure_seq(l_emit + 1);
        if (seq_text == NULL) return -1;
        if (orig_seq != NULL) {
            /* Apply is_rev RC + supp soft-clip trim over pre-c2t bases. */
            if (!p.is_rev) {
                for (size_t i = 0; i < l_emit; ++i) {
                    char c = orig_seq[qb + (int)i];
                    seq_text[i] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : c;
                }
            } else {
                for (size_t i = 0; i < l_emit; ++i) {
                    unsigned char c = (unsigned char)orig_seq[qe - 1 - (int)i];
                    /* nst_nt4_table maps '-' to 5, which indexes the literal's
                     * NUL terminator (in bounds, but emits '\0' into SEQ --
                     * corrupt output, not an OOB read); clamp anything >=4 to N.
                     * Byte-identical for A/C/G/T/N. */
                    int bi = nst_nt4_table[c];
                    seq_text[i] = "TGCAN"[bi < 4 ? bi : 4];
                }
            }
        } else if (!p.is_rev) {
            /* Fallback: YS:Z (orig_seq) was missing. s->seq holds ASCII
             * (post-c2t in meth mode); map ASCII → 0..4 via nst_nt4_table. */
            for (size_t i = 0; i < l_emit; ++i) {
                unsigned char c = (unsigned char)s->seq[qb + (int)i];
                int bi = nst_nt4_table[c];
                seq_text[i] = "ACGTN"[bi < 4 ? bi : 4];
            }
        } else {
            for (size_t i = 0; i < l_emit; ++i) {
                unsigned char c = (unsigned char)s->seq[qe - 1 - (int)i];
                int bi = nst_nt4_table[c];
                seq_text[i] = "TGCAN"[bi < 4 ? bi : 4];
            }
        }
        seq_text[l_emit] = '\0';
        if (s->qual) {
            qual_bin = bs.ensure_qual(l_emit);
            if (qual_bin == NULL) return -1;
            if (!p.is_rev) {
                for (size_t i = 0; i < l_emit; ++i) qual_bin[i] = (char)((unsigned char)s->qual[qb + (int)i] - 33);
            } else {
                for (size_t i = 0; i < l_emit; ++i) qual_bin[i] = (char)((unsigned char)s->qual[qe - 1 - (int)i] - 33);
            }
        }
    }

    /* Chimera QC + set-as-failed applied here so group propagation upstream
     * only scans flags. */
    uint8_t mapq = p.mapq;
    int mapped = !(flag16 & 0x4) && direction != 0;
    if (mapped) {
        if (opt->meth_set_as_failed != 0 && opt->meth_set_as_failed == direction) {
            flag16 |= 0x200;
        }
        if (opt->meth_chimera_qc && p.n_cigar > 0 && s->l_seq > 0) {
            int lm = cigar_longest_m_mem(p.cigar, p.n_cigar);
            if (100 * lm < MIN_LONGEST_M_PCT * s->l_seq) {
                flag16 |= 0x200;
                flag16 &= ~0x2;
                if (mapq > 1) mapq = 1;
            }
        }
    }

    /* Build XM:Z BEFORE bam_set1 frees seq_text and bam_cigar. Stashed for
     * appending after the bam_set1 call below — bam_aux_append needs the
     * record to be initialized first.
     *
     * D3 (PR-5): is_top_strand keys off the winning hypothesis (= XG:Z), NOT
     * the SAM 0x10 RC flag: methylation is on the top strand for OT
     * (p.meth_hypothesis==1, any read including CTOT is_rev=1 R2s) and on the
     * bottom strand for OB. The XM ref source is the ORIGINAL bns/pac (tid is
     * the original rid); meth_build_xm reads the original read-C-vs-T at ref-C
     * from `seq_text`, which is restored from meth_orig_seq above. */
    char *xm = NULL;
    if ((opt->meth_tags & MEM_METH_TAG_XM)
        && mapped && seq_text != NULL && bam_cigar != NULL && l_emit > 0) {
        /* The `>= 0` term cannot fire here: this block is gated on `mapped`,
         * which requires `direction != 0`, and `direction` is only set when
         * meth_hypothesis >= 0. So -1 is already excluded before this line --
         * the term is redundant under the current control flow, not a guard
         * against something that happens.
         *
         * It is kept for symmetry with the XG `direction` guard above, and so
         * the two stay consistent if `mapped` is ever loosened: should a -1
         * reach here then, it is treated as bottom strand, matching the
         * XG:Z:GA that guard would emit. Dropping the term instead would let
         * ((-1)&1)==1 mislabel such a record top-strand and pair XG:GA with a
         * top-strand XM. */
        int is_top_strand = (p.meth_hypothesis >= 0 && (p.meth_hypothesis & 1)) ? 1 : 0;
        xm = meth_build_xm(bns, pac, tid, (int64_t)p.pos,
                           is_top_strand,
                           bam_cigar, (int)bam_n_cigar,
                           seq_text, (int)l_emit,
                           (meth_chem_t)opt->meth_chem);
    }

    /* Build the bam1_t. bam_set1 handles 4-bit packing, name storage, etc. */
    int ret = bam_set1(b,
                       strlen(s->name), s->name,
                       flag16,
                       tid,
                       (hts_pos_t)p.pos,
                       mapq,
                       bam_n_cigar, bam_cigar,
                       mtid,
                       mp ? (hts_pos_t)mp->pos : -1,
                       tlen,
                       l_emit, seq_text, qual_bin,
                       /* l_aux */ 0);

    /* bam_cigar/seq_text/qual_bin are the reused thread-local scratch — no free
     * here; bam_set1 has already copied them into b->data. Same for xm, which
     * aliases meth_xm.cpp's own thread-local scratch. */
    if (ret < 0) return -1;

    /* Aux tags — roughly match mem_aln2sam emission order */
    if (p.n_cigar > 0) {
        int32_t nm = (int32_t)p.NM;
        if (bam_aux_append(b, "NM", 'i', sizeof(nm), (const uint8_t *)&nm) < 0) return -1;
        const char *md = (const char *)(p.cigar + p.n_cigar);
        if (bam_aux_append(b, "MD", 'Z', (int)strlen(md) + 1, (const uint8_t *)md) < 0) return -1;
    }
    if (mp && mp->n_cigar > 0) {
        /* Reused thread-local kstring, grown as needed. */
        kstring_t *mc = &bs.mc;
        mc->l = 0;   /* reset the reused scratch; capacity/mc->s persist */
        /* Propagate OOM: on failure ks_resize leaves the buffer too small, and
         * kputw/kputc swallow their own failures, so without this the record
         * could emit with MC silently truncated/omitted. */
        if (ks_resize(mc, (size_t)mp->n_cigar * 12 + 1) < 0) return -1;   /* worst case: 10 digits + op + NUL */
        for (int i = 0; i < mp->n_cigar; ++i) {
            int op = mp->cigar[i] & 0xf;
            int len = mp->cigar[i] >> 4;
            if (!(opt->flag & MEM_F_SOFTCLIP) && !mp->is_alt && (op == 3 || op == 4))
                op = which ? 4 : 3;
            /* kputw + kputc instead of ksprintf("%d%c"): same bytes, no vsnprintf
             * (which ran twice per op — measure then format). */
            kputw(len, mc);
            kputc("MIDSH"[op], mc);
        }
        if (mc->l > 0)
            if (bam_aux_append(b, "MC", 'Z', (int)mc->l + 1, (const uint8_t *)mc->s) < 0) return -1;
        /* no free: bs.mc.s persists across records, freed on thread exit */
    }
    /* MQ:i — guarded on `mp` alone, not on the MC block's `mp->n_cigar > 0`, so
     * a mate with no CIGAR still gets its MAPQ. The compat gate is always open
     * under --meth (fastmap.cpp refuses --compat with --meth) but is read here
     * anyway so this stays a copy of bam_writer.cpp rather than a divergence. */
    if (mp && opt->compat->emit_mq) {
        int32_t mq = (int32_t)mp->mapq;
        if (bam_aux_append(b, "MQ", 'i', sizeof(mq), (const uint8_t *)&mq) < 0) return -1;
    }
    if (p.score >= 0) {
        int32_t as = (int32_t)p.score;
        if (bam_aux_append(b, "AS", 'i', sizeof(as), (const uint8_t *)&as) < 0) return -1;
    }
    if (p.sub >= 0) {
        int32_t xs = (int32_t)p.sub;
        if (bam_aux_append(b, "XS", 'i', sizeof(xs), (const uint8_t *)&xs) < 0) return -1;
    }
    if (bwa_rg_id[0]) {
        if (bam_aux_append(b, "RG", 'Z', (int)strlen(bwa_rg_id) + 1, (const uint8_t *)bwa_rg_id) < 0) return -1;
    }
    /* SA:Z (other primary hits) and pa:f — mirrors mem_aln2sam, which emits
     * both only for non-secondary records and in this order.
     *
     * D3 (PR-5), SA:Z only: rids are already ORIGINAL, so the contig name comes
     * straight from bns->anns (no f/r→chrom rewrite). */
    if (!(p.flag & 0x100)) {
        int has_other = 0;
        for (int i = 0; i < n_alns; ++i)
            if (i != which && !(list[i].flag & 0x100)) { has_other = 1; break; }
        if (has_other) {
            kstring_t *sa = &bs.sa;
            sa->l = 0;   /* reset the reused scratch; capacity/sa->s persist */
            /* Propagate OOM like the MC:Z path above: kputs/kputw/kputc swallow
             * their own failures, so without pre-sizing the record could emit
             * with SA silently truncated/omitted. Worst case per hit: rname +
             * n_cigar*(10 digits + op) + 64 (pos/strand/mapq/NM/commas/;). Both
             * loops resolve the name through meth_sa_ref_name so the length
             * measured here is the length written below. */
            size_t max_sa_len = 0;
            for (int i = 0; i < n_alns; ++i) {
                if (i == which || (list[i].flag & 0x100)) continue;
                const mem_aln_t *r = &list[i];
                max_sa_len += strlen(meth_sa_ref_name(bns, r->rid))
                            + (size_t)r->n_cigar * 12 + 64;
            }
            if (ks_resize(sa, max_sa_len + 1) < 0) return -1;
            for (int i = 0; i < n_alns; ++i) {
                const mem_aln_t *r = &list[i];
                if (i == which || (r->flag & 0x100)) continue;
                const char *r_name = meth_sa_ref_name(bns, r->rid);
                kputs(r_name, sa); kputc(',', sa);
                kputl(r->pos + 1, sa); kputc(',', sa);
                kputc("+-"[r->is_rev], sa); kputc(',', sa);
                for (int k = 0; k < r->n_cigar; ++k) {
                    kputw(r->cigar[k] >> 4, sa);
                    kputc("MIDSH"[r->cigar[k] & 0xf], sa);
                }
                kputc(',', sa); kputw(r->mapq, sa);
                kputc(',', sa); kputw(r->NM, sa);
                kputc(';', sa);
            }
            if (sa->l > 0)
                if (bam_aux_append(b, "SA", 'Z', (int)sa->l + 1, (const uint8_t *)sa->s) < 0) return -1;
            /* no free: bs.sa.s persists across records, freed on thread exit */
        }
        if (p.alt_sc > 0) {
            /* Same shared value as the SAM-text and --bam writers; see
             * bwa_pa_tag_value. --compat is refused under --meth, but a
             * --meth --bam consumer still has to see the SAM rendering. */
            float pa_f = bwa_pa_tag_value(p.score, p.alt_sc);
            if (bam_aux_append(b, "pa", 'f', sizeof(pa_f), (const uint8_t *)&pa_f) < 0) return -1;
        }
    }
    /* XA:Z — D3 (PR-5): p.XA is produced by mem_gen_alt against the ORIGINAL
     * bns, so its `name,+/-pos,cigar,NM;` entries already carry original contig
     * names. No f/r prefix stripping or chrom rewrite is needed; emit verbatim. */
    if (p.XA != NULL) {
        if (bam_aux_append(b, "XA", 'Z', (int)strlen(p.XA) + 1, (const uint8_t *)p.XA) < 0) return -1;
    }
    /* HN:i — the same mem_gen_alt call that fills p.XA fills p.HN, so this
     * counts the hits XA enumerates. Like XA above it is already stated against
     * the ORIGINAL bns, so the doubled C->T/G->A reference does not double-count
     * it. See the MQ:i note above re: the compat gate. */
    if (p.HN >= 0 && opt->compat->emit_hn) {
        int32_t hn = (int32_t)p.HN;
        if (bam_aux_append(b, "HN", 'i', sizeof(hn), (const uint8_t *)&hn) < 0) return -1;
    }
    /* Bismark-compatible XR:Z (read conversion) emitted on every record;
     * XG:Z (genome strand) and XM:Z (methylation call string) only on
     * mapped records. Each is additionally gated on its --meth-tags bit, and
     * the gate wraps the value's derivation rather than just the append, so a
     * deselected tag costs nothing to compute -- the contract --help and the
     * methylation docs state. The YC:Z payload in s->comment carries the (CT|GA)
     * value (the comment buffer is "YS:Z:<seq>\tYC:Z:<dir>", see
     * src/fastmap.cpp:415-425). Locate it with a marker search rather
     * than l_seq arithmetic so any future YS framing change (alternate
     * prefix length, prior FASTQ comments folded in) stays detectable. */
    if (opt->meth_tags & MEM_METH_TAG_XR) {
        const char *xr = NULL;
        if (s->comment != NULL) {
            const char *yc = strstr(s->comment, "\tYC:Z:");
            if (yc != NULL) {
                const char *p2 = yc + 6;  /* skip "\tYC:Z:" */
                if      (p2[0] == 'C' && p2[1] == 'T') xr = "CT";
                else if (p2[0] == 'G' && p2[1] == 'A') xr = "GA";
            }
        }
        if (xr != NULL) {
            if (bam_aux_append(b, "XR", 'Z', 3, (const uint8_t *)xr) < 0) return -1;
        }
    }
    if (mapped) {
        if (opt->meth_tags & MEM_METH_TAG_XG) {
            /* XG:Z genome strand from the winning hypothesis: OT→CT, OB→GA
             * (direction is the 'f'/'r' encoding of p.meth_hypothesis above). */
            const char *xg = (direction == 'f') ? "CT" : "GA";
            if (bam_aux_append(b, "XG", 'Z', 3, (const uint8_t *)xg) < 0) return -1;
        }
        if (xm != NULL) {
            /* xm aliases thread-local scratch in meth_xm.cpp; do not free. */
            if (bam_aux_append(b, "XM", 'Z', (int)l_emit + 1,
                               (const uint8_t *)xm) < 0) return -1;
        }
    }

    /* Generic aux shared with the --bam path so --meth -C emits FASTQ tags.
     * The YS:Z/YC:Z meth carriers in s->comment are filtered inside
     * append_sam_aux_tokens under opt->meth_mode so they don't leak into
     * the BAM output (they're internal carriers only — XR:Z replaces YC,
     * SEQ restoration replaces YS). D3 (PR-5): p.rid is the ORIGINAL contig
     * index, which is what the original `bns` passed here uses. */
    if (bam_writer_append_generic_aux(b, s, opt, bns, p.rid) < 0) return -1;

    return 0;
}

void meth_bam_group_propagate_qcfail(bam1_t **group, int n)
{
    if (group == NULL || n <= 0) return;
    int any_fail = 0;
    for (int i = 0; i < n; ++i) {
        if (group[i] != NULL && (group[i]->core.flag & 0x200)) { any_fail = 1; break; }
    }
    if (!any_fail) return;
    for (int i = 0; i < n; ++i) {
        if (group[i] == NULL) continue;
        group[i]->core.flag |= 0x200;
        group[i]->core.flag &= (uint16_t)~0x2;
    }
}
