/* src/compat_target.h — `--compat` output-compatibility targets.
 *
 * A compat target describes the exact SAM/BAM output shaping needed to
 * reproduce another aligner's byte stream. It is DATA, not control flow: the
 * table in compat_target.cpp holds one row per target, and every consumer reads
 * a field off the selected row instead of testing a flag bit. Adding a target
 * is a new row.
 *
 * A single boolean would not do. bwa and bwa-mem2 diverge from each other on
 * the very fields this shapes -- bwa emits MQ:i and a default @HD, bwa-mem2
 * emits neither -- so "compat" is a choice among targets, not an on/off switch.
 *
 * A compat target reproduces ITS upstream's output, records and header alike.
 * Almost every field here is header or tag shaping and changes no alignment,
 * no score and no flag -- and that remains what a new field should expect to
 * be.
 *
 * `chain_flt_resurrect_empty` and `sa_sentinel_drop_offset` are the two known
 * records on which bwa and bwa-mem2 emit DIFFERENT ALIGNMENTS for the same
 * read -- both places where bwa-mem2's port is not faithful to bwa -- so a
 * row records whichever behavior ITS target has, alignment-affecting or not:
 * a `--compat=bwa-mem2` that returned bwa's alignment there would not be a
 * weaker guarantee, it would be a false one. The `off` row (no --compat) takes
 * bwa's answer on both, the principled one: without --compat, bwa-mem3 is
 * bwa-mem2 plus bug fixes.
 *
 * This does not license `--fast` or `--proper-pair-from-emitted` into a row:
 * those deviate from BOTH targets, so asking for target parity and for a
 * deviation from it in one command stays incoherent, and main_mem still
 * rejects the pair. See docs/src/whats-different/equivalence.md.
 */

#ifndef BWAMEM3_COMPAT_TARGET_H
#define BWAMEM3_COMPAT_TARGET_H

#ifdef __cplusplus
extern "C" {
#endif

/* The default @HD record bwa-mem3 emits when neither -H nor the index sidecar
 * supplies one. Byte-identical to upstream bwa (bwa.c:426, added in 0.7.18
 * 6b18630). ONE definition on purpose: the SAM-text, BAM and --meth writers
 * each used to hardcode their own, and the BAM ones had drifted to
 * "VN:1.6 SO:unsorted" -- so the same run emitted a different @HD depending on
 * --bam (fg-labs/bwa-mem3#288). Every emission site now spells it this way. */
#define BWAMEM3_DEFAULT_HD_LINE "@HD\tVN:1.5\tSO:unsorted\tGO:query"

typedef struct compat_target_t {
    /* Canonical spelling, as documented and as reported in diagnostics. */
    const char *name;

    /* One additional accepted spelling, or NULL. */
    const char *alias;

    /* NULL when the target is selectable. Otherwise, WHY it is not: the row is
     * fully specified and unit-tested, but --compat refuses it and prints this
     * instead of a generic "unknown target". For a target whose output shaping
     * we know exactly but whose byte-identity we cannot deliver for other
     * reasons. The reason lives here, next to the evidence the row is built
     * from, so the two cannot drift apart.
     *
     * NO ROW SETS THIS TODAY -- `bwa-mem` was the one, until its blocking
     * measurement was retracted (see compat_target.cpp). The field and its
     * parser arm are kept because they are the row grammar's way of saying
     * "specified, not yet offered", and the next target to need staging will
     * want it rather than a hardcoded string in main_mem's getopt arm. The
     * table test asserts every row is currently selectable, so a row that sets
     * this has to say so deliberately. */
    const char *unavailable_reason;

    /* Emit a default @HD when neither -H nor the index sidecar supplies one. */
    int emit_hd;

    /* Exact @HD text (no trailing newline). Always non-NULL when emit_hd is
     * set, so an emitting site never has to fall back to a literal of its own
     * -- that is exactly how the three writers drifted apart before #288. */
    const char *hd_line;

    /* Honor the bwa-mem3-only <prefix>.hdr / <baseprefix>.dict sidecar. Both
     * upstreams lack the feature entirely (lh3/bwa#348 was closed unmerged), so
     * every non-`off` target must skip it to match. */
    int read_sidecar;

    /* Emit the MQ:i mate mapping quality tag. NOTE: this is not a bwa-mem3
     * invention -- bwa emits it (bwamem.c:935; lh3/bwa#330, merged 2022-03-06)
     * and bwa-mem2 does not, having forked at 0.7.17 before that landed. */
    int emit_mq;

    /* Emit the HN:i hit-count tag. Genuinely bwa-mem3-only: absent from both
     * upstreams. */
    int emit_hn;

    /* When mem_chain_flt's weight filter drops EVERY chain for a read, hand
     * slot 0 -- a chain the filter just rejected -- back to the caller with
     * kept = 3 (1), or report zero survivors (0).
     *
     * One of the two records on which bwa-mem2's port is not faithful to bwa
     * (the other is `sa_sentinel_drop_offset`). bwa returns 0: `n_chn = k`
     * with no post-filter check, so its tail loops are bounded by 0 and the
     * function returns 0 (bwa 0.7.19 bwamem.c). bwa-mem2's seqid-range
     * machinery synthesizes the range {0,1} for an emptied array, so n_chn
     * comes back as 1, the unconditional `kept[0] = 3` becomes load-bearing,
     * and the rejected chain is extended. bwa-mem3 inherited the latter at the
     * fork point (fg-labs/bwa-mem3#310).
     *
     * Reachable only when min_chain_weight > 0 -- never the default -- via -W
     * or the -x pacbio/pbref/ont2d presets. Measured on 500 HiFi reads at
     * -x pacbio it never fires (real long reads build chains far above the
     * threshold); with -W above the read length it fires on every read.
     *
     * `off` and `bwa-mem` report zero survivors: bwa's answer, and the
     * principled one, since extending a chain the filter just rejected is a
     * bookkeeping artifact rather than a decision. `bwa-mem2` resurrects,
     * because reproducing that release's records, this divergence included,
     * is the entire contract of that target. */
    int chain_flt_resurrect_empty;

    /* When the compressed suffix-array lookup's LF walk reaches the sentinel
     * ($) row, report coordinate 0 (1) or the accumulated walk offset (0).
     *
     * The suffix at the sentinel row starts at text position 0, so a walk of
     * `offset` steps that lands there belongs to text position `offset`. bwa's
     * bwt_sa accounts for that (bwt_invPsi returns 0 at the primary and
     * bwt->sa[0] = -1 compensates); bwa-mem2's pipelined lookup (call_one_step)
     * set sa_entry = 0 there and dropped the walk, placing any hit whose walk
     * reaches the sentinel before a sampled row up to `offset` bases too far
     * left. Reachable only for hits in the first few bases of the concatenated
     * reference (contig 0's opening bases), i.e. never on hg38, which opens
     * with an N run, but real on small and custom references.
     *
     * `off` and `bwa-mem` report the correct coordinate; `bwa-mem2` keeps the
     * drop, because reproducing bwa-mem2 v2.2.1's records is that target's
     * contract and this is one of its alignments (fg-labs/bwa-mem3#469). */
    int sa_sentinel_drop_offset;
} compat_target_t;

/* The `off` row: bwa-mem3's own native output. Never NULL on any mem_opt_t
 * built by mem_opt_init(), so consumers can dereference opt->compat
 * unconditionally. (The `opt0` "was this set explicitly" sentinel in main_mem
 * is memset to zero and is NOT such a struct -- it is only ever read for
 * scalar fields.) */
extern const compat_target_t COMPAT_TARGET_OFF;

/* Resolve a user-supplied --compat value, by canonical name or alias.
 *
 * Returns NULL for an unrecognized name. A name that IS recognized but is
 * unavailable resolves to its row, so the caller can print
 * ->unavailable_reason rather than a generic "unknown target"; callers must
 * check that field before use. */
const compat_target_t *compat_target_from_name(const char *name);

/* Comma-separated list of the selectable target names with their aliases, for
 * usage and error text (e.g. "bwa-mem2 (alias mem2), off"). Derived from the
 * table, so a new row cannot leave it stale. Statically allocated. */
const char *compat_target_selectable_list(void);

/* The table itself, for tests and diagnostics. Sets *n to the row count. */
const compat_target_t *const *compat_targets(int *n);

#ifdef __cplusplus
}
#endif

#endif /* BWAMEM3_COMPAT_TARGET_H */
