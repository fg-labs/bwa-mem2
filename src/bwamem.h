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

#ifndef BWAMEM_HPP
#define BWAMEM_HPP

/* --adaptive-band start band; the chain-geometry retry expands per pair from here */
#define ADAPTIVE_BAND_START 20

#include "bwt.h"
#include "bntseq.h"
#include "bwa.h"
#include "kthread.h"
#include "compat_target.h"
#include "macro.h"
#include "bandedSWA.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>   /* INT32_MAX -- seqbuf_grow_capacity() bound */
#include <math.h>
#include <vector>
#include "kstring.h"
#include "ksw.h"
#include "kvec.h"
#include "ksort.h"
#include "utils.h"
#include "macro.h"
#include "profiling.h"
#include "FMI_search.h"

#define MEM_MAPQ_COEF 30.0
#define MEM_MAPQ_MAX  60

/* Bounds on the --rescue-kmer / --rescue-band options (mem_opt_t::rescue_kmer,
 * ::rescue_band below). K is capped where a k-mer code stops fitting the uint32
 * the anchor index packs it into; beyond that every K would behave as K=16, so
 * the CLI rejects it rather than silently clamping. The band is capped well past
 * any insert-sized rescue window -- a wider band already covers the whole window,
 * so asking for one is a mistake worth naming -- and the cap also keeps the
 * `anchor_diagonal + read_length + band` arithmetic clear of int overflow. */
#define MEM_RESCUE_KMER_MAX 16
#define MEM_RESCUE_BAND_MAX 1000000
/* K used by both bare `--rescue-kmer` and the --fast preset (the measured
 * wall-time optimum); named so the two parse sites and the help text cannot
 * drift apart. The band default is named for the same reason: mem_opt_init sets
 * it and matesw_kmer_narrow repeats it as a defensive fallback for a caller that
 * sets the field directly, and those two must not drift. */
#define MEM_RESCUE_KMER_DEFAULT 6
#define MEM_RESCUE_BAND_DEFAULT 50

/* Vote thresholds on the anchor scan (matesw_kmer_anchor in bwamem_pair.cpp).
 * Fixed constants rather than CLI knobs: they are chosen from a truth-based
 * sweep, and exposing them would ship combinations no sweep ever covered. The
 * #ifndef guards exist so a characterization build can override one from the
 * command line (-DMEM_RESCUE_MIN_VOTES=N) without editing this header.
 *
 * MIN_VOTES gates NARROWING: the winning diagonal must gather at least this many
 * k-mer votes before the rescue SW is banded to it.
 *
 * SKIP_MIN_VOTES / SKIP_FRAC gate SKIPPING the rescue SW outright
 * (opt->rescue_skip): skip when the best diagonal clears neither an absolute
 * floor nor a fraction of the distinct query k-mers the anchor index holds
 * (n_kmer, which is the read's full distinct count until the index stops
 * inserting at 768 codes). The fractional term only binds when n_kmer < ~30
 * (reads under ~36 bp at k=6).
 *
 * Note the skip floor is an absolute count while the vote budget is not: a
 * 150 bp read offers ~145 6-mers and a 75 bp read ~70, so the same 10 votes is
 * twice as harsh a test on short reads. Measured skip rates bear this out --
 * 14% on 150 bp simulated reads, 42% on 125 bp, 79% on 75 bp -- which is why
 * --rescue-skip is opt-in and is NOT part of --fast. */
#ifndef MEM_RESCUE_MIN_VOTES
#  define MEM_RESCUE_MIN_VOTES 3
#endif
#ifndef MEM_RESCUE_SKIP_MIN_VOTES
#  define MEM_RESCUE_SKIP_MIN_VOTES 10
#endif
#ifndef MEM_RESCUE_SKIP_FRAC
#  define MEM_RESCUE_SKIP_FRAC 0.33
#endif

struct __smem_i;
typedef struct __smem_i smem_i;

#define MEM_F_PE        0x2
#define MEM_F_NOPAIRING 0x4
#define MEM_F_ALL       0x8
#define MEM_F_NO_MULTI  0x10
#define MEM_F_NO_RESCUE 0x20
#define MEM_F_REF_HDR   0x100
#define MEM_F_SOFTCLIP  0x200
#define MEM_F_SMARTPE   0x400

// V17
#define MEM_F_PRIMARY5  0x800
#define MEM_F_KEEP_SUPP_MAPQ 0x1000
#define MEM_F_XB        0x2000
// --compat is NOT a flag bit: it selects one of several output-compatibility
// targets, which disagree with each other on the fields it shapes (bwa emits
// MQ:i and a default @HD; bwa-mem2 emits neither). See mem_opt_t::compat and
// src/compat_target.h.


/* D3 (--meth): bisulfite/TAPS SW scoring model, selected by --meth-scoring. All
 * three free the conversion cell (OT: ref-C x read-T; OB: ref-G x read-A); they
 * differ in to what value, and whether the mirror cell is freed too:
 *   COLLAPSED: conversion + mirror both freed to +a (C/T interchangeable,
 *              bwameth-compatible placement, -B 2).
 *   GENOMIC:   conversion freed to +a, mirror kept as a mismatch (variant-aware,
 *              -B 4, rank-1 batched-expressible).
 *   NEUTRAL:   conversion freed to 0 (tolerated but not rewarded), mirror kept a
 *              mismatch (-B 4). For TAPS, whose conversions are sparse (~3% of C),
 *              a full-match reward over-credits spurious C->T alignments; scoring
 *              the conversion neutrally measured ~+0.25 pp placement over GENOMIC
 *              at every methylation load. Not rank-1 (the freed value is neither
 *              match nor mismatch), but the generalized kswv freed-cell blend
 *              scores the cell to its matrix value, so mate rescue is batched on
 *              the freed-capable tiers (NEON/AVX2/AVX512BW) exactly like
 *              GENOMIC/COLLAPSED, and falls back to scalar ksw_align2 on the
 *              freed-less x86 tiers (sse41/sse42/avx) as all three modes do. */
enum mem_meth_scoring { MEM_METH_SCORING_COLLAPSED = 0, MEM_METH_SCORING_GENOMIC = 1,
                        MEM_METH_SCORING_NEUTRAL = 2 };

/* --meth-seed-prune modes (mem_opt_t.meth_seed_prune). Shared here so the CLI
 * (fastmap.cpp) and the prune (bwamem.cpp) agree on the numeric values. */
enum mem_meth_seed_prune { MEM_METH_PRUNE_OFF = 0, MEM_METH_PRUNE_SPEC30 = 1,
                           MEM_METH_PRUNE_BASELINE = 2 };

/* Bismark tag selection under --meth (--meth-tags). XR:Z and XG:Z are two-byte
 * strand labels; XM:Z is a read-length methylation-call string and dominates the
 * BAM's aux payload, so it is the one worth being able to turn off. */
#define MEM_METH_TAG_XR   0x1
#define MEM_METH_TAG_XG   0x2
#define MEM_METH_TAG_XM   0x4
#define MEM_METH_TAGS_ALL (MEM_METH_TAG_XR | MEM_METH_TAG_XG | MEM_METH_TAG_XM)

/* Parse a --meth-tags spec into a MEM_METH_TAG_* bitmask.
 *
 * Grammar: `all` | `none` | a comma-separated list of tag names, either all
 * plain (an inclusion set: `XR,XG`) or all exclusions (subtracted from the full
 * set: `^XM`). An exclusion may be written `^XM` or `-XM`; the latter needs no
 * shell quoting. Mixing plain and excluded terms is rejected. Tag names are
 * case-insensitive. Returns 0 on success and writes *out; returns -1 on a
 * malformed spec and writes a human-readable reason to *err (a static string).
 * Exposed for unit testing. */
int mem_opt_parse_meth_tags(const char *spec, int *out, const char **err);

typedef enum {
    SEED_ORDER_OFF = 0,
    SEED_ORDER_GLOBAL_LONGEST,
    SEED_ORDER_LOCAL_LONGEST,
    SEED_ORDER_ABSORB_COUNT,
    SEED_ORDER_MOST_ABSORB
} seed_order_t;

typedef struct mem_opt_t {
    int a, b;               // match score and mismatch penalty
    int o_del, e_del;
    int o_ins, e_ins;
    int pen_unpaired;       // phred-scaled penalty for unpaired reads
    int pen_clip5,pen_clip3;// clipping penalty. This score is not deducted from the DP score.
    int w;                  // band width
    int zdrop;              // Z-dropoff

    uint64_t max_mem_intv;

    int T;                  // output score threshold; only affecting output
    int flag;               // see MEM_F_* macros
    int min_seed_len;       // minimum seed length
    int min_ext_len;        // seeds shorter than this are not extended (0 = off)
    int max_extend_chains;  // cap on chains extended per read: keep only the top-N by weight before banded-SW (0 = off). Opt-in speed lever; NOT byte-identical.
    int mate_concordant_window;  // --extend-mate-concordant: when max_extend_chains caps a PE read, also retain chains concordant with a mate chain within this many bp. 0 = off; -1 = auto (use the estimated proper-pair insert high bound); >0 = fixed window. Recovers the true pair's low-weight chain (mainly --meth). NOT byte-identical.
    int est_insert_high;         // runtime state (NOT a user option): upper proper-pair insert bound (pes[FR].high) estimated from data during the run, or from -I; 0 = not yet estimated. Read by the mate-concordant cap when mate_concordant_window == -1 (auto).
    float extend_tie_frac;       // --extend-tie-frac: extend a chain (ranked at/after extend_tie_floor) only if its weight >= this fraction of the best chain's weight. Complements max_extend_chains (the count cap) and also gates on its own when that cap is off. 0 = off (byte-identical); clamped to [0,1]. Trims non-competitive tail chains from banded-SW. NOT byte-identical when > 0.
    int extend_csub;             // --extend-csub: when the cap/gate drops chains, seed the primary region's competitor score (calibrated dropped-chain estimate, clamped < primary score) so MAPQ is not inflated by the pruning. 0 = off (byte-identical). Opt-in.
    int extend_tie_floor;        // --extend-tie-floor: always extend at least this many top-ranked chains regardless of extend_tie_frac. 0 = no unconditional floor (the fraction gate governs from rank 0 up; the best chain always survives). Only meaningful when extend_tie_frac > 0.
    seed_order_t seed_emit_order;  // --seed-order; SEED_ORDER_OFF = byte-identical
    int min_chain_weight;
    int max_chain_extend;
    float split_factor;     // split into a seed if MEM is longer than min_seed_len*split_factor
    int split_width;        // split into a seed if its occurence is smaller than this value
    int max_occ;            // skip a seed if its occurence is larger than this value
    int max_chain_gap;      // do not chain seed if it is max_chain_gap-bp away from the closest seed
    int n_threads;          // number of threads
    int64_t chunk_size;         // process chunk_size-bp sequences in a batch
    float mask_level;       // regard a hit as redundant if the overlap with another better hit is over mask_level times the min length of the two hits
    float drop_ratio;       // drop a chain if its seed coverage is below drop_ratio times the seed coverage of a better chain overlapping with the small chain
    float XA_drop_ratio;    // when counting hits for the XA tag, ignore alignments with score < XA_drop_ratio * max_score; only effective for the XA tag
    float mask_level_redun;
    float mapQ_coef_len;
    int mapQ_coef_fac;
    int max_ins;            // when estimating insert size distribution, skip pairs with insert longer than this value
    int max_matesw;         // perform maximally max_matesw rounds of mate-SW for each end
    int rescue_kmer;        // k-mer-anchored banded mate rescue: 0=off, else anchor k-mer length in [1,MEM_RESCUE_KMER_MAX] (opt-in, not byte-identical)
    int rescue_band;        // half-width (bp) of the band around the k-mer anchor diagonal, in [1,MEM_RESCUE_BAND_MAX]
    int rescue_skip;        // 1 = skip the mate-rescue SW outright when no k-mer anchor clears the vote floor; requires rescue_kmer (opt-in, not byte-identical)
    int max_XA_hits, max_XA_hits_alt; // if there are max_hits or fewer, output them all
    int8_t mat[25];         // scoring matrix; mat[0] == 0 if unset
    /* D3 (--meth): per-hypothesis substitution matrices, built from `mat` by
     * mem_opt_fill_meth_mat() per opt->meth_scoring (target-major mat[ref*5+read],
     * ACGT order A,C,G,T,N). Each frees the conversion cell — mat_ot frees
     * mat[C][T]=mat[1*5+3] (top strand C→T), mat_ob frees mat[G][A]=mat[2*5+0]
     * (bottom strand G→A) — to a value that depends on the mode:
     *   GENOMIC: that cell alone is freed, to a MATCH (+a); the mirror
     *     (mat[T][C], mat[A][G]) STAYS at −b so genuine variants score as
     *     mismatches → one freed cell at +a ⇒ the SIMD rank-1 fast path.
     *     Variant-aware: real variants stay visible in NM/MD.
     *   NEUTRAL: that cell alone is freed, to 0 (tolerated, not rewarded); the
     *     mirror STAYS at −b. One freed cell, but NOT rank-1 (the value is
     *     neither match nor mismatch) → bandedSWA's general path; the kswv
     *     freed-cell blend expresses it directly. Variant-aware: real variants
     *     stay visible in NM/MD.
     *   COLLAPSED: the mirror cell is ALSO freed (two cells, both to +a) so C/T
     *     and G/A are interchangeable → reproduces bwameth; uses bandedSWA's
     *     general path.
     * Outside --meth these are unused; selection is by mem_chain_t.meth_hypothesis
     * (1 = OT, 0 = OB) via mem_opt_meth_mat(). */
    int8_t mat_ot[25];      // OT (C→T) meth matrix; valid only under --meth
    int8_t mat_ob[25];      // OB (G→A) meth matrix; valid only under --meth
    int    bam_mode;        // 1 = emit BAM instead of SAM text (--bam). Orthogonal
                            // to meth_mode: --meth picks alignment semantics, --bam
                            // picks the container, on every mode alike.
    int    bam_level;       // 0..9, BGZF deflate level (0 = uncompressed)
    int    bam_threads;     // BGZF deflate threads for --bam output (--bam-threads).
                            // -1 = unset -> auto (n_threads/8 when bam_level>0);
                            // 0 = serial on the writer thread; N = that many.
    int    meth_mode;       // 1 = bisulfite mode (--meth)
    int    meth_scoring;    // bisulfite matrix mode (--meth-scoring):
                            // MEM_METH_SCORING_{COLLAPSED,GENOMIC,NEUTRAL}
    int    meth_seed_prune; // --meth-seed-prune: prune 3-letter over-seeding before
                            // SA resolution. 0 = off, 1 = spec30 (per-read two-regime),
                            // 2 = baseline (len<25 & hits>1). mem_opt_init sets 0, but
                            // the CLI defaults it to spec30 under --meth (bwa-mem3-native
                            // feature, no upstream reference to preserve); --meth-seed-prune
                            // =off|baseline (or env BWAMEM3_METH_SEED_PRUNE) overrides.
                            // ~30% faster --meth at ~0 truth accuracy cost; NOT byte-identical.
    int    meth_chem;       // methylation chemistry (--meth=emseq|taps): meth_chem_t.
                            // Selects XM:Z call polarity ONLY -- seeding, the
                            // converted index and the scoring matrices are shared,
                            // because both chemistries produce the same C->T /
                            // G->A base change as far as the aligner can see.
    int    meth_tags;       // bitmask of Bismark tags to emit (--meth-tags):
                            // MEM_METH_TAG_{XR,XG,XM}. Tags not selected are
                            // neither computed nor emitted -- clearing XM skips
                            // the per-read meth_build_xm() pass entirely.
    char   meth_set_as_failed;// 'f', 'r', or 0 — flag reads on that strand 0x200
    int    meth_chimera_qc; // 1 to enable bwameth.py-style longest-M <44% chimera heuristic (default off; not in Bismark)
    /* Derive the proper-pair FLAG bit (0x2) from the alignment actually EMITTED
     * (a[which]) instead of the top-scoring region (a[0]). Opt-in; 0 = match
     * bwa and bwa-mem2, which both use a[0] unconditionally.
     *
     * fg-labs/bwa-mem3#17 made a[which] the default, reasoning that deriving the
     * bit from a[0] lets it describe a placement no emitted record carries. That
     * reasoning stands, which is why the behavior is still reachable -- but 0x2
     * is aligner-defined ("properly aligned according to the aligner"), so it is
     * a choice rather than a correction, and as a default it made bwa-mem3
     * differ from BOTH upstreams at once (fg-labs/bwa-mem3#362).
     *
     * Inert without a `.alt` sidecar: a[which] != a[0] requires n_pri < a.n (the
     * read has ALT hits), and is_alt is never set without one.
     *
     * Deliberately NOT a compat_target_t field. Making the default match both
     * upstreams means no target needs a say in it, which keeps that table's
     * "output shaping only, never an alignment change" invariant intact -- FLAG
     * is an alignment-record field. --compat and this option are instead
     * mutually exclusive (see main_mem). */
    int    proper_pair_from_emitted;
    int    supp_rep_hard_cap; // supp alnregs whose chain's seeds share >=this many genome hits are forced to MAPQ=0; 0 disables
    int    smem_dedup;        // 1 = dedup fully-identical SMEMs before SA expansion (--smem-dedup); 0 = off (default, byte-identical to baseline)
    int    alnreg_sort_fast;  // 1 = strict-total-order comparator + pdqsort at the mem_sort_dedup_patch sort sites (set by --fast); 0 = bwa-mem2's re-only comparator + ks_introsort (default, bwa-mem2-compatible)
    int    skip_contained_ext; // 1 = skip banded-SW extension of seeds contained (same diagonal) in a longer in-chain seed (--skip-contained-ext); 0 = off. Byte-identical to baseline: the skip set is a subset of the post-extension containment purge (PE18).
    int    band_start;       // >0 = adaptive chain-geometry banding active (start band; set to ADAPTIVE_BAND_START by --adaptive-band); 0 = off (byte-identical). Long-read speed lever; no-op on the 8-bit short-read tier.
    int    band_cert;        // 1 = sound (byte-identical) adaptive band via per-pair tie-break certificate (default); 0 = off, set by --fast/--adaptive-band (which use the aggressive band_start heuristic instead). Skips the wide DP on provably-narrow pairs with bit-for-bit-identical output.
    /* --compat: the selected output-compatibility target. Non-NULL on any
     * mem_opt_t from mem_opt_init(), which sets it to &COMPAT_TARGET_OFF
     * (bwa-mem3's native output), so consumers can dereference it
     * unconditionally. (main_mem's `opt0` "was this set explicitly" sentinel is
     * memset to zero and is NOT such a struct; only its scalars are ever read.)
     * Points into the static table in compat_target.cpp -- not owned, never
     * freed, so the shallow struct copy in the MEM_F_SMARTPE path is correct.
     * Shapes OUTPUT ONLY: no alignment, score, flag, or tag VALUE depends on it. */
    const compat_target_t *compat;
} mem_opt_t;


struct mem_alnreg_t;
// * Chaining *
typedef struct abc {
    abc() {
        rbeg = qbeg = len = score = aln = 0;
        n_hits = 1;
    }
    int64_t rbeg;
    int32_t qbeg;
    int32_t len;
    int32_t score;
    int aln;
    int32_t n_hits;  // SMEM SA occurrence count this seed came from; 1 = unique
} mem_seed_t; // unaligned memory

/* Width of mem_chain_t::w, and the largest value that field can hold.
 *
 * These exist so the bitfield and mem_chain_weight()'s saturating clamp cannot
 * drift apart. They used to disagree: the clamp saturated at (1<<30)-1 while
 * the field was 27 bits, so a weight in [2^27, 2^30) wrapped modulo 2^27 on
 * store and a very heavy chain could come back with a tiny w -- dropped by the
 * `c->w < opt->min_chain_weight` gate, or losing a `drop_ratio` shadowing
 * comparison it should have won (fg-labs/bwa-mem3#309).
 *
 * Unreachable on real data, and the fix is byte-identical because of it: chain
 * weight accumulates non-overlapping seed spans and takes min(query, ref)
 * coverage, so it is bounded by the query length and saturating needs a single
 * ~134 Mbp read. The point is that the two constants now have one definition,
 * not that the old arithmetic ever fired. */
#define MEM_CHAIN_W_BITS 27
#define MEM_CHAIN_W_MAX  ((1u << MEM_CHAIN_W_BITS) - 1u)

typedef struct {
    int32_t seqid, cseed;
    int32_t n, m, first, rid;
    /* meth_hypothesis lives in this bitfield word — NOT as a trailing byte — so
     * sizeof(mem_chain_t) is unchanged from upstream bwa-mem2 (48 B). This struct
     * is the klib kbtree key type (KBTREE_INIT(chn, mem_chain_t, chain_cmp) in
     * bwamem.cpp), and kbtree derives its B-tree node fan-out from sizeof(key_t)
     * (ext .../kbtree.h: b->t = ((size-4-sizeof(void*))/(sizeof(void*)+sizeof(key_t))+1)>>1).
     * Growing the struct changes the tree shape, which changes WHICH chain kb_getp
     * returns as the merge neighbour when two chains tie on chain_cmp (== .pos),
     * which regroups seeds into different chains and silently moves default output.
     * Adding it as a trailing `int8_t` (padded to +8 B) did exactly that — it moved
     * non-meth output away from bwa-mem2 (see the static_assert in bwamem.cpp).
     * Values: 1 = OT (C→T, odd "f" seed contig), 0 = OB (G→A, even "r" seed
     * contig), -1 = not a meth chain (non-meth runs leave this at -1). A signed
     * 2-bit field represents {-2,-1,0,1}, covering all three values.
     *
     * For directional libraries (the --meth contract) each read is projected under
     * a SINGLE hypothesis (R1→OT, R2→OB), so all of a read's seeds carry the same
     * label and no cross-hypothesis merge can occur; test_and_merge is therefore
     * NOT hypothesis-guarded. NOTE (--meth directional invariant): if non-
     * directional / dual-hypothesis-per-read support is ever added, test_and_merge
     * MUST gain a hypothesis guard (it is currently safe only because each read is
     * single-hypothesis). */
    uint32_t w:MEM_CHAIN_W_BITS, kept:2, is_alt:1;   /* unsigned: w/kept/is_alt are 0..N flags (kept reaches 3) */
    int32_t  meth_hypothesis:2;        /* signed: needs -1; packs into the same 4-byte word (static_assert below) */
    float frac_rep;
    int64_t pos;
    mem_seed_t *seeds;
} mem_chain_t;

typedef struct { size_t n, m, cc; mem_chain_t *a; int capped_w; } mem_chain_v;

typedef struct mem_alnreg_t {
    // mem_alnreg_t() {c=NULL;}
    int64_t rb, re; // [rb,re): reference sequence in the alignment
    int qb, qe;     // [qb,qe): query sequence in the alignment
    int rid;        // reference seq ID
    mem_chain_t *c;
    int score;      // best local SW score
    int truesc;     // actual score corresponding to the aligned region; possibly smaller than $score
    int sub;        // 2nd best SW score
    int alt_sc;
    int csub;       // SW score of a tandem hit
    int sub_n;      // approximate number of suboptimal hits
    int w;          // actual band width used in extension
    int seedcov;    // length of regions coverged by seeds
    int secondary;  // index of the parent hit shadowing the current hit; <0 if primary
    int secondary_all;
    int seedlen0;   // length of the starting seed
    int chain_n_hits; // max SMEM SA-occurrence count across this chain's seeds (1 = no repetitive seed)
    int n_comp:30, is_alt:2; // number of sub-alignments chained together
    float frac_rep;
    uint64_t hash;
    int flg;
    /* D3 (--meth, PR-3): bisulfite hypothesis (1=OT, 0=OB, -1=non-meth),
     * propagated from the originating chain so PR-4's asymmetric extension and
     * the output (XG strand) layers can select the right matrix/strand. */
    int8_t meth_hypothesis;
    /* D3 (--meth, fix): STRAND-ADJUSTED extension hypothesis = meth_hypothesis
     * XOR is_rev. bwa-mem extends reverse-strand seeds against the reverse-
     * complemented reference window, so a conversion that the OT matrix frees
     * (ref-C x read-T) presents as the OB-freed cell (ref-G x read-A) there, and
     * vice-versa. Extension/regen MUST select the matrix from THIS field, not the
     * raw hypothesis, or reverse-strand reads (the bulk of PE) get the wrong
     * matrix, their real conversions are scored as mismatches, and the 5' end
     * soft-clips. Set at alnreg creation from the seed strand; -1 = non-meth.
     * (Output XG/XM still use the raw meth_hypothesis — the genome strand.) */
    int8_t meth_strand_hyp;
} mem_alnreg_t;

typedef struct { size_t n, m; mem_alnreg_t *a; int capped_w; } mem_alnreg_v;
/* capped_w: max WEIGHT among chains the cap/gate dropped before extension, carried
 * to worker_sam so --extend-csub can seed the primary region's competitor score
 * (a calibrated estimate, clamped below the primary's own score) and keep MAPQ from
 * inflating when a real competitor was pruned. 0 = nothing dropped / feature off. */

typedef struct {
    int low, high;   // lower and upper bounds within which a read pair is considered to be properly paired
    int failed;      // non-zero if the orientation is not supported by sufficient data
    double avg, std; // mean and stddev of the insert size distribution
} mem_pestat_t;

typedef struct { // This struct is only used for the convenience of API.
    int64_t pos;     // forward strand 5'-end mapping position
    int rid;         // reference sequence index in bntseq_t; <0 for unmapped
    int flag;        // extra flag
    uint32_t is_rev:1, is_alt:1, mapq:8, NM:22; // is_rev: whether on the reverse strand; mapq: mapping quality; NM: edit distance
    int n_cigar;     // number of CIGAR operations
    uint32_t *cigar; // CIGAR in the BAM encoding: opLen<<4|op; op to integer mapping: MIDSH=>01234
    char *XA;        // alternative mappings
    int HN;          // total # of hits clustered with this primary under XA_drop_ratio; -1 when not computed (e.g., MEM_F_ALL)

    int score, sub, alt_sc;
    /* D3 (--meth, PR-5): bisulfite hypothesis (1=OT, 0=OB, -1=non-meth),
     * copied from the source mem_alnreg_t in mem_reg2aln. The output layer
     * (meth_mem_aln_to_bam) sources the XG strand tag and the XM source
     * strand from THIS, not from the (now retired) f/r contig direction. */
    int8_t meth_hypothesis;
} mem_aln_t;

// struct
typedef struct {
    bwtintv_v mem, mem1, *tmpv[2];
} smem_aux_t;

typedef struct
{
    SeqPair *seqPairArrayAux[MAX_THREADS];
    SeqPair *seqPairArrayLeft128[MAX_THREADS];
    SeqPair *seqPairArrayRight128[MAX_THREADS];
    
    int64_t wsize[MAX_THREADS];

    int64_t wsize_buf_ref[MAX_THREADS*CACHE_LINE]; 
    int64_t wsize_buf_qer[MAX_THREADS*CACHE_LINE];

    uint8_t *seqBufLeftRef[MAX_THREADS*CACHE_LINE];
    uint8_t *seqBufRightRef[MAX_THREADS*CACHE_LINE];
    uint8_t *seqBufLeftQer[MAX_THREADS*CACHE_LINE];
    uint8_t *seqBufRightQer[MAX_THREADS*CACHE_LINE];    

    SMEM *matchArray[MAX_THREADS];
    int32_t *min_intv_ar[MAX_THREADS];
    int32_t *rid[MAX_THREADS];
    int32_t *lim[MAX_THREADS];
    int32_t *query_pos_ar[MAX_THREADS];
    uint8_t *enc_qdb[MAX_THREADS];
    
    int64_t wsize_mem[MAX_THREADS];
    int64_t wsize_mem_s[MAX_THREADS];
    int64_t wsize_mem_r[MAX_THREADS];
    // enc_qdb byte-size tracker. Separate from wsize_mem because
    // mem_collect_smem repurposes wsize_mem as an SMEM-entry count, which
    // breaks the byte-count invariant for enc_qdb on short-read workloads
    // (≤50 bp aDNA), where SMEM count >> tot_len bytes can leave enc_qdb
    // silently undersized between batches.
    int64_t wsize_qdb[MAX_THREADS];

    // Lockstep SMEM batching per-slot state. One pair of contiguous SMEM
    // buffers per thread, each of size SMEM_LOCKSTEP_N * lockstep_buf_cap[tid].
    // Per-slot views are stride-offset into these at lockstep driver entry.
    // Grown on demand when the batch's max_readlength exceeds the per-slot
    // capacity.
    SMEM    *lockstep_prev[MAX_THREADS];
    SMEM    *lockstep_match_buf[MAX_THREADS];
    int64_t  lockstep_buf_cap[MAX_THREADS];

    // Reusable per-thread scratch for FMI_search::sortSMEMs' rid counting sort
    // (audit SEED-15). Owned here so the count/offset and stable-scatter buffers
    // are allocated once and grown on demand instead of malloc/free'd per batch.
    // Each worker thread touches only its own [tid] slot, so the reuse is safe
    // by construction even though the FMI_search instance itself is shared.
    SmemSortScratch smem_sort_scratch[MAX_THREADS];

    // Pointer into worker_t::ref_string (the unpacked .0123 reference).
    // Set once in the worker_aln/worker_sam entry points; lets helpers like
    // mem_seed_sw and the mem_matesw_* family invoke bns_fetch_seq_v2 without
    // threading ref_string through 8 function signatures. Read-only and
    // shared across threads — every thread sees the same pointer.
    uint8_t *ref_string;
} mem_cache;

// chain moved to .h
typedef struct worker_t {
    const mem_opt_t      *opt;
    //const bntseq_t         *bns;
    // const uint8_t         *pac;
    const mem_pestat_t   *pes;
    smem_aux_t      **aux;
    bseq1_t          *seqs;
    mem_alnreg_v     *regs;
    int64_t           n_processed;
    /* Per-THREAD chaining scratch: nthreads * BATCH_SIZE entries, indexed by
     * tid (see worker_alloc). Renamed from chain_ar/seedBuf when they stopped
     * being nreads-sized seq_id-indexed arrays, so that any out-of-tree code
     * still doing `w.chain_ar + seq_id` fails to compile rather than silently
     * running off the end of a much smaller allocation. */
    mem_chain_v      *chain_scratch;
    mem_cache         mmc;
    mem_seed_t       *seed_scratch;
    int64_t           seed_scratch_size;   /* one thread's window, in seeds */
    mem_seed_t       *auxSeedBuf;
    int64_t           auxSeedBufSize;
    uint8_t          *ref_string;
    int16_t           nthreads;
    int32_t           nreads;
    FMI_search       *fmi;
    /* D3 (--meth, PR-3) coordinate cutover. The seed FM-index in `fmi` lives in
     * f/r-doubled SEED coordinates and is used ONLY for candidate generation;
     * every downstream consumer (chaining merge tests, extension ref fetch,
     * pairing/insert-size, mate rescue, output) must run in ORIGINAL-reference
     * coordinates. These handles are the original (un-converted) bns/pac and the
     * original unpacked `.0123` ref_string; seed hits are remapped into the
     * original-doubled-pac space in mem_chain_seeds before chaining, after which
     * these (NOT fmi->idx->bns/pac) are the bns/pac/ref_string the rest of the
     * pipeline uses. NULL outside --meth. */
    const bntseq_t   *meth_orig_bns;
    const uint8_t    *meth_orig_pac;
    uint8_t          *meth_orig_ref_string;
    /* [dedup-reads] per-chunk memoization state (Phase 2 consumer). NULL = no
     * memoization; Phase 1 always leaves it NULL (measure-only). */
    const struct read_memo_state *memo;
} worker_t;

/* D3 (--meth, PR-3) helpers. In --meth, this returns the ORIGINAL bns/pac/
 * ref_string the chaining/extension/pairing/output layers must use; outside
 * --meth it returns the single seed/normal index handles. Centralizes the
 * "which coordinate space" decision so every consumer is converted in one place
 * (the coordinate-cutover audit, B4). */
static inline const bntseq_t *mem_aln_bns(const worker_t *w) {
    return (w->opt->meth_mode && w->meth_orig_bns != NULL)
           ? w->meth_orig_bns : w->fmi->idx->bns;
}
static inline const uint8_t *mem_aln_pac(const worker_t *w) {
    return (w->opt->meth_mode && w->meth_orig_pac != NULL)
           ? w->meth_orig_pac : w->fmi->idx->pac;
}
static inline uint8_t *mem_aln_ref_string(const worker_t *w) {
    return (w->opt->meth_mode && w->meth_orig_ref_string != NULL)
           ? w->meth_orig_ref_string : w->ref_string;
}

/* D3 (--meth, PR-4): select the substitution matrix for an extension by the
 * per-chain/per-alnreg OT/OB hypothesis. Outside --meth (or hypothesis < 0,
 * i.e. a non-meth chain) this returns the symmetric `opt->mat` unchanged so the
 * non-meth path is byte-for-byte identical. hypothesis: 1 = OT (mat_ot, frees
 * C→T), 0 = OB (mat_ob, frees G→A). See mem_opt_t::mat_ot/mat_ob. */
static inline const int8_t *mem_opt_meth_mat(const mem_opt_t *opt, int meth_hypothesis) {
    if (!opt->meth_mode || meth_hypothesis < 0) return opt->mat;
    return (meth_hypothesis & 1) ? opt->mat_ot : opt->mat_ob;
}

/* True when mem_aln2sam() builds records as bam1_t (pushed to bseq1_t.bams) and
 * leaves the SAM kstring_t untouched, rather than formatting SAM text into it.
 *
 * This is NOT the same question as "is the output file BAM". --meth always
 * builds bam1_t — the bisulfite overlay (XM:Z, XG:Z, chimera QC) is written into
 * the aux block, and htslib serializes that to either container — so a --meth run
 * WITHOUT --bam still takes the bam1_t path and then writes SAM text. Callers
 * that skip the `str.s` handoff must test THIS, not opt->bam_mode; testing
 * bam_mode alone would send `--meth` (no `--bam`) down the text branch and trip
 * its `assert(str.s != 0)` on a string nothing ever wrote to. */
static inline int mem_opt_records_are_bam(const mem_opt_t *opt) {
    return opt->bam_mode || opt->meth_mode;
}


typedef kvec_t(int) int_v;

smem_i *smem_itr_init(const bwt_t *bwt);
void smem_itr_destroy(smem_i *itr);
void smem_set_query(smem_i *itr, int len, const uint8_t *query);
void smem_config(smem_i *itr, int min_intv, int max_len, uint64_t max_intv);
const bwtintv_v *smem_next(smem_i *itr);

mem_opt_t *mem_opt_init(void);
void mem_fill_scmat(int a, int b, int8_t mat[25]);
/* True iff the certified adaptive band (opt->band_cert) is safe to apply under the
 * current scoring/gap/zdrop parameters. The certificate bounds the optimal score but
 * not the extension kernel's early-termination heuristics; outside a conservative
 * parameter envelope (small zdrop, large clip penalties, a matrix scoring above a)
 * the caller must fall back to the exact full-width ladder. No-op at default
 * parameters. Defined in bwamem.cpp. */
int mem_band_cert_params_safe(const mem_opt_t *opt);
/* (Re)derive the --meth per-hypothesis matrices (mat_ot/mat_ob) from opt->mat +
 * opt->a. Call after any rebuild of opt->mat (e.g. CLI -A/-B/-x parsing) so meth
 * scoring tracks the user's options instead of the init-time defaults. */
void mem_opt_fill_meth_mat(mem_opt_t *opt);

/* Apply the --meth scoring defaults to `opt`, honouring the user's explicit
 * settings recorded as sentinels in `opt0` (non-zero field == user supplied).
 *
 * The constants are bwameth's (`bwa-mem2 mem -T 40 -B 2 -L 10 -CM`, plus
 * -U 100 for paired), and bwameth runs bwa at its default match score a == 1.
 * They are therefore SCALED BY opt->a here: every one of these options is
 * expressed in units of the match score, and bwa's update_a() scales the
 * non-meth defaults the same way. Applying them flat would silently discard
 * -A -- `--meth -A 2` would leave T at 40 while the alignment scores it gates
 * had doubled. At the default a == 1 the scaling is a no-op, so this is
 * byte-identical to the historical behaviour for every run that does not pass
 * -A.
 *
 * Must be called AFTER -A/-B/-T/-L/-U parsing and after update_a(). */
void mem_opt_apply_meth_defaults(mem_opt_t *opt, const mem_opt_t *opt0);

/* Configure extension-DP job dedup. mode_arg is the --dedup CLI value (NULL =>
 * fall back to env BWAMEM3_DEDUP, then default 'auto'). The controller knobs z
 * and reprobe are expert-only, resolved from env (BWAMEM3_DEDUP_Z default 2.0,
 * BWAMEM3_DEDUP_REPROBE default 12000000) -- no CLI flag, matching the house
 * style for BWAMEM3_* tuning knobs. Must be called before the first alignment
 * batch; invalid values are fatal. CLI(mode) > env > default. */
void mem_dedup_configure(const char *mode_arg);

// Skip-short-seed extension filter: drop seeds shorter than min_ext_len from a
// chain in place (stable; surviving seeds keep their order). Returns the new
// seed count. min_ext_len <= 0 is a no-op. See mem_opt_t::min_ext_len.
int mem_chain_drop_short_seeds(mem_chain_t *c, int min_ext_len);

// Chain weight: the smaller of the chain's query-axis and reference-axis
// coverage, counting each base once however many seeds span it. Saturates at
// MEM_CHAIN_W_MAX, the largest value mem_chain_t::w can hold (#309). Declared
// here (it was file-local to bwamem.cpp) so the clamp is unit-testable; the
// only callers in the aligner remain the two stores in mem_chain_flt.
int mem_chain_weight(const mem_chain_t *c);

void mem_reg2sam(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                 bseq1_t *s, mem_alnreg_v *a, int extra_flag, const mem_aln_t *m);

int mem_approx_mapq_se(const mem_opt_t *opt, const mem_alnreg_t *a) ;

int mem_mark_primary_se(const mem_opt_t *opt, int n, mem_alnreg_t *a, int64_t id);
void mem_seed_capped_sub(mem_alnreg_v *a, int match_a);  /* --extend-csub: seed primary sub from dropped chains */
/* Exposed for unit tests (test/unit/test_chain_cap.cpp). */
int  mem_chain_cap_extend(mem_chain_t *a, int n, int max_n, float tie_frac, int tie_floor,
                          int *capped_w_out);   /* --max-extend-chains count cap + --extend-tie-frac gate */
int  mem_capped_pair_subo(const mem_alnreg_v a[2], const int z[2], int o);  /* --extend-csub PE competitor floor */

static void mem_mark_primary_se_core(const mem_opt_t *opt, int n, mem_alnreg_t *a, int_v *z);

/* ONLY work after mem_mark_primary_se(); out_hn may be NULL.
 * `meth_orig_query` is forwarded to mem_reg2aln for each XA sub-entry so they
 * regenerate under the same NM/MD policy as the primary record (see
 * mem_reg2aln). NULL (the default) keeps the legacy literal regen. */
char **mem_gen_alt(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                   const mem_alnreg_v *a, int l_query, const char *query,
                   int **out_hn, const char *meth_orig_query = NULL);
void mem_aln2sam(const mem_opt_t *opt, const bntseq_t *bns, kstring_t *str, bseq1_t *s,
                 int n, const mem_aln_t *list, int which, const mem_aln_t *m_);

static inline int get_rlen(int n_cigar, const uint32_t *cigar);
static inline int infer_bw(int l1, int l2, int score, int a, int q, int r);

int mem_kernel1_core(FMI_search *fmi, const mem_opt_t *opt,
                     bseq1_t *seq_,
                     int nseq,
                     mem_chain_v *chain_ar,
                     mem_seed_t *seedBuf,
                     int64_t seedBufSize,
                     mem_cache *mmc,
                     int tid,
                     /* D3 (--meth, PR-3): ORIGINAL bns/pac remap target +
                      * filtering ref. NULL outside --meth → legacy behavior. */
                     const bntseq_t *meth_orig_bns = NULL,
                     const uint8_t  *meth_orig_pac = NULL);

int mem_kernel2_core(FMI_search *fmi, const mem_opt_t *opt,
                     bseq1_t *seq_, mem_alnreg_v *regs, int nseq,
                     mem_chain_v *chain_ar, mem_cache *mmc,
                     uint8_t *ref_string, int tid,
                     /* D3 (--meth, PR-3): ORIGINAL bns/pac for extension/dedup.
                      * NULL outside --meth → legacy behavior. ref_string above
                      * is already the original .0123 in --meth (see worker_aln). */
                     const bntseq_t *meth_orig_bns = NULL,
                     const uint8_t  *meth_orig_pac = NULL);

void* _mm_realloc(void *ptr, int64_t csize, int64_t nsize, int16_t dsize);

/* Sentinel returned by seqbuf_grow_capacity() when the doubled capacity could
 * no longer be addressed by an int32 offset. */
#define SEQBUF_CAPACITY_OVERFLOW ((int64_t)-1)

/* Double a seqBuf* capacity, refusing to exceed the int32 offset range.
 *
 * Offsets into the extension buffers reach the SW kernels via SeqPair.idr/.idq
 * (src/bandedSWA.h), which are int32_t, while the accumulators that feed them
 * are int64_t. A capacity above INT32_MAX therefore yields offsets that narrow
 * to a negative index on assignment, and the kernel then indexes outside the
 * allocation. Long-read blocks reach this in ~5 doublings because the buffers
 * are sized on a short-read model (MAX_SEQ_LEN_REF bytes of reference span per
 * seed), so returning the sentinel here is what stops silent corruption.
 *
 * Returns the doubled capacity, or SEQBUF_CAPACITY_OVERFLOW if it would exceed
 * what an int32 offset can address. */
static inline int64_t seqbuf_grow_capacity(int64_t current) {
    if (current > ((int64_t)INT32_MAX) / 2) return SEQBUF_CAPACITY_OVERFLOW;
    return current * 2;
}

/* Report an unrepresentable seqBuf* growth and abort.
 *
 * Bounding memory properly requires flushing the accumulated SW batch when the
 * buffers fill rather than growing without limit; until that exists, failing
 * loudly with an actionable message beats corrupting the heap. */
void seqbuf_capacity_fatal(const char *buf_name, const char *func,
                           int64_t current);

void mem_chain2aln_across_reads_V2(const mem_opt_t *opt, const bntseq_t *bns,
                                   const uint8_t *pac, bseq1_t *seq_, int nseq,
                                   mem_chain_v* chain_ar, mem_alnreg_v *av_v,
                                   mem_cache *mmc, uint8_t *ref_string, int tid);

int mem_sam_pe_batch_pre(const mem_opt_t *opt, const bntseq_t *bns,
                         const uint8_t *pac, const mem_pestat_t pes[4],
                         uint64_t id, bseq1_t s[2], mem_alnreg_v a[2],
                         mem_cache *mmc,  int64_t &pcnt, int32_t &gcnt,
                         int32_t&, int32_t&, int tid);

int mem_matesw_batch_pre(const mem_opt_t *opt, const bntseq_t *bns,
                         const uint8_t *pac, const mem_pestat_t pes[4],
                         const mem_alnreg_t *a, int l_ms, const uint8_t *ms,
                         const char *ms_orig,
                         mem_alnreg_v *ma, mem_cache *mmc, int pcnt, int32_t gcnt,
                         int32_t &maxRefLen, int32_t &maxQerLen, int32_t tid,
                         int mate_meth_ot = -1);

/* Given two alignment begin positions (rb) on the 2-bit-packed concat-with-
 * reverse-complement index, infer the pair orientation (0=FF, 1=FR, 2=RF,
 * 3=RR) and the distance between 5' ends. Exposed so that external consumers
 * can share the same proper-pair classification used by mem_sam_pe's
 * no_pairing fallback. */
int mem_infer_dir(int64_t l_pac, int64_t b1, int64_t b2, int64_t *dist);

/* Proper-pair bit (FLAG 0x2) for a pair emitted on mem_sam_pe's no-pairing
 * path. Returns 2 when properly paired and 0 otherwise, so the result ORs
 * straight into extra_flag. `which[i]` is the index of the region mate i
 * actually emits; both entries must be >= 0 and the two EMITTED regions
 * (a[i].a[which[i]]) must already be known to share a reference sequence. That
 * is all the caller checks, so on the default path -- which reads a[i].a[0] --
 * the two coordinates may sit on different reference sequences and yield a
 * cross-sequence distance. bwa and bwa-mem2 compute it the same way.
 *
 * The single definition of a decision the two no-pairing blocks make
 * identically: derive 0x2 from the top-scoring region a[0] (bwa and bwa-mem2,
 * the default) or from the emitted a[which] (#17, opt-in via
 * --proper-pair-from-emitted). Inert without a `.alt` sidecar, since the two
 * indices coincide unless the read has ALT hits. */
int mem_proper_pair_extra_flag(const mem_opt_t *opt, int64_t l_pac,
                               const mem_alnreg_v a[2], const int which[2],
                               const mem_pestat_t pes[4]);

int mem_sam_pe_batch(const mem_opt_t *opt, mem_cache *mmc,
                     int64_t &pcnt, int64_t &pcnt8, kswr_t *aln,
                     int32_t, int32_t, int tid);

int mem_sam_pe_batch_post(const mem_opt_t *opt, const bntseq_t *bns,
                          const uint8_t *pac, const mem_pestat_t pes[4],
                          uint64_t id, bseq1_t s[2], mem_alnreg_v a[2],
                          kswr_t **myaln, mem_cache *mmc,
                          int32_t &gcnt, int tid);

int mem_matesw_batch_post(const mem_opt_t *opt, const bntseq_t *bns,
                          const uint8_t *pac, const mem_pestat_t pes[4],
                          const mem_alnreg_t *a, int l_ms, const uint8_t *ms,
                          mem_alnreg_v *ma, kswr_t **myaln, int32_t gcnt,
                          int32_t *gar, mem_cache *mmc, const char *ms_orig = NULL,
                          const int8_t *mat = NULL, int mate_meth_ot = -1);

int mem_sam_pe(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
               const mem_pestat_t pes[4], uint64_t id, bseq1_t s[2],
               mem_alnreg_v a[2]);

// Core pairing decision for a single read pair. Runs mate-rescue SW,
// mem_mark_primary_se, optional MEM_F_PRIMARY5 reorder, mem_pair, is-multi
// sanity check, q_pe/q_se computation, and the secondary<->primary
// secondary_all patch. Does NOT emit SAM/BAM — callers do that.
//
// Output contract:
//   * paired_out: set to 1 iff the paired branch was taken; 0 otherwise.
//     Callers must always inspect *paired_out before reading z[] / q_se[].
//   * z[], q_se[]: valid only when *paired_out == 1. On the no-pairing path
//     they must be treated as undefined — mem_pair() can populate z[]
//     before the is_multi early return, so their contents do not signal
//     anything when *paired_out == 0.
//   * extra_flag_out: on the paired path it is fully assembled (and
//     includes 0x2 iff the paired alignment was preferred); on the
//     no-pairing path it is only partial — the caller (or mem_sam_pe's
//     no_pairing emission block) is responsible for OR-ing in 0x2 itself.
//   * n_pri[]: always populated.
//
// Returns the number of mate-rescue hits produced (same meaning as the
// historical `n` return of mem_sam_pe).
int mem_pair_resolve(const mem_opt_t *opt, const bntseq_t *bns,
                     const uint8_t *pac, const mem_pestat_t pes[4],
                     uint64_t id, bseq1_t s[2], mem_alnreg_v a[2],
                     int n_pri[2], int z[2], int q_se[2],
                     int *extra_flag_out, int *paired_out);
/**
 * Align a batch of sequences and generate the alignments in the SAM format
 *
 * This routine requires $seqs[i].{l_seq,seq,name} and write $seqs[i].sam.
 * Note that $seqs[i].sam may consist of several SAM lines if the
 * corresponding sequence has multiple primary hits.
 *
 * In the paired-end mode (i.e. MEM_F_PE is set in $opt->flag), query
 * sequences must be interleaved: $n must be an even number and the 2i-th
 * sequence and the (2i+1)-th sequence constitute a read pair. In this
 * mode, there should be enough (typically >50) unique pairs for the
 * routine to infer the orientation and insert size.
 *
 * @param opt    alignment parameters
 * @param bwt    FM-index of the reference sequence
 * @param bns    Information of the reference
 * @param pac    2-bit encoded reference
 * @param n      number of query sequences
 * @param seqs   query sequences; $seqs[i].seq/sam to be modified after the call
 * @param pes0   insert-size info; if NULL, infer from data; if not NULL, it should be an array with 4 elements,
 *               corresponding to each FF, FR, RF and RR orientation. See mem_pestat() for more info.
 */
void mem_process_seqs(mem_opt_t *opt, int64_t n_processed,
                      int n, bseq1_t *seqs, const mem_pestat_t *pes0,
                      worker_t &w);

/**
 * Align one slice of a pestat cohort: seeding + banded SW only, no pairing.
 *
 * mem_process_seqs() is the single-slice special case of
 * mem_align_cohort_slice() + mem_pair_and_emit_cohort(). Splitting them lets the
 * pipeline read and align a cohort in several smaller pieces -- so the first
 * read of a run does not stall the whole compute pipeline -- while mem_pestat
 * still sees exactly the read set it would have seen for the un-sliced batch.
 * That is what keeps the output byte-identical: seeding and BSW are per-read
 * independent, and read ids come from the global n_processed counter rather
 * than from a position within the batch.
 *
 * @param n_processed global id of this slice's FIRST read (cohort base + offset)
 * @param n           reads in this slice; must be even for paired input so a
 *                    slice boundary never splits a pair
 * @param seqs        this slice's reads (cohort array + offset)
 * @param regs        this slice's alnreg output (cohort regs + the same offset);
 *                    must remain live until mem_pair_and_emit_cohort() runs
 *
 * w.seqs / w.regs / w.n_processed are saved and restored around the call.
 */
void mem_align_cohort_slice(mem_opt_t *opt, int64_t n_processed,
                            int n, bseq1_t *seqs, mem_alnreg_v *regs,
                            worker_t &w);

/**
 * Infer the insert size over a complete cohort, then pair and emit it.
 *
 * Must be called with `w.regs` still pointing at the cohort's base and with
 * every slice's alnregs intact: mem_pestat() reads regs[0..n) to build the
 * insert-size distribution, and worker_sam then consumes and frees them.
 *
 * @param n_processed global id of the cohort's first read
 * @param n           reads in the whole cohort
 * @param seqs        the cohort's contiguous reads
 * @param pes0        insert-size info from -I, or NULL to infer from data
 */
void mem_pair_and_emit_cohort(mem_opt_t *opt, int64_t n_processed,
                              int n, bseq1_t *seqs, const mem_pestat_t *pes0,
                              worker_t &w);


/**
 * Generate CIGAR and forward-strand position from alignment region
 *
 * @param opt    alignment parameters
 * @param bns    Information of the reference
 * @param pac    2-bit encoded reference
 * @param l_seq  length of query sequence
 * @param seq    query sequence
 * @param ar     one alignment region
 *
 * @return       CIGAR, strand, mapping quality and forward-strand position
 */
/* D3 (--meth, PR-5): `meth_orig_query` is the read's ORIGINAL (un-projected)
 * bases (bseq1_t.meth_orig_seq), same orientation/order as `seq`. When non-NULL
 * under --meth, the final CIGAR/NM/MD regen runs the ORIGINAL read against the
 * ORIGINAL reference with the per-hypothesis asymmetric matrix
 * (mem_opt_meth_mat(opt, ar->meth_hypothesis)) so output is native-alphabet.
 * NULL (the default) preserves the legacy symmetric `seq`-vs-ref regen exactly
 * for the non-meth path. */
mem_aln_t mem_reg2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                      int l_seq, const char *seq, const mem_alnreg_t *ar,
                      const char *meth_orig_query = NULL);


/**
 * Infer the insert size distribution from interleaved alignment regions
 *
 * This function can be called after mem_align1(), as long as paired-end
 * reads are properly interleaved.
 *
 * @param opt    alignment parameters
 * @param l_pac  length of concatenated reference sequence
 * @param n      number of query sequences; must be an even number
 * @param regs   region array of size $n; 2i-th and (2i+1)-th elements constitute a pair
 * @param pes    inferred insert size distribution (output)
 */
void mem_pestat(const mem_opt_t *opt, int64_t l_pac, int n, const mem_alnreg_v *regs,
                mem_pestat_t pes[4]);

void mem_reorder_primary5(int T, mem_alnreg_v *a);

#endif
