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

#include "bwamem.h"
#include "read_memo.h"     /* --dedup-reads whole-read-pair memoization (Phase 1: measure-only) */
#include "meth_xm.h"   /* meth_chem_t for the --meth chemistry default */
#include "FMI_search.h"
#include "smem_dedup.h"
#include "bam_writer.h"
#include "meth_bam.h"
#include "u8vec_scratch.h"
#include "kvec.h"          /* kvec_t/kv_push/kv_resize used directly below */
#include "grow_capacity.h" /* shared geometric-growth policy for scratch buffers */
#include "pdqsort_wrap.h"
#include "robin_hood.h"    /* per-batch DP-job dedup: missidx candidate map */
#include <inttypes.h>      /* PRId64 for int64_t fprintf format strings */
#include <errno.h>         /* errno/ERANGE: full-string parse of the dedup env knobs */
#include <strings.h>       /* strcasecmp in mem_opt_parse_meth_tags */
#include <atomic>          /* net-cycles auto-dedup controller state */
#include <chrono>          /* net-cycles auto-dedup controller timing */
#include <cmath>           /* std::isfinite: reject nan/inf BWAMEM3_DEDUP_Z */
#include <mutex>           /* BWAMEM3_DUMP_PAIRS append lock; net-cycles Welford state */
#include <string>          /* BWAMEM3_DUMP_PAIRS per-batch row buffer */

#include "sam_encode.h"
#include "stage_prof.h"
#include "seed_order.h"
/* Not including <htslib/sam.h>: its kstring.h shares the KSTRING_H guard
 * with bwa-mem3's. Opaque bam1_t wrappers live in bam_writer.h. */

/* RAII bracket for the SAM/BAM build (stage_prof --profile). Accumulates the
 * calling (compute) thread's CPU time across all of mem_aln2sam's return paths
 * into the per-thread encode accumulator; harvested by kt_for. Zero cost when off. */
namespace { struct SpEncodeScope {
    double t0; bool on;
    SpEncodeScope() : t0(0.0), on(sp_enabled()) { if (on) t0 = sp_thread_cpu(); }
    ~SpEncodeScope() { if (on) sp_encode_add(sp_thread_cpu() - t0); }
}; }

/* D3 (--meth, PR-5): original-reference pac for XM:Z, set in fastmap.cpp.
 * The meth output path is gated on opt->meth_mode (the chrom map that used to
 * gate it was retired with the f/r output layer). */
const uint8_t *g_meth_orig_pac = NULL;

//----------------
extern uint64_t tprof[LIM_R][LIM_C];
//----------------
#include "kbtree.h"

#define chain_cmp(a, b) (((b).pos < (a).pos) - ((a).pos < (b).pos))
KBTREE_INIT(chn, mem_chain_t, chain_cmp)

/* chain_cmp orders chains by .pos ONLY, so chains that share a .pos are a tie
 * whose resolution is left to the kbtree's internal structure. klib derives the
 * B-tree node fan-out from sizeof(mem_chain_t) (kbtree.h), so changing that size
 * changes which chain kb_getp returns at a tie, which regroups seeds and silently
 * moves default (non-meth) output. This guard fails the build if mem_chain_t ever
 * grows again — repack new fields into the existing bitfield word (see bwamem.h),
 * do not append. 48 B is upstream bwa-mem2's size; keeping it preserves bwa-mem2
 * chaining parity. */
static_assert(sizeof(mem_chain_t) == 48,
              "sizeof(mem_chain_t) feeds klib kbtree node fan-out; changing it moves default "
              "output at chain_cmp .pos ties. Repack fields into the bitfield word, don't append.");

/* RAII holder for the per-thread persistent chaining B-tree used in
 * mem_chain_seeds. The tree is init'd once per thread (not once per read) and
 * kb_reset between reads to remove per-read kb_init/kb_destroy malloc churn;
 * this destructor kb_destroy's it at thread exit so the thread_local stays
 * leak-clean under ASan, matching the house-style u8vec_scratch_t. */
namespace {
struct ChnTreeScratch {
    kbtree_t(chn) *t;
    ChnTreeScratch() : t(kb_init(chn, KB_DEFAULT_SIZE + 8)) {} // +8, counters in chain
    ~ChnTreeScratch() { if (t) kb_destroy(chn, t); }
    ChnTreeScratch(const ChnTreeScratch&) = delete;
    ChnTreeScratch& operator=(const ChnTreeScratch&) = delete;
};
} // namespace

/* #309: `w` shares its 32-bit word with kept:2, is_alt:1 and meth_hypothesis:2.
 * Narrowing `w` to make room for something else is the drift this catches --
 * mem_chain_weight() saturates at MEM_CHAIN_W_MAX, so a narrower field would
 * silently start wrapping again. Widening it past the word is caught by the
 * sizeof assert above; this one catches the other direction. */
static_assert(MEM_CHAIN_W_BITS + 2 + 1 + 2 == 32,
              "mem_chain_t's bitfield word is full: w + kept + is_alt + meth_hypothesis must be "
              "exactly 32 bits. If w is narrowed, MEM_CHAIN_W_MAX follows it and mem_chain_weight "
              "keeps saturating correctly -- but re-check that the new max still exceeds any "
              "reachable chain weight (bounded by query length).");

#define intv_lt1(a, b) ((((uint64_t)(a).m) <<32 | ((uint64_t)(a).n)) < (((uint64_t)(b).m) <<32 | ((uint64_t)(b).n)))  // trial
KSORT_INIT(mem_intv1, SMEM, intv_lt1)  // debug

#define max_(x, y) ((x)>(y)?(x):(y))
#define min_(x, y) ((x)>(y)?(y):(x))

#define MAX_BAND_TRY  4

/* cap initial band-width at this value. The retry loop
 * doubles w each iteration (up to MAX_BAND_TRY-1 iters), so pairs whose
 * alignment needs a wider band are caught by the retry. Setting this below
 * opt->w forces the kernel to start tight, accept tight-fit pairs early
 * (via sp->max_off heuristic + sp->tight_band), and only expand on demand.
 * BUCKET_MAX_INIT_W=8 with MAX_BAND_TRY=4 gives w-sequence 8, 16, 32, 64. */
#ifndef BUCKET_MAX_INIT_W
#define BUCKET_MAX_INIT_W 8
#endif

/* ------------------------------------------------------------------------
 * 8-bit (16-lane) SW safe-envelope gate.
 *
 * The recovered 8-bit kernel `smithWaterman128_8` (and the AVX2/AVX-512 ports)
 * runs an UNSIGNED [0,255] score recurrence and is byte-exact for long reads
 * ONLY inside the envelope below. The decisive constraint is that the per-lane
 * score re-baselining must NEVER fire for an admitted pair: its saturating
 * subtract is not generally lossless (it can zero a still-positive off-diagonal
 * cell, which the recurrence then misreads as the h00==0 local-restart sentinel
 * and truncates a valid alignment — z-drop is a row-level early-exit, not a
 * per-cell guarantee). Held inert, the kernel is a plain exact unsigned SW.
 *
 *   - len1, len2 < MAX_SEQ_LEN8  : slab width + the wide per-row position
 *     side channel are sized MAX_SEQ_LEN8; rows/cols at or beyond it would
 *     run off the slab.
 *   - len1 >= len2 (target >= query) : the gscore (query-end) capture is
 *     byte-identical to scalar only when the query-end column lies on/left of
 *     the main diagonal. See smithWaterman128_8 gscore capture.
 *   - w <= BSW8_MAX_W (= 124)     : the band/position quantities are encoded as
 *     signed-int8 diagonal offsets spanning [-(w+1), w+3] (the band-grow term
 *     `myband+1` and the tail-trim term `index+2`, which reaches w+3). The
 *     positive edge w+3 must fit signed int8, so w <= 124; at w >= 125 it wraps
 *     past +127, the band collapses and lanes die mid-alignment. The default
 *     opt->w = 100 qualifies; wide-band retries (w doubling to 200/400/800) do
 *     NOT and fall back to 16-bit.
 *   - zdrop + maxStep <= 253      : zdrop is broadcast into the DP as a byte
 *     (_mm256_set1_epi8), so it must fit one, and the maxStep headroom keeps the
 *     z-drop comparison from wrapping at the top of the range. The DP body AND
 *     the h0-prefix column/row seed (smithWaterman*_8 setup) are both unsigned-
 *     saturating [0,255], and the seed is the raw h0 clamped to uint8 (the old
 *     min(h0, zdrop+1) prefix clamp went away with the re-baseline floor).
 *     maxStep is the largest per-step score increment,
 *     max(w_match, w_ambig, 1) = max(opt->a, 1).
 *   - h0 + min(len1,len2)*a < 255 - maxStep : the MAX ATTAINABLE score (seed
 *     plus an all-match diagonal over the shorter sequence) stays below the byte
 *     ceiling, so no row max can ever reach it. This is also what bounds h0
 *     itself, which is why there is no separate h0 gate: the former
 *     h0 <= zdrop + 1 condition existed to zero the removed re-baseline floor
 *     and was dropped in EXT-4 (see bsw8_envelope_ok below).
 *
 * NOTE: minval (= h0 + min(len)*a) is still used by sortPairsLenExt as the
 * counting-sort bin index (hist[minval]) and must stay < MAX_SEQ_LEN8 there for
 * the bin to be in range; that is a histogram-sizing constraint, NOT the tier
 * decision. Pairs failing this envelope fall through to the existing 16-bit
 * (then scalar) buckets exactly as before. */
#define BSW8_MAX_W 124                 /* max band: the diagonal-offset encoding spans
                                          [-(w+1), w+3] (band-grow `myband+1`, tail-trim
                                          `index+2` reaching w+3), all int8, so w <= 124.
                                          At w>=125 the positive edge wraps past +127:
                                          the band collapses and lanes die mid-alignment. */
#define BSW8_MAX_ZDROP_STEP 253        /* zdrop is broadcast into the DP as a byte
                                          (_mm256_set1_epi8), so it must fit one, and the
                                          max_step headroom keeps the z-drop comparison from
                                          wrapping: zdrop+max_step <= 253. The DP body AND the
                                          h0-prefix column/row seed in the 8-bit wrappers are
                                          both unsigned-saturating [0,255], and the seed is the
                                          raw h0 clamped to uint8. NOTE: the 253 value dates from
                                          the re-baseline era, where it also kept
                                          255-max_step > zdrop+1; with re-baseline gone that
                                          second role is moot, so this bound is now conservative
                                          rather than tight. See smithWaterman128_8 h0-prefix seed. */

static inline int bsw8_envelope_ok(int len1, int len2, int w,
                                   int score_a, int zdrop, int h0)
{
    /* Dev A/B hook: BWAMEM3_DISABLE_BSW8=1 forces every pair off the 8-bit path
     * onto the 16-bit (then scalar) buckets. Used to validate that the 8-bit
     * kernel is byte-identical to the 16-bit reference in the full pipeline
     * (diff the SAM with vs without). Off by default. The function-local static
     * with a non-trivial initializer gets C++11 thread-safe one-time init, so the
     * getenv runs exactly once even under the OpenMP/pthread worker fan-out (a
     * plain `static int x = -1; if (x<0) x = ...;` would be a data race). */
    static const int disable = []() {
        const char *e = getenv("BWAMEM3_DISABLE_BSW8");
        return (e && atoi(e)) ? 1 : 0;
    }();
    if (disable) return 0;
    /* 64-bit math: len < MAX_SEQ_LEN8 and a small score_a keep these tiny in
     * practice, but compute the bounds in int64_t so a pathological scoring
     * parameter cannot wrap an int and admit a pair the guards would reject. */
    int64_t max_step = score_a > 1 ? (int64_t)score_a : 1;   /* max(w_match, w_ambig, 1) */
    int64_t shorter  = len1 < len2 ? (int64_t)len1 : (int64_t)len2;
    /* Max attainable SW score: seed h0 plus an all-match diagonal over the
     * shorter sequence. Must stay strictly below the byte ceiling 255 - max_step
     * so no row max can ever reach it; that alone makes the kernel an exact
     * unsigned [0,255] SW, and since h0 <= max_score it also bounds the seed
     * byte, so no separate h0 gate is needed. */
    int64_t max_score = (int64_t)h0 + shorter * (int64_t)score_a;
    return len1 < MAX_SEQ_LEN8 && len2 < MAX_SEQ_LEN8 &&
           /* target (rows, len1) >= query (cols, len2). The re-baseline 8-bit
            * kernel's gscore (query-end score) is byte-identical to scalar ONLY
            * when the query-end column (len2-1) lies on or left of the main
            * diagonal, i.e. len1 >= len2. When len2 > len1 the query end is
            * off-diagonal to the right, where the re-baseline saturating-subtract
            * zeroes those low-scoring cells (they sit >zdrop below the row max),
            * the band trims them, and gscore is lost. Such pairs (rare: only when
            * the reference window is shorter than the read segment, e.g. at contig
            * edges) route to the 16-bit tier, which has no re-baseline and is
            * byte-identical for them. See smithWaterman128_8 gscore capture. */
           len1 >= len2 &&
           w <= BSW8_MAX_W &&
           zdrop + max_step <= BSW8_MAX_ZDROP_STEP &&
           /* EXT-4: the former `h0 <= zdrop + 1` gate has been dropped. It was
            * introduced to force the per-lane re-baseline floor B0 = max(0, h0 -
            * (zdrop+1)) to zero; that re-baseline machinery was removed (the 8-bit
            * DP is now a plain unsigned [0,255] SW), so h0's only remaining
            * constraint is that the seed byte fit a uint8 — guaranteed by the
            * max_score bound below (h0 <= max_score < 255 - max_step). A second,
            * subtler role — bounding the seed against the z-drop horizon so a
            * high-h0 lane does not z-drop before its row max builds up — was fixed
            * in the kernel by #273 (correct 8-bit z-drop/seed clamp at high seed
            * scores). getScores8 is now byte-identical to scalarBandedSWA across
            * the newly-admitted h0 > zdrop+1 region for the default, moderate, and
            * harsh scoring sweeps (see test/bandedswa_high_h0_zdrop_test.cpp).
            * Admitting these pairs moves ~7.41% of Illumina extension pairs from
            * the 8-lane 16-bit tier to the 16-lane 8-bit tier. */
           max_score < 255 - max_step;
}

/* ---------------------------------------------------------------------------
 * BWAMEM3_DUMP_PAIRS=<path>: record the extension pairs handed to the banded-SW
 * batch, one TSV row per SeqPair: len1, len2, h0, w, tier (8 | 16 | 1).
 *
 * Why this exists: the shape of an extension problem is defined by seeding and
 * chaining, so it cannot be recovered from an alignment record after the fact —
 * a BAM says where a read landed, not which DP problems were posed to get it
 * there. Anything that needs to reproduce a representative extension workload
 * (benchmarks, kernel harnesses) has to observe the pairs here.
 *
 * The recorded tier is the routing decision this build actually made, which
 * lets an external mirror of bsw8_envelope_ok() be checked against ground truth
 * rather than against a re-reading of the rule.
 *
 * Off by default and inert when off: one getenv behind a function-local static
 * (C++11 thread-safe init, so the OpenMP/pthread fan-out cannot race it), and
 * the per-batch buffer is only touched when armed. Rows are buffered per batch
 * and appended under a lock — a batch is thousands of pairs of DP work, so the
 * contention is irrelevant next to what it is measuring.
 * ------------------------------------------------------------------------- */
static FILE *bsw_dump_fp(void)
{
    static FILE *fp = []() -> FILE * {
        const char *path = getenv("BWAMEM3_DUMP_PAIRS");
        if (!path || !*path) return NULL;
        FILE *f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "[bsw-dump] cannot open '%s' for writing; dump disabled\n", path);
            return NULL;
        }
        fprintf(f, "len1\tlen2\th0\tw\ttier\n");
        return f;
    }();
    return fp;
}

static inline bool bsw_dump_armed(void) { return bsw_dump_fp() != NULL; }

static inline void bsw_dump_pair(std::string &buf, const SeqPair &sp, int w, int tier)
{
    char line[64];
    int n = snprintf(line, sizeof line, "%d\t%d\t%d\t%d\t%d\n",
                     sp.len1, sp.len2, sp.h0, w, tier);
    if (n > 0) {
        /* snprintf returns the would-be length; cap to what actually landed in
         * `line` so append never over-reads the stack buffer on truncation. */
        if ((size_t)n >= sizeof line) n = (int)(sizeof line - 1);
        buf.append(line, (size_t)n);
    }
}

static void bsw_dump_flush(std::string &buf)
{
    if (buf.empty()) return;
    static std::mutex mtx;
    FILE *fp = bsw_dump_fp();
    if (!fp) return;   /* defensive: fp is non-null whenever dumping is armed */
    {
        std::lock_guard<std::mutex> lock(mtx);
        fwrite(buf.data(), 1, buf.size(), fp);
    }
    buf.clear();
}

            int tcnt = 0;
/********************
 * Filtering chains *
 ********************/

#define chn_beg(ch) ((ch).seeds->qbeg)
#define chn_end(ch) ((ch).seeds[(ch).n-1].qbeg + (ch).seeds[(ch).n-1].len)

#define flt_lt(a, b) ((a).w > (b).w)
KSORT_INIT(mem_flt, mem_chain_t, flt_lt)

/* Chain-geometry adaptive band (--adaptive-band). Start the banded-SW tight at
 * opt->band_start on the 16-bit and scalar (long-extension) tiers; carry each
 * pair's chain_band (= its chain's seed diagonal spread, capped at opt->w = the
 * implied inter-seed indel). ACCEPT_PAIR gates the score/max_off "converged"
 * accepts on w >= chain_band, so a pair whose chain implies a larger indel keeps
 * retrying to that band while a diagonal-hugging pair (chain_band=0) accepts in
 * one tight pass. The 8-bit tier stays at opt->w (its int8 diagonal encoding caps
 * at 127, and short extensions are sub-band so gain nothing). tight_band keeps its
 * ungapped-estimate accept-early role. band_start<=0 with band_cert off: INIT_W==opt->w and
 * ACCEPT_PAIR reduces to the original full-width condition -> byte-identical.
 *
 * NOTE: the DEFAULT is now the sound band_cert path below (band_cert=1), which narrows to
 * ADAPTIVE_BAND_START and is byte-identical via a per-pair certificate. This band_start path is
 * the separate opt-in AGGRESSIVE narrowing selected by --adaptive-band / --fast (band_cert=0),
 * which trades byte-identity for more speed via the max_off "converged" heuristic above. */
/* ------------------------------------------------------------------------
 * Sound (byte-identical) adaptive extension band  [opt->band_cert, default on]
 *
 * Runs one narrow probe at ADAPTIVE_BAND_START and finalizes only the pairs it can
 * PROVE are already optimal there; every other pair falls through to the standard
 * ceiling ladder and is scored exactly as the non-adaptive path scores it. So the
 * output is bit-for-bit identical to a full-width extension, while the proven-narrow
 * pairs skip the wide DP.
 *
 * Certificate: a gapped alignment reaching diagonal offset d costs >= o_min + d*e_min,
 * so its best possible score is h0 + min_len*a - o_min - d*e_min. That can tie-or-beat
 * the achieved score S only for d <= floor((h0 + min_len*a - o_min - S)/e_min) =: d_max.
 * If the width-w band covers every such offset (w >= d_max) no tying/better alignment
 * exists beyond w -> the width-w cell set (hence the optimal score) matches the ceiling.
 * Recall of the optimal score is 100% by construction. */
static inline bool band_cert_ok(int S, int h0, int len1, int len2, int w, const mem_opt_t *opt) {
    int min_len = len1 < len2 ? len1 : len2;
    int o_min = opt->o_del < opt->o_ins ? opt->o_del : opt->o_ins;
    int e_min = opt->e_del < opt->e_ins ? opt->e_del : opt->e_ins;
    if (e_min <= 0) return false;
    long num = (long)h0 + (long)min_len * opt->a - o_min - (long)S;
    if (num < 0) return true;                 /* no gapped alt can even tie */
    long d_max = num / e_min;                  /* largest offset that could tie/beat S */
    return (long)w >= d_max;                   /* band covers all tying offsets */
}
/* Rung count: narrowing tiers (init_w < opt->w) get one extra rung so the narrow probe
 * sits ahead of the full ceiling ladder [opt->w, 2w, 4w, 8w]. */
static inline int band_cert_nband(int init_w, const mem_opt_t *opt) {
    return (init_w < opt->w) ? MAX_BAND_TRY + 1 : MAX_BAND_TRY;
}
/* Width per rung: narrowing tiers -> [init_w, opt->w, 2w, 4w, 8w]; already-wide tiers ->
 * the standard ladder [opt->w, 2w, 4w, 8w]. */
static inline int32_t band_cert_width(int init_w, int i, const mem_opt_t *opt) {
    if (init_w < opt->w) return (i == 0) ? (int32_t)init_w : (int32_t)(opt->w << (i - 1));
    return (int32_t)(opt->w << i);
}
/* Accept: at the narrow probe (w < opt->w) finalize only on the certificate AND the clip
 * guard (the pair must reach the query end within the clip penalty of the local max, so a
 * band-truncated gscore cannot drive the clip-vs-extend choice at bwamem.cpp's clip
 * branch). Uncertified pairs fall to w >= opt->w and use the standard ceiling-ladder
 * accept clause -> identical to the non-adaptive path.
 *
 * GAP2 (the "restore" at each retry site): before this accept test the caller runs
 * `prev = a->score; a->score = sp->score;`, so the narrow probe overwrites a->score with
 * the width-w0 score. For a pair that fails to certify here, the retry loops restore
 * `a->score = prev` so the opt->w rung reads the pre-probe a->score as its `prev` -- exactly
 * what the non-adaptive ladder reads -- and the `score == prev` converged-accept fires
 * identically. The narrow rung writes no other a->* field before acceptance, so restoring
 * a->score fully un-pollutes the ladder. */
static inline bool band_cert_accept(int sc, int pv, int mo, int w, int tb, int i, int nband,
                                    const SeqPair *sp, const mem_opt_t *opt) {
    if (w < opt->w) {
        /* Left extensions clip on pen_clip5, right on pen_clip3; this accept is shared by both
         * directions, so guard with the smaller penalty -> the strictest threshold, which can only
         * under-certify (widen), never wrongly certify, under an asymmetric -L. */
        int pen_clip = opt->pen_clip5 < opt->pen_clip3 ? opt->pen_clip5 : opt->pen_clip3;
        /* Anchor the certificate on S - pen_clip, not S. band_cert_ok(S) alone only proves the
         * LOCAL max S (and its qle/tle) is band-invariant; a certified branch-B pair, however,
         * finalizes with the query-END gscore VALUE and gtle (truesc = gscore, rb -= gtle). Those
         * are a max over the last query row and can still grow when the band widens -- unless we
         * prove no out-of-band cell reaches even S - pen_clip. Since an accepted pair has
         * gscore > S - pen_clip (branch B), certifying at S - pen_clip forces every out-of-band
         * cell (incl. last-row) below the narrow gscore, so gscore's value, gtle, AND the
         * clip-vs-extend decision (threshold S - pen_clip5/pen_clip3 >= S - pen_clip) are all
         * band-invariant. This subsumes the old separate clip guard's soundness role. */
        if (!band_cert_ok(sc - pen_clip, sp->h0, sp->len1, sp->len2, w, opt)) return false;
        return sp->gscore > 0 && sp->gscore > sc - pen_clip;
    }
    return (i + 1 == nband) || (tb > 0 && w >= tb) || (sc == pv || mo < ((w >> 1) + (w >> 2)));
}
static inline int band_cert_mat_max(const int8_t *mat) {
    int mx = mat[0];
    for (int k = 1; k < 25; ++k) if (mat[k] > mx) mx = mat[k];
    return mx;
}
/* Runtime safety envelope for the certified adaptive band.
 *
 * band_cert_ok proves the OPTIMAL SCORE (and, with the S - pen_clip anchor, the
 * accepted pair's gscore value / gtle / clip decision) is band-invariant. It does
 * NOT bound the extension kernel's early-termination CONTROL FLOW: the zdrop break,
 * the all-zero-row (m==0) break, and the band-edge shrink read cells (out-of-band,
 * and in-band cells near the edge whose out-of-band neighbours differ between a
 * narrow and a wide run) that the certificate does not pin. In principle the narrow
 * probe and the full-width run can therefore terminate at different rows and report
 * a different gscore/gtle. On real reads this does not manifest even far from
 * default parameters, but it is only PROVABLY absent inside a parameter envelope
 * where those heuristics cannot diverge; outside it we fall back to the exact
 * full-width ladder (byte-identical for any -d/-L/-O/-E/-A/-B).
 *
 * Each condition is conservative -- it can only DISABLE the probe, never wrongly
 * enable it -- and default parameters are inside the envelope (so this is a no-op
 * at defaults and the ~3% win is preserved):
 *   - e_min > 0                                 division guard.
 *   - max(mat) <= opt->a                        the certificate uses opt->a as the
 *       per-column ceiling (num = h0 + min_len*a - ...); a custom or meth matrix
 *       scoring above a would break that bound. opt->a >= max(mat) keeps it a valid
 *       (conservative) ceiling, so band_cert_ok needs no matrix argument.
 *   - max(pen_clip5,pen_clip3) < o_min + e_min  no gapped path can reach within a
 *       clip penalty of the local max, closing the clip/gtle tie channel on either
 *       extension direction.
 *   - zdrop > o_min + (2*w0+1)*e_max + w0*a      a certified pair wastes <= w0*e_min
 *       and its optimal excursion is <= w0; this bounds the running-max lead an
 *       out-of-band cell can give the wide run, so neither run can zdrop where the
 *       other survives. Default zdrop=100 clears the default bound of 67.
 *   - (opt->w>>1)+(opt->w>>2) > w0 + 1 + (opt->a-1)/e_min   a certified pair finalizes
 *       at the narrow rung with a->w = max(a->w_init=opt->w, w0) = opt->w. The full-width
 *       ladder must reach the same a->w, i.e. accept the pair at its first rung (w=opt->w)
 *       via the max_off heuristic max_off < (w>>1)+(w>>2). max_off advances only on a cell
 *       that beats the running max (bandedSWA.cpp: `if (m > max) max_off = max(max_off,
 *       |mj-i|)`), and that running max is propped by the optimal path itself; combined
 *       with the certificate bounding gains, every record-setting offset on the wide run
 *       is < w0 + 1 + (a-pen_clip-1)/e_min <= w0 + 1 + (a-1)/e_min. Requiring the rung
 *       threshold to exceed that guarantees rung-1 acceptance, so a->w converges on both
 *       paths (else it would diverge -- opt->w vs a widened value -- feeding mem_patch_reg
 *       / mem_reg2aln). A within-w0 pair meets it at defaults; a custom small -w
 *       (opt->w ~ 21..26) or a large -A can fail it, and we fall back to the exact ladder.
 *       Default opt->w=100, a=1 -> 75 > 21. */
/* Envelope check at an arbitrary probe width w0. The two w0-dependent conditions (the
 * zdrop margin and the a->w first-rung-accept condition) TIGHTEN monotonically as w0 grows,
 * so `_w(opt, W) == 1` implies `_w(opt, W') == 1` for every W' <= W. The startup gate below
 * evaluates this at w0 = ADAPTIVE_BAND_START = 20; the certified probe rung re-evaluates it
 * at each wider bucket ceiling (Gate B) so it may only route a pair to a wider band when the
 * control-flow envelope still provably holds AT that band. The parameter-only conditions
 * (e_min>0, a_max<=a, pen_clip<o_min+e_min) are w0-independent and carry over unchanged. */
static int mem_band_cert_params_safe_w(const mem_opt_t *opt, int w0) {
    int o_min = min_(opt->o_del, opt->o_ins);
    int e_min = min_(opt->e_del, opt->e_ins);
    int e_max = max_(opt->e_del, opt->e_ins);
    int pen_clip_max = max_(opt->pen_clip5, opt->pen_clip3);
    if (e_min <= 0) return 0;
    int a_max = band_cert_mat_max(opt->mat);
    if (opt->meth_mode) {
        a_max = max_(a_max, max_(band_cert_mat_max(opt->mat_ot), band_cert_mat_max(opt->mat_ob)));
    }
    if (a_max > opt->a) return 0;
    if (pen_clip_max >= o_min + e_min) return 0;
    long zmargin = (long)o_min + (long)(2 * w0 + 1) * e_max + (long)w0 * opt->a;
    if ((long)opt->zdrop <= zmargin) return 0;
    if ((opt->w >> 1) + (opt->w >> 2) <= w0 + 1 + (opt->a - 1) / e_min) return 0;
    return 1;
}
int mem_band_cert_params_safe(const mem_opt_t *opt) {
    return mem_band_cert_params_safe_w(opt, ADAPTIVE_BAND_START);
}
#define BAND_NBAND(init_w) (opt->band_cert ? band_cert_nband((init_w), opt) : MAX_BAND_TRY)
#define BAND_WIDTH(init_w,i) (opt->band_cert ? band_cert_width((init_w), (i), opt) : ((int32_t)((init_w) << (i))))
#define INIT_W(w) (opt->band_cert ? min_(ADAPTIVE_BAND_START, (w)) \
                   : (opt->band_start > 0 ? min_(opt->band_start, (w)) : (w)))
/* The band_cert branch does not consult chain_band (cb): cb only gates the
 * aggressive band_start narrowing, and band_cert is mutually exclusive with it
 * -- band_cert=1 implies band_start=0 (the default), while --adaptive-band/--fast
 * set band_start>0 AND band_cert=0. No CLI path enables both, so cb has no role
 * under the certificate; band_cert_accept is a pure per-pair proof + clip guard.
 *
 * Implicit captures (like the sibling INIT_W/BAND_WIDTH macros): ACCEPT_PAIR reads
 * `opt`, `sp`, and `init_w` from the calling scope in addition to its parameters --
 * every retry-loop call site has all three in scope with those exact names. */
#define ACCEPT_PAIR(sc,pv,mo,w,tb,cb,i) \
    (opt->band_cert \
      ? band_cert_accept((sc),(pv),(mo),(w),(tb),(i),band_cert_nband(init_w,opt),sp,opt) \
      : ((i)+1==MAX_BAND_TRY || ((tb)>0 && (w)>=(tb)) || \
         (((sc)==(pv) || (mo) < ((w)>>1)+((w)>>2)) && (opt->band_start <= 0 || (w) >= (cb)))))
//------------------------------------------------------------------
// Alignment: Construct the alignment from a chain *

static inline int cal_max_gap(const mem_opt_t *opt, int qlen)
{
    int l_del = (int)((double)(qlen * opt->a - opt->o_del) / opt->e_del + 1.);
    int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) / opt->e_ins + 1.);
    //int l_del = (int)((double)(qlen * opt->a - opt->o_del) + 1.);
    //int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) + 1.);

    int l = l_del > l_ins? l_del : l_ins;
    l = l > 1? l : 1;
    return l < opt->w<<1? l : opt->w<<1;
}

//------------------------------------------------------------------
// SMEMs
static smem_aux_t *smem_aux_init()
{
    smem_aux_t *a;
    if ((a = (smem_aux_t *) calloc(BATCH_SIZE, sizeof(smem_aux_t))) == NULL) { fprintf(stderr, "ERROR: out of memory %s\n", __func__); exit(EXIT_FAILURE); }
    for (int i=0; i<BATCH_SIZE; i++)
    {
        a[i].tmpv[0] = (bwtintv_v *) calloc(1, sizeof(bwtintv_v));
        a[i].tmpv[1] = (bwtintv_v *) calloc(1, sizeof(bwtintv_v));
        if (!a[i].tmpv[0] || !a[i].tmpv[1]) { fprintf(stderr, "ERROR: out of memory %s\n", __func__); exit(EXIT_FAILURE); }
    }
    return a;
}

static void smem_aux_destroy(smem_aux_t *a)
{
    for (int i=0; i<BATCH_SIZE; i++)
    {
        free(a[i].tmpv[0]->a);
        free(a[i].tmpv[0]);
        free(a[i].tmpv[1]->a);
        free(a[i].tmpv[1]);
        free(a[i].mem.a);
        free(a[i].mem1.a);
    }
    free(a);
}

mem_opt_t *mem_opt_init()
{
    mem_opt_t *o;
    if ((o = (mem_opt_t *) calloc(1, sizeof(mem_opt_t))) == NULL)  { fprintf(stderr, "ERROR: out of memory\n"); exit(1); }
    o->flag = 0;
    o->a = 1; o->b = 4;
    o->o_del = o->o_ins = 6;
    o->e_del = o->e_ins = 1;
    o->w = 100;
    o->T = 30;
    o->zdrop = 100;
    o->pen_unpaired = 17;
    o->pen_clip5 = o->pen_clip3 = 5;

    o->max_mem_intv = 20;

    o->min_seed_len = 19;
    o->min_ext_len = 0;   // off by default -> byte-identical to baseline
    o->max_extend_chains = 0;   // off by default; opt-in speed lever (--max-extend-chains / --fast)
    o->mate_concordant_window = 0;   // off by default; opt-in (--extend-mate-concordant / --fast --meth)
    o->est_insert_high = 0;          // runtime state; set from data (mem_pestat) or -I during the run
    o->extend_csub = 0;          // off by default -> byte-identical; opt-in via --extend-csub
    o->extend_tie_frac = 0.0f;   // off by default -> byte-identical to baseline; opt-in via --extend-tie-frac / --fast
    o->extend_tie_floor = 0;     // no unconditional floor; only meaningful when extend_tie_frac > 0
    o->seed_emit_order = SEED_ORDER_OFF;  // byte-identical default
    o->smem_dedup  = 0;   // off by default -> byte-identical to baseline; opt-in via --smem-dedup
    o->alnreg_sort_fast = 0;  // off by default -> bwa-mem2's dedup sort (see mem_sort_dedup_patch); set by --fast
    o->skip_contained_ext = 0;   // off by default; opt-in via --skip-contained-ext (byte-identical)
    o->band_start  = 0;   // off by default (adaptive chain-geometry band); opt-in via --adaptive-band
    o->band_cert   = 1;   // on by default: sound (byte-identical) adaptive band via per-pair tie-break certificate
    o->split_width = 10;
    o->max_occ = 500;
    o->max_chain_gap = 10000;
    o->max_ins = 10000;
    o->mask_level = 0.50;
    o->drop_ratio = 0.50;
    o->XA_drop_ratio = 0.80;
    o->split_factor = 1.5;
    o->chunk_size = 10000000;
    o->n_threads = 1;
    o->bam_threads = -1;    // -1 = unset: auto-resolved from -t after parsing
    o->max_XA_hits = 5;
    o->max_XA_hits_alt = 200;
    o->max_matesw = 50;
    o->rescue_kmer = 0;      /* off by default (opt-in, not byte-identical) */
    o->rescue_band = MEM_RESCUE_BAND_DEFAULT;
    o->rescue_skip = 0;      /* off by default; requires rescue_kmer (opt-in) */
    o->mask_level_redun = 0.95;
    o->min_chain_weight = 0;
    o->max_chain_extend = 1<<30;
    o->mapQ_coef_len = 50; o->mapQ_coef_fac = log(o->mapQ_coef_len);
    o->meth_scoring = MEM_METH_SCORING_COLLAPSED;  /* --meth default: bwameth-compatible */
    o->meth_seed_prune = 0;                        /* base default off; --meth turns it on
                                                      (SPEC30), set in fastmap.cpp */
    o->meth_chem    = METH_CHEM_EMSEQ;             /* --meth default chemistry: bisulfite/em-seq */
    o->meth_tags    = MEM_METH_TAGS_ALL;           /* --meth default: emit XR/XG/XM (Bismark set) */
    o->compat       = &COMPAT_TARGET_OFF;          /* --compat off: bwa-mem3's native output */
    /* Match bwa and bwa-mem2, which both derive FLAG 0x2 from a[0]. Explicit
     * rather than left to calloc: this default is what keeps the proper-pair
     * bit agreeing with both upstreams on ALT-aware runs (fg-labs/bwa-mem3#362)
     * -- it settles FLAG 0x2 alone, not the rest of the record, which still
     * carries bwa-mem3's own MQ/HN tags and header unless --compat shapes them
     * away -- and a default that important should be visible where the others
     * are set. */
    o->proper_pair_from_emitted = 0;
    bwa_fill_scmat(o->a, o->b, o->mat);
    mem_opt_fill_meth_mat(o);
    return o;
}

/* D3 (--meth, PR-4): (re)derive the per-hypothesis ASYMMETRIC matrices from the
 * symmetric `mat` + match score `a`. Target-major mat[ref*5+read], ACGT order
 * A,C,G,T,N. Free the conversion cell; leave everything else (incl. the mirror
 * cell, a real variant) at the symmetric value:
 *   OT: free mat[C][T] = mat[1*5+3]  (ref C, read T = unmethylated C→T)
 *   OB: free mat[G][A] = mat[2*5+0]  (ref G, read A = bottom-strand G→A)
 * The freed VALUE and whether the mirror is freed too depend on opt->meth_scoring:
 * GENOMIC frees only that cell, to a MATCH (+a) ⇒ the mirror stays a real mismatch
 * and the SIMD kernel takes its rank-1 fast path. NEUTRAL frees only that cell, to
 * 0 (tolerated but not rewarded) ⇒ the mirror likewise stays a mismatch; one freed
 * cell but not rank-1, so bandedSWA uses its general-matrix path. COLLAPSED: the
 * mirror cell is ALSO freed (two cells, both to +a) so C/T and G/A are
 * interchangeable (bwameth-compatible), via bandedSWA's general-matrix path.
 * Selected per chain
 * via mem_opt_meth_mat(); valid only under --meth. MUST be called after EVERY
 * rebuild of opt->mat (mem_opt_init AND after main_mem parses -A/-B/-x and
 * re-runs bwa_fill_scmat) — otherwise the meth matrices keep the default
 * match/mismatch and silently ignore the user's scoring options. */
/* Map one tag name to its MEM_METH_TAG_* bit; 0 if unknown. Case-insensitive. */
static int meth_tag_bit(const char *name, size_t len)
{
    if (len != 2 || (name[0] != 'X' && name[0] != 'x')) return 0;
    switch (name[1]) {
        case 'R': case 'r': return MEM_METH_TAG_XR;
        case 'G': case 'g': return MEM_METH_TAG_XG;
        case 'M': case 'm': return MEM_METH_TAG_XM;
        default:            return 0;
    }
}

int mem_opt_parse_meth_tags(const char *spec, int *out, const char **err)
{
    if (spec == NULL || *spec == '\0') { *err = "empty tag spec"; return -1; }
    if (strcasecmp(spec, "all")  == 0) { *out = MEM_METH_TAGS_ALL; return 0; }
    if (strcasecmp(spec, "none") == 0) { *out = 0;                 return 0; }

    /* A spec is either all-inclusion (`XR,XG`) or all-exclusion (`^XM`).
     * Mixing the two has no obvious reading -- is `XR,^XM` "only XR" or
     * "everything but XM"? -- so reject it rather than pick a meaning. */
    int included = 0, excluded = 0, n_inc = 0, n_exc = 0;
    const char *p = spec;
    while (*p != '\0') {
        const char *comma = strchr(p, ',');
        size_t len = comma != NULL ? (size_t)(comma - p) : strlen(p);
        /* Separate from the unknown-tag case below: in `,XR` or `XR,,XG` there
         * is no tag name to be unknown, and saying so sends the user hunting
         * for a misspelling that isn't there. `XR,` is caught as a trailing
         * comma at the bottom of the loop. */
        if (len == 0) { *err = "empty tag in list"; return -1; }
        /* '-' is accepted as a synonym for '^' because a bare ^XM is a negated
         * glob in zsh with EXTENDED_GLOB (the oh-my-zsh/prezto default), where
         * it silently expands to every file in the directory except XM. '-XM'
         * needs no quoting in any common shell. */
        int is_exclusion = (*p == '^' || *p == '-');
        const char *name = is_exclusion ? p + 1 : p;
        size_t name_len  = is_exclusion ? len - 1 : len;
        int bit = meth_tag_bit(name, name_len);
        if (bit == 0) { *err = "unknown tag (expected XR, XG or XM)"; return -1; }
        if (is_exclusion) { excluded |= bit; ++n_exc; }
        else              { included |= bit; ++n_inc; }
        if (comma == NULL) break;
        p = comma + 1;
        if (*p == '\0') { *err = "trailing comma"; return -1; }
    }
    if (n_inc > 0 && n_exc > 0) {
        *err = "cannot mix included and ^excluded tags in one spec";
        return -1;
    }
    *out = n_exc > 0 ? (MEM_METH_TAGS_ALL & ~excluded) : included;
    return 0;
}

void mem_opt_fill_meth_mat(mem_opt_t *o) {
    memcpy(o->mat_ot, o->mat, sizeof(o->mat));
    memcpy(o->mat_ob, o->mat, sizeof(o->mat));
    /* Conversion cell: +a (rewarded) for COLLAPSED/GENOMIC, 0 (tolerated but not
     * rewarded) for NEUTRAL. See enum mem_meth_scoring for the rationale. */
    int8_t conv = (o->meth_scoring == MEM_METH_SCORING_NEUTRAL) ? 0 : (int8_t)o->a;
    o->mat_ot[1 * 5 + 3] = conv;   /* OT: ref C / read T (C→T conversion)   */
    o->mat_ob[2 * 5 + 0] = conv;   /* OB: ref G / read A (G→A conversion)   */
    if (o->meth_scoring == MEM_METH_SCORING_COLLAPSED) {
        /* bwameth-compatible: free the MIRROR cell too so C/T (and G/A) are
         * mutually interchangeable (collapsed 3-letter space). Two freed cells ⇒
         * bandedSWA general path, not the rank-1 fast path. Loses variant vs.
         * conversion discrimination; reproduces bwameth placement. */
        o->mat_ot[3 * 5 + 1] = o->a;   /* OT: ref T / read C → match */
        o->mat_ob[0 * 5 + 2] = o->a;   /* OB: ref A / read G → match */
    }
}

/* See bwamem.h for the contract. The `* opt->a` factors are the fix for -A
 * being discarded under --meth: bwameth's constants assume a == 1, and bwa
 * scales every other score-derived default by opt->a in update_a(). */
void mem_opt_apply_meth_defaults(mem_opt_t *opt, const mem_opt_t *opt0)
{
    if (!opt0->pen_clip5)    opt->pen_clip5    = 10 * opt->a;   /* bwameth -L 10 */
    if (!opt0->pen_clip3)    opt->pen_clip3    = 10 * opt->a;
    if (!opt0->pen_unpaired) opt->pen_unpaired = 100 * opt->a;  /* bwameth -U 100 (paired) */
    if (!opt0->T)            opt->T            = 40 * opt->a;   /* bwameth -T 40 */
    opt->flag |= MEM_F_NO_MULTI;                                /* -M */
    if (opt->meth_scoring == MEM_METH_SCORING_COLLAPSED) {
        if (!opt0->b) opt->b = 2 * opt->a;                      /* bwameth's lenient -B 2 */
    }
    /* GENOMIC and NEUTRAL keep bwa's default b (already scaled by update_a):
     * both are variant-aware, so the mirror cell must stay a real mismatch. */
}

/******************************
 * De-overlap single-end hits *
 ******************************/

/* The two comparators below are the two halves of the same trade-off, and
 * mem_sort_dedup_patch picks between them on opt->alnreg_sort_fast. Its first
 * sort feeds an ORDER-SENSITIVE dedup loop, so the comparator decides which of
 * several equal-`re` regions survives, propagating to `sub_n` and MAPQ:
 *
 *   mem_ars2_m2 (default) -- bwa-mem2's ORIGINAL comparator, a PARTIAL order on
 *     `re` alone, run under klib's unstable introsort. The surviving set depends
 *     on the *permutation* introsort produces, which is precisely what
 *     bwa-mem2's output is defined by. Kept so the default build reproduces
 *     upstream's dedup outcome exactly.
 *
 *   mem_ars2 (--fast) -- a STRICT TOTAL order: `re`, then the tie-break keys
 *     (rb, score-desc, qb). Every sort algorithm then produces the same
 *     surviving set, so the SAM is bit-equal regardless of which sort runs
 *     (klib introsort is unstable, pdqsort is unstable differently, radix is
 *     stable). Paired with pdqsort here.
 *
 * The two properties are mutually exclusive: any strict total order necessarily
 * picks different winners among equal-`re` records than the permutation
 * introsort happens to leave, so robustness-to-sort-choice and bwa-mem2 parity
 * cannot both hold. Covered by test/unit/test_alnreg_sort_dedup.cpp. */
#define alnreg_slt2_m2(a, b) ((a).re < (b).re)
KSORT_INIT(mem_ars2_m2, mem_alnreg_t, alnreg_slt2_m2)

#define alnreg_slt2(a, b) \
    ((a).re < (b).re || ((a).re == (b).re && \
     ((a).rb < (b).rb || ((a).rb == (b).rb && \
      ((a).score > (b).score || ((a).score == (b).score && \
       (a).qb < (b).qb))))))
KSORT_INIT(mem_ars2, mem_alnreg_t, alnreg_slt2)
PDQSORT_INIT(mem_ars2, mem_alnreg_t, alnreg_slt2)

PDQSORT_INIT(mem_ars2_m2, mem_alnreg_t, alnreg_slt2_m2)

#define alnreg_slt(a, b) ((a).score > (b).score || ((a).score == (b).score && ((a).rb < (b).rb || ((a).rb == (b).rb && (a).qb < (b).qb))))
KSORT_INIT(mem_ars, mem_alnreg_t, alnreg_slt)
PDQSORT_INIT(mem_ars, mem_alnreg_t, alnreg_slt)

/*----------------------------------------------------------------------------
 * Tie-free fast path for the default (bwa-mem2-compatible) dedup sorts.
 *
 * Under alnreg_sort_fast == 0 the surviving alnreg set is defined by the
 * *permutation* klib's unstable introsort happens to produce, so substituting a
 * faster sort changes which regions survive -- but only when the comparator
 * actually sees a tie. With no tied pair the sorted sequence is unique, so
 * pdqsort's result is bit-for-bit what introsort would have produced and the
 * substitution is free. Ties are therefore detected, not avoided.
 *
 * Ties are rare. Instrumented over 2.32M calls / 186M regions (533,334 HG00096
 * WGS pairs, -t 16): the `re` sort ties on 0.98% of calls and the closing
 * by-score sort on 0 of them -- two survivors cannot share (rb, qb), see the
 * comment at the end of mem_sort_dedup_patch. So ~99% of calls get pdqsort and
 * the rest pay pdqsort plus a restore plus introsort.
 *
 * Saving the input first is what makes the fallback exact: ks_introsort is
 * input-order-dependent, so running it on the pdqsort-permuted array would give
 * a different permutation -- and a different SAM -- than bwa-mem2.
 *--------------------------------------------------------------------------*/

/* Below ~9 elements pdqsort wins nothing: on the alnreg microbench
 * (an arm64 dedup-sort microbenchmark, 2-8 bucket) klib's introsort
 * leads on the `re` sort (127.4 vs 119.0 Mcell/s) and ties it on the by-score
 * sort (130.8 vs 130.1). Short arrays are ~46% of calls, so the save-copy is not
 * worth paying there for either. */
#define DEDUP_SORT_PDQ_MIN 9

/* Geometric-growth backing for the per-thread dedup scratch buffers below.
 * Grows `s` to hold at least `want` bytes, doubling to amortize repeated calls.
 * realloc returns storage suitably aligned for mem_alnreg_t and the pair types.
 * Returns the buffer, or NULL on allocation failure -- leaving the existing
 * buffer intact so the caller can take the exact (introsort) fallback. */
static inline void *scratch_grow(u8vec_scratch_t &s, size_t want)
{
    if (s.v.m < want) {
        size_t cap = grow_capacity(s.v.m, want, 1);   /* byte-counted: shared growth policy */
        uint8_t *p = (uint8_t *)realloc(s.v.a, cap);
        if (p == NULL) return NULL;   /* keep the existing buffer intact */
        s.v.a = p;
        s.v.m = cap;
    }
    return s.v.a;
}

/* Per-thread save buffer, grown geometrically and released at thread exit. Held
 * as bytes so it can reuse the u8vec scratch holder. Returns NULL only if the
 * allocation fails, in which case the caller takes the exact (introsort) path. */
static inline mem_alnreg_t *alnreg_save_buf(int n)
{
    static thread_local u8vec_scratch_t save;
    return (mem_alnreg_t *)scratch_grow(save, (size_t)n * sizeof(mem_alnreg_t));
}

/* Emit `static void fn(int n, mem_alnreg_t *a)` that sorts `a` exactly as
 * ks_introsort(name, n, a) would, but via pdqsort_##name whenever `lt` sees no
 * tie. `lt` must be the comparator KSORT_INIT and PDQSORT_INIT were given. */
#define DEDUP_TIE_SORT_INIT(fn, name, lt)                                        \
    static void fn(int n, mem_alnreg_t *a)                                       \
    {                                                                            \
        mem_alnreg_t *save;                                                      \
        if (n < DEDUP_SORT_PDQ_MIN || (save = alnreg_save_buf(n)) == NULL) {     \
            ks_introsort(name, n, a);                                            \
            return;                                                              \
        }                                                                        \
        memcpy(save, a, (size_t)n * sizeof(*a));                                 \
        pdqsort_##name(n, a);                                                    \
        for (int i = 1; i < n; ++i) {                                            \
            /* `a` is sorted, so !lt(a[i-1], a[i]) means the pair is tied. */    \
            if (!lt(a[i - 1], a[i])) {                                           \
                memcpy(a, save, (size_t)n * sizeof(*a));                         \
                ks_introsort(name, n, a);                                        \
                return;                                                          \
            }                                                                    \
        }                                                                        \
    }

DEDUP_TIE_SORT_INIT(dedup_sort_by_re,    mem_ars2_m2, alnreg_slt2_m2)
DEDUP_TIE_SORT_INIT(dedup_sort_by_score, mem_ars,     alnreg_slt)

/* Exposed for test/unit/test_alnreg_sort_dedup.cpp, which asserts each fast path
 * against the introsort it claims to reproduce, array for array. Thin enough not
 * to perturb the call sites above, which still call the static helpers. Each
 * takes the region count `n` and the array `a`, sorted in place. */

/* The `re` sort as mem_sort_dedup_patch runs it by default: pdqsort, or restore
 * plus introsort on a tie. */
void bwamem3_dedup_sort_by_re(int n, mem_alnreg_t *a)          { dedup_sort_by_re(n, a); }
/* The `re` sort bwa-mem2 runs: klib introsort, unconditionally. The oracle the
 * fast path above must match record for record. */
void bwamem3_dedup_sort_by_re_exact(int n, mem_alnreg_t *a)    { ks_introsort(mem_ars2_m2, n, a); }
/* The closing by-score sort as mem_sort_dedup_patch runs it by default. */
void bwamem3_dedup_sort_by_score(int n, mem_alnreg_t *a)       { dedup_sort_by_score(n, a); }
/* The closing by-score sort bwa-mem2 runs, i.e. the oracle for the above. */
void bwamem3_dedup_sort_by_score_exact(int n, mem_alnreg_t *a) { ks_introsort(mem_ars, n, a); }

/*----------------------------------------------------------------------------
 * Permutation-gather dedup sorts (default path).
 *
 * The in-place dedup_sort_by_re/_by_score above keep bwa-mem2 parity on tied
 * `re` (klib introsort's unstable permutation) by saving the whole array aside
 * on every call >= DEDUP_SORT_PDQ_MIN, so a tie can be resolved by restoring and
 * re-running ks_introsort. That 112-byte-per-element save-copy is paid on 100%
 * of calls to cover the ~1% that actually tie.
 *
 * These variants remove it. They sort (key, index) pairs -- 16 or 24 bytes, not
 * 112 -- with the SAME algorithms (pdqsort when tie-free, klib ks_introsort on a
 * tie), so the index order is exactly the permutation the in-place sort would
 * apply, ties included; then one gather materialises the records into a separate
 * `dst`. Since `src` is never overwritten, the tie fallback just rebuilds and
 * re-sorts the cheap pairs -- there is no struct save-copy. Each swap during the
 * sort moves a pair; the records move exactly once, at the gather.
 *
 * `dst` differs from `src`, so mem_sort_dedup_patch alternates its two sorts
 * src<->scratch (an A/B ping-pong) to land the final records back in the
 * caller's own array. Byte-identity to unconditional ks_introsort is asserted in
 * test/unit/test_alnreg_sort_dedup.cpp via the same harness as the in-place
 * fast paths.
 *--------------------------------------------------------------------------*/

typedef struct { int64_t re;    uint32_t idx; } dedup_pair_re;
typedef struct { int score; int64_t rb; int qb; uint32_t idx; } dedup_pair_sc;

/* Pair comparators MUST mirror the record comparators' keys exactly:
 * dedup_pre_lt == alnreg_slt2_m2 (re only); dedup_psc_lt == alnreg_slt. */
#define dedup_pre_lt(a, b) ((a).re < (b).re)
#define dedup_psc_lt(a, b) ((a).score > (b).score || ((a).score == (b).score && \
                            ((a).rb < (b).rb || ((a).rb == (b).rb && (a).qb < (b).qb))))
KSORT_INIT(dedup_pre, dedup_pair_re, dedup_pre_lt)
KSORT_INIT(dedup_psc, dedup_pair_sc, dedup_psc_lt)
PDQSORT_INIT(dedup_pre, dedup_pair_re, dedup_pre_lt)
PDQSORT_INIT(dedup_psc, dedup_pair_sc, dedup_psc_lt)

/* Below this the pair indirection costs more than the 112-byte moves it saves,
 * and the in-place path pays no save-copy there anyway (< DEDUP_SORT_PDQ_MIN). */
#define DEDUP_PERM_MIN 9

/* Thread-local scratch for the (key,index) pairs and the record gather target,
 * each grown geometrically and reused across calls (bytes-backed to share the
 * u8vec holder). Return NULL only on allocation failure, leaving the existing
 * buffer intact; the caller then takes the in-place path. */
static inline void *dedup_pair_buf(size_t want)
{
    static thread_local u8vec_scratch_t pairs;
    return scratch_grow(pairs, want);
}
static inline mem_alnreg_t *dedup_gather_buf(int n)
{
    static thread_local u8vec_scratch_t gather;
    return (mem_alnreg_t *)scratch_grow(gather, (size_t)n * sizeof(mem_alnreg_t));
}

/* Sort `src` into `dst` by `re` (the default comparator), reproducing
 * ks_introsort(mem_ars2_m2)'s array exactly. On pair-buffer OOM, falls back to an
 * in-place introsort on a straight copy -- still byte-identical. */
static void dedup_perm_sort_by_re(int n, const mem_alnreg_t *src, mem_alnreg_t *dst)
{
    /* The gather below reads `src` and writes `dst`; they must be distinct
     * buffers. mem_sort_dedup_patch guarantees this by ping-ponging the caller
     * array against the thread-local gather scratch (dedup_gather_buf), so `scr`
     * owns that buffer for the whole call. Aliasing would corrupt records with
     * no diagnostic, so catch it in all builds -- xassert fires under -DNDEBUG,
     * unlike assert, which would compile the check out on the release path. */
    xassert(src != dst, "dedup permutation sort requires distinct source and destination buffers");
    dedup_pair_re *p =
        (dedup_pair_re *)dedup_pair_buf((size_t)n * sizeof(dedup_pair_re));
    if (p == NULL) {
        memcpy(dst, src, (size_t)n * sizeof(*src));
        ks_introsort(mem_ars2_m2, n, dst);
        return;
    }
    for (int i = 0; i < n; ++i) { p[i].re = src[i].re; p[i].idx = (uint32_t)i; }
    pdqsort_dedup_pre((size_t)n, p);
    int tie = 0;
    for (int i = 1; i < n; ++i)
        if (!dedup_pre_lt(p[i - 1], p[i])) { tie = 1; break; }
    if (tie) {  /* restore the pairs and reproduce introsort's permutation */
        for (int i = 0; i < n; ++i) { p[i].re = src[i].re; p[i].idx = (uint32_t)i; }
        ks_introsort_dedup_pre((size_t)n, p);
    }
    for (int i = 0; i < n; ++i) dst[i] = src[p[i].idx];  /* the single gather */
}

/* The by-score analogue, reproducing ks_introsort(mem_ars). */
static void dedup_perm_sort_by_score(int n, const mem_alnreg_t *src, mem_alnreg_t *dst)
{
    xassert(src != dst, "dedup permutation sort requires distinct source and destination buffers");  /* see dedup_perm_sort_by_re: src/dst must not alias */
    dedup_pair_sc *p =
        (dedup_pair_sc *)dedup_pair_buf((size_t)n * sizeof(dedup_pair_sc));
    if (p == NULL) {
        memcpy(dst, src, (size_t)n * sizeof(*src));
        ks_introsort(mem_ars, n, dst);
        return;
    }
    for (int i = 0; i < n; ++i) {
        p[i].score = src[i].score; p[i].rb = src[i].rb; p[i].qb = src[i].qb;
        p[i].idx = (uint32_t)i;
    }
    pdqsort_dedup_psc((size_t)n, p);
    int tie = 0;
    for (int i = 1; i < n; ++i)
        if (!dedup_psc_lt(p[i - 1], p[i])) { tie = 1; break; }
    if (tie) {
        for (int i = 0; i < n; ++i) {
            p[i].score = src[i].score; p[i].rb = src[i].rb; p[i].qb = src[i].qb;
            p[i].idx = (uint32_t)i;
        }
        ks_introsort_dedup_psc((size_t)n, p);
    }
    for (int i = 0; i < n; ++i) dst[i] = src[p[i].idx];
}

/* Exposed for test/unit/test_alnreg_sort_dedup.cpp: thin in-place wrappers (perm
 * into the gather scratch, then copy back) so the byte-identity harness -- which
 * drives an in-place sort_fn -- can validate the src->dst core. The hot path in
 * mem_sort_dedup_patch calls the core directly and never copies back. */
void bwamem3_dedup_perm_sort_by_re(int n, mem_alnreg_t *a)
{
    mem_alnreg_t *tmp = dedup_gather_buf(n);
    if (tmp == NULL) { ks_introsort(mem_ars2_m2, n, a); return; }
    dedup_perm_sort_by_re(n, a, tmp);
    memcpy(a, tmp, (size_t)n * sizeof(*a));
}
void bwamem3_dedup_perm_sort_by_score(int n, mem_alnreg_t *a)
{
    mem_alnreg_t *tmp = dedup_gather_buf(n);
    if (tmp == NULL) { ks_introsort(mem_ars, n, a); return; }
    dedup_perm_sort_by_score(n, a, tmp);
    memcpy(a, tmp, (size_t)n * sizeof(*a));
}

#define alnreg_hlt(a, b)  ((a).score > (b).score || ((a).score == (b).score && ((a).is_alt < (b).is_alt || ((a).is_alt == (b).is_alt && (a).hash < (b).hash))))
KSORT_INIT(mem_ars_hash, mem_alnreg_t, alnreg_hlt)

#define alnreg_hlt2(a, b) ((a).is_alt < (b).is_alt || ((a).is_alt == (b).is_alt && ((a).score > (b).score || ((a).score == (b).score && (a).hash < (b).hash))))
KSORT_INIT(mem_ars_hash2, mem_alnreg_t, alnreg_hlt2)

#if MATE_SORT
void sort_alnreg_re(int n, mem_alnreg_t* a) {
    pdqsort_mem_ars2(n, a);
}

void sort_alnreg_score(int n, mem_alnreg_t* a) {
    pdqsort_mem_ars(n, a);
}

#endif

#define PATCH_MAX_R_BW 0.05f
#define PATCH_MIN_SC_RATIO 0.90f

int mem_patch_reg(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                  uint8_t *query, const mem_alnreg_t *a, const mem_alnreg_t *b,
                  int *_w, const int8_t *mat)
{
    /* D3 (--meth, PR-4): `mat` is the per-hypothesis substitution matrix and
     * `query` the original read bases; the caller passes opt->mat + projected
     * read outside --meth so behavior is unchanged there. */
    int w, score, q_s, r_s;
    double r;
    if (bns == 0 || pac == 0 || query == 0) return 0;
    assert(a->rid == b->rid && a->rb <= b->rb);

    if (a->rb < bns->l_pac && b->rb >= bns->l_pac) return 0; // on different strands
    if (a->qb >= b->qb || a->qe >= b->qe || a->re >= b->re) return 0; // not colinear
    w = (a->re - b->rb) - (a->qe - b->qb); // required bandwidth
    w = w > 0? w : -w; // l = abs(l)
    r = (double)(a->re - b->rb) / (b->re - a->rb) - (double)(a->qe - b->qb) / (b->qe - a->qb); // relative bandwidth
    r = r > 0.? r : -r; // r = fabs(r)

    if (bwa_verbose >= 4)
        fprintf(stderr, "* potential hit merge between [%d,%d)<=>[%ld,%ld) and "
               "[%d,%d)<=>[%ld,%ld), @ %s; w=%d, r=%.4g\n",
               a->qb, a->qe, (long)a->rb, (long)a->re, b->qb, b->qe,
               (long)b->rb, (long)b->re, bns->anns[a->rid].name, w, r);

    if (a->re < b->rb || a->qe < b->qb) // no overlap on query or on ref
    {
        if (w > opt->w<<1 || r >= PATCH_MAX_R_BW) return 0; // the bandwidth or the relative bandwidth is too large
    } else if (w > opt->w<<2 || r >= PATCH_MAX_R_BW*2) return 0; // more permissive if overlapping on both ref and query

    // global alignment
    w += a->w + b->w;
    w = w < opt->w<<2? w : opt->w<<2;

    if (bwa_verbose >= 4)
        fprintf(stderr, "* test potential hit merge with global alignment; w=%d\n", w);

    bwa_gen_cigar2(mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, w,
                   bns->l_pac, pac, b->qe - a->qb, query + a->qb, a->rb, b->re,
                   &score, 0, 0);

    q_s = (int)((double)(b->qe - a->qb) / ((b->qe - b->qb) + (a->qe - a->qb)) *
                (b->score + a->score) + .499); // predicted score from query

    r_s = (int)((double)(b->re - a->rb) / ((b->re - b->rb) + (a->re - a->rb)) *
                (b->score + a->score) + .499); // predicted score from ref

    if (bwa_verbose >= 4)
        fprintf(stderr, "* score=%d;(%d,%d)\n", score, q_s, r_s);

    if ((double)score / (q_s > r_s? q_s : r_s) < PATCH_MIN_SC_RATIO) return 0;
    *_w = w;
    return score;
}
/*********************************
 * Test if a seed is good enough *
 *********************************/

#define MEM_SHORT_EXT 50
#define MEM_SHORT_LEN 200

#define MEM_HSP_COEF 1.1f
#define MEM_MINSC_COEF 5.5f
#define MEM_SEEDSW_COEF 0.05f

#if MATE_SORT
int mem_dedup_patch(const mem_opt_t *opt, const bntseq_t *bns,
                    const uint8_t *pac, uint8_t *query, int n,
                    mem_alnreg_t *a)
{
    int m, i, j;
    if (n <= 1) return n;
    for (i = 0; i < n; ++i) a[i].n_comp = 1;

    for (i = 1; i < n; ++i)
    {
        mem_alnreg_t *p = &a[i];
        if (p->rid != a[i-1].rid || p->rb >= a[i-1].re + opt->max_chain_gap)
            continue; // then no need to go into the loop below

        for (j = i - 1; j >= 0 && p->rid == a[j].rid && p->rb < a[j].re + opt->max_chain_gap; --j) {
            mem_alnreg_t *q = &a[j];
            int64_t or_, oq, mr, mq;
            int score, w;
            if (q->qe == q->qb) continue; // a[j] has been excluded
            or_ = q->re - p->rb; // overlap length on the reference
            oq = q->qb < p->qb? q->qe - p->qb : p->qe - q->qb; // overlap length on the query
            mr = q->re - q->rb < p->re - p->rb? q->re - q->rb : p->re - p->rb; // min ref len in alignment
            mq = q->qe - q->qb < p->qe - p->qb? q->qe - q->qb : p->qe - p->qb; // min qry len in alignment
            if (or_ > opt->mask_level_redun * mr && oq > opt->mask_level_redun * mq) { // one of the hits is redundant
                if (p->score < q->score)
                {
                    p->qe = p->qb;
                    break;
                }
                else q->qe = q->qb;
            }
            else if (q->rb < p->rb && (score = mem_patch_reg(opt, bns, pac, query, q, p, &w, opt->mat)) > 0) { // then merge q into p
                p->n_comp += q->n_comp + 1;
                p->seedcov = p->seedcov > q->seedcov? p->seedcov : q->seedcov;
                p->sub = p->sub > q->sub? p->sub : q->sub;
                p->csub = p->csub > q->csub? p->csub : q->csub;
                p->qb = q->qb, p->rb = q->rb;
                p->truesc = p->score = score;
                p->w = w;
                q->qb = q->qe;
            }
        }
    }
    for (i = 0, m = 0; i < n; ++i) // exclude identical hits
        if (a[i].qe > a[i].qb) {
            if (m != i) a[m++] = a[i];
            else ++m;
        }
    n = m;
    return m;
}
#endif

int mem_sort_dedup_patch(const mem_opt_t *opt, const bntseq_t *bns,
                         const uint8_t *pac, uint8_t *query, int n,
                         mem_alnreg_t *a, const int8_t *mat)
{
    /* D3 (--meth, PR-4): `mat` is the per-read OT/OB matrix and `query` the
     * original read bases (the caller threads both under --meth); outside --meth
     * `mat` is opt->mat and `query` the normal read, so behavior is unchanged.
     * All of a read's alnregs share one hypothesis under directional --meth, so a
     * single per-read matrix is correct for the colinear merge below. mat == NULL
     * (the default used by dedup-only callers, e.g. mate rescue passing 0,0,0)
     * resolves to opt->mat — those callers also pass bns==0 so no patch SW runs. */
    if (mat == NULL) mat = opt->mat;
    int m, i, j;
    if (n <= 1) return n;

    /* The default path reorders the 112-byte records by sorting a (key,index)
     * permutation and gathering once, which drops the per-call save-copy the
     * in-place tie-detection scheme pays (dedup_perm_sort_by_re/_by_score). The
     * gather writes a DIFFERENT buffer than it reads, so the two sorts ping-pong
     * a<->scr and the final records land back in the caller's own array `a_orig`.
     * `scr` is the gather target, sized to the entry n; NULL disables the
     * permutation form (small n, --fast, or scratch OOM), leaving the exact
     * in-place behaviour. The dedup loop below operates on `a`, which the sort1
     * swap may repoint at `scr`. */
    mem_alnreg_t *const a_orig = a;
    mem_alnreg_t *scr = (!opt->alnreg_sort_fast && n >= DEDUP_PERM_MIN)
                        ? dedup_gather_buf(n) : NULL;

    /* Default: bwa-mem2's re-only comparator, ordered exactly as klib introsort
     * would order it -- reproducing upstream's dedup outcome. --fast: strict total
     * order + pdqsort unconditionally, resolving equal-`re` ties differently. */
    if (opt->alnreg_sort_fast) pdqsort_mem_ars2(n, a);       // sort by the END position, not START!
    else if (scr) { dedup_perm_sort_by_re(n, a, scr); mem_alnreg_t *t = a; a = scr; scr = t; }
    else           dedup_sort_by_re(n, a);

    for (i = 0; i < n; ++i) a[i].n_comp = 1;
    for (i = 1; i < n; ++i)
    {
        mem_alnreg_t *p = &a[i];
        if (p->rid != a[i-1].rid || p->rb >= a[i-1].re + opt->max_chain_gap)
            continue; // then no need to go into the loop below

        for (j = i - 1; j >= 0 && p->rid == a[j].rid && p->rb < a[j].re + opt->max_chain_gap; --j) {
            mem_alnreg_t *q = &a[j];
            int64_t or_, oq, mr, mq;
            int score, w;
            if (q->qe == q->qb) continue; // a[j] has been excluded
            or_ = q->re - p->rb; // overlap length on the reference
            oq = q->qb < p->qb? q->qe - p->qb : p->qe - q->qb; // overlap length on the query
            mr = q->re - q->rb < p->re - p->rb? q->re - q->rb : p->re - p->rb; // min ref len in alignment
            mq = q->qe - q->qb < p->qe - p->qb? q->qe - q->qb : p->qe - p->qb; // min qry len in alignment
            if (or_ > opt->mask_level_redun * mr && oq > opt->mask_level_redun * mq) { // one of the hits is redundant
                if (p->score < q->score)
                {
                    p->qe = p->qb;
                    break;
                }
                else q->qe = q->qb;
            }
            else if (q->rb < p->rb && (score = mem_patch_reg(opt, bns, pac, query, q, p, &w, mat)) > 0) { // then merge q into p
                p->n_comp += q->n_comp + 1;
                p->seedcov = p->seedcov > q->seedcov? p->seedcov : q->seedcov;
                p->sub = p->sub > q->sub? p->sub : q->sub;
                p->csub = p->csub > q->csub? p->csub : q->csub;
                p->qb = q->qb, p->rb = q->rb;
                p->truesc = p->score = score;
                p->w = w;
                q->qb = q->qe;
            }
        }
    }
    for (i = 0, m = 0; i < n; ++i) // exclude identical hits
        if (a[i].qe > a[i].qb) {
            if (m != i) a[m++] = a[i];
            else ++m;
        }
    n = m;
    if (opt->alnreg_sort_fast)                 pdqsort_mem_ars(n, a);
    else if (scr && n >= DEDUP_PERM_MIN) { dedup_perm_sort_by_score(n, a, scr); mem_alnreg_t *t = a; a = scr; scr = t; }
    else                                       dedup_sort_by_score(n, a);
    /* Return the records to the caller's own buffer. With both sorts permuting
     * (the common large-n case) the a<->scr ping-pong already leaves `a == a_orig`
     * and this is a no-op; the copy only runs when an odd number of gathers landed
     * the data in `scr` (e.g. dedup shrank n below DEDUP_PERM_MIN before sort2),
     * and then only over the few surviving records. */
    if (a != a_orig) memcpy(a_orig, a, (size_t)n * sizeof(*a));
    /* The historical post-sort exact-duplicate passes (mark then exclude regions
     * sharing (score, rb, qb)) have been removed: they are provably dead. Any two
     * regions with equal (rb, qb) have the shorter fully contained in the longer,
     * so on the reference or_ == mr and on the query oq >= mq; with the hardcoded
     * mask_level_redun = 0.95 (< 1) the sliding-window merge above always takes
     * its redundant branch (or_ > 0.95*mr && oq > 0.95*mq) and drops one of them.
     * Same-read alignments sharing rb are always within max_chain_gap, so the pair
     * is always in-window. Hence the exact-(score,rb,qb)-duplicate case is a strict
     * subset of what the merge already removes. Confirmed empirically: 0 removals
     * across 337M regions on HG002 WGS, and SAM output byte-identical. The by-score
     * sort is retained — its ordering is relied on downstream by mem_mark_primary_se
     * / mem_pair (removing it changes primary selection). */
    return n;
}


// return 1 if the seed is merged into the chain
static int test_and_merge(const mem_opt_t *opt, int64_t l_pac, mem_chain_t *c,
                          const mem_seed_t *p, int seed_rid, int tid)
{
    int64_t qend, rend, x, y;
    const mem_seed_t *last = &c->seeds[c->n-1];
    qend = last->qbeg + last->len;
    rend = last->rbeg + last->len;

    if (seed_rid != c->rid) return 0; // different chr; request a new chain
    if (p->qbeg >= c->seeds[0].qbeg && p->qbeg + p->len <= qend &&
        p->rbeg >= c->seeds[0].rbeg && p->rbeg + p->len <= rend)
        return 1; // contained seed; do nothing

    if ((last->rbeg < l_pac || c->seeds[0].rbeg < l_pac) &&
        p->rbeg >= l_pac) return 0; // don't chain if on different strand

    x = p->qbeg - last->qbeg; // always non-negtive
    y = p->rbeg - last->rbeg;
    if (y >= 0 && x - y <= opt->w && y - x <= opt->w &&
        x - last->len < opt->max_chain_gap &&
        y - last->len < opt->max_chain_gap) { // grow the chain
        if (c->n == c->m)
        {
            mem_seed_t *auxSeedBuf = NULL;
            int pm = c->m;
            c->m <<= 1;
            if (pm == SEEDS_PER_CHAIN) {  // re-new memory
                /* CHN-11: was calloc; the zero-fill it did over [0, m) was pure
                 * waste -- [0, n) is immediately overwritten by the memcpy below
                 * and [n, m) by the shared memset that follows (identical to the
                 * realloc branch). malloc leaves exactly the same final bytes. */
                if ((auxSeedBuf = (mem_seed_t *) malloc(c->m * sizeof(mem_seed_t))) == NULL) { fprintf(stderr, "ERROR: out of memory auxSeedBuf\n"); exit(1); }
                memcpy((char*) (auxSeedBuf), c->seeds, c->n * sizeof(mem_seed_t));
                c->seeds = auxSeedBuf;
            } else {  // new memory
                // fprintf(stderr, "[%0.4d] re-allocing old seed, m: %d\n", tid, c->m);
                if ((auxSeedBuf = (mem_seed_t *) realloc(c->seeds, c->m * sizeof(mem_seed_t))) == NULL) { fprintf(stderr, "ERROR: out of memory auxSeedBuf\n"); exit(1); }
                c->seeds = auxSeedBuf;
            }
            memset((char*) (c->seeds + c->n), 0, (c->m - c->n) * sizeof(mem_seed_t));
        }
        c->seeds[c->n++] = *p;
        return 1;
    }
    return 0; // request to add a new chain
}

int mem_seed_sw(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                int l_query, const uint8_t *query, const mem_seed_t *s,
                const int8_t *mat)
{
    /* D3 (--meth, PR-4): `mat` is the (possibly asymmetric, per-hypothesis)
     * substitution matrix for this chain; the caller passes opt->mat outside
     * --meth so behavior is unchanged there. */
    int qb, qe, rid;
    int64_t rb, re, mid, l_pac = bns->l_pac;
    uint8_t *rseq = 0;
    kswr_t x;

    if (s->len >= MEM_SHORT_LEN) return -1; // the seed is longer than the max-extend; no need to do SW
    qb = s->qbeg, qe = s->qbeg + s->len;
    rb = s->rbeg, re = s->rbeg + s->len;
    mid = (rb + re) >> 1;
    qb -= MEM_SHORT_EXT; qb = qb > 0? qb : 0;
    qe += MEM_SHORT_EXT; qe = qe < l_query? qe : l_query;
    rb -= MEM_SHORT_EXT; rb = rb > 0? rb : 0;
    re += MEM_SHORT_EXT; re = re < l_pac<<1? re : l_pac<<1;
    if (rb < l_pac && l_pac < re) {
        if (mid < l_pac) re = l_pac;
        else rb = l_pac;
    }
    if (qe - qb >= MEM_SHORT_LEN || re - rb >= MEM_SHORT_LEN) return -1; // the seed seems good enough; no need to do SW

    {
        /* Prefetch the reference window's first pac[] cache line a few instructions ahead of the
         * fetch (cf. AP6 on the CIGAR-emission path): bns_fetch_seq_into makes a random
         * ~DRAM-latency load. The head start is modest — only the scratch-resize check and
         * bns_pos2rid sit between the hint and the load — but the window is bounded by
         * MEM_SHORT_EXT (~13 packed bytes), so the pre-clamp rb used here and the post-clamp start
         * bns_get_seq_into actually reads almost always land in the same cache line. bns_depos maps
         * the strand-dependent window start (fwd pac[rb>>2]; rev pac[(2*l_pac-1-rb)>>2]). Pure hint
         * -> byte-identical. */
        {
            int is_rev_pf;  // discarded; bns_depos also reports the strand
            __builtin_prefetch(&pac[bns_depos(bns, rb, &is_rev_pf) >> 2], 0, 3);
        }
        static thread_local u8vec_scratch_t t_rseq;
        size_t want = (size_t)((re - rb) + 64);
        if (t_rseq.v.m < want) kv_resize(uint8_t, t_rseq.v, want);
        int64_t rlen;
        bns_fetch_seq_into(bns, pac, &rb, mid, &re, &rid, t_rseq.v.a, &rlen);
        rseq = t_rseq.v.a;
        // No qry-profile cache: each seed slices a different sub-query
        // (query+qb, qe-qb), so ksw_align2 always builds a fresh profile.
        x = ksw_align2(qe - qb, (uint8_t*)query + qb, re - rb, rseq, 5, mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, KSW_XSTART, NULL);
        /* rseq aliases thread-local scratch; do not free. */
    }
    return x.score;
}

int mem_chain_weight(const mem_chain_t *c)
{
    /* CL-1: fused query- and reference-axis coverage into a single seed sweep.
     * The two accumulations are independent (query coverage keys off qbeg/len,
     * reference coverage off rbeg/len), so interleaving them in one pass with a
     * separate running end per axis is byte-identical to the former two sweeps
     * -- it only halves the loop iterations (the second sweep re-read the same
     * L1-resident seeds), a tidy-up rather than a speedup. */
    int64_t qend, rend;
    int j, wq = 0, wr = 0, w;
    for (j = 0, qend = 0, rend = 0; j < c->n; ++j) {
        const mem_seed_t *s = &c->seeds[j];
        if (s->qbeg >= qend) wq += s->len;
        else if (s->qbeg + s->len > qend) wq += s->qbeg + s->len - qend;
        qend = qend > s->qbeg + s->len? qend : s->qbeg + s->len;
        if (s->rbeg >= rend) wr += s->len;
        else if (s->rbeg + s->len > rend) wr += s->rbeg + s->len - rend;
        rend = rend > s->rbeg + s->len? rend : s->rbeg + s->len;
    }
    w = wq < wr? wq : wr;
    /* Saturate at what mem_chain_t::w can actually hold. The clamp used to be
     * (1<<30)-1, three bits wider than the field, so a weight in [2^27, 2^30)
     * wrapped to a small number on store instead of staying large (#309). The
     * shared MEM_CHAIN_W_MAX keeps the two in step. */
    return w < (int)MEM_CHAIN_W_MAX ? w : (int)MEM_CHAIN_W_MAX;
}

void mem_print_chain(const bntseq_t *bns, mem_chain_v *chn)
{
    int i, j;
    for (i = 0; i < chn->n; ++i)
    {
        mem_chain_t *p = &chn->a[i];
        fprintf(stderr, "* Found CHAIN(%d): n=%d; weight=%d", i, p->n, mem_chain_weight(p));
        for (j = 0; j < p->n; ++j)
        {
            bwtint_t pos;
            int is_rev;
            pos = bns_depos(bns, p->seeds[j].rbeg, &is_rev);
            if (is_rev) pos -= p->seeds[j].len - 1;
            fprintf(stderr, "\t%d;%d;%d,%ld(%s:%c%ld)",
                       p->seeds[j].score, p->seeds[j].len, p->seeds[j].qbeg,
                       (long)p->seeds[j].rbeg, bns->anns[p->rid].name,
                       "+-"[is_rev], (long)(pos - bns->anns[p->rid].offset) + 1);
        }
        fputc('\n', stderr);
    }
}

// Skip-short-seed extension filter: in a chain that still has a long-enough
// anchor, drop seeds shorter than min_ext_len so they are never extended (their
// banded Smith-Waterman is skipped downstream) -- those short seeds are collinear
// with the anchor and its extension already covers them, so dropping them is
// near output-neutral and pure speed.
//
// A chain with NO seed >= min_ext_len is left untouched: its only evidence is
// short, so dropping it would empty the chain and the read would go unmapped.
// Such all-short chains (common on low-mappability / repetitive reads) therefore
// extend exactly as the default does. This guarantees the filter never empties a
// non-empty chain, so it can only reduce extension work, never lose a read.
//
// Stable compaction -- surviving seeds keep their order. Returns the new seed
// count. min_ext_len <= 0 is a no-op (default), byte-identical to baseline.
int mem_chain_drop_short_seeds(mem_chain_t *c, int min_ext_len)
{
    if (min_ext_len <= 0) return c->n;
    int j, k;
    for (j = 0; j < c->n; ++j)
        if (c->seeds[j].len >= min_ext_len) break;
    if (j == c->n) return c->n;            /* no anchor: all-short chain, leave intact */
    for (j = k = 0; j < c->n; ++j)
        if (c->seeds[j].len >= min_ext_len)
            c->seeds[k++] = c->seeds[j];
    c->n = k;
    return c->n;
}

void mem_flt_chained_seeds(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                           bseq1_t *seq_, int n_chn, mem_chain_t *a)
{
    //double min_l = opt->min_chain_weight?
    // MEM_HSP_COEF * opt->min_chain_weight : MEM_MINSC_COEF * log(l_query);

    int i, j, k;// min_HSP_score = (int)(opt->a * min_l + .499);
    //if (min_l > MEM_SEEDSW_COEF * l_query) return; // don't run the following for short reads

    /* min_l/min_HSP_score depend only on (opt, l_query). Upstream hoisted them out
     * of the loop because it was called per read; here `a` holds chains for a whole
     * batch, so l_query = seq_[c->seqid].l_seq varies and a plain hoist is wrong.
     * But chains arrive grouped by seqid and libraries are usually fixed-length, so
     * a one-entry memo on l_query collapses ~all of the per-chain libm log() calls
     * (order 10 chains/read => ~1e8 log() on a 10M-read run). Byte-identical: the
     * same inputs produce the same IEEE double, we just stop recomputing it. */
    int    memo_l_query = -1;
    double memo_min_l = 0.0;
    int    memo_min_HSP_score = 0;

    /* D3 (--meth, PR-4): the chained-seed SW filter must score the ORIGINAL read
     * against the original ref with the per-chain asymmetric matrix, same as
     * extension — otherwise it over-penalizes bisulfite conversions and drops
     * borderline seeds (the PR-3 placeholder concern). meth_qbuf is the reusable
     * 2-bit encode of meth_orig_seq (same orientation as seq, per bwa.h). */
    uint8_t *meth_qbuf = NULL;
    int      meth_qbuf_cap = 0;

    for (i = 0; i < n_chn; ++i)
    {
        mem_chain_t *c = &a[i];
        const uint8_t *query = (uint8_t*) seq_[c->seqid].seq;
        int l_query = seq_[c->seqid].l_seq;

        // Skip-short-seed extension filter (opt->min_ext_len). Placed BEFORE the
        // MEM_SEEDSW_COEF screen `continue` below, which is skipped for typical
        // read lengths -- dropping seeds inside that loop would no-op on the main
        // workload. Off (min_ext_len==0): no-op, output byte-identical.
        mem_chain_drop_short_seeds(c, opt->min_ext_len);
        if (c->n == 0) continue;  /* Defensive: drop_short_seeds never empties a
                                   * non-empty chain (it leaves all-short chains
                                   * intact), but the meth block below reads
                                   * seeds[0], so guard regardless. */

        /* D3 (--meth, PR-4): swap to the original read + per-hypothesis matrix. */
        const int8_t *mat = opt->mat;
        if (opt->meth_mode && seq_[c->seqid].meth_orig_seq != NULL) {
            if (l_query > meth_qbuf_cap) {
                meth_qbuf_cap = l_query;
                /* CodeRabbit: don't assign realloc() result back to meth_qbuf
                 * directly — a NULL return would leak the old buffer. */
                uint8_t *meth_qbuf_new = (uint8_t *) realloc(meth_qbuf, (size_t) meth_qbuf_cap);
                xassert(meth_qbuf_new != NULL, "out of memory: meth_qbuf_new");
                meth_qbuf = meth_qbuf_new;
            }
            const char *os = seq_[c->seqid].meth_orig_seq;
            for (int q = 0; q < l_query; ++q) {
                unsigned char ch = (unsigned char) os[q];
                meth_qbuf[q] = nst_nt4_decode(ch, 4);
            }
            query = meth_qbuf;
            /* D3 (--meth, fix): strand-adjusted hypothesis — this chain's seeds
             * are single-strand; a reverse chain (seeds[0].rbeg >= l_pac) extends
             * against the RC reference, flipping the conversion's freed cell. */
            int hyp = c->meth_hypothesis;
            if (hyp >= 0 && c->seeds[0].rbeg >= bns->l_pac) hyp ^= 1;
            mat = mem_opt_meth_mat(opt, hyp);
        }

        double min_l;
        int min_HSP_score;
        if (l_query == memo_l_query) {          /* same read length as the previous chain */
            min_l = memo_min_l;
            min_HSP_score = memo_min_HSP_score;
        } else {
            min_l = opt->min_chain_weight?
                MEM_HSP_COEF * opt->min_chain_weight : MEM_MINSC_COEF * log(l_query);
            min_HSP_score = (int)(opt->a * min_l + .499);
            memo_l_query = l_query; memo_min_l = min_l; memo_min_HSP_score = min_HSP_score;
        }
        if (min_l > MEM_SEEDSW_COEF * l_query) continue;

        for (j = k = 0; j < c->n; ++j)
        {
            mem_seed_t *s = &c->seeds[j];
            s->score = mem_seed_sw(opt, bns, pac, l_query, query, s, mat);
            if (s->score < 0 || s->score >= min_HSP_score)
            {
                s->score = s->score < 0? s->len * opt->a : s->score;
                c->seeds[k++] = *s;
            }
        }
        c->n = k;
    }
    free(meth_qbuf);  /* NULL outside --meth; free() is NULL-safe */
}

/* --max-extend-chains: cap the number of chains that reach banded-SW extension
 * to the top `max_n` by chain weight (applied after mem_chain_flt). The dropped
 * chains are the lowest-weight secondaries; on holodeck truth (sim-wgs-place)
 * this keeps high-confidence placement accuracy essentially unchanged while
 * cutting extension work -- errors it does add land in low-MAPQ multimappers,
 * not confident calls. Opt-in speed lever (bundled into --fast); NOT
 * byte-identical (drops candidate secondaries, so XS/secondary/MAPQ can move on
 * multi-mapping reads). Always keeps >= 1 chain. Returns the new chain count. */
#define MAX_EXTEND_CHAINS_CAP 4096
int mem_chain_cap_extend(mem_chain_t *a, int n, int max_n, float tie_frac, int tie_floor,
                         int *capped_w_out)
{
    if (capped_w_out) *capped_w_out = 0;  /* max weight among dropped chains, for csub */
    /* The competitiveness gate applies even when the count cap does NOT engage
     * (n <= max_n): a read with a handful of chains still pays banded-SW for every
     * non-competitive one, and those reads are the bulk of the workload. Gating only
     * reads that exceed max_n removes almost nothing (measured: -1% to -6% of
     * extension work, vs -20% for a plain top-5 cap). With tie_frac <= 0 the original
     * early-return is preserved exactly, so the gate-off path stays byte-identical. */
    const int cap_on = max_n > 0;
    if (n > MAX_EXTEND_CHAINS_CAP) return n;
    if (tie_frac <= 0.0f && (!cap_on || n <= max_n)) return n;
    int w[MAX_EXTEND_CHAINS_CAP];
    int w_best = 0;
    for (int i = 0; i < n; i++) {
        w[i] = a[i].w > 0 ? a[i].w : mem_chain_weight(&a[i]);
        if (w[i] > w_best) w_best = w[i];
    }
    /* competitiveness gate (--extend-tie-frac): a chain ranked at/after tie_floor is
     * extended only if its weight is within tie_frac of the best chain's -- i.e. it is
     * a genuine near-tie that could change placement/MAPQ. tie_frac<=0 disables the gate,
     * reducing this to the pure top-max_n cap (baseline). tie_frac is validated to [0,1]
     * upstream (out-of-range is rejected), so (int)(tie_frac*w_best) <= w_best and the best chain (rank 0) always
     * clears the gate -- the "keep >= 1 chain" invariant holds. */
    int gate = tie_frac > 0.0f ? (int)(tie_frac * w_best) : 0;
    /* keep chain i iff fewer than max_n other chains outrank it by
     * (weight desc, then original index asc) -- a stable top-max_n selection --
     * AND (it is within the always-extend floor OR it clears the competitiveness gate). */
    int k = 0;
    for (int i = 0; i < n; i++) {
        int rank = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            if (w[j] > w[i] || (w[j] == w[i] && j < i)) rank++;
        }
        if ((!cap_on || rank < max_n) && (rank < tie_floor || w[i] >= gate)) a[k++] = a[i];
        else {
            if (capped_w_out && w[i] > *capped_w_out) *capped_w_out = w[i];
            if (a[i].m > SEEDS_PER_CHAIN) free(a[i].seeds);
        }
    }
    return k;
}

/* --extend-mate-concordant: like mem_chain_cap_extend but additionally RETAINS
 * any chain concordant with a mate chain (same contig, FR orientation, within
 * `win` bp) even if it ranks below max_n. Targets the meth --fast pairing
 * regression: the true chain is often low-weight under bisulfite (collapsed
 * alphabet) and gets capped, yet it anchors the true concordant pair, so PE
 * pairing flips both mates to a wrong concordant locus. chain.pos is in the
 * original 2*l_pac coordinate space, so bns_depos gives the forward coord.
 * `win` is the estimated proper-pair insert high bound (pes[FR].high) when
 * available, else MATE_CONCORDANT_WINDOW_FALLBACK, else a fixed CLI value;
 * matching the aligner's own concordance bound keeps only genuine pair anchors
 * (far/spurious concordant chains stay capped, bounding the extra extension).
 * The mate scan is bounded (MATE_SCAN_MAX) to keep this O(n) in practice; the
 * true anchor is virtually always among a read's higher-ranked chains. */
#define MATE_CONCORDANT_WINDOW_FALLBACK 1000  /* used only in --auto before the insert size is estimated (e.g. first chunk) */
#define MATE_SCAN_MAX 256
static int mem_chain_cap_extend_mate(mem_chain_t *a, int n, int max_n,
        const mem_chain_t *mate, int mate_n, const bntseq_t *bns, int64_t win,
        float tie_frac, int tie_floor, int *capped_w_out)
{
    if (capped_w_out) *capped_w_out = 0;  /* max weight among dropped chains, for csub */
    /* See mem_chain_cap_extend: the competitiveness gate also applies when the count
     * cap does not engage. tie_frac <= 0 preserves the original early-return exactly. */
    const int cap_on = max_n > 0;
    if (n > MAX_EXTEND_CHAINS_CAP) return n;
    if (tie_frac <= 0.0f && (!cap_on || n <= max_n)) return n;
    const int MSCAN = MATE_SCAN_MAX;
    int mn = mate_n < MSCAN ? mate_n : MSCAN;
    int64_t mfp[MSCAN]; int mrid[MSCAN], mrev[MSCAN];
    for (int j = 0; j < mn; j++) {
        int is_rev; mfp[j] = bns_depos(bns, mate[j].pos, &is_rev);
        mrid[j] = mate[j].rid; mrev[j] = is_rev;
    }
    int w[MAX_EXTEND_CHAINS_CAP];
    int w_best = 0;
    for (int i = 0; i < n; i++) {
        w[i] = a[i].w > 0 ? a[i].w : mem_chain_weight(&a[i]);
        if (w[i] > w_best) w_best = w[i];
    }
    /* competitiveness gate (--extend-tie-frac); see mem_chain_cap_extend. The
     * mate-concordant override below still runs after the gate, so a gate-failed
     * (low-weight) chain that anchors the true concordant pair is still retained --
     * exactly the #202 fix -- and the gate cannot drop a mate-concordant true pair. */
    int gate = tie_frac > 0.0f ? (int)(tie_frac * w_best) : 0;
    int k = 0;
    for (int i = 0; i < n; i++) {
        int rank = 0;
        for (int j = 0; j < n; j++)
            if (j != i && (w[j] > w[i] || (w[j] == w[i] && j < i))) rank++;
        int keep = (!cap_on || rank < max_n) && (rank < tie_floor || w[i] >= gate);
        if (!keep && mn > 0) {
            int is_rev; int64_t fp = bns_depos(bns, a[i].pos, &is_rev);
            for (int j = 0; j < mn; j++)
                if (mrid[j] == a[i].rid && mrev[j] != is_rev &&
                    /* true FR ("innie"): the forward-strand chain must sit
                     * at/upstream of the reverse-strand chain, else an RF
                     * ("outie") pair within win bp would falsely qualify. */
                    ((!is_rev && fp <= mfp[j]) || (is_rev && mfp[j] <= fp))) {
                    int64_t d = fp > mfp[j] ? fp - mfp[j] : mfp[j] - fp;
                    if (d <= win) { keep = 1; break; }
                }
        }
        if (keep) a[k++] = a[i];
        else {
            /* --extend-csub: remember the heaviest chain we dropped, so the SAM stage
             * can seed the pruned competitor back into MAPQ (mem_seed_capped_sub). */
            if (capped_w_out && w[i] > *capped_w_out) *capped_w_out = w[i];
            if (a[i].m > SEEDS_PER_CHAIN) free(a[i].seeds);
        }
    }
    return k;
}

/* ---- mem_chain_flt greedy pairwise overlap pass (CHN-FLT-2) ----------------
 * The pass compares each weight-sorted chain i against the already-kept chains
 * and decides which chains survive. Two implementations share one signature so
 * a debug build can run both and assert they agree byte-for-byte:
 *   - chn_flt_pairwise_ref : the historical O(n^2) naive scan (reference).
 *   - chn_flt_pairwise_fast: the BI-1 skip that visits only effect-bearing
 *     comparisons (the case analysis is in the block comment above it).
 *
 * Both read the sorted chain array `a` (weight-descending) and the SoA-hoisted
 * query coordinates cb/ce, and write their results into caller-owned arrays:
 *   first[t]  first shadowing chain of sorted-chain t, or -1 (init by caller)
 *   kept[t]   0/2/3 kept-class of sorted-chain t (init 0 by caller)
 *   *surv     survivor list, in scan (weight-descending) order
 * Nothing in `a` is mutated, so the two runs are independent. All indices are
 * into the post-ks_introsort array `a`.
 *
 * The reference is compiled only for the opt-in cross-check build (see the
 * BWA_MEM3_DEBUG_CHN_FLT_XCHECK note at the call site); production builds
 * contain the fast pass alone. */
typedef kvec_t(int) kvi_t;

#ifdef BWA_MEM3_DEBUG_CHN_FLT_XCHECK
static void chn_flt_pairwise_ref(int n_chn, const mem_chain_t *a,
    const int *cb, const int *ce,
    float mask_level, float drop_ratio, int max_chain_gap, int min_seed_gap,
    int *first, int *kept, kvi_t *surv)
{
    int i, k;
    kept[0] = 3;
    kv_push(int, *surv, 0);
    for (i = 1; i < n_chn; ++i)
    {
        int large_ovlp = 0;
        const int cbi = cb[i], cei = ce[i];
        const int li = cei - cbi;
        const int wi = a[i].w;
        const int ai_alt = a[i].is_alt;
        for (k = 0; k < (int)surv->n; ++k)
        {
            int j = surv->a[k];
            int b_max = cb[j] > cbi? cb[j] : cbi;
            int e_min = ce[j] < cei? ce[j] : cei;
            if (e_min > b_max && (!a[j].is_alt || ai_alt)) { // have overlap; don't consider ovlp where the kept chain is ALT while the current chain is primary
                int lj = ce[j] - cb[j];
                int min_l = li < lj? li : lj;
                if (e_min - b_max >= min_l * mask_level && min_l < max_chain_gap) { // significant overlap
                    large_ovlp = 1;
                    if (first[j] < 0) first[j] = i; // keep the first shadowed hit s.t. mapq can be more accurate
                    if (wi < a[j].w * drop_ratio && a[j].w - wi >= min_seed_gap)
                        break;
                }
            }
        }
        if (k == (int)surv->n)
        {
            kv_push(int, *surv, i);
            kept[i] = large_ovlp? 2 : 3;
        }
    }
}
#endif /* BWA_MEM3_DEBUG_CHN_FLT_XCHECK */

/* BI-1 skip: the naive pass wastes the vast majority of its comparisons on kept
 * chains j that can have no observable effect on i -- j already has a first
 * shadower, is too light to dominate i, and is reached after i's large_ovlp latch
 * has fired. This visits only the effect-bearing comparisons and is byte-identical
 * to chn_flt_pairwise_ref (proven by case analysis; asserted per-read in debug).
 *
 * Per i, the three observable effects of a comparison are localized:
 *   (C) large_ovlp latch: fires only on i's FIRST significant overlap; found by a
 *       linear scan (phase 1) that stops there -- exactly what the ref does before
 *       any skipping could begin.
 *   (B) domination break: needs `wi < wj*drop_ratio && wj-wi >= gap`, a pure weight
 *       test. Since `surv` is weight-descending, any chain reached after i's first
 *       significant overlap is no heavier than that one, so if that first overlap did
 *       not dominate i, nothing later can either: the break can only fire at the first
 *       significant overlap (handled in phase 1) or never. i then survives.
 *   (A) first[j]=i: only for j with first<0 that i significantly overlaps. Tracked in
 *       an intrusive doubly-linked list of surv positions whose chain still has
 *       first<0 (phase 2); the list only shrinks (a chain leaves once shadowed) and
 *       grows by one per newly kept chain. */
static void chn_flt_pairwise_fast(int n_chn, const mem_chain_t *a,
    const int *cb, const int *ce,
    float mask_level, float drop_ratio, int max_chain_gap, int min_seed_gap,
    int *first, int *kept, kvi_t *surv)
{
    /* Intrusive DLL over surv positions (0..surv->n-1) with first<0. A position is
     * appended once (when its chain is kept) and removed once (when its chain gets
     * a first shadower), so each entry is live for one contiguous span and phase 2
     * visits every still-unshadowed chain exactly once.
     * Grow-only thread_local scratch; only touched entries are initialized. */
    static thread_local std::vector<int> nxt_v, prv_v;
    if ((int)nxt_v.size() < n_chn) { nxt_v.resize(n_chn); prv_v.resize(n_chn); }
    int *nxt = nxt_v.data(), *prv = prv_v.data();
    int f0_head, f0_tail;

    kept[0] = 3;
    kv_push(int, *surv, 0);
    nxt[0] = -1; prv[0] = -1; f0_head = 0; f0_tail = 0;   // f0 = { position 0 }

    for (int i = 1; i < n_chn; ++i)
    {
        const int cbi = cb[i], cei = ce[i];
        const int li = cei - cbi;
        const int wi = a[i].w;
        const int ai_alt = a[i].is_alt;

        /* significant overlap of i with chain j (identical test to the ref). */
        auto sig = [&](int j) -> bool {
            int b_max = cb[j] > cbi? cb[j] : cbi;
            int e_min = ce[j] < cei? ce[j] : cei;
            if (!(e_min > b_max && (!a[j].is_alt || ai_alt))) return false;
            int lj = ce[j] - cb[j];
            int min_l = li < lj? li : lj;
            return (e_min - b_max >= min_l * mask_level && min_l < max_chain_gap);
        };
        /* j is heavy enough to dominate i (pure weight test; overlap-independent). */
        auto wdom = [&](int j) -> bool {
            return (wi < a[j].w * drop_ratio && a[j].w - wi >= min_seed_gap);
        };
        auto f0_remove = [&](int p) {
            int pv = prv[p], nx = nxt[p];
            if (pv >= 0) nxt[pv] = nx; else f0_head = nx;
            if (nx >= 0) prv[nx] = pv; else f0_tail = pv;
        };
        auto f0_append = [&](int p) {
            nxt[p] = -1; prv[p] = f0_tail;
            if (f0_tail >= 0) nxt[f0_tail] = p; else f0_head = p;
            f0_tail = p;
        };

        int large_ovlp = 0, dominated = 0, k0 = -1, k;
        const int nsurv = (int)surv->n;

        /* Phase 1: linear scan to i's first significant overlap (latches large_ovlp).
         * Nothing is skippable before the latch, so this mirrors the ref exactly. */
        for (k = 0; k < nsurv; ++k) {
            int j = surv->a[k];
            if (sig(j)) {
                large_ovlp = 1;
                if (first[j] < 0) { first[j] = i; f0_remove(k); }
                if (wdom(j)) dominated = 1;   // first sig overlap is a dominator -> break here
                k0 = k;
                break;
            }
        }

        if (large_ovlp == 0) {                // no significant overlap: keep, kept=3
            int p = surv->n;
            kv_push(int, *surv, i);
            kept[i] = 3; first[i] = -1;
            f0_append(p);
            continue;
        }
        if (dominated) continue;              // dominated at k0: i dropped, ref broke here too

        /* i survives (kept=2): no chain after k0 can dominate it (see (B) above), so
         * the ref would scan on without breaking. Phase 2: record i as the first
         * shadower of every still-unshadowed chain it significantly overlaps at a
         * position after k0. Positions <= k0 either did not overlap (k0 is the first
         * overlap) or were handled in phase 1. */
        for (int p = f0_head; p >= 0; ) {
            int nx = nxt[p];
            if (p > k0) {
                int j = surv->a[p];
                if (sig(j)) { first[j] = i; f0_remove(p); }
            }
            p = nx;
        }

        int p = surv->n;                      // survives with a large overlap: kept=2
        kv_push(int, *surv, i);
        kept[i] = 2; first[i] = -1;
        f0_append(p);
    }
}

int mem_chain_flt(const mem_opt_t *opt, int n_chn_, mem_chain_t *a_, int tid)
{
    int i, k;
    if (n_chn_ == 0) return 0; // no need to filter
    // compute the weight of each chain and drop chains with small weight
    /* CHN-17: with min_chain_weight <= 0 no chain can ever be dropped -- c->w is
     * an unsigned 27-bit bitfield, so it is always >= 0 and `c->w < min_chain_weight`
     * is never true. The general loop below then degenerates to `a_[k++] = *c` with
     * k == i throughout: a per-chain self-copy of a 48-byte mem_chain_t. Take a fast
     * path that updates the gate fields in place and skips the copy. Byte-identical:
     * the surviving array (all chains), their order, and every field value are
     * unchanged, and nothing is freed because nothing is dropped. min_chain_weight
     * defaults to 0, so this is the dominant case. */
    if (opt->min_chain_weight <= 0)
    {
        for (i = 0; i < n_chn_; ++i)
        {
            mem_chain_t *c = &a_[i];
            c->first = -1; c->kept = 0;
            c->w = mem_chain_weight(c);
        }
        /* No chain can be dropped, so there is no compaction and n_chn_ is
         * unchanged; the general path's `k` would have ended at n_chn_. */
    }
    else
    {
        /* Slot 0's seeds are freed LATE *when the target resurrects slot 0*. If
         * every chain is dropped the array is never compacted, so a_[0] still
         * holds chain 0, and the empty-array path below hands exactly that slot
         * back to the caller with kept = 3 (bwa-mem2 behaviour; see the CHN-6/19
         * note). bwa-mem2 frees those seeds here and then returns the chain
         * pointing at the freed block, so what the caller extends depends on
         * whether the allocator has reused it. Deferring this one free yields the
         * same chain with its own seeds still intact -- the result bwa-mem2
         * produces whenever the block is untouched -- without the use-after-free.
         * When at least one chain survives, slot 0 was either kept or overwritten
         * by compaction, and the seeds are freed immediately after the loop.
         *
         * Under a target that does NOT resurrect (--compat=bwa-mem), nothing
         * reads slot 0 again, so the deferral has nothing to protect and holding
         * the pointer past the early return below would simply leak it. Free it
         * in the loop, exactly as bwa does. */
        const int resurrect_empty =
            (opt->compat == NULL) || opt->compat->chain_flt_resurrect_empty;
        mem_seed_t *slot0_seeds = NULL;
        for (i = k = 0; i < n_chn_; ++i)
        {
            mem_chain_t *c = &a_[i];
            c->first = -1; c->kept = 0;
            c->w = mem_chain_weight(c);
            if (c->w < opt->min_chain_weight)
            {
                if (c->m > SEEDS_PER_CHAIN)  // CHN-16: dead tprof[PE11] counter removed
                {
                    if (i == 0 && resurrect_empty) slot0_seeds = c->seeds;  // may be resurrected below
                    else free(c->seeds);
                }
            }
            else a_[k++] = *c;
        }
        // Not resurrected -- drop it now. free(NULL) when chain 0 was kept.
        if (k != 0) free(slot0_seeds);
        n_chn_ = k;

        /* #310: the weight filter emptied the array. bwa reports that honestly
         * (its tail loops are bounded by n_chn == 0, so it returns 0 and the read
         * goes out unmapped); bwa-mem2's seqid-range machinery instead rebuilds a
         * count of 1 and extends the chain it just rejected. Returning here is
         * bwa's answer -- selected per target rather than unconditionally,
         * because a `bwa-mem2` or `off` run that stopped resurrecting would no
         * longer be the drop-in it advertises. */
        if (n_chn_ == 0 && !resurrect_empty) return 0;
    }

    /* Fast path for the dominant single-chain (unique-mapper) case. With one
     * surviving chain the pairwise-overlap pass reduces to `a_[0].kept = 3` (set
     * unconditionally, kept through compaction) returning 1 — but the general path
     * still builds a kvec and calls ks_introsort to get there. a_[0].first was
     * already reset to -1 in the weight-filter loop above. (Only n_chn_ == 1 is
     * short-circuited; n_chn_ == 0 falls through to preserve the existing
     * general-path behavior.) */
    if (n_chn_ == 1) {
        a_[0].kept = 3;
        return 1;
    }

    /* CHN-6/19: mem_chain_flt runs once per read (chn = &chain_ar[l]); every chain
     * chain_add_one_seed emits for that read is stamped seqid = l, so all chains
     * reaching here share a single seqid and the removed seqid-run scan always
     * produced exactly one range [0, n_chn_) -- the `c->seqid != pseqid` test could
     * never fire. Run the former per-range body once over the whole array.
     * Byte-identical: with one range `ilag` stays 0, so `a[k++ - ilag] == a[k++]`,
     * and the return value equals `k` (the old `n_numc`). The post-weight-filter
     * n_chn_ == 0 case is preserved exactly: the old scan built the range {0,1} and
     * ran the body over one uncompacted slot, so n_chn falls back to 1 and slot 0
     * is returned with kept = 3. Its seeds are still live -- see the deferred free
     * in the weight filter above. */
    mem_chain_t *a = a_;
    int n_chn = n_chn_ ? n_chn_ : 1;
    kvi_t chains = {0,0,0};
    kv_resize(int, chains, n_chn);   // CHN-19: reserve up front (kept count <= n_chn)
    ks_introsort(mem_flt, n_chn, a);

    /* SoA hoist (#265, byte-identical). chn_beg()/chn_end() each chase a[].seeds --
     * a dependent load of the seeds pointer, then a load from the 65k-entry seedBuf
     * arena -- and the pairwise loop below evaluates them for EVERY (i,j) pair, so
     * the j-side is re-chased on every inner iteration. Cache them once into two
     * contiguous int arrays (typical ~25 chains/read, so a couple hundred bytes,
     * L1-resident). The seed arrays are not modified anywhere in this loop, so
     * these are exactly the values the macros would have recomputed. thread_local
     * + grow-only: no per-read allocation. */
    static thread_local std::vector<int> cb_v, ce_v;
    int *cb = NULL, *ce = NULL;
    /* Skipped for n_chn < 2. cb/ce are read only by the `i >= 1` pairwise loop
     * below, so with a single chain they are dead -- and the one way to reach
     * here with n_chn == 1 is the resurrected-empty-array path, where bwa-mem2
     * evaluates chn_beg()/chn_end() inline inside that same never-entered loop
     * and so never touches a_[0].seeds at all. Hoisting unconditionally would
     * add a seeds dereference bwa-mem2 does not perform. */
    if (n_chn > 1) {
        if ((int)cb_v.size() < n_chn) { cb_v.resize(n_chn); ce_v.resize(n_chn); }
        cb = cb_v.data(); ce = ce_v.data();
        for (int t = 0; t < n_chn; ++t) { cb[t] = chn_beg(a[t]); ce[t] = chn_end(a[t]); }
    }

    // pairwise chain comparisons
    /* CL-4: hoist the filter constants read every iteration of the nested pass. */
    const float mask_level    = opt->mask_level;
    const float drop_ratio    = opt->drop_ratio;
    const int   max_chain_gap = opt->max_chain_gap;
    const int   min_seed_gap  = opt->min_seed_len << 1;

    /* Run the pairwise pass into caller-owned first[]/kept[] scratch, then write
     * the results back into a[]. Keeping the pass out of a[] lets a debug build
     * cross-check the fast skip against the naive reference on identical inputs.
     * Grow-only thread_local scratch: no per-read allocation. */
    static thread_local std::vector<int> first_v, kept_v;
    if ((int)first_v.size() < n_chn) { first_v.resize(n_chn); kept_v.resize(n_chn); }
    int *first = first_v.data(), *kept = kept_v.data();
    for (int t = 0; t < n_chn; ++t) { first[t] = -1; kept[t] = 0; }

    chn_flt_pairwise_fast(n_chn, a, cb, ce,
                          mask_level, drop_ratio, max_chain_gap, min_seed_gap,
                          first, kept, &chains);

    /* Adversarial cross-check: run the naive reference into separate scratch and
     * assert the fast skip produced identical survivors, first[] and kept[] for
     * this read.
     *
     * Opt-in via BWA_MEM3_DEBUG_CHN_FLT_XCHECK rather than `#ifndef NDEBUG`: no
     * target in this build system ever defines NDEBUG, so an NDEBUG gate would
     * leave the O(n^2) reference -- precisely the work this pass exists to skip --
     * running on every read of every shipped binary. That is the same trap already
     * recorded for the pac-fetch poison in bntseq.cpp. Enable the cross-check with
     *   make EXTRA_CXXFLAGS=-DBWA_MEM3_DEBUG_CHN_FLT_XCHECK
     * Byte-identical: production never defines the macro, so nothing here runs. */
#ifdef BWA_MEM3_DEBUG_CHN_FLT_XCHECK
    {
        static thread_local std::vector<int> rf_v, rk_v;
        if ((int)rf_v.size() < n_chn) { rf_v.resize(n_chn); rk_v.resize(n_chn); }
        int *rfirst = rf_v.data(), *rkept = rk_v.data();
        for (int t = 0; t < n_chn; ++t) { rfirst[t] = -1; rkept[t] = 0; }
        /* Reference survivor list. Grow-only thread_local like rf_v/rk_v above, so
         * the cross-check adds no per-read malloc/free on top of the one the pass
         * itself already does. The RAII holder mirrors u8vec_scratch_t: kv_destroy
         * runs at thread exit, so a sanitizer debug build sees no leak (a bare
         * thread_local kvi_t would never be freed). kv_resize reallocs
         * unconditionally, hence the capacity guard. */
        struct kvi_scratch_t {
            kvi_t v;
            kvi_scratch_t() { kv_init(v); }
            ~kvi_scratch_t() { kv_destroy(v); }
        };
        static thread_local kvi_scratch_t rs;
        rs.v.n = 0;
        if (rs.v.m < (size_t)n_chn) kv_resize(int, rs.v, n_chn);
        chn_flt_pairwise_ref(n_chn, a, cb, ce,
                             mask_level, drop_ratio, max_chain_gap, min_seed_gap,
                             rfirst, rkept, &rs.v);
        /* xassert, not assert: defining this macro alongside -DNDEBUG would
         * otherwise compile the quadratic reference into the hot path while
         * stripping every check -- full cost, zero verification, which is the
         * one configuration in which enabling the cross-check is strictly
         * worse than not. xassert survives NDEBUG (utils.h), so the check is
         * live whenever the reference is compiled. */
        xassert(rs.v.n == chains.n, "BI-1: survivor count differs from reference");
        for (int t = 0; t < (int)chains.n; ++t)
            xassert(rs.v.a[t] == chains.a[t], "BI-1: survivor set/order differs");
        for (int t = 0; t < n_chn; ++t) {
            xassert(rfirst[t] == first[t], "BI-1: first[] differs from reference");
            xassert(rkept[t] == kept[t], "BI-1: kept[] differs from reference");
        }
    }
#endif /* BWA_MEM3_DEBUG_CHN_FLT_XCHECK */

    for (int t = 0; t < n_chn; ++t) { a[t].first = first[t]; a[t].kept = kept[t]; }

    for (i = 0; i < chains.n; ++i)
    {
        mem_chain_t *c = &a[chains.a[i]];
        if (c->first >= 0) a[c->first].kept = 1;
    }
    free(chains.a);
    for (i = k = 0; i < n_chn; ++i) { // don't extend more than opt->max_chain_extend .kept=1/2 chains
        if (a[i].kept == 0 || a[i].kept == 3) continue;
        if (++k >= opt->max_chain_extend) break;
    }
    /* CL-5: the former standalone demotion loop (demote kept<3 chains from the
     * extend-cap cutoff onward to 0) is fused into the free+compact pass below,
     * one scan instead of two. `cap_cutoff` is the index the cap-count loop above
     * stopped at (n_chn if the cap was never reached); chains at or beyond it with
     * kept<3 are demoted inline as we compact. Byte-identical to the two-loop form. */
    const int cap_cutoff = i;

    for (i = k = 0; i < n_chn; ++i)  // demote-beyond-cap + free discarded + compact
    {
        mem_chain_t *c = &a[i];
        if (i >= cap_cutoff && c->kept < 3) c->kept = 0;   // CL-5: former step-10 demotion
        if (c->kept == 0)
        {
            if (c->m > SEEDS_PER_CHAIN)  // CHN-16: dead tprof[PE11] counter removed
                free(c->seeds);
        }
        else a[k++] = a[i];
    }

    return k;
}

SMEM *mem_collect_smem(FMI_search *fmi, const mem_opt_t *opt,
                       const bseq1_t *seq_,
                       int nseq,
                       SMEM *matchArray,
                       int32_t *min_intv_ar,
                       int32_t *query_pos_ar,
                       uint8_t *enc_qdb,
                       int32_t *rid,
                       mem_cache *mmc,
                       int64_t &tot_smem,
                       int tid)
{
    int64_t pos = 0;
    int split_len = (int)(opt->min_seed_len * opt->split_factor + .499);
    int64_t num_smem1 = 0, num_smem2 = 0, num_smem3 = 0;
    int max_readlength = -1;

    int32_t *query_cum_len_ar = (int32_t *)_mm_malloc(nseq * sizeof(int32_t), 64);

    int offset = 0;
    for (int l=0; l<nseq; l++)
    {
        min_intv_ar[l] = 1;
        memcpy(enc_qdb + offset, seq_[l].seq, (size_t)seq_[l].l_seq);
        offset += seq_[l].l_seq;
        rid[l] = l;
    }

    max_readlength = seq_[0].l_seq;
    query_cum_len_ar[0] = 0;
    for(int i = 1; i < nseq; i++) {
        query_cum_len_ar[i] = query_cum_len_ar[i - 1] + seq_[i-1].l_seq;
        if (max_readlength < seq_[i].l_seq)
            max_readlength = seq_[i].l_seq;
    }

    fmi->getSMEMsAllPosOneThread(enc_qdb, min_intv_ar, rid, nseq, nseq,
                                 seq_, query_cum_len_ar, max_readlength, opt->min_seed_len,
                                 matchArray, &num_smem1);

    for (int64_t i=0; i<num_smem1; i++)
    {
        SMEM *p = &matchArray[i];
        int start = p->m, end = p->n +1;
        if (end - start < split_len || p->s > opt->split_width)
            continue;

        int len = seq_[p->rid].l_seq;

        rid[pos] = p->rid;

        query_pos_ar[pos] = (end + start)>>1;

        assert(query_pos_ar[pos] < len);

        min_intv_ar[pos] = p->s + 1;
        pos ++;
    }

    #if 1
    // Pre-walk grow: getSMEMsOnePosOneThread emits at most readlength SMEMs
    // per re-walk position (the scalar walker's worst case is one SMEM per
    // backward step plus the final prev[0]), so the safe upper bound on
    // num_smem2 is pos * max_readlength.
    const int64_t safe_smem2_cap = (int64_t)pos * max_readlength;
    if (mmc->wsize_mem[tid] < num_smem1 + safe_smem2_cap) {
        int64_t tmp = mmc->wsize_mem[tid];
        mmc->wsize_mem[tid] = num_smem1 + safe_smem2_cap;

        SMEM* ptr1 = mmc->matchArray[tid];
        // ptr2 = w.mmc.min_intv_ar;
        // ptr3 =
        if (bwa_verbose >= 4) {
            fprintf(stderr, "[%0.4d] REA Re-allocating SMEM after num_smem1: "
                    "%" PRId64 " -> %" PRId64 "\n",
                    tid, tmp, mmc->wsize_mem[tid]);
        }
        mmc->matchArray[tid]    = (SMEM *) _mm_malloc(mmc->wsize_mem[tid] * sizeof(SMEM), 64);
        xassert(mmc->matchArray[tid] != NULL, "out of memory: mmc->matchArray[tid]");
        matchArray = mmc->matchArray[tid];
        // w.mmc.min_intv_ar[l]   = (int32_t *) malloc(w.mmc.wsize_mem[l] * sizeof(int32_t));
        // w.mmc.query_pos_ar[tid]  = (int16_t *) malloc(w.mmc.wsize_mem[l] * sizeof(int16_t));
        // w.mmc.enc_qdb[l]       = (uint8_t *) malloc(w.mmc.wsize_mem[l] * sizeof(uint8_t));
        // w.mmc.rid[l]           = (int32_t *) malloc(w.mmc.wsize_mem[l] * sizeof(int32_t));
        //w.mmc.lim[l]         = (int32_t *) _mm_malloc((BATCH_SIZE + 32) * sizeof(int32_t), 64);
        // SMEM is trivially copyable POD; ptr1 (old buffer) and the freshly
        // _mm_malloc'd destination are distinct, non-overlapping allocations, so
        // a bulk memcpy of num_smem1 records is byte-identical to the per-element
        // copy it replaces.
        memcpy(mmc->matchArray[tid], ptr1, (size_t)num_smem1 * sizeof(SMEM));
        _mm_free(ptr1);
    }
    #endif
    
#if SMEM_LOCKSTEP_N > 1
    fmi->getSMEMsOnePosOneThread_lockstep(enc_qdb,
                                          query_pos_ar,
                                          min_intv_ar,
                                          rid,
                                          pos,
                                          pos,
                                          seq_,
                                          query_cum_len_ar,
                                          max_readlength,
                                          opt->min_seed_len,
                                          matchArray + num_smem1,
                                          &num_smem2);
#else
    fmi->getSMEMsOnePosOneThread(enc_qdb,
                                 query_pos_ar,
                                 min_intv_ar,
                                 rid,
                                 pos,
                                 pos,
                                 seq_,
                                 query_cum_len_ar,
                                 max_readlength,
                                 opt->min_seed_len,
                                 matchArray + num_smem1,
                                 &num_smem2);
#endif
    // assert(mmc->wsize_mem[tid] > (num_smem1 + num_smem2));
    // fprintwsize_mem_rf(stderr, "num_smem2: %d\n", num_smem2);
    if (opt->max_mem_intv > 0)
    {
        #if 1
		// Pre-walk grow for the third pass. bwtSeedStrategyAllPosOneThread's
		// documented worst case (FMI_search.h) is nseq * max_seq_length SMEMs
		// — one per query position per read — which dwarfs the prior
		// offset/(min_seed_len+1) heuristic. Use the safe bound so the
		// caller cannot overflow matchArray when this pass writes at
		// (num_smem1 + num_smem2).
		const int64_t safe_smem3_cap = (int64_t)nseq * max_readlength;
		const int64_t mem_intv_cap = num_smem1 + num_smem2 + safe_smem3_cap;
		if (mmc->wsize_mem[tid] < mem_intv_cap) {
            int64_t tmp = mmc->wsize_mem[tid];
		    mmc->wsize_mem[tid] = mem_intv_cap;
		    SMEM* ptr1 = mmc->matchArray[tid];
            if (bwa_verbose >= 4) {
                fprintf(stderr, "[%0.4d] REA Re-allocating SMEM after num_smem2: "
                        "%" PRId64 " -> %" PRId64 "\n",
                        tid, tmp, mmc->wsize_mem[tid]);
            }
		    mmc->matchArray[tid]    = (SMEM *) _mm_malloc(mmc->wsize_mem[tid] * sizeof(SMEM), 64);
		    xassert(mmc->matchArray[tid] != NULL, "out of memory: mmc->matchArray[tid]");
		    matchArray = mmc->matchArray[tid];
		    // First pass wrote num_smem1 records at offset 0; the second pass
		    // (getSMEMsOnePosOneThread*) just wrote num_smem2 records at offset
		    // num_smem1. Preserve both — bwtSeedStrategyAllPosOneThread() below
		    // appends at offset num_smem1 + num_smem2.
		    int64_t already_written = (int64_t)num_smem1 + num_smem2;
		    // Bulk-copy the already-written prefix: SMEM is trivially copyable
		    // POD and src/dst are distinct non-overlapping allocations, so this
		    // memcpy is byte-identical to the per-element loop it replaces.
		    memcpy(mmc->matchArray[tid], ptr1, (size_t)already_written * sizeof(SMEM));
		    _mm_free(ptr1);
		}
        #endif
        for (int l=0; l<nseq; l++)
            min_intv_ar[l] = opt->max_mem_intv;

// Third-pass re-seeding lockstep is gated to arm64. It overlaps the serial
// forward-extension chain's cp_occ misses with N independent reads' chains,
// which is a large seeding win (~-16%, ~-1.7% end-to-end) on non-SMT arm
// (Graviton4, Apple Silicon). On SMT x86 the sibling hyperthread already hides
// that latency, so the lockstep is redundant overhead and regresses ~+2% at
// full-vCPU thread counts — measured across BWTSEED_LOCKSTEP_N in {4,8,16},
// all depths regress x86-SMT. Byte-identical either way (see bwtseed lockstep
// parity harness); the scalar path stays the x86 default.
#if defined(__aarch64__) && BWTSEED_LOCKSTEP_N > 1
        num_smem3 = fmi->bwtSeedStrategyAllPosOneThread_lockstep(enc_qdb, min_intv_ar,
                                                                 nseq, seq_, query_cum_len_ar,
                                                                 opt->min_seed_len + 1,
                                                                 matchArray + num_smem1 + num_smem2,
                                                                 max_readlength);
#else
        num_smem3 = fmi->bwtSeedStrategyAllPosOneThread(enc_qdb, min_intv_ar,
                                                        nseq, seq_, query_cum_len_ar,
                                                        opt->min_seed_len + 1,
                                                        matchArray + num_smem1 + num_smem2);
#endif
    }
    tot_smem = num_smem1 + num_smem2 + num_smem3;
    // assert(mmc->wsize_mem[tid] > (tot_smem));
    // fprintf(stderr, "num_smems: %d %d %d, %d\n", num_smem1, num_smem2, num_smem3, tot_smem);
    fmi->sortSMEMs(matchArray, &tot_smem, nseq, seq_[0].l_seq, 1, mmc->smem_sort_scratch[tid]); // seq_[0].l_seq - only used for blocking when using nthreads

    pos = 0;
    int64_t smem_ptr = 0;
    for (int l=0; l<nseq && pos < tot_smem - 1; l++) {
        pos = smem_ptr - 1;
        do {
            pos++;
        } while (pos < tot_smem - 1 && matchArray[pos].rid == matchArray[pos + 1].rid);
        int64_t n = pos + 1 - smem_ptr;

        if (n > 0)
            ks_introsort(mem_intv1, n, &matchArray[smem_ptr]);
        smem_ptr = pos + 1;
    }

    if (opt->smem_dedup)
        tot_smem = smem_dedup_inplace(matchArray, tot_smem);

    _mm_free(query_cum_len_ar);

    return matchArray;
}

/* D3 (--meth, PR-3): remap one seed hit from SEED-index (f/r-doubled) coordinates
 * to ORIGINAL-reference coordinates + a bisulfite hypothesis label.
 *
 * The seed reference has 2N contigs in r0,f0,r1,f1,... order (rchrK = G→A / OB,
 * fchrK = C→T / OT); both are FORWARD projections of the original contig K, so
 * the seed-local position equals the original-local position. The seed FM-index
 * additionally doubles its own pac (forward + reverse-complement), so a seed hit
 * `seed_rbeg` lives in [0, 2*L_seed) where >= L_seed means the hit is on the RC
 * of the seed reference. That RC strand is the genomic alignment strand and is
 * INDEPENDENT of the OT/OB hypothesis (which is the f/r contig parity).
 *
 * Inputs:
 *   seed_bns   — the seed (f/r-doubled) BNS, used to decode seed_rbeg.
 *   orig_bns   — the original BNS (N contigs), the remap target.
 *   seed_rbeg  — seed-index doubled-pac coordinate of the seed start.
 *   seed_len   — seed length (>0); used to project the start across strands.
 * Outputs (only written on success):
 *   *orig_rbeg   — start coordinate in ORIGINAL doubled-pac space [0, 2*L_orig).
 *   *orig_rid    — original contig id.
 *   *hypothesis  — 1 = OT (odd "f" seed contig), 0 = OB (even "r" seed contig).
 * Returns 0 on success, -1 if the hit cannot be remapped (bridging / out of
 * range), in which case the caller should drop the seed.
 *
 * NOTE: the reverse-strand re-encoding below is the single most placement-
 * critical step in --meth (spec §5.2 / plan §5: "strand bookkeeping can silently
 * corrupt placement"). It is exercised end-to-end and covered by the seed→
 * original remap + OT/OB PE/strand placement regression tests
 * (test/regression/meth_seed_index.sh and the live mem --meth placement tests).
 */
/* --meth seed-chemistry filter (default ON; BWAMEM3_METH_SEED_FILTER=0 disables).
 * A directional read is projected under a single chemistry by read number
 * (R1=OT/C->T, R2=OB/G->A). Its true seeds therefore only land on converted
 * copies whose implied chemistry matches: a seed on copy parity P at genomic
 * strand S implies chemistry (P XOR S), and the true locus always satisfies
 * (P XOR S) == read#. A seed that disagrees is a paralog/low-complexity
 * coincidence on the wrong converted copy; dropping it stops it from seeding a
 * confident wrong-chromosome placement (the #202 tail). */
static inline bool meth_seed_filter_enabled()
{
    static const bool on = []() {
        const char *e = getenv("BWAMEM3_METH_SEED_FILTER");
        return !(e != NULL && strcmp(e, "0") == 0);
    }();
    return on;
}

/* --meth-seed-prune (mem_opt_t.meth_seed_prune, enum mem_meth_seed_prune).
 * Under --meth the 3-letter alphabet over-seeds with short, high-multiplicity
 * spurious SMEMs while the true placement keeps a long/unique seed. Pruning
 * those before SA resolution removes ~2/3 of meth seed volume at ~0 truth-based
 * accuracy cost (validated on WGS/WES/twist-emseq). ON by default under --meth
 * (SPEC30; see fastmap.cpp), overridable by BWAMEM3_METH_SEED_PRUNE for A/B:
 *   OFF      -> no prune (byte-identical to the pre-prune seeding)
 *   BASELINE -> drop SMEM iff (len < L AND SA-count > Kb)   [L=25,Kb=1]
 *   SPEC30   -> per read, M = longest SMEM length; regime A (M < T) keeps all;
 *               regime B (M >= T) keeps iff (len >= L) OR
 *               (SA-count == 1 AND len >= U).               [T=30,L=25,U=22]
 * The read's longest SMEM is always kept (never empty a read's rid-group).
 * Thresholds overridable via BWAMEM3_METH_PRUNE_{T,L,U,KB} (integer 1..4096;
 * a malformed value warns to stderr and keeps the default). */
static int meth_prune_env_int(const char *name, int def)
{
    const char *e = getenv(name);
    if (e == NULL || *e == '\0') return def;
    char *end = NULL;
    errno = 0;
    long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || v < 1 || v > 4096) {
        fprintf(stderr, "[W::meth-seed-prune] ignoring malformed %s='%s' "
                        "(want integer 1..4096); using %d\n", name, e, def);
        return def;
    }
    return (int)v;
}
/* Cached BWAMEM3_METH_SEED_PRUNE mode override (for A/B against a shipped
 * default): -1 = no override, else a MEM_METH_PRUNE_* value. A C++11 magic
 * static so the getenv/parse/warn runs exactly once and thread-safely -- the
 * worker threads that call meth_seed_prune_mode concurrently from
 * mem_chain_seeds neither re-read the environment on the hot path nor race a
 * warn-once flag. "0" is accepted as off, matching the sibling env knobs
 * (BWAMEM3_METH_SEED_FILTER=0, BWA3_SA_PER_READ=0). */
static inline int meth_prune_env_mode_override()
{
    static const int ov = []() -> int {
        const char *m = getenv("BWAMEM3_METH_SEED_PRUNE");
        if (m == NULL || *m == '\0') return -1;
        if (strcmp(m, "spec30") == 0)                     return MEM_METH_PRUNE_SPEC30;
        if (strcmp(m, "baseline") == 0)                   return MEM_METH_PRUNE_BASELINE;
        if (strcmp(m, "off") == 0 || strcmp(m, "0") == 0) return MEM_METH_PRUNE_OFF;
        fprintf(stderr, "[W::meth-seed-prune] ignoring BWAMEM3_METH_SEED_PRUNE='%s' "
                        "(want off|spec30|baseline); using the --meth-seed-prune setting\n", m);
        return -1;
    }();
    return ov;
}
static inline int meth_seed_prune_mode(const mem_opt_t *opt, int *T, int *L, int *U, int *Kb)
{
    /* Thresholds: compile-time defaults, env-overridable (validated) for A/B.
     * Each is a C++11 function-local static (a "magic static"): its initializer
     * runs exactly once and thread-safely, so the worker threads that call this
     * concurrently from mem_chain_seeds cannot race the cache -- unlike the
     * hand-rolled `if (!tinit)` flag this replaces, which was a data race on
     * non-atomic statics (matches meth_seed_filter_enabled's IIFE pattern). */
    static const int t  = meth_prune_env_int("BWAMEM3_METH_PRUNE_T",  30);
    static const int l  = meth_prune_env_int("BWAMEM3_METH_PRUNE_L",  25);
    static const int u  = meth_prune_env_int("BWAMEM3_METH_PRUNE_U",  22);
    static const int kb = meth_prune_env_int("BWAMEM3_METH_PRUNE_KB",  1);
    *T = t; *L = l; *U = u; *Kb = kb;
    const int ov = meth_prune_env_mode_override();   /* cached; env wins over the opt field */
    if (ov >= 0) return ov;
    return opt ? opt->meth_seed_prune : MEM_METH_PRUNE_OFF;
}

/* BWAMEM3_METH_SEED_PRUNE_STATS=1: sum input vs kept SMEMs across all pruned
 * batches and print the pruned fraction once at exit. Measurement only (a
 * single cached-bool branch per pruned batch, inert unless armed); mirrors the
 * BWAMEM3_DEDUP_STATS dumper. Lets a test prove the prune actually removes
 * seeds -- an identity/placement-invariance check alone passes even if the
 * prune silently became a no-op. */
static std::atomic<uint64_t> g_meth_prune_total{0};
static std::atomic<uint64_t> g_meth_prune_kept{0};
static inline bool meth_prune_stats_on()
{
    static const bool on = []() {
        const char *e = getenv("BWAMEM3_METH_SEED_PRUNE_STATS");
        return e && *e && !(e[0] == '0' && e[1] == '\0');
    }();
    return on;
}
static struct MethPruneStatsDumper {
    ~MethPruneStatsDumper() {
        if (!meth_prune_stats_on()) return;
        uint64_t t = g_meth_prune_total.load(), k = g_meth_prune_kept.load();
        fprintf(stderr,
            "[meth-seed-prune-stats] total_smem=%llu kept_smem=%llu pruned_frac=%.4f\n",
            (unsigned long long)t, (unsigned long long)k,
            t ? 1.0 - (double)k / (double)t : 0.0);
    }
} g_meth_prune_stats_dumper;

/* Compact matchArray in place under the resolved prune mode, returning the new
 * SMEM count. rid grouping/order is preserved and each read's longest SMEM
 * (imax, the first on a tie) is always kept, so a read's rid-group is never
 * emptied. spec30: per read M = longest SMEM length; regime A (M < T) keeps
 * all, regime B (M >= T) keeps len >= L or (unique and len >= U). baseline:
 * drop iff len < L and SA-count > Kb. Extracted from mem_chain_seeds so the
 * rule is a single pure function (unit-checkable over hand-built arrays). */
static int64_t meth_prune_compact(SMEM *matchArray, int64_t num_smem,
                                  int mode, int T, int L, int U, int Kb)
{
    int64_t w = 0, ri = 0;
    while (ri < num_smem) {
        int32_t rid_cur = matchArray[ri].rid;
        int64_t rj = ri, imax = ri; int32_t M = -1;
        while (rj < num_smem && matchArray[rj].rid == rid_cur) {
            int32_t ln = matchArray[rj].n + 1 - matchArray[rj].m;
            if (ln > M) { M = ln; imax = rj; }
            rj++;
        }
        for (int64_t k = ri; k < rj; k++) {
            int32_t ln = matchArray[k].n + 1 - matchArray[k].m;
            int keep;
            if (k == imax) keep = 1;                 /* never empty a read */
            else if (mode == MEM_METH_PRUNE_SPEC30) {
                if (M < T) keep = 1;                 /* regime A: keep all */
                else keep = (ln >= L) ||             /* regime B */
                            (matchArray[k].s == 1 && ln >= U);
            } else {                                 /* baseline */
                keep = (ln >= L) || (matchArray[k].s <= (int64_t)Kb);
            }
            if (keep) { if (w != k) matchArray[w] = matchArray[k]; w++; }
        }
        ri = rj;
    }
    if (meth_prune_stats_on()) {
        g_meth_prune_total.fetch_add((uint64_t)num_smem, std::memory_order_relaxed);
        g_meth_prune_kept.fetch_add((uint64_t)w, std::memory_order_relaxed);
    }
    return w;
}

static inline int meth_seed_to_orig(const bntseq_t *seed_bns,
                                    const bntseq_t *orig_bns,
                                    int64_t seed_rbeg, int32_t seed_len,
                                    int read_meth_base_ot,
                                    int64_t *orig_rbeg, int *orig_rid,
                                    int8_t *hypothesis)
{
    if (seed_bns == NULL || orig_bns == NULL) return -1;
    if (seed_rbeg < 0 || seed_rbeg >= (seed_bns->l_pac << 1)) return -1;

    /* Decode the seed start to a forward seed coordinate + RC flag. */
    int seed_is_rev = 0;
    int64_t seed_fwd = bns_depos(seed_bns, seed_rbeg, &seed_is_rev); /* [0,L_seed) */
    int seed_rid = bns_pos2rid(seed_bns, seed_fwd);
    if (seed_rid < 0) return -1;

    /* Seed contig order is r0,f0,r1,f1,...: orig contig = rid/2, OT iff odd. */
    int o_rid = seed_rid >> 1;
    if (o_rid < 0 || o_rid >= orig_bns->n_seqs) return -1;
    *hypothesis = (int8_t)(seed_rid & 1); /* 1=f=OT(C→T), 0=r=OB(G→A) */

    /* Drop cross-chemistry seeds by read number (see meth_seed_filter_enabled).
     * parity = seed_rid & 1 (which converted copy: 1=OT/f, 0=OB/r); seed_is_rev =
     * genomic strand; read# = read_meth_base_ot (R1=1/OT, R2=0/OB). Keep iff the
     * seed's implied chemistry (parity XOR strand) equals the read number. */
    if (read_meth_base_ot >= 0 && meth_seed_filter_enabled()) {
        if ((((int)(seed_rid & 1)) ^ seed_is_rev) != (read_meth_base_ot & 1))
            return -1; /* cross-chemistry seed: drop */
    }

    /* f/r seed contigs are forward projections of the original, so the seed-local
     * offset is the original-local offset. Reject seeds that bridge the contig
     * end (parity with bns_intv2rid, which returns -1 on a multi-contig / f-r
     * boundary span). seed_fwd is the FORWARD coordinate of the seed start: for a
     * forward seed the forward span is [seed_local, seed_local+span); for a
     * reverse seed bns_depos returns the rightmost forward base, so the forward
     * span is (seed_local-span, seed_local]. Require that whole span to fit the
     * seed contig (and, since the original shares the layout, the original too). */
    int64_t seed_local = seed_fwd - seed_bns->anns[seed_rid].offset;
    int64_t span = (seed_len > 0) ? (int64_t)seed_len : 1;
    int64_t span_lo = seed_is_rev ? (seed_local - span + 1) : seed_local;
    int64_t span_hi = seed_is_rev ? (seed_local + 1) : (seed_local + span); /* exclusive */
    int64_t seed_contig_len = seed_bns->anns[seed_rid].len;
    int64_t orig_contig_len = orig_bns->anns[o_rid].len;
    if (span_lo < 0 || span_hi > seed_contig_len) return -1;
    if (span_hi > orig_contig_len)                return -1;
    int64_t orig_fwd = orig_bns->anns[o_rid].offset + seed_local; /* [0,L_orig) */

    /* Re-encode the genomic strand into the ORIGINAL doubled-pac space.
     * NOTE: this mirrors the doubled-pac arithmetic used throughout pairing /
     * mate rescue (e.g. b.rb = (l_pac<<1) - (rb + te + 1)).
     * The seed and original references share identical per-contig length and
     * forward layout, so the transform is a pure per-contig forward-coordinate
     * offset shift (seed_offset → orig_offset, already applied to get orig_fwd),
     * re-encoded to the SAME strand. bns_depos is its own inverse on a single
     * position, so the START coordinate is preserved exactly across the shift;
     * extension re-derives the span. */
    if (!seed_is_rev) {
        *orig_rbeg = orig_fwd;                                   /* forward */
    } else {
        *orig_rbeg = (orig_bns->l_pac << 1) - 1 - orig_fwd;      /* reverse */
    }
    /* NOTE: the reverse-strand re-encode above is THE placement-critical line; it
     * is covered by the seed→original remap round-trip and the OT/OB paired-end
     * strand placement regression tests. */
    if (*orig_rbeg < 0 || *orig_rbeg >= (orig_bns->l_pac << 1)) return -1;
    *orig_rid = o_rid;
    return 0;
}

/* Chain one fully-resolved seed into the per-read kbtree: find the closest
 * existing chain and test_and_merge, else open a new chain. Extracted verbatim
 * from mem_chain_seeds's per-seed body so the two seed-emit paths share it: the
 * default (SEED_ORDER_OFF) streams each seed straight here as it is resolved
 * (single pass, no buffering), while the opt-in reorder modes materialize all
 * seeds, order them, then replay them through here. Behaviour is identical to
 * the previous inline body; only the call site differs. */
static inline void chain_add_one_seed(const mem_opt_t *opt, int64_t l_pac,
                                      const bntseq_t *chain_bns, kbtree_t(chn) *tree,
                                      mem_seed_t *seedBuf, int64_t *seedBufCount,
                                      int64_t seedBufSize, int tid, int seqid,
                                      const mem_seed_t *seed_in,
                                      int rid, int8_t meth_hyp)
{
    /* CHN-15: no local `mem_seed_t s = *seed_in` copy -- seed_in is const and is
     * never mutated here (test_and_merge takes const mem_seed_t*), so the probe
     * and the new-chain store read straight from seed_in. */
    mem_chain_t tmp, *lower;   // CHN-13: `upper` dropped; kb_intervalp never read it
    int to_add = 0;
    tmp.pos = seed_in->rbeg;

    if (kb_size(tree))
    {
        kb_intervalp(chn, tree, &tmp, &lower, NULL); // find the closest chain

        if (!lower || !test_and_merge(opt, l_pac, lower, seed_in, rid, tid))
            to_add = 1;
    }
    else to_add = 1;

    if (to_add) // add the seed as a new chain
    {
        tmp.n = 1; tmp.m = SEEDS_PER_CHAIN;
        if((*seedBufCount + tmp.m) > seedBufSize)
        {
            tmp.m += 1;
            /* CHN-16: dead tprof[PE13] counter removed. The allocation is guarded
             * the same way test_and_merge guards its seed-buffer growth: an
             * `assert` alone compiles out under NDEBUG, so a release build would
             * fall through and dereference NULL at `tmp.seeds[0] = *seed_in`. */
            if ((tmp.seeds = (mem_seed_t *)calloc(tmp.m, sizeof(mem_seed_t))) == NULL) { fprintf(stderr, "ERROR: out of memory tmp.seeds\n"); exit(1); }
        }
        else {
            tmp.seeds = seedBuf + *seedBufCount;
            *seedBufCount += tmp.m;
        }
        /* CHN-11: the per-new-chain memset(tmp.seeds, 0, ...) was dead. tmp.n == 1,
         * so only slot 0 is live and it is overwritten immediately below; any
         * further slots (the +1 overflow buffer, already calloc-zeroed) are never
         * read while zero -- every seed read is bounded by c->n and each slot is
         * written before n reaches it (test_and_merge: c->seeds[c->n++] = *p). */
        tmp.seeds[0] = *seed_in;
        tmp.rid = rid;
        tmp.seqid = seqid;
        /* is_alt indexes the chain-side bns (original in --meth). */
        tmp.is_alt = !!chain_bns->anns[rid].is_alt;
        /* D3: carry the OT/OB hypothesis on the chain (-1 non-meth). */
        tmp.meth_hypothesis = meth_hyp;
        kb_putp(chn, tree, &tmp);
        // CHN-12: (*num_seqid)++ removed with the dead num[] array (see mem_chain_seeds)
    }
}

/** NEW ONE **/
void mem_chain_seeds(FMI_search *fmi, const mem_opt_t *opt,
                     const bntseq_t *bns,
                     const bntseq_t *meth_orig_bns,
                     const bseq1_t *seq_,
                     int nseq,
                     int tid,
                     mem_chain_v *chain_ar,
                     mem_seed_t *seedBuf,
                     int64_t seedBufSize,
                     SMEM *matchArray,
                     int64_t num_smem)
{
    int b, e, l_rep, size = 0;
    int64_t i, pos = 0;
    int64_t smem_ptr = 0;
    /* `bns` is the SEED (f/r-doubled) BNS used to decode SA coordinates. In
     * --meth we remap each seed hit into ORIGINAL coordinates before chaining, so
     * the chain-side bns / l_pac (rid lookup, strand-boundary merge test) must be
     * the ORIGINAL ones. Outside --meth chain_bns == bns. */
    const int meth_remap = (opt->meth_mode && meth_orig_bns != NULL);
    const bntseq_t *chain_bns = meth_remap ? meth_orig_bns : bns;
    int64_t l_pac = chain_bns->l_pac;
    /* Seed-emit strategy. OFF (the default) streams each resolved seed straight
     * into the kbtree in resolve order — no recs[] buffer, no order_seeds pass —
     * which is byte-identical to, and as cheap as, the pre-seed-order baseline.
     * Only a selected reorder mode needs the materialize/order/replay path. */
    const int reorder = (opt->seed_emit_order != SEED_ORDER_OFF);

    /* CHN-12: the per-read `int num[nseq]` counter (zeroed each batch, incremented
     * once per opened chain via a stack-passed pointer) was never read anywhere --
     * pure dead work. Removed along with chain_add_one_seed's num_seqid parameter. */
    int smem_buf_size = 6000;
    int64_t *sa_coord = (int64_t *) _mm_malloc(sizeof(int64_t) * opt->max_occ * smem_buf_size, 64);
    int64_t seedBufCount = 0;

    /* Seed-order refactor (Phase A buffer, Spec S4): one fully-resolved seed per
     * SA hit, sized by RESOLVED-seed count (the prefetch ceiling), NOT SMEM count.
     * Persisted across reads in this call and freed at function end with sa_coord. */
    seed_rec_t *recs = NULL;
    int64_t recs_cap = 0;

    for (int l=0; l<nseq; l++)
        kv_init(chain_ar[l]);

    /* --meth-seed-prune: compact matchArray in place BEFORE SA resolution so the
     * prefetch, resolve loop and chaining all see the pruned set (this is what
     * captures the SA-lookup saving). Preserves rid grouping/order and always
     * keeps each read's longest SMEM, so no read's rid-group is emptied (the loop
     * below maps chains by read-run position). Only active under --meth with a
     * remap bns; OFF (--meth-seed-prune=off) is byte-identical to the pre-prune
     * seeding. Default is SPEC30 under --meth (set in fastmap.cpp). */
    {
        int mp_T, mp_L, mp_U, mp_Kb;
        int mp_mode = meth_seed_prune_mode(opt, &mp_T, &mp_L, &mp_U, &mp_Kb);
        if (meth_remap && mp_mode != MEM_METH_PRUNE_OFF && num_smem > 0)
            num_smem = meth_prune_compact(matchArray, num_smem,
                                          mp_mode, mp_T, mp_L, mp_U, mp_Kb);
    }

    // filter seq at early stage than this!, shifted to collect!!!
    // if (len < opt->min_seed_len) return chain; // if the query is shorter than the seed length, no match

    /* Cross-read SA resolution (default ON; BWA3_SA_PER_READ=1 restores the
     * per-read call for A/B).
     *
     * get_sa_entries_prefetch is a 20-lane refill pipeline over independent
     * compressed-SA walks. Calling it once per read starves it: the median read
     * supplies ~13 positions, so mean lane occupancy is only 10.5/20 and ~20% of
     * all walk rounds run with a SINGLE lane active (fully serialized DRAM
     * latency). The whole batch's SMEMs are already resident in matchArray,
     * rid-sorted and rid-grouped, so resolving a run of reads per call keeps the
     * lanes full. Measured -21..-28% on MEM_SA / -2.0..-2.4% on kernel time,
     * byte-identical, across Graviton4 / Sapphire Rapids / Zen3.
     *
     * Order-preserving by construction: the per-read loop below consumes SMEM
     * runs contiguously in matchArray order, and the staging loop inside
     * get_sa_entries_prefetch enumerates each SMEM's occurrences with the
     * identical stride/cap rule. A chunk's coordArray is therefore the exact
     * concatenation of the per-read ones it covers, so each read's positions stay
     * in that read's order and `mypos` is a simple cursor from the chunk base. */
#if SA_COMPRESSION
    /* Treat an explicit "0" as off, matching meth_seed_filter_enabled(): a
     * presence-only check would let BWA3_SA_PER_READ=0 silently flip on per-read
     * mode and quietly invalidate an A/B comparison run. */
    static const int sa_per_read = []() {
        const char *e = getenv("BWA3_SA_PER_READ");
        return e != NULL && strcmp(e, "0") != 0;
    }();
    const int sa_across = !sa_per_read;
#else
    /* The walk inside get_sa_entries_prefetch assumes the compressed-SA index
     * layout; an uncompressed build resolves via get_sa_entries below instead. */
    const int sa_across = 0;
#endif
    /* Resolve in read-run-aligned chunks rather than one whole-batch call. Two
     * independent reasons, and the tighter one sets the size:
     *   1. Memory. A whole-batch position count is unbounded on repetitive input,
     *      and get_sa_entries_prefetch's staging (t_sa) is thread_local and
     *      grow-only, so one pathological batch would pin that high-water for the
     *      life of the process.
     *   2. Cache. Staging is 16 B/position across pos_ar+map_ar. Per-read it was
     *      L1-resident; a whole batch (~58k positions) is ~930 KB and spills L2.
     * 16k positions = 256 KB of staging, still three orders of magnitude more than
     * the ~13-57 a single read supplies -- far past where occupancy saturates.
     * A chunk sweep (1k..256k) is flat, so this is a comfortable default, not a
     * tuned one. */
    constexpr int64_t sa_chunk_positions = 16384;
    int64_t mypos = 0;          /* cursor into sa_coord, relative to chunk start */
    int64_t chunk_end = 0;      /* exclusive SMEM index resolved so far */
    int64_t sa_cap = (int64_t)opt->max_occ * smem_buf_size;   /* sa_coord elements */

    uint64_t tim = __rdtsc();
    /* Guard on `smem_ptr < num_smem` (unconsumed SMEMs remain), NOT the legacy
     * `pos < num_smem - 1`. Since `smem_ptr == pos + 1` after any read is
     * processed, the two are identical for every batch with num_smem >= 2 (the
     * entire non-meth path and all normal meth batches). They diverge only on the
     * first iteration when num_smem == 1, where `0 < 0` is false: the legacy form
     * skipped the whole loop and dropped the sole surviving read as unmapped. The
     * --meth-seed-prune compaction above makes a single-SMEM batch reachable (a
     * ragged trailing batch whose one rid-group prunes to its longest seed), so
     * fix the pre-existing off-by-one here. The inner do-while keeps its
     * `pos < num_smem - 1` -- that is a real bounds guard for matchArray[pos+1]. */
    for (int l=0; l<nseq && smem_ptr < num_smem; l++)
    {
        // addition, FIX FIX FIX!!!!! THIS!!!!
        //if (aux_->mem.n == 0) continue;

        if (matchArray[smem_ptr].rid > l) continue;
        if (seq_[l].l_seq < opt->min_seed_len) continue;
        assert(matchArray[smem_ptr].rid == l);

        /* Per-thread persistent chaining B-tree (see ChnTreeScratch above):
         * init once per thread, kb_reset (below) between reads instead of a
         * per-read kb_init/kb_destroy. Byte-identical -- tree structure,
         * comparisons, and traversal order depend only on the seeds inserted,
         * never on node provenance, and chaining never deletes, so the reset
         * reproduces the fresh kb_init state exactly -- and the common
         * single-leaf-root read then does zero allocations. */
        static thread_local ChnTreeScratch chn_scratch;
        kbtree_t(chn) *tree = chn_scratch.t;
        mem_chain_v *chain = &chain_ar[l];
        size = 0;

        b = e = l_rep = 0;
        pos = smem_ptr - 1;
        //for (i = 0, b = e = l_rep = 0; i < aux_->mem.n; ++i) // compute frac_rep
        do
        {
            pos ++;
            SMEM *p = &matchArray[pos];
            int sb = p->m, se = p->n + 1;
            if (p->s <= opt->max_occ) continue;
            if (sb > e) l_rep += e - b, b = sb, e = se;
            else e = e > se? e : se;
        } while (pos < num_smem - 1 && matchArray[pos].rid == matchArray[pos + 1].rid);
        l_rep += e - b;

        // bwt_sa
        // assert(pos - smem_ptr + 1 < 6000);
        /* sa_across grows sa_coord per chunk below; a grow here would move the
         * buffer out from under the chunk it already filled. */
        if (!sa_across && pos - smem_ptr + 1 >= smem_buf_size)
        {
            /* csize is in ELEMENTS, matching nsize (see _mm_realloc). sa_coord
             * holds max_occ entries per SMEM, so the old element count is
             * max_occ * smem_buf_size -- not smem_buf_size, which understated it
             * by a factor of max_occ and left the tail of the buffer uncopied.
             * Harmless today only because sa_coord's contents are dead here (the
             * resolve that fills it runs after this grow), but it silently
             * corrupts any caller that makes them live across the grow. */
            int64_t csize = (int64_t)opt->max_occ * smem_buf_size;
            /* A single read can yield more than 2x the current SMEM count, so
             * one doubling may not be enough -- grow until the buffer holds the
             * whole run, or the subsequent get_sa_entries* writes overrun it. */
            while (pos - smem_ptr + 1 >= smem_buf_size)
                smem_buf_size *= 2;
            sa_coord = (int64_t *) _mm_realloc(sa_coord, csize,
                                               (int64_t)opt->max_occ * smem_buf_size,
                                               sizeof(int64_t));
            xassert(sa_coord != NULL, "out of memory: sa_coord");
        }

        /* Phase A buffer (Spec S4): resolved-seed records for the reorder path.
         * Grow with realloc, persist across reads. Reset nrec per read. Only the
         * opt-in reorder modes populate it; OFF streams straight to chaining. */
        int64_t nrec = 0;
        if (reorder)
        {
            /* Tight ceiling: each SMEM contributes at most min(occ, max_occ)
             * resolved seeds (the resolve loop below stops at the occurrence
             * count and caps at max_occ), so summing that is a far snugger bound
             * than the old (#SMEMs * max_occ) worst case whenever most SMEMs are
             * low-occurrence. Still a strict upper bound on nrec. */
            int64_t recs_need = 0;
            for (int64_t si = smem_ptr; si <= pos; ++si)
            {
                int64_t occ = matchArray[si].s;
                recs_need += occ < opt->max_occ ? occ : opt->max_occ;
            }
            if (recs_need > recs_cap)
            {
                recs_cap = recs_need;
                /* CodeRabbit: temp pointer so a NULL realloc doesn't leak recs. */
                seed_rec_t *recs_new = (seed_rec_t *) realloc(recs, recs_cap * sizeof(seed_rec_t));
                xassert(recs_new != NULL, "out of memory: recs_new");
                recs = recs_new;
            }
        }

        int64_t id = 0, cnt_ = 0;
        #if SA_COMPRESSION
        if (sa_across)
        {
            /* Start a new chunk when this read's run falls outside the resolved
             * one. Chunk boundaries are read-run boundaries, so a read's
             * positions are never split across two calls and `mypos` stays a
             * simple cursor from the chunk's base. */
            if (smem_ptr >= chunk_end)
            {
                int64_t need = 0, si = smem_ptr;
                while (si < num_smem)
                {
                    /* Extend by one whole read-run (matchArray is rid-grouped). */
                    int64_t run_start = si, run_need = 0;
                    int32_t rid_cur = matchArray[si].rid;
                    while (si < num_smem && matchArray[si].rid == rid_cur)
                    {
                        int64_t occ = matchArray[si].s;
                        run_need += occ < opt->max_occ ? occ : opt->max_occ;
                        si++;
                    }
                    /* Always take at least one run, even if it alone exceeds the
                     * budget -- a single read must never be split. */
                    if (need > 0 && need + run_need > sa_chunk_positions)
                    {
                        si = run_start;
                        break;
                    }
                    need += run_need;
                    if (need >= sa_chunk_positions) break;
                }
                chunk_end = si;
                if (need > sa_cap)
                {
                    sa_coord = (int64_t *) _mm_realloc(sa_coord, sa_cap, need, sizeof(int64_t));
                    xassert(sa_coord != NULL, "out of memory: sa_coord");
                    sa_cap = need;
                }
                uint64_t tim_sa = __rdtsc();
                fmi->get_sa_entries_prefetch(&matchArray[smem_ptr], sa_coord, &cnt_,
                                             chunk_end - smem_ptr, opt->max_occ, tid, id);
                tprof[MEM_SA][tid] += __rdtsc() - tim_sa;
                mypos = 0;
            }
        }
        else
        {
            mypos = 0;   /* per-read path: sa_coord is rewritten for each read */
            uint64_t tim = __rdtsc();
            fmi->get_sa_entries_prefetch(&matchArray[smem_ptr], sa_coord, &cnt_,
                                         pos - smem_ptr + 1, opt->max_occ, tid, id);  // sa compressed prefetch
            tprof[MEM_SA][tid] += __rdtsc() - tim;
        }
        #else
        mypos = 0;
        #endif

        for (i = smem_ptr; i <= pos; i++)
        {
            SMEM *p = &matchArray[i];
            int64_t step;
            int32_t count, slen = p->n + 1 - p->m; // seed length
            int64_t k;
            step = p->s > opt->max_occ? p->s / opt->max_occ : 1;

            int cnt = 0;
            #if !SA_COMPRESSION
            uint64_t tim = __rdtsc();
            fmi->get_sa_entries(p, sa_coord, &cnt, 1, opt->max_occ);
            tprof[MEM_SA][tid] += __rdtsc() - tim;
            #endif

            cnt = 0;
            for (k = count = 0; k < p->s && count < opt->max_occ; k += step, ++count)
            {
                mem_seed_t s;
                int rid;

                /* Phase A (resolve). B2/B3: the SA cursor advances BEFORE any drop
                 * (rid<0 / un-remappable meth) — a dropped hit still consumes a
                 * slot. The #if SA_COMPRESSION / #else regimes index different
                 * cursors (global mypos vs per-SMEM cnt); keep both. */
                #if SA_COMPRESSION
                s.rbeg = sa_coord[mypos++];
                #else
                s.rbeg = sa_coord[cnt++];
                #endif

                s.qbeg = p->m;
                s.score= s.len = slen;
                // Propagate SMEM SA-count so chain_n_hits gates --supp-rep-hard-cap.
                s.n_hits = static_cast<int32_t>(p->s);
                /* CHN-18: dropped the per-seed `if (s.rbeg < 0 || s.len < 0) fprintf`
                 * sanity branch -- a never-firing debug guard (valid SA decodes give
                 * rbeg >= 0 and len = p->n+1-p->m >= 0) that only wrote to stderr and
                 * never altered s or control flow, so removal is output-identical. */

                /* D3 (--meth, PR-3) seed→original remap (spec §5.2). The SA coord
                 * just decoded (s.rbeg) is in SEED (f/r-doubled) coordinates. Map
                 * it to ORIGINAL doubled-pac coords + an OT/OB hypothesis so all
                 * downstream chaining/extension/output sees original space only.
                 * Non-meth: chain_bns == bns and rid is the seed rid as before. */
                int8_t meth_hyp = -1;
                if (meth_remap) {
                    int64_t o_rbeg = 0; int o_rid = -1;
                    if (meth_seed_to_orig(bns, meth_orig_bns, s.rbeg, s.len,
                                          seq_[l].meth_base_ot,
                                          &o_rbeg, &o_rid, &meth_hyp) != 0)
                        continue; /* un-remappable, or cross-chemistry: drop the seed */
                    s.rbeg = o_rbeg;
                    rid = o_rid;
                } else {
                    rid = bns_intv2rid(bns, s.rbeg, s.rbeg + s.len);
                }
                // bridging multiple reference sequences or the
                // forward-reverse boundary; TODO: split the seed;
                // don't discard it!!!
                if (rid < 0) continue;

                if (!reorder)
                {
                    /* OFF (default): chain the seed immediately, in resolve
                     * order — the single-pass streaming path. No recs[] write,
                     * no order_seeds; byte-identical to buffering with the
                     * identity order but without the double memory traffic. */
                    chain_add_one_seed(opt, l_pac, chain_bns, tree, seedBuf,
                                       &seedBufCount, seedBufSize, tid, l,
                                       &s, rid, meth_hyp);
                }
                else
                {
                    // Phase A: append the fully-resolved seed (drops above already
                    // consumed the SA slot). B1: copy the FULL mem_seed_t.
                    seed_rec_t *r = &recs[nrec++];
                    r->seed = s;
                    r->rid = rid;
                    r->meth_hyp = meth_hyp; // -1 when !meth_remap
                    r->orig_ix = (uint32_t)(nrec - 1);
                }
            }
        } // seeds

        if (reorder)
        {
            // Phase B (order). Stable on orig_ix.
            order_seeds(recs, nrec, opt->seed_emit_order);

            // Phase C (chain). Replay the ordered buffer through the shared
            // chaining helper. S5: equal-pos insertion order into the kbtree is
            // preserved (no dedup/compact beyond order_seeds).
            for (int64_t ri = 0; ri < nrec; ++ri)
                chain_add_one_seed(opt, l_pac, chain_bns, tree, seedBuf,
                                   &seedBufCount, seedBufSize, tid, l,
                                   &recs[ri].seed, recs[ri].rid, recs[ri].meth_hyp);
        } // reorder

        smem_ptr = pos + 1;
        size = kb_size(tree);
        // tprof[PE21][0] += kb_size(tree) * sizeof(mem_chain_t);

        kv_resize(mem_chain_t, *chain, size);

#define traverse_func(p_) (chain->a[chain->n++] = *(p_))
        __kb_traverse(mem_chain_t, tree, traverse_func);
#undef traverse_func

        for (i = 0; i < chain->n; ++i)
            chain->a[i].frac_rep = (float)l_rep / seq_[l].l_seq;

        kb_reset(chn, tree); // reuse the tree for the next read (no destroy/re-init)

    } // iterations over input reads
    tprof[MEM_SA_BLOCK][tid] += __rdtsc() - tim;

    _mm_free(sa_coord);
    free(recs);
}

int mem_kernel1_core(FMI_search *fmi,
                     const mem_opt_t *opt,
                     bseq1_t *seq_,
                     int nseq,
                     mem_chain_v *chain_ar,
                     mem_seed_t *seedBuf,
                     int64_t seedBufSize,
                     mem_cache *mmc,
                     int tid,
                     const bntseq_t *meth_orig_bns,
                     const uint8_t  *meth_orig_pac)
{
    /* D3 (--meth, PR-3): in --meth chaining/filtering run in ORIGINAL coords;
     * outside --meth (or if a handle is missing) fall back to the seed index so
     * the non-meth path is byte-for-byte unchanged. */
    const bntseq_t *chain_bns = (opt->meth_mode && meth_orig_bns != NULL)
                                ? meth_orig_bns : fmi->idx->bns;
    const uint8_t  *chain_pac = (opt->meth_mode && meth_orig_pac != NULL)
                                ? meth_orig_pac : fmi->idx->pac;
    int i;
    int64_t num_smem = 0, tot_len = 0;
    mem_chain_v *chn;

    uint64_t tim;
    /* convert to 2-bit encoding if we have not done so */
    for (int l=0; l<nseq; l++)
    {
        char *seq = seq_[l].seq;
        int len = seq_[l].l_seq;
        tot_len += len;

        for (i = 0; i < len; ++i) {
            unsigned char ch = (unsigned char) seq[i];
            seq[i] = nst_nt4_decode(ch, 4); //nst_nt4??
        }
    }
    // Empty-batch fast path: a 0-base batch (e.g. malformed FASTQ parsed as a
    // single zero-length record) tripped the lazy-init grow block below with
    // tot_len(0) >= wsize_mem(0), routing through _mm_realloc(NULL, 0, 0, …)
    // and exiting via the post-collect overflow check. With no bases, there
    // are no SMEMs to collect — initialize chain_ar to empty so mem_kernel2_core
    // (which walks chain_ar unconditionally and free()s chain_ar[l].a) sees
    // a valid empty state, then let downstream stages emit unmapped records.
    if (tot_len == 0) {
        for (int l = 0; l < nseq; ++l) {
            kv_init(chain_ar[l]);
            chain_ar[l].capped_w = 0;  /* Pass 2 (which sets this) is skipped on this
                                        * early return; kv_init leaves it uninitialized.
                                        * Harmless for output (no regions) but the
                                        * --dedup-reads VERIFY instrument compares it. */
        }
        return 1;
    }
    // enc_qdb is sized in *bytes* (= tot_len). Track its capacity via the
    // dedicated wsize_qdb so it stays correct independent of wsize_mem,
    // which mem_collect_smem grows in *SMEM-entry* units. On short-read
    // workloads (≤50 bp aDNA), the post-mem_collect_smem wsize_mem in SMEM
    // units can exceed tot_len in bytes, which would make the legacy
    // `if (tot_len >= wsize_mem)` gate skip the enc_qdb realloc on
    // subsequent batches even when enc_qdb is undersized.
    if (tot_len > mmc->wsize_qdb[tid]) {
        int64_t tmp = mmc->wsize_qdb[tid];
        uint8_t *new_enc_qdb = (uint8_t *) realloc(mmc->enc_qdb[tid],
                                                   tot_len * sizeof(uint8_t));
        if (new_enc_qdb == NULL) err_fatal(__func__, "out of memory growing enc_qdb");
        mmc->enc_qdb[tid] = new_enc_qdb;
        mmc->wsize_qdb[tid] = tot_len;
        if (bwa_verbose >= 4) {
            fprintf(stderr, "[%0.4d] Re-allocating enc_qdb: "
                    "%" PRId64 " -> %" PRId64 "\n",
                    tid, tmp, mmc->wsize_qdb[tid]);
        }
    }
    // tot_len *= N_SMEM_KERNEL;
    // fprintf(stderr, "wsize: %d, tot_len: %d\n", mmc->wsize_mem[tid], tot_len);
    // This covers matchArray and the per-read int arrays. enc_qdb is grown
    // separately above (wsize_qdb).
    if (tot_len >= mmc->wsize_mem[tid])
    {
        int64_t tmp = mmc->wsize_mem[tid];
        mmc->wsize_mem[tid] = tot_len;
        if (bwa_verbose >= 4) {
            fprintf(stderr, "[%0.4d] Re-allocating SMEM scratch: "
                    "%" PRId64 " -> %" PRId64 "\n",
                    tid, tmp, mmc->wsize_mem[tid]);
        }
        mmc->wsize_mem_s[tid] = tot_len;
        mmc->wsize_mem_r[tid] = tot_len;
        mmc->matchArray[tid]   = (SMEM *) _mm_realloc(mmc->matchArray[tid],
                                                      tmp, mmc->wsize_mem[tid], sizeof(SMEM));
            //realloc(mmc->matchArray[tid], mmc->wsize_mem[tid] *   sizeof(SMEM));
        int32_t *new_min_intv_ar = (int32_t *) realloc(mmc->min_intv_ar[tid],
                                                       mmc->wsize_mem[tid] * sizeof(int32_t));
        if (new_min_intv_ar == NULL) err_fatal(__func__, "out of memory growing min_intv_ar");
        mmc->min_intv_ar[tid] = new_min_intv_ar;

        int32_t *new_query_pos_ar = (int32_t *) realloc(mmc->query_pos_ar[tid],
                                                        mmc->wsize_mem[tid] * sizeof(int32_t));
        if (new_query_pos_ar == NULL) err_fatal(__func__, "out of memory growing query_pos_ar");
        mmc->query_pos_ar[tid] = new_query_pos_ar;

        int32_t *new_rid = (int32_t *) realloc(mmc->rid[tid],
                                               mmc->wsize_mem[tid] * sizeof(int32_t));
        if (new_rid == NULL) err_fatal(__func__, "out of memory growing rid");
        mmc->rid[tid] = new_rid;
        // w.mmc.lim[l]        = (int32_t *) _mm_malloc((BATCH_SIZE + 32) * sizeof(int32_t), 64);
    }

    SMEM    *matchArray   = mmc->matchArray[tid];
    int32_t *min_intv_ar  = mmc->min_intv_ar[tid];
    int32_t *query_pos_ar = mmc->query_pos_ar[tid];
    uint8_t *enc_qdb      = mmc->enc_qdb[tid];
    int32_t *rid          = mmc->rid[tid];
    int64_t  *wsize_mem   = &mmc->wsize_mem[tid];

    tim = __rdtsc();
    /********************** Kernel 1: FM+SMEMs *************************/
    printf_(VER, "6. Calling mem_collect_smem.., tid: %d\n", tid);
    matchArray = mem_collect_smem(fmi, opt,
                                  seq_,
                                  nseq,
                                  matchArray,
                                  min_intv_ar,
                                  query_pos_ar,
                                  enc_qdb,
                                  rid,
                                  mmc,
                                  num_smem,
                                  tid);

    // Exact-fit (num_smem == *wsize_mem) is valid: the last write lands at
    // matchArray[*wsize_mem - 1]. Trip only on a true overrun.
    if (num_smem > *wsize_mem){
        fprintf(stderr, "Error [bug]: num_smem: %lld are more than allocated space %lld.\n",
                (long long)num_smem, (long long)*wsize_mem);
        exit(EXIT_FAILURE);
    }
    printf_(VER, "6. Done! mem_collect_smem, num_smem: %lld\n", (long long)num_smem);
    tprof[MEM_COLLECT][tid] += __rdtsc() - tim;


    /********************* Kernel 1.1: SA2REF **********************/
    tim = __rdtsc();
    printf_(VER, "6.1. Calling mem_chain..\n");
    mem_chain_seeds(fmi, opt, fmi->idx->bns,
                    /* remap target (NULL outside --meth) */ meth_orig_bns,
                    seq_, nseq, tid,
                    chain_ar,
                    seedBuf,
                    seedBufSize,
                    matchArray,
                    num_smem);

    printf_(VER, "5. Done mem_chain..\n");
    tprof[MEM_CHAIN][tid] += __rdtsc() - tim;

    /************** Post-processing of collected smems/chains ************/
    // tim = __rdtsc();
    printf_(VER, "6.1. Calling mem_chain_flt..\n");
    /* Pass 1: filter every read first, so the mate's filtered chains are
     * available to the mate-concordance-aware cap (--extend-mate-concordant) below. */
    for (int l=0; l<nseq; l++) {
        chn = &chain_ar[l];
        chn->n = mem_chain_flt(opt, chn->n, chn->a, tid);
    }
    int cap_mate_concordant = opt->mate_concordant_window != 0 && (opt->flag & MEM_F_PE)
                              && opt->max_extend_chains > 0;
    /* Resolve the concordance window: fixed CLI value if >0; else (auto == -1) the
     * estimated proper-pair insert high bound if known, else the fallback. */
    int64_t mate_win = 0;
    if (cap_mate_concordant)
        mate_win = opt->mate_concordant_window > 0 ? opt->mate_concordant_window
                 : (opt->est_insert_high > 0 ? opt->est_insert_high
                                             : MATE_CONCORDANT_WINDOW_FALLBACK);
    /* Pass 2: cap chains (mate-concordance-aware when --extend-mate-concordant). */
    for (int l=0; l<nseq; l++)
    {
        chn = &chain_ar[l];
        int capped_w = 0;
        if (cap_mate_concordant && (l ^ 1) < nseq) {
            int ml = l ^ 1;  /* interleaved PE: mate is the sibling index */
            chn->n = mem_chain_cap_extend_mate(chn->a, chn->n, opt->max_extend_chains,
                                               chain_ar[ml].a, chain_ar[ml].n,
                                               chain_bns, mate_win,
                                               opt->extend_tie_frac, opt->extend_tie_floor,
                                               &capped_w);
        } else {
            chn->n = mem_chain_cap_extend(chn->a, chn->n, opt->max_extend_chains,
                                          opt->extend_tie_frac, opt->extend_tie_floor,
                                          &capped_w);
        }
        chn->capped_w = opt->extend_csub ? capped_w : 0;  /* carried to worker_sam via regs */
    }
    printf_(VER, "7. Done mem_chain_flt..\n");
    // tprof[MEM_ALN_M1][tid] += __rdtsc() - tim;


    printf_(VER, "8. Calling mem_flt_chained_seeds..\n");
    for (int l=0; l<nseq; l++) {
        chn = &chain_ar[l];
        /* D3: seeds are now in ORIGINAL coords (remapped in mem_chain_seeds), so
         * the seed-SW filter fetches the ORIGINAL ref window.
         * D3 (--meth, PR-4): mem_flt_chained_seeds now scores the ORIGINAL read
         * (meth_orig_seq) against the original ref with the per-chain asymmetric
         * OT/OB matrix (mem_opt_meth_mat), so bisulfite conversions are no longer
         * mis-penalized at the seed-filter stage. */
        mem_flt_chained_seeds(opt, chain_bns, chain_pac, seq_, chn->n, chn->a);
    }
    printf_(VER, "8. Done mem_flt_chained_seeds..\n");
    // tprof[MEM_ALN_M2][tid] += __rdtsc() - tim;


    return 1;
}

int mem_kernel2_core(FMI_search *fmi,
                     const mem_opt_t *opt,
                     bseq1_t *seq_,
                     mem_alnreg_v *regs,
                     int nseq,
                     mem_chain_v *chain_ar,
                     mem_cache *mmc,
                     uint8_t *ref_string,
                     int tid,
                     const bntseq_t *meth_orig_bns,
                     const uint8_t  *meth_orig_pac)
{
    int i;
    /* D3 (--meth, PR-3): extension/dedup must run in ORIGINAL coords. ref_string
     * is already the original .0123 (worker_aln passes mem_aln_ref_string).
     * Outside --meth these fall back to the seed/normal index unchanged. */
    const bntseq_t *aln_bns = (opt->meth_mode && meth_orig_bns != NULL)
                              ? meth_orig_bns : fmi->idx->bns;
    const uint8_t  *aln_pac = (opt->meth_mode && meth_orig_pac != NULL)
                              ? meth_orig_pac : fmi->idx->pac;
    for (int l=0; l<nseq; l++)
    {
        kv_init(regs[l]);
    }
    /****************** Kernel 2: B-SWA *********************/
    uint64_t tim = __rdtsc();
    printf_(VER, "9. Calling mem_chain2aln...\n");
    /* D3 (--meth, PR-4): mem_chain2aln_across_reads_V2 now extends the ORIGINAL
     * read bases (seq_[].meth_orig_seq, 2-bit-encoded internally) against the
     * ORIGINAL ref window with the per-hypothesis asymmetric matrix selected from
     * chain->meth_hypothesis. The ungapped fast path is disabled under --meth
     * (it cannot express the matrix). a->score / a->truesc — and therefore
     * selection (mem_mark_primary_se) and MAPQ (mem_approx_mapq_se) downstream —
     * are now original-space γ scores. A MIXED-hypothesis batch (interleaved
     * R1/OT + R2/OB — the common paired-end case) is handled by bsw_run_tier's
     * per-hypothesis partition pass inside the function: each tier's pairs are
     * split by meth_hypothesis and scored against a matching OT/OB SW object, so
     * every pair gets its asymmetric matrix. Homogeneous (single-hypothesis)
     * batches collapse to a single partition with no extra cost. */
    mem_chain2aln_across_reads_V2(opt,
                                  aln_bns,
                                  aln_pac,
                                  seq_,
                                  nseq,
                                  chain_ar,
                                  regs,
                                  mmc,
                                  ref_string,
                                  tid);

    printf_(VER, "9. Done mem_chain2aln...\n\n");
    tprof[MEM_ALN2][tid] += __rdtsc() - tim;

    // tim = __rdtsc();
    for (int l=0; l<nseq; l++) {
        mem_chain_v *chain = &chain_ar[l];
        for (int i = 0; i < chain->n; ++i)
        {
            mem_chain_t chn = chain->a[i];
            // CHN-16: dead tprof[PE11]/[PE12] false-sharing counters removed
            if (chn.m > SEEDS_PER_CHAIN)
                free(chn.seeds);
        }
        free(chain_ar[l].a);

        /* mem_chain2aln_across_reads_V2 parks a back-pointer to the originating
         * chain on every alnreg (a->c) and walks a->c->seeds to compute
         * a->seedcov. Every one of those reads happens above, inside that
         * function; nothing downstream of this point touches a->c. The chain it
         * points at has just been freed, and its seeds lived in this thread's
         * seed_scratch window -- which the very next work item on this thread
         * will overwrite. Clear the pointer so a future reader gets a null
         * dereference instead of silently-recycled seeds. */
        for (int i2 = 0; i2 < (int) regs[l].n; ++i2)
            regs[l].a[i2].c = NULL;
    }

    int m = 0;
    for (int l=0; l<nseq; l++)
    {
        mem_alnreg_t *a = regs[l].a;
        int n = regs[l].n;
        for (i = 0, m = 0; i < n; ++i) // exclude identical hits
            if (a[i].qe > a[i].qb) {
                if (m != i) a[m++] = a[i];
                else ++m;
            }
        regs[l].n = m;
    }

    /* D3 (--meth, PR-4): the colinear alnreg merge in mem_sort_dedup_patch
     * recomputes p->score/p->truesc (feeds selection + MAPQ), so it must use the
     * ORIGINAL read + per-read OT/OB matrix like extension. Under directional
     * --meth every alnreg of a read shares one hypothesis, so one matrix per read
     * is correct. Outside --meth (or with no orig seq) it stays the projected
     * read + opt->mat — unchanged. The original read is 2-bit-encoded into a
     * reusable scratch buffer (same orientation as seq, per bwa.h). */
    uint8_t *dd_qbuf = NULL;
    int      dd_qbuf_cap = 0;
    for (int l=0; l<nseq; l++) {
        uint8_t      *dd_query = (uint8_t*) seq_[l].seq;
        const int8_t *dd_mat   = opt->mat;
        if (opt->meth_mode && seq_[l].meth_orig_seq != NULL) {
            int l_query = seq_[l].l_seq;
            if (l_query > dd_qbuf_cap) {
                dd_qbuf_cap = l_query;
                dd_qbuf = (uint8_t *) realloc(dd_qbuf, (size_t) dd_qbuf_cap);
                xassert(dd_qbuf != NULL, "out of memory: dd_qbuf");
            }
            const char *os = seq_[l].meth_orig_seq;
            for (int q = 0; q < l_query; ++q) {
                unsigned char ch = (unsigned char) os[q];
                dd_qbuf[q] = nst_nt4_decode(ch, 4);
            }
            dd_query = dd_qbuf;
            /* per-read STRAND-ADJUSTED hypothesis (see meth_strand_hyp); take the
             * first alnreg's (dedup-patch shares one matrix across the read). */
            int hyp = (regs[l].n > 0) ? regs[l].a[0].meth_strand_hyp : -1;
            dd_mat = mem_opt_meth_mat(opt, hyp);
        }
        regs[l].n = mem_sort_dedup_patch(opt, aln_bns,
                                         aln_pac,
                                         dd_query,
                                         regs[l].n, regs[l].a, dd_mat);
        /* SAM-A18: stamp is_alt in this same pass. Each read's regions are now
         * final (dedup done) and is_alt is not read by any later read's dedup,
         * so folding the former separate full pass in here is byte-identical. */
        regs[l].capped_w = chain_ar[l].capped_w;  /* ride the dropped-weight to worker_sam */
        for (i = 0; i < regs[l].n; ++i)
        {
            mem_alnreg_t *p = &regs[l].a[i];
            if (p->rid >= 0 && aln_bns->anns[p->rid].is_alt)
                p->is_alt = 1;
        }
    }
    free(dd_qbuf);  /* NULL outside --meth; free() is NULL-safe */
    // tprof[POST_SWA][tid] += __rdtsc() - tim;

    return 1;
}

static void worker_aln(void *data, int seq_id, int batch_size, int tid)
{
    worker_t *w = (worker_t*) data;

    printf_(VER, "11. Calling mem_kernel2_core..\n");
    /* D3 (--meth): pass the ORIGINAL .0123 ref_string and original bns/pac so
     * extension/dedup fetch original bases for the (now original-coord) seeds.
     * mem_aln_* return the seed/normal handles outside --meth (no-op there). */
    mem_kernel2_core(w->fmi, w->opt,
                     w->seqs + seq_id,
                     w->regs + seq_id,
                     batch_size,
                     /* This thread's chain window -- the SAME one worker_bwt
                      * just filled for this work item. Per-tid, not per-seq_id;
                      * see worker_alloc. */
                     w->chain_scratch + (size_t) tid * BATCH_SIZE,
                     &w->mmc,
                     mem_aln_ref_string(w),
                     tid,
                     w->meth_orig_bns, w->meth_orig_pac);
    printf_(VER, "11. Done mem_kernel2_core....\n");

}


/* [dedup-reads] Phase 2 seed/chain stage for a work item when the memo is armed:
 * run kernel1 on the REP pairs only, then scatter each REP's chains back to its
 * original slot and leave DUP slots empty. Kernel2 (worker_aln) then runs
 * UNMODIFIED over the original view -- a DUP's empty chains yield empty regs,
 * which read_memo_copy_regs later overwrites with the REP's. The work item
 * [seq_id, seq_id+batch_size) is pair-aligned (kt_for splits on the even
 * BATCH_SIZE and the PE tail is even), so whole pairs move as a unit and
 * kernel1's l^1 mate-concordant cap still pairs mates correctly. */
static void worker_bwt_memo(worker_t *w, int seq_id, int batch_size, int tid)
{
    const read_memo_state *memo = w->memo;
    bseq1_t     *seqs = w->seqs + seq_id;
    mem_chain_v *cw   = w->chain_scratch + (size_t) tid * BATCH_SIZE;

    /* D1: kernel1 converts bases ASCII->2-bit in place, but DUP reads skip
     * kernel1 -- so convert EVERY read here (idempotent, guarded on seq[i] < 4)
     * so worker_sam sees 2-bit bases for the DUPs too; REPs re-convert harmlessly
     * inside kernel1. */
    for (int l = 0; l < batch_size; ++l) {
        char *s = seqs[l].seq;
        const int len = seqs[l].l_seq;
        for (int i = 0; i < len; ++i) {
            unsigned char ch = (unsigned char) s[i];
            s[i] = nst_nt4_decode(ch, 4);
        }
    }

    /* Compact REP pairs into a dense view (struct copies; seq/name/qual pointers
     * shared). REPs preserve their relative order, so the j-th REP read maps to
     * the j-th compacted slot. */
    static thread_local bseq1_t compact[BATCH_SIZE];
    int n_rep = 0;
    for (int l = 0; l < batch_size; ++l)
        if (memo->role[(seq_id + l) >> 1] == READ_MEMO_ROLE_REP)
            compact[n_rep++] = seqs[l];

    if (n_rep > 0)
        mem_kernel1_core(w->fmi, w->opt, compact, n_rep, cw,
                         w->seed_scratch + (size_t) tid * BATCH_SIZE * AVG_SEEDS_PER_READ,
                         w->seed_scratch_size, &(w->mmc), tid,
                         w->meth_orig_bns, w->meth_orig_pac);

    /* Scatter cw[0..n_rep) back to the original slots. The j-th REP read sits at
     * original index R[j] >= j, so a single descending pull (original slot l,
     * compact cursor j) moves each REP's chains to its slot and empties every DUP
     * slot, touching each slot exactly once with no aliasing / no double-free.
     * mem_chain_seeds stamped each chain's seqid = its COMPACT index; kernel2
     * asserts c->seqid == l and routes extension results to regs[seqid], so
     * re-stamp seqid to the original slot after moving. */
    int j = n_rep - 1;
    for (int l = batch_size - 1; l >= 0; --l) {
        if (memo->role[(seq_id + l) >> 1] == READ_MEMO_ROLE_REP) {
            if (l != j) cw[l] = cw[j];
            for (size_t k = 0; k < cw[l].n; ++k) cw[l].a[k].seqid = l;
            --j;
        } else {
            kv_init(cw[l]);
        }
    }
    xassert(j == -1, "read_memo: REP scatter cursor mismatch");
}

/* Kernel, called by threads */
static void worker_bwt(void *data, int seq_id, int batch_size, int tid)
{
    worker_t *w = (worker_t*) data;
    /* Armed: compact to REP-only. VERIFY suppresses the compaction so DUP pairs
     * align normally and worker_copy_regs can compare instead of copy. */
    if (w->memo && !read_memo_verify()) { worker_bwt_memo(w, seq_id, batch_size, tid); return; }
    printf_(VER, "4. Calling mem_kernel1_core..%d %d\n", seq_id, tid);

    /* One fixed-size window per thread. The old code additionally granted the
     * final (ragged) work item the whole unused remainder of a global,
     * nreads-sized arena via `(w->nreads - seq_id) * AVG_SEEDS_PER_READ` --
     * which, because nreads is a deliberate over-estimate of the read count
     * (chunk_bases / NREADS_ESTIMATE_AVG_BASES + 10), handed that item millions
     * of extra seed slots. With per-thread windows there is no remainder to
     * grant, so that adjustment is gone.
     *
     * Dropping it only moves where a chain's seeds are allocated: on overflow
     * chain_add_one_seed falls back to calloc and marks the chain with
     * m == SEEDS_PER_CHAIN + 1. That flag is output-dead -- it is read only by
     * the capacity-growth ladder in test_and_merge (which preserves
     * seeds[0..n-1] either way) and by the `m > SEEDS_PER_CHAIN` free sites --
     * so a chain's contents, and therefore every alignment, are unchanged. */
    int seedBufSz = w->seed_scratch_size;

    mem_kernel1_core(w->fmi, w->opt,
                     w->seqs + seq_id,
                     batch_size,
                     w->chain_scratch + (size_t) tid * BATCH_SIZE,
                     w->seed_scratch + (size_t) tid * BATCH_SIZE * AVG_SEEDS_PER_READ,
                     seedBufSz,
                     &(w->mmc),
                     tid,
                     /* D3 (--meth) remap target; NULL outside --meth */
                     w->meth_orig_bns, w->meth_orig_pac);
    printf_(VER, "4. Done mem_kernel1_core....\n");
}

int64_t sort_classify(mem_cache *mmc, int64_t pcnt, int tid)
{

    SeqPair *seqPairArray = mmc->seqPairArrayLeft128[tid];
    // SeqPair *seqPairArrayAux = mmc->seqPairArrayAux[tid];
    SeqPair *seqPairArrayAux = mmc->seqPairArrayRight128[tid];

    int64_t pos8 = 0, pos16 = 0;
    for (int i=0; i<pcnt; i++)
    {
        SeqPair *s = seqPairArray + i;
        int xtra = s->h0;
        int size = (xtra & KSW_XBYTE)? 1 : 2;
        if (size == 1) // 8
        {
            seqPairArray[pos8++] = seqPairArray[i];
        } else { // 16
            seqPairArrayAux[pos16++] = seqPairArray[i];
        }
    }
    assert(pos8 + pos16 == pcnt);

    for (int i=pos8; i<pcnt; i++) {
        seqPairArray[i] = seqPairArrayAux[i-pos8];
    }

    return pos8;
}

/* Fused seed+extend work item: runs mem_kernel1_core then mem_kernel2_core on the
 * SAME reads while they are still hot in this thread's cache, instead of paying a
 * global barrier and a full DRAM round-trip of chain_ar/seedBuf between two
 * separate kt_for passes. Both halves are per-read independent, so this is
 * byte-identical; worker_bwt's seedBufSz tail adjustment is preserved verbatim by
 * delegating rather than inlining. */
static void worker_bwt_aln(void *data, int seq_id, int batch_size, int tid)
{
    worker_bwt(data, seq_id, batch_size, tid);
    worker_aln(data, seq_id, batch_size, tid);
}

/* [dedup-reads] Phase 2: deep-copy one read's regs from its representative.
 * mem_alnreg_t is POD (its only pointer, ->c, is dangling-by-design and NULL'd
 * at kernel2 exit; ->hash is (re)stamped later per-read in the SAM stage), so a
 * plain memcpy is a valid deep copy. Capacity m := n is fine -- the SAM stage's
 * rescue kv_push appends realloc on n == m. Ownership matches worker_sam's
 * free(w->regs[i].a): each DUP owns an independent allocation. */
static inline void read_memo_copy_regs(mem_alnreg_v *dst, const mem_alnreg_v *src)
{
    free(dst->a);
    const int n = (int) src->n;
    if (n > 0) {
        dst->a = (mem_alnreg_t *) malloc((size_t) n * sizeof(mem_alnreg_t));
        xassert(dst->a != NULL, "read_memo: out of memory copying regs");
        memcpy(dst->a, src->a, (size_t) n * sizeof(mem_alnreg_t));
    } else {
        dst->a = NULL;
    }
    dst->n = dst->m = (size_t) n;
    /* --extend-csub side channel: the dropped-chain weight rides the vector, not
     * the regions, so it must be copied too or a DUP loses its csub seed and
     * --dedup-reads on/off stop being byte-identical under --extend-csub. */
    dst->capped_w = src->capped_w;
}

/* [dedup-reads] VERIFY (BWAMEM3_DEDUP_READS_VERIFY): the first mem_alnreg_t field
 * that differs between a duplicate's own regs and its representative's, or NULL
 * if identical. Excludes ->c (dangling, NULL'd at kernel2 exit) and ->hash (set
 * per-read later in the SAM stage); everything else is the position-invariant
 * payload the memo copies. */
static const char *read_memo_alnreg_field_diff(const mem_alnreg_t *a, const mem_alnreg_t *b)
{
#define RM_D(f) do { if ((a)->f != (b)->f) return #f; } while (0)
    RM_D(rb); RM_D(re); RM_D(qb); RM_D(qe); RM_D(rid);
    RM_D(score); RM_D(truesc); RM_D(sub); RM_D(alt_sc); RM_D(csub); RM_D(sub_n);
    RM_D(w); RM_D(seedcov); RM_D(secondary); RM_D(secondary_all);
    RM_D(seedlen0); RM_D(chain_n_hits); RM_D(frac_rep); RM_D(flg);
    RM_D(meth_hypothesis); RM_D(meth_strand_hyp);
#undef RM_D
    if (a->n_comp != b->n_comp) return "n_comp";
    if (a->is_alt != b->is_alt) return "is_alt";
    return NULL;
}

/* [dedup-reads] VERIFY: assert a duplicate read's independently-computed regs
 * equal its representative's (the position-invariance claim, on real data). */
static void read_memo_verify_regs(const mem_alnreg_v *dup, const mem_alnreg_v *rep, int read_idx)
{
    if (dup->n != rep->n)
        err_fatal(__func__, "dedup-reads VERIFY: read %d has n=%zu but its representative has n=%zu",
                  read_idx, dup->n, rep->n);
    if (dup->capped_w != rep->capped_w)
        err_fatal(__func__, "dedup-reads VERIFY: read %d has capped_w=%d but its representative has capped_w=%d",
                  read_idx, dup->capped_w, rep->capped_w);
    for (size_t i = 0; i < dup->n; ++i) {
        const char *f = read_memo_alnreg_field_diff(&dup->a[i], &rep->a[i]);
        if (f)
            err_fatal(__func__, "dedup-reads VERIFY: read %d alnreg %zu field '%s' diverges from its representative",
                      read_idx, i, f);
    }
}

/* [dedup-reads] Phase 2 copy pass: for each DUP pair in this work item, replicate
 * its representative pair's post-extension regs (both mates). Runs as its own
 * kt_for after worker_bwt_aln (all REP regs are final) and strictly before either
 * mem_pestat site, so pairing/MAPQ see the same regs the DUP would have computed.
 * Each DUP read is written by exactly one work item and a REP is never a DUP, so
 * there are no write/write or read/write conflicts across threads.
 *
 * Under VERIFY the DUP pairs were aligned normally (worker_bwt did not compact),
 * so instead of copying we compare each DUP's own regs to the REP's. */
static void worker_copy_regs(void *data, int seq_id, int batch_size, int tid)
{
    worker_t *w = (worker_t*) data;
    const read_memo_state *memo = w->memo;
    const int verify = read_memo_verify();
    for (int l = 0; l < batch_size; l += 2) {
        const int gr = seq_id + l;                 /* global read index (pair R1) */
        if (memo->role[gr >> 1] != READ_MEMO_ROLE_DUP) continue;
        const int rep_r1 = (int) memo->rep_pair[gr >> 1] << 1;
        if (verify) {
            read_memo_verify_regs(&w->regs[gr],     &w->regs[rep_r1],     gr);
            read_memo_verify_regs(&w->regs[gr + 1], &w->regs[rep_r1 + 1], gr + 1);
        } else {
            read_memo_copy_regs(&w->regs[gr],     &w->regs[rep_r1]);
            read_memo_copy_regs(&w->regs[gr + 1], &w->regs[rep_r1 + 1]);
        }
    }
    (void) tid;
}

/* Prefetch the first pac[] cache line of the CIGAR-emission reference window
 * (bwa_gen_cigar3 -> bns_get_seq_into) for an alignment beginning at `rb`. That
 * fetch is a random ~DRAM-latency load; a hint issued ahead of it hides the miss.
 * The window-start byte mirrors bns_get_seq_into's strand split: forward
 * (rb < l_pac) touches pac[rb>>2]; reverse touches pac[((2*l_pac-1-rb)>>2)].
 * locality=0 (streaming/NTA): bns_get_seq_into reads the window once and never
 * revisits it, so per simd_compat.h's locality convention this must not evict
 * L1-resident data (L1-keep, locality 3, is reserved there for genuine reuse
 * like the FM-index walk). The `rb < 0` skip is defense-in-depth for the emit-loop
 * look-ahead callers, which pass a raw a[0].rb that mem_reg2aln itself would
 * reject; mem_reg2aln only reaches here after its own rb>=0 check. A prefetch is
 * a pure hint -> byte-identical whatever region is ultimately emitted. */
static inline void mem_prefetch_cigar_ref(const bntseq_t *bns, const uint8_t *pac, int64_t rb)
{
    if (rb < 0) return;
    int64_t l_pac = bns->l_pac;
    int64_t pf = (rb < l_pac) ? (rb >> 2) : (((l_pac << 1) - 1 - rb) >> 2);
    __builtin_prefetch(&pac[pf], 0, 0);
}

static void worker_sam(void *data, int seqid, int batch_size, int tid)
{
    worker_t *w = (worker_t*) data;

    if (w->opt->flag & MEM_F_PE)
    {
        int64_t pcnt = 0;
        int start = seqid;
        int end = seqid + batch_size;
        int pos = start >> 1;

#if !BWAMEM_BATCHED_MATESW
        // Scalar mem_sam_pe path. Selected when no SIMD kswv kernel is
        // available (e.g. plain SSE2/AVX2 x86 builds, or forced via
        // DISABLE_BATCHED_MATESW=1 for the proto-neon-kswv CI A/B).
        for (int i=start; i< end; i+=2)
        {
            // orig mem_sam_pe() function
            /* D3 (--meth, PR-3): pairing + mate rescue run in ORIGINAL coords
             * (anchors already remapped). mem_aln_bns/pac return the original
             * bns/pac in --meth; the seed/normal index otherwise. This is where
             * the mate-rescue coordinate reconciliation (6a) lands: mem_matesw's
             * window math derives l_pac from bns->l_pac, so passing original
             * bns/pac fixes the doubled-pac↔real-chrom mismatch.
             * D3 (--meth, PR-6): mem_pair_resolve now drives mem_matesw with the
             * ORIGINAL mate bases (s[!i].meth_orig_seq) + the OPPOSITE-strand
             * asymmetric matrix of the anchor; the rescued mate's meth_hypothesis
             * is set to !anchor. (This scalar mem_sam_pe path is selected only
             * when no batched kswv kernel is available.) */
            mem_sam_pe(w->opt, mem_aln_bns(w),
                       mem_aln_pac(w), w->pes,
                       (w->n_processed >> 1) + pos++,   // check!
                       &w->seqs[i],
                       &w->regs[i]);

            free(w->regs[i].a);
            free(w->regs[i+1].a);
        }
#else   // re-structured
        // pre-processing
        // uint64_t tim = __rdtsc();
        int32_t maxRefLen = 0, maxQerLen = 0;
        int32_t gcnt = 0;
        for (int i=start; i< end; i+=2)
        {
            mem_sam_pe_batch_pre(w->opt, mem_aln_bns(w),
                                 mem_aln_pac(w), w->pes,
                                 (w->n_processed >> 1) + pos++,   // check!
                                 &w->seqs[i],
                                 &w->regs[i],
                                 &w->mmc,
                                 pcnt, gcnt,
                                 maxRefLen,
                                 maxQerLen,
                                 tid);
        }

        // tprof[SAM1][tid] += __rdtsc() - tim;
        int64_t pcnt8 = sort_classify(&w->mmc, pcnt, tid);

        kswr_t *aln = (kswr_t *) _mm_malloc ((pcnt + SIMD_WIDTH8) * sizeof(kswr_t), 64);
        xassert(aln != NULL, "out of memory: aln");

        // processing
        mem_sam_pe_batch(w->opt, &w->mmc, pcnt, pcnt8, aln, maxRefLen, maxQerLen, tid);

        // post-processing
        // tim = __rdtsc();
        gcnt = 0;
        pos = start >> 1;
        kswr_t *myaln = aln;
        for (int i=start; i< end; i+=2)
        {
            /* Look-ahead: prefetch the next pair's two emit windows (both mates)
             * while this pair's emit (rescue readback + pairing + 2x CIGAR gen +
             * SAM formatting) hides the ~DRAM-latency loads on the next pair's
             * scattered loci. a[0].rb is a best-effort guess of the emitted window:
             * mem_pair() may emit a non-top region (a[z[i]], z[i]!=0) for multi-
             * mappers, so the guess holds for concordant unique pairs and degrades
             * -- harmlessly, still a hint -- for repeats. Pure hint -> byte-identical. */
            if (i + 3 < end) { // next pair (i+2,i+3); i+3 is the highest index touched
                const bntseq_t *nbns = mem_aln_bns(w);
                const uint8_t  *npac = mem_aln_pac(w);
                if (w->regs[i+2].n > 0) mem_prefetch_cigar_ref(nbns, npac, w->regs[i+2].a[0].rb);
                if (w->regs[i+3].n > 0) mem_prefetch_cigar_ref(nbns, npac, w->regs[i+3].a[0].rb);
            }
            mem_sam_pe_batch_post(w->opt, mem_aln_bns(w),
                                  mem_aln_pac(w), w->pes,
                                  (w->n_processed >> 1) + pos++,   // check!
                                  &w->seqs[i],
                                  &w->regs[i],
                                  &myaln,
                                  &w->mmc,
                                  gcnt,
                                  tid);

            free(w->regs[i].a);
            free(w->regs[i+1].a);
        }
        //tprof[SAM3][tid] += __rdtsc() - tim;
        _mm_free(aln);  // kswr_t
#endif
    }
    else
    {
        for (int i=seqid; i<seqid + batch_size; i++)
        {
            /* Look-ahead: prefetch the next read's emit window while this read's
             * emit hides the ~DRAM-latency load on its scattered locus. a[0] is
             * the score-top region -- the emitted primary in the default case;
             * mem_reorder_primary5 (--primary5) may promote another, harmlessly.
             * Pure hint -> byte-identical. */
            if (i + 1 < seqid + batch_size && w->regs[i+1].n > 0)
                mem_prefetch_cigar_ref(mem_aln_bns(w), mem_aln_pac(w), w->regs[i+1].a[0].rb);
            mem_mark_primary_se(w->opt, w->regs[i].n,
                                w->regs[i].a,
                                w->n_processed + i);
#if V17  // Feature from v0.7.17 of orig. bwa-mem
            if (w->opt->flag & MEM_F_PRIMARY5) mem_reorder_primary5(w->opt->T, &w->regs[i]);
#endif
            mem_seed_capped_sub(&w->regs[i], w->opt->a);  /* --extend-csub (no-op when off) */
            /* D3 (--meth): emit in ORIGINAL coords/alphabet via mem_aln_bns/pac. */
            mem_reg2sam(w->opt, mem_aln_bns(w), mem_aln_pac(w), &w->seqs[i],
                        &w->regs[i], 0, 0);
            free(w->regs[i].a);
        }
    }
}

/* Align phase for one slice of a pestat cohort: seeding + BSW only.
 *
 * Split out of mem_process_seqs so the pipeline can align a cohort in several
 * smaller physical reads while mem_pestat still sees the whole cohort. Nothing
 * here depends on how the cohort is partitioned: seeding and BSW are per-read
 * independent, and the read ids that feed the tie-break hashes come from the
 * global n_processed counter, so a read gets the same id under any slicing.
 *
 * `seqs` and `regs` point at THIS slice (i.e. cohort base + offset), and
 * `n_processed` is the global id of the slice's first read. The caller's
 * w.seqs / w.regs / w.n_processed are saved and restored, so a slice can never
 * leave w.regs pointing into the middle of the allocation -- worker_free()
 * frees that pointer. */
void mem_align_cohort_slice(mem_opt_t *opt,
                            int64_t n_processed,
                            int n,
                            bseq1_t *seqs,
                            mem_alnreg_v *regs,
                            worker_t &w)
{
    bseq1_t      *saved_seqs = w.seqs;
    mem_alnreg_v *saved_regs = w.regs;
    int64_t       saved_np   = w.n_processed;

    w.opt = opt;
    w.seqs = seqs;
    w.regs = regs;
    w.n_processed = n_processed;
    /* [dedup-reads] the cohort-slice align path is not memoized in this version
     * (a DUP's representative can live in another slice, so the copy pass would
     * have to run after all slices, before pestat in mem_pair_and_emit_cohort);
     * keep it byte-identical by leaving the memo unarmed. Note it once under
     * verbose so a --dedup-reads run on this path isn't silently un-accelerated. */
    if (read_memo_mode() != READMEMO_OFF && (opt->flag & MEM_F_PE) && bwa_verbose >= 3) {
        static bool warned = false;   /* serialized caller (pipeline step 1) */
        if (!warned) {
            warned = true;
            fprintf(stderr, "[dedup-reads] cohort/multi-slice align path is not "
                    "memoized in this build; --dedup-reads has no effect here\n");
        }
    }
    w.memo = NULL;

    uint64_t tim = __rdtsc();
    kt_for(worker_bwt_aln, &w, n);
    tprof[WORKER10][0] += __rdtsc() - tim;

    w.seqs = saved_seqs;
    w.regs = saved_regs;
    w.n_processed = saved_np;
}

/* Pair + emit phase for a complete pestat cohort.
 *
 * mem_pestat runs here, over the cohort's contiguous regs -- exactly the read
 * set bwa-mem2 would have handed it for the same batch -- so the inferred
 * insert-size distribution, and therefore pairing, mate rescue and MAPQ, are
 * unchanged by however many slices the align phase used. */
void mem_pair_and_emit_cohort(mem_opt_t *opt,
                              int64_t n_processed,
                              int n,
                              bseq1_t *seqs,
                              const mem_pestat_t *pes0,
                              worker_t &w)
{
    mem_pestat_t pes[4];
    double ctime = cputime(), rtime = realtime();

    w.opt = opt;
    w.seqs = seqs;
    w.n_processed = n_processed;
    w.pes = &pes[0];

    if (opt->flag & MEM_F_PE) {
        if (pes0)
            memcpy(pes, pes0, 4 * sizeof(mem_pestat_t));
        else {
            /* D3 (--meth): insert-size inference operates on alnreg.rb which are
             * in ORIGINAL doubled-pac space, so use the ORIGINAL l_pac. */
            const bntseq_t *pestat_bns = mem_aln_bns(&w);
            fprintf(stderr, "[0000] Inferring insert size distribution of PE reads from data, "
                    "l_pac: %lld, n: %d\n", (long long)pestat_bns->l_pac, n);
            mem_pestat(opt, pestat_bns->l_pac, n, w.regs, pes);
        }
        /* Publish the FR proper-pair insert high bound onto opt so the NEXT
         * cohort's mate-concordant chain cap (which runs before pairing) can use
         * it as its window (--extend-mate-concordant auto). pes index 1 = FR. */
        if (!pes[1].failed && pes[1].high > 0)
            opt->est_insert_high = pes[1].high;
    }

    uint64_t tim = __rdtsc();
    fprintf(stderr, "[0000] 3. Calling kt_for - worker_sam\n");
    kt_for(worker_sam, &w, n);
    tprof[WORKER20][0] += __rdtsc() - tim;

    fprintf(stderr, "\t[0000][ M::%s] Processed %d reads in %.3f "
            "CPU sec, %.3f real sec\n",
            __func__, n, cputime() - ctime, realtime() - rtime);
}

void mem_process_seqs(mem_opt_t *opt,
                      int64_t n_processed,
                      int n,
                      bseq1_t *seqs,
                      const mem_pestat_t *pes0,
                      worker_t &w)
{
    mem_pestat_t pes[4];
    double ctime, rtime;

    ctime = cputime(); rtime = realtime();
    w.opt = opt;
    w.seqs = seqs; w.n_processed = n_processed;
    w.pes = &pes[0];

    /* When -I supplied the insert-size distribution (pes0), publish its FR high
     * bound onto opt BEFORE worker_bwt so this chunk's mate-concordant chain cap
     * (--extend-mate-concordant auto) honors it. Otherwise the bound is only set
     * post-pairing at the end of this function, so a single-chunk -I run would
     * cap with the fallback window and never use the user-provided insert size.
     * The inferred-from-data path (pes0 == NULL) still publishes below for the
     * next chunk. pes index 1 = FR orientation. */
    if ((opt->flag & MEM_F_PE) && opt->mate_concordant_window == -1 &&
        pes0 && !pes0[1].failed && pes0[1].high > 0)
        opt->est_insert_high = pes0[1].high;

    //int n_ = (opt->flag & MEM_F_PE) ? n : n;   // this requires n%2==0
    int n_ = n;

    /* [dedup-reads] Fingerprint this chunk's read-pairs and, when armed, align
     * each distinct pair once and copy its regs to the duplicates. read_memo_should_arm
     * returns the latch (auto/on) and, while auto is measuring, alternates the arm
     * decision so the A/B controller gathers both armed and unarmed samples; the
     * measured realized cost of this chunk (align + copy) is fed back below.
     * PE + non-meth + even n only; the cohort-slice entry paths are not memoized
     * in this version (they stay byte-identical, unarmed).
     * EXIT GATE: a regression pinning `--dedup-reads off == on == auto == baseline`
     * SAM byte-identity (compat_byte_identical.sh / cohort_slice_identity.sh
     * style), plus the BWAMEM3_DEDUP_READS_VERIFY regs-invariance instrument. */
    static read_memo_state g_readmemo_state = { 0, NULL, NULL, 0 };
    read_memo_result rm_r; bool rm_measure = false, rm_arm = false;
    /* VERIFY (BWAMEM3_DEDUP_READS_VERIFY) is a correctness instrument, not a
     * memoization mode: it must run whenever the prepass can identify DUP pairs,
     * independently of whether the memo is armed. Force the prepass on under
     * VERIFY even in `off` mode (and in `auto` while latched OFF, where rm_arm
     * stays false), so the regs-invariance check actually fires instead of
     * silently doing nothing. */
    const bool rm_verify = read_memo_verify();
    if ((read_memo_mode() != READMEMO_OFF || rm_verify) && (opt->flag & MEM_F_PE)
        && !opt->meth_mode && n_ > 0 && (n_ & 1) == 0) {
        rm_r = read_memo_prepass(opt, seqs, n_, &g_readmemo_state);
        rm_measure = true;
        rm_arm = read_memo_should_arm(rm_r.dup_pairs) != 0;
    }
    /* Set w.memo (so the per-pair pass can find DUP/REP roles) when the memo is
     * armed OR when VERIFY needs to compare unarmed DUP pairs. Under VERIFY the
     * compaction in worker_bwt is suppressed, so DUP pairs are aligned normally
     * and their independently-computed regs are compared to the representative's;
     * real memo copying stays gated on rm_arm below. */
    const bool rm_run_pairs = rm_arm || (rm_measure && rm_verify);
    w.memo = rm_run_pairs ? &g_readmemo_state : NULL;
    const auto rm_t0 = std::chrono::steady_clock::now();

    uint64_t tim = __rdtsc();
    fprintf(stderr, "[0000] 1. Calling kt_for - worker_bwt_aln (fused)\n");

    /* FUSED: these were two separate kt_for passes over the whole chunk. Both are
     * strictly per-work-item -- worker_bwt writes chain_ar[seq_id..] / seedBuf for
     * its own reads and worker_aln reads exactly those, with no cross-read
     * dependency, and the only thing that sat between them was an fprintf. Running
     * them as one pass removes a 16-way global barrier per chunk and, more
     * importantly, keeps that chunk's chains and seeds in L2 instead of streaming
     * them out to DRAM at the end of pass 1 and back in at the start of pass 2
     * (order of GB per chunk). Byte-identical: per-read results are independent and
     * the only tid-keyed state (w.mmc) is scratch consumed inside a single call.
     * mem_pestat's barrier below is unaffected and must stay. */
    kt_for(worker_bwt_aln, &w, n_); // SMEMs (+SAL) then BSW, fused
    tprof[WORKER10][0] += __rdtsc() - tim;
    const uint64_t rm_align_ns = rm_measure
        ? (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - rm_t0).count()
        : 0;

    /* [dedup-reads] Copy each DUP pair's regs from its representative. Runs after
     * the (REP-only) align pass and strictly before pestat/pairing, so the
     * inferred insert-size distribution and every downstream stage see the same
     * regs a full alignment would have produced. Time it: the A/B controller
     * charges this copy cost (plus the compaction folded into the align phase
     * above) into the armed sample, so it never latches ON when the memo's own
     * overhead exceeds the align work saved (e.g. cheap reads at a high dup rate). */
    uint64_t rm_copy_ns = 0;
    if (rm_arm) {
        const auto rm_c0 = std::chrono::steady_clock::now();
        kt_for(worker_copy_regs, &w, n_);
        rm_copy_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - rm_c0).count();
    }
    if (rm_measure)
        read_memo_controller_observe(rm_r, rm_align_ns, rm_copy_ns, rm_arm);


    // PAIRED_END
    if (opt->flag & MEM_F_PE) { // infer insert sizes if not provided
        if (pes0)
            memcpy(pes, pes0, 4 * sizeof(mem_pestat_t)); // if pes0 != NULL, set the insert-size
                                                         // distribution as pes0
        else {
            /* D3 (--meth): insert-size inference operates on alnreg.rb which are
             * now in ORIGINAL doubled-pac space, so use the ORIGINAL l_pac. */
            const bntseq_t *pestat_bns = mem_aln_bns(&w);
            fprintf(stderr, "[0000] Inferring insert size distribution of PE reads from data, "
                    "l_pac: %lld, n: %d\n", (long long)pestat_bns->l_pac, n);
            mem_pestat(opt, pestat_bns->l_pac, n, w.regs, pes); // otherwise, infer the insert size
                                                         // distribution from data
        }
        /* Publish the FR proper-pair insert high bound onto opt so the NEXT
         * chunk's mate-concordant chain cap (which runs before pairing) can use
         * it as its window (--extend-mate-concordant auto). opt is shared and
         * mutable across chunks; this write is single-threaded (between kt_for
         * passes), and the cap only reads it. pes index 1 = FR orientation. */
        if (!pes[1].failed && pes[1].high > 0)
            opt->est_insert_high = pes[1].high;
    }

    tim = __rdtsc();
    fprintf(stderr, "[0000] 3. Calling kt_for - worker_sam\n");

    kt_for(worker_sam, &w,  n_);   // SAM
    tprof[WORKER20][0] += __rdtsc() - tim;

    /* [dedup-reads] The memo points at g_readmemo_state, whose role[]/rep_pair[]
     * describe THIS chunk only; clear it so it can never be reused by a later
     * chunk or a different align entry path. */
    w.memo = NULL;

    fprintf(stderr, "\t[0000][ M::%s] Processed %d reads in %.3f "
            "CPU sec, %.3f real sec\n",
            __func__, n, cputime() - ctime, realtime() - rtime);

}

static void mem_mark_primary_se_core(const mem_opt_t *opt, int n, mem_alnreg_t *a, int_v *z)
{ // similar to the loop in mem_chain_flt()
    int i, k, tmp;
    tmp = opt->a + opt->b;
    tmp = opt->o_del + opt->e_del > tmp? opt->o_del + opt->e_del : tmp;
    tmp = opt->o_ins + opt->e_ins > tmp? opt->o_ins + opt->e_ins : tmp;
    z->n = 0;
    kv_push(int, *z, 0);
    for (i = 1; i < n; ++i) {
        for (k = 0; k < z->n; ++k) {
            int j = z->a[k];
            int b_max = a[j].qb > a[i].qb? a[j].qb : a[i].qb;
            int e_min = a[j].qe < a[i].qe? a[j].qe : a[i].qe;
            if (e_min > b_max) { // have overlap
                int min_l = a[i].qe - a[i].qb < a[j].qe - a[j].qb? a[i].qe - a[i].qb : a[j].qe - a[j].qb;
                if (e_min - b_max >= min_l * opt->mask_level) { // significant overlap
                    if (a[j].sub == 0) a[j].sub = a[i].score;
                    if (a[j].score - a[i].score <= tmp && (a[j].is_alt || !a[i].is_alt))
                        ++a[j].sub_n;
                    break;
                }
            }
        }
        if (k == z->n) kv_push(int, *z, i);
        else a[i].secondary = z->a[k];
    }
}

/* Thread-local holder for mem_mark_primary_se's index scratch. Frees the buffer
 * in its destructor, i.e. at thread exit, which is the whole reason it is a
 * struct rather than a bare `static thread_local int_v`: a buffer that is never
 * released is a leak by LeakSanitizer's definition — still-reachable at exit is
 * still unfreed — and the ASan CI row aborts on it. Same shape and rationale as
 * u8vec_scratch_t, kept local because this is its only user. */
struct mark_primary_scratch_t {
    int_v v;
    mark_primary_scratch_t() { kv_init(v); }
    ~mark_primary_scratch_t() { kv_destroy(v); }
};

int mem_mark_primary_se(const mem_opt_t *opt, int n, mem_alnreg_t *a, int64_t id)
{
    int i, n_pri;
    /* Per-thread scratch for the overlap index set. Reused across every read on
     * this thread instead of malloc/free per call: it is pure scratch (only
     * indices, never observable in output), so reuse is byte-identical. Held for
     * the thread's lifetime rather than freed per call, and released at thread
     * exit by the holder's destructor; mem_mark_primary_se_core resets z.n = 0
     * before repopulating, so no stale entry is ever read. */
    static thread_local mark_primary_scratch_t z_scratch;
    int_v &z = z_scratch.v;
    if (n == 0) return 0;

    /* Fast path for the dominant unique-mapper case. With a single region the
     * general path below still runs two ks_introsort calls and a kv_push in
     * mem_mark_primary_se_core (which heap-allocates z.a) only to leave a[0]
     * with these exact values: mem_mark_primary_se_core touches nothing for
     * n==1 (its inner loop starts at i=1), the first rewrite loop sets
     * secondary_all to the rank 0 then normalizes it to secondary (-1), and
     * sub_n is never touched here. Reproduce that directly. */
    if (n == 1) {
        a[0].sub = a[0].alt_sc = 0;
        a[0].secondary = a[0].secondary_all = -1;
        a[0].hash = hash_64(id);
        return a[0].is_alt ? 0 : 1;
    }

    for (i = n_pri = 0; i < n; ++i)
    {
        a[i].sub = a[i].alt_sc = 0, a[i].secondary = a[i].secondary_all = -1, a[i].hash = hash_64(id+i);
        if (!a[i].is_alt) ++n_pri;
    }
    ks_introsort(mem_ars_hash, n, a);
    mem_mark_primary_se_core(opt, n, a, &z);
    for (i = 0; i < n; ++i)
    {
        mem_alnreg_t *p = &a[i];
        p->secondary_all = i; // keep the rank in the first round
        if (!p->is_alt && p->secondary >= 0 && a[p->secondary].is_alt)
            p->alt_sc = a[p->secondary].score;
    }
    if (n_pri >= 0 && n_pri < n)
    {
        kv_resize(int, z, n);
        if (n_pri > 0) ks_introsort(mem_ars_hash2, n, a);
        for (i = 0; i < n; ++i) z.a[a[i].secondary_all] = i;
        for (i = 0; i < n; ++i)
        {
            if (a[i].secondary >= 0)
            {
                a[i].secondary_all = z.a[a[i].secondary];
                if (a[i].is_alt) a[i].secondary = INT_MAX;
            } else a[i].secondary_all = -1;
        }
        if (n_pri > 0) { // mark primary for hits to the primary assembly only
            for (i = 0; i < n_pri; ++i) a[i].sub = 0, a[i].secondary = -1;
            mem_mark_primary_se_core(opt, n_pri, a, &z);
        }
    }
    else {
        for (i = 0; i < n; ++i)
            a[i].secondary_all = a[i].secondary;
    }
    /* z.a intentionally NOT freed here: it is a thread-lifetime scratch buffer,
     * released by z_scratch's destructor at thread exit. */
    return n_pri;
}

/************************
 * Integrated interface *
 ************************/

/* --extend-csub: seed each PRIMARY region's csub with a calibrated estimate of the
 * best chain the cap/gate dropped before extension, so MAPQ (which reads
 * max(sub, csub)) reflects the pruned competitor instead of inflating. The estimate
 * scales the dropped chain's weight by the all-match rate (weight*a; the PE path in
 * mem_capped_pair_subo deliberately uses the observed per-base rate instead, since a
 * pair score sums two estimates), and is clamped strictly below the region's own
 * score, so it lowers MAPQ proportionally rather than forcing it to 0 (the UNCLAMPED
 * weight*a version tripped `sub >= score -> return 0`, which the PE MAPQ
 * recombination then turned into spurious promotions). No-op when nothing was
 * dropped (capped_w == 0), so --extend-csub off is byte-identical. */
void mem_seed_capped_sub(mem_alnreg_v *a, int match_a)
{
    if (a->capped_w <= 0) return;
    for (size_t r = 0; r < a->n; ++r) {
        mem_alnreg_t *p = &a->a[r];
        if (p->secondary >= 0) continue;            /* primary regions only */
        int span = p->qe - p->qb;
        if (span <= 0 || p->score <= 0) continue;
        /* TUNING(csub): all-match rate (weight*a). The clamp below is what makes this
         * safe (the naive unclamped weight*a tripped `sub>=score -> mapq 0`). */
        long est = (long)a->capped_w * match_a;
        if (est >= p->score) est = p->score - 1;    /* clamp: never force mapq 0 */
        /* Seed the pruned competitor into `sub`, NOT `csub`. csub is the tandem-hit slot,
         * capped mate-BLIND after the +40 PE pair promotion (bwamem_pair.cpp) -- routing a
         * dropped (mate-non-concordant under the mate-aware --fast cap) chain there over-demotes
         * correct paired reads in paralog/pseudogene regions. `sub` feeds the promotable q_se, so
         * pairing can rescue reads the mate resolves. SE MAPQ and XS both read max(sub,csub), so
         * single-end output is byte-identical. */
        if (est > p->sub) p->sub = (int)est;
    }
}

int mem_approx_mapq_se(const mem_opt_t *opt, const mem_alnreg_t *a)
{
    int mapq, l, sub = a->sub? a->sub : opt->min_seed_len * opt->a;
    double identity;
    sub = a->csub > sub? a->csub : sub;
    if (sub >= a->score) return 0;
    l = a->qe - a->qb > a->re - a->rb? a->qe - a->qb : a->re - a->rb;
    identity = 1. - (double)(l * opt->a - a->score) / (opt->a + opt->b) / l;
    if (a->score == 0) {
        mapq = 0;
    } else if (opt->mapQ_coef_len > 0) {
        double tmp;
        tmp = l < opt->mapQ_coef_len? 1. : opt->mapQ_coef_fac / log(l);
        tmp *= identity * identity;
        mapq = (int)(6.02 * (a->score - sub) / opt->a * tmp * tmp + .499);
    } else {
        mapq = (int)(MEM_MAPQ_COEF * (1. - (double)sub / a->score) * log(a->seedcov) + .499);
        mapq = identity < 0.95? (int)(mapq * identity * identity + .499) : mapq;
    }
    if (a->sub_n > 0) mapq -= (int)(4.343 * log(a->sub_n+1) + .499);
    if (mapq > 60) mapq = 60;
    if (mapq < 0) mapq = 0;
    mapq = (int)(mapq * (1. - a->frac_rep) + .499);
    return mapq;
}

void mem_reorder_primary5(int T, mem_alnreg_v *a)
{
    int k, n_pri = 0, left_st = INT_MAX, left_k = -1;
    mem_alnreg_t t;
    for (k = 0; k < a->n; ++k)
        if (a->a[k].secondary < 0 && !a->a[k].is_alt && a->a[k].score >= T) ++n_pri;
    if (n_pri <= 1) return; // only one alignment
    for (k = 0; k < a->n; ++k) {
        mem_alnreg_t *p = &a->a[k];
        if (p->secondary >= 0 || p->is_alt || p->score < T) continue;
        if (p->qb < left_st) left_st = p->qb, left_k = k;
    }
    assert(a->a[0].secondary < 0);
    if (left_k == 0) return; // no need to reorder
    t = a->a[0], a->a[0] = a->a[left_k], a->a[left_k] = t;
    for (k = 1; k < a->n; ++k) { // update secondary and secondary_all
        mem_alnreg_t *p = &a->a[k];
        if (p->secondary == 0) p->secondary = left_k;
        else if (p->secondary == left_k) p->secondary = 0;
        if (p->secondary_all == 0) p->secondary_all = left_k;
        else if (p->secondary_all == left_k) p->secondary_all = 0;
    }
}

// TODO (future plan): group hits into a uint64_t[] array. This will be cleaner and more flexible
void mem_reg2sam(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                 bseq1_t *s, mem_alnreg_v *a, int extra_flag, const mem_aln_t *m)
{
    kstring_t str;
    kvec_t(mem_aln_t) aa;
    int k, l;
    char **XA = 0;
    int *HN = 0;

    if (!(opt->flag & MEM_F_ALL))
        XA = mem_gen_alt(opt, bns, pac, a, s->l_seq, s->seq, &HN, s->meth_orig_seq);

    kv_init(aa);
    str.l = str.m = 0; str.s = 0;
    for (k = l = 0; k < a->n; ++k)
    {
        mem_alnreg_t *p = &a->a[k];
        mem_aln_t *q;
        if (p->score < opt->T) continue;
        //fprintf(stderr, "%d %d %d %d\n", p->secondary, p->is_alt, opt->flag&MEM_F_ALL,
        //      (p->secondary >= 0 && (p->is_alt || !(opt->flag&MEM_F_ALL))));
        if (p->secondary >= 0 && (p->is_alt || !(opt->flag&MEM_F_ALL))) continue;
        // assert(p->secondary < INT_MAX);
        if (p->secondary >= 0 && p->secondary < INT_MAX && p->score < a->a[p->secondary].score * opt->drop_ratio) continue;
        q = kv_pushp(mem_aln_t, aa);

        *q = mem_reg2aln(opt, bns, pac, s->l_seq, s->seq, p, s->meth_orig_seq);
        assert(q->rid >= 0); // this should not happen with the new code
        q->XA = XA? XA[k] : 0;
        q->HN = HN? HN[k] : -1;
        q->flag |= extra_flag; // flag secondary
        if (p->secondary >= 0) q->sub = -1; // don't output sub-optimal score
        if (l && p->secondary < 0) // if supplementary
            q->flag |= (opt->flag&MEM_F_NO_MULTI)? 0x10000 : 0x800;
#if V17
        if (!(opt->flag & MEM_F_KEEP_SUPP_MAPQ) && l && !p->is_alt && q->mapq > aa.a[0].mapq)
            q->mapq = aa.a[0].mapq; // lower mapq for supplementary mappings, unless -5 or -q is applied
#else
        if (l && !p->is_alt && q->mapq > aa.a[0].mapq) q->mapq = aa.a[0].mapq;
#endif
        // fg-labs: supp alnregs whose chain contains a repetitive seed (many
        // genome occurrences) inherit primary MAPQ despite being ambiguous on
        // their own — upstream issue bwa-mem2/bwa-mem2#260. Opt-in override:
        // force MAPQ=0 when the chain's most-repetitive seed exceeds the cap.
        if (opt->supp_rep_hard_cap > 0 && l && p->secondary < 0 &&
            p->chain_n_hits >= opt->supp_rep_hard_cap)
            q->mapq = 0;
        ++l;
    }
    if (aa.n == 0) { // no alignments good enough; then write an unaligned record
        mem_aln_t t;
        t = mem_reg2aln(opt, bns, pac, s->l_seq, s->seq, 0);
        t.flag |= extra_flag;
        mem_aln2sam(opt, bns, &str, s, 1, &t, 0, m);
    } else {
        for (k = 0; k < aa.n; ++k)
            mem_aln2sam(opt, bns, &str, s, aa.n, aa.a, k, m);
        for (k = 0; k < aa.n; ++k) free(aa.a[k].cigar);
        free(aa.a);
    }
    s->sam = str.s;
    if (XA) {
        for (k = 0; k < a->n; ++k) free(XA[k]);
        free(XA);
    }
    free(HN);
}

/* Caller must have called ks_resize beforehand to leave room for
 * n_cigar*16 bytes (worst-case: each op is 11 digits + 1 letter). */
static inline void add_cigar(const mem_opt_t *opt, mem_aln_t *p, kstring_t *str, int which)
{
    int i;
    if (p->n_cigar) { // aligned
        for (i = 0; i < p->n_cigar; ++i) {
            int c = p->cigar[i]&0xf;
            if (!(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt && (c == 3 || c == 4))
                c = which? 4 : 3; // use hard clipping for supplementary alignments
            kputw_u(p->cigar[i]>>4, str); kputc_u("MIDSH"[c], str);
        }
    } else kputc_u('*', str); // having a coordinate but unaligned (e.g. when copy_mate is true)
}

void mem_aln2sam(const mem_opt_t *opt, const bntseq_t *bns, kstring_t *str,
                 bseq1_t *s, int n, const mem_aln_t *list, int which, const mem_aln_t *m_)
{
    SpEncodeScope _enc;   /* stage_prof: time the SAM/BAM build (all return paths) */
    /* bam1_t short-circuit: meth_mode applies the bisulfite overlay (Bismark
     * XR:Z/XG:Z/XM:Z, opt-in chimera QC), plain bam_mode uses the generic
     * writer. Either path leaves `str` untouched, so callers must branch on
     * mem_opt_records_are_bam(), not opt->bam_mode — --meth builds bam1_t even
     * when the container is SAM text. */
    if (opt->meth_mode) {
        struct bam1_t *b = bam_writer_alloc();
        if (b == NULL)
            err_fatal(__func__, "out of memory allocating bam1_t for meth record");
        /* `bns` here is the ORIGINAL bns under --meth (mem_aln_bns()); pair
         * it with the original pac global for the XM:Z reference window. */
        if (meth_mem_aln_to_bam(b, opt, bns, g_meth_orig_pac, s, n, list, which, m_) != 0) {
            bam_writer_free(b);
            err_fatal(__func__, "meth BAM conversion failed for read \"%s\"", s->name);
        }
        bam_writer_bseq_push(s, b);
        return;
    }
    if (opt->bam_mode) {
        if (bam_writer_push_aln(s, opt, bns, n, list, which, m_) != 0)
            err_fatal(__func__, "BAM conversion failed for read \"%s\"", s->name);
        return;
    }

    int i, l_name;
    mem_aln_t ptmp = list[which], *p = &ptmp, mtmp, *m = 0; // make a copy of the alignment to convert

    if (m_) mtmp = *m_, m = &mtmp;
    // set flag
    p->flag |= m? 0x1 : 0; // is paired in sequencing
    p->flag |= p->rid < 0? 0x4 : 0; // is mapped
    p->flag |= m && m->rid < 0? 0x8 : 0; // is mate mapped
    if (p->rid < 0 && m && m->rid >= 0) // copy mate to alignment
        p->rid = m->rid, p->pos = m->pos, p->is_rev = m->is_rev, p->n_cigar = 0;
    if (m && m->rid < 0 && p->rid >= 0) // copy alignment to mate
        m->rid = p->rid, m->pos = p->pos, m->is_rev = p->is_rev, m->n_cigar = 0;
    p->flag |= p->is_rev? 0x10 : 0; // is on the reverse strand
    p->flag |= m && m->is_rev? 0x20 : 0; // is mate on the reverse strand

    /* ─── Pre-size kstring once for the worst-case record length, then use
     * unsafe (no-realloc, no-bound-check) writes throughout. The estimate
     * covers: QNAME + FLAG + RNAME + POS + MAPQ + CIGAR + mate fields +
     * SEQ + QUAL + tags + headers; rare overflow falls back to ks_resize.
     */
    l_name = (int)strlen(s->name);
    /* Cache lengths that are needed both for pre-sizing and for the actual
     * unsafe writes below, so each string is measured exactly once per record:
     *   - md_len   : MD:Z string (SAM-A16)
     *   - rname_len: RNAME contig name (SAM-A14)
     *   - rnext_len: RNEXT contig name (SAM-A14)
     * Each is only *used* at its write site under the same guard that makes it
     * hold a real strlen here, so the cached value is byte-identical to the
     * strlen kputs_u() would have recomputed. */
    size_t md_len = 0;
    if (p->n_cigar)
        md_len = strlen((const char*)(p->cigar + p->n_cigar));
    /* RNAME/RNEXT contig names can be very long (T2T-style assemblies).
     * Mate CIGAR (MC:Z) under V17 is also written via add_cigar(). */
    size_t rname_len = (p->rid >= 0) ? strlen(bns->anns[p->rid].name) : 1;
    size_t rnext_len = (m && m->rid >= 0 && m->rid != p->rid)
                       ? strlen(bns->anns[m->rid].name) : 1;
    {
        size_t comm_len = s->comment ? strlen(s->comment) : 0;
        size_t anno_len = ((opt->flag & MEM_F_REF_HDR) && p->rid >= 0
                           && bns->anns[p->rid].anno && bns->anns[p->rid].anno[0])
                          ? strlen(bns->anns[p->rid].anno) : 0;
        size_t xa_len   = p->XA ? strlen(p->XA) : 0;
        size_t rg_len   = bwa_rg_id[0] ? strlen(bwa_rg_id) : 0;
        size_t mate_cigar_len = (m && m->n_cigar) ? (size_t)m->n_cigar * 16 : 0;
        /* SA:Z worst case (only emitted when n>1 and there are other primary
         * hits, but priced unconditionally so the local ks_resize at the
         * SA:Z site can be dropped — single resize keeps the unsafe-fast
         * write invariant for the post-SA:Z tags (XA, HN, comment, XR:Z). */
        size_t sa_need = 16;
        for (int j = 0; j < n; ++j) {
            if (j == which || (list[j].flag&0x100)) continue;
            sa_need += strlen(bns->anns[list[j].rid].name) + 32 + list[j].n_cigar * 16;
        }
        size_t need = (size_t)l_name
                    + (size_t)s->l_seq * 2     /* SEQ + QUAL */
                    + (size_t)p->n_cigar * 16  /* primary CIGAR */
                    + mate_cigar_len           /* MC:Z mate CIGAR */
                    + rname_len + rnext_len    /* RNAME + RNEXT */
                    + md_len + xa_len + rg_len + comm_len + anno_len
                    + sa_need                  /* SA:Z (multi-supp) */
                    + 256;                      /* slack for ints + tags
                                                 * (incl. pa:f:, bounded by
                                                 * BWA_PA_TEXT_MAX) */
        ks_resize(str, str->l + need);
    }

    // print up to CIGAR
    kputsn_u(s->name, l_name, str); kputc_u('\t', str); // QNAME
    kputw_u((p->flag&0xffff) | (p->flag&0x10000? 0x100 : 0), str); kputc_u('\t', str); // FLAG
    if (p->rid >= 0) { // with coordinate
        kputsn_u(bns->anns[p->rid].name, (int)rname_len, str); kputc_u('\t', str); // RNAME
        kputl_u(p->pos + 1, str); kputc_u('\t', str); // POS
        kputw_u(p->mapq, str); kputc_u('\t', str); // MAPQ
#if OLD // from v15
        if (p->n_cigar) { // aligned
            for (i = 0; i < p->n_cigar; ++i) {
                int c = p->cigar[i]&0xf;
                if (!(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt && (c == 3 || c == 4))
                    c = which? 4 : 3; // use hard clipping for supplementary alignments
                kputw_u(p->cigar[i]>>4, str); kputc_u("MIDSH"[c], str);
            }
        } else kputc_u('*', str);
#else
        // modification from v17
        add_cigar(opt, p, str, which);
#endif
    } else kputsn_u("*\t0\t0\t*", 7, str); // without coordinte
    kputc_u('\t', str);

    // print the mate position if applicable
    if (m && m->rid >= 0) {
        if (p->rid == m->rid) kputc_u('=', str);
        else kputsn_u(bns->anns[m->rid].name, (int)rnext_len, str);
        kputc_u('\t', str);
        kputl_u(m->pos + 1, str); kputc_u('\t', str);
        if (p->rid == m->rid) {
            int64_t p0 = p->pos + (p->is_rev? get_rlen(p->n_cigar, p->cigar) - 1 : 0);
            int64_t p1 = m->pos + (m->is_rev? get_rlen(m->n_cigar, m->cigar) - 1 : 0);
            if (m->n_cigar == 0 || p->n_cigar == 0) kputc_u('0', str);
            else kputl_u(-(p0 - p1 + (p0 > p1? 1 : p0 < p1? -1 : 0)), str);
        } else kputc_u('0', str);
    } else kputsn_u("*\t0\t0", 5, str);
    kputc_u('\t', str);

    // print SEQ and QUAL — SIMD via NEON tbl1q_u8 byte LUT.
    if (p->flag & 0x100) { // for secondary alignments, don't write SEQ and QUAL
        kputsn_u("*\t*", 3, str);
    } else if (!p->is_rev) { // forward strand
        int qb = 0, qe = s->l_seq;
        if (p->n_cigar && which && !(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt) {
            if ((p->cigar[0]&0xf) == 4 || (p->cigar[0]&0xf) == 3) qb += p->cigar[0]>>4;
            if ((p->cigar[p->n_cigar-1]&0xf) == 4 || (p->cigar[p->n_cigar-1]&0xf) == 3) qe -= p->cigar[p->n_cigar-1]>>4;
        }
        const int n_seq = qe - qb;
        sam_encode_seq_fwd(str->s + str->l, (const uint8_t*)(s->seq + qb), n_seq);
        str->l += n_seq;
        kputc_u('\t', str);
        if (s->qual) {
            memcpy(str->s + str->l, s->qual + qb, (size_t)n_seq);
            str->l += n_seq;
        } else kputc_u('*', str);
    } else { // reverse strand
        int qb = 0, qe = s->l_seq;
        if (p->n_cigar && which && !(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt) {
            if ((p->cigar[0]&0xf) == 4 || (p->cigar[0]&0xf) == 3) qe -= p->cigar[0]>>4;
            if ((p->cigar[p->n_cigar-1]&0xf) == 4 || (p->cigar[p->n_cigar-1]&0xf) == 3) qb += p->cigar[p->n_cigar-1]>>4;
        }
        const int n_seq = qe - qb;
        sam_encode_seq_rev(str->s + str->l, (const uint8_t*)(s->seq + qb), n_seq);
        str->l += n_seq;
        kputc_u('\t', str);
        if (s->qual) {
            sam_encode_qual_rev(str->s + str->l, (const char*)(s->qual + qb), n_seq);
            str->l += n_seq;
        } else kputc_u('*', str);
    }

    // print optional tags
    if (p->n_cigar) {
        kputsn_u("\tNM:i:", 6, str); kputw_u(p->NM, str);
        kputsn_u("\tMD:Z:", 6, str); kputsn_u((char*)(p->cigar + p->n_cigar), (int)md_len, str);
    }
#if V17
    if (m && m->n_cigar) { kputsn_u("\tMC:Z:", 6, str); add_cigar(opt, m, str, which); }
#endif
    if (m && opt->compat->emit_mq) { kputsn_u("\tMQ:i:", 6, str); kputw_u(m->mapq, str); }
    if (p->score >= 0) { kputsn_u("\tAS:i:", 6, str); kputw_u(p->score, str); }
    if (p->sub >= 0) { kputsn_u("\tXS:i:", 6, str); kputw_u(p->sub, str); }
    if (bwa_rg_id[0]) { kputsn_u("\tRG:Z:", 6, str); kputs_u(bwa_rg_id, str); }
    if (!(p->flag & 0x100)) { // not multi-hit
        for (i = 0; i < n; ++i)
            if (i != which && !(list[i].flag&0x100)) break;
        if (i < n) { // there are other primary hits; output them
            /* SA:Z capacity is reserved in the outer pre-size above. */
            kputsn_u("\tSA:Z:", 6, str);
            for (i = 0; i < n; ++i) {
                const mem_aln_t *r = &list[i];
                int k;
                if (i == which || (r->flag&0x100)) continue;
                kputs_u(bns->anns[r->rid].name, str); kputc_u(',', str);
                kputl_u(r->pos+1, str); kputc_u(',', str);
                kputc_u("+-"[r->is_rev], str); kputc_u(',', str);
                for (k = 0; k < r->n_cigar; ++k) {
                    kputw_u(r->cigar[k]>>4, str); kputc_u("MIDSH"[r->cigar[k]&0xf], str);
                }
                kputc_u(',', str); kputw_u(r->mapq, str);
                kputc_u(',', str); kputw_u(r->NM, str);
                kputc_u(';', str);
            }
        }
        if (p->alt_sc > 0) {
            /* Shared with the --bam and --meth writers so the three cannot
             * drift again; see bwa_format_pa_value. The token is at most
             * BWA_PA_TEXT_MAX-1 bytes, covered by the tag slack in the
             * pre-size above, so the unsafe writes stay in bounds. */
            char pa[BWA_PA_TEXT_MAX];
            const int pa_len = bwa_format_pa_value(pa, p->score, p->alt_sc);
            kputsn_u("\tpa:f:", 6, str); kputsn_u(pa, pa_len, str);
        }
    }

    if (p->XA) { kputsn_u("\tXA:Z:", 6, str); kputs_u(p->XA, str); }
    if (p->HN >= 0 && opt->compat->emit_hn) { kputsn_u("\tHN:i:", 6, str); kputw_u(p->HN, str); }

    if (s->comment) { kputc_u('\t', str); kputs_u(s->comment, str); }
    if ((opt->flag&MEM_F_REF_HDR) && !opt->meth_mode && p->rid >= 0 && bns->anns[p->rid].anno != 0 && bns->anns[p->rid].anno[0] != 0) {
        /* Suppressed under --meth: that mode repurposes XR:Z for the Bismark
         * read-conversion direction (see docs/src/methylation/tags.md). */
        int tmp;
        kputsn_u("\tXR:Z:", 6, str);
        tmp = (int)str->l;
        kputs_u(bns->anns[p->rid].anno, str);
        for (i = tmp; i < (int)str->l; ++i) // replace TAB in the comment to SPACE
            if (str->s[i] == '\t') str->s[i] = ' ';
    }
    kputc_u('\n', str);
    str->s[str->l] = 0;  /* single trailing NUL — unsafe writers skipped this */
}

/* Does this region's CIGAR/NM/MD regeneration run in NATIVE (--meth) mode?
 *
 * One predicate, because it selects THREE things at once and they must never
 * disagree: the query bases (original vs projected), the scoring matrix
 * (per-hypothesis asymmetric vs symmetric), and the NM/MD predicate
 * (matrix-derived vs literal). A caller that satisfies two of the three inputs
 * but not the third silently produces a record with the OTHER tag semantics and
 * no diagnostic — which is exactly how XA:Z sub-entries ended up reporting
 * conversion-counting NM alongside a conversion-hiding primary (fixed by
 * threading meth_orig_query through mem_gen_alt). Keeping the decision in one
 * named place is what makes that class of mistake reviewable.
 *
 * On the `meth_hypothesis >= 0` term: for a MAPPED region under --meth this is
 * always true, so the term is defensive rather than a real branch.
 *   - Chain seeds that cannot be remapped to an OT/OB hypothesis are DROPPED,
 *     not carried with -1 (mem_chain_seeds -> meth_seed_to_orig returns
 *     non-zero and the seed is skipped).
 *   - meth_orig_bns is NULL only if the reference prefix overflowed PATH_MAX,
 *     which warns loudly at startup.
 *   - Every mem_matesw call site passes `opt->meth_mode ? i : -1`, so a rescued
 *     mate under --meth inherits 0 or 1, never -1.
 * Were it ever false, the fallback is still self-consistent rather than mixed:
 * the symmetric opt->mat has every off-diagonal at -b < 0 (--meth rejects
 * -B 0), so the matrix-derived and literal NM/MD predicates agree on it. */
static int mem_use_native_regen(const mem_opt_t *opt, const mem_alnreg_t *ar,
                                const char *meth_orig_query)
{
    return opt->meth_mode && meth_orig_query != NULL && ar->meth_hypothesis >= 0;
}

mem_aln_t mem_reg2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const char *query_, const mem_alnreg_t *ar, const char *meth_orig_query)
{
    mem_aln_t a;
    int i, w2, tmp, qb, qe, NM, score, is_rev, last_sc = -(1<<30), l_MD;
    int64_t pos, rb, re;
    uint8_t *query;

    memset(&a, 0, sizeof(mem_aln_t));
    a.HN = -1; // sentinel: HN not computed unless caller fills from mem_gen_alt out_hn
    a.meth_hypothesis = -1; // non-meth default; overwritten below from ar
    if (ar == 0 || ar->rb < 0 || ar->re < 0) { // generate an unmapped record
        a.rid = -1; a.pos = -1; a.flag |= 0x4;
        return a;
    }
    a.meth_hypothesis = ar->meth_hypothesis; // carry hypothesis to the output layer
    qb = ar->qb, qe = ar->qe;
    rb = ar->rb, re = ar->re;
    /* Shallow same-call prefetch of this alignment's CIGAR-emission window,
     * hidden behind the ~dozens of setup instructions below. The emit loops also
     * issue a deeper look-ahead prefetch of the NEXT read/pair's window (see
     * mem_prefetch_cigar_ref callers), so on the hot path this call's window is
     * already resident and this becomes a no-op hit; it still covers callers
     * that reach mem_reg2aln without a look-ahead. Pure hint -> byte-identical. */
    mem_prefetch_cigar_ref(bns, pac, rb);
    /* D3 (--meth, PR-5): output CIGAR/NM/MD must reflect the ORIGINAL read vs the
     * ORIGINAL reference in the original alphabet, so placement and gap shape are
     * expressed in real coordinates and a real (non-converting) SNP stays a
     * mismatch. When `meth_orig_query` is supplied, regen from the original
     * bases; else fall back to the projected `query_` (non-meth path,
     * byte-for-byte identical to legacy). How conversions themselves are treated
     * in NM/MD is governed by the policy note below, not here. */
    const int use_meth_orig = mem_use_native_regen(opt, ar, meth_orig_query);
    const char *regen_query = use_meth_orig ? meth_orig_query : query_;
    /* Reuse a per-thread nt4 scratch instead of a malloc/free per region. The
     * buffer is consumed within this call (copied into a.cigar via the DP), so
     * one live instance per thread is safe; grown to the high-water l_query. */
    static thread_local u8vec_scratch_t t_query;
    if (t_query.v.m < (size_t)l_query) kv_resize(uint8_t, t_query.v, (size_t)l_query);
    query = t_query.v.a;
    /* SAM-A10: only [qb,qe) of `query` is ever read (bwa_gen_cigar2 below is
     * handed &query[qb] with length qe-qb); the encode was over the whole read.
     * Encode only the consumed window — the untouched prefix/suffix are never
     * referenced, so output is byte-identical. */
    for (i = qb; i < qe; ++i) { // convert to the nt4 encoding
        unsigned char ch = (unsigned char) regen_query[i];
        query[i] = nst_nt4_decode(ch, 5);
    }
    a.mapq = ar->secondary < 0? mem_approx_mapq_se(opt, ar) : 0;
    if (ar->secondary >= 0) a.flag |= 0x100; // secondary alignment
    tmp = infer_bw(qe - qb, re - rb, ar->truesc, opt->a, opt->o_del, opt->e_del);
    w2  = infer_bw(qe - qb, re - rb, ar->truesc, opt->a, opt->o_ins, opt->e_ins);
    w2 = w2 > tmp? w2 : tmp;
    if (bwa_verbose >= 4) fprintf(stderr, "* Band width: inferred=%d, cmd_opt=%d, alnreg=%d\n", w2, opt->w, ar->w);
    if (w2 > opt->w) w2 = w2 < ar->w? w2 : ar->w;
    i = 0; a.cigar = 0;
    /* D3 (--meth, PR-5): native-alphabet output regen.
     *   - `regen_query` is the ORIGINAL read (meth_orig_query) under --meth, else
     *     the projected read for non-meth (legacy, unchanged).
     *   - `regen_mat` is the per-hypothesis ASYMMETRIC matrix under --meth so the
     *     CIGAR/gap shape matches the asymmetric extension that selected this
     *     region (using the symmetric matrix with the ORIGINAL read would
     *     re-penalize every conversion as a mismatch and can shift gaps); for
     *     non-meth it is the symmetric opt->mat (legacy, unchanged).
     *
     * NOTE (--meth NM/MD-vs-conversion policy): under --meth, NM/MD are derived
     * from `regen_mat`, not from literal base inequality — a column is a mismatch
     * iff the matrix penalises it (`bwa_gen_cigar3(..., nm_from_mat=1)`). Because
     * the asymmetric matrix scores the conversion cell as a match, a bisulfite
     * C→T (OT) / G→A (OB) at a ref-C/ref-G is a MATCH for NM/MD exactly as it
     * already is for the DP. One definition drives both; there is no bisulfite
     * special case in the NM/MD pass itself.
     *
     * Consequences, by design and by scoring mode:
     *   - Conversions never reach NM/MD. A perfectly converted read is NM:i:0,
     *     and MD does not enumerate every ref-C (which otherwise dominates the
     *     record; see issue #327).
     *   - `genomic` frees only the conversion direction, so a real variant in the
     *     opposite direction stays a mismatch and remains visible.
     *   - `collapsed` frees the mirror cell too, so C/T (and G/A) are fully
     *     interchangeable and real variants in that class are hidden as well —
     *     the documented cost of bwameth-compatible placement.
     *
     * A read alone cannot distinguish a bisulfite C→T from a real C→T SNP; that
     * aliasing is resolved downstream at the pileup, not here. The aligner
     * therefore reports what its scoring model treats as divergence rather than
     * adjudicating chemistry-vs-genotype per base. NOTE: this makes NM/MD
     * deliberately non-conformant with the SAM spec's "edit distance to the
     * reference" under --meth (the same divergence class as BISCUIT's NM and
     * bwameth's collapsed-space NM); the non-meth path is unchanged. */
    /* D3 (--meth, fix): the CIGAR regen must use the SAME strand-adjusted matrix
     * as the extension (see mem_alnreg_t.meth_strand_hyp). ar->rb is final here,
     * so derive is_rev directly (rb in doubled-pac; >= l_pac = reverse) and flip
     * the hypothesis; otherwise reverse-strand reads regen with the wrong matrix
     * and their conversions become CIGAR mismatches / soft-clips. */
    int regen_hyp = ar->meth_hypothesis;
    if (regen_hyp >= 0 && ar->rb >= bns->l_pac) regen_hyp ^= 1;
    const int8_t *regen_mat = use_meth_orig
        ? mem_opt_meth_mat(opt, regen_hyp) : opt->mat;
    do {
        free(a.cigar);
        w2 = w2 < opt->w<<2? w2 : opt->w<<2;
        a.cigar = bwa_gen_cigar3(regen_mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, w2, bns->l_pac, pac, qe - qb, (uint8_t*)&query[qb], rb, re, &score, &a.n_cigar, &NM, use_meth_orig);
        if (bwa_verbose >= 4) fprintf(stderr, "* Final alignment: w2=%d, global_sc=%d, local_sc=%d\n", w2, score, ar->truesc);
        if (score == last_sc || w2 == opt->w<<2) break; // it is possible that global alignment and local alignment give different scores
        last_sc = score;
        w2 <<= 1;
    } while (++i < 3 && score < ar->truesc - opt->a);
    assert(a.cigar != NULL);
    l_MD = strlen((char*)(a.cigar + a.n_cigar)) + 1;
    a.NM = NM;
    pos = bns_depos(bns, rb < bns->l_pac? rb : re - 1, &is_rev);
    a.is_rev = is_rev;
    if (a.n_cigar > 0) { // squeeze out leading or trailing deletions
        assert(a.cigar != NULL);
        if ((a.cigar[0]&0xf) == 2) {
            pos += a.cigar[0]>>4;
            --a.n_cigar;
            memmove(a.cigar, a.cigar + 1, a.n_cigar * 4 + l_MD);
        } else if ((a.cigar[a.n_cigar-1]&0xf) == 2) {
            --a.n_cigar;
            memmove(a.cigar + a.n_cigar, a.cigar + a.n_cigar + 1, l_MD); // MD needs to be moved accordingly
        }
    }
    if (qb != 0 || qe != l_query) { // add clipping to CIGAR
        int clip5, clip3;
        clip5 = is_rev? l_query - qe : qb;
        clip3 = is_rev? qb : l_query - qe;
        a.cigar = (uint32_t*) realloc(a.cigar, 4 * (a.n_cigar + 2) + l_MD);
        if (clip5) {
            memmove(a.cigar+1, a.cigar, a.n_cigar * 4 + l_MD); // make room for 5'-end clipping
            a.cigar[0] = clip5<<4 | 3;
            ++a.n_cigar;
        }
        if (clip3) {
            memmove(a.cigar + a.n_cigar + 1, a.cigar + a.n_cigar, l_MD); // make room for 3'-end clipping
            a.cigar[a.n_cigar++] = clip3<<4 | 3;
        }
    }
    /* SAM-A9: the region already carries its rid and bns_pos2rid(bns, pos) is
     * provably equal to it (the original code asserted exactly this). Use the
     * known value rather than recomputing. Output is byte-identical:
     * a.rid == ar->rid == old bns_pos2rid result.
     *
     * The equality is kept as a cross-check, but opt-in rather than under a
     * bare `assert`. Nothing in this build system defines NDEBUG, so an assert
     * here is live in every shipped binary -- meaning the recompute this
     * optimization exists to avoid ran on every emitted alignment, and the
     * saving the comment claimed was never realized anywhere. Enable with
     *   make EXTRA_CXXFLAGS=-DBWA_MEM3_DEBUG_RID_CHECK */
    a.rid = ar->rid;
#ifdef BWA_MEM3_DEBUG_RID_CHECK
    assert(a.rid == bns_pos2rid(bns, pos));
#endif
    a.pos = pos - bns->anns[a.rid].offset;
    a.score = ar->score; a.sub = ar->sub > ar->csub? ar->sub : ar->csub;
    a.is_alt = ar->is_alt; a.alt_sc = ar->alt_sc;
    /* query is the reused thread-local nt4 scratch — no free here. */
    return a;
}

/*****************************
 * Basic hit->SAM conversion *
 *****************************/

/* Infer the DP band width needed to realize an alignment of the given lengths and score.
 *
 * The l1 == l2 early-return is the load-bearing ungapped-CIGAR fast path. On an equal-length
 * window, closing any gap requires a balanced insertion + deletion (two gaps), so the guard
 * returns a zero band whenever the score deficit (l1*a - score) is below the implemented
 * two-gap threshold 2*(q + r - a): no balanced indel could recover that deficit, so the
 * optimal alignment is provably gap-free and a zero band suffices. Emission (mem_reg2aln)
 * feeds that 0 to bwa_gen_cigar3, which then takes bwa.cpp's no-DP equal-length block and
 * emits a single <len>M with no ksw_global2 fill or traceback. This shortcut is what makes the
 * common ungapped case skip the banded DP; keep it. */
static inline int infer_bw(int l1, int l2, int score, int a, int q, int r)
{
    int w;
    if (l1 == l2 && l1 * a - score < (q + r - a)<<1) return 0; // to get equal alignment length, we need at least two gaps
    w = ((double)((l1 < l2? l1 : l2) * a - score - q) / r + 2.);
    if (w < abs(l1 - l2)) w = abs(l1 - l2);
    return w;
}

static inline int get_rlen(int n_cigar, const uint32_t *cigar)
{
    int k, l;
    for (k = l = 0; k < n_cigar; ++k) {
        int op = cigar[k]&0xf;
        if (op == 0 || op == 2)
            l += cigar[k]>>4;
    }
    return l;
}

/************************ New functions, version2*****************************************/
#define _get_pac(pac, l) ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)

void seqbuf_capacity_fatal(const char *buf_name, const char *func,
                           int64_t current) {
    fprintf(stderr,
            "ERROR: %s in %s cannot grow past the int32 offset range "
            "(at %lld bytes).\n"
            "  The extension buffers are sized on a short-read model, so a "
            "block of very long reads\n"
            "  (HiFi/ONT) can exhaust them. Reduce the reads per block "
            "(lower -K, or fewer threads),\n"
            "  or split the input. Continuing would index outside the "
            "allocation.\n",
            buf_name, func, (long long)current);
    exit(EXIT_FAILURE);
}

void* _mm_realloc(void *ptr, int64_t csize, int64_t nsize, int16_t dsize) {
    if (nsize <= csize)
    {
        fprintf(stderr, "Shringking not supported yet.\n");
        return ptr;
    }
    void *nptr = _mm_malloc(nsize * dsize, 64);
    xassert(nptr != NULL, "out of memory: nptr");
    /* csize is in elements (matching nsize), not bytes — multiply by dsize so
     * callers passing dsize > 1 (sa_coord with sizeof(int64_t), matchArray
     * with sizeof(SMEM)) get the full pre-grow contents copied across, not
     * just the first csize bytes. Guard the NULL/empty lazy-init path
     * (matchArray[tid] starts NULL with wsize_mem[tid]=0). */
    if (ptr != NULL && csize > 0) {
        memcpy(nptr, ptr, (size_t)csize * (size_t)dsize);
    }
    _mm_free(ptr);

    return nptr;
}

inline void sortPairsLenExt(SeqPair *pairArray, int32_t count, SeqPair *tempArray,
                            int32_t *hist, int &numPairs128, int &numPairs16,
                            int &numPairs1, int score_a, int w, int zdrop)
{
    int32_t i;
    numPairs128 = numPairs16 = numPairs1 = 0;
    if (count <= 0) return;   /* empty extension side: nothing to sort/count */

    int32_t *hist2 = hist + MAX_SEQ_LEN8;
    int32_t *hist3 = hist + MAX_SEQ_LEN8 + MAX_SEQ_LEN16;

    /* The counting-sort bin space is three concatenated regions: 8-bit
     * (hist[0..MAX_SEQ_LEN8)), 16-bit (hist2[0..MAX_SEQ_LEN16)) and scalar
     * (hist3[0]). The 16-bit region alone is 32768 bins, so clearing +
     * cumulating the whole range every call was O(32768) irrespective of the
     * pair count. Window the 16-bit region to the actual minval range: a 16-bit
     * pair's bin is exactly its minval, so every touched bin lies within
     * [min minval, max minval]; bins below the window hold no pairs (they add 0
     * to the cross-region cumulative sum, so starting the sum at that offset is
     * exact) and bins above are never read. The sorted output and the numPairs*
     * tier counts are therefore byte-for-byte identical to the full-range sort.
     * The 8-bit region (1088 bins) is small, so it is still cleared in full to
     * avoid reasoning about its clamped bins. */
    int64_t mn = 0, mx = 0;
    for (i = 0; i < count; i++) {
        SeqPair sp = pairArray[i];
        int64_t minval = (int64_t)sp.h0 + (int64_t)min_(sp.len1, sp.len2) * (int64_t)score_a;
        if (i == 0) { mn = mx = minval; }
        else { if (minval < mn) mn = minval; if (minval > mx) mx = minval; }
    }
    int32_t w_lo = mn < 0 ? 0 : (mn >= MAX_SEQ_LEN16 ? MAX_SEQ_LEN16 - 1 : (int32_t)mn);
    int32_t w_hi = mx < 0 ? 0 : (mx >= MAX_SEQ_LEN16 ? MAX_SEQ_LEN16 - 1 : (int32_t)mx);

    for (i = 0; i < MAX_SEQ_LEN8; i++) hist[i] = 0;
    for (i = w_lo; i <= w_hi; i++)     hist2[i] = 0;
    hist3[0] = 0;

    /* minval is the counting-sort bin (sorts the 8-bit bucket by score for
     * lane packing). The tier decision is the safe envelope, not minval:
     * re-baselining handles arbitrary score, so a pair can be 8-bit even with
     * minval >= MAX_SEQ_LEN8. Clamp the bin to keep it in range — only the
     * ordering, not correctness, depends on the exact key. */
    /* BWAMEM3_DUMP_PAIRS observation buffer (empty unless the hook is armed).
     * Recording the tier from inside the branches below — rather than
     * re-deriving it afterwards — is deliberate: the dump is used to validate
     * external mirrors of this routing rule, so it must report the branch this
     * function actually took, not a second opinion that could drift from it. */
    std::string dump_buf;
    const bool dumping = bsw_dump_armed();
    if (dumping) dump_buf.reserve((size_t)count * 24);

    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        // int minval = sp.h0 + max_(sp.len1, sp.len2);
        int64_t minval = (int64_t)sp.h0 + (int64_t)min_(sp.len1, sp.len2) * (int64_t)score_a;
        if (bsw8_envelope_ok(sp.len1, sp.len2, w, score_a, zdrop, sp.h0)) {
            int bin = minval < 0 ? 0 : (minval < MAX_SEQ_LEN8 ? (int)minval : MAX_SEQ_LEN8 - 1);
            hist[bin]++;
            if (dumping) bsw_dump_pair(dump_buf, sp, w, 8);
        }
        else if(sp.len1 < MAX_SEQ_LEN16 && sp.len2 < MAX_SEQ_LEN16 && minval >= 0 && minval < MAX_SEQ_LEN16) {
            hist2[(int)minval] ++;
            if (dumping) bsw_dump_pair(dump_buf, sp, w, 16);
        }
        else {
            hist3[0] ++;
            if (dumping) bsw_dump_pair(dump_buf, sp, w, 1);
        }
    }
    if (dumping) bsw_dump_flush(dump_buf);

    int32_t cumulSum = 0;
    for(i = 0; i < MAX_SEQ_LEN8; i++)
    {
        int32_t cur = hist[i];
        hist[i] = cumulSum;
        cumulSum += cur;
    }
    /* Skipping hist2[0..w_lo) is exact: those bins were cleared to (and hold) no
     * counts, so they would contribute 0 to cumulSum anyway. */
    for(i = w_lo; i <= w_hi; i++)
    {
        int32_t cur = hist2[i];
        hist2[i] = cumulSum;
        cumulSum += cur;
    }
    hist3[0] = cumulSum;

    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        // int minval = sp.h0 + max_(sp.len1, sp.len2);
        int64_t minval = (int64_t)sp.h0 + (int64_t)min_(sp.len1, sp.len2) * (int64_t)score_a;

        if (bsw8_envelope_ok(sp.len1, sp.len2, w, score_a, zdrop, sp.h0))
        {
            int bin = minval < 0 ? 0 : (minval < MAX_SEQ_LEN8 ? (int)minval : MAX_SEQ_LEN8 - 1);
            int32_t pos = hist[bin];
            tempArray[pos] = sp;
            hist[bin]++;
            numPairs128 ++;
        }
        else if (sp.len1 < MAX_SEQ_LEN16 && sp.len2 < MAX_SEQ_LEN16 && minval >= 0 && minval < MAX_SEQ_LEN16) {
            int32_t pos = hist2[(int)minval];
            tempArray[pos] = sp;
            hist2[(int)minval]++;
            numPairs16 ++;
        }
        else {
            int32_t pos = hist3[0];
            tempArray[pos] = sp;
            hist3[0]++;
            numPairs1 ++;
        }
    }

    for(i = 0; i < count; i++) {
        pairArray[i] = tempArray[i];
    }
}

/* Counting-sort the tier's pairs so lanes packed into one SIMD group have
 * similar dimensions. `key_by_max` selects the sort key:
 *   false (16-bit tier): key = len1 (the historical bwa-mem2 behavior).
 *   true  (8-bit tier, EXT-13): key = max(len1, len2). The int8 kernel derives
 *     its per-group column count (`ncol`, from maxLen2) and gscore threshold
 *     (`minq`, from min len2) across the whole lane group, so packing purely by
 *     len1 leaves both pessimal whenever len2 varies. Keying on max(len1,len2)
 *     groups pairs of similar total extent, tightening ncol/minq.
 *
 * This changes only the ORDER in which pairs are packed into lane groups: each
 * pair's banded-SW result is computed per-lane from its own len1/len2/band, and
 * the caller scatters results back by the pair's unique (seqid, regid), so the
 * reorder is output-invariant (byte-identical). The counting sort remains stable
 * (equal keys keep input order) regardless of which key is chosen. */
inline void sortPairsLen(SeqPair *pairArray, int32_t count, SeqPair *tempArray, int32_t *hist,
                         bool key_by_max)
{

    int32_t i;
    if (count <= 0) return;

    /* Bins are keyed directly by the sort key, so only [minLen, maxLen] is ever
     * touched; clearing + cumulating the full [0, MAX_SEQ_LEN16] range every call
     * was O(32768) regardless of count. Window to the actual key range — output
     * is identical because every pair falls in [minLen, maxLen], bins below add
     * 0 to the cumulative sum, and bins above are never read. */
    int32_t key0 = key_by_max ? max_(pairArray[0].len1, pairArray[0].len2) : pairArray[0].len1;
    int32_t minLen = key0, maxLen = key0;
    for(i = 1; i < count; i++)
    {
        int32_t L = key_by_max ? max_(pairArray[i].len1, pairArray[i].len2) : pairArray[i].len1;
        if (L < minLen) minLen = L;
        if (L > maxLen) maxLen = L;
    }
    for(i = minLen; i <= maxLen; i++) hist[i] = 0;

    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        int32_t key = key_by_max ? max_(sp.len1, sp.len2) : sp.len1;
        hist[key]++;
    }
    int32_t cumulSum = 0;
    for(i = minLen; i <= maxLen; i++)
    {
        int32_t cur = hist[i];
        hist[i] = cumulSum;
        cumulSum += cur;
    }

    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        int32_t key = key_by_max ? max_(sp.len1, sp.len2) : sp.len1;
        int32_t pos = hist[key];

        tempArray[pos] = sp;
        hist[key]++;
    }

    for(i = 0; i < count; i++) {
        pairArray[i] = tempArray[i];
    }
}

/* ------------------------------------------------------------------------- *
 * D3 (--meth, A1/PR-4): per-hypothesis banded-SW tier dispatch.
 *
 * A worker batch interleaves R1 (OT, C->T) and R2 (OB, G->A) reads, so a single
 * per-object substitution matrix cannot score every pair in the batch correctly
 * (the PR-3 placeholder fell back to the SYMMETRIC matrix for mixed batches).
 * bsw_run_tier() partitions a tier's pair array in place by the owning alnreg's
 * meth_hypothesis (OT = odd, OB = even, none = <0) and runs the matching SW
 * object on each contiguous run, so each pair is scored in original space with
 * its asymmetric OT/OB matrix. Each run is single-hypothesis, so the kernel's
 * rank-1 fast path still applies. The kernel re-sorts pairs by length internally
 * and the retry loop reads results back via seqid/regid, so the in-place reorder
 * is safe.
 *
 * Outside --meth the partition is skipped and the call is byte-for-byte the
 * original single kernel invocation on the symmetric object.
 * ------------------------------------------------------------------------- */
enum BswMethTier { BSW_TIER_SCALAR, BSW_TIER_16, BSW_TIER_8 };

/* One tier kernel call over pa[0..n). `sort_scratch` must be a SeqPair buffer
 * DISTINCT from `pa` (sortPairsLen is a counting sort that scatters through it;
 * aliasing pa would corrupt the sort). Scalar tier ignores sort_scratch. */
static inline void bsw_score_batch(BswMethTier tier, IBandedPairWiseSW *bsw,
        SeqPair *pa, uint8_t *ref, uint8_t *qer, int n, int nthreads, int32_t w,
        SeqPair *sort_scratch, int32_t *hist)
{
    switch (tier) {
    case BSW_TIER_SCALAR:
        bsw->scalarBandedSWAWrapper(pa, ref, qer, n, nthreads, w);
        break;
    case BSW_TIER_16:
#if !HAVE_BSW_VECTOR_8_16
        bsw->scalarBandedSWAWrapper(pa, ref, qer, n, nthreads, w);
#else
        sortPairsLen(pa, n, sort_scratch, hist, /*key_by_max=*/false);
        bsw->getScores16(pa, ref, qer, n, nthreads, w);
#endif
        break;
    case BSW_TIER_8:
#if !HAVE_BSW_VECTOR_8_16
        bsw->scalarBandedSWAWrapper(pa, ref, qer, n, nthreads, w);
#else
        /* EXT-13: pack the int8 lane groups by max(len1,len2) so ncol/minq are
         * tight (len2 varies within a len1-keyed group). Byte-identical: only
         * the packing order changes; results scatter back by (seqid,regid). */
        sortPairsLen(pa, n, sort_scratch, hist, /*key_by_max=*/true);
        /* int8 diagonal encoding can't represent a band >= 128; divert to the
         * 16-bit kernel once the doubling retry pushes w past the envelope. */
        if (w <= BSW8_MAX_W) bsw->getScores8(pa, ref, qer, n, nthreads, w);
        else                 bsw->getScores16(pa, ref, qer, n, nthreads, w);
#endif
        break;
    }
}

struct BswJobResult { int32_t score, tle, qle, gscore, gtle, max_off; };

static inline void bsw_apply_result(SeqPair &sp, const BswJobResult &r)
{
    sp.score = r.score; sp.tle = r.tle; sp.qle = r.qle;
    sp.gscore = r.gscore; sp.gtle = r.gtle; sp.max_off = r.max_off;
}

enum { BSW_DEDUP_OFF = 0, BSW_DEDUP_ON = 1, BSW_DEDUP_AUTO = 2 };
/* Parse a dedup mode string. Returns -1 on an unrecognized value. Legacy
 * numeric values from the experimental phase are accepted for pipeline
 * compatibility: "1"/"perbatch" -> on; "4"/"net" -> auto; "2"(crossbatch) and
 * "3"(adaptive) were removed -- they map to auto with a warning. */
static int bsw_parse_dedup_mode(const char *e)
{
    if (!strcmp(e, "0") || !strcmp(e, "off"))                            return BSW_DEDUP_OFF;
    if (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "perbatch"))   return BSW_DEDUP_ON;
    if (!strcmp(e, "auto") || !strcmp(e, "net") || !strcmp(e, "4"))      return BSW_DEDUP_AUTO;
    if (!strcmp(e, "2") || !strcmp(e, "3")) {
        fprintf(stderr, "[dedup] mode '%s' (crossbatch/adaptive) was removed; using 'auto'\n", e);
        return BSW_DEDUP_AUTO;
    }
    return -1;
}
/* Extension-DP dedup configuration store. mode -1 means "unresolved" -- the
 * only valid state before mem_dedup_configure() has run once; every reader
 * goes through bsw_dedup_cfg(), which lazily resolves via mem_dedup_configure
 * if some entry point (unit binaries, library use) never called it. */
struct BswDedupCfg { int mode; double z; int64_t reprobe_jobs; };
static BswDedupCfg g_dedup_cfg = { -1, 2.0, 12000000 };   /* mode -1 = unresolved */

void mem_dedup_configure(const char *mode_arg)
{
    /* A non-empty --dedup wins outright (and overrides even an explicitly-empty
     * BWAMEM3_DEDUP); otherwise consult the env, distinguishing "unset" (use the
     * default) from "set but empty" (an explicit misconfiguration -> fatal, so a
     * cleared BWAMEM3_DEDUP= cannot silently fall back to the default). An empty
     * CLI value is rejected earlier by the getopt handler. */
    if (mode_arg && *mode_arg) {
        const int v = bsw_parse_dedup_mode(mode_arg);
        if (v < 0) { fprintf(stderr, "ERROR: --dedup: expected off|on|auto, got '%s'\n", mode_arg); exit(1); }
        g_dedup_cfg.mode = v;
    } else {
        const char *m = getenv("BWAMEM3_DEDUP");
        if (m == NULL) g_dedup_cfg.mode = BSW_DEDUP_AUTO;   /* unset -> default */
        else {
            const int v = bsw_parse_dedup_mode(m);          /* "" -> -1 -> fatal */
            if (v < 0) { fprintf(stderr, "ERROR: BWAMEM3_DEDUP: expected off|on|auto, got '%s'\n", m); exit(1); }
            g_dedup_cfg.mode = v;
        }
    }
    const char *z = getenv("BWAMEM3_DEDUP_Z");            /* expert knob, env-only */
    if (z) {                                             /* set (empty -> rejected by the full-string parse) */
        char        *end = NULL;
        errno = 0;
        const double zv  = strtod(z, &end);
        /* Full-string parse: reject trailing junk ("2x") rather than silently
         * taking the leading number, the way atof would. */
        if (end == z || end == NULL || *end != '\0' || errno == ERANGE || !std::isfinite(zv) || zv <= 0.0) {
            fprintf(stderr, "ERROR: BWAMEM3_DEDUP_Z: must be a finite number > 0, got '%s'\n", z);
            exit(1);
        }
        g_dedup_cfg.z = zv;
    }
    const char *r = getenv("BWAMEM3_DEDUP_REPROBE");      /* expert knob, env-only */
    if (r) {                                             /* set (empty -> rejected by the full-string parse) */
        char           *end = NULL;
        errno = 0;
        const long long rv  = strtoll(r, &end, 10);
        /* Full-string parse: reject trailing junk ("12M") rather than silently
         * taking the leading number, the way atoll would. */
        if (end == r || end == NULL || *end != '\0' || errno == ERANGE || rv < 0) {
            fprintf(stderr, "ERROR: BWAMEM3_DEDUP_REPROBE: must be a non-negative integer number of jobs (0 = off), got '%s'\n", r);
            exit(1);
        }
        g_dedup_cfg.reprobe_jobs = rv;
    }
}
/* Lazy resolution for entry points that never call mem_dedup_configure()
 * (unit binaries, library use): resolve from env/defaults on first touch. */
static inline const BswDedupCfg &bsw_dedup_cfg(void)
{
    static std::once_flag once;
    std::call_once(once, []() { if (g_dedup_cfg.mode < 0) mem_dedup_configure(NULL); });
    return g_dedup_cfg;
}
static inline int bsw_dedup_mode(void)
{
    return bsw_dedup_cfg().mode;
}

/* ---------------------------------------------------------------------------
 * Net-cycles adaptive dedup (BWAMEM3_DEDUP=auto): decide ON/OFF from
 * MEASURED wall time, not a dup-rate threshold. Dedup wins exactly when the
 * bookkeeping it adds (fingerprint + map build + result scatter) costs less
 * than the kernel time it removes by scoring m distinct reps instead of n. Both
 * sides are observable per batch, so the decision is the sign of a measured net:
 *
 *   overhead_ns = whole dedup-wrapper time  -  kernel time on the m reps
 *   c_cell      = kernel_ns / dp_cells       (this host's real per-cell DP cost)
 *   benefit_ns  = cells_saved * c_cell       (cells_saved = total_cells - dp_cells)
 *   net_ns      = overhead_ns - benefit_ns   (< 0  =>  dedup won this batch)
 *
 * c_cell is measured, so the break-even self-calibrates to CPU / compiler /
 * SIMD tier / substrate -- there is no magic dup-rate constant. We run dedup ON
 * while measuring (real data is dup-rich, so ON is the productive default and
 * pays ~zero tax; instrumenting a batch we already run is two clock reads), pool
 * per-batch net/job across threads (Welford), and latch the moment a two-sided
 * z-test clears |z| >= NET_Z (a dimensionless confidence level, not a rate).
 * mean net < 0 -> latch ON (keep dedup); > 0 -> latch OFF (flip to plain score).
 * Below NET_MIN_BATCHES we never latch (CLT warmup); past NET_MAX_JOBS with no
 * significance the data is at break-even (cost ~0 either way) so we keep ON.
 * The running net sum is the warmup tax in nanoseconds, reported at latch. */
enum { NET_MEASURING = 0, NET_LATCH_ON = 1, NET_LATCH_OFF = 2 };
static std::atomic<int>      g_net_state{NET_MEASURING};
static const uint64_t        NET_MIN_BATCHES = 24;         /* CLT floor before any latch */
static const uint64_t        NET_MAX_JOBS    = 3000000;    /* give up measuring past this */
/* Confidence level for the latch, overridable (the only knob, and a benign one:
 * a wrong value costs a little decision latency, never a wrong ON/OFF call). */
static inline double net_z_crit(void)
{
    return bsw_dedup_cfg().z;
}
/* Periodic re-probe (C4): once latched, the controller stays ON or OFF
 * forever unless re-checked. `g_net_incumbent` records the current latched
 * direction (0 = none yet, i.e. still in the initial MEASURING phase) so a
 * re-probe can tell "confirm" from "flip" apart, and `g_net_since_latch`
 * counts jobs run under the current latch toward the next re-probe cadence
 * (BWAMEM3_DEDUP_REPROBE jobs; cfg in C5). */
static std::atomic<int>     g_net_incumbent{0};     /* 0 = none (initial phase) */
static std::atomic<int64_t> g_net_since_latch{0};   /* jobs since last latch */
static const uint64_t       NET_PROBE_MAX_JOBS = 1500000; /* probe cap -> keep incumbent */
static inline int64_t bsw_dedup_reprobe_jobs(void); /* env BWAMEM3_DEDUP_REPROBE, default 12000000, 0=off */
static inline void net_maybe_reprobe(int n);
/* Pooled measurement state, guarded by g_net_mtx (held for a few doubles per
 * instrumented batch; batches are milliseconds, so contention is negligible). */
static std::mutex   g_net_mtx;
static uint64_t     g_net_k    = 0;    /* instrumented batches */
static uint64_t     g_net_jobs = 0;    /* jobs measured */
static double       g_net_mean = 0.0;  /* Welford mean of net_ns/job */
static double       g_net_m2   = 0.0;  /* Welford sum of squares */
static double       g_net_sum  = 0.0;  /* raw sum of net_ns (the warmup tax) */

/* Feed one measured batch into the controller and latch when significant.
 * Incumbent-aware (C4): the initial decision (incumbent == 0) uses a plain
 * z-test with the MAX_JOBS break-even fallback, same as before. A re-probe
 * (incumbent != 0) instead asks whether the NEW evidence still supports the
 * incumbent direction: agreement only needs to *confirm* at the ordinary z
 * (or is kept on an inconclusive probe-cap timeout); disagreement needs a
 * higher bar (z+1) to *flip* the incumbent, so noise near break-even cannot
 * flap the decision back and forth every cadence. */
static inline void net_observe(double net_ns, int n)
{
    if (n <= 0) return;
    std::lock_guard<std::mutex> lk(g_net_mtx);
    if (g_net_state.load(std::memory_order_relaxed) != NET_MEASURING) return;
    g_net_sum  += net_ns;
    g_net_jobs += (uint64_t)n;
    g_net_k    += 1;
    const double d  = net_ns - g_net_mean;            /* Welford, per-BATCH statistic */
    g_net_mean += d / (double)g_net_k;
    g_net_m2   += d * (net_ns - g_net_mean);
    const int incumbent = g_net_incumbent.load(std::memory_order_relaxed);
    int decision = 0; const char *why = "";
    if (g_net_k >= NET_MIN_BATCHES) {
        const double var = g_net_m2 / (double)(g_net_k - 1);
        const double sd  = var > 0.0 ? sqrt(var) : 0.0;
        const double z   = sd > 0.0 ? g_net_mean * sqrt((double)g_net_k) / sd : 0.0;
        const int    fav = (g_net_mean < 0.0) ? NET_LATCH_ON : NET_LATCH_OFF;
        const double zc  = net_z_crit();
        if (incumbent == 0) {                       /* initial: plain test, cap -> ON */
            if      (fabs(z) >= zc)               { decision = fav;           why = "measured"; }
            else if (g_net_jobs >= NET_MAX_JOBS)  { decision = NET_LATCH_ON;  why = "break-even, keeping default"; }
        } else if (fav == incumbent) {              /* re-probe, evidence agrees */
            if      (fabs(z) >= zc)                       { decision = incumbent; why = "re-probe confirmed"; }
            else if (g_net_jobs >= NET_PROBE_MAX_JOBS)    { decision = incumbent; why = "re-probe inconclusive, kept"; }
        } else {                                    /* re-probe, evidence opposes: flip bar is z+1 */
            if      (fabs(z) >= zc + 1.0)                 { decision = fav;       why = "re-probe FLIPPED"; }
            else if (g_net_jobs >= NET_PROBE_MAX_JOBS)    { decision = incumbent; why = "below flip margin, kept"; }
        }
    }
    if (decision) {
        g_net_incumbent.store(decision, std::memory_order_relaxed);
        g_net_since_latch.store(0, std::memory_order_relaxed);
        g_net_state.store(decision, std::memory_order_relaxed);
        if (bwa_verbose >= 3)
            fprintf(stderr, "[dedup-auto] %s: %llu jobs / %llu batches, mean net=%.1f us/batch -> dedup %s\n",
                    why, (unsigned long long)g_net_jobs, (unsigned long long)g_net_k,
                    g_net_mean / 1e3, decision == NET_LATCH_ON ? "ON" : "OFF");
    }
}

/* Env-configured re-probe cadence, in jobs (same style as net_z_crit): 0 or a
 * negative/unset value falls back to the default; 0 explicitly disables
 * re-probing (net_maybe_reprobe short-circuits on period <= 0). */
static inline int64_t bsw_dedup_reprobe_jobs(void)
{
    return bsw_dedup_cfg().reprobe_jobs;
}

/* Work-based re-probe: called from BOTH latched paths with this batch's job
 * count. When the cadence elapses, reset the accumulator and re-enter
 * MEASURING. All stats mutation is under g_net_mtx; the reset happens before
 * the state store, so no thread can feed a stale accumulator. Byte-identity is
 * unaffected: a probe only changes when the kernel is skipped. */
static inline void net_maybe_reprobe(int n)
{
    const int64_t period = bsw_dedup_reprobe_jobs();
    if (period <= 0) return;
    if (g_net_since_latch.fetch_add(n, std::memory_order_relaxed) + n < period) return;
    std::lock_guard<std::mutex> lk(g_net_mtx);
    const int st = g_net_state.load(std::memory_order_relaxed);
    if (st == NET_MEASURING) return;                                   /* probe already running */
    if (g_net_since_latch.load(std::memory_order_relaxed) < period) return;  /* lost the race */
    g_net_k = 0; g_net_jobs = 0; g_net_mean = 0.0; g_net_m2 = 0.0; g_net_sum = 0.0;
    g_net_since_latch.store(0, std::memory_order_relaxed);
    if (bwa_verbose >= 3)
        fprintf(stderr, "[dedup-auto] re-probe (incumbent %s)\n", st == NET_LATCH_ON ? "ON" : "OFF");
    g_net_state.store(NET_MEASURING, std::memory_order_relaxed);
}

/* Fold up to LIMIT bytes from each END of `s` into `h`, eight bytes (one 64-bit
 * word) at a time. Fingerprinting only the ends as words -- rather than hashing
 * every byte scalar -- is what keeps dedup's cost a few percent of a job's DP
 * instead of a third of it: a full byte-wise hash over the ~230-byte target
 * would eat most of what the dedup saves. Fingerprint collisions are harmless
 * because every candidate is byte-verified (bsw_job_eq / cache memcmp) before a
 * result is reused, so this only needs to separate distinct jobs *well*, not
 * perfectly. */
static inline void bsw_hash_fold(uint64_t &h, const uint8_t *s, int len)
{
    const int LIMIT = 32;
    auto mix = [&](const uint8_t *p, int n) {
        int k = 0;
        for (; k + 8 <= n; k += 8) {
            uint64_t v; memcpy(&v, p + k, 8);
            h ^= v; h *= 1099511628211ULL;
        }
        if (k < n) {                                   /* < 8-byte tail */
            uint64_t v = 0;
            for (int i = 0; k + i < n; i++) v |= (uint64_t)p[k + i] << (8 * i);
            h ^= v; h *= 1099511628211ULL;
        }
    };
    if (len <= 2 * LIMIT) mix(s, len);
    else { mix(s, LIMIT); mix(s + len - LIMIT, LIMIT); }
}

static inline uint64_t bsw_job_hash(const uint8_t *s1, int len1,
                                    const uint8_t *s2, int len2, int h0)
{
    uint64_t h = 1469598103934665603ULL;              /* FNV-1a offset basis */
    h ^= ((uint64_t)(uint32_t)len1 << 32) | (uint32_t)len2;
    h *= 1099511628211ULL;
    h ^= (uint64_t)(uint32_t)h0;
    h *= 1099511628211ULL;
    bsw_hash_fold(h, s1, len1);
    bsw_hash_fold(h, s2, len2);
    return h;
}

/* BWAMEM3_DEDUP_STATS=1: count distinct-within-batch (m) vs total (n) extension
 * DP jobs and print the batch dup rate once at exit. Measurement only; a single
 * cached-bool branch per batch, inert unless armed. */
static std::atomic<uint64_t> g_dedup_total{0};
static std::atomic<uint64_t> g_dedup_distinct{0};
static inline bool bsw_dedup_stats_on(void)
{
    static const bool on = []() {
        const char *e = getenv("BWAMEM3_DEDUP_STATS");
        return e && *e && !(e[0] == '0' && e[1] == '\0');
    }();
    return on;
}
static struct BswDedupStatsDumper {
    ~BswDedupStatsDumper() {
        if (!bsw_dedup_stats_on()) return;
        uint64_t t = g_dedup_total.load(), d = g_dedup_distinct.load();
        fprintf(stderr,
            "[dedup-stats] total_jobs=%llu distinct_jobs=%llu perbatch_dup_rate=%.4f\n",
            (unsigned long long)t, (unsigned long long)d,
            t ? 1.0 - (double)d / (double)t : 0.0);
    }
} g_bsw_dedup_stats_dumper;

/* BWAMEM3_EXT_TIME=1: sum the wall time spent inside bsw_tier_kernel (the
 * extension dedup + banded-SW path) across all worker threads, print at exit.
 * Dedup changes only affect this phase, so a small-read smoke run reports the
 * extension delta directly instead of burying it under the fixed index-load +
 * I/O wall. Sums thread-time, which is fine for A/B ratios. Inert unless armed. */
static std::atomic<uint64_t> g_ext_ns{0};
static inline bool bsw_ext_time_on() {
    static const bool on = []() {
        const char *e = getenv("BWAMEM3_EXT_TIME");
        return e && *e && !(e[0] == '0' && e[1] == '\0');
    }();
    return on;
}
static struct BswExtTimeDumper {
    ~BswExtTimeDumper() {
        if (bsw_ext_time_on())
            fprintf(stderr, "[ext-time] bsw_tier_kernel_thread_ms=%.1f\n",
                    g_ext_ns.load() / 1e6);
    }
} g_bsw_ext_time_dumper;
struct BswExtTimer {
    std::chrono::steady_clock::time_point t0;
    bool on;
    BswExtTimer() : on(bsw_ext_time_on()) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~BswExtTimer() {
        if (on) g_ext_ns.fetch_add(
            (uint64_t)std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - t0).count(),
            std::memory_order_relaxed);
    }
};

/* Dedup wrapper over bsw_score_batch: score each distinct (seq1,seq2,h0) job
 * once within this batch and copy its result to the repeats. Byte-identical to
 * scoring all n: every candidate sharing a fingerprint is byte-verified
 * (memcmp on both sequences) before its result is reused, so a fingerprint
 * collision can only cost an extra memcmp, never a wrong or lost dedup. */
static inline void bsw_tier_kernel(BswMethTier tier, IBandedPairWiseSW *bsw,
        SeqPair *pa, uint8_t *ref, uint8_t *qer, int n, int nthreads, int32_t w,
        SeqPair *sort_scratch, int32_t *hist)
{
    if (n <= 0) return;
    BswExtTimer _ext_timer;           /* inert unless BWAMEM3_EXT_TIME set */
    const int mode = bsw_dedup_mode();
    if (mode == BSW_DEDUP_OFF || n < 2) {
        bsw_score_batch(tier, bsw, pa, ref, qer, n, nthreads, w, sort_scratch, hist);
        return;
    }
    if (mode == BSW_DEDUP_AUTO) {
        net_maybe_reprobe(n);
        if (g_net_state.load(std::memory_order_relaxed) == NET_LATCH_OFF) {
            bsw_score_batch(tier, bsw, pa, ref, qer, n, nthreads, w, sort_scratch, hist);
            return;   /* net-cycles controller measured dedup net-negative here */
        }
    }

    /* net-cycles controller: while measuring (AUTO), run the dedup path AND
     * time it, so the ON/OFF decision comes from observed net wall (see
     * net_observe). net_t0 spans everything the OFF path would NOT do; the
     * kernel call is timed separately and subtracted out to isolate the
     * dedup-wrapper overhead from the DP-kernel cost. */
    const bool net_measure = (mode == BSW_DEDUP_AUTO
                        && g_net_state.load(std::memory_order_relaxed) == NET_MEASURING);
    std::chrono::steady_clock::time_point net_t0;
    int64_t net_total_cells = 0, net_dp_cells = 0;
    double  net_kernel_ns = 0.0;
    /* Banded-SW cost model: a job computes ~ len2 * band cells, NOT len1*len2 --
     * the band (2w+1, capped by the target length) is far narrower than len1, so
     * len1*len2 wildly over-weights long targets and skews the per-cell rate,
     * biasing the ON/OFF decision toward OFF. min(len1, 2w+1) is the true DP width. */
    const int64_t net_band = 2 * (int64_t)w + 1;
    if (net_measure) net_t0 = std::chrono::steady_clock::now();

    static thread_local std::vector<int>          miss_grp;    /* per pair: miss-group, or -1 if cache-resolved */
    static thread_local std::vector<SeqPair>      repbuf;      /* one rep per distinct miss (to score) */
    static thread_local std::vector<int>          next_grp;    /* per miss-group: next group (1-based) in its hash chain, 0 = end (intrusive bucket) */
    static thread_local std::vector<BswJobResult> miss_res;    /* per miss-group: scored result */
    static thread_local robin_hood::unordered_flat_map<uint64_t,int> missidx; /* content hash -> head miss-group of the chain (this batch) */

    miss_grp.assign(n, -1);
    repbuf.clear();
    next_grp.clear();
    missidx.clear();

    for (int i = 0; i < n; i++) {
        const uint8_t *s1 = ref + pa[i].idr; const int len1 = pa[i].len1;
        const uint8_t *s2 = qer + pa[i].idq; const int len2 = pa[i].len2;
        const int      h0 = pa[i].h0;
        if (net_measure) {
            const int64_t cells = (int64_t)len2 * ((int64_t)len1 < net_band ? (int64_t)len1 : net_band);
            net_total_cells += cells;
        }
        const uint64_t hv = bsw_job_hash(s1, len1, s2, len2, h0);

        /* miss: dedup against distinct misses already seen in THIS batch. Every
         * candidate sharing this fingerprint is byte-compared (walking the
         * intrusive chain), so a fingerprint collision costs an extra memcmp --
         * it never SPLITS a true group and loses the dedup, which a single-slot
         * map would on collision. */
        int g = -1;
        int &head = missidx[hv];        /* value-inits to 0 on a new key; groups are stored 1-based, 0 = empty chain */
        for (int cand = head; cand > 0; cand = next_grp[cand - 1]) {
            const SeqPair &rep = repbuf[cand - 1];
            if (rep.len1 == len1 && rep.len2 == len2 && rep.h0 == h0
                && memcmp(ref + rep.idr, s1, (size_t)len1) == 0
                && memcmp(qer + rep.idq, s2, (size_t)len2) == 0) { g = cand - 1; break; }
        }
        if (g < 0) {
            g = (int)repbuf.size();
            repbuf.push_back(pa[i]);
            next_grp.push_back(head);   /* prepend: new group's chain-next is the old head (0 = end) */
            head = g + 1;               /* store 1-based so a fresh map slot (0) reads as "empty" */
        }
        miss_grp[i] = g;
    }

    const int m = (int)repbuf.size();
    if (bsw_dedup_stats_on()) {   /* count distinct-within-batch vs total */
        g_dedup_total.fetch_add((uint64_t)n, std::memory_order_relaxed);
        g_dedup_distinct.fetch_add((uint64_t)m, std::memory_order_relaxed);
    }
    if (m > 0) {
        /* tag each rep's group in the unused `id` field before scoring permutes
         * repbuf. The kernels write `id` only on the padding lanes
         * (pairArray[ii].id = ii for ii >= numPairs), never on a real slot, and
         * sortPairsLen moves whole SeqPairs -- so id survives the reorder and
         * identifies each scored rep's group with no second map (the old
         * (seqid,regid) repkey2grp). */
        for (int g = 0; g < m; g++) {
            if (net_measure) { const int64_t l1 = repbuf[g].len1;
                net_dp_cells += (int64_t)repbuf[g].len2 * (l1 < net_band ? l1 : net_band); }
            repbuf[g].id = g;
        }

        /* The kernels write and (guardedly) prefetch padding lanes past
         * numPairs; the caller's seqPair buffers are ALWAYS allocated as
         * (numPairs-bound + MAX_LINE_LEN), so every kernel is written against
         * that guaranteed slack. repbuf must honor the same contract -- a tight
         * roundup(m, SIMD_WIDTH8) is not enough and overruns the heap. Value-
         * initialization zeroes len1/len2/idr/idq so the slack lanes stay inert. */
        repbuf.resize((size_t)((m + SIMD_WIDTH8 - 1) / SIMD_WIDTH8) * SIMD_WIDTH8
                      + MAX_LINE_LEN);

        std::chrono::steady_clock::time_point net_k0;
        if (net_measure) net_k0 = std::chrono::steady_clock::now();
        bsw_score_batch(tier, bsw, repbuf.data(), ref, qer, m, nthreads, w, sort_scratch, hist);
        if (net_measure)
            net_kernel_ns = std::chrono::duration<double, std::nano>(
                                std::chrono::steady_clock::now() - net_k0).count();

        /* map each scored rep back to its miss-group by the `id` tag set above.
         * `id` is kernel-controlled: guard it before indexing so a tier that
         * violates the "id survives untouched on real slots" contract fails
         * loudly (xassert fires under -DNDEBUG too) instead of scattering out of
         * bounds and corrupting the heap. */
        miss_res.resize(m);
        for (int k = 0; k < m; k++) {
            const int32_t rk = repbuf[k].id;
            xassert(rk >= 0 && rk < m, "dedup scatter: kernel-controlled repbuf.id out of [0, m)");
            miss_res[rk] =
                BswJobResult{repbuf[k].score, repbuf[k].tle, repbuf[k].qle,
                             repbuf[k].gscore, repbuf[k].gtle, repbuf[k].max_off};
        }

        for (int i = 0; i < n; i++)   /* apply miss results to their pending pairs */
            if (miss_grp[i] >= 0) bsw_apply_result(pa[i], miss_res[miss_grp[i]]);
    }

    /* net-cycles decision: overhead = whole dedup wrapper minus the kernel; the
     * benefit is the kernel time the skipped duplicate cells would have cost, at
     * this batch's measured per-cell rate. Feed the signed net to the controller. */
    if (net_measure && net_dp_cells > 0 && net_kernel_ns > 0.0) {
        const double total_ns = std::chrono::duration<double, std::nano>(
                                    std::chrono::steady_clock::now() - net_t0).count();
        const double overhead_ns  = total_ns - net_kernel_ns;
        const double c_cell       = net_kernel_ns / (double)net_dp_cells;
        const double cells_saved  = (double)(net_total_cells - net_dp_cells);
        const double benefit_ns   = cells_saved * c_cell;
        net_observe(overhead_ns - benefit_ns, n);
    }
}

/* Dispatch one tier of `nump` pairs. Non-meth: a single call on `sym` with the
 * original `base_scratch` (= seqPairArrayAux). Meth: 3-way in-place partition by
 * hypothesis, then one call per non-empty run on the matching object, using
 * `meth_scratch` (= the retry loop's pair_ar_aux, always distinct from pair_ar)
 * as the sort scratch. */
static inline void bsw_run_tier(BswMethTier tier, const mem_opt_t *opt,
        const mem_alnreg_v *av_v,
        IBandedPairWiseSW *sym, IBandedPairWiseSW *ot, IBandedPairWiseSW *ob,
        SeqPair *pair_ar, uint8_t *ref, uint8_t *qer, int nump,
        int nthreads, int32_t w,
        SeqPair *base_scratch, SeqPair *meth_scratch, int32_t *hist)
{
    if (nump <= 0) return;
    if (!opt->meth_mode) {
        bsw_tier_kernel(tier, sym, pair_ar, ref, qer, nump, nthreads, w,
                        base_scratch, hist);
        return;
    }
    /* Dutch-flag partition into [OT (key 0) | OB (key 1) | SYM (key 2)]. Order
     * within/among runs is irrelevant: the kernel re-sorts by length and the
     * caller maps each result back to its alnreg by seqid/regid. */
    int lo = 0, mid = 0, hi = nump - 1;
    while (mid <= hi) {
        const SeqPair *sp = &pair_ar[mid];
        /* strand-adjusted hypothesis (see mem_alnreg_t.meth_strand_hyp): the
         * extension matrix must match the conversion direction AS SEEN against
         * the (possibly RC) reference window, not the raw genome-strand hypothesis. */
        int h = av_v[sp->seqid].a[sp->regid].meth_strand_hyp;
        int key = (h < 0) ? 2 : ((h & 1) ? 0 : 1);
        if (key == 0)      { SeqPair t = pair_ar[lo]; pair_ar[lo]  = pair_ar[mid]; pair_ar[mid] = t; lo++; mid++; }
        else if (key == 1) { mid++; }
        else               { SeqPair t = pair_ar[mid]; pair_ar[mid] = pair_ar[hi]; pair_ar[hi]  = t; hi--; }
    }
    int n_ot = lo, n_ob = mid - lo, n_sym = nump - mid;
    bsw_tier_kernel(tier, ot,  pair_ar,       ref, qer, n_ot,  nthreads, w, meth_scratch, hist);
    bsw_tier_kernel(tier, ob,  pair_ar + lo,  ref, qer, n_ob,  nthreads, w, meth_scratch, hist);
    bsw_tier_kernel(tier, sym, pair_ar + mid, ref, qer, n_sym, nthreads, w, meth_scratch, hist);
}

/* The rung's finalize blocks call the ugp_record_*_outcome profiling hooks, which are
 * defined further below (after the rung). Those calls compile only under the profiling
 * build (-DBWAMEM3_UGP_PROFILE=1, off by default), so forward-declare them here to avoid
 * a use-before-definition under that flag. No effect on the default (non-profiling) build. */
#if BWAMEM3_UGP_PROFILE
static inline void ugp_record_left_outcome(const SeqPair *sp, int a_match, int tid);
static inline void ugp_record_right_outcome(const SeqPair *sp, int tid);
#endif

/* ------------------------------------------------------------------------
 * Certified tight_band probe rung (tier-parameterized; see `tier`).
 *
 * 8-bit tier: its ladder starts at opt->w, so short-read extensions that carry a proven
 * tight_band still pay the full-width band. This rung gives it a narrow probe it lacked.
 * 16-bit tier: its ladder already probes narrow (first rung INIT_W(opt->w) = 20); here the
 * rung REPLACES that single uniform-20 probe with a per-w_need bucketed one and the caller
 * then starts the ladder at opt->w. Both are byte-identical by the same certificate, and both
 * feed the identical downstream ladder. This rung runs the eligible pairs at the probe width
 * BEFORE the ladder and finalizes only those band_cert_accept certifies there --
 * byte-identical to the full-width ladder result, by the same certificate the
 * other tiers ship (band_cert_accept's w < opt->w branch: the S - pen_clip anchor
 * pins score/qle/tle/gscore/gtle and the clip decision; mem_band_cert_params_safe
 * -- implied by opt->band_cert, see fastmap.cpp -- pins the control flow, and its
 * w0 = 20 conditions cover a 20-wide probe verbatim; wider bucket ceilings re-check
 * that envelope at their own width via mem_band_cert_params_safe_w, see Gate B below).
 *
 * Eligibility (gate A): opt->band_cert AND 0 < tight_band AND w_need(tb) <= w_cap,
 * where w_cap is the widest bucket ceiling whose control-flow envelope still holds
 * (ADAPTIVE_BAND_START = 20 at minimum, 30 at default params -- Gate B). Ineligible
 * pairs (no proof, or a band wider than w_cap) skip the rung and reach the untouched
 * ladder unchanged. Uncertified pairs GAP2-restore a->score and fall through to the
 * ladder too. The rung returns the count of pairs left for the ladder (ineligible +
 * uncertified), compacted to the front of pair_ar.
 *
 * The bucket call scores a sub-slice of pair_ar, so its padding-lane overshoot
 * would clobber the following region: guard the sym object for its duration
 * (OT/OB are already guarded under --meth). pen_clip / is_left select the
 * extension-direction clip penalty and profiling hook.
 * ------------------------------------------------------------------------- */
/* Forward declarations: the certified probe rung below records post-SW outcomes
 * through these profiling helpers, whose definitions live after the ladder
 * blocks further down this file. */
static inline void ugp_record_left_outcome(const SeqPair *sp, int a_match, int tid);
static inline void ugp_record_right_outcome(const SeqPair *sp, int tid);

static inline int bsw_tb_probe_rung(BswMethTier tier, const mem_opt_t *opt, mem_alnreg_v *av_v,
        IBandedPairWiseSW *sym, IBandedPairWiseSW *ot, IBandedPairWiseSW *ob,
        SeqPair *pair_ar, uint8_t *ref, uint8_t *qer, int nump,
        int nthreads, SeqPair *scratch, int32_t *hist,
        const bseq1_t *seq_, int pen_clip, int is_left, int tid)
{
    if (nump <= 0 || !opt->band_cert) return nump;

    /* Quantized bucket ceilings. A pair runs at the smallest ceiling >= its w_need, so a
     * proven-narrow pair does its DP at a band near its tb rather than the uniform probe
     * width -- the whole point, since on the short 8-bit extensions band 20 costs about
     * what band opt->w does. The certificate is evaluated at the bucket ceiling W_b >= tb,
     * so acceptance is byte-identical for every ceiling.
     *
     * Gate B: ceilings ABOVE ADAPTIVE_BAND_START (20) recover the ~10-17% of eligible pairs
     * whose w_need exceeds the startup probe width. They may only be used where the certificate's
     * control-flow envelope still holds AT the wider band, so each such ceiling is admitted only
     * when mem_band_cert_params_safe_w(opt, W_b) passes. Because that predicate tightens
     * monotonically in W_b, the admissible ceilings form a prefix up to a single cap w_cap
     * (= 30 at default params, where the zdrop margin binds: zdrop 100 > 7 + 3*30). Ceilings
     * <= 20 are already covered by the startup gate (opt->band_cert on <=> _w(opt,20) passed). */
    static const int32_t BUCKET_CEIL[] = {4, 8, 12, 16, ADAPTIVE_BAND_START, 24, 28, 30};
    const int NBUCK = (int)(sizeof(BUCKET_CEIL) / sizeof(BUCKET_CEIL[0]));
    int w_cap = ADAPTIVE_BAND_START;
    for (int b = 0; b < NBUCK; b++) {
        const int32_t Wc = BUCKET_CEIL[b];
        if (Wc <= ADAPTIVE_BAND_START) { w_cap = Wc; continue; }
        if (!mem_band_cert_params_safe_w(opt, Wc)) break;  /* monotone: wider ceilings also fail */
        w_cap = Wc;
    }

    /* The certificate at run width W_b needs W_b >= tb + ceil(pen_clip/e_min): tight_band is
     * anchored on the raw ungapped score with no clip slack, but the accept below certifies at
     * S - pen_clip (the direction's clip). So key eligibility and buckets on the width the
     * certificate actually demands, w_need = tb + clip_slack, NOT tb -- otherwise the modal
     * ungapped-optimal pair (sc == max_sc_proof) fails the certificate by exactly that margin and
     * re-runs the ladder. This is a heuristic width choice only; the certificate is still evaluated
     * at the actual run width, so byte-identity is unchanged. */
    const int e_min = opt->e_del < opt->e_ins ? opt->e_del : opt->e_ins;
    const int clip_slack = (e_min > 0) ? (pen_clip + e_min - 1) / e_min : ADAPTIVE_BAND_START;
    auto w_need_of = [&](int tb) { return tb + clip_slack; };

    /* Partition [ineligible | eligible]; eligible = proven narrow (w_need <= w_cap). */
    int n_inelig = 0;
    for (int l = 0; l < nump; l++) {
        int tb = pair_ar[l].tight_band;
        if (!(tb > 0 && w_need_of(tb) <= w_cap)) {
            SeqPair t = pair_ar[n_inelig]; pair_ar[n_inelig] = pair_ar[l]; pair_ar[l] = t;
            n_inelig++;
        }
    }
    int n_elig = nump - n_inelig;
    if (n_elig <= 0) return nump;
    SeqPair *elig = pair_ar + n_inelig;

    /* Counting-sort the eligible pairs into contiguous ceiling-buckets keyed on w_need (scratch
     * as temp; freed before any bsw_run_tier call reuses it as sort scratch). */
    int cnt[NBUCK] = {0}, off[NBUCK] = {0};
    auto bucket_of = [&](int w_need) { int b = 0; while (b < NBUCK - 1 && BUCKET_CEIL[b] < w_need) b++; return b; };
    for (int l = 0; l < n_elig; l++) cnt[bucket_of(w_need_of(elig[l].tight_band))]++;
    for (int b = 1; b < NBUCK; b++) off[b] = off[b - 1] + cnt[b - 1];
    { int cur[NBUCK]; for (int b = 0; b < NBUCK; b++) cur[b] = off[b];
      for (int l = 0; l < n_elig; l++) { int b = bucket_of(w_need_of(elig[l].tight_band)); scratch[cur[b]++] = elig[l]; }
      for (int l = 0; l < n_elig; l++) elig[l] = scratch[l]; }

    /* Score each non-empty bucket at its ceiling and certify there; failures from
     * all buckets compact after the ineligible run (write cursor <= read cursor)
     * for the untouched ladder. Guard the sym object: each bucket is a sub-slice. */
    int n_fail = 0;
    for (int b = 0; b < NBUCK; b++) {
      if (cnt[b] <= 0) continue;
      const int32_t Wb = BUCKET_CEIL[b];
      SeqPair *bkt = elig + off[b];
      sym->set_guard_overshoot(true);
      bsw_run_tier(tier, opt, av_v, sym, ot, ob,
                   bkt, ref, qer, cnt[b], nthreads, Wb, scratch, scratch, hist);
      sym->set_guard_overshoot(false);
      for (int l = 0; l < cnt[b]; l++) {
        SeqPair *sp = &bkt[l];
        mem_alnreg_t *a = &(av_v[sp->seqid].a[sp->regid]);
        int prev = a->score;
        a->score = sp->score;
        /* Fix 2 (branch-A accept): certify at the direction's pen_clip anchor with NO gscore
         * precondition, then let the finalize block below pick branch A (clipped: qle/tle) or
         * branch B (query-end: gscore/gtle) by the same threshold the ladder uses. band_cert_ok
         * at S - pen_clip proves every out-of-band cell scores strictly below S - pen_clip, so the
         * wide run's branch decision and both branches' finalized fields are band-invariant
         * (Fable-verified against the kernel update rules). The direction's pen_clip is REQUIRED:
         * min(pen_clip5,pen_clip3) is unsound for branch A under asymmetric -L. This is a SEPARATE
         * accept from the shared band_cert_accept, so the shipped scalar/16-bit probes are untouched. */
        if (band_cert_ok(a->score - pen_clip, sp->h0, sp->len1, sp->len2, Wb, opt)) {
            /* Finalize exactly as the direction's ladder does (LEFT: qb/rb backward, truesc=;
             * RIGHT: qe/re forward, truesc+=, branch B sets qe=l_query). */
            if (is_left) {
#if BWAMEM3_UGP_PROFILE
                ugp_record_left_outcome(sp, opt->a, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - pen_clip) {
                    a->qb -= sp->qle; a->rb -= sp->tle;
                    a->truesc = a->score;
                } else {
                    a->qb = 0; a->rb -= sp->gtle;
                    a->truesc = sp->gscore;
                }
            } else {
#if BWAMEM3_UGP_PROFILE
                ugp_record_right_outcome(sp, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - pen_clip) {
                    a->qe += sp->qle; a->re += sp->tle;
                    a->truesc += a->score - sp->h0;
                } else {
                    int l_query = seq_[sp->seqid].l_seq;
                    a->qe = l_query; a->re += sp->gtle;
                    a->truesc += sp->gscore - sp->h0;
                }
            }
            a->w = max_(a->w, Wb);
            if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_) {
                int ii = 0;
                for (ii = 0, a->seedcov = 0; ii < a->c->n; ++ii) {
                    const mem_seed_t *t = &(a->c->seeds[ii]);
                    if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                        t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                        a->seedcov += t->len;
                }
            }
        } else {
            a->score = prev;              /* GAP2 restore (unconditional) */
            elig[n_fail++] = *sp;          /* carry to the ladder */
        }
      }
    }
    return n_inelig + n_fail;
}

/* Restructured BSW parent function */
#define FAC 8
#define PFD 2

// ungapped diagonal-extension analyzer.
//
// Walks the N-step diagonal once (SIMD scan + bitmap + scalar trajectory).
// Emits three possible status codes:
//
//   FP_STATUS_HIT      — ungapped is provably optimal (≤ x_threshold
//                        mismatches, no ambig). Caller skips banded SW
//                        and uses sp.score/qle/gscore/gtle directly.
//
//   FP_STATUS_TIGHT    — fast-path fails (too many mismatches) but the
//                        ungapped score bounds the useful SW band:
//                          tight_band = ceil((min(len1,len2)·a - ungapped_score)
//                                            / (o_min + e_min))
//                        Caller runs SW with this tight band instead of
//                        opt->w, and skips MAX_BAND_TRY retries (the
//                        bound is an upper bound on any gapped score).
//
//   FP_STATUS_FALLBACK — ambig base or out-of-range length. Caller uses
//                        opt->w with the full retry loop.
//
// Derivation of tight_band: for any alignment with band offset B from
// diagonal, min B gaps are required; cost ≥ B · (o_min + e_min). Max
// alignment score (all matches) ≤ min(len1, len2) · a. For any gapped
// alignment to beat the observed ungapped max score S:
//   min_len·a - B·(o_min+e_min) > S
//   B < (min_len·a - S) / (o_min + e_min)
// So any band ≥ tight_band is sufficient; narrower bands suffice too
// when the gapped alternative score is < min_len·a. Starting SW at
// tight_band is strictly correct and avoids over-banded DP work.
#define FP_N_MAX 512
/* TIGHT routing (accepting a narrow banded-SW result early via the retry
 * ladder's tight_band clause) is byte-identity-proven only up to this length.
 * Beyond it, a width-w accepted result and the width-2w result a FALLBACK pair
 * would retry can differ in extent fields (qle/tle/gscore/max_off and a->w)
 * even at an identical optimal score -- the ladder clause is score-sound but
 * not extent-invariant. So a pair longer than this may only HIT (skip SW,
 * whose scalar walk mirrors kernel semantics at any length) or FALLBACK to the
 * exact full-width ladder; it must never emit a tight_band. Keep <= the old
 * scanner cap so behavior at those lengths is unchanged. */
#define FP_TIGHT_MAX 128
/* Words in the per-pair mismatch bitmap. Sized to FP_N_MAX so every scanned
 * position 0..FP_N_MAX-1 has a bit; a shift by the in-word offset (< 64) can
 * never alias. (Must track FP_N_MAX: at 128 this is 2 words = the old
 * mis_lo/mis_hi pair.) */
#define FP_MIS_NWORDS ((FP_N_MAX + 63) / 64)
#define FP_STATUS_FALLBACK  0
#define FP_STATUS_HIT       1
#define FP_STATUS_TIGHT     2

/* Q2 helper: bin an extension's final alignment score into one of 8 buckets.
 * Bins: {0-10, 11-25, 26-50, 51-75, 76-100, 101-125, 126-150, 151+}. */
static inline int ugp_score_bin(int s)
{
    if      (s <=  10) return 0;
    else if (s <=  25) return 1;
    else if (s <=  50) return 2;
    else if (s <=  75) return 3;
    else if (s <= 100) return 4;
    else if (s <= 125) return 5;
    else if (s <= 150) return 6;
    else               return 7;
}

/* Q3 helper: bin a "delta from perfect extension" into one of 8 buckets.
 * delta = h0 + a * len2 - aln_score, where (h0 + a * len2) is the score of a
 * hypothetical all-match ungapped extension. delta == 0 ⇒ extension was
 * perfect (HIT path with all matches); delta grows with mismatches and gaps.
 * Bins: {0, 1-5, 6-10, 11-25, 26-50, 51-75, 76-100, 101+}. */
static inline int ugp_delta_bin(int d)
{
    if      (d ==   0) return 0;
    else if (d <=   5) return 1;
    else if (d <=  10) return 2;
    else if (d <=  25) return 3;
    else if (d <=  50) return 4;
    else if (d <=  75) return 5;
    else if (d <= 100) return 6;
    else               return 7;
}

/* Q1+Q2+Q3 outcome helpers. Each post-SW dispatch (LEFT scalar / 16-bit /
 * 8-bit; RIGHT scalar / 16-bit / 8-bit) commits a final outcome and bumps
 * the same set of profiling counters. Hoisted so categorization changes
 * (extra category, adjusted bin edges, fourth tier, ...) live in one place
 * instead of drifting across six near-identical inlined blocks. */
static inline void ugp_record_left_outcome(const SeqPair *sp, int a_match, int tid)
{
    int _u = (sp->qle == sp->tle);
    tprof[UGP_OUTCOME_BASE + 0 * 2 + (_u ? 0 : 1)][tid]++;
    tprof[UGP_SCORE_HIST_BASE + 0 * UGP_SCORE_HIST_NBINS
          + ugp_score_bin(sp->score)][tid]++;
    /* Q3: per-category delta-from-perfect histograms. Non-HIT by
     * construction at the post-SW commit. cat0=ALL; cat1 (ungapped final)
     * or cat2 (gapped final) by sp->qle == sp->tle proxy; cat4
     * (ungap_final ∩ ~HIT) when ungapped. */
    int _perfect = sp->h0 + a_match * sp->len2;
    int _bin_ung = ugp_delta_bin(_perfect - sp->ugp_walk_score);
    int _bin_fin = ugp_delta_bin(_perfect - sp->score);
    tprof[UGP_L_CAT_UNG_BASE + 0 * UGP_CAT_NBINS + _bin_ung][tid]++;
    tprof[UGP_L_CAT_FIN_BASE + 0 * UGP_CAT_NBINS + _bin_fin][tid]++;
    if (_u) {
        tprof[UGP_L_CAT_UNG_BASE + 1 * UGP_CAT_NBINS + _bin_ung][tid]++;
        tprof[UGP_L_CAT_FIN_BASE + 1 * UGP_CAT_NBINS + _bin_fin][tid]++;
        tprof[UGP_L_CAT_UNG_BASE + 4 * UGP_CAT_NBINS + _bin_ung][tid]++;
        tprof[UGP_L_CAT_FIN_BASE + 4 * UGP_CAT_NBINS + _bin_fin][tid]++;
    } else {
        tprof[UGP_L_CAT_UNG_BASE + 2 * UGP_CAT_NBINS + _bin_ung][tid]++;
        tprof[UGP_L_CAT_FIN_BASE + 2 * UGP_CAT_NBINS + _bin_fin][tid]++;
    }
}

static inline void ugp_record_right_outcome(const SeqPair *sp, int tid)
{
    int _u = (sp->qle == sp->tle);
    tprof[UGP_OUTCOME_BASE + 1 * 2 + (_u ? 0 : 1)][tid]++;
    tprof[UGP_SCORE_HIST_BASE + 1 * UGP_SCORE_HIST_NBINS
          + ugp_score_bin(sp->score)][tid]++;
}

/* Q3 helper: would-be ungapped extension score for arbitrary N. Mirrors the
 * HIT-path scalar walk in ungapped_analyze (cur with floor at 0; max_sc
 * tracker; ambig terminates the walk to match analyze's FALLBACK semantics).
 * Returned score is what an ungapped extension would produce on this pair
 * regardless of whether the fast-path actually triggered HIT. */
static inline int ungapped_walk_score(const uint8_t *qs, const uint8_t *rs,
                                       int N, int h0, int a, int b)
{
    int cur = h0, max_sc = h0;
    for (int j = 0; j < N; j++) {
        uint8_t qj = qs[j], rj = rs[j];
        if (qj >= 4 || rj >= 4) break;       /* ambig: terminate the walk */
        if (cur == 0) continue;
        if (qj == rj) cur += a;
        else {
            cur -= b;
            if (cur < 0) cur = 0;
        }
        if (cur >= max_sc) max_sc = cur;
    }
    return max_sc;
}

static inline int ungapped_analyze(const uint8_t *qs, const uint8_t *rs, int N,
                                    int h0, int a, int b,
                                    int o_min, int e_min,
                                    int x_threshold, int default_w,
                                    int *out_score, int *out_qle,
                                    int *out_gscore, int *out_gtle,
                                    int *out_tight_band)
{
    if (N <= 0 || N > FP_N_MAX || x_threshold < 0) return FP_STATUS_FALLBACK;

    const __m128i v3 = _mm_set1_epi8(3);
    uint64_t mis[FP_MIS_NWORDS] = {0};
    int total_mis = 0;
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m128i qv = _mm_loadu_si128((const __m128i *)(qs + i));
        __m128i rv = _mm_loadu_si128((const __m128i *)(rs + i));
        // AMBIG (code >= 4) detected via max > 3. ACGT = 0..3.
        __m128i mxv = _mm_max_epu8(qv, rv);
        if ((unsigned)_mm_movemask_epi8(_mm_cmpgt_epi8(mxv, v3)))
            return FP_STATUS_FALLBACK;  // ambig: give up on both paths
        __m128i eqv = _mm_cmpeq_epi8(qv, rv);
        unsigned mism_mask = (~(unsigned)_mm_movemask_epi8(eqv)) & 0xFFFFu;
        total_mis += __builtin_popcount(mism_mask);
        /* i is 16-aligned, so (i & 63) in {0,16,32,48}: a 16-bit mask shifted
         * by at most 48 fills bits [i&63, (i&63)+15] within a single word and
         * never straddles a 64-bit boundary. */
        mis[i >> 6] |= ((uint64_t)mism_mask) << (i & 63);
    }
    for (; i < N; i++) {
        uint8_t qi = qs[i], ri = rs[i];
        if (qi >= 4 || ri >= 4) return FP_STATUS_FALLBACK;
        if (qi != ri) {
            total_mis++;
            mis[i >> 6] |= (1ULL << (i & 63));
        }
    }

    if (total_mis > x_threshold) {
        /* TIGHT routing is extent-invariant only up to FP_TIGHT_MAX (see the
         * define). A longer pair that is not a HIT falls back to the exact
         * full-width ladder rather than accepting a narrow banded-SW result. */
        if (N > FP_TIGHT_MAX) return FP_STATUS_FALLBACK;
        // S in the band proof must be REALIZABLE by an actual offset-0 extension.
        // The tight_band derivation shows every out-of-band alignment (band
        // offset B >= tb) scores <= S, so a band >= tb is sufficient ONLY IF the
        // in-band (offset-0) run actually achieves S. ungapped_walk_score is the
        // floored ungapped score under ksw_extend local-truncation semantics
        // (once the running score hits 0 it stays 0) -- exactly the score the
        // rung-1 banded DP can reach on the diagonal, so it is a valid lower
        // bound on the in-band optimum.
        //
        // A no-floor score (which lets a negative prefix recover via later
        // matches) can EXCEED the floor-killed value the DP actually reaches;
        // feeding that larger S shrinks tb below what is sound, letting the retry
        // ladder skip the wider rung a gapped alignment in (default_w, wider]
        // genuinely needs -- a CIGAR/coordinate divergence from the full-width
        // ladder (breaking byte-identity). The walk is O(N) and runs only on the
        // TIGHT branch after the FP_TIGHT_MAX gate above (N <= 128), so its cost
        // is negligible.
        //
        // Band derivation (see comment above):
        //     B < (min_len·a − S − o_min) / e_min        with S = max_sc − h0
        // For LEFT extensions where the caller gates on len1 ≥ len2,
        // min_len = N. Substituting:
        //     numerator = N·a − (max_sc_proof − h0) − o_min
        //               = N·a + h0 − max_sc_proof − o_min
        int max_sc_proof = ungapped_walk_score(qs, rs, N, h0, a, b);
        int64_t numerator = (int64_t)N * a + h0 - max_sc_proof - o_min;
        int band;
        if (numerator <= 0) {
            // Ungapped-optimal (same "tight_band = 0 sentinel" semantic as
            // walk-derived numerator ≤ 0: let SW run with fallback width).
            band = 0;
        } else if (e_min <= 0) {
            band = default_w;
        } else {
            band = (int)((numerator + e_min - 1) / e_min);
            // The band proof only excludes diagonal offsets >= band. When the
            // proven band exceeds default_w, clamping to default_w would falsely
            // certify that a width-default_w run is complete, letting the retry
            // ladder short-circuit and skip the wider rungs a gapped alignment
            // in the unproven window (default_w, band) genuinely needs. Emit the
            // tight_band = 0 fallback sentinel instead, so such a pair runs the
            // exact full-width ladder (byte-identical to non-adaptive extension).
            if (band > default_w) band = 0;
        }
        *out_tight_band = band;
        // Outputs unused on TIGHT; set sane values for any future caller.
        *out_score  = max_sc_proof;
        *out_qle    = 0;
        *out_gscore = 0;
        *out_gtle   = N;
        return FP_STATUS_TIGHT;
    }

    // Closed-form fast path: total_mis == 0 (perfect-match HIT).
    //
    // With no mismatches the scalar walk below is deterministic:
    //   cur starts at h0 and only increases (+a per iter), so it never
    //   touches the `cur == 0` early-skip after the first iteration.
    //   max_sc rises to h0 + N*a; max_i ends at N (the >= tie-break
    //   updates max_i on every step).
    //
    // The h0 > 0 guard preserves bit-identicality with the loop: when
    // h0 == 0 the loop's `if (cur == 0) continue` inhibits all updates,
    // leaving max_sc = 0 / max_i = 0; the closed form would instead
    // compute max_sc = N*a, breaking parity. h0 == 0 is pathological in
    // practice (caller passes a seed score) but the guard is a single
    // compare and free.
    if (total_mis == 0 && h0 > 0) {
        int score = h0 + N * a;
        *out_score      = score;
        *out_qle        = N;
        *out_gscore     = score;
        *out_gtle       = N;
        *out_tight_band = 0;   // unused on HIT (SW skipped)
        return FP_STATUS_HIT;
    }

    // HIT candidate: run the scalar walk for precise qle / gscore / max_sc.
    //
    // MAIN_CODE* local-SW semantics: once cur==0 in the ungapped path it
    // stays 0 (no e/f restart).
    //
    // Tie-break: SW's maxRS tracker (bandedSWA.cpp MAIN_CODE) updates the
    // position on BOTH strictly-greater and tied equal-to-current
    // comparisons — equivalent to "pick the rightmost position where the
    // max was achieved". We must mirror that (use >=) or qle/tle diverge
    // from SW on tied-score walks, breaking byte-identical SAM.
    int cur = h0, max_sc = h0, max_i = 0;
    for (int j = 0; j < N; j++) {
        int is_mis = (int)((mis[j >> 6] >> (j & 63)) & 1ULL);
        if (cur == 0) continue;
        if (!is_mis) cur += a;
        else {
            cur -= b;
            if (cur < 0) cur = 0;
        }
        if (cur >= max_sc) { max_sc = cur; max_i = j + 1; }
    }

    *out_score      = max_sc;
    *out_qle        = max_i;
    *out_gscore     = cur;
    *out_gtle       = N;
    *out_tight_band = 0;   // unused on HIT (SW skipped)
    return FP_STATUS_HIT;
}

/* ------------------------------------------------------------------------
 * Byte-identical contained-seed extension skip (--skip-contained-ext).
 *
 * A seed s strictly contained (same diagonal, query subinterval) in a longer
 * seed of the same chain is dominated by that longer seed's extension: the
 * longest seed on a diagonal is always extended, its alnreg covers s's region,
 * so s's own alnreg is purged post-extension anyway (the PE18 containment purge
 * later in this function). We detect this before extension and skip building s's
 * banded-SW pair, marking its alnreg purged (qb=qe=-1) exactly as PE18 would --
 * but s stays in c->seeds, so seedcov/MAPQ are untouched.
 *
 * Byte-identity: the skip set is a proven subset of PE18's purge set. (a) seed
 * containment implies containment in the longer seed's (extension-grown) alnreg,
 * so PE18 would purge it; (b) we replicate PE18's interference guard (a
 * comparably-long seed overlapping s on a *different* diagonal forces
 * extension) using seed coordinates; (c) dropping s as a container for other
 * seeds is safe because the longest same-diagonal seed (always extended) is a
 * superset container.
 * ---------------------------------------------------------------------- */
static inline int mem_seed_ext_redundant(const mem_chain_t *c, int si)
{
    const mem_seed_t *s = &c->seeds[si];
    long sd = (long)s->rbeg - s->qbeg;
    int has_container = 0, j;
    for (j = 0; j < c->n; ++j) {
        if (j == si) continue;
        const mem_seed_t *t = &c->seeds[j];
        if (t->len <= s->len) continue;                    /* strictly longer */
        if ((long)t->rbeg - t->qbeg != sd) continue;       /* same diagonal */
        if (s->qbeg >= t->qbeg && s->qbeg + s->len <= t->qbeg + t->len) { has_container = 1; break; }
    }
    if (!has_container) return 0;
    /* interference guard: mirror PE18 (a seed >= .95*len overlapping s on a
     * different diagonal by >= s->len/4 means s could lead to a distinct aln). */
    for (j = 0; j < c->n; ++j) {
        if (j == si) continue;
        const mem_seed_t *u = &c->seeds[j];
        if (u->len < s->len * .95) continue;
        if (s->qbeg <= u->qbeg && s->qbeg + s->len - u->qbeg >= s->len >> 2 &&
            u->qbeg - s->qbeg != u->rbeg - s->rbeg) return 0;
        if (u->qbeg <= s->qbeg && u->qbeg + u->len - s->qbeg >= s->len >> 2 &&
            s->qbeg - u->qbeg != s->rbeg - u->rbeg) return 0;
    }
    return 1;
}


void mem_chain2aln_across_reads_V2(const mem_opt_t *opt_in, const bntseq_t *bns,
                                   const uint8_t *pac, bseq1_t *seq_, int nseq,
                                   mem_chain_v* chain_ar, mem_alnreg_v *av_v,
                                   mem_cache *mmc, uint8_t *ref_string, int tid)
{
    /* Enforce the certified-band safety envelope at this public entry point.
     * band_cert is default-on, but the certificate is sound only inside the
     * parameter envelope (see mem_band_cert_params_safe). main_mem() applies
     * this same guard for the CLI, so for the shipping binary this is a no-op
     * (opt_in->band_cert already agrees with the guard) and the run stays
     * byte-identical. A library caller that mutates scoring/gap fields after
     * mem_opt_init(), or sets an aggressive adaptive-band start (band_start > 0)
     * without also clearing band_cert the way the --adaptive-band CLI path does,
     * and calls this entry point directly would otherwise select certified
     * narrowing outside the envelope and change alignment records; sanitize a
     * local copy here so every band_cert decision and restore below falls back
     * to the exact full-width ladder when band_start is set or the parameters
     * are unsafe. The short-circuit keeps the mat scan off the hot path whenever
     * band_cert is already off. */
    mem_opt_t opt_sanitized;
    const mem_opt_t *opt = opt_in;
    if (opt_in->band_cert &&
        (opt_in->band_start > 0 || !mem_band_cert_params_safe(opt_in))) {
        opt_sanitized = *opt_in;
        opt_sanitized.band_cert = 0;
        opt = &opt_sanitized;
    }

    SeqPair *seqPairArrayAux      = mmc->seqPairArrayAux[tid];
    SeqPair *seqPairArrayLeft128  = mmc->seqPairArrayLeft128[tid];
    SeqPair *seqPairArrayRight128 = mmc->seqPairArrayRight128[tid];
    int64_t *wsize_pair = &(mmc->wsize[tid]);

    uint8_t *seqBufLeftRef  = mmc->seqBufLeftRef[tid*CACHE_LINE];
    uint8_t *seqBufRightRef = mmc->seqBufRightRef[tid*CACHE_LINE];
    uint8_t *seqBufLeftQer  = mmc->seqBufLeftQer[tid*CACHE_LINE];
    uint8_t *seqBufRightQer = mmc->seqBufRightQer[tid*CACHE_LINE];
    int64_t *wsize_buf_ref = &(mmc->wsize_buf_ref[tid*CACHE_LINE]);
    int64_t *wsize_buf_qer = &(mmc->wsize_buf_qer[tid*CACHE_LINE]);

    // int32_t *lim_g = mmc->lim + (BATCH_SIZE + 32) * tid;
    int32_t *lim_g = mmc->lim[tid];

    mem_seed_t *s;
    int64_t l_pac = bns->l_pac, rmax[8] __attribute__((aligned(64)));
    // std::vector<int8_t> nexitv(nseq, 0);

    int numPairsLeft = 0, numPairsRight = 0;
    int numPairsLeft1 = 0, numPairsRight1 = 0;
    int numPairsLeft128 = 0, numPairsRight128 = 0;
    int numPairsLeft16 = 0, numPairsRight16 = 0;

    int64_t leftRefOffset = 0, rightRefOffset = 0;
    int64_t leftQerOffset = 0, rightQerOffset = 0;

    // Ungapped fast-path threshold, computed once per call from the scoring
    // model: the largest mismatch count X for which an ungapped diagonal is
    // provably optimal, so ungapped_analyze may return FP_STATUS_HIT and skip
    // the banded SW entirely.
    //
    // An ungapped walk carrying X mismatches scores X*(a+b) below the all-match
    // diagonal. The cheapest gapped alternative pays at least one gap open plus
    // extend, o_min + e_min, and can at best convert every one of those X
    // mismatches back into a match. So the ungapped walk is safe only while it
    // wins STRICTLY:
    //
    //     X * (a + b) < o_min + e_min   =>   X <= (o_min + e_min - 1) / (a + b)
    //
    // Strictness is load-bearing, not pedantry: on an exact tie the gapped path
    // scores the same, and ksw_extend2 breaks ties with `if (m > max)` -- the
    // cell reached first keeps the max -- so the DP can return the gapped CIGAR
    // while this fast path returns the ungapped one.
    //
    // This previously read o_min / (a + b - e_min), which coincides with the
    // correct bound at bwa's defaults (6/4 == 1 and (6+1-1)/5 == 1) but is too
    // permissive elsewhere. Measured on 250k HG002 WGS pairs vs hg38 against a
    // build with the fast path disabled outright, i.e. always running the DP
    // (what bwa-mem2 does -- it has no ungapped fast path at all):
    //
    //   -B 3             old bound admits X = 2, yet 2*(1+3) = 8 > 6+1, so one
    //                    deletion beats the diagonal:    51/501,821 records
    //                    (0.010%) differ from the DP -- CIGAR in all 51, POS in
    //                    27, MAPQ in none. The fast path emits e.g. 151M and
    //                    148M3S where the DP finds 138M1D13M and 148M1D3M.
    //   -A1 -B1 -O1 -E1  old bound admits X = 1 and 1*(1+1) == 1+1 exactly, the
    //                    tie case:                    1,522/501,091 records
    //                    (0.30%) differ, including 10 MAPQ changes.
    //
    // Defaults are unchanged in both directions, so this costs no fast-path
    // coverage on a stock run.
    //
    // That second scheme is what -x pacbio and -x ont2d set, but long reads
    // never reached it: FP_N_MAX caps the fast path at 128 columns and their
    // extensions are longer, so 2,000 HG002 HiFi reads under -x pacbio come out
    // byte-identical with the bound either way. Note the cap is a coverage
    // limit, not a correctness guard -- the bound above is sound at every
    // scoring scheme tested, so raising FP_N_MAX does not reopen this.
    //
    // The fast path is disabled entirely under --meth (see the three call sites
    // below), since ungapped_analyze hardcodes symmetric opt->a/opt->b and
    // cannot express the asymmetric OT/OB matrix. This bound is not consulted
    // there.
    const int fp_o_min = opt->o_del < opt->o_ins ? opt->o_del : opt->o_ins;
    const int fp_e_min = opt->e_del < opt->e_ins ? opt->e_del : opt->e_ins;
    // Cheapest single gap the scoring scheme allows; a degenerate scheme that
    // makes gaps free (or a non-positive a+b) leaves nothing provable, so the
    // fast path is disabled with -1 rather than guessed at.
    const int fp_gap_min = fp_o_min + fp_e_min;
    const int fp_denom   = opt->a + opt->b;
    const int fp_x_threshold =
        (fp_denom > 0 && fp_gap_min > 0) ? ((fp_gap_min - 1) / fp_denom) : -1;
    // (fp_o_min, fp_e_min above are reused directly by ungapped_analyze.)

    int srt_size = MAX_SEEDS_PER_READ, fac = FAC;
    uint64_t *srt = (uint64_t *) malloc(srt_size * 8);
    uint32_t *srtgg = (uint32_t*) malloc(nseq * SEEDS_PER_READ * fac * sizeof(uint32_t));

    int spos = 0;

    /* D3 (--meth, PR-4): extension scores the ORIGINAL (unconverted) read bases
     * against the original reference, not the projected (C→T / G→A) read used for
     * seeding. seq_[l].meth_orig_seq holds the original bases as NUL-terminated
     * ASCII in the SAME orientation/order as seq_[l].seq (see the bseq1_t
     * orientation contract in bwa.h), so a 2-bit encode of it indexes identically
     * to the projected query the extension index math already assumes — every
     * qs[i] = query[...] below stays correct with no index change. We 2-bit-encode
     * into this reusable scratch buffer per read. NULL/unused outside --meth. */
    uint8_t *meth_qbuf = NULL;
    int      meth_qbuf_cap = 0;

    // uint64_t timUP = __rdtsc();
    for (int l=0; l<nseq; l++)
    {
        int max = 0;
        uint8_t *rseq = 0;

        uint32_t *srtg = srtgg;
        lim_g[l+1] = 0;

        const uint8_t *query = (uint8_t *) seq_[l].seq;
        int l_query = seq_[l].l_seq;

        /* D3 (--meth, PR-4): swap the extension query to the ORIGINAL read bases.
         * seq_[l].seq is the projected (already C→T/G→A) read, used only for
         * seeding; extension must score the original read against the original
         * ref with the per-hypothesis asymmetric matrix. We 2-bit-encode
         * meth_orig_seq (ASCII, same orientation as seq) into meth_qbuf and point
         * `query` at it. If meth_orig_seq is missing (should not happen under
         * --meth once ingest populates it) we fall back to the projected read so
         * the path still runs. */
        if (opt->meth_mode && seq_[l].meth_orig_seq != NULL) {
            if (l_query > meth_qbuf_cap) {
                meth_qbuf_cap = l_query;
                /* CodeRabbit: temp pointer so a NULL realloc doesn't leak. */
                uint8_t *meth_qbuf_new = (uint8_t *) realloc(meth_qbuf, (size_t) meth_qbuf_cap);
                xassert(meth_qbuf_new != NULL, "out of memory: meth_qbuf_new");
                meth_qbuf = meth_qbuf_new;
            }
            const char *os = seq_[l].meth_orig_seq;
            for (int i = 0; i < l_query; ++i) {
                unsigned char c = (unsigned char) os[i];
                meth_qbuf[i] = nst_nt4_decode(c, 4);
            }
            query = meth_qbuf;
        }

        mem_chain_v *chn = &chain_ar[l];
        mem_alnreg_v *av = &av_v[l];  // alignment
        mem_chain_t *c;

        _mm_prefetch((const char*) query, _MM_HINT_NTA);

        // aln mem allocation
        av->m = 0;
        for (int j=0; j<chn->n; j++) {
            c = &chn->a[j]; av->m += c->n;
        }
        av->a = (mem_alnreg_t*)calloc(av->m, sizeof(mem_alnreg_t));

        // aln mem allocation ends
        for (int j=0; j<chn->n; j++)
        {
            c = &chn->a[j];
            assert(c->seqid == l);

            int64_t tmp = 0;
            if (c->n == 0) continue;

            _mm_prefetch((const char*) (srtg + spos + 64), _MM_HINT_NTA);
            _mm_prefetch((const char*) (lim_g), _MM_HINT_NTA);

            // get the max possible span
            rmax[0] = l_pac<<1; rmax[1] = 0;

            int chain_max_n_hits = 1;
            int64_t cb_lo=0, cb_hi=0; int cb_f=1, chain_band=0;
            for (int i = 0; i < c->n; ++i) {
                int64_t b, e;
                const mem_seed_t *t = &c->seeds[i];
                b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg));
                e = t->rbeg + t->len + ((l_query - t->qbeg - t->len) +
                                        cal_max_gap(opt, l_query - t->qbeg - t->len));

                tmp = rmax[0];
                rmax[0] = tmp < b? rmax[0] : b;
                rmax[1] = (rmax[1] > e)? rmax[1] : e;
                if (t->len > max) max = t->len;
                if (t->n_hits > chain_max_n_hits) chain_max_n_hits = t->n_hits;
                if (opt->band_start > 0) { int64_t _d = t->rbeg - t->qbeg; if (cb_f){cb_lo=cb_hi=_d; cb_f=0;} else { if(_d<cb_lo)cb_lo=_d; if(_d>cb_hi)cb_hi=_d; } }
            }
            int64_t cb_span = cb_hi - cb_lo;
            if (cb_span > opt->w) cb_span = opt->w;
            chain_band = (int)cb_span;

            rmax[0] = rmax[0] > 0? rmax[0] : 0;
            rmax[1] = rmax[1] < l_pac<<1? rmax[1] : l_pac<<1;
            if (rmax[0] < l_pac && l_pac < rmax[1])
            {
                if (c->seeds[0].rbeg < l_pac) rmax[1] = l_pac;
                else rmax[0] = l_pac;
            }

            /* retrieve the reference sequence */
            {
                int rid = 0;
                // free rseq
                rseq = bns_fetch_seq_v2(bns, pac, &rmax[0],
                                        c->seeds[0].rbeg,
                                        &rmax[1], &rid, ref_string,
                                        (uint8_t*) seqPairArrayAux);
                assert(c->rid == rid);
            }

            _mm_prefetch((const char*) rseq, _MM_HINT_NTA);
            // _mm_prefetch((const char*) rseq + 64, _MM_HINT_NTA);

            // assert(c->n < MAX_SEEDS_PER_READ);  // temp
            if (c->n > srt_size) {
                srt_size = c->n + 10;
                srt = (uint64_t *) realloc(srt, srt_size * 8);
            }

            for (int i = 0; i < c->n; ++i)
                srt[i] = (uint64_t)c->seeds[i].score<<32 | i;

            if (c->n > 1)
                ks_introsort_64(c->n, srt);

            // assert((spos + c->n) < SEEDS_PER_READ * FAC * nseq);
            if ((spos + c->n) > SEEDS_PER_READ * fac * nseq) {
                fac <<= 1;
                srtgg = (uint32_t *) realloc(srtgg, nseq * SEEDS_PER_READ * fac * sizeof(uint32_t));
            }

            for (int i = 0; i < c->n; ++i)
                srtg[spos++] = srt[i];

            lim_g[l+1] += c->n;

            // uint64_t tim = __rdtsc();
            for (int k=c->n-1; k >= 0; k--)
            {
                s = &c->seeds[(uint32_t)srt[k]];

                mem_alnreg_t *a;
                // a = kv_pushp(mem_alnreg_t, *av);
                a = &av->a[av->n++];
                /* av->a was calloc'd to av->m entries just above; av->n only
                 * ever increments (no slot is reused), so each slot is already
                 * zero on first write — the per-slot memset was redundant. */

                s->aln = av->n-1;

                a->w = opt->w;
                a->score = a->truesc = -1;
                a->rid = c->rid;
                a->frac_rep = c->frac_rep;
                a->seedlen0 = s->len;
                a->c = c; //ptr
                /* D3 (--meth, PR-3): carry the OT/OB hypothesis from chain to
                 * alnreg (-1 outside --meth) so PR-4's asymmetric extension and
                 * the output (XG) layer can select the matrix/strand. */
                a->meth_hypothesis = c->meth_hypothesis;
                /* D3 (--meth, fix): strand-adjusted hypothesis for the extension
                 * matrix. Reverse-strand seeds (s->rbeg >= l_pac, original
                 * doubled-pac) extend against the RC reference window, which
                 * flips the conversion's freed cell — so flip the hypothesis. */
                a->meth_strand_hyp = (c->meth_hypothesis < 0) ? -1
                    : (int8_t)((c->meth_hypothesis ^ (s->rbeg >= l_pac ? 1 : 0)) & 1);
                /* D3 (--meth): the seed matched in projected 3-letter space, but
                 * its 4-letter score against the ORIGINAL read + ref is not
                 * necessarily len*a — a seed-internal variant (e.g. a C/T mirror
                 * that collapsed in seed space) scores as a mismatch under the
                 * per-strand matrix. Recompute the true ungapped seed score so the
                 * alnreg score (the h0 the extension starts from, hence AS, MAPQ,
                 * and mate rescue) is honest rather than optimistically len*a.
                 * --meth-scoring collapsed frees C/T (and G/A) both ways, so this
                 * reproduces len*a (bwameth-like); genomic penalizes the variant.
                 * query is the ORIGINAL read (meth_orig_seq, set above); rseq is
                 * the original 4-letter ref window. mat[ref*5+read] is target-major
                 * (matches mem_opt_fill_meth_mat). */
                int meth_seed_sc = s->len * opt->a;
                if (opt->meth_mode && a->meth_strand_hyp >= 0) {
                    const int8_t *seed_mat = mem_opt_meth_mat(opt, a->meth_strand_hyp);
                    const int64_t roff = s->rbeg - rmax[0];
                    meth_seed_sc = 0;
                    for (int _i = 0; _i < s->len; ++_i)
                        meth_seed_sc += seed_mat[ rseq[roff + _i] * 5 + query[s->qbeg + _i] ];
                }
                a->chain_n_hits = chain_max_n_hits;
                a->rb = a->qb = a->re = a->qe = H0_;

                tprof[PE19][tid] ++;

                /* NB: gated off under --meth. The dominance argument assumes a
                 * longer same-diagonal seed's extension covers the shorter one;
                 * meth's asymmetric C->T matrix breaks this (a contained seed can
                 * extend to a differently-scored aln), so the skip is NOT
                 * byte-identical under --meth (measured ~0.17% of pairs diverge). */
                if (opt->skip_contained_ext && !opt->meth_mode &&
                    mem_seed_ext_redundant(c, (uint32_t)srt[k])) {
                    a->qb = a->qe = -1;   /* pre-purge exactly as PE18 would */
                    continue;
                }

                int flag = 0;
                std::pair<int, int> pr;
                if (s->qbeg)  // left extension
                {
                    SeqPair sp;
                    sp.h0 = meth_seed_sc;   /* D3: true seed score (== len*a outside --meth) */
                    sp.seqid = c->seqid;
                    sp.regid = av->n - 1;

                    if (numPairsLeft >= *wsize_pair) {
                        if (bwa_verbose >= 4) fprintf(stderr, "[0000][%0.4d] Re-allocating seqPairArrays, in Left\n", tid);
                        *wsize_pair +=  1024;
                        // assert(*wsize_pair > numPairsLeft);
                        *wsize_pair += numPairsLeft + 1024;
                        seqPairArrayAux = (SeqPair *) realloc(seqPairArrayAux,
                                                              (*wsize_pair + MAX_LINE_LEN)
                                                              * sizeof(SeqPair));
                        mmc->seqPairArrayAux[tid] = seqPairArrayAux;
                        seqPairArrayLeft128 = (SeqPair *) realloc(seqPairArrayLeft128,
                                                                  (*wsize_pair + MAX_LINE_LEN)
                                                                  * sizeof(SeqPair));
                        mmc->seqPairArrayLeft128[tid] = seqPairArrayLeft128;
                        seqPairArrayRight128 = (SeqPair *) realloc(seqPairArrayRight128,
                                                                   (*wsize_pair + MAX_LINE_LEN)
                                                                   * sizeof(SeqPair));
                        mmc->seqPairArrayRight128[tid] = seqPairArrayRight128;
                    }


                    sp.idq = leftQerOffset;
                    sp.idr = leftRefOffset;

                    leftQerOffset += s->qbeg;
                    if (leftQerOffset >= *wsize_buf_qer)
                    {
                        if (bwa_verbose >= 4) fprintf(stderr, "[%0.4d] Re-allocating (doubling) seqBufQers in %s (left)\n",
                                tid, __func__);
                        int64_t tmp = *wsize_buf_qer;
                        *wsize_buf_qer = seqbuf_grow_capacity(tmp);
                        if (*wsize_buf_qer == SEQBUF_CAPACITY_OVERFLOW)
                            seqbuf_capacity_fatal("seqBufQer", __func__, tmp);
                        assert(*wsize_buf_qer > leftQerOffset);

                        uint8_t *seqBufQer_ = (uint8_t*)
                            _mm_realloc(seqBufLeftQer, tmp, *wsize_buf_qer, sizeof(uint8_t));
                        mmc->seqBufLeftQer[tid*CACHE_LINE] = seqBufLeftQer = seqBufQer_;

                        seqBufQer_ = (uint8_t*)
                            _mm_realloc(seqBufRightQer, tmp, *wsize_buf_qer, sizeof(uint8_t));
                        mmc->seqBufRightQer[tid*CACHE_LINE] = seqBufRightQer = seqBufQer_;
                    }

                    uint8_t *qs = seqBufLeftQer + sp.idq;
                    for (int i = 0; i < s->qbeg; ++i) qs[i] = query[s->qbeg - 1 - i];

                    tmp = s->rbeg - rmax[0];
                    leftRefOffset += tmp;
                    if (leftRefOffset >= *wsize_buf_ref)
                    {
                        if (bwa_verbose >= 4) fprintf(stderr, "[%0.4d] Re-allocating (doubling) seqBufRefs in %s (left)\n",
                                tid, __func__);
                        int64_t tmp = *wsize_buf_ref;
                        *wsize_buf_ref = seqbuf_grow_capacity(tmp);
                        if (*wsize_buf_ref == SEQBUF_CAPACITY_OVERFLOW)
                            seqbuf_capacity_fatal("seqBufRef", __func__, tmp);
                        assert(*wsize_buf_ref > leftRefOffset);
                        uint8_t *seqBufRef_ = (uint8_t*)
                            _mm_realloc(seqBufLeftRef, tmp, *wsize_buf_ref, sizeof(uint8_t));
                        mmc->seqBufLeftRef[tid*CACHE_LINE] = seqBufLeftRef = seqBufRef_;

                        seqBufRef_ = (uint8_t*)
                            _mm_realloc(seqBufRightRef, tmp, *wsize_buf_ref, sizeof(uint8_t));
                        mmc->seqBufRightRef[tid*CACHE_LINE] = seqBufRightRef = seqBufRef_;
                    }

                    uint8_t *rs = seqBufLeftRef + sp.idr;
                    for (int64_t i = 0; i < tmp; ++i) rs[i] = rseq[tmp - 1 - i]; //seq1

                    sp.len2 = s->qbeg;
                    sp.len1 = tmp;
                    sp.tight_band = 0; sp.chain_band = chain_band;  // 0 sentinel: "no tight bound known"

                    // ungapped analysis.
                    //   HIT      → skip SW; fill a->* from ungapped.
                    //   TIGHT    → save sp.tight_band; SW will use it.
                    //   FALLBACK → use opt->w.
                    /* D3 (--meth, PR-4): the ungapped fast path scores with
                     * opt->a/opt->b hardcoded (ungapped_analyze) — it CANNOT
                     * express the asymmetric OT/OB matrix, so a HIT would
                     * mis-penalize every bisulfite conversion (read T at ref C)
                     * as a mismatch and commit a wrong a->score. Disable it under
                     * --meth so all extension flows through the matrix-aware SW
                     * kernel. (Perf-only fast path; correctness-neutral to skip.) */
                    if (!opt->meth_mode && sp.len1 >= sp.len2 && sp.len2 <= FP_N_MAX) {
                        tprof[UGP_L_ATTEMPT][tid]++;
                        int fp_score, fp_qle, fp_gscore, fp_gtle, fp_band;
                        int fp_st = ungapped_analyze(qs, rs, sp.len2,
                                                     sp.h0, opt->a, opt->b,
                                                     fp_o_min, fp_e_min,
                                                     fp_x_threshold, opt->w,
                                                     &fp_score, &fp_qle,
                                                     &fp_gscore, &fp_gtle,
                                                     &fp_band);
                        if (fp_st == FP_STATUS_HIT) {
#if BWAMEM3_UGP_PROFILE
                            tprof[UGP_L_HIT][tid]++;
                            tprof[UGP_L_UNGAPPED][tid]++;
                            tprof[UGP_SCORE_HIST_BASE + 0 * UGP_SCORE_HIST_NBINS
                                  + ugp_score_bin(fp_score)][tid]++;
                            /* Q3: cat0=ALL, cat1=UNGAP_FINAL, cat3=HIT.
                             * For HIT, fp_score is both the would-be ungapped
                             * score and the committed score, so the delta
                             * (perfect_score - aln_score) is the same for
                             * both histograms. */
                            {
                                int _perfect = sp.h0 + opt->a * sp.len2;
                                int _delta = _perfect - fp_score;
                                int _bin = ugp_delta_bin(_delta);
                                tprof[UGP_L_CAT_UNG_BASE + 0 * UGP_CAT_NBINS + _bin][tid]++;
                                tprof[UGP_L_CAT_UNG_BASE + 1 * UGP_CAT_NBINS + _bin][tid]++;
                                tprof[UGP_L_CAT_UNG_BASE + 3 * UGP_CAT_NBINS + _bin][tid]++;
                                tprof[UGP_L_CAT_FIN_BASE + 0 * UGP_CAT_NBINS + _bin][tid]++;
                                tprof[UGP_L_CAT_FIN_BASE + 1 * UGP_CAT_NBINS + _bin][tid]++;
                                tprof[UGP_L_CAT_FIN_BASE + 3 * UGP_CAT_NBINS + _bin][tid]++;
                            }
#endif
                            // Roll back the qs/rs buffer offsets we just
                            // consumed; the batch won't reference them.
                            leftQerOffset -= s->qbeg;
                            leftRefOffset -= tmp;
                            // Mirror post-SW extraction (~line 2560).
                            a->score = fp_score;
                            if (fp_gscore <= 0 || fp_gscore <= a->score - opt->pen_clip5) {
                                a->qb = s->qbeg - fp_qle;
                                a->rb = s->rbeg - fp_qle; // tle == qle on diagonal
                                a->truesc = a->score;
                            } else {
                                a->qb = 0;
                                a->rb = s->rbeg - fp_gtle;
                                a->truesc = fp_gscore;
                            }
                            a->w = max_(a->w, opt->w);
                            if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_) {
                                int ii;
                                for (ii = 0, a->seedcov = 0; ii < a->c->n; ++ii) {
                                    const mem_seed_t *t = &(a->c->seeds[ii]);
                                    if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                                        t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                                        a->seedcov += t->len;
                                }
                            }
                            goto LEFT_DONE;
                        }
                        if (fp_st == FP_STATUS_TIGHT) {
                            // Fast-path failed but tight_band valid. Pipe
                            // to SW via sp.tight_band.
                            sp.tight_band = fp_band;
                            tprof[UGP_L_TIGHT][tid]++;
                            if (fp_band <= 8)        tprof[UGP_L_TB_1_8][tid]++;
                            else if (fp_band <= 32)  tprof[UGP_L_TB_9_32][tid]++;
                            else                     tprof[UGP_L_TB_33_MAX][tid]++;

                        }
                        // FALLBACK: sp.tight_band stays 0.
                    }

#if BWAMEM3_UGP_PROFILE
                    /* Per-pair tier routing + tight_band histograms. The
                     * numPairsLeft* increments here are dead (sortPairsLenExt
                     * below resets and recomputes all three counters, and
                     * nothing reads them before that) — only t_tier feeds the
                     * UGP histograms, so the whole block is instrumentation. */
                    {
                        int64_t minval = (int64_t)sp.h0 + (int64_t)min_(sp.len1, sp.len2) * (int64_t)opt->a;
                        int t_tier;
                        /* Mirror the routing decision in sortPairsLenExt: 8-bit
                         * iff the safe envelope holds (initial band opt->w). */
                        if (bsw8_envelope_ok(sp.len1, sp.len2, opt->w, opt->a, opt->zdrop, sp.h0)) {
                            numPairsLeft128++; t_tier = 0;
                        }
                        else if (sp.len1 < MAX_SEQ_LEN16 && sp.len2 < MAX_SEQ_LEN16 && minval >= 0 && minval < MAX_SEQ_LEN16){
                            numPairsLeft16++;  t_tier = 1;
                        }
                        else {
                            numPairsLeft1++;   t_tier = 2;  /* scalar; not bucketed */
                        }
                        /* tight_band histogram bin. */
                        int tb_ = sp.tight_band;
                        int fine_bin;
                        if      (tb_ ==  0) fine_bin = 0;
                        else if (tb_ <=  2) fine_bin = 1;
                        else if (tb_ <=  4) fine_bin = 2;
                        else if (tb_ <=  8) fine_bin = 3;
                        else if (tb_ <= 16) fine_bin = 4;
                        else if (tb_ <= 32) fine_bin = 5;
                        else if (tb_ <= 48) fine_bin = 6;
                        else if (tb_ <= 64) fine_bin = 7;
                        else if (tb_ <= 80) fine_bin = 8;
                        else                fine_bin = 9;
                        tprof[UGP_FINE_BASE + 0 * UGP_FINE_NBINS + fine_bin][tid]++;
                        if (t_tier <= 1) {
                            int band_bin;
                            if      (tb_ ==  0) band_bin = 0;
                            else if (tb_ <=  8) band_bin = 1;
                            else if (tb_ <= 32) band_bin = 2;
                            else                band_bin = 3;
                            tprof[UGP_TIER_TB_BASE + 0 * 8 + t_tier * 4 + band_bin][tid]++;
                        }
                    }
#endif

#if BWAMEM3_UGP_PROFILE
                    /* Q3: compute would-be ungapped extension score for this
                     * non-HIT LEFT pair. The walk handles arbitrary N
                     * (including the bypass case len2 > FP_N_MAX where
                     * analyze did not run). Cost: O(len2) scalar; instrumentation
                     * only (feeds ugp_record_left_outcome's tprof histograms). */
                    sp.ugp_walk_score = ungapped_walk_score(qs, rs, sp.len2,
                                                            sp.h0, opt->a, opt->b);
#endif

                    seqPairArrayLeft128[numPairsLeft] = sp;
                    numPairsLeft ++;
                    a->qb = s->qbeg; a->rb = s->rbeg;
                    LEFT_DONE: ;
                }
                else
                {
                    flag = 1;
                    a->score = a->truesc = meth_seed_sc, a->qb = 0, a->rb = s->rbeg;  /* D3: true seed score */
                }

                if (s->qbeg + s->len != l_query)  // right extension
                {
                    int64_t qe = s->qbeg + s->len;
                    int64_t re = s->rbeg + s->len - rmax[0];
                    assert(re >= 0);
                    SeqPair sp;

                    sp.h0 = H0_; //random number
                    sp.seqid = c->seqid;
                    sp.regid = av->n - 1;

                    if (numPairsRight >= *wsize_pair)
                    {
                        if (bwa_verbose >= 4) fprintf(stderr, "[0000] [%0.4d] Re-allocating seqPairArrays Right\n", tid);
                        *wsize_pair += 1024;
                        // assert(*wsize_pair > numPairsRight);
                        *wsize_pair += numPairsLeft + 1024;
                        seqPairArrayAux = (SeqPair *) realloc(seqPairArrayAux,
                                                              (*wsize_pair + MAX_LINE_LEN)
                                                              * sizeof(SeqPair));
                        mmc->seqPairArrayAux[tid] = seqPairArrayAux;
                        seqPairArrayLeft128 = (SeqPair *) realloc(seqPairArrayLeft128,
                                                                  (*wsize_pair + MAX_LINE_LEN)
                                                                  * sizeof(SeqPair));
                        mmc->seqPairArrayLeft128[tid] = seqPairArrayLeft128;
                        seqPairArrayRight128 = (SeqPair *) realloc(seqPairArrayRight128,
                                                                   (*wsize_pair + MAX_LINE_LEN)
                                                                   * sizeof(SeqPair));
                        mmc->seqPairArrayRight128[tid] = seqPairArrayRight128;
                    }

                    sp.len2 = l_query - qe;
                    sp.len1 = rmax[1] - rmax[0] - re;

                    sp.idq = rightQerOffset;
                    sp.idr = rightRefOffset;

                    rightQerOffset += sp.len2;
                    if (rightQerOffset >= *wsize_buf_qer)
                    {
                        if (bwa_verbose >= 4) fprintf(stderr, "[%0.4d] Re-allocating (doubling) seqBufQers in %s (right)\n",
                                tid, __func__);
                        int64_t tmp = *wsize_buf_qer;
                        *wsize_buf_qer = seqbuf_grow_capacity(tmp);
                        if (*wsize_buf_qer == SEQBUF_CAPACITY_OVERFLOW)
                            seqbuf_capacity_fatal("seqBufQer", __func__, tmp);
                        assert(*wsize_buf_qer > rightQerOffset);

                        uint8_t *seqBufQer_ = (uint8_t*)
                            _mm_realloc(seqBufLeftQer, tmp, *wsize_buf_qer, sizeof(uint8_t));
                        mmc->seqBufLeftQer[tid*CACHE_LINE] = seqBufLeftQer = seqBufQer_;

                        seqBufQer_ = (uint8_t*)
                            _mm_realloc(seqBufRightQer, tmp, *wsize_buf_qer, sizeof(uint8_t));
                        mmc->seqBufRightQer[tid*CACHE_LINE] = seqBufRightQer = seqBufQer_;
                    }

                    rightRefOffset += sp.len1;
                    if (rightRefOffset >= *wsize_buf_ref)
                    {
                        if (bwa_verbose >= 4) fprintf(stderr, "[%0.4d] Re-allocating (doubling) seqBufRefs in %s (right)\n",
                                tid, __func__);
                        int64_t tmp = *wsize_buf_ref;
                        *wsize_buf_ref = seqbuf_grow_capacity(tmp);
                        if (*wsize_buf_ref == SEQBUF_CAPACITY_OVERFLOW)
                            seqbuf_capacity_fatal("seqBufRef", __func__, tmp);
                        assert(*wsize_buf_ref > rightRefOffset);
                        uint8_t *seqBufRef_ = (uint8_t*)
                            _mm_realloc(seqBufLeftRef, tmp, *wsize_buf_ref, sizeof(uint8_t));
                        mmc->seqBufLeftRef[tid*CACHE_LINE] = seqBufLeftRef = seqBufRef_;

                        seqBufRef_ = (uint8_t*)
                            _mm_realloc(seqBufRightRef, tmp, *wsize_buf_ref, sizeof(uint8_t));
                        mmc->seqBufRightRef[tid*CACHE_LINE] = seqBufRightRef = seqBufRef_;
                    }

                    tprof[PE23][tid] += sp.len1 + sp.len2;

                    uint8_t *qs = seqBufRightQer + sp.idq;
                    uint8_t *rs = seqBufRightRef + sp.idr;

                    for (int i = 0; i < sp.len2; ++i) qs[i] = query[qe + i];

                    for (int i = 0; i < sp.len1; ++i) rs[i] = rseq[re + i]; //seq1

                    sp.tight_band = 0; sp.chain_band = chain_band;
                    sp.ugp_r_attempted = 0;

                    // ungapped analysis on right ext.
                    // Precondition a->score != -1 means left either didn't
                    // need extension or was fast-pathed — in both cases
                    // h0 is known. If left was batched, we skip (can't
                    // know h0 yet); SW will run with default band.
                    /* D3 (--meth, PR-4): disable the ungapped fast path under
                     * --meth — see the LEFT-side rationale above (it can't score
                     * the asymmetric OT/OB matrix). */
                    if (!opt->meth_mode && a->score != -1 && sp.len1 >= sp.len2 && sp.len2 <= FP_N_MAX) {
                        sp.ugp_r_attempted = 1;
                        tprof[UGP_R_ATTEMPT][tid]++;
                        int fp_h0 = a->score;  // the real h0 for right ext
                        int fp_score, fp_qle, fp_gscore, fp_gtle, fp_band;
                        int fp_st = ungapped_analyze(qs, rs, sp.len2,
                                                     fp_h0, opt->a, opt->b,
                                                     fp_o_min, fp_e_min,
                                                     fp_x_threshold, opt->w,
                                                     &fp_score, &fp_qle,
                                                     &fp_gscore, &fp_gtle,
                                                     &fp_band);
                        if (fp_st == FP_STATUS_HIT) {
                            tprof[UGP_R_HIT][tid]++;
                            tprof[UGP_R_UNGAPPED][tid]++;
                            tprof[UGP_SCORE_HIST_BASE + 1 * UGP_SCORE_HIST_NBINS
                                  + ugp_score_bin(fp_score)][tid]++;
                            // Roll back the qs/rs buffer offsets.
                            rightQerOffset -= sp.len2;
                            rightRefOffset -= sp.len1;
                            // Mirror post-SW extraction (~line 2777).
                            a->score = fp_score;
                            if (fp_gscore <= 0 || fp_gscore <= a->score - opt->pen_clip3) {
                                a->qe = qe + fp_qle;
                                a->re = rmax[0] + re + fp_qle; // tle == qle on diagonal
                                a->truesc += a->score - fp_h0;
                            } else {
                                a->qe = l_query;
                                a->re = rmax[0] + re + fp_gtle;
                                a->truesc += fp_gscore - fp_h0;
                            }
                            a->w = max_(a->w, opt->w);
                            if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_) {
                                int ii;
                                for (ii = 0, a->seedcov = 0; ii < a->c->n; ++ii) {
                                    const mem_seed_t *t = &a->c->seeds[ii];
                                    if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                                        t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                                        a->seedcov += t->len;
                                }
                            }
                            goto RIGHT_DONE;
                        }
                        if (fp_st == FP_STATUS_TIGHT) {
                            sp.tight_band = fp_band;
                            tprof[UGP_R_TIGHT][tid]++;
                            if (fp_band <= 8)        tprof[UGP_R_TB_1_8][tid]++;
                            else if (fp_band <= 32)  tprof[UGP_R_TB_9_32][tid]++;
                            else                     tprof[UGP_R_TB_33_MAX][tid]++;

                        }
                    }

                    /* The RIGHT-side tier counters (numPairsRight128/16/1) are
                     * recomputed from scratch by sortPairsLenExt below (which
                     * resets them to 0 first), and nothing reads them before
                     * that call — so the per-pair envelope check and routing
                     * that used to live here were dead work. Just stage the
                     * pair; the sort assigns its tier. RIGHT histograms (Groups
                     * A & B) are populated post-left-SW once the right pass has
                     * finalised sp->tight_band. */
                    seqPairArrayRight128[numPairsRight] = sp;
                    numPairsRight ++;
                    a->qe = qe; a->re = rmax[0] + re;
                    RIGHT_DONE: ;
                }
                else
                {
                    a->qe = l_query, a->re = s->rbeg + s->len;
                    // Do NOT remove this seedcov recompute: it is load-bearing.
                    // This is the no-right-extension branch (s->qbeg + s->len ==
                    // l_query). When the seed also needs no left extension
                    // (s->qbeg == 0, i.e. it spans the whole read, as perfect-match
                    // reads do) no SW pair is staged on either side, so none of the
                    // other seedcov recompute sites run -- neither the right-HIT
                    // path above, nor the post-SW LEFT-accept loops, nor the
                    // RIGHT-accept loops. In that case this block is the ONLY writer
                    // of a->seedcov. Dropping it would leave seedcov stale and change
                    // mem_approx_mapq_se (log(seedcov)), altering the emitted MAPQ.
                    if (a->rb != H0_ && a->qb != H0_)
                    {
                        int i;
                        for (i = 0, a->seedcov = 0; i < c->n; ++i)
                        {
                            const mem_seed_t *t = &c->seeds[i];
                            if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                                t->rbeg >= a->rb && t->rbeg + t->len <= a->re) // seed fully contained
                                a->seedcov += t->len;
                        }
                    }
                }
            }
            // tprof[MEM_ALN2_DOWN1][tid] += __rdtsc() - tim;
        }
    }
    // tprof[MEM_ALN2_UP][tid] += __rdtsc() - timUP;


    int32_t *hist = (int32_t *)_mm_malloc((MAX_SEQ_LEN8 + MAX_SEQ_LEN16 + 32) *
                                          sizeof(int32_t), 64);

    /* Sorting based score is required as that affects the use of SIMD lanes */
    sortPairsLenExt(seqPairArrayLeft128, numPairsLeft, seqPairArrayAux, hist,
                    numPairsLeft128, numPairsLeft16, numPairsLeft1, opt->a,
                    opt->w, opt->zdrop);
    assert(numPairsLeft == (numPairsLeft128 + numPairsLeft16 + numPairsLeft1));

#if BWAMEM3_UGP_PROFILE
    /* instrumentation: per-batch narrow bucket size (Group C, LEFT).
     * Counts pairs in the 8-bit tier with tight_band ∈ [1,8] for THIS worker
     * batch — the size a future narrow bucket would have. */
    {
        int narrow_sz = 0;
        for (int l = 0; l < numPairsLeft128; l++) {
            int tb = seqPairArrayLeft128[l].tight_band;
            if (tb > 0 && tb <= 8) narrow_sz++;
        }
        int bin;
        if      (narrow_sz ==   0) bin = 0;
        else if (narrow_sz <   16) bin = 1;
        else if (narrow_sz <   32) bin = 2;
        else if (narrow_sz <   64) bin = 3;
        else if (narrow_sz <  128) bin = 4;
        else if (narrow_sz <  256) bin = 5;
        else if (narrow_sz <  512) bin = 6;
        else                       bin = 7;
        tprof[UGP_NARROW_SZ_BASE + 0 * UGP_NARROW_SZ_NBINS + bin][tid]++;
    }
#endif


    // SWA
    // uint64_t timL = __rdtsc();
    int nthreads = 1;

    /* D3 (--meth, A1/PR-4): per-batch substitution-matrix objects. The banded-SW
     * kernel reads this->mat once per getScores16/getScores8/scalar call for ALL
     * pairs it processes (the matrix is per-SW-object, not per-pair).
     *
     * The symmetric objects (bswLeft/bswRight, built from opt->mat) drive the
     * entire non-meth pipeline and, under --meth, the non-meth (hypothesis < 0)
     * pairs. Under --meth we additionally build per-hypothesis OT/OB objects;
     * bsw_run_tier() partitions each tier's pairs by hypothesis and scores each
     * partition against its matching object, so a MIXED-hypothesis batch (the
     * common paired-end case: R1 OT + R2 OB interleaved) is scored correctly in
     * original space — the remaining PR-4 work, replacing the PR-3 symmetric
     * fallback. A homogeneous batch (e.g. single-end directional input) yields a
     * single non-empty partition, so there is no extra cost there.
     *
     * Outside --meth only the symmetric objects are built and used, and the path
     * is byte-for-byte identical to baseline. */
    // Now, process all the collected seq-pairs
    // First, left alignment, move out these calls
    auto bswLeft = make_banded_pair_wise_sw(
        opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
        opt->zdrop, opt->pen_clip5,
        opt->mat, opt->a, opt->b, nthreads);

    auto bswRight = make_banded_pair_wise_sw(
        opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
        opt->zdrop, opt->pen_clip3,
        opt->mat, opt->a, opt->b, nthreads);

    /* Per-hypothesis OT/OB objects (only under --meth). Left/right differ solely
     * in the clip-extension end bonus (pen_clip5 vs pen_clip3), matching the
     * symmetric objects above. */
    std::unique_ptr<IBandedPairWiseSW> bswLeftOt, bswLeftOb, bswRightOt, bswRightOb;
    if (opt->meth_mode) {
        const int8_t *m_ot = mem_opt_meth_mat(opt, 1);   /* OT: frees C->T */
        const int8_t *m_ob = mem_opt_meth_mat(opt, 0);   /* OB: frees G->A */
        bswLeftOt  = make_banded_pair_wise_sw(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                                              opt->zdrop, opt->pen_clip5, m_ot, opt->a, opt->b, nthreads);
        bswLeftOb  = make_banded_pair_wise_sw(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                                              opt->zdrop, opt->pen_clip5, m_ob, opt->a, opt->b, nthreads);
        bswRightOt = make_banded_pair_wise_sw(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                                              opt->zdrop, opt->pen_clip3, m_ot, opt->a, opt->b, nthreads);
        bswRightOb = make_banded_pair_wise_sw(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                                              opt->zdrop, opt->pen_clip3, m_ob, opt->a, opt->b, nthreads);
        /* OT/OB kernels score a SUB-SLICE of the pair array in bsw_run_tier, so
         * their padding-lane overshoot would corrupt the next slice: enable the
         * guard on exactly these objects. The SYM object (bswLeft/bswRight) is a
         * whole-array / last-slice caller and stays unguarded. */
        bswLeftOt->set_guard_overshoot(true);
        bswLeftOb->set_guard_overshoot(true);
        bswRightOt->set_guard_overshoot(true);
        bswRightOb->set_guard_overshoot(true);
    }

    int i;
    // Left
    SeqPair *pair_ar = seqPairArrayLeft128 + numPairsLeft128 + numPairsLeft16;
    SeqPair *pair_ar_aux = seqPairArrayAux;
    int nump = numPairsLeft1;

    // per-pair tight_band proofs are still piped in via
    // sp->tight_band (and short-circuit the retry loop below once w >=
    // tight_band), but we no longer narrow init_w from opt->w. A batched
    // SW pass shares one band across all pairs in the batch, so narrowing
    // would force FALLBACK pairs (no tight_band proof) to start with a
    // band insufficient for indels their alignment really needs. The
    // heuristic exits (a->score == prev / max_off < 3w/4) can then fire
    // on a suboptimal alignment found within the narrow band, breaking
    // chr22 parity.
    int init_w = INIT_W(opt->w);

    // scalar
    for ( i=0; i<BAND_NBAND(init_w); i++)
    {
        int32_t w = BAND_WIDTH(init_w, i);
        // uint64_t tim = __rdtsc();
        bsw_run_tier(BSW_TIER_SCALAR, opt, av_v,
                     bswLeft.get(), bswLeftOt.get(), bswLeftOb.get(),
                     pair_ar, seqBufLeftRef, seqBufLeftQer, nump, nthreads, w,
                     pair_ar_aux, pair_ar_aux, hist);  /* CodeRabbit: base_scratch must
                     * track pair_ar_aux (distinct from pair_ar after the retry swap),
                     * not the static seqPairArrayAux which pair_ar can alias. */
        // tprof[PE5][0] += nump;
        // tprof[PE6][0] ++;
        // tprof[MEM_ALN2_B][tid] += __rdtsc() - tim;

        int num = 0;
        for (int l=0; l<nump; l++)
        {
            mem_alnreg_t *a;
            SeqPair *sp = &pair_ar[l];
            a = &(av_v[sp->seqid].a[sp->regid]); // prev
            int prev = a->score;
            a->score = sp->score;

            if (ACCEPT_PAIR(a->score, prev, sp->max_off, w, sp->tight_band, sp->chain_band, i))
            {
#if BWAMEM3_UGP_PROFILE
                ugp_record_left_outcome(sp, opt->a, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip5) {
                    a->qb -= sp->qle; a->rb -= sp->tle;
                    a->truesc = a->score;
                } else {
                    a->qb = 0; a->rb -= sp->gtle;
                    a->truesc = sp->gscore;
                }

                a->w = max_(a->w, w);
                if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_)
                {
                    int i = 0;
                    for (i = 0, a->seedcov = 0; i < a->c->n; ++i){
                        const mem_seed_t *t = &(a->c->seeds[i]);
                        if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                            t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                            a->seedcov += t->len;
                    }
                }

            } else {
                /* GAP2 restore (see band_cert_accept): un-pollute the ladder's `prev`. */
                if (opt->band_cert && w < opt->w) a->score = prev;
                pair_ar_aux[num++] = *sp;
            }
        }
        nump = num;
        SeqPair *tmp = pair_ar;
        pair_ar = pair_ar_aux;
        pair_ar_aux = tmp;
    }


    //****************** Left - vector int16 ***********************
    assert(numPairsLeft == (numPairsLeft128 + numPairsLeft16 + numPairsLeft1));

    pair_ar = seqPairArrayLeft128 + numPairsLeft128;
    pair_ar_aux = seqPairArrayAux;

    nump = numPairsLeft16;
    init_w = INIT_W(opt->w);
    /* item 2: certified tight_band probe rung on the 16-bit tier, as a PURE PREPEND. The 16-bit
     * ladder already probes narrow (its first rung is INIT_W(opt->w) = 20); the rung certifies the
     * proven-narrow (tb>0, w_need <= w_cap) pairs at a per-w_need bucket <= 20/30 and removes them
     * early, byte-identically (a certified pair's fields are band-invariant beyond its bucket, so
     * the bucket and the width-20 probe finalize identically). Everything else -- ineligible pairs
     * (tb=0, or w_need > w_cap) and the rare uncertified pair -- falls through to the UNCHANGED
     * [20, opt->w, ...] ladder and keeps its cheap w=20 first probe.
     *
     * init_w is deliberately NOT raised to opt->w here (it is on the 8-bit tier, which has no w=20
     * probe to preserve). The 16-bit tier is ~all tb=0 on short reads; raising init_w would force
     * those ineligible pairs to probe at opt->w first (5x wider) -- a measured ~0.12% regression on
     * 2x150 with no offsetting narrow population. As a pure prepend the rung is zero-cost on short
     * reads (no eligible pairs -> immediate return) and wins on long-read/SBX (~25% 16-bit tier). */
    nump = bsw_tb_probe_rung(BSW_TIER_16, opt, av_v, bswLeft.get(), bswLeftOt.get(), bswLeftOb.get(),
                             pair_ar, seqBufLeftRef, seqBufLeftQer, nump, nthreads,
                             pair_ar_aux, hist, seq_, opt->pen_clip5, /*is_left=*/1, tid);
    for ( i=0; i<BAND_NBAND(init_w); i++)
    {
        int32_t w = BAND_WIDTH(init_w, i);
        // int64_t tim = __rdtsc();
        bsw_run_tier(BSW_TIER_16, opt, av_v,
                     bswLeft.get(), bswLeftOt.get(), bswLeftOb.get(),
                     pair_ar, seqBufLeftRef, seqBufLeftQer, nump, nthreads, w,
                     pair_ar_aux, pair_ar_aux, hist);  /* CodeRabbit: base_scratch must
                     * track pair_ar_aux (distinct from pair_ar after the retry swap),
                     * not the static seqPairArrayAux which pair_ar can alias. */

        tprof[PE5][0] += nump;
        tprof[PE6][0] ++;
        // tprof[MEM_ALN2_B][tid] += __rdtsc() - tim;

        int num = 0;
        for (int l=0; l<nump; l++)
        {
            mem_alnreg_t *a;
            SeqPair *sp = &pair_ar[l];
            a = &(av_v[sp->seqid].a[sp->regid]); // prev

            int prev = a->score;
            a->score = sp->score;


            if (ACCEPT_PAIR(a->score, prev, sp->max_off, w, sp->tight_band, sp->chain_band, i))
            {
#if BWAMEM3_UGP_PROFILE
                ugp_record_left_outcome(sp, opt->a, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip5) {
                    a->qb -= sp->qle; a->rb -= sp->tle;
                    a->truesc = a->score;
                } else {
                    a->qb = 0; a->rb -= sp->gtle;
                    a->truesc = sp->gscore;
                }

                a->w = max_(a->w, w);
                if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_)
                {
                    int i = 0;
                    for (i = 0, a->seedcov = 0; i < a->c->n; ++i){
                        const mem_seed_t *t = &(a->c->seeds[i]);
                        if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                            t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                            a->seedcov += t->len;
                    }
                }
            } else {
                /* GAP2 restore (see band_cert_accept): un-pollute the ladder's `prev`. */
                if (opt->band_cert && w < opt->w) a->score = prev;
                pair_ar_aux[num++] = *sp;
            }
        }
        nump = num;
        SeqPair *tmp = pair_ar;
        pair_ar = pair_ar_aux;
        pair_ar_aux = tmp;
    }

    //****************** Left - vector int8 ***********************
    pair_ar = seqPairArrayLeft128;
    pair_ar_aux = seqPairArrayAux;

    nump = numPairsLeft128;
    init_w = opt->w;
    /* commit 4: certified tight_band probe rung. Finalizes the proven-narrow
     * 8-bit pairs at the probe width (byte-identical); returns the count left
     * (ineligible + uncertified), compacted to the front, for the ladder. */
    nump = bsw_tb_probe_rung(BSW_TIER_8, opt, av_v, bswLeft.get(), bswLeftOt.get(), bswLeftOb.get(),
                             pair_ar, seqBufLeftRef, seqBufLeftQer, nump, nthreads,
                             pair_ar_aux, hist, seq_, opt->pen_clip5, /*is_left=*/1, tid);
    for ( i=0; i<BAND_NBAND(init_w); i++)
    {
        int32_t w = BAND_WIDTH(init_w, i);
        // int64_t tim = __rdtsc();

        /* These pairs were bucketed for the 8-bit path at the initial band
         * opt->w (which the envelope proved <= BSW8_MAX_W). Once the doubling
         * retry pushes w >= 128 the diagonal-offset int8 encoding can no longer
         * represent the band, so bsw_run_tier (BSW_TIER_8) diverts that
         * iteration to the 16-bit kernel rather than feeding getScores8 an
         * unrepresentable band. The tight_band early-exit below is unaffected
         * (it short-circuits the retry regardless of which width ran). */
        bsw_run_tier(BSW_TIER_8, opt, av_v,
                     bswLeft.get(), bswLeftOt.get(), bswLeftOb.get(),
                     pair_ar, seqBufLeftRef, seqBufLeftQer, nump, nthreads, w,
                     pair_ar_aux, pair_ar_aux, hist);  /* CodeRabbit: base_scratch must
                     * track pair_ar_aux (distinct from pair_ar after the retry swap),
                     * not the static seqPairArrayAux which pair_ar can alias. */

        tprof[PE1][0] += nump;
        tprof[PE2][0] ++;
        // tprof[MEM_ALN2_D][tid] += __rdtsc() - tim;

        int num = 0;
        for (int l=0; l<nump; l++)
        {
            mem_alnreg_t *a;
            SeqPair *sp = &pair_ar[l];
            a = &(av_v[sp->seqid].a[sp->regid]); // prev

            int prev = a->score;
            a->score = sp->score;

            if (ACCEPT_PAIR(a->score, prev, sp->max_off, w, sp->tight_band, sp->chain_band, i))
            {
#if BWAMEM3_UGP_PROFILE
                ugp_record_left_outcome(sp, opt->a, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip5) {
                    a->qb -= sp->qle; a->rb -= sp->tle;
                    a->truesc = a->score;
                } else {
                    a->qb = 0; a->rb -= sp->gtle;
                    a->truesc = sp->gscore;
                }

                a->w = max_(a->w, w);
                if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_)
                {
                    int i = 0;
                    for (i = 0, a->seedcov = 0; i < a->c->n; ++i){
                        const mem_seed_t *t = &(a->c->seeds[i]);
                        if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                            t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                            a->seedcov += t->len;
                    }
                }
            } else {
                /* GAP2 restore (see band_cert_accept): un-pollute the ladder's `prev`. */
                if (opt->band_cert && w < opt->w) a->score = prev;
                pair_ar_aux[num++] = *sp;
            }
        }
        nump = num;
        SeqPair *tmp = pair_ar;
        pair_ar = pair_ar_aux;
        pair_ar_aux = tmp;
    }

    // tprof[CLEFT][tid] += __rdtsc() - timL;

    // uint64_t timR = __rdtsc();
    // **********************************************************
    // h0 fixup: right-extension h0 is the left-extension alnreg score.
    for (int l=0; l<numPairsRight; l++) {
        mem_alnreg_t *a;
        SeqPair *sp = &seqPairArrayRight128[l];
        a = &(av_v[sp->seqid].a[sp->regid]); // prev
        sp->h0 = a->score;
    }

    // post-left-SW right-side ungapped pass. The per-seed fast-
    // path skipped right ext for 93% of pairs because a->score was still
    // -1 at construction time (left SW hadn't run yet). Now that left
    // SW is done, a->score is final and h0 is set above. Run ungapped
    // analysis on every right pair:
    //   HIT    → fill a->* directly, compact the pair out of the array.
    //   TIGHT  → set sp->tight_band for the SW dispatch.
    //   FALL   → leave tight_band at construction default (typically 0).
    {
        int compacted = 0;
        for (int l = 0; l < numPairsRight; l++) {
            SeqPair *sp = &seqPairArrayRight128[l];
            mem_alnreg_t *a = &(av_v[sp->seqid].a[sp->regid]);
            int keep = 1;
            // Skip pairs already analyzed at construction time (a->score
            // was non-(-1) then; UGP_R_ATTEMPT and outcome counters fired
            // there). Without this guard, those pairs would double-count.
            // !opt->meth_mode: ungapped_analyze hardcodes symmetric opt->a/opt->b
            // and cannot honor the per-strand asymmetric matrix, so a HIT here
            // would fill a->score / qe / re from symmetric scoring and compact
            // the pair out, bypassing the asymmetric banded SW. The two
            // construction-time passes are gated the same way (see ~3397/3618).
            if (!opt->meth_mode && !sp->ugp_r_attempted &&
                sp->len1 >= sp->len2 && sp->len2 > 0 && sp->len2 <= FP_N_MAX) {
                tprof[UGP_R_ATTEMPT][tid]++;
                const uint8_t *qs = seqBufRightQer + sp->idq;
                const uint8_t *rs = seqBufRightRef + sp->idr;
                int fp_score, fp_qle, fp_gscore, fp_gtle, fp_band;
                int fp_st = ungapped_analyze(qs, rs, sp->len2, sp->h0,
                                              opt->a, opt->b,
                                              fp_o_min, fp_e_min,
                                              fp_x_threshold, opt->w,
                                              &fp_score, &fp_qle,
                                              &fp_gscore, &fp_gtle,
                                              &fp_band);
                if (fp_st == FP_STATUS_HIT) {
                    tprof[UGP_R_HIT][tid]++;
                    tprof[UGP_R_UNGAPPED][tid]++;
                    tprof[UGP_SCORE_HIST_BASE + 1 * UGP_SCORE_HIST_NBINS
                          + ugp_score_bin(fp_score)][tid]++;
                    // Mirror post-right-SW extraction. a->qe / a->re were
                    // set in the per-seed construction loop to the pre-
                    // extension anchor positions (s->qbeg+s->len / s->rbeg+s->len).
                    int l_query = seq_[sp->seqid].l_seq;
                    a->score = fp_score;
                    if (fp_gscore <= 0 || fp_gscore <= a->score - opt->pen_clip3) {
                        a->qe += fp_qle;
                        a->re += fp_qle; // tle == qle on diagonal
                        a->truesc += a->score - sp->h0;
                    } else {
                        a->qe = l_query;
                        a->re += fp_gtle;
                        a->truesc += fp_gscore - sp->h0;
                    }
                    a->w = max_(a->w, opt->w);
                    if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_) {
                        int ii;
                        for (ii = 0, a->seedcov = 0; ii < a->c->n; ++ii) {
                            const mem_seed_t *t = &a->c->seeds[ii];
                            if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                                t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                                a->seedcov += t->len;
                        }
                    }
                    keep = 0;
                } else if (fp_st == FP_STATUS_TIGHT) {
                    sp->tight_band = fp_band;
                    tprof[UGP_R_TIGHT][tid]++;
                    if (fp_band <= 8)        tprof[UGP_R_TB_1_8][tid]++;
                    else if (fp_band <= 32)  tprof[UGP_R_TB_9_32][tid]++;
                    else                     tprof[UGP_R_TB_33_MAX][tid]++;

                }
            }
            if (keep) {
                if (compacted != l)
                    seqPairArrayRight128[compacted] = *sp;
                compacted++;
            }
        }
        numPairsRight = compacted;
    }

    sortPairsLenExt(seqPairArrayRight128, numPairsRight, seqPairArrayAux,
                    hist, numPairsRight128, numPairsRight16, numPairsRight1, opt->a,
                    opt->w, opt->zdrop);

    assert(numPairsRight == (numPairsRight128 + numPairsRight16 + numPairsRight1));

#if BWAMEM3_UGP_PROFILE
    /* instrumentation (Groups A, B, C — RIGHT). Run after the post-
     * left-SW analyze pass + tier sort: sp->tight_band is now final and the
     * 128 region is contiguous at the head of seqPairArrayRight128. */
    {
        int narrow_sz = 0;
        for (int l = 0; l < numPairsRight; l++) {
            SeqPair *sp_ = &seqPairArrayRight128[l];
            int tb_ = sp_->tight_band;
            int t_tier;
            int64_t minval_ = (int64_t)sp_->h0 + (int64_t)min_(sp_->len1, sp_->len2) * (int64_t)opt->a;
            if (bsw8_envelope_ok(sp_->len1, sp_->len2, opt->w, opt->a, opt->zdrop, sp_->h0))             t_tier = 0;
            else if (sp_->len1 < MAX_SEQ_LEN16 && sp_->len2 < MAX_SEQ_LEN16 && minval_ >= 0 && minval_ < MAX_SEQ_LEN16) t_tier = 1;
            else                                                                                       t_tier = 2;
            int fine_bin;
            if      (tb_ ==  0) fine_bin = 0;
            else if (tb_ <=  2) fine_bin = 1;
            else if (tb_ <=  4) fine_bin = 2;
            else if (tb_ <=  8) fine_bin = 3;
            else if (tb_ <= 16) fine_bin = 4;
            else if (tb_ <= 32) fine_bin = 5;
            else if (tb_ <= 48) fine_bin = 6;
            else if (tb_ <= 64) fine_bin = 7;
            else if (tb_ <= 80) fine_bin = 8;
            else                fine_bin = 9;
            tprof[UGP_FINE_BASE + 1 * UGP_FINE_NBINS + fine_bin][tid]++;
            if (t_tier <= 1) {
                int band_bin;
                if      (tb_ ==  0) band_bin = 0;
                else if (tb_ <=  8) band_bin = 1;
                else if (tb_ <= 32) band_bin = 2;
                else                band_bin = 3;
                tprof[UGP_TIER_TB_BASE + 1 * 8 + t_tier * 4 + band_bin][tid]++;
            }
            if (t_tier == 0 && tb_ > 0 && tb_ <= 8) narrow_sz++;
        }
        int bin;
        if      (narrow_sz ==   0) bin = 0;
        else if (narrow_sz <   16) bin = 1;
        else if (narrow_sz <   32) bin = 2;
        else if (narrow_sz <   64) bin = 3;
        else if (narrow_sz <  128) bin = 4;
        else if (narrow_sz <  256) bin = 5;
        else if (narrow_sz <  512) bin = 6;
        else                       bin = 7;
        tprof[UGP_NARROW_SZ_BASE + 1 * UGP_NARROW_SZ_NBINS + bin][tid]++;
    }
#endif

    pair_ar = seqPairArrayRight128 + numPairsRight128 + numPairsRight16;
    pair_ar_aux = seqPairArrayAux;
    nump = numPairsRight1;
    init_w = INIT_W(opt->w);

    for ( i=0; i<BAND_NBAND(init_w); i++)
    {
        int32_t w = BAND_WIDTH(init_w, i);
        // tim = __rdtsc();
        bsw_run_tier(BSW_TIER_SCALAR, opt, av_v,
                     bswRight.get(), bswRightOt.get(), bswRightOb.get(),
                     pair_ar, seqBufRightRef, seqBufRightQer, nump, nthreads, w,
                     pair_ar_aux, pair_ar_aux, hist);  /* CodeRabbit: base_scratch must
                     * track pair_ar_aux (distinct from pair_ar after the retry swap),
                     * not the static seqPairArrayAux which pair_ar can alias. */
        // tprof[PE7][0] += nump;
        // tprof[PE8][0] ++;
        // tprof[MEM_ALN2_C][tid] += __rdtsc() - tim;
        int num = 0;

        for (int l=0; l<nump; l++)
        {
            mem_alnreg_t *a;
            SeqPair *sp = &pair_ar[l];
            a = &(av_v[sp->seqid].a[sp->regid]); // prev
            int prev = a->score;
            a->score = sp->score;

            // no further banding
            if (ACCEPT_PAIR(a->score, prev, sp->max_off, w, sp->tight_band, sp->chain_band, i))
            {
#if BWAMEM3_UGP_PROFILE
                ugp_record_right_outcome(sp, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip3) {
                    a->qe += sp->qle, a->re += sp->tle;
                    a->truesc += a->score - sp->h0;
                } else {
                    int l_query = seq_[sp->seqid].l_seq;
                    a->qe = l_query, a->re += sp->gtle;
                    a->truesc += sp->gscore - sp->h0;
                }
                a->w = max_(a->w, w);
                if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_)
                {
                    int i = 0;
                    for (i = 0, a->seedcov = 0; i < a->c->n; ++i) {
                        const mem_seed_t *t = &a->c->seeds[i];
                        if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                            t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                            a->seedcov += t->len;
                    }
                }
            } else {
                /* GAP2 restore (see band_cert_accept): un-pollute the ladder's `prev`. */
                if (opt->band_cert && w < opt->w) a->score = prev;
                pair_ar_aux[num++] = *sp;
            }
        }
        nump = num;
        SeqPair *tmp = pair_ar;
        pair_ar = pair_ar_aux;
        pair_ar_aux = tmp;
    }

    // ************************* Right - vector int16 **********************
    pair_ar = seqPairArrayRight128 + numPairsRight128;
    pair_ar_aux = seqPairArrayAux;
    nump = numPairsRight16;
    init_w = INIT_W(opt->w);
    /* item 2: certified tight_band probe rung on the 16-bit tier (mirror of the LEFT 16-bit
     * rung; pen_clip3, is_left=0). See the LEFT block for the byte-identity argument and why
     * init_w stays at INIT_W(opt->w) (pure prepend; no short-read regression). */
    nump = bsw_tb_probe_rung(BSW_TIER_16, opt, av_v, bswRight.get(), bswRightOt.get(), bswRightOb.get(),
                             pair_ar, seqBufRightRef, seqBufRightQer, nump, nthreads,
                             pair_ar_aux, hist, seq_, opt->pen_clip3, /*is_left=*/0, tid);

    for ( i=0; i<BAND_NBAND(init_w); i++)
    {
        int32_t w = BAND_WIDTH(init_w, i);
        // uint64_t tim = __rdtsc();
        bsw_run_tier(BSW_TIER_16, opt, av_v,
                     bswRight.get(), bswRightOt.get(), bswRightOb.get(),
                     pair_ar, seqBufRightRef, seqBufRightQer, nump, nthreads, w,
                     pair_ar_aux, pair_ar_aux, hist);  /* CodeRabbit: base_scratch must
                     * track pair_ar_aux (distinct from pair_ar after the retry swap),
                     * not the static seqPairArrayAux which pair_ar can alias. */

        tprof[PE7][0] += nump;
        tprof[PE8][0] ++;
        // tprof[MEM_ALN2_C][tid] += __rdtsc() - tim;

        int num = 0;

        for (int l=0; l<nump; l++)
        {
            mem_alnreg_t *a;
            SeqPair *sp = &pair_ar[l];
            a = &(av_v[sp->seqid].a[sp->regid]); // prev
            //OutScore *o = outScoreArray + l;
            int prev = a->score;
            a->score = sp->score;

            // no further banding
            if (ACCEPT_PAIR(a->score, prev, sp->max_off, w, sp->tight_band, sp->chain_band, i))
            {
#if BWAMEM3_UGP_PROFILE
                ugp_record_right_outcome(sp, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip3) {
                    a->qe += sp->qle, a->re += sp->tle;
                    a->truesc += a->score - sp->h0;
                } else {
                    int l_query = seq_[sp->seqid].l_seq;
                    a->qe = l_query, a->re += sp->gtle;
                    a->truesc += sp->gscore - sp->h0;
                }
                a->w = max_(a->w, w);
                if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_)
                {
                    int i = 0;
                    for (i = 0, a->seedcov = 0; i < a->c->n; ++i) {
                        const mem_seed_t *t = &a->c->seeds[i];
                        if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                            t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                            a->seedcov += t->len;
                    }
                }
            } else {
                /* GAP2 restore (see band_cert_accept): un-pollute the ladder's `prev`. */
                if (opt->band_cert && w < opt->w) a->score = prev;
                pair_ar_aux[num++] = *sp;
            }
        }
        nump = num;
        SeqPair *tmp = pair_ar;
        pair_ar = pair_ar_aux;
        pair_ar_aux = tmp;
    }


    // ************************* Right, vector int8 **********************
    pair_ar = seqPairArrayRight128;
    pair_ar_aux = seqPairArrayAux;
    nump = numPairsRight128;
    init_w = opt->w;
    /* commit 4: certified tight_band probe rung (mirror of the LEFT rung; pen_clip3). */
    nump = bsw_tb_probe_rung(BSW_TIER_8, opt, av_v, bswRight.get(), bswRightOt.get(), bswRightOb.get(),
                             pair_ar, seqBufRightRef, seqBufRightQer, nump, nthreads,
                             pair_ar_aux, hist, seq_, opt->pen_clip3, /*is_left=*/0, tid);

    for ( i=0; i<BAND_NBAND(init_w); i++)
    {
        int32_t w = BAND_WIDTH(init_w, i);
        // uint64_t tim = __rdtsc();

        /* See the LEFT int8 loop: bsw_run_tier (BSW_TIER_8) diverts w >= 128
         * retry iterations to the 16-bit kernel (the diagonal-offset int8 band
         * can't represent it). tight_band early-exit is preserved. */
        bsw_run_tier(BSW_TIER_8, opt, av_v,
                     bswRight.get(), bswRightOt.get(), bswRightOb.get(),
                     pair_ar, seqBufRightRef, seqBufRightQer, nump, nthreads, w,
                     pair_ar_aux, pair_ar_aux, hist);  /* CodeRabbit: base_scratch must
                     * track pair_ar_aux (distinct from pair_ar after the retry swap),
                     * not the static seqPairArrayAux which pair_ar can alias. */

        tprof[PE3][0] += nump;
        tprof[PE4][0] ++;
        // tprof[MEM_ALN2_E][tid] += __rdtsc() - tim;
        int num = 0;

        for (int l=0; l<nump; l++)
        {
            mem_alnreg_t *a;
            SeqPair *sp = &pair_ar[l];
            a = &(av_v[sp->seqid].a[sp->regid]); // prev
            //OutScore *o = outScoreArray + l;
            int prev = a->score;
            a->score = sp->score;
            // no further banding
            if (ACCEPT_PAIR(a->score, prev, sp->max_off, w, sp->tight_band, sp->chain_band, i))
            {
#if BWAMEM3_UGP_PROFILE
                ugp_record_right_outcome(sp, tid);
#endif
                if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip3) {
                    a->qe += sp->qle, a->re += sp->tle;
                    a->truesc += a->score - sp->h0;
                } else {
                    int l_query = seq_[sp->seqid].l_seq;
                    a->qe = l_query, a->re += sp->gtle;
                    a->truesc += sp->gscore - sp->h0;
                }
                a->w = max_(a->w, w);
                if (a->rb != H0_ && a->qb != H0_ && a->qe != H0_ && a->re != H0_)
                {
                    int i = 0;
                    for (i = 0, a->seedcov = 0; i < a->c->n; ++i) {
                        const mem_seed_t *t = &a->c->seeds[i];
                        if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe &&
                            t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
                            a->seedcov += t->len;
                    }
                }
            } else {
                /* GAP2 restore (see band_cert_accept): un-pollute the ladder's `prev`. */
                if (opt->band_cert && w < opt->w) a->score = prev;
                pair_ar_aux[num++] = *sp;
            }
        }
        nump = num;
        SeqPair *tmp = pair_ar;
        pair_ar = pair_ar_aux;
        pair_ar_aux = tmp;
    }

    _mm_free(hist);
    // tprof[CRIGHT][tid] += __rdtsc() - timR;

    if (numPairsLeft >= *wsize_pair || numPairsRight >= *wsize_pair)
    {   // refine it!
        fprintf(stderr, "Error: Unexpected behaviour!!!\n");
        fprintf(stderr, "Error: assert failed for seqPair size, "
                "numPairsLeft: %d, numPairsRight %d, lim: %lld\nExiting.\n",
                numPairsLeft, numPairsRight, (long long)*wsize_pair);
        exit(EXIT_FAILURE);
    }
    /* Discard seeds and hence their alignemnts */

    lim_g[0] = 0;
    for (int l=1; l<nseq; l++)
        lim_g[l] += lim_g[l-1];

    // uint64_t tim = __rdtsc();
    // BATCH_SIZE is a compile-time constant (1024 on arm64, 512 elsewhere);
    // sizeof(int)*BATCH_SIZE = 4096/2048 bytes — safe on stack, skips an
    // allocator round-trip per call to this function. nseq is bounded by
    // kt_for's BATCH_SIZE-chunked work distribution (worker_bwt at the
    // call site) — assert it for future-proofing.
    assert(nseq <= BATCH_SIZE);
    int lim[BATCH_SIZE] = {0};

    for (int l=0; l<nseq; l++)
    {
        int s_start = 0, s_end = 0;
        uint32_t *srtg = srtgg + lim_g[l];

        int l_query = seq_[l].l_seq;
        mem_chain_v *chn = &chain_ar[l];
        mem_alnreg_v *av = &av_v[l];  // alignment
        mem_chain_t *c;

        for (int j=0; j<chn->n; j++)
        {
            c = &chn->a[j];
            assert(c->seqid == l);

            s_end = s_start + c->n;

            uint32_t *srt2 = srtg + s_start;
            s_start += c->n;

            int k = 0;
            for (k = c->n-1; k >= 0; k--)
            {
                s = &c->seeds[srt2[k]];
                int i = 0;
                int v = 0;
                for (i = 0; i < av->n && v < lim[l]; ++i)  // test whether extension has been made before
                {
                    mem_alnreg_t *p = &av->a[i];
                    if (p->qb == -1 && p->qe == -1) {
                        continue;
                    }

                    int64_t rd;
                    int qd, w, max_gap;
                    if (s->rbeg < p->rb || s->rbeg + s->len > p->re || s->qbeg < p->qb
                        || s->qbeg + s->len > p->qe) {
                        v++; continue; // not fully contained
                    }

                    if (s->len - p->seedlen0 > .1 * l_query) { v++; continue;}
                    // qd: distance ahead of the seed on query; rd: on reference
                    qd = s->qbeg - p->qb; rd = s->rbeg - p->rb;
                    // the maximal gap allowed in regions ahead of the seed
                    max_gap = cal_max_gap(opt, qd < rd? qd : rd);
                    w = max_gap < p->w? max_gap : p->w; // bounded by the band width
                    if (qd - rd < w && rd - qd < w) break; // the seed is "around" a previous hit
                    // similar to the previous four lines, but this time we look at the region behind
                    qd = p->qe - (s->qbeg + s->len); rd = p->re - (s->rbeg + s->len);
                    max_gap = cal_max_gap(opt, qd < rd? qd : rd);
                    w = max_gap < p->w? max_gap : p->w;
                    if (qd - rd < w && rd - qd < w) break;

                    v++;
                }

                // the seed is (almost) contained in an existing alignment;
                // further testing is needed to confirm it is not leading
                // to a different aln
                if (v < lim[l])
                {
                    for (v = k + 1; v < c->n; ++v)
                    {
                        const mem_seed_t *t;
                        if (srt2[v] == UINT_MAX) continue;
                        t = &c->seeds[srt2[v]];
                        // only check overlapping if t is long enough;
                        // TODO: more efficient by early stopping
                        if (t->len < s->len * .95) continue;
                        if (s->qbeg <= t->qbeg && s->qbeg + s->len - t->qbeg >= s->len>>2 &&
                            t->qbeg - s->qbeg != t->rbeg - s->rbeg) break;
                        if (t->qbeg <= s->qbeg && t->qbeg + t->len - s->qbeg >= s->len>>2 &&
                            s->qbeg - t->qbeg != s->rbeg - t->rbeg) break;
                    }
                    if (v == c->n) {                  // no overlapping seeds; then skip extension
                        mem_alnreg_t *ar = &(av_v[l].a[s->aln]);
                        ar->qb = ar->qe = -1;         // purge the alingment
                        srt2[k] = UINT_MAX;
                        tprof[PE18][tid]++;
                        continue;
                    }
                }
                lim[l]++;
            }
        }
    }
    free(srtgg);
    free(srt);
    free(meth_qbuf);  /* D3 (--meth, PR-4): original-read 2-bit scratch; NULL outside --meth */
    // tprof[MEM_ALN2_DOWN][tid] += __rdtsc() - tim;
}
