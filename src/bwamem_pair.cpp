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

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "kstring.h"
#include "bwamem.h"
#include "bam_writer.h"
#include "kvec.h"
#include "u8vec_scratch.h"
#include "utils.h"
#include "ksw.h"
#include "bandedSWA.h"
#include "kswv.h"
#include "simd_dispatch.h"

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

/* ---- k-mer-anchored banded mate rescue (opt->rescue_kmer) --------------------
 * Mate rescue Smith-Watermans a read against a ~kilobase reference window chosen
 * by insert size (mem_matesw_batch_pre/post). With --rescue-kmer set, find the
 * read's dominant k-mer exact-match diagonal within that window and band the SW
 * to that diagonal +/- rescue_band, falling back to the full window when no
 * anchor gathers enough votes. Opt-in and NOT byte-identical: the narrowed
 * window yields a different suboptimal score (csub -> MAPQ), so it is off by
 * default and enabled by --fast. The kswv kernel is unchanged -- narrowing just
 * hands it a shorter reference window. */
#include <vector>
#include <cstdint>
#include <algorithm>

/* BWA_MEM3_DEBUG_RESCUE_STATS: count how the anchor gate actually resolved.
 * Wall time alone cannot distinguish "narrowing helped" from "narrowing almost
 * never fired", and the vote floor and uniqueness guard can silently drive the
 * narrowing rate to zero while the scan still costs its O(M + l_ms) pass. These
 * rates are the metric that explains a wall delta, so they are instrumented
 * rather than inferred. Off in normal builds (the counters do not exist), and
 * named per the BWA_MEM3_DEBUG_* convention the ci.yml -D list is linted
 * against. */
#ifdef BWA_MEM3_DEBUG_RESCUE_STATS
#  include <atomic>
std::atomic<uint64_t> g_rescue_stat_scans{0};      /* anchor scans that ran the vote */
std::atomic<uint64_t> g_rescue_stat_narrowed{0};   /* ... that banded the SW         */
std::atomic<uint64_t> g_rescue_stat_skipped{0};    /* ... that dropped the SW        */
#endif

namespace {

/* Result of one anchor scan. `scanned` is the contract that matters: it is false
 * on the two paths that return BEFORE the diagonal vote runs (window or read
 * shorter than k; no k-mer indexed at all), and on those paths best_v and
 * n_kmer are meaningless and MUST NOT be consumed. In particular the skip gate
 * keys on `scanned && best_v < ...`, never on `!narrowed` -- the edge-clamp
 * decline below scans successfully and can carry a strong best_v, so treating
 * "did not narrow" as "no anchor" would skip exactly the windows where the read
 * hangs off the window edge. */
struct matesw_anchor_t {
    int  best_v;     /* votes on the winning diagonal (valid iff scanned)       */
    int  n_kmer;     /* distinct query k-mers indexed (valid iff scanned)       */
    int  ob, oe;     /* narrowed window sub-range (valid iff narrowed)          */
    bool narrowed;   /* band the rescue SW to [ob,oe)                           */
    bool scanned;    /* the diagonal vote ran; the counts above are meaningful  */
};

/* Storage model: the index keeps only the NEWEST query position per k-mer code,
 * so each window k-mer casts exactly one vote.
 * The alternative -- chaining every query position, so a code occurring m times
 * in the read and n times in the window casts m*n votes -- never misses a vote
 * on the true diagonal, but costs a chain walk whose length grows as
 * read_length / alphabet^k and is what makes small k expensive.
 *
 * Measured on a 644k-pair bisulfite set: single-position storage narrows 98.29%
 * of scans versus 98.84% for chaining, at equal-or-better accuracy (recall at
 * MAPQ>=60 was 5 reads higher) and ~1% less wall time. The lost votes are the
 * repeated-k-mer case, where keeping the newest occurrence can retain the wrong
 * one; that costs half a percent of narrowing and buys back the chain walk. */

/* Find the dominant k-mer exact-match diagonal of the (oriented) read within the
 * reference window ref[0..M) and report both the banding decision and the raw
 * vote statistics the caller's skip gate needs. Narrowing requires the winning
 * diagonal to clear MEM_RESCUE_MIN_VOTES; the band is opt->rescue_band either
 * side, and is declined when the clamped span would come out shorter than the
 * query.
 *
 * Only mem_matesw_batch_pre calls this: it runs the scan once per enqueued pair
 * and stores the resulting offset in g_rescue_narrow_off (keyed by regid) for
 * mem_matesw_batch_post to read back, so _post never re-derives the window.
 * `collapse` reduces the alphabet for the anchor scan only (1 = C->T / OT,
 * 2 = G->A / OB, 0 = none) so exact k-mers survive the bisulfite conversion
 * under --meth; the SW still sees the real bases. */
static matesw_anchor_t matesw_kmer_anchor(const mem_opt_t *opt, const uint8_t *ref, int M,
                                          const uint8_t *ms, int l_ms, int is_rev, int collapse)
{
    matesw_anchor_t res = { 0, 0, 0, 0, false, false };

    /* The CLI rejects K outside [1,MEM_RESCUE_KMER_MAX]; clamp defensively so a
     * caller that sets opt->rescue_kmer directly cannot overflow the uint32 code. */
    int k = opt->rescue_kmer;
    if (k < 1) k = 1;
    if (k > MEM_RESCUE_KMER_MAX) k = MEM_RESCUE_KMER_MAX;
    /* Returns before the indexing loop below, so no slot was claimed and the
     * touched-slot reset is vacuous. See the reset-invariant note there. */
    if (M < k || l_ms < k) return res;

    auto col = [collapse](uint8_t b) -> uint8_t {
        if (b > 3) return b;
        if (collapse == 1 && b == 1) return 3;   /* C -> T */
        if (collapse == 2 && b == 2) return 0;   /* G -> A */
        return b;
    };

    /* oriented (RC for is_rev), collapsed query bases */
    static thread_local std::vector<uint8_t> q;
    if ((int)q.size() < l_ms) q.resize(l_ms);
    if (is_rev) for (int i = 0; i < l_ms; ++i) { uint8_t c = ms[l_ms-1-i]; q[i] = col((c < 4)? 3-c : 4); }
    else        for (int i = 0; i < l_ms; ++i) q[i] = col(ms[i]);

    /* index the read's k-mers in a fixed open-addressing hash (O(read) build,
     * O(1) lookup); reset only the touched slots each call.
     *
     * HMAX caps occupancy so both probe loops below always terminate on an empty
     * slot: a read offers up to l_ms - k + 1 distinct k-mers, which exceeds the
     * HS slots for a mate of ~1 kb or longer (the K=6 code space is 4096), and a
     * full table would make linear probing spin forever. touched.size() IS the
     * occupancy -- every insert pushes exactly one slot and the reset loop below
     * empties exactly those -- so no separate counter is needed. Stopping early
     * leaves a partial index, which is safe: the vote scan then sees fewer
     * anchors and falls back to the full window if votes stay under threshold. */
    constexpr int HS = 1024, HMASK = HS - 1;
    constexpr int HMAX = HS - (HS >> 2);                   /* stop inserting at 75% load */
    static_assert(HMAX < HS, "the table must retain empty slots or linear probing cannot terminate");
    static thread_local std::vector<uint32_t> slot_code;   /* 0xFFFFFFFF = empty */
    static thread_local std::vector<int>      slot_pos;     /* newest query pos per code */
    static thread_local std::vector<int>      touched;
    if ((int)slot_code.size() < HS) { slot_code.assign(HS, 0xFFFFFFFFu); slot_pos.assign(HS, -1); }
    touched.clear();

    uint32_t mask = (k >= 16) ? 0xffffffffu : ((1u << (2*k)) - 1);
    { uint32_t code = 0; int valid = 0;
      for (int i = 0; i < l_ms; ++i) {
          uint8_t c = q[i];
          if (c > 3) { valid = 0; code = 0; continue; }
          code = ((code << 2) | c) & mask;
          if (++valid >= k) {
              int pos = i - k + 1;
              uint32_t h = (code * 2654435761u) >> (32 - 10);   /* HS = 1<<10 */
              while (slot_code[h] != 0xFFFFFFFFu && slot_code[h] != code) h = (h + 1) & HMASK;
              if (slot_code[h] == 0xFFFFFFFFu) {
                  if ((int)touched.size() >= HMAX) break;   /* table full enough: stop indexing */
                  slot_code[h] = code; touched.push_back((int)h);
              }
              /* Keep only the NEWEST occurrence of this code, so each window
               * k-mer casts exactly one vote. See the storage-model note above
               * the function. */
              slot_pos[h] = pos;
          }
      } }
    /* Nothing was inserted, so the reset below would be a no-op: returning here
     * cannot leave a stale slot behind. Every path that DID insert falls through
     * to the single reset + return at the bottom -- see the invariant there. */
    if (touched.empty()) return res;

    /* vote diagonals d = (ref k-mer start) - (read k-mer start), index by d+l_ms */
    int span = M + l_ms + 1;
    static thread_local std::vector<int> dv;
    if ((int)dv.size() < span) dv.resize(span);
    std::fill(dv.begin(), dv.begin() + span, 0);
    int best_d = 0, best_v = 0;
    { uint32_t code = 0; int valid = 0;
      for (int o = 0; o < M; ++o) {
          uint8_t c = col(ref[o]);
          if (c > 3) { valid = 0; code = 0; continue; }
          code = ((code << 2) | c) & mask;
          if (++valid >= k) {
              int opos = o - k + 1;
              uint32_t h = (code * 2654435761u) >> (32 - 10);
              while (slot_code[h] != 0xFFFFFFFFu && slot_code[h] != code) h = (h + 1) & HMASK;
              if (slot_code[h] == code) {
                  int p = slot_pos[h];
                  int idx = (opos - p) + l_ms;
                  if (idx >= 0 && idx < span) {
                      int v = ++dv[idx];
                      if (v > best_v) { best_v = v; best_d = opos - p; }
                  }
              }
          }
      } }
    /* RESET INVARIANT: every path that claimed a slot must reach this line. The
     * table is static thread_local and initialized exactly once, so a return
     * placed above it leaves stale slot_code/slot_pos entries; the next call's
     * probe would then find a live-looking slot pointing at the PREVIOUS read's
     * position, casting votes on a garbage diagonal. That corruption does not
     * crash and is invisible to the byte-identity gate, because --rescue-kmer
     * output is not covered by it. Keep the single exit below: do not add an
     * early return past this point. */
    for (int h : touched) slot_code[h] = 0xFFFFFFFFu;

    res.scanned = true;
    res.best_v  = best_v;
    res.n_kmer  = (int)touched.size();

    if (best_v >= MEM_RESCUE_MIN_VOTES) {
        int W = opt->rescue_band > 0 ? opt->rescue_band : MEM_RESCUE_BAND_DEFAULT;
        int lo_off = best_d - W;         /* read-0 maps to window offset best_d */
        int hi_off = best_d + l_ms + W;
        if (lo_off < 0) lo_off = 0;
        if (hi_off > M) hi_off = M;
        /* An anchor diagonal near a window edge clamps to a span that can be far
         * shorter than the query (down to ~k + W when the read hangs off the left
         * edge). Such a window cannot host a full-length rescue alignment, so the
         * SW would return a truncated, low score that the caller's
         * `score >= min_seed_len` gate is liable to reject -- a rescue the full
         * window might have kept. Keep the full window in that case. Note this
         * declines NARROWING only: the scan succeeded, so res.scanned stays true
         * and the caller's skip gate still sees a valid best_v. */
        if (hi_off - lo_off >= l_ms) {
            res.ob = lo_off; res.oe = hi_off; res.narrowed = true;
        }
    }
    return res;
}

/* --rescue-skip: should this orientation's rescue SW be dropped entirely rather
 * than run against the full window? Only when the scan actually ran and the best
 * diagonal clears neither the absolute floor nor a fraction of the distinct
 * query k-mers the anchor index holds (n_kmer = touched.size(), the read's full
 * distinct count until indexing stops at HMAX = 768 codes). The fractional term
 * binds only for n_kmer below ~30 -- reads under ~36 bp at k=6 -- and is inert
 * at 150 bp.
 *
 * The floor is an absolute vote count while the vote budget is not: a read offers
 * l_ms - k + 1 k-mers, so ~145 at 150 bp and ~70 at 75 bp, and the same 10 votes
 * is twice as harsh a test on the shorter read. Measured skip rates track read
 * length accordingly -- 14% at 150 bp, 42% at 125 bp, 79% at 75 bp -- so the rate
 * this gate produces has to be measured per read length rather than assumed. */
static bool matesw_anchor_declines_rescue(const mem_opt_t *opt, const matesw_anchor_t &anc)
{
    if (!opt->rescue_skip || !anc.scanned) return false;
    return anc.best_v < MEM_RESCUE_SKIP_MIN_VOTES
        && (double)anc.best_v < (double)anc.n_kmer * MEM_RESCUE_SKIP_FRAC;
}

/* gar[gcnt+r] sentinel for --rescue-skip: "the anchor gate declined this
 * orientation; no rescue SW runs for it". A DISTINCT value from -1 is required:
 * -1 already means "re-run this orientation through the scalar ksw_align2" in
 * mem_matesw_batch_post, so reusing it would run the full UNBANDED scalar SW --
 * the exact opposite of skipping, and slower than doing nothing. */
static const int32_t MATESW_GAR_DECLINED = -2;

/* mem_matesw_batch_pre stores the narrowing offset per enqueued pair (keyed by
 * regid, stable under the batch's sort_classify / length-sort reorder) so
 * mem_matesw_batch_post reads it back instead of re-running the anchor scan. */
static thread_local std::vector<int> g_rescue_narrow_off;

/* --rescue-kmer: length-sort each SIMD-width partition of `sp[0,pcnt)` (8-bit
 * below pcnt8, 16-bit above) so the narrowed short windows group together --
 * getScores8/16 runs each SIMD-lane group out to the group's longest len1, so a
 * full fallback window mixed into a narrowed group wastes the saving. The
 * 8/16-bit split at pcnt8 is preserved by sorting each side independently.
 * Result-safe: aln[] and g_rescue_narrow_off are keyed by regid and seqBuf* by
 * idr/idq, so every key travels with the SeqPair it belongs to. */
static void matesw_sort_partitions_by_len(SeqPair *sp, int64_t pcnt8, int64_t pcnt)
{
    auto by_len1 = [](const SeqPair &x, const SeqPair &y) { return x.len1 < y.len1; };
    std::sort(sp, sp + pcnt8, by_len1);
    if (pcnt > pcnt8) std::sort(sp + pcnt8, sp + pcnt, by_len1);
}

/* T5/L3: RAII holder for mem_pair's per-thread scratch vectors. Retaining a
 * vector's grown capacity across calls (reset n=0, keep .a/.m) removes two
 * malloc/free pairs per read-pair, but a bare `static thread_local pair64_v`
 * is a POD with no destructor: its backing .a allocation would never be freed
 * and LeakSanitizer (the ASan CI row runs LSan) flags it as a per-thread leak
 * at exit. Wrapping the vector in a type whose destructor frees .a releases the
 * buffer when the worker thread ends, so the reuse is genuinely leak-free. */
struct pair64_scratch {
    pair64_v v = {0, 0, 0};
    ~pair64_scratch() { free(v.a); }
};
} // namespace

/* File-scope forward declaration of the dedup/patch entry point defined in
 * src/bwamem.cpp (kept default-free there). The `mat = NULL` default lives
 * here only, so [dcl.fct.default]/4 (no redefining a default in one TU) is
 * satisfied — the historical per-block-scope re-declarations each repeated the
 * default and only compiled under -fpermissive. */
extern int mem_sort_dedup_patch(const mem_opt_t *opt, const bntseq_t *bns,
                                const uint8_t *pac, uint8_t *query, int n,
                                mem_alnreg_t *a, const int8_t *mat = NULL);


#define MIN_RATIO     0.8
#define MIN_DIR_CNT   10
#define MIN_DIR_RATIO 0.05
#define OUTLIER_BOUND 2.0
#define MAPPING_BOUND 3.0
#define MAX_STDDEV    4.0

extern uint64_t tprof[LIM_R][LIM_C];

int mem_infer_dir(int64_t l_pac, int64_t b1, int64_t b2, int64_t *dist)
{
    int64_t p2;
    int r1 = (b1 >= l_pac), r2 = (b2 >= l_pac);
    p2 = r1 == r2? b2 : (l_pac<<1) - 1 - b2; // p2 is the coordinate of read 2 on the read 1 strand
    *dist = p2 > b1? p2 - b1 : b1 - p2;
    return (r1 == r2? 0 : 1) ^ (p2 > b1? 0 : 3);
}

/* Proper-pair bit (FLAG 0x2) for a pair emitted on the no-pairing path.
 * Returns 2 if the pair is properly paired, 0 otherwise, so callers can OR the
 * result straight into extra_flag. `which[i]` is the index of the region mate i
 * actually EMITS; both entries must already be >= 0 and the two EMITTED regions
 * must be on the same reference sequence (the caller checks h[].rid, which is
 * derived from a[i].a[which[i]] and so is not recoverable here). Note what that
 * does NOT cover: on the default path the coordinates read are a[i].a[0].rb,
 * whose rids nobody compared, so an ALT read can produce a distance spanning
 * two reference sequences. It then lands outside every pes window and the bit
 * stays clear. bwa (bwamem_pair.c:411) and bwa-mem2 do exactly the same.
 *
 * Which alignment the bit is derived from is the whole point of this function.
 * Both upstreams use a[0] unconditionally, even though the record they emit is
 * a[which] (bwa bwamem_pair.c:411; bwa-mem2 inherited it verbatim). #17 switched
 * bwa-mem3 to a[which] so the bit describes the record it rides on -- sound
 * reasoning, but 0x2 is aligner-defined, so it is a choice, and as a default it
 * made bwa-mem3 differ from BOTH upstreams (#362). The default now matches them;
 * #17's behavior is opt-in via --proper-pair-from-emitted.
 *
 * The two expressions differ only when which != 0, which requires the read to
 * have ALT hits, so this is inert on any run without a `.alt` sidecar.
 *
 * Extracted rather than inlined at both mem_sam_pe sites on purpose. The blocks
 * are verbatim copies of each other, the behavioral difference is unreachable
 * without an ALT sidecar, and a fix applied to one copy and not the other would
 * pass every fixture in the suite -- so the duplication is the actual hazard.
 * One definition also gives the option a testable seam
 * (test/unit/test_proper_pair_source.cpp). */
int mem_proper_pair_extra_flag(const mem_opt_t *opt, int64_t l_pac,
                               const mem_alnreg_v a[2], const int which[2],
                               const mem_pestat_t pes[4])
{
    int64_t dist;
    int d;
    const int w0 = opt->proper_pair_from_emitted ? which[0] : 0;
    const int w1 = opt->proper_pair_from_emitted ? which[1] : 0;
    d = mem_infer_dir(l_pac, a[0].a[w0].rb, a[1].a[w1].rb, &dist);
    if (!pes[d].failed && dist >= pes[d].low && dist <= pes[d].high) return 2;
    return 0;
}

static int cal_sub(const mem_opt_t *opt, mem_alnreg_v *r)
{
    int j;
    for (j = 1; j < r->n; ++j) { // choose unique alignment
        int b_max = r->a[j].qb > r->a[0].qb? r->a[j].qb : r->a[0].qb;
        int e_min = r->a[j].qe < r->a[0].qe? r->a[j].qe : r->a[0].qe;
        if (e_min > b_max) { // have overlap
            int min_l = r->a[j].qe - r->a[j].qb < r->a[0].qe - r->a[0].qb? r->a[j].qe - r->a[j].qb : r->a[0].qe - r->a[0].qb;
            if (e_min - b_max >= min_l * opt->mask_level) break; // significant overlap
        }
    }
    return j < r->n? r->a[j].score : opt->min_seed_len * opt->a;
}

/* Value at index `k` of the ascending order described by `cnt[0..max_ins]`.
 *
 * The counts are a permutation-free description of the sorted array: value v
 * occupies the index range [cumulative(v-1), cumulative(v)). So the element the
 * sorted array would hold at `k` is the first v whose cumulative count exceeds k.
 * Callers only ever pass k < total, which the percentile expressions guarantee.
 */
static inline int isize_at_rank(const int64_t *cnt, int max_ins, int64_t k)
{
    int64_t seen = 0;
    for (int v = 0; v <= max_ins; ++v) {
        seen += cnt[v];
        if (seen > k) return v;
    }
    return max_ins;   /* unreachable while k < total; keeps the return total */
}

void mem_pestat(const mem_opt_t *opt, int64_t l_pac, int n,
                const mem_alnreg_v *regs, mem_pestat_t pes[4])
{
    int i, d;
    int64_t max;
    /* Insert-size distribution as counts rather than a collected-then-sorted
     * array. The candidate loop below discards anything above max_ins, so
     * the values are bounded and a direct-address count array is EXACT -- there
     * is no bucketing here and nothing is approximated.
     *
     * Walking the counts in ascending order visits the same values, the same
     * number of times, in the same order as walking the sorted array would, and
     * that is what makes this bit-identical rather than merely equal to within
     * rounding. In particular the mean and variance loops below add each value
     * `count` times instead of multiplying by count: multiplying is algebraically
     * the same and numerically different, because floating-point addition is not
     * associative and the original summed the elements one at a time.
     *
     * This drops an O(m log m) sort and the kvec growth it fed on. mem_pestat is
     * serial (it needs the whole cohort before pairing can start), so its cost is
     * a fixed term that thread count cannot reduce -- measured at 30 ns/read,
     * invariant in both -t and cohort size, i.e. ~0.30 s of every run's wall. */
    /* Bounded before it sizes an allocation. The old code used max_ins purely as
     * a filter on collected values, so any value at all was harmless; here it is
     * a direct-address array bound, and 4*(max_ins+1)*8 bytes with an unbounded
     * max_ins is a caller-controlled allocation. mem_opt_init() sets 10000 (320
     * kB) and no CLI option changes it, but max_ins is read from the caller's
     * mem_opt_t and this is a library entry point -- an INT_MAX field would ask
     * for 64 GB and a negative one would convert to a huge size_t. Neither is a
     * value any caller means, so clamp rather than trust:
     *
     *   < 0        -> 0. No insert size satisfies `is <= 0`, so every
     *                 orientation ends up failed for want of pairs, which is
     *                 the same outcome as a cohort with no usable pairs.
     *   > the cap  -> the cap. 1 Mbase is orders of magnitude past any real
     *                 paired-end library and still only 32 MB of counters.
     *
     * The candidate filter below uses this bounded value, not opt->max_ins:
     * filtering on the unclamped field while indexing a clamped array is a
     * heap overflow, not a fallback. */
    const int kMaxInsCap = 1000000;
    int max_ins = opt->max_ins;
    if (max_ins < 0 || max_ins > kMaxInsCap) {
        int clamped = max_ins < 0 ? 0 : kMaxInsCap;
        fprintf(stderr, "[0000][PE] max_ins of %d is out of range; using %d\n",
                max_ins, clamped);
        max_ins = clamped;
    }
    int64_t *cnt_buf = (int64_t *) calloc((size_t)4 * ((size_t)max_ins + 1), sizeof(int64_t));
    int64_t *cnt[4], total[4];
    xassert(cnt_buf != NULL, "out of memory allocating insert-size histogram");
    for (d = 0; d < 4; ++d) { cnt[d] = cnt_buf + (size_t)d * ((size_t)max_ins + 1); total[d] = 0; }

    memset(pes, 0, 4 * sizeof(mem_pestat_t));
    for (i = 0; i < n>>1; ++i) {
        int dir;
        int64_t is;
        mem_alnreg_v *r[2];
        r[0] = (mem_alnreg_v*)&regs[i<<1|0];
        r[1] = (mem_alnreg_v*)&regs[i<<1|1];
        if (r[0]->n == 0 || r[1]->n == 0) continue;
        if (cal_sub(opt, r[0]) > MIN_RATIO * r[0]->a[0].score) continue;
        if (cal_sub(opt, r[1]) > MIN_RATIO * r[1]->a[0].score) continue;
        if (r[0]->a[0].rid != r[1]->a[0].rid) continue; // not on the same chr
        dir = mem_infer_dir(l_pac, r[0]->a[0].rb, r[1]->a[0].rb, &is);
        if (is && is <= max_ins) { ++cnt[dir][is]; ++total[dir]; }   /* bounded, not opt->max_ins */
    }
    if (bwa_verbose >= 3) fprintf(stderr, "[0000][PE] # candidate unique pairs for (FF, FR, RF, RR): (%lld, %lld, %lld, %lld)\n",
                                  (long long)total[0], (long long)total[1], (long long)total[2], (long long)total[3]);
    for (d = 0; d < 4; ++d) { // TODO: this block is nearly identical to the one in bwtsw2_pair.c. It would be better to merge these two.
        mem_pestat_t *r = &pes[d];
        int64_t *q = cnt[d], nq = total[d];
        int p25, p50, p75, x;
        if (nq < MIN_DIR_CNT) {
            fprintf(stderr, "[0000][PE] skip orientation %c%c as there are not enough pairs\n", "FR"[d>>1&1], "FR"[d&1]);
            r->failed = 1;
            continue;
        } else fprintf(stderr, "[0000][PE] analyzing insert size distribution for orientation %c%c...\n", "FR"[d>>1&1], "FR"[d&1]);
        /* Same indices the sorted array was subscripted with, so the same values. */
        p25 = isize_at_rank(q, max_ins, (int64_t)(.25 * nq + .499));
        p50 = isize_at_rank(q, max_ins, (int64_t)(.50 * nq + .499));
        p75 = isize_at_rank(q, max_ins, (int64_t)(.75 * nq + .499));
        r->low  = (int)(p25 - OUTLIER_BOUND * (p75 - p25) + .499);
        if (r->low < 1) r->low = 1;
        r->high = (int)(p75 + OUTLIER_BOUND * (p75 - p25) + .499);
        fprintf(stderr, "[0000][PE] (25, 50, 75) percentile: (%d, %d, %d)\n", p25, p50, p75);
        fprintf(stderr, "[0000][PE] low and high boundaries for computing mean and std.dev: (%d, %d)\n", r->low, r->high);
        /* The sorted array is ascending, so its [low, high] filter selects one
         * contiguous run -- exactly this range of values, each repeated its own
         * count of times. Hence the inner loops, and hence identical arithmetic. */
        int lo = r->low  > 0       ? r->low  : 0;
        int hi = r->high < max_ins ? r->high : max_ins;
        for (x = 0, r->avg = 0; lo <= hi; ++lo)
            for (int64_t k = 0; k < q[lo]; ++k)
                r->avg += lo, ++x;
        assert(x != 0);
        r->avg /= x;
        lo = r->low > 0 ? r->low : 0;
        for (r->std = 0; lo <= hi; ++lo)
            for (int64_t k = 0; k < q[lo]; ++k)
                r->std += (lo - r->avg) * (lo - r->avg);
        r->std = sqrt(r->std / x);
        fprintf(stderr, "[0000][PE] mean and std.dev: (%.2f, %.2f)\n", r->avg, r->std);
        r->low  = (int)(p25 - MAPPING_BOUND * (p75 - p25) + .499);
        r->high = (int)(p75 + MAPPING_BOUND * (p75 - p25) + .499);
        if (r->low  > r->avg - MAX_STDDEV * r->std) r->low  = (int)(r->avg - MAX_STDDEV * r->std + .499);
        if (r->high < r->avg + MAX_STDDEV * r->std) r->high = (int)(r->avg + MAX_STDDEV * r->std + .499);
        if (r->low < 1) r->low = 1;
        fprintf(stderr, "[0000][PE] low and high boundaries for proper pairs: (%d, %d)\n", r->low, r->high);
    }
    free(cnt_buf);
    for (d = 0, max = 0; d < 4; ++d)
        max = max > total[d]? max : total[d];
    for (d = 0; d < 4; ++d)
        if (pes[d].failed == 0 && total[d] < max * MIN_DIR_RATIO) {
            pes[d].failed = 1;
            fprintf(stderr, "[0000][PE] skip orientation %c%c\n", "FR"[d>>1&1], "FR"[d&1]);
        }
}

/* ------------------------------------------------------------------------- *
 * u8 mate-rescue admission (tier selection)
 *
 * Mate rescue routes each pair to the 8-bit (u8) or 16-bit SW kernel. The u8
 * kernel biases scores by `shift` and stores them in a uint8_t, so it is only
 * byte-identical to the 16-bit kernel while the max attainable biased score
 * cannot saturate/early-break the u8 DP. The binding limit is the u8-only
 * early break `gmax + q->shift >= 255` in ksw.cpp (ksw_u8), i.e. the biased
 * max score must be <= 254; the batched kswv u8 kernel saturates one unit
 * later, so 254 is the tighter, shared bound.
 *
 * `shift` is the kernel's score bias: ksw derives it as `256 - min(matrix)`
 * taken as a uint8_t, which equals `-(min matrix entry)` for every matrix
 * bwa-mem3 builds (>= 1 in practice; the floor guards a degenerate matrix).
 *
 * There is a second, geometric provability condition. The u8 and 16-bit paths
 * pad the query to different widths (16*ceil(l/16) vs 8*ceil(l/8)); a pair may
 * flip tier only when those pad widths are equal, otherwise the two kernels run
 * different padded column counts and their tie-break/XS scan can diverge. That
 * equality holds exactly when l % 16 == 0 || l % 16 >= 9. Pairs already routed
 * to u8 under the historical `l_ms*a < 250` bound keep their routing regardless
 * (they were u8 before and stay u8), so the geometry clause only gates the new
 * admissions the tightened bound opens up.
 * ------------------------------------------------------------------------- */

/* Kernel score bias for a 5x5 scoring matrix, matching ksw_qinit's
 * `256 - (uint8_t)min` reduction. >= 1 for any real matrix (min <= 0). */
static inline int matesw_u8_shift(const int8_t *mat) {
    int mn = 0;
    for (int i = 0; i < 25; ++i) if (mat[i] < mn) mn = mat[i];
    int s = -mn;
    return s < 1 ? 1 : s;
}

/* Worst-case (max) bias across every matrix the SW at a site could use. Taking
 * the max only ever routes MORE pairs to the safe 16-bit tier, so it can never
 * over-admit. Under --meth the actual matrix is one of mat_ot/mat_ob (chosen by
 * the rescued mate's read#); folding both in keeps the batched pre/post split
 * consistent no matter which hypothesis a pair ends up scored under. */
static inline int matesw_u8_shift_for(const mem_opt_t *opt, const int8_t *base_mat) {
    int s = matesw_u8_shift(base_mat);
    if (opt->meth_mode) {
        int so = matesw_u8_shift(opt->mat_ot);
        int sb = matesw_u8_shift(opt->mat_ob);
        if (so > s) s = so;
        if (sb > s) s = sb;
    }
    return s;
}

/* True iff a mate of query length l_ms may be scored on the u8 tier
 * byte-identically. `a` is the match reward (opt->a); `shift` the kernel bias
 * from matesw_u8_shift_for(). */
static inline int matesw_use_u8(int l_ms, int a, int shift) {
    int prod = l_ms * a;                              /* max attainable SW score */
    return (prod + shift <= 254)                      /* no u8 saturation / early break */
        && (prod < 250                                /* historical admissions keep routing */
            || l_ms % 16 == 0 || (l_ms % 16) >= 9);   /* new admissions: pad geometry equal */
}

int mem_matesw(const mem_opt_t *opt, const bntseq_t *bns,
               const uint8_t *pac, const mem_pestat_t pes[4],
               const mem_alnreg_t *a, int l_ms, const uint8_t *ms,
               mem_alnreg_v *ma, const char *ms_orig = NULL,
               const int8_t *mat = NULL, int mate_meth_ot = -1)
{
    /* D3 (--meth): the mate-rescue dedup at the bottom of this function still
     * passes mat=NULL (resolving to opt->mat) — but those calls pass
     * bns/pac/query = 0 (dedup-only, no patch SW), so the matrix is unused
     * there regardless; leaving them on opt->mat is correct.
     *
     * The per-hypothesis scorer below no longer consults the `mat` parameter
     * under --meth: it derives the matrix from the rescued mate's own read#
     * chemistry (mate_meth_ot ^ is_rev — see the use_mat selection below). The
     * caller-supplied `mat` (rmat = mem_opt_meth_mat(opt, !a->meth_hypothesis)
     * in mem_pair_resolve) is therefore dead code under --meth: it is discarded
     * here and recomputed from the mate read#. sw_mat/`mat` survives only as the
     * non-meth default. */
    /* Outside --meth, mat is NULL and ms_orig is NULL: behavior is byte-for-byte
     * identical to the historical symmetric/projected mate rescue. */
    const int8_t *sw_mat = mat ? mat : opt->mat;
    /* D3 (--meth, PR-6): when ms_orig is set, score the ORIGINAL (unconverted)
     * mate bases against the original ref window instead of the projected mate.
     * meth_orig_seq is ASCII in the SAME orientation as `ms`/`seq` (bwa.h
     * contract), so 2-bit-encode it here and let the existing per-orientation
     * RC below reverse-complement it EXACTLY as it does the projected mate.
     * Allocated AFTER the skip-all early return below so the consistent-pair
     * fast path (the common case) does not leak it. */
    uint8_t *ms2 = NULL;
    #if MATE_SORT
    extern int mem_dedup_patch(const mem_opt_t *opt, const bntseq_t *bns,
                               const uint8_t *pac, uint8_t *query, int n, mem_alnreg_t *a);
    extern void sort_alnreg_re(int n, mem_alnreg_t* a);
    extern void sort_alnreg_score(int n, mem_alnreg_t* a);
    #endif
    
    //int tid = omp_get_thread_num();
    int64_t l_pac = bns->l_pac;
    int i, r, skip[4], n = 0, rid = -1;
    for (r = 0; r < 4; ++r)
        skip[r] = pes[r].failed? 1 : 0;


    for (i = 0; i < ma->n; ++i) { // check which orinentation has been found
        int64_t dist;
        r = mem_infer_dir(l_pac, a->rb, ma->a[i].rb, &dist);
        
        if (dist >= pes[r].low && dist <= pes[r].high) {
            skip[r] = 1;
        }
    }

    if (skip[0] + skip[1] + skip[2] + skip[3] == 4) return 0; // consistent pair exist; no need to perform SW

    if (ms_orig != NULL) {
        ms2 = (uint8_t*) malloc(l_ms);
        xassert(ms2 != NULL, "out of memory: ms2");
        for (int k = 0; k < l_ms; ++k) {
            unsigned char c = (unsigned char) ms_orig[k];
            ms2[k] = nst_nt4_decode(c, 4);
        }
        ms = ms2; // alias the projected-mate pointer to the original bases
    }

    for (r = 0; r < 4; ++r) {
        int is_rev, is_larger;
        uint8_t *seq, *rev = 0, *ref = 0;
        int64_t rb, re;
        if (skip[r]) continue;
        is_rev = (r>>1 != (r&1)); // whether to reverse complement the mate
        is_larger = !(r>>1); // whether the mate has larger coordinate
        if (is_rev) {
            rev = (uint8_t*) malloc(l_ms); // this is the reverse complement of $ms
            xassert(rev != NULL, "out of memory: rev");
            for (i = 0; i < l_ms; ++i) rev[l_ms - 1 - i] = ms[i] < 4? 3 - ms[i] : 4;
            seq = rev;
        } else seq = (uint8_t*)ms;
        if (!is_rev) {
            rb = is_larger? a->rb + pes[r].low : a->rb - pes[r].high;
            re = (is_larger? a->rb + pes[r].high: a->rb - pes[r].low) + l_ms; // if on the same strand, end position should be larger to make room for the seq length
        } else {
            rb = (is_larger? a->rb + pes[r].low : a->rb - pes[r].high) - l_ms; // similarly on opposite strands
            re = is_larger? a->rb + pes[r].high: a->rb - pes[r].low;
        }
        if (rb < 0) rb = 0;
        if (re > l_pac<<1) re = l_pac<<1;
        static thread_local u8vec_scratch_t t_ref;
        if (rb < re) {
            /* Prefetch the rescue window's first pac[] cache line ahead of the fetch (cf. AP6).
             * Best-effort: the rescue window (a->rb ± pes[r]) is unbounded, so when it straddles a
             * contig boundary bns_fetch_seq_into clamps the read start up to far_beg and this hint
             * — computed from the pre-clamp rb — can target a line the fetch never reads. Harmless
             * (it just misses), never wrong output. bns_depos maps the strand-dependent window
             * start; the outer is_rev is load-bearing, so discard bns_depos's strand into a
             * scratch and do NOT reuse that name. Pure hint -> byte-identical. */
            {
                int is_rev_pf;  // discarded; do NOT reuse the outer is_rev
                __builtin_prefetch(&pac[bns_depos(bns, rb, &is_rev_pf) >> 2], 0, 3);
            }
            size_t want = (size_t)((re - rb) + 64);
            if (t_ref.v.m < want) kv_resize(uint8_t, t_ref.v, want);
            int64_t rlen;
            bns_fetch_seq_into(bns, pac, &rb, (rb+re)>>1, &re, &rid, t_ref.v.a, &rlen);
            ref = t_ref.v.a;
        }
        if (a->rid == rid && re - rb >= opt->min_seed_len) { // no funny things happening
            kswr_t aln;
            mem_alnreg_t b;
            int u8_shift = matesw_u8_shift_for(opt, sw_mat);
            int tmp, xtra = KSW_XSUBO | KSW_XSTART | (matesw_use_u8(l_ms, opt->a, u8_shift)? KSW_XBYTE : 0) | (opt->min_seed_len * opt->a);

            /* D3 (--meth): score the rescued mate under ITS OWN read-number
             * chemistry (mate_meth_ot: R1=1/OT, R2=0/OB), flipped by the rescue
             * strand because this path RC's the READ (seq=rev) against the forward
             * reference window. One SW pass with the correct matrix — no try-both:
             * the mate's read# is the chemistry that frees its conversions at its
             * true locus, so the crippling that a wrong seed-derived guess caused
             * cannot happen. matesw_hyp is the genome-strand hypothesis recorded on
             * the rescued alnreg for the XG/XM output layer. */
            const int8_t *use_mat = sw_mat;
            int matesw_hyp = -1;
            if (opt->meth_mode && mate_meth_ot >= 0) {
                matesw_hyp = (mate_meth_ot ^ is_rev) & 1;
                use_mat = mem_opt_meth_mat(opt, matesw_hyp);
            }
            assert(ref !=0 && re - rb >= 0);
            aln = ksw_align2(l_ms, seq, re - rb, ref, 5,
                             use_mat, opt->o_del, opt->e_del,
                             opt->o_ins, opt->e_ins, xtra, 0);

            memset(&b, 0, sizeof(mem_alnreg_t));
            if (aln.score >= opt->min_seed_len && aln.qb >= 0 && aln.qe < l_ms) { // something goes wrong if aln.qb < 0, or if aln.qe runs past the read
                b.rid = a->rid;
                b.is_alt = a->is_alt;
                /* D3 (--meth): record the rescued mate's genome-strand hypothesis
                 * (matesw_hyp = mate read# XOR rescue strand, computed with the SW
                 * above) so the XG/XM output layer sources the right strand. -1
                 * anchor (non-meth) stays -1. Coordinates are already ORIGINAL
                 * (l_pac is the original l_pac via the original bns), so the 6a
                 * coordinate fix holds. */
                b.meth_hypothesis = matesw_hyp;
                b.qb = is_rev? l_ms - (aln.qe + 1) : aln.qb;
                b.qe = is_rev? l_ms - aln.qb : aln.qe + 1;
                b.rb = is_rev? (l_pac<<1) - (rb + aln.te + 1) : rb + aln.tb;
                b.re = is_rev? (l_pac<<1) - (rb + aln.tb) : rb + aln.te + 1;
                b.score = aln.score;
                b.csub = aln.score2;
                b.secondary = -1;
                b.seedcov = (b.re - b.rb < b.qe - b.qb? b.re - b.rb : b.qe - b.qb) >> 1;
                b.chain_n_hits = 1; // mate-rescue has no SMEM evidence; treat as unique anchor

                kv_push(mem_alnreg_t, *ma, b); // make room for a new element

                #if !MATE_SORT
                // move b s.t. ma is sorted
                for (i = 0; i < ma->n - 1; ++i) // find the insertion point
                    if (ma->a[i].score < b.score) break;
                tmp = i;
                for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
                ma->a[i] = b;
                
                #else
                
                int resort = 0;
                for (i = 0; i < ma->n - 1; ++i) { // find the insertion point
                    if (ma->a[i].re == b.re) {
                        resort = 1;
                        break;
                    }
                    if (ma->a[i].re > b.re) {
                        break;
                    }
                }
                if (resort) {
                    // Don't know where to put this alignment. So let the scores decide
                    sort_alnreg_score(ma->n - 1, ma->a);
                    for (i = 0; i < ma->n - 1; ++i) { // find the insertion point
                        if (ma->a[i].score < b.score) {
                            break;
                        }
                    }
                    tmp = i;
                    for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
                    ma->a[i] = b;
                    // Now we can sort based on end position
                    sort_alnreg_re(ma->n, ma->a);
                }
                else {
                    tmp = i;
                    for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
                    ma->a[i] = b;
                }
                #endif
                tprof[PE26][0] ++;
            }
            ++n;
        }
        #if !MATE_SORT
        if (n) ma->n = mem_sort_dedup_patch(opt, 0, 0, 0, ma->n, ma->a);
        #else
        if (n) ma->n = mem_dedup_patch(opt, 0, 0, 0, ma->n, ma->a); // sam_improvements
        #endif
        if (rev) free(rev);
        /* ref aliases t_ref thread-local scratch; do not free. */
    }
    if (ms2) free(ms2); // D3 (--meth): original-mate 2-bit scratch
    return n;
}

int mem_pair(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_pestat_t pes[4], bseq1_t s[2], mem_alnreg_v a[2], int id, int *sub, int *n_sub, int z[2], int n_pri[2])
{
    /* T5/L3: v and u are pure per-call working buffers -- rebuilt from n=0 every
     * call, only [0,n) is ever read, and no state survives the return. Making
     * them thread-local persistent buffers (reset n=0, keep the grown capacity)
     * removes two malloc/free pairs per read-pair without changing any bytes:
     * a reset-to-empty vector with retained capacity behaves identically to a
     * fresh kv_init'd one. The pair64_scratch wrapper frees .a at thread exit
     * (LSan-clean; see its definition above). */
    static thread_local pair64_scratch v_buf, u_buf;
    pair64_v &v = v_buf.v, &u = u_buf.v;
    int r, i, k, y[4], ret; // y[] keeps the last hit
    int64_t l_pac = bns->l_pac;
    /* Bound the retained capacity so one pathological high-cardinality pair does
     * not pin a large allocation for the worker thread's lifetime. Below the cap
     * the grown buffer is reused as-is; above it we release and let the next call
     * re-grow from empty. Capacity-only -- [0,n) content and output are
     * byte-identical either way. The cap (64Ki entries = 1 MiB per buffer) sits
     * orders of magnitude above any realistic per-pair candidate count, so the
     * common path never trims. */
    const size_t PAIR_SCRATCH_MAX_M = (size_t)1 << 16;
    if (v.m > PAIR_SCRATCH_MAX_M) { free(v.a); v.a = 0; v.m = 0; }
    if (u.m > PAIR_SCRATCH_MAX_M) { free(u.a); u.a = 0; u.m = 0; }
    v.n = 0; u.n = 0;
    for (r = 0; r < 2; ++r) { // loop through read number
        for (i = 0; i < n_pri[r]; ++i) {
            pair64_t key;
            mem_alnreg_t *e = &a[r].a[i];
            key.x = e->rb < l_pac? e->rb : (l_pac<<1) - 1 - e->rb; // forward position
            key.x = (uint64_t)e->rid<<32 | (key.x - bns->anns[e->rid].offset);
            key.y = (uint64_t)e->score << 32 | i << 2 | (e->rb >= l_pac)<<1 | r;
            kv_push(pair64_t, v, key);
        }
    }

    ks_introsort_128(v.n, v.a);
    y[0] = y[1] = y[2] = y[3] = -1;
    for (i = 0; i < v.n; ++i) {
        for (r = 0; r < 2; ++r) { // loop through direction
            int dir = r<<1 | (v.a[i].y>>1&1), which;
            if (pes[dir].failed) continue; // invalid orientation
            which = r<<1 | ((v.a[i].y&1)^1);
            if (y[which] < 0) continue; // no previous hits
            for (k = y[which]; k >= 0; --k) { // TODO: this is a O(n^2) solution in the worst case; remember to check if this loop takes a lot of time (I doubt)
                int64_t dist;
                int q;
                double ns;
                pair64_t *p;
                if ((v.a[k].y&3) != which) continue;
                dist = (int64_t)v.a[i].x - v.a[k].x;
                //printf("%d: %lld\n", k, dist);
                if (dist > pes[dir].high) break;
                if (dist < pes[dir].low)  continue;
                // Guard std==0 (from -I mean,0 or a fixed-insert cohort where
                // mem_pestat yields std=0): every surviving candidate already has
                // dist==avg, so ns=0 is the right limit. Without this the divide
                // is 0/0=NaN, the (int) cast of NaN is UB, and pairing silently
                // collapses (q=0 for every pair).
                ns = pes[dir].std > 0. ? (dist - pes[dir].avg) / pes[dir].std : 0.;
                q = (int)((v.a[i].y>>32) + (v.a[k].y>>32) + .721 * log(2. * erfc(fabs(ns) * M_SQRT1_2)) * opt->a + .499); // .721 = 1/log(4)
                if (q < 0) q = 0;
                p = kv_pushp(pair64_t, u);
                p->y = (uint64_t)k<<32 | i;
                p->x = (uint64_t)q<<32 | (hash_64(p->y ^ id<<8) & 0xffffffffU);
            }
        }
        y[v.a[i].y&3] = i;
    }
    if (u.n) { // found at least one proper pair
        int tmp = opt->a + opt->b;
        tmp = tmp > opt->o_del + opt->e_del? tmp : opt->o_del + opt->e_del;
        tmp = tmp > opt->o_ins + opt->e_ins? tmp : opt->o_ins + opt->e_ins;
        ks_introsort_128(u.n, u.a);
        i = u.a[u.n-1].y >> 32; k = u.a[u.n-1].y << 32 >> 32;
        z[v.a[i].y&1] = v.a[i].y<<32>>34; // index of the best pair
        z[v.a[k].y&1] = v.a[k].y<<32>>34;
        ret = u.a[u.n-1].x >> 32;
        *sub = u.n > 1? u.a[u.n-2].x>>32 : 0;
        for (i = (long)u.n - 2, *n_sub = 0; i >= 0; --i)
            if (*sub - (int)(u.a[i].x>>32) <= tmp) ++*n_sub;

    } else ret = 0, *sub = 0, *n_sub = 0;
    /* T5/L3: no per-call free -- v/u keep their capacity for reuse across calls;
     * the pair64_scratch destructor releases .a when the worker thread exits. */
    return ret;
}

void mem_aln2sam(const mem_opt_t *opt, const bntseq_t *bns, kstring_t *str, bseq1_t *s, int n, const mem_aln_t *list, int which, const mem_aln_t *m);

#define raw_mapq(diff, a) ((int)(6.02 * (diff) / (a) + .499))


// Core pairing decision for a single read pair.
//
// Performs, in order:
//   1. Mate-rescue SW (unless MEM_F_NO_RESCUE) — may add entries to a[].
//   2. mem_mark_primary_se on both reads — mutates a[] and fills n_pri[].
//   3. mem_reorder_primary5 if MEM_F_PRIMARY5 — mutates a[].
//   4. Pairing attempt via mem_pair, is-multi sanity check, q_pe / q_se
//      computation, and the secondary<->primary secondary_all patch.
//
// On exit:
//   *paired_out = 1 if the paired branch was taken (extra_flag includes 0x2
//                   iff o > score_un, i.e. paired alignment is preferred).
//               = 0 if pairing fell through (MEM_F_NOPAIRING set, neither
//                   read has a primary, mem_pair returned 0, or is_multi).
//                   Callers must treat z[]/q_se[] as undefined in this case:
//                   mem_pair() may already have populated z[] before an
//                   early return (e.g. is_multi), so paired_out is the only
//                   validity check. Callers take the "no_pairing" emission
//                   path here.
//   extra_flag_out carries the common extra_flag bits (always has 0x1).
//                   On the paired path it is fully assembled and includes
//                   0x2 iff the paired alignment was preferred. On the
//                   no-pairing path it is only partial — the caller (or
//                   mem_sam_pe's no_pairing emission block) is responsible
//                   for OR-ing in 0x2 itself.
//   n_pri[] is always populated.
//   z[], q_se[] are valid only when *paired_out == 1.
//
// Returns the number of mate-rescue hits produced (same meaning as the
// historical `n` return from mem_sam_pe — i.e. a caller-visible accounting
// of mate-SW work done).
/* --extend-csub (PE): estimate the pair score of a competitor pruned before
 * extension, so the pairing MAPQ raw_mapq(o - subo) reflects it. Each mate contributes
 * its pruned-competitor SE score, estimated from the dropped chain weight scaled by that
 * mate's OBSERVED per-base score rate (score/span). Note this is deliberately gentler
 * than the all-match upper bound (weight * a) that mem_seed_capped_sub uses on the SE
 * side: a pair score is the sum of two such estimates, so the generous bound would
 * overstate subo roughly twice over. The sum is clamped strictly below o, so q_pe drops
 * proportionally to a near-tie rather than being forced to 0.
 * Returns a floor for subo, or 0 to leave subo unchanged. */
int mem_capped_pair_subo(const mem_alnreg_v a[2], const int z[2], int o)
{
    long est = 0;
    for (int i = 0; i < 2; i++) {
        if (a[i].capped_w <= 0 || z[i] < 0 || (size_t)z[i] >= a[i].n) continue;
        const mem_alnreg_t *p = &a[i].a[z[i]];
        int span = p->qe - p->qb;
        if (span > 0 && p->score > 0)
            est += (long)a[i].capped_w * p->score / span;
    }
    if (est <= 0) return 0;
    if (est > (long)o - 1) est = (long)o - 1;   /* clamp: never force q_pe to 0 */
    return (int)est;
}

int mem_pair_resolve(const mem_opt_t *opt, const bntseq_t *bns,
                     const uint8_t *pac, const mem_pestat_t pes[4],
                     uint64_t id, bseq1_t s[2], mem_alnreg_v a[2],
                     int n_pri[2], int z[2], int q_se[2],
                     int *extra_flag_out, int *paired_out)
{
    extern int mem_mark_primary_se(const mem_opt_t *opt, int n, mem_alnreg_t *a, int64_t id);
    extern void mem_seed_capped_sub(mem_alnreg_v *a, int match_a);
    extern int mem_approx_mapq_se(const mem_opt_t *opt, const mem_alnreg_t *a);

    #if MATE_SORT
    extern void sort_alnreg_re(int n, mem_alnreg_t* a);
    extern void sort_alnreg_score(int n, mem_alnreg_t* a);
    /* D3 (--meth, PR-6): this dedup call passes bns/pac/query = 0 (dedup-only,
     * no patch SW), so its matrix is never consulted — opt->mat default is
     * correct; the per-hypothesis asymmetric scoring lives in mem_matesw. */
    #endif

    int n = 0, i, j, o, subo, n_sub, extra_flag = 1;

    *paired_out = 0;
    *extra_flag_out = extra_flag;

    if (!(opt->flag & MEM_F_NO_RESCUE)) { // then perform SW for the best alignment

        mem_alnreg_v b[2];
        kv_init(b[0]); kv_init(b[1]);
        for (i = 0; i < 2; ++i)
            for (j = 0; j < a[i].n; ++j)
                if (a[i].a[j].score >= a[i].a[0].score  - opt->pen_unpaired)
                    kv_push(mem_alnreg_t, b[i], a[i].a[j]);

        #if MATE_SORT
        for (i = 0; i < 2; ++i) {
            sort_alnreg_re(a[!i].n, a[!i].a);
            int val = 0, swcount = 0;
            for (j = 0; j < b[i].n && j < opt->max_matesw; ++j) {
                /* D3 (--meth): rescue read !i against the original ref using its
                 * ORIGINAL bases (ms_orig). `rmat` is vestigial — mem_matesw now
                 * derives the matrix from the mate's own read# chemistry
                 * (mate_meth_ot ^ is_rev) and discards `mat` under --meth, so rmat
                 * is dead code (recomputed inside mem_matesw). Outside --meth both
                 * stay NULL and mem_matesw is identical to before. */
                const char  *ms_orig = opt->meth_mode ? s[!i].meth_orig_seq : NULL;
                const int8_t *rmat    = opt->meth_mode
                    ? mem_opt_meth_mat(opt, !b[i].a[j].meth_hypothesis) : NULL;
                int val = mem_matesw(opt, bns, pac, pes, &b[i].a[j], s[!i].l_seq, (uint8_t*)s[!i].seq, &a[!i], ms_orig, rmat, opt->meth_mode ? i : -1);
                n += val;
                swcount += val;
            }
            if (swcount > 0) {
                mem_alnreg_v* ma = &a[!i];
                ma->n = mem_sort_dedup_patch(opt, 0, 0, 0, ma->n, ma->a);
            }
            else {
                sort_alnreg_score(a[!i].n, a[!i].a);
            }
        }

        #else

        for (i = 0; i < 2; ++i)
            for (j = 0; j < b[i].n && j < opt->max_matesw; ++j) {
                /* D3 (--meth): see MATE_SORT branch above — original mate bases;
                 * rmat is vestigial (mem_matesw derives its matrix from the mate
                 * read# and discards `mat` under --meth). */
                const char  *ms_orig = opt->meth_mode ? s[!i].meth_orig_seq : NULL;
                const int8_t *rmat    = opt->meth_mode
                    ? mem_opt_meth_mat(opt, !b[i].a[j].meth_hypothesis) : NULL;
                int val = mem_matesw(opt, bns, pac, pes, &b[i].a[j], s[!i].l_seq, (uint8_t*)s[!i].seq, &a[!i], ms_orig, rmat, opt->meth_mode ? i : -1);
                n += val;
            }
        #endif
        free(b[0].a); free(b[1].a);
    }

    n_pri[0] = mem_mark_primary_se(opt, a[0].n, a[0].a, id<<1|0);
    n_pri[1] = mem_mark_primary_se(opt, a[1].n, a[1].a, id<<1|1);

    #if V17
    if (opt->flag & MEM_F_PRIMARY5) {
        mem_reorder_primary5(opt->T, &a[0]);
        mem_reorder_primary5(opt->T, &a[1]);
    }
    #endif

    /* --extend-csub: seed each end's primary competitor score from the chains the
     * cap/gate dropped, before pairing/MAPQ below reads csub. No-op when off. */
    mem_seed_capped_sub(&a[0], opt->a);
    mem_seed_capped_sub(&a[1], opt->a);

    if (opt->flag & MEM_F_NOPAIRING) {
        *extra_flag_out = extra_flag;
        return n;
    }

    // pairing single-end hits
    if (!(n_pri[0] && n_pri[1] &&
          (o = mem_pair(opt, bns, pac, pes, s, a, id, &subo, &n_sub, z, n_pri)) > 0)) {
        *extra_flag_out = extra_flag;
        return n;
    }

    int is_multi[2], q_pe, score_un;
    for (i = 0; i < 2; ++i) {
        for (j = 1; j < n_pri[i]; ++j)
            if (a[i].a[j].secondary < 0 && a[i].a[j].score >= opt->T) break;
        is_multi[i] = j < n_pri[i] ? 1 : 0;
    }
    if (is_multi[0] || is_multi[1]) { // TODO: in rare cases, the true hit may be long but with low score
        *extra_flag_out = extra_flag;
        return n;
    }

    // compute mapQ for the best SE hit
    score_un = a[0].a[0].score + a[1].a[0].score - opt->pen_unpaired;
    subo = subo > score_un ? subo : score_un;
    if (opt->extend_csub) { int fl = mem_capped_pair_subo(a, z, o); if (fl > subo) subo = fl; }
    q_pe = raw_mapq(o - subo, opt->a);
    if (n_sub > 0) q_pe -= (int)(4.343 * log(n_sub + 1) + .499);
    if (q_pe < 0) q_pe = 0;
    if (q_pe > 60) q_pe = 60;
    q_pe = (int)(q_pe * (1. - .5 * (a[0].a[0].frac_rep + a[1].a[0].frac_rep)) + .499);

    // the following assumes no split hits
    if (o > score_un) { // paired alignment is preferred
        mem_alnreg_t *c[2];
        c[0] = &a[0].a[z[0]]; c[1] = &a[1].a[z[1]];
        for (i = 0; i < 2; ++i) {
            if (c[i]->secondary >= 0)
                c[i]->sub = a[i].a[c[i]->secondary].score, c[i]->secondary = -2;
            q_se[i] = mem_approx_mapq_se(opt, c[i]);
        }
        q_se[0] = q_se[0] > q_pe ? q_se[0] : q_pe < q_se[0] + 40 ? q_pe : q_se[0] + 40;
        q_se[1] = q_se[1] > q_pe ? q_se[1] : q_pe < q_se[1] + 40 ? q_pe : q_se[1] + 40;
        extra_flag |= 2;

        // cap at the tandem repeat score
        q_se[0] = q_se[0] < raw_mapq(c[0]->score - c[0]->csub, opt->a) ? q_se[0] : raw_mapq(c[0]->score - c[0]->csub, opt->a);
        q_se[1] = q_se[1] < raw_mapq(c[1]->score - c[1]->csub, opt->a) ? q_se[1] : raw_mapq(c[1]->score - c[1]->csub, opt->a);
    } else { // the unpaired alignment is preferred
        z[0] = z[1] = 0;
        q_se[0] = mem_approx_mapq_se(opt, &a[0].a[0]);
        q_se[1] = mem_approx_mapq_se(opt, &a[1].a[0]);
    }

    for (i = 0; i < 2; ++i) {
        int k = a[i].a[z[i]].secondary_all;
        if (k >= 0 && k < n_pri[i]) { // switch secondary and primary if both of them are non-ALT
            assert(a[i].a[k].secondary_all < 0);
            for (j = 0; j < a[i].n; ++j)
                if (a[i].a[j].secondary_all == k || j == k)
                    a[i].a[j].secondary_all = z[i];
            a[i].a[z[i]].secondary_all = -1;
        }
    }

    *paired_out = 1;
    *extra_flag_out = extra_flag;
    return n;
}


int mem_sam_pe(const mem_opt_t *opt, const bntseq_t *bns,
               const uint8_t *pac, const mem_pestat_t pes[4],
               uint64_t id, bseq1_t s[2], mem_alnreg_v a[2])
{
    extern void mem_reg2sam(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, bseq1_t *s, mem_alnreg_v *a, int extra_flag, const mem_aln_t *m);
    extern char **mem_gen_alt(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_alnreg_v *a, int l_query, const char *query, int **out_hn, const char *meth_orig_query);

    int i, j, z[2], extra_flag, n_pri[2], q_se[2], n_aa[2], paired;
    kstring_t str;
    mem_aln_t h[2], g[2], aa[2][2];

    str.l = str.m = 0; str.s = 0;
    memset(h, 0, sizeof(mem_aln_t) * 2);
    memset(g, 0, sizeof(mem_aln_t) * 2);
    n_aa[0] = n_aa[1] = 0;

    int n = mem_pair_resolve(opt, bns, pac, pes, id, s, a, n_pri, z, q_se,
                             &extra_flag, &paired);

    if (paired) {
        char **XA[2];
        int *HN[2] = { 0, 0 };
        if (!(opt->flag & MEM_F_ALL)) {
            for (i = 0; i < 2; ++i)
                XA[i] = mem_gen_alt(opt, bns, pac, &a[i], s[i].l_seq, s[i].seq, &HN[i], s[i].meth_orig_seq);
        } else XA[0] = XA[1] = 0;
        // write SAM
        for (i = 0; i < 2; ++i) {
            h[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, &a[i].a[z[i]], s[i].meth_orig_seq);
            h[i].mapq = q_se[i];

            h[i].flag |= 0x40<<i | extra_flag;
            h[i].XA = XA[i]? XA[i][z[i]] : 0;
            h[i].HN = HN[i]? HN[i][z[i]] : -1;
            aa[i][n_aa[i]++] = h[i];
            if (n_pri[i] < a[i].n) { // the read has ALT hits
                mem_alnreg_t *p = &a[i].a[n_pri[i]];
                if (p->score < opt->T || p->secondary >= 0 || !p->is_alt) continue;
                g[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, p, s[i].meth_orig_seq);
                g[i].flag |= 0x800 | 0x40<<i | extra_flag;
                g[i].XA = XA[i]? XA[i][n_pri[i]] : 0;
                g[i].HN = HN[i]? HN[i][n_pri[i]] : -1;
                if (opt->supp_rep_hard_cap > 0 && p->chain_n_hits >= opt->supp_rep_hard_cap)
                    g[i].mapq = 0; // fg-labs: force repetitive-supp MAPQ to 0
                aa[i][n_aa[i]++] = g[i];
            }
        }
        for (i = 0; i < n_aa[0]; ++i)
            mem_aln2sam(opt, bns, &str, &s[0], n_aa[0], aa[0], i, &h[1]);

        if (mem_opt_records_are_bam(opt)) {
            /* bam1_t path (meth or generic): mem_aln2sam short-circuited into
             * s->bams, leaving str untouched. Skip the str.s dance. Note this
             * is NOT opt->bam_mode — --meth without --bam still builds bam1_t
             * and would hit the `assert(str.s != 0)` below. */
            s[0].sam = NULL;
            str.l = 0;
            for (i = 0; i < n_aa[1]; ++i)
                mem_aln2sam(opt, bns, &str, &s[1], n_aa[1], aa[1], i, &h[0]);
            s[1].sam = NULL;
            free(str.s); str.s = NULL; str.m = 0;
        } else {
            assert(str.s != 0);
            s[0].sam = strdup(str.s); str.l = 0;
            for (i = 0; i < n_aa[1]; ++i)
                mem_aln2sam(opt, bns, &str, &s[1], n_aa[1], aa[1], i, &h[0]); // write read2 hits
            s[1].sam = str.s;
        }
        if (strcmp(s[0].name, s[1].name) != 0) err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n", s[0].name, s[1].name);
        // free
        for (i = 0; i < 2; ++i) {
            free(h[i].cigar); free(g[i].cigar);
            free(HN[i]);
            if (XA[i] == 0) continue;
            for (j = 0; j < a[i].n; ++j) free(XA[i][j]);
            free(XA[i]);
        }
        return n;
    }

    // no_pairing
    int which[2] = { -1, -1 };
    for (i = 0; i < 2; ++i) {
        if (a[i].n) {
            if (a[i].a[0].score >= opt->T) which[i] = 0;
            else if (n_pri[i] < a[i].n && a[i].a[n_pri[i]].score >= opt->T)
                which[i] = n_pri[i];
        }
        if (which[i] >= 0) h[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, &a[i].a[which[i]], s[i].meth_orig_seq);
        else h[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, 0);
    }
    // Proper-pair bit. Which alignment it is derived from -- the top-scoring
    // region a[0] (both upstreams, the default) or the emitted a[which] (#17,
    // opt-in via --proper-pair-from-emitted) -- lives in one place so this block
    // and its verbatim twin cannot drift apart; see mem_proper_pair_extra_flag.
    if (!(opt->flag & MEM_F_NOPAIRING) && which[0] >= 0 && which[1] >= 0 &&
        h[0].rid == h[1].rid && h[0].rid >= 0)
        extra_flag |= mem_proper_pair_extra_flag(opt, bns->l_pac, a, which, pes);
    mem_reg2sam(opt, bns, pac, &s[0], &a[0], 0x41|extra_flag, &h[1]);
    mem_reg2sam(opt, bns, pac, &s[1], &a[1], 0x81|extra_flag, &h[0]);
    if (strcmp(s[0].name, s[1].name) != 0)
        err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n",
                  s[0].name, s[1].name);

    free(h[0].cigar); free(h[1].cigar);
    return n;
}

int mem_sam_pe_batch_pre(const mem_opt_t *opt, const bntseq_t *bns,
                         const uint8_t *pac, const mem_pestat_t pes[4],
                         uint64_t id, bseq1_t s[2], mem_alnreg_v a[2],
                         mem_cache *mmc, int64_t &pcnt, int32_t &gcnt,
                         int32_t &maxRefLen, int32_t &maxQerLen,
                         int tid)
{
    //uint8_t *seqBufRef = mmc->seqBufLeftRef[tid*CACHE_LINE];
    //uint8_t *seqBufQer = mmc->seqBufLeftQer[tid*CACHE_LINE];
    // int64_t *wsize_buf = &(mmc->wsize_buf[tid]);

    //SeqPair *seqPairArray = mmc->seqPairArrayLeft128[tid];
    //int32_t *gar = (int32_t*) (mmc->seqPairArrayAux[tid]);
    // int64_t *wsize = &(mmc->wsize[tid]);
    
    int i, j, n_aa[2];
    kstring_t str;
    mem_aln_t h[2], g[2];
    // int tid = omp_get_thread_num();
    
    str.l = str.m = 0; str.s = 0;
    memset(h, 0, sizeof(mem_aln_t) * 2);
    memset(g, 0, sizeof(mem_aln_t) * 2);
    n_aa[0] = n_aa[1] = 0;
    
    if (!(opt->flag & MEM_F_NO_RESCUE)) { // then perform SW for the best alignment
        mem_alnreg_v b[2];
        kv_init(b[0]); kv_init(b[1]);
        for (i = 0; i < 2; ++i)
            for (j = 0; j < a[i].n; ++j)
                if (a[i].a[j].score >= a[i].a[0].score  - opt->pen_unpaired)
                    kv_push(mem_alnreg_t, b[i], a[i].a[j]);

        // NEW, batching
        for (i = 0; i < 2; ++i) {
            for (j = 0; j < b[i].n && j < opt->max_matesw; ++j) {
                int64_t val = mem_matesw_batch_pre(opt, bns, pac, pes, &b[i].a[j],
                                                   s[!i].l_seq, (uint8_t*)s[!i].seq,
                                                   opt->meth_mode ? s[!i].meth_orig_seq : NULL,
                                                   &a[!i], mmc, pcnt, gcnt,
                                                   maxRefLen, maxQerLen, tid,
                                                   opt->meth_mode ? i : -1);

                pcnt = val;
                gcnt += 4;
            }
        }       
        free(b[0].a); free(b[1].a);
    }
    
    return 1;
}

static inline void revseq(int l, uint8_t *s)
{
    int i, t;
    for (i = 0; i < l>>1; ++i)
        t = s[i], s[i] = s[l - 1 - i], s[l - 1 - i] = t;
}

/* Task 5: Read the BWAMEM3_METH_BATCHED_RESCUE escape hatch once. Default ON
 * (batched meth mate rescue). Set to "0" to force the legacy scalar ksw_align2
 * rescue path (used by the regression to A/B the same binary). Only consulted
 * under opt->meth_mode; the non-meth path is unaffected. */
static bool meth_batched_rescue_enabled()
{
    static const bool enabled = []() {
        /* Default ON: meth mate rescue runs on the batched SIMD kswv path,
         * which scores the rescued mate under its read#-derived bisulfite matrix
         * (one SW per rescue). Set BWAMEM3_METH_BATCHED_RESCUE=0 to force the
         * scalar ksw_align2 path (same single matrix; slower, used to A/B for
         * parity). */
        const char *e = getenv("BWAMEM3_METH_BATCHED_RESCUE");
        return !(e != NULL && strcmp(e, "0") == 0);
    }();
    return enabled;
}

/* Whether the RUNNING SIMD tier implements the kswv freed-cell override. Only
 * the NEON, AVX2, and AVX-512BW kernel bodies do; on sse41/sse42/avx the
 * mat-aware kswv ctor reports needsScalar() for any freed-cell matrix (see
 * kswv.cpp's `freed_simd_supported`). This must be a RUNTIME query, not a
 * compile-time `#if`: bwamem_pair.cpp is a non-kernel TU compiled once at
 * BASELINE_ARCH, while make_kswv dispatches to a per-tier kernel chosen at
 * runtime (and downgradable via BWAMEM3_FORCE_TIER), so the two need not agree.
 * Cached because the callers below sit in per-pair loops; the tier is fixed
 * once bwamem3_simd_init() has run. */
static bool meth_freed_cell_tier_supported()
{
    static const bool supported = []() {
        bwamem3_simd_init();          /* idempotent; guarantees the tier is set */
        const int tier = bwamem3_simd_tier();
        return tier == BWAMEM3_TIER_AVX2 || tier == BWAMEM3_TIER_AVX512BW
                || tier == BWAMEM3_TIER_NEON;
    }();
    return supported;
}

/* Whether the current --meth-scoring matrix is expressible by the batched kswv
 * kernel. The kernel handles exactly what mem_opt_fill_meth_mat produces:
 * COLLAPSED (conversion cell + its mirror, both freed to +a), GENOMIC (the single
 * conversion cell freed to +a, rank-1), and NEUTRAL (the single conversion cell
 * freed to 0). The kernel's freed-cell blend now scores the freed cell to its
 * matrix value (fr_val), so a single freed cell of ANY value — match (genomic) or
 * 0 (neutral) — is expressible; only a changed diagonal or a non-mirror multi-cell
 * matrix forces scalar. Expressibility is therefore purely a tier question: every
 * scoring mode is batched on a freed-capable tier, and every scoring mode takes
 * the scalar ksw_align2 rescue on the freed-less x86 tiers. Keep this in lockstep
 * with mem_opt_fill_meth_mat and the kswv mat-aware ctor. */
static inline bool meth_scoring_batched_expressible(const mem_opt_t *opt)
{
    (void)opt;   /* every mem_opt_fill_meth_mat matrix is kernel-expressible */
    return meth_freed_cell_tier_supported();
}

/* Task 5: run the two-phase batched kswv over a contiguous SeqPair slice
 * pairs[0..slice_pcnt) whose 8-bit pairs occupy [0, slice_pcnt8) and 16-bit
 * pairs [slice_pcnt8, slice_pcnt). The slice must have MAX_LINE_LEN trailing
 * headroom (the 16-bit shift writes into pairs[slice_pcnt + MAX_LINE_LEN)).
 * aln is the shared per-batch result array indexed by SeqPair.regid; seqBufRef
 * /seqBufQer are the shared (idr/idq-keyed) sequence buffers. This is the exact
 * body of the pre-Task-5 mem_sam_pe_batch, hoisted verbatim so the non-meth
 * call (one slice = the whole batch) stays byte-identical. */
static void mem_sam_pe_batch_run(Ikswv *pwsw, SeqPair *pairs,
                                 uint8_t *seqBufRef, uint8_t *seqBufQer,
                                 kswr_t *aln, int64_t slice_pcnt,
                                 int64_t slice_pcnt8, int nthreads)
{
    // Shift 16-bit
    for (int i=0; i<slice_pcnt-slice_pcnt8; i++)
        pairs[slice_pcnt + MAX_LINE_LEN - 1 - i] = pairs[slice_pcnt-i-1];

#if BWAMEM_BATCHED_MATESW
    pwsw->getScores8(pairs, seqBufRef, seqBufQer, aln, slice_pcnt8, nthreads, 0);
    pwsw->getScores16(pairs + slice_pcnt8 + MAX_LINE_LEN, seqBufRef, seqBufQer,
                      aln, slice_pcnt-slice_pcnt8, nthreads, 0);
#else
    fprintf(stderr, "Error: mem_sam_pe_batch reached without a batched kswv kernel\n");
    exit(EXIT_FAILURE);
#endif

    // Post-processing
    int pos = 0, pos8 = 0, pos16 = 0;
    for (int i=0; i<slice_pcnt8; i++)
    {
        SeqPair sp = pairs[i];
        int ind = sp.regid;
        kswr_t r = aln[ind];
        int xtra = sp.h0;
        if ((xtra & KSW_XSTART) == 0 || ((xtra & KSW_XSUBO) && r.score < (xtra & 0xffff))) continue;

        sp.h0 = KSW_XSTOP | r.score;
        sp.len2 = r.qe + 1;
        uint8_t *qs = seqBufQer + sp.idq;
        uint8_t *rs = seqBufRef + sp.idr;
        revseq(r.qe + 1, qs); revseq(r.te + 1, rs);
        pairs[pos++] = sp;
        pos8 ++;
    }

    int id = slice_pcnt8 + MAX_LINE_LEN;
    for (int i=0; i<slice_pcnt-slice_pcnt8; i++)
    {
        SeqPair sp = pairs[i + id];
        int ind = sp.regid;
        kswr_t r = aln[ind];
        int xtra = sp.h0;
        if ((xtra & KSW_XSTART) == 0 || ((xtra & KSW_XSUBO) && r.score < (xtra & 0xffff))) continue;

        sp.h0 = KSW_XSTOP | r.score;
        sp.len2 = r.qe + 1;
        uint8_t *qs = seqBufQer + sp.idq;
        uint8_t *rs = seqBufRef + sp.idr;
        revseq(r.qe + 1, qs); revseq(r.te + 1, rs);
        pairs[pos++] = sp;
        pos16 ++;
    }

    int pcnt2 = pos;
    assert(pos8 + pos16 == pcnt2);
    (void) pcnt2;

#if BWAMEM_BATCHED_MATESW
    pwsw->getScores16(pairs + pos8, seqBufRef, seqBufQer, aln, pos16, nthreads, 1);
    pwsw->getScores8(pairs, seqBufRef, seqBufQer, aln, pos8, nthreads, 1);
#else
    fprintf(stderr, "Error: mem_sam_pe_batch reached without a batched kswv kernel\n");
    exit(EXIT_FAILURE);
#endif
}

// This function is equivalent to align2() for axv512
int mem_sam_pe_batch(const mem_opt_t *opt, mem_cache *mmc,
                     int64_t &pcnt, int64_t &pcnt8, kswr_t *aln,
                     int32_t maxRefLen, int32_t maxQerLen, int tid)
{
    uint8_t *seqBufRef = mmc->seqBufLeftRef[tid*CACHE_LINE];
    uint8_t *seqBufQer = mmc->seqBufLeftQer[tid*CACHE_LINE];

    SeqPair *seqPairArray = mmc->seqPairArrayLeft128[tid];

#if DEBUG    // orig function from bwa-mem -- for debugging purpose. Disabled by default.
    // uint64_t tim = __rdtsc();
    SeqPair sp;
    for (int i=0; i<pcnt; i++) {
        sp = seqPairArray[i];
        int xtra = sp.h0;
        uint8_t *qs = seqBufQer + sp.idq;
        uint8_t *rs = seqBufRef + sp.idr;
        aln[i] = ksw_align2(sp.len2, qs, sp.len1, rs, 5,
                            opt->mat, opt->o_del, opt->e_del,
                            opt->o_ins, opt->e_ins, xtra, 0);
    }
    // tprof[SAM2][0] += __rdtsc() - tim;

#else   // avx512, vectorized function

    for (int i=0; i<pcnt; i++) {
        kswr_t *r = &aln[i];
        r->tb = r->qb = -1;
    }

    int nthreads = 1; // no multi-threading here

    /* Task 5: under --meth (batched rescue enabled), partition the enqueued
     * mate-rescue pairs by bisulfite hypothesis (OT/OB) and run the kernel once
     * per group with the matching mat-aware kswv object. mat_ot frees C->T (top
     * strand), mat_ob frees G->A (bottom strand); make_kswv installs the
     * freed-cell override for each, so the batched score equals the scalar
     * ksw_align2 score with the asymmetric matrix. Non-meth and the env-OFF
     * escape hatch fall through to the single-object path below, byte-identical
     * to the pre-Task-5 code. */
    if (opt->meth_mode && meth_batched_rescue_enabled()
            && meth_scoring_batched_expressible(opt)) {
        // Scratch slice (Right128 is free here; sort_classify only used it as
        // a transient and the final layout lives in Left128). Sized
        // wsize_pair + MAX_LINE_LEN, which comfortably holds one group plus
        // the 16-bit shift headroom (a group is <= pcnt <= wsize_pair pairs).
        SeqPair *scratch = mmc->seqPairArrayRight128[tid];

        auto pwsw_ot = make_kswv(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                                 opt->a, -1*opt->b, nthreads,
                                 maxRefLen, maxQerLen, opt->mat_ot);
        auto pwsw_ob = make_kswv(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                                 opt->a, -1*opt->b, nthreads,
                                 maxRefLen, maxQerLen, opt->mat_ob);
        // The batched kernel expresses exactly the matrices mem_opt_fill_meth_mat
        // produces: GENOMIC (one freed cell to +a), NEUTRAL (one freed cell to 0),
        // and COLLAPSED (the conversion cell PLUS its mirror — a symmetric
        // (i,j)/(j,i) pair). The freed-less x86 tiers never reach here —
        // meth_scoring_batched_expressible() gates this branch on the running
        // tier — so needsScalar() is always false. Guard it at RUNTIME rather
        // than with a bare assert(): under NDEBUG assert is a no-op, which would
        // let a future unsupported matrix run the batched kernel and silently
        // mis-score. Fail loudly instead — this can only fire on a programming
        // error in mem_opt_fill_meth_mat (a matrix that is neither a single freed
        // cell nor a mirror pair) or a tier gate that drifted out of lockstep
        // with the kswv mat-aware ctor.
        if (pwsw_ot->needsScalar() || pwsw_ob->needsScalar())
            err_fatal(__func__,
                      "meth mate-rescue matrix not expressible by the batched kernel "
                      "(neither a single freed cell nor a collapsed mirror pair)");

        // hyp == 1 -> OT (mat_ot / pwsw_ot); hyp == 0 -> OB (mat_ob / pwsw_ob).
        // -1 (non-meth) cannot occur here: every enqueued pair under meth_mode
        // was tagged with a real hypothesis in mem_matesw_batch_pre. Process
        // each group in the shared scratch buffer one at a time; aln[] is keyed
        // by regid and seqBuf* by idr/idq, so each pair is scored exactly once
        // regardless of grouping.
        int64_t scored = 0;
        for (int hyp = 1; hyp >= 0; --hyp) {
            int64_t n8 = 0, n16 = 0;
            // 8-bit pairs of this hyp first, then 16-bit, mirroring
            // sort_classify's layout so mem_sam_pe_batch_run's split is valid.
            for (int64_t i = 0; i < pcnt; ++i) {
                SeqPair *s = &seqPairArray[i];
                if (s->meth_hyp != hyp) continue;
                if (s->h0 & KSW_XBYTE) scratch[n8++] = *s;
            }
            int64_t group_pcnt8 = n8;
            for (int64_t i = 0; i < pcnt; ++i) {
                SeqPair *s = &seqPairArray[i];
                if (s->meth_hyp != hyp) continue;
                if (!(s->h0 & KSW_XBYTE)) scratch[group_pcnt8 + (n16++)] = *s;
            }
            int64_t group_pcnt = n8 + n16;
            scored += group_pcnt;
            if (group_pcnt == 0) continue;

            /* --rescue-kmer: same length sort as the non-meth path below, applied
             * per OT/OB partition -- otherwise --fast --meth would enqueue narrowed
             * windows but leave them mixed with full ones inside each SIMD group,
             * paying the scan cost without collecting the saving. */
            if (opt->rescue_kmer)
                matesw_sort_partitions_by_len(scratch, group_pcnt8, group_pcnt);

            Ikswv *pwsw = (hyp == 1) ? pwsw_ot.get() : pwsw_ob.get();
            mem_sam_pe_batch_run(pwsw, scratch, seqBufRef, seqBufQer,
                                 aln, group_pcnt, group_pcnt8, nthreads);
        }
        // Release-safe invariant guard: the OT/OB partition must cover every
        // enqueued pair. A pair tagged with a hypothesis outside {0,1} would be
        // skipped by both groups, leaving its aln[] entry uninitialized for
        // mem_matesw_batch_post to read. The assert in mem_matesw_batch_pre
        // catches this in debug; fail loud under NDEBUG too rather than emit a
        // silently mis-scored rescue.
        if (scored != pcnt)
            err_fatal(__func__,
                      "batched meth rescue partition missed %ld of %ld pairs "
                      "(a pair carried a hypothesis outside {OT,OB})",
                      (long)(pcnt - scored), (long)pcnt);
        return 1;
    }

    auto pwsw = make_kswv(opt->o_del, opt->e_del, opt->o_ins, opt->e_ins,
                          opt->a, -1*opt->b, nthreads,
                          maxRefLen, maxQerLen);

    if (opt->rescue_kmer) matesw_sort_partitions_by_len(seqPairArray, pcnt8, pcnt);

    mem_sam_pe_batch_run(pwsw.get(), seqPairArray, seqBufRef, seqBufQer,
                         aln, pcnt, pcnt8, nthreads);

#endif

    return 1;
}

int mem_sam_pe_batch_post(const mem_opt_t *opt, const bntseq_t *bns,
                          const uint8_t *pac, const mem_pestat_t pes[4],
                          uint64_t id, bseq1_t s[2], mem_alnreg_v a[2],
                          kswr_t **myaln, mem_cache *mmc, 
                          int32_t &gcnt, int tid)
{
    extern int mem_mark_primary_se(const mem_opt_t *opt, int n, mem_alnreg_t *a, int64_t id);
    extern void mem_seed_capped_sub(mem_alnreg_v *a, int match_a);
    extern int mem_approx_mapq_se(const mem_opt_t *opt, const mem_alnreg_t *a);
    extern void mem_reg2sam(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                            bseq1_t *s, mem_alnreg_v *a, int extra_flag, const mem_aln_t *m);
    extern char **mem_gen_alt(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                              const mem_alnreg_v *a, int l_query, const char *query,
                              int **out_hn, const char *meth_orig_query);
    #if MATE_SORT
    extern void sort_alnreg_re(int n, mem_alnreg_t* a);
    extern void sort_alnreg_score(int n, mem_alnreg_t* a);
    /* D3 (--meth, PR-6): dedup-only call (query = 0, no patch SW) — matrix
     * unused, opt->mat default correct; asymmetric scoring is in
     * mem_matesw_batch_post's scalar fallback. */
    #endif

    int32_t *gar = (int32_t*) mmc->seqPairArrayAux[tid];
    
    int n = 0, i, j, z[2], o, subo, n_sub, extra_flag = 1, n_pri[2], n_aa[2];
    kstring_t str;
    mem_aln_t h[2], g[2], aa[2][2];
    // int tid = omp_get_thread_num();
    
    str.l = str.m = 0; str.s = 0;
    memset(h, 0, sizeof(mem_aln_t) * 2);
    memset(g, 0, sizeof(mem_aln_t) * 2);
    n_aa[0] = n_aa[1] = 0;
    
    if (!(opt->flag & MEM_F_NO_RESCUE)) { // then perform SW for the best alignment
        mem_alnreg_v b[2];
        kv_init(b[0]); kv_init(b[1]);
        for (i = 0; i < 2; ++i)
            for (j = 0; j < a[i].n; ++j)
                if (a[i].a[j].score >= a[i].a[0].score  - opt->pen_unpaired)
                    kv_push(mem_alnreg_t, b[i], a[i].a[j]);
                
        for (int l=0; l<a[0].n; l++)
            a[0].a[l].flg = 0;
        for (int l=0; l<a[1].n; l++)
            a[1].a[l].flg = 0;

        #if MATE_SORT
        for (i = 0; i < 2; ++i) {
            sort_alnreg_re(a[!i].n, a[!i].a);
            int val = 0, swcount = 0;
            for (j = 0; j < b[i].n && j < opt->max_matesw; ++j) {
                /* D3 (--meth): original mate bases (ms_orig). rmat is vestigial —
                 * under --meth `mat` is consulted on neither path: the batched
                 * kswv result was scored in-kernel via the per-pair sp.meth_hyp
                 * (set in mem_matesw_batch_pre), and the scalar index==-1 fallback
                 * derives its matrix from the mate's own read# chemistry
                 * (mate_meth_ot ^ is_rev). So rmat is dead code under --meth. */
                const char  *ms_orig = opt->meth_mode ? s[!i].meth_orig_seq : NULL;
                const int8_t *rmat    = opt->meth_mode
                    ? mem_opt_meth_mat(opt, !b[i].a[j].meth_hypothesis) : NULL;
                val = mem_matesw_batch_post(opt, bns, pac, pes, &b[i].a[j],
                                                s[!i].l_seq, (uint8_t*)s[!i].seq,
                                                &a[!i], myaln, gcnt, gar, mmc,
                                                ms_orig, rmat, opt->meth_mode ? i : -1);
                n += val;
                swcount += val;
                // ncnt++;
                gcnt += 4;
            }
            if (swcount > 0) {
                mem_alnreg_v* ma = &a[!i];
                ma->n = mem_sort_dedup_patch(opt, 0, 0, 0, ma->n, ma->a);
            }
            else {
                sort_alnreg_score(a[!i].n, a[!i].a);
            }
        }
        #else
        for (i = 0; i < 2; ++i) {
            for (j = 0; j < b[i].n && j < opt->max_matesw; ++j) {
                /* D3 (--meth, PR-6, B3): see MATE_SORT branch above. */
                const char  *ms_orig = opt->meth_mode ? s[!i].meth_orig_seq : NULL;
                const int8_t *rmat    = opt->meth_mode
                    ? mem_opt_meth_mat(opt, !b[i].a[j].meth_hypothesis) : NULL;
                int val = mem_matesw_batch_post(opt, bns, pac, pes, &b[i].a[j],
                                                s[!i].l_seq, (uint8_t*)s[!i].seq,
                                                &a[!i], myaln, gcnt, gar, mmc,
                                                ms_orig, rmat, opt->meth_mode ? i : -1);
                n += val;
                // ncnt++;
                gcnt += 4;
            }
        }
        #endif
        free(b[0].a); free(b[1].a);
    }

    n_pri[0] = mem_mark_primary_se(opt, a[0].n, a[0].a, id<<1|0);
    n_pri[1] = mem_mark_primary_se(opt, a[1].n, a[1].a, id<<1|1);

    #if V17
    if (opt->flag & MEM_F_PRIMARY5) {
        mem_reorder_primary5(opt->T, &a[0]);
        mem_reorder_primary5(opt->T, &a[1]);
    }
    #endif

    /* --extend-csub: seed each end's primary competitor score from the chains the
     * cap/gate dropped, before pairing/MAPQ below reads csub. No-op when off. */
    mem_seed_capped_sub(&a[0], opt->a);
    mem_seed_capped_sub(&a[1], opt->a);
    
    if (opt->flag&MEM_F_NOPAIRING) goto no_pairing;

    // pairing single-end hits
    if (n_pri[0] && n_pri[1] && (o = mem_pair(opt, bns, pac, pes, s, a, id, &subo, &n_sub, z, n_pri)) > 0)
    {
        int is_multi[2], q_pe, score_un, q_se[2];
        char **XA[2];
        // check if an end has multiple hits even after mate-SW
        for (i = 0; i < 2; ++i) {
            for (j = 1; j < n_pri[i]; ++j)
                if (a[i].a[j].secondary < 0 && a[i].a[j].score >= opt->T) break;
            is_multi[i] = j < n_pri[i]? 1 : 0;
        }
        if (is_multi[0] || is_multi[1]) goto no_pairing; // TODO: in rare cases, the true hit may be long but with low score
        // compute mapQ for the best SE hit
        score_un = a[0].a[0].score + a[1].a[0].score - opt->pen_unpaired;
        //q_pe = o && subo < o? (int)(MEM_MAPQ_COEF * (1. - (double)subo / o) * log(a[0].a[z[0]].seedcov + a[1].a[z[1]].seedcov) + .499) : 0;
        subo = subo > score_un? subo : score_un;
        if (opt->extend_csub) { int fl = mem_capped_pair_subo(a, z, o); if (fl > subo) subo = fl; }
        {
            /* Meth PE MAPQ hardening (dragmap-style, cf. minibwa r404): fold each
             * end's SECOND-best single-end hit into the pair MAPQ so a repeat on
             * either end deflates confidence even when one paired alignment looks
             * clean. Meth-gated so non-meth PE MAPQ is byte-identical. */
            int qdiff = o - subo;
            if (opt->meth_mode) {
                int se2_0 = (n_pri[0] > 1) ? a[0].a[1].score : 0;
                int se2_1 = (n_pri[1] > 1) ? a[1].a[1].score : 0;
                int cap = o + 4 * opt->a - (se2_0 + se2_1);
                if (qdiff > cap) qdiff = cap;
            }
            if (qdiff < 0) qdiff = 0;
            q_pe = raw_mapq(qdiff, opt->a);
        }

        if (n_sub > 0) q_pe -= (int)(4.343 * log(n_sub+1) + .499);
        if (q_pe < 0) q_pe = 0;
        if (q_pe > 60) q_pe = 60;

        q_pe = (int)(q_pe * (1. - .5 * (a[0].a[0].frac_rep + a[1].a[0].frac_rep)) + .499);

        // the following assumes no split hits
        if (o > score_un) { // paired alignment is preferred
            mem_alnreg_t *c[2];
            c[0] = &a[0].a[z[0]]; c[1] = &a[1].a[z[1]];
            for (i = 0; i < 2; ++i) {
                if (c[i]->secondary >= 0)
                    c[i]->sub = a[i].a[c[i]->secondary].score, c[i]->secondary = -2;
                q_se[i] = mem_approx_mapq_se(opt, c[i]);
            }

            q_se[0] = q_se[0] > q_pe? q_se[0] : q_pe < q_se[0] + 40? q_pe : q_se[0] + 40;
            q_se[1] = q_se[1] > q_pe? q_se[1] : q_pe < q_se[1] + 40? q_pe : q_se[1] + 40;
            extra_flag |= 2;

            // cap at the tandem repeat score
            q_se[0] = q_se[0] < raw_mapq(c[0]->score - c[0]->csub, opt->a)? q_se[0] : raw_mapq(c[0]->score - c[0]->csub, opt->a);
            q_se[1] = q_se[1] < raw_mapq(c[1]->score - c[1]->csub, opt->a)? q_se[1] : raw_mapq(c[1]->score - c[1]->csub, opt->a);

        } else { // the unpaired alignment is preferred
            z[0] = z[1] = 0;
            q_se[0] = mem_approx_mapq_se(opt, &a[0].a[0]);
            q_se[1] = mem_approx_mapq_se(opt, &a[1].a[0]);

        }
        for (i = 0; i < 2; ++i) {
            int k = a[i].a[z[i]].secondary_all;
            if (k >= 0 && k < n_pri[i]) { // switch secondary and primary if both of them are non-ALT
                assert(a[i].a[k].secondary_all < 0);
                for (j = 0; j < a[i].n; ++j)
                    if (a[i].a[j].secondary_all == k || j == k)
                        a[i].a[j].secondary_all = z[i];
                a[i].a[z[i]].secondary_all = -1;
            }
        }
        int *HN[2] = { 0, 0 };
        if (!(opt->flag & MEM_F_ALL)) {
            for (i = 0; i < 2; ++i)
                XA[i] = mem_gen_alt(opt, bns, pac, &a[i], s[i].l_seq, s[i].seq, &HN[i], s[i].meth_orig_seq);
        } else XA[0] = XA[1] = 0;
        // write SAM
        for (i = 0; i < 2; ++i) {
            h[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, &a[i].a[z[i]], s[i].meth_orig_seq);
            h[i].mapq = q_se[i];

            h[i].flag |= 0x40<<i | extra_flag;
            h[i].XA = XA[i]? XA[i][z[i]] : 0;
            h[i].HN = HN[i]? HN[i][z[i]] : -1;
            aa[i][n_aa[i]++] = h[i];
            if (n_pri[i] < a[i].n) { // the read has ALT hits
                mem_alnreg_t *p = &a[i].a[n_pri[i]];
                if (p->score < opt->T || p->secondary >= 0 || !p->is_alt) continue;
                g[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, p, s[i].meth_orig_seq);
                g[i].flag |= 0x800 | 0x40<<i | extra_flag;
                g[i].XA = XA[i]? XA[i][n_pri[i]] : 0;
                g[i].HN = HN[i]? HN[i][n_pri[i]] : -1;
                if (opt->supp_rep_hard_cap > 0 && p->chain_n_hits >= opt->supp_rep_hard_cap)
                    g[i].mapq = 0; // fg-labs: force repetitive-supp MAPQ to 0
                aa[i][n_aa[i]++] = g[i];
            }
        }
        for (i = 0; i < n_aa[0]; ++i)
            mem_aln2sam(opt, bns, &str, &s[0], n_aa[0], aa[0], i, &h[1]);

        if (mem_opt_records_are_bam(opt)) {
            /* bam1_t path (meth or generic): mem_aln2sam short-circuited into
             * s->bams, leaving str untouched. Skip the str.s dance. Note this
             * is NOT opt->bam_mode — --meth without --bam still builds bam1_t
             * and would hit the `assert(str.s != 0)` below. */
            s[0].sam = NULL;
            str.l = 0;
            for (i = 0; i < n_aa[1]; ++i)
                mem_aln2sam(opt, bns, &str, &s[1], n_aa[1], aa[1], i, &h[0]);
            s[1].sam = NULL;
            free(str.s); str.s = NULL; str.m = 0;
        } else {
            assert(str.s != 0);
            s[0].sam = strdup(str.s); str.l = 0;
            for (i = 0; i < n_aa[1]; ++i)
                mem_aln2sam(opt, bns, &str, &s[1], n_aa[1], aa[1], i, &h[0]);
            s[1].sam = str.s;
        }
        if (strcmp(s[0].name, s[1].name) != 0) err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n", s[0].name, s[1].name);
        // free
        for (i = 0; i < 2; ++i) {
            free(h[i].cigar); free(g[i].cigar);
            free(HN[i]);
            if (XA[i] == 0) continue;
            for (j = 0; j < a[i].n; ++j) free(XA[i][j]);
            free(XA[i]);
        }
    } else goto no_pairing;
    return n;

no_pairing:
    int which[2] = { -1, -1 };
    for (i = 0; i < 2; ++i) {
        if (a[i].n) {
            if (a[i].a[0].score >= opt->T) which[i] = 0;
            else if (n_pri[i] < a[i].n && a[i].a[n_pri[i]].score >= opt->T)
                which[i] = n_pri[i];
        }
        if (which[i] >= 0) h[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, &a[i].a[which[i]], s[i].meth_orig_seq);
        else h[i] = mem_reg2aln(opt, bns, pac, s[i].l_seq, s[i].seq, 0);
    }
    // Proper-pair bit. Which alignment it is derived from -- the top-scoring
    // region a[0] (both upstreams, the default) or the emitted a[which] (#17,
    // opt-in via --proper-pair-from-emitted) -- lives in one place so this block
    // and its verbatim twin cannot drift apart; see mem_proper_pair_extra_flag.
    if (!(opt->flag & MEM_F_NOPAIRING) && which[0] >= 0 && which[1] >= 0 &&
        h[0].rid == h[1].rid && h[0].rid >= 0)
        extra_flag |= mem_proper_pair_extra_flag(opt, bns->l_pac, a, which, pes);
    mem_reg2sam(opt, bns, pac, &s[0], &a[0], 0x41|extra_flag, &h[1]);
    mem_reg2sam(opt, bns, pac, &s[1], &a[1], 0x81|extra_flag, &h[0]);
    if (strcmp(s[0].name, s[1].name) != 0)
        err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n",
                  s[0].name, s[1].name);

    free(h[0].cigar); free(h[1].cigar);
    return n;
}


int mem_matesw_batch_pre(const mem_opt_t *opt, const bntseq_t *bns,
                         const uint8_t *pac, const mem_pestat_t pes[4],
                         const mem_alnreg_t *a, int l_ms, const uint8_t *ms,
                         const char *ms_orig,
                         mem_alnreg_v *ma, mem_cache *mmc, int pcnt, int32_t gcnt,
                         int32_t &maxRefLen, int32_t &maxQerLen, int32_t tid,
                         int mate_meth_ot)
{
    /* D3 (--meth, A1/PR-6 → issue 173 Task 5): _pre stages the batched-SIMD
     * mate-rescue query for the kswv getScores8/16 kernel. As of the mat-aware
     * kernels (Tasks 1-4) the batched kernel CAN express the bisulfite OT/OB
     * freed-cell matrices, so under --meth we now stage meth pairs here just
     * like non-meth pairs and tag each with its bisulfite hypothesis
     * (sp.meth_hyp, set near the enqueue below). mem_sam_pe_batch then
     * partitions the enqueued pairs by OT/OB and scores each group with the
     * matching mat-aware kswv object, byte-identical to the per-hypothesis
     * scalar ksw_align2. The legacy scalar path (gar[gcnt+r] = -1 →
     * mem_matesw_batch_post re-runs ksw_align2) is retained as the escape hatch
     * (BWAMEM3_METH_BATCHED_RESCUE=0). The OT/OB matrices mem_opt_fill_meth_mat
     * produces are always expressible on a freed-cell tier (a single cell freed
     * to +a for genomic or to 0 for neutral, or a collapsed mirror pair), so
     * the kswv objects never report needsScalar() here; if a future
     * matrix or tier ever did, mem_sam_pe_batch fails loud via err_fatal rather
     * than silently mis-scoring (see the needsScalar() guard there). The dedup
     * below passes query = 0 (no patch SW), so its mat default (opt->mat) is
     * unused. */

    uint8_t *seqBufRef = mmc->seqBufLeftRef[tid*CACHE_LINE];
    uint8_t *seqBufQer = mmc->seqBufLeftQer[tid*CACHE_LINE];
    SeqPair *seqPairArray = mmc->seqPairArrayLeft128[tid];
    int32_t *gar = (int32_t*) (mmc->seqPairArrayAux[tid]);

    int64_t *wsize_pair = &(mmc->wsize[tid]);
    int64_t *wsize_buf_ref = &(mmc->wsize_buf_ref[tid*CACHE_LINE]);
    int64_t *wsize_buf_qer = &(mmc->wsize_buf_qer[tid*CACHE_LINE]);
    
    int64_t l_pac = bns->l_pac;
    int i, r, skip[4], rid = -1;
    for (r = 0; r < 4; ++r)
        skip[r] = pes[r].failed? 1 : 0;

    for (i = 0; i < ma->n; ++i) { // check which orinentation has been found        
        int64_t dist;
        r = mem_infer_dir(l_pac, a->rb, ma->a[i].rb, &dist);
        if (dist >= pes[r].low && dist <= pes[r].high) {
            skip[r] = 1;
        }
    }

    
    if (skip[0] + skip[1] + skip[2] + skip[3] == 4) //return pcnt; // consistent pair exist; no need to perform SW
    {
        gar[gcnt + 3] = gar[gcnt + 2] = gar[gcnt + 1] = gar[gcnt + 0] = -1;
        return pcnt;
    }

    /* D3 (--meth): score the ORIGINAL (unconverted) mate bases, exactly as
     * mem_matesw_batch_post's scalar path does. Without this the batched kernel
     * aligns the C→T/G→A-projected query (s->seq) — a strict information loss
     * (original-C vs genomic-T is already collapsed), which degrades repeat-mate
     * placement (wrongCHR@30 117 vs the scalar path's 88). 2-bit-encode
     * meth_orig_seq here; the per-orientation RC in the loop below then handles
     * it identically to the projected mate. Freed before the return. */
    uint8_t *ms2 = NULL;
    if (ms_orig != NULL) {
        ms2 = (uint8_t*) malloc(l_ms);
        xassert(ms2 != NULL, "out of memory: ms2");
        for (int k = 0; k < l_ms; ++k) {
            unsigned char c = (unsigned char) ms_orig[k];
            ms2[k] = nst_nt4_decode(c, 4);
        }
        ms = ms2;
    }

    for (r = 0; r < 4; ++r)
    {
        int is_rev, is_larger;
        uint8_t *ref = 0;
        int64_t rb, re;
        if (skip[r]) {
            gar[gcnt + r] = -1;
            continue;
        }
        is_rev = (r>>1 != (r&1)); // whether to reverse complement the mate
        is_larger = !(r>>1); // whether the mate has larger coordinate

        /* RESC-A7: the mate query bases ($ms) are directly readable, so the
         * reverse-complement (is_rev orientations) is built straight into the
         * kernel input buffer (seqBufQer) at enqueue time below — eliminating
         * the per-orientation `rev` scratch malloc/free and the redundant
         * second copy that staged rev -> seqBufQer. The forward orientation
         * likewise copies $ms directly. Byte-identical to the old rev buffer
         * (see the enqueue copy below for the equivalence argument). */

        if (!is_rev) {
            rb = is_larger? a->rb + pes[r].low : a->rb - pes[r].high;
            re = (is_larger? a->rb + pes[r].high: a->rb - pes[r].low) + l_ms; // if on the same strand, end position should be larger to make room for the seq length
        }
        else {
            rb = (is_larger? a->rb + pes[r].low : a->rb - pes[r].high) - l_ms; // similarly on opposite strands
            re = is_larger? a->rb + pes[r].high: a->rb - pes[r].low;
        }

        if (rb < 0) rb = 0;
        if (re > l_pac<<1) re = l_pac<<1;
        // bns_fetch_seq_v2 is zero-copy: it returns a pointer into ref_string
        // (the unpacked .0123 reference). The legacy bns_fetch_seq malloc'd a
        // fresh buffer, which the trailing free(ref) below released; v2 has
        // no allocation, so we drop that free. The seqPairArrayAux scratch
        // is unused by v2 but kept for signature parity with the existing v2
        // caller in bwamem.cpp::mem_chain2aln_across_reads_V2.
        if (rb < re) ref = bns_fetch_seq_v2(bns, pac, &rb, (rb+re)>>1, &re, &rid,
                                            mmc->ref_string,
                                            (uint8_t*) mmc->seqPairArrayAux[tid]);

        if (a->rid == rid && re - rb >= opt->min_seed_len) { // no funny things happening
            /* D3 (--meth, Task 5): meth mate-rescue is now enqueued for the
             * batched kswv kernel just like a non-meth pair. The kernels are
             * mat-aware (Task 2/4): make_kswv(..., mat_ot/mat_ob) installs the
             * freed-cell override for the bisulfite OT/OB matrices, so
             * the batched score matches the scalar ksw_align2 score with the
             * asymmetric matrix. The per-pair hypothesis tag (sp.meth_hyp,
             * below) drives the OT/OB partition in mem_sam_pe_batch, which runs
             * getScores8/16 once per hypothesis group with the matching object.
             * The rescued mate is scored under its OWN read# chemistry
             * (sp.meth_hyp = (mate_meth_ot ^ is_rev) & 1, set below), matching
             * mem_matesw_batch_post's scalar index==-1 fallback. The legacy scalar
             * path remains as a safety fallback for any pair whose object
             * reports needsScalar() (meth never does on a freed-capable tier) and
             * is reachable via BWAMEM3_METH_BATCHED_RESCUE=0 (escape hatch) or on
             * a freed-less SIMD tier (sse41/sse42/avx), where the kswv kernel has
             * no freed-cell override and every scoring mode must score scalar. */
            if (opt->meth_mode && (!meth_batched_rescue_enabled()
                    || !meth_scoring_batched_expressible(opt))) {
                // Escape hatch (env=0) or freed-less tier: keep the legacy scalar
                // rescue. Leave gar = -1 so mem_matesw_batch_post re-runs this
                // orientation through ksw_align2 with the asymmetric matrix.
                gar[gcnt + r] = -1;
                continue;
            }
            /* --rescue-kmer: band this window to the anchor diagonal before enqueue
             * (meth collapses the anchor scan to 3 letters on this pair's strand).
             * mem_matesw_batch_post applies the same offset, so coords stay
             * consistent; narrow_ob (0 = full window) is recorded at enqueue. */
            int narrow_ob = 0;
            if (opt->rescue_kmer) {
                int collapse = 0;
                if (opt->meth_mode) { int hyp = (mate_meth_ot ^ is_rev) & 1; collapse = hyp ? 1 : 2; }
                matesw_anchor_t anc = matesw_kmer_anchor(opt, ref, (int)(re - rb),
                                                         ms, l_ms, is_rev, collapse);
                const bool declined = matesw_anchor_declines_rescue(opt, anc);
#ifdef BWA_MEM3_DEBUG_RESCUE_STATS
                if (anc.scanned) {
                    g_rescue_stat_scans.fetch_add(1, std::memory_order_relaxed);
                    if (anc.narrowed) g_rescue_stat_narrowed.fetch_add(1, std::memory_order_relaxed);
                    if (declined)     g_rescue_stat_skipped.fetch_add(1, std::memory_order_relaxed);
                }
#endif
                /* --rescue-skip: no anchor worth the SW. Mark the orientation
                 * declined and do not enqueue. mem_matesw_batch_post reads this
                 * same gar slot under the identical `a->rid == rid && re - rb >=
                 * min_seed_len` guard, so the sentinel is always observed. */
                if (declined) {
                    gar[gcnt + r] = MATESW_GAR_DECLINED;
                    continue;
                }
                if (anc.narrowed) {
                    int64_t rb_full = rb;
                    ref += anc.ob; rb = rb_full + anc.ob; re = rb_full + anc.oe;
                    narrow_ob = anc.ob;
                }
            }
            //kswr_t aln;
            //mem_alnreg_t b;
            int u8_shift = matesw_u8_shift_for(opt, opt->mat);
            int xtra = KSW_XSUBO | KSW_XSTART | (matesw_use_u8(l_ms, opt->a, u8_shift)? KSW_XBYTE : 0) | (opt->min_seed_len * opt->a);

            /* D3 (--meth): enqueue ONE SW, scored under the rescued mate's own
             * read-number chemistry (mate_meth_ot: R1=1/OT, R2=0/OB) flipped by
             * the rescue strand (this path RC's the READ, so is_rev toggles the
             * freed cell). This is the chemistry that frees the mate's conversions
             * at its true locus, so the single matrix is correct — replacing the
             * old two-hypothesis (OT+OB) enqueue + keep-max, which cost 2x the
             * rescue SW to empirically rediscover this same matrix. Non-meth also
             * enqueues one pair. */
            int n_hyp = 1;
            for (int hi = 0; hi < n_hyp; ++hi)
            {
                int qerOffset = 0, refOffset = 0;
                if (pcnt != 0)
                {
                    SeqPair prev = seqPairArray[pcnt - 1];
                    refOffset = prev.idr + prev.len1;
                    qerOffset = prev.idq + prev.len2;
                }
                SeqPair sp;
                sp.h0 = xtra;
                // assert(pcnt < BATCH_SIZE * SEEDS_PER_READ);
                // Array is allocated (*wsize_pair + MAX_LINE_LEN); the +1024 grow
                // below keeps pcnt <= *wsize_pair, but a rescue enqueue can
                // momentarily hit pcnt == *wsize_pair before that grow fires, so
                // bound against the true allocation to avoid a debug false-positive.
                assert(pcnt < *wsize_pair + MAX_LINE_LEN);
            
                sp.idq = qerOffset;
                sp.idr = refOffset;
                sp.len1 = re - rb;
                sp.len2 = l_ms;
                sp.id = sp.score = sp.seqid = sp.gtle = sp.tle = sp.qle = sp.max_off = sp.gscore = -1; // not needed, remove while code cleaning
            
                assert(sp.len1 >= 0 && sp.len2 >= 0);
                if (refOffset + sp.len1 >= *wsize_buf_ref)
                {
                    if (bwa_verbose >= 4) fprintf(stderr, "[0000][%0.4d] Re-allocating (doubling) seqBufRefs in %s\n",
                            tid, __func__);
                    int64_t tmp = *wsize_buf_ref;
                    *wsize_buf_ref = seqbuf_grow_capacity(tmp);
                    if (*wsize_buf_ref == SEQBUF_CAPACITY_OVERFLOW)
                        seqbuf_capacity_fatal("seqBufRef", __func__, tmp);
                    assert(*wsize_buf_ref > refOffset + sp.len1);

                    uint8_t *seqBufRef_ = (uint8_t*)
                        _mm_realloc(seqBufRef, tmp, *wsize_buf_ref, sizeof(uint8_t)); 
                    mmc->seqBufLeftRef[tid*CACHE_LINE] = seqBufRef = seqBufRef_;

                    seqBufRef_ = (uint8_t*)
                        _mm_realloc(mmc->seqBufRightRef[tid*CACHE_LINE], tmp,
                                    *wsize_buf_ref, sizeof(uint8_t)); 
                    mmc->seqBufRightRef[tid*CACHE_LINE] = seqBufRef_;               
                }
            
                if (qerOffset + sp.len2 >= *wsize_buf_qer)
                {
                    if (bwa_verbose >= 4) fprintf(stderr, "[0000][%0.4d] Re-allocating (doubling) seqBufQers in %s\n",
                            tid, __func__);
                    int64_t tmp = *wsize_buf_qer;
                    *wsize_buf_qer = seqbuf_grow_capacity(tmp);
                    if (*wsize_buf_qer == SEQBUF_CAPACITY_OVERFLOW)
                        seqbuf_capacity_fatal("seqBufQer", __func__, tmp);
                    assert(*wsize_buf_qer > qerOffset + sp.len2);

                    uint8_t *seqBufQer_ = (uint8_t*)
                        _mm_realloc(seqBufQer, tmp, *wsize_buf_qer, sizeof(uint8_t)); 
                    mmc->seqBufLeftQer[tid*CACHE_LINE] = seqBufQer = seqBufQer_;

                    seqBufQer_ = (uint8_t*)
                        _mm_realloc(mmc->seqBufRightQer[tid*CACHE_LINE], tmp,
                                    *wsize_buf_qer, sizeof(uint8_t)); 
                    mmc->seqBufRightQer[tid*CACHE_LINE] = seqBufQer_;               
                }
            
                if (pcnt >= *wsize_pair)
                {
                    if (bwa_verbose >= 4) fprintf(stderr, "[0000][%0.4d] Re-allocating seqPairs in %s\n", tid, __func__);
                    *wsize_pair += 1024;
                    // realloc into a temp + xassert: a failed grow must not leak
                    // the old block and leave a NULL slot that later writes deref.
                    SeqPair *tmp_aux = (SeqPair *) realloc(mmc->seqPairArrayAux[tid],
                                                        (*wsize_pair + MAX_LINE_LEN) * sizeof(SeqPair));
                    xassert(tmp_aux != NULL, "out of memory: seqPairArrayAux");
                    mmc->seqPairArrayAux[tid] = tmp_aux;
                    SeqPair *tmp_left = (SeqPair *) realloc(mmc->seqPairArrayLeft128[tid],
                                                        (*wsize_pair + MAX_LINE_LEN) * sizeof(SeqPair));
                    xassert(tmp_left != NULL, "out of memory: seqPairArrayLeft128");
                    mmc->seqPairArrayLeft128[tid] = tmp_left;
                    SeqPair *tmp_right = (SeqPair *) realloc(mmc->seqPairArrayRight128[tid],
                                                        (*wsize_pair + MAX_LINE_LEN) * sizeof(SeqPair));
                    xassert(tmp_right != NULL, "out of memory: seqPairArrayRight128");
                    mmc->seqPairArrayRight128[tid] = tmp_right;
                    seqPairArray = mmc->seqPairArrayLeft128[tid];
                    gar = (int32_t*) (mmc->seqPairArrayAux[tid]);
                }

                if (maxRefLen < sp.len1) maxRefLen = sp.len1;
                if (maxQerLen < sp.len2) maxQerLen = sp.len2;
            
                uint8_t *qs = seqBufQer + sp.idq;
                uint8_t *rs = seqBufRef + sp.idr;
                /* RESC-A7: the ref-window copy into seqBufRef is IRREDUCIBLE and
                 * is retained. It cannot be replaced by aliasing mmc->ref_string
                 * (the "phase-0 reads ref_string directly" idea) because:
                 *   (1) mem_sam_pe_batch_run reverses seqBufRef+idr IN PLACE
                 *       (revseq) between the two kswv phases, and ref_string may be
                 *       a PROT_READ shm mapping — writing it would corrupt the
                 *       shared reference / SIGSEGV (mirrored by the ref_rw scratch
                 *       copy in mem_matesw_batch_post's scalar fallback); and
                 *   (2) SeqPair.idr is int32, so seqBufRef+idr cannot address the
                 *       doubled reference (~6.4e9 on hg38) even if we tried.
                 * The per-byte loop is kept as memcpy (same bytes, matches the
                 * _post ref_rw memcpy), which is byte-identical. */
                memcpy(rs, ref, (size_t)sp.len1);
                /* Query: stage the mate bases directly from $ms (no `rev`
                 * scratch). For is_rev, reverse-complement into qs; this is
                 * byte-identical to the removed two-step (build rev, then
                 * qs[l]=rev[l]) because rev[l] == ms[l_ms-1-l]<4 ? 3-ms[l_ms-1-l]
                 * : 4 and sp.len2 == l_ms. */
                if (is_rev)
                    for (int l = 0; l < sp.len2; l++)
                        qs[l] = ms[sp.len2 - 1 - l] < 4 ? 3 - ms[sp.len2 - 1 - l] : 4;
                else
                    memcpy(qs, ms, (size_t)sp.len2);

                /* Tag the enqueued meth pair with a single hypothesis = the
                 * rescued mate's read# chemistry XOR rescue strand
                 * ((mate_meth_ot ^ is_rev) & 1) — the matrix that frees the mate's
                 * conversions at this locus. The kswv OT/OB partition in
                 * mem_sam_pe_batch scores this pair with the matching matrix;
                 * mem_matesw_batch_post's scalar index==-1 fallback derives the
                 * same value. Non-meth keeps the SeqPair default (-1), which the
                 * kernels ignore. The tag is always in {0,1} under --meth, which
                 * the OT/OB partition requires; the assert (and the runtime guard
                 * in mem_sam_pe_batch) documents that invariant so a future
                 * violation fails loud instead of silently dropping a pair. */
                sp.meth_hyp = opt->meth_mode ? (int8_t)((mate_meth_ot ^ is_rev) & 1) : (int8_t)-1;
                assert(!opt->meth_mode || sp.meth_hyp == 0 || sp.meth_hyp == 1);

                /* gar[gcnt+r] points at this rescue's single enqueued regid. */
                if (hi == 0) gar[gcnt + r] = pcnt;
                sp.regid = pcnt;
                if (opt->rescue_kmer) {   /* record narrow offset by regid for _post */
                    if ((int)g_rescue_narrow_off.size() <= pcnt)
                        g_rescue_narrow_off.resize(pcnt + 1024, 0);
                    g_rescue_narrow_off[pcnt] = narrow_ob;
                }
                seqPairArray[pcnt++] = sp;
            }
        }
        // RESC-A7: no `rev` scratch to free (mate query staged directly above).
        // ref aliases ref_string (see bns_fetch_seq_v2 above); no free.
    }
    if (ms2) free(ms2);
    return pcnt;
}

int mem_matesw_batch_post(const mem_opt_t *opt, const bntseq_t *bns,
                          const uint8_t *pac, const mem_pestat_t pes[4],
                          const mem_alnreg_t *a, int l_ms, const uint8_t *ms,
                          mem_alnreg_v *ma, kswr_t **myaln, int32_t gcnt,
                          int32_t *gar, mem_cache *mmc, const char *ms_orig,
                          const int8_t *mat, int mate_meth_ot)
{
    extern int mem_sort_dedup_patch_rev(const mem_opt_t *opt, const bntseq_t *bns,
                                        const uint8_t *pac, uint8_t *query, int n,
                                        mem_alnreg_t *a);
    /* The mate-rescue dedup at the bottom passes bns/pac/query = 0 (dedup-only,
     * no patch SW), so its matrix is unused; opt->mat default is correct there. */
    /* D3 (--meth, A1/PR-6, B3): meth mate rescue is now scored by the BATCHED
     * kswv kernel (via the OT/OB hypothesis partition in mem_sam_pe_batch) by
     * default — _pre enqueues each pair and tags it with sp.meth_hyp ∈ {0,1}.
     * The scalar ksw_align2 path below (index == -1) is the ESCAPE HATCH: it
     * is reached only when BWAMEM3_METH_BATCHED_RESCUE=0 forces the legacy
     * scalar path (gar[gcnt+r] == -1 from _pre), or for any matrix where the
     * kswv object reports needsScalar() (an asymmetric matrix the kernel cannot
     * express, or any freed-cell matrix on a tier without the override). All
     * three --meth-scoring modes are expressible, so on a freed-cell tier the
     * current OT/OB matrices never trigger this. The path is retained as a
     * safety net, not the primary route. */
    const int8_t *sw_mat = mat ? mat : opt->mat;
    #if MATE_SORT    
    extern int mem_dedup_patch(const mem_opt_t *opt, const bntseq_t *bns,
                               const uint8_t *pac, uint8_t *query, int n, mem_alnreg_t *a);
    extern void sort_alnreg_re(int n, mem_alnreg_t* a);
    extern void sort_alnreg_score(int n, mem_alnreg_t* a);
    #endif
    
    int64_t l_pac = bns->l_pac;
    int i, r, skip[4], n = 0, rid = -1;

    /* D3 (--meth): point the mate query at the ORIGINAL bases (ASCII
     * meth_orig_seq, same orientation as `ms`/seq); the per-orientation RC below
     * reverse-complements them exactly as it would the projected mate. Both the
     * batched kswv path (mem_matesw_batch_pre now stages meth_orig_seq too) and
     * the scalar ksw_align2 fallback (index == -1) score these ORIGINAL bases, so
     * the two paths are byte-identical. Aligning the C→T/G→A-projected s->seq
     * instead would be a strict information loss and degrade repeat-mate
     * placement. Allocated AFTER the skip-all early return below so the
     * consistent-pair fast path (the common case) does not leak it. */
    uint8_t *ms2 = NULL;

    for (r = 0; r < 4; ++r) {
        skip[r] = pes[r].failed? 1 : 0;
    }

    for (i = 0; i < ma->n; ++i) { // check which orinentation has been found
        int64_t dist;
        r = mem_infer_dir(l_pac, a->rb, ma->a[i].rb, &dist);
        if (dist >= pes[r].low && dist <= pes[r].high)
            skip[r] = 1;
    }


    if (skip[0] + skip[1] + skip[2] + skip[3] == 4) {
        return 0; // consistent pair exist; no need to perform SW
    }

    if (ms_orig != NULL) {
        ms2 = (uint8_t*) malloc(l_ms);
        xassert(ms2 != NULL, "out of memory: ms2");
        for (int k = 0; k < l_ms; ++k) {
            unsigned char c = (unsigned char) ms_orig[k];
            ms2[k] = nst_nt4_decode(c, 4);
        }
        ms = ms2;
    }

    for (r = 0; r < 4; ++r) {
        int is_rev, is_larger;
        uint8_t *seq = 0, *rev = 0, *ref = 0;
        int64_t rb, re;
        if (skip[r]) {
                continue;
        }
        is_rev = (r>>1 != (r&1)); // whether to reverse complement the mate
        is_larger = !(r>>1); // whether the mate has larger coordinate
        // The mate query `seq` (and its reverse-complement buffer `rev`) is read
        // only by the scalar ksw_align2 fallback (index == -1); the batched kswv
        // path reads its result out of *myaln and never touches `seq`. Defer the
        // malloc + fill into that branch so the common batched path skips it.
        if (!is_rev) {
            rb = is_larger? a->rb + pes[r].low : a->rb - pes[r].high;
            re = (is_larger? a->rb + pes[r].high: a->rb - pes[r].low) + l_ms; // if on the same strand, end position should be larger to make room for the seq length
        } else {
            rb = (is_larger? a->rb + pes[r].low : a->rb - pes[r].high) - l_ms; // similarly on opposite strands
            re = is_larger? a->rb + pes[r].high: a->rb - pes[r].low;
        }
        if (rb < 0) rb = 0;
        if (re > l_pac<<1) re = l_pac<<1;
        // Zero-copy ref slice via bns_fetch_seq_v2 (see mem_matesw_batch_pre
        // for rationale). The scratch arg is unused by v2; pass NULL since
        // mem_matesw_batch_post has no tid in scope to index seqPairArrayAux.
        if (rb < re) ref = bns_fetch_seq_v2(bns, pac, &rb, (rb+re)>>1, &re, &rid,
                                            mmc->ref_string, NULL);

        if (a->rid == rid && re - rb >= opt->min_seed_len) { // no funny things happening
            kswr_t aln;
            mem_alnreg_t b;
            // Default -1 (non-meth anchor stays -1); under --meth this is always
            // overwritten below with the mate read#-derived hypothesis
            // ((mate_meth_ot ^ is_rev) & 1), mirroring the scalar mem_matesw.
            int meth_won_hyp = -1;
            int u8_shift = matesw_u8_shift_for(opt, opt->mat);
            int tmp, xtra = KSW_XSUBO | KSW_XSTART | (matesw_use_u8(l_ms, opt->a, u8_shift)? KSW_XBYTE : 0) | (opt->min_seed_len * opt->a);

            //aln = **myaln;
            //(*myaln)++;
            int index = gar[gcnt + r];
            /* --rescue-skip: _pre found no anchor worth an SW for this orientation
             * and did not enqueue it. Nothing to read back -- `rev` and `ref_rw`
             * are not allocated yet, so this leaves nothing to unwind.
             *
             * `++n` still fires. n counts SW ATTEMPTS, not successes, and its only
             * live effect is gating mem_sort_dedup_patch at the bottom of this
             * loop (the caller discards the return value). Skipping an SW should
             * not silently change how often the dedup runs -- that coupling is
             * incidental, and letting it into the ROC signal would confound the
             * skip gate's measured accuracy cost with a dedup difference. */
            if (index == MATESW_GAR_DECLINED) { ++n; continue; }
            /* --rescue-kmer: apply the same narrowing offset _pre stored for this
             * regid, so aln.tb/te map against the narrowed window start (batched
             * pairs only; index==-1 scalar-fallback pairs were never narrowed). */
            if (opt->rescue_kmer && index >= 0)
                rb += g_rescue_narrow_off[index];
            if (index == -1) {
                // fprintf(stderr, "Re-routing: Encountered -ve index for "
                // "gcnt: %d, look into pre.\n", gcnt + r);
                assert(ref != 0);
                // Build the mate query here (only the scalar path reads it):
                // reverse-complement `ms` into `rev` for the RC orientations,
                // else point straight at `ms`. Freed by the `if (rev) free(rev)`
                // at the bottom of the r-loop (rev stays 0 on the batched path).
                if (is_rev) {
                    rev = (uint8_t*) malloc(l_ms); // reverse complement of $ms
                    xassert(rev != NULL, "out of memory: rev");
                    for (i = 0; i < l_ms; ++i) rev[l_ms - 1 - i] = ms[i] < 4? 3 - ms[i] : 4;
                    seq = rev;
                } else seq = (uint8_t*)ms;
                // ksw_align2 reverses its target argument in place via
                // revseq (see ksw.cpp:375,381). When mmc->ref_string is
                // shm-backed (PROT_READ mmap of /dev/shm/bwaidx-*), that
                // write SIGSEGVs. Copy the slice into a writable scratch
                // buffer before handing it to ksw_align2.
                int64_t ref_len = re - rb;
                uint8_t *ref_rw = (uint8_t*) malloc((size_t)ref_len);
                xassert(ref_rw != NULL, "out of memory: ref_rw");
                memcpy(ref_rw, ref, (size_t)ref_len);
                if (opt->meth_mode && mate_meth_ot >= 0) {
                    /* D3: score the rescued mate under ITS OWN read-number chemistry
                     * (mate_meth_ot) flipped by the rescue strand (this path RC's
                     * the READ). One SW pass with the correct matrix — the matrix
                     * that frees the mate's conversions at its true locus, so the
                     * true-locus cripple that a wrong guess caused cannot happen.
                     * (Replaces the old both-matrices keep-max, which cost 2x SW to
                     * rediscover this same matrix.) */
                    meth_won_hyp = (mate_meth_ot ^ is_rev) & 1;
                    const int8_t *mm = mem_opt_meth_mat(opt, meth_won_hyp);
                    aln = ksw_align2(l_ms, seq, ref_len, ref_rw, 5, mm,
                                     opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, xtra, 0);
                } else {
                    aln = ksw_align2(l_ms, seq, ref_len, ref_rw, 5,
                                     sw_mat, opt->o_del, opt->e_del,
                                     opt->o_ins, opt->e_ins, xtra, 0);
                }
                free(ref_rw);
            }
            else if (opt->meth_mode && mate_meth_ot >= 0) {
                /* Single-hypothesis batched rescue: mem_matesw_batch_pre enqueued
                 * ONE pair at `index`, scored by the kswv OT/OB partition under the
                 * mate's read# chemistry (XOR rescue strand). Read it and record the
                 * genome-strand hypothesis for the XG/XM output layer — identical to
                 * the scalar single-matrix path (index==-1 branch above). */
                aln = *(*myaln + index);
                meth_won_hyp = (mate_meth_ot ^ is_rev) & 1;
            }
            else
                aln = *(*myaln + index);

            memset(&b, 0, sizeof(mem_alnreg_t));
            if (aln.score >= opt->min_seed_len && aln.qb >= 0 && aln.qe < l_ms) { // something goes wrong if aln.qb < 0, or if aln.qe runs past the read
                b.rid = a->rid;
                b.is_alt = a->is_alt;
                /* D3 (--meth): rescued mate hypothesis = the mate's own read#
                 * chemistry XOR rescue strand ((mate_meth_ot ^ is_rev) & 1,
                 * meth_won_hyp above) so the output layer (XG/XM) sources the
                 * right strand. -1 anchor stays -1. The score came from the batched
                 * kswv result (or the scalar ksw_align2 index==-1 fallback) under
                 * that same single matrix — see above.
                 * Coordinates are already ORIGINAL (l_pac is the original l_pac via
                 * the original bns), so the 6a coordinate fix holds. */
                b.meth_hypothesis = meth_won_hyp;
                b.qb = is_rev? l_ms - (aln.qe + 1) : aln.qb;
                b.qe = is_rev? l_ms - aln.qb : aln.qe + 1;
                b.rb = is_rev? (l_pac<<1) - (rb + aln.te + 1) : rb + aln.tb;
                b.re = is_rev? (l_pac<<1) - (rb + aln.tb) : rb + aln.te + 1;
                b.score = aln.score;
                b.csub = aln.score2;
                b.secondary = -1;
                b.seedcov = (b.re - b.rb < b.qe - b.qb? b.re - b.rb : b.qe - b.qb) >> 1;
                b.chain_n_hits = 1; // mate-rescue has no SMEM evidence; treat as unique anchor

                kv_push(mem_alnreg_t, *ma, b); // make room for a new element

                #if !MATE_SORT

                // move b s.t. ma is sorted
                for (i = 0; i < ma->n - 1; ++i) // find the insertion point
                    if (ma->a[i].score < b.score) break;
                tmp = i;
                for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
                ma->a[i] = b;

                #else
                int resort = 0;
                // move b s.t. ma is sorted
                for (i = 0; i < ma->n - 1; ++i) { // find the insertion point
                    if (ma->a[i].re == b.re) {
                        resort = 1;
                        break;
                    }
                    if (ma->a[i].re > b.re) {
                        break;
                    }
                }
                if (resort) {
                    // Don't know where to put this alignment. So let the scores decide
                    sort_alnreg_score(ma->n - 1, ma->a);
                    for (i = 0; i < ma->n - 1; ++i) { // find the insertion point
                        if (ma->a[i].score < b.score) {
                            break;
                        }
                    }
                    tmp = i;
                    for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
                    ma->a[i] = b;
                    // Now we can sort based on end position
                    sort_alnreg_re(ma->n, ma->a);
                }
                else {
                    tmp = i;
                    for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
                    ma->a[i] = b;
                }
                #endif
                tprof[PE26][0] ++;
            }
            ++n;
        }
        #if !MATE_SORT
        if (n) ma->n = mem_sort_dedup_patch(opt, 0, 0, 0, ma->n, ma->a);
        #else
        if (n) ma->n = mem_dedup_patch(opt, 0, 0, 0, ma->n, ma->a);
        #endif

        if (rev) free(rev);
        // ref aliases ref_string (see bns_fetch_seq_v2 above); no free.
    }
    if (ms2) free(ms2); // D3 (--meth): original-mate 2-bit scratch
    return n;
}

