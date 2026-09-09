/* Whole-read-pair memoization ("--dedup-reads") -- Phase 1: CLI + config +
 * fingerprint pre-pass + net-cycles controller, all wired but not yet consumed.
 * See read_memo.h for the design. This module deliberately mirrors the shipped
 * extension-DP dedup controller in bwamem.cpp ("#415"): same off|on|auto surface,
 * same self-calibrating net-cycles controller, same ends-only word fingerprint
 * with full-byte verify. Byte-identity is structural in Phase 1 (nothing reads
 * role[]/rep_pair[]). */

#include "read_memo.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "bwa.h"          /* bseq1_t */
#include "utils.h"        /* xassert (survives a hypothetical -DNDEBUG build) */
#include "robin_hood.h"

extern int bwa_verbose;   /* from utils */

/* ------------------------------------------------------------------------- *
 * Config: --dedup-reads / BWAMEM3_DEDUP_READS{,_Z,_REPROBE}
 *   (clone of mem_dedup_configure / bsw_parse_dedup_mode, bwamem.cpp:4436-4520)
 * ------------------------------------------------------------------------- */

static int readmemo_parse_mode(const char *e)
{
    if (!e) return -1;
    if (!strcmp(e, "0") || !strcmp(e, "off"))               return READMEMO_OFF;
    if (!strcmp(e, "1") || !strcmp(e, "on"))                return READMEMO_ON;
    if (!strcmp(e, "auto") || !strcmp(e, "2"))              return READMEMO_AUTO;
    return -1;
}

struct ReadMemoCfg { int mode; double z; int64_t reprobe_pairs; };
/* mode -1 = unresolved; default z=2.0; default re-probe every 25M pairs. */
static ReadMemoCfg g_cfg = { -1, 2.0, 25000000 };

static double readmemo_parse_pos_double(const char *v, const char *what)
{
    char *end = NULL; errno = 0;
    const double x = strtod(v, &end);
    if (end == v || end == NULL || *end != '\0' || errno == ERANGE || !std::isfinite(x) || x <= 0.0) {
        fprintf(stderr, "ERROR: %s: must be a finite number > 0, got '%s'\n", what, v);
        exit(1);
    }
    return x;
}

static int64_t readmemo_parse_nonneg_ll(const char *v, const char *what)
{
    char *end = NULL; errno = 0;
    const long long x = strtoll(v, &end, 10);
    if (end == v || end == NULL || *end != '\0' || errno == ERANGE || x < 0) {
        fprintf(stderr, "ERROR: %s: must be a non-negative integer (0 = off), got '%s'\n", what, v);
        exit(1);
    }
    return (int64_t)x;
}

void mem_dedup_reads_configure(const char *mode_arg)
{
    /* A non-empty --dedup-reads wins outright; otherwise consult the env,
     * distinguishing "unset" (default) from "set but empty" (fatal). An empty CLI
     * value is rejected by the getopt handler. Mirrors mem_dedup_configure. */
    if (mode_arg && *mode_arg) {
        const int v = readmemo_parse_mode(mode_arg);
        if (v < 0) { fprintf(stderr, "ERROR: --dedup-reads: expected off|on|auto, got '%s'\n", mode_arg); exit(1); }
        g_cfg.mode = v;
    } else {
        const char *m = getenv("BWAMEM3_DEDUP_READS");
        if (m == NULL) g_cfg.mode = READMEMO_AUTO;      /* unset -> default */
        else {
            const int v = readmemo_parse_mode(m);        /* "" -> -1 -> fatal */
            if (v < 0) { fprintf(stderr, "ERROR: BWAMEM3_DEDUP_READS: expected off|on|auto, got '%s'\n", m); exit(1); }
            g_cfg.mode = v;
        }
    }
    if (const char *z = getenv("BWAMEM3_DEDUP_READS_Z"))
        g_cfg.z = readmemo_parse_pos_double(z, "BWAMEM3_DEDUP_READS_Z");
    if (const char *r = getenv("BWAMEM3_DEDUP_READS_REPROBE"))
        g_cfg.reprobe_pairs = readmemo_parse_nonneg_ll(r, "BWAMEM3_DEDUP_READS_REPROBE");
}

static const ReadMemoCfg &readmemo_cfg(void)
{
    static std::once_flag once;
    std::call_once(once, []() { if (g_cfg.mode < 0) mem_dedup_reads_configure(NULL); });
    return g_cfg;
}

int read_memo_mode(void) { return readmemo_cfg().mode; }

/* ------------------------------------------------------------------------- *
 * Fingerprint: ends-only word hash (clone of bsw_hash_fold/bsw_job_hash,
 *   bwamem.cpp:4665-4688). Fingerprint collisions are harmless -- every hit is
 *   byte-verified before a pair is marked DUP.
 * ------------------------------------------------------------------------- */

static inline void readmemo_hash_fold(uint64_t &h, const uint8_t *s, int len)
{
    const int LIMIT = 32;
    auto mix = [&](const uint8_t *p, int n) {
        int k = 0;
        for (; k + 8 <= n; k += 8) { uint64_t v; memcpy(&v, p + k, 8); h ^= v; h *= 1099511628211ULL; }
        if (k < n) { uint64_t v = 0; for (int i = 0; k + i < n; i++) v |= (uint64_t)p[k + i] << (8 * i); h ^= v; h *= 1099511628211ULL; }
    };
    if (len <= 2 * LIMIT) mix(s, len);
    else { mix(s, LIMIT); mix(s + len - LIMIT, LIMIT); }
}

static inline uint64_t readmemo_pair_hash(const bseq1_t *r1, const bseq1_t *r2)
{
    uint64_t h = 1469598103934665603ULL;                 /* FNV-1a offset basis */
    h ^= ((uint64_t)(uint32_t)r1->l_seq << 32) | (uint32_t)r2->l_seq;
    h *= 1099511628211ULL;
    readmemo_hash_fold(h, (const uint8_t *)r1->seq, r1->l_seq);
    readmemo_hash_fold(h, (const uint8_t *)r2->seq, r2->l_seq);
    return h;
}

static inline bool readmemo_pair_eq(const bseq1_t *a1, const bseq1_t *a2,
                                    const bseq1_t *b1, const bseq1_t *b2)
{
    return a1->l_seq == b1->l_seq && a2->l_seq == b2->l_seq
        && memcmp(a1->seq, b1->seq, (size_t)a1->l_seq) == 0
        && memcmp(a2->seq, b2->seq, (size_t)a2->l_seq) == 0;
}

/* ------------------------------------------------------------------------- *
 * Pre-pass. Serial, single-threaded (called from the pipeline's main loop
 *   between chunk load and align). Intrusive-chain + byte-verify: the map keys a
 *   fingerprint to a chain head (a pair index); each pair links to the previous
 *   same-fingerprint pair via chain_next[], so a collision walks the chain and
 *   memcmps rather than splitting a duplicate set.
 * ------------------------------------------------------------------------- */

/* PRECONDITION: read_memo_prepass runs SINGLE-THREADED. These module statics
 * (and the file-static read_memo_state in bwamem.cpp) carry no synchronization;
 * they are safe only because the pre-pass runs on the pipeline's main thread
 * between chunk load and align -- it is NOT under kt_for. A future caller that
 * runs it concurrently must shard this state per thread. */
static robin_hood::unordered_flat_map<uint64_t, int32_t> g_map;
static int32_t *g_chain_next = NULL;                     /* module scratch, grown in place */
static int64_t  g_chain_cap  = 0;

static void readmemo_state_grow(read_memo_state *st, int npairs)
{
    if (npairs <= st->cap) return;
    int64_t ncap = st->cap ? st->cap : 1024;
    while (ncap < npairs) ncap <<= 1;
    /* Capture each realloc into a temp and xassert before assigning: a bare
     * `p = realloc(p, ...)` leaks the original block and NULLs the live pointer
     * on OOM. xassert (not assert) so the guard survives a -DNDEBUG build. */
    uint8_t *role_    = (uint8_t *) realloc(st->role,     (size_t)ncap * sizeof(uint8_t));
    int32_t *reppair_ = (int32_t *) realloc(st->rep_pair, (size_t)ncap * sizeof(int32_t));
    xassert(role_ != NULL && reppair_ != NULL, "out of memory: read_memo state");
    st->role     = role_;
    st->rep_pair = reppair_;
    st->cap      = ncap;
    if (ncap > g_chain_cap) {
        int32_t *next_ = (int32_t *) realloc(g_chain_next, (size_t)ncap * sizeof(int32_t));
        xassert(next_ != NULL, "out of memory: read_memo chain");
        g_chain_next = next_;
        g_chain_cap  = ncap;
    }
}

/* Release a read_memo_state's grown-in-place buffers. Call at shutdown, after
 * all worker threads have joined, so these one-shot allocations are not left
 * still-reachable at process exit. Safe on a zeroed/partial state and idempotent. */
void read_memo_state_free(read_memo_state *st)
{
    if (st == NULL) return;
    free(st->role);
    free(st->rep_pair);
    st->role     = NULL;
    st->rep_pair = NULL;
    st->npairs   = 0;
    st->cap      = 0;
}

/* Release the module-scope chain scratch (g_chain_next). Pairs with
 * read_memo_state_free; same shutdown-only, idempotent contract. */
void read_memo_teardown(void)
{
    free(g_chain_next);
    g_chain_next = NULL;
    g_chain_cap  = 0;
}

read_memo_result read_memo_prepass(const mem_opt_t * /*opt*/, const bseq1_t *seqs,
                                   int n, read_memo_state *st)
{
    /* PE-only, even n: the sole caller (bwamem.cpp) gates on
     * (flag & MEM_F_PE) && n>0 && (n&1)==0. Enforce it fail-closed -- an odd or
     * zero n would silently drop the tail read and leave role[npairs] stale for
     * the bwamem consumers (worker_bwt_memo/worker_copy_regs), a silent wrong
     * regs copy rather than a crash. xassert (not assert) so the guard survives
     * a -DNDEBUG build. */
    xassert(n > 0 && (n & 1) == 0, "read_memo_prepass requires n>0 and even n");
    const auto t0 = std::chrono::steady_clock::now();
    const int npairs = n >> 1;
    readmemo_state_grow(st, npairs);
    st->npairs = npairs;
    g_map.clear();                     /* retains capacity; reserve is a no-op after chunk 1 */
    g_map.reserve((size_t)npairs);

    int64_t dup = 0;
    for (int i = 0; i < npairs; i++) {
        const bseq1_t *r1 = &seqs[2 * i], *r2 = &seqs[2 * i + 1];
        const uint64_t fp = readmemo_pair_hash(r1, r2);
        int32_t rep = -1;
        auto it = g_map.find(fp);
        if (it != g_map.end()) {
            for (int32_t c = it->second; c >= 0; c = g_chain_next[c]) {
                if (readmemo_pair_eq(&seqs[2 * c], &seqs[2 * c + 1], r1, r2)) { rep = c; break; }
            }
        }
        if (rep >= 0) {
            st->role[i] = READ_MEMO_ROLE_DUP;
            st->rep_pair[i] = rep;
            dup++;
        } else {
            st->role[i] = READ_MEMO_ROLE_REP;
            st->rep_pair[i] = -1;
            g_chain_next[i] = (it != g_map.end()) ? it->second : -1;
            g_map[fp] = i;                               /* new chain head */
        }
    }

    const uint64_t probe_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t0).count();
    return read_memo_result{ (int64_t)npairs, dup, probe_ns };
}

/* ------------------------------------------------------------------------- *
 * Two-sample A/B controller. The observation unit is one align invocation; the
 * quantity is that chunk's REALIZED per-pair total cost:
 *
 *   cost_per_pair = (probe_ns + align_ns + (armed ? copy_ns : 0)) / pairs
 *
 * For an ARMED chunk align_ns already reflects the memo (REP-only align, minus
 * the skipped duplicates, PLUS the compaction overhead), and copy_ns is the
 * regs copy pass -- so EVERY overhead the memo adds is charged. For an UNARMED
 * chunk it is the full align. While auto is measuring we alternate the arm
 * decision (read_memo_should_arm) so both samples accumulate, then latch on the
 * MEASURED difference of the two means -- not a benefit model that could omit
 * the copy/compaction cost (which is exactly how a dup-rate-blind model would
 * wrongly latch ON on cheap reads, where skipping cheap alignments saves less
 * than the memo's own overhead).
 *
 *   delta = mean(cost | ON) - mean(cost | OFF)      [< 0 => memoizing is cheaper]
 *
 * Chunks are huge, so the per-arm CLT floor is small (3) rather than #415's 24;
 * a decision needs NET_MIN_OBS of EACH arm (so ~2*NET_MIN_OBS chunks).
 * ------------------------------------------------------------------------- */

enum { NET_MEASURING = 0, NET_LATCH_ON = 1, NET_LATCH_OFF = 2 };
static std::atomic<int>     g_net_state{NET_MEASURING};
static const uint64_t       NET_MIN_OBS      = 3;         /* per-arm CLT floor       */
static const uint64_t       NET_MAX_PAIRS    = 20000000;  /* give up measuring past this */
static const uint64_t       NET_PROBE_MAX_PAIRS = 10000000;
static std::atomic<int>     g_net_incumbent{0};
static std::atomic<int64_t> g_net_since_latch{0};

static std::mutex g_net_mtx;
/* Welford accumulators of per-pair cost (ns), split by arm state. */
struct ReadMemoAcc { uint64_t n; double mean, m2; };
static ReadMemoAcc g_on  = {0, 0.0, 0.0};
static ReadMemoAcc g_off = {0, 0.0, 0.0};
static uint64_t    g_meas_pairs = 0;   /* pairs measured this window (for the cap) */
static uint64_t    g_meas_arm   = 0;   /* arm-alternation counter while measuring  */

/* STATS accounting (BWAMEM3_DEDUP_READS_STATS=1). */
static std::atomic<uint64_t> g_stat_pairs{0}, g_stat_dups{0};
static inline bool readmemo_stats_on(void)
{
    static const bool on = []() { const char *e = getenv("BWAMEM3_DEDUP_READS_STATS"); return e && *e && !(e[0]=='0'&&e[1]=='\0'); }();
    return on;
}

int read_memo_verify(void)
{
    static const int on = []() {
        const char *e = getenv("BWAMEM3_DEDUP_READS_VERIFY");
        return (e && *e && !(e[0] == '0' && e[1] == '\0')) ? 1 : 0;
    }();
    return on;
}

static void readmemo_reprobe(int64_t pairs)
{
    const int64_t period = readmemo_cfg().reprobe_pairs;
    if (period <= 0) return;
    if (g_net_since_latch.fetch_add(pairs, std::memory_order_relaxed) + pairs < period) return;
    std::lock_guard<std::mutex> lk(g_net_mtx);
    if (g_net_state.load(std::memory_order_relaxed) == NET_MEASURING) return;
    if (g_net_since_latch.load(std::memory_order_relaxed) < period) return;
    g_on = ReadMemoAcc{0, 0.0, 0.0}; g_off = ReadMemoAcc{0, 0.0, 0.0};
    g_meas_pairs = 0; g_meas_arm = 0;
    g_net_since_latch.store(0, std::memory_order_relaxed);
    if (bwa_verbose >= 3)
        fprintf(stderr, "[dedup-reads] re-probe (incumbent %s)\n",
                g_net_state.load() == NET_LATCH_ON ? "ON" : "OFF");
    g_net_state.store(NET_MEASURING, std::memory_order_relaxed);
}

int read_memo_should_arm(int64_t dup_pairs)
{
    const int mode = readmemo_cfg().mode;
    if (mode == READMEMO_OFF) return 0;
    if (dup_pairs <= 0)       return 0;      /* nothing to memoize this chunk */
    if (mode == READMEMO_ON)  return 1;
    /* auto: honor the latch; while measuring, alternate so the A/B gets both
     * armed and unarmed samples on comparable (dup-bearing) chunks. */
    const int st = g_net_state.load(std::memory_order_relaxed);
    if (st == NET_LATCH_ON)  return 1;
    if (st == NET_LATCH_OFF) return 0;
    std::lock_guard<std::mutex> lk(g_net_mtx);
    const int st2 = g_net_state.load(std::memory_order_relaxed);
    if (st2 != NET_MEASURING) return st2 == NET_LATCH_ON ? 1 : 0;
    return ((g_meas_arm++ & 1ULL) == 0ULL) ? 1 : 0;   /* arm even, unarm odd */
}

void read_memo_controller_observe(const read_memo_result &r, uint64_t align_ns,
                                  uint64_t copy_ns, bool armed)
{
    if (readmemo_stats_on()) {
        g_stat_pairs.fetch_add((uint64_t)r.pairs, std::memory_order_relaxed);
        g_stat_dups.fetch_add((uint64_t)r.dup_pairs, std::memory_order_relaxed);
    }
    if (r.pairs <= 0) return;
    if (readmemo_cfg().mode != READMEMO_AUTO) return;   /* forced on/off: controller inert */

    /* Advance the work-based re-probe counter first; this may reset a latched
     * state back to MEASURING within this call. */
    readmemo_reprobe(r.pairs);

    std::lock_guard<std::mutex> lk(g_net_mtx);
    if (g_net_state.load(std::memory_order_relaxed) != NET_MEASURING) return;

    /* Only dup-bearing chunks inform the A/B: a dup-free chunk is never armed
     * (read_memo_should_arm returns 0 for it), so folding it into the OFF arm
     * would blend "alternated-OFF" with "structurally-can't-arm" samples and bias
     * the comparison. Excluding them also means an all-unique workload (WGS/exome)
     * accumulates nothing, stays MEASURING (== auto OFF), and never arms -- zero
     * measuring-phase exposure there. */
    if (r.dup_pairs <= 0) return;

    /* Realized per-pair total cost of THIS chunk (see the header comment): the
     * pre-pass, the align phase (armed => REP-only + compaction; unarmed => full),
     * and the copy pass (armed only). Route it to the matching A/B accumulator. */
    const double cost = (double)(r.probe_ns + align_ns + (armed ? copy_ns : 0))
                        / (double)r.pairs;
    ReadMemoAcc &acc = armed ? g_on : g_off;
    acc.n += 1;
    const double d = cost - acc.mean;
    acc.mean += d / (double)acc.n;
    acc.m2   += d * (cost - acc.mean);
    g_meas_pairs += (uint64_t)r.pairs;

    const int incumbent = g_net_incumbent.load(std::memory_order_relaxed);
    int decision = 0; const char *why = "";
    if (g_on.n >= NET_MIN_OBS && g_off.n >= NET_MIN_OBS) {
        const double var_on  = g_on.n  > 1 ? g_on.m2  / (double)(g_on.n  - 1) : 0.0;
        const double var_off = g_off.n > 1 ? g_off.m2 / (double)(g_off.n - 1) : 0.0;
        const double se = sqrt(var_on / (double)g_on.n + var_off / (double)g_off.n);
        const double delta = g_on.mean - g_off.mean;   /* < 0 => armed cheaper */
        const double z = se > 0.0 ? delta / se : (delta < 0.0 ? -1e9 : (delta > 0.0 ? 1e9 : 0.0));
        const int    fav = (delta < 0.0) ? NET_LATCH_ON : NET_LATCH_OFF;
        const double zc  = readmemo_cfg().z;
        if (incumbent == 0) {
            if      (fabs(z) >= zc)                 { decision = fav;          why = "measured"; }
            else if (g_meas_pairs >= NET_MAX_PAIRS) { decision = NET_LATCH_OFF; why = "break-even, default off"; }
        } else if (fav == incumbent) {
            if      (fabs(z) >= zc)                        { decision = incumbent; why = "re-probe confirmed"; }
            else if (g_meas_pairs >= NET_PROBE_MAX_PAIRS)  { decision = incumbent; why = "re-probe inconclusive, kept"; }
        } else {
            if      (fabs(z) >= zc + 1.0)                  { decision = fav;       why = "re-probe FLIPPED"; }
            else if (g_meas_pairs >= NET_PROBE_MAX_PAIRS)  { decision = incumbent; why = "below flip margin, kept"; }
        }
    } else if (g_meas_pairs >= NET_MAX_PAIRS) {
        decision = incumbent ? incumbent : NET_LATCH_OFF; why = "insufficient A/B, default off";
    }
    if (decision) {
        g_net_incumbent.store(decision, std::memory_order_relaxed);
        g_net_since_latch.store(0, std::memory_order_relaxed);
        g_net_state.store(decision, std::memory_order_relaxed);
        if (bwa_verbose >= 3)
            fprintf(stderr, "[dedup-reads] %s: on=%.1f off=%.1f ns/pair (n=%llu/%llu) -> dedup %s\n",
                    why, g_on.mean, g_off.mean,
                    (unsigned long long)g_on.n, (unsigned long long)g_off.n,
                    decision == NET_LATCH_ON ? "ON" : "OFF");
    }
}

void read_memo_reset_for_testing(void)
{
    std::lock_guard<std::mutex> lk(g_net_mtx);
    g_net_state.store(NET_MEASURING, std::memory_order_relaxed);
    g_net_incumbent.store(0, std::memory_order_relaxed);
    g_net_since_latch.store(0, std::memory_order_relaxed);
    g_on = ReadMemoAcc{0, 0.0, 0.0}; g_off = ReadMemoAcc{0, 0.0, 0.0};
    g_meas_pairs = 0; g_meas_arm = 0;
    g_stat_pairs.store(0, std::memory_order_relaxed);
    g_stat_dups.store(0, std::memory_order_relaxed);
}

int read_memo_active(void)
{
    const int mode = readmemo_cfg().mode;
    if (mode == READMEMO_OFF) return READMEMO_OFF;
    if (mode == READMEMO_ON)  return READMEMO_ON;
    return g_net_state.load(std::memory_order_relaxed) == NET_LATCH_ON ? READMEMO_ON : READMEMO_OFF;
}

/* STATS dump at exit (clone of the #415 dumper, bwamem.cpp:4689-4710). */
namespace {
struct ReadMemoStatsDump {
    ~ReadMemoStatsDump() {
        if (!readmemo_stats_on()) return;
        const uint64_t p = g_stat_pairs.load(), d = g_stat_dups.load();
        if (p == 0) return;
        fprintf(stderr, "[dedup-reads-stats] pairs=%llu dup=%llu rate=%.2f%% final=%s\n",
                (unsigned long long)p, (unsigned long long)d, 100.0 * (double)d / (double)p,
                read_memo_active() == READMEMO_ON ? "ON" : "OFF");
    }
};
static ReadMemoStatsDump g_readmemo_stats_dump;
}
