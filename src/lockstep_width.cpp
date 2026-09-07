#include "lockstep_width.h"

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <mutex>

#if defined(__linux__)
#include <sched.h>   /* sched_getaffinity, CPU_* -- the effective-affinity core count */
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

/* Runtime phase-2 lockstep width. Defaults to the compile-time floor so a
 * binary that never calls the startup probe is identical to the old constant. */
int32_t g_smem_lockstep_n = SMEM_LOCKSTEP_N;

int32_t bwa3_lockstep_width_from_probe(int32_t raw_mlp) {
    if (raw_mlp < SMEM_LOCKSTEP_N)     return SMEM_LOCKSTEP_N;      /* floor (also covers <=0) */
    if (raw_mlp > SMEM_LOCKSTEP_N_MAX) return SMEM_LOCKSTEP_N_MAX;  /* ceiling */
    return raw_mlp;
}

int32_t bwa3_lockstep_width_parse_env(const char *env) {
    if (env == NULL || env[0] == '\0') return 0;  /* unset/empty: no override */

    errno = 0;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || *end != '\0') return -1;  /* not a clean integer */
    if (errno == ERANGE)            return -1;  /* overflowed the parse */
    if (v < 1)                      return -1;  /* non-positive */
    if (v > SMEM_LOCKSTEP_N_MAX)    return SMEM_LOCKSTEP_N_MAX;  /* clamp large to ceiling */
    return (int32_t)v;
}

int bwa3_lockstep_probe_enabled(const char *env) {
    /* Truthy opt-in: enable only for a present, non-empty value that is not "0".
     * Unset (NULL), empty, and "0" all leave the probe off, so neither an empty
     * value nor an explicit =0 can silently pay the startup sweep's cost. */
    return env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
}

/* ---- startup memory-level-parallelism probe --------------------------------
 *
 * A pointer chase whose next address is derived from the value just loaded is
 * a pure chain of dependent loads: the prefetcher cannot run ahead and every
 * cache-missing hop is a serialized memory stall. Running k such chains
 * interleaved lets the core keep up to k misses outstanding at once; per-access
 * latency falls as k rises while the core still has miss slots to spare and
 * flattens once it does not. The smallest k at (near) that floor is the core's
 * usable memory-level parallelism -- exactly how many reads' FM-index walks the
 * lockstep seeding driver can profitably keep in flight.
 *
 * The chase walks the FM-index's own cp_occ checkpoint array, NOT a scratch
 * buffer. A fresh allocation measures whatever cache level it happens to fit
 * in -- on a large system-level cache it reports near-cache latency and picks
 * the widest candidate for the wrong reason. cp_occ is gigabytes, already
 * resident, carries the real TLB behaviour, and is the array seeding actually
 * hammers, so probing it measures the thing that matters and costs no memory.
 */

static int64_t bwa3_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int32_t bwa3_measure_mlp(const void *base, int64_t n_blocks,
                         size_t stride, size_t word_off) {
    /* Too small to measure DRAM behaviour (e.g. a tiny test index): let the
     * caller fall back to the compiled default. */
    if (base == NULL || n_blocks < 4096) return 0;

    const char    *bp  = (const char *)base;
    const uint64_t MIX = 0x9E3779B97F4A7C15ULL;  /* golden-ratio odd multiplier */
    const int32_t  max_c = SMEM_LOCKSTEP_N_MAX;
    const int64_t  accesses = 200000;            /* equal work per candidate */

    /* Candidates span the plausible MLP range; sub-floor values still get
     * measured (a low knee just floors to the default in from_probe). */
    static const int32_t sweep[] = {8, 12, 16, 24, 32, 48, 64};
    const size_t n_sweep = sizeof(sweep) / sizeof(sweep[0]);
    double  ns[sizeof(sweep) / sizeof(sweep[0])] = {0.0};  /* ns/access; lower better */

    uint64_t cur[SMEM_LOCKSTEP_N_MAX];
    volatile uint64_t sink = 0;

    /* One hop: read a 64-bit word from block cur, mix with the index so the
     * value alone chooses the next block -- a genuine data dependency. */
    #define BWA3_HOP(idx) do { \
        uint64_t v = *(const uint64_t *)(bp + ((idx) % (uint64_t)n_blocks) * stride + word_off) ^ (idx); \
        (idx) = (v * MIX >> 24) % (uint64_t)n_blocks; \
    } while (0)

    size_t n_run = 0;
    for (size_t si = 0; si < n_sweep; si++) {
        int32_t k = sweep[si];
        if (k > max_c) break;
        const int64_t steps = accesses / k;
        /* Best-of-2: the faster timing rejects a scheduling hiccup or migration
         * that would otherwise inflate a single point. */
        int64_t best_dt = 0;
        for (int rep = 0; rep < 2; rep++) {
            for (int32_t j = 0; j < k; j++)
                cur[j] = (uint64_t)((int64_t)j * (n_blocks / k));
            /* Untimed warm-up so every width starts from the same TLB/cache
             * state and does not eat first-touch cost. */
            for (int w = 0; w < 64; w++)
                for (int32_t j = 0; j < k; j++) BWA3_HOP(cur[j]);
            const int64_t t0 = bwa3_now_ns();
            for (int64_t s = 0; s < steps; s++)
                for (int32_t j = 0; j < k; j++) BWA3_HOP(cur[j]);
            const int64_t t1 = bwa3_now_ns();
            for (int32_t j = 0; j < k; j++) sink += cur[j];  /* defeat DCE */
            const int64_t dt = (t1 - t0) > 0 ? (t1 - t0) : 1;
            if (best_dt == 0 || dt < best_dt) best_dt = dt;
        }
        ns[si] = (double)best_dt / (double)(steps * k);
        n_run = si + 1;
    }
    (void)sink;
    #undef BWA3_HOP

    if (n_run == 0) return 0;

    /* Best (lowest) per-access latency, then the knee = smallest candidate
     * within 5% of it. A flat top thus picks the cheaper (fewer-slot) width. */
    double best_ns = ns[0];
    for (size_t si = 1; si < n_run; si++)
        if (ns[si] < best_ns) best_ns = ns[si];
    int32_t knee = sweep[n_run - 1];
    for (size_t si = 0; si < n_run; si++)
        if (ns[si] <= best_ns * 1.05) { knee = sweep[si]; break; }

    /* BWA3_MLP_DEBUG: dump the sweep so the knee choice is auditable. */
    if (getenv("BWA3_MLP_DEBUG") != NULL) {
        fprintf(stderr, "[mlp] ns/access chasing cp_occ:");
        for (size_t si = 0; si < n_run; si++)
            fprintf(stderr, " %d=%.2f", sweep[si], ns[si]);
        fprintf(stderr, "  knee=%d\n", knee);
    }
    return knee;
}

void bwa3_init_smem_lockstep_width(const void *base, int64_t n_blocks,
                                   size_t stride, size_t word_off) {
    /* One-time resolution. The sole intended caller (fastmap.cpp) runs this
     * once before the seeding workers spawn, so today it is single-threaded;
     * std::call_once matches the repo's other once-init sites
     * (simd_dispatch.cpp, read_memo.cpp) and keeps it correct if that ever
     * changes. */
    static std::once_flag once;
    std::call_once(once, [&]() {
    const char *env = getenv("BWA3_SMEM_LOCKSTEP_N");
    const int32_t pinned = bwa3_lockstep_width_parse_env(env);
    if (pinned > 0) {
        /* A valid override pins the width explicitly, so gated/CI runs pay no
         * measurement cost and stay deterministic. */
        g_smem_lockstep_n = pinned;
    } else {
        if (pinned < 0)
            /* Set but invalid: do NOT silently accept it. Diagnose, then fall
             * through to the same resolution as an unset value below. */
            fprintf(stderr,
                    "ERROR: BWA3_SMEM_LOCKSTEP_N=\"%s\" is not a positive integer "
                    "(<= %d); ignoring it (resolving as if unset).\n",
                    env, SMEM_LOCKSTEP_N_MAX);
        /* No explicit pin. The startup MLP probe is OPT-IN, not the default:
         * measured across architectures (x86/arm), core counts, and page sizes,
         * the lockstep width is a flat ~1% knob end-to-end, while the probe
         * itself costs a non-trivial single-threaded startup pass. So by default
         * we keep the compile-time SMEM_LOCKSTEP_N (the value the probe converges
         * to on all benchmarked hardware -- output is unchanged, the probe cost
         * is not paid). BWA3_SMEM_LOCKSTEP_PROBE=1 opts back into the probe to
         * self-calibrate on new/untested hardware. Truthy opt-in
         * (bwa3_lockstep_probe_enabled): any value other than unset, empty, or
         * "0" enables it, so unset it (or set it to 0) to disable. */
        if (bwa3_lockstep_probe_enabled(getenv("BWA3_SMEM_LOCKSTEP_PROBE")))
            g_smem_lockstep_n = bwa3_lockstep_width_from_probe(
                bwa3_measure_mlp(base, n_blocks, stride, word_off));
        /* else: g_smem_lockstep_n keeps its compile-time SMEM_LOCKSTEP_N init. */
    }
    });
}

/* ---- third-pass bwtseed lockstep: on/off policy (see lockstep_width.h) ----- */

/* The compile-time platform default for the third-pass bwtseed lockstep, and the
 * single place it is written down: arm64 has no SMT so keep it on, elsewhere the
 * rule decides and this is the fallback (off). Used both to initialize
 * g_bwtseed_lockstep and, in bwa3_init_bwtseed_lockstep, as the default_on fed to
 * the resolver -- so a later unset run always falls back here, never to a leaked
 * prior value. */
#if defined(__aarch64__)
#define BWA3_BWTSEED_LOCKSTEP_DEFAULT 1
#else
#define BWA3_BWTSEED_LOCKSTEP_DEFAULT 0
#endif

int32_t g_bwtseed_lockstep = BWA3_BWTSEED_LOCKSTEP_DEFAULT;

/* Does the sysfs CPU list `list` ("0,4", "0-1", "0,4,8,12", ...) name any CPU the
 * process may run on? A CPU c is allowed when c < allowed_len and allowed[c] != 0.
 * With allowed == NULL every well-formed list is allowed (the host-wide count).
 * Sets *ok = 0 on a malformed list so the caller can treat the whole count as
 * unknown, mirroring the online-list parser's rigor. Copies `list` before
 * tokenizing so the caller's buffer is left intact. */
static int list_any_allowed(const char *list, const unsigned char *allowed,
                            int32_t allowed_len, int *ok) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", list);
    int any = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",\n", &save); tok != NULL; tok = strtok_r(NULL, ",\n", &save)) {
        long lo, hi;
        char *end = NULL;
        lo = strtol(tok, &end, 10);
        if (end == tok) { *ok = 0; return 0; }
        if (*end == '-') {
            char *he = NULL;
            hi = strtol(end + 1, &he, 10);
            if (he == end + 1) { *ok = 0; return 0; }
            end = he;
        } else {
            hi = lo;
        }
        while (*end == ' ' || *end == '\t') end++;
        if (*end != '\0') { *ok = 0; return 0; }
        if (hi < lo || lo < 0) { *ok = 0; return 0; }
        if (allowed == NULL) { any = 1; continue; }
        for (long c = lo; c <= hi; c++)
            if (c < (long) allowed_len && allowed[c]) any = 1;
    }
    return any;
}

int32_t bwa3_physical_core_count_masked_from(const char *cpu_root,
                                             const unsigned char *allowed, int32_t allowed_len) {
    /* Walk the online CPU list ("0-31", "0-7,16-23", ...); each physical core is
     * counted once, at the leader of its thread_siblings_list (its lowest-numbered
     * sibling). A core counts only when the process may run on at least one of its
     * siblings -- so a cpuset/taskset restriction to a subset of the SMT threads
     * counts the cores actually available, not the host's full set. allowed == NULL
     * counts every online core (the host-wide count). A read or parse failure
     * anywhere returns 0 (unknown) rather than a partial count. */
    char path[256];
    snprintf(path, sizeof(path), "%s/online", cpu_root);
    FILE *fp = fopen(path, "r");
    if (fp == NULL) return 0;
    char online[8192];
    if (fgets(online, sizeof(online), fp) == NULL) { fclose(fp); return 0; }
    fclose(fp);
    /* A list too long for the buffer would be counted short: treat as unknown. */
    if (strchr(online, '\n') == NULL) return 0;

    int32_t cores = 0;
    char *save = NULL;
    for (char *tok = strtok_r(online, ",\n", &save); tok != NULL; tok = strtok_r(NULL, ",\n", &save)) {
        long lo, hi;
        char *end = NULL;
        lo = strtol(tok, &end, 10);
        if (end == tok) return 0;
        if (*end == '-') {
            char *hi_end = NULL;
            hi = strtol(end + 1, &hi_end, 10);
            if (hi_end == end + 1) return 0;  /* a '-' with no numeric endpoint */
            end = hi_end;
        } else {
            hi = lo;
        }
        /* Reject any trailing non-whitespace on the token or range endpoint: a
         * value like "0x" or "0-1x" is malformed, so treat the whole list as
         * unknown rather than silently counting its numeric prefix. strtok_r
         * already split on ',' and '\n', so nothing but optional trailing
         * whitespace may remain. Mirrors the *end != '\0' rigor of the env
         * parsers above. */
        while (*end == ' ' || *end == '\t') end++;
        if (*end != '\0') return 0;
        if (hi < lo || lo < 0) return 0;
        for (long cpu = lo; cpu <= hi; cpu++) {
            snprintf(path, sizeof(path), "%s/cpu%ld/topology/thread_siblings_list", cpu_root, cpu);
            FILE *tp = fopen(path, "r");
            if (tp == NULL) return 0;
            char sib[512];
            const char *got = fgets(sib, sizeof(sib), tp);
            fclose(tp);
            if (got == NULL) return 0;
            char *e2 = NULL;
            long leader = strtol(sib, &e2, 10);  /* first entry: lowest sibling */
            if (e2 == sib) return 0;
            if (leader != cpu) continue;   /* count each physical core once, at its leader */
            int ok = 1;
            const int any = list_any_allowed(sib, allowed, allowed_len, &ok);
            if (!ok) return 0;
            if (any) cores++;
        }
    }
    return cores;
}

int32_t bwa3_physical_core_count_from(const char *cpu_root) {
    /* Host-wide count: every online physical core (no affinity restriction). */
    return bwa3_physical_core_count_masked_from(cpu_root, NULL, 0);
}

int32_t bwa3_physical_core_count(void) {
#if defined(__APPLE__)
    int32_t n = 0;
    size_t size = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &size, NULL, 0) == 0 && n > 0) return n;
    return 0;
#elif defined(__linux__)
    /* Count only the physical cores the process may actually run on. A cpuset or
     * taskset restriction (containers, batch schedulers) can confine the workers
     * to a subset of the host -- even to the SMT siblings of a single core -- and
     * the lockstep rule must compare n_threads against that, not the host's full
     * core count. Build an allowed[] from the effective affinity mask and pass it
     * to the masked count. Whenever the effective affinity cannot be determined,
     * return the unknown sentinel (0) rather than the host-wide count: reporting
     * the host's full core count for a possibly-restricted process could enable
     * lockstep when n_threads exceeds the process's available cores but not the
     * host's. The rule already treats physical_cores <= 0 as "no basis to move"
     * and keeps the platform default. */
    static const char *const root = "/sys/devices/system/cpu";
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return 0;  /* affinity unknown: unknown topology, not the host-wide count */
    unsigned char allowed[CPU_SETSIZE];
    for (int i = 0; i < CPU_SETSIZE; i++) allowed[i] = CPU_ISSET(i, &set) ? 1 : 0;
    /* masked == 0 means the sysfs read failed (unknown); a running process always
     * has at least one allowed CPU, so it never legitimately means "no cores". Pass
     * it through as the unknown sentinel rather than retrying host-wide. */
    return bwa3_physical_core_count_masked_from(root, allowed, (int32_t) CPU_SETSIZE);
#else
    return 0;
#endif
}

int32_t bwa3_bwtseed_lockstep_parse_env(const char *env) {
    if (env == NULL || env[0] == '\0') return -1;  /* unset/empty: apply the rule */
    if (strcmp(env, "0") == 0) return 0;
    if (strcmp(env, "1") == 0) return 1;
    return -2;                                     /* invalid: report, apply the rule */
}

int bwa3_bwtseed_lockstep_rule(int32_t n_threads, int32_t physical_cores, int default_on) {
    if (physical_cores <= 0) return default_on;    /* unknown topology: no basis to move */
    return n_threads <= physical_cores;
}

int bwa3_bwtseed_lockstep_resolve(int32_t pinned, int is_arm64, int32_t n_threads,
                                  int32_t physical_cores, int default_on) {
    if (pinned == 0 || pinned == 1) return pinned;   /* explicit pin, taken alone */
    if (is_arm64) return default_on;                 /* no SMT: keep the compiled default, no rule */
    return bwa3_bwtseed_lockstep_rule(n_threads, physical_cores, default_on);
}

int32_t bwa3_init_bwtseed_lockstep(int32_t n_threads) {
    /* Resolve on every run, not once process-wide. A library caller can invoke
     * main_mem more than once with different thread counts, and the third-pass
     * driver dispatched in bwamem.cpp must reflect the current run's policy --
     * not whichever thread count happened to come first. The sole caller runs
     * this on the main thread before the seeding workers spawn, so the plain
     * write to g_bwtseed_lockstep needs no synchronization. The first call still
     * resolves from the compiled default, so single-run behavior is unchanged. */
    const int32_t cores_used = bwa3_physical_core_count();  /* for the caller's log, pinned or not */
    const char *env = getenv("BWA3_BWTSEED_LOCKSTEP");
    const int32_t pinned = bwa3_bwtseed_lockstep_parse_env(env);
    if (pinned == -2)
        fprintf(stderr,
                "ERROR: BWA3_BWTSEED_LOCKSTEP=\"%s\" is not 0 or 1; ignoring it "
                "(resolving as if unset).\n", env);
#if defined(__aarch64__)
    const int is_arm64 = 1;
#else
    const int is_arm64 = 0;
#endif
    /* Feed the compile-time default -- never the mutable g_bwtseed_lockstep --
     * so an earlier explicit pin (or unknown-topology resolution) does not leak
     * into a later run whose environment is unset. */
    g_bwtseed_lockstep = bwa3_bwtseed_lockstep_resolve(pinned, is_arm64, n_threads,
                                                      cores_used, BWA3_BWTSEED_LOCKSTEP_DEFAULT);
    return cores_used;
}
