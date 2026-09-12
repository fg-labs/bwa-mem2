// src/simd_dispatch.cpp
#include "simd_dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

static int g_tier = BWAMEM3_TIER_NONE;
static int g_host_capability = BWAMEM3_TIER_NONE;  /* raw detect_x86_tier(),
                                                      preserved before
                                                      BWAMEM3_FORCE_TIER */
static std::once_flag g_init_flag;

/* Build-time tier — what the compiler emitted for this TU and (because the
 * BASELINE_ARCH knob in the Makefile propagates ARCH_FLAGS into CXXFLAGS for
 * every non-kernel compile) what every non-kernel TU was compiled at. Used
 * at init time to flag a gap between the binary's build baseline and the
 * host's capability — when the host is more capable than the build, hot
 * non-kernel paths (chain extension, FMI walks, mate scoring) won't be
 * auto-vectorized at the higher width and the binary leaves measurable
 * performance on the table. The arm64 branch comes first because the
 * sse2neon shim sets the SSE feature macros, which would otherwise cause
 * the SSE branches to fire on aarch64. */
#if defined(__aarch64__) || defined(__ARM_NEON)
static constexpr int g_build_tier = BWAMEM3_TIER_NEON;
#elif defined(__AVX512BW__)
static constexpr int g_build_tier = BWAMEM3_TIER_AVX512BW;
#elif defined(__AVX2__)
static constexpr int g_build_tier = BWAMEM3_TIER_AVX2;
#elif defined(__AVX__)
static constexpr int g_build_tier = BWAMEM3_TIER_AVX;
#elif defined(__SSE4_2__)
static constexpr int g_build_tier = BWAMEM3_TIER_SSE42;
#elif defined(__SSE4_1__)
static constexpr int g_build_tier = BWAMEM3_TIER_SSE41;
#else
static constexpr int g_build_tier = BWAMEM3_TIER_NONE;
#endif

#if defined(__x86_64__) || defined(__i386__)
static int detect_x86_tier(void)
{
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512bw")) return BWAMEM3_TIER_AVX512BW;
    if (__builtin_cpu_supports("avx2"))     return BWAMEM3_TIER_AVX2;
    if (__builtin_cpu_supports("avx"))      return BWAMEM3_TIER_AVX;
    if (__builtin_cpu_supports("sse4.2"))   return BWAMEM3_TIER_SSE42;
    if (__builtin_cpu_supports("sse4.1"))   return BWAMEM3_TIER_SSE41;
    return BWAMEM3_TIER_NONE;
}
#endif

const char *bwamem3_simd_tier_name(int tier)
{
    switch (tier) {
        case BWAMEM3_TIER_AVX512BW: return "avx512bw";
        case BWAMEM3_TIER_AVX2:     return "avx2";
        case BWAMEM3_TIER_AVX:      return "avx";
        case BWAMEM3_TIER_SSE42:    return "sse42";
        case BWAMEM3_TIER_SSE41:    return "sse41";
        case BWAMEM3_TIER_NEON:     return "neon";
        case BWAMEM3_TIER_NONE:     return "scalar";
        default:                    return "unknown";
    }
}

int bwamem3_simd_tier(void) { return g_tier; }

/* Internal: parse the same strings bwamem3_simd_tier_name returns. */
static int parse_tier_name(const char *name)
{
    if (name == NULL || name[0] == '\0') return -1;
    if (!strcmp(name, "avx512bw")) return BWAMEM3_TIER_AVX512BW;
    if (!strcmp(name, "avx2"))     return BWAMEM3_TIER_AVX2;
    if (!strcmp(name, "avx"))      return BWAMEM3_TIER_AVX;
    if (!strcmp(name, "sse42"))    return BWAMEM3_TIER_SSE42;
    if (!strcmp(name, "sse4.2"))   return BWAMEM3_TIER_SSE42;
    if (!strcmp(name, "sse41"))    return BWAMEM3_TIER_SSE41;
    if (!strcmp(name, "sse4.1"))   return BWAMEM3_TIER_SSE41;
    if (!strcmp(name, "neon"))     return BWAMEM3_TIER_NEON;
    if (!strcmp(name, "scalar"))   return BWAMEM3_TIER_NONE;
    return -1;
}

/* Pure helper: returns 1 if host_tier can run a binary built at build_tier.
 * x86 tiers and NEON are orthogonal — a binary built for NEON cannot run on
 * x86 and vice versa, so cross-family always returns 0. NEON is its own
 * floor: NEON-on-NEON returns 1; anything else paired with NEON returns 0. */
extern "C" int bwamem3_check_host_floor(int host_tier, int build_tier)
{
    if (build_tier == BWAMEM3_TIER_NEON) {
        return host_tier == BWAMEM3_TIER_NEON ? 1 : 0;
    }
    if (host_tier == BWAMEM3_TIER_NEON) {
        return 0;  /* x86 build on NEON host: impossible by ELF, defensive */
    }
    return host_tier >= build_tier ? 1 : 0;
}

/* Pure helper: format the host-floor error message into a caller-provided
 * buffer. Returns the number of bytes written (excluding terminator), or
 * -1 if the buffer was too small or buf is NULL. When buf != NULL &&
 * bufsz > 0, the buffer is left NUL-terminated within its bounds on
 * any return path. Pure function — no I/O, no exit. */
extern "C" int bwamem3_format_host_floor_error(char *buf, size_t bufsz,
                                               int host_tier, int build_tier)
{
    if (buf == NULL || bufsz == 0) return -1;
    const char *host_name  = bwamem3_simd_tier_name(host_tier);
    const char *build_name = bwamem3_simd_tier_name(build_tier);
    int n = snprintf(buf, bufsz,
        "[E::bwamem3] this binary was compiled for SIMD floor %s and emits %s "
        "instructions in non-kernel translation units. The host CPU does not "
        "support %s (detected: %s). Running would SIGILL on the first %s "
        "instruction.\n"
        "\n"
        "To run on this host, rebuild bwa-mem3 with BASELINE_ARCH=%s (or "
        "lower), or use a binary built for a lower SIMD floor.\n",
        build_name, build_name, build_name, host_name, build_name, host_name);
    if (n < 0 || (size_t)n >= bufsz) {
        buf[bufsz - 1] = '\0';
        return -1;
    }
    return n;
}

extern "C" int bwamem3_host_meets_floor(void)
{
    bwamem3_simd_init();
    return bwamem3_check_host_floor(g_host_capability, g_build_tier);
}

extern "C" void bwamem3_enforce_host_floor(void)
{
    bwamem3_simd_init();
    if (bwamem3_check_host_floor(g_host_capability, g_build_tier)) {
        return;
    }
    char buf[1024];
    bwamem3_format_host_floor_error(buf, sizeof(buf),
                                    g_host_capability, g_build_tier);
    fputs(buf, stderr);
    /* exit (not _exit) so stdio flushes before the process tears down. */
    exit(2);
}

extern "C" void bwamem3_print_version_simd(FILE *f)
{
    bwamem3_simd_init();

#if defined(__x86_64__) || defined(__i386__)
    const char *floor_paren = "x86 floor";
    switch (g_build_tier) {
      case BWAMEM3_TIER_AVX512BW: floor_paren = "x86-64-v4, Skylake-X 2017+"; break;
      case BWAMEM3_TIER_AVX2:     floor_paren = "x86-64-v3, Haswell 2013+";   break;
      case BWAMEM3_TIER_AVX:      floor_paren = "Sandy Bridge 2011+";          break;
      case BWAMEM3_TIER_SSE42:    floor_paren = "x86-64-v2, Nehalem 2008+";    break;
      case BWAMEM3_TIER_SSE41:    floor_paren = "Penryn 2007+";                break;
      default:                    floor_paren = "scalar";                       break;
    }
    fprintf(f, "SIMD floor: %s (%s); kernels: sse41 sse42 avx avx2 avx512bw\n",
            bwamem3_simd_tier_name(g_build_tier), floor_paren);
#elif defined(__aarch64__) || defined(__ARM_NEON)
    fprintf(f, "SIMD floor: neon (aarch64 baseline); kernels: neon\n");
#else
    fprintf(f, "SIMD floor: scalar; kernels: scalar\n");
#endif

    /* Runtime line: reflects FORCE_TIER if set. When FORCE_TIER asks for
     * an unknown / up-tier / cross-family value, init_body() emits a
     * [W::*] warning and leaves g_tier unchanged — annotate the runtime
     * line as ", ignored" so the banner doesn't look like the request
     * was accepted. Comparison is on the parsed tier value, not the raw
     * string, so aliases like "sse4.2" / "sse42" both report correctly. */
    const char *force = getenv("BWAMEM3_FORCE_TIER");
    if (force != NULL && force[0] != '\0') {
        int requested_tier = parse_tier_name(force);
        const char *suffix = (requested_tier >= 0 && requested_tier == g_tier)
                             ? "" : ", ignored";
        fprintf(f, "SIMD runtime: %s (BWAMEM3_FORCE_TIER=%s%s)\n",
                bwamem3_simd_tier_name(g_tier), force, suffix);
    } else {
        fprintf(f, "SIMD runtime: %s (BWAMEM3_FORCE_TIER unset)\n",
                bwamem3_simd_tier_name(g_tier));
    }

    /* Optional warning when raw host capability is below build floor.
     * Goes to stderr (not f) to match the convention of every other
     * [W::*] warning in the codebase, and so 'bwa-mem3 version | grep ^SIMD'
     * in CI scripts isn't contaminated by the warning line. */
    if (!bwamem3_check_host_floor(g_host_capability, g_build_tier)) {
        fprintf(stderr, "[W::bwa-mem3] this host (%s) is below the binary's floor "
                   "(%s); 'bwa-mem3 mem' will refuse to run. Rebuild with "
                   "BASELINE_ARCH=%s (or lower) to run on this host.\n",
                bwamem3_simd_tier_name(g_host_capability),
                bwamem3_simd_tier_name(g_build_tier),
                bwamem3_simd_tier_name(g_host_capability));
    }
}

/* The body of bwamem3_simd_init. Called at most once via std::call_once. */
static void bwamem3_simd_init_body(void)
{
#if defined(__x86_64__) || defined(__i386__)
    g_host_capability = detect_x86_tier();
#elif defined(__aarch64__) || defined(__ARM_NEON)
    g_host_capability = BWAMEM3_TIER_NEON;
#else
    g_host_capability = BWAMEM3_TIER_NONE;
#endif

#ifdef BWAMEM3_TESTING
    /* Test-only: BWAMEM3_TESTING_HOST_TIER overrides the detected host
     * capability, so regression tests can simulate too-old hosts without
     * actually being on one. Production builds (without -DBWAMEM3_TESTING)
     * never compile this path. */
    {
        const char *injected = getenv("BWAMEM3_TESTING_HOST_TIER");
        if (injected != NULL && injected[0] != '\0') {
            int tier = parse_tier_name(injected);
            if (tier >= 0) {
                g_host_capability = tier;
            } else {
                fprintf(stderr, "[W::%s] ignoring BWAMEM3_TESTING_HOST_TIER=%s "
                                "(unknown tier)\n", __func__, injected);
            }
        }
    }
#endif

    g_tier = g_host_capability;

    /* Build-vs-host gap (debug-only).
     *
     * The per-tier kernel TUs (KERNEL_SRCS in the Makefile) are compiled
     * at every supported tier and dispatched correctly via this file, so
     * kernel calls (BSW, kswv, ksw_extend, sam_encode_*) will pick the
     * host's tier regardless of BASELINE_ARCH. Every non-kernel TU
     * (bwamem.cpp, bwamem_pair.cpp, FMI_search.cpp, fastmap.cpp, ...) is
     * compiled once at BASELINE_ARCH, so a build_tier < host_tier mismatch
     * means the compiler couldn't auto-vectorize those hot paths at the
     * higher width.
     *
     * Earlier versions of bwa-mem3 emitted a [W::] warning here promising
     * "10-15%% slower hot paths, rebuild with BASELINE_ARCH=<host_tier>
     * to recover". Empirically (c7a / c7i, a 5M-read WGS slice,
     * shm-warmed bare-metal) the gap is much smaller than that and the
     * recommendation is not always applicable:
     *   - avx2 -> avx512bw: c7a -2.2%%, c7i -0.7%% (wash, both cases)
     *   - avx2 -> avx512bw + -mprefer-vector-width=256 (the default for
     *     arch=avx512bw): c7a -4.4%%, c7i -0.7%%
     *   - sse41 -> avx2: ~+10-15%% on AVX2 hosts (the original PR #84
     *     measurement that motivated raising the default to avx2)
     * The "10-15%%" figure was the sse41->avx2 transition on AVX2-only
     * hosts; it does not generalize to avx2->avx512bw.
     *
     * The warning is now BWAMEM3_DEBUG_SIMD-gated to avoid spamming a
     * misleading recommendation in production logs. The per-stage tier
     * report below already covers the diagnostic case. */
#if defined(__x86_64__) || defined(__i386__)
    if (g_build_tier > BWAMEM3_TIER_NONE && g_build_tier < g_tier
        && getenv("BWAMEM3_DEBUG_SIMD")) {
        fprintf(stderr,
                "[M::%s] build baseline %s < host tier %s; non-kernel TUs "
                "compiled for %s. Hot kernel paths self-dispatch at host tier "
                "regardless. Rebuilding with BASELINE_ARCH=%s typically "
                "yields <2%% wall-time gain on AVX-512 hosts (PR #84's "
                "10-15%% figure was the sse41->avx2 transition).\n",
                __func__,
                bwamem3_simd_tier_name(g_build_tier),
                bwamem3_simd_tier_name(g_tier),
                bwamem3_simd_tier_name(g_build_tier),
                bwamem3_simd_tier_name(g_tier));
    }
#endif

    /* Optional override: BWAMEM3_FORCE_TIER=<name> downgrades only.
     * Up-tier requests would SIGILL on the first wider instruction. */
    const char *force = getenv("BWAMEM3_FORCE_TIER");
    if (force != NULL && force[0] != '\0') {
        int forced = parse_tier_name(force);
        if (forced < 0) {
            fprintf(stderr, "[W::%s] ignoring BWAMEM3_FORCE_TIER=%s (unknown tier)\n",
                    __func__, force);
        } else {
#if defined(__x86_64__) || defined(__i386__)
            /* Only x86 tiers are ordered; reject NEON forcing on x86. */
            if (forced == BWAMEM3_TIER_NEON) {
                fprintf(stderr, "[W::%s] ignoring BWAMEM3_FORCE_TIER=neon on x86\n", __func__);
            } else if (forced > g_tier) {
                fprintf(stderr, "[W::%s] ignoring BWAMEM3_FORCE_TIER=%s (host is %s; cannot up-tier)\n",
                        __func__, force, bwamem3_simd_tier_name(g_tier));
            } else {
                g_tier = forced;
            }
#else
            /* arm64 and scalar: allow only the matching tier or scalar. */
            if (forced != g_tier && forced != BWAMEM3_TIER_NONE) {
                fprintf(stderr, "[W::%s] ignoring BWAMEM3_FORCE_TIER=%s on this arch\n",
                        __func__, force);
            } else {
                g_tier = forced;
            }
#endif
        }
    }

    if (getenv("BWAMEM3_DEBUG_SIMD")) {
        fprintf(stderr, "[M::%s] SIMD tier: %s (build baseline: %s)\n",
                __func__,
                bwamem3_simd_tier_name(g_tier),
                bwamem3_simd_tier_name(g_build_tier));
    }
}

void bwamem3_simd_init(void)
{
    /* std::call_once: thread-safe, exactly-once execution of the init body.
     * Any number of concurrent callers will block until the running one
     * finishes, then return; subsequent calls are a single relaxed load. */
    std::call_once(g_init_flag, bwamem3_simd_init_body);
}

/* ── Per-tier kernel factory dispatch ──────────────────────────────────
 *
 * Each per-tier kernel TU compile produces a distinct concrete class
 * (BandedPairWiseSW_avx2 etc.). The factories below construct the right
 * one based on g_tier and return ownership via unique_ptr<IBandedPairWiseSW>
 * / unique_ptr<Ikswv>. On arm64 there's only one tier (NEON, unsuffixed)
 * so the factories construct the unmangled class directly.
 */

#include "bandedSWA.h"
#include "kswv.h"
#include "ksw.h"
#include "sam_encode.h"

#if defined(__x86_64__) || defined(__i386__)

std::unique_ptr<IBandedPairWiseSW> make_banded_pair_wise_sw(
    int o_del, int e_del, int o_ins, int e_ins, int zdrop,
    int end_bonus, const int8_t *mat,
    int8_t w_match, int8_t w_mismatch, int numThreads)
{
    bwamem3_simd_init();
    IBandedPairWiseSW *raw;
    switch (g_tier) {
      case BWAMEM3_TIER_AVX512BW:
        raw = make_bsw_kernel_avx512bw(o_del, e_del, o_ins, e_ins, zdrop, end_bonus,
                                       mat, w_match, w_mismatch, numThreads); break;
      case BWAMEM3_TIER_AVX2:
        raw = make_bsw_kernel_avx2(o_del, e_del, o_ins, e_ins, zdrop, end_bonus,
                                   mat, w_match, w_mismatch, numThreads); break;
      case BWAMEM3_TIER_AVX:
        raw = make_bsw_kernel_avx(o_del, e_del, o_ins, e_ins, zdrop, end_bonus,
                                  mat, w_match, w_mismatch, numThreads); break;
      case BWAMEM3_TIER_SSE42:
        raw = make_bsw_kernel_sse42(o_del, e_del, o_ins, e_ins, zdrop, end_bonus,
                                    mat, w_match, w_mismatch, numThreads); break;
      case BWAMEM3_TIER_SSE41:
      default:
        raw = make_bsw_kernel_sse41(o_del, e_del, o_ins, e_ins, zdrop, end_bonus,
                                    mat, w_match, w_mismatch, numThreads); break;
    }
    return std::unique_ptr<IBandedPairWiseSW>(raw);
}

/* Mat-aware factory (issue 173): routes through the per-tier _mat overloads so
 * the chosen tier's concrete kswv runs freed-cell detection in its ctor. */
std::unique_ptr<Ikswv> make_kswv(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen,
    const int8_t *mat25)
{
    bwamem3_simd_init();
    Ikswv *raw;
    switch (g_tier) {
      case BWAMEM3_TIER_AVX512BW:
        raw = make_kswv_kernel_avx512bw_mat(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                                            numThreads, maxRefLen, maxQerLen, mat25); break;
      case BWAMEM3_TIER_AVX2:
        raw = make_kswv_kernel_avx2_mat(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                                        numThreads, maxRefLen, maxQerLen, mat25); break;
      case BWAMEM3_TIER_AVX:
        raw = make_kswv_kernel_avx_mat(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                                       numThreads, maxRefLen, maxQerLen, mat25); break;
      case BWAMEM3_TIER_SSE42:
        raw = make_kswv_kernel_sse42_mat(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                                         numThreads, maxRefLen, maxQerLen, mat25); break;
      case BWAMEM3_TIER_SSE41:
      default:
        raw = make_kswv_kernel_sse41_mat(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                                         numThreads, maxRefLen, maxQerLen, mat25); break;
    }
    return std::unique_ptr<Ikswv>(raw);
}

/* 9-arg factory: delegates to the mat-aware path with mat25 == nullptr, which
 * reproduces the original symmetric-kernel behavior exactly. */
std::unique_ptr<Ikswv> make_kswv(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen)
{
    return make_kswv(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                     numThreads, maxRefLen, maxQerLen, nullptr);
}

#else  /* arm64 / scalar fallback */

std::unique_ptr<IBandedPairWiseSW> make_banded_pair_wise_sw(
    int o_del, int e_del, int o_ins, int e_ins, int zdrop,
    int end_bonus, const int8_t *mat,
    int8_t w_match, int8_t w_mismatch, int numThreads)
{
    return std::unique_ptr<IBandedPairWiseSW>(
        new BandedPairWiseSW(o_del, e_del, o_ins, e_ins, zdrop, end_bonus,
                             mat, w_match, w_mismatch, numThreads));
}

/* Mat-aware factory (issue 173): single unsuffixed arm64 kswv, constructed via
 * the mat-aware ctor so freed-cell detection runs. */
std::unique_ptr<Ikswv> make_kswv(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen,
    const int8_t *mat25)
{
    return std::unique_ptr<Ikswv>(
        new kswv(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                 numThreads, maxRefLen, maxQerLen, mat25));
}

/* 9-arg factory: delegates with mat25 == nullptr (original behavior). */
std::unique_ptr<Ikswv> make_kswv(
    int o_del, int e_del, int o_ins, int e_ins,
    int8_t w_match, int8_t w_mismatch,
    int numThreads, int32_t maxRefLen, int32_t maxQerLen)
{
    return make_kswv(o_del, e_del, o_ins, e_ins, w_match, w_mismatch,
                     numThreads, maxRefLen, maxQerLen, nullptr);
}

#endif  /* x86 vs arm64 factory dispatch */

/* ── ksw C-linkage dispatch wrappers ────────────────────────────────────
 *
 * ksw.cpp's free functions are SIMD-bearing and get per-tier mangled by
 * kernel_dispatch.h. Non-kernel TUs (bwa.cpp, bwamem.cpp, bwamem_pair.cpp)
 * call the unmangled names, so we provide thin wrappers here that pick
 * the right mangled symbol at runtime via a switch on g_tier.
 *
 * On arm64 the unmangled symbol IS the only build, so the wrappers just
 * call through directly.
 */

#if defined(__x86_64__) || defined(__i386__)

/* Forward-declare the per-tier mangled symbols. Each kernel TU compile
 * with KERNEL_VARIANT=_avx2 etc. will define exactly one of each set. */

#define BWAMEM3_DECLARE_KSW_TIER(suffix)                                              \
extern "C" int ksw_extend2##suffix(int qlen, const uint8_t *query, int tlen,          \
    const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,            \
    int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0,                    \
    int *qle, int *tle, int *gtle, int *gscore, int *max_off);                        \
extern "C" int ksw_extend##suffix(int qlen, const uint8_t *query, int tlen,           \
    const uint8_t *target, int m, const int8_t *mat, int gapo, int gape,              \
    int w, int end_bonus, int zdrop, int h0, int *qle, int *tle,                      \
    int *gtle, int *gscore, int *max_off);                                            \
extern "C" int ksw_global2##suffix(int qlen, const uint8_t *query, int tlen,          \
    const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,            \
    int o_ins, int e_ins, int w, int *n_cigar, uint32_t **cigar);                     \
extern "C" int ksw_global2_scalar_ref##suffix(int qlen, const uint8_t *query,         \
    int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,  \
    int o_ins, int e_ins, int w, int *n_cigar, uint32_t **cigar);                     \
extern "C" unsigned long ksw_g2_wave_exec_count##suffix(void);                        \
extern "C" unsigned long ksw_g2_wave16_exec_count##suffix(void);                      \
extern "C" int ksw_g2_wave16_wmin##suffix(void);                                      \
extern "C" int ksw_g2_wave_wmin##suffix(void);                                        \
extern "C" unsigned long ksw_g2_wave_zr_capacity##suffix(void);                       \
extern "C" int ksw_global##suffix(int qlen, const uint8_t *query, int tlen,           \
    const uint8_t *target, int m, const int8_t *mat, int gapo, int gape,              \
    int w, int *n_cigar, uint32_t **cigar);                                           \
extern "C" kswr_t ksw_align2##suffix(int qlen, uint8_t *query, int tlen,              \
    uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,                  \
    int o_ins, int e_ins, int xtra, kswq_t **qry);                                    \
extern "C" kswr_t ksw_align##suffix(int qlen, uint8_t *query, int tlen,               \
    uint8_t *target, int m, const int8_t *mat, int gapo, int gape, int xtra,          \
    kswq_t **qry);

BWAMEM3_DECLARE_KSW_TIER(_sse41)
BWAMEM3_DECLARE_KSW_TIER(_sse42)
BWAMEM3_DECLARE_KSW_TIER(_avx)
BWAMEM3_DECLARE_KSW_TIER(_avx2)
BWAMEM3_DECLARE_KSW_TIER(_avx512bw)
#undef BWAMEM3_DECLARE_KSW_TIER

/* Helper macro to pick a tier-specific symbol based on g_tier. */
#define KSW_DISPATCH_CALL(name, ...) \
    do { \
        switch (g_tier) { \
          case BWAMEM3_TIER_AVX512BW: return name##_avx512bw(__VA_ARGS__); \
          case BWAMEM3_TIER_AVX2:     return name##_avx2(__VA_ARGS__); \
          case BWAMEM3_TIER_AVX:      return name##_avx(__VA_ARGS__); \
          case BWAMEM3_TIER_SSE42:    return name##_sse42(__VA_ARGS__); \
          default:                    return name##_sse41(__VA_ARGS__); \
        } \
    } while(0)

extern "C" int ksw_extend2(int qlen, const uint8_t *query, int tlen,
    const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,
    int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0,
    int *qle, int *tle, int *gtle, int *gscore, int *max_off) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_extend2, qlen, query, tlen, target, m, mat, o_del, e_del,
                      o_ins, e_ins, w, end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
}

extern "C" int ksw_extend(int qlen, const uint8_t *query, int tlen,
    const uint8_t *target, int m, const int8_t *mat, int gapo, int gape,
    int w, int end_bonus, int zdrop, int h0, int *qle, int *tle,
    int *gtle, int *gscore, int *max_off) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_extend, qlen, query, tlen, target, m, mat, gapo, gape, w,
                      end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
}

extern "C" int ksw_global2(int qlen, const uint8_t *query, int tlen,
    const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,
    int o_ins, int e_ins, int w, int *n_cigar, uint32_t **cigar) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_global2, qlen, query, tlen, target, m, mat, o_del, e_del,
                      o_ins, e_ins, w, n_cigar, cigar);
}

/* Test-only: routes to the current tier's scalar reference behind ksw_global2.
 * Lets the wavefront byte-identity unit test call the scalar oracle for the
 * same (input, w) the SIMD path takes. Not used on any production path. */
extern "C" int ksw_global2_scalar_ref(int qlen, const uint8_t *query, int tlen,
    const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,
    int o_ins, int e_ins, int w, int *n_cigar, uint32_t **cigar) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_global2_scalar_ref, qlen, query, tlen, target, m, mat, o_del, e_del,
                      o_ins, e_ins, w, n_cigar, cigar);
}

/* Test-only: wavefront-exec counters for the active tier (see ksw.cpp). The
 * int16 variant counts only int16-kernel entries. */
extern "C" unsigned long ksw_g2_wave_exec_count(void) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_g2_wave_exec_count);
}

extern "C" unsigned long ksw_g2_wave16_exec_count(void) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_g2_wave16_exec_count);
}

/* Test-only: the wavefront width crossovers the active tier was compiled with
 * (see ksw.cpp); 0 where the tier has no such kernel. */
extern "C" int ksw_g2_wave16_wmin(void) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_g2_wave16_wmin);
}

extern "C" int ksw_g2_wave_wmin(void) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_g2_wave_wmin);
}

/* Test-only: retained zr capacity for the active tier (see ksw.cpp). */
extern "C" unsigned long ksw_g2_wave_zr_capacity(void) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_g2_wave_zr_capacity);
}

extern "C" int ksw_global(int qlen, const uint8_t *query, int tlen,
    const uint8_t *target, int m, const int8_t *mat, int gapo, int gape,
    int w, int *n_cigar, uint32_t **cigar) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_global, qlen, query, tlen, target, m, mat, gapo, gape, w,
                      n_cigar, cigar);
}

extern "C" kswr_t ksw_align2(int qlen, uint8_t *query, int tlen,
    uint8_t *target, int m, const int8_t *mat, int o_del, int e_del,
    int o_ins, int e_ins, int xtra, kswq_t **qry) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_align2, qlen, query, tlen, target, m, mat, o_del, e_del,
                      o_ins, e_ins, xtra, qry);
}

extern "C" kswr_t ksw_align(int qlen, uint8_t *query, int tlen,
    uint8_t *target, int m, const int8_t *mat, int gapo, int gape, int xtra,
    kswq_t **qry) {
    bwamem3_simd_init();
    KSW_DISPATCH_CALL(ksw_align, qlen, query, tlen, target, m, mat, gapo, gape, xtra, qry);
}

#endif  /* x86 ksw dispatch wrappers */

/* ── sam_encode C-linkage dispatch wrappers ─────────────────────────────────
 *
 * sam_encode.cpp's three functions are compiled per-tier on x86 so the
 * compiler can emit VEX/EVEX-encoded SIMD where available. Non-kernel TUs
 * (bwamem.cpp) call the unmangled names; these wrappers resolve those calls
 * to the right tier-mangled body at runtime.
 *
 * On arm64 there is only one build (unmangled NEON/scalar), so the wrappers
 * are not needed — bwamem.cpp resolves directly to the body in sam_encode.o.
 */

#if defined(__x86_64__) || defined(__i386__)

#define BWAMEM3_DECLARE_SAM_ENCODE_TIER(suffix)                                        \
extern "C" void sam_encode_seq_fwd##suffix(char *dst, const uint8_t *src, int n);      \
extern "C" void sam_encode_seq_rev##suffix(char *dst, const uint8_t *src, int n);      \
extern "C" void sam_encode_qual_rev##suffix(char *dst, const char *src, int n);

BWAMEM3_DECLARE_SAM_ENCODE_TIER(_sse41)
BWAMEM3_DECLARE_SAM_ENCODE_TIER(_sse42)
BWAMEM3_DECLARE_SAM_ENCODE_TIER(_avx)
BWAMEM3_DECLARE_SAM_ENCODE_TIER(_avx2)
BWAMEM3_DECLARE_SAM_ENCODE_TIER(_avx512bw)
#undef BWAMEM3_DECLARE_SAM_ENCODE_TIER

#define SAM_ENCODE_DISPATCH_VOID(name, ...) \
    do { \
        switch (g_tier) { \
          case BWAMEM3_TIER_AVX512BW: name##_avx512bw(__VA_ARGS__); return; \
          case BWAMEM3_TIER_AVX2:     name##_avx2(__VA_ARGS__);     return; \
          case BWAMEM3_TIER_AVX:      name##_avx(__VA_ARGS__);      return; \
          case BWAMEM3_TIER_SSE42:    name##_sse42(__VA_ARGS__);    return; \
          default:                    name##_sse41(__VA_ARGS__);    return; \
        } \
    } while(0)

extern "C" void sam_encode_seq_fwd(char *dst, const uint8_t *src, int n) {
    bwamem3_simd_init();
    SAM_ENCODE_DISPATCH_VOID(sam_encode_seq_fwd, dst, src, n);
}

extern "C" void sam_encode_seq_rev(char *dst, const uint8_t *src, int n) {
    bwamem3_simd_init();
    SAM_ENCODE_DISPATCH_VOID(sam_encode_seq_rev, dst, src, n);
}

extern "C" void sam_encode_qual_rev(char *dst, const char *src, int n) {
    bwamem3_simd_init();
    SAM_ENCODE_DISPATCH_VOID(sam_encode_qual_rev, dst, src, n);
}

#endif  /* x86 sam_encode dispatch wrappers */
