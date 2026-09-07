#ifndef LOCKSTEP_WIDTH_H
#define LOCKSTEP_WIDTH_H

#include <stdint.h>
#include <stddef.h>  /* size_t */

/* Phase-2 SMEM lockstep width.
 *
 * SMEM_LOCKSTEP_N is three things at once: the compile-time default, the
 * regression floor for the runtime auto-tune (the startup probe may only
 * RAISE the width, never pick below this), and the enable guard
 * (`#if SMEM_LOCKSTEP_N > 1`) that selects the lockstep driver over the
 * scalar path at build time. Overridable at build time (a `-D` on the
 * command line wins over this default). See getSMEMsOnePosOneThread_lockstep
 * in FMI_search.cpp for the driver this width feeds. */
#ifndef SMEM_LOCKSTEP_N
#define SMEM_LOCKSTEP_N 16
#endif

/* Upper bound on the runtime width. Sizes the driver's on-stack slot array
 * (BatchSlot slots[SMEM_LOCKSTEP_N_MAX]) and caps the value the startup
 * memory-level-parallelism probe may select. Raise only in step with the
 * stack-array cost (~80 B per slot). */
#ifndef SMEM_LOCKSTEP_N_MAX
#define SMEM_LOCKSTEP_N_MAX 64
#endif

/* Both widths are `-D`-overridable, and the driver indexes an on-stack
 * BatchSlot slots[SMEM_LOCKSTEP_N_MAX] array with a runtime width in
 * [SMEM_LOCKSTEP_N, SMEM_LOCKSTEP_N_MAX]. An override that inverts the range
 * (SMEM_LOCKSTEP_N > SMEM_LOCKSTEP_N_MAX) or a non-positive bound would index
 * out of bounds, so reject it at compile time rather than miscompile.
 *
 * The upper bound is a STORAGE-safe cap, not merely INT32_MAX: both the
 * driver's BatchSlot slots[SMEM_LOCKSTEP_N_MAX] (~100 B/slot, FMI_search.cpp)
 * and the probe's uint64_t cur[SMEM_LOCKSTEP_N_MAX] (8 B/slot, lockstep_width.cpp)
 * are fixed-size STACK arrays sized by this bound. An INT32_MAX bound is
 * representable in int32_t but would demand a multi-gigabyte stack frame and
 * crash before the probe ever runs. Cap the override at a width whose
 * worst-case frame (~28 KB at 256) is safe on any thread stack, while still
 * leaving 4x headroom over the shipped default 64. */
#define SMEM_LOCKSTEP_N_MAX_CAP 256
#if !(SMEM_LOCKSTEP_N >= 1 && SMEM_LOCKSTEP_N <= SMEM_LOCKSTEP_N_MAX && SMEM_LOCKSTEP_N_MAX <= SMEM_LOCKSTEP_N_MAX_CAP)
#error "require 1 <= SMEM_LOCKSTEP_N <= SMEM_LOCKSTEP_N_MAX <= SMEM_LOCKSTEP_N_MAX_CAP (256; stack-array storage-safe)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime phase-2 lockstep width, resolved once at startup. Initialized to
 * SMEM_LOCKSTEP_N so a binary that never runs the probe behaves exactly like
 * the compile-time constant. Read (not written) on the seeding hot path. */
extern int32_t g_smem_lockstep_n;

/* Clamp a raw memory-level-parallelism estimate to the usable width range
 * [SMEM_LOCKSTEP_N, SMEM_LOCKSTEP_N_MAX]. A non-positive estimate (a probe
 * that failed to measure) returns the floor. The floor is what guarantees the
 * auto-tune never selects a width below the shipped default. */
int32_t bwa3_lockstep_width_from_probe(int32_t raw_mlp);

/* Classify a BWA3_SMEM_LOCKSTEP_N override (env may be NULL/empty) WITHOUT
 * running the probe or touching global state. Returns:
 *   > 0  the width to pin, clamped up to SMEM_LOCKSTEP_N_MAX -- a valid positive
 *        integer override (it MAY sit below the floor, an explicit operator/gate
 *        escape hatch); the caller skips the probe and uses this;
 *     0  env is unset or empty -- no override; the caller keeps the compile-time
 *        default (and probes only if BWA3_SMEM_LOCKSTEP_PROBE is set);
 *    -1  env is set but malformed, non-positive, or overflowed -- an invalid
 *        override; the caller reports it and resolves it as if unset.
 * Pure and side-effect-free (emits no diagnostic), so it is unit-testable; the
 * ERROR message for the -1 case is the initializer's responsibility. */
int32_t bwa3_lockstep_width_parse_env(const char *env);

/* Gate the startup MLP probe opt-in on a BWA3_SMEM_LOCKSTEP_PROBE value (env may
 * be NULL/empty). Returns nonzero (enable the probe) only for a truthy value:
 * present, non-empty, and not "0"; unset, empty, and "0" return 0 (disabled).
 * Pure and side-effect-free, so the gating is unit-testable independently of the
 * initializer that consults it. */
int bwa3_lockstep_probe_enabled(const char *env);

/* Measure the core's memory-level parallelism: the number of independent
 * dependent-load chains it keeps outstanding before per-access latency stops
 * falling. The chase walks a value-dependent pseudo-random cycle THROUGH the
 * FM-index's own cp_occ checkpoint array -- gigabytes, already resident, the
 * same array the seeding walk hammers, with the same TLB behaviour -- rather
 * than a scratch buffer (a fresh buffer measures whatever cache level it fits
 * in and over-picks). `base`/`n_blocks`/`stride` describe cp_occ as an opaque
 * array of `n_blocks` fixed-size blocks; `word_off` is the byte offset of a
 * 64-bit word to read from each block. Returns the knee width (raw, pre-clamp),
 * or 0 if the array is absent or too small to measure (caller floors to the
 * default). Cost is host-dependent: ~10-20 ms on low-latency memory (e.g. Apple
 * Silicon), ~0.4 s on server-class DRAM (a ~100 ns dependent-load latency over
 * the sweep). This is the raw measurement primitive: it retains only its input
 * guard (returns 0 when base == NULL or n_blocks < 4096) and has no one-shot or
 * opt-in guard, so a direct caller with a large enough array runs the full sweep
 * on every call. The "at most once at startup, and only when opted in" contract
 * is enforced by the sole intended caller, bwa3_init_smem_lockstep_width
 * (idempotent, gated on BWA3_SMEM_LOCKSTEP_PROBE) -- see below. */
int32_t bwa3_measure_mlp(const void *base, int64_t n_blocks,
                         size_t stride, size_t word_off);

/* Resolve and install g_smem_lockstep_n once. Idempotent (subsequent calls are
 * no-ops). Resolution order:
 *   1. BWA3_SMEM_LOCKSTEP_N=<n>   -- explicit pin, taken alone.
 *   2. BWA3_SMEM_LOCKSTEP_PROBE=1 -- opt into the startup MLP probe, which
 *      chases the cp_occ array (base/n_blocks/stride/word_off) to self-calibrate
 *      the width for this host. Truthy opt-in (bwa3_lockstep_probe_enabled): any
 *      value other than unset, empty, or "0" enables it, so unset the variable
 *      (or set it to 0) to disable -- BWA3_SMEM_LOCKSTEP_PROBE=0 does NOT enable.
 *   3. neither set (the default)  -- keep the compile-time SMEM_LOCKSTEP_N.
 * The probe is OPT-IN because the width is a measured ~1% flat knob end-to-end
 * across architectures, core counts, and page sizes, so paying its startup cost
 * on every run buys no throughput; the default constant is what the probe
 * converges to on all benchmarked hardware. An invalid env value is reported
 * (ERROR to stderr) and ignored, resolving as if unset.
 * Call after the index is loaded and before the seeding workers spawn. */
void bwa3_init_smem_lockstep_width(const void *base, int64_t n_blocks,
                                   size_t stride, size_t word_off);


/* ---- Third-pass bwtseed re-seeding lockstep: on/off policy -------------------
 *
 * FMI_search::bwtSeedStrategyAllPosOneThread_lockstep overlaps BWTSEED_LOCKSTEP_N
 * reads' forward-extension walks so their cp_occ cache misses issue together.
 * That hides memory latency the core would otherwise idle on -- which is only a
 * win where nothing else hides it. On a non-SMT core (arm64: Graviton, Apple
 * Silicon) it is a large seeding win at any thread count. On an SMT core the
 * sibling hyperthread hides the same latency once both siblings are busy, and
 * the lockstep's extra bookkeeping then shows up as wall time; below that
 * point (threads <= physical cores, one thread per core) the sibling is idle
 * and the lockstep wins as it does on arm64. Measured on x86 (5M-pair WGS,
 * clang 19): -8 to -10% user CPU at -t = physical cores on both AMD Zen 3 and
 * Intel Sapphire Rapids; at -t = 2x cores (both siblings busy) user CPU still
 * falls ~7% but wall is bimodal on Zen 3, so the rule keeps it off there.
 * Whole-aligner wall at -t = physical cores: -9.8% on both hosts (61.4 -> 55.3 s
 * and 56.1 -> 50.6 s for the 5M-pair slice).
 *
 * Output is byte-identical either way (same SMEM emission order; the lockstep
 * parity harness pins it) -- this is a scheduling choice, never a result one. */

/* Runtime switch read by mem_collect_smem: nonzero selects the lockstep driver,
 * zero the scalar one. Initialized to the pre-rule shipping default (on for
 * arm64, off elsewhere) so a binary that never runs the resolver behaves as
 * before. Read (not written) on the seeding hot path. */
extern int32_t g_bwtseed_lockstep;

/* Count the distinct physical cores the process may actually run on (Linux: one
 * per thread_siblings_list leader under /sys/devices/system/cpu, intersected with
 * the effective sched_getaffinity mask so a cpuset/taskset restriction to a subset
 * of CPUs -- even to the SMT siblings of a single core -- counts only the cores
 * available, not the host's full set; macOS: hw.physicalcpu). Returns 0 when the
 * topology cannot be read, which the rule treats as "unknown": keep the platform
 * default. */
int32_t bwa3_physical_core_count(void);

/* The sysfs parser behind the Linux branch, on an explicit cpu directory
 * (`<cpu_root>/online`, `<cpu_root>/cpu<N>/topology/thread_siblings_list`).
 * Counts the CPUs that lead their own sibling list, so each physical core is
 * counted once whatever the list's spelling ("0,16" or "0-1"). Returns 0 on any
 * read or parse failure, never a partial count. Exposed so the unit test can
 * point it at a synthetic tree with a known answer; production calls go through
 * bwa3_physical_core_count. */
int32_t bwa3_physical_core_count_from(const char *cpu_root);

/* As bwa3_physical_core_count_from, but a physical core counts only when at least
 * one of its SMT siblings is allowed: CPU c is allowed when c < allowed_len and
 * allowed[c] != 0. `allowed == NULL` counts every online core (identical to
 * bwa3_physical_core_count_from). This is how bwa3_physical_core_count applies the
 * affinity mask; exposed for the unit test to drive with a synthetic mask. */
int32_t bwa3_physical_core_count_masked_from(const char *cpu_root,
                                             const unsigned char *allowed, int32_t allowed_len);

/* Classify a BWA3_BWTSEED_LOCKSTEP override (env may be NULL/empty) without
 * touching global state:  1 = pin on, 0 = pin off, -1 = unset/empty (apply
 * the rule), -2 = set but not "0"/"1" (invalid: the caller reports it and
 * applies the rule). Pure, so it is unit-testable. */
int32_t bwa3_bwtseed_lockstep_parse_env(const char *env);

/* The rule itself, pure: enable when every worker thread gets its own physical
 * core (n_threads <= physical_cores). An unknown topology (physical_cores <= 0)
 * returns `default_on` unchanged -- there is no basis to move off the platform
 * default. */
int bwa3_bwtseed_lockstep_rule(int32_t n_threads, int32_t physical_cores, int default_on);

/* The resolution order as one pure function (unit-tested), fed by the pieces
 * above: `pinned` is bwa3_bwtseed_lockstep_parse_env's result (1 / 0 pin, -1
 * unset, -2 invalid), `is_arm64` selects the platform that keeps its compiled
 * default without consulting the rule (no SMT on any shipping arm64 host; the
 * measured shipping state), `default_on` is that compiled default
 * (BWA3_BWTSEED_LOCKSTEP_DEFAULT in lockstep_width.cpp, the single place it is
 * written down; it also initializes g_bwtseed_lockstep).
 *   1. a 0/1 pin is taken alone (gates / CI);
 *   2. arm64 returns default_on;
 *   3. otherwise bwa3_bwtseed_lockstep_rule(n_threads, physical_cores, default_on). */
int bwa3_bwtseed_lockstep_resolve(int32_t pinned, int is_arm64, int32_t n_threads,
                                  int32_t physical_cores, int default_on);

/* Resolve and install g_bwtseed_lockstep, re-resolving on every call: reads the
 * env pin, counts the host's physical cores, and applies
 * bwa3_bwtseed_lockstep_resolve. Re-resolving (rather than caching the first
 * decision) lets a library caller that runs main_mem repeatedly with different
 * thread counts get the right policy each time. An invalid env value is reported
 * (ERROR to stderr) and resolved as if unset. Returns the physical core count it
 * read (0 = unknown) so the caller can log the decision. Call on the main thread
 * before the seeding workers spawn. */
int32_t bwa3_init_bwtseed_lockstep(int32_t n_threads);

#ifdef __cplusplus
}
#endif

#endif /* LOCKSTEP_WIDTH_H */
