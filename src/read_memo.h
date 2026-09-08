#ifndef READ_MEMO_H
#define READ_MEMO_H

#include <cstdint>

/* Whole-read-pair memoization ("--dedup-reads").
 *
 * When two input read-PAIRS are byte-identical in both mates' bases, their
 * post-extension alignment regions (mem_alnreg_v) are byte-identical too (proven
 * by the Phase-0 kernel-invariance audit: the alignment kernels are a pure
 * function of read content + index; every id/name/qual/position dependence lives
 * in the replayed SAM stage). So we can run seed->chain->extend once per distinct
 * pair (the REPRESENTATIVE) and copy its regs to the DUPLICATES, then replay the
 * SAM stage per read for byte-identical output.
 *
 * This header exposes only what bwamem.cpp / fastmap.cpp need. The dedup window
 * is one align invocation (the -K chunk): duplicates are collapsed within a
 * chunk only, because the alignment result depends on chunk-scoped state
 * (est_insert_high from per-chunk mem_pestat), which is constant within a chunk
 * but varies across chunks -- so a cross-chunk cache would not be byte-identical.
 *
 * The memoize path (Phase 2): when the controller latches ON (or mode==on), the
 * aligner runs seed->chain->extend only on the REP pairs (compacted at the
 * worker_bwt level; kernels unmodified) and a parallel copy pass replicates each
 * REP's regs into its DUP pairs before pestat/pairing. The per-read SAM stage
 * then replays with each read's own id/name/qual, so output is byte-identical.
 * In auto mode the first chunks measure with the memo OFF (clean align-cost
 * baseline for the controller); the latch turns the copy path on thereafter.
 *
 * The design mirrors the shipped extension-DP dedup controller (BWAMEM3_DEDUP /
 * --dedup, "#415") deliberately: same off|on|auto surface, same net-cycles
 * self-calibrating controller (no user-facing dup-rate threshold), same
 * ends-only word-fingerprint + full-byte-verify, same env-knob family.
 */

#include "bwa.h"        /* bseq1_t (anonymous-struct typedef; cannot be fwd-declared) */
struct mem_opt_t;

/* Resolved mode. */
enum { READMEMO_OFF = 0, READMEMO_ON = 1, READMEMO_AUTO = 2 };

/* Per-pair role assigned by the pre-pass. NONE is an odd tail (never happens on
 * the PE-gated path); REP pairs are aligned; DUP pairs skip alignment and copy
 * their representative's regs. */
enum { READ_MEMO_ROLE_NONE = 0, READ_MEMO_ROLE_REP = 1, READ_MEMO_ROLE_DUP = 2 };

/* Resolve the mode from --dedup-reads (mode_arg) > BWAMEM3_DEDUP_READS env >
 * default "auto". Fatal (exit 1) on an unrecognized value. An empty CLI value is
 * rejected earlier by the getopt handler; an explicitly-empty env value is fatal
 * (cannot silently fall back to the default). Mirrors mem_dedup_configure. */
void mem_dedup_reads_configure(const char *mode_arg);

/* The resolved mode (lazy env/default resolution for entry points -- unit
 * binaries, library use -- that never call mem_dedup_reads_configure). */
int read_memo_mode(void);

/* Per-align-invocation working state. Owned by the module and grown-in-place
 * across invocations (sized from the pair count). Phase 1 fills role[]/rep_pair[]
 * but no consumer reads them. */
struct read_memo_state {
    int      npairs;       /* n/2 for the current invocation                 */
    uint8_t *role;         /* [npairs] 0=NONE (odd tail), 1=REP, 2=DUP        */
    int32_t *rep_pair;     /* [npairs] DUP pair -> its REP pair index; else -1 */
    int64_t  cap;          /* allocated capacity (pairs)                      */
};

/* Result of one pre-pass, fed to the controller. */
struct read_memo_result {
    int64_t  pairs;        /* pairs examined this invocation      */
    int64_t  dup_pairs;    /* pairs marked DUP (collapsible)       */
    uint64_t probe_ns;     /* measured pre-pass wall cost (ns)     */
};

/* Serial fingerprint pre-pass over the chunk's pairs (seqs[2i], seqs[2i+1]),
 * run on RAW (pre-kernel1) bases. Fills st->role / st->rep_pair (grows st as
 * needed) and returns counts + its own measured cost. Byte-verifies every
 * fingerprint hit (a collision costs a memcmp, never splits a duplicate set).
 * Caller must ensure n is even and > 0. */
read_memo_result read_memo_prepass(const mem_opt_t *opt, const bseq1_t *seqs,
                                   int n, read_memo_state *st);

/* Whether to arm the memo for the NEXT align invocation of `dup_pairs` duplicate
 * pairs. OFF mode / no duplicates -> 0; ON mode -> 1; auto -> the current latch,
 * except while measuring, when it alternates so the A/B controller gathers both
 * armed and unarmed samples. Call exactly once per invocation (it advances the
 * alternation counter) and pass the same value to read_memo_controller_observe. */
int read_memo_should_arm(int64_t dup_pairs);

/* Feed one invocation's pre-pass result, its measured align-phase cost (WORKER10
 * delta, ns), and -- when armed -- the measured regs copy-pass cost (ns) into the
 * two-sample A/B controller. It records this chunk's REALIZED per-pair total cost
 * (probe + align + copy) into the ON or OFF accumulator (per `armed`, which MUST
 * match read_memo_should_arm) and latches on the measured mean difference, so the
 * copy/compaction overhead is charged rather than assumed away. `copy_ns` is
 * ignored when `armed` is false (no copy pass ran). Controller runs in auto only. */
void read_memo_controller_observe(const read_memo_result &r, uint64_t align_ns,
                                  uint64_t copy_ns, bool armed);

/* Current controller decision for the memoize path:
 * READMEMO_ON while auto is latched-on or mode==on; READMEMO_OFF otherwise. */
int read_memo_active(void);

/* Diagnostic: BWAMEM3_DEDUP_READS_VERIFY=1 makes the armed path align DUP pairs
 * NORMALLY (no compaction) and, instead of copying, COMPARE each duplicate's
 * independently-computed regs against its representative's field-by-field --
 * i.e. it executes the position-invariance claim on real data. Non-zero only
 * when the env knob is set; production runs it off. Returns 1 if set. */
int read_memo_verify(void);

/* Shutdown teardown for the module's grown-in-place scratch (call after worker
 * threads join). read_memo_state_free releases a state's role[]/rep_pair[];
 * read_memo_teardown releases the module-scope chain scratch. Both idempotent. */
void read_memo_state_free(read_memo_state *st);
void read_memo_teardown(void);

/* Test-only: reset the process-global controller + stats to their initial
 * (unlatched, MEASURING) state so a unit test does not depend on prior state or
 * case ordering. Not used in production. */
void read_memo_reset_for_testing(void);

#endif /* READ_MEMO_H */
