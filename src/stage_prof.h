#ifndef STAGE_PROF_H
#define STAGE_PROF_H
#include <stddef.h>

/* Force-inline so the (compile-time-constant) hooks leave no out-of-line copy
 * in any object file -- C `static inline` otherwise emits dead orphan bodies. */
#if defined(__GNUC__)
#define SP_INLINE static inline __attribute__((always_inline, unused))
#else
#define SP_INLINE static inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Per-pipeline-chunk stage breakdown for aligner profiling (--profile).
 *
 * Profiling is a COMPILE-TIME feature, gated on the STAGE_PROF macro:
 *   - Built WITHOUT -DSTAGE_PROF (the default): sp_enabled() is the compile-time
 *     constant 0, every sp_* hook below is an empty inline, and the --profile
 *     CLI option does not exist. All instrumentation constant-folds away, so the
 *     hot paths are byte-identical to an un-instrumented build (zero overhead).
 *   - Built WITH -DSTAGE_PROF (e.g. `make STAGE_PROF=1`): the real out-of-line
 *     implementation in stage_prof.cpp is linked and --profile is available.
 *
 * The aggregate row carries "ALL" in the chunk column (per-chunk rows are
 * numbered from 0). All times are in seconds. Fields a tool/
 * format cannot separate are NaN (see sp_chunk_init) and printed as empty cells,
 * so the schema stays rectangular across bwa-mem3 and minibwa.
 *
 * Per-tool/format feasibility (documented; not faked):
 *   read_diskwait/read_decompress/read_parse:
 *       bwa-mem3 = all three (fast_reader splits read/inflate/parse);
 *                  legacy reader leaves them NaN (uninstrumented).
 *       minibwa  = read_diskwait NaN; read_decompress = fused gzread
 *                  (disk+inflate+tokenize); read_parse = kseq2bseq copy.
 *   read_bytes_in / bgzf_blocks:
 *       bwa-mem3 = bytes read from the fd (compressed-on-disk) + BGZF block count
 *                  (0 for plain/gzip); minibwa = NaN (zlib gzFile hides them).
 *   encode:
 *       bwa-mem3 = accurate clock around the SAM/BAM build (step 1);
 *       minibwa  = accurate (mb_format, step 2).
 *   write_compress/write_diskwrite:
 *       SAM          = write_compress 0, write_diskwrite clean.
 *       bwa-mem3 BAM = fused in htslib sam_write1 -> write_compress; diskwrite NaN.
 */
typedef struct {
    long   chunk, n_reads, n_bp, write_bytes, read_bytes_in, bgzf_blocks;
    double chunk_start;   /* wall offset of this chunk's read-start from run start */
    double read_wall, read_diskwait, read_decompress, read_parse;
    double proc_wall, proc_cpu, compute, encode;
    double thr_busy_min, thr_busy_max, thr_busy_mean, thr_busy_stdev;
    double write_wall, write_compress, write_diskwrite;
} prof_chunk_t;

/* Per-thread compute balance for a kt_for fan-out (see stage_prof.cpp). Declared
 * unconditionally so the (compile-time-dead, STAGE_PROF-off) instrumentation
 * blocks that name it still type-check; defined unconditionally in stage_prof.cpp. */
extern __thread prof_chunk_t g_ktfor;

/* sp_read_add() sub-stage buckets. Declared unconditionally (outside STAGE_PROF)
 * so call sites name them in both the instrumented and no-op builds. */
enum {
    SP_READ_DISKWAIT   = 0,   /* time blocked on the underlying read()    */
    SP_READ_DECOMPRESS = 1,   /* time in gzip inflate / BGZF block decode */
    SP_READ_PARSE      = 2,   /* time tokenizing records into bseq1_t     */
};

#ifdef STAGE_PROF

/* The on/off flag is exposed so sp_enabled() is a header inline: when --profile
 * is off (but STAGE_PROF compiled in) this collapses to a single global load +
 * branch at every call site (no out-of-line call) even in non-LTO builds. */
extern int sp_g_on;
SP_INLINE int sp_enabled(void) { return sp_g_on; }

/* Arm profiling; captures the run-start wall anchor. No-op if path NULL/empty. */
void   sp_init(const char *path, const char *tool, const char *version,
               const char *arch, int cores, const char *output_format,
               int compression_level, const char *input);
/* Record the pipeline-worker count (run-level context; the suspected ceiling). */
void   sp_set_workers(int n_pipeline_workers);

double sp_wall(void);          /* CLOCK_MONOTONIC seconds */
double sp_thread_cpu(void);    /* CLOCK_THREAD_CPUTIME_ID seconds (calling thread) */
double sp_run_elapsed(void);   /* sp_wall() - run-start anchor (for chunk_start) */

void   sp_chunk_init(prof_chunk_t *c);
void   sp_add_chunk(const prof_chunk_t *c);
/* Test-only fault-injection seam: arm the next `n` chunk-buffer growths in
 * sp_add_chunk to behave as if realloc returned NULL, so the out-of-memory
 * drop-and-count path (g_n_dropped plus the "report is incomplete" diagnostic)
 * can be exercised deterministically without depending on host OOM. Present only
 * in STAGE_PROF builds; the aligner never calls it. */
void   sp_test_arm_realloc_fail(int n);
/* Accumulate pipeline-worker idle seconds, attributed to the step it was about
 * to run (0=read 1=process 2=write); also adds to the total. */
void   sp_add_idle(int next_step, double seconds);
void   sp_thread_stats(prof_chunk_t *c, const double *busy, int n);
void   sp_finish(double total_wall, double mean_cores_busy, double peak_rss_mb);

/* Per-thread read-stage accumulators (the step-0 worker owns these).
 * sp_read_add()'s `which` selects the sub-stage bucket (see SP_READ_* above);
 * sp_read_bytes() adds fd/compressed bytes and BGZF block counts harvested via
 * sp_read_get_bytes(). */
void   sp_read_reset(void);
void   sp_read_add(int which, double seconds);
void   sp_read_get(double *diskwait, double *decompress, double *parse);
void   sp_read_bytes(long fd_bytes, long bgzf_blocks);
void   sp_read_get_bytes(long *fd_bytes, long *bgzf_blocks);

/* Per-thread encode accumulator (bwa-mem3 brackets the SAM/BAM build deep in
 * mem_process_seqs). sp_encode_reset() at step-1 entry; sp_encode_get() after. */
void   sp_encode_reset(void);
void   sp_encode_add(double seconds);
double sp_encode_get(void);

#else  /* !STAGE_PROF: profiling compiled out -> every hook is a no-op inline. */

/* sp_enabled() is a compile-time 0, so `if (sp_enabled()) { ... }` and
 * `sp_enabled() ? sp_wall() : 0.0` constant-fold to nothing. Parameters are
 * named and (void)-cast so these compile cleanly as C (no unnamed-parameter or
 * unused-parameter diagnostics) in the C translation units that include this. */
SP_INLINE int    sp_enabled(void) { return 0; }
SP_INLINE void   sp_init(const char *path, const char *tool, const char *version,
                             const char *arch, int cores, const char *output_format,
                             int compression_level, const char *input) {
    (void)path; (void)tool; (void)version; (void)arch; (void)cores;
    (void)output_format; (void)compression_level; (void)input;
}
SP_INLINE void   sp_set_workers(int n_pipeline_workers) { (void)n_pipeline_workers; }
SP_INLINE double sp_wall(void) { return 0.0; }
SP_INLINE double sp_thread_cpu(void) { return 0.0; }
SP_INLINE double sp_run_elapsed(void) { return 0.0; }
SP_INLINE void   sp_chunk_init(prof_chunk_t *c) { (void)c; }
SP_INLINE void   sp_add_chunk(const prof_chunk_t *c) { (void)c; }
SP_INLINE void   sp_add_idle(int next_step, double seconds) { (void)next_step; (void)seconds; }
SP_INLINE void   sp_thread_stats(prof_chunk_t *c, const double *busy, int n) {
    (void)c; (void)busy; (void)n;
}
SP_INLINE void   sp_finish(double total_wall, double mean_cores_busy, double peak_rss_mb) {
    (void)total_wall; (void)mean_cores_busy; (void)peak_rss_mb;
}
SP_INLINE void   sp_read_reset(void) {}
SP_INLINE void   sp_read_add(int which, double seconds) { (void)which; (void)seconds; }
SP_INLINE void   sp_read_get(double *diskwait, double *decompress, double *parse) {
    (void)diskwait; (void)decompress; (void)parse;
}
SP_INLINE void   sp_read_bytes(long fd_bytes, long bgzf_blocks) { (void)fd_bytes; (void)bgzf_blocks; }
SP_INLINE void   sp_read_get_bytes(long *fd_bytes, long *bgzf_blocks) { (void)fd_bytes; (void)bgzf_blocks; }
SP_INLINE void   sp_encode_reset(void) {}
SP_INLINE void   sp_encode_add(double seconds) { (void)seconds; }
SP_INLINE double sp_encode_get(void) { return 0.0; }

#endif /* STAGE_PROF */

#ifdef __cplusplus
}
#endif
#endif /* STAGE_PROF_H */
