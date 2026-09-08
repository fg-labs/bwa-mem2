/* stage_prof: per-chunk read/process/write timing for aligner --profile mode.
 * Self-contained; no dependency on the rest of bwa-mem3. See stage_prof.h. */
#include "stage_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <assert.h>

/* Defined unconditionally so the STAGE_PROF-off (compile-time-dead) instrumentation
 * blocks that reference it still link at any optimization level. */
__thread prof_chunk_t g_ktfor;   /* per-thread kt_for balance handoff (see header) */

#ifdef STAGE_PROF

int    sp_g_on = 0;   /* declared extern in stage_prof.h; read by inline sp_enabled() */
static char   g_path[4096];
static char   g_tool[64], g_version[64], g_arch[16], g_fmt[16], g_input[4096];
static int    g_cores, g_level, g_workers = 0;
static double g_run_start = 0.0;
static prof_chunk_t *g_rows = NULL;
static size_t  g_n = 0, g_cap = 0;
static double  g_idle = 0.0, g_idle_split[3] = {0,0,0};
static size_t  g_n_dropped = 0;   /* chunks dropped on realloc failure (OOM) */
static int     g_test_fail_realloc = 0;   /* test-only: pending forced realloc failures */
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

/* See stage_prof.h: arm the next `n` growth reallocs to fail deterministically. */
void sp_test_arm_realloc_fail(int n) { g_test_fail_realloc = n; }

double sp_wall(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
double sp_thread_cpu(void) {
    struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
double sp_run_elapsed(void) { return sp_wall() - g_run_start; }

void sp_init(const char *path, const char *tool, const char *version,
             const char *arch, int cores, const char *fmt, int level, const char *input) {
    if (!path || !*path) return;
    snprintf(g_path,    sizeof g_path,    "%s", path);
    snprintf(g_tool,    sizeof g_tool,    "%s", tool ? tool : "");
    snprintf(g_version, sizeof g_version, "%s", version ? version : "");
    snprintf(g_arch,    sizeof g_arch,    "%s", arch ? arch : "");
    snprintf(g_fmt,     sizeof g_fmt,     "%s", fmt ? fmt : "");
    snprintf(g_input,   sizeof g_input,   "%s", input ? input : "");
    g_cores = cores; g_level = level; g_run_start = sp_wall();
    /* The caller (main_mem) can run more than once in a single process (e.g. a
     * host embedding bwa-mem3 as a library, or repeated invocations in one test
     * binary); reset every run-scoped accumulator so a fresh run starts clean
     * instead of inheriting counts/timings left over from a prior run. The
     * chunk buffer itself (g_rows/g_cap) is intentionally kept allocated and
     * reused -- only its logical length (g_n) resets. */
    g_n = 0; g_n_dropped = 0; g_idle = 0.0;
    g_idle_split[0] = g_idle_split[1] = g_idle_split[2] = 0.0;
    sp_g_on = 1;
}
void sp_set_workers(int n) { g_workers = n; }

void sp_chunk_init(prof_chunk_t *c) {
    memset(c, 0, sizeof *c);
    c->read_diskwait = c->read_decompress = c->read_parse = NAN;
    c->compute = c->encode = NAN;
    c->thr_busy_min = c->thr_busy_max = c->thr_busy_mean = c->thr_busy_stdev = NAN;
    c->write_compress = c->write_diskwrite = NAN;
    c->chunk_start = NAN;
    c->read_bytes_in = c->bgzf_blocks = -1;   /* -1 -> blank (N/A) */
}

void sp_add_chunk(const prof_chunk_t *c) {
    if (!sp_g_on) return;
    int lk = pthread_mutex_lock(&g_mu); assert(lk == 0); (void)lk;
    if (g_n == g_cap) {
        size_t new_cap = g_cap ? g_cap * 2 : 64;
        prof_chunk_t *new_rows;
        if (g_test_fail_realloc > 0) {   /* test-only fault injection: mimic realloc == NULL */
            --g_test_fail_realloc;
            new_rows = NULL;             /* g_rows left untouched, exactly as on a real failure */
        } else {
            new_rows = (prof_chunk_t*)realloc(g_rows, new_cap * sizeof *new_rows);
        }
        if (!new_rows) {   /* keep old g_rows intact; drop this chunk rather than crash */
            ++g_n_dropped;   /* count it so sp_finish can report the report is incomplete */
            int ul = pthread_mutex_unlock(&g_mu); assert(ul == 0); (void)ul;
            return;
        }
        g_rows = new_rows;
        g_cap = new_cap;
    }
    g_rows[g_n++] = *c;
    int ul = pthread_mutex_unlock(&g_mu); assert(ul == 0); (void)ul;
}

void sp_add_idle(int next_step, double s) {
    if (!sp_g_on) return;
    int lk = pthread_mutex_lock(&g_mu); assert(lk == 0); (void)lk;
    g_idle += s;
    if (next_step >= 0 && next_step < 3) g_idle_split[next_step] += s;
    int ul = pthread_mutex_unlock(&g_mu); assert(ul == 0); (void)ul;
}

void sp_thread_stats(prof_chunk_t *c, const double *b, int n) {
    if (n <= 0) return;
    double mn = b[0], mx = b[0], sum = 0;
    for (int i = 0; i < n; i++) { if (b[i] < mn) mn = b[i]; if (b[i] > mx) mx = b[i]; sum += b[i]; }
    double mean = sum / n, var = 0;
    for (int i = 0; i < n; i++) { double d = b[i] - mean; var += d * d; }
    c->thr_busy_min = mn; c->thr_busy_max = mx;
    c->thr_busy_mean = mean; c->thr_busy_stdev = sqrt(var / n);
}

/* ---- per-thread read-stage accumulators ---- */
static __thread double tl_disk = 0.0, tl_dec = 0.0, tl_parse = 0.0;
static __thread long   tl_fdbytes = 0, tl_blocks = 0;
void sp_read_reset(void) { tl_disk = tl_dec = tl_parse = 0.0; tl_fdbytes = 0; tl_blocks = 0; }
void sp_read_add(int which, double s) {
    if (which == SP_READ_DISKWAIT) tl_disk += s;
    else if (which == SP_READ_DECOMPRESS) tl_dec += s;
    else tl_parse += s;   /* SP_READ_PARSE */
}
void sp_read_get(double *d, double *c, double *p) {
    if (d) *d = tl_disk; if (c) *c = tl_dec; if (p) *p = tl_parse;
}
void sp_read_bytes(long fd_bytes, long blocks) { tl_fdbytes += fd_bytes; tl_blocks += blocks; }
void sp_read_get_bytes(long *fd_bytes, long *blocks) {
    if (fd_bytes) *fd_bytes = tl_fdbytes; if (blocks) *blocks = tl_blocks;
}

/* ---- per-thread encode accumulator ---- */
static __thread double tl_enc = 0.0;
void   sp_encode_reset(void) { tl_enc = 0.0; }
void   sp_encode_add(double s) { tl_enc += s; }
double sp_encode_get(void) { return tl_enc; }

/* ---- emit ---- */
static void cell(FILE *f, double v) { if (isnan(v)) fputc('\t', f); else fprintf(f, "%.4f\t", v); }
static void lcell(FILE *f, long v)  { if (v < 0) fputc('\t', f); else fprintf(f, "%ld\t", v); }

static void emit_row(FILE *f, const prof_chunk_t *c, int agg,
                     double total_wall, double mcb, double rss) {
    fprintf(f, "%s\t%s\t%s\t%d\t%d\t%s\t%d\t%s\t",
            g_tool, g_version, g_arch, g_cores, g_workers, g_fmt, g_level, g_input);
    if (agg) fputs("ALL\t", f); else fprintf(f, "%ld\t", c->chunk);
    cell(f, c->chunk_start);
    fprintf(f, "%ld\t%ld\t", c->n_reads, c->n_bp);
    cell(f, c->read_wall); cell(f, c->read_diskwait); cell(f, c->read_decompress); cell(f, c->read_parse);
    lcell(f, c->read_bytes_in); lcell(f, c->bgzf_blocks);
    cell(f, c->proc_wall); cell(f, c->proc_cpu); cell(f, c->compute); cell(f, c->encode);
    cell(f, c->thr_busy_min); cell(f, c->thr_busy_max); cell(f, c->thr_busy_mean); cell(f, c->thr_busy_stdev);
    cell(f, c->write_wall); cell(f, c->write_compress); cell(f, c->write_diskwrite);
    fprintf(f, "%ld\t", c->write_bytes);
    if (agg) fprintf(f, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.1f\n",
                     total_wall, g_idle, g_idle_split[0], g_idle_split[1], g_idle_split[2], mcb, rss);
    else     fputs("\t\t\t\t\t\t\n", f);   /* 7 aggregate-only cols blank on chunk rows */
}

void sp_finish(double total_wall, double mean_cores_busy, double peak_rss_mb) {
    if (!sp_g_on) return;
    FILE *f = fopen(g_path, "w");
    if (!f) { fprintf(stderr, "[stage_prof] cannot open %s for writing\n", g_path); return; }
    fputs("tool\tversion\tarch\tcores\tn_pipeline_workers\toutput_format\tcompression_level\tinput\t"
          "chunk\tchunk_start\tn_reads\tn_bp\t"
          "read_wall\tread_diskwait\tread_decompress\tread_parse\tread_bytes_in\tbgzf_blocks\t"
          "proc_wall\tproc_cpu\tcompute\tencode\tthr_busy_min\tthr_busy_max\tthr_busy_mean\tthr_busy_stdev\t"
          "write_wall\twrite_compress\twrite_diskwrite\twrite_bytes\t"
          "total_wall\tidle_worker_s\tidle_read_s\tidle_proc_s\tidle_write_s\tmean_cores_busy\tpeak_rss_mb\n", f);
    prof_chunk_t agg; memset(&agg, 0, sizeof agg);   /* sums start at 0 */
    agg.read_bytes_in = agg.bgzf_blocks = -1;        /* -1 -> blank until a chunk supplies a real value */
    double sum_proc_cpu = 0.0;
    for (size_t i = 0; i < g_n; i++) {
        emit_row(f, &g_rows[i], 0, 0, 0, 0);
        agg.n_reads += g_rows[i].n_reads;
        agg.n_bp    += g_rows[i].n_bp;
        agg.write_bytes += g_rows[i].write_bytes;
        if (g_rows[i].read_bytes_in >= 0) {
            if (agg.read_bytes_in < 0) agg.read_bytes_in = 0;
            agg.read_bytes_in += g_rows[i].read_bytes_in;
        }
        if (g_rows[i].bgzf_blocks >= 0) {
            if (agg.bgzf_blocks < 0) agg.bgzf_blocks = 0;
            agg.bgzf_blocks += g_rows[i].bgzf_blocks;
        }
        agg.read_wall  += g_rows[i].read_wall;
        agg.proc_wall  += g_rows[i].proc_wall;
        agg.write_wall += g_rows[i].write_wall;
        if (!isnan(g_rows[i].proc_cpu)) sum_proc_cpu += g_rows[i].proc_cpu;
    }
    if (total_wall > 0 && sum_proc_cpu > 0) mean_cores_busy = sum_proc_cpu / total_wall;
    /* aggregate: blank the per-tool sub-splits; keep summed walls + byte totals */
    agg.read_diskwait = agg.read_decompress = agg.read_parse = NAN;
    agg.compute = agg.encode = NAN;
    agg.thr_busy_min = agg.thr_busy_max = agg.thr_busy_mean = agg.thr_busy_stdev = NAN;
    agg.proc_cpu = NAN; agg.write_compress = agg.write_diskwrite = NAN; agg.chunk_start = NAN;
    emit_row(f, &agg, 1, total_wall, mean_cores_busy, peak_rss_mb);
    int write_failed = ferror(f);
    if (fclose(f) != 0 || write_failed) {
        fprintf(stderr, "[stage_prof] error writing profile output to %s\n", g_path);
        return;
    }
    if (g_n_dropped > 0)
        fprintf(stderr, "[stage_prof] wrote %zu chunk rows + aggregate to %s (%zu chunk(s) dropped, out of memory: report is incomplete)\n", g_n, g_path, g_n_dropped);
    else
        fprintf(stderr, "[stage_prof] wrote %zu chunk rows + aggregate to %s\n", g_n, g_path);
}

#endif /* STAGE_PROF */
