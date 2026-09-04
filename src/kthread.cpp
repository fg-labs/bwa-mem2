/*************************************************************************************
                           The MIT License

   Copyright Attractive Chaos <attractor@live.co.uk>
   BWA-MEM2  (Sequence alignment using Burrows-Wheeler Transform),

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

   Modified Copyright (C) 2019  Intel Corporation, Heng Li.
   Authors: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>;
         Heng Li <hli@jimmy.harvard.edu>.
*****************************************************************************************/

#include "kthread.h"
#include "stage_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


/* Apple Silicon QoS (Quality of Service) support
 * This helps the scheduler preferentially place compute threads on P-cores
 * (performance cores) rather than E-cores (efficiency cores) */
#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/sysctl.h>

/* Get the number of performance cores on Apple Silicon
 * Returns -1 if unable to detect (e.g., on Intel Macs) */
static int get_pcore_count() {
    int pcore_count = 0;
    size_t size = sizeof(pcore_count);

    /* Try Apple Silicon specific sysctl first */
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcore_count, &size, NULL, 0) == 0) {
        return pcore_count;
    }

    /* Fallback: assume half of physical cores are P-cores on Apple Silicon,
     * or return -1 on Intel (no hybrid cores) */
    int total_cores = 0;
    size = sizeof(total_cores);
    if (sysctlbyname("hw.physicalcpu", &total_cores, &size, NULL, 0) == 0) {
        /* Check if this is Apple Silicon by looking for perflevel */
        int levels = 0;
        size = sizeof(levels);
        if (sysctlbyname("hw.nperflevels", &levels, &size, NULL, 0) == 0 && levels > 1) {
            /* Apple Silicon with hybrid cores - rough estimate */
            return total_cores / 2;
        }
    }
    return -1;  /* Intel Mac or detection failed */
}
#endif /* __APPLE__ */

extern uint64_t tprof[LIM_R][LIM_C];

static inline long steal_work(kt_for_t *t)
{
	int i, min_i = -1;
	long k, min = LONG_MAX;
	for (i = 0; i < t->n_threads; ++i)
		if (min > t->w[i].i) min = t->w[i].i, min_i = i;
	k = __sync_fetch_and_add(&t->w[min_i].i, t->n_threads);
	// return k >= t->n? -1 : k;
	return k*BATCH_SIZE >= t->n? -1 : k;
}

/* The per-thread work loop: strided dispatch followed by work-stealing.
 * Extracted from the old per-call thread body so it can be driven by a
 * persistent pool worker (see below). Distribution is byte-for-byte the
 * legacy behaviour; tid passed to func() is the worker's slot index
 * (w - w->t->w). */
static void ktf_run(ktf_worker_t *w)
{
	long i;
	double _c0 = sp_enabled() ? sp_thread_cpu() : 0.0;
	if (sp_enabled()) sp_encode_reset();


	for (;;) {
		i = __sync_fetch_and_add(&w->i, w->t->n_threads);
		int st = i * BATCH_SIZE;
		if (st >= w->t->n) break;
		int ed = (i + 1) * BATCH_SIZE < w->t->n? (i + 1) * BATCH_SIZE : w->t->n;
		w->t->func(w->t->data, st, ed-st, w - w->t->w);
	}

	while ((i = steal_work(w->t)) >= 0) {
		int st = i * BATCH_SIZE;
		int ed = (i + 1) * BATCH_SIZE < w->t->n? (i + 1) * BATCH_SIZE : w->t->n;
		w->t->func(w->t->data, st, ed-st, w - w->t->w);
	}
	if (sp_enabled()) { w->cpu_busy = sp_thread_cpu() - _c0; w->encode = sp_encode_get(); }
}

/* ---------------------------------------------------------------------------
 * Persistent worker pool for kt_for().
 *
 * The original kt_for() spawned n worker pthreads on every call and joined
 * them at the end. bwa-mem3 calls kt_for() three times per chunk (worker_bwt /
 * worker_aln / worker_sam) across many chunks, so a per-thread scratch buffer
 * (e.g. mmc->enc_qdb[tid]) is malloc()'d by one chunk's worker thread and then
 * realloc()'d by a *different* OS thread on a later chunk — the first thread
 * has already exited. That cross-thread realloc of an abandoned-heap block is
 * mishandled by mimalloc v3.x and silently corrupts the heap, crashing with
 * SIGSEGV several chunks later (it is benign under glibc / mimalloc v2.x, which
 * is why no sanitizer flags it).
 *
 * A persistent pool keeps the same n worker threads alive for the whole run, so
 * worker slot `tid` is always the same OS thread: every realloc of that slot's
 * buffers happens on the thread that allocated them — no cross-thread realloc.
 * It also removes the per-chunk thread create/join overhead.
 * ------------------------------------------------------------------------- */
typedef struct {
	int started;
	int n_threads;
	pthread_t *threads;
	ktf_worker_t *w;          /* per-worker steal state (n_threads entries) */
	kt_for_t job;             /* current job: func / data / n / w           */
	pthread_mutex_t mtx;
	pthread_cond_t cv_go;     /* signalled when a new generation is posted  */
	pthread_cond_t cv_done;   /* signalled when the last worker finishes    */
	long generation;          /* bumped once per kt_for() call              */
	int n_left;               /* workers not yet done with `generation`     */
	int shutdown;
} kt_pool_t;

static kt_pool_t g_kt_pool = {0};

static void *kt_pool_worker(void *arg)
{
	long k = (long)(intptr_t)arg;
	long seen = 0;
	for (;;) {
		pthread_mutex_lock(&g_kt_pool.mtx);
		while (g_kt_pool.generation == seen && !g_kt_pool.shutdown)
			pthread_cond_wait(&g_kt_pool.cv_go, &g_kt_pool.mtx);
		if (g_kt_pool.shutdown) { pthread_mutex_unlock(&g_kt_pool.mtx); break; }
		seen = g_kt_pool.generation;
		pthread_mutex_unlock(&g_kt_pool.mtx);

		ktf_run(&g_kt_pool.w[k]);

		pthread_mutex_lock(&g_kt_pool.mtx);
		if (--g_kt_pool.n_left == 0) pthread_cond_signal(&g_kt_pool.cv_done);
		pthread_mutex_unlock(&g_kt_pool.mtx);
	}
	return NULL;
}

static void kt_pool_init(int n_threads)
{
	g_kt_pool.n_threads = n_threads;
	g_kt_pool.threads = (pthread_t*) malloc(n_threads * sizeof(pthread_t));
	/* L27: cache-line-aligned so each worker's steal counter sits on its own
	 * line (sizeof(ktf_worker_t) is padded to 64, so n_threads*sizeof is a
	 * multiple of 64 as aligned_alloc requires). free() handles it. */
	g_kt_pool.w       = (ktf_worker_t*) aligned_alloc(64, n_threads * sizeof(ktf_worker_t));
	if (g_kt_pool.threads == NULL || g_kt_pool.w == NULL) {
		perror("Allocation of kt_for worker pool failed");
		exit(EXIT_FAILURE);
	}
	pthread_mutex_init(&g_kt_pool.mtx, NULL);
	pthread_cond_init(&g_kt_pool.cv_go, NULL);
	pthread_cond_init(&g_kt_pool.cv_done, NULL);
	g_kt_pool.generation = 0;
	g_kt_pool.n_left = 0;
	g_kt_pool.shutdown = 0;

	pthread_attr_t attr;
	pthread_attr_init(&attr);
#ifdef __APPLE__
	/* Prefer P-cores on Apple Silicon for compute-intensive alignment work. */
	int pcore_count = get_pcore_count();
	if (pcore_count > 0)
		pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INITIATED, 0);
#endif
	for (int i = 0; i < n_threads; ++i) {
		/* A failed worker would leave kt_for() waiting on cv_done for a
		 * completion that never arrives (deadlock), so fail loudly instead.
		 * pthread_create() returns the error code directly rather than via
		 * errno, so report it with strerror(rc), not perror(). */
		int rc = pthread_create(&g_kt_pool.threads[i], &attr, kt_pool_worker, (void*)(intptr_t)i);
		if (rc != 0) {
			fprintf(stderr, "ERROR: kt_for worker pool: pthread_create failed (worker %d): %s\n", i, strerror(rc));
			exit(EXIT_FAILURE);
		}
	}
	pthread_attr_destroy(&attr);
	g_kt_pool.started = 1;
}

/* Tear the pool down (joins the workers). Safe to call when no pool exists. */
void kt_pool_destroy(void)
{
	if (!g_kt_pool.started) return;
	pthread_mutex_lock(&g_kt_pool.mtx);
	g_kt_pool.shutdown = 1;
	pthread_cond_broadcast(&g_kt_pool.cv_go);
	pthread_mutex_unlock(&g_kt_pool.mtx);
	for (int i = 0; i < g_kt_pool.n_threads; ++i)
		pthread_join(g_kt_pool.threads[i], NULL);
	free(g_kt_pool.threads);
	free(g_kt_pool.w);
	pthread_mutex_destroy(&g_kt_pool.mtx);
	pthread_cond_destroy(&g_kt_pool.cv_go);
	pthread_cond_destroy(&g_kt_pool.cv_done);
	memset(&g_kt_pool, 0, sizeof(g_kt_pool));
}

void kt_for(void (*func)(void*, int, int, int), void *data, int n)
{
	worker_t *w = (worker_t*) data;
	int n_threads = w->nthreads;
	if (n_threads < 1) n_threads = 1;  /* nthreads is int16_t; never spin up a 0/negative pool */

	/* kt_for() is only ever called from the serialized step-1 of kt_pipeline
	 * (one chunk in flight at a time), so lazy init here is race-free, and
	 * n_threads (= opt->n_threads) is constant for the whole run. Rebuild the
	 * pool defensively if that invariant ever changes. */
	if (!g_kt_pool.started)
		kt_pool_init(n_threads);
	else if (g_kt_pool.n_threads != n_threads) {
		kt_pool_destroy();
		kt_pool_init(n_threads);
	}

	pthread_mutex_lock(&g_kt_pool.mtx);
	g_kt_pool.job.func = func;
	g_kt_pool.job.data = data;
	g_kt_pool.job.n = n;
	g_kt_pool.job.n_threads = n_threads;
	g_kt_pool.job.w = g_kt_pool.w;
	for (int i = 0; i < n_threads; ++i) {
		g_kt_pool.w[i].t = &g_kt_pool.job;
		g_kt_pool.w[i].i = i;
	}
	g_kt_pool.n_left = n_threads;
	++g_kt_pool.generation;
	pthread_cond_broadcast(&g_kt_pool.cv_go);
	while (g_kt_pool.n_left > 0)
		pthread_cond_wait(&g_kt_pool.cv_done, &g_kt_pool.mtx);
	pthread_mutex_unlock(&g_kt_pool.mtx);

	/* All workers are done with this generation (n_left == 0) and will not
	 * touch their per-slot stats again until the next kt_for() call, so the
	 * pool's worker slots can be read without the lock here. */
	if (sp_enabled()) {
		double *busy = (double*) malloc(n_threads * sizeof(double));
		assert(busy != NULL);
		double sum = 0, esum = 0;
		for (int i = 0; i < n_threads; ++i) { busy[i] = g_kt_pool.w[i].cpu_busy; sum += busy[i]; esum += g_kt_pool.w[i].encode; }
		g_ktfor.proc_cpu += sum;                        /* accumulate across kt_for calls in a step */
		g_ktfor.encode   += esum;                       /* SAM/BAM-build CPU (only worker_sam adds) */
		sp_thread_stats(&g_ktfor, busy, n_threads);     /* balance stats from the most recent call */
		free(busy);
	}
}
