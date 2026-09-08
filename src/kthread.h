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
         Heng Li <hli@jimmy.harvard.edu>
*****************************************************************************************/

#ifndef KTHREAD_HPP
#define KTHREAD_HPP

#include <stdint.h>
#include "macro.h"
#include "bwamem.h"
#include <pthread.h>

// ----------------
//  kt_pipeline() 
// -------------------

struct ktp_t;
struct mem_opt_t;
struct worker_t;

typedef struct {
	struct ktp_t *pl;
	int64_t index;
	int step;
	void *data;
	worker_t *w;
	mem_opt_t *opt;
	int i;
} ktp_worker_t;

typedef struct ktp_t {
	void *shared;
	int64_t index;
	int n_workers, n_steps;
	ktp_worker_t *workers;
	pthread_mutex_t mutex;
	pthread_cond_t cv;
} ktp_t;

// ---------------
// kt_for() 
// ---------------

struct kt_for_t;

/* L27: pad each worker to a full cache line so adjacent workers' steal counter
 * `i` (bumped with __sync_fetch_and_add during the end-of-pass steal phase) do
 * not share a line and false-share. Layout-only: no field is added or reordered
 * that any code reads, so output is byte-identical. The backing array is
 * allocated cache-line-aligned (see kthread.cpp) so the stride actually lands
 * each element on its own line. */
typedef struct {
	struct kt_for_t *t;
	long i;
	double cpu_busy;   /* stage_prof: per-thread CLOCK_THREAD_CPUTIME seconds (--profile) */
	double encode;     /* stage_prof: per-thread SAM/BAM-build CPU seconds */
} __attribute__((aligned(64))) ktf_worker_t;

typedef struct kt_for_t {
	int n_threads;
	long n;
	ktf_worker_t *w;
	void (*func)(void*, int, int, int);
	void *data;
} kt_for_t;


void kt_for(void (*func)(void*,int,int,int), void *data, int n);
/* Tear down the persistent kt_for() worker pool (joins the worker threads).
 * Call once after the last kt_for()/kt_pipeline() of a run; a no-op if the pool
 * was never created. */
void kt_pool_destroy(void);
#endif
