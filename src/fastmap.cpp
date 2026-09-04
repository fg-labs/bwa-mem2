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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> /* strcasecmp(): POSIX declares it here, not in string.h */
#include "bwa_madvise.h"
#if NUMA_ENABLED
#include <numa.h>
#endif
#include <sstream>
#include <getopt.h>
#include <errno.h>    /* errno/ERANGE: strtoll validation of the INT options */
#include <unistd.h>   /* access(): --compat's "you passed a file" diagnostic */
#ifdef BWA_MEM3_DEBUG_RESCUE_STATS
#  include <atomic>
#  include <cstdint>
#endif
#include "fastmap.h"
#include "read_memo.h"
#include "FMI_search.h"
#include "bam_writer.h"
#include "meth_bam.h"
#include "meth_xm.h"   /* meth_chem_t for --meth=emseq|taps */
#include "stage_prof.h"
#include "seed_order.h"
#include "version.h"
#include <sys/resource.h>
#include "bwa_shm.h"
#include "bwa_hugepages.h"
#include "fast_reader_bseq.h"

#if AFF && (__linux__)
#include <sys/sysinfo.h>
int affy[256];
#endif

// --------------
extern uint64_t tprof[LIM_R][LIM_C];
// ---------------

/* --cohort-slices / BWA_MEM3_COHORT_SLICES. Named here rather than repeated at
 * each site because the value is needed in four places -- the ramp's shift
 * clamp, the flag's range check, that check's error message, and the `mem
 * --help` default -- and a help text or a validator that disagrees with the
 * ramp is exactly the drift this consolidates away.
 *
 * MAX is 20 because the ramp shift saturates there (task_size >> 20): a larger
 * value is indistinguishable from 20, so asking for one is a mistake worth
 * naming rather than silently accepting. See the ramp in the read step for how
 * the default is derived. */
#define COHORT_SLICES_DEFAULT 6
#define COHORT_SLICES_MAX     20

/* Ramp SHAPE defaults (--cohort-ramp-first / --cohort-ramp-ratio). At file scope
 * for the same reason as the constants above: usage() is defined before
 * main_mem, so a function-local default cannot be printed in `mem --help` and
 * would have to be duplicated as a literal there. See the derivation at the
 * cohort_ramp_ratio declaration in main_mem.
 *
 * Kept ABOVE the architecture split below, like the constants above: these are
 * read by kt_pipeline, usage() and main_mem on every architecture, so defining
 * them inside the ARM branch would leave them undeclared on x86. */
static const double  cohort_ramp_ratio_default = 1.6;
static const int64_t cohort_ramp_first_default = 16000000;

/* Strict parsers shared by the batch-shaping options (--chunk-cap,
 * --cohort-slices, --cohort-ramp-ratio, --cohort-ramp-first) and by each one's
 * BWA_MEM3_* env override.
 *
 * Every one of those options had to reject what atoll()/atof() silently accept:
 * a typo parses as 0, and 0 means "off" or "use the default" for all of them,
 * so an unvalidated parse quietly disables the very thing the option was set to
 * configure -- and atoll("16M") is 16, a SIXTEEN-BYTE first slice rather than
 * 16 Mbases. That made every site carry the same errno/end-pointer/range
 * checks, which is how one site once shipped a bare atoll() while its siblings
 * were already hardened, and how the env spelling of --cohort-slices came to
 * hardcode its upper bound instead of naming COHORT_SLICES_MAX. One
 * implementation each removes both failure modes.
 *
 * Both return 0 on success and -1 on rejection, leaving *out untouched, so the
 * caller keeps its own choice of fatal (CLI, nothing loaded yet) versus warn
 * and carry on (env, index already read). */

/* Reject unless `s` is a complete decimal integer within [min, max]. */
static int parse_bounded_i64(const char *s, int64_t min, int64_t max, int64_t *out)
{
    char *end = NULL;
    errno = 0;
    const long long v = strtoll(s, &end, 10);
    if (end == s || end == NULL || *end != '\0' || errno == ERANGE ||
        (int64_t)v < min || (int64_t)v > max)
        return -1;
    *out = (int64_t)v;
    return 0;
}

/* Reject unless `s` is a complete floating-point number. Range is deliberately
 * NOT checked: the ramp-ratio sites share a documented `<= 1` fallback that
 * reports and substitutes the default rather than failing, so only parseability
 * belongs here. */
static int parse_full_double(const char *s, double *out)
{
    char *end = NULL;
    errno = 0;
    const double v = strtod(s, &end);
    if (end == s || end == NULL || *end != '\0' || errno == ERANGE)
        return -1;
    *out = v;
    return 0;
}

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
/* ARM/Apple Silicon - no CPUID instruction, use system calls */

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

void __cpuid(unsigned int i, unsigned int cpuid[4]) {
    /* ARM doesn't have CPUID - return zeros */
    cpuid[0] = cpuid[1] = cpuid[2] = cpuid[3] = 0;
}

/* Get L2 cache size in bytes (for dynamic tuning) */
static int64_t get_l2_cache_size() {
#ifdef __APPLE__
    int64_t l2_size = 0;
    size_t size = sizeof(l2_size);
    if (sysctlbyname("hw.l2cachesize", &l2_size, &size, NULL, 0) == 0) {
        return l2_size;
    }
#endif
    return 4 * 1024 * 1024;  /* Default 4MB */
}

/* Get optimal batch size based on L2 cache */
int get_dynamic_batch_size() {
    int64_t l2_size = get_l2_cache_size();
    /* Heuristic: ~1KB working set per read pair, use 1/4 of L2 for batching
     * to leave room for other data structures */
    int batch_size = (int)(l2_size / 4 / 1024);
    /* Clamp to reasonable range */
    if (batch_size < 256) batch_size = 256;
    if (batch_size > 4096) batch_size = 4096;
    /* Round down to power of 2 for alignment */
    int pow2 = 256;
    while (pow2 * 2 <= batch_size) pow2 *= 2;
    return pow2;
}

int HTStatus()
{
    /* ARM/Apple Silicon doesn't have hyperthreading in the x86 sense.
     * Apple Silicon has P-cores and E-cores, which we handle differently.
     * Return 0 to indicate no HT (we handle core types via QoS instead).
     */
#ifdef __APPLE__
    int pcore_count = 0, ecore_count = 0;
    size_t size = sizeof(int);

    /* Try to get P-core and E-core counts on Apple Silicon */
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcore_count, &size, NULL, 0) == 0) {
        sysctlbyname("hw.perflevel1.physicalcpu", &ecore_count, &size, NULL, 0);
        fprintf(stderr, "Platform vendor: Apple Silicon.\n");
        fprintf(stderr, "P-cores: %d, E-cores: %d\n", pcore_count, ecore_count);
    } else {
        /* Fallback for older macOS or non-Apple Silicon */
        int total_cores = 0;
        sysctlbyname("hw.physicalcpu", &total_cores, &size, NULL, 0);
        fprintf(stderr, "Platform: ARM64, Physical CPUs: %d\n", total_cores);
    }

    /* Report L2 cache and dynamic batch size */
    int64_t l2_size = get_l2_cache_size();
    int dynamic_batch = get_dynamic_batch_size();
    fprintf(stderr, "L2 Cache: %lld MB, Dynamic batch size: %d (compile-time: %d)\n",
            l2_size / (1024 * 1024), dynamic_batch, BATCH_SIZE);
#else
    fprintf(stderr, "Platform vendor: ARM64.\n");
#endif
    return 0;  /* No hyperthreading on ARM */
}

#else
/* x86 CPUID implementation */

void __cpuid(unsigned int i, unsigned int cpuid[4]) {
#ifdef _WIN32
    __cpuid((int *) cpuid, (int)i);

#else
    asm volatile
        ("cpuid" : "=a" (cpuid[0]), "=b" (cpuid[1]), "=c" (cpuid[2]), "=d" (cpuid[3])
            : "0" (i), "2" (0));
#endif
}


int HTStatus()
{
    unsigned int cpuid[4];
    char platform_vendor[12];
    __cpuid(0, cpuid);
    ((unsigned int *)platform_vendor)[0] = cpuid[1]; // B
    ((unsigned int *)platform_vendor)[1] = cpuid[3]; // D
    ((unsigned int *)platform_vendor)[2] = cpuid[2]; // C
    std::string platform = std::string(platform_vendor, 12);

    __cpuid(1, cpuid);
    unsigned int platform_features = cpuid[3]; //D

    // __cpuid(1, cpuid);
    unsigned int num_logical_cpus = (cpuid[1] >> 16) & 0xFF; // B[23:16]
    // fprintf(stderr, "#logical cpus: ", num_logical_cpus);

    unsigned int num_cores = -1;
    if (platform == "GenuineIntel") {
        __cpuid(4, cpuid);
        num_cores = ((cpuid[0] >> 26) & 0x3f) + 1; //A[31:26] + 1
        fprintf(stderr, "Platform vendor: Intel.\n");
    } else  {
        fprintf(stderr, "Platform vendor unknown.\n");
    }

    // fprintf(stderr, "#physical cpus: ", num_cores);

    int ht = platform_features & (1 << 28) && num_cores < num_logical_cpus;
    if (ht)
        fprintf(stderr, "CPUs support hyperThreading !!\n");

    return ht;
}

#endif /* ARM vs x86 */


/*** Memory pre-allocations ***/
// Core allocation routine, parameterised only on mem_opt_t. Exposed so that
// library consumers which build up a worker_t themselves (e.g. language
// bindings that don't construct a ktp_aux_t) can reuse the exact same
// allocation sequence instead of re-implementing it and drifting out of sync.
void worker_alloc(const mem_opt_t *opt, worker_t &w, int32_t nreads, int32_t nthreads)
{
    assert(opt != NULL);
    assert(nreads >= 0);
    assert(nthreads > 0);
    /* The assert above compiles out under NDEBUG, and the chaining scratch is
     * now sized FROM nthreads rather than from nreads -- so a release build
     * given nthreads <= 0 would allocate a zero-entry scratch that kt_for then
     * writes into, because kt_for makes the same input safe by clamping
     * (`if (n_threads < 1) n_threads = 1`, kthread.cpp) instead of refusing.
     * Clamp identically so the two agree on how many windows exist. */
    if (nthreads < 1) nthreads = 1;

    // Record the thread count on the worker so worker_free can validate the
    // paired call and the per-thread loops can never walk past the slots
    // populated here.
    w.nthreads = nthreads;

    int32_t memSize = nreads;

    /* Mem allocation section for core kernels */
    w.regs = NULL; w.chain_scratch = NULL; w.seed_scratch = NULL;
    w.memo = NULL;   /* [dedup-reads] armed per-chunk by mem_process_seqs when the
                      * controller latches ON; NULL elsewhere (unarmed = no memo) */

    /* regs is genuinely chunk-lifetime, and the only nreads-sized allocation
     * left here: it is filled by the align pass, read by mem_pestat, and
     * consumed (and freed per read) by worker_sam.
     *
     * PIPE-F24: when memSize == 0 the caller is deferring that sizing to the
     * grow-on-demand path in kt_pipeline (step 1), which allocates regs EXACTLY
     * from the parsed read count (ret->n_seqs) and reuses it across chunks.
     * Leaving the pointer NULL here is a valid state: that path free()s it
     * (free(NULL) is a no-op) before its own calloc, and worker_free is
     * NULL-safe too. For memSize > 0 (e.g. language-binding consumers that size
     * regs themselves) the behavior is byte-identical to before. */
    if (memSize > 0) {
        w.regs = (mem_alnreg_v *) calloc(memSize, sizeof(mem_alnreg_v));
        xassert(w.regs != NULL, "out of memory: w.regs");
    }

    /* chain_scratch / seed_scratch are PER-THREAD scratch, not per-read arrays.
     * They used to be sized by nreads and indexed by seq_id, because worker_bwt
     * and worker_aln were two separate kt_for passes and the chains + their
     * seeds had to survive the barrier between them. The fused worker_bwt_aln
     * removed that requirement: mem_kernel2_core frees chain->a for every read
     * in the item before returning, so both buffers are dead the moment a work
     * item ends and a thread can reuse its own window for the next item.
     *
     * kt_for dispatches items of at most BATCH_SIZE reads and never hands the
     * same tid to two concurrent invocations, so nthreads windows of BATCH_SIZE
     * entries each is exactly sufficient.
     *
     * At -t 64 and the default -K this takes seed_scratch from 13.1 GB to 67 MB
     * and chain_scratch from 204.8 MB to 1.0 MB. The old sizing came off the
     * read-count estimate (chunk_bases / NREADS_ESTIMATE_AVG_BASES, and
     * chunk_bases is -K * -t), so it grew in BOTH -t and -K -- 39.3 GB of
     * seed_scratch alone at -t 192. The new sizing drops the -K dependence
     * entirely and leaves a per-thread constant.
     *
     * Those new figures are for BATCH_SIZE 512; arm64 uses 1024, so double them
     * there (134 MB / 2.1 MB). The old figures do not depend on BATCH_SIZE.
     *
     * Both are sized from nthreads, so PIPE-F24's memSize == 0 deferral does not
     * apply to them: that exists so an nreads-shaped buffer is not sized before
     * the read count is known, and neither of these is nreads-shaped any more.
     * They are allocated unconditionally, and the reallocation in kt_pipeline
     * step 1 correspondingly regrows only regs. */
    const int64_t scratch_reads = (int64_t) nthreads * BATCH_SIZE;
    w.chain_scratch = (mem_chain_v*) malloc (scratch_reads * sizeof(mem_chain_v));
    w.seed_scratch  = (mem_seed_t *) calloc(scratch_reads * AVG_SEEDS_PER_READ,
                                            sizeof(mem_seed_t));

    xassert(w.seed_scratch  != NULL, "out of memory: w.seed_scratch");
    xassert(w.chain_scratch != NULL, "out of memory: w.chain_scratch");

    /* Size of ONE thread's seed window. */
    w.seed_scratch_size = BATCH_SIZE * AVG_SEEDS_PER_READ;

    /*** printing ***/
    int64_t scratchMem = scratch_reads * sizeof(mem_chain_v) +
        sizeof(mem_seed_t) * scratch_reads * AVG_SEEDS_PER_READ;
    int64_t allocMem = memSize * sizeof(mem_alnreg_v) + scratchMem;
    fprintf(stderr, "------------------------------------------\n");
    fprintf(stderr, "1. Memory pre-allocation for Chaining: %0.4lf MB\n", allocMem/1e6);
    /* Reported separately from the total because the two scale differently and
     * conflating them hides regressions: regs is legitimately per-read (it
     * tracks -K), whereas the chaining scratch must depend only on the thread
     * count. test/regression/chain_scratch_per_thread.sh asserts this figure is
     * invariant across -K. */
    fprintf(stderr, "   per-thread chaining scratch: %0.4lf MB (%d threads)\n",
            scratchMem/1e6, nthreads);


    /* SWA mem allocation.
     *
     * These are a STARTING size, not a bound: seqBufRef/Qer double via
     * seqbuf_grow_capacity() and the seqPairArrays realloc on demand, both in
     * mem_chain2aln_across_reads_V2 and the batched mate-rescue path. So the
     * only thing the initial value buys is avoiding a few early reallocs.
     *
     * It used to start at BATCH_SIZE * SEEDS_PER_READ. SEEDS_PER_READ is 500 --
     * a worst-case seeds-per-read bound -- but these arrays hold extension
     * PAIRS, roughly a couple per chain, not one per seed. The result was
     * ~504 MB per thread reserved up front and, being per-thread, growth linear
     * in -t: 32.2 GB at -t 64 and ~97 GB at -t 192, dwarfing even the index.
     *
     * Start from AVG_SEEDS_PER_READ instead -- the same per-read seed estimate
     * the chaining scratch is sized with -- for an 8x smaller reservation, and
     * let the existing growth paths cover anything heavier. Byte-identical:
     * capacity affects only where the extension data lives, never its
     * contents. */
    int64_t wsize = BATCH_SIZE * AVG_SEEDS_PER_READ;
    for(int l=0; l<nthreads; l++)
    {
        w.mmc.seqBufLeftRef[l*CACHE_LINE]  = (uint8_t *)
            _mm_malloc(wsize * MAX_SEQ_LEN_REF * sizeof(int8_t) + MAX_LINE_LEN, 64);
        w.mmc.seqBufLeftQer[l*CACHE_LINE]  = (uint8_t *)
            _mm_malloc(wsize * MAX_SEQ_LEN_QER * sizeof(int8_t) + MAX_LINE_LEN, 64);
        w.mmc.seqBufRightRef[l*CACHE_LINE] = (uint8_t *)
            _mm_malloc(wsize * MAX_SEQ_LEN_REF * sizeof(int8_t) + MAX_LINE_LEN, 64);
        w.mmc.seqBufRightQer[l*CACHE_LINE] = (uint8_t *)
            _mm_malloc(wsize * MAX_SEQ_LEN_QER * sizeof(int8_t) + MAX_LINE_LEN, 64);

        w.mmc.wsize_buf_ref[l*CACHE_LINE] = wsize * MAX_SEQ_LEN_REF;
        w.mmc.wsize_buf_qer[l*CACHE_LINE] = wsize * MAX_SEQ_LEN_QER;

        assert(w.mmc.seqBufLeftRef[l*CACHE_LINE]  != NULL);
        assert(w.mmc.seqBufLeftQer[l*CACHE_LINE]  != NULL);
        assert(w.mmc.seqBufRightRef[l*CACHE_LINE] != NULL);
        assert(w.mmc.seqBufRightQer[l*CACHE_LINE] != NULL);
    }

    for(int l=0; l<nthreads; l++) {
        w.mmc.seqPairArrayAux[l]      = (SeqPair *) malloc((wsize + MAX_LINE_LEN)* sizeof(SeqPair));
        w.mmc.seqPairArrayLeft128[l]  = (SeqPair *) malloc((wsize + MAX_LINE_LEN)* sizeof(SeqPair));
        w.mmc.seqPairArrayRight128[l] = (SeqPair *) malloc((wsize + MAX_LINE_LEN)* sizeof(SeqPair));
        w.mmc.wsize[l] = wsize;

        xassert(w.mmc.seqPairArrayAux[l] != NULL, "out of memory: w.mmc.seqPairArrayAux[l]");
        xassert(w.mmc.seqPairArrayLeft128[l] != NULL, "out of memory: w.mmc.seqPairArrayLeft128[l]");
        xassert(w.mmc.seqPairArrayRight128[l] != NULL, "out of memory: w.mmc.seqPairArrayRight128[l]");
    }


    allocMem = (wsize * MAX_SEQ_LEN_REF * sizeof(int8_t) + MAX_LINE_LEN) * opt->n_threads * 2+
        (wsize * MAX_SEQ_LEN_QER * sizeof(int8_t) + MAX_LINE_LEN) * opt->n_threads  * 2 +
        wsize * sizeof(SeqPair) * opt->n_threads * 3;
    fprintf(stderr, "2. Memory pre-allocation for BSW: %0.4lf MB\n", allocMem/1e6);

    // SMEM buffers (matchArray / min_intv_ar / query_pos_ar / enc_qdb / rid)
    // and the lockstep-batch slot buffers are sized from the observed max
    // read length on each batch in mem_collect_smem; they're NULL here and
    // grow on first use. `lim` is still a fixed BATCH_SIZE+32 allocation
    // because its size does not depend on read length.
    for (int l=0; l<nthreads; l++)
    {
        w.mmc.wsize_mem[l]     = 0;
        w.mmc.wsize_mem_s[l]   = 0;
        w.mmc.wsize_mem_r[l]   = 0;
        w.mmc.wsize_qdb[l]     = 0;
        w.mmc.matchArray[l]    = NULL;
        w.mmc.min_intv_ar[l]   = NULL;
        w.mmc.query_pos_ar[l]  = NULL;
        w.mmc.enc_qdb[l]       = NULL;
        w.mmc.rid[l]           = NULL;
        w.mmc.lim[l]           = (int32_t *) _mm_malloc((BATCH_SIZE + 32) * sizeof(int32_t), 64);

        w.mmc.lockstep_prev[l]      = NULL;
        w.mmc.lockstep_match_buf[l] = NULL;
        w.mmc.lockstep_buf_cap[l]   = 0;

        w.mmc.smem_sort_scratch[l].cnt    = NULL;
        w.mmc.smem_sort_scratch[l].cntCap = 0;
        w.mmc.smem_sort_scratch[l].tmp    = NULL;
        w.mmc.smem_sort_scratch[l].tmpCap = 0;
    }

    allocMem = nthreads * (BATCH_SIZE + 32) * sizeof(int32_t);
    fprintf(stderr, "3. Memory pre-allocation for BWT (lazy SMEM buffers, %ld B fixed): %0.4lf MB\n",
            (long)(nthreads * (BATCH_SIZE + 32) * sizeof(int32_t)), allocMem/1e6);
    fprintf(stderr, "------------------------------------------\n");
}

// Release every per-worker scratch buffer allocated by worker_alloc. The
// nthreads argument must match the value passed to the paired worker_alloc
// call so the per-thread loops iterate over exactly the slots that were
// populated.
void worker_free(worker_t &w, int32_t nthreads)
{
    assert(nthreads > 0);
    // Catch mismatched alloc/free pairs before they drive out-of-bounds frees.
    assert(w.nthreads == nthreads);

    free(w.chain_scratch);
    free(w.regs);
    free(w.seed_scratch);

    for(int l=0; l<nthreads; l++) {
        _mm_free(w.mmc.seqBufLeftRef[l*CACHE_LINE]);
        _mm_free(w.mmc.seqBufRightRef[l*CACHE_LINE]);
        _mm_free(w.mmc.seqBufLeftQer[l*CACHE_LINE]);
        _mm_free(w.mmc.seqBufRightQer[l*CACHE_LINE]);
    }

    for(int l=0; l<nthreads; l++) {
        free(w.mmc.seqPairArrayAux[l]);
        free(w.mmc.seqPairArrayLeft128[l]);
        free(w.mmc.seqPairArrayRight128[l]);
    }

    // NULL-safe: SMEM buffers are now allocated lazily on first batch;
    // workers that never ran a batch leave them as NULL. _mm_free / free
    // are both well-defined on NULL.
    for(int l=0; l<nthreads; l++) {
        _mm_free(w.mmc.matchArray[l]);
        free(w.mmc.min_intv_ar[l]);
        free(w.mmc.query_pos_ar[l]);
        free(w.mmc.enc_qdb[l]);
        free(w.mmc.rid[l]);
        _mm_free(w.mmc.lim[l]);

        _mm_free(w.mmc.lockstep_prev[l]);
        _mm_free(w.mmc.lockstep_match_buf[l]);

        _mm_free(w.mmc.smem_sort_scratch[l].cnt);
        _mm_free(w.mmc.smem_sort_scratch[l].tmp);
    }
}

// Back-compat wrapper used by the bwa-mem3 pipeline.
void memoryAlloc(ktp_aux_t *aux, worker_t &w, int32_t nreads, int32_t nthreads)
{
    worker_alloc(aux->opt, w, nreads, nthreads);
}

/* Copy the compute step's --profile counters out of the per-thread kt_for
 * accumulator and into this chunk's profile row (see stage_prof.h).
 *
 * Step 1 has two exits -- a whole cohort, and a partial cohort that is still
 * accumulating slices -- and g_ktfor is reset at every step-1 entry, so a path
 * that forgets to harvest does not merely lose detail: that slice's compute CPU
 * is gone. Both exits call this, so adding a third cannot silently drop it.
 *
 * `sp_p0` is the sp_wall() reading taken at step-1 entry.
 */
static void sp_harvest_proc(prof_chunk_t *p, double sp_p0)
{
    p->proc_wall      = sp_wall() - sp_p0;
    p->proc_cpu       = g_ktfor.proc_cpu;
    p->thr_busy_min   = g_ktfor.thr_busy_min;
    p->thr_busy_max   = g_ktfor.thr_busy_max;
    p->thr_busy_mean  = g_ktfor.thr_busy_mean;
    p->thr_busy_stdev = g_ktfor.thr_busy_stdev;
    /* encode = SAM/BAM-build CPU (accurate, summed over compute threads);
     * compute = the rest of the alignment CPU. Same clock, so subtractable.
     * A slice that only seeds and extends never enters mem_aln2sam, so its
     * encode is legitimately 0 and compute is the whole of proc_cpu. */
    p->encode  = g_ktfor.encode;
    p->compute = (g_ktfor.proc_cpu > g_ktfor.encode)
                 ? g_ktfor.proc_cpu - g_ktfor.encode : NAN;
}

ktp_data_t *kt_pipeline(void *shared, int step, void *data, mem_opt_t *opt, worker_t &w)
{
    ktp_aux_t *aux = (ktp_aux_t*) shared;
    ktp_data_t *ret = (ktp_data_t*) data;

    if (step == 0)
    {
        ktp_data_t *ret = (ktp_data_t *) calloc(1, sizeof(ktp_data_t));
        xassert(ret != NULL, "out of memory: ret");
        uint64_t tim = __rdtsc();
        double sp_r0 = 0.0;
        if (sp_enabled()) {
            sp_chunk_init(&ret->prof); sp_read_reset();
            ret->prof.chunk_start = sp_run_elapsed();   /* timeline anchor */
            sp_r0 = sp_wall();
        }

        /* Read "reads" from input file (fread).
         *
         * A cohort is one task_size batch. Normally it is read in a single call
         * and this is exactly the old code path. When slicing is enabled the
         * FIRST cohort is read as a ramp of growing slices: the first is an
         * absolute cohort_ramp_first bases (16 Mbase by default, capped at
         * task/2) and each next is cohort_ramp_ratio times the previous (1.6 by
         * default), for at most cohort_slices slices. Compute therefore starts
         * working while the rest is still being read -- only the first read of a
         * run overlaps nothing, and it is the largest single unhidden cost in
         * the pipeline.
         *
         * The target for every slice is clamped to the bases the cohort still
         * needs. That clamp is load-bearing, not defensive: both readers stop at
         * the first whole-record, even-n boundary at or PAST the requested size,
         * so each slice overshoots independently. Reading T/8 + T/4 + T/2 + T/8
         * without the clamp lands at T + e1+e2+e3+e4, whereas one read of T lands
         * at T + e -- a different last record, hence a different cohort, hence a
         * different mem_pestat. With the clamp, the first even-n boundary at or
         * past (cohort_bases + slice_target) is provably the same record as the
         * first at or past T whenever the slice crosses T, because every earlier
         * boundary sits strictly below it. */
        int64_t slice_target = aux->task_size - aux->cohort_bases;
        if (slice_target <= 0) slice_target = aux->task_size;   /* new cohort */
        /* Normally only the first cohort ramps -- later ones are already
         * overlapped by the pipeline. slice_all is a STRESS knob for the
         * identity test: it slices EVERY cohort and drops the efficiency floor,
         * so a short run exercises the accumulator, the cohort-id advance and
         * the boundary clamp at dozens of boundaries instead of one. It is not
         * a performance mode -- tiny slices cost more in kt_for passes and lost
         * SIMD batching than they can ever recover in overlap. */
        if (aux->cohort_slices > 0 &&
            (aux->cohort_slice_all || aux->cohort_index == 0) &&
            aux->cohort_slice < aux->cohort_slices) {
            /* How fast the ramp is allowed to grow. Doubling every slice is only
             * free while the reader can deliver slice k+1 inside the time step 1
             * spends on slice k; past that the ramp starves the very pipeline it
             * exists to fill. See cohort_ramp_ratio in fastmap.h. */
            double ratio = aux->cohort_ramp_ratio;
            int64_t ramped;
            if (aux->cohort_slice == 0 || aux->ramp_prev_target <= 0) {
                /* The first slice is an ABSOLUTE size, not a fraction of the
                 * cohort. Both costs it trades against are absolute -- its own
                 * read overlaps nothing, and every extra slice costs one more
                 * kt_for pass -- so nothing about the right answer scales with
                 * task_size. Sizing it as task_size/ratio^depth made it grow with
                 * -t, which is backwards: task_size is chunk_size * n_threads, so
                 * at -t 128 a 1280 Mbase cohort would open with a 76 Mbase slice
                 * and 0.37 s of pure unhidden fill, against 0.077 s for a fixed
                 * 16 Mbase. The penalty grows linearly with thread count.
                 *
                 * Capped at half the cohort so a small task_size (a short test,
                 * or an explicit -K) still gets at least two slices.
                 *
                 * The stress knob keeps the fractional shape on purpose: it
                 * exists to make a short run cross as many cohort boundaries as
                 * possible, and is explicitly not a performance mode. */
                if (aux->cohort_ramp_first > 0 && !aux->cohort_slice_all) {
                    ramped = aux->cohort_ramp_first;
                    if (ramped > aux->task_size >> 1) ramped = aux->task_size >> 1;
                } else if (ratio == 2.0) {
                    /* Ratio 2 keeps the original integer shift, so pinning the
                     * ratio to 2 with --cohort-ramp-first 0 reproduces the
                     * pre-existing slice sizes exactly rather than approximately
                     * via pow(). That exactness is what makes an A/B against the
                     * old shape meaningful. */
                    int shift = (int)(aux->cohort_slices - aux->cohort_slice);
                    if (shift > COHORT_SLICES_MAX) shift = COHORT_SLICES_MAX;
                    ramped = aux->task_size >> shift;
                } else {
                    double denom = pow(ratio, (double)aux->cohort_slices);
                    ramped = (denom > 1.0)
                             ? (int64_t)((double)aux->task_size / denom)
                             : aux->task_size;
                }
            } else {
                /* Saturate in double space, before the narrowing cast. The
                 * ratio is only validated as > 1.0, so a sweep value like 30
                 * or 100 compounds ramp_prev_target past INT64_MAX within a
                 * handful of slices, and converting an out-of-range double to
                 * int64_t is undefined behaviour. Saturating at task_size
                 * costs nothing: slice_target is never above task_size, so a
                 * ramp at or past it already means "the whole cohort", and it
                 * keeps the stored ramp_prev_target from compounding away. */
                const double grown = (double)aux->ramp_prev_target * ratio;
                ramped = (grown >= (double)aux->task_size)
                         ? aux->task_size
                         : (int64_t)grown;
            }

            /* A slice must stay large enough for its own SMEM/BSW batching to be
             * efficient; below this the extra kt_for passes cost more than the
             * overlap they buy. The stress knob deliberately bypasses this. */
            int64_t floor_bases = aux->cohort_slice_all ? 1 : 1000000;
            if (ramped < floor_bases) ramped = floor_bases;
            /* Remember the REQUESTED size, not what the reader returns: the ramp
             * shape must not drift with each slice's whole-record overshoot. */
            aux->ramp_prev_target = ramped;
            if (ramped < slice_target) slice_target = ramped;
        }

        int64_t sz = 0;
        /* ONE arena per cohort, not per read call. The accumulator below copies
         * only the bseq1_t structs into aux->cohort_seqs and shares name/seq/qual
         * BY POINTER -- so those bytes must outlive the slice that carved them and
         * stay alive until the whole cohort has been paired and written. Carrying
         * the cohort's arena into each slice's read gives exactly that: the reader
         * appends to it (see read_arena.h -- blocks are chained and never moved,
         * so earlier pointers stay valid), and the completing slice hands the
         * single arena to the write stage, which destroys it once.
         *
         * Scoping the arena per read call instead is a use-after-free: the write
         * stage destroys every item's arena unconditionally, including the empty
         * items partial slices return, freeing the strings the cohort still
         * references. */
        ret->read_arena = aux->cohort_arena;
        ret->seqs = aux->legacy_reader
            ? bseq_read_orig(slice_target, &ret->n_seqs, aux->ks, aux->ks2, &sz, &ret->read_arena)
            : bseq_read_fast(slice_target, &ret->n_seqs, aux->frks, aux->frks2, &sz, &ret->read_arena, aux->copy_comment);
        aux->cohort_arena = ret->read_arena;

        /* A short read means the input ran out, which ends the cohort early. */
        aux->cohort_bases += sz;
        /* Is a cohort mid-accumulation? cohort_slice counts the partial slices of
         * the current cohort and is reset the moment one completes, so a non-zero
         * value means step 1 is holding earlier slices that still have to be
         * paired and emitted. Read from cohort_slice rather than cohort_n because
         * only step 0 touches it: step 0 and step 1 run concurrently on different
         * items, and cohort_n belongs to step 1. Captured before the bookkeeping
         * below resets it. */
        const int mid_cohort = (aux->cohort_slice > 0);
        ret->cohort_complete = (sz < slice_target) ||
                               (aux->cohort_bases >= aux->task_size) ||
                               (ret->seqs == NULL) || (ret->n_seqs == 0);
        if (ret->cohort_complete) {
            aux->cohort_bases = 0;
            aux->cohort_slice = 0;
            aux->ramp_prev_target = 0;
            aux->cohort_index++;
            /* This item carries the cohort's arena to the write stage, which
             * destroys it. Drop our reference so the NEXT cohort starts fresh. */
            aux->cohort_arena = NULL;
        } else {
            aux->cohort_slice++;
            /* Partial slice: the arena stays with the cohort, not this item. The
             * write stage destroys ret->read_arena unconditionally, so it must
             * not see it here -- the strings are still live. */
            ret->read_arena = NULL;
        }

        tprof[READ_IO][0] += __rdtsc() - tim;

        if (sp_enabled()) {
            ret->prof.read_wall = sp_wall() - sp_r0;
            /* Only the fast reader is instrumented; the legacy reader leaves the
             * sub-splits at their NaN init (blank) rather than reporting a fake 0. */
            if (!aux->legacy_reader) {
                sp_read_get(&ret->prof.read_diskwait, &ret->prof.read_decompress, &ret->prof.read_parse);
                sp_read_get_bytes(&ret->prof.read_bytes_in, &ret->prof.bgzf_blocks);
            }
            ret->prof.n_reads = ret->n_seqs;
            ret->prof.n_bp = sz;
        }

        /* `read_chunk` keeps meaning the COHORT (batch) target, as it did before
         * slicing existed. Reporting slice_target here instead would repurpose an
         * existing field rather than add one: the batch size is the thing this
         * change is careful NOT to move -- byte-identity depends on it -- so it is
         * exactly what a reader of this line wants, and
         * test/regression/chunk_cap_optin.sh asserts on it. The slice request is
         * reported as its own field, and only when the cohort is actually being
         * sliced, so the common single-slice line stays character-for-character
         * what it was. `work_chunk_size` is unchanged: bases actually delivered by
         * this read, which for a slice is that slice's real size (the readers stop
         * at the first whole-record boundary at or past the request, so delivered
         * and requested differ). */
        if (slice_target != aux->task_size)
            fprintf(stderr, "[0000] read_chunk: %lld, slice: %lld, work_chunk_size: %lld, nseq: %d%s\n",
                    (long long)aux->task_size, (long long)slice_target,
                    (long long)sz, ret->n_seqs,
                    ret->cohort_complete ? "" : " (cohort slice)");
        else
            fprintf(stderr, "[0000] read_chunk: %lld, work_chunk_size: %lld, nseq: %d%s\n",
                    (long long)aux->task_size, (long long)sz, ret->n_seqs,
                    ret->cohort_complete ? "" : " (cohort slice)");

        /* No reads. Normally that is clean EOF: returning 0 retires this worker.
         * But if a cohort is still accumulating, the input ended exactly ON a
         * slice boundary (the previous slice delivered at least its target, so it
         * did not end the cohort, and there is nothing left to read). Its earlier
         * slices are already aligned and still have to be paired, emitted, and
         * have their arena destroyed. Retiring here dropped them silently --
         * short output, exit status 0. Hand step 1 an empty item instead: the
         * accumulator appends nothing, aligns nothing, and flushes the cohort. */
        if (ret->seqs == 0 && !mid_cohort) {
            free(ret);
            return 0;
        }
        if (!aux->copy_comment){
            for (int i = 0; i < ret->n_seqs; ++i){
                free(ret->seqs[i].comment);
                ret->seqs[i].comment = 0;
            }
        }

        /* Inline bwameth-style c2t conversion on read ingest. Matches
         * `bwameth.py c2t R1.fq R2.fq`: R1 reads get C→T, R2 reads get G→A.
         * The original sequence is stashed as a YS:Z comment tag and the
         * conversion type as YC:Z — these pass through to SAM via
         * copy_comment (which --meth sets).
         *
         * R1/R2 classification:
         *   - Non-smart-pair PE (two FASTQs): records are strictly
         *     interleaved R1/R2/R1/R2… in ret->seqs, so record parity
         *     (i & 1) identifies R2.
         *   - Smart-pair (-p, single stream): the stream can contain
         *     orphans, so parity mis-tags them. Classify by adjacent-name
         *     pairing (same rule as bseq_classify) — if this read's name
         *     matches the previous unmatched read's name, it is R2;
         *     otherwise it starts a new pair as R1.
         *   - SE: always R1. */
        if (aux->opt->meth_mode) {
            int is_pe    = (aux->opt->flag & MEM_F_PE) != 0;
            int is_smart = (aux->opt->flag & MEM_F_SMARTPE) != 0;
            int prev_is_r1 = 0;
            for (int i = 0; i < ret->n_seqs; ++i) {
                bseq1_t *s = &ret->seqs[i];
                int is_r2;
                if (!is_pe) {
                    is_r2 = 0;
                } else if (!is_smart) {
                    is_r2 = (i & 1);
                } else if (prev_is_r1 && i > 0
                           && strcmp(s->name, ret->seqs[i-1].name) == 0) {
                    is_r2 = 1;
                    prev_is_r1 = 0;
                } else {
                    is_r2 = 0;
                    prev_is_r1 = 1;
                }
                const char *yc = is_r2 ? "GA" : "CT";
                char from = is_r2 ? 'G' : 'C';
                char from_lo = is_r2 ? 'g' : 'c';
                char to   = is_r2 ? 'A' : 'T';
                int l = s->l_seq;
                /* Build the YS:Z/YC:Z comment. Preserve any prior FASTQ
                 * comment (e.g. -C carries barcode/UMI SAM tags) by appending
                 * it after YC; otherwise --meth silently strips -C metadata. */
                const char *prior = s->comment;
                size_t prior_len = prior ? strlen(prior) : 0;
                size_t yslen = (size_t)l + 32 + (prior_len ? prior_len + 1 : 0);
                char *comment = (char *)malloc(yslen);
                xassert(comment != NULL, "out of memory: comment");
                int off = snprintf(comment, yslen, "YS:Z:");
                memcpy(comment + off, s->seq, (size_t)l);
                off += l;
                off += snprintf(comment + off, yslen - off, "\tYC:Z:%s", yc);
                if (prior_len)
                    snprintf(comment + off, yslen - off, "\t%s", prior);
                free(s->comment);
                s->comment = comment;
                /* Retain the ORIGINAL (unconverted) read bases as a first-class
                 * field BEFORE the in-place projection below overwrites s->seq.
                 * Same orientation/order as s->seq (original read order, ASCII);
                 * downstream consumers must RC it wherever they RC s->seq — see
                 * the bseq1_t.meth_orig_seq orientation contract in bwa.h.
                 * strdup is fine for the draft; freed in the per-batch free
                 * loop below alongside s->seq. */
                s->meth_orig_seq = strdup(s->seq);
                xassert(s->meth_orig_seq != NULL, "out of memory: s->meth_orig_seq");
                /* --meth: read-number chemistry (R1=OT=1, R2=OB=0) for the
                 * seed-chemistry filter in meth_seed_to_orig. */
                s->meth_base_ot = is_r2 ? 0 : 1;
                /* Project in place. */
                for (int j = 0; j < l; ++j) {
                    char c = s->seq[j];
                    if (c == from || c == from_lo) s->seq[j] = to;
                }
            }
        }
        {
            int64_t size = 0;
            for (int i = 0; i < ret->n_seqs; ++i) size += ret->seqs[i].l_seq;

            fprintf(stderr, "\t[0000][ M::%s] read %d sequences (%ld bp)...\n",
                    __func__, ret->n_seqs, (long)size);
        }

        return ret;
    } // Step 0
    else if (step == 1)  /* Step 2: Main processing-engine */
    {
        static int task = 0;
        /* regs must hold the whole COHORT, not just the reads this item carries.
         * A sliced cohort aligns each slice as it arrives and keeps the results
         * until mem_pair_and_emit_cohort consumes them, so growing this array
         * must PRESERVE what the earlier slices already wrote -- hence realloc,
         * not free() + calloc(). The old free()+calloc() sized from this slice's
         * ret->n_seqs alone and discarded every earlier slice of the cohort,
         * which emitted the cohort's leading reads as unmapped.
         *
         * Not zeroed: mem_kernel2_core kv_init()s every entry of the range it is
         * given before writing it (src/bwamem.cpp), so no reader ever observes
         * an entry the align phase has not initialized. That is also why the
         * pre-existing `w.nreads >= n_seqs` case has always been safe without
         * zeroing, and why dropping the whole-array calloc is output-neutral.
         *
         * Only regs is sized by the read count. chain_scratch/seed_scratch are
         * per-thread windows sized by nthreads * BATCH_SIZE and are unaffected
         * by how many reads a chunk turns out to hold. */
        int32_t regs_want = aux->cohort_n + ret->n_seqs;
        if (w.nreads < regs_want)
        {
            fprintf(stderr, "[0000] Reallocating initial memory allocations!!\n");
            /* Into a temporary, and err_fatal rather than assert, for the same
             * reason as the seed-scratch check below: assert compiles out under
             * NDEBUG, so in a release build a failed realloc would null the live
             * pointer, drop the previous allocation, and let the align phase
             * dereference NULL a few lines later. */
            mem_alnreg_v *regs_tmp = (mem_alnreg_v *) realloc(w.regs,
                                              (size_t) regs_want * sizeof(mem_alnreg_v));
            if (regs_tmp == NULL)
                err_fatal(__func__, "failed to grow the cohort's regs array to %d reads",
                          regs_want);
            w.regs = regs_tmp;
            w.nreads = regs_want;
        }

        /* The per-thread chaining scratch must never be resized per chunk: it is
         * sized from nthreads * BATCH_SIZE in worker_alloc and indexed by tid, so
         * an nreads-shaped reallocation here would give back the memory this
         * sizing reclaimed, and a SMALLER one would hand two threads overlapping
         * windows.
         *
         * Enforced rather than merely commented because the mistake is invisible
         * downstream: the alignments are byte-identical either way, so every
         * parity and determinism test in the suite would still pass. Deliberately
         * not an assert -- this has to survive NDEBUG, which is exactly the build
         * a memory regression would be noticed in. Costs one comparison per
         * chunk. */
        if (w.seed_scratch_size != (int64_t) BATCH_SIZE * AVG_SEEDS_PER_READ) {
            fprintf(stderr,
                    "ERROR: per-thread seed scratch was resized per chunk "
                    "(%lld seeds, expected %lld); the chaining scratch must be "
                    "sized only in worker_alloc (src/fastmap.cpp).\n",
                    (long long) w.seed_scratch_size,
                    (long long) BATCH_SIZE * AVG_SEEDS_PER_READ);
            exit(EXIT_FAILURE);
        }

        fprintf(stderr, "[0000] Calling mem_process_seqs.., task: %d\n", task++);

        double sp_p0 = 0.0;
        if (sp_enabled()) {
            sp_chunk_init(&g_ktfor);          /* reset balance + encode accumulator */
            g_ktfor.encode = 0.0;             /* accumulate (sp_chunk_init left it NaN) */
            sp_p0 = sp_wall();
        }

        uint64_t tim = __rdtsc();
        if (opt->flag & MEM_F_SMARTPE)
        {
            bseq1_t *sep[2];
            int n_sep[2];
            mem_opt_t tmp_opt = *opt;

            bseq_classify(ret->n_seqs, ret->seqs, n_sep, sep);

            fprintf(stderr, "[M::%s] %d single-end sequences; %d paired-end sequences.....\n",
                    __func__, n_sep[0], n_sep[1]);

            if (n_sep[0]) {
                tmp_opt.flag &= ~MEM_F_PE;
                /* single-end sequences, in the mixture */
                mem_process_seqs(&tmp_opt,
                                 aux->n_processed,
                                 n_sep[0],
                                 sep[0],
                                 0,
                                 w);

                for (int i = 0; i < n_sep[0]; ++i) {
                    bseq1_t *dst = &ret->seqs[sep[0][i].id];
                    bseq1_t *src = &sep[0][i];
                    dst->sam      = src->sam;      src->sam      = NULL;
                    dst->bams     = src->bams;     src->bams     = NULL;
                    dst->n_bams   = src->n_bams;   src->n_bams   = 0;
                    dst->cap_bams = src->cap_bams; src->cap_bams = 0;
                }
            }
            if (n_sep[1]) {
                tmp_opt.flag |= MEM_F_PE;
                /* paired-end sequences, in the mixture */
                mem_process_seqs(&tmp_opt,
                                 aux->n_processed + n_sep[0],
                                 n_sep[1],
                                 sep[1],
                                 aux->pes0,
                                 w);

                for (int i = 0; i < n_sep[1]; ++i) {
                    bseq1_t *dst = &ret->seqs[sep[1][i].id];
                    bseq1_t *src = &sep[1][i];
                    dst->sam      = src->sam;      src->sam      = NULL;
                    dst->bams     = src->bams;     src->bams     = NULL;
                    dst->n_bams   = src->n_bams;   src->n_bams   = 0;
                    dst->cap_bams = src->cap_bams; src->cap_bams = 0;
                }
            }
            free(sep[0]); free(sep[1]);
        }
        else if (ret->cohort_complete && aux->cohort_n == 0) {
            /* The common case: this slice IS the whole cohort. Byte-for-byte the
             * pre-slicing code path -- no accumulation, no copy. */
            mem_process_seqs(opt,
                             aux->n_processed,
                             ret->n_seqs,
                             ret->seqs,
                             aux->pes0,
                             w);
        }
        else {
            /* Multi-slice cohort. Append this slice to the cohort array, align
             * just this slice, and only pair + emit once the cohort is whole.
             *
             * Only the bseq1_t structs are copied; the per-read name/seq/qual
             * allocations are shared by pointer and the slice's own array is
             * freed. The pipeline is one-item-in / one-item-out, so the cohort
             * must be handed to the write step as a single contiguous array. */
            if (aux->cohort_n == 0) aux->cohort_first_id = aux->n_processed;

            if (aux->cohort_n + ret->n_seqs > aux->cohort_cap) {
                int want = aux->cohort_n + ret->n_seqs;
                int cap  = aux->cohort_cap ? aux->cohort_cap : 1024;
                /* A cohort with more slices still coming: project its final read
                 * count from this slice's mean read length and reserve it in one
                 * go.
                 *
                 * Doubling from 1024 instead takes ~13 reallocs to reach a
                 * default t=64 cohort (task_size = chunk_size * nthreads = 640
                 * Mbase, ~4.27M reads) and lands on the next power of two --
                 * 8388608 slots for 4266668 reads, i.e. ~330 MB of bseq1_t held
                 * and never used. The cohort's size is not a surprise: the ramp
                 * appends slices until it reaches aux->task_size bases, so one
                 * projection is enough.
                 *
                 * Deliberately NOT projected on the cohort's first slice, even
                 * though that is where the reallocs would be saved. On the first
                 * slice "more is coming" is not yet known: a file that ends
                 * exactly at the slice boundary returns sz == slice_target with
                 * seqs != NULL, which fails every cohort_complete test, so the
                 * slice looks mid-cohort and EOF only surfaces on the next read
                 * as the empty flush item (see the memcpy guard below). Projecting
                 * there would reserve a whole task_size -- ~330 MB for what turns
                 * out to be one ~16 Mbase slice, strictly worse than the doubling
                 * this replaces. Waiting for cohort_n > 0 makes a second slice the
                 * proof that the input did not end, and costs only the first
                 * slice's doublings, which are small and cheap.
                 *
                 * Idempotent across later slices: the projected total is the same
                 * each time and cap only ever moves up, so slices 3+ recompute it
                 * and find nothing to do.
                 *
                 * The doubling below is kept as the fallback: if reads later in
                 * the cohort are shorter than this slice's mean, the projection
                 * undershoots and growth proceeds exactly as before. Capacity
                 * only ever affects allocation, never which reads land in the
                 * cohort, so output is unchanged either way. */
                if (aux->cohort_n > 0 && !ret->cohort_complete && ret->n_seqs > 0) {
                    int64_t slice_bases = 0;
                    for (int i = 0; i < ret->n_seqs; ++i)
                        slice_bases += ret->seqs[i].l_seq;
                    if (slice_bases > 0 && aux->task_size > slice_bases) {
                        /* +2 for the rounding slack: both readers stop at the
                         * first whole-record, even-n boundary at or PAST each
                         * request, so the cohort can end a record or two past
                         * task_size. */
                        double projected = (double) ret->n_seqs *
                                           ((double) aux->task_size / (double) slice_bases) + 2.0;
                        if (projected > (double) cap && projected < (double) INT_MAX)
                            cap = (int) projected;
                    }
                }
                while (cap < want) cap <<= 1;
                /* Temporary + err_fatal, not assert -- see the regs growth in
                 * step 1. A release-build realloc failure here would null the
                 * accumulated cohort, leak every slice already copied into it,
                 * and be dereferenced by the memcpy immediately below. */
                bseq1_t *cohort_tmp = (bseq1_t *) realloc(aux->cohort_seqs,
                                                       (size_t) cap * sizeof(bseq1_t));
                if (cohort_tmp == NULL)
                    err_fatal(__func__, "failed to grow the cohort buffer to %d reads", cap);
                aux->cohort_seqs = cohort_tmp;
                aux->cohort_cap = cap;
                /* -v 4 only: capacity has no effect on output, so this exists
                 * purely so a test can tell a projected reservation from a
                 * doubled one. Default verbosity is 3, so ordinary runs and every
                 * stderr-parsing test see exactly what they saw before. */
                if (bwa_verbose >= 4)
                    fprintf(stderr, "[0000] cohort_reserve: cap: %d, held: %d\n",
                            cap, aux->cohort_n + ret->n_seqs);
            }
            /* Guarded because the EOF-on-slice-boundary item reaches here with
             * seqs == NULL and n_seqs == 0 -- it exists only to flush the cohort
             * (see step 0). memcpy(dst, NULL, 0) is UB in C/C++ even at zero
             * length, and UBSan's nonnull-attribute check reports it. */
            if (ret->n_seqs > 0)
                memcpy(aux->cohort_seqs + aux->cohort_n, ret->seqs,
                       (size_t) ret->n_seqs * sizeof(bseq1_t));
            int slice_off = aux->cohort_n;
            aux->cohort_n += ret->n_seqs;
            free(ret->seqs);            /* structs copied out; strings still owned */
            ret->seqs = NULL; ret->n_seqs = 0;

            /* Re-base ids to the cohort. Only bseq_classify reads .id, and that
             * path is excluded from slicing, but a stale per-slice id would be a
             * trap for the next reader. */
            for (int i = 0; i < aux->cohort_n - slice_off; ++i)
                aux->cohort_seqs[slice_off + i].id = slice_off + i;

            /* regs is sized once per item, at the top of this step, to the whole
             * cohort (cohort_n + this slice's reads) and grown with realloc so
             * the earlier slices' alignments survive -- they are computed as each
             * slice arrives but consumed only by mem_pair_and_emit_cohort once
             * the cohort is whole. Nothing to do here but rely on that.
             *
             * Sizing there rather than here is what makes the invariant hold for
             * BOTH paths: the single-slice fast path above needs the same array
             * and never reaches this branch. */
            assert(w.nreads >= aux->cohort_n);

            mem_align_cohort_slice(opt,
                                   aux->cohort_first_id + slice_off,
                                   aux->cohort_n - slice_off,
                                   aux->cohort_seqs + slice_off,
                                   w.regs + slice_off,
                                   w);

            if (!ret->cohort_complete) {
                /* Partial cohort: hand the pipeline a non-NULL empty item. NULL
                 * would retire this worker (see ktp_worker's step advance). */
                tprof[MEM_PROCESS2][0] += __rdtsc() - tim;
                if (sp_enabled()) sp_harvest_proc(&ret->prof, sp_p0);
                return ret;
            }

            mem_pair_and_emit_cohort(opt,
                                     aux->cohort_first_id,
                                     aux->cohort_n,
                                     aux->cohort_seqs,
                                     aux->pes0,
                                     w);

            /* Hand the cohort to the write step, which frees the array. */
            ret->seqs   = aux->cohort_seqs;
            ret->n_seqs = aux->cohort_n;
            aux->cohort_seqs = NULL; aux->cohort_n = 0; aux->cohort_cap = 0;
        }
        tprof[MEM_PROCESS2][0] += __rdtsc() - tim;

        if (sp_enabled()) sp_harvest_proc(&ret->prof, sp_p0);

        aux->n_processed += ret->n_seqs;
        return ret;
    }
    /* Step 3: Write output */
    else if (step == 2)
    {
        uint64_t tim = __rdtsc();
        double sp_w0 = sp_enabled() ? sp_wall() : 0.0;
        long sp_wbytes = 0;

        /* L17: hoist the chunk-constant writer/mode selectors out of the
         * per-record loop. opt->meth_mode and opt->flag never change within a
         * chunk, and neither writer pointer is reassigned during output; the
         * loop body only calls bam_writer_write/free, which touch records, not
         * these. Without this the compiler must reload aux->opt->meth_mode (a
         * two-level deref) and the g_meth_bam_writer global after every external
         * write call. Byte-identical (same control flow, same values). */
        const int   out_meth_mode = aux->opt->meth_mode;
        const int   out_flag_pe   = (aux->opt->flag & MEM_F_PE);
        meth_bam_writer_t *const out_meth_bw = g_meth_bam_writer;
        struct bam_writer_s *const out_bam_bw = aux->bam_writer;

        for (int i = 0; i < ret->n_seqs; )
        {
            int group_size = 1;
            if (out_meth_mode && out_flag_pe
                && i + 1 < ret->n_seqs
                && strcmp(ret->seqs[i].name, ret->seqs[i+1].name) == 0) {
                group_size = 2;
            }

            if (out_meth_mode && out_meth_bw != NULL) {
                /* Gather all bam1_t* in the QNAME group, propagate QC fail, emit. */
                int total = 0;
                for (int k = 0; k < group_size; ++k) total += ret->seqs[i+k].n_bams;
                if (total > 0) {
                    struct bam1_t **group = (struct bam1_t **)malloc(total * sizeof(struct bam1_t *));
                    if (group == NULL)
                        err_fatal(__func__, "out of memory gathering meth BAM group of %d", total);
                    int idx = 0;
                    for (int k = 0; k < group_size; ++k) {
                        for (int j = 0; j < ret->seqs[i+k].n_bams; ++j) {
                            group[idx++] = (struct bam1_t *)ret->seqs[i+k].bams[j];
                        }
                    }
                    meth_bam_group_propagate_qcfail(group, total);
                    for (int j = 0; j < total; ++j) {
#ifndef DISABLE_OUTPUT
                        if (meth_bam_writer_write(out_meth_bw, group[j]) < 0)
                            err_fatal(__func__, "failed to write meth BAM record");
#endif
                        bam_writer_free(group[j]);
                    }
                    free(group);
                }
            } else if (out_bam_bw != NULL) {
                for (int k = 0; k < group_size; ++k) {
                    for (int j = 0; j < ret->seqs[i+k].n_bams; ++j) {
#ifndef DISABLE_OUTPUT
                        if (bam_writer_write(out_bam_bw, (struct bam1_t *)ret->seqs[i+k].bams[j]) < 0)
                            err_fatal(__func__, "failed to write BAM record");
#endif
                        bam_writer_free((struct bam1_t *)ret->seqs[i+k].bams[j]);
                    }
                }
            } else {
                for (int k = 0; k < group_size; ++k) {
                    if (ret->seqs[i+k].sam) {
#ifndef DISABLE_OUTPUT
                        fputs(ret->seqs[i+k].sam, aux->fp);
#endif
                    }
                    /* meth_mode populates bams[] regardless of writer state;
                     * under DISABLE_OUTPUT both writers are forced NULL and
                     * we land here, so the per-record bam1_t allocations
                     * would otherwise leak (only the pointer array below is
                     * freed). Profile-build only, but it skews the very
                     * Maximum RSS that bench/run.sh records. */
                    for (int j = 0; j < ret->seqs[i+k].n_bams; ++j) {
                        bam_writer_free((struct bam1_t *)ret->seqs[i+k].bams[j]);
                    }
                }
            }

            for (int k = 0; k < group_size; ++k) {
                if (sp_enabled() && ret->seqs[i+k].sam) sp_wbytes += (long)strlen(ret->seqs[i+k].sam);
                /* PIPE-F6: name/seq/qual are carved from ret->read_arena, which
                 * is freed once below — do NOT free them individually here.
                 * comment stays heap-owned (see the reader / --meth notes), and
                 * sam/bams are allocated during processing; those still free
                 * per-read. meth_orig_seq is a step-0 heap strdup (NULL outside
                 * --meth; free() is NULL-safe). */
                free(ret->seqs[i+k].comment);
                free(ret->seqs[i+k].sam);
                free(ret->seqs[i+k].bams);
                free(ret->seqs[i+k].meth_orig_seq);
            }
            i += group_size;
        }
        if (sp_enabled()) {
            ret->prof.write_wall = sp_wall() - sp_w0;
            ret->prof.write_bytes = sp_wbytes;
            /* bam_mode, NOT mem_opt_records_are_bam(): this asks whether the
             * write path DEFLATES, and only --bam does. A --meth run without
             * --bam builds bam1_t but htslib serializes them as plain text, so
             * there is no compression stage to fuse — it belongs in `else`. */
            if (aux->opt->bam_mode) {            /* htslib fuses compress+diskwrite */
                ret->prof.write_compress = ret->prof.write_wall;   /* diskwrite stays NaN */
            } else {
                ret->prof.write_diskwrite = ret->prof.write_wall;
                ret->prof.write_compress = 0.0;
            }
            static long g_sp_chunk = 0;
            ret->prof.chunk = __sync_fetch_and_add(&g_sp_chunk, 1);
            sp_add_chunk(&ret->prof);
        }
        /* PIPE-F6: every name/seq/qual freed above by the former per-field loop
         * now lives in this one arena; release it once for the whole chunk.
         * All uses are done: output was written above and the seq/qual bytes
         * are never read after the SAM/BAM string was built. NULL-safe. */
        read_arena_destroy(ret->read_arena);
        free(ret->seqs);
        free(ret);
        tprof[SAM_IO][0] += __rdtsc() - tim;

        return 0;
    } // step 2

    return 0;
}

static void *ktp_worker(void *data)
{
    ktp_worker_t *w = (ktp_worker_t*) data;
    ktp_t *p = w->pl;

    while (w->step < p->n_steps) {
        // test whether we can kick off the job with this worker
        double sp_i0 = sp_enabled() ? sp_wall() : 0.0;   // idle = time waiting for our turn
        int pthread_ret = pthread_mutex_lock(&p->mutex);
        assert(pthread_ret == 0);
        for (;;) {
            int i;
            // test whether another worker is doing the same step
            for (i = 0; i < p->n_workers; ++i) {
                if (w == &p->workers[i]) continue; // ignore itself
                if (p->workers[i].step <= w->step && p->workers[i].index < w->index)
                    break;
            }
            if (i == p->n_workers) break; // no workers with smaller indices are doing w->step or the previous steps
            pthread_ret = pthread_cond_wait(&p->cv, &p->mutex);
            assert(pthread_ret == 0);
        }
        pthread_ret = pthread_mutex_unlock(&p->mutex);
        assert(pthread_ret == 0);
        if (sp_enabled()) sp_add_idle(w->step, sp_wall() - sp_i0);   /* idle attributed to next step */

        // working on w->step
        w->data = kt_pipeline(p->shared, w->step, w->step? w->data : 0, w->opt, *(w->w)); // for the first step, input is NULL

        // update step and let other workers know
        pthread_ret = pthread_mutex_lock(&p->mutex);
        assert(pthread_ret == 0);
        w->step = w->step == p->n_steps - 1 || w->data? (w->step + 1) % p->n_steps : p->n_steps;

        if (w->step == 0) w->index = p->index++;
        pthread_ret = pthread_cond_broadcast(&p->cv);
        assert(pthread_ret == 0);
        pthread_ret = pthread_mutex_unlock(&p->mutex);
        assert(pthread_ret == 0);
    }
    pthread_exit(0);
}

static int process(void *shared, gzFile gfp, gzFile gfp2, int pipe_threads)
{
    ktp_aux_t   *aux = (ktp_aux_t*) shared;
    worker_t     w;
    mem_opt_t   *opt = aux->opt;
    int32_t nthreads = opt->n_threads; // global variable for profiling!
    w.nthreads = opt->n_threads;

#if NUMA_ENABLED
    int  deno = 1;
    int tc = numa_num_task_cpus();
    int tn = numa_num_task_nodes();
    int tcc = numa_num_configured_cpus();
    fprintf(stderr, "num_cpus: %d, num_numas: %d, configured cpus: %d\n", tc, tn, tcc);
    int ht = HTStatus();
    if (ht) deno = 2;

    if (nthreads < tcc/tn/deno) {
        fprintf(stderr, "Enabling single numa domain...\n\n");
        // numa_set_preferred(0);
        // bitmask mask(0);
        struct bitmask *mask = numa_bitmask_alloc(numa_num_possible_nodes());
        numa_bitmask_clearall(mask);
        numa_bitmask_setbit(mask, 0);
        numa_bind(mask);
        numa_bitmask_free(mask);
    }
#else
    /* Report platform info on non-NUMA systems (e.g., macOS/Apple Silicon) */
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
    HTStatus();
#endif
#endif
#if AFF && (__linux__)
    { // Affinity/HT stuff
        unsigned int cpuid[4];
        asm volatile
            ("cpuid" : "=a" (cpuid[0]), "=b" (cpuid[1]), "=c" (cpuid[2]), "=d" (cpuid[3])
             : "0" (0xB), "2" (1));
        int num_logical_cpus = cpuid[1] & 0xFFFF;

        asm volatile
            ("cpuid" : "=a" (cpuid[0]), "=b" (cpuid[1]), "=c" (cpuid[2]), "=d" (cpuid[3])
             : "0" (0xB), "2" (0));
        int num_ht = cpuid[1] & 0xFFFF;
        int num_total_logical_cpus = get_nprocs_conf();
        int num_sockets = num_total_logical_cpus / num_logical_cpus;
        fprintf(stderr, "#sockets: %d, #cores/socket: %d, #logical_cpus: %d, #ht/core: %d\n",
                num_sockets, num_logical_cpus/num_ht, num_total_logical_cpus, num_ht);

        for (int i=0; i<num_total_logical_cpus; i++) affy[i] = i;
        int slookup[256] = {-1};

        if (num_ht == 2 && num_sockets == 2)  // generalize it for n sockets
        {
            for (int i=0; i<num_total_logical_cpus; i++) {
                std::ostringstream ss;
                ss << i;
                std::string str = "/sys/devices/system/cpu/cpu"+ ss.str();
                str = str +"/topology/thread_siblings_list";
                // std::cout << str << std::endl;
                // std::string str = "cpu.txt";
                FILE *fp = fopen(str.c_str(), "r");
                if (fp == NULL) {
                    fprintf("Error: Cant open the file..\n");
                    break;
                }
                else {
                    int a, b, v;
                    char ch[10] = {'\0'};
                    fgets(ch, 10, fp);
                    v = sscanf(ch, "%u,%u",&a,&b);
                    if (v == 1) v = sscanf(ch, "%u-%u",&a,&b);
                    if (v == 1) {
                        fprintf(stderr, "Mis-match between HT and threads_sibling_list...%s\n", ch);
                        fprintf(stderr, "Continuing with default affinity settings..\n");
                        break;
                    }
                    slookup[a] = 1;
                    slookup[b] = 2;
                    fclose(fp);
                }
            }
            int a = 0, b = num_total_logical_cpus / num_ht;
            for (int i=0; i<num_total_logical_cpus; i++) {
                if (slookup[i] == -1) {
                    fprintf(stderr, "Unseen cpu topology..\n");
                    break;
                }
                if (slookup[i] == 1) affy[a++] = i;
                else affy[b++] = i;
            }
        }
    }
#endif

    /* PIPE-F24: do NOT pre-size the read-count-sized scratch from a
     * bytes/NREADS_ESTIMATE_AVG_BASES heuristic. That estimate (chunk_bytes/100
     * + 10) is right only for ~100 bp reads: it OVER-allocates for longer reads
     * (pure waste — the pool is never shrunk) and UNDER-allocates for shorter
     * ones, forcing a free() + large calloc() on the first chunk. Start at 0 and
     * let the grow-on-demand path in kt_pipeline (step 1) size it EXACTLY from
     * the actual parsed read count (ret->n_seqs) and reuse it across chunks.
     * Correctness is unchanged: that path runs BEFORE any read indexes the
     * buffer and guarantees w.nreads >= n_seqs, with identical calloc zero-init.
     *
     * `regs` is the only such buffer now — chain_scratch/seed_scratch are sized
     * from nthreads * BATCH_SIZE in worker_alloc and never depend on nreads, so
     * a 0 here costs them nothing. That is also what retires the worst case this
     * comment was written about: the multi-GB overshoot was seed_scratch's. */
    int32_t nreads = 0;

    /* All memory allocation */
    memoryAlloc(aux, w, nreads, nthreads);
    fprintf(stderr, "* Threads used (compute): %d\n", nthreads);

    /* pipeline using pthreads */
    ktp_t aux_;
    int p_nt = pipe_threads; // 2;
    int n_steps = 3;

    w.ref_string = aux->ref_string;
    // Mirror into mem_cache so helpers that take only `mmc` (e.g.
    // mem_matesw_batch_pre/post) can call bns_fetch_seq_v2 without
    // threading ref_string through every signature on the way down.
    w.mmc.ref_string = aux->ref_string;
    w.fmi = aux->fmi;
    /* D3 (--meth, PR-3): hand the ORIGINAL bns/pac/.0123 to the worker so the
     * (remapped, original-coord) seeds chain/extend/pair/output against the
     * original reference. NULL outside --meth → mem_aln_* fall back to the seed
     * index and the non-meth path is unchanged. The batched mate-rescue helpers
     * read the ref via w.mmc.ref_string, so point THAT at the original .0123 too
     * in --meth (else mate rescue would fetch SEED bases at original coords). */
    w.meth_orig_bns        = aux->meth_orig_bns;
    w.meth_orig_pac        = aux->meth_orig_pac;
    w.meth_orig_ref_string = aux->meth_orig_ref_string;
    if (aux->opt->meth_mode && aux->meth_orig_ref_string != NULL)
        w.mmc.ref_string = aux->meth_orig_ref_string;
    w.nreads  = nreads;
    // w.memSize = nreads;

    aux_.n_workers = p_nt;
    aux_.n_steps = n_steps;
    aux_.shared = aux;
    aux_.index = 0;
    int pthread_ret = pthread_mutex_init(&aux_.mutex, 0);
    assert(pthread_ret == 0);
    pthread_ret = pthread_cond_init(&aux_.cv, 0);
    assert(pthread_ret == 0);

    fprintf(stderr, "* No. of pipeline threads: %d\n\n", p_nt);
    aux_.workers = (ktp_worker_t*) malloc(p_nt * sizeof(ktp_worker_t));
    xassert(aux_.workers != NULL, "out of memory: aux_.workers");

    for (int i = 0; i < p_nt; ++i) {
        ktp_worker_t *wr = &aux_.workers[i];
        wr->step = 0; wr->pl = &aux_; wr->data = 0;
        wr->index = aux_.index++;
        wr->i = i;
        wr->opt = opt;
        wr->w = &w;
    }

    pthread_t *ptid = (pthread_t *) calloc(p_nt, sizeof(pthread_t));
    xassert(ptid != NULL, "out of memory: ptid");

    for (int i = 0; i < p_nt; ++i)
        pthread_create(&ptid[i], 0, ktp_worker, (void*) &aux_.workers[i]);

    for (int i = 0; i < p_nt; ++i)
        pthread_join(ptid[i], 0);

    pthread_ret = pthread_mutex_destroy(&aux_.mutex);
    assert(pthread_ret == 0);
    pthread_ret = pthread_cond_destroy(&aux_.cv);
    assert(pthread_ret == 0);

    free(ptid);
    free(aux_.workers);
    /***** pipeline ends ******/

    /* Retire the kt_for() worker pool (created lazily on the first chunk).
     * No kt_for() calls remain past this point. */
    kt_pool_destroy();

    fprintf(stderr, "[0000] Computation ends..\n");

    /* Dealloc per-worker scratch buffers allocated in the header section */
    worker_free(w, nthreads);

    return 0;
}

static void update_a(mem_opt_t *opt, const mem_opt_t *opt0)
{
    if (opt0->a) { // matching score is changed
        if (!opt0->b) opt->b *= opt->a;
        if (!opt0->T) opt->T *= opt->a;
        if (!opt0->o_del) opt->o_del *= opt->a;
        if (!opt0->e_del) opt->e_del *= opt->a;
        if (!opt0->o_ins) opt->o_ins *= opt->a;
        if (!opt0->e_ins) opt->e_ins *= opt->a;
        if (!opt0->zdrop) opt->zdrop *= opt->a;
        if (!opt0->pen_clip5) opt->pen_clip5 *= opt->a;
        if (!opt0->pen_clip3) opt->pen_clip3 *= opt->a;
        if (!opt0->pen_unpaired) opt->pen_unpaired *= opt->a;
    }
}

/* True if `p` names something on disk, or is the base of a bwa index (the
 * <idxbase> argument is a prefix, not a file: "ref.fa" with "ref.fa.amb"
 * alongside it). Used only to decide whether a positional that *looks* like an
 * option value is in fact a real path the user meant. */
static int path_exists(const char *p)
{
    char buf[PATH_MAX];
    if (p == NULL || *p == '\0') return 0;
    if (access(p, F_OK) == 0) return 1;
    /* Probe one index sidecar rather than all of them: .amb is written by every
     * index variant (plain and --meth) and is the smallest. */
    if (snprintf(buf, sizeof(buf), "%s.amb", p) < (int)sizeof(buf)
        && access(buf, F_OK) == 0) return 1;
    return 0;
}

/* If `s` is a value that belongs to one of the --meth family's options rather
 * than a path, return the flag it belongs to; else NULL. Deliberately narrow:
 * every token here is a closed-vocabulary keyword no one would name a reference
 * after, and the caller additionally requires that no such path exists. */
static const char *stray_option_value_flag(const char *s)
{
    static const struct { const char *value, *flag; } known[] = {
        { "taps",      "--meth"         }, { "emseq",     "--meth" },
        { "em-seq",    "--meth"         }, { "bisulfite", "--meth" },
        { "collapsed", "--meth-scoring" }, { "genomic",   "--meth-scoring" },
        { "neutral",   "--meth-scoring" },
        { "XR",        "--meth-tags"    }, { "XG",        "--meth-tags" },
        { "XM",        "--meth-tags"    }, { "all",       "--meth-tags" },
        { "none",      "--meth-tags"    },
        { "spec30",    "--meth-seed-prune" }, { "baseline",  "--meth-seed-prune" },
        { "off",       "--meth-seed-prune" },
    };
    if (s == NULL) return NULL;
    /* A `^`-prefixed exclusion can only have come from --meth-tags. Its `-XM`
     * synonym needs no case here: `X` takes an argument in the short optstring,
     * so getopt binds `-XM` as `-X M` and it never reaches a positional slot. */
    if (s[0] == '^') return "--meth-tags";
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); ++i)
        if (strcasecmp(s, known[i].value) == 0) return known[i].flag;
    return NULL;
}

/* The corrective spelling for a stray value, tailored to how it got orphaned. */
static const char *stray_option_value_advice(const char *s)
{
    const char *flag = stray_option_value_flag(s);
    if (flag == NULL) return "";
    if (strcmp(flag, "--meth") == 0)
        return "       --meth takes an OPTIONAL argument, which getopt only binds with '=':\n"
               "       write --meth=taps, not --meth taps.";
    if (strcmp(flag, "--meth-tags") == 0)
        return "       --meth-tags takes ONE comma-separated list, not a space-separated one:\n"
               "       write --meth-tags XR,XG (or --meth-tags '^XM'), not --meth-tags XR XG.";
    if (strcmp(flag, "--meth-seed-prune") == 0)
        return "       --meth-seed-prune takes an OPTIONAL argument, which getopt only binds with '=':\n"
               "       write --meth-seed-prune=baseline, not --meth-seed-prune baseline.";
    return "       Pass it as the argument to that flag, e.g. --meth-scoring genomic.";
}

static void usage(const mem_opt_t *opt)
{
    fprintf(stderr, "Usage: bwa-mem3 mem [options] <idxbase> <in1.fq> [in2.fq]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  Algorithm options:\n");
    fprintf(stderr, "    -o STR        Output SAM file name\n");
    fprintf(stderr, "    --bam[=N]     Emit BAM instead of SAM text. N=0 (default) = uncompressed;\n");
    fprintf(stderr, "                  1..9 = BGZF deflate levels. Writes to stdout; redirect with `>`.\n");
    fprintf(stderr, "    -t INT        number of threads, up to %d; values above %d are\n"
                     "                  clamped to %d [%d]\n",
            MAX_THREADS, MAX_THREADS, MAX_THREADS, opt->n_threads);
    fprintf(stderr, "    -k INT        minimum seed length [%d]\n", opt->min_seed_len);
    fprintf(stderr, "    -w INT        band width for banded alignment [%d]\n", opt->w);
    fprintf(stderr, "    -d INT        off-diagonal X-dropoff [%d]\n", opt->zdrop);
    fprintf(stderr, "    -r FLOAT      look for internal seeds inside a seed longer than {-k} * FLOAT [%g]\n", opt->split_factor);
    fprintf(stderr, "    -y INT        seed occurrence for the 3rd round seeding [%ld]\n", (long)opt->max_mem_intv);
    fprintf(stderr, "    -c INT        skip seeds with more than INT occurrences [%d]\n", opt->max_occ);
    fprintf(stderr, "    --smem-dedup  dedup identical SMEMs before chaining: fewer SA lookups, ~10%% fewer; opt-in, NOT byte-identical (changes XS/secondary on a small fraction of reads) [off]\n");
    fprintf(stderr, "    --dedup STR   extension-DP job dedup: 'off', 'on' (dedup identical jobs within each batch), or 'auto' (measure net benefit at runtime, latch, and periodically re-probe); alignment records byte-identical in every mode (@PG excluded, it embeds argv) [auto]\n");
    fprintf(stderr, "    --dedup-reads STR  whole-read-pair memoization: 'off', 'on', or 'auto' (measure the duplicate rate and net benefit at runtime, latch, and periodically re-probe); aligns once per distinct pair within a chunk and replays the per-read SAM stage, so alignment records are byte-identical in every mode. Benefits amplicon/UMI panels with PCR duplicates; ~no effect on WGS/exome [auto]\n");
    fprintf(stderr, "    --huge-pages  back the index with 1 GB huge pages via mimalloc when the host has enough free 1 GB pages reserved; cuts dTLB misses in seeding; Linux only, alignment records byte-identical (only @PG CL differs, recording the flag), safe no-op otherwise [off]\n");
    fprintf(stderr, "    --skip-contained-ext  skip banded-SW extension of seeds contained (same diagonal) in a longer in-chain seed; byte-identical on short/medium non-meth reads (NOT on kilobase-scale long reads); no effect under --meth [off]\n");
    fprintf(stderr, "    --max-extend-chains INT  cap chains extended per read to the top-INT by weight; ~23%% less alignment CPU, high-confidence placement unaffected; ignored for reads with >4096 chains; opt-in, NOT byte-identical (0 = off) [%d]\n", opt->max_extend_chains);
    fprintf(stderr, "    --adaptive-band  adaptive banded-SW: start tight and expand each pair to its chain-geometry band on long-extension reads; ~1.3x on medium reads (SBX ~240bp), no-op on short reads; kilobase-scale HiFi/ONT do not run at default settings; opt-in, NOT byte-identical [%s]\n", opt->band_start? "on":"off");
    fprintf(stderr, "    --no-adaptive-band  disable adaptive banded-SW (exact, byte-identical full-width extension; also disables the certified band); overrides --adaptive-band and the --adaptive-band that --fast enables\n");
    fprintf(stderr, "    --no-band-cert  disable the certified adaptive extension band (on by default): run the full-width extension ladder for every pair instead of the narrow-probe-plus-certificate. The certified band is byte-identical to full-width, so on a plain run this only removes the speedup; it has no effect under --fast, --adaptive-band, or --no-adaptive-band (which already disable the certified band). Escape hatch / A-B handle [%s]\n", opt->band_cert? "on":"off");
    fprintf(stderr, "    --extend-mate-concordant[=INT]  when --max-extend-chains caps a PE read, also keep any chain concordant (same contig, FR, within INT bp) with a mate chain; recovers the true pair's low-weight chain the cap would drop (mainly --meth). Bare = auto (window = estimated proper-pair insert high bound); =INT = fixed bp; =0 = off. Opt-in, NOT byte-identical [%s]\n", opt->mate_concordant_window? (opt->mate_concordant_window<0? "auto":"fixed") : "off");
    fprintf(stderr, "    --extend-tie-frac FLOAT  extend a chain (ranked at/after --extend-tie-floor) only if its (integer) weight >= floor(FLOAT * the best chain's weight) -- trims non-competitive tail chains from banded-SW while still extending genuine near-ties. The threshold is truncated to an integer, so e.g. best=19 with FLOAT=0.95 gates at 18, not 18.05. Complements --max-extend-chains (the count cap) and also gates on its own when that cap is off; must be in [0,1]; opt-in, NOT byte-identical (0 = off) [%g]\n", opt->extend_tie_frac);
    fprintf(stderr, "    --extend-csub  when --extend-tie-frac/--max-extend-chains drops chains, seed the primary's competitor score with a calibrated estimate of the best dropped chain so MAPQ is not inflated by the pruning; opt-in, NOT byte-identical [%s]\n", opt->extend_csub? "on":"off");
    fprintf(stderr, "    --extend-tie-floor INT  extend at least the top-INT chains regardless of --extend-tie-frac, but only within --max-extend-chains (the count cap runs first, so a floor above --max-extend-chains still keeps at most --max-extend-chains); 0 = no floor (the fraction gate governs from rank 0, best chain always kept); only meaningful with --extend-tie-frac > 0 [%d]\n", opt->extend_tie_floor);
    fprintf(stderr, "    -D FLOAT      drop chains shorter than FLOAT fraction of the longest overlapping chain [%.2f]\n", opt->drop_ratio);
    fprintf(stderr, "    -W INT        discard a chain if seeded bases shorter than INT [0]\n");
    fprintf(stderr, "    -m INT        perform at most INT rounds of mate rescues for each read [%d]\n", opt->max_matesw);
    fprintf(stderr, "    --rescue-kmer[=K]  band the mate-rescue Smith-Waterman to a K-mer exact-match anchor\n");
    fprintf(stderr, "                  diagonal, falling back to the full insert window when no anchor. K is\n");
    fprintf(stderr, "                  1..%d (bare = %d, =0 = off); the value must be attached with '='.\n",
            MEM_RESCUE_KMER_MAX, MEM_RESCUE_KMER_DEFAULT);
    fprintf(stderr, "                  Opt-in speedup, NOT byte-identical; enabled by --fast [off]\n");
    fprintf(stderr, "    --rescue-band INT  half-width (bp) of the band around the anchor diagonal, 1..%d [%d]\n",
            MEM_RESCUE_BAND_MAX, opt->rescue_band);
    fprintf(stderr, "    --rescue-skip  skip the mate-rescue Smith-Waterman outright when no K-mer anchor\n");
    fprintf(stderr, "                  clears the vote floor, instead of falling back to the full window.\n");
    fprintf(stderr, "                  Requires --rescue-kmer. Drops rescues rather than shortening them,\n");
    fprintf(stderr, "                  so it can lose alignments; NOT part of --fast [off]\n");
    fprintf(stderr, "    -S            skip mate rescue\n");
    fprintf(stderr, "    -P            skip pairing; mate rescue performed unless -S also in use\n");
    fprintf(stderr, "    --hic         map Hi-C reads; equivalent to -5SP (note: -P alone still runs\n");
    fprintf(stderr, "                  mate rescue -- use --hic or -5SP to skip it too) [off]\n");
    fprintf(stderr, "    --fast        speed preset: -m 10 -y 0 --min-ext-len 30 --smem-dedup --rescue-kmer=6\n");
    fprintf(stderr, "                  --skip-contained-ext --max-extend-chains 20 --adaptive-band\n");
    fprintf(stderr, "                  --extend-mate-concordant --extend-tie-frac 0.95 --extend-tie-floor 1\n");
    fprintf(stderr, "                  --extend-csub (under --meth: --max-extend-chains 10, -s 2). Opt-in; explicit\n");
    fprintf(stderr, "                  flags override where applicable; --smem-dedup,\n");
    fprintf(stderr, "                  --skip-contained-ext and --adaptive-band are enabled\n");
    fprintf(stderr, "                  (pass --no-adaptive-band to keep exact extension under --fast).\n");
    fprintf(stderr, "                  Also switches the alignment-region dedup sort to a strict\n");
    fprintf(stderr, "                  total order (faster, but resolves equal-end-position ties\n");
    fprintf(stderr, "                  differently from bwa-mem2, which the default reproduces).\n");
    fprintf(stderr, "                  NOT byte-identical to the default (divergence confined to the\n");
    fprintf(stderr, "                  low-confidence tail). Also implies --chunk-cap 256000000.\n");
    fprintf(stderr, "    --compat STR  shape output to be byte-identical to another aligner.\n");
    fprintf(stderr, "                  Targets: %s\n", compat_target_selectable_list());
    fprintf(stderr, "                  Both targets drop the HN:i tag and ignore the <prefix>.hdr /\n");
    fprintf(stderr, "                  <baseprefix>.dict sidecar, so @SQ is generated as bare SN/LN\n");
    fprintf(stderr, "                  (+AH:* on ALT contigs). They differ where the upstreams do:\n");
    fprintf(stderr, "                  bwa-mem2: also suppress MQ:i and the default @HD -- bwa-mem2\n");
    fprintf(stderr, "                  v2.2.1 forked at bwa 0.7.17, before either landed.\n");
    fprintf(stderr, "                  bwa-mem:  keep both -- bwa 0.7.18+ emits them. Pinned at 0.7.19.\n");
    fprintf(stderr, "                  Shapes output only; changes no alignment. @PG still differs (it\n");
    fprintf(stderr, "                  is run-specific) -- exclude it when comparing. Mutually exclusive\n");
    fprintf(stderr, "                  with --fast (which changes alignments) and with --meth (neither\n");
    fprintf(stderr, "                  target has a bisulfite mode) -- combining them is an error [off]\n");
    fprintf(stderr, "Scoring options:\n");
    fprintf(stderr, "   -A INT        score for a sequence match, which scales options -TdBOELU unless overridden [%d]\n", opt->a);
    fprintf(stderr, "   -B INT        penalty for a mismatch [%d]\n", opt->b);
    fprintf(stderr, "   -O INT[,INT]  gap open penalties for deletions and insertions [%d,%d]\n", opt->o_del, opt->o_ins);
    fprintf(stderr, "   -E INT[,INT]  gap extension penalty; a gap of size k cost '{-O} + {-E}*k' [%d,%d]\n", opt->e_del, opt->e_ins);
    fprintf(stderr, "   -L INT[,INT]  penalty for 5'- and 3'-end clipping [%d,%d]\n", opt->pen_clip5, opt->pen_clip3);
    fprintf(stderr, "   -U INT        penalty for an unpaired read pair [%d]\n", opt->pen_unpaired);
//  fprintf(stderr, "   -x STR        read type. Setting -x changes multiple parameters unless overriden [null]\n");
//  fprintf(stderr, "                 pacbio: -k17 -W40 -r10 -A1 -B1 -O1 -E1 -L0  (PacBio reads to ref)\n");
//  fprintf(stderr, "                 ont2d: -k14 -W20 -r10 -A1 -B1 -O1 -E1 -L0  (Oxford Nanopore 2D-reads to ref)\n");
//  fprintf(stderr, "                 intractg: -B9 -O16 -L5  (intra-species contigs to ref)\n");
    fprintf(stderr, "Input/output options:\n");
    fprintf(stderr, "   -p            smart pairing (ignoring in2.fq)\n");
    fprintf(stderr, "   -R STR        read group header line such as '@RG\\tID:foo\\tSM:bar' [null]\n");
    fprintf(stderr, "   -H STR/FILE   insert STR to header if it starts with @; or insert lines in FILE [null]\n");
    fprintf(stderr, "   -j            treat ALT contigs as part of the primary assembly (i.e. ignore <idxbase>.alt file)\n");
    fprintf(stderr, "   -5            for split alignment, take the alignment with the smallest coordinate as primary\n");
    fprintf(stderr, "   -q            don't modify mapQ of supplementary alignments\n");
    fprintf(stderr, "   -K INT        process INT input bases in each batch regardless of nThreads (for reproducibility) []\n");
    fprintf(stderr, "   --chunk-cap INT\n");
    fprintf(stderr, "                 upper bound (bases) on the default nThreads-scaled batch size;\n");
    fprintf(stderr, "                 0 = off. Off by default so batching matches bwa/bwa-mem2 exactly\n");
    fprintf(stderr, "                 at any -t. Capping re-partitions the input and is NOT\n");
    fprintf(stderr, "                 byte-identical. Prefer -K if you want a fixed batch size AND\n");
    fprintf(stderr, "                 reproducibility [0]\n");
    fprintf(stderr, "   --cohort-slices INT\n");
    fprintf(stderr, "                 read the FIRST batch as a ramp of up to INT growing slices, so\n");
    fprintf(stderr, "                 alignment starts before the whole batch has been read. 0 = off\n");
    fprintf(stderr, "                 (single read). The slice SIZES come from --cohort-ramp-first and\n");
    fprintf(stderr, "                 --cohort-ramp-ratio. Later batches are always read whole, since\n");
    fprintf(stderr, "                 their read already overlaps the previous batch's compute. Does NOT\n");
    fprintf(stderr, "                 move the batch boundary, so output is unchanged. Max %d. Overridden\n",
            (int)COHORT_SLICES_MAX);
    fprintf(stderr, "                 by BWA_MEM3_COHORT_SLICES [%d]\n", (int)COHORT_SLICES_DEFAULT);
    fprintf(stderr, "   --cohort-ramp-first INT\n");
    fprintf(stderr, "                 bases in the FIRST ramp slice. An absolute size, not a fraction of\n");
    fprintf(stderr, "                 the batch: this slice's read overlaps nothing and each extra slice\n");
    fprintf(stderr, "                 costs one more pipeline pass, and neither cost scales with the\n");
    fprintf(stderr, "                 batch. Capped at half the batch so a small batch still gets two\n");
    fprintf(stderr, "                 slices. 0 selects the fractional shape (batch / ratio^slices).\n");
    fprintf(stderr, "                 Output-neutral. Overridden by BWA_MEM3_COHORT_RAMP_FIRST [%lld]\n",
            (long long)cohort_ramp_first_default);
    fprintf(stderr, "   --cohort-ramp-ratio FLOAT\n");
    fprintf(stderr, "                 growth factor between consecutive ramp slices. Growing faster than\n");
    fprintf(stderr, "                 the reader can deliver the next slice while the current one is\n");
    fprintf(stderr, "                 being aligned stalls the pipeline, and that ceiling FALLS as -t\n");
    fprintf(stderr, "                 rises because only compute scales with threads. Output-neutral.\n");
    fprintf(stderr, "                 Overridden by BWA_MEM3_COHORT_RAMP_RATIO [%.1f]\n",
            cohort_ramp_ratio_default);
    fprintf(stderr, "   -v INT        verbose level: 1=error, 2=warning, 3=message, 4+=debugging [%d]\n", bwa_verbose);
    fprintf(stderr, "   -T INT        minimum score to output [%d]\n", opt->T);
    fprintf(stderr, "   -h INT[,INT]  if there are <INT hits with score >%.2f%% of the max score, output all in XA [%d,%d]\n",
            opt->XA_drop_ratio * 100.0, opt->max_XA_hits, opt->max_XA_hits_alt);
    fprintf(stderr, "   -z FLOAT      the fraction of the max score to use with -h [%.2f]\n", opt->XA_drop_ratio);
    fprintf(stderr, "   -u            output XB instead of XA; XB is XA with the alignment score and mapping quality added\n");
    fprintf(stderr, "   -a            output all alignments for SE or unpaired PE\n");
    fprintf(stderr, "   -C            append FASTA/FASTQ comment to SAM output\n");
    fprintf(stderr, "   -V            output the reference FASTA header in the XR tag\n");
    fprintf(stderr, "   -Y            use soft clipping for supplementary alignments\n");
    fprintf(stderr, "   -M            mark shorter split hits as secondary\n");
    fprintf(stderr, "   -I FLOAT[,FLOAT[,INT[,INT]]]\n");
    fprintf(stderr, "                 specify the mean, standard deviation (10%% of the mean if absent), max\n");
    fprintf(stderr, "                 (4 sigma from the mean if absent) and min of the insert size distribution.\n");
    fprintf(stderr, "                 Sets the FR distribution only and skips inference, so FF/RF/RR get none:\n");
    fprintf(stderr, "                 no proper-pair flag or mate rescue there. As in bwa/bwa-mem2. [inferred]\n");
    fprintf(stderr, "Methylation (--meth) options:\n");
    fprintf(stderr, "   --meth[=CHEM] enable inline bwameth-style C→T/G→A read conversion + meth-aware\n");
    fprintf(stderr, "                 record emission (XM:Z/XG:Z/XR:Z). Output is SAM text by default,\n");
    fprintf(stderr, "                 as without --meth; add --bam for BAM. Requires the reference to\n");
    fprintf(stderr, "                 have been built with `bwa-mem3 index --meth` (emits the original\n");
    fprintf(stderr, "                 index plus a ref.fa.meth.* converted seed index).\n");
    fprintf(stderr, "                 CHEM selects the chemistry and thus the XM:Z call polarity:\n");
    fprintf(stderr, "                   emseq  bisulfite/EM-seq, UNmethylated C converts [default]\n");
    fprintf(stderr, "                          (aliases: em-seq, bisulfite)\n");
    fprintf(stderr, "                   taps   TET-assisted pyridine borane, METHYLATED C converts;\n");
    fprintf(stderr, "                          also defaults --meth-scoring to neutral\n");
    fprintf(stderr, "                 NOTE: use --meth=taps (with '='), not --meth taps. Running TAPS\n");
    fprintf(stderr, "                 data without =taps inverts every methylation call.\n");
    fprintf(stderr, "   --meth-scoring collapsed|genomic|neutral\n");
    fprintf(stderr, "                 scoring mode. Default depends on chemistry: collapsed for\n");
    fprintf(stderr, "                 --meth/--meth=emseq, neutral for --meth=taps (TAPS conversions\n");
    fprintf(stderr, "                 are sparse, so collapsing costs specificity it can't repay).\n");
    fprintf(stderr, "                 collapsed: C/T (and G/A) interchangeable, bwameth-compatible\n");
    fprintf(stderr, "                 placement (sets -B 2).\n");
    fprintf(stderr, "                 genomic: free only the conversion direction, scored as a full\n");
    fprintf(stderr, "                 match, keep variants as mismatches (variant-aware: variants\n");
    fprintf(stderr, "                 outside the conversion direction visible in NM/MD; -B 4).\n");
    fprintf(stderr, "                 neutral: free only the conversion direction but score it 0\n");
    fprintf(stderr, "                 (tolerated, not rewarded); best for TAPS (variant-aware:\n");
    fprintf(stderr, "                 variants visible in NM/MD as in genomic; -B 4).\n");
    fprintf(stderr, "                 In genomic/neutral a real variant in the conversion direction\n");
    fprintf(stderr, "                 itself (C->T at a reference C) is indistinguishable from a\n");
    fprintf(stderr, "                 conversion and stays hidden in NM/MD.\n");
    fprintf(stderr, "   --meth-seed-prune[=spec30|baseline|off]\n");
    fprintf(stderr, "                 prune the 3-letter alphabet's short, repetitive spurious SMEMs\n");
    fprintf(stderr, "                 before SA resolution: ~30%% faster --meth at ~0 accuracy cost\n");
    fprintf(stderr, "                 vs truth (measured WGS/WES/em-seq, mapq>=20 accuracy identical).\n");
    fprintf(stderr, "                 ON by default under --meth (spec30) -- --meth is bwa-mem3-native,\n");
    fprintf(stderr, "                 so there is no upstream reference output to preserve. spec30:\n");
    fprintf(stderr, "                 per-read two-regime rule; baseline: drop len<25 & SA-count>1;\n");
    fprintf(stderr, "                 off: restore the un-pruned seeding (byte-identical). NOT\n");
    fprintf(stderr, "                 byte-identical vs off. Env BWAMEM3_METH_SEED_PRUNE (off|spec30|\n");
    fprintf(stderr, "                 baseline) overrides this flag for A/B testing.\n");
    fprintf(stderr, "   --meth-tags SPEC\n");
    fprintf(stderr, "                 which Bismark tags to emit: 'all' (default), 'none', a\n");
    fprintf(stderr, "                 comma-separated list (XR,XG), or ^-prefixed exclusions (^XM).\n");
    fprintf(stderr, "                 Comma-separated, NOT space-separated (a second word becomes\n");
    fprintf(stderr, "                 the reference). An exclusion may be written ^XM or -XM; prefer\n");
    fprintf(stderr, "                 -XM in scripts (bare ^ is a negated glob in zsh EXTENDED_GLOB).\n");
    fprintf(stderr, "                 Unselected tags are not computed. XM:Z is a read-length string\n");
    fprintf(stderr, "                 and dominates the BAM's aux payload; '^XM' drops it for callers\n");
    fprintf(stderr, "                 that recompute from the reference (MethylDackel, biscuit).\n");
    fprintf(stderr, "                 Keep XM for Bismark-family tools (bismark_methylation_extractor,\n");
    fprintf(stderr, "                 methylKit, methtuple, DMRfinder, epialleleR).\n");
    fprintf(stderr, "   --set-as-failed f|r\n");
    fprintf(stderr, "                 flag alignments to the matching strand ('f' or 'r') as QC-fail (0x200)\n");
    fprintf(stderr, "   --chimera-qc\n");
    fprintf(stderr, "                 enable the bwameth.py-style longest-match <44%% chimera heuristic\n");
    fprintf(stderr, "                 (sets 0x200, clears 0x2, caps MAPQ at 1). Off by default; not in Bismark.\n");
    fprintf(stderr, "Supplementary MAPQ rescoring (fg-labs extension):\n");
    fprintf(stderr, "   --supp-rep-hard-cap INT\n");
    fprintf(stderr, "                 force MAPQ=0 for supplementary alignments whose chain contains any seed\n");
    fprintf(stderr, "                 with >=INT genome occurrences (i.e. the supp region is repetitive on its\n");
    fprintf(stderr, "                 own). 0 disables (default). Typical values 5-20; lower = more aggressive.\n");
    fprintf(stderr, "                 Primary MAPQ is unaffected.\n");
    fprintf(stderr, "   --proper-pair-from-emitted\n");
    fprintf(stderr, "                 derive the proper-pair FLAG bit (0x2) from the alignment actually\n");
    fprintf(stderr, "                 emitted rather than the top-scoring one. bwa and bwa-mem2 both use\n");
    fprintf(stderr, "                 the top-scoring one, so this deviates from both and is mutually\n");
    fprintf(stderr, "                 exclusive with --compat. Has no effect unless the index has a .alt\n");
    fprintf(stderr, "                 sidecar: the two differ only for reads with ALT hits [off]\n");
    fprintf(stderr, "Seed ordering (fg-labs extension):\n");
    fprintf(stderr, "   --seed-order STR\n");
    fprintf(stderr, "                 seed emission order before chaining: off|local-longest [off]\n");
    fprintf(stderr, "                 (advanced modes: global-longest, absorb-count, most-absorb; see docs)\n");
    fprintf(stderr, "Input reader:\n");
    fprintf(stderr, "   --legacy-reader\n");
    fprintf(stderr, "                 use the legacy gzFile/kseq input reader instead of the default\n");
    fprintf(stderr, "                 content-detecting fast reader (escape hatch / A-B baseline).\n");
#ifdef STAGE_PROF
    fprintf(stderr, "Profiling:\n");
    fprintf(stderr, "   --profile FILE\n");
    fprintf(stderr, "                 write per-chunk stage profiling TSV to FILE (off by default).\n");
#endif
    fprintf(stderr, "Help:\n");
    fprintf(stderr, "   --help        print this help message and exit\n");
    fprintf(stderr, "Note: Please read the man page for detailed description of the command line and options.\n");
}

/* D3 (--meth) only: load the ORIGINAL reference's bns + pac (un-converted, real
 * chrom names) from `prefix` as resident handles for the future extension/scoring
 * phase, distinct from the seed FM-index. Mirrors indexEle::bwa_idx_load_ele's
 * disk path (bns_restore then slurp the full .pac into memory and close fp_pac).
 * On success writes *bns_out / *pac_out and returns 0; on failure frees any
 * partial allocation, leaves the out-params NULL, and returns -1. */
static int meth_orig_ref_load_handles(const char *prefix,
                                      bntseq_t **bns_out, uint8_t **pac_out)
{
    *bns_out = NULL;
    *pac_out = NULL;
    bntseq_t *bns = bns_restore(prefix);
    if (bns == NULL) {
        fprintf(stderr,
                "ERROR: --meth could not load the original reference bns from "
                "'%s.{amb,ann,pac}'\n", prefix);
        return -1;
    }
    int64_t pac_bytes = bns->l_pac / 4 + 1;
    uint8_t *pac = (uint8_t*) calloc(pac_bytes, 1);
    if (pac == NULL) {
        fprintf(stderr, "ERROR: --meth failed to allocate %lld bytes for the "
                "original reference pac\n", (long long)pac_bytes);
        bns_destroy(bns);
        return -1;
    }
    bwamem_madv_hugepage(pac, pac_bytes);
    /* bns_restore left .pac open in bns->fp_pac; slurp it whole, then close. */
    err_fread_noeof(pac, 1, pac_bytes, bns->fp_pac);
    err_fclose(bns->fp_pac);
    bns->fp_pac = NULL;

    *bns_out = bns;
    *pac_out = pac;
    return 0;
}

/* Free the original-reference handles loaded by meth_orig_ref_load_handles and
 * NULL them. Idempotent (NULL-safe) so it can sit on every exit path the seed
 * index is freed on without double-free risk. */
static void meth_orig_ref_free_handles(ktp_aux_t *aux)
{
    if (aux->meth_orig_pac != NULL) { free(aux->meth_orig_pac); aux->meth_orig_pac = NULL; }
    if (aux->meth_orig_bns != NULL) { bns_destroy(aux->meth_orig_bns); aux->meth_orig_bns = NULL; }
    /* D3 (--meth): meth_orig_ref_string is NULL under pac-fetch (the only path
     * now — the original reference is unpacked from `.pac` on demand, never
     * materialized). This NULL-safe free is retained defensively; if it is ever
     * non-NULL it would be an _mm_malloc'd buffer, freed with _mm_free. */
    if (aux->meth_orig_ref_string != NULL) {
        _mm_free(aux->meth_orig_ref_string);
        aux->meth_orig_ref_string = NULL;
    }
}

int main_mem(int argc, char *argv[])
{
    int          i, c, ignore_alt = 0, no_mt_io = 0;
    int          fixed_chunk_size          = -1;
    char        *p, *rg_line               = 0, *hdr_line = 0;
    const char  *mode                      = 0;
    int          fast                      = 0;
    int          want_huge_pages           = 0;
    /* --chunk-cap: upper bound (bases) on the default `chunk_size * n_threads`
     * batch size. 0 = off, which is the DEFAULT and matches bwa and bwa-mem2
     * exactly (both compute `chunk_size * n_threads` with no cap). Capping
     * re-partitions the input, which changes each batch's mem_pestat cohort and
     * therefore the output, so it must never be on by default -- see the
     * task_size block below. */
    int64_t      chunk_cap                 = 0;
    int          chunk_cap_set             = 0;
    /* --no-adaptive-band: user opted out of adaptive banded-SW extension. Tracked
     * separately so the opt-out beats both an explicit --adaptive-band (either
     * order) and the --adaptive-band that --fast would otherwise turn on. */
    int          no_adaptive_band          = 0;
    /* --cohort-slices: how many geometric slices to read the FIRST batch in, so
     * compute can start before the whole batch has been read. Byte-identical --
     * the batch (pestat cohort) boundary is unchanged, only the physical read
     * granularity inside it. 0 = one slice, i.e. the pre-slicing behaviour.
     *
     * The default was swept, not guessed. This is the ramp's DEPTH only -- an
     * upper bound on how many slices the first cohort is carved into. Their sizes
     * come from cohort_ramp_first and cohort_ramp_ratio below, so the depth binds
     * only until the ramp reaches the cohort or the 1 Mbase slice floor, which is
     * about where 6 lands on ordinary inputs. The measured sweep is in the PR that
     * introduced this option rather than here: it is a wall-clock result for one
     * host, thread count and input, so it dates in a way the shape of the curve
     * does not, and it would read as live justification long after it stopped
     * being one. */
    int64_t      cohort_slices             = COHORT_SLICES_DEFAULT;
    /* --cohort-ramp-first / --cohort-ramp-ratio: the ramp's shape. Output-neutral
     * at any setting -- these change slice SIZES, and the cohort boundary is
     * pinned by the slice-target clamp, not by how the cohort is carved up.
     *
     * A ramp schedule pays exactly three costs, all measured on a 5M-read WGS
     * slice/hg38 (c8g.16xlarge, warm cache, from --profile):
     *
     *   fill      the first slice's read, which by definition overlaps nothing.
     *             Reading is single-threaded and flat at 4.83 ms/Mbase at every -t.
     *   stalls    while step 1 computes slice k the reader must deliver slice k+1,
     *             so growth of r stalls unless r <= (compute s/base)/(read s/base).
     *             Compute is 31.5 / 15.4 / 7.87 ms/Mbase at -t 16 / 32 / 64, so
     *             that ceiling is 6.5 / 3.2 / 1.63 -- it FALLS as threads are
     *             added, because only compute scales.
     *   overhead  every extra slice is one more kt_for pass over the pipeline:
     *             0.0351 / 0.0498 / 0.0689 s per slice at -t 16 / 32 / 64,
     *             measured from Sproc_wall against ramp slice count.
     *
     * Those pull in opposite directions and the third is easy to forget: at low
     * -t stalls are impossible and overhead rules, so few large slices win; at
     * high -t the stall ceiling binds and the ratio must stay under it.
     *
     * A cost model over those three terms suggested r=1.80, and measurement
     * refused it: at -t 64 that is ABOVE the 1.63 ceiling, and the run sat right on
     * the cliff -- mid-run stall 0.109 s on one rep and 0.614 s on the next, with a
     * 0.54 s PROCESS spread. The model under-predicts stalls near the ceiling, so
     * the ratio stays at 1.6, under it.
     *
     * Measured in one binary, three shapes selected by the env knobs, 3 reps
     * interleaved, warm cache, warmup discarded, EVERY arm writing SAM to a real
     * file (head / mid / overhead from --profile; ramp = slice count):
     *
     *    -t   shape                 ramp    head     mid    ovhd    PROCESS
     *    16   fractional, r=2.0        7   0.018   0.000   0.211   86.44
     *    16   fractional, r=1.6        6   0.051   0.000   0.175   86.51
     *    16   absolute 16 Mb, r=1.6    5   0.081   0.000   0.140   86.50
     *    32   fractional, r=2.0        7   0.034   0.000   0.299   44.45
     *    32   fractional, r=1.6        6   0.099   0.000   0.249   44.47
     *    32   absolute 16 Mb, r=1.6    6   0.087   0.000   0.249   44.49
     *    64   fractional, r=2.0        7   0.066   0.416   0.413   23.99
     *    64   fractional, r=1.6        6   0.199   0.000   0.345   23.62
     *    64   absolute 16 Mb, r=1.6    7   0.095   0.018   0.413   23.61
     *
     * The ratio is what buys wall time: at -t 64 it takes the mid-run stall from
     * 0.416 s to zero, -1.5% end to end. It costs 0.06-0.07 s (0.07%) at -t 16/32,
     * where no stall was possible to begin with.
     *
     * The absolute first slice is a WASH on this input at every rung -- -0.005 /
     * -0.012 / -0.017 s in the ramp terms, and inside the run-to-run spread end to
     * end. It is here for what the table shows about how the fractional form
     * SCALES, not for a speedup. Sizing the first slice as task_size/ratio^depth
     * makes it grow with -t, because task_size is chunk_size * n_threads, and the
     * measured head fill tracks that exactly: 0.051 -> 0.099 -> 0.199 s across
     * -t 16 -> 32 -> 64, doubling with every rung. The absolute form is flat at
     * 0.081 -> 0.087 -> 0.095 s. Extending the same arithmetic, -t 128 would open
     * a 1280 Mbase cohort with a 76 Mbase slice and ~0.37 s of wholly unhidden
     * fill. Fill and overhead are both absolute costs, so nothing about the right
     * first slice scales with the cohort; the fractional form only looked free
     * because it was never measured above -t 64.
     *
     * The honest cost of pinning it: at -t 64 the absolute slice needs one extra
     * ramp slice (7 vs 6), and that +0.068 s of kt_for overhead eats most of the
     * 0.104 s of head fill it saves. That trade gets better with -t, since fill
     * would keep doubling while the per-slice cost does not.
     *
     * Deriving the ratio at run time from the ramp's own slices was implemented
     * and rejected. Beyond a sign error, it is not robust: the estimate must come
     * from the small early slices, which under-report compute throughput because
     * fixed kt_for costs dominate, and its outcome at -t 64 flips sign on whether
     * step 1 is one or two slices behind step 0 -- a scheduling artifact, not
     * something a policy can pin. A fixed pair is deterministic and measured
     * better. */
    double       cohort_ramp_ratio         = cohort_ramp_ratio_default;
    int64_t      cohort_ramp_first         = cohort_ramp_first_default;

    mem_opt_t    *opt, opt0;
    gzFile        fp = 0, fp2 = 0;
    void         *ko = 0, *ko2 = 0;
    int           fd, fd2;
    mem_pestat_t  pes[4];
    ktp_aux_t     aux;
    bool          is_o       = 0;
    const char   *out_path   = NULL;   /* -o/-f path; opened lazily so --bam can claim it */
    bool          out_opened = false;  /* true iff aux.fp is a real fopen()'d FILE* we own */
    uint8_t      *ref_string;

    memset(&aux, 0, sizeof(ktp_aux_t));
    memset(pes, 0, 4 * sizeof(mem_pestat_t));
    for (i = 0; i < 4; ++i) pes[i].failed = 1;

    // opterr = 0;
    aux.fp = stdout;
    aux.opt = opt = mem_opt_init();
    memset(&opt0, 0, sizeof(mem_opt_t));

    /* Parse input arguments */
    // comment: added option '5' in the list
    //
    // Long-only options for bisulfite mode (bwa-mem3 meth fork):
    //   --meth              Enable inline bwameth-style c2t + post-processing. Output
    //                       container is --bam's job, same as without --meth.
    //                       Expects a reference built with `bwa-mem3 index --meth`.
    //   --set-as-failed f|r Flag alignments to this strand as QC-fail (0x200)
    //   --chimera-qc        Enable the bwameth.py-style longest-M <44% chimera heuristic
    //                       (off by default; not part of Bismark)
    enum {
        OPT_BAM = 1000,
        OPT_METH,
        OPT_METH_SCORING,
        OPT_METH_SEED_PRUNE,
        OPT_METH_TAGS,
        OPT_METH_SET_AS_FAILED,
        OPT_METH_CHIMERA_QC,
        OPT_SUPP_REP_HARD_CAP,
        OPT_PROPER_PAIR_FROM_EMITTED,
        OPT_LEGACY_READER,
        OPT_MIN_EXT_LEN,
        OPT_MAX_EXTEND_CHAINS,
        OPT_SEED_ORDER,
        OPT_SMEM_DEDUP,
        OPT_FAST,
        OPT_HUGE_PAGES,
        OPT_SKIP_CONTAINED_EXT,
        OPT_ADAPTIVE_BAND,
        OPT_NO_ADAPTIVE_BAND,
        OPT_NO_BAND_CERT,
        OPT_EXTEND_MATE_CONCORDANT,
        OPT_COMPAT,
        OPT_CHUNK_CAP,
        OPT_COHORT_SLICES,
        OPT_RESCUE_KMER,
        OPT_RESCUE_BAND,
        OPT_RESCUE_SKIP,
        OPT_COHORT_RAMP_RATIO,
        OPT_COHORT_RAMP_FIRST,
        OPT_HIC,
        OPT_EXTEND_TIE_FRAC,
        OPT_EXTEND_TIE_FLOOR,
        OPT_EXTEND_CSUB,
#ifdef STAGE_PROF
        OPT_PROFILE,
#endif
        OPT_DEDUP,
        OPT_DEDUP_READS,
        OPT_HELP,
    };
    static struct option long_opts[] = {
        {"bam",                      optional_argument, 0, OPT_BAM},
        {"min-ext-len",              required_argument, 0, OPT_MIN_EXT_LEN},
        {"max-extend-chains",        required_argument, 0, OPT_MAX_EXTEND_CHAINS},
        {"smem-dedup",               no_argument,       0, OPT_SMEM_DEDUP},
        {"dedup",                    required_argument, 0, OPT_DEDUP},
        {"dedup-reads",              required_argument, 0, OPT_DEDUP_READS},
        {"fast",                     no_argument,       0, OPT_FAST},
        {"huge-pages",               no_argument,       0, OPT_HUGE_PAGES},
        {"skip-contained-ext",       no_argument,       0, OPT_SKIP_CONTAINED_EXT},
        {"adaptive-band",            no_argument,       0, OPT_ADAPTIVE_BAND},
        {"no-adaptive-band",         no_argument,       0, OPT_NO_ADAPTIVE_BAND},
        {"no-band-cert",             no_argument,       0, OPT_NO_BAND_CERT},
        {"extend-mate-concordant",   optional_argument, 0, OPT_EXTEND_MATE_CONCORDANT},
        {"chunk-cap",                required_argument, 0, OPT_CHUNK_CAP},
        {"cohort-slices",            required_argument, 0, OPT_COHORT_SLICES},
        {"rescue-kmer",              optional_argument, 0, OPT_RESCUE_KMER},
        {"rescue-band",              required_argument, 0, OPT_RESCUE_BAND},
        {"rescue-skip",              no_argument,       0, OPT_RESCUE_SKIP},
        {"cohort-ramp-ratio",        required_argument, 0, OPT_COHORT_RAMP_RATIO},
        {"cohort-ramp-first",        required_argument, 0, OPT_COHORT_RAMP_FIRST},
        {"extend-tie-frac",          required_argument, 0, OPT_EXTEND_TIE_FRAC},
        {"extend-tie-floor",         required_argument, 0, OPT_EXTEND_TIE_FLOOR},
        {"extend-csub",              no_argument,       0, OPT_EXTEND_CSUB},
        {"meth",                     optional_argument, 0, OPT_METH},
        {"meth-scoring",             required_argument, 0, OPT_METH_SCORING},
        {"meth-seed-prune",          optional_argument, 0, OPT_METH_SEED_PRUNE},
        {"meth-tags",                required_argument, 0, OPT_METH_TAGS},
        {"set-as-failed",            required_argument, 0, OPT_METH_SET_AS_FAILED},
        {"chimera-qc",               no_argument,       0, OPT_METH_CHIMERA_QC},
        {"supp-rep-hard-cap",        required_argument, 0, OPT_SUPP_REP_HARD_CAP},
        {"proper-pair-from-emitted", no_argument,       0, OPT_PROPER_PAIR_FROM_EMITTED},
        {"seed-order",               required_argument, 0, OPT_SEED_ORDER},
        {"compat",                   required_argument, 0, OPT_COMPAT},
        {"legacy-reader",            no_argument,       0, OPT_LEGACY_READER},
        {"hic",                      no_argument,       0, OPT_HIC},
#ifdef STAGE_PROF
        {"profile",                  required_argument, 0, OPT_PROFILE},
#endif
        {"help",                     no_argument,       0, OPT_HELP},
        {0, 0, 0, 0}
    };
#ifdef STAGE_PROF
    const char *profile_path = NULL;   /* --profile <path>: stage_prof TSV output */
#endif
    const char *dedup_mode_arg = NULL; /* --dedup <off|on|auto>; resolved via mem_dedup_configure after getopt */
    const char *dedup_reads_mode_arg = NULL; /* --dedup-reads <off|on|auto>; resolved via mem_dedup_reads_configure after getopt */
    while ((c = getopt_long(argc, argv, "51qpaMCSPVYjuk:c:v:s:r:t:R:A:B:O:E:U:w:L:d:T:Q:D:m:I:N:W:x:G:h:y:K:X:H:o:f:z:",
                            long_opts, NULL)) >= 0)
    {
        if (c == 'k') opt->min_seed_len = atoi(optarg), opt0.min_seed_len = 1;
        else if (c == OPT_MIN_EXT_LEN) opt->min_ext_len = atoi(optarg), opt0.min_ext_len = 1;
        else if (c == OPT_MAX_EXTEND_CHAINS) opt->max_extend_chains = atoi(optarg), opt0.max_extend_chains = 1;
        else if (c == OPT_EXTEND_TIE_FRAC) {
            /* Validated like the other numeric options rather than via a bare
             * atof, which maps unparseable text to 0 (= off, silently disabling
             * the gate) and accepts "nan"/"inf". The single !(d >= 0 && d <= 1)
             * test rejects NaN (which compares false to everything), +/-inf, and
             * any value outside [0,1] at once -- no <math.h> needed. */
            double d = 0.0;
            if (parse_full_double(optarg, &d) != 0 || !(d >= 0.0 && d <= 1.0)) {
                fprintf(stderr, "ERROR: --extend-tie-frac requires a number in "
                                "0..1 (0 = off), got '%s'\n", optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->extend_tie_frac = (float)d; opt0.extend_tie_frac = 1;
        }
        else if (c == OPT_EXTEND_CSUB) opt->extend_csub = 1;
        else if (c == OPT_EXTEND_TIE_FLOOR) {
            /* Validated like --extend-tie-frac above: a bare atoi accepted
             * unparseable text and negatives, both of which the consumer then
             * treats as "no floor" -- so a typo silently ran the un-floored gate. */
            int64_t v = 0;
            if (parse_bounded_i64(optarg, 0, INT_MAX, &v) != 0) {
                fprintf(stderr, "ERROR: --extend-tie-floor requires an integer in "
                                "0..%d (0 = no floor), got '%s'\n", INT_MAX, optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->extend_tie_floor = (int)v; opt0.extend_tie_floor = 1;
        }
        else if (c == '1') no_mt_io = 1;
        else if (c == 'x') mode = optarg;
        else if (c == 'w') opt->w = atoi(optarg), opt0.w = 1;
        else if (c == 'A') opt->a = atoi(optarg), opt0.a = 1, assert(opt->a >= INT_MIN && opt->a <= INT_MAX);
        else if (c == 'B') opt->b = atoi(optarg), opt0.b = 1, assert(opt->b >= INT_MIN && opt->b <= INT_MAX);
        else if (c == 'T') opt->T = atoi(optarg), opt0.T = 1, assert(opt->T >= INT_MIN && opt->T <= INT_MAX);
        else if (c == 'U')
            opt->pen_unpaired = atoi(optarg), opt0.pen_unpaired = 1, assert(opt->pen_unpaired >= INT_MIN && opt->pen_unpaired <= INT_MAX);
        else if (c == 't') {
            /* atoi() silently maps unparseable input to 0 and ignores
             * trailing garbage (e.g. "-t nope" -> 0, "-t 300oops" -> 300), so
             * a typo would read as a valid-looking thread count with no
             * diagnostic. Reject anything that is not a complete integer; a
             * parseable value that is <= 0 still floors to 1 as before (not
             * an error -- "at least one thread" is the existing contract),
             * and a value above MAX_THREADS is still clamped, with its own
             * warning, after getopt below. */
            int64_t v;
            if (parse_bounded_i64(optarg, INT_MIN, INT_MAX, &v) != 0) {
                fprintf(stderr, "ERROR: -t requires an integer thread count, got '%s'\n", optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->n_threads = (int)(v > 1 ? v : 1);
        }
        else if (c == 'o' || c == 'f')
        {
            /* Capture the path; defer opening until after --bam is parsed so
             * BAM mode can hand the path to htslib instead of truncating it
             * here as a SAM-text FILE*. */
            is_o = 1;
            out_path = optarg;
        }
        else if (c == 'P') opt->flag |= MEM_F_NOPAIRING;
        else if (c == 'a') opt->flag |= MEM_F_ALL;
        else if (c == 'p') opt->flag |= MEM_F_PE | MEM_F_SMARTPE;
        else if (c == 'M') opt->flag |= MEM_F_NO_MULTI;
        else if (c == 'S') opt->flag |= MEM_F_NO_RESCUE;
        else if (c == 'Y') opt->flag |= MEM_F_SOFTCLIP;
        else if (c == 'V') opt->flag |= MEM_F_REF_HDR;
        else if (c == '5') opt->flag |= MEM_F_PRIMARY5 | MEM_F_KEEP_SUPP_MAPQ; // always apply MEM_F_KEEP_SUPP_MAPQ with -5
        /* --hic: exactly -5SP, no more. Set here rather than deferred like
         * --fast because there is no opt0 interaction to resolve -- these are
         * plain flag bits, so ORing them at parse time composes with -5/-S/-P
         * in any order, and `--hic -P` stays idempotent.
         *
         * -S is the letter that matters and the one a reader is least likely
         * to infer: mate rescue runs BEFORE the pairing bail-out in mem_sam_pe
         * (bwamem_pair.cpp -- the MEM_F_NO_RESCUE block precedes the
         * MEM_F_NOPAIRING goto), so -P alone skips pairing while leaving the
         * full rescue SW in place. Note minibwa gates rescue inside pairing
         * instead, so its --hic is -5P; the flag letters differ but the
         * intended behavior is the same, which is the point of the alias. */
        else if (c == OPT_HIC) opt->flag |= MEM_F_PRIMARY5 | MEM_F_KEEP_SUPP_MAPQ | MEM_F_NO_RESCUE | MEM_F_NOPAIRING;
        else if (c == 'q') opt->flag |= MEM_F_KEEP_SUPP_MAPQ;
        else if (c == 'u') opt->flag |= MEM_F_XB;
        else if (c == 'c') opt->max_occ = atoi(optarg), opt0.max_occ = 1;
        else if (c == 'd') opt->zdrop = atoi(optarg), opt0.zdrop = 1;
        else if (c == 'v') bwa_verbose = atoi(optarg);
        else if (c == 'j') ignore_alt = 1;
        else if (c == 'r')
            opt->split_factor = atof(optarg), opt0.split_factor = 1.;
        else if (c == 'D') opt->drop_ratio = atof(optarg), opt0.drop_ratio = 1.;
        else if (c == 'm') opt->max_matesw = atoi(optarg), opt0.max_matesw = 1;
        else if (c == 's') opt->split_width = atoi(optarg), opt0.split_width = 1;
        else if (c == 'G')
            opt->max_chain_gap = atoi(optarg), opt0.max_chain_gap = 1;
        else if (c == 'N')
            opt->max_chain_extend = atoi(optarg), opt0.max_chain_extend = 1;
        else if (c == 'W')
            opt->min_chain_weight = atoi(optarg), opt0.min_chain_weight = 1;
        else if (c == 'y')
            opt->max_mem_intv = atol(optarg), opt0.max_mem_intv = 1;
        else if (c == 'C') aux.copy_comment = 1;
        else if (c == 'K') fixed_chunk_size = atoi(optarg);
        else if (c == 'X') opt->mask_level = atof(optarg);
        else if (c == 'h')
        {
            opt0.max_XA_hits = opt0.max_XA_hits_alt = 1;
            opt->max_XA_hits = opt->max_XA_hits_alt = strtol(optarg, &p, 10);
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                opt->max_XA_hits_alt = strtol(p+1, &p, 10);
        }
        else if (c == 'z') opt->XA_drop_ratio = atof(optarg);
        else if (c == 'Q')
        {
            opt0.mapQ_coef_len = 1;
            opt->mapQ_coef_len = atoi(optarg);
            opt->mapQ_coef_fac = opt->mapQ_coef_len > 0? log(opt->mapQ_coef_len) : 0;
        }
        else if (c == 'O')
        {
            opt0.o_del = opt0.o_ins = 1;
            opt->o_del = opt->o_ins = strtol(optarg, &p, 10);
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                opt->o_ins = strtol(p+1, &p, 10);
        }
        else if (c == 'E')
        {
            opt0.e_del = opt0.e_ins = 1;
            opt->e_del = opt->e_ins = strtol(optarg, &p, 10);
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                opt->e_ins = strtol(p+1, &p, 10);
        }
        else if (c == 'L')
        {
            opt0.pen_clip5 = opt0.pen_clip3 = 1;
            opt->pen_clip5 = opt->pen_clip3 = strtol(optarg, &p, 10);
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                opt->pen_clip3 = strtol(p+1, &p, 10);
        }
        else if (c == 'R')
        {
            if ((rg_line = bwa_set_rg(optarg)) == 0) {
                free(opt);
                if (out_opened)
                    fclose(aux.fp);
                return 1;
            }
        }
        else if (c == 'H')
        {
            if (optarg[0] != '@')
            {
                FILE *fp;
                if ((fp = fopen(optarg, "r")) != 0)
                {
                    hdr_line = bwa_insert_header_file(fp, hdr_line);
                    fclose(fp);
                }
            } else hdr_line = bwa_insert_header(optarg, hdr_line);
        }
        else if (c == OPT_BAM) {
            opt->bam_mode = 1;
            opt->bam_level = 0;
            if (optarg != NULL && optarg[0] != '\0') {
                int lvl = atoi(optarg);
                if (lvl < 0 || lvl > 9) {
                    fprintf(stderr, "ERROR: --bam level must be 0..9 (got '%s')\n", optarg);
                    free(opt);
                    if (out_opened) fclose(aux.fp);
                    return 1;
                }
                opt->bam_level = lvl;
                /* The compressed-BAM warning is emitted once after parsing,
                 * from the resolved bam_level -- see below. Warning here would
                 * fire per occurrence and would warn for a level a later
                 * --bam=0 goes on to override. */
            }
        }
#ifdef STAGE_PROF
        else if (c == OPT_PROFILE) {
            profile_path = optarg;
        }
#endif
        else if (c == OPT_METH) {
            opt->meth_mode = 1;
            /* --meth selects bisulfite ALIGNMENT semantics only; the output
             * container is --bam's job here exactly as it is without --meth.
             * (Through 0.7.x this set bam_mode=1, which made text SAM
             * unreachable under --meth — there was no flag that undid it.) */
            /* Optional chemistry argument. getopt_long's optional_argument only
             * accepts the `--meth=taps` form (a separate word is treated as a
             * positional), so a bare `--meth` leaves optarg NULL => em-seq. */
            if (optarg != NULL) {
                if (strcmp(optarg, "emseq") == 0 || strcmp(optarg, "em-seq") == 0
                        || strcmp(optarg, "bisulfite") == 0) {
                    opt->meth_chem = METH_CHEM_EMSEQ;
                } else if (strcmp(optarg, "taps") == 0) {
                    opt->meth_chem = METH_CHEM_TAPS;
                } else {
                    fprintf(stderr, "ERROR: --meth accepts 'emseq' (default) or 'taps', got '%s'\n"
                                    "       note: use --meth=taps, not --meth taps\n", optarg);
                    free(opt);
                    if (out_opened) fclose(aux.fp);
                    return 1;
                }
            }
        }
        else if (c == OPT_METH_SCORING) {
            if (optarg != NULL && strcmp(optarg, "collapsed") == 0) {
                opt->meth_scoring = MEM_METH_SCORING_COLLAPSED;
                opt0.meth_scoring = 1;
            } else if (optarg != NULL && strcmp(optarg, "genomic") == 0) {
                opt->meth_scoring = MEM_METH_SCORING_GENOMIC;
                opt0.meth_scoring = 1;
            } else if (optarg != NULL && strcmp(optarg, "neutral") == 0) {
                opt->meth_scoring = MEM_METH_SCORING_NEUTRAL;
                opt0.meth_scoring = 1;
            } else {
                fprintf(stderr, "ERROR: --meth-scoring requires 'collapsed', 'genomic', or 'neutral'\n");
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
        }
        else if (c == OPT_METH_SEED_PRUNE) {
            /* --meth-seed-prune[=spec30|baseline|off]: prune the 3-letter
             * over-seeding before SA resolution (~30% faster --meth at ~0 truth
             * accuracy cost; NOT byte-identical). ON by default under --meth
             * (SPEC30, set in the meth-default block below); pass =off to restore
             * byte-identical seeding. getopt_long's optional_argument only binds
             * with '=' (a bare word is a positional), so a plain --meth-seed-prune
             * leaves optarg NULL => spec30 (the recommended rule). */
            if (optarg == NULL || strcmp(optarg, "spec30") == 0) {
                opt->meth_seed_prune = MEM_METH_PRUNE_SPEC30;
            } else if (strcmp(optarg, "baseline") == 0) {
                opt->meth_seed_prune = MEM_METH_PRUNE_BASELINE;
            } else if (strcmp(optarg, "off") == 0) {
                opt->meth_seed_prune = MEM_METH_PRUNE_OFF;
            } else {
                /* optarg is non-NULL here (NULL took the spec30 branch), so the
                 * value was bound with '=' -- the orphaned-space case cannot
                 * reach this branch, and a "use =, not a space" note would
                 * misdirect. Just name the accepted vocabulary. */
                fprintf(stderr, "ERROR: --meth-seed-prune accepts 'spec30' (default), 'baseline', or 'off'\n");
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt0.meth_seed_prune = 1;   /* explicit: wins over the --meth spec30 default below */
        }
        else if (c == OPT_METH_TAGS) {
            const char *tag_err = NULL;
            if (mem_opt_parse_meth_tags(optarg, &opt->meth_tags, &tag_err) != 0) {
                fprintf(stderr, "ERROR: --meth-tags '%s': %s\n"
                                "       expected 'all', 'none', a comma-separated list "
                                "(e.g. XR,XG), or ^-prefixed exclusions (e.g. ^XM)\n",
                        optarg != NULL ? optarg : "", tag_err);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt0.meth_tags = 1;
        }
        else if (c == OPT_METH_SET_AS_FAILED) {
            if (optarg == NULL || !(optarg[0] == 'f' || optarg[0] == 'r') || optarg[1] != '\0') {
                fprintf(stderr, "ERROR: --set-as-failed requires 'f' or 'r'\n");
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->meth_set_as_failed = optarg[0];
        }
        else if (c == OPT_METH_CHIMERA_QC) {
            opt->meth_chimera_qc = 1;
        }
        else if (c == OPT_SUPP_REP_HARD_CAP) {
            char *end = NULL;
            errno = 0;
            long v = strtol(optarg, &end, 10);
            if (end == optarg || end == NULL || *end != '\0' ||
                errno == ERANGE || v < 0 || v > INT_MAX) {
                fprintf(stderr, "ERROR: --supp-rep-hard-cap requires a non-negative integer\n");
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->supp_rep_hard_cap = (int)v;
        }
        else if (c == OPT_LEGACY_READER) aux.legacy_reader = 1;
        else if (c == OPT_SEED_ORDER) {
            opt->seed_emit_order = seed_order_from_str(optarg);
            if ((int)opt->seed_emit_order < 0) {
                fprintf(stderr, "[E::%s] unknown --seed-order '%s' (off|local-longest; "
                        "see docs for advanced modes)\n", __func__, optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
        }
        else if (c == OPT_COMPAT) {
            const compat_target_t *t = compat_target_from_name(optarg);
            if (t == NULL) {
                fprintf(stderr, "[E::%s] unknown --compat target '%s' (%s)\n",
                        __func__, optarg, compat_target_selectable_list());
                /* --compat takes a required argument, so a bare `--compat`
                 * silently swallows the next token -- usually the index
                 * prefix. Recognize that shape and say so, rather than leaving
                 * the user staring at their own reference path being called a
                 * compat target. */
                if (access(optarg, R_OK) == 0)
                    fprintf(stderr, "[E::%s] ('%s' is an existing file -- "
                            "--compat requires a target, e.g. --compat=bwa-mem2)\n",
                            __func__, optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            /* A recognized-but-unavailable target gets its own diagnostic:
             * "unknown target" would be a lie, and a user asking for bwa-mem
             * deserves the real reason rather than a shrug. The reason travels
             * with the table row, not with this parser. */
            if (t->unavailable_reason != NULL) {
                fprintf(stderr,
                        "[E::%s] compat target '%s' is recognized but not yet selectable: "
                        "%s. Supported: %s\n",
                        __func__, t->name, t->unavailable_reason,
                        compat_target_selectable_list());
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->compat = t;
        }
        else if (c == OPT_SMEM_DEDUP) opt->smem_dedup = 1;
        else if (c == OPT_DEDUP) {
            /* Reject an explicit-but-empty CLI value (`--dedup=` / `--dedup ''`):
             * mem_dedup_configure() treats an empty mode_arg as "no CLI value" and
             * falls back to BWAMEM3_DEDUP, so without this guard a malformed flag
             * would silently inherit the env instead of being fatal. */
            if (!*optarg) {
                fprintf(stderr, "ERROR: --dedup: expected off|on|auto, got empty value\n");
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            dedup_mode_arg = optarg;
        }
        else if (c == OPT_DEDUP_READS) {
            /* Reject an explicit-but-empty CLI value, same as --dedup: an empty
             * mode_arg reads as "no CLI value" in mem_dedup_reads_configure() and
             * would silently inherit BWAMEM3_DEDUP_READS instead of being fatal. */
            if (!*optarg) {
                fprintf(stderr, "ERROR: --dedup-reads: expected off|on|auto, got empty value\n");
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            dedup_reads_mode_arg = optarg;
        }
        else if (c == OPT_FAST) fast = 1;
        else if (c == OPT_HUGE_PAGES) want_huge_pages = 1;
        else if (c == OPT_PROPER_PAIR_FROM_EMITTED) opt->proper_pair_from_emitted = 1;
        else if (c == OPT_RESCUE_KMER) {
            /* Validated rather than atoi'd for the same reason as --chunk-cap
             * below: atoi maps every unparseable value to 0, and 0 here means
             * "off" -- so `--rescue-kmer=x` would silently disable the option it
             * was meant to configure, and would keep it disabled under --fast
             * (which defers to any explicitly-set value). K above the uint32 code
             * width is rejected rather than clamped so `--rescue-kmer=20` cannot
             * masquerade as a distinct setting from K=16. */
            int64_t k = MEM_RESCUE_KMER_DEFAULT;   /* bare --rescue-kmer */
            if (optarg && parse_bounded_i64(optarg, 0, MEM_RESCUE_KMER_MAX, &k) != 0) {
                fprintf(stderr, "ERROR: --rescue-kmer requires an integer in "
                                "0..%d (0 = off), got '%s'\n",
                        MEM_RESCUE_KMER_MAX, optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->rescue_kmer = (int)k;
            opt0.rescue_kmer = 1;
        }
        else if (c == OPT_RESCUE_BAND) {
            /* Validated like --rescue-kmer above. A bare atoi accepted both
             * unparseable text and negatives, and the kernel's `band > 0 ? band
             * : 50` guard then turned either into the default -- so a typo read
             * as a deliberate band width but silently ran the default one. */
            int64_t band = 0;
            if (parse_bounded_i64(optarg, 1, MEM_RESCUE_BAND_MAX, &band) != 0) {
                fprintf(stderr, "ERROR: --rescue-band requires an integer in "
                                "1..%d bp, got '%s'\n",
                        MEM_RESCUE_BAND_MAX, optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            opt->rescue_band = (int)band;
        }
        else if (c == OPT_RESCUE_SKIP) {
            /* A plain switch: there is no `=0` form because there is no preset to
             * opt out of -- --fast deliberately does not enable this. Validated
             * AFTER getopt (below), not here: it needs a non-zero rescue_kmer,
             * but --rescue-kmer may not be parsed yet and --fast resolves it
             * later still, so an inline check would make the diagnostic depend
             * on flag order. */
            opt->rescue_skip = 1;
            opt0.rescue_skip = 1;
        }
        else if (c == OPT_CHUNK_CAP) {
            /* Validated the same way as --supp-rep-hard-cap below rather than via
             * a bare atoll, which maps every unparseable value to 0 -- and 0 here
             * means "no cap". A typo would therefore silently change the batch
             * size, which is the exact class of invisible batching change this
             * option exists to make explicit. */
            if (parse_bounded_i64(optarg, 0, INT64_MAX, &chunk_cap) != 0) {
                fprintf(stderr, "ERROR: --chunk-cap requires a non-negative "
                                "integer number of bases (0 = off), got '%s'\n", optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
            chunk_cap_set = 1;
        }
        else if (c == OPT_COHORT_SLICES) {
            /* Same validation as --chunk-cap above, for the same reason: a bare
             * atoll maps every unparseable value to 0, and 0 here means "no
             * slicing". `--cohort-slices 3x` would silently disable the feature
             * it was meant to configure. Capped at COHORT_SLICES_MAX because the
             * ramp shift is clamped there anyway, so a larger value is
             * indistinguishable from it and asking for it is a mistake worth
             * naming. */
            if (parse_bounded_i64(optarg, 0, COHORT_SLICES_MAX, &cohort_slices) != 0) {
                fprintf(stderr, "ERROR: --cohort-slices requires an integer in "
                                "0..%d (0 = off), got '%s'\n",
                        (int)COHORT_SLICES_MAX, optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
        }
        else if (c == OPT_COHORT_RAMP_RATIO) {
            /* Validated like the two above. atof() maps anything unparseable to
             * 0.0, which the `ratio <= 1` guard below then reports as a ratio the
             * user never asked for and silently replaces with the default -- a
             * confusing way to learn you made a typo. Only PARSEABILITY is
             * enforced here; the `<= 1` range check stays where it is, because it
             * is shared with the env override and is a documented fallback rather
             * than an error. */
            if (parse_full_double(optarg, &cohort_ramp_ratio) != 0) {
                fprintf(stderr, "ERROR: --cohort-ramp-ratio requires a number, "
                                "got '%s'\n", optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
        }
        else if (c == OPT_COHORT_RAMP_FIRST) {
            /* Validated for the same reason, and this one is the sharpest of the
             * four: atoll("16M") is 16, which is > 0 and therefore accepted as a
             * SIXTEEN-BYTE first slice rather than 16 Mbases. That is a silent
             * ~million-fold error in the option this PR exists to introduce.
             * --chunk-cap already rejects '100M' for exactly this reason. */
            if (parse_bounded_i64(optarg, 0, INT64_MAX, &cohort_ramp_first) != 0) {
                fprintf(stderr, "ERROR: --cohort-ramp-first requires a "
                                "non-negative integer number of bases "
                                "(0 = fraction of the batch), got '%s'\n", optarg);
                free(opt);
                if (out_opened) fclose(aux.fp);
                return 1;
            }
        }
        else if (c == OPT_SKIP_CONTAINED_EXT) opt->skip_contained_ext = 1;
        else if (c == OPT_ADAPTIVE_BAND) { if (!no_adaptive_band) opt->band_start = ADAPTIVE_BAND_START; opt->band_cert = 0; }
        else if (c == OPT_NO_ADAPTIVE_BAND) { no_adaptive_band = 1; opt->band_start = 0; opt->band_cert = 0; }  /* disable adaptive
                                                               * banding entirely: aggressive band off
                                                               * AND certified band off -> exact
                                                               * full-width extension, by construction. */
        else if (c == OPT_NO_BAND_CERT) opt->band_cert = 0;   /* opt out of the certified adaptive
                                                               * band -> full-width exact ladder;
                                                               * output stays byte-identical, only
                                                               * slower. Escape hatch + test A/B. */
        else if (c == OPT_EXTEND_MATE_CONCORDANT) {
            /* bare flag = auto (-1, use the estimated insert-size high bound);
             * =INT = fixed window in bp; =0 = off. */
            opt->mate_concordant_window = optarg ? atoi(optarg) : -1;
            opt0.mate_concordant_window = 1;
        }
        else if (c == OPT_HELP) {
            usage(opt);
            free(opt);
            free(hdr_line);
            free(rg_line);
            if (out_opened) fclose(aux.fp);
            return 0;
        }
        else if (c == 'I')
        {
            aux.pes0 = pes;
            pes[1].failed = 0;
            pes[1].avg = strtod(optarg, &p);
            pes[1].std = pes[1].avg * .1;
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                pes[1].std = strtod(p+1, &p);
            pes[1].high = (int)(pes[1].avg + 4. * pes[1].std + .499);
            pes[1].low  = (int)(pes[1].avg - 4. * pes[1].std + .499);
            if (pes[1].low < 1) pes[1].low = 1;
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                pes[1].high = (int)(strtod(p+1, &p) + .499);
            if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                pes[1].low  = (int)(strtod(p+1, &p) + .499);
        }
        else {
            free(opt);
            if (out_opened)
                fclose(aux.fp);
            return 1;
        }
    }

    /* Check output file name */
    if (rg_line)
    {
        hdr_line = bwa_insert_header(rg_line, hdr_line);
        free(rg_line);
    }

    if (opt->n_threads < 1) opt->n_threads = 1;
    if (opt->n_threads > MAX_THREADS) {
        /* Per-thread profiling arrays (tprof[][tid], sized to LIM_C >= MAX_THREADS
         * -- see the static_assert in macro.h) are indexed by tid up to
         * n_threads-1, so n_threads must not exceed MAX_THREADS or the write
         * corrupts adjacent globals. Clamp rather than abort, matching how the
         * pipeline already floors n_threads to 1. */
        fprintf(stderr, "[W::%s] -t %d exceeds the %d-thread maximum; clamping to %d.\n",
                __func__, opt->n_threads, MAX_THREADS, MAX_THREADS);
        opt->n_threads = MAX_THREADS;
    }
    mem_dedup_configure(dedup_mode_arg);   /* --dedup CLI > env BWAMEM3_DEDUP > default 'auto'; fatal on bad value */
    mem_dedup_reads_configure(dedup_reads_mode_arg); /* --dedup-reads CLI > env BWAMEM3_DEDUP_READS > default 'auto'; fatal on bad value */
    /* A stray word on the command line slides silently into a positional slot.
     * Two spellings invite it:
     *   --meth taps        (optional argument: getopt_long only binds it with '=')
     *   --meth-tags XR XG  (required argument: 'XR' binds, 'XG' does not)
     * In both cases the orphan lands in <idxbase>, and with a single-end read
     * file the result still has three positionals -- a perfectly well-formed
     * paired-end invocation whose reference happens to be "taps" or "XG". The
     * user then gets a missing-index error naming a token they never meant as a
     * path. Diagnose it here, before any index work, whenever a positional
     * matches a known option-value vocabulary and is not an actual path. */
    for (int pos = optind; pos < argc; ++pos) {
        const char *flag = stray_option_value_flag(argv[pos]);
        if (flag == NULL || path_exists(argv[pos])) continue;
        fprintf(stderr,
                "ERROR: '%s' was taken as a positional argument (%s), but it looks like\n"
                "       the value for %s and no such file exists.\n"
                "%s\n",
                argv[pos], pos == optind ? "the <idxbase> reference" : "a read file",
                flag, stray_option_value_advice(argv[pos]));
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }
    if (optind + 2 != argc && optind + 3 != argc) {
        usage(opt);
        free(opt);
        if (out_opened)
            fclose(aux.fp);
        return 1;
    }

    /* In-process BGZF deflate runs on the single writer thread, so for large
     * outputs it -- not alignment -- is usually what caps throughput. Warn
     * once, here rather than in the parsing loop, so the message describes the
     * *resolved* setting: `--bam=6 --bam=0` must stay silent (the last --bam
     * wins and it is uncompressed) and `--bam=6 --bam=6` must warn once, not
     * per occurrence. Placed after the positional-argument check so a usage
     * error is not preceded by a warning about output it never writes. */
    if (opt->bam_mode && opt->bam_level > 0)
        fprintf(stderr,
            "WARNING: --bam=%d writes compressed BAM on a single writer "
            "thread; BGZF deflate is not parallelized here, so for large "
            "outputs this serial compression is usually the bottleneck. "
            "Prefer uncompressed output (--bam, i.e. --bam=0) piped to a "
            "threaded compressor, e.g. "
            "`bwa-mem3 mem --bam ... | samtools view -@ N -b -o out.bam`.\n",
            opt->bam_level);

    /* Further input parsing */
    if (mode)
    {
        fprintf(stderr, "WARNING: bwa-mem3 doesn't work well with long reads or contigs; please use minimap2 instead.\n");
        if (strcmp(mode, "intractg") == 0)
        {
            if (!opt0.o_del) opt->o_del = 16;
            if (!opt0.o_ins) opt->o_ins = 16;
            if (!opt0.b) opt->b = 9;
            if (!opt0.pen_clip5) opt->pen_clip5 = 5;
            if (!opt0.pen_clip3) opt->pen_clip3 = 5;
        }
        else if (strcmp(mode, "pacbio") == 0 || strcmp(mode, "pbref") == 0 || strcmp(mode, "ont2d") == 0)
        {
            if (!opt0.o_del) opt->o_del = 1;
            if (!opt0.e_del) opt->e_del = 1;
            if (!opt0.o_ins) opt->o_ins = 1;
            if (!opt0.e_ins) opt->e_ins = 1;
            if (!opt0.b) opt->b = 1;
            if (opt0.split_factor == 0.) opt->split_factor = 10.;
            if (strcmp(mode, "ont2d") == 0)
            {
                if (!opt0.min_chain_weight) opt->min_chain_weight = 20;
                if (!opt0.min_seed_len) opt->min_seed_len = 14;
                if (!opt0.pen_clip5) opt->pen_clip5 = 0;
                if (!opt0.pen_clip3) opt->pen_clip3 = 0;
            }
            else
            {
                if (!opt0.min_chain_weight) opt->min_chain_weight = 40;
                if (!opt0.min_seed_len) opt->min_seed_len = 17;
                if (!opt0.pen_clip5) opt->pen_clip5 = 0;
                if (!opt0.pen_clip3) opt->pen_clip3 = 0;
            }
        }
        else
        {
            fprintf(stderr, "[E::%s] unknown read type '%s'\n", __func__, mode);
            free(opt);
            if (out_opened)
                fclose(aux.fp);
            return 1;
        }
    } else update_a(opt, &opt0);

    /* --fast: one-flag shorthand for the characterized speed levers
     *   -m 10  -y 0  --min-ext-len 30  --smem-dedup  --skip-contained-ext
     *   --max-extend-chains 20  --adaptive-band  --extend-mate-concordant
     *   --extend-tie-frac 0.95  --extend-tie-floor 1  --extend-csub
     *   (under --meth: --max-extend-chains 10 and also adds -s 2),
     *   plus the strict-total-order + pdqsort dedup sort (alnreg_sort_fast),
     *   which has no flag of its own -- see the alnreg_sort_fast assignment
     *   below and the comparator commentary in src/bwamem.cpp.
     * Mirrors the -x preset: each lever is applied only when the user did not
     * set it explicitly (opt0), so explicit flags win where applicable. The
     * exceptions are --smem-dedup, --skip-contained-ext and the dedup sort,
     * which are plain on/off booleans forced on unconditionally (no opt-out
     * flag exists; the dedup sort has no flag at all).
     * --skip-contained-ext is byte-identical on non-meth SE/PE and no-ops under
     * --meth via its own internal gate (see bwamem.cpp), so forcing it on here is
     * safe for --fast --meth too.
     * Output is NOT byte-identical to the default; divergence is confined to the
     * low-confidence tail (see docs/best-practices/settings-profiles.md).
     * meth_mode is already resolved here (parsed in the getopt loop above). */
    /* The two guards and the warning below all key on "a target other than
     * `off` is selected" -- --compat=off is exactly equivalent to not passing
     * the flag, so it must trip none of them. */
    const int compat_on = (opt->compat != &COMPAT_TARGET_OFF);
    /* --fast and --compat are mutually exclusive. --compat suppresses only
     * additive output (MQ:i/HN:i, @HD, sidecar @SQ tags) to reproduce another
     * aligner byte-for-byte, but --fast deliberately CHANGES alignments;
     * combining them would yield a diff-clean-looking stream over genuinely
     * different alignments, defeating the parity-validation purpose of
     * --compat. Reject up front. */
    if (fast && compat_on) {
        fprintf(stderr, "[E::%s] --compat and --fast are mutually exclusive: "
                "--compat targets byte-identical %s output, but --fast changes alignments\n",
                __func__, opt->compat->name);
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }
    /* --proper-pair-from-emitted deliberately derives FLAG 0x2 differently from
     * both upstreams (fg-labs/bwa-mem3#17, #362), so pairing it with a --compat
     * target asks for byte-identity and for a documented deviation from it in
     * the same command. Same shape as --fast above: refuse rather than emit a
     * stream that diffs clean everywhere except the ALT records. */
    if (opt->proper_pair_from_emitted && compat_on) {
        fprintf(stderr, "[E::%s] --compat and --proper-pair-from-emitted are mutually exclusive: "
                "--compat targets byte-identical %s output, but --proper-pair-from-emitted "
                "derives FLAG 0x2 from the emitted alignment, which %s does not\n",
                __func__, opt->compat->name, opt->compat->name);
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }
    /* --compat is an output-parity target, but no target has a bisulfite mode,
     * so "byte-identical" is undefined under --meth, which also emits
     * meth-specific tags that no target models. Reject the combination rather
     * than silently half-suppress. */
    if (opt->meth_mode && compat_on) {
        fprintf(stderr, "[E::%s] --compat is not supported with --meth: "
                "--compat reproduces %s output, which has no methylation mode\n",
                __func__, opt->compat->name);
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }
    /* The score-gated chain-extension cap (--extend-tie-frac / --extend-tie-floor
     * / --extend-csub) prunes chains before banded-SW and reseeds the pruned
     * competitor into MAPQ, so a setting that can actually prune a chain changes
     * alignments and MAPQ exactly as --fast does. It is folded into --fast
     * (already rejected above), but the levers are also settable on their own,
     * so a command such as `--compat=bwa-mem2 --extend-tie-frac=0.95 --extend-csub`
     * would otherwise emit a diff-clean-looking stream over genuinely
     * non-compatible alignments.
     *
     * Reject only configurations that can actually prune (mem_chain_cap_extend /
     * mem_chain_cap_extend_mate in bwamem.cpp):
     *  - extend_tie_frac > 0 gates on its own, independent of --max-extend-chains
     *    (cap_on), so any nonzero value is unsafe by itself.
     *  - extend_tie_floor only ever changes a keep decision when the fraction
     *    gate is active (with tie_frac <= 0 the gate is 0, so `w[i] >= gate` is
     *    always true and the floor comparison never matters) -- a bare floor is
     *    always a no-op and must not trip the guard.
     *  - extend_csub only has an effect once a chain was actually dropped
     *    (capped_w != 0). With extend_tie_frac == 0 that still happens whenever
     *    --max-extend-chains is explicitly set (the plain count cap populates
     *    capped_w on its own), so csub is unsafe there too, even though the
     *    count cap itself is a pre-existing, separately-scoped lever.
     *
     * Compared by value, not opt0: an explicit `--extend-tie-frac 0` resolves to
     * the byte-identical off-state and must not trip the guard. */
    if (compat_on &&
        (opt->extend_tie_frac != 0.0f ||
         (opt->extend_csub != 0 && opt->max_extend_chains != 0))) {
        fprintf(stderr, "[E::%s] --compat and the chain-extension cap "
                "(--extend-tie-frac, or --extend-csub combined with "
                "--max-extend-chains) are mutually exclusive: --compat targets "
                "byte-identical %s output, but these levers prune chains and "
                "recalibrate MAPQ, which changes alignments\n",
                __func__, opt->compat->name);
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }
    /* --compat with an @HD in -H: WARN, do not reject. Emitted only after every
     * rejection above (--fast, --proper-pair-from-emitted, --meth), so a run
     * that is about to be refused does not also collect a warning about how its
     * header would have been ordered. A new --compat guard belongs above this
     * comment, not below it.
     *
     * bwa-mem3 hoists a LEADING user @HD above the @SQ block, so the header is
     * spec-valid (@HD must come first). Neither target does that: bwa 0.7.19
     * emits -H records after @SQ and only warns (bwa.c:426-428), bwa-mem2 has
     * no @HD logic at all. So the header differs from the target in line ORDER.
     *
     * This is not rejected, unlike --fast and --meth, because it is an explicit
     * and coherent request: "give me a valid SAM header, everything else the
     * same". --fast silently moves alignments and --meth is a different mode --
     * the user cannot see those in their own command line. An @HD they typed
     * themselves, they can. Records are unaffected either way.
     *
     * Only a LEADING @HD is hoisted, and only that case diverges; a later @HD
     * is emitted inline after @SQ exactly as upstream does. Warn precisely, so
     * the warning means something when it fires. */
    if (compat_on && hdr_line != NULL &&
        strncmp(hdr_line, "@HD\t", 4) == 0 && bwa_verbose >= 2) {
        fprintf(stderr, "[W::%s] --compat=%s with an @HD from -H: bwa-mem3 emits it before "
                "@SQ (the SAM spec requires @HD first), but %s emits -H records after @SQ, "
                "so the header will differ from %s in line order. Records are unaffected. "
                "Continue anyway.\n",
                __func__, opt->compat->name, opt->compat->name, opt->compat->name);
    }
    if (fast) {
        if (!opt0.max_matesw)   opt->max_matesw   = 10;  /* -m 10 */
        if (!opt0.max_mem_intv) opt->max_mem_intv = 0;   /* -y 0  */
        if (!opt0.min_ext_len)  opt->min_ext_len  = 30;  /* --min-ext-len 30 */
        /* Band mate rescue to a 6-mer anchor diagonal (--rescue-kmer=6). Opt-in,
         * not byte-identical, but truth-based ROC (holodeck, hg38 + bisulfite
         * chr20) shows accuracy neutral-to-positive and confident (MAPQ>=60)
         * mismaps unchanged; k=6 is the measured wall-time peak (~-22% on the
         * rescue-heavy WGS tail). */
        if (!opt0.rescue_kmer)  opt->rescue_kmer  = MEM_RESCUE_KMER_DEFAULT;
        /* --rescue-skip is deliberately NOT part of --fast. It drops mate
         * rescues rather than shortening them, and on real reads that costs
         * confident alignments: measured against --rescue-kmer alone, 16 losses
         * at MAPQ>=30 per 200k primaries on 125 bp WGBS and 420 per 2M on 75 bp
         * em-seq, because the fixed vote floor is read-length-blind and skips
         * 42%/79% of scans at those lengths. --fast is meant to be a
         * characterized speed preset, not a recall trade. */
        /* --max-extend-chains: 5 for non-meth; 10 under --meth. A 7-point ablation
         * ({0,5,10,20,50,100,1000}) on 1M sim-meth PE pairs (with mate-concordant
         * rescue on, below) shows chr-accuracy flat (0.9908) at every cap but the
         * confident wrong-chromosome rate is U-shaped, minimized at 10 (cap 5: 592
         * MAPQ>=30 mismaps; cap 10: 382; uncapped: 1056), for +0.7s wall (20.2->20.9s,
         * still -6% vs uncapped). Non-meth uses 20 + --extend-mate-concordant
         * (below): the non-meth sweep in #202 shows the plain cap 5 inflates the
         * confident (MAPQ>=1) mis-placement tail 3.8x on sim-wgs-place (3,626 ->
         * 13,921 vs uncapped); cap 20 + mate-concordant lands within +827 of that
         * floor (verified --fast) while keeping ~-20% aligner CPU. */
        if (!opt0.max_extend_chains) opt->max_extend_chains = opt->meth_mode ? 10 : 20;
        opt->smem_dedup = 1;                             /* --smem-dedup (plain on/off) */
        opt->alnreg_sort_fast = 1;                       /* strict-total-order + pdqsort dedup sort
                                                          * (~35-55% faster for n>=9; diverges from
                                                          * bwa-mem2 on equal-`re` ties) */
        opt->skip_contained_ext = 1;                     /* --skip-contained-ext (plain on/off;
                                                          * meth-gated internally) */
        if (!no_adaptive_band)
            opt->band_start = ADAPTIVE_BAND_START;        /* --adaptive-band: no-op on short reads
                                                          * (8-bit tier untouched), ~25% faster on
                                                          * long-read (SBX/HiFi/ONT) runs.
                                                          * --no-adaptive-band opts back out, keeping
                                                          * band_start=0 (exact extension) here. */
        opt->band_cert  = 0;                             /* --fast opts out of the certified (byte-identical)
                                                          * band: either the aggressive band_start heuristic
                                                          * above is in force, or --no-adaptive-band opted
                                                          * back out to exact full-width extension. */
        /* --extend-mate-concordant (meth only): the top-5 chain cap regresses
         * bisulfite PE placement. Mechanism (instrumented on 50k sim-meth-place
         * pairs vs truth): NOT chain-dropping -- in 89% of regressions the read's
         * true alignment is still a candidate, but the collapsed 3-letter alphabet
         * flattens chain weights so the read carries many chains, and capping to 5
         * starves PE pairing/mate-rescue of the secondary anchors that let the true
         * concordant pair win; both mates then flip together to a wrong concordant
         * locus (99% proper-pair). Keeping any capped chain that is concordant with
         * a mate chain retains exactly the true pair's low-weight chain while still
         * dropping the far/redundant ones, recovering placement to default parity
         * (97.64% -> 98.08%, == cap-off 98.09%). Non-meth --fast benefits too:
         * the same top-5 cap inflates confident (MAPQ>=1) mismaps 3.8x on
         * sim-wgs-place (3,626 -> 13,921 vs uncapped), for the same reason -- the
         * true chain is low-weight but mate-concordant in 98% of cap-dropped reads.
         * Mate-concordant retention recovers ~60% of that (-> 5,521) at ~neutral
         * CPU, so it is enabled for non-meth --fast as well (fg-labs/bwa-mem3#202).
         * Auto (-1) sizes the concordance window to the estimated proper-pair
         * insert bound so only genuine pair anchors are retained. */
        if (!opt0.mate_concordant_window)
            opt->mate_concordant_window = -1;
        if (opt->meth_mode && !opt0.split_width)
            opt->split_width = 2;                        /* -s 2 (meth only): light Pass-2 reseed.
                                                          * -s 0 (no reseed) inflates MAPQ on bisulfite
                                                          * reads (interior-repeat competitors go unfound);
                                                          * -s 2 reseeds the occurrence-1 SMEMs that inflate,
                                                          * recovering MAPQ+placement at ~the same speed. */
        /* Score-gated chain-extension cap (fg-labs/bwa-mem3#269), applied on top
         * of --max-extend-chains above (incl. meth's cap of 10): extend a capped
         * chain only if its weight is >= 0.95x the best chain's weight
         * (--extend-tie-frac 0.95), always keeping the top-1 chain
         * (--extend-tie-floor 1), and seed the primary's competitor score with the
         * best dropped chain so the pruning does not inflate MAPQ (--extend-csub).
         * Truth-graded placement-neutral on wgs/exome/panel/meth and on a substrate
         * matched to real low-AS panel reads (net-favorable there: reduces confident
         * and total mismapping); -11 to -17% wall on the bulk workloads. The two
         * --extend-tie-* levers honor an explicit user value (opt0); --extend-csub
         * has no opt-out flag and is forced on like --smem-dedup, since it only
         * matters when the cap/gate prunes, which --fast now always does. */
        if (!opt0.extend_tie_frac)  opt->extend_tie_frac  = 0.95;
        if (!opt0.extend_tie_floor) opt->extend_tie_floor = 1;
        opt->extend_csub = 1;
    }

    /* Meth-mode default tuning. bwameth.py runs bwa as
     * `bwa mem -T 40 -B 2 -L 10 -CM`, adding `-U 100 -p` for paired-end. We adopt
     * the soft-clip (-L 10), unpaired (-U 100), output-threshold (-T 40), -M and
     * -C defaults for ALL THREE --meth-scoring modes; the ONLY mode-dependent knob
     * is the mismatch penalty (the leniency gate):
     *   COLLAPSED (bwameth drop-in): -B 2 — bwameth's lenient mismatch. Combined
     *     with the two-cell matrix (C/T and G/A free both ways) this reproduces
     *     bwameth's collapsed-space placement.
     *   GENOMIC (variant-aware): keep bwa's default -B 4 — the full-hg38 variant
     *     A/B with the asymmetric matrix showed b=4 places better and is better
     *     MAPQ-calibrated than b=2 (placement 92.6 vs 92.5, discordant MAPQ
     *     1.8 vs 2.1).
     *   NEUTRAL (variant-aware, the --meth=taps default): also keeps -B 4. It
     *     differs from GENOMIC only in the freed cell's VALUE (0 vs +a), not in
     *     the leniency gate, so the same b=4 argument applies unchanged.
     * pen_unpaired is only consulted for paired-end rescue, so setting it
     * unconditionally is a no-op for single-end. -A/-B always override and reach
     * the matrices (mem_opt_fill_meth_mat below).
     * NB: those constants are quoted at bwameth's match score (a == 1) and are
     * scaled by opt->a inside mem_opt_apply_meth_defaults — see bwamem.h. Before
     * that, they were applied flat and silently discarded -A. */
    if (opt->meth_mode) {
        /* TAPS defaults to NEUTRAL scoring. TAPS converts only the METHYLATED
         * cytosines, so conversions are ~20-30x rarer than under em-seq (measured:
         * 3.2% of C vs 94.6% on chr22 @ 12x). The collapsed 3-letter alphabet buys
         * little at that density and costs specificity. NEUTRAL goes one step
         * further: it scores the conversion cell as 0 rather than a full match,
         * so a sparse TAPS conversion is tolerated without over-crediting spurious
         * C->T alignments. Measured on 4.07M simulated TAPS reads vs full hg38,
         * across three methylation loads: NEUTRAL placed 95.95-96.01% (essentially
         * the 96.02% unconverted ceiling) vs GENOMIC 95.68-95.73% and COLLAPSED
         * 95.37-95.43% -- a robust +0.24-0.28 pp over genomic -- with IDENTICAL
         * MAPQ calibration (60+ bin 99.99% both) and NM (1.966 vs 1.969), so the
         * gain is not bought with over-confidence or inflated edit distance.
         * NEUTRAL's freed cell is not rank-1, but the generalized kswv freed-cell
         * blend scores it to its matrix value, so mate rescue stays on the batched
         * kernel on the freed-capable tiers (NEON/AVX2/AVX512BW) and falls back to
         * scalar ksw_align2 only on the freed-less x86 tiers (sse41/sse42/avx),
         * exactly as GENOMIC and COLLAPSED do. An explicit --meth-scoring still wins.
         * Set before mem_opt_apply_meth_defaults so its COLLAPSED -B 2 branch keys
         * off the resolved scoring mode. */
        if (opt->meth_chem == METH_CHEM_TAPS && !opt0.meth_scoring)
            opt->meth_scoring = MEM_METH_SCORING_NEUTRAL;
        /* --meth-seed-prune defaults to spec30 under --meth. Unlike non-meth, --meth
         * is a bwa-mem3-native feature -- bwa/bwa-mem2 never had a bisulfite mode, so
         * there is no upstream reference output to stay byte-identical to. spec30 is
         * ~30% faster on em-seq at ~0 truth-based accuracy cost (mapq>=20 accuracy
         * identical; 8x fewer newly-unmapped than the `baseline` rule). An explicit
         * --meth-seed-prune=off|baseline (or the BWAMEM3_METH_SEED_PRUNE env) still
         * wins; no effect outside --meth (the prune is gated on the meth remap). */
        if (!opt0.meth_seed_prune)
            opt->meth_seed_prune = MEM_METH_PRUNE_SPEC30;
        /* Scored defaults live in mem_opt_apply_meth_defaults so they scale with
         * -A (bwameth's constants assume a==1) and can be unit-tested. */
        mem_opt_apply_meth_defaults(opt, &opt0);
        aux.copy_comment = 1;          /* -C, needed for YS:Z/YC:Z passthrough */
        /* The certified adaptive band is a non-meth optimization: --meth extension
         * scores against the original 4-letter reference through per-strand
         * asymmetric matrices, and its reads are short (no band win to reclaim), so
         * keep the exact full-width ladder here rather than reason about the
         * certificate under the meth matrices. (Also avoids the safety-envelope
         * downgrade note firing on every --meth run.) */
        opt->band_cert = 0;
    }

    /* Under --meth, NM/MD are derived from the scoring matrix (a column is a
     * mismatch iff the matrix penalizes it), so a non-positive -B makes every
     * substitution cell non-negative and silently collapses NM to 0 and MD to
     * a bare match run -- hiding real variants, not just conversions. Refuse it
     * rather than emit output that looks clean because scoring is degenerate.
     * The bound is `<= 0`, not `== 0`: bwa_fill_scmat stores -b, so a NEGATIVE
     * -B turns every substitution into a positive reward, which hides real
     * variants at least as thoroughly as -B 0 does.
     * The message reports the EFFECTIVE penalty rather than echoing "-B":
     * update_a() scales b by -A when -B was not given, so `-A 0` reaches b == 0
     * without the user ever passing -B, and naming -B there would misdirect.
     * The non-meth path keeps the literal comparison and is unaffected. */
    if (opt->meth_mode && opt->b <= 0) {
        fprintf(stderr, "ERROR: --meth requires a positive mismatch penalty, but the "
                        "effective penalty is %d; a non-positive penalty makes every "
                        "substitution free or rewarded, which collapses NM/MD to zero "
                        "and hides real variants (check -B and -A)\n", opt->b);
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }

    /* Matrix for SWA */
    bwa_fill_scmat(opt->a, opt->b, opt->mat);
    /* D3 (--meth): re-derive the per-hypothesis asymmetric matrices from the matrix
     * we just rebuilt, so -A/-B and the -x presets reach meth scoring (they set
     * opt->a/opt->b above; without this the meth matrices keep init-time defaults). */
    mem_opt_fill_meth_mat(opt);

    /* Certified adaptive band: apply the narrow probe only inside the parameter
     * envelope where the extension kernel's early-termination heuristics are
     * provably quiescent (see mem_band_cert_params_safe). Outside it -- small
     * -d/zdrop, large -L clip penalties, a custom -A/-B matrix scoring above the
     * match reward -- fall back to the exact full-width ladder so output stays
     * byte-identical for any parameters. Default parameters are inside the
     * envelope, so this never fires on a plain run. Checked after the matrices are
     * final because the envelope depends on max(mat). */
    if (opt->band_cert && !mem_band_cert_params_safe(opt)) {
        opt->band_cert = 0;
        fprintf(stderr, "[M::%s] extension parameters (-d/-L/-A/-B/-O/-E) are outside the "
                        "certified-band safety envelope; using exact full-width extension\n",
                __func__);
    }

    /* In --meth (D3) the canonical UX is "bwa-mem3 mem --meth ref.fa": we
     * auto-append ".meth" to find the converted SEED FM-index built by
     * "bwa-mem3 index --meth" (the original-alphabet index lives at the bare
     * prefix and supplies BNS+PAC via FMI_search::set_meth_ref_prefix below).
     * If the user already passed the ".meth" path directly, use it as-is. */
    char c2t_ref[PATH_MAX];
    char orig_ref_buf[PATH_MAX];
    /* In --meth, the ORIGINAL (un-projected) reference prefix: supplies BNS+PAC
     * for extension/coords (dual-index, set_meth_ref_prefix) and the
     * .hdr/.dict sidecar for @SQ M5/UR enrichment. NULL outside --meth. */
    const char *meth_orig_ref_prefix = NULL;
    const char *ref_prefix = argv[optind];
    if (opt->meth_mode) {
        const char *suffix = ".meth";
        size_t slen = strlen(suffix);
        size_t alen = strlen(argv[optind]);
        int already_c2t = (alen >= slen) &&
                          (strcmp(argv[optind] + alen - slen, suffix) == 0);
        if (!already_c2t) {
            int n = snprintf(c2t_ref, sizeof(c2t_ref), "%s%s", argv[optind], suffix);
            if (n <= 0 || (size_t)n >= sizeof(c2t_ref)) {
                fprintf(stderr, "ERROR: ref path too long for --meth\n");
                exit(EXIT_FAILURE);
            }
            ref_prefix = c2t_ref;
            meth_orig_ref_prefix = argv[optind];
        } else {
            /* User passed the ".meth" seed-index path directly; recover the
             * original reference prefix by stripping the suffix so its
             * sidecar (not the seed index's) supplies @SQ identity tags. */
            size_t base = alen - slen;
            if (base < sizeof(orig_ref_buf)) {
                memcpy(orig_ref_buf, argv[optind], base);
                orig_ref_buf[base] = '\0';
                meth_orig_ref_prefix = orig_ref_buf;
            } else {
                /* Degenerate: the prefix doesn't fit. The original sidecar is
                 * optional (the c2t index alone aligns), so warn and continue
                 * without @SQ M5/UR enrichment rather than aborting. */
                fprintf(stderr,
                        "[bwa-mem3:--meth] WARNING: reference path too long to "
                        "derive the original prefix; @SQ M5/UR enrichment "
                        "skipped.\n");
            }
        }
    }

    /* D3: fail fast with an actionable message if the `.meth` seed index is
     * absent — distinguishing a never-built index from a stale D1 `.bwameth.c2t`
     * index (the format changed; the user must re-run `index --meth`). */
    if (opt->meth_mode) {
        char probe[PATH_MAX];
        snprintf(probe, sizeof(probe), "%s%s", ref_prefix, ".bwt.2bit.64");
        if (access(probe, F_OK) != 0) {
            const char *orig = (meth_orig_ref_prefix != NULL) ? meth_orig_ref_prefix
                                                              : argv[optind];
            char old_probe[PATH_MAX];
            snprintf(old_probe, sizeof(old_probe), "%s.bwameth.c2t.bwt.2bit.64", orig);
            if (access(old_probe, F_OK) == 0)
                fprintf(stderr,
                        "ERROR: --meth found a legacy '.bwameth.c2t' index, but the "
                        "format changed. Re-run: bwa-mem3 index --meth %s\n", orig);
            else
                fprintf(stderr,
                        "ERROR: --meth seed index '%s.*' not found. Run: "
                        "bwa-mem3 index --meth %s\n", ref_prefix, orig);
            free(opt);
            if (out_opened) fclose(aux.fp);
            return 1;
        }
    }

    if (opt->seed_emit_order != SEED_ORDER_OFF)
        fprintf(stderr, "[M::%s] seed order: %s\n", __func__, seed_order_to_str(opt->seed_emit_order));
    if (fast) {
        /* `alnreg-sort=fast` is deliberately spelled WITHOUT a leading `--`: unlike
         * every other item on this line it is not a flag the user can pass, only a
         * lever --fast turns on. That is also why it must be here: with no flag to
         * grep for, this line is the only record a run leaves of a lever that
         * changes output (see the comparator commentary in src/bwamem.cpp). It is
         * not meth-gated, so it appears on both branches. */
        /* --rescue-kmer reports its RESOLVED K, and reports `=0` when the user opted
         * back out, because 0 is the one --fast lever whose off-state is invisible
         * anywhere else in the run -- it changes MAPQ on rescued reads, so which way
         * it resolved has to be on the record. It applies under --meth too (the
         * anchor scan collapses to 3 letters there), so it is on both branches. */
        /* Report the RESOLVED adaptive-band state: --no-adaptive-band opts back out
         * (band_start=0), and like --rescue-kmer=0 that off-state is otherwise
         * invisible in the run record, so spell it out on the audit line. */
        const char *adaptive_band_label = opt->band_start ? "--adaptive-band" : "--no-adaptive-band";
        /* --extend-tie-frac prints with %g, not %.2f: a value like 0.004 would
         * truncate to 0.00 and misreport the resolved gate. %g (not %.9g) because
         * the field is a float -- %.9g would surface the double-promotion noise
         * (0.95 -> 0.949999988); %g's 6 significant digits round-trip every
         * realistically-typed fraction cleanly. */
        if (opt->meth_mode)
            /* --skip-contained-ext is set but no-ops under --meth (internal gate), so it is
             * intentionally omitted from the meth audit line to reflect the effective levers.
             * --adaptive-band applies under --meth, so it stays (unless opted out). */
            fprintf(stderr, "[M::%s] --fast: -m %d -y %ld --min-ext-len %d --smem-dedup --max-extend-chains %d %s -s %d --extend-mate-concordant --extend-tie-frac %g --extend-tie-floor %d%s --rescue-kmer=%d%s alnreg-sort=fast\n",
                    __func__, opt->max_matesw, (long)opt->max_mem_intv, opt->min_ext_len, opt->max_extend_chains, adaptive_band_label, opt->split_width, opt->extend_tie_frac, opt->extend_tie_floor, opt->extend_csub ? " --extend-csub" : "", opt->rescue_kmer, opt->rescue_skip ? " --rescue-skip" : "");
        else
            fprintf(stderr, "[M::%s] --fast: -m %d -y %ld --min-ext-len %d --smem-dedup --skip-contained-ext --max-extend-chains %d %s --extend-mate-concordant --extend-tie-frac %g --extend-tie-floor %d%s --rescue-kmer=%d%s alnreg-sort=fast\n",
                    __func__, opt->max_matesw, (long)opt->max_mem_intv, opt->min_ext_len, opt->max_extend_chains, adaptive_band_label, opt->extend_tie_frac, opt->extend_tie_floor, opt->extend_csub ? " --extend-csub" : "", opt->rescue_kmer, opt->rescue_skip ? " --rescue-skip" : "");
        /* --fast also caps the batch size, which keeps the read/compute/write
         * pipeline overlapped at high -t. It re-partitions the input and so is not
         * byte-identical -- which --fast already is not -- hence it rides here
         * rather than in the default path. An explicit --chunk-cap still wins. */
        if (!chunk_cap_set) chunk_cap = 256000000;
    }

    /* --rescue-skip keys entirely off the --rescue-kmer anchor scan, so without
     * that scan it has nothing to decide on and would silently do nothing. Reject
     * rather than no-op, matching how --rescue-kmer and --rescue-band already
     * reject out-of-range values instead of clamping: a flag that changes which
     * reads get rescued must not be quietly inert.
     *
     * Deliberately placed AFTER getopt and after the --fast block: rescue_kmer is
     * not final until --fast has resolved it, so checking inline in the getopt
     * case would make the diagnostic depend on the order the flags were typed.
     * --fast does not set rescue_skip, so `--fast --rescue-kmer=0` reaches here
     * with rescue_skip == 0 and passes. */
    if (opt->rescue_skip && !opt->rescue_kmer) {
        fprintf(stderr, "ERROR: --rescue-skip requires --rescue-kmer (the skip decision "
                        "reuses the k-mer anchor scan); got --rescue-kmer=0\n");
        free(opt);
        if (out_opened) fclose(aux.fp);
        return 1;
    }

    /* Load bwt2/FMI index */
    uint64_t tim = __rdtsc();

    fprintf(stderr, "* Ref file: %s\n", ref_prefix);
    /* Opt-in --huge-pages: reserve 1 GB pages for the index BEFORE it is
     * allocated below, so the FM-index / SA arrays land on them. Safe no-op when
     * the host has no reserved 1 GB pool. See bwa_hugepages.{h,cpp}, issue #402. */
    if (want_huge_pages) bwamem_reserve_huge_pages(ref_prefix);
    aux.fmi = new FMI_search(ref_prefix);
    /* D3 dual-index: the FM-index AND its BNS/PAC come from the `.meth` SEED prefix.
     * The seed BNS is required to decode seed positions into (seed contig, local pos,
     * strand): the seed reference has the f/r-doubled contig layout (r0,f0,r1,f1,...).
     * Seed contigs are remapped to ORIGINAL coordinates arithmetically
     * (orig_tid = seed_rid/2; hypothesis = seed_rid & 1; pos preserved). The ORIGINAL
     * reference's BNS/PAC are loaded separately as the remap/extension target in the
     * extension phase (NOT a replacement of the seed BNS). */
    /* D3 --meth: load the SEED index's FM + bns but NOT its pac. The seed pac is
     * never read in --meth (extension/scoring/mate-rescue use meth_orig_pac);
     * skipping it saves ~1.6 GB on hg38. Outside --meth, load the pac as before. */
    aux.fmi->load_index(/*load_pac=*/!opt->meth_mode, /*n_threads=*/opt->n_threads);
    aux.shm_base = aux.fmi->shm_attached_base();
    /* Report the LOADED sample rate, not the compile-time SA_COMPX default:
     * a `bwa-mem3 index -u INT`-built index (or one attached from shm) can
     * differ from it, and this is the first point after load_index() where
     * aux.fmi->sa_compx reflects the real value (disk path: tail-detected
     * in load_index; shm path: read from the packed FMI_SCALARS section). */
    #if SA_COMPRESSION
    fprintf(stderr, "* SA compression enabled with xfactor: %d\n", 1 << (int)aux.fmi->sa_compx);
    #endif
    tprof[FMI][0] += __rdtsc() - tim;

#if SMEM_LOCKSTEP_N > 1
    /* Resolve the phase-2 SMEM lockstep width once, before the seeding workers
     * spawn: how many reads' FM-index walks the driver keeps in flight. Defaults
     * to the compile-time SMEM_LOCKSTEP_N; BWA3_SMEM_LOCKSTEP_N pins an explicit
     * value, and BWA3_SMEM_LOCKSTEP_PROBE opts into a startup memory-level-
     * parallelism probe that chases the just-loaded cp_occ checkpoint array
     * (opaque here: base, block count, block stride, and the byte offset of a
     * 64-bit word per block). Width changes scheduling only, never output. */
    bwa3_init_smem_lockstep_width(
        aux.fmi->cp_occ_data(),
        aux.fmi->cp_occ_size_bytes() / (int64_t)sizeof(CP_OCC),
        sizeof(CP_OCC),
        offsetof(CP_OCC, one_hot_bwt_str));
    if (bwa_verbose >= 3)
        fprintf(stderr, "[M::%s] phase-2 SMEM lockstep width: %d\n",
                __func__, g_smem_lockstep_n);
#endif

    /* D3: load the ORIGINAL reference's bns/pac as resident handles for the
     * (future) extension/scoring phase — distinct from the seed FM-index above.
     * The seed BNS (aux.fmi->idx->bns) is the f/r-doubled converted reference
     * used to decode seed positions; these handles are the un-converted original
     * (real chrom names, N contigs) that extension will score against. No
     * consumer yet (extension is behind the meth-mode checkpoint below); this is
     * a load-only building block. Freed on every exit path the seed index is. */
    if (opt->meth_mode && meth_orig_ref_prefix != NULL) {
        if (meth_orig_ref_load_handles(meth_orig_ref_prefix,
                                       &aux.meth_orig_bns, &aux.meth_orig_pac) != 0) {
            delete aux.fmi;
            free(opt);
            if (out_opened) fclose(aux.fp);
            return 1;
        }
        fprintf(stderr,
                "[bwa-mem3:--meth] original reference bns/pac loaded for "
                "extension (%d contig(s)).\n", aux.meth_orig_bns->n_seqs);
    }

    /* D3 (--meth, PR-3): the early-return seeding checkpoint is REMOVED — the
     * pipeline now runs end to end. Seeds generated against the `.meth` seed
     * FM-index are remapped to ORIGINAL coordinates + an OT/OB hypothesis inside
     * mem_chain_seeds (see meth_seed_to_orig), and chaining/extension/pairing/
     * mate-rescue/output all operate against the ORIGINAL bns/pac carried on
     * worker_t::meth_orig_* (the original reference bases are pac-fetched from
     * `.pac` on demand; no unpacked `.0123` is loaded for either the seed or the
     * original — see below).
     * Extension and mate-rescue score the ORIGINAL read against the original
     * reference with the per-hypothesis asymmetric OT/OB matrix (PR-4 + A1: the
     * batched extension partitions a mixed PE batch by hypothesis; mate rescue
     * routes meth pairs through the scalar ksw_align2 path). The projected read is
     * used only for seeding against the `.meth` FM-index. */
    /* pac-fetch: reconstruct the ORIGINAL reference from its `.pac` on demand
     * (bns_get_seq_v2's ref_string==NULL path) instead of loading the unpacked
     * `.0123` (~6.4 GB on hg38). This is the only reference path: `index` no
     * longer builds `.0123` and `mem` never reads it (byte-identical to the
     * historical `.0123` load, verified on plain + --meth incl. batched rescue). */
    if (opt->meth_mode && meth_orig_ref_prefix != NULL) {
        fprintf(stderr,
                "[bwa-mem3:--meth] pac-fetch: original reference unpacked from "
                ".pac on demand (.0123 not loaded).\n");
    }

    // reference bases are pac-fetched from .pac on demand; no .0123 is loaded
    tim = __rdtsc();
    uint64_t timer;
    if (opt->meth_mode) {
        /* D3 --meth: the SEED `.0123` is dead weight (~13 GB on hg38). Every
         * downstream consumer reads the ORIGINAL unpacked reference via
         * mem_aln_ref_string()/mmc.ref_string (= meth_orig_ref_string, loaded
         * above); seeding uses the FM-index, not the unpacked seed `.0123`. So
         * skip loading it entirely and poison aux.ref_string to NULL — any
         * consumer that bypasses the mem_aln_* helpers will then crash loudly
         * rather than silently read seed bases at original coordinates. */
        ref_string             = NULL;
        aux.ref_string         = NULL;
        aux.ref_string_is_shm  = 0;
        timer = __rdtsc();
        fprintf(stderr, "* [--meth] seed reference `.0123` not loaded "
                "(extension uses the original reference)\n");
    } else {
        /* plain pac-fetch: unpack the original reference from `.pac` on demand
         * (bns_get_seq_v2's ref_string==NULL path). idx->pac is resident on both
         * the disk path (load_pac=true) and the shm-attach path (aliases the
         * staged PAC section). aux.ref_string==NULL routes every consumer —
         * extension + batched mate rescue — to pac-fetch (the rescue copies each
         * window into seqBufRef in-iteration, so the single-live-window contract
         * holds). −6.4 GB on hg38, byte-identical. */
        ref_string             = NULL;
        aux.ref_string         = NULL;
        aux.ref_string_is_shm  = 0;
        timer = __rdtsc();
        fprintf(stderr, "* pac-fetch: reference `.0123` not loaded; "
                "unpacking original bases from .pac on demand\n");
    }
    tprof[REF_IO][0] += timer - tim;

    if (ignore_alt)
        for (i = 0; i < aux.fmi->idx->bns->n_seqs; ++i)
            aux.fmi->idx->bns->anns[i].is_alt = 0;
    /* D3 (--meth, PR-3): --ignore-alt clears is_alt on the SEED bns above, but
     * --meth chaining/extension/output read is_alt from the ORIGINAL bns
     * (aux.meth_orig_bns) — mirror the clear there so --ignore-alt takes effect
     * in --meth. (Original genome indexes rarely carry ALT contigs, but keep the
     * two views consistent.) */
    if (ignore_alt && opt->meth_mode && aux.meth_orig_bns != NULL)
        for (i = 0; i < aux.meth_orig_bns->n_seqs; ++i)
            aux.meth_orig_bns->anns[i].is_alt = 0;

    /* READS file operations */
    ko = kopen(argv[optind + 1], &fd);
	if (ko == 0) {
		fprintf(stderr, "[E::%s] fail to open file `%s'.\n", __func__, argv[optind + 1]);
        free(opt);
        if (out_opened)
            fclose(aux.fp);
        delete aux.fmi;
        // kclose(ko);
        return 1;
    }
    // fp = gzopen(argv[optind + 1], "r");
    if (aux.legacy_reader) {
        fp = gzdopen(fd, "r");
        aux.ks = kseq_init(fp);
    } else {
        const char *fr_err = NULL;
        aux.fr1 = fast_reader_dopen(fd, &fr_err);
        if (aux.fr1 == NULL) {
            fprintf(stderr, "[E::%s] %s\n", __func__, fr_err ? fr_err : "failed to open input");
            free(opt);
            if (out_opened) fclose(aux.fp);
            delete aux.fmi;
            kclose(ko);
            return 1;
        }
        aux.frks = fast_kseq_init(aux.fr1);
    }

    // PAIRED_END
    /* Handling Paired-end reads */
    aux.ks2 = 0;
    if (optind + 2 < argc) {
        if (opt->flag & MEM_F_PE) {
            fprintf(stderr, "[W::%s] when '-p' is in use, the second query file is ignored.\n",
                    __func__);
        }
        else
        {
            ko2 = kopen(argv[optind + 2], &fd2);
            if (ko2 == 0) {
                fprintf(stderr, "[E::%s] failed to open file `%s'.\n", __func__, argv[optind + 2]);
                free(opt);
                // kclose(ko) below owns and frees the handle; a free(ko) here
                // would make that a use-after-free + double free.
                if (aux.legacy_reader) { err_gzclose(fp); kseq_destroy(aux.ks); }
                else { fast_kseq_destroy(aux.frks); fast_reader_close(aux.fr1); }
                if (out_opened)
                    fclose(aux.fp);
                delete aux.fmi;
                kclose(ko);
                // kclose(ko2);
                return 1;
            }
            if (aux.legacy_reader) {
                fp2 = gzdopen(fd2, "r");
                aux.ks2 = kseq_init(fp2);
                assert(aux.ks2 != 0);
            } else {
                const char *fr_err2 = NULL;
                aux.fr2 = fast_reader_dopen(fd2, &fr_err2);
                if (aux.fr2 == NULL) {
                    fprintf(stderr, "[E::%s] %s\n", __func__, fr_err2 ? fr_err2 : "failed to open input");
                    free(opt);
                    fast_kseq_destroy(aux.frks); fast_reader_close(aux.fr1);
                    if (out_opened) fclose(aux.fp);
                    delete aux.fmi; kclose(ko); kclose(ko2);
                    return 1;
                }
                aux.frks2 = fast_kseq_init(aux.fr2);
            }
            opt->flag |= MEM_F_PE;
        }
    }

    /* Look up optional per-index header records (<prefix>.hdr or
     * <baseprefix>.dict) once and route them into both output paths. The
     * SAM text path merges these with user -H per lh3/bwa#348 precedence;
     * the --bam path forwards them to htslib's sam_hdr_add_lines so the
     * rich @SQ (AS/M5/SP/AH/…) also makes it into the BAM header.
     *
     * Skipped entirely under a compat target: the sidecar is a bwa-mem3-only
     * feature (a port of lh3/bwa#348, which lh3 closed unmerged), so neither
     * bwa nor bwa-mem2 has anything to load. Its @SQ block adds M5/AS/UR/SP
     * that no upstream emits, and on the bench hg38 index its @HD as well.
     * Skipping restores the generated bare SN/LN @SQ block -- including the
     * AH:* on ALT contigs that both upstreams emit -- which is exactly the
     * target output. */
    char *idx_hdr_lines = opt->compat->read_sidecar
                        ? bwa_load_hdr_from_index(ref_prefix)
                        : NULL;
    /* A sidecar @SQ block that omits AH on ALT contigs silently strips ALT
     * status from the output. We do not rewrite it (it is authoritative --
     * see the function's comment), but we do say so. Not reached under a
     * compat target: idx_hdr_lines is NULL there and @SQ is regenerated from
     * bns, AH included. */
    if (idx_hdr_lines != NULL && !opt->meth_mode)
        bwa_warn_sidecar_missing_AH(aux.fmi->idx->bns, idx_hdr_lines, ref_prefix);
    /* --meth only: the original (pre-c2t) reference's .hdr/.dict sidecar, for
     * @SQ M5/UR enrichment and @CO/@PG/@RG pass-through in meth_bam_writer_open.
     * NULL outside --meth or when the original has no sidecar.
     *
     * Deliberately NOT gated on opt->compat->read_sidecar, unlike the load
     * above: --compat with --meth is rejected during option validation (see the
     * guard earlier in this function), so this load is unreachable under any
     * compat target. If that exclusion is ever relaxed, this needs the gate. */
    char *meth_orig_hdr_lines = (meth_orig_ref_prefix != NULL)
                                ? bwa_load_hdr_from_index(meth_orig_ref_prefix)
                                : NULL;

    /* Output path. --meth picks the WRITER (it owns the meth @SQ/@PG header and
     * the bam1_t overlay); --bam picks the CONTAINER within whichever writer.
     *  - --meth [--bam]: open meth_bam_writer, which serializes its bam1_t to
     *    BGZF under --bam and to SAM text without it. Honors -o/-f or stdout.
     *  - --bam (no --meth): open generic bam_writer; htslib writes its own
     *    @HD + @SQ + @PG header. Honors -o/-f or stdout.
     *  - SAM text (no --meth, no --bam): open -o/-f path (if any) as a FILE*;
     *    bwa_print_sam_hdr2. Note the meth writer claims -o itself in the first
     *    branch, so this fopen stays unreachable under --meth. */
    bam_writer_t *bam_writer = NULL;
#ifdef DISABLE_OUTPUT
    /* profile-build (-DDISABLE_OUTPUT) skips ALL filesystem-touching output
     * code so wall-clock measurements aren't gated on disk speed. The
     * per-record write sites below are also #ifndef-guarded; this block
     * additionally skips writer open + header emit so a non-writable -o
     * (or read-only fs) doesn't fail before compute begins. */
    aux.bam_writer = NULL;
    g_meth_bam_writer = NULL;
    if (opt->meth_mode) {
        /* D3 (PR-5): output is native original-alphabet. RNAME/POS/XM/XG all
         * derive from the ORIGINAL bns/pac (loaded into aux.meth_orig_bns/pac
         * by PR-1) and the per-alignment hypothesis — no f/r chrom map, no
         * un-converted ref view. The per-record path needs only the original
         * pac global (the original bns reaches it via mem_aln_bns()). With
         * output disabled there is no writer to open, but the global must still
         * be set so meth_mem_aln_to_bam builds correct XM/coords. Mirror the
         * non-DISABLE_OUTPUT guard below: a NULL original bns/pac (e.g. the
         * degenerate "prefix too long" path never loaded the handles) would
         * otherwise let chaining run against a NULL original reference and
         * silently corrupt coordinates. */
        if (aux.meth_orig_bns == NULL || aux.meth_orig_pac == NULL) {
            fprintf(stderr, "ERROR: meth: original reference (bns/pac) not loaded\n");
            free(opt);
            delete aux.fmi;
            return 1;
        }
        g_meth_orig_pac = aux.meth_orig_pac;
    }
    (void)is_o;
    (void)hdr_line;
    (void)idx_hdr_lines;
    (void)meth_orig_hdr_lines;
    (void)out_path;
#else
    if (opt->meth_mode) {
        /* D3 (PR-5): native original-alphabet output. The @SQ header is built
         * straight from the ORIGINAL (un-converted) bns (aux.meth_orig_bns),
         * and the per-record path consults the original pac (g_meth_orig_pac)
         * for XM:Z; alignments already carry original rids/coords (PR-3) and the
         * hypothesis (XG strand), so there is no f/r chrom map and no
         * un-converted ref fold. */
        if (aux.meth_orig_bns == NULL || aux.meth_orig_pac == NULL) {
            fprintf(stderr, "ERROR: meth: original reference (bns/pac) not loaded\n");
            free(opt);
            delete aux.fmi;
            return 1;
        }
        g_meth_orig_pac = aux.meth_orig_pac;
        const char *meth_out_path = is_o ? out_path : "-";
        extern char *bwa_pg;
        /* meth_orig_hdr_lines is the *original* reference's .hdr/.dict sidecar;
         * the writer enriches each @SQ with its M5/UR tags and forwards
         * @CO/@PG/@RG provenance. */
        g_meth_bam_writer = meth_bam_writer_open(meth_out_path, aux.meth_orig_bns,
                                                 bwa_pg, NULL,
                                                 hdr_line, meth_orig_hdr_lines,
                                                 opt->bam_mode, opt->bam_level);
        if (g_meth_bam_writer == NULL) {
            fprintf(stderr, "ERROR: meth: failed to open %s writer for '%s'\n",
                    opt->bam_mode ? "BAM" : "SAM", meth_out_path);
            g_meth_orig_pac = NULL;
            free(opt);
            delete aux.fmi;
            return 1;
        }
    } else if (opt->bam_mode) {
        const char *bam_path = is_o ? out_path : "-";
        extern char *bwa_pg;
        /* Suppress idx .hdr/.dict records entirely when the user's -H
         * supplies any @SQ, matching bwa_print_sam_hdr2's SAM precedence. */
        const char *bam_idx_hdr = bwa_hdr_text_has_type(hdr_line, "@SQ\t")
                                ? NULL : idx_hdr_lines;
        bam_writer = bam_writer_open(bam_path, aux.fmi->idx->bns,
                                     bam_idx_hdr, hdr_line,
                                     bwa_pg, opt->bam_level, opt->compat);
        if (bam_writer == NULL) {
            fprintf(stderr, "ERROR: failed to open BAM writer at '%s'\n", bam_path);
            free(opt);
            delete aux.fmi;
            return 1;
        }
        aux.bam_writer = bam_writer;
    } else {
        aux.bam_writer = NULL;
        if (is_o) {
            aux.fp = fopen(out_path, "w");
            if (aux.fp == NULL) {
                fprintf(stderr, "Error: can't open %s output file\n", out_path);
                free(opt);
                delete aux.fmi;
                return 1;
            }
            out_opened = true;
        }
        bwa_print_sam_hdr2(aux.fmi->idx->bns, idx_hdr_lines, hdr_line, aux.fp,
                           opt->compat);
    }
#endif

    if (fixed_chunk_size > 0) {
        /* -K: the user pinned the batch size (for reproducibility) — honor it
         * exactly, never cap. */
        aux.task_size = fixed_chunk_size;
    } else {
        /* Default batch size is chunk_size (~10M bases) per thread -- byte-for-byte
         * the same formula as bwa (fastmap.c: `opt->chunk_size * opt->n_threads`)
         * and bwa-mem2 v2.2.1 (fastmap.cpp: same), with NO cap.
         *
         * A cap is tempting: at very high -t a single chunk becomes enormous
         * (10M * 192 ~= 1.9G bases), so the input is only ~3-4 chunks and the
         * pipeline starves -- the first chunk's read and the last chunk's write
         * overlap nothing (fill/drain). Measured on c8g.16xlarge / a
         * 5M-read WGS slice, that costs ~1.6s of a 25.8s PROCESS() at -t 64.
         *
         * But capping RE-PARTITIONS THE INPUT, and the partition is not an
         * implementation detail: mem_pestat() infers the insert-size distribution
         * from whatever reads land in a batch (bwamem.cpp), and those percentile
         * bounds feed pairing, mate rescue and MAPQ. Different batches => different
         * pes => different output. Everything else in the aligner is batch-invariant
         * (read ids come from the global n_processed counter, so the hash_64 tie
         * breaks are stable), which makes mem_pestat the single reason a cap is
         * observable at all -- and it is enough.
         *
         * A default cap of 256M therefore silently broke byte-identity with
         * bwa/bwa-mem2 for every -t >= 26 (10M * 26 > 256M) -- ordinary production
         * settings -- while looking safe because the benchmark aligns at -t 16,
         * where the cap never engaged. So the cap is now OPT-IN:
         *
         *   default            no cap; identical batching to bwa/bwa-mem2 at any -t
         *   --chunk-cap N      cap at N bases (0 = off)
         *   --fast             implies --chunk-cap 256000000 (--fast already does
         *                      not promise byte-identical output)
         *   BWA_MEM3_CHUNK_CAP env override, for sweeps; wins over both
         *
         * If you want a specific batch size AND reproducibility, use -K: it pins
         * the size exactly and is never capped (see the branch above), which also
         * makes output independent of -t. */
        int64_t cap = chunk_cap;
        /* An unset or EMPTY value leaves the CLI/--fast cap alone -- atoll("") is
         * 0, which would otherwise read as "no cap" and silently switch an
         * explicit --chunk-cap off. A malformed value is reported rather than
         * silently treated as 0 for the same reason: this variable changes how
         * the input is partitioned, so a typo in it must not quietly change the
         * output. Reported, not fatal -- unlike the --chunk-cap flag, the index
         * is already loaded here, and an unusable sweep value should not throw
         * that work away when the documented default is still correct. */
        const char *cap_env = getenv("BWA_MEM3_CHUNK_CAP");
        if (cap_env && *cap_env) {
            if (parse_bounded_i64(cap_env, 0, INT64_MAX, &cap) != 0)
                fprintf(stderr, "WARNING: BWA_MEM3_CHUNK_CAP='%s' is not a "
                                "non-negative integer; ignoring it and using "
                                "chunk cap %lld\n", cap_env, (long long)cap);
        }
        int64_t scaled = (int64_t)opt->chunk_size * (int64_t)opt->n_threads;
        aux.task_size = (cap > 0 && scaled > cap) ? cap : scaled;
        if (cap > 0 && scaled > cap)
            fprintf(stderr, "[M::%s] chunk cap engaged: batch %lld -> %lld bases; "
                    "output is NOT byte-identical to bwa/bwa-mem2 at this -t\n",
                    __func__, (long long)scaled, (long long)cap);
    }
    tprof[MISC][1] = opt->chunk_size = aux.actual_chunk_size = aux.task_size;

    /* Cohort slicing. The first batch's read overlaps nothing -- the compute
     * pipeline has nothing to chew on until it lands -- and with the cap now
     * opt-in that first read is `chunk_size * n_threads` bases, the largest
     * single unhidden cost in the run. Reading the FIRST cohort as a geometric
     * ramp lets compute start on the first slice -- cohort_ramp_first bases,
     * 16 Mbase by default, independent of -t -- while the rest is still
     * arriving.
     *
     * This does not move the cohort boundary, so mem_pestat sees exactly the
     * read set it would have seen otherwise and the output is unchanged. See
     * the slice-target clamp in the read step for why the boundary is preserved
     * even though each read overshoots its request.
     *
     * Steady-state cohorts stay single-slice: once the pipeline is full,
     * read(N+1) already overlaps compute(N), so slicing them would only add
     * kt_for passes and shrink the SIMD batches for no overlap gain. */
    {
        int64_t slices = cohort_slices;
        /* Validated like BWA_MEM3_CHUNK_CAP, and non-fatal for the same reason
         * (the index is already loaded here): a bare atoll turns a typo into 0,
         * which means "no slicing" -- silently disabling the thing the variable
         * was set to configure. An empty value leaves the CLI/default alone. */
        const char *cs_env = getenv("BWA_MEM3_COHORT_SLICES");
        if (cs_env && *cs_env) {
            if (parse_bounded_i64(cs_env, 0, COHORT_SLICES_MAX, &slices) != 0)
                fprintf(stderr, "WARNING: BWA_MEM3_COHORT_SLICES='%s' is not an "
                                "integer in 0..%d; ignoring it and using %lld\n",
                        cs_env, (int)COHORT_SLICES_MAX, (long long)slices);
        }
        /* -p re-derives pairing from adjacency across the WHOLE array
         * (bseq_classify carries has_last state), so a slice boundary between
         * two mates would classify them as two SEs, change n_sep[], and shift
         * the read-id base handed to the second mem_process_seqs. Classification
         * must therefore precede alignment, which is incompatible with aligning
         * slices early. Fall back to un-sliced cohorts. */
        if (opt->flag & MEM_F_SMARTPE) {
            if (slices > 0)
                fprintf(stderr, "[M::%s] -p (smart pairing) is incompatible with "
                        "cohort slicing; reading each batch in one slice\n", __func__);
            slices = 0;
        }
        aux.cohort_slices = slices > 0 ? slices : 0;
        /* Stress knob for test/regression/cohort_slice_identity.sh: slice every
         * cohort, not just the first, so a small input exercises the accumulator
         * and the boundary clamp many times over. Deliberately env-only -- it
         * trades throughput for coverage and is not a mode users should pick. */
        const char *sa_env = getenv("BWA_MEM3_COHORT_SLICE_ALL");
        aux.cohort_slice_all = (sa_env && *sa_env && strcmp(sa_env, "0") != 0) ? 1 : 0;
        if (aux.cohort_slice_all && aux.cohort_slices > 0)
            fprintf(stderr, "[M::%s] BWA_MEM3_COHORT_SLICE_ALL set: slicing every "
                    "cohort (stress mode; slower)\n", __func__);

        /* Ramp growth ratio. Env override for sweeps, as with the slice count.
         * A ratio at or below 1 would shrink the ramp rather than grow it, so it
         * cannot reach the cohort at all; that is user error, not a mode, and the
         * default is restored rather than reading the input in equal dribbles. */
        double ramp_ratio = cohort_ramp_ratio;
        const char *rr_env = getenv("BWA_MEM3_COHORT_RAMP_RATIO");
        if (rr_env && *rr_env) {
            /* Parsed exactly like --cohort-ramp-ratio, and for the same reason:
             * atof() accepts a prefix, so '1.5x' silently becomes 1.5 and
             * 'abc' becomes 0.0. The 0.0 case would at least surface via the
             * `<= 1` guard below, but as a ratio the user never typed. Only
             * PARSEABILITY is checked here -- the `<= 1` range check stays
             * below, shared with the CLI value, because it is a documented
             * fallback rather than an error. */
            if (parse_full_double(rr_env, &ramp_ratio) != 0)
                fprintf(stderr, "WARNING: BWA_MEM3_COHORT_RAMP_RATIO='%s' is not "
                                "a number; ignoring it and using %.2f\n",
                        rr_env, ramp_ratio);
        }
        /* Spelled as the negation of the ACCEPT condition, not `<= 1.0`, so NaN
         * lands here too: strtod("nan") parses cleanly and every comparison
         * against NaN is false, so `<= 1.0` would wave it through as a valid
         * ratio. It would then poison the ramp -- prev * NaN is NaN, which is
         * neither >= task_size nor convertible to int64_t, putting the ramp back
         * on the undefined conversion the saturation in the read step exists to
         * avoid. A ratio that cannot grow the ramp is user error either way. */
        if (!(ramp_ratio > 1.0)) {
            fprintf(stderr, "[M::%s] a cohort ramp ratio of %.3f would never grow "
                    "the ramp to the cohort; using %.2f\n",
                    __func__, ramp_ratio, cohort_ramp_ratio_default);
            ramp_ratio = cohort_ramp_ratio_default;
        }
        aux.cohort_ramp_ratio = ramp_ratio;

        /* First-slice size, also env-overridable for sweeps. 0 selects the old
         * fractional shape (task_size / ratio^depth), which is what makes an
         * exact A/B against the pre-absolute behaviour possible; negative is
         * meaningless, so it is treated as 0 rather than silently clamped. */
        int64_t ramp_first = cohort_ramp_first;
        const char *rf_env = getenv("BWA_MEM3_COHORT_RAMP_FIRST");
        if (rf_env && *rf_env) {
            /* Validated exactly like --cohort-ramp-first. Without this, the env
             * spelling reintroduces the bug the flag's validation exists to
             * prevent: atoll("16M") is 16, a positive value, so
             * BWA_MEM3_COHORT_RAMP_FIRST=16M would silently request a SIXTEEN
             * BYTE first slice instead of 16 Mbases. Non-fatal here, unlike the
             * flag, because the index is already loaded by this point -- same
             * choice BWA_MEM3_CHUNK_CAP and BWA_MEM3_COHORT_SLICES make above. */
            if (parse_bounded_i64(rf_env, 0, INT64_MAX, &ramp_first) != 0)
                fprintf(stderr, "WARNING: BWA_MEM3_COHORT_RAMP_FIRST='%s' is not "
                                "a non-negative integer number of bases; ignoring "
                                "it and using %lld\n",
                        rf_env, (long long)ramp_first);
        }
        aux.cohort_ramp_first = ramp_first > 0 ? ramp_first : 0;
    }

    /* Pipeline depth. The 3-step pipeline (read / process / write) is gated by
     * how many of those steps can run at once. With 2 workers only 2 of the 3
     * run concurrently, so the single I/O-side worker serialises read(N+1) and
     * write(N-1) around the compute worker; at high thread counts that
     * read+write chain, not compute, binds the wall. A 3rd worker lets
     * read || process || write triple-overlap. It is not oversubscription: the
     * step mutex still admits only one worker into the compute step at a time,
     * so the extra worker only ever does single-threaded I/O. Cost is one more
     * chunk in flight. -1 (no_mt_io) forces single-threaded I/O as before. */
    const int pipe_workers = no_mt_io ? 1 : (opt->n_threads > 2 ? 3 : opt->n_threads);

    double sp_t0 = 0.0;
#ifdef STAGE_PROF
    /* stage_prof: arm per-chunk read/process/write profiling if --profile given */
    if (profile_path && profile_path[0]) {
#if defined(__x86_64__) || defined(_M_X64)
        const char *sp_arch = "x86_64";
#elif defined(__aarch64__) || defined(__arm64__)
        const char *sp_arch = "arm64";
#else
        const char *sp_arch = "unknown";
#endif
        const char *sp_in = (optind + 1 < argc) ? argv[optind + 1]
                          : (optind < argc ? argv[optind] : "");
        sp_init(profile_path, "bwa-mem3", PACKAGE_VERSION, sp_arch, opt->n_threads,
                opt->bam_mode ? "bam" : "sam",
                opt->bam_mode ? opt->bam_level : -1, sp_in);
        sp_set_workers(pipe_workers);   /* pipeline depth (process() arg) */
        sp_t0 = sp_wall();
    }
#endif

    tim = __rdtsc();

    /* Relay process function */
    process(&aux, fp, fp2, pipe_workers);

    tprof[PROCESS][0] += __rdtsc() - tim;

    if (sp_enabled()) {
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
#ifdef __linux__
        double rss_mb = (double)ru.ru_maxrss / 1024.0;          /* Linux: ru_maxrss is KiB */
#else
        double rss_mb = (double)ru.ru_maxrss / (1024.0*1024.0); /* macOS/BSD: ru_maxrss is bytes */
#endif
        /* mean_cores_busy is computed inside sp_finish from per-chunk proc_cpu */
        sp_finish(sp_wall() - sp_t0, 0.0, rss_mb);
    }

    /* Close meth BAM writer BEFORE free(opt) — opt->meth_mode is checked here. */
    int meth_mode_local = opt->meth_mode;

    // free memory
    int32_t nt = aux.opt->n_threads;
    if (!aux.ref_string_is_shm) {
        _mm_free(ref_string);
    }
    /* When ref_string aliases shm, the segment is owned by the loader
     * process; the kernel reclaims the mapping at process exit. */
    free(hdr_line);
    free(idx_hdr_lines);
    free(meth_orig_hdr_lines);
    free(opt);
    if (aux.legacy_reader) { kseq_destroy(aux.ks); err_gzclose(fp); }
    else { fast_kseq_destroy(aux.frks); fast_reader_close(aux.fr1); }
    // kclose reaps a `<cmd` input producer and returns its non-zero exit status;
    // capture it so a failed producer can't masquerade as a successful run.
    int ko_close_rc = kclose(ko);

    // PAIRED_END
    int ko2_close_rc = 0;
    if (aux.ks2 || aux.fr2) {
        if (aux.legacy_reader) { kseq_destroy(aux.ks2); err_gzclose(fp2); }
        else { fast_kseq_destroy(aux.frks2); fast_reader_close(aux.fr2); }
        ko2_close_rc = kclose(ko2);
    }

    /* BGZF flush + EOF marker errors surface only on close. Propagate to the
     * exit code so a truncated BAM doesn't masquerade as a successful run. */
    int exit_code = 0;
    // Name the failing input and its status (kclose returns the producer's exit
    // code) so a two-input run tells the user which `<cmd` producer died.
    if (ko_close_rc != 0) {
        fprintf(stderr, "ERROR: input command `%s' exited with status %d\n",
                argv[optind + 1], ko_close_rc);
        exit_code = 1;
    }
    if (ko2_close_rc != 0) {
        fprintf(stderr, "ERROR: input command `%s' exited with status %d\n",
                argv[optind + 2], ko2_close_rc);
        exit_code = 1;
    }
    if (meth_mode_local && g_meth_bam_writer != NULL) {
        int rc = meth_bam_writer_close(g_meth_bam_writer);
        if (rc != 0) {
            fprintf(stderr, "[meth] ERROR: BAM writer close rc=%d\n", rc);
            exit_code = 1;
        }
        g_meth_bam_writer = NULL;
    }
    /* D3 (PR-5): the original pac global is a borrowed pointer into
     * aux.meth_orig_pac (freed by meth_orig_ref_free_handles below); just
     * clear it. The retired f/r chrom map and un-converted ref fold no longer
     * exist. */
    g_meth_orig_pac = NULL;
    if (out_opened) {
        fclose(aux.fp);
    }

    if (bam_writer != NULL) {
        if (bam_writer_close(bam_writer) != 0) {
            fprintf(stderr, "[bam_writer] ERROR: close returned non-zero\n");
            exit_code = 1;
        }
        aux.bam_writer = NULL;
    }

    // new bwt/FMI
    /* D3 (--meth, PR-3): free the original-reference handles (bns/pac/.0123)
     * alongside the seed FM-index. NULL/no-op outside --meth; idempotent and
     * NULL-safe. Now reached on every run since the seeding checkpoint is gone. */
    meth_orig_ref_free_handles(&aux);
    delete(aux.fmi);

    /* Display runtime profiling stats */
    tprof[MEM][0] = __rdtsc() - tprof[MEM][0];
    display_stats(nt);

#ifdef BWA_MEM3_DEBUG_RESCUE_STATS
    /* How the --rescue-kmer anchor gate resolved over the whole run. A wall-time
     * delta is uninterpretable without these: a vote floor tuned too high drives
     * the narrowing rate toward zero while the scan still costs its full pass,
     * which reads as "the optimization did nothing" rather than "the gate
     * rejected everything". Printed unconditionally under the macro so a run with
     * the knob off shows 0/0/0 rather than nothing at all. */
    {
        extern std::atomic<uint64_t> g_rescue_stat_scans;
        extern std::atomic<uint64_t> g_rescue_stat_narrowed;
        extern std::atomic<uint64_t> g_rescue_stat_skipped;
        uint64_t sc = g_rescue_stat_scans.load(std::memory_order_relaxed);
        uint64_t nw = g_rescue_stat_narrowed.load(std::memory_order_relaxed);
        uint64_t sk = g_rescue_stat_skipped.load(std::memory_order_relaxed);
        fprintf(stderr, "[M::%s] rescue-anchor: scans %llu narrowed %llu (%.2f%%) "
                        "skipped %llu (%.2f%%)\n", __func__,
                (unsigned long long)sc, (unsigned long long)nw,
                sc ? 100.0 * (double)nw / (double)sc : 0.0,
                (unsigned long long)sk,
                sc ? 100.0 * (double)sk / (double)sc : 0.0);
    }
#endif

    return exit_code;
}
