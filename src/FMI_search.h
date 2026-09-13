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

Authors: Sanchit Misra <sanchit.misra@intel.com>; Vasimuddin Md <vasimuddin.md@intel.com>;
*****************************************************************************************/

#ifndef _FMI_SEARCH_H
#define _FMI_SEARCH_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

/* SIMD compatibility for ARM/x86 */
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
    #include "simd_compat.h"
#else
    #include <immintrin.h>
#endif
#include <fstream>

#include "read_index_ele.h"
#include "bwa.h"
#include "lockstep_width.h"  /* SMEM_LOCKSTEP_N (+ _MAX), runtime width + probe */
#include "fmi_seed_api.h"

#define DUMMY_CHAR 6

#define assert_not_null(x, size, cur_alloc) \
        if (x == NULL) { fprintf(stderr, "Allocation of %0.2lf GB for " #x " failed.\nCurrent Allocation = %0.2lf GB\n", size * 1.0 /(1024*1024*1024), cur_alloc * 1.0 /(1024*1024*1024)); exit(EXIT_FAILURE); }

#define \
GET_OCC(pp, c, occ_id_pp, y_pp, occ_pp, one_hot_bwt_str_c_pp, match_mask_pp) \
                int64_t occ_id_pp = pp >> CP_SHIFT; \
                int64_t y_pp = pp & CP_MASK; \
                int64_t occ_pp = cp_occ[occ_id_pp].cp_count[c]; \
                uint64_t one_hot_bwt_str_c_pp = cp_occ[occ_id_pp].one_hot_bwt_str[c]; \
                uint64_t match_mask_pp = one_hot_bwt_str_c_pp & one_hot_mask_array[y_pp]; \
                occ_pp += _mm_countbits_64(match_mask_pp);

/* Reusable scratch for FMI_search::sortSMEMs' rid counting sort. Hoisted out of
 * the per-batch malloc/free + memcpy (audit SEED-15): the caller owns one of
 * these per worker thread (see mem_cache in bwamem.h) and passes it in on every
 * batch, so the count/offset array (`cnt`) and the stable-scatter buffer (`tmp`)
 * are allocated once and grown on demand instead of allocated and freed per
 * call. Zero-initialize the struct before first use (all-NULL, zero caps); the
 * owner frees `cnt`/`tmp` with _mm_free at teardown (both NULL-safe). One
 * instance is single-threaded scratch — never share it across threads. */
typedef struct smem_sort_scratch
{
    int64_t *cnt;     /* counting-sort count/offset array; >= observed rid range */
    int64_t  cntCap;  /* allocated capacity of cnt, in int64_t entries           */
    SMEM    *tmp;     /* stable-scatter output buffer; >= observed SMEM count     */
    int64_t  tmpCap;  /* allocated capacity of tmp, in SMEM entries              */
}SmemSortScratch;

#define SAL_PFD 16

/* SMEM_LOCKSTEP_N (phase-2 SMEM lockstep width, the compile-time floor/default
 * and the enable guard) and SMEM_LOCKSTEP_N_MAX now live in lockstep_width.h,
 * included above, alongside the runtime width g_smem_lockstep_n and the startup
 * MLP probe that resolves it. */

/* Lockstep depth for the third-pass (bwtSeedStrategy) re-seeding, tuned
 * separately from the phase-2 SMEM depth above. Whether the lockstep driver
 * runs is a per-run decision (g_bwtseed_lockstep, resolved once at startup:
 * arm64 always on; x86 on iff every worker thread gets its own physical core;
 * BWA3_BWTSEED_LOCKSTEP pins it), because it only wins where nothing else hides
 * the cp_occ latency it overlaps -- see the policy in lockstep_width.h. N=8 is
 * the measured whole-aligner optimum on Graviton4 (-1.7% vs the scalar third
 * pass, -16.7% on the seeding stage). Set to 1 to compile the lockstep driver
 * out entirely (the `#if BWTSEED_LOCKSTEP_N > 1` guard) and always take the
 * scalar third-pass path. */
#ifndef BWTSEED_LOCKSTEP_N
#define BWTSEED_LOCKSTEP_N 8
#endif

/* Smallest slice of an index array a parallel-load worker is given. The load is
 * memory-bandwidth bound, so below this the per-thread create/join overhead is
 * a larger share of the work than the bandwidth it unlocks. */
#define FMI_PREAD_MIN_CHUNK (8UL << 20)

/* Number of workers to split an `nbytes` index-array read across, given the
 * caller's requested `nthreads`.
 *
 * Never returns more than `nthreads` when `nthreads` is positive, and never so
 * many that a chunk would fall below FMI_PREAD_MIN_CHUNK -- except for the
 * unavoidable single-worker case where `nbytes` is itself below the floor. A
 * non-positive `nthreads` clamps UP to 1: the result is always >= 1, since the
 * caller divides `nbytes` by it. Small references and a large BWA3_LOAD_THREADS
 * are what push against the floor; on a GB-scale index the load's own 8-worker
 * cap binds first.
 *
 * Exposed (rather than kept file-local with the pread machinery) so the chunk
 * arithmetic is unit-testable without a real index on disk. */
int fmi_pread_worker_count(size_t nbytes, int nthreads);

/* Bytes to request from a single pread() call, given how many remain in this
 * worker's chunk. macOS fails a pread() whose count exceeds INT_MAX with EINVAL,
 * so a chunk larger than 2GiB can never be read in one call -- and with the
 * 8-worker load cap that is every index over ~17.2GB.
 *
 * Exposed (rather than kept file-local with the pread machinery) so the clamp
 * is unit-testable without materialising a multi-GB file. */
size_t fmi_pread_request_size(size_t remaining);

/* Read the next `nbytes` of `fp` into `dst` using up to `nthreads` pread
 * workers, then leave the stream positioned exactly past them so a following
 * sequential read (the trailing sentinel index) still lands correctly.
 *
 * Aborts the process on a read error or short file. Exposed alongside the
 * worker count so the chunk-splitting and the stream postcondition are
 * unit-testable against a synthetic file. */
void fmi_pread_from_stream(FILE *fp, void *dst, size_t nbytes, int nthreads);

/* Detect the trailing sa_compx tag on a loaded FM-index (.bwt.2bit.64) file,
 * falling back to `default_compx` when no valid tag is present (a legacy
 * index built before this field existed).
 *
 * `fd` is the open file descriptor to pread() from; `file_size` is its size
 * on disk (e.g. from fstat); `ref_seq_len` is the reference_seq_len already
 * read from the file's header, which the implied layout size depends on.
 *
 * Aborts the process (EXIT_FAILURE) on a tail-read error or short/EOF read:
 * silently degrading to `default_compx` there would mis-size the SA-sample
 * arrays of a genuinely non-default index with no diagnostic. `default_compx`
 * is returned only for a fully-read tag that fails the layout test (a legacy,
 * pre-tag index) or a file too small to carry a tail.
 *
 * Shared by FMI_search::load_index (the disk loader, FMI_search.cpp) and
 * bwa_shm_layout_load (the shm sizing path, bwa_shm.cpp) -- both must agree
 * on the detected rate, since bwa_shm_compute sizes the shm segment the
 * loader then attaches to.
 *
 * This tail-detection heuristic is provably unambiguous and MUST NOT be
 * replaced by a dedicated magic-number/version field in the on-disk format:
 * see the implementation comment in FMI_search.cpp for the proof (the size
 * delta a false positive would require, 5*Delta == 8, has no integer
 * solution since 5 does not divide 8). Adding a version field would change
 * the on-disk format and break this branch's byte-identity guarantee with
 * stock indexes for no benefit. */
int64_t detect_sa_compx(int fd, int64_t file_size, int64_t ref_seq_len, int64_t default_compx);

/* Worker count for the index load: the caller's `n_threads` clamped to [1, 8],
 * overridable via the BWA3_LOAD_THREADS environment variable. A malformed
 * override (non-numeric, trailing garbage, non-positive, or unrepresentable as
 * a long) is warned about and ignored, keeping the computed default; a valid
 * override is honored and clamped to a 64-thread ceiling. Always returns >= 1.
 *
 * Exposed (rather than kept file-local with the pread machinery) so the
 * fail-closed env-parse validation is unit-testable without loading an index. */
int index_load_threads(int n_threads);

class FMI_search: public indexEle
{
    public:
    FMI_search(const char *fname);
    ~FMI_search();
    //int64_t beCalls;

    /* Read-only size accessors. Used by load_index_from_shm for section
     * validation, and by the round-trip test for byte-equality checks. */
    int64_t cp_occ_size_bytes()      const;
    int64_t sa_sample_count()        const;
    int64_t sa_ms_byte_size_bytes()  const { return sa_sample_count() * (int64_t)sizeof(int8_t); }
    int64_t sa_ls_word_size_bytes() const { return sa_sample_count() * (int64_t)sizeof(uint32_t); }

    /* Read-only data accessors. Used by the round-trip test to byte-compare
     * the packed segment payload against the in-memory loader's buffers. */
    const void     *cp_occ_data()     const { return cp_occ; }
    const int8_t   *sa_ms_byte_data() const { return sa_ms_byte; }
    const uint32_t *sa_ls_word_data() const { return sa_ls_word; }
    const int64_t  *count_data()      const { return count; }

    /* Return the shm segment base if load_index attached from shm, else NULL.
     * fastmap reuses this so it doesn't have to re-attach for the ref string. */
    uint8_t *shm_attached_base()     const { return shm_base; }

    /* emit_unpacked_ref defaults false: skip writing the unpacked `<prefix>.0123`.
     * `mem` pac-fetches the original reference from `.pac`, so `.0123` is never
     * read; pass true only to emit it for an external consumer (e.g. bwa-mem2).
     * sa_compx defaults to 3 (the historical rate); `bwa-mem3 index -u INT`
     * threads its validated [0,6] value down to here via bwa_idx_build. */
    int build_index(bool emit_unpacked_ref = false, int sa_compx = 3);
    /* load_pac=false skips loading the 2-bit packed reference (BNS only). D3
     * --meth uses this for the SEED index: seeding needs the FM-index + bns
     * (for the seed->original remap) but never the seed pac — extension/scoring
     * runs against the ORIGINAL pac (meth_orig_pac). Saves ~1.6 GB on hg38. */
    /* n_threads: worker count for the big-array disk reads (capped internally,
     * overridable via BWA3_LOAD_THREADS). Default 1 preserves prior behavior. */
    void load_index(bool load_pac = true, int n_threads = 1);

    /* Attach to a packed bwa-mem3 index segment from bwa_shm_attach. Sets
     * scalars and the cp_occ / sa_ms_byte / sa_ls_word pointers; the
     * destructor munmaps `base` and leaves the aliased buffers untouched. */
    void load_index_from_shm(uint8_t *base, size_t len);

    /* matchArray sizing contract (applies to all four SMEM-emitting methods
     * below). The previous internal `max_smem` capacity guard was removed;
     * the caller MUST pre-size matchArray to hold at least:
     *
     *   - getSMEMsOnePosOneThread, getSMEMsOnePosOneThread_lockstep,
     *     getSMEMsAllPosOneThread:   numReads * max_readlength SMEMs,
     *     where max_readlength is the function parameter passed in.
     *
     *   - bwtSeedStrategyAllPosOneThread:   numReads * max_seq_length SMEMs,
     *     where max_seq_length = max_i(seq_[i].l_seq), computed by the
     *     caller (this method takes no max_readlength parameter).
     *
     * Writing past the pre-sized capacity is undefined behavior. The actual
     * number of SMEMs written is reported via *__numTotalSmem (or the
     * int64_t return value on bwtSeedStrategyAllPosOneThread). */
    /* matchArray must hold at least numReads * max_readlength SMEMs (caller-sized). */
    void getSMEMsOnePosOneThread(uint8_t *enc_qdb,
                                 int32_t *query_pos_array,
                                 int32_t *min_intv_array,
                                 int32_t *rid_array,
                                 int32_t numReads,
                                 int32_t batch_size,
                                 const bseq1_t *seq_,
                                 int32_t *query_cum_len_ar,
                                 int32_t  max_readlength,
                                 int32_t minSeedLen,
                                 SMEM *matchArray,
                                 int64_t *__numTotalSmem);

    /* Lockstep-batched variant of getSMEMsOnePosOneThread: advances
     * SMEM_LOCKSTEP_N reads' SMEM walks in slot-interleaved order to expose
     * N independent backwardExt dependency chains to the CPU's out-of-order
     * engine. Per-read algorithm is byte-identical to the scalar path; only
     * the cross-read interleaving is new. Output in matchArray is written
     * in the same (read, smem) order as the scalar path via per-slot match
     * buffers flushed by input-index cursor.
     * matchArray must hold at least numReads * max_readlength SMEMs (caller-sized). */
    void getSMEMsOnePosOneThread_lockstep(uint8_t *enc_qdb,
                                          int32_t *query_pos_array,
                                          int32_t *min_intv_array,
                                          int32_t *rid_array,
                                          int32_t numReads,
                                          int32_t batch_size,
                                          const bseq1_t *seq_,
                                          int32_t *query_cum_len_ar,
                                          int32_t  max_readlength,
                                          int32_t minSeedLen,
                                          SMEM *matchArray,
                                          int64_t *__numTotalSmem);

    /* matchArray must hold at least numReads * max_readlength SMEMs (caller-sized). */
    void getSMEMsAllPosOneThread(uint8_t *enc_qdb,
                                 int32_t *min_intv_array,
                                 int32_t *rid_array,
                                 int32_t numReads,
                                 int32_t batch_size,
                                 const bseq1_t *seq_,
                                 int32_t *query_cum_len_ar,
                                 int32_t max_readlength,
                                 int32_t minSeedLen,
                                 SMEM *matchArray,
                                 int64_t *__numTotalSmem);


    /* matchArray must hold at least numReads * (longest l_seq in seq_) SMEMs;
     * returns the actual count written. */
    int64_t bwtSeedStrategyAllPosOneThread(uint8_t *enc_qdb,
                                           int32_t *max_intv_array,
                                           int32_t numReads,
                                           const bseq1_t *seq_,
                                           int32_t *query_cum_len_ar,
                                           int32_t minSeedLen,
                                           SMEM *matchArray);

    /* Lockstep batched variant of bwtSeedStrategyAllPosOneThread. Processes
     * BWTSEED_LOCKSTEP_N reads' forward-extension walks in slot-interleaved
     * order so the CP_OCC cache-line misses from independent reads issue
     * concurrently. Same SMEM emission order as the scalar (reads in input
     * order; within a read, outer-x ascending). matchArray must hold at
     * least numReads * max_readlength SMEMs. */
    int64_t bwtSeedStrategyAllPosOneThread_lockstep(uint8_t *enc_qdb,
                                                    int32_t *max_intv_array,
                                                    int32_t numReads,
                                                    const bseq1_t *seq_,
                                                    int32_t *query_cum_len_ar,
                                                    int32_t minSeedLen,
                                                    SMEM *matchArray,
                                                    int32_t max_readlength);
        
    void sortSMEMs(SMEM *matchArray,
                   int64_t numTotalSmem[],
                   int32_t numReads,
                   int32_t readlength,
                   int nthreads,
                   SmemSortScratch &scratch);
    // Uncompressed SA accessors -- index sa_ms_byte/sa_ls_word by raw BWT row, so
    // they read out of bounds against the SA_COMPRESSION-sized arrays. Fenced to
    // match their definitions in FMI_search.cpp; the compressed equivalents below
    // are the shipped paths. Every in-tree caller is likewise `#if !SA_COMPRESSION`.
#if !SA_COMPRESSION
    int64_t get_sa_entry(int64_t pos);
    void get_sa_entries(int64_t *posArray,
                        int64_t *coordArray,
                        uint32_t count,
                        int32_t nthreads);
    void get_sa_entries(SMEM *smemArray,
                        int64_t *coordArray,
                        int32_t *coordCountArray,
                        uint32_t count,
                        int32_t max_occ);
#endif // !SA_COMPRESSION
    int64_t get_sa_entry_compressed(int64_t pos, int tid=0);
    void get_sa_entries(SMEM *smemArray,
                        int64_t *coordArray,
                        int32_t *coordCountArray,
                        uint32_t count,
                        int32_t max_occ,
                        int tid);
    /* drop_sentinel_offset selects the coordinate reported when the LF walk
     * reaches the sentinel row: 0 = the accumulated walk offset (correct, bwa's
     * behaviour), 1 = 0 (bwa-mem2 v2.2.1's behaviour, selected by
     * --compat=bwa-mem2; see compat_target_t::sa_sentinel_drop_offset). */
    int64_t call_one_step(int64_t pos, int64_t &sa_entry, int64_t &offset,
                          int drop_sentinel_offset = 0);
    void get_sa_entries_prefetch(SMEM *smemArray, int64_t *coordArray,
                                 int64_t *coordCountArray, int64_t count,
                                 const int32_t max_occ, int tid, int64_t &id_,
                                 int drop_sentinel_offset = 0);
    
    int64_t reference_seq_len;
    int64_t sentinel_index;
    int64_t sa_compx;        // SA sample shift; sample rate 1/(1<<sa_compx). Loaded per-index.
    int64_t sa_compx_mask;   // (1<<sa_compx)-1
private:
        char file_name[PATH_MAX];
        int64_t count[5];
        uint32_t *sa_ls_word;
        int8_t *sa_ms_byte;
        CP_OCC *cp_occ;

        /* If non-NULL, cp_occ / sa_ms_byte / sa_ls_word point into a shared
         * memory mapping owned by the shm segment rather than being
         * _mm_malloc'd. shm_len is the byte length of the mapping so the
         * destructor can munmap it; the destructor also skips _mm_free on
         * shm-backed buffers. */
        uint8_t *shm_base;
        size_t   shm_len;

        /* Defined inline so all hot callers (getSMEMs* and ls_advance_*)
         * fully absorb the body and pay no struct-by-value pass / return
         * cost. backwardExt is called ~10^9 times per 5M-pair WGS run, so
         * the SysV-ABI hidden-pointer dance for the 24-byte SMEM struct
         * dominates self-time on gcc 12+ (issue #87): a single output store
         * `vmovdqu %ymm0, (%r8)` writing the return SMEM hits 42% of the
         * function's CPU samples on c7a (Zen 4) with gcc-14, with the
         * matching argument load close behind. always_inline removes the
         * call boundary and recovers the full gcc-11 baseline (and beats
         * it by ~4% wall-clock on c7a with a 5M-read WGS slice, shm-warmed). */
        __attribute__((always_inline)) inline
        SMEM backwardExt(SMEM smem, uint8_t a) const
        {
            /* sp/ep and their checkpoint block + one-hot mask do not depend on
             * the base b, so hoist them out of the per-base occ computation.
             * Only k[a] and s[a] are consumed for the result, but all four s[b]
             * feed the l cumulation below, so the full 4-lane occ is computed. */
            const int64_t sp = (int64_t)smem.k;
            const int64_t ep = (int64_t)smem.k + (int64_t)smem.s;
            const CP_OCC &blk_sp = cp_occ[sp >> CP_SHIFT];
            const CP_OCC &blk_ep = cp_occ[ep >> CP_SHIFT];
            const uint64_t mask_sp = one_hot_mask_array[sp & CP_MASK];
            const uint64_t mask_ep = one_hot_mask_array[ep & CP_MASK];

            int64_t occ_sp[4], s[4], l[4];

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
            /* arm64 has no scalar popcount instruction, so the scalar GET_OCC
             * path pays a GPR<->SIMD round-trip for every __builtin_popcountl
             * (eight per call). Compute all four bases in-lane instead: AND the
             * one-hot BWT words with the position mask, popcount each 64-bit
             * lane (cnt + pairwise-add reduction), and add the checkpoint
             * counts. Bit-identical to the scalar popcount, no cross-domain
             * moves, one load per checkpoint block. */
            const uint64x2_t msp = vdupq_n_u64(mask_sp);
            const uint64x2_t mep = vdupq_n_u64(mask_ep);
            #define BWA3_PC64(v) vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(v)))))
            const uint64x2_t psp01 = BWA3_PC64(vandq_u64(vld1q_u64(&blk_sp.one_hot_bwt_str[0]), msp));
            const uint64x2_t psp23 = BWA3_PC64(vandq_u64(vld1q_u64(&blk_sp.one_hot_bwt_str[2]), msp));
            const uint64x2_t pep01 = BWA3_PC64(vandq_u64(vld1q_u64(&blk_ep.one_hot_bwt_str[0]), mep));
            const uint64x2_t pep23 = BWA3_PC64(vandq_u64(vld1q_u64(&blk_ep.one_hot_bwt_str[2]), mep));
            #undef BWA3_PC64
            const uint64x2_t occ_sp01 = vaddq_u64(vld1q_u64((const uint64_t*)&blk_sp.cp_count[0]), psp01);
            const uint64x2_t occ_sp23 = vaddq_u64(vld1q_u64((const uint64_t*)&blk_sp.cp_count[2]), psp23);
            const uint64x2_t occ_ep01 = vaddq_u64(vld1q_u64((const uint64_t*)&blk_ep.cp_count[0]), pep01);
            const uint64x2_t occ_ep23 = vaddq_u64(vld1q_u64((const uint64_t*)&blk_ep.cp_count[2]), pep23);
            vst1q_u64((uint64_t*)&occ_sp[0], occ_sp01);
            vst1q_u64((uint64_t*)&occ_sp[2], occ_sp23);
            vst1q_u64((uint64_t*)&s[0], vsubq_u64(occ_ep01, occ_sp01));
            vst1q_u64((uint64_t*)&s[2], vsubq_u64(occ_ep23, occ_sp23));
#else
            for (uint8_t b = 0; b < 4; b++) {
                int64_t occ_s = blk_sp.cp_count[b] + _mm_countbits_64(blk_sp.one_hot_bwt_str[b] & mask_sp);
                int64_t occ_e = blk_ep.cp_count[b] + _mm_countbits_64(blk_ep.one_hot_bwt_str[b] & mask_ep);
                occ_sp[b] = occ_s;
                s[b]      = occ_e - occ_s;
            }
#endif

            int64_t sentinel_offset = 0;
            if ((smem.k <= sentinel_index) && ((smem.k + smem.s) > sentinel_index))
                sentinel_offset = 1;
            l[3] = smem.l + sentinel_offset;
            l[2] = l[3] + s[3];
            l[1] = l[2] + s[2];
            l[0] = l[1] + s[1];

            smem.k = count[a] + occ_sp[a];
            smem.l = l[a];
            smem.s = s[a];
            return smem;
        }

        /* Population count of (*word & *mask) for one 64-bit BWT word.
         * arm64 has no scalar popcount: __builtin_popcountll lowers to
         * fmov d,x; cnt.8b; addv.8b; fmov x,d -- two GPR<->SIMD moves on the
         * k' dependency chain. Loading the word and the mask straight into d
         * registers (both are memory operands here) leaves one move, the
         * result. Bit-identical. Other targets keep the scalar builtin. */
        static inline int64_t occ_popcount64(const uint64_t *word, const uint64_t *mask)
        {
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
            const uint8x8_t bits = vreinterpret_u8_u64(vand_u64(vld1_u64(word), vld1_u64(mask)));
            return (int64_t) vaddv_u8(vcnt_u8(bits));
#else
            return (int64_t) __builtin_popcountll(*word & *mask);
#endif
        }

        /* K-only backward extension for the pure-backward SMEM phase
         * (ls_advance_backward_step). Computes ONLY smem.k and smem.s, leaving
         * smem.l untouched: in the backward phase the reverse-complement interval
         * start `l` is dead — it is consumed only to compute the next step's `l`,
         * and every downstream reader of an emitted seed's `l` is redundant by the
         * (rid,m,n) => (k,l,s) invariant. Those readers were the smem_dedup
         * comparator (smem_dedup_inplace) and the scalar/lockstep field comparator
         * in the parity harness (test/smem_lockstep_parity_test.cpp); both have
         * been updated to drop `l`, so an emitted seed's identity is (rid,m,n,k,s).
         * NOTE this means the l this phase emits is intentionally stale (frozen at
         * its forward-phase value); it must never be resurrected as an observable
         * field without also restoring the l-chain here. Dropping the l-chain
         * removes the sentinel test, the l cumulation, and (on x86) six of eight
         * popcounts — same two cache lines read, far fewer instructions.
         * Byte-identical in observable (SAM) output.
         *
         * s==1 fast path: when the interval is a single suffix (common once the
         * match is unique), occ(a, sp+1) - occ(a, sp) is exactly the one BWT bit
         * at sp, so s' is a single bit test on the sp block — no ep block, no
         * second popcount. one_hot_mask_array is MSB-first (mask[i] = top i bits),
         * so BWT position `pos` lives at bit (CP_MASK - pos). On the s'->0 exit the
         * new k is left stale (unset): this is safe ONLY because every caller
         * discards a seed with s < min_intv while emitting the OLD smem, and every
         * caller uses min_intv >= 1 (enforced by an xassert in ls_init_slot), so an
         * s'==0 result is always discarded and its k is never read. Under that
         * precondition the remaining popcount is skipped too. */
        __attribute__((always_inline)) inline
        SMEM backwardExt_konly(SMEM smem, uint8_t a) const
        {
            const int64_t sp = (int64_t)smem.k;
            if (smem.s == 1) {
                const CP_OCC &blk = cp_occ[sp >> CP_SHIFT];
                const int pos = (int)(sp & CP_MASK);
                const uint64_t oh = blk.one_hot_bwt_str[a];
                if ((oh >> (CP_MASK - pos)) & 1ULL) {  /* BWT[sp] == a  => s' = 1 */
                    smem.k = count[a] + blk.cp_count[a] +
                             occ_popcount64(&blk.one_hot_bwt_str[a], &one_hot_mask_array[pos]);
                    smem.s = 1;
                } else {                               /* s' = 0; new k is dead */
                    smem.s = 0;
                }
                return smem;
            }
            const int64_t ep = sp + (int64_t)smem.s;
            const CP_OCC &blk_sp = cp_occ[sp >> CP_SHIFT];
            const CP_OCC &blk_ep = cp_occ[ep >> CP_SHIFT];
            const int64_t occ_s = blk_sp.cp_count[a] +
                occ_popcount64(&blk_sp.one_hot_bwt_str[a], &one_hot_mask_array[(sp & CP_MASK)]);
            const int64_t occ_e = blk_ep.cp_count[a] +
                occ_popcount64(&blk_ep.one_hot_bwt_str[a], &one_hot_mask_array[(ep & CP_MASK)]);
            smem.k = count[a] + occ_s;
            smem.s = occ_e - occ_s;
            return smem;
        }

    // ----- Lockstep SMEM batching internals -----
    // Defined in FMI_search.cpp. Phases and per-slot state are scoped to
    // the lockstep driver; not used anywhere else.
    struct BatchSlot;

    void ls_init_slot(BatchSlot *s, int32_t input_idx,
                      const int32_t *query_pos_array,
                      const int32_t *min_intv_array,
                      const int32_t *rid_array,
                      const bseq1_t *seq_,
                      const int32_t *query_cum_len_ar,
                      const uint8_t *enc_qdb);
    void ls_prefetch_cp_occ(const BatchSlot *s);
    void ls_prefetch_cp_occ_t1(const BatchSlot *s);
    void ls_advance_forward_step(BatchSlot *s, const uint8_t *enc_qdb);
    void ls_prepare_backward(BatchSlot *s);
    void ls_advance_backward_step(BatchSlot *s,
                                  const uint8_t *enc_qdb,
                                  int32_t minSeedLen);

    // ----- Lockstep bwtSeed batching internals -----
    struct BwtSeedSlot;

    void bsd_init_slot(BwtSeedSlot *s, int32_t input_idx,
                       const int32_t *max_intv_array,
                       const bseq1_t *seq_,
                       const int32_t *query_cum_len_ar,
                       const uint8_t *enc_qdb);
    void bsd_prefetch_cp_occ(const BwtSeedSlot *s);
    void bsd_prefetch_cp_occ_t1(const BwtSeedSlot *s);
    void bsd_advance_step(BwtSeedSlot *s,
                          const uint8_t *enc_qdb,
                          int32_t minSeedLen);
};

#endif
