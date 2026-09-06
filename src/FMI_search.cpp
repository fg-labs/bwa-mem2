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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <climits>
#include <cstring>
#include <vector>
#include <cstdarg>
#include <pthread.h>
#include <unistd.h>       /* pread, _exit */
#include <sys/mman.h>     /* munmap */
#include <sys/stat.h>     /* fstat */
#if defined(__linux__)
#include <fcntl.h>        /* posix_fadvise */
#endif
#include "bwa_madvise.h"
#include "bwa_shm.h"
#include "utils.h"        /* ATTRIBUTE, err_fread_noeof */
#include "io_utils.h"     /* io_request_size, IO_MAX_ONCE */
#include "FMI_search.h"
#include "profiling.h"
#include "libsais_build.h"

/*
 * Parallel index load.
 *
 * The bwa-mem3 FM-index (cp_occ + compressed SA samples, ~10 GB on hg38) is
 * otherwise slurped by a single-threaded fread whose warm-cache cost is one
 * core's page-fault + memcpy bandwidth out of the page cache (~0.8 s on hg38).
 * Splitting each big array across a few workers is memory-bandwidth bound and
 * cuts that ~4x. pread (not read) lets every worker share one fd without
 * touching the shared file offset, so no locking is needed.
 *
 * The destination stays the _mm_malloc'd + MADV_HUGEPAGE buffer, so the
 * transparent-hugepage coverage the hot Occ-lookup loop relies on is preserved
 * -- unlike an mmap/shm alias of the file, which lands on non-THP pages and
 * slows alignment enough to erase the load saving.
 */

/* Declared in FMI_search.h; see there for the contract. Uses a floor division
 * so the chunk size never dips below FMI_PREAD_MIN_CHUNK: at `nbytes` a whole
 * multiple of the floor, `nbytes / min_chunk + 1` would hand out one chunk more
 * than the array can cover at that size (8 MB read across 2 workers = 4 MB
 * chunks). Only reachable for small arrays -- a caller asking for more workers
 * than a test-sized reference can feed, or a large BWA3_LOAD_THREADS. */
int fmi_pread_worker_count(size_t nbytes, int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    size_t max_threads = nbytes / FMI_PREAD_MIN_CHUNK;
    if (max_threads < 1) max_threads = 1;
    if ((size_t)nthreads > max_threads) nthreads = (int)max_threads;
    return nthreads;
}

size_t fmi_pread_request_size(size_t remaining)
{
    // Delegate to the one shared clamp so read and write agree on the cap; see
    // io_utils.h. pread() has the same macOS >INT_MAX EINVAL cap as pwrite().
    return io_request_size(remaining, IO_MAX_ONCE);
}

namespace {

struct PreadChunk { int fd; char *dst; size_t nbytes; off_t off; };

/* Bail out of a failed chunk read.
 *
 * _exit, not exit: this runs on a worker thread while its siblings are still
 * inside pread(). exit() would run atexit handlers and static destructors
 * (mimalloc teardown, htslib) against live threads, and two workers reaching it
 * at once -- what a truncated index produces, since every chunk past the real
 * EOF fails together -- is undefined. stderr is unbuffered, but flush anyway so
 * the diagnostic survives if a caller made it buffered.
 *
 * noreturn keeps the caller's `if (r < 0) { ... }` terminating the way the
 * inline exit() it replaced did; format(printf) keeps -Wformat checking the
 * call sites, matching utils.h's err_fatal family. */
void pread_chunk_fail(const char *fmt, ...)
    ATTRIBUTE((noreturn)) ATTRIBUTE((format(printf, 1, 2)));

void pread_chunk_fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
    _exit(EXIT_FAILURE);
}

void *pread_chunk_worker(void *arg)
{
    PreadChunk *c = static_cast<PreadChunk *>(arg);
    size_t done = 0;
    /* macOS pread() rejects any count > INT_MAX with EINVAL, so a chunk larger
     * than 2GiB fails outright -- with the 8-worker cap that is every index over
     * ~17.2GB (the hg38 --meth FM-index is 20.9GB). Request a bounded slice per
     * call; the loop already accumulates partial reads. Linux has no such limit,
     * which is why this only ever bit local macOS runs. */
    while (done < c->nbytes) {
        ssize_t r = pread(c->fd, c->dst + done,
                          fmi_pread_request_size(c->nbytes - done),
                          c->off + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) continue;
            pread_chunk_fail("ERROR: pread failed during index load: %s\n", strerror(errno));
        }
        if (r == 0) {  // index file shorter than the header says it should be
            // Counts are for THIS worker's chunk, not the whole array: a
            // truncated index trips every chunk past the real EOF, so several
            // of these can interleave, each describing a different slice.
            pread_chunk_fail("ERROR: unexpected EOF during index load "
                             "(chunk at offset %lld: %zu of %zu bytes)\n",
                             (long long)c->off, done, c->nbytes);
        }
        done += (size_t)r;
    }
    return NULL;
}

// Read `nbytes` at file offset `off` into `dst` using up to `nthreads` workers.
void parallel_pread(int fd, void *dst, size_t nbytes, off_t off, int nthreads)
{
    if (nbytes == 0) return;
    nthreads = fmi_pread_worker_count(nbytes, nthreads);

    std::vector<PreadChunk> chunks((size_t)nthreads);
    std::vector<pthread_t>  tids((size_t)nthreads);
    std::vector<bool>       spawned((size_t)nthreads, false);
    size_t base = nbytes / (size_t)nthreads, rem = nbytes % (size_t)nthreads, cum = 0;
    for (int i = 0; i < nthreads; i++) {
        size_t len = base + ((size_t)i < rem ? 1 : 0);
        chunks[i] = PreadChunk{ fd, (char *)dst + cum, len, off + (off_t)cum };
        cum += len;
    }
    // Main thread takes chunk 0; spawn workers for the rest. If a spawn fails
    // (e.g. thread limit), fall back to reading that chunk inline so the load
    // still completes correctly, just less parallel.
    for (int i = 1; i < nthreads; i++) {
        if (pthread_create(&tids[i], NULL, pread_chunk_worker, &chunks[i]) == 0)
            spawned[i] = true;
        else
            pread_chunk_worker(&chunks[i]);
    }
    pread_chunk_worker(&chunks[0]);
    for (int i = 1; i < nthreads; i++)
        if (spawned[i]) pthread_join(tids[i], NULL);
}

}  // namespace

// Worker count for the index load: the caller's -t, capped (bandwidth bound
// past ~8), with an explicit BWA3_LOAD_THREADS override for tuning.
// Exposed (declared in FMI_search.h) so the env-parse validation is
// unit-testable without loading a real index.
int index_load_threads(int n_threads)
{
    int t = n_threads > 0 ? n_threads : 1;
    if (t > 8) t = 8;
    const char *e = getenv("BWA3_LOAD_THREADS");
    if (e != NULL && *e != '\0') {
        // Strict parse: atoi silently accepts garbage (abc -> 0, 0x8 -> 0) and
        // any huge value (100000 -> ~1280 pthread_creates). On malformed input
        // warn and keep the computed default -- ignore-and-continue like
        // BWA3_SMEM_LOCKSTEP_N (bwa3_lockstep_width_parse_env), NOT the hard
        // exit(1) that BWA_INDEX_THREADS's parse_ll takes. Clamp to a 64-thread
        // ceiling (the load is bandwidth-bound past ~8).
        char *end = NULL;
        errno = 0;
        long v = strtol(e, &end, 10);
        if (errno != 0 || end == e || *end != '\0' || v <= 0) {
            fprintf(stderr, "ERROR: BWA3_LOAD_THREADS=\"%s\" is not a positive integer; ignoring it\n", e);
        } else {
            if (v > 64) v = 64;   // ceiling; the load is bandwidth-bound past ~8
            t = (int)v;
        }
    }
    return t;
}

/* Declared in FMI_search.h; see there for the contract. ftello reports the
 * stream's logical position (buffered bytes included), so it is the right
 * offset to hand pread; the fseeko afterwards drops the now-stale buffer and
 * re-anchors the stream past the array we just read behind its back. */
void fmi_pread_from_stream(FILE *fp, void *dst, size_t nbytes, int nthreads)
{
    off_t off = ftello(fp);
    if (off < 0) {
        fprintf(stderr, "ERROR: ftello failed during index load: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    parallel_pread(fileno(fp), dst, nbytes, off, nthreads);
    if (fseeko(fp, off + (off_t)nbytes, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: fseeko failed during index load: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

/* Declared in FMI_search.h; see there for the contract.
 *
 * The on-disk layout is: header(48B: reference_seq_len + count[5], i.e.
 * 6*sizeof(int64_t)) + cp_occ + SA-sample arrays (sized by sa_compx) +
 * sentinel_index(8B) [+ sa_compx tag(8B), new format only]. Since the
 * SA-sample-array size depends on sa_compx, the tag's own file offset can't
 * be computed without already knowing sa_compx -- so instead we read the
 * file's last 8 bytes as a `candidate` sa_compx and verify it reproduces the
 * file's actual size when plugged back into the layout-size formula
 * (`off_sent_for`). If it doesn't -- including on a legacy, pre-this-feature
 * index, which carries no tag at all -- we fall back to `default_compx`.
 *
 * Why this is safe without a version field: a legacy index (no tag, always
 * built at the fixed compile-time SA_COMPX rate) has file size
 *
 *   legacy_size = off_sent_for(SA_COMPX) + 8            (sentinel only)
 *
 * while a false-positive match against some candidate `c` would require
 *
 *   legacy_size == off_sent_for(c) + 16                  (sentinel + tag)
 *
 * Each unit of SA-sample count changes off_sent_for by exactly 5 bytes
 * (1-byte int8_t + 4-byte uint32_t per sample), so equating the two reduces
 * to 5*Delta == 8 for some integer Delta = sa_sample_cnt(SA_COMPX) -
 * sa_sample_cnt(c). Since 5 does not divide 8, no integer Delta -- and hence
 * no candidate `c`, including c == SA_COMPX itself (Delta == 0, 0 != 8) --
 * can satisfy it. A legacy index can therefore never be misdetected as
 * carrying a valid tail. This is a property of the byte accounting, not of
 * a chosen constant, so it must not be "shored up" with a magic-number or
 * version field: doing so would change the on-disk format and break this
 * branch's byte-identity guarantee against stock indexes for no benefit. */
int64_t detect_sa_compx(int fd, int64_t file_size, int64_t ref_seq_len, int64_t default_compx)
{
    int64_t candidate = -1;
    if (file_size >= (int64_t)(2 * sizeof(int64_t))) {
        /* Robustly read the trailing 8-byte candidate tag. A read error or a
         * short/EOF read here must NOT silently fall back to default_compx: a
         * genuinely non-default index would then be mis-sized (wrong SA-sample
         * offsets, no diagnostic) -- the same failure the fstat() guards at
         * both call sites already prevent. Retry EINTR and treat any other
         * failure as a fatal load error, mirroring pread_chunk_worker() above.
         * Only a fully-read candidate that fails the layout test below
         * legitimately falls back to default_compx. */
        char   *dst  = (char *)&candidate;
        off_t   off  = (off_t)(file_size - (int64_t)sizeof(int64_t));
        size_t  done = 0;
        while (done < sizeof(int64_t)) {
            ssize_t r = pread(fd, dst + done, sizeof(int64_t) - done,
                              off + (off_t)done);
            if (r < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "ERROR: pread failed reading SA-rate tag "
                                "during index load: %s\n", strerror(errno));
                exit(EXIT_FAILURE);
            }
            if (r == 0) {
                fprintf(stderr, "ERROR: unexpected EOF reading SA-rate tag "
                                "during index load\n");
                exit(EXIT_FAILURE);
            }
            done += (size_t)r;
        }
    }

    auto off_sent_for = [ref_seq_len](int64_t compx) -> int64_t {
        // Mirrors HDR_BYTES in fm_index_writer.cpp / BWA_BWT_2BIT_HEADER_BYTES
        // in bwa_shm.cpp: ref_seq_len + count[5].
        const int64_t hdr_bytes      = 6 * (int64_t)sizeof(int64_t);
        const int64_t cp_occ_cnt     = (ref_seq_len >> CP_SHIFT) + 1;
        const int64_t sa_sample_cnt  = (ref_seq_len >> compx) + 1;
        return hdr_bytes + cp_occ_cnt * (int64_t)sizeof(CP_OCC)
                         + sa_sample_cnt * (int64_t)sizeof(int8_t)
                         + sa_sample_cnt * (int64_t)sizeof(uint32_t);
    };

    if (candidate >= 0 && candidate <= CP_SHIFT &&
        off_sent_for(candidate) + 2 * (int64_t)sizeof(int64_t) == file_size) {
        return candidate;
    }
    return default_compx;
}

/* Build "<prefix><suffix>" into `out` (sized `outsz`); aborts on overflow.
 * Replaces the prior strcpy_s/strcat_s pattern for assembling FMI sidecar
 * paths from a user-supplied prefix. */
static void fmi_build_path(char *out, size_t outsz, const char *prefix, const char *suffix)
{
    int n = snprintf(out, outsz, "%s%s", prefix, suffix);
    if (n < 0 || (size_t)n >= outsz) {
        fprintf(stderr, "ERROR: FMI path too long for prefix '%s' (suffix '%s')\n",
                prefix, suffix);
        exit(EXIT_FAILURE);
    }
}

FMI_search::FMI_search(const char *fname)
{
    fprintf(stderr, "* Entering FMI_search\n");
    fmi_build_path(file_name, sizeof(file_name), fname, "");
    reference_seq_len = 0;
    sentinel_index = 0;
    sa_compx = SA_COMPX;
    sa_compx_mask = SA_COMPX_MASK;
    sa_ls_word = NULL;
    sa_ms_byte = NULL;
    cp_occ = NULL;
    shm_base = NULL;
    shm_len = 0;

    /* one_hot_mask_array is now a file-scope compile-time constant table (see
     * FMI_search.h); nothing to allocate or initialize here. */
}

FMI_search::~FMI_search()
{
    /* When attached from shm, cp_occ/sa_*_byte/sa_*_word alias mmap'd pages
     * owned by the shm segment, so we don't _mm_free them. We do munmap the
     * mapping itself — leaving it would leak VA in long-lived processes that
     * construct and destroy FMI_search repeatedly. */
    if (shm_base != NULL) {
        munmap(shm_base, shm_len);
        shm_base = NULL;
        shm_len  = 0;
    } else {
        if (sa_ms_byte) _mm_free(sa_ms_byte);
        if (sa_ls_word) _mm_free(sa_ls_word);
        if (cp_occ)     _mm_free(cp_occ);
    }
}

int64_t FMI_search::cp_occ_size_bytes() const {
    return ((reference_seq_len >> CP_SHIFT) + 1) * (int64_t)sizeof(CP_OCC);
}

int64_t FMI_search::sa_sample_count() const {
    return (reference_seq_len >> sa_compx) + 1;
}

void FMI_search::load_index_from_shm(uint8_t *base, size_t len)
{
    if (base == NULL) {
        fprintf(stderr, "ERROR! load_index_from_shm called with NULL base\n");
        exit(EXIT_FAILURE);
    }
    uint64_t off = 0, sz = 0;

    if (bwa_shm_section_find(base, BWA_SHM_SEC_FMI_SCALARS, &off, &sz) != 0
        || sz != BWA_SHM_FMI_SCALARS_BYTES) {
        fprintf(stderr, "ERROR! shm segment missing or malformed FMI_SCALARS\n");
        exit(EXIT_FAILURE);
    }
    memcpy(&reference_seq_len, base + off,                                    sizeof(int64_t));
    memcpy(count,              base + off + sizeof(int64_t),                  sizeof(int64_t) * 5);
    memcpy(&sentinel_index,    base + off + sizeof(int64_t) * 6,              sizeof(int64_t));
    /* sa_compx: the SA sample-rate shift the staged index was built with.
     * IMPORTANT FIX: this used to stay at the constructor's compile-time
     * default (SA_COMPX) because nothing here overrode it, so a non-default
     * `-u`-built index attached via shm was mis-sized (sa_sample_count() and
     * every sa_ms_byte/sa_ls_word index below used the wrong shift). Reading
     * it from the packed scalars (see BWA_SHM_FMI_SCALARS_BYTES / the PACK
     * side in bwa_shm.cpp) makes shm-attach agree with the disk loader's
     * tail-detected rate. */
    memcpy(&sa_compx,          base + off + sizeof(int64_t) * 7,              sizeof(int64_t));

    /* Validate scalars before we use reference_seq_len in cp_occ_size_bytes()
     * and the SA size accessors. Bounds match the disk path's asserts in
     * load_index() and the writer-side checks in bwa_shm_compute(). A corrupt
     * segment would otherwise drive negative or overflowing section sizes. */
    if (reference_seq_len <= 0 || reference_seq_len > 0x7fffffffffLL) {
        fprintf(stderr,
            "ERROR! shm FMI_SCALARS: reference_seq_len=%lld out of bounds\n",
            (long long)reference_seq_len);
        exit(EXIT_FAILURE);
    }
    /* count[] is +1-adjusted by bwa_shm_compute; range [1, ref_seq_len+1], and
     * as the cumulative C[] array it must be non-decreasing. These bounds match
     * the disk path's post-adjustment asserts in load_index(): a count[i] == 0
     * drives smem.k out of cp_occ on the first backwardExt, and a non-monotone
     * pair yields a negative smem.s (interval width) during seed extraction, so
     * a corrupt segment must fail closed here rather than attach with an invalid
     * C array. */
    for (int i = 0; i < 5; ++i) {
        if (count[i] < 1 || count[i] > reference_seq_len + 1) {
            fprintf(stderr,
                "ERROR! shm FMI_SCALARS: count[%d]=%lld out of bounds (ref_seq_len=%lld)\n",
                i, (long long)count[i], (long long)reference_seq_len);
            exit(EXIT_FAILURE);
        }
        if (i > 0 && count[i] < count[i - 1]) {
            fprintf(stderr,
                "ERROR! shm FMI_SCALARS: count[%d]=%lld < count[%d]=%lld "
                "(not monotonically non-decreasing)\n",
                i, (long long)count[i], i - 1, (long long)count[i - 1]);
            exit(EXIT_FAILURE);
        }
    }
    if (sentinel_index < 0 || sentinel_index >= reference_seq_len) {
        fprintf(stderr,
            "ERROR! shm FMI_SCALARS: sentinel_index=%lld out of bounds (ref_seq_len=%lld)\n",
            (long long)sentinel_index, (long long)reference_seq_len);
        exit(EXIT_FAILURE);
    }
    /* [0,6]: same range the -u CLI validates and write_fm_index_streaming
     * enforces (the sample period 1<<sa_compx must divide CP_BLOCK_SIZE=64). */
    if (sa_compx < 0 || sa_compx > 6) {
        fprintf(stderr,
            "ERROR! shm FMI_SCALARS: sa_compx=%lld out of bounds\n",
            (long long)sa_compx);
        exit(EXIT_FAILURE);
    }
    /* Only compute the mask once sa_compx is known to be in [0,6]; shifting by
     * an out-of-range value (e.g. a corrupt -1 or 64) would be UB. */
    sa_compx_mask = (1LL << sa_compx) - 1;

    if (bwa_shm_section_find(base, BWA_SHM_SEC_FMI_CP_OCC, &off, &sz) != 0
        || (int64_t)sz != cp_occ_size_bytes()) {
        fprintf(stderr, "ERROR! shm segment missing or sized FMI_CP_OCC\n");
        exit(EXIT_FAILURE);
    }
    cp_occ = (CP_OCC *)(base + off);

    if (bwa_shm_section_find(base, BWA_SHM_SEC_FMI_SA_MS, &off, &sz) != 0
        || (int64_t)sz != sa_ms_byte_size_bytes()) {
        fprintf(stderr, "ERROR! shm segment missing or sized FMI_SA_MS\n");
        exit(EXIT_FAILURE);
    }
    sa_ms_byte = (int8_t *)(base + off);

    if (bwa_shm_section_find(base, BWA_SHM_SEC_FMI_SA_LS, &off, &sz) != 0
        || (int64_t)sz != sa_ls_word_size_bytes()) {
        fprintf(stderr, "ERROR! shm segment missing or sized FMI_SA_LS\n");
        exit(EXIT_FAILURE);
    }
    sa_ls_word = (uint32_t *)(base + off);

    shm_base = base;
    shm_len  = len;

    fprintf(stderr, "* FMI attached from shm: ref_seq_len=%ld sentinel_index=%ld\n",
            (long)reference_seq_len, (long)sentinel_index);
}

int FMI_search::build_index(bool emit_unpacked_ref, int sa_compx) {

    char *prefix = file_name;

    // Read single-strand length from .ann to compute doubled pac_len.
    char ann_path[PATH_MAX];
    fmi_build_path(ann_path, sizeof(ann_path), prefix, ".ann");
    FILE* fann = fopen(ann_path, "r");
    if (fann == NULL) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", ann_path);
        return 1;
    }
    int64_t l_pac = 0;
    int n_seqs = 0, seed = 0;
    if (fscanf(fann, "%" SCNd64 " %d %d", &l_pac, &n_seqs, &seed) != 3) {
        fprintf(stderr, "ERROR: malformed '%s'\n", ann_path);
        fclose(fann);
        return 1;
    }
    fclose(fann);
    // Defensive: a non-positive l_pac (corrupt or zero-length .ann) would
    // pass a bad pac_len into libsais and only fail much later. Catch it
    // up front with an actionable message, mirroring bntseq_restore_core.
    if (l_pac <= 0 || n_seqs < 0) {
        fprintf(stderr, "ERROR: malformed '%s' (l_pac=%" PRId64 ", n_seqs=%d)\n",
                ann_path, l_pac, n_seqs);
        return 1;
    }
    int64_t pac_len = 2 * l_pac;

    auto parse_ll = [](const char* s, const char* name,
                       long long max_val = LLONG_MAX) -> long long {
        char* end = nullptr;
        errno = 0;
        long long v = strtoll(s, &end, 10);
        if (errno || end == s || *end != '\0' || v <= 0 || v > max_val) {
            fprintf(stderr, "ERROR: invalid %s='%s' (expected positive integer)\n",
                    name, s);
            exit(1);
        }
        return v;
    };
    LibsaisBuildOpts opts;
    if (const char* th = getenv("BWA_INDEX_THREADS"))
        opts.num_threads      = (int)parse_ll(th, "BWA_INDEX_THREADS", INT_MAX);
    if (const char* mm = getenv("BWA_INDEX_MAX_MEMORY"))
        opts.max_memory_bytes = parse_ll(mm, "BWA_INDEX_MAX_MEMORY");
    if (const char* td = getenv("BWA_INDEX_TMPDIR"))
        opts.tmpdir           = td;
    if (const char* mu = getenv("BWA_INDEX_MAX_MEMORY_USER"))
        opts.max_memory_user_specified = (mu[0] == '1');
    opts.emit_unpacked_ref = emit_unpacked_ref;
    opts.sa_compx          = sa_compx;
    return libsais_build_fm_index(prefix, pac_len, opts);
}

void FMI_search::load_index(bool load_pac, int n_threads)
{
    /* Try the staged shm segment first. On hit, both the FMI internals
     * (cp_occ / sa_*) and the BNS+PAC are attached as views into the
     * mapping; the segment lifetime belongs to the shm-loading process. */
    {
        size_t   shm_attach_len = 0;
        uint8_t *shm_base_local = bwa_shm_attach(file_name, &shm_attach_len);
        if (shm_base_local != NULL) {
            load_index_from_shm(shm_base_local, shm_attach_len);
            /* D3 --meth seed segment is staged bns_only: no PAC section, and
             * idx->pac stays NULL (extension uses meth_orig_pac). */
            bwa_idx_load_ele_from_shm(shm_base_local, shm_attach_len, load_pac);
            fprintf(stderr, "* FMI+BNS%s attached from shm; "
                    "skipping disk load.\n", load_pac ? "+PAC" : "");
            return;
        }
    }

    // Running total of index bytes allocated so far in this function.
    // Passed to assert_not_null so a failed allocation reports both the
    // attempted size and how much we'd already committed before failing.
    int64_t index_alloc = 0;

    // Worker count for the big-array reads below (read once, out of any loop).
    const int load_nt = index_load_threads(n_threads);

    char *ref_file_name = file_name;
    //beCalls = 0;
    char cp_file_name[PATH_MAX];
    fmi_build_path(cp_file_name, sizeof(cp_file_name), ref_file_name, CP_FILENAME_SUFFIX);

    // Read the BWT and FM index of the reference sequence
    FILE *cpstream = NULL;
    cpstream = fopen(cp_file_name,"rb");
    if (cpstream == NULL)
    {
        fprintf(stderr, "ERROR! Unable to open the file: %s\n", cp_file_name);
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "* Index file found. Loading index from %s\n", cp_file_name);
    }

#if defined(__linux__)
    // Kick readahead for the whole index so the parallel preads below hit warm
    // pages on a cold cache (no-op benefit when already cached).
    posix_fadvise(fileno(cpstream), 0, 0, POSIX_FADV_WILLNEED);
#endif

    err_fread_noeof(&reference_seq_len, sizeof(int64_t), 1, cpstream);
    // Fail closed on a truncated/corrupt index, mirroring load_index_from_shm's
    // scalar bounds checks (a bad reference_seq_len/count[]/sentinel_index drives
    // smem.k past cp_occ on the first backwardExt -> wrong seeds, not a crash).
    // xassert survives a hypothetical -DNDEBUG build where a plain assert would not.
    xassert(reference_seq_len > 0 && reference_seq_len <= 0x7fffffffffLL,
            "FMI index: reference_seq_len out of bounds (corrupt .bwt.2bit.64?)");

    fprintf(stderr, "* Reference seq len for bi-index = %lld\n", (long long)reference_seq_len);

    // Peek the trailing sa_compx field (if present) before sizing the SA
    // sample arrays below, via the shared detect_sa_compx() helper (see its
    // doc comment in FMI_search.h and implementation comment above for why
    // the tail-detection heuristic is safe). pread doesn't disturb
    // cpstream's buffered read position, so this doesn't affect the
    // sequential reads that follow.
    {
        struct stat st;
        /* A failed fstat() must not silently fall back to file_size=0: that
         * would make detect_sa_compx() treat a genuinely non-default-rate
         * index as legacy (rate 3), mis-sizing the SA-sample arrays below
         * and producing wrong seed coordinates with no diagnostic. Fail the
         * load instead -- consistent with the ftello/fseeko error handling
         * just above in fmi_pread_from_stream(). */
        if (fstat(fileno(cpstream), &st) != 0) {
            fprintf(stderr, "ERROR: fstat failed during index load: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
        sa_compx = detect_sa_compx(fileno(cpstream), (int64_t)st.st_size, reference_seq_len, SA_COMPX);
        sa_compx_mask = (1LL << sa_compx) - 1;
    }

    // create checkpointed occ
    int64_t cp_occ_size = (reference_seq_len >> CP_SHIFT) + 1;
    cp_occ = NULL;

    err_fread_noeof(&count[0], sizeof(int64_t), 5, cpstream);
    int64_t cp_occ_bytes = cp_occ_size * sizeof(CP_OCC);
    cp_occ = (CP_OCC *)_mm_malloc(cp_occ_bytes, 64);
    index_alloc += cp_occ_bytes;
    assert_not_null(cp_occ, cp_occ_bytes, index_alloc);
    bwamem_madv_hugepage(cp_occ, cp_occ_bytes);

    fmi_pread_from_stream(cpstream, cp_occ, (size_t)cp_occ_bytes, load_nt);
    int64_t ii = 0;
    // Validate the RAW on-disk count[] before the +1 adjustment below. Doing the
    // bounds/monotonicity checks on the raw values (rather than after +1) is what
    // makes this overflow-safe: a corrupt count[ii] == INT64_MAX would trigger
    // signed-overflow UB in `count[ii] + 1` before any post-adjustment check
    // could fail closed. On disk count[] is the cumulative C[] array of the
    // reference, so a valid raw entry lies in [0, reference_seq_len]; the shm
    // producer (bwa_shm_compute) applies the same +1 and load_index_from_shm
    // then validates the adjusted [1, ref_seq_len+1] range.
    for (ii = 0; ii < 5; ii++) {
        // A raw count[ii] outside [0, ref_seq_len] (e.g. a disk -1, or a value
        // exceeding the reference length) drives smem.k out of cp_occ on the
        // first backwardExt once adjusted.
        xassert(count[ii] >= 0 && count[ii] <= reference_seq_len,
                "FMI index: count[] out of bounds (corrupt .bwt.2bit.64?)");
        // count[] is the cumulative C[] array, so it must be non-decreasing; a
        // non-monotone pair yields a negative smem.s (interval width) downstream.
        // The +1 is applied uniformly, so raw monotonicity == adjusted monotonicity.
        xassert(ii == 0 || count[ii] >= count[ii - 1],
                "FMI index: count[] not monotonically non-decreasing (corrupt .bwt.2bit.64?)");
    }
    for(ii = 0; ii < 5; ii++)// update read count structure
    {
        count[ii] = count[ii] + 1;
    }
    // Validate count[] (post +1-adjustment, range [1, ref_seq_len+1]) as
    // load_index_from_shm does; a count[a] exceeding the reference length drives
    // smem.k out of cp_occ on the first backwardExt.
    for (ii = 0; ii < 5; ii++) {
        // Lower bound is 1, not 0: the +1 adjustment above is applied before
        // this check (load_index_from_shm instead validates the raw disk value
        // in [0, ref_seq_len] and adjusts after), so a disk count[ii] == -1
        // becomes 0 here and would slip past a >= 0 test -- leaving the disk
        // loader open to an invalid FM interval that the shm producer rejects.
        xassert(count[ii] >= 1 && count[ii] <= reference_seq_len + 1,
                "FMI index: count[] out of bounds (corrupt .bwt.2bit.64?)");
        // count[] is the cumulative C[] array, so it must be non-decreasing; a
        // non-monotone pair yields a negative smem.s (interval width) downstream.
        xassert(ii == 0 || count[ii] >= count[ii - 1],
                "FMI index: count[] not monotonically non-decreasing (corrupt .bwt.2bit.64?)");
    }

    #if SA_COMPRESSION

    int64_t reference_seq_len_ = (reference_seq_len >> sa_compx) + 1;
    int64_t sa_ms_bytes = reference_seq_len_ * sizeof(int8_t);
    int64_t sa_ls_bytes = reference_seq_len_ * sizeof(uint32_t);
    sa_ms_byte = (int8_t *)_mm_malloc(sa_ms_bytes, 64);
    index_alloc += sa_ms_bytes;
    assert_not_null(sa_ms_byte, sa_ms_bytes, index_alloc);
    sa_ls_word = (uint32_t *)_mm_malloc(sa_ls_bytes, 64);
    index_alloc += sa_ls_bytes;
    assert_not_null(sa_ls_word, sa_ls_bytes, index_alloc);
    bwamem_madv_hugepage(sa_ms_byte, sa_ms_bytes);
    bwamem_madv_hugepage(sa_ls_word, sa_ls_bytes);
    fmi_pread_from_stream(cpstream, sa_ms_byte, (size_t)sa_ms_bytes, load_nt);
    fmi_pread_from_stream(cpstream, sa_ls_word, (size_t)sa_ls_bytes, load_nt);

    #else

    int64_t sa_ms_bytes = reference_seq_len * sizeof(int8_t);
    int64_t sa_ls_bytes = reference_seq_len * sizeof(uint32_t);
    sa_ms_byte = (int8_t *)_mm_malloc(sa_ms_bytes, 64);
    index_alloc += sa_ms_bytes;
    assert_not_null(sa_ms_byte, sa_ms_bytes, index_alloc);
    sa_ls_word = (uint32_t *)_mm_malloc(sa_ls_bytes, 64);
    index_alloc += sa_ls_bytes;
    assert_not_null(sa_ls_word, sa_ls_bytes, index_alloc);
    bwamem_madv_hugepage(sa_ms_byte, sa_ms_bytes);
    bwamem_madv_hugepage(sa_ls_word, sa_ls_bytes);
    fmi_pread_from_stream(cpstream, sa_ms_byte, (size_t)sa_ms_bytes, load_nt);
    fmi_pread_from_stream(cpstream, sa_ls_word, (size_t)sa_ls_bytes, load_nt);

    #endif

    sentinel_index = -1;
    #if SA_COMPRESSION
    err_fread_noeof(&sentinel_index, sizeof(int64_t), 1, cpstream);
    xassert(sentinel_index >= 0 && sentinel_index < reference_seq_len,
            "FMI index: sentinel_index out of bounds (corrupt .bwt.2bit.64?)");
    fprintf(stderr, "* sentinel-index: %lld\n", (long long)sentinel_index);
    #endif
    fclose(cpstream);

    int64_t x;
    #if !SA_COMPRESSION
    for(x = 0; x < reference_seq_len; x++)
    {
        // fprintf(stderr, "x: %ld\n", x);
        #if SA_COMPRESSION
        if(get_sa_entry_compressed(x) == 0) {
            sentinel_index = x;
            break;
        }
        #else
        if(get_sa_entry(x) == 0) {
            sentinel_index = x;
            break;
        }
        #endif
    }
    fprintf(stderr, "\nsentinel_index: %lld\n", (long long)x);
    #endif

    fprintf(stderr, "* Count:\n");
    for(x = 0; x < 5; x++)
    {
        fprintf(stderr, "%lld,\t%lld\n", (long long)x, (long long)count[x]);
    }
    fprintf(stderr, "\n");  

    fprintf(stderr, "* Reading other elements of the index from files %s\n",
            ref_file_name);
    /* D3 --meth: BNS only for the seed index (skip the ~1.6 GB seed pac). The
     * seed bns drives the seed->original remap; extension uses meth_orig_pac. */
    bwa_idx_load_ele(ref_file_name, load_pac ? BWA_IDX_ALL : BWA_IDX_BNS);

    fprintf(stderr, "* Done reading Index!!\n");
}

void FMI_search::getSMEMsOnePosOneThread(uint8_t *enc_qdb,
                                         int32_t *query_pos_array,
                                         int32_t *min_intv_array,
                                         int32_t *rid_array,
                                         int32_t numReads,
                                         int32_t batch_size,
                                         const bseq1_t *seq_,
                                         int32_t *query_cum_len_ar,
                                         int32_t max_readlength,
                                         int32_t minSeedLen,
                                         SMEM *matchArray,
                                         int64_t *__numTotalSmem)
{
    int64_t numTotalSmem = *__numTotalSmem;
    // Heap-allocate to avoid stack overflow on long reads (PacBio HiFi /
    // ONT 1 Mbp+); mirrors the lockstep path's max_readlength sizing.
    int64_t prevArray_bytes = (int64_t)max_readlength * sizeof(SMEM);
    SMEM *prevArray = (SMEM *)_mm_malloc((size_t)prevArray_bytes, 64);
    assert_not_null(prevArray, prevArray_bytes, prevArray_bytes);

    // Hoist the FM-index cumulative-count table into a local for the batch (as
    // the bwa-mem2 reference does). count[] is a lifetime-constant member never
    // mutated during seeding, so this is a pure cache of identical values: it
    // trades the per-seed `this->count[..]` member loads for stack reads. Byte-
    // identical to reading the member directly.
    const int64_t counts[5] = {count[0], count[1], count[2], count[3], count[4]};

    uint32_t i;
    // Perform SMEM for original reads
    for(i = 0; i < numReads; i++)
    {
        int x = query_pos_array[i];
        int32_t rid = rid_array[i];
        int next_x = x + 1;

        int readlength = seq_[rid].l_seq;
        int offset = query_cum_len_ar[rid];
        // uint8_t a = enc_qdb[rid * readlength + x];
        uint8_t a = enc_qdb[offset + x];

        if(a < 4)
        {
            SMEM smem;
            smem.rid = rid;
            smem.m = x;
            smem.n = x;
            smem.k = counts[a];
            smem.l = counts[3 - a];
            smem.s = counts[a+1] - counts[a];
            int numPrev = 0;
            
            int j;
            for(j = x + 1; j < readlength; j++)
            {
                // a = enc_qdb[rid * readlength + j];
                a = enc_qdb[offset + j];
                next_x = j + 1;
                if(a < 4)
                {
                    SMEM smem_ = smem;

                    // Forward extension is backward extension with the BWT of reverse complement:
                    // swap k <-> l, backwardExt, swap back. `l` is LIVE here — after the swap it is
                    // the k that drives the next forward step, and the full backwardExt's
                    // l-cumulation is what produces it. So backwardExt_konly (which drops the
                    // backward-phase-dead l-chain) does NOT apply to the forward pass; the full
                    // backwardExt is required. konly is backward-only by construction.
                    smem_.k = smem.l;
                    smem_.l = smem.k;
                    SMEM newSmem_ = backwardExt(smem_, 3 - a);
                    //SMEM newSmem_ = forwardExt(smem_, 3 - a);
                    SMEM newSmem = newSmem_;
                    newSmem.k = newSmem_.l;
                    newSmem.l = newSmem_.k;
                    newSmem.n = j;

                    int32_t s_neq_mask = newSmem.s != smem.s;

                    prevArray[numPrev] = smem;
                    numPrev += s_neq_mask;
                    if(newSmem.s < min_intv_array[i])
                    {
                        next_x = j;
                        break;
                    }
                    smem = newSmem;
#ifdef ENABLE_PREFETCH
                    /* Next iteration swaps k<->l then backwardExt reads
                     * sp = smem.l, ep = smem.l + smem.s (see ls_advance_forward_step
                     * for the full derivation). Prefetching smem.k targeted an
                     * address never touched; ep was never prefetched. Pure hint. */
                    _mm_prefetch((const char *)(&cp_occ[(smem.l) >> CP_SHIFT]), _MM_HINT_T0);
                    _mm_prefetch((const char *)(&cp_occ[(smem.l + smem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
                }
                else
                {
                    break;
                }
            }
            if(smem.s >= min_intv_array[i])
            {

                prevArray[numPrev] = smem;
                numPrev++;
            }

            SMEM *prev;
            prev = prevArray;

            int p;
            for(p = 0; p < (numPrev/2); p++)
            {
                SMEM temp = prev[p];
                prev[p] = prev[numPrev - p - 1];
                prev[numPrev - p - 1] = temp;
            }

            // Backward search.
            //
            // This scalar path deliberately keeps the FULL backwardExt (which
            // also maintains the reverse-complement start `l`), NOT the
            // backwardExt_konly used by the lockstep production path
            // (ls_advance_backward_step). Two reasons, both intentional — do not
            // "unify" this to konly:
            //   1. Oracle independence. This function is the reference the
            //      lockstep path is validated against in
            //      test/smem_lockstep_parity_test.cpp; that test cross-checks
            //      konly's k/s against this full implementation's k/s. Making
            //      both use konly would remove the independent check.
            //   2. It is compiled out of production at the shipping
            //      SMEM_LOCKSTEP_N > 1 (getSMEMsAllPosOneThread dispatches to the
            //      lockstep driver), so the extra l-chain work here is off every
            //      shipped path — there is nothing to optimize.
            // (A SMEM_LOCKSTEP_N <= 1 build would run this in production and pay
            // the full-backwardExt cost, but that config is unsupported and the
            // parity test #errors on it.)
            int cur_j = readlength;
            for(j = x - 1; j >= 0; j--)
            {
                int numCurr = 0;
                int curr_s = -1;
                // a = enc_qdb[rid * readlength + j];
                a = enc_qdb[offset + j];

                if(a > 3)
                {
                    break;
                }
                for(p = 0; p < numPrev; p++)
                {
                    SMEM smem = prev[p];
                    SMEM newSmem = backwardExt(smem, a);
                    newSmem.m = j;

                    if((newSmem.s < min_intv_array[i]) && ((smem.n - smem.m + 1) >= minSeedLen))
                    {
                        cur_j = j;

                        matchArray[numTotalSmem++] = smem;
                        break;
                    }
                    if((newSmem.s >= min_intv_array[i]) && (newSmem.s != curr_s))
                    {
                        curr_s = newSmem.s;
                        prev[numCurr++] = newSmem;
#ifdef ENABLE_PREFETCH
                        _mm_prefetch((const char *)(&cp_occ[(newSmem.k) >> CP_SHIFT]), _MM_HINT_T0);
                        _mm_prefetch((const char *)(&cp_occ[(newSmem.k + newSmem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
                        break;
                    }
                }
                p++;
                for(; p < numPrev; p++)
                {
                    SMEM smem = prev[p];

                    SMEM newSmem = backwardExt(smem, a);
                    newSmem.m = j;


                    if((newSmem.s >= min_intv_array[i]) && (newSmem.s != curr_s))
                    {
                        curr_s = newSmem.s;
                        prev[numCurr++] = newSmem;
#ifdef ENABLE_PREFETCH
                        _mm_prefetch((const char *)(&cp_occ[(newSmem.k) >> CP_SHIFT]), _MM_HINT_T0);
                        _mm_prefetch((const char *)(&cp_occ[(newSmem.k + newSmem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
                    }
                }
                numPrev = numCurr;
                if(numCurr == 0)
                {
                    break;
                }
            }
            if(numPrev != 0)
            {
                SMEM smem = prev[0];
                if(((smem.n - smem.m + 1) >= minSeedLen))
                {

                    matchArray[numTotalSmem++] = smem;
                }
                numPrev = 0;
            }
        }
        query_pos_array[i] = next_x;
    }
    (*__numTotalSmem) = numTotalSmem;
    _mm_free(prevArray);
}

// ===== Lockstep SMEM batching =====
// Per-slot state for the lockstep SMEM walk. One instance per in-flight read
// in the batch. Every field mirrors a per-read local in the scalar
// getSMEMsOnePosOneThread body, so parity with the scalar path follows from
// composing the existing primitives (backwardExt, count[], cp_occ) in the
// same sequence the scalar would.
//
// The prev[] and match_buf[] buffers are heap-allocated by the driver
// (getSMEMsOnePosOneThread_lockstep), sized from the batch's
// max_readlength. No compile-time cap — long reads (PacBio HiFi, ONT)
// fit cleanly.

enum LockstepPhase : uint8_t {
    PH_FWD      = 0,  // forward extension inner loop active
    PH_BWD_INIT = 1,  // between-phases housekeeping pending
    PH_BWD      = 2,  // backward search outer loop active
    PH_DONE     = 3   // slot finished; match_buf ready for flush
};

// Per-thread cache for the lockstep SMEM driver's prev[]/match_buf[]
// buffers. RAII wrapper exists so the buffers are released at thread
// exit instead of leaking until process exit (LSan complaint, issue
// #116). Holding the cache as `static thread_local LockstepSmemCache`
// inside getSMEMsOnePosOneThread_lockstep keeps the per-call reuse
// behavior unchanged; only the cleanup point moves.
struct LockstepSmemCache {
    SMEM  *prev     = nullptr;
    SMEM  *match    = nullptr;
    size_t per_slot = 0;
    int32_t n_slots = 0;   // lockstep width the buffers were sized for
    ~LockstepSmemCache() {
        if (prev  != nullptr) _mm_free(prev);
        if (match != nullptr) _mm_free(match);
    }
};

// LISA trick #4: hybrid SoA. The per-slot "hot" state stays compact
// (~80 bytes) so cross-slot access (e.g. T1 prefetch lookahead against
// slots[(s+N/2)%N]) hits a tight L1-resident array instead of jumping
// 32KB strides across embedded buffers. The bulk arrays (prev[],
// match_buf[]; ~32 KB each) live in separate per-slot allocations,
// referenced by pointer. Accessor syntax (s->prev[i], s->match_buf[i])
// is unchanged from the embedded-array version.
struct FMI_search::BatchSlot {
    // Input identity — copied at init, never mutated thereafter.
    int32_t input_idx;           // index into the caller's input arrays
    int32_t rid;                 // rid_array[input_idx]
    int32_t start_pos;           // query_pos_array[input_idx] (saved for bwd init)
    int32_t min_intv;            // min_intv_array[input_idx]
    int32_t readlength;          // seq_[rid].l_seq
    int32_t offset;              // query_cum_len_ar[rid]

    // Output for query_pos_array[input_idx] write-back at flush time.
    int32_t next_x;

    // Walk state (mirrors scalar locals).
    SMEM smem;                   // current SA interval
    int32_t j;                   // current query position in the active phase's inner loop
    int32_t cur_j;               // backward phase bookkeeping (scalar's cur_j)
    LockstepPhase phase;

    // Per-slot bulk-buffer state.
    int32_t numPrev;
    int32_t match_count;
    bool    ready;

    // Pointers into per-slot bulk arrays (allocated by the driver, sized
    // by max_readlength).
    SMEM    *prev;
    SMEM    *match_buf;
};

void FMI_search::ls_prefetch_cp_occ(const BatchSlot *s)
{
#ifdef ENABLE_PREFETCH
    _mm_prefetch((const char *)(&cp_occ[(s->smem.k) >> CP_SHIFT]), _MM_HINT_T0);
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l) >> CP_SHIFT]), _MM_HINT_T0);
    _mm_prefetch((const char *)(&cp_occ[(s->smem.k + s->smem.s) >> CP_SHIFT]), _MM_HINT_T0);
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l + s->smem.s) >> CP_SHIFT]), _MM_HINT_T0);
#else
    (void)s;
#endif
}

/* T1 (L2) variant — used for cross-slot N/2-step lookahead. The T0 prefetch
 * above lands at "this slot's next step" granularity; T1 here lands at
 * "this slot's step ~N/2 from now". For N=8 that's 4 stepping-passes ahead,
 * giving DRAM-latency-class lookahead when the cp_occ working set spills
 * out of L2/L3.
 *
 * Target the interval the lookahead slot will actually read next, which
 * differs by phase:
 *
 *   PH_FWD: smem is live and the next forward step consumes it, so warm all
 *           four of its checkpoint blocks (k, l, k+s, l+s).
 *
 *   PH_BWD: ls_advance_backward_step runs entirely out of prev[] and NEVER
 *           rewrites smem, so smem here is the STALE leftover from the end of
 *           the forward phase -- warming it prefetches lines the backward walk
 *           will not touch. The next backward step instead calls backwardExt on
 *           each live prev[p], reading cp_occ[prev[p].k] and cp_occ[prev[p].k +
 *           prev[p].s]. Aim the lookahead at those. This keeps the N/2-ahead L2
 *           memory-level parallelism (the part that was actually hiding DRAM
 *           latency) but points it at blocks that are really read. Pure hint,
 *           so output is unchanged. */
void FMI_search::ls_prefetch_cp_occ_t1(const BatchSlot *s)
{
#ifdef ENABLE_PREFETCH
    if (s->phase == PH_BWD) {
        const int32_t np = s->numPrev;
        for (int32_t p = 0; p < np; p++) {
            const SMEM m = s->prev[p];
            _mm_prefetch((const char *)(&cp_occ[(m.k) >> CP_SHIFT]), _MM_HINT_T1);
            _mm_prefetch((const char *)(&cp_occ[(m.k + m.s) >> CP_SHIFT]), _MM_HINT_T1);
        }
    } else {
        _mm_prefetch((const char *)(&cp_occ[(s->smem.k) >> CP_SHIFT]), _MM_HINT_T1);
        _mm_prefetch((const char *)(&cp_occ[(s->smem.l) >> CP_SHIFT]), _MM_HINT_T1);
        _mm_prefetch((const char *)(&cp_occ[(s->smem.k + s->smem.s) >> CP_SHIFT]), _MM_HINT_T1);
        _mm_prefetch((const char *)(&cp_occ[(s->smem.l + s->smem.s) >> CP_SHIFT]), _MM_HINT_T1);
    }
#else
    (void)s;
#endif
}

// Populate a slot from the caller's input arrays at the given input_idx.
// After this call either:
//   phase == PH_FWD  — slot is ready to step the forward-extension inner loop
//   phase == PH_DONE — the first base is non-ACGT; scalar skips this read, so
//                      we match that (zero matches emitted, ready to flush).
void FMI_search::ls_init_slot(BatchSlot *s,
                              int32_t input_idx,
                              const int32_t *query_pos_array,
                              const int32_t *min_intv_array,
                              const int32_t *rid_array,
                              const bseq1_t *seq_,
                              const int32_t *query_cum_len_ar,
                              const uint8_t *enc_qdb)
{
    s->input_idx   = input_idx;
    s->rid         = rid_array[input_idx];
    s->start_pos   = query_pos_array[input_idx];
    s->min_intv    = min_intv_array[input_idx];
    /* backwardExt_konly leaves the new k stale on its s'->0 exit and relies on
     * the caller discarding that seed via `newSmem.s < min_intv` — which only
     * holds when min_intv >= 1 (an s'==0 result must compare < min_intv). Every
     * current source of min_intv satisfies this (mem_collect_smem seeds 1 or
     * p->s+1 >= 2; max_mem_intv is guarded > 0 and never reaches this path), so
     * this guard documents and enforces the precondition rather than papering
     * over a live bug. Once per slot init — off the per-step hot path. */
    xassert(s->min_intv >= 1, "ls_init_slot: min_intv must be >= 1");
    s->readlength  = seq_[s->rid].l_seq;
    s->offset      = query_cum_len_ar[s->rid];
    s->next_x      = s->start_pos + 1;
    s->numPrev     = 0;
    s->match_count = 0;
    s->ready       = false;

    int32_t x = s->start_pos;
    uint8_t a = enc_qdb[s->offset + x];
    if (a < 4) {
        s->smem.rid = s->rid;
        s->smem.m   = x;
        s->smem.n   = x;
        s->smem.k   = count[a];
        s->smem.l   = count[3 - a];
        s->smem.s   = count[a + 1] - count[a];
        s->j        = x + 1;
        s->phase    = PH_FWD;
    } else {
        // Scalar path skips the whole read when the first base is N.
        // Match that by going straight to DONE with zero matches.
        s->phase = PH_DONE;
        s->ready = true;
    }
}
// Advance one slot through one step of forward extension.
// Mirrors the inner j-loop body of the scalar getSMEMsOnePosOneThread
// (src/FMI_search.cpp — the for(j = x+1; j < readlength; j++) loop).
// On the step that exits forward-ext (end-of-read, non-ACGT at j, or
// s < min_intv), transitions phase to PH_BWD_INIT.
void FMI_search::ls_advance_forward_step(BatchSlot *s, const uint8_t *enc_qdb)
{
    if (s->j >= s->readlength) {
        // Ran off the end of the read; keep the still-valid smem if any.
        if (s->smem.s >= s->min_intv) {
            s->prev[s->numPrev++] = s->smem;
        }
        s->phase = PH_BWD_INIT;
        return;
    }

    uint8_t a = enc_qdb[s->offset + s->j];
    s->next_x = s->j + 1;
    if (a >= 4) {
        // Non-ACGT base terminates forward ext.
        if (s->smem.s >= s->min_intv) {
            s->prev[s->numPrev++] = s->smem;
        }
        s->phase = PH_BWD_INIT;
        return;
    }

    SMEM smem_ = s->smem;
    smem_.k = s->smem.l;
    smem_.l = s->smem.k;
    SMEM newSmem_ = backwardExt(smem_, 3 - a);
    SMEM newSmem  = newSmem_;
    newSmem.k = newSmem_.l;
    newSmem.l = newSmem_.k;
    newSmem.n = s->j;

    int32_t s_neq_mask = (newSmem.s != s->smem.s);
    s->prev[s->numPrev] = s->smem;
    s->numPrev += s_neq_mask;

    if (newSmem.s < s->min_intv) {
        s->next_x = s->j;
        s->phase = PH_BWD_INIT;
        return;
    }

    s->smem = newSmem;
    s->j++;
#ifdef ENABLE_PREFETCH
    /* The NEXT forward step swaps k<->l (see smem_ above) before calling
     * backwardExt, which reads sp = smem.k and ep = smem.k + smem.s. After the
     * swap that is OUR smem.l and smem.l + smem.s -- so those are the two lines
     * to prefetch. The old code prefetched smem.k (an address the next step
     * never touches) and smem.l, leaving the ep line to always miss cold.
     * Same targets as bsd_prefetch_cp_occ(), which fixed this for round 3 in
     * #242; the fix was simply never propagated here. Prefetch is a pure hint,
     * so this cannot change output. */
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l) >> CP_SHIFT]), _MM_HINT_T0);
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l + s->smem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
}

// Between-phases housekeeping: reverse prev[] and set up backward-phase
// cursors. After this call the slot is either PH_BWD (ready for
// ls_advance_backward_step) or PH_DONE (nothing to do — backward outer
// loop would terminate on the first step, so we short-circuit to the
// final prev[0] emit and done).
void FMI_search::ls_prepare_backward(BatchSlot *s)
{
    // Reverse prev[] in place (matches scalar behavior before backward loop).
    for (int p = 0; p < (s->numPrev / 2); p++) {
        SMEM tmp = s->prev[p];
        s->prev[p] = s->prev[s->numPrev - p - 1];
        s->prev[s->numPrev - p - 1] = tmp;
    }
    s->cur_j = s->readlength;
    s->j = s->start_pos - 1;  // first position for the backward outer loop

    if (s->numPrev == 0) {
        // Nothing to emit; scalar's final `if (numPrev != 0)` block is a no-op.
        s->phase = PH_DONE;
        s->ready = true;
        return;
    }
    if (s->j < 0) {
        // Backward outer loop cannot execute (start_pos == 0). Transition to
        // PH_BWD so the first ls_advance_backward_step call sees j < 0, jumps
        // to DONE, and runs the final prev[0] emit with the correct minSeedLen.
        s->phase = PH_BWD;
        return;
    }
    s->phase = PH_BWD;
}

// Advance one slot through ONE outer-j iteration of backward search.
// Mirrors the body of the scalar `for (j = x-1; j >= 0; j--)` outer loop
// in getSMEMsOnePosOneThread. On terminating conditions (numCurr == 0,
// j < 0 after decrement, or non-ACGT at j), runs the scalar's final
// prev[0] emit and transitions to PH_DONE.
void FMI_search::ls_advance_backward_step(BatchSlot *s,
                                          const uint8_t *enc_qdb,
                                          int32_t minSeedLen)
{
    if (s->j < 0) goto DONE;

    {
        int32_t numCurr = 0;
        int32_t curr_s = -1;
        uint8_t a = enc_qdb[s->offset + s->j];
        if (a > 3) goto DONE;

        /* numPrev==1 straight-line path: the two-loop machinery and the curr_s
         * dedup collapse to a single element. Byte-identical. This is a hand-kept
         * specialization of the two general loops below (the emit predicate, the
         * prefetch pair, and the numCurr/j-- bookkeeping are duplicated here) — a
         * future edit to that logic in the general loops MUST be mirrored into
         * this block, or numPrev==1 reads (unique matches — the common case this
         * path accelerates) silently diverge. The lockstep parity test guards
         * this (see Case 13, which drives real interior-start backward walks). */
        if (s->numPrev == 1) {
            SMEM smem = s->prev[0];
            SMEM newSmem = backwardExt_konly(smem, a);
            newSmem.m = s->j;
            if (newSmem.s < s->min_intv) {
                if ((smem.n - smem.m + 1) >= minSeedLen) {
                    s->cur_j = s->j;
                    s->match_buf[s->match_count++] = smem;
                }
                /* numCurr stays 0 */
            } else {
                s->prev[0] = newSmem;
#ifdef ENABLE_PREFETCH
                _mm_prefetch((const char *)(&cp_occ[(newSmem.k) >> CP_SHIFT]), _MM_HINT_T0);
                _mm_prefetch((const char *)(&cp_occ[(newSmem.k + newSmem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
                numCurr = 1;
            }
            s->numPrev = numCurr;
            s->j--;
            if (numCurr == 0) goto DONE;
            return;
        }

        int p;
        for (p = 0; p < s->numPrev; p++) {
            SMEM smem = s->prev[p];
            SMEM newSmem = backwardExt_konly(smem, a);
            newSmem.m = s->j;
            if ((newSmem.s < s->min_intv) && ((smem.n - smem.m + 1) >= minSeedLen)) {
                s->cur_j = s->j;
                s->match_buf[s->match_count++] = smem;
                break;
            }
            if ((newSmem.s >= s->min_intv) && (newSmem.s != curr_s)) {
                curr_s = newSmem.s;
                s->prev[numCurr++] = newSmem;
#ifdef ENABLE_PREFETCH
                _mm_prefetch((const char *)(&cp_occ[(newSmem.k) >> CP_SHIFT]), _MM_HINT_T0);
                _mm_prefetch((const char *)(&cp_occ[(newSmem.k + newSmem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
                break;
            }
        }
        p++;
        for (; p < s->numPrev; p++) {
            SMEM smem = s->prev[p];
            SMEM newSmem = backwardExt_konly(smem, a);
            newSmem.m = s->j;
            if ((newSmem.s >= s->min_intv) && (newSmem.s != curr_s)) {
                curr_s = newSmem.s;
                s->prev[numCurr++] = newSmem;
#ifdef ENABLE_PREFETCH
                _mm_prefetch((const char *)(&cp_occ[(newSmem.k) >> CP_SHIFT]), _MM_HINT_T0);
                _mm_prefetch((const char *)(&cp_occ[(newSmem.k + newSmem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif
            }
        }
        s->numPrev = numCurr;
        s->j--;                // advance for the next outer step
        if (numCurr == 0) goto DONE;
        return;                // stay in PH_BWD; next call continues
    }

DONE:
    // Scalar end-of-function final prev[0] emit (lines ~650-659 of scalar).
    if (s->numPrev != 0) {
        SMEM smem = s->prev[0];
        if ((smem.n - smem.m + 1) >= minSeedLen) {
            s->match_buf[s->match_count++] = smem;
        }
        s->numPrev = 0;
    }
    s->phase = PH_DONE;
    s->ready = true;
}
// ===== End lockstep SMEM batching =====

// ===== Lockstep bwtSeed batching =====
// Per-slot state for the forward-only bwtSeed walk. Mirrors the locals of
// the scalar bwtSeedStrategyAllPosOneThread inner-j loop, plus an outer-x
// cursor. Each slot owns the entire scalar `while (x < readlength)` outer
// loop — outer transitions are handled inline in bsd_advance_step (Heng
// review action: don't burn a stepping-pass on outer-only housekeeping
// when the inner-j loop is shallow, which is exactly bwtSeed's regime).
//
// match_buf[] is a per-slot accumulator flushed in input order; sized
// by the driver from the batch's max_readlength.
enum BwtSeedPhase : uint8_t {
    BSD_FWD  = 0,  // in active inner-j extension (s->j is next j to step;
                   // when s->j < 0 the next step does outer seek first)
    BSD_DONE = 1,  // slot retired; match_buf ready for in-order flush
};

// Per-thread cache for the lockstep bwtSeed driver's match_buf[]. RAII
// wrapper parallels LockstepSmemCache above; released at thread exit so
// LSan stays quiet (issue #116 family). Tracked separately from the SMEM
// cache because the access pattern is different (single buffer, not
// prev/match pair) and the watermark can move independently.
struct LockstepBwtSeedCache {
    SMEM  *match    = nullptr;
    size_t per_slot = 0;
    ~LockstepBwtSeedCache() {
        if (match != nullptr) _mm_free(match);
    }
};

struct FMI_search::BwtSeedSlot {
    int32_t input_idx;     // index into the caller's input arrays; doubles as smem.rid
    int32_t max_intv;      // max_intv_array[input_idx] cached at init
    int32_t readlength;    // seq_[input_idx].l_seq
    int32_t offset;        // query_cum_len_ar[input_idx]

    SMEM    smem;          // current SA interval; rid/m fixed at outer-x init
    int32_t x;             // outer cursor: position of current smem's seed base
    int32_t j;             // inner cursor: next j to step; j < 0 = need outer seek

    BwtSeedPhase phase;
    int32_t match_count;
    bool    ready;

    SMEM   *match_buf;     // -> thread-local cache slice, sized = max_readlength
};

// Same-slot T0 prefetch for the next inner-j step. The bwtSeed walk is
// forward-only: bsd_advance_step swaps k<->l and calls backwardExt on the
// swapped interval, so the next occ read lands in cp_occ at (smem.l >> CP_SHIFT)
// and ((smem.l + smem.s) >> CP_SHIFT) — i.e. sp and ep of the swapped interval,
// NOT smem.k. Prefetching smem.k (the un-swapped field) targets the wrong lines,
// so every continuation-step prefetch was a no-op and each step paid full cp_occ
// latency. smem.s is unchanged by the k<->l swap.
void FMI_search::bsd_prefetch_cp_occ(const BwtSeedSlot *s)
{
#ifdef ENABLE_PREFETCH
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l) >> CP_SHIFT]), _MM_HINT_T0);
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l + s->smem.s) >> CP_SHIFT]), _MM_HINT_T0);
#else
    (void)s;
#endif
}

// Cross-slot T1 (L2) prefetch — same two-line target as T0 (the swapped
// interval at smem.l; see bsd_prefetch_cp_occ) but at L2 hint, used in the
// driver's (s + N/2) % N lookahead to cover DRAM-class latency when cp_occ
// spills out of L3.
void FMI_search::bsd_prefetch_cp_occ_t1(const BwtSeedSlot *s)
{
#ifdef ENABLE_PREFETCH
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l) >> CP_SHIFT]), _MM_HINT_T1);
    _mm_prefetch((const char *)(&cp_occ[(s->smem.l + s->smem.s) >> CP_SHIFT]), _MM_HINT_T1);
#else
    (void)s;
#endif
}

// Initialize a fresh slot for the given input read. Scans forward from
// x=0 for the first ACGT base (matching the scalar's outer while-loop
// skipping non-ACGT) and either enters BSD_FWD with smem seeded, or jumps
// straight to BSD_DONE when the read is entirely non-ACGT.
void FMI_search::bsd_init_slot(BwtSeedSlot *s,
                               int32_t input_idx,
                               const int32_t *max_intv_array,
                               const bseq1_t *seq_,
                               const int32_t *query_cum_len_ar,
                               const uint8_t *enc_qdb)
{
    s->input_idx   = input_idx;
    s->max_intv    = max_intv_array[input_idx];
    s->readlength  = seq_[input_idx].l_seq;
    s->offset      = query_cum_len_ar[input_idx];
    s->match_count = 0;
    s->ready       = false;

    int32_t x = 0;
    while (x < s->readlength && enc_qdb[s->offset + x] >= 4) x++;
    s->x = x;
    if (x >= s->readlength) {
        s->phase = BSD_DONE;
        s->ready = true;
        return;
    }
    uint8_t a = enc_qdb[s->offset + x];
    s->smem.rid = (uint32_t)input_idx;
    s->smem.m   = x;
    s->smem.n   = x;
    s->smem.k   = count[a];
    s->smem.l   = count[3 - a];
    s->smem.s   = count[a + 1] - count[a];
    s->j        = x + 1;
    s->phase    = BSD_FWD;
}

// Advance one slot by one unit of work. Always does at least one cp_occ
// access per call when the slot is active, so the lockstep pipeline keeps
// uniform per-pass memory parallelism (Heng action 2a — collapse outer
// transition into the step function rather than burning a stepping-pass
// on housekeeping).
//
// The semantics are identical to the scalar bwtSeedStrategyAllPosOneThread
// inner-j body + outer-x advancement:
//   for(j = x+1; j < readlength; j++) {
//       next_x = j+1; a = enc_qdb[offset + j];
//       if (a < 4) {
//           SMEM newSmem = forward-ext(smem, 3-a); newSmem.n = j;
//           smem = newSmem;
//           if (s < max_intv && (n - m + 1) >= minSeedLen) {
//               if (s > 0) emit smem;
//               break;
//           }
//       } else { break; }
//   }
//   x = next_x;
void FMI_search::bsd_advance_step(BwtSeedSlot *s,
                                  const uint8_t *enc_qdb,
                                  int32_t minSeedLen)
{
    // Outer seek: if j < 0 the slot just broke out of an inner-j extension
    // and needs to (re-)init smem at the next ACGT x. Spin through non-ACGT
    // bases inline (scalar does the same — at most O(N_run) here, and N runs
    // are rare in Illumina data). Falls through to one inner-j step.
    if (s->j < 0) {
        while (s->x < s->readlength && enc_qdb[s->offset + s->x] >= 4) s->x++;
        if (s->x >= s->readlength) {
            s->phase = BSD_DONE;
            s->ready = true;
            return;
        }
        uint8_t a = enc_qdb[s->offset + s->x];
        s->smem.rid = (uint32_t)s->input_idx;
        s->smem.m   = s->x;
        s->smem.n   = s->x;
        s->smem.k   = count[a];
        s->smem.l   = count[3 - a];
        s->smem.s   = count[a + 1] - count[a];
        s->j        = s->x + 1;
        // Fall through to inner-j step.
    }

    // Inner-j step.
    if (s->j >= s->readlength) {
        // Scalar: inner for-loop exhausted; next_x stays at last iteration's
        // j+1 = readlength. Setting x = readlength and j = -1 signals the
        // next stepping pass to retire (the seek above will hit x >= L).
        s->x = s->readlength;
        s->j = -1;
        return;
    }

    uint8_t a = enc_qdb[s->offset + s->j];
    if (a >= 4) {
        // Scalar: non-ACGT in inner-j sets next_x = j+1 and breaks.
        s->x = s->j + 1;
        s->j = -1;
        return;
    }

    // Forward extension = backward extension on the BWT of the reverse
    // complement: swap k <-> l around backwardExt(., 3-a) and swap back.
    SMEM smem_ = s->smem;
    smem_.k = s->smem.l;
    smem_.l = s->smem.k;
    SMEM newSmem_ = backwardExt(smem_, 3 - a);
    SMEM newSmem  = newSmem_;
    newSmem.k = newSmem_.l;
    newSmem.l = newSmem_.k;
    newSmem.n = s->j;

    if ((newSmem.s < s->max_intv) && ((newSmem.n - newSmem.m + 1) >= minSeedLen)) {
        if (newSmem.s > 0) {
            s->match_buf[s->match_count++] = newSmem;
        }
        s->x = s->j + 1;
        s->j = -1;
        return;
    }

    s->smem = newSmem;
    s->j++;
    bsd_prefetch_cp_occ(s);
}
// ===== End lockstep bwtSeed batching =====

void FMI_search::getSMEMsAllPosOneThread(uint8_t *enc_qdb,
                                         int32_t *min_intv_array,
                                         int32_t *rid_array,
                                         int32_t numReads,
                                         int32_t batch_size,
                                         const bseq1_t *seq_,
                                         int32_t *query_cum_len_ar,
                                         int32_t max_readlength,
                                         int32_t minSeedLen,
                                         SMEM *matchArray,
                                         int64_t *__numTotalSmem)
{
    size_t query_pos_bytes = (size_t)numReads * sizeof(int32_t);
    int32_t *query_pos_array = (int32_t *)_mm_malloc(query_pos_bytes, 64);
    assert_not_null(query_pos_array, query_pos_bytes, query_pos_bytes);

    int32_t i;
    for(i = 0; i < numReads; i++)
        query_pos_array[i] = 0;

    int32_t numActive = numReads;
    (*__numTotalSmem) = 0;

    do
    {
        int32_t head = 0;
        int32_t tail = 0;
        for(head = 0; head < numActive; head++)
        {
            int readlength = seq_[rid_array[head]].l_seq;
            if(query_pos_array[head] < readlength)
            {
                rid_array[tail] = rid_array[head];
                query_pos_array[tail] = query_pos_array[head];
                min_intv_array[tail] = min_intv_array[head];
                tail++;
            }
        }
#if SMEM_LOCKSTEP_N > 1
        getSMEMsOnePosOneThread_lockstep(enc_qdb,
                                         query_pos_array,
                                         min_intv_array,
                                         rid_array,
                                         tail,
                                         batch_size,
                                         seq_,
                                         query_cum_len_ar,
                                         max_readlength,
                                         minSeedLen,
                                         matchArray,
                                         __numTotalSmem);
#else
        getSMEMsOnePosOneThread(enc_qdb,
                                query_pos_array,
                                min_intv_array,
                                rid_array,
                                tail,
                                batch_size,
                                seq_,
                                query_cum_len_ar,
                                max_readlength,
                                minSeedLen,
                                matchArray,
                                __numTotalSmem);
#endif
        numActive = tail;
    } while(numActive > 0);

    _mm_free(query_pos_array);
}

void FMI_search::getSMEMsOnePosOneThread_lockstep(uint8_t *enc_qdb,
                                                   int32_t *query_pos_array,
                                                   int32_t *min_intv_array,
                                                   int32_t *rid_array,
                                                   int32_t numReads,
                                                   int32_t batch_size,
                                                   const bseq1_t *seq_,
                                                   int32_t *query_cum_len_ar,
                                                   int32_t max_readlength,
                                                   int32_t minSeedLen,
                                                   SMEM *matchArray,
                                                   int64_t *__numTotalSmem)
{
    (void)batch_size;

    if (numReads == 0) return;

    const int32_t N = g_smem_lockstep_n;
    // LISA trick #4: hybrid SoA layout. `slots[]` holds only the small hot
    // state (~80 B per slot, full array fits in 1-2 cache lines for N=8).
    // Bulk per-slot buffers (prev/match_buf) live separately and are reused
    // across calls via thread_local caches sized from the batch's
    // max_readlength so reads of any length fit (see issue #44 / PR #55).
    //
    // The outer driver getSMEMsAllPosOneThread runs this in a do/while
    // loop, so allocating per-call (~2*N*max_readlength*sizeof(SMEM) ≈
    // 384 KB at N=8, max_readlength=1500) imposed measurable allocator
    // pressure. Cache per-thread (so OMP workers don't share) and grow
    // monotonically — max_readlength is bounded by the driver batch and
    // increases rarely in practice.
    //
    // TODO(memory): the cache only grows. For mixed-length workloads (e.g.
    // a single ONT-class read with max_readlength≈1e6 mid-stream — sized
    // to ~640 MB per thread, ~10 GB across 16 OMP workers — followed by
    // short reads), the high-water mark is held until thread exit.
    // Acceptable for the smoke1M Illumina PE150 benchmark (max_readlength≈
    // 150 → ~38 KB/thread). Revisit if a streaming aligner or long-running
    // service pipeline appears: gate the realloc on a configured upper
    // bound (e.g. MAX_SMEM_PER_SLOT) or shrink when cache.per_slot greatly
    // exceeds the current batch.
    /* Fixed-size on the stack (compile-time MAX); only the first N are used,
     * where N = g_smem_lockstep_n is the startup-probed runtime width. */
    BatchSlot slots[SMEM_LOCKSTEP_N_MAX] = {};
    static thread_local LockstepSmemCache cache;
    const size_t per_slot_smems = (size_t)max_readlength;
    if (per_slot_smems > cache.per_slot || N > cache.n_slots) {
        if (cache.prev  != nullptr) _mm_free(cache.prev);
        if (cache.match != nullptr) _mm_free(cache.match);
        // Grow on EITHER dimension: g_smem_lockstep_n (N) can rise after first
        // use (BWA3_SMEM_LOCKSTEP_N / the MLP probe), and sizing only on per_slot
        // would leave slots[N-1] pointing past the allocation.
        const size_t alloc_slots = (size_t)(N > cache.n_slots ? N : cache.n_slots);
        const size_t alloc_per   = per_slot_smems > cache.per_slot ? per_slot_smems : cache.per_slot;
        const size_t total_slot_bytes = alloc_slots * alloc_per * sizeof(SMEM);
        cache.prev  = (SMEM *)_mm_malloc(total_slot_bytes, 64);
        assert_not_null(cache.prev, total_slot_bytes, total_slot_bytes);
        cache.match = (SMEM *)_mm_malloc(total_slot_bytes, 64);
        assert_not_null(cache.match, total_slot_bytes, (size_t)2 * total_slot_bytes);
        cache.per_slot = alloc_per;
        cache.n_slots  = (int32_t)alloc_slots;
    }
    for (int32_t s = 0; s < N; s++) {
        slots[s].prev      = cache.prev  + (size_t)s * cache.per_slot;
        slots[s].match_buf = cache.match + (size_t)s * cache.per_slot;
    }

    // Seed the first min(N, numReads) slots.
    int32_t initial = (numReads < N) ? numReads : N;
    for (int32_t s = 0; s < initial; s++) {
        ls_init_slot(&slots[s], s, query_pos_array, min_intv_array,
                     rid_array, seq_, query_cum_len_ar, enc_qdb);
        if (slots[s].phase == PH_FWD) ls_prefetch_cp_occ(&slots[s]);
    }
    // Any unused slots (numReads < N) are marked DONE with invalid input_idx
    // so the stepping pass skips them and the flush pass ignores them.
    for (int32_t s = initial; s < N; s++) {
        slots[s].phase = PH_DONE;
        slots[s].ready = false;
        slots[s].input_idx = -1;
    }

    int32_t next_input   = initial;
    int32_t flush_cursor = 0;
    int64_t numTotalSmem = *__numTotalSmem;

    while (flush_cursor < numReads) {
        // --- Stepping pass: advance each non-DONE slot by one phase-step. ---
        for (int32_t s = 0; s < N; s++) {
            switch (slots[s].phase) {
                case PH_FWD:
                    ls_advance_forward_step(&slots[s], enc_qdb);
                    // If forward just transitioned to PH_BWD_INIT, run the
                    // between-phases housekeeping immediately. prepare_backward
                    // may transition straight to PH_DONE (e.g. empty prev[]).
                    if (slots[s].phase == PH_BWD_INIT) {
                        ls_prepare_backward(&slots[s]);
                    }
                    break;
                case PH_BWD_INIT:
                    ls_prepare_backward(&slots[s]);
                    break;
                case PH_BWD:
                    ls_advance_backward_step(&slots[s], enc_qdb, minSeedLen);
                    break;
                case PH_DONE:
                    break;
            }
            // LISA trick #5: cross-slot T1 (L2) prefetch with N/2-step lookahead.
            // The same-slot T0 prefetch issued inside ls_advance_*_step covers
            // the next single-step access; this T1 prefetch on slot[s+N/2] keeps
            // a copy in L2 for that slot's access ~N/2 stepping-passes from now,
            // hiding DRAM-class latency when cp_occ entries spill out of L3.
            // N is runtime (g_smem_lockstep_n), so `% N` is a real idiv every
            // step. s < N and N/2 < N => s + N/2 < 2N, so one compare + subtract
            // replaces the division. Byte-identical.
            int32_t s_la = s + (N >> 1);
            if (s_la >= N) s_la -= N;
            if (slots[s_la].phase == PH_FWD || slots[s_la].phase == PH_BWD) {
                ls_prefetch_cp_occ_t1(&slots[s_la]);
            }
        }

        // --- Flush pass: in-order emit + slot recycle. ---
        bool progress = true;
        while (progress) {
            progress = false;
            for (int32_t s = 0; s < N; s++) {
                if (slots[s].phase == PH_DONE &&
                    slots[s].ready &&
                    slots[s].input_idx == flush_cursor) {
                    for (int32_t m = 0; m < slots[s].match_count; m++) {
                        matchArray[numTotalSmem++] = slots[s].match_buf[m];
                    }
                    query_pos_array[slots[s].input_idx] = slots[s].next_x;
                    flush_cursor++;

                    if (next_input < numReads) {
                        ls_init_slot(&slots[s], next_input,
                                     query_pos_array, min_intv_array,
                                     rid_array, seq_, query_cum_len_ar, enc_qdb);
                        if (slots[s].phase == PH_FWD) ls_prefetch_cp_occ(&slots[s]);
                        next_input++;
                    } else {
                        slots[s].input_idx = -1;  // retired
                        slots[s].ready = false;
                    }
                    progress = true;
                }
            }
        }
    }

    *__numTotalSmem = numTotalSmem;
    /* prev/match buffers are owned by the thread_local cache above; intentionally
     * not freed here so the next call (same thread) reuses them without a
     * round-trip through the allocator. They are released by
     * LockstepSmemCache's destructor at thread exit. */
}

int64_t FMI_search::bwtSeedStrategyAllPosOneThread(uint8_t *enc_qdb,
                                                   int32_t *max_intv_array,
                                                   int32_t numReads,
                                                   const bseq1_t *seq_,
                                                   int32_t *query_cum_len_ar,
                                                   int32_t minSeedLen,
                                                   SMEM *matchArray)
{
    int32_t i;

    int64_t numTotalSeed = 0;

    for(i = 0; i < numReads; i++)
    {
        int readlength = seq_[i].l_seq;
        int32_t x = 0;
        while(x < readlength)
        {
            int next_x = x + 1;

            // Forward search
            SMEM smem;
            smem.rid = i;
            smem.m = x;
            smem.n = x;
            
            int offset = query_cum_len_ar[i];
            uint8_t a = enc_qdb[offset + x];
            // uint8_t a = enc_qdb[i * readlength + x];

            if(a < 4)
            {
                smem.k = count[a];
                smem.l = count[3 - a];
                smem.s = count[a+1] - count[a];


                int j;
                for(j = x + 1; j < readlength; j++)
                {
                    next_x = j + 1;
                    // a = enc_qdb[i * readlength + j];
                    a = enc_qdb[offset + j];
                    if(a < 4)
                    {
                        SMEM smem_ = smem;

                        // Forward extension is backward extension with the BWT of reverse complement
                        smem_.k = smem.l;
                        smem_.l = smem.k;
                        SMEM newSmem_ = backwardExt(smem_, 3 - a);
                        //SMEM smem = backwardExt(smem, 3 - a);
                        //smem.n = j;
                        SMEM newSmem = newSmem_;
                        newSmem.k = newSmem_.l;
                        newSmem.l = newSmem_.k;
                        newSmem.n = j;
                        smem = newSmem;
#ifdef ENABLE_PREFETCH
                        /* Same correction as the lockstep path: the next step reads
                         * sp = smem.l, ep = smem.l + smem.s after the k<->l swap.
                         * This is the x86 default round-3 path (the lockstep variant
                         * is arm64-gated), so it was missing the #242 fix entirely. */
                        _mm_prefetch((const char *)(&cp_occ[(smem.l) >> CP_SHIFT]), _MM_HINT_T0);
                        _mm_prefetch((const char *)(&cp_occ[(smem.l + smem.s) >> CP_SHIFT]), _MM_HINT_T0);
#endif


                        if((smem.s < max_intv_array[i]) && ((smem.n - smem.m + 1) >= minSeedLen))
                        {

                            if(smem.s > 0)
                            {
                                matchArray[numTotalSeed++] = smem;
                            }
                            break;
                        }
                    }
                    else
                    {

                        break;
                    }
                }

            }
            x = next_x;
        }
    }
    return numTotalSeed;
}

// Lockstep-batched bwtSeed driver. Drop-in for bwtSeedStrategyAllPosOneThread
// — same input contract, same emission order (reads in input order; within
// a read, outer-x ascending; at most one SMEM per outer x). The interleaving
// runs BWTSEED_LOCKSTEP_N reads' forward-extension walks together so the
// independent cp_occ cache-line misses issue concurrently into Zen 4's
// load queue rather than serializing on per-read load latency.
//
// Structurally parallel to getSMEMsOnePosOneThread_lockstep above; see that
// function's comments for the prefetch model (same-slot T0 inside the step
// function + cross-slot T1 at (s + N/2) % N in the driver) and the hybrid
// SoA per-slot/bulk-buffer layout rationale.
int64_t FMI_search::bwtSeedStrategyAllPosOneThread_lockstep(uint8_t *enc_qdb,
                                                            int32_t *max_intv_array,
                                                            int32_t numReads,
                                                            const bseq1_t *seq_,
                                                            int32_t *query_cum_len_ar,
                                                            int32_t minSeedLen,
                                                            SMEM *matchArray,
                                                            int32_t max_readlength)
{
    if (numReads <= 0) return 0;

    const int32_t N = BWTSEED_LOCKSTEP_N;

    // Per-thread cache for match_buf[] slices. One pointer (no prev[]
    // needed — bwtSeed has only a forward pass). Sized from max_readlength;
    // scalar's worst-case emit is one SMEM per x ∈ [0, readlength) so the
    // bound holds. Grown monotonically across calls.
    BwtSeedSlot slots[BWTSEED_LOCKSTEP_N] = {};
    static thread_local LockstepBwtSeedCache cache;
    const size_t per_slot_smems = (size_t)max_readlength;
    if (per_slot_smems > cache.per_slot) {
        if (cache.match != nullptr) _mm_free(cache.match);
        const size_t total_bytes = (size_t)N * per_slot_smems * sizeof(SMEM);
        cache.match = (SMEM *)_mm_malloc(total_bytes, 64);
        assert_not_null(cache.match, total_bytes, total_bytes);
        cache.per_slot = per_slot_smems;
    }
    for (int32_t s = 0; s < N; s++) {
        slots[s].match_buf = cache.match + (size_t)s * cache.per_slot;
    }

    // Seed the first min(N, numReads) slots.
    const int32_t initial = (numReads < N) ? numReads : N;
    for (int32_t s = 0; s < initial; s++) {
        bsd_init_slot(&slots[s], s, max_intv_array, seq_, query_cum_len_ar, enc_qdb);
        if (slots[s].phase == BSD_FWD) bsd_prefetch_cp_occ(&slots[s]);
    }
    // Unused slots get input_idx = -1 so the flush pass skips them.
    for (int32_t s = initial; s < N; s++) {
        slots[s].phase     = BSD_DONE;
        slots[s].ready     = false;
        slots[s].input_idx = -1;
    }

    int32_t next_input   = initial;
    int32_t flush_cursor = 0;
    int64_t numTotalSeed = 0;

    while (flush_cursor < numReads) {
        // Stepping pass: advance each active slot by one unit of work.
        for (int32_t s = 0; s < N; s++) {
            if (slots[s].phase == BSD_FWD) {
                bsd_advance_step(&slots[s], enc_qdb, minSeedLen);
            }
            // Cross-slot T1 lookahead — same pattern as the SMEM lockstep's
            // (s + N/2) % N L2 prefetch.
            const int32_t s_la = (s + (N / 2)) % N;
            if (slots[s_la].phase == BSD_FWD && slots[s_la].j >= 0) {
                bsd_prefetch_cp_occ_t1(&slots[s_la]);
            }
        }

        // Flush pass: emit retired slots in input order; recycle into the
        // next pending read. Loop until no progress so a chain of newly
        // retired slots at the cursor flushes in one pass.
        bool progress = true;
        while (progress) {
            progress = false;
            for (int32_t s = 0; s < N; s++) {
                if (slots[s].phase == BSD_DONE &&
                    slots[s].ready &&
                    slots[s].input_idx == flush_cursor) {
                    for (int32_t m = 0; m < slots[s].match_count; m++) {
                        matchArray[numTotalSeed++] = slots[s].match_buf[m];
                    }
                    flush_cursor++;

                    if (next_input < numReads) {
                        bsd_init_slot(&slots[s], next_input,
                                      max_intv_array, seq_, query_cum_len_ar, enc_qdb);
                        if (slots[s].phase == BSD_FWD) bsd_prefetch_cp_occ(&slots[s]);
                        next_input++;
                    } else {
                        slots[s].input_idx = -1;
                        slots[s].ready     = false;
                    }
                    progress = true;
                }
            }
        }
    }

    return numTotalSeed;
}


void FMI_search::sortSMEMs(SMEM *matchArray,
        int64_t numTotalSmem[],
        int32_t numReads,
        int32_t readlength,
        int nthreads,
        SmemSortScratch &scratch)
{
    /* The only property the caller needs from this sort is that every SMEM of a
     * given read (rid) is contiguous and the reads appear in ascending rid
     * order: mem_collect_smem then walks each rid block and re-sorts it with
     * ks_introsort(mem_intv1) keyed purely on (m, n). The old qsort's secondary
     * (m asc, n desc) ordering was therefore thrown away — and SMEMs that tie on
     * (rid, m, n) are byte-identical (same read span => same SA interval), so the
     * unstable re-sort's handling of them cannot depend on their incoming order.
     *
     * rid is a dense index in [0, numReads), so a counting sort by rid replaces
     * the O(n log n) function-pointer qsort with an O(n + rid_range) pass.
     *
     * The count/offset array (`cnt`) and the stable-scatter buffer (`tmp`) live
     * in caller-owned scratch that is allocated once and reused across batches
     * (audit SEED-15), replacing the old per-batch calloc/_mm_malloc + free.
     * FMI_search is shared across worker threads, so no member scratch may be
     * used here; the scratch instead comes from the caller's per-thread
     * mem_cache slot (mmc->smem_sort_scratch[tid]) and is never shared. */
    int tid;
    int32_t perThreadQuota = (numReads + (nthreads - 1)) / nthreads;
    for(tid = 0; tid < nthreads; tid++)
    {
        int64_t smem_count = numTotalSmem[tid];   /* scalar SMEM count; not the class `count[5]` array */
        if (smem_count <= 1) continue;   /* 0 or 1 element is already sorted */

        int32_t first = tid * perThreadQuota;
        SMEM *myMatchArray = matchArray + (int64_t)first * readlength;

        /* rid range actually present in this block (dense, but a per-thread
         * block only holds its own reads, so offset by the observed minimum). */
        uint32_t minRid = myMatchArray[0].rid, maxRid = myMatchArray[0].rid;
        for (int64_t i = 1; i < smem_count; i++) {
            uint32_t r = myMatchArray[i].rid;
            if (r < minRid) minRid = r;
            if (r > maxRid) maxRid = r;
        }
        int64_t range = (int64_t)maxRid - (int64_t)minRid + 1;

        /* Grow the reused scratch on demand; it is never shrunk. cnt is zeroed
         * over [0, range) below — exactly what the old per-call calloc did — so
         * only the used prefix must be reset, not the whole capacity. tmp is
         * fully overwritten by the scatter, so it needs no initialization. */
        if (range > scratch.cntCap) {
            _mm_free(scratch.cnt);
            scratch.cnt = (int64_t *) _mm_malloc((size_t)range * sizeof(int64_t), 64);
            if (scratch.cnt == NULL) {
                fprintf(stderr, "ERROR: out of memory in %s\n", __func__);
                exit(EXIT_FAILURE);
            }
            scratch.cntCap = range;
        }
        if (smem_count > scratch.tmpCap) {
            _mm_free(scratch.tmp);
            scratch.tmp = (SMEM *) _mm_malloc((size_t)smem_count * sizeof(SMEM), 64);
            if (scratch.tmp == NULL) {
                fprintf(stderr, "ERROR: out of memory in %s\n", __func__);
                exit(EXIT_FAILURE);
            }
            scratch.tmpCap = smem_count;
        }
        int64_t *cnt = scratch.cnt;
        SMEM    *tmp = scratch.tmp;

        memset(cnt, 0, (size_t)range * sizeof(int64_t));

        for (int64_t i = 0; i < smem_count; i++)
            cnt[myMatchArray[i].rid - minRid]++;

        int64_t sum = 0;
        for (int64_t r = 0; r < range; r++) { int64_t c = cnt[r]; cnt[r] = sum; sum += c; }

        /* stable scatter: same-rid SMEMs keep their incoming relative order */
        for (int64_t i = 0; i < smem_count; i++) {
            int64_t b = (int64_t)myMatchArray[i].rid - minRid;
            tmp[cnt[b]++] = myMatchArray[i];
        }
        /* tmp is distinct scratch storage and cannot overlap myMatchArray */
        memcpy(myMatchArray, tmp, (size_t)smem_count * sizeof(SMEM));
    }
}


/* FMI_search::backwardExt is now defined inline in FMI_search.h
 * (issue #87) so the body inlines into all 9 hot callers, eliminating
 * the SysV-ABI struct-by-value pass and return-slot store that dominate
 * self-time on gcc 12+. */

/* Uncompressed SA accessors: these index sa_ms_byte[pos] / sa_ls_word[pos] by the
 * raw BWT row. Under SA_COMPRESSION (the shipped build) those arrays are sized to
 * reference_seq_len >> SA_COMPX, so a raw-row index reads ~SA_COMPX-fold out of
 * bounds. They are only correct for a full (uncompressed) SA, and every in-tree
 * caller is already guarded by `#if !SA_COMPRESSION` (see FMI_search.cpp's
 * sentinel scan and bwamem.cpp's per-read SA resolve), so fence the definitions to
 * match: under SA_COMPRESSION they must not be compiled or callable. The compressed
 * paths (get_sa_entry_compressed / get_sa_entries(..., tid) / get_sa_entries_prefetch)
 * below are the shipped equivalents. */
#if !SA_COMPRESSION
int64_t FMI_search::get_sa_entry(int64_t pos)
{
    int64_t sa_entry = sa_ms_byte[pos];
    sa_entry = sa_entry << 32;
    sa_entry = sa_entry + sa_ls_word[pos];
    return sa_entry;
}

void FMI_search::get_sa_entries(int64_t *posArray, int64_t *coordArray, uint32_t count, int32_t nthreads)
{
    uint32_t i;
// #pragma omp parallel for num_threads(nthreads)
    for(i = 0; i < count; i++)
    {
        /* Prefetch the SAL slot SAL_PFD iterations ahead. Both
         * sa_ms_byte and sa_ls_word are large random-access arrays
         * (separate cache lines), so issue two prefetches per slot. */
        if (i + SAL_PFD < count) {
            int64_t pf_pos = posArray[i + SAL_PFD];
            _mm_prefetch((const char *)(sa_ms_byte + pf_pos), _MM_HINT_T0);
            _mm_prefetch((const char *)(sa_ls_word + pf_pos), _MM_HINT_T0);
        }
        int64_t pos = posArray[i];
        int64_t sa_entry = sa_ms_byte[pos];
        sa_entry = sa_entry << 32;
        sa_entry = sa_entry + sa_ls_word[pos];
        coordArray[i] = sa_entry;
    }
}

void FMI_search::get_sa_entries(SMEM *smemArray, int64_t *coordArray, int32_t *coordCountArray, uint32_t count, int32_t max_occ)
{
    uint32_t i;
    int32_t totalCoordCount = 0;
    for(i = 0; i < count; i++)
    {
        int32_t c = 0;
        SMEM smem = smemArray[i];
        int64_t hi = smem.k + smem.s;
        int64_t step = (smem.s > max_occ) ? smem.s / max_occ : 1;
        int64_t j;
        for(j = smem.k; (j < hi) && (c < max_occ); j+=step, c++)
        {
            int64_t pos = j;
            /* Prefetch SAL_PFD iterations ahead. Both arrays sit on
             * separate cache lines, so issue both prefetches per slot. */
            int64_t pf_pos = pos + SAL_PFD * step;
            if (pf_pos < hi) {
                _mm_prefetch((const char *)(sa_ms_byte + pf_pos), _MM_HINT_T0);
                _mm_prefetch((const char *)(sa_ls_word + pf_pos), _MM_HINT_T0);
            }
            int64_t sa_entry = sa_ms_byte[pos];
            sa_entry = sa_entry << 32;
            sa_entry = sa_entry + sa_ls_word[pos];
            coordArray[totalCoordCount + c] = sa_entry;
        }
        coordCountArray[i] = c;
        totalCoordCount += c;
    }
}
#endif // !SA_COMPRESSION

// sa_compression
int64_t FMI_search::get_sa_entry_compressed(int64_t pos, int tid)
{
    if ((pos & sa_compx_mask) == 0) {
        
        #if  SA_COMPRESSION
        int64_t sa_entry = sa_ms_byte[pos >> sa_compx];
        #else
        int64_t sa_entry = sa_ms_byte[pos];     // simulation
        #endif
        
        sa_entry = sa_entry << 32;
        
        #if  SA_COMPRESSION
        sa_entry = sa_entry + sa_ls_word[pos >> sa_compx];
        #else
        sa_entry = sa_entry + sa_ls_word[pos];   // simulation
        #endif
        
        return sa_entry;        
    }
    else {
        // tprof[MEM_CHAIN][tid] ++;
        int64_t offset = 0; 
        int64_t sp = pos;
        while(true)
        {
            int64_t occ_id_pp_ = sp >> CP_SHIFT;
            int64_t y_pp_ = CP_BLOCK_SIZE - (sp & CP_MASK) - 1; 
            uint64_t *one_hot_bwt_str = cp_occ[occ_id_pp_].one_hot_bwt_str;
            uint8_t b;

            if((one_hot_bwt_str[0] >> y_pp_) & 1)
                b = 0;
            else if((one_hot_bwt_str[1] >> y_pp_) & 1)
                b = 1;
            else if((one_hot_bwt_str[2] >> y_pp_) & 1)
                b = 2;
            else if((one_hot_bwt_str[3] >> y_pp_) & 1)
                b = 3;
            else
                b = 4;

            if (b == 4) {
                return offset;
            }

            GET_OCC(sp, b, occ_id_sp, y_sp, occ_sp, one_hot_bwt_str_c_sp, match_mask_sp);

            sp = count[b] + occ_sp;
            
            offset ++;
            // tprof[ALIGN1][tid] ++;
            if ((sp & sa_compx_mask) == 0) break;
        }
        // assert((reference_seq_len >> sa_compx) - 1 >= (sp >> sa_compx));
        #if  SA_COMPRESSION
        int64_t sa_entry = sa_ms_byte[sp >> sa_compx];
        #else
        int64_t sa_entry = sa_ms_byte[sp];      // simultion
        #endif
        
        sa_entry = sa_entry << 32;

        #if  SA_COMPRESSION
        sa_entry = sa_entry + sa_ls_word[sp >> sa_compx];
        #else
        sa_entry = sa_entry + sa_ls_word[sp];      // simulation
        #endif
        
        sa_entry += offset;
        return sa_entry;
    }
}

void FMI_search::get_sa_entries(SMEM *smemArray, int64_t *coordArray, int32_t *coordCountArray, uint32_t count, int32_t max_occ, int tid)
{
    
    uint32_t i;
    int32_t totalCoordCount = 0;
    for(i = 0; i < count; i++)
    {
        int32_t c = 0;
        SMEM smem = smemArray[i];
        int64_t hi = smem.k + smem.s;
        int64_t step = (smem.s > max_occ) ? smem.s / max_occ : 1;
        int64_t j;
        for(j = smem.k; (j < hi) && (c < max_occ); j+=step, c++)
        {
            int64_t pos = j;
            int64_t sa_entry = get_sa_entry_compressed(pos, tid);
            coordArray[totalCoordCount + c] = sa_entry;
        }
        // coordCountArray[i] = c;
        *coordCountArray += c;
        totalCoordCount += c;
    }
}

// SA_COPMRESSION w/ PREFETCH
int64_t FMI_search::call_one_step(int64_t pos, int64_t &sa_entry, int64_t &offset,
                                  int drop_sentinel_offset)
{
    if ((pos & sa_compx_mask) == 0) {        
        sa_entry = sa_ms_byte[pos >> sa_compx];        
        sa_entry = sa_entry << 32;        
        sa_entry = sa_entry + sa_ls_word[pos >> sa_compx];        
        // return sa_entry;
        return 1;
    }
    else {
        // int64_t offset = 0; 
        int64_t sp = pos;

        int64_t occ_id_pp_ = sp >> CP_SHIFT;
        int64_t y_pp_ = CP_BLOCK_SIZE - (sp & CP_MASK) - 1; 
        uint64_t *one_hot_bwt_str = cp_occ[occ_id_pp_].one_hot_bwt_str;
        uint8_t b;

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)
        /* The BWT symbol at sp is one-hot across the four bit-planes, and the
         * bases are close to uniformly distributed, so testing the planes one
         * branch at a time mispredicts on most steps. Test all four at once
         * and select the symbol without a branch; the empty case (no plane
         * set) is the sentinel row, kept as the one real branch since it is
         * nearly never taken. The occurrence count then reuses the k-only
         * backward step's NEON popcount, whose d-register loads keep the
         * masked word out of the general-purpose file. Symbol priority
         * (lowest plane wins) matches the plane-at-a-time test exactly. */
        const unsigned t0 = (unsigned)((one_hot_bwt_str[0] >> y_pp_) & 1);
        const unsigned t1 = (unsigned)((one_hot_bwt_str[1] >> y_pp_) & 1);
        const unsigned t2 = (unsigned)((one_hot_bwt_str[2] >> y_pp_) & 1);
        const unsigned t3 = (unsigned)((one_hot_bwt_str[3] >> y_pp_) & 1);
        if ((t0 | t1 | t2 | t3) == 0) {
            // Sentinel ($) row (no bit-plane set, the scalar path's b == 4): its
            // suffix-array value is 0 and we walked `offset` LF steps to reach it,
            // so the SA entry is 0 + offset. drop_sentinel_offset (=> 0) reproduces
            // bwa-mem2's dropped walk for --compat=bwa-mem2; the default keeps the
            // accumulated offset, matching the scalar branch and the compressed
            // sibling get_sa_entry_compressed.
            sa_entry = drop_sentinel_offset ? 0 : offset;
            return 1;
        }
        b = (uint8_t)(t0 ? 0 : t1 ? 1 : t2 ? 2 : 3);
        const int64_t occ_sp = cp_occ[occ_id_pp_].cp_count[b] +
            occ_popcount64(&one_hot_bwt_str[b], &one_hot_mask_array[sp & CP_MASK]);
#else
        if((one_hot_bwt_str[0] >> y_pp_) & 1)
            b = 0;
        else if((one_hot_bwt_str[1] >> y_pp_) & 1)
            b = 1;
        else if((one_hot_bwt_str[2] >> y_pp_) & 1)
            b = 2;
        else if((one_hot_bwt_str[3] >> y_pp_) & 1)
            b = 3;
        else
            b = 4;
        if (b == 4) {
            // Sentinel ($) row: its suffix-array value is 0, and we have walked
            // `offset` LF steps to reach it, so the SA entry is 0 + offset. The
            // compressed sibling get_sa_entry_compressed returns `offset` here for
            // the same reason. bwa-mem2 set sa_entry = 0 here, dropping the walk
            // and reporting positions within the first `offset` bases of the
            // concatenated reference up to `offset` bases too far left; that is
            // kept only for --compat=bwa-mem2 (drop_sentinel_offset), whose
            // contract is to reproduce bwa-mem2's records.
            sa_entry = drop_sentinel_offset ? 0 : offset;
            return 1;
        }
        
        GET_OCC(sp, b, occ_id_sp, y_sp, occ_sp, one_hot_bwt_str_c_sp, match_mask_sp);
#endif
        
        sp = count[b] + occ_sp;
        
        offset ++;
        if ((sp & sa_compx_mask) == 0) {
    
            sa_entry = sa_ms_byte[sp >> sa_compx];        
            sa_entry = sa_entry << 32;
            sa_entry = sa_entry + sa_ls_word[sp >> sa_compx];
            
            sa_entry += offset;
            // return sa_entry;
            return 1;
        }
        else {
            sa_entry = sp;
            return 0;
        }
    } // else
}

/* Lane count of the pipelined compressed-SA resolver (get_sa_entries_prefetch);
 * see the BWA3_SA_LANES note there. 40 on x86_64, 20 elsewhere: a 20-vs-40 A/B
 * on a 5M-pair WGS slice (3 interleaved reps, user CPU) measured -0.30% on
 * AMD Zen3 (every rep faster) and no change on Intel Sapphire Rapids, while
 * a 16..40 sweep on Graviton4 was flat. The count only sets how many
 * checkpoint-block fetches are in flight; results are stored by position and
 * are byte-identical at every width. */
#if defined(__x86_64__) || defined(_M_X64)
#define SA_RESOLVE_LANES_DEFAULT 40
#else
#define SA_RESOLVE_LANES_DEFAULT 20
#endif
#define SA_RESOLVE_LANES_MAX 64
/* The resolver's lane arrays (working_set/map_pos/offset) are sized to
 * SA_RESOLVE_LANES_MAX and indexed by sa_batch_size, which is clamped into
 * [1, SA_RESOLVE_LANES_MAX] for env overrides but defaults to
 * SA_RESOLVE_LANES_DEFAULT unclamped -- guard the default against silently
 * overrunning those stack arrays if it is ever raised past the array size. */
static_assert(SA_RESOLVE_LANES_DEFAULT >= 1 &&
              SA_RESOLVE_LANES_DEFAULT <= SA_RESOLVE_LANES_MAX,
              "SA_RESOLVE_LANES_DEFAULT must be within [1, SA_RESOLVE_LANES_MAX]");

/* Thread-local scratch for the pos_ar/map_ar staging buffers below. These were
 * two _mm_malloc/_mm_free per call, and this function runs once per read — so on
 * a WGS run that is tens of millions of aligned-allocation round-trips. The pair
 * is pure scratch (no state carried between calls), so a per-thread grow-only
 * holder reused across reads removes the churn. FMI_search is shared across
 * worker threads, hence thread_local (not a member). Freed on thread exit. */
namespace {
struct SaPrefetchScratch {
    // Only pos is staged now: the former map_ar array held map_ar[k] == k for
    // every entry (the staging loop writes map_ar[id] = totalCoordCount + c,
    // and totalCoordCount == id on entry to each SMEM while both advance in
    // lockstep with c), so the map index is just the buffer index and the
    // second allocation was dead. Downstream consumers use the index directly.
    int64_t *pos = nullptr;
    int64_t  cap = 0;
    void ensure(int64_t n) {
        if (n <= cap) return;
        _mm_free(pos);
        pos = (int64_t *) _mm_malloc((size_t)n * sizeof(int64_t), 64);
        // Fail loudly on OOM: without this, pos==NULL but cap=n reports capacity,
        // so the next call returns early and the staging loop writes through NULL.
        xassert(pos != NULL, "out of memory: SA prefetch staging buffer");
        cap = n;
    }
    ~SaPrefetchScratch() { _mm_free(pos); }
};
} // namespace

void FMI_search::get_sa_entries_prefetch(SMEM *smemArray, int64_t *coordArray,
                                         int64_t *coordCountArray, int64_t count,
                                         const int32_t max_occ, int tid, int64_t &id_,
                                         int drop_sentinel_offset)
{

    // uint32_t i;
    // totalCoordCount and id (below) both count entries staged into the int64
    // coordArray/pos_ar/map_ar buffers and are paired with the int64 mem_lim,
    // so keep them int64 to avoid truncating the running offset.
    int64_t totalCoordCount = 0;
    // mem_lim is the exact number of entries the staging loop below writes:
    // each SMEM contributes min(smem.s, max_occ) (the inner loop stops at
    // c < max_occ). Summing min(s, max_occ) instead of the uncapped interval
    // size s both right-sizes the pos_ar/map_ar allocation -- repetitive seeds
    // have s up to the reference length but still write only max_occ entries --
    // and keeps every term bounded by max_occ so the int64 sum cannot run away.
    // id indexes those buffers and is paired with mem_lim, so it is int64 too.
    int64_t mem_lim = 0, id = 0;

    for(int i = 0; i < count; i++)
    {
        SMEM smem = smemArray[i];
        mem_lim += (smem.s > max_occ) ? max_occ : smem.s;
    }

    /* Reuse the per-thread staging buffers, grown to the high-water mem_lim.
     * mem_lim == 0 leaves the pointers null, but the staging loop below writes
     * nothing in that case (id stays 0), so they are never dereferenced. */
    static thread_local SaPrefetchScratch t_sa;
    t_sa.ensure(mem_lim);
    int64_t *pos_ar = t_sa.pos;

    for(int i = 0; i < count; i++)
    {
        int32_t c = 0;
        SMEM smem = smemArray[i];
        int64_t hi = smem.k + smem.s;
        int64_t step = (smem.s > max_occ) ? smem.s / max_occ : 1;
        int64_t j;
        for(j = smem.k; (j < hi) && (c < max_occ); j+=step, c++)
        {
            int64_t pos = j;
             pos_ar[id++]  = pos;
            // map_ar[k] == k (== id here), so the staging index is stored
            // implicitly; map_pos below reads the index directly.
            // int64_t sa_entry = get_sa_entry_compressed(pos, tid);
            // coordArray[totalCoordCount + c] = sa_entry;
        }
        //coordCountArray[i] = c;
        *coordCountArray += c;
        totalCoordCount += c;
    }
    
    id_ += id;
    
    /* Number of SA rows walked concurrently. Each lane's next checkpoint block
     * is prefetched when the lane is (re)staged, so the lane count sets how
     * many block fetches are in flight; it does not affect results, which are
     * stored by position. BWA3_SA_LANES overrides the default for tuning; out
     * of range values (below 1 or above SA_RESOLVE_LANES_MAX) are ignored. */
    static const int32_t sa_batch_size = []() {
        int32_t lanes = SA_RESOLVE_LANES_DEFAULT;
        const char *e = getenv("BWA3_SA_LANES");
        if (e != NULL) {
            // Strict parse: atoi accepts a numeric prefix ("1x" -> 1) and has
            // undefined behavior when the value overflows int. Match the
            // BWA3_LOAD_THREADS parse in index_load_threads -- reject empty,
            // trailing-garbage, and overflowed input, then keep values in
            // [1, SA_RESOLVE_LANES_MAX]. On anything else warn and keep the
            // default (ignore-and-continue, no hard exit).
            char *end = NULL;
            errno = 0;
            const long v = strtol(e, &end, 10);
            if (errno != 0 || end == e || *end != '\0' ||
                v < 1 || v > SA_RESOLVE_LANES_MAX) {
                fprintf(stderr, "[W::get_sa_entries_prefetch] ignoring BWA3_SA_LANES=%s (expected 1..%d)\n",
                        e, (int) SA_RESOLVE_LANES_MAX);
            } else {
                lanes = (int32_t) v;
            }
        }
        return lanes;
    }();
    int64_t working_set[SA_RESOLVE_LANES_MAX], map_pos[SA_RESOLVE_LANES_MAX];
    int64_t offset[SA_RESOLVE_LANES_MAX] = {-1};
    
    int i = 0, j = 0;    
    while(i<id && j<sa_batch_size)
    {
        int64_t pos =  pos_ar[i];
        working_set[j] = pos;
        map_pos[j] = i;   // map_ar[i] == i (see staging loop invariant)
        offset[j] = 0;
        
        if ((pos & sa_compx_mask) == 0) {
            _mm_prefetch(&sa_ms_byte[pos >> sa_compx], _MM_HINT_T0);
            _mm_prefetch(&sa_ls_word[pos >> sa_compx], _MM_HINT_T0);
        }
        else {
            int64_t occ_id_pp_ = pos >> CP_SHIFT;
            _mm_prefetch(&cp_occ[occ_id_pp_], _MM_HINT_T0);
        }
        i++;
        j++;
    }
        
    // all_quit counts up to id (int64); lim stays int (bounded by sa_batch_size).
    int lim = j;
    int64_t all_quit = 0;
    while (all_quit < id)
    {
        
        for (int k=0; k<lim; k++)
        {
            int64_t sp = 0, pos = 0;
            bool quit;
            if (offset[k] >= 0) {
                quit = call_one_step(working_set[k], sp, offset[k], drop_sentinel_offset);
            }
            else
                continue;
            
            if (quit) {
                coordArray[map_pos[k]] = sp;
                all_quit ++;
                
                if (i < id)
                {
                    pos = pos_ar[i];
                    working_set[k] = pos;
                    map_pos[k] = i++;   // map_ar[i] == i (staging invariant)
                    offset[k] = 0;
                    
                    if ((pos & sa_compx_mask) == 0) {
                        _mm_prefetch(&sa_ms_byte[pos >> sa_compx], _MM_HINT_T0);
                        _mm_prefetch(&sa_ls_word[pos >> sa_compx], _MM_HINT_T0);
                    }
                    else {
                        int64_t occ_id_pp_ = pos >> CP_SHIFT;
                        _mm_prefetch(&cp_occ[occ_id_pp_], _MM_HINT_T0);
                    }
                }
                else
                    offset[k] = -1;
            }
            else {
                working_set[k] = sp;
                if ((sp & sa_compx_mask) == 0) {
                    _mm_prefetch(&sa_ms_byte[sp >> sa_compx], _MM_HINT_T0);
                    _mm_prefetch(&sa_ls_word[sp >> sa_compx], _MM_HINT_T0);
                }
                else {
                    int64_t occ_id_pp_ = sp >> CP_SHIFT;
                    _mm_prefetch(&cp_occ[occ_id_pp_], _MM_HINT_T0);
                }                
            }
        }
    }
    /* pos_ar is the reused thread-local scratch — no free here. */
}
