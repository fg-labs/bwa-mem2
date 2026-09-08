#include "libsais_build.h"

#include "fm_index_writer.h"
#include "index_prelude.h"
#include "io_utils.h"
#include "macro.h"
#include "packed_text.h"
#include "system.h"
#include "utils.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <new>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// libsais headers use `SA` as a parameter name in their public API; bwa's
// macro.h defines SA as a preprocessor constant. Undefine around the include
// so we get the real declarations. We don't need the SA macro here anyway.
#ifdef SA
# pragma push_macro("SA")
# undef SA
# define SA_MACRO_PUSHED 1
#endif
extern "C" {
#include "libsais.h"
#include "libsais64.h"
}
#ifdef SA_MACRO_PUSHED
# pragma pop_macro("SA")
# undef SA_MACRO_PUSHED
#endif

namespace {

using clock_t_ = std::chrono::steady_clock;
double elapsed_since(clock_t_::time_point t) {
    return std::chrono::duration<double>(clock_t_::now() - t).count();
}


// Reference length in the unit a user thinks in, so a small fixture does not
// report "0.00 Gbp".
std::string fmt_bases(int64_t bp) {
    char out[32];
    if      (bp >= 1000000000LL) std::snprintf(out, sizeof(out), "%.2f Gbp", (double)bp / 1e9);
    else if (bp >= 1000000LL)    std::snprintf(out, sizeof(out), "%.1f Mbp", (double)bp / 1e6);
    else                         std::snprintf(out, sizeof(out), "%lld bp",  (long long)bp);
    return std::string(out);
}

using bwa::fmt_bytes;

// Pack the forward-only .pac (l_pac bases) plus its reverse-complement into
// a doubled .pac on disk. The sais-lite path built this doubled text in
// memory via pac2nt; writing it packed lets PackedText / emit_0123 /
// compute_counts consume both strands without another ASCII round-trip.
// Parallelises by striping output bytes: each thread fills a disjoint
// [byte_start, byte_end) range of the output buffer, so there's no
// aliasing at the forward/reverse-complement seam even when l_pac isn't
// a multiple of 4.
void write_doubled_pac(const char* fwd_pac_path, int64_t l_pac,
                       const char* out_path, int num_threads)
{
    PackedText fwd(fwd_pac_path, l_pac);
    int64_t doubled = 2 * l_pac;
    int64_t nbytes  = (doubled + 3) / 4;
    // Default-init via new[]: every byte is unconditionally overwritten in
    // the parallel loop below, so the value-init zero-fill that
    // std::vector<uint8_t>(n, 0) performed was wasted bandwidth.
    // new (nothrow): a throwing new[] on OOM calls std::terminate, which runs no
    // atexit handlers -- bypassing scratch cleanup. err_fatal prints a `[func]
    // message` diagnostic and exit()s, running the cleanup path.
    std::unique_ptr<uint8_t[]> buf(new (std::nothrow) uint8_t[(size_t)nbytes]);
    if (!buf) err_fatal(__func__, "out of memory: %lld bytes for the 2-bit pack buffer", (long long)nbytes);

    auto base_at = [&](int64_t idx) -> uint8_t {
        // Forward half: fwd[idx]. RC half: complement of fwd[l_pac-1-(idx-l_pac)].
        if (idx < l_pac) return fwd.get_base(idx);
        int64_t src = l_pac - 1 - (idx - l_pac);
        return (uint8_t)(3u - fwd.get_base(src));
    };

#ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t by = 0; by < nbytes; ++by) {
        uint8_t b = 0;
        for (int k = 0; k < 4; ++k) {
            int64_t idx = by * 4 + k;
            if (idx >= doubled) break;
            uint8_t base = base_at(idx);
            b |= (uint8_t)(base << ((3 - k) << 1));
        }
        buf[(size_t)by] = b;
    }

    int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) err_fatal(__func__, "open('%s'): %s", out_path, strerror(errno));
    // Linux caps a single write(2) on regular files at 0x7ffff000 bytes (~2 GiB)
    // regardless of the requested length; for references where doubled-pac
    // grows past that (large custom genomes / methylation doubled wheat) the
    // raw write() returned a short count and aborted with a half-written file
    // in $TMPDIR. pwrite_all loops on EINTR + short writes.
    pwrite_all(fd, buf.get(), (size_t)nbytes, 0, "doubled.pac data");
    off_t cur = (off_t)nbytes;
    uint8_t ct = 0;
    if (doubled % 4 == 0) {
        pwrite_all(fd, &ct, 1, cur, "doubled.pac pad-byte");
        ++cur;
    }
    ct = (uint8_t)(doubled % 4);
    pwrite_all(fd, &ct, 1, cur, "doubled.pac count-byte");
    if (close(fd) != 0)
        err_fatal(__func__, "close('%s'): %s", out_path, strerror(errno));
}

std::string resolve_tmpdir(const LibsaisBuildOpts& opts) {
    if (!opts.tmpdir.empty()) return opts.tmpdir;
    if (const char* e = std::getenv("TMPDIR")) return std::string(e);
    return std::string("/tmp");
}

} // anonymous namespace

bool libsais_sa_is_64bit(int64_t N) {
    const int64_t kSais32Threshold = (int64_t)INT32_MAX - 10000;
    return (N + 1) >= kSais32Threshold;
}

int64_t libsais_estimate_peak_bytes(int64_t N) {
    // Per-base cost, calibrated against measured peak RSS rather than derived
    // from the buffer arithmetic alone (buf is 1 B/base, the SA 4 or 8, and
    // libsais's aux arrays plus the OMP/allocator arenas add the rest).
    //
    // Two properties have to hold, and the second is the binding one because
    // `index --meth` runs two builds in one process -- the original over
    // N = 2*l_pac, then a seed over 4*l_pac -- and does not return the first
    // build's memory before the second starts. So the estimate for a build must
    // also cover a --meth invocation whose *dominant* build is that size.
    //
    // Worst measured cost per base of N, over a size ladder from chr17 (0.08
    // Gbp) to chr1-9 (1.67 Gbp) on a 64 GiB 12-core host:
    //
    //             single build   --meth process   estimate   margins
    //   int32 SA      5.90 B         7.07 B          8         1.36 / 1.13
    //   int64 SA      9.84 B        10.04 B         12         1.22 / 1.20
    //
    // The int32 constant was 6, i.e. only 1.02x over a single build's own peak
    // and *below* the --meth process peak -- so a small-reference `--meth` build
    // could exceed its budget by ~17% while every individual estimate cleared
    // it. int64's 1.22x margin already absorbed the same carry-over, which is
    // why the defect was confined to the int32 path. 8 restores a comparable
    // margin; no --meth-specific term is needed, and none is wanted, since the
    // carry-over measures as bounded rather than proportional.
    return (N + 1) * (libsais_sa_is_64bit(N) ? 12 : 8);
}

void libsais_report_budget_shortfall(const char* what, int64_t need_bytes,
                                     int64_t budget_bytes, bool budget_user_specified,
                                     int64_t ref_bases) {
    const bool use_int64_sa = libsais_sa_is_64bit(2 * ref_bases);
    // Round the --max-memory suggestion up to a whole unit the flag accepts, in
    // the unit the requirement actually lands in: ceil-to-GiB overshoots ~44x on
    // a 23 MiB requirement.
    const bool      suggest_g   = need_bytes >= (1LL << 30);
    const int64_t   unit        = suggest_g ? (1LL << 30) : (1LL << 20);
    const long long suggest_n   = (long long)((need_bytes + unit - 1) / unit);
    const char      suggest_sfx = suggest_g ? 'G' : 'M';

    std::fprintf(stderr,
            "ERROR: %s needs ~%s (%d-bit SA) to index a %s reference; "
            "--max-memory is %s.\n",
            what, fmt_bytes(need_bytes).c_str(), use_int64_sa ? 64 : 32,
            fmt_bases(ref_bases).c_str(), fmt_bytes(budget_bytes).c_str());

    // A host-size hint is noise when the user pinned the budget themselves: the
    // binding constraint is their flag, not the machine. "Use a smaller
    // reference" is never an option for a whole-genome aligner, so it is gone.
    const int64_t required_total = budget_user_specified
                                 ? -1
                                 : bwa::required_total_for_batch_budget(need_bytes);
    if (required_total > 0)
        std::fprintf(stderr,
                "       Retry on a host with >= %s of RAM, or pass --max-memory %lld%c\n"
                "       to build within a smaller budget (which may swap). Bounded-memory\n"
                "       SA construction is not yet implemented.\n",
                fmt_bytes(required_total).c_str(), suggest_n, suggest_sfx);
    else
        std::fprintf(stderr,
                "       Raise --max-memory to at least %lld%c. Bounded-memory SA "
                "construction is not yet implemented.\n",
                suggest_n, suggest_sfx);
}

int libsais_build_fm_index(const char* prefix, int64_t pac_len,
                           const LibsaisBuildOpts& opts)
{
    int     T_user = opts.num_threads > 0 ? opts.num_threads : 1;
    int64_t N = pac_len;   // doubled-text length
    int64_t l_pac = N / 2;

    // Cap effective parallelism at (N+1) / 16 Mbp. On tiny inputs the OMP
    // pool spin-up + per-thread heap arenas cost more than the parallel
    // work saves, which breaks the --max-memory contract on small fixtures.
    constexpr int64_t kMinBasesPerThread = 16LL << 20;
    int T = (int)std::min<int64_t>(T_user,
                                   std::max<int64_t>(1, (N + 1) / kMinBasesPerThread));

    auto t0 = clock_t_::now();
    std::fprintf(stderr, "[libsais_build] N=%lld threads=%d (user-requested=%d)\n",
            (long long)N, T, T_user);

    const bool    use_int64_sa = libsais_sa_is_64bit(N);
    const int64_t est_bytes    = libsais_estimate_peak_bytes(N);
    if (opts.max_memory_bytes > 0 && est_bytes > opts.max_memory_bytes) {
        libsais_report_budget_shortfall("libsais", est_bytes, opts.max_memory_bytes,
                                        opts.max_memory_user_specified, l_pac);
        return 3;
    }
    std::fprintf(stderr, "[libsais_build] estimate ~%s (budget %s)\n",
            fmt_bytes(est_bytes).c_str(),
            opts.max_memory_bytes > 0 ? fmt_bytes(opts.max_memory_bytes).c_str() : "unset");

    // Phase 0: materialise a doubled .pac so downstream helpers can mmap it
    // via PackedText. bns_fasta2bntseq writes forward-only; we append the
    // RC half in a scratch temp file.
    std::string tmp_base = resolve_tmpdir(opts);
    char workdir[PATH_MAX];
    std::snprintf(workdir, sizeof(workdir), "%s/bwa_libsais_XXXXXX", tmp_base.c_str());
    if (!mkdtemp(workdir))
        err_fatal(__func__, "mkdtemp under '%s' failed: %s",
                  tmp_base.c_str(), strerror(errno));
    std::string doubled_pac = std::string(workdir) + "/doubled.pac";
    // RAII guard so the workdir + doubled.pac are removed on every exit
    // path, including the err_fatal calls below. err_fatal calls exit(),
    // so we also wire an atexit hook for non-stack-unwinding paths.
    struct TmpGuard {
        std::string dir;
        std::string pac;
        static TmpGuard*& current() { static TmpGuard* g = nullptr; return g; }
        static void atexit_cleanup() {
            if (auto* g = current()) g->cleanup();
        }
        void cleanup() {
            if (!pac.empty()) { ::unlink(pac.c_str()); pac.clear(); }
            if (!dir.empty()) { ::rmdir(dir.c_str());  dir.clear(); }
        }
        ~TmpGuard() { cleanup(); current() = nullptr; }
    } tmp_guard{workdir, doubled_pac};
    TmpGuard::current() = &tmp_guard;
    std::atexit(&TmpGuard::atexit_cleanup);

    write_doubled_pac((std::string(prefix) + ".pac").c_str(), l_pac,
                      doubled_pac.c_str(), T);

    PackedText pac(doubled_pac.c_str(), N);

    /* D3 --meth seed index passes emit_unpacked_ref=false: the seed `.0123` is
     * never read at runtime (extension uses the original reference), so skip
     * the ~13 GB write. The FM build below uses `pac`, not the `.0123` file. */
    if (opts.emit_unpacked_ref) {
        if (emit_0123(pac, prefix, T) != 0)
            err_fatal(__func__, "emit_0123 failed");
    }

    int64_t count[5] = {0};
    compute_counts(pac, count, T);
    std::fprintf(stderr, "[libsais_build] phase 0 (doubled-pac%s + count) %.2fs\n",
            opts.emit_unpacked_ref ? " + .0123" : "", elapsed_since(t0));

    // Phase 1: unpack .pac into a libsais-ready byte buffer. Alphabet is
    // {0=$, 1..4=ACGT}; the trailing 0 at index N is the GSA terminator.
    auto t1 = clock_t_::now();
    // Default-init: positions [0, N) are written by the parallel loop and
    // buf[N] is set explicitly to the GSA terminator below, so the
    // value-init zero-fill std::vector<uint8_t>(N+1) performed was a wasted
    // ~6.2 GiB write on a doubled-human input.
    // new (nothrow) + err_fatal: OOM here must run the TmpGuard cleanup (see
    // below), which a throwing new[] -> std::terminate would skip.
    std::unique_ptr<uint8_t[]> buf(new (std::nothrow) uint8_t[(size_t)(N + 1)]);
    if (!buf) err_fatal(__func__, "out of memory: %lld bytes for the SAIS input buffer", (long long)(N + 1));
    // Capture out-of-range bases without crashing inside the OpenMP region:
    // err_fatal terminates via exit(), which interacts poorly with libomp's
    // teardown when other threads are mid-loop. Record the first offending
    // index atomically and report after the parallel region.
    std::atomic<int64_t> bad_pos{-1};
    std::atomic<unsigned> bad_val{0};
    uint8_t* buf_ptr = buf.get();
#ifdef _OPENMP
    #pragma omp parallel for num_threads(T) schedule(static)
#endif
    for (int64_t i = 0; i < N; ++i) {
        uint8_t b = pac.get_base(i);
        if (b >= 4) {
            int64_t expect = -1;
            if (bad_pos.compare_exchange_strong(expect, i)) bad_val.store(b);
        }
        buf_ptr[(size_t)i] = (uint8_t)(b + 1);
    }
    buf_ptr[(size_t)N] = 0;
    if (bad_pos.load() >= 0)
        err_fatal(__func__,
                  "unexpected non-2bit base %u at doubled-pac[%lld]",
                  bad_val.load(), (long long)bad_pos.load());
    std::fprintf(stderr, "[libsais_build] phase 1 (unpack %lld bases) %.2fs\n",
            (long long)(N + 1), elapsed_since(t1));

    // Phase 2: libsais GSA. Extra-space `fs` lets libsais avoid a realloc
    // inside the induced-sorting passes; 10000 matches upstream's example.
    // SA buffer is allocated default-init (uninitialized): libsais writes
    // every output suffix and uses the trailing `fs` slots as scratch, so
    // the implicit zero-fill from std::vector<T>(n) was wasted bandwidth
    // (~50 GiB on a doubled-human input with int64_t SA).
    auto t2 = clock_t_::now();
    const int64_t fs = 10000;
    std::unique_ptr<int64_t[]> sa64;
    std::unique_ptr<int32_t[]> sa32;
    if (!use_int64_sa) {
        sa32.reset(new (std::nothrow) int32_t[(size_t)(N + 1 + fs)]);
        if (!sa32) err_fatal(__func__, "out of memory: %lld bytes for the 32-bit suffix array", (long long)((N + 1 + fs) * (int64_t)sizeof(int32_t)));
        int32_t rc = libsais_gsa_omp(buf_ptr, sa32.get(),
                                     (int32_t)(N + 1), (int32_t)fs,
                                     /*freq=*/nullptr, T);
        if (rc != 0) err_fatal(__func__, "libsais_gsa_omp failed (rc=%d)", rc);
    } else {
        sa64.reset(new (std::nothrow) int64_t[(size_t)(N + 1 + fs)]);
        if (!sa64) err_fatal(__func__, "out of memory: %lld bytes for the 64-bit suffix array", (long long)((N + 1 + fs) * (int64_t)sizeof(int64_t)));
        int64_t rc = libsais64_gsa_omp(buf_ptr, sa64.get(),
                                       N + 1, fs,
                                       /*freq=*/nullptr, T);
        if (rc != 0) err_fatal(__func__, "libsais64_gsa_omp failed (rc=%lld)", (long long)rc);
    }
    std::fprintf(stderr, "[libsais_build] phase 2 (libsais_gsa) %.2fs\n", elapsed_since(t2));

    // Phase 3: streaming FM-index writer reads BWT bytes and SA samples
    // in flight from sa + buf; no bwt[] or samples[] is materialised.
    auto t3 = clock_t_::now();
    std::string bwt_path = std::string(prefix) + ".bwt.2bit.64";
    int64_t sentinel_index = -1;
    // Drive the dispatch off the explicit width flag rather than
    // checking which unique_ptr is non-null; the latter is correct today
    // but would silently flip if a future refactor allocated both buffers
    // for RAII reasons.
    const void* sa_ptr = use_int64_sa
        ? (const void*)sa64.get()
        : (const void*)sa32.get();
    write_fm_index_streaming(
        bwt_path.c_str(), buf_ptr,
        sa_ptr,
        /*sa_is_64bit=*/use_int64_sa,
        N, count, &sentinel_index, opts.sa_compx, T);
    std::fprintf(stderr,
            "[libsais_build] phase 3 (streaming FM-index write) %.2fs, sentinel_index=%lld; total %.2fs\n",
            elapsed_since(t3), (long long)sentinel_index, elapsed_since(t0));

    return 0;
}
