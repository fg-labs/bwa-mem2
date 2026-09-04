#include "index_prelude.h"

#include "io_utils.h"
#include "packed_text.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <vector>

// Forward-declare the two OpenMP runtime entry points we actually call,
// rather than #include <omp.h>. omp.h doesn't ship with Apple's clang or
// (by default) with Linux clang; the runtime symbols are still resolved
// via -fopenmp / -lomp at link time. Pragmas don't need omp.h either.
#ifdef _OPENMP
extern "C" {
    int omp_get_thread_num(void);
    int omp_get_num_threads(void);
}
#endif

int emit_0123(const PackedText& pac, const char* prefix, int num_threads)
{
    std::string out = std::string(prefix) + ".0123";
    // Write-only: fd is never read on the success path.
    int fd = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        err_fatal(__func__, "failed to open '%s': %s", out.c_str(), strerror(errno));
    }
    int64_t N = pac.length();
    if (ftruncate(fd, N) < 0) {
        err_fatal(__func__, "ftruncate '%s' failed: %s", out.c_str(), strerror(errno));
    }

    if (num_threads < 1) num_threads = 1;
    // Each thread streams its stripe through a small fixed-size buffer so
    // per-thread peak stays O(KB) regardless of input size. OpenMP reuses
    // the process-wide libomp thread pool (shared with libsais) so no new
    // OS threads are created.
    constexpr size_t kEmitChunk = 64 * 1024;

    // Capture the first write failure inside the OpenMP region and report it
    // AFTER the region: pwrite_all's err_fatal()->exit() would tear down libomp
    // while sibling threads are still mid-loop (hang/UB). Mirrors the
    // capture-then-fatal pattern in libsais_build.cpp's unpack loop.
    std::atomic<int>       io_errno{0};
    std::atomic<long long> io_off{-1};
#ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
#endif
    {
        std::vector<uint8_t> thr_buf(kEmitChunk);
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
#else
        int tid = 0, nt = 1;
#endif
        int64_t s = (N * (int64_t)tid) / nt;
        int64_t e = (N * (int64_t)(tid + 1)) / nt;
        int64_t off = s;
        while (off < e) {
            size_t n = (size_t)std::min<int64_t>(e - off, (int64_t)kEmitChunk);
            for (size_t k = 0; k < n; ++k) thr_buf[k] = pac.get_base(off + (int64_t)k);
            int rc = pwrite_all_status(fd, thr_buf.data(), n, off);
            if (rc != 0) {
                int expect = 0;
                if (io_errno.compare_exchange_strong(expect, rc)) io_off.store((long long)off);
                break;  // stop this thread; the region ends, then we err_fatal
            }
            off += (int64_t)n;
        }
    }
    // Report the captured failure outside the region. `off` is the failing
    // chunk's start (pwrite_all_status advances internally), so "at or after".
    // rc < 0 is a pwrite-returned-0; rc > 0 is an errno.
    int io_rc = io_errno.load();
    if (io_rc < 0)
        err_fatal(__func__, "pwrite('%s') returned 0 at or after offset %lld",
                  out.c_str(), io_off.load());
    else if (io_rc > 0)
        err_fatal(__func__, "pwrite('%s') failed at or after offset %lld: %s",
                  out.c_str(), io_off.load(), strerror(io_rc));
    // Surface deferred I/O errors (delayed-writeback EIO, full filesystem,
    // etc.) so a truncated .0123 never masquerades as complete. This runs only
    // when the caller opted into emitting .0123 (off by default; `mem`
    // pac-fetches from .pac instead).
    if (close(fd) != 0) {
        err_fatal(__func__, "close('%s') failed: %s", out.c_str(), strerror(errno));
    }
    return 0;
}

void compute_counts(const PackedText& pac, int64_t count[5], int num_threads)
{
    int64_t N = pac.length();
    if (num_threads < 1) num_threads = 1;

    int64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static) \
                             reduction(+:r0,r1,r2,r3)
#endif
    for (int64_t i = 0; i < N; ++i) {
        uint8_t c = pac.get_base(i);
        switch (c) {
            case 0: ++r0; break;
            case 1: ++r1; break;
            case 2: ++r2; break;
            case 3: ++r3; break;
            default: break;
        }
    }

    count[0] = 0;
    count[1] = r0;
    count[2] = r0 + r1;
    count[3] = r0 + r1 + r2;
    count[4] = r0 + r1 + r2 + r3;
}
