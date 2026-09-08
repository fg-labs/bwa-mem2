#ifndef BWA_IO_UTILS_H
#define BWA_IO_UTILS_H

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>

#include "utils.h"

// Single per-call byte cap shared by both the read and the write path. macOS
// rejects any pread()/pwrite() whose count exceeds INT_MAX (2GiB) with EINVAL
// outright -- it is not reported as a short transfer, so the retry loops never
// get a chance and the index build aborts. The meth doubled .pac for a
// human-scale genome is ~3.2GB, which trips this on the write side; the base
// doubled .pac (~1.6GB) stays under it, which is why only --meth failed. Linux
// glibc caps a single read/write at 0x7ffff000 but returns a short count the
// loops already handle. Clamping every request to 1GiB (well under INT_MAX)
// makes both platforms take the short-transfer path. This must NOT be
// "simplified" away. This is the ONE place the cap is defined; the read-side
// fmi_pread_request_size() delegates here too.
static const size_t IO_MAX_ONCE = (size_t)1 << 30;   /* 1GiB, well under INT_MAX */

// Bytes to hand a single pread()/pwrite() call: the smaller of what remains and
// `cap`. Shared by pwrite_request_size() and fmi_pread_request_size().
static inline size_t io_request_size(size_t remaining, size_t cap)
{
    return remaining > cap ? cap : remaining;
}

// Bytes to hand a single pwrite() call: the smaller of what remains and `cap`
// (the 1GiB IO_MAX_ONCE by default). Kept as a distinct, defaulted-cap entry
// point so the write-side tests can call it directly.
static inline size_t pwrite_request_size(size_t remaining, size_t cap = IO_MAX_ONCE)
{
    return io_request_size(remaining, cap);
}

// Test-only observability: counts the completed write chunks (one per positive
// pwrite(), not per EINTR retry) in the most
// recent pwrite_all()/pwrite_all_status() call on this thread, so a test can prove it
// actually split its buffer into chunks. A regression that drops the clamp
// would write the whole buffer in one pwrite() and the count would fall to 1.
// The single increment is negligible next to a 1GiB write, so production
// behaviour is unchanged. `inline` (not `static inline`) so every translation
// unit shares one counter -- the test reads the same one pwrite_all() bumps.
inline unsigned long& pwrite_all_chunk_counter()
{
    static thread_local unsigned long count = 0;
    return count;
}
inline void          pwrite_all_reset_chunk_count() { pwrite_all_chunk_counter() = 0; }
inline unsigned long pwrite_all_chunk_count()       { return pwrite_all_chunk_counter(); }

// Write the entire `len`-byte buffer at file offset `off`, retrying on EINTR
// and looping on short writes. Both are permitted by POSIX; treating either as
// a hard failure can turn a transient signal into a spurious index-build abort.
// Each request is clamped to `max_chunk` (see pwrite_request_size) so a buffer
// larger than 2GiB does not trip the macOS single-write EINVAL cap. `max_chunk`
// defaults to the real 1GiB cap and is a parameter only so a test can force the
// multi-chunk path with a small cap instead of a real >2GiB buffer.
//
// Returns 0 on success, -1 for a `pwrite` that returned 0 (an unexpected
// zero-byte short write), or a positive errno on a write error. This is the
// shared core: the fatal `pwrite_all` wrapper below aborts on a non-zero return,
// while callers inside an OpenMP region (where err_fatal()->exit() would tear
// down libomp with sibling threads mid-loop) call this directly, record the
// first non-zero return, leave the region, and err_fatal outside it.
static inline int pwrite_all_status(int fd, const void* buf, size_t len, off_t off,
                                    size_t max_chunk = IO_MAX_ONCE)
{
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t remaining = len;
    pwrite_all_reset_chunk_count();
    while (remaining > 0) {
        ssize_t w = pwrite(fd, p, pwrite_request_size(remaining, max_chunk), off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return errno ? errno : EIO;
        }
        if (w == 0) return -1;   // pwrite returned 0 (distinct from an errno failure)
        ++pwrite_all_chunk_counter();   // one bump per completed write chunk (not per
                                        // EINTR retry, which does not advance the buffer)
        p         += (size_t)w;
        remaining -= (size_t)w;
        off       += (off_t)w;
    }
    return 0;
}

// Fatal wrapper: aborts via err_fatal on any failure. `what` names the buffer.
static inline void pwrite_all(int fd, const void* buf, size_t len, off_t off,
                              const char* what, size_t max_chunk = IO_MAX_ONCE)
{
    int rc = pwrite_all_status(fd, buf, len, off, max_chunk);
    if (rc < 0) err_fatal("pwrite_all", "pwrite(%s) returned 0", what);
    if (rc > 0) err_fatal("pwrite_all", "pwrite(%s) failed: %s", what, strerror(rc));
}

#endif
