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

   SIMD Compatibility Header for Apple Silicon (ARM64/NEON)
   Based on PR #281 from BenjaminDEMAILLE
*****************************************************************************************/

#ifndef SIMD_COMPAT_H
#define SIMD_COMPAT_H

/*
 * ARM/sse2neon compatibility shim -- ARM targets ONLY.
 *
 * This header is the sse2neon-based SSE->NEON translation layer plus the
 * native-NEON hot-path helper (_mm_movemask_epi16). It is NOT a cross-tier SIMD
 * abstraction: every
 * consumer includes it only under `#if defined(__ARM_NEON) || defined(__aarch64__)`,
 * so on x86 it is never seen and x86 uses <immintrin.h> directly. The `#else`
 * guard below enforces that invariant -- do not add x86/AVX/SSE branches here.
 */

#if defined(__ARM_NEON) || defined(__aarch64__)
    /* ARM/Apple Silicon */
    #define APPLE_SILICON 1
    #define SIMDE_ENABLE_NATIVE_ALIASES
    #include "sse2neon.h"
    #include <arm_neon.h>

    /* Define SIMD widths for NEON (128-bit) */
    #ifndef SIMD_WIDTH8
    #define SIMD_WIDTH8 16   /* 128-bit / 8-bit = 16 elements */
    #endif
    #ifndef SIMD_WIDTH16
    #define SIMD_WIDTH16 8   /* 128-bit / 16-bit = 8 elements */
    #endif

    /* Memory allocation compatibility.
     * Default cache-line size is 128 (Apple Silicon). A build targeting a
     * 64-byte-line ARM core (e.g. Graviton4 / Neoverse V2) overrides this on the
     * command line via -DCACHE_LINE_BYTES=64; the guard lets that value win. */
    #ifndef CACHE_LINE_BYTES
    #define CACHE_LINE_BYTES 128  /* Apple Silicon uses 128-byte cache lines */
    #endif

    static inline void* _mm_malloc_compat(size_t size, size_t align) {
        void* ptr = NULL;
        if (posix_memalign(&ptr, align, size) != 0) {
            return NULL;
        }
        return ptr;
    }

    static inline void _mm_free_compat(void* ptr) {
        free(ptr);
    }

    /* Use compatibility functions on ARM
     * Apple Silicon uses 128-byte cache lines, so we enforce minimum 128-byte
     * alignment for all SIMD allocations (vs 64-byte on x86) to avoid false sharing */
    #define _mm_malloc(size, align) _mm_malloc_compat(size, (align) < CACHE_LINE_BYTES ? CACHE_LINE_BYTES : (align))
    #define _mm_free(ptr) _mm_free_compat(ptr)

    /* Prefetch compatibility.
     *
     * Do NOT pass the SSE hint enum straight through as GCC's locality argument:
     * the two scales are different, and for _MM_HINT_T0 they actively disagree.
     *
     *   SSE hint:            NTA=0    T0=1 (L1)   T1=2 (L2)   T2=3 (L3)
     *   __builtin_prefetch:  0=stream 1=L3        2=L2        3=L1
     *
     * Passing the enum through maps T0 -> locality 1 -> `prfm pldl3keep`, i.e. the
     * one hint that means "bring this to L1" was the one that landed in L3. NTA and
     * T1 happen to coincide on both scales, so only T0 was wrong -- which is the hint
     * used by the FM-index walk, whose entire lockstep design exists to make those
     * lines L1-resident by the time the dependent load issues.
     *
     * Measured before this fix on arm64 (objdump src/FMI_search.o): 47 pldl3keep,
     * 12 pldl2keep, ZERO pldl1keep.
     *
     * Note sse2neon does provide a correct _mm_prefetch, but as a FORCE_INLINE
     * *function*, so `#ifndef _mm_prefetch` does not see it and this macro shadowed
     * it. Mapping explicitly here is correct whether or not sse2neon is in play.
     *
     * Prefetches are pure hints: this cannot change output bytes. */
    #ifndef _mm_prefetch
    #define BWA3_PF_LOCALITY(hint) \
        ((hint) == 0 ? 0 : (hint) == 1 ? 3 : (hint) == 2 ? 2 : 1)
    #define _mm_prefetch(addr, hint) \
        __builtin_prefetch((const void*)(addr), 0, BWA3_PF_LOCALITY(hint))
    #endif

    /* Mask types for SSE compatibility (sse2neon should provide these) */
    #ifndef __mmask8
    typedef uint8_t __mmask8;
    #endif
    #ifndef __mmask16
    typedef uint16_t __mmask16;
    #endif
    #ifndef __mmask32
    typedef uint32_t __mmask32;
    #endif
    #ifndef __mmask64
    typedef uint64_t __mmask64;
    #endif

    /* __rdtsc compatibility - sse2neon provides _rdtsc */
    #ifndef __rdtsc
    #define __rdtsc _rdtsc
    #endif

    /*
     * Optimized movemask for 16-bit elements (used heavily in bandedSWA.cpp)
     * Instead of _mm_movemask_epi8(v) & 0xAAAA, use _mm_movemask_epi16(v) directly.
     * This extracts the MSB of each 16-bit element into an 8-bit result.
     */
    static inline int _mm_movemask_epi16(__m128i v) {
        /* Broadcast each 16-bit lane's MSB across the lane (0xFFFF/0x0000) via an
         * arithmetic right shift, then narrow to 0xFF/0x00 per byte.
         * NOTE: a *logical* shift (vshrq_n_u16) yields 0x0001, and after narrowing
         * 0x01 & (1<<lane) is 0 for every lane except 0 -- that silently collapses
         * the mask. The signed shift is required for vand to match the old vmul. */
        uint16x8_t msb = vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_m128i(v), 15));
        /* Narrow to 8-bit (low byte of each 16-bit element is now 0xFF/0x00) */
        uint8x8_t narrow = vmovn_u16(msb);
        /* Weight each lane's mask by its bit position, then horizontal add.
         * vand against a compile-time {1<<lane} literal (materialized via MOVI,
         * 1-cycle) replaces vmul_u8 + a data-section weights load (3-cycle). */
        const uint8x8_t bit_mask = {1, 2, 4, 8, 16, 32, 64, 128};
        return (int)vaddv_u8(vand_u8(narrow, bit_mask));
    }

#else
    /* This header is the ARM/sse2neon compatibility shim only. It must never be
     * compiled on a non-ARM target: on x86 the consumers include <immintrin.h>
     * directly and never reach this header. A non-ARM include here means a guard
     * regressed -- fail loudly rather than silently first-activating dead code. */
    #error "simd_compat.h is ARM-only; include it under #if defined(__ARM_NEON) || defined(__aarch64__)"
#endif

#endif /* SIMD_COMPAT_H */
