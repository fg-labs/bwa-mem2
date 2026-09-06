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

   NEON helpers for the arm64 kernels: the saturating signed-into-unsigned add
   and the byte movemask; everything else uses arm_neon.h directly
*****************************************************************************************/

#ifndef NEON_UTILS_H
#define NEON_UTILS_H

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(APPLE_SILICON)

#include <arm_neon.h>
#include <stdint.h>

/* Only the helpers with live callers are kept here. The header once carried a
 * full NEON macro set (loads, stores, saturating arithmetic, compares, blends,
 * shifts, horizontal maxima, shuffles, prefetch, aligned allocation) that no
 * kernel ever used; the blend macros also had the select polarity inverted
 * relative to the _mm_blendv_epi8 they claimed to match. Kernels use the
 * arm_neon.h intrinsics directly, which is what the compiler sees anyway. */

/* Unsigned-saturating add of a SIGNED addend into an unsigned accumulator:
 * clamp(a + (int8)b, 0, 255). NEON's USQADD; `b` is passed as uint8x16_t and
 * reinterpreted, matching how the scoring table is laid out. The u8 mate-rescue
 * diagonal uses it to fold the de-bias vqsubq_u8 out of the h00->m11 chain. */
#define NEON_SQADD_U8(a, b)  vsqaddq_u8((a), vreinterpretq_s8_u8(b))

/*
 * Extract high bit from each byte (movemask equivalent)
 * NEON doesn't have a direct movemask, so we need to implement it
 */
static inline uint16_t neon_movemask_u8(uint8x16_t v) {
    /* Per-byte positional weights (1<<lane), repeated across both 8-byte
     * halves. A compile-time vector literal, so it materializes via MOVI
     * immediates rather than a data-section load. */
    const uint8x16_t bit_mask = {1, 2, 4, 8, 16, 32, 64, 128,
                                 1, 2, 4, 8, 16, 32, 64, 128};

    /* Broadcast each byte's high bit across the whole lane (0xFF/0x00) via an
     * arithmetic right shift, then keep that lane's positional weight with vand.
     * NOTE: a *logical* shift (vshrq_n_u8) yields 0x01, and 0x01 & (1<<lane) is
     * 0 for every lane except 0 and 8 -- that silently collapses the mask. The
     * signed shift is required for vand to be equivalent to the vshl packing. */
    uint8x16_t hi = vreinterpretq_u8_s8(vshrq_n_s8(vreinterpretq_s8_u8(v), 7));
    uint8x16_t bits = vandq_u8(hi, bit_mask);

    /* Sum each 8-byte half: low half -> low byte of result, high -> high. */
    return (uint16_t)vaddv_u8(vget_low_u8(bits)) |
           ((uint16_t)vaddv_u8(vget_high_u8(bits)) << 8);
}

#endif /* __ARM_NEON || __aarch64__ || APPLE_SILICON */

#endif /* NEON_UTILS_H */
