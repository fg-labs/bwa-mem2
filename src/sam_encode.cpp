// src/sam_encode.cpp
//
// SIMD-accelerated SEQ/QUAL byte encoders for SAM record building.
// Moved from bwamem.cpp; symbols renamed to their public sam_encode_* names.

#include "kernel_dispatch.h"
#include "sam_encode.h"
#include <stdint.h>

/* ─── SIMD-accelerated SEQ/QUAL encoders for SAM record building ──────────
 * Replaces the per-byte `dst[i] = "ACGTN"[src[i]]` loop in mem_aln2sam.
 * Works on 16 bytes at a time via a byte-LUT shuffle. Three implementations
 * compiled in: NEON (vqtbl1q_u8) on aarch64, SSSE3 (_mm_shuffle_epi8) on
 * x86 — covers SSSE3, SSE4, AVX2, AVX-512BW — and a scalar fallback for
 * everything else. AVX2/AVX-512BW could process 32/64 bytes per iter but
 * the inner loop is already short for typical 150-bp reads, so the SSSE3
 * 16-byte width is the right compromise of code volume vs. perf. */
#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#  define SAM_FAST_IMPL 1   /* NEON */
#elif defined(__SSSE3__) || defined(__SSE4_1__) || defined(__AVX__) \
   || defined(__AVX2__) || defined(__AVX512BW__)
#  include <tmmintrin.h>    /* SSSE3 _mm_shuffle_epi8 */
#  define SAM_FAST_IMPL 2   /* SSSE3+ */
#endif

#ifdef SAM_FAST_IMPL
static const uint8_t enc_fwd_lut[16] = {
    'A','C','G','T','N','N','N','N',
    'N','N','N','N','N','N','N','N',
};
static const uint8_t enc_rev_lut[16] = {
    'T','G','C','A','N','N','N','N',
    'N','N','N','N','N','N','N','N',
};
#endif

/* Scalar tail clamp — mirrors the SIMD path's vminq_u8/_mm_min_epu8 against 4
 * so any code >= 5 in src[] decodes to 'N' instead of indexing OOB into the
 * 5-byte "ACGTN"/"TGCAN" literals. Today s->seq is always 2-bit ACGTN before
 * mem_aln2sam, so this is latent — but it keeps SIMD and scalar paths in
 * lockstep if that invariant ever drifts. */
#define SAM_NT_CLAMP4(c) ((unsigned)(c) <= 4u ? (unsigned)(c) : 4u)

extern "C" {

#if SAM_FAST_IMPL == 1
/* NEON 16-byte: vqtbl1q_u8 LUT + vminq_u8 clamp; the reversed encoders
 * reverse each 16-byte chunk with a single vqtbl1q_u8 against a {15..0}
 * index vector (one op, where vextq + vrev64q was two). */
static const uint8_t rev16_idx[16] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};
void sam_encode_seq_fwd(char *dst, const uint8_t *src, int n) {
    const uint8x16_t lut = vld1q_u8(enc_fwd_lut);
    const uint8x16_t four = vdupq_n_u8(4);
    int i = 0;
    while (i + 16 <= n) {
        uint8x16_t v   = vld1q_u8(src + i);
        uint8x16_t cl  = vminq_u8(v, four);
        uint8x16_t out = vqtbl1q_u8(lut, cl);
        vst1q_u8((uint8_t*)dst + i, out);
        i += 16;
    }
    if (i < n) {
        if (n >= 16) {
            /* Tail as one overlapped vector: the last 16 input bytes map to
             * the last 16 output bytes, and the bytes recomputed under the
             * overlap are the same pure function of the same inputs, so they
             * are rewritten with identical values. Load and store both stay
             * inside [0, n). */
            uint8x16_t v   = vld1q_u8(src + n - 16);
            uint8x16_t cl  = vminq_u8(v, four);
            uint8x16_t out = vqtbl1q_u8(lut, cl);
            vst1q_u8((uint8_t*)dst + n - 16, out);
        } else {
            while (i < n) { dst[i] = "ACGTN"[SAM_NT_CLAMP4(src[i])]; ++i; }
        }
    }
}
void sam_encode_seq_rev(char *dst, const uint8_t *src, int n) {
    const uint8x16_t lut = vld1q_u8(enc_rev_lut);
    const uint8x16_t four = vdupq_n_u8(4);
    const uint8x16_t rev = vld1q_u8(rev16_idx);
    int i = 0;
    while (i + 16 <= n) {
        uint8x16_t v = vld1q_u8(src + n - 16 - i);
        v = vqtbl1q_u8(v, rev);
        uint8x16_t cl  = vminq_u8(v, four);
        uint8x16_t out = vqtbl1q_u8(lut, cl);
        vst1q_u8((uint8_t*)dst + i, out);
        i += 16;
    }
    if (i < n) {
        if (n >= 16) {
            /* Overlapped tail: the first 16 input bytes, reversed, are the
             * last 16 output bytes (see sam_encode_seq_fwd). */
            uint8x16_t v = vld1q_u8(src);
            v = vqtbl1q_u8(v, rev);
            uint8x16_t cl  = vminq_u8(v, four);
            uint8x16_t out = vqtbl1q_u8(lut, cl);
            vst1q_u8((uint8_t*)dst + n - 16, out);
        } else {
            while (i < n) { dst[i] = "TGCAN"[SAM_NT_CLAMP4(src[n - 1 - i])]; ++i; }
        }
    }
}
void sam_encode_qual_rev(char *dst, const char *src, int n) {
    const uint8x16_t rev = vld1q_u8(rev16_idx);
    int i = 0;
    while (i + 16 <= n) {
        uint8x16_t v = vld1q_u8((const uint8_t*)(src + n - 16 - i));
        v = vqtbl1q_u8(v, rev);
        vst1q_u8((uint8_t*)dst + i, v);
        i += 16;
    }
    if (i < n) {
        if (n >= 16) {
            /* Overlapped tail (see sam_encode_seq_fwd). */
            uint8x16_t v = vld1q_u8((const uint8_t*)src);
            v = vqtbl1q_u8(v, rev);
            vst1q_u8((uint8_t*)dst + n - 16, v);
        } else {
            while (i < n) { dst[i] = src[n - 1 - i]; ++i; }
        }
    }
}
#elif SAM_FAST_IMPL == 2
/* SSSE3 16-byte: _mm_shuffle_epi8 LUT + _mm_min_epu8 clamp + reverse-mask
 * shuffle for the reverse direction. Works on every x86 since 2006 (SSSE3
 * is in Core 2 and later); AVX2/AVX-512 hosts pick this path too. */
static const uint8_t rev16_mask[16] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};
void sam_encode_seq_fwd(char *dst, const uint8_t *src, int n) {
    const __m128i lut  = _mm_loadu_si128((const __m128i*)enc_fwd_lut);
    const __m128i four = _mm_set1_epi8(4);
    int i = 0;
    while (i + 16 <= n) {
        __m128i v   = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i cl  = _mm_min_epu8(v, four);
        __m128i out = _mm_shuffle_epi8(lut, cl);
        _mm_storeu_si128((__m128i*)(dst + i), out);
        i += 16;
    }
    if (i < n) {
        if (n >= 16) {
            /* Tail as one overlapped vector, as on NEON: the last 16 input
             * bytes map to the last 16 output bytes, and the bytes recomputed
             * under the overlap are the same pure function of the same
             * inputs, so they are rewritten with identical values. Load and
             * store both stay inside [0, n). */
            __m128i v   = _mm_loadu_si128((const __m128i*)(src + n - 16));
            __m128i cl  = _mm_min_epu8(v, four);
            __m128i out = _mm_shuffle_epi8(lut, cl);
            _mm_storeu_si128((__m128i*)(dst + n - 16), out);
        } else {
            while (i < n) { dst[i] = "ACGTN"[SAM_NT_CLAMP4(src[i])]; ++i; }
        }
    }
}
void sam_encode_seq_rev(char *dst, const uint8_t *src, int n) {
    const __m128i lut  = _mm_loadu_si128((const __m128i*)enc_rev_lut);
    const __m128i four = _mm_set1_epi8(4);
    const __m128i rev  = _mm_loadu_si128((const __m128i*)rev16_mask);
    int i = 0;
    while (i + 16 <= n) {
        __m128i v   = _mm_loadu_si128((const __m128i*)(src + n - 16 - i));
        v           = _mm_shuffle_epi8(v, rev);   /* reverse all 16 bytes */
        __m128i cl  = _mm_min_epu8(v, four);
        __m128i out = _mm_shuffle_epi8(lut, cl);
        _mm_storeu_si128((__m128i*)(dst + i), out);
        i += 16;
    }
    if (i < n) {
        if (n >= 16) {
            /* Overlapped tail: the first 16 input bytes, reversed, are the
             * last 16 output bytes (see sam_encode_seq_fwd). */
            __m128i v   = _mm_loadu_si128((const __m128i*)src);
            v           = _mm_shuffle_epi8(v, rev);
            __m128i cl  = _mm_min_epu8(v, four);
            __m128i out = _mm_shuffle_epi8(lut, cl);
            _mm_storeu_si128((__m128i*)(dst + n - 16), out);
        } else {
            while (i < n) { dst[i] = "TGCAN"[SAM_NT_CLAMP4(src[n - 1 - i])]; ++i; }
        }
    }
}
void sam_encode_qual_rev(char *dst, const char *src, int n) {
    const __m128i rev = _mm_loadu_si128((const __m128i*)rev16_mask);
    int i = 0;
    while (i + 16 <= n) {
        __m128i v = _mm_loadu_si128((const __m128i*)(src + n - 16 - i));
        v = _mm_shuffle_epi8(v, rev);
        _mm_storeu_si128((__m128i*)(dst + i), v);
        i += 16;
    }
    if (i < n) {
        if (n >= 16) {
            /* Overlapped tail (see sam_encode_seq_fwd). */
            __m128i v = _mm_loadu_si128((const __m128i*)src);
            v = _mm_shuffle_epi8(v, rev);
            _mm_storeu_si128((__m128i*)(dst + n - 16), v);
        } else {
            while (i < n) { dst[i] = src[n - 1 - i]; ++i; }
        }
    }
}
#else
/* Scalar fallback (no SIMD available). */
void sam_encode_seq_fwd(char *dst, const uint8_t *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = "ACGTN"[SAM_NT_CLAMP4(src[i])];
}
void sam_encode_seq_rev(char *dst, const uint8_t *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = "TGCAN"[SAM_NT_CLAMP4(src[n - 1 - i])];
}
void sam_encode_qual_rev(char *dst, const char *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[n - 1 - i];
}
#endif

}  /* extern "C" */
