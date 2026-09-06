// src/sam_encode.h
//
// SIMD-accelerated SEQ/QUAL byte encoders for SAM record building.
// Replaces the per-byte `dst[i] = "ACGTN"[src[i]]` loop in mem_aln2sam.
//
// Implementations: NEON (vqtbl1q_u8) on aarch64, SSSE3 (_mm_shuffle_epi8)
// on x86 (covers all x86 tiers >= sse41 we ship), and a scalar fallback
// when neither macro is set.
//
// Per-tier x86 builds (sse41/sse42/avx/avx2/avx512bw) are compiled from the
// same source under different `-m...` flags so the compiler may emit
// VEX/EVEX-encoded forms of the same 128-bit intrinsics + auto-vectorize
// the scalar tail. Symbols are mangled by kernel_dispatch.h.

#ifndef BWAMEM3_SAM_ENCODE_H
#define BWAMEM3_SAM_ENCODE_H

#include <stdint.h>

#include "kernel_dispatch.h"  /* must come before any project header */

#ifdef __cplusplus
extern "C" {
#endif

/* All three encoders require `dst` and `src` not to overlap: the NEON
 * implementations finish with one full 16-byte vector on the last 16 output
 * bytes (recomputing up to 15 already-written bytes from the corresponding
 * inputs), which is only correct when the inputs are not being overwritten.
 * Every caller encodes from the read's own seq/qual into a separate output
 * buffer (kstring or BAM record scratch). */
/* Encode `n` 2-bit-packed bases (ACGTN with N=4) from `src` into `n` ASCII
 * bytes in `dst` ('ACGTN'). Forward direction. */
void sam_encode_seq_fwd(char *dst, const uint8_t *src, int n);

/* Encode `n` 2-bit-packed bases reverse-complemented ('TGCAN'). */
void sam_encode_seq_rev(char *dst, const uint8_t *src, int n);

/* Reverse-copy `n` ASCII quality bytes from `src` to `dst`. */
void sam_encode_qual_rev(char *dst, const char *src, int n);

#ifdef __cplusplus
}
#endif

#endif /* BWAMEM3_SAM_ENCODE_H */
