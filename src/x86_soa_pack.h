/* SPDX-License-Identifier: MIT */
/* x86_soa_pack.h -- tiled SoA packing for the AVX2 / AVX-512BW 8-bit batch
 * wrappers (kswv mate rescue, bandedSWA extension): the x86 counterpart of
 * the tiled packing the NEON batch wrappers use (neon_soa_pack.h, which lands
 * with the NEON wrapper changes; the two headers share the same contract).
 *
 * The 8-bit kernels read their sequences in SoA form: row k holds position k
 * of every lane, SIMD_WIDTH8 bytes per row. The wrappers built that layout
 * with one strided byte store per base. Here each lane's bytes are loaded 16
 * at a time and a register transpose turns 16 positions of every lane into 16
 * SoA rows.
 *
 * The transpose is the 16x16 byte zip network (bytes, halfwords, words,
 * doublewords). On x86 the unpacklo/unpackhi instructions operate within
 * each 128-bit lane of a wider register, so one network on __m256i transposes
 * two independent 16-lane groups at once and one on __m512i four: register j
 * holds lane j in its first 128 bits, lane j+16 in its second, and so on, and
 * after the transpose register k holds SoA row k for all SIMD_WIDTH8 lanes in
 * the order the kernels expect. */
#ifndef BWAMEM3_X86_SOA_PACK_H
#define BWAMEM3_X86_SOA_PACK_H

#if defined(__AVX2__)

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

/* One 16-byte piece of the SoA input: positions [kb, kb + 16) of one lane,
 * with the pad and ambiguity contract the scalar fill wrote:
 *     seq[k]  (remapFrom -> remapTo when remap)   for k <  len
 *     padA                                         for len <= k < padStart
 *     padB                                         for k >= padStart
 * The three ranges are ordered, so the caller must pass len <= padStart
 * (x86_soa_pack asserts it): the broadcast below returns padB for a whole
 * tile once kb >= padStart, which is only right when no base lies past
 * padStart. A tile entirely inside the sequence is one load (plus the remap
 * on the whole vector); a tile entirely past padStart is a broadcast; the
 * boundary tile is assembled byte-wise so nothing is read past the sequence.
 * The remap is applied only to real bases, so pad bytes are never rewritten
 * even if padA or padB equal remapFrom. */
static inline __m128i x86_soa_piece16(const uint8_t *seq, int len, int padStart, int kb,
                                      uint8_t padA, uint8_t padB,
                                      bool remap, uint8_t remapFrom, uint8_t remapTo,
                                      __m128i fromv, __m128i tov, __m128i padBv)
{
    if (kb + 16 <= len) {
        __m128i v = _mm_loadu_si128((const __m128i *)(seq + kb));
        if (remap) v = _mm_blendv_epi8(v, tov, _mm_cmpeq_epi8(v, fromv));
        return v;
    }
    if (kb >= padStart) return padBv;
    uint8_t tmp[16];
    for (int t = 0; t < 16; t++) {
        const int k = kb + t;
        if (k < len)
            tmp[t] = (remap && seq[k] == remapFrom) ? remapTo : seq[k];
        else
            tmp[t] = (k < padStart) ? padA : padB;
    }
    return _mm_loadu_si128((const __m128i *)tmp);
}

/* 16x16 byte transpose per 128-bit lane of 16 __m256i: on entry r[j] holds
 * 16 consecutive positions of lane j (low half) and lane j+16 (high half); on
 * return r[k] holds position k of lanes 0..15 then 16..31 -- SoA row k. */
static inline void x86_transpose16x16_u8_x2(__m256i r[16])
{
    __m256i a[16];
    for (int p = 0; p < 8; p++) {                 /* lanes (2p, 2p+1): k 0-7 | 8-15 */
        a[2 * p]     = _mm256_unpacklo_epi8(r[2 * p], r[2 * p + 1]);
        a[2 * p + 1] = _mm256_unpackhi_epi8(r[2 * p], r[2 * p + 1]);
    }
    __m256i b[16];
    for (int g = 0; g < 4; g++) {                 /* lanes 4g..4g+3: k 0-3 | 4-7 | 8-11 | 12-15 */
        b[4 * g]     = _mm256_unpacklo_epi16(a[4 * g],     a[4 * g + 2]);
        b[4 * g + 1] = _mm256_unpackhi_epi16(a[4 * g],     a[4 * g + 2]);
        b[4 * g + 2] = _mm256_unpacklo_epi16(a[4 * g + 1], a[4 * g + 3]);
        b[4 * g + 3] = _mm256_unpackhi_epi16(a[4 * g + 1], a[4 * g + 3]);
    }
    __m256i c[16];
    for (int h = 0; h < 2; h++) {                 /* lanes 8h..8h+7: k pairs (0,1) .. (14,15) */
        for (int q = 0; q < 4; q++) {
            c[8 * h + 2 * q]     = _mm256_unpacklo_epi32(b[8 * h + q], b[8 * h + 4 + q]);
            c[8 * h + 2 * q + 1] = _mm256_unpackhi_epi32(b[8 * h + q], b[8 * h + 4 + q]);
        }
    }
    for (int m = 0; m < 8; m++) {                 /* lanes 0-7 | 8-15 -> rows 2m, 2m+1 */
        r[2 * m]     = _mm256_unpacklo_epi64(c[m], c[8 + m]);
        r[2 * m + 1] = _mm256_unpackhi_epi64(c[m], c[8 + m]);
    }
}

#if defined(__AVX512BW__)
/* Same network on __m512i: four 16-lane groups per register (lanes j, j+16,
 * j+32, j+48), so r[k] is the 64-byte SoA row k. */
static inline void x86_transpose16x16_u8_x4(__m512i r[16])
{
    __m512i a[16];
    for (int p = 0; p < 8; p++) {
        a[2 * p]     = _mm512_unpacklo_epi8(r[2 * p], r[2 * p + 1]);
        a[2 * p + 1] = _mm512_unpackhi_epi8(r[2 * p], r[2 * p + 1]);
    }
    __m512i b[16];
    for (int g = 0; g < 4; g++) {
        b[4 * g]     = _mm512_unpacklo_epi16(a[4 * g],     a[4 * g + 2]);
        b[4 * g + 1] = _mm512_unpackhi_epi16(a[4 * g],     a[4 * g + 2]);
        b[4 * g + 2] = _mm512_unpacklo_epi16(a[4 * g + 1], a[4 * g + 3]);
        b[4 * g + 3] = _mm512_unpackhi_epi16(a[4 * g + 1], a[4 * g + 3]);
    }
    __m512i c[16];
    for (int h = 0; h < 2; h++) {
        for (int q = 0; q < 4; q++) {
            c[8 * h + 2 * q]     = _mm512_unpacklo_epi32(b[8 * h + q], b[8 * h + 4 + q]);
            c[8 * h + 2 * q + 1] = _mm512_unpackhi_epi32(b[8 * h + q], b[8 * h + 4 + q]);
        }
    }
    for (int m = 0; m < 8; m++) {
        r[2 * m]     = _mm512_unpacklo_epi64(c[m], c[8 + m]);
        r[2 * m + 1] = _mm512_unpackhi_epi64(c[m], c[8 + m]);
    }
}
#endif /* __AVX512BW__ */

/* Pack W lanes' byte sequences into the W-lane SoA layout (row k = position
 * k of every lane, W bytes per row, rows [0, nrows) written), 16 rows per
 * transposed tile instead of one strided byte store per base. W is the
 * kernel's SIMD_WIDTH8: 32 on the AVX2 tier, 64 on AVX-512BW; W == 64 is
 * accepted only in an AVX-512BW build, where the 64-lane path exists (an AVX2
 * build rejects it at compile time rather than packing half the lanes).
 * `soa` is the kernel's 64-byte-aligned SoA buffer, so every row store is
 * aligned. Every lane needs len[j] <= padStart[j] (see x86_soa_piece16).
 * Byte-for-byte the same SoA the scalar fill produced (see x86_soa_piece16
 * for the per-lane contract). */
template <int W>
static inline void x86_soa_pack(uint8_t *soa,
                                const uint8_t *const seq[W],
                                const int len[W],
                                const int padStart[W],
                                int nrows, uint8_t padA, uint8_t padB,
                                bool remap, uint8_t remapFrom, uint8_t remapTo)
{
#if defined(__AVX512BW__)
    static_assert(W == 32 || W == 64, "x86_soa_pack: W must be the AVX2 (32) or AVX-512BW (64) lane count");
#else
    static_assert(W == 32, "x86_soa_pack: W == 64 needs an AVX-512BW build; this build packs 32 lanes");
#endif
    for (int j = 0; j < W; j++) assert(len[j] <= padStart[j]);
    const __m128i fromv = _mm_set1_epi8((char)remapFrom);
    const __m128i tov   = _mm_set1_epi8((char)remapTo);
    const __m128i padBv = _mm_set1_epi8((char)padB);
    for (int kb = 0; kb < nrows; kb += 16) {
        const int kend = (kb + 16 < nrows) ? kb + 16 : nrows;
#if defined(__AVX512BW__)
        if (W == 64) {
            __m512i t[16];
            for (int j = 0; j < 16; j++) {
                __m512i v = _mm512_castsi128_si512(
                    x86_soa_piece16(seq[j], len[j], padStart[j], kb, padA, padB, remap, remapFrom, remapTo, fromv, tov, padBv));
                v = _mm512_inserti32x4(v, x86_soa_piece16(seq[j + 16], len[j + 16], padStart[j + 16], kb, padA, padB, remap, remapFrom, remapTo, fromv, tov, padBv), 1);
                v = _mm512_inserti32x4(v, x86_soa_piece16(seq[j + 32], len[j + 32], padStart[j + 32], kb, padA, padB, remap, remapFrom, remapTo, fromv, tov, padBv), 2);
                v = _mm512_inserti32x4(v, x86_soa_piece16(seq[j + 48], len[j + 48], padStart[j + 48], kb, padA, padB, remap, remapFrom, remapTo, fromv, tov, padBv), 3);
                t[j] = v;
            }
            x86_transpose16x16_u8_x4(t);
            for (int k = kb; k < kend; k++)
                _mm512_store_si512((__m512i *)(soa + (size_t)k * W), t[k - kb]);
            continue;
        }
#endif
        {
            __m256i t[16];
            for (int j = 0; j < 16; j++) {
                const __m128i lo = x86_soa_piece16(seq[j],      len[j],      padStart[j],      kb, padA, padB, remap, remapFrom, remapTo, fromv, tov, padBv);
                const __m128i hi = x86_soa_piece16(seq[j + 16], len[j + 16], padStart[j + 16], kb, padA, padB, remap, remapFrom, remapTo, fromv, tov, padBv);
                t[j] = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
            }
            x86_transpose16x16_u8_x2(t);
            for (int k = kb; k < kend; k++)
                _mm256_store_si256((__m256i *)(soa + (size_t)k * W), t[k - kb]);
        }
    }
}

#endif /* __AVX2__ */
#endif /* BWAMEM3_X86_SOA_PACK_H */
