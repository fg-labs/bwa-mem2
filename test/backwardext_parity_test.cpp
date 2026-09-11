/* Isolated parity gate for the full backwardExt (k, l, s) against a scalar
 * reference computed straight from the checkpoint blocks.
 *
 * backwardExt has an arm64 vector path (all four bases' occurrence counts in
 * SIMD lanes) and a scalar path; both must reproduce, for every (sp, s, a),
 *   occ(b, p) = cp_count[b] + popcount(one_hot_bwt_str[b] & mask[p & CP_MASK])
 *   k' = count[a] + occ(a, sp)
 *   s' = occ(a, ep) - occ(a, sp)                       (ep = sp + s)
 *   l' = l + [sp <= sentinel < ep] + sum_{b > a} (occ(b, ep) - occ(b, sp))
 * The sweep covers every in-block offset at both block edges for sp, several
 * interval sizes (so sp and ep fall in the same block and in different
 * blocks), and all four bases, so every lane of the vector path and the
 * suffix sum feeding l are checked against the portable formula. The
 * reference popcount is the compiler builtin on a GPR value, independent of
 * any SIMD lowering.
 *
 * Usage: backwardext_parity_test <bwa-mem3 index prefix>
 */

#define private public
#include "FMI_search.h"
#undef private
#include "bwa.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int64_t ref_occ(const FMI_search *fmi, int b, int64_t p) {
    const CP_OCC &blk = fmi->cp_occ[p >> CP_SHIFT];
    const uint64_t m = one_hot_mask_array[p & CP_MASK];
    return blk.cp_count[b] + (int64_t) __builtin_popcountll(blk.one_hot_bwt_str[b] & m);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <bwa-mem3 index prefix>\n", argv[0]);
        return 2;
    }
    FMI_search *fmi = new FMI_search(argv[1]);
    fmi->load_index();

    const int64_t ref_len = fmi->reference_seq_len;
    const int64_t sentinel = fmi->sentinel_index;
    fprintf(stderr, "backwardext_parity_test: reference_seq_len=%lld sentinel_index=%lld CP_MASK=%d\n",
            (long long) ref_len, (long long) sentinel, (int) CP_MASK);

    const int64_t bases[] = {0, CP_BLOCK_SIZE, 10 * CP_BLOCK_SIZE, ref_len / 2,
                             sentinel - (sentinel & CP_MASK), ref_len - 2 * CP_BLOCK_SIZE};
    const int32_t s_values[] = {1, 2, 3, 5, 8, 17, 63, 64, 65, 200};

    int64_t checked = 0, k_mismatch = 0, l_mismatch = 0, s_mismatch = 0;
    for (size_t bi = 0; bi < sizeof(bases) / sizeof(bases[0]); bi++) {
        for (int32_t off = 0; off <= (int32_t) CP_MASK; off++) {
            const int64_t sp = bases[bi] + off;
            if (sp < 0) continue;
            for (size_t si = 0; si < sizeof(s_values) / sizeof(s_values[0]); si++) {
                const int32_t s = s_values[si];
                const int64_t ep = sp + s;
                if (ep >= ref_len) continue;
                for (uint8_t a = 0; a < 4; a++) {
                    SMEM smem;
                    smem.rid = 0; smem.m = 0; smem.n = 0;
                    smem.k = sp; smem.l = 777 + off; smem.s = s;

                    const SMEM got = fmi->backwardExt(smem, a);

                    const int64_t occ_s = ref_occ(fmi, a, sp);
                    const int64_t want_k = fmi->count[a] + occ_s;
                    const int64_t want_s = ref_occ(fmi, a, ep) - occ_s;
                    int64_t want_l = smem.l + ((sp <= sentinel && ep > sentinel) ? 1 : 0);
                    for (int b = a + 1; b < 4; b++) want_l += ref_occ(fmi, b, ep) - ref_occ(fmi, b, sp);

                    checked++;
                    if (got.k != want_k) { if (k_mismatch < 10) fprintf(stderr, "[FAIL] k sp=%lld s=%d a=%d got=%lld want=%lld\n", (long long) sp, s, a, (long long) got.k, (long long) want_k); k_mismatch++; }
                    if (got.s != want_s) { if (s_mismatch < 10) fprintf(stderr, "[FAIL] s sp=%lld s=%d a=%d got=%lld want=%lld\n", (long long) sp, s, a, (long long) got.s, (long long) want_s); s_mismatch++; }
                    if (got.l != want_l) { if (l_mismatch < 10) fprintf(stderr, "[FAIL] l sp=%lld s=%d a=%d got=%lld want=%lld\n", (long long) sp, s, a, (long long) got.l, (long long) want_l); l_mismatch++; }
                }
            }
        }
    }
    fprintf(stderr, "checked=%lld  k_mismatch=%lld  s_mismatch=%lld  l_mismatch=%lld\n",
            (long long) checked, (long long) k_mismatch, (long long) s_mismatch, (long long) l_mismatch);
    delete fmi;
    if (checked == 0) { fprintf(stderr, "[FAIL] no comparisons ran\n"); return 1; }
    if (k_mismatch || s_mismatch || l_mismatch) { fprintf(stderr, "backwardExt PARITY FAILED\n"); return 1; }
    fprintf(stderr, "backwardExt PARITY PASS\n");
    return 0;
}
