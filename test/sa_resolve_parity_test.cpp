/* Isolated parity gate for the pipelined compressed-SA resolver
 * (FMI_search::get_sa_entries_prefetch, stepping each lane with call_one_step)
 * against a scalar LF walk computed straight from the checkpoint blocks.
 *
 * Every SA row that is not a sampled row (row & SA_COMPX_MASK != 0) is
 * resolved by walking LF until a sampled row is reached:
 *   b        = the one-hot BWT symbol at row sp (bit CP_MASK - (sp & CP_MASK))
 *   occ(b,p) = cp_count[b] + popcount(one_hot_bwt_str[b] & mask[p & CP_MASK])
 *   sp'      = count[b] + occ(b, sp),   offset' = offset + 1
 * and the answer is sa[sp] + offset. A row with no symbol set is the sentinel
 * row: its suffix-array value is 0 and the walk took `offset` LF steps to reach
 * it, so the resolver reports 0 + offset (the accumulated offset). That is the
 * production contract call_one_step and get_sa_entry_compressed share; the
 * --compat=bwa-mem2 arm (drop_sentinel_offset) instead reports 0.
 *
 * The sweep runs every row of the fixture through the pipeline as an s = 1
 * interval, then as strided multi-occurrence intervals (s > max_occ, so the
 * staging loop's stride and cap are exercised), and compares every returned
 * coordinate with the scalar walk. The reference popcount is the compiler
 * builtin on a GPR value, independent of any SIMD lowering.
 *
 * Usage: sa_resolve_parity_test <bwa-mem3 index prefix>
 */

#define private public
#include "FMI_search.h"
#undef private
#include "bwa.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

/* Scalar LF walk with the pipeline's contract. Returns the resolved text
 * position; at the sentinel row that is the accumulated LF `offset` (sa value 0
 * plus the walk), matching call_one_step's default branch (sets *hit_sentinel). */
static int64_t ref_resolve(const FMI_search *fmi, int64_t pos, int *hit_sentinel) {
    int64_t sp = pos, offset = 0;
    *hit_sentinel = 0;
    while ((sp & fmi->sa_compx_mask) != 0) {
        const CP_OCC &blk = fmi->cp_occ[sp >> CP_SHIFT];
        const int y = (int)(CP_MASK - (sp & CP_MASK));
        int b = 4;
        for (int c = 0; c < 4; c++)
            if ((blk.one_hot_bwt_str[c] >> y) & 1ULL) { b = c; break; }
        if (b == 4) { *hit_sentinel = 1; return offset; }
        const int64_t occ = blk.cp_count[b] +
            (int64_t) __builtin_popcountll(blk.one_hot_bwt_str[b] & one_hot_mask_array[sp & CP_MASK]);
        sp = fmi->count[b] + occ;
        offset++;
    }
    int64_t sa = fmi->sa_ms_byte[sp >> fmi->sa_compx];
    sa = (sa << 32) + fmi->sa_ls_word[sp >> fmi->sa_compx];
    return sa + offset;
}

/* Run one batch of intervals through the pipeline and check every coordinate
 * against the scalar walk, mirroring the staging rule (stride s / max_occ,
 * at most max_occ entries per interval). */
static void check_batch(FMI_search *fmi, std::vector<SMEM> &smems, int32_t max_occ,
                        int64_t &checked, int64_t &mismatches, int64_t &sentinel_walks) {
    int64_t total = 0;
    for (size_t i = 0; i < smems.size(); i++)
        total += smems[i].s > max_occ ? max_occ : smems[i].s;
    std::vector<int64_t> coords((size_t) total + 1, -1);
    int64_t coord_count = 0, id = 0;
    fmi->get_sa_entries_prefetch(smems.data(), coords.data(), &coord_count,
                                 (int64_t) smems.size(), max_occ, 0, id);
    if (coord_count != total || id != total) {
        fprintf(stderr, "  staging count mismatch: expected %lld, coord_count=%lld id=%lld\n",
                (long long) total, (long long) coord_count, (long long) id);
        mismatches++;
        return;
    }
    int64_t c = 0;
    for (size_t i = 0; i < smems.size(); i++) {
        const SMEM &m = smems[i];
        const int64_t step = m.s > max_occ ? m.s / max_occ : 1;
        int32_t n = 0;
        for (int64_t j = m.k; j < m.k + m.s && n < max_occ; j += step, n++, c++) {
            int hit = 0;
            const int64_t expect = ref_resolve(fmi, j, &hit);
            sentinel_walks += hit;
            checked++;
            if (coords[(size_t) c] != expect) {
                if (mismatches < 10)
                    fprintf(stderr, "  MISMATCH row=%lld got=%lld expected=%lld\n",
                            (long long) j, (long long) coords[(size_t) c], (long long) expect);
                mismatches++;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <bwa-mem3 index prefix>\n", argv[0]);
        return 2;
    }
    FMI_search *fmi = new FMI_search(argv[1]);
    fmi->load_index();

    const int64_t ref_len = fmi->reference_seq_len;
    fprintf(stderr, "sa_resolve_parity_test: reference_seq_len=%lld sentinel_index=%lld SA_COMPX=%d\n",
            (long long) ref_len, (long long) fmi->sentinel_index, (int) SA_COMPX);

    int64_t checked = 0, mismatches = 0, sentinel_walks = 0;

    /* 1. Every row as its own s = 1 interval: exercises every residue mod the
     * SA sampling, every in-block offset, and every walk length. */
    {
        std::vector<SMEM> smems((size_t) ref_len);
        for (int64_t k = 0; k < ref_len; k++) {
            SMEM m = {}; m.k = k; m.l = 0; m.s = 1; smems[(size_t) k] = m;
        }
        check_batch(fmi, smems, 500, checked, mismatches, sentinel_walks);
    }

    /* 2. Strided multi-occurrence intervals (s > max_occ): the staging loop's
     * stride and cap, with several caps so the stride takes different values,
     * plus intervals that end exactly at the last row. */
    {
        const int32_t caps[] = {1, 2, 7, 20, 64, 500};
        for (size_t ci = 0; ci < sizeof(caps) / sizeof(caps[0]); ci++) {
            std::vector<SMEM> smems;
            const int64_t sizes[] = {1, 2, 3, 19, 20, 21, 64, 65, 1000, ref_len / 3, ref_len - 1};
            for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
                for (int64_t k0 = 0; k0 + sizes[si] <= ref_len; k0 += (ref_len / 7) + 1) {
                    SMEM m = {}; m.k = k0; m.s = sizes[si]; smems.push_back(m);
                }
                SMEM tail = {}; tail.k = ref_len - sizes[si]; tail.s = sizes[si]; smems.push_back(tail);
            }
            check_batch(fmi, smems, caps[ci], checked, mismatches, sentinel_walks);
        }
    }

    /* 3. A small batch and an empty one: fewer intervals than pipeline lanes,
     * and no work at all. */
    {
        std::vector<SMEM> smems;
        SMEM m = {}; m.k = ref_len / 2 + 1; m.s = 3; smems.push_back(m);
        check_batch(fmi, smems, 500, checked, mismatches, sentinel_walks);
        std::vector<SMEM> none;
        check_batch(fmi, none, 500, checked, mismatches, sentinel_walks);
    }

    fprintf(stderr, "sa_resolve_parity_test: checked=%lld mismatches=%lld sentinel_walks=%lld\n",
            (long long) checked, (long long) mismatches, (long long) sentinel_walks);
    delete fmi;
    if (mismatches != 0) {
        printf("SA resolve PARITY FAIL (%lld mismatches)\n", (long long) mismatches);
        return 1;
    }
    /* The sentinel-row branch is the one real branch left in the arm64 step;
     * the fixture must reach it (phix does, from three rows) or the sweep is
     * not covering it. */
    if (sentinel_walks == 0) {
        printf("SA resolve PARITY FAIL (sentinel branch never exercised)\n");
        return 1;
    }
    printf("SA resolve PARITY PASS (%lld coordinates)\n", (long long) checked);
    return 0;
}
