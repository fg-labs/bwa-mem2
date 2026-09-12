#!/usr/bin/env bash
# Local unit-test harness for the standalone C++ test binaries under test/
# (the $(EXE) list in test/Makefile). Builds, indexes the phiX fixture, runs
# each binary, asserts exit 0 and non-empty output. No hash pinning — that lives in the CI workflow.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BWAMEM3="$ROOT/bwa-mem3"
FIXTURES="$HERE/fixtures"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}
ok() { echo "OK:   $*"; }
skip() { echo "SKIP: $*"; }

[[ -x "$BWAMEM3" ]] || fail "bwa-mem3 not built at $BWAMEM3 (run 'make' or 'make arm64' first)"

# Build the standalone test binaries.
(cd "$HERE" && make) || fail "test/ make failed"

# Build any test binaries that live outside test/Makefile.
(cd "$ROOT" && make -j4 shm_section_find_test shm_pack_round_trip_test shm_lock_destroy_test fmi_seed_api_guards_test) > /dev/null

# synthetic_1mb.fa is checked in alongside its committed baselines so the
# byte-diff test stays reproducible across Python versions and platforms.
# A previous version of this script regenerated the FASTA via random.choice;
# random.choice's bytes are deterministic per Python version, but the
# baselines were built from a different generator, so CI ran the diff
# against a mismatched source and flaked.
[[ -s "$FIXTURES/synthetic_1mb.fa" ]] || fail "synthetic_1mb.fa fixture missing at $FIXTURES/synthetic_1mb.fa"

# Build the phiX FMI index if not already present. Check all four artifacts
# that `bwa-mem3 index` produces by default (no `.0123`: mem pac-fetches from
# `.pac`) so a corrupt/partial prior run is re-indexed.
if [[ ! -s "$FIXTURES/phix.fa.bwt.2bit.64" ||
    ! -s "$FIXTURES/phix.fa.amb" ||
    ! -s "$FIXTURES/phix.fa.ann" ||
    ! -s "$FIXTURES/phix.fa.pac" ]]; then
    "$BWAMEM3" index "$FIXTURES/phix.fa" > /dev/null 2>&1 || fail "bwa-mem3 index on phix.fa failed"
fi

# --- fmi_test --------------------------------------------------------------
OUT="$(cd "$HERE" && ./fmi_test "$FIXTURES/phix.fa" "$FIXTURES/reads.fa" 10 19 1 2>&1)"
echo "$OUT" | grep -q 'numReads = 10' || fail "fmi_test: numReads line missing"
echo "$OUT" | grep -q 'totalSmems =' || fail "fmi_test: totalSmems line missing"
echo "$OUT" | grep -qE '\[[0-9]+,[0-9]+\]' || fail "fmi_test: no SMEM output (PRINT_OUTPUT patch not applied?)"
ok "fmi_test"

# --- smem2_test ------------------------------------------------------------
OUT="$(cd "$HERE" && ./smem2_test "$FIXTURES/phix.fa" "$FIXTURES/smem2_input.txt" 5 50 19 1 2>&1)"
echo "$OUT" | grep -q 'numReads = 5' || fail "smem2_test: numReads line missing"
echo "$OUT" | grep -q 'totalSmems =' || fail "smem2_test: totalSmems line missing"
ok "smem2_test"

# --- bwt_seed_strategy_test ------------------------------------------------
# Note: unlike fmi_test and smem2_test, this binary doesn't emit a
# "numReads = N" line. Only assert the post-SW totalSmems marker.
OUT="$(cd "$HERE" && ./bwt_seed_strategy_test "$FIXTURES/phix.fa" "$FIXTURES/bwt_seed_input.fa" 5 50 19 20 2>&1)"
echo "$OUT" | grep -q 'minSeedLen =' || fail "bwt_seed_strategy_test: minSeedLen line missing (read-parsing broken?)"
echo "$OUT" | grep -q 'totalSmems =' || fail "bwt_seed_strategy_test: totalSmems line missing"
ok "bwt_seed_strategy_test"

# --- sa2ref_test -----------------------------------------------------------
OUT_FILE="$(mktemp)"
FKSW="$HERE/fksw.txt"
HEADER_OUT="$(mktemp)"
trap 'rm -f "$OUT_FILE" "$FKSW" "$HEADER_OUT"' EXIT
(cd "$HERE" && ./sa2ref_test "$FIXTURES/phix.fa" "$FIXTURES/sa2ref_input.txt" "$OUT_FILE" 20 > /dev/null 2>&1) || fail "sa2ref_test crashed"
[[ -s "$OUT_FILE" ]] || fail "sa2ref_test: output file empty"
ok "sa2ref_test (wrote $(wc -l < "$OUT_FILE" | tr -d ' ') coords)"

# --- header_insert_test ----------------------------------------------------
(cd "$HERE" && ./header_insert_test 2>&1) | tee "$HEADER_OUT" > /dev/null || fail "header_insert_test crashed"
grep -q 'ALL HEADER INSERT TESTS PASSED' "$HEADER_OUT" \
    || fail "header_insert_test: final banner missing ($(tail -n1 "$HEADER_OUT"))"
ok "header_insert_test"

# --- xeonbsw (main_banded) -------------------------------------------------
(cd "$HERE" && ./xeonbsw -pairs "$FIXTURES/pairs.txt" > /dev/null 2>&1) || fail "xeonbsw crashed"
[[ -s "$FKSW" ]] || fail "xeonbsw: fksw.txt empty (main_banded fksw-write patch not applied?)"
LINES="$(wc -l < "$FKSW" | tr -d ' ')"
[[ "$LINES" -eq 3 ]] || fail "xeonbsw: expected 3 lines in fksw.txt, got $LINES"
ok "xeonbsw ($LINES pair scores emitted)"

# --- bandedswa_zdrop_eweight_test (z-drop gap-extend weighting parity) ------
# Self-contained: generates tandem-repeat fixtures and compares getScores8 to
# the scalar oracle at non-default gap-extend (e_del/e_ins != 1). Exits non-zero
# on any score-field mismatch.
(cd "$HERE" && ./bandedswa_zdrop_eweight_test > /dev/null 2>&1) || fail "bandedswa_zdrop_eweight_test (z-drop e_del/e_ins weighting)"
ok "bandedswa_zdrop_eweight_test (z-drop gap-extend weighting parity)"

# --- pg_cl_escape_test ----------------------------------------------------
# Regression for issue #45 / upstream #293: tabs inside `-R` must not
# bleed into the @PG CL: value. Uses the same phiX fixture as the rest
# of the harness (indexed above if needed).
"$HERE/pg_cl_escape_test.sh" "$BWAMEM3" "$FIXTURES" || fail "pg_cl_escape_test failed"
ok "pg_cl_escape_test"

# --- meth_rg_header_test --------------------------------------------------
# A `-R` read group must be emitted as an @RG header line in --meth mode,
# not just stamped as RG:Z on records (the default path always was). Guards
# the malformed-BAM regression where --meth records referenced an undeclared
# read group ID.
"$HERE/meth_rg_header_test.sh" "$BWAMEM3" "$FIXTURES" || fail "meth_rg_header_test failed"
ok "meth_rg_header_test"

# --- meth_sidecar_enrich_test ---------------------------------------------
# --meth must enrich its consolidated @SQ with the original reference's
# identity tags (M5/UR/AS/SP) from that reference's .hdr/.dict sidecar
# (matched by SN) and forward its @CO/@PG/@RG provenance, while never
# consulting the c2t index's own sidecar.
"$HERE/meth_sidecar_enrich_test.sh" "$BWAMEM3" "$FIXTURES" || fail "meth_sidecar_enrich_test failed"
ok "meth_sidecar_enrich_test"

# --- index_alt_sidecar_warn_test ------------------------------------------
# `index` must warn when the reference has ALT contigs (<prefix>.alt) but the
# header sidecar supplies @SQ with no AH tag -- the Picard/GATK .dict case,
# where ALT status is silently lost from every output header. It must stay quiet
# with no sidecar, with no ALT contig, and when the sidecar already carries
# AH:*. The remaining --compat and --bam gating is covered at `mem` level by
# regression/header_parity.sh.
"$HERE/index_alt_sidecar_warn_test.sh" "$BWAMEM3" || fail "index_alt_sidecar_warn_test failed"
ok "index_alt_sidecar_warn_test"

# --- help_prescan_test ----------------------------------------------------
# `mem --help` pre-scan must not match `--help` when it is the value of
# an option that takes an argument (-R, -o, --set-as-failed, ...).
"$HERE/help_prescan_test.sh" "$BWAMEM3" "$FIXTURES" || fail "help_prescan_test failed"
ok "help_prescan_test"

# --- proc_freq_calibration_test -------------------------------------------
# Startup must not sleep to calibrate proc_freq (it used to sleep(1) on
# every invocation, which dominated this very suite), and the replacement
# short-window calibration must still report a plausible tick rate.
"$HERE/proc_freq_calibration_test.sh" "$BWAMEM3" "$FIXTURES" || fail "proc_freq_calibration_test failed"
ok "proc_freq_calibration_test"

# --- bam_compress_warn_test -----------------------------------------------
# The compressed-BAM (--bam=N, N>0) single-writer-thread warning must fire
# once, from the resolved --bam level, not once per occurrence and not for a
# level a later --bam=0 overrides.
"$HERE/bam_compress_warn_test.sh" "$BWAMEM3" "$FIXTURES" || fail "bam_compress_warn_test failed"
ok "bam_compress_warn_test"

# --- fast_preset_test -----------------------------------------------------
# --fast bundles the four characterized speed levers; explicit flags override;
# default path stays clean. Asserts on the resolved-opts audit line.
"$HERE/fast_preset_test.sh" "$BWAMEM3" "$FIXTURES" || fail "fast_preset_test failed"
ok "fast_preset_test"

# --- min_ext_len_safety_test ----------------------------------------------
# The --min-ext-len short-seed filter must never empty a chain: an all-short
# chain is left intact, so a huge --min-ext-len is a no-op (== default), not a
# read-dropping cliff. Regression for the smoke-1M low-mappability collapse.
"$HERE/min_ext_len_safety_test.sh" "$BWAMEM3" "$FIXTURES" || fail "min_ext_len_safety_test failed"
ok "min_ext_len_safety_test"

# --- smem_lockstep_parity_test --------------------------------------------
OUT="$(cd "$HERE" && ./smem_lockstep_parity_test "$FIXTURES/phix.fa" 2>&1)"
CASES_PASSED="$(echo "$OUT" | sed -nE 's/^([0-9]+) \/ ([0-9]+) cases passed$/\1/p')"
CASES_TOTAL="$(echo "$OUT" | sed -nE 's/^([0-9]+) \/ ([0-9]+) cases passed$/\2/p')"
[[ -n "$CASES_TOTAL" ]] || fail "smem_lockstep_parity_test: no summary line"
[[ "$CASES_PASSED" == "$CASES_TOTAL" ]] || fail "smem_lockstep_parity_test: $CASES_PASSED / $CASES_TOTAL cases passed"
ok "smem_lockstep_parity_test ($CASES_PASSED / $CASES_TOTAL)"

# --- bwtseed_lockstep_parity_test -----------------------------------------
OUT="$(cd "$HERE" && ./bwtseed_lockstep_parity_test "$FIXTURES/phix.fa" 2>&1)"
CASES_PASSED="$(echo "$OUT" | sed -nE 's/^([0-9]+) \/ ([0-9]+) cases passed$/\1/p')"
CASES_TOTAL="$(echo "$OUT" | sed -nE 's/^([0-9]+) \/ ([0-9]+) cases passed$/\2/p')"
[[ -n "$CASES_TOTAL" ]] || fail "bwtseed_lockstep_parity_test: no summary line"
[[ "$CASES_PASSED" == "$CASES_TOTAL" ]] || fail "bwtseed_lockstep_parity_test: $CASES_PASSED / $CASES_TOTAL cases passed"
ok "bwtseed_lockstep_parity_test ($CASES_PASSED / $CASES_TOTAL)"

# --- backwardext_konly_parity_test ----------------------------------------
# Direct kernel parity: backwardExt_konly vs the full backwardExt across a sweep
# of block offsets, locking the s==1 MSB-first bit path (bit CP_MASK-pos), which
# had a real byte-identity bug during development.
if OUT="$(cd "$HERE" && ./backwardext_konly_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
    echo "$OUT" | grep -q "backwardExt_konly PARITY PASS" \
        || fail "backwardext_konly_parity_test: no PASS line"
    ok "backwardext_konly_parity_test"
else
    echo "$OUT"
    fail "backwardext_konly_parity_test: non-zero exit"
fi

# --- sa_lookup_sentinel_parity_test ---------------------------------------
# Prefetch SA-lookup pipeline (get_sa_entries_prefetch/call_one_step) vs the
# scalar get_sa_entry_compressed over EVERY suffix-array row. The sentinel ($)
# row is where call_one_step dropped the accumulated LF offset (reported hits in
# the first few bases of the concatenated reference too far left); this sweep
# catches that row and is byte-identical elsewhere.
if OUT="$(cd "$HERE" && ./sa_lookup_sentinel_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
    echo "$OUT" | grep -q "^PASS:" \
        || fail "sa_lookup_sentinel_parity_test: no PASS line"
    ok "sa_lookup_sentinel_parity_test"
else
    echo "$OUT"
    fail "sa_lookup_sentinel_parity_test: non-zero exit"
fi

# --- backwardext_parity_test ----------------------------------------------
# Full backwardExt (k, l, s) vs a scalar reference computed straight from the
# checkpoint blocks, across block offsets, interval sizes and all four bases;
# gates the arm64 vector occurrence path lane by lane, including the suffix
# sum that feeds l.
if OUT="$(cd "$HERE" && ./backwardext_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
    echo "$OUT" | grep -q "backwardExt PARITY PASS" \
        || fail "backwardext_parity_test: no PASS line"
    ok "backwardext_parity_test"
else
    echo "$OUT"
    fail "backwardext_parity_test: non-zero exit"
fi

# --- sa_resolve_parity_test ------------------------------------------------
# The pipelined compressed-SA resolver (get_sa_entries_prefetch / call_one_step)
# vs a scalar LF walk computed straight from the checkpoint blocks, over every
# SA row of the fixture plus strided multi-occurrence intervals; gates the
# arm64 branchless symbol test and NEON occurrence popcount in call_one_step.
if OUT="$(cd "$HERE" && env -u BWA3_SA_LANES ./sa_resolve_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
    if echo "$OUT" | grep -q "SA resolve PARITY PASS"; then
        ok "sa_resolve_parity_test"
    elif echo "$OUT" | grep -q "SA resolve PARITY SKIP"; then
        skip "sa_resolve_parity_test (non-AArch64)"
    else
        echo "$OUT"
        fail "sa_resolve_parity_test: no PASS or SKIP line"
    fi
else
    echo "$OUT"
    fail "sa_resolve_parity_test: non-zero exit"
fi

# Same parity gate, but with the resolver's lane count forced to 1 (the
# non-default extreme of BWA3_SA_LANES). The lane count is a prefetch-depth
# knob only -- coordinates are stored by position -- so a PASS here against the
# same scalar oracle pins that "the lane count does not affect results".
if OUT="$(cd "$HERE" && BWA3_SA_LANES=1 ./sa_resolve_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
    if echo "$OUT" | grep -q "SA resolve PARITY PASS"; then
        ok "sa_resolve_parity_test (BWA3_SA_LANES=1)"
    elif echo "$OUT" | grep -q "SA resolve PARITY SKIP"; then
        skip "sa_resolve_parity_test (BWA3_SA_LANES=1, non-AArch64)"
    else
        echo "$OUT"
        fail "sa_resolve_parity_test (BWA3_SA_LANES=1): no PASS or SKIP line"
    fi
else
    echo "$OUT"
    fail "sa_resolve_parity_test (BWA3_SA_LANES=1): non-zero exit"
fi

# Same parity gate, but with the resolver's lane count forced to 64 (the
# non-default extreme of BWA3_SA_LANES). The lane count is a prefetch-depth
# knob only -- coordinates are stored by position -- so a PASS here against the
# same scalar oracle pins that "the lane count does not affect results".
if OUT="$(cd "$HERE" && BWA3_SA_LANES=64 ./sa_resolve_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
    if echo "$OUT" | grep -q "SA resolve PARITY PASS"; then
        ok "sa_resolve_parity_test (BWA3_SA_LANES=64)"
    elif echo "$OUT" | grep -q "SA resolve PARITY SKIP"; then
        skip "sa_resolve_parity_test (BWA3_SA_LANES=64, non-AArch64)"
    else
        echo "$OUT"
        fail "sa_resolve_parity_test (BWA3_SA_LANES=64): no PASS or SKIP line"
    fi
else
    echo "$OUT"
    fail "sa_resolve_parity_test (BWA3_SA_LANES=64): non-zero exit"
fi

# And at the two compiled-in defaults, 20 (arm64) and 40 (x86_64), so each
# host's parity run exercises the other architecture's default too: the
# no-env run above pins only whichever default this build compiled in.
for lanes in 20 40; do
    if OUT="$(cd "$HERE" && BWA3_SA_LANES=$lanes ./sa_resolve_parity_test "$FIXTURES/phix.fa" 2>&1)"; then
        if echo "$OUT" | grep -q "SA resolve PARITY PASS"; then
            ok "sa_resolve_parity_test (BWA3_SA_LANES=$lanes)"
        elif echo "$OUT" | grep -q "SA resolve PARITY SKIP"; then
            skip "sa_resolve_parity_test (BWA3_SA_LANES=$lanes, non-AArch64)"
        else
            echo "$OUT"
            fail "sa_resolve_parity_test (BWA3_SA_LANES=$lanes): no PASS or SKIP line"
        fi
    else
        echo "$OUT"
        fail "sa_resolve_parity_test (BWA3_SA_LANES=$lanes): non-zero exit"
    fi
done

# --- long-read end-to-end (issue 44) --------------------------------------
# Pre-fix, reads > 151 bp overran the per-thread SMEM buffer (segfault) and
# reads > 512 bp tripped the MAX_READ_LEN_FOR_LOCKSTEP assert. Post-fix the
# SMEM and lockstep-slot buffers are sized per-batch, so these must align
# cleanly and produce SAM lines.
for LEN_FQ in long_read_300bp long_read_1kbp long_read_3kbp; do
    RAW="$("$BWAMEM3" mem "$FIXTURES/phix.fa" "$FIXTURES/${LEN_FQ}.fq" 2> /dev/null)" \
        || fail "bwa-mem3 mem ${LEN_FQ}.fq: non-zero exit (crash regression)"
    OUT="$(printf '%s\n' "$RAW" | grep -v '^@' || true)"
    [[ -n "$OUT" ]] || fail "bwa-mem3 mem ${LEN_FQ}.fq: no SAM records emitted"
    # Every record should be a successful alignment (not flag 4 = unmapped).
    UNMAPPED="$(echo "$OUT" | awk '$2 == 4 || $2 == 2052' | wc -l | tr -d ' ')"
    [[ "$UNMAPPED" == "0" ]] || fail "bwa-mem3 mem ${LEN_FQ}.fq: $UNMAPPED unmapped record(s)"
    ok "bwa-mem3 mem ${LEN_FQ}.fq (all records mapped)"
done

# --- ultra-long-read crash regression (read length > INT16_MAX) -----------
# Pre-fix, read positions in the SMEM seeding path were int16_t:
# query_pos_array (FMI_search getSMEMsAllPos/OnePos/lockstep + caller's
# mmc->query_pos_ar), BatchSlot.start_pos/next_x, and the walk index `x`
# in bwtSeedStrategyAllPosOneThread. For a read longer than INT16_MAX
# (32767) bp the position incremented past 32767 and wrapped negative, so
# the re-seed loops (`while (x < readlength)` and the getSMEMsAllPos
# do/while) never advanced past the wrap point. They emitted SMEMs without
# bound until matchArray overran -> heap corruption / SIGSEGV. The
# 300bp-3kbp cases above all stay under 32767 and never exposed it; phiX
# (~5 kb) is too small to host the alignment, so use the checked-in 1 Mb
# synthetic reference and a 40 kb read sliced from it.
SYN="$FIXTURES/synthetic_1mb.fa"
if [[ ! -s "$SYN.bwt.2bit.64" || ! -s "$SYN.amb" ||
    ! -s "$SYN.ann" || ! -s "$SYN.pac" ]]; then
    "$BWAMEM3" index "$SYN" > /dev/null 2>&1 || fail "bwa-mem3 index on synthetic_1mb.fa failed"
fi
# Slice a 40 kb read (> 32767) from the middle of the reference so it maps
# back cleanly. Derived at run time rather than committed as a fixture.
ULTRALONG_FA="$(mktemp "$HERE/.ultralong_40kbp.XXXXXX.fa")"
# Append to (don't replace) the EXIT trap installed earlier so the temp FASTA
# is removed on early failure or in parallel runs, not just on success.
trap 'rm -f "$OUT_FILE" "$FKSW" "$HEADER_OUT" "$ULTRALONG_FA"' EXIT
awk 'NR==1 && /^>/ {next} {seq = seq $0}
     END { print ">ultralong_40kbp"; print substr(seq, 100001, 40000) }' \
    "$SYN" > "$ULTRALONG_FA"
LR_LEN="$(awk 'NR==2 {print length($0)}' "$ULTRALONG_FA")"
[[ "$LR_LEN" -gt 32767 ]] || fail "ultralong fixture is ${LR_LEN}bp, need > 32767 (INT16_MAX)"
RAW="$("$BWAMEM3" mem "$SYN" "$ULTRALONG_FA" 2> /dev/null)" \
    || fail "bwa-mem3 mem ultralong ${LR_LEN}bp: non-zero exit (int16 position-overflow regression)"
OUT="$(printf '%s\n' "$RAW" | grep -v '^@' || true)"
[[ -n "$OUT" ]] || fail "bwa-mem3 mem ultralong: no SAM records emitted"
# The primary record (flag < 256, not 4=unmapped) must align.
PRIMARY_MAPPED="$(echo "$OUT" | awk '$2 < 256 && $2 != 4 {c++} END {print c+0}')"
[[ "$PRIMARY_MAPPED" -ge 1 ]] || fail "bwa-mem3 mem ultralong ${LR_LEN}bp: primary read unmapped"
rm -f "$ULTRALONG_FA" # cleaned here on success; EXIT trap covers early-failure paths
ok "bwa-mem3 mem ultralong ${LR_LEN}bp (> 32767, mapped)"

# --- interleaved -p mode regression --------------------------------------
# Bugs covered:
#  - Empty/zero-length-read batches (which a malformed FASTQ that bseq parses
#    as a single 0-bp record would trigger) used to fall through the lazy-init
#    grow checks in mem_collect_smem with NULL per-thread buffers and SEGV in
#    the populate loop. mem_collect_smem now early-returns on zero-base batches.
#  - The OLD `tot_len >= mmc->wsize_mem[tid]` grow block in mem_kernel1_core
#    used to fire when both sides were 0, calling _mm_realloc(NULL, 0, 0, ...)
#    which returns NULL and then exit(1) on the post-collect check. That block
#    is gone; mem_collect_smem owns all SMEM-family sizing.
PE_TMP="$(mktemp -d)"
# Append to (don't replace) the EXIT trap installed earlier for OUT_FILE /
# FKSW / HEADER_OUT — overwriting it would leak those paths once this block
# runs.
trap 'rm -f "$OUT_FILE" "$FKSW" "$HEADER_OUT"; rm -rf "$PE_TMP"' EXIT

# Build a small properly-interleaved FASTQ from phix. Use python so the
# fixture is generated programmatically (per repo convention) and so the
# interleaving is correct at FASTQ-record granularity.
python3 - "$FIXTURES/phix.fa" "$PE_TMP/interleaved.fq" << 'PY'
import random, sys
ref_path, out_path = sys.argv[1], sys.argv[2]
ref = ''
with open(ref_path) as f:
    next(f)
    for line in f:
        ref += line.strip()
random.seed(42)
N, RL, INS = 50, 120, 280
comp = str.maketrans('ACGTN', 'TGCAN')
with open(out_path, 'w') as out:
    for i in range(N):
        pos = random.randint(0, len(ref) - INS)
        r1 = ref[pos:pos+RL]
        r2 = ref[pos+INS-RL:pos+INS][::-1].translate(comp)
        q  = 'I' * RL
        out.write(f'@pe{i}/1\n{r1}\n+\n{q}\n')
        out.write(f'@pe{i}/2\n{r2}\n+\n{q}\n')
PY

OUT="$("$BWAMEM3" mem -p "$FIXTURES/phix.fa" "$PE_TMP/interleaved.fq" 2> /dev/null)" \
    || fail "bwa-mem3 mem -p interleaved.fq: non-zero exit (crash regression)"
RECORDS="$(printf '%s\n' "$OUT" | grep -cv '^@' || true)"
[[ "$RECORDS" -ge 100 ]] || fail "bwa-mem3 mem -p: expected >=100 records (50 pairs × 2), got $RECORDS"
ok "bwa-mem3 mem -p interleaved.fq ($RECORDS records)"

# Empty FASTQ + zero-length-read FASTQ regressions for the empty-batch
# defensive path in mem_collect_smem / mem_kernel1_core.
: > "$PE_TMP/empty.fq"
"$BWAMEM3" mem "$FIXTURES/phix.fa" "$PE_TMP/empty.fq" > /dev/null 2>&1 \
    || fail "bwa-mem3 mem on empty FASTQ: non-zero exit"
ok "bwa-mem3 mem on empty FASTQ (no crash)"

printf '@zero\n\n+\n\n' > "$PE_TMP/zero.fq"
"$BWAMEM3" mem "$FIXTURES/phix.fa" "$PE_TMP/zero.fq" > /dev/null 2>&1 \
    || fail "bwa-mem3 mem on zero-length read: non-zero exit"
ok "bwa-mem3 mem on zero-length read (no crash)"

# --- packed_text_test ------------------------------------------------------
(cd "$HERE" && ./packed_text_test) || fail "packed_text_test"
ok "packed_text_test"

# --- system_test -----------------------------------------------------------
(cd "$HERE" && ./system_test) || fail "system_test"
ok "system_test"

# --- libsais byte-diff against committed baselines -------------------------
(cd "$HERE" && ./libsais_index_diff_test.sh) || fail "libsais_index_diff_test"
ok "libsais_index_diff_test"

# --- libsais determinism across thread counts ------------------------------
(cd "$HERE" && ./libsais_determinism_test.sh) || fail "libsais_determinism_test"
ok "libsais_determinism_test"

# --- libsais chr22 byte-diff (skips if scratch fixture absent) -------------
(cd "$HERE" && ./libsais_chr22_diff_test.sh) || fail "libsais_chr22_diff_test"
ok "libsais_chr22_diff_test"

# --- libsais 10 Mbp hg38 slice byte-diff (skips if scratch fixture absent) -
(cd "$HERE" && ./libsais_hg38_slice_diff_test.sh) || fail "libsais_hg38_slice_diff_test"
ok "libsais_hg38_slice_diff_test"

# --- libsais --max-memory peak-RSS budget test -----------------------------
(cd "$HERE" && ./libsais_memory_budget_test.sh) || fail "libsais_memory_budget_test"
ok "libsais_memory_budget_test"

# --- shm_section_find_test (unit: section-find helper) -----------------------
echo "==> shm_section_find_test"
(cd "$ROOT" && ./shm_section_find_test)
echo "OK:   shm_section_find_test"

# --- shm_lock_destroy_test (unit: bwa_shm_destroy unlinks /bwactl_lock) ------
echo "==> shm_lock_destroy_test"
(cd "$ROOT" && ./shm_lock_destroy_test)
echo "OK:   shm_lock_destroy_test"

# --- fmi_seed_api_guards_test (unit: fmi_seed_sa_prefetch max_occ guard) -----
echo "==> fmi_seed_api_guards_test"
(cd "$ROOT" && ./fmi_seed_api_guards_test "$FIXTURES/phix.fa")
echo "OK:   fmi_seed_api_guards_test"

# --- shm pack round-trip (unit: pack/unpack a phiX segment) ------------------
(cd "$HERE" && ./shm_pack_round_trip_test.sh) || fail "shm_pack_round_trip_test"
ok "shm_pack_round_trip_test"

# --- shm end-to-end round-trip (byte-identical SAM with and without shm) -----
(cd "$HERE" && ./shm_round_trip_test.sh) || fail "shm_round_trip_test"
ok "shm_round_trip_test"

# --- shm --meth flag (parity with mem --meth's c2t-suffix resolution) --------
(cd "$HERE" && ./shm_meth_test.sh) || fail "shm_meth_test"
ok "shm_meth_test"

echo "ALL UNIT TESTS PASSED"
