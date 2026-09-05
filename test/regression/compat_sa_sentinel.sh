#!/usr/bin/env bash
# test/regression/compat_sa_sentinel.sh
#
# Regression: the compressed suffix-array lookup's sentinel-row offset is
# delivered end to end through `mem --compat` (fg-labs/bwa-mem3#469).
#
# When the LF walk from an unsampled SA row reaches the sentinel (`$`) row
# before a sampled one, the coordinate is the number of steps walked. bwa's
# `bwt_sa` reports that offset; bwa-mem2 v2.2.1's pipelined lookup dropped it
# and reported 0, placing the hit up to a few bases too far left. bwa-mem3
# reports the offset by default and under `--compat=bwa-mem`, and reproduces
# the dropped offset under `--compat=bwa-mem2` (compat_target_t's
# `sa_sentinel_drop_offset`, read once per batch in `mem_chain_seeds`).
#
# The resolver itself is pinned by test/sa_lookup_sentinel_parity_test.cpp,
# which calls it directly with the flag in both positions. What that test
# cannot see is the plumbing from `--compat` to the resolver: a build that
# never passed the flag through would pass it and every other test in the
# suite. This script closes that gap by running `mem` itself.
#
# Fixture (deterministic, no PRNG): SE reads sliced from the committed phix.fa
# at offsets 0, 1 and 2 of contig 0. Only a walk that starts within the first
# few bases of the concatenated text can reach the sentinel; on phix the read
# at offset 1 is the one whose resolved seed does. Which seed the aligner
# resolves for a read depends on the read's length, so the offset-1 read is
# sliced at several lengths and the drop arm is required to show on at least
# one of them, rather than pinning the seeding of any single length.
#
# Assertions:
#   1. default and --compat=bwa-mem: every read maps to POS offset+1 with a
#      full-length match, at every length (the correct coordinate).
#   2. --compat=bwa-mem2: the offset-0 and offset-2 reads map exactly as in 1
#      (the drop moves only sentinel-reaching walks), the offset-1 read maps to
#      POS 1 or POS 2 at every length, and to POS 1 (bwa-mem2's dropped
#      offset) on at least one length. The offset-0/2 reads are the guard
#      against a vacuous pass: they prove the fixture aligns and that the only
#      moving record is the one the sentinel path owns.
#
# Inputs (env vars):
#   BWA_MEM3              — path to the bwa-mem3 binary under test
#   SA_SENTINEL_PHIX_FA   — path to test/fixtures/phix.fa (the reference source)
#   SA_SENTINEL_WORK_DIR  — directory for fixture-private intermediates

set -Eeuo pipefail
# Any command that fails unexpectedly (cp/index/mem) exits via set -e; surface it
# as a FAIL: record with the failing line and status rather than dying silently.
# shellcheck disable=SC2154  # rc is assigned by `rc=$?` at the start of the trap body
trap 'rc=$?; echo "FAIL: unexpected command failure at line $LINENO (exit $rc)" >&2; exit "$rc"' ERR

if [ -z "${BWA_MEM3:-}" ]; then
    echo "FAIL: BWA_MEM3 must be set" >&2
    exit 1
fi
if [ -z "${SA_SENTINEL_PHIX_FA:-}" ]; then
    echo "FAIL: SA_SENTINEL_PHIX_FA must be set" >&2
    exit 1
fi
if [ -z "${SA_SENTINEL_WORK_DIR:-}" ]; then
    echo "FAIL: SA_SENTINEL_WORK_DIR must be set" >&2
    exit 1
fi

mkdir -p "$SA_SENTINEL_WORK_DIR"

ref="$SA_SENTINEL_WORK_DIR/phix.fa"
cp "$SA_SENTINEL_PHIX_FA" "$ref"

# --- Build SE FASTQ by slicing phix (deterministic, no PRNG). ---
seq="$(grep -v '^>' "$ref" | tr -d '\n' | tr 'acgt' 'ACGT')"
reads="$SA_SENTINEL_WORK_DIR/reads.fq"
: > "$reads"
lengths="30 50 150"
n_reads=0
for L in $lengths; do
    for off in 0 1 2; do
        s="${seq:$off:$L}"
        if [ "${#s}" -ne "$L" ]; then
            echo "FAIL: fixture -- phix shorter than offset $off + $L" >&2
            exit 1
        fi
        q="$(printf 'I%.0s' $(seq 1 "$L"))"
        printf '@off%d_L%d\n%s\n+\n%s\n' "$off" "$L" "$s" "$q" >> "$reads"
        n_reads=$((n_reads + 1))
    done
done
echo "fixture: $n_reads SE reads sliced from phix at offsets 0-2, lengths $lengths"

"$BWA_MEM3" index "$ref" > "$SA_SENTINEL_WORK_DIR/index.log" 2>&1

# run <label> [extra flags...]: writes "<qname> <pos> <cigar>" per record.
run() {
    local label=$1
    shift
    local sam="$SA_SENTINEL_WORK_DIR/$label.sam"
    "$BWA_MEM3" mem -t 1 "$@" "$ref" "$reads" > "$sam" 2> /dev/null
    awk '!/^@/ { print $1, $4, $6 }' "$sam" > "$SA_SENTINEL_WORK_DIR/$label.tsv"
}

run default
run compat-mem --compat=bwa-mem
run compat-mem2 --compat=bwa-mem2

fail=0

# --- Assertion 1: default and --compat=bwa-mem report the correct coordinate.
for label in default compat-mem; do
    got=$(wc -l < "$SA_SENTINEL_WORK_DIR/$label.tsv" | tr -d ' ')
    if [ "$got" -ne "$n_reads" ]; then
        echo "FAIL: $label emitted $got records for $n_reads reads (expected one primary each)" >&2
        fail=1
    fi
    while read -r qname pos cigar; do
        off=${qname#off}
        off=${off%%_*}
        L=${qname#*_L}
        if [ "$pos" -ne $((off + 1)) ] || [ "$cigar" != "${L}M" ]; then
            echo "FAIL: $label placed $qname at POS $pos CIGAR $cigar (expected POS $((off + 1)) ${L}M)" >&2
            fail=1
        fi
    done < "$SA_SENTINEL_WORK_DIR/$label.tsv"
done

# --- Assertion 2: --compat=bwa-mem2 drops the walk offset on the sentinel
# --- path and nowhere else.
got=$(wc -l < "$SA_SENTINEL_WORK_DIR/compat-mem2.tsv" | tr -d ' ')
if [ "$got" -ne "$n_reads" ]; then
    echo "FAIL: compat-mem2 emitted $got records for $n_reads reads (expected one primary each)" >&2
    fail=1
fi
dropped=0
while read -r qname pos cigar; do
    off=${qname#off}
    off=${off%%_*}
    L=${qname#*_L}
    if [ "$cigar" != "${L}M" ]; then
        echo "FAIL: --compat=bwa-mem2 placed $qname with CIGAR $cigar (expected ${L}M)" >&2
        fail=1
        continue
    fi
    if [ "$off" -ne 1 ]; then
        if [ "$pos" -ne $((off + 1)) ]; then
            echo "FAIL: guard -- --compat=bwa-mem2 moved $qname to POS $pos (expected $((off + 1)); only the sentinel-reaching read may move)" >&2
            fail=1
        fi
    elif [ "$pos" -eq 1 ]; then
        dropped=$((dropped + 1))
    elif [ "$pos" -ne 2 ]; then
        echo "FAIL: --compat=bwa-mem2 placed $qname at POS $pos (expected 1 or 2)" >&2
        fail=1
    fi
done < "$SA_SENTINEL_WORK_DIR/compat-mem2.tsv"

if [ "$dropped" -eq 0 ]; then
    echo "FAIL: --compat=bwa-mem2 never reported the dropped offset (offset-1 read at POS 1) at any length; the compat flag is not reaching the SA resolver" >&2
    fail=1
fi

[ "$fail" -eq 0 ] || exit 1

echo "PASS: default and --compat=bwa-mem place all $n_reads reads at their true coordinate; --compat=bwa-mem2 reproduces the dropped sentinel offset on $dropped of $(echo "$lengths" | wc -w | tr -d ' ') lengths and moves nothing else (#469)"
