#!/usr/bin/env bash
# test/regression/chain_flt_empty_compat.sh
#
# Regression: when `mem_chain_flt`'s weight filter drops EVERY chain for a
# read, the two upstreams disagree about the read, and each `--compat` target
# must reproduce its own (fg-labs/bwa-mem3#310).
#
# bwa returns zero survivors -- its tail loops are bounded by `n_chn == 0`, so
# the read goes out unmapped. bwa-mem2's seqid-range machinery instead
# synthesizes a count of 1 for the emptied array, the unconditional
# `kept[0] = 3` becomes load-bearing, and the chain the filter just rejected is
# extended into an alignment. bwa-mem3 inherited bwa-mem2's behavior at the
# fork point. The default path and `--compat=bwa-mem` now take bwa's answer
# (the filter's decision stands); only `--compat=bwa-mem2` reproduces the port
# as shipped, because reproducing that release's records is its contract.
#
# Reachable only when `min_chain_weight > 0`, which is never the default: `-W`,
# or the `-x pacbio`/`pbref`/`ont2d` presets. This script uses `-W` because it
# makes the trigger exact rather than probabilistic -- with `-W` above the read
# length NO chain can clear the threshold, so every read takes the path. (On
# real long reads the presets essentially never trigger it: measured over 500
# HiFi reads at `-x pacbio`, zero reads had every chain dropped, which is why
# this is pinned with `-W` and not with a preset.)
#
# The three assertions:
#   1. default        -- every read UNMAPPED (bwa's behavior: the fix)
#   2. --compat=bwa-mem2 -- every read mapped (the target IS bwa-mem2)
#   3. --compat=bwa-mem  -- every read UNMAPPED (bwa's behavior)
#
# Guard against a vacuous pass: the same reads at `-W 0` must map under all
# three, proving the fixture aligns at all and that assertions 1 and 3 are the
# weight filter firing rather than reads that simply do not align.
#
# Fixture (deterministic, no PRNG): 150 bp SE reads sliced from the committed
# phix.fa, same construction as compat_byte_identical.sh. `-W 200` exceeds the
# read length, so every chain's weight is below the threshold.
#
# Inputs (env vars):
#   BWA_MEM3           — path to the bwa-mem3 binary under test
#   CHAIN_FLT_PHIX_FA  — path to test/fixtures/phix.fa (the reference source)
#   CHAIN_FLT_WORK_DIR — directory for fixture-private intermediates

set -euo pipefail

: "${BWA_MEM3:?BWA_MEM3 must be set}"
: "${CHAIN_FLT_PHIX_FA:?CHAIN_FLT_PHIX_FA must be set}"
: "${CHAIN_FLT_WORK_DIR:?CHAIN_FLT_WORK_DIR must be set}"

mkdir -p "$CHAIN_FLT_WORK_DIR"

ref="$CHAIN_FLT_WORK_DIR/phix.fa"
cp "$CHAIN_FLT_PHIX_FA" "$ref"

# --- Build SE FASTQ by slicing phix (deterministic, no PRNG). ---
seq="$(grep -v '^>' "$ref" | tr -d '\n' | tr 'acgt' 'ACGT')"
r1="$CHAIN_FLT_WORK_DIR/r1.fq"
: > "$r1"
L=150
n_reads=0
for off in 200 700 1200 1700 2200 2700 3200 3700 4200; do
    s="${seq:$off:$L}"
    [ "${#s}" -eq "$L" ] || continue
    q="$(printf 'I%.0s' $(seq 1 "$L"))"
    printf '@read%d\n%s\n+\n%s\n' "$n_reads" "$s" "$q" >> "$r1"
    n_reads=$((n_reads + 1))
done

if [ "$n_reads" -lt 5 ]; then
    echo "FAIL: fixture built only $n_reads reads (phix slicing regression?)" >&2
    exit 1
fi
echo "fixture: $n_reads SE reads sliced from phix"

"$BWA_MEM3" index "$ref" > "$CHAIN_FLT_WORK_DIR/index.log" 2>&1

# run_arm <label> <-W value> [extra flags...]: one `mem` run per arm. A nonzero
# exit is a FAIL here rather than a silently empty SAM -- the counts below would
# read an empty file as "0 mapped", which is exactly what assertions 1 and 3
# expect, so the exit status has to be checked where it cannot be swallowed by
# a command substitution.
run_arm() {
    local label=$1 wt=$2
    shift 2
    local sam="$CHAIN_FLT_WORK_DIR/$label.sam"
    if ! "$BWA_MEM3" mem "$@" -W "$wt" "$ref" "$r1" > "$sam" 2> "$CHAIN_FLT_WORK_DIR/$label.log"; then
        echo "FAIL: $label -- bwa-mem3 mem exited nonzero (see $CHAIN_FLT_WORK_DIR/$label.log)" >&2
        exit 1
    fi
}

# record_count <label>: every non-header line, mapped or not.
record_count() {
    awk '!/^@/' "$CHAIN_FLT_WORK_DIR/$1.sam" | wc -l | tr -d ' '
}

# mapped_count <label>: records with FLAG bit 0x4 (unmapped) clear. The bit is
# tested by arithmetic: `and()` is a gawk builtin and is absent from BSD awk
# (macOS) and mawk, both of which run this suite.
mapped_count() {
    awk '!/^@/ && int($2 / 4) % 2 == 0' "$CHAIN_FLT_WORK_DIR/$1.sam" | wc -l | tr -d ' '
}

# --- Guard: at -W 0 the weight filter cannot drop anything, so all three
# --- configurations must map every read. Without this, assertions 1 and 3
# --- below (both expect 0 mapped) would also pass on a fixture that simply
# --- fails to align.
for cfg in "default:" "compat-mem2:--compat=bwa-mem2" "compat-mem:--compat=bwa-mem"; do
    label=${cfg%%:*}
    flag=${cfg#*:}
    if [ -n "$flag" ]; then
        run_arm "w0-$label" 0 "$flag"
    else run_arm "w0-$label" 0; fi
    got=$(mapped_count "w0-$label")
    if [ "$got" -ne "$n_reads" ]; then
        echo "FAIL: guard -- at -W 0, $label mapped $got/$n_reads reads (expected all)" >&2
        exit 1
    fi
done
echo "guard: all $n_reads reads map at -W 0 under all three configurations"

# --- The load-bearing assertions, at -W 200 (> read length). ---
run_arm "w200-default" 200
run_arm "w200-compat-mem2" 200 --compat=bwa-mem2
run_arm "w200-compat-mem" 200 --compat=bwa-mem

fail=0

# Every arm must emit one record per read: "unmapped" means present with 0x4
# set, not absent. Without this, a run that dropped reads instead of emitting
# them unmapped would satisfy the 0-mapped assertions below.
for label in w200-default w200-compat-mem2 w200-compat-mem; do
    got=$(record_count "$label")
    if [ "$got" -ne "$n_reads" ]; then
        echo "FAIL: $label emitted $got records for $n_reads reads (expected one per read, mapped or not)" >&2
        fail=1
    fi
done

def_mapped=$(mapped_count "w200-default")
mem2_mapped=$(mapped_count "w200-compat-mem2")
mem_mapped=$(mapped_count "w200-compat-mem")

if [ "$def_mapped" -ne 0 ]; then
    echo "FAIL: default at -W 200 mapped $def_mapped/$n_reads (expected 0 -- the default takes bwa's answer)" >&2
    fail=1
fi
if [ "$mem2_mapped" -ne "$n_reads" ]; then
    echo "FAIL: --compat=bwa-mem2 at -W 200 mapped $mem2_mapped/$n_reads (expected all)" >&2
    fail=1
fi
if [ "$mem_mapped" -ne 0 ]; then
    echo "FAIL: --compat=bwa-mem at -W 200 mapped $mem_mapped/$n_reads (expected 0 -- bwa leaves them unmapped)" >&2
    fail=1
fi
[ "$fail" -eq 0 ] || exit 1

echo "PASS: default=0/$n_reads mapped, --compat=bwa-mem2=$mem2_mapped/$n_reads, --compat=bwa-mem=0/$n_reads, $n_reads records emitted per arm (#310)"
