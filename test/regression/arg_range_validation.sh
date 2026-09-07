#!/usr/bin/env bash
# test/regression/arg_range_validation.sh
#
# Asserts that `mem` rejects a non-positive gap-extension penalty (-E) and a
# non-positive insert-size standard deviation (-I mean,std) at parse time, with
# a clear ERROR: message, instead of dividing by zero downstream:
#   - -E 0 divides by zero in cal_max_gap; the (int) cast of the inf/NaN result
#     is UB that resolves differently per architecture (arch-divergent bands).
#   - -I mean,0 divides by zero in mem_pair; the NaN collapses every pair score
#     to 0 and silently disables pair-aware placement.
#
# Validation fires during option parsing, before the reference is opened, so
# /dev/null placeholders suffice — no fixture is needed. The message (not just a
# non-zero exit) is asserted, so a non-zero exit from a later missing-input
# error cannot masquerade as the rejection under test.

set -euo pipefail

BWA_MEM3="${BWA_MEM3:-$(dirname "$0")/../../bwa-mem3}"
if [[ ! -x "$BWA_MEM3" ]]; then
    echo "FAIL: BWA_MEM3 ('$BWA_MEM3') is not an executable binary" >&2
    exit 1
fi

OUT_DIR="$(mktemp -d -t bwamem3-argval-XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

# check_reject <description> <expected-stderr-substring> <mem-args...>
check_reject() {
    local desc=$1 pattern=$2
    shift 2
    local rc=0
    "$BWA_MEM3" mem "$@" /dev/null /dev/null /dev/null \
        > "$OUT_DIR/out" 2> "$OUT_DIR/err" || rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "FAIL: $desc: expected non-zero exit, got 0" >&2
        cat "$OUT_DIR/err" >&2
        exit 1
    fi
    if ! grep -qF "$pattern" "$OUT_DIR/err"; then
        echo "FAIL: $desc: stderr does not contain '$pattern'" >&2
        cat "$OUT_DIR/err" >&2
        exit 1
    fi
}

check_reject "-E 0" "ERROR: -E gap-extension penalty must be a positive integer" -E 0
check_reject "-E 5,0" "ERROR: -E gap-extension penalty must be a positive integer" -E 5,0
# An out-of-range -E value must be rejected before it is narrowed to int: 2^32+1
# narrows to 1, which would pass the old >= 1 check and silently apply a
# gap-extension penalty different from the one requested (and feed cal_max_gap's
# divide with a wrong value). Reject on the parsed long, not the wrapped int.
check_reject "-E 4294967297" "ERROR: -E gap-extension penalty must be a positive integer" -E 4294967297
check_reject "-E 5,4294967297" "ERROR: -E gap-extension penalty must be a positive integer" -E 5,4294967297
check_reject "-I 300,0" "ERROR: -I standard deviation must be a positive number" -I 300,0

# A non-finite -I mean or std (nan/inf, or an overflowing token that strtod maps
# to inf) reaches an (int) cast whose result is undefined and arch-divergent;
# reject it at parse rather than casting inf/NaN downstream.
check_reject "-I nan,1" "ERROR: -I mean insert size must be a finite number" -I nan,1
check_reject "-I inf" "ERROR: -I mean insert size must be a finite number" -I inf
check_reject "-I 300,1e400" "ERROR: -I standard deviation must be a positive number" -I 300,1e400
check_reject "-I 300,1,1e400" "ERROR: -I insert-size max must be a finite number" -I 300,1,1e400

echo "PASS: arg_range_validation (-E and -I reject non-positive and non-finite values)"
