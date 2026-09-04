#!/usr/bin/env bash
# test/kopen_pipe_e2e_test.sh
#
# End-to-end regression for the `<cmd` input producer status path: a producer
# that emits valid FASTA/FASTQ and then exits non-zero must fail the WHOLE
# `mem` run, not just the isolated kopen/kclose contract (see
# kopen_pipe_status_test.cpp for that). This exercises src/fastmap.cpp's
# post-kclose exit-code propagation in main_mem: kclose's non-zero return
# folds into the process exit code, and stderr names the failing input's
# argv position and its exact producer exit status.
#
# Three cases:
#   1. Primary input fails.
#   2. Paired inputs, only the SECOND fails (diagnostic must name it, not
#      the first, which exited cleanly).
#   3. Paired inputs, only the FIRST fails (diagnostic must name it, not
#      the second).
#
# Usage: test/kopen_pipe_e2e_test.sh <bwa-mem3-binary> <fixtures-dir>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <bwa-mem3-binary> <fixtures-dir>" >&2
    exit 2
fi

abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
bin="$(abspath "$1")"
fixtures="$(abspath "$2")"
ref="$fixtures/phix.fa"
reads="$fixtures/reads.fa"

[[ -x "$bin" ]] || {
    echo "FAIL: bwa-mem3 binary not executable at $bin" >&2
    exit 1
}
[[ -s "$ref" ]] || {
    echo "FAIL: phix.fa missing at $ref" >&2
    exit 1
}
[[ -s "$reads" ]] || {
    echo "FAIL: reads.fa missing at $reads" >&2
    exit 1
}

# Build the phiX FMI index if not already present.
if [[ ! -s "$ref.bwt.2bit.64" || ! -s "$ref.amb" ||
    ! -s "$ref.ann" || ! -s "$ref.pac" ]]; then
    "$bin" index "$ref" > /dev/null 2>&1 \
        || {
            echo "FAIL: bwa-mem3 index on phix.fa failed" >&2
            exit 1
        }
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Run `mem` with the given `<cmd` input args and assert the exit status AND that
# the diagnostic names the CORRECT failing input. main_mem prints one line per
# failed producer: `ERROR: input command `<cmd>' exited with status N` (see
# src/fastmap.cpp). `want_cmd` is the exact command string of the input that must
# be reported as failed, and `want_status` its exit code; both must appear on the
# SAME line so a mis-attributed input position (right status, wrong command) is
# caught, not just the status in isolation.
expect_fail() {
    local label="$1" want_cmd="$2" want_status="$3"
    shift 3
    local err="$tmpdir/$label.err"
    local rc=0

    "$bin" mem "$ref" "$@" > /dev/null 2> "$err" || rc=$?

    if ((rc == 0)); then
        echo "FAIL: [$label] mem exited 0, expected non-zero" >&2
        cat "$err" >&2
        exit 1
    fi
    # Fixed-string match of the whole diagnostic line ties the command that is
    # named to the status reported for it. grep -F so the backticks, quotes, and
    # path in the command string are matched literally.
    local want_line="input command \`$want_cmd' exited with status $want_status"
    if ! grep -qF -- "$want_line" "$err"; then
        echo "FAIL: [$label] stderr missing diagnostic '$want_line'" >&2
        echo "---- stderr ----" >&2
        cat "$err" >&2
        exit 1
    fi
    echo "OK:   [$label] rc=$rc, failing input correctly named"
}

# --- 1. Primary `<cmd` input emits valid FASTA then exits non-zero. --------
primary_cmd="<cat '$reads'; exit 7"
expect_fail primary "$primary_cmd" 7 "$primary_cmd"

# --- 2. Paired: only the SECOND producer fails. -----------------------------
# The diagnostic must name the second input (the one that exited 9), not the
# first (which exited 0) -- asserting the failing command string tied to its
# status pins the reported position, not merely that some input failed.
p2_first="<cat '$reads'; exit 0"
p2_second="<cat '$reads'; exit 9"
expect_fail paired_second "$p2_second" 9 "$p2_first" "$p2_second"

# --- 3. Paired: only the FIRST producer fails. ------------------------------
p3_first="<cat '$reads'; exit 5"
p3_second="<cat '$reads'; exit 0"
expect_fail paired_first "$p3_first" 5 "$p3_first" "$p3_second"

echo "PASS: kopen_pipe_e2e_test"
