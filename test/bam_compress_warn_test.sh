#!/usr/bin/env bash
# test/bam_compress_warn_test.sh
#
# `--bam=N` (N>0) compresses BGZF. The deflate pool is auto-sized from -t
# (`--bam-threads` defaults to n_threads/8 when compressing), so the common case
# is NOT serial and must NOT warn. The only serial-deflate throughput trap left
# is a user *explicitly* forcing `--bam-threads 0` at a compressing level; that
# -- and only that -- warns, once, from the resolved settings.
#
# The warning describes the *resolved* output setting, so it is emitted once,
# after option parsing, from the final opt->bam_mode / opt->bam_level /
# opt->bam_threads -- not from inside the parsing loop. The cases below pin that:
#
#   1. `--bam=6`                       auto pool -> none.
#   2. `--bam=6 --bam-threads=0`       forced serial at a compressing level -> one.
#   3. `--bam=6 --bam-threads=4`       explicit positive pool -> none.
#   4. `--bam=6 --bam-threads=0 --bam=0`  last --bam wins (uncompressed) -> none.
#   5. `--bam=6 --bam=6 --bam-threads=0`  once, not per --bam occurrence.
#   6. `--bam=0 --bam-threads=0`, `--bam=0`, bare `--bam`, SAM -> none.
#   7. usage error (`--bam=6 --bam-threads=0`, no reads) -> none: the run writes
#                                     nothing, so a warning about that output's
#                                     compression only buries the usage message.
#
# Every case also asserts the run's exit status, so a run that died before
# reaching the warning site cannot pass a "expected 0 warnings" case vacuously.
#
# Usage: test/bam_compress_warn_test.sh <bwa-mem3-binary> <fixtures-dir>

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

# expect <label> <want-rc-shape: zero|nonzero> <want-warnings> [mem args...]
# The exit status is asserted because it is what makes a zero-warning
# expectation meaningful: without it, a run that crashed during startup would
# also print zero warnings and satisfy the "none" cases vacuously.
expect() {
    local label="$1" want_rc="$2" want_warns="$3"
    shift 3
    local err="$tmpdir/$label.err"
    local rc=0 warns grep_rc=0

    "$bin" mem "$@" > /dev/null 2> "$err" || rc=$?
    warns="$(grep -c 'WARNING: --bam=' "$err")" || grep_rc=$?

    if ((grep_rc > 1)); then
        echo "FAIL: [$label] could not scan $err for warnings (grep exit $grep_rc)" >&2
        exit 1
    fi

    case "$want_rc" in
        zero) ((rc == 0)) || {
            echo "FAIL: [$label] 'mem $*' exited $rc, expected 0" >&2
            cat "$err" >&2
            exit 1
        } ;;
        nonzero) ((rc != 0)) || {
            echo "FAIL: [$label] 'mem $*' exited 0, expected non-zero" >&2
            cat "$err" >&2
            exit 1
        } ;;
        *)
            echo "FAIL: [$label] bad want_rc '$want_rc' (use zero|nonzero)" >&2
            exit 1
            ;;
    esac

    if [[ "$warns" != "$want_warns" ]]; then
        echo "FAIL: [$label] 'mem $*' printed $warns compressed-BAM warning(s), expected $want_warns" >&2
        echo "---- stderr ----" >&2
        cat "$err" >&2
        exit 1
    fi
}

# --- 1. Auto pool: compressed BAM does NOT warn. ----------------------------
expect auto zero 0 --bam=6 "$ref" "$reads"

# --- 2. Forced serial at a compressing level warns exactly once. ------------
expect forced_serial zero 1 --bam=6 --bam-threads=0 "$ref" "$reads"

# --- 3. Explicit positive pool does not warn. -------------------------------
expect explicit_pool zero 0 --bam=6 --bam-threads=4 "$ref" "$reads"

# --- 4. The resolved level is what matters: last --bam wins (uncompressed). --
expect forced_serial_then_off zero 0 --bam=6 --bam-threads=0 --bam=0 "$ref" "$reads"

# --- 5. Warns once, not per --bam occurrence. -------------------------------
expect repeated_forced_serial zero 1 --bam=6 --bam=6 --bam-threads=0 "$ref" "$reads"

# --- 6. Uncompressed BAM and SAM output are silent (nothing to deflate). -----
expect level_zero_forced_serial zero 0 --bam=0 --bam-threads=0 "$ref" "$reads"
expect level_zero zero 0 --bam=0 "$ref" "$reads"
expect bare_bam zero 0 --bam "$ref" "$reads"
expect sam zero 0 "$ref" "$reads"

# --- 7. A usage error produces no warning about output that never happens. ---
expect usage_error nonzero 0 --bam=6 --bam-threads=0 "$ref"

# --- 8. --bam-threads on SAM output (no --bam) warns it is inert. ------------
# Separate assertion because this warning's text is distinct from the
# compressed-BAM one the expect() helper counts.
no_bam_err="$tmpdir/no_bam.err"
"$bin" mem --bam-threads=4 "$ref" "$reads" > /dev/null 2> "$no_bam_err" \
    || {
        echo "FAIL: [no_bam] 'mem --bam-threads=4' (no --bam) exited non-zero" >&2
        cat "$no_bam_err" >&2
        exit 1
    }
if ! grep -q 'has no effect without --bam' "$no_bam_err"; then
    echo "FAIL: [no_bam] --bam-threads=4 without --bam did not warn it is inert" >&2
    cat "$no_bam_err" >&2
    exit 1
fi

echo "PASS: compressed-BAM warning fires once for user-forced serial deflate; --bam-threads-without-bam warns"
