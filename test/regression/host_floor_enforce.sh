#!/usr/bin/env bash
# test/regression/host_floor_enforce.sh
#
# Asserts that bwa-mem3 refuses cleanly (exit 2, clear stderr) when the
# host doesn't meet the build's SIMD floor. Uses the BWAMEM3_TESTING_HOST_TIER
# injection hook (only present in builds compiled with -DBWAMEM3_TESTING).
#
# Three scenarios:
#   1. 'bwa-mem3 mem ref r1.fq r2.fq' with a below-floor injected tier must
#      exit 2 with a stderr message containing the [E::bwamem3] precheck
#      header AND a "detected: <tier>" clause naming the injected tier.
#   2. 'bwa-mem3 version' with the same injection must exit 0 and emit
#      the [W::bwa-mem3] warning line on stderr (warnings always go to
#      stderr regardless of the version banner's stdout stream).
#   3. 'bwa-mem3 mem -h N ...' must ALSO exit 2: for mem, -h is the XA-hits
#      option (an INT), not a help alias, so it must not bypass the precheck.
#
# Inputs:
#   BWA_MEM3_TESTING — path to a binary built with `make TESTING_BUILD=1`
#   INJECTED_TIER    — tier name to inject (e.g. "sse41" for an avx2 build)
#   PARITY_FA        — pre-indexed reference fasta (any will do — precheck
#                      fires before the file is opened)

set -euo pipefail

: "${BWA_MEM3_TESTING:?BWA_MEM3_TESTING must be set (a TESTING_BUILD=1 binary)}"
: "${INJECTED_TIER:?INJECTED_TIER must be set (e.g. 'sse41')}"
: "${PARITY_FA:?PARITY_FA must be set}"

OUT_DIR="$(mktemp -d -t bwamem3-enforce-XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

# --- Scenario 1: bwa-mem3 mem refuses ---
rc=0
BWAMEM3_TESTING_HOST_TIER="$INJECTED_TIER" \
    "$BWA_MEM3_TESTING" mem "$PARITY_FA" /dev/null /dev/null \
    > "$OUT_DIR/mem.stdout" 2> "$OUT_DIR/mem.stderr" || rc=$?

if [[ $rc -ne 2 ]]; then
    echo "FAIL: bwa-mem3 mem exited $rc (expected 2)" >&2
    cat "$OUT_DIR/mem.stderr" >&2
    exit 1
fi

# Match the precheck error header rather than just the tier name. The bare
# tier name also appears in the unrelated "Executing in <tier> mode!!"
# banner that main.cpp prints to stderr — without anchoring on the
# [E::bwamem3] header, a regression that silently disabled the precheck
# would still pass the test by matching the banner.
if ! grep -q '\[E::bwamem3\]' "$OUT_DIR/mem.stderr"; then
    echo "FAIL: stderr does not contain the [E::bwamem3] precheck error header" >&2
    cat "$OUT_DIR/mem.stderr" >&2
    exit 1
fi

if ! grep -qE "detected:[[:space:]]*$INJECTED_TIER" "$OUT_DIR/mem.stderr"; then
    echo "FAIL: stderr does not name the injected host tier in the 'detected:' clause" >&2
    cat "$OUT_DIR/mem.stderr" >&2
    exit 1
fi

if ! grep -q 'BASELINE_ARCH' "$OUT_DIR/mem.stderr"; then
    echo "FAIL: stderr does not mention BASELINE_ARCH remediation" >&2
    cat "$OUT_DIR/mem.stderr" >&2
    exit 1
fi

# --- Scenario 2: bwa-mem3 version warns but exits 0 ---
rc=0
BWAMEM3_TESTING_HOST_TIER="$INJECTED_TIER" \
    "$BWA_MEM3_TESTING" version > "$OUT_DIR/version.stdout" 2> "$OUT_DIR/version.stderr" || rc=$?

if [[ $rc -ne 0 ]]; then
    echo "FAIL: bwa-mem3 version exited $rc (expected 0)" >&2
    cat "$OUT_DIR/version.stdout" "$OUT_DIR/version.stderr" >&2
    exit 1
fi

# Warnings go to stderr by [W::*] convention; the version banner's
# floor/runtime lines go to stdout. CI scripts that grep '^SIMD' on
# stdout must not see the warning.
if ! grep -qE '\[W::bwa-mem3\]' "$OUT_DIR/version.stderr"; then
    echo "FAIL: version stderr does not include [W::bwa-mem3] warning line" >&2
    cat "$OUT_DIR/version.stderr" >&2
    exit 1
fi

if grep -qE '\[W::bwa-mem3\]' "$OUT_DIR/version.stdout"; then
    echo "FAIL: warning leaked to stdout (must go to stderr only)" >&2
    cat "$OUT_DIR/version.stdout" >&2
    exit 1
fi

# --- Scenario 3: 'mem -h N' still runs the precheck (must exit 2) ---
# For `mem`, -h is the XA-hits option (takes an INT), NOT a help alias, so it
# must not bypass the host-floor precheck the way `mem --help` / `index -h` do.
rc=0
BWAMEM3_TESTING_HOST_TIER="$INJECTED_TIER" \
    "$BWA_MEM3_TESTING" mem -h 5 "$PARITY_FA" /dev/null /dev/null \
    > "$OUT_DIR/memh.stdout" 2> "$OUT_DIR/memh.stderr" || rc=$?

if [[ $rc -ne 2 ]]; then
    echo "FAIL: bwa-mem3 mem -h 5 exited $rc (expected 2 — -h must not bypass the precheck)" >&2
    cat "$OUT_DIR/memh.stderr" >&2
    exit 1
fi
if ! grep -q '\[E::bwamem3\]' "$OUT_DIR/memh.stderr"; then
    echo "FAIL: 'mem -h 5' stderr lacks the [E::bwamem3] precheck header (precheck skipped?)" >&2
    cat "$OUT_DIR/memh.stderr" >&2
    exit 1
fi

echo "PASS: bwa-mem3 mem (and 'mem -h N') exits 2 and bwa-mem3 version warns on injected too-old host"
