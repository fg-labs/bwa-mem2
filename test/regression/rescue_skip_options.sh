#!/usr/bin/env bash
# test/regression/rescue_skip_options.sh
#
# Regression: --rescue-skip must be rejected when it cannot do anything, and
# must NOT reject the supported `--fast --rescue-kmer=0` opt-out.
#
# --rescue-skip drops the mate-rescue Smith-Waterman entirely for a pair whose
# k-mer anchor does not clear the vote floor. The decision reuses the
# --rescue-kmer anchor scan, so with the scan off there is nothing to decide on
# and the flag would be silently inert -- a flag that changes which reads get
# rescued going quietly nowhere. Hence: hard error, not a no-op.
#
# --fast enables --rescue-kmer=6 but deliberately NOT --rescue-skip: the skip
# gate drops rescues rather than shortening them, and on real reads that costs
# confident alignments (16 losses at MAPQ>=30 per 200k primaries on 125 bp WGBS,
# 420 per 2M on 75 bp em-seq). The validation still runs AFTER getopt rather than
# inline in the case, so the diagnostic does not depend on flag order.
#
# This test pins that contract:
#   * --rescue-skip alone                -> hard error
#   * --rescue-skip --rescue-kmer=0      -> hard error, either order
#   * --rescue-skip --rescue-kmer=6      -> accepted, either order
#   * --rescue-skip=false                -> accepted, needs no anchor scan (off)
#   * --rescue-skip=true --rescue-kmer=6 -> accepted; =true without it -> error
#   * --rescue-skip=bogus / =1           -> hard error (surface is exactly true|false)
#   * --rescue-skip false (space form)   -> stray-value diagnostic, not silent enable
#   * --fast                             -> does NOT enable it (opt-in only)
#   * --fast --rescue-skip                -> accepted, named on the audit line
#   * --fast --rescue-kmer=0             -> accepted, clean no-op
#
# Inputs:
#   BWA_MEM3 — path to bwa-mem3 binary
#   FIXTURES — directory containing phix.fa and reads.fa (default: test/fixtures)
set -euo pipefail

: "${BWA_MEM3:?BWA_MEM3 must be set}"
FIXTURES="${FIXTURES:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../fixtures" && pwd)}"

src_ref="$FIXTURES/phix.fa"
reads="$FIXTURES/reads.fa"
[[ -x "$BWA_MEM3" ]] || {
    echo "FAIL: binary not executable: $BWA_MEM3" >&2
    exit 1
}
[[ -s "$src_ref" ]] || {
    echo "FAIL: phix.fa missing: $src_ref" >&2
    exit 1
}
[[ -s "$reads" ]] || {
    echo "FAIL: reads.fa missing: $reads" >&2
    exit 1
}

# Index a private copy so the test never writes into the fixtures tree.
mdir="$(mktemp -d)"
err="$mdir/err.log"
trap 'rm -rf "$mdir"' EXIT
ref="$mdir/phix.fa"
cp "$src_ref" "$ref"
"$BWA_MEM3" index "$ref" > /dev/null 2>&1 || {
    echo "FAIL: index phix.fa" >&2
    exit 1
}

fails=0
skip_err='ERROR: --rescue-skip requires --rescue-kmer'

# Run with an arbitrary number of option words. "$@" = options only.
run() { "$BWA_MEM3" mem "$@" "$ref" "$reads" > /dev/null 2> "$err"; }

# All args are option words; the whole invocation must fail with $skip_err.
reject() {
    local rc
    set +e
    run "$@"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
        echo "FAIL: '$*' was accepted; expected a non-zero exit"
        fails=$((fails + 1))
    elif ! grep -q "$skip_err" "$err"; then
        echo "FAIL: '$*' exited $rc but without the expected diagnostic ('$skip_err')"
        cat "$err" >&2
        fails=$((fails + 1))
    else
        echo "  ok: '$*' rejected"
    fi
}

# All args are option words; the whole invocation must succeed.
accept() {
    if run "$@"; then
        echo "  ok: '$*' accepted"
    else
        echo "FAIL: '$*' was rejected; expected it to run"
        cat "$err" >&2
        fails=$((fails + 1))
    fi
}

# --- --rescue-skip needs an anchor scan to key on ----------------------------
reject "--rescue-skip"
reject "--rescue-skip" "--rescue-kmer=0"
# Order-independent: the check runs after getopt, so neither ordering may slip
# through. Before the check was moved out of the getopt case, one of these two
# passed and the other did not, purely on parse order.
reject "--rescue-kmer=0" "--rescue-skip"

# --- with a live anchor scan it is accepted, either order --------------------
accept "--rescue-kmer=6" "--rescue-skip"
accept "--rescue-skip" "--rescue-kmer=6"
accept "--rescue-skip" "--rescue-kmer" # bare --rescue-kmer resolves to the default K

# --- --rescue-skip[=true|false]: optional true|false argument ----------------
# The flag takes an OPTIONAL true|false value (bare = true). =false is the
# explicit opt-out and, being off, needs no anchor scan; =true behaves like the
# bare form and still requires --rescue-kmer. Any other value hard-errors, and
# the value must be attached with '=' -- a space-separated `--rescue-skip false`
# orphans `false` into a positional, caught by the stray-value diagnostic rather
# than silently enabling the skip.

# Reject with a specific diagnostic: exit non-zero AND the message present.
reject_with() {
    local want=$1
    shift
    local rc
    set +e
    run "$@"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
        echo "FAIL: '$*' was accepted; expected a non-zero exit"
        fails=$((fails + 1))
    elif ! grep -qF "$want" "$err"; then
        echo "FAIL: '$*' exited $rc but without the expected diagnostic ('$want')"
        cat "$err" >&2
        fails=$((fails + 1))
    else
        echo "  ok: '$*' rejected"
    fi
}

accept "--rescue-skip=false"                  # off: no anchor scan needed
accept "--rescue-skip=true" "--rescue-kmer=6" # on, explicit
reject "--rescue-skip=true"                   # on without --rescue-kmer -> requires it
# A malformed value hard-errors, naming the accepted vocabulary. 1/0 are NOT
# aliases (the surface is exactly true|false).
reject_with "accepts true|false" "--rescue-skip=bogus"
reject_with "accepts true|false" "--rescue-skip=1"
# The value must be attached with '='. A space form orphans 'false' into a
# positional; the stray-value diagnostic points at the '=' form rather than
# silently enabling the skip and swallowing 'false' as <idxbase>.
reject_with "write --rescue-skip=false, not --rescue-skip false" \
    "--rescue-skip" "false" "--rescue-kmer=6"

# --- interaction with --fast -------------------------------------------------
# --rescue-skip is deliberately NOT in --fast: it drops rescues rather than
# shortening them, and on real reads that costs confident alignments. --fast
# turns on --rescue-kmer only, so the audit line must NOT name --rescue-skip.
# Pinning the absence keeps a future "add it to the preset" from landing without
# the recall evidence that would justify it.
if run "--fast"; then
    if grep -q -- '--rescue-skip' "$err"; then
        echo "FAIL: --fast audit line names --rescue-skip; the skip gate is opt-in only"
        grep -- '--fast:' "$err" >&2 || true
        fails=$((fails + 1))
    else
        echo "  ok: --fast does not enable --rescue-skip"
    fi
else
    echo "FAIL: '--fast' was rejected"
    cat "$err" >&2
    fails=$((fails + 1))
fi

# Explicitly combining them is still supported, and then it IS on the audit line
# -- the line is the only run-time record of a lever that changes output.
if run "--fast" "--rescue-skip"; then
    if grep -q -- '--rescue-skip' "$err"; then
        echo "  ok: '--fast --rescue-skip' reports --rescue-skip on the audit line"
    else
        echo "FAIL: '--fast --rescue-skip' does not report --rescue-skip"
        fails=$((fails + 1))
    fi
else
    echo "FAIL: '--fast --rescue-skip' was rejected"
    cat "$err" >&2
    fails=$((fails + 1))
fi

# The opt-out that must keep working: --rescue-kmer=0 disables the anchor scan,
# and since --fast does not preset --rescue-skip this stays a clean no-op rather
# than tripping the "requires --rescue-kmer" error.
if run "--fast" "--rescue-kmer=0"; then
    echo "  ok: '--fast --rescue-kmer=0' accepted"
else
    echo "FAIL: '--fast --rescue-kmer=0' was rejected; it is a supported opt-out"
    cat "$err" >&2
    fails=$((fails + 1))
fi

if [[ "$fails" -ne 0 ]]; then
    echo "FAIL: rescue-skip option validation ($fails failure(s))"
    exit 1
fi
echo "PASS: --rescue-skip requires --rescue-kmer and composes with --fast"
