#!/usr/bin/env bash
# test/pg_cl_escape_test.sh
#
# Asserts that `bwa-mem3 mem -R $'@RG\tID:x\tSM:y\tLB:z' ...` produces a
# SAM header whose `@PG` line contains no literal tabs, newlines, or
# carriage returns inside the `CL:` value. Regression for issue #45 /
# upstream #293: without escaping, whitespace from the user-supplied `-R`
# argument bleeds into the `@PG` line and strict SAM validators
# (noodles, some fgbio configs) reject it.
#
# The assertion is structural, not textual:
#   * The @PG line must have exactly 5 tab-separated fields
#     (@PG, ID:bwa-mem3, PN:bwa-mem3, VN:..., CL:...).
#   * There must be exactly one @PG line.
#   * The CL: value must be free of raw \t, \n, and \r.
#   * The @RG line must still contain the user's tab-separated fields
#     (tab-only case), because argv itself is not mutated.
#
# Usage: test/pg_cl_escape_test.sh <bwa-mem3-binary> <fixtures-dir>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <bwa-mem3-binary> <fixtures-dir>" >&2
    exit 2
fi

bin="$1"
fixtures="$2"
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
    "$bin" index "$ref" > /dev/null 2>&1 || {
        echo "FAIL: bwa-mem3 index on phix.fa failed" >&2
        exit 1
    }
fi

sam="$(mktemp)"
err="$(mktemp)"
trap 'rm -f "$sam" "$err"' EXIT

# Run `bwa-mem3 mem` with the given -R value and assert that the emitted
# @PG line is structurally well-formed. $1 is a short label used in
# failure messages; $2 is the -R string.
assert_pg_well_formed() {
    local label="$1"
    local rg="$2"
    # Optional: the exact sanitized -R token expected to survive intact inside
    # the CL: value (tabs collapsed to single spaces by append_pg_cl_sanitized).
    # When set, assert the whole token is present, not just the structural and
    # trailing-token checks below.
    local expected_token="${3:-}"

    "$bin" mem -R "$rg" "$ref" "$reads" > "$sam" 2> "$err" || {
        echo "FAIL [$label]: bwa-mem3 mem exited non-zero" >&2
        echo "---- stderr ----" >&2
        cat "$err" >&2
        echo "----------------" >&2
        exit 1
    }

    local pg_lines
    pg_lines="$(grep -c '^@PG' "$sam" || true)"
    if [[ "$pg_lines" -ne 1 ]]; then
        echo "FAIL [$label]: expected exactly one @PG line, got $pg_lines" >&2
        grep '^@PG' "$sam" >&2 || true
        exit 1
    fi

    local pg_line
    pg_line="$(grep '^@PG' "$sam")"

    # Field count: @PG plus 4 tags (ID, PN, VN, CL) = 5 tab-separated columns.
    # Any extra column means a tab leaked into one of the tag values.
    local pg_field_count
    pg_field_count="$(awk -F'\t' '{print NF}' <<< "$pg_line")"
    if [[ "$pg_field_count" -ne 5 ]]; then
        echo "FAIL [$label]: @PG line has $pg_field_count tab-separated fields, expected 5" >&2
        echo "---- @PG line (tabs shown as <TAB>) ----" >&2
        printf '%s\n' "$pg_line" | sed $'s/\t/<TAB>/g' >&2
        echo "----------------------------------------" >&2
        exit 1
    fi

    # Belt-and-braces: exactly one `ID:` tag and exactly one `CL:` tag.
    # Stray tabs from -R would produce a second `ID:` tag (from the RG).
    #
    # `|| true` on each grep: a zero count is exactly the failure this block
    # exists to report, but grep exits 1 when it matches nothing, and under
    # `set -o pipefail` that would end the script before the assertion could
    # print which tag was missing.
    local id_count cl_count
    id_count="$({ grep -o $'\tID:' <<< "$pg_line" || true; } | wc -l | tr -d ' ')"
    cl_count="$({ grep -o $'\tCL:' <<< "$pg_line" || true; } | wc -l | tr -d ' ')"
    if [[ "$id_count" -ne 1 || "$cl_count" -ne 1 ]]; then
        echo "FAIL [$label]: @PG line has ID:=$id_count CL:=$cl_count (expected 1 of each)" >&2
        printf '%s\n' "$pg_line" | sed $'s/\t/<TAB>/g' >&2
        exit 1
    fi

    # Extract the CL: value. grep above was line-oriented, so if the raw
    # argv contained a \n it would have terminated $pg_line early and
    # $cl_value would be missing the trailing argv tokens. Assert that
    # the CL: value ends with the reads path (the last argv token) —
    # this catches an embedded \n that truncated the @PG line.
    local cl_value
    cl_value="$(awk -F'\t' '{for (i=1;i<=NF;i++) if ($i ~ /^CL:/) { sub(/^CL:/, "", $i); print $i; exit }}' <<< "$pg_line")"
    if [[ "$cl_value" != *"$reads" ]]; then
        echo "FAIL [$label]: CL: value does not end with reads path '$reads'" >&2
        echo "CL: was: $cl_value" >&2
        exit 1
    fi
    if [[ "$cl_value" == *$'\t'* ]]; then
        echo "FAIL [$label]: CL: value contains a literal tab" >&2
        exit 1
    fi
    if [[ "$cl_value" == *$'\r'* ]]; then
        echo "FAIL [$label]: CL: value contains a literal carriage return" >&2
        exit 1
    fi

    # When an expected sanitized token is supplied, assert it appears verbatim
    # in the CL: value. The trailing-token and no-tab checks above still pass on
    # a value whose SM: field was silently truncated (later argv tokens follow
    # the -R value), so a growth-math bug can corrupt the token without tripping
    # them — this substring check is what actually catches it.
    if [[ -n "$expected_token" && "$cl_value" != *"$expected_token"* ]]; then
        echo "FAIL [$label]: CL: value does not contain expected -R token" >&2
        echo "expected substring: $expected_token" >&2
        echo "CL: was: $cl_value" >&2
        exit 1
    fi
}

# --- @PG checks: embed each whitespace variant in -R and verify CL: clean -
# All three values parse as a valid @RG (they start with "@RG\tID:" before
# the injected character); bwa-mem3 will emit a @RG line but we only
# assert @PG structure here — that is the fix under test.
assert_pg_well_formed "tab" $'@RG\tID:x\tSM:y\tLB:z'
assert_pg_well_formed "newline" $'@RG\tID:x\tSM:y\tLB:z\nTRAIL'
assert_pg_well_formed "carriage-return" $'@RG\tID:x\tSM:y\tLB:z\rTRAIL'

# A long -R value drives the @PG CL: kstring through many grow-and-realloc
# cycles (append_pg_cl_sanitized calls kputc once per input byte). Regression
# for the buffer-growth bookkeeping added alongside the CL: kstring's OOM
# guard: assert_pg_well_formed's checks (single well-formed @PG line, no
# leaked tab/CR, CL: ends with the reads path) still catch a growth-math bug
# that corrupts or truncates the value.
long_sm="$(printf 'x%.0s' $(seq 1 100000))"
assert_pg_well_formed "long-arg-many-growths" \
    $'@RG\tID:x\tSM:'"$long_sm"$'\tLB:z' \
    "@RG ID:x SM:${long_sm} LB:z"

# --- @RG assertions (tab-only case) ----------------------------------------
# argv is not mutated, so the @RG line the user asked for must still be
# well-formed and carry ID:x, SM:y, LB:z as distinct tab-separated tags.
# Re-run with the plain tab-only -R; $sam already has output from the
# "tab" call above but re-running keeps the test self-contained if the
# call order above ever changes.
rg=$'@RG\tID:x\tSM:y\tLB:z'
"$bin" mem -R "$rg" "$ref" "$reads" > "$sam" 2> "$err" || {
    echo "FAIL: bwa-mem3 mem exited non-zero (@RG assertion)" >&2
    echo "---- stderr ----" >&2
    cat "$err" >&2
    echo "----------------" >&2
    exit 1
}
rg_line="$(grep '^@RG' "$sam" || true)"
[[ -n "$rg_line" ]] || {
    echo "FAIL: no @RG line in output" >&2
    exit 1
}

for tag in $'\tID:x' $'\tSM:y' $'\tLB:z'; do
    if ! grep -qF -- "$tag" <<< "$rg_line"; then
        echo "FAIL: @RG line missing tag '${tag//$'\t'/<TAB>}'" >&2
        printf '%s\n' "$rg_line" | sed $'s/\t/<TAB>/g' >&2
        exit 1
    fi
done

echo "PASS: @PG line contains no embedded tabs/newlines/CRs; @RG preserved"
