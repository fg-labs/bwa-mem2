#!/usr/bin/env bash
# test/regression/compat_byte_identical.sh
#
# Regression: `bwa-mem3 mem --compat=bwa-mem2` must suppress exactly the
# three things that keep the drop-in profile from being a byte-for-byte
# match to bwa-mem2 — the `MQ:i` and `HN:i` tags and the default `@HD`
# line — and must change NOTHING else.
#
# `@HD`: bwa-mem2 v2.2.1 emits none at all (its bwa_print_sam_hdr has no
# @HD logic); bwa gained one only in 0.7.18, after bwa-mem2 forked at
# 0.7.17. Suppressing ours is deliberate. The sidecar `@SQ` half of header
# parity is covered by compat_header_parity.sh, which needs an ALT-aware
# fixture; phix here has no sidecar, so its `@SQ` block is identical
# either way and this script's byte-diff still covers it.
#
# `@PG` is deliberately NOT suppressed: bwa-mem2 emits its own, so
# dropping ours would turn a changed line into a missing one, and `CL:`
# embeds the invocation either way. It is excluded from the comparison
# below (the default and --compat runs have different argv) and asserted
# to still be present.
#
# The load-bearing assertion is byte-identity: a default run with its
# MQ:i / HN:i tags stripped must be identical, byte-for-byte, to a
# `--compat` run of the same inputs. If --compat ever suppresses too much
# (drops another tag, drops @HD/@SQ) or too little (leaks MQ/HN), or
# perturbs an alignment, the streams diverge and this fails. Both the
# SAM-text and --bam paths are checked, since the tags are emitted from
# two independent code paths (bwamem.cpp and bam_writer.cpp) reading the
# same compat target row (src/compat_target.cpp).
#
# Fixture (deterministic, no PRNG): PE reads are sliced directly from the
# committed phix.fa so the FASTQ is byte-identical across awk
# implementations. read1 is a forward window; read2 is the reverse
# complement of a downstream window, so every pair aligns concordantly
# and carries a mate (=> MQ:i is emitted) and a primary hit count
# (=> HN:i is emitted). Several pairs guarantee non-zero tag counts.
#
# Inputs (env vars):
#   BWA_MEM3         — path to the bwa-mem3 binary under test
#   COMPAT_PHIX_FA   — path to test/fixtures/phix.fa (the reference source)
#   COMPAT_WORK_DIR  — directory for fixture-private intermediates

set -euo pipefail

: "${BWA_MEM3:?BWA_MEM3 must be set}"
: "${COMPAT_PHIX_FA:?COMPAT_PHIX_FA must be set}"
: "${COMPAT_WORK_DIR:?COMPAT_WORK_DIR must be set}"

mkdir -p "$COMPAT_WORK_DIR"

ref="$COMPAT_WORK_DIR/phix.fa"
cp "$COMPAT_PHIX_FA" "$ref"

# --- Build the PE FASTQs by slicing phix (deterministic, no PRNG). ---
# Flatten the reference to a single sequence, then emit N pairs at fixed
# offsets: read1 = ref[off .. off+L], read2 = revcomp(ref[off+G .. off+G+L]).
seq="$(grep -v '^>' "$ref" | tr -d '\n' | tr 'acgt' 'ACGT')"
r1="$COMPAT_WORK_DIR/r1.fq"
r2="$COMPAT_WORK_DIR/r2.fq"
: > "$r1"
: > "$r2"
L=120 # read length
G=300 # inner gap between the two windows (fragment ~ G+L)
n_pairs=0
for off in 200 900 1600 2300 3000 3700; do
    s1="${seq:$off:$L}"
    s2fwd="${seq:$((off + G)):$L}"
    # reverse complement of s2fwd
    s2="$(printf '%s' "$s2fwd" | rev | tr 'ACGT' 'TGCA')"
    [ "${#s1}" -eq "$L" ] && [ "${#s2}" -eq "$L" ] || continue
    q="$(printf 'I%.0s' $(seq 1 "$L"))"
    printf '@pair%d/1\n%s\n+\n%s\n' "$n_pairs" "$s1" "$q" >> "$r1"
    printf '@pair%d/2\n%s\n+\n%s\n' "$n_pairs" "$s2" "$q" >> "$r2"
    n_pairs=$((n_pairs + 1))
done

if [ "$n_pairs" -lt 3 ]; then
    echo "FAIL: fixture built only $n_pairs pairs (phix slicing regression?)" >&2
    exit 1
fi
echo "fixture: $n_pairs PE pairs sliced from phix"

# --- Index the reference. ---
"$BWA_MEM3" index "$ref" > "$COMPAT_WORK_DIR/index.log" 2>&1

# --- SAM-text path: default vs --compat. ---
def_sam="$COMPAT_WORK_DIR/default.sam"
cmp_sam="$COMPAT_WORK_DIR/compat.sam"
"$BWA_MEM3" mem "$ref" "$r1" "$r2" > "$def_sam" 2> /dev/null
"$BWA_MEM3" mem --compat=bwa-mem2 "$ref" "$r1" "$r2" > "$cmp_sam" 2> /dev/null

# The default run must actually contain the tags we claim to suppress,
# else the byte-identity check below would pass vacuously.
def_mq=$(grep -c 'MQ:i:' "$def_sam" || true)
def_hn=$(grep -c 'HN:i:' "$def_sam" || true)
if [ "$def_mq" -lt 1 ] || [ "$def_hn" -lt 1 ]; then
    echo "FAIL: default SAM lacks MQ/HN to suppress (MQ=$def_mq HN=$def_hn) — fixture too weak" >&2
    exit 1
fi

# --compat must emit neither tag.
cmp_mq=$(grep -c 'MQ:i:' "$cmp_sam" || true)
cmp_hn=$(grep -c 'HN:i:' "$cmp_sam" || true)
if [ "$cmp_mq" -ne 0 ] || [ "$cmp_hn" -ne 0 ]; then
    echo "FAIL: --compat SAM still emits MQ=$cmp_mq HN=$cmp_hn (expected 0/0)" >&2
    exit 1
fi

# The default run must emit an @HD (else the suppression check is vacuous),
# and --compat must emit none: bwa-mem2 writes no @HD at all.
if ! grep -q '^@HD' "$def_sam"; then
    echo "FAIL: default SAM has no @HD to suppress — fixture or default changed" >&2
    exit 1
fi
if grep -q '^@HD' "$cmp_sam"; then
    echo "FAIL: --compat SAM still emits @HD (bwa-mem2 emits none):" >&2
    grep '^@HD' "$cmp_sam" >&2
    exit 1
fi

# @SQ must survive. Whether --compat SUPPRESSES a sidecar's identity tags is
# not testable here -- phix has no .hdr/.dict, so there would be nothing to
# suppress and the assertion would pass vacuously. header_parity.sh covers it
# against a real `samtools dict` sidecar.
if ! grep -q '^@SQ' "$cmp_sam"; then
    echo "FAIL: --compat SAM dropped the @SQ header block" >&2
    exit 1
fi

# @PG is preserved -- suppressing it would only trade a changed line for a
# missing one (bwa-mem2 emits its own).
if ! grep -q '^@PG.*ID:bwa-mem3' "$cmp_sam"; then
    echo "FAIL: --compat SAM dropped the bwa-mem3 @PG line (it must be preserved)" >&2
    exit 1
fi

# The load-bearing check: default (minus MQ/HN/@HD) is byte-identical to
# --compat. @PG is excluded from BOTH sides -- its CL: records the actual
# argv, which necessarily differs between the two invocations. @HD is
# excluded from both because its presence/absence is asserted directly
# above; excluding it here keeps THIS diff about everything else.
def_stripped="$COMPAT_WORK_DIR/default.stripped.sam"
cmp_stripped="$COMPAT_WORK_DIR/compat.stripped.sam"
grep -vE '^@PG|^@HD' "$def_sam" \
    | sed -E 's/\tMQ:i:[0-9-]+//; s/\tHN:i:[0-9-]+//' > "$def_stripped"
grep -vE '^@PG|^@HD' "$cmp_sam" > "$cmp_stripped"
if ! diff -q "$def_stripped" "$cmp_stripped" > /dev/null; then
    echo "FAIL: --compat SAM differs from default beyond MQ/HN/@HD:" >&2
    diff "$def_stripped" "$cmp_stripped" | head -20 >&2
    exit 1
fi
echo "PASS: SAM-text --compat is byte-identical to default minus MQ/HN/@HD (@PG excluded)"

# --- BAM path: same assertion, if samtools is available. ---
if command -v samtools > /dev/null 2>&1; then
    def_bam="$COMPAT_WORK_DIR/default.bam"
    cmp_bam="$COMPAT_WORK_DIR/compat.bam"
    "$BWA_MEM3" mem --bam "$ref" "$r1" "$r2" > "$def_bam" 2> /dev/null
    "$BWA_MEM3" mem --bam --compat=bwa-mem2 "$ref" "$r1" "$r2" > "$cmp_bam" 2> /dev/null

    # --no-PG so samtools does not inject its own @PG line into the header dump.
    if [ "$(samtools view -H --no-PG "$cmp_bam" | grep -c '^@PG.*ID:bwa-mem3')" -ne 1 ]; then
        echo "FAIL: --compat BAM lacks the bwa-mem3 @PG line (it must be preserved)" >&2
        exit 1
    fi
    # The BAM header is built by a separate writer (bam_writer.cpp), so
    # assert @HD suppression independently of the SAM-text path above.
    if [ "$(samtools view -H --no-PG "$def_bam" | grep -c '^@HD')" -lt 1 ]; then
        echo "FAIL: default BAM has no @HD to suppress" >&2
        exit 1
    fi
    if [ "$(samtools view -H --no-PG "$cmp_bam" | grep -c '^@HD')" -ne 0 ]; then
        echo "FAIL: --compat BAM still emits @HD (bwa-mem2 emits none)" >&2
        exit 1
    fi
    bam_def_rec="$COMPAT_WORK_DIR/default.bam.rec"
    bam_cmp_rec="$COMPAT_WORK_DIR/compat.bam.rec"
    samtools view --no-PG "$def_bam" \
        | sed -E 's/\tMQ:i:[0-9-]+//; s/\tHN:i:[0-9-]+//' > "$bam_def_rec"
    samtools view --no-PG "$cmp_bam" > "$bam_cmp_rec"
    if ! diff -q "$bam_def_rec" "$bam_cmp_rec" > /dev/null; then
        echo "FAIL: --compat BAM records differ from default beyond MQ/HN:" >&2
        diff "$bam_def_rec" "$bam_cmp_rec" | head -20 >&2
        exit 1
    fi
    echo "PASS: BAM --compat suppresses @HD; records byte-identical to default minus MQ/HN"
else
    echo "SKIP: samtools not on PATH — BAM-path assertion skipped"
fi

# --- --compat and --fast are mutually exclusive: must be a hard error. ---
# (A diff-clean-looking stream over --fast's changed alignments would defeat
# the parity-validation purpose of --compat, so combining them is rejected.)
if "$BWA_MEM3" mem --compat=bwa-mem2 --fast "$ref" "$r1" "$r2" > /dev/null 2> "$COMPAT_WORK_DIR/mutex.log"; then
    echo "FAIL: --compat --fast exited 0 (expected a hard error)" >&2
    exit 1
fi
if ! grep -q 'mutually exclusive' "$COMPAT_WORK_DIR/mutex.log"; then
    echo "FAIL: --compat --fast failed without the expected 'mutually exclusive' message:" >&2
    cat "$COMPAT_WORK_DIR/mutex.log" >&2
    exit 1
fi
echo "PASS: --compat --fast is rejected as a hard error"

# --- --compat is non-meth only: --compat --meth must be a hard error. ---
# bwa-mem2 has no methylation mode, so byte-identity is undefined under --meth.
# The guard fires during option validation, before any index load, so the
# non-meth phix reference here is sufficient to exercise it.
if "$BWA_MEM3" mem --compat=bwa-mem2 --meth "$ref" "$r1" "$r2" > /dev/null 2> "$COMPAT_WORK_DIR/meth.log"; then
    echo "FAIL: --compat --meth exited 0 (expected a hard error)" >&2
    exit 1
fi
if ! grep -q 'not supported with --meth' "$COMPAT_WORK_DIR/meth.log"; then
    echo "FAIL: --compat --meth failed without the expected 'not supported with --meth' message:" >&2
    cat "$COMPAT_WORK_DIR/meth.log" >&2
    exit 1
fi
echo "PASS: --compat --meth is rejected as a hard error"

# --- --compat is an enum: check the grammar the CLI advertises. ---
# `--compat` takes a required argument, so every accepted spelling and every
# rejection below is a contract the docs state. A regression here is silent
# otherwise -- a wrong optstring turns `--compat=off` into a hard error, or
# swallows the index prefix as if it were a target name.
compat_ok() { # spelling... -- must exit 0
    if ! "$BWA_MEM3" mem "$@" "$ref" "$r1" "$r2" > /dev/null 2> "$COMPAT_WORK_DIR/cli.log"; then
        echo "FAIL: 'mem $*' exited nonzero, expected success:" >&2
        cat "$COMPAT_WORK_DIR/cli.log" >&2
        exit 1
    fi
}
compat_err() { # <expected-substring> <spelling...> -- must exit nonzero AND say why
    want="$1"
    shift
    if "$BWA_MEM3" mem "$@" "$ref" "$r1" "$r2" > /dev/null 2> "$COMPAT_WORK_DIR/cli.log"; then
        echo "FAIL: 'mem $*' exited 0, expected a hard error" >&2
        exit 1
    fi
    if ! grep -q "$want" "$COMPAT_WORK_DIR/cli.log"; then
        echo "FAIL: 'mem $*' failed without the expected message '$want':" >&2
        cat "$COMPAT_WORK_DIR/cli.log" >&2
        exit 1
    fi
}
compat_warn() { # <expected-substring> <spelling...> -- must exit 0 AND warn
    want="$1"
    shift
    if ! "$BWA_MEM3" mem "$@" "$ref" "$r1" "$r2" > /dev/null 2> "$COMPAT_WORK_DIR/cli.log"; then
        echo "FAIL: 'mem $*' exited nonzero, expected success with a warning:" >&2
        cat "$COMPAT_WORK_DIR/cli.log" >&2
        exit 1
    fi
    if ! grep -q "$want" "$COMPAT_WORK_DIR/cli.log"; then
        echo "FAIL: 'mem $*' succeeded but without the expected warning '$want':" >&2
        cat "$COMPAT_WORK_DIR/cli.log" >&2
        exit 1
    fi
}

compat_ok --compat=bwa-mem2   # canonical, '=' form
compat_ok --compat bwa-mem2   # canonical, space form (required_argument)
compat_ok --compat=mem2       # documented alias
compat_ok --compat=off        # explicit no-op
compat_ok --compat=off --fast # off must NOT trip the --fast guard
compat_ok --compat=bwa-mem    # selectable since 0.9.0
compat_ok --compat bwa-mem    # space form, and NOT confusable with bwa-mem2
compat_err "unknown --compat target" --compat=bogus
# Exact match only. `bwa-mem` is a prefix of `bwa-mem2`, so a lookup that ever
# grew prefix or case folding would silently route one target to the other --
# and they disagree on MQ:i and @HD, which is the entire point of the enum.
compat_err "unknown --compat target" --compat=bwa
compat_err "unknown --compat target" --compat=BWA-MEM
# --proper-pair-from-emitted deviates from BOTH upstreams on FLAG 0x2, so asking
# for it alongside a byte-identity target is contradictory (fg-labs/bwa-mem3#362).
# Same contract as --fast. It must be accepted on its own, or the opt-in escape
# hatch #17's reasoning justifies would not exist.
compat_err "mutually exclusive" --compat=bwa-mem2 --proper-pair-from-emitted
compat_err "mutually exclusive" --compat=bwa-mem --proper-pair-from-emitted
compat_ok --proper-pair-from-emitted
compat_ok --compat=off --proper-pair-from-emitted
# The chain-extension cap (--extend-tie-frac / --extend-tie-floor / --extend-csub)
# prunes chains and recalibrates MAPQ, so a setting that can actually prune a
# chain changes alignments just like --fast. It is folded into --fast (rejected
# above) but the levers are also settable on their own, so
# `--compat --extend-tie-frac=0.95 --extend-csub` would otherwise diff clean
# over non-compatible alignments. Reject only configurations that can actually
# prune: --extend-tie-frac > 0 gates on its own, regardless of
# --max-extend-chains; --extend-csub only matters once something was dropped,
# which --max-extend-chains alone (the plain count cap) can still do even with
# --extend-tie-frac at 0. --extend-tie-floor is meaningful only when
# --extend-tie-frac > 0 (mem_chain_cap_extend's gate is 0 otherwise, so the
# floor never changes a keep decision), so a bare floor is always a no-op and
# must NOT trip the guard.
compat_err "mutually exclusive" --compat=bwa-mem2 --extend-tie-frac=0.95
compat_err "mutually exclusive" --extend-tie-frac=0.95 --compat=bwa-mem2
compat_err "mutually exclusive" --compat=bwa-mem2 --extend-tie-frac=0.95 --extend-csub
compat_err "mutually exclusive" --compat=bwa-mem2 --extend-csub --max-extend-chains=5
# --extend-tie-floor alone (--extend-tie-frac at 0) is always a no-op, so --compat
# must accept it. But --max-extend-chains != 0 is itself an independent divergent
# lever (the plain count cap prunes chains on its own), so pairing it with --compat
# IS an error now, regardless of --extend-tie-floor.
compat_ok --compat=bwa-mem2 --extend-tie-floor=1
compat_err "mutually exclusive" --compat=bwa-mem2 --extend-tie-floor=1 --max-extend-chains=5
# --extend-csub alone, with no cap active to ever populate capped_w, is also a
# no-op.
compat_ok --compat=bwa-mem2 --extend-csub
# An explicit off-state value resolves to the byte-identical default, so it must
# NOT trip the guard -- the guard keys off the effective value, not whether the
# flag was typed.
compat_ok --compat=bwa-mem2 --extend-tie-frac=0
compat_ok --compat=off --extend-tie-frac=0.95 --extend-csub

# --- Every bwa-mem3-only alignment/MAPQ lever is guarded under --compat. -------
# These are all levers no upstream (bwa-mem / bwa-mem2) can express, so engaging
# any of them under a byte-identity target changes alignments/MAPQ and must be
# refused. The guard is centralized, so one row per lever; test them here so a
# newly-added lever that forgets the guard fails loudly. Checked against BOTH
# targets for the count cap (they share the reject list; only the output-shaping
# struct differs by target).
compat_err "mutually exclusive" --compat=bwa-mem2 --max-extend-chains=5
compat_err "mutually exclusive" --compat=bwa-mem --max-extend-chains=5
compat_err "mutually exclusive" --compat=bwa-mem2 --min-ext-len 30
compat_err "mutually exclusive" --compat=bwa-mem2 --smem-dedup
compat_err "mutually exclusive" --compat=bwa-mem2 --adaptive-band
compat_err "mutually exclusive" --compat=bwa-mem2 --rescue-kmer
compat_err "mutually exclusive" --compat=bwa-mem2 --supp-rep-hard-cap=10
compat_err "mutually exclusive" --compat=bwa-mem2 --seed-order local-longest
compat_err "mutually exclusive" --compat=bwa-mem2 --skip-contained-ext
compat_err "mutually exclusive" --compat=bwa-mem2 --chunk-cap 1000
# The error must NAME the offending flag(s) (the guard builds the list at runtime),
# so assert on the flag name, not just the boilerplate. A two-lever case pins that
# the accumulation lists more than one.
compat_err "smem-dedup" --compat=bwa-mem2 --smem-dedup
compat_err "adaptive-band" --compat=bwa-mem2 --smem-dedup --adaptive-band
# Order independence: the guard must fire whichever side the target is on.
compat_err "mutually exclusive" --smem-dedup --compat=bwa-mem2
# Off-state / no-cap values are the byte-identical default and must NOT trip it --
# the guard keys off the effective value, not whether the flag was typed. Cover the
# value-taking levers explicitly (their nonzero forms are exercised above).
compat_ok --compat=bwa-mem2 --chunk-cap 0
compat_ok --compat=bwa-mem2 --seed-order off
compat_ok --compat=bwa-mem2 --rescue-kmer=0
compat_ok --compat=bwa-mem2 --max-extend-chains=0
compat_ok --compat=bwa-mem2 --supp-rep-hard-cap=0
compat_ok --compat=bwa-mem2 --min-ext-len 0
# A divergent lever without --compat is of course fine.
compat_ok --smem-dedup
compat_ok --compat=off --smem-dedup

# --- --compat-allow-divergent: opt-in escape hatch. ---------------------------
# It downgrades the divergent-lever refusal to a warning and proceeds, so the
# user keeps the target's OUTPUT CONVENTIONS with a bwa-mem3 lever engaged. Must
# still WARN (output is no longer byte-identical), and must NOT relax the two
# category errors (--fast, --meth), which stay hard-rejected even with the hatch.
compat_warn "NOT byte-identical" --compat=bwa-mem2 --smem-dedup --compat-allow-divergent
compat_warn "NOT byte-identical" --compat=bwa-mem2 --max-extend-chains=5 --compat-allow-divergent
compat_warn "NOT byte-identical" --compat=bwa-mem2 --proper-pair-from-emitted --compat-allow-divergent
# The warning, like the error, must NAME the forced lever (same runtime-built list).
compat_warn "smem-dedup" --compat=bwa-mem2 --smem-dedup --compat-allow-divergent
# Hatch does not rescue the category errors.
compat_err "mutually exclusive" --compat=bwa-mem2 --fast --compat-allow-divergent
compat_err "no methylation mode" --compat=bwa-mem2 --meth --compat-allow-divergent
# Hatch is a harmless no-op without --compat, and with --compat but no lever.
compat_ok --compat-allow-divergent --smem-dedup
compat_ok --compat=bwa-mem2 --compat-allow-divergent
echo "PASS: --compat enum grammar (=/space, alias, off, both targets, unknown, exactness)"

# --- Source guard: both proper-pair sites share one derivation. --------------
# The behavioural difference is reachable ONLY with a `.alt` sidecar, which no
# fixture here has, so a consumer that ignored `proper_pair_from_emitted` and
# hardcoded #17's `which[i]` would pass every other test in this suite and would
# only surface as 3,013-per-10M FLAG diffs on a real ALT-aware run
# (fg-labs/bwa-mem3#362). mem_sam_pe's scalar and batched no-pairing blocks are
# verbatim copies of each other, so a fix applied to one and not the other is the
# specific mistake to catch. Both now call mem_proper_pair_extra_flag, whose
# behavior proper_pair_alt.sh checks end to end and whose region selection
# test/unit/test_proper_pair_source.cpp pins against an independent oracle -- so
# what is left to assert here is structural: exactly one definition, and no block
# that quietly grew a private copy of the decision.
PAIR_SRC="$(dirname "$0")/../../src/bwamem_pair.cpp"
[[ -f "$PAIR_SRC" ]] || {
    echo "FAIL: cannot find src/bwamem_pair.cpp at $PAIR_SRC" >&2
    exit 1
}
n_calls=$(grep -c 'extra_flag |= mem_proper_pair_extra_flag(' "$PAIR_SRC" || true)
n_guard=$(grep -c 'opt->proper_pair_from_emitted ? which\[' "$PAIR_SRC" || true)
if [[ "$n_calls" -ne 2 ]]; then
    echo "FAIL: expected 2 mem_proper_pair_extra_flag call sites (scalar + batched)," >&2
    echo "      found $n_calls in bwamem_pair.cpp -- a no-pairing block either lost" >&2
    echo "      the call or re-inlined the derivation" >&2
    exit 1
fi
# Exactly one w0/w1 selection, i.e. inside the helper and nowhere else. A block
# that re-inlined the ternary would push this to 4 and fail here even if it also
# kept its call above.
if [[ "$n_guard" -ne 2 ]]; then
    echo "FAIL: expected exactly 2 'opt->proper_pair_from_emitted ? which[' lines -- the" >&2
    echo "      w0 and w1 selections inside mem_proper_pair_extra_flag, and nowhere else" >&2
    echo "      -- but found $n_guard in bwamem_pair.cpp" >&2
    exit 1
fi
echo "PASS: both proper-pair sites derive FLAG 0x2 through one guarded helper"

# --- The two targets must actually differ, not merely both be accepted. ---
# bwa-mem keeps MQ:i and the default @HD; bwa-mem2 suppresses both. If a future
# refactor collapsed the rows, every assertion above would still pass while the
# flag silently stopped meaning anything.
mem_sam="$COMPAT_WORK_DIR/target-bwa-mem.sam"
"$BWA_MEM3" mem --compat=bwa-mem "$ref" "$r1" "$r2" > "$mem_sam" 2> /dev/null
if ! grep -q '^@HD' "$mem_sam"; then
    echo "FAIL: --compat=bwa-mem emitted no @HD (bwa 0.7.19 emits one; bwa.c:426)" >&2
    exit 1
fi
if [[ "$(grep -c '^@HD' "$mem_sam")" -ne 1 ]]; then
    echo "FAIL: --compat=bwa-mem emitted more than one @HD" >&2
    exit 1
fi
if [[ "$(grep '^@HD' "$mem_sam")" != "$(printf '@HD\tVN:1.5\tSO:unsorted\tGO:query')" ]]; then
    echo "FAIL: --compat=bwa-mem @HD is not byte-identical to bwa 0.7.19's (bwa.c:426):" >&2
    grep '^@HD' "$mem_sam" >&2
    exit 1
fi
# The tag assertions below are single greps, not `grep -v '^@' | grep -q ...`
# pipelines: under `set -o pipefail` a downstream `grep -q` closes the pipe on
# its first match, so the upstream `grep -v` can die of SIGPIPE and fail the
# whole pipeline on output that is in fact correct. `^[^@]` selects record lines
# without a pipe, and anchoring the tag to a leading tab additionally pins it to
# a field boundary rather than matching the same bytes inside SEQ or QUAL.
if ! grep -q '^[^@]' "$mem_sam"; then
    echo "FAIL: --compat=bwa-mem emitted a header but no alignment records" >&2
    exit 1
fi
if ! grep -q $'^[^@].*\tMQ:i:' "$mem_sam"; then
    echo "FAIL: --compat=bwa-mem suppressed MQ:i (bwa emits it; bwamem.c:935)" >&2
    exit 1
fi
if grep -q $'^[^@].*\tHN:i:' "$mem_sam"; then
    echo "FAIL: --compat=bwa-mem emitted HN:i, which bwa has no counterpart for" >&2
    exit 1
fi
# ...and the bwa-mem2 stream, already written above, must be its opposite.
if grep -q '^@HD' "$cmp_sam" || grep -q $'^[^@].*\tMQ:i:' "$cmp_sam"; then
    echo "FAIL: --compat=bwa-mem2 emitted @HD or MQ:i; the two targets have collapsed" >&2
    exit 1
fi
echo "PASS: bwa-mem keeps @HD + MQ:i and drops HN:i; bwa-mem2 drops all three"

# --- --compat with an @HD in -H warns, and does NOT reject. --------------
# bwa-mem3 hoists a LEADING user @HD above @SQ so the header is spec-valid;
# bwa emits -H records after @SQ and bwa-mem2 has no @HD logic, so the header
# differs from the target in line ORDER. That is an explicit, coherent request
# ("valid header, everything else the same") -- unlike --fast/--meth, which the
# user cannot see in their own command line -- so it warns and continues.
hd_h=$(printf '@HD\tVN:1.6\tSO:coordinate')
rg_h=$(printf '@RG\tID:x\tSM:y')
if ! "$BWA_MEM3" mem --compat=bwa-mem2 -H "$hd_h" "$ref" "$r1" "$r2" \
    > "$COMPAT_WORK_DIR/hdwarn.sam" 2> "$COMPAT_WORK_DIR/hdwarn.log"; then
    echo "FAIL: --compat with -H @HD exited nonzero; it must warn, not reject" >&2
    cat "$COMPAT_WORK_DIR/hdwarn.log" >&2
    exit 1
fi
grep -q 'will differ from bwa-mem2 in line order' "$COMPAT_WORK_DIR/hdwarn.log" \
    || {
        echo "FAIL: --compat with -H @HD did not warn about line order:" >&2
        cat "$COMPAT_WORK_DIR/hdwarn.log" >&2
        exit 1
    }
# The user's @HD must win and be hoisted above @SQ (that is the whole point).
head -1 "$COMPAT_WORK_DIR/hdwarn.sam" | grep -q '^@HD' \
    || {
        echo "FAIL: user @HD from -H is not the first header line" >&2
        exit 1
    }
echo "PASS: --compat with -H @HD warns and continues, user @HD wins"

# A LATER @HD is emitted inline after @SQ, exactly as upstream does, so there
# is nothing to warn about -- the warning must stay quiet or it means nothing.
"$BWA_MEM3" mem --compat=bwa-mem2 -H "$rg_h" -H "$hd_h" "$ref" "$r1" "$r2" \
    > /dev/null 2> "$COMPAT_WORK_DIR/hdquiet.log"
if grep -q 'in line order' "$COMPAT_WORK_DIR/hdquiet.log"; then
    echo "FAIL: warned about a non-leading @HD, which does not diverge" >&2
    cat "$COMPAT_WORK_DIR/hdquiet.log" >&2
    exit 1
fi
# ...and no warning at all without a compat target.
"$BWA_MEM3" mem -H "$hd_h" "$ref" "$r1" "$r2" > /dev/null 2> "$COMPAT_WORK_DIR/hdoff.log"
if grep -q 'in line order' "$COMPAT_WORK_DIR/hdoff.log"; then
    echo "FAIL: warned without a compat target selected" >&2
    exit 1
fi
echo "PASS: the -H @HD warning fires only when the header actually diverges"

# --- Exactly one @HD, always, on both output paths. -----------------------
# A non-leading @HD in -H used to leave the SAM path emitting the DEFAULT @HD
# too, for two @HD records in one header -- a spec violation, and one bwa does
# not have (bwa.c:412-426 counts @HD at any line start). The BAM writer was
# already correct, so this also re-aligns the two paths.
one_hd() { # <description> [extra bwa-mem3 args...]
    local desc="$1"
    shift
    local n m
    n=$("$BWA_MEM3" mem "$@" "$ref" "$r1" "$r2" 2> /dev/null | grep -c '^@HD' || true)
    [ "$n" -eq 1 ] \
        || {
            echo "FAIL: SAM emitted $n @HD lines ($desc), expected exactly 1" >&2
            exit 1
        }
    if command -v samtools > /dev/null 2>&1; then
        m=$("$BWA_MEM3" mem --bam "$@" "$ref" "$r1" "$r2" 2> /dev/null \
            | samtools view -H --no-PG - 2> /dev/null | grep -c '^@HD' || true)
        [ "$m" -eq 1 ] \
            || {
                echo "FAIL: BAM emitted $m @HD lines ($desc), expected exactly 1" >&2
                exit 1
            }
    fi
}
one_hd "no -H"
one_hd "leading @HD" -H "$hd_h"
one_hd "later @HD" -H "$rg_h" -H "$hd_h"
echo "PASS: exactly one @HD on both paths, leading or later -H @HD"

echo "PASS: compat byte-identical regression"
