#!/usr/bin/env bash
# test/regression/meth_rescue_batched_identical.sh
#
# Regression (issue 173 / Task 5): --meth mate rescue routed through the BATCHED
# kswv kernel (OT/OB driver-partition) must produce a BAM byte-identical to the
# legacy SCALAR ksw_align2 rescue path. Both legs are the SAME binary; only the
# BWAMEM3_METH_BATCHED_RESCUE escape hatch differs:
#
#   BWAMEM3_METH_BATCHED_RESCUE=0  -> legacy scalar meth rescue (reference)
#   BWAMEM3_METH_BATCHED_RESCUE=1  -> batched per-hypothesis meth rescue (candidate)
#
# The fixture is built programmatically (no committed test data): a constructed
# bisulfite reference, a bank of plain concordant FR pairs to seed the insert-size
# distribution (single-pair batches leave every pes[r].failed, which suppresses
# rescue entirely), plus one PE fixture pair where the anchor maps cleanly and the
# mate is rendered unseedable by spaced non-bisulfite errors so it is brought in
# by SW mate rescue. The decisive assertion is a full `samtools view` diff: the
# two rescue paths must agree byte-for-byte.
#
# SCOPE NOTE: this whole-aligner test is a no-regression / byte-identity guard for
# the routing — it proves the batched OT/OB-partition path emits the same BAM as
# the scalar path. It is NOT the kernel-correctness oracle: the rescue candidate's
# kswv score is re-derived downstream (mem_sort_dedup_patch + the asymmetric
# mem_reg2aln re-scoring), so the SIMD vs scalar score difference does not surface
# in the final BAM. The authoritative byte-identity gate for the mat-aware
# OT/OB rescue kernel is a separate, standalone SW-kernel byte-identity check
# that compares the batched kswv kernel against scalar ksw_align2 with the same
# asymmetric matrix, field-by-field.
#
# Inputs:
#   BWA_MEM3 — path to the bwa-mem3 binary under test
set -euo pipefail
: "${BWA_MEM3:?BWA_MEM3 must be set}"
command -v samtools > /dev/null 2>&1 || {
    echo "SKIP: samtools not on PATH (--meth emits BAM)"
    exit 0
}

# Resolve BWA_MEM3 to an absolute path before we cd into the temp workdir.
case "$BWA_MEM3" in
    /*) BIN="$BWA_MEM3" ;;
    *) BIN="$PWD/$BWA_MEM3" ;;
esac
[ -x "$BIN" ] || {
    echo "FAIL: BWA_MEM3 ($BIN) is not executable" >&2
    exit 1
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# Deterministic 1500 bp reference (same PRNG seed 4242 as the sibling meth tests).
REF=TCATTGGCTATCCTAACCCGACCCTAGGAGCGGTTGGCGTGTATGCCGTGAATTTTCTCATTTCCGCTAGACATAATCGTTCTGCCTATATCTGGACAACATCCCGGCGACTTAGGCGACCCACAGAATCGTCCCTTCTAACGTAGTTCGCATAGTTCCCGTCCGTAGCCGGACTATTCGAACACCCAGTATTCGATTAACTCGGGCTTGACGTATTAGAGGCGTTAGTGTGCCAGGTAAGATACGCCAACGGAATTAACCTCTGTGACACTCCGCGGAGCCTTCGGACATATAAGTGATCGGGTCTACGTTTGTTAGACTTGAGACGTCTGTTAAGAGTTGGGTCTAATAAATCGCCTACACGTGGAGTCTAACGGGGAAGCGTCGAATCCTGATACATCATATAATGGAGCGTGTTATGAAAAAAGAGCATTCCATTGTACGAGCCGTGCCAGAAACGGCTTGACTACGTGAGCGTAGTGTTAGATAAACAGGAAACACTGACGCGGTTAGAAGGCGGATTGCCGGTAGGTTTTGGAAACATAAATACACACGGTATCATGTTGGGTCACGATTCCTATCACCGCACAGGGCCAACCATAGAAGAACTGAAAGAACTAATCTGGCGGCGGGCTCGGTGCTTATATTTTCCACCCAACATCGTGCACATTAGGCTCACCGCGCCCTACGGGCGAAGGGTGCGTACGGTGTTTATAAGGCGTGACGGCCCCAAGTAGAGGGTAATTCTGTGAAAGAATCTCAGGACGGTGGCATGAATTCAATTCCTTTTAAACCTATCGTTCCGACCTTATGCAATCCTTCAATGAAGATCGTCAACGACCATCGTTCTTCTGCTTTAAGTGTGAGTTCTCTCTTACAAGCTAATACACCCCAGCGTTCTCCGTACTCTTCACTGCCCAAGCGAGGCTAACCTTTTGAAATGTCACAGTCGAAGCATATCTCCCGTACATCTTTTTCGGAGATCGCAGCTCGCGGAGCTATAAGCGACTTAAGCCCTTGTGTCGGTGATCCCAAGGGTCTGACTCCTGTACCAGGGTTACTGTTTCGCTTTACGGAGTAGCCTGTGAGGTGAACTGAAAGGAGCATATTTGAGATCTAAGATAGGGTCCTCCTCTGCGTCTACGTTCTCTCCGTTACGTACGGCTTCGCACCGGAGTGCATCTTGGCCCCGAAACGCACTGTGTGTGCTGATACAGCGTCCCTGGCCGGCCATGGGTTCAGAACTCCCGGGAACGCTTTTCAACTTAGAGGAACCCCGTCATGGAAGTAGATCGCGTCGAATGAGGGAGTTAGTCCTCGTTCCAGCTGGTAATTGTTTTACCGCTTGGGACCACTATAGGCCGCGGGTAGAAGTTGCTGGGTGTTGATTCCAACCCTCGAACCACGATACGACCTGCCATTTATGGCACAGTAAGGTTCAAACAGCATAATGAATACAGTTATAGTAACTTCCTCACGTACGATTAGGACGCAGCCTTG

printf '>chrA\n%s\n' "$REF" > ref.fa
"$BIN" index --meth ref.fa > /dev/null 2>&1 || fail "index --meth nonzero exit"

python3 - "$REF" > fixture.env << 'PY'
import sys, random
ref = sys.argv[1]
L = len(ref)
def rc(s):
    c = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    return ''.join(c[b] for b in reversed(s))
def err(b):                          # spaced NON-bisulfite mismatch (breaks seeds)
    return {'A':'G','G':'A','T':'A','C':'C'}.get(b, 'A')

q = 'I'*60
INS = 250                            # fixed insert so pes estimates tightly
r1, r2 = [], []

# Bank of plain (unconverted) concordant FR pairs at distinct loci. These exist
# only to populate the insert-size distribution: bwa refuses mate rescue until
# enough unique pairs let it estimate pes (a single pair leaves all pes failed).
rnd = random.Random(7)
for k in range(80):
    p  = rnd.randint(40, L - 360)
    a  = ref[p:p+60]
    mp = p + INS - 60
    mf = ref[mp:mp+60]
    r1.append('@c%d\n%s\n+\n%s\n' % (k, a, q))
    r2.append('@c%d\n%s\n+\n%s\n' % (k, rc(mf), q))

# The rescue pair. Anchor (R1) is a clean forward OT read that maps strongly.
a_pos = 101
a = list(ref[a_pos-1:a_pos-1+60])
for i, b in enumerate(a):
    if b == 'C': a[i] = 'T'          # OT C->T conversions on the anchor

# Mate (R2) sits at the proper-pair distance but is made UNSEEDABLE by two spaced
# non-bisulfite errors (bisulfite conversions alone never break seeding — they are
# collapsed by the meth index). It therefore only places via SW mate rescue, which
# is exactly the path this test routes through the batched OT/OB kernel.
m_pos = a_pos + INS - 60
m = list(ref[m_pos-1:m_pos-1+60])
for i, b in enumerate(m):
    if b == 'C': m[i] = 'T'          # dense OT conversions (free only under the OT/OB matrix)
for pos in (7, 31):
    m[pos] = err(m[pos])             # seed-breaking real mismatches
r1.append('@resc\n%s\n+\n%s\n' % (''.join(a), q))
r2.append('@resc\n%s\n+\n%s\n' % (rc(''.join(m)), q))

with open('r1.fq','w') as f: f.write(''.join(r1))
with open('r2.fq','w') as f: f.write(''.join(r2))
print('OK')
PY
[ -s fixture.env ] || fail "fixture generation failed (python3 required)"

run_leg() { # $1 = env value (0|1) -> writes $2.sam
    BWAMEM3_METH_BATCHED_RESCUE="$1" "$BIN" mem --meth -t 1 ref.fa r1.fq r2.fq 2> /dev/null \
        | samtools view - > "$2.sam" || fail "leg env=$1 nonzero exit"
    [ -s "$2.sam" ] || fail "leg env=$1 produced empty SAM"
}

run_leg 0 scalar
run_leg 1 batched

if ! diff -u scalar.sam batched.sam > diff.txt; then
    echo "FAIL: batched meth rescue (env=1) differs from scalar meth rescue (env=0):" >&2
    cat diff.txt >&2
    exit 1
fi

echo "PASS: meth_rescue_batched_identical (batched OT/OB-partition rescue == scalar rescue, byte-identical BAM)"
