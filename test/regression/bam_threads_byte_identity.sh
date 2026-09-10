#!/usr/bin/env bash
# test/regression/bam_threads_byte_identity.sh
#
# Regression: the --bam-threads BGZF compression pool is byte-identical to the
# serial writer. htslib's ordered thread pool emits blocks in dispatch order, so
# --bam=6 --bam-threads=N must decode to exactly the same records as
# --bam=6 --bam-threads=0 (the same guarantee `samtools -@` relies on). This is
# the correctness contract that justifies the auto-parallelized default, so it
# is pinned here rather than left to manual checking.
#
# Also checks: the threaded output is deterministic (same compressed bytes across
# runs), and the --bam-threads parser rejects malformed / out-of-range values.
#
# Uses the chr22 holodeck fixture (multi-block BAM — a single BGZF block would
# not exercise the ordered-tpool block-ordering path).
#
# Inputs:
#   BWA_MEM3       — path to bwa-mem3 binary
#   CHR22_FA       — path to chr22.fa (pre-indexed with bwa-mem3 by caller)
#   CHR22_SIM_DIR  — directory containing holodeck reads.r[12].fastq.gz
set -euo pipefail
# Emit a FAIL: line (and preserve the exit status) for any uncaught command
# failure -- e.g. bwa-mem3 mem or samtools quickcheck dying outside an explicit
# check -- so set -e never exits silently. Inherited (set -E) so it also fires
# inside run()/mrun().
set -E
# shellcheck disable=SC2154  # rc is assigned by `rc=$?`, the trap body's first statement
trap 'rc=$?; printf "FAIL: bam_threads_byte_identity.sh failed (exit %d)\n" "$rc" >&2; exit "$rc"' ERR

: "${BWA_MEM3:?BWA_MEM3 must be set}"
: "${CHR22_FA:?CHR22_FA must be set}"
: "${CHR22_SIM_DIR:?CHR22_SIM_DIR must be set}"

# Resolve caller-relative inputs to absolute paths BEFORE the cd below, which
# otherwise re-roots relative BWA_MEM3 / CHR22_FA at $CHR22_SIM_DIR and breaks a
# standalone invocation with relative paths. A bare command name (resolved via
# PATH) and an already-absolute path are left untouched.
resolve_input() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;                                               # absolute
        */*) printf '%s\n' "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;; # relative path
        *) printf '%s\n' "$1" ;;                                                # bare PATH name
    esac
}
BWA_MEM3="$(resolve_input "$BWA_MEM3")"
CHR22_FA="$(resolve_input "$CHR22_FA")"

cd "$CHR22_SIM_DIR"
[ -f reads.r1.fastq.gz ] && [ -f reads.r2.fastq.gz ] || {
    echo "FAIL: expected $CHR22_SIM_DIR/reads.r[12].fastq.gz" >&2
    exit 1
}

run() { # run <out.bam> <bam-threads>
    local err
    # Capture stderr (stdout is the BAM); a positive --bam-threads must actually
    # engage the BGZF pool. hts_set_threads() failure is non-fatal and degrades
    # to serial with a "could not be applied" warning, which would let every
    # comparison below pass without ever testing the thread pool.
    err="$("$BWA_MEM3" mem --bam=6 --bam-threads="$2" "$CHR22_FA" \
        reads.r1.fastq.gz reads.r2.fastq.gz 2>&1 > "$1")"
    if [ "$2" -gt 0 ] && printf '%s' "$err" | grep -q "could not be applied"; then
        echo "FAIL: --bam-threads=$2 fell back to serial compression: $err" >&2
        exit 1
    fi
    samtools quickcheck "$1"
}

run serial.bam 0
run threaded_a.bam 4
run threaded_b.bam 4

# 1. Byte-identity of the decoded records (the contract). `samtools view` with no
#    -h emits records only, so the @PG command-line difference (--bam-threads=0
#    vs =4) does not enter the comparison.
serial_md5=$(samtools view serial.bam | md5sum | cut -d' ' -f1)
threaded_md5=$(samtools view threaded_a.bam | md5sum | cut -d' ' -f1)
serial_n=$(samtools view -c serial.bam)
threaded_n=$(samtools view -c threaded_a.bam)
if [ "$serial_md5" != "$threaded_md5" ] || [ "$serial_n" != "$threaded_n" ]; then
    echo "FAIL: --bam-threads=4 records differ from serial" \
        "(serial $serial_n/$serial_md5 vs threaded $threaded_n/$threaded_md5)" >&2
    exit 1
fi

# 2. Threaded output is deterministic: identical flags -> identical compressed
#    bytes (no block-ordering race in the tpool).
if ! cmp -s threaded_a.bam threaded_b.bam; then
    echo "FAIL: two --bam-threads=4 runs produced different compressed BAM (ordering race)" >&2
    exit 1
fi

# 3. The parser rejects malformed / out-of-range values (must exit non-zero and
#    write nothing usable). MAX_THREADS is 256.
for bad in abc -1 999 4x; do
    if "$BWA_MEM3" mem --bam=6 --bam-threads="$bad" "$CHR22_FA" \
        reads.r1.fastq.gz reads.r2.fastq.gz > /dev/null 2> /dev/null; then
        echo "FAIL: --bam-threads=$bad was accepted, expected a parse error" >&2
        exit 1
    fi
done

# 4. Same contract for the --meth BAM writer (src/meth_bam.cpp) -- a second
#    htsFile emitter alongside src/bam_writer.cpp that gets its own
#    hts_set_threads pool. The chr22 fixture above is not --meth-indexed, so
#    build a small self-contained --meth reference and enough reads to emit a
#    multi-block compressed BAM (a single BGZF block would not exercise the
#    ordered-tpool block-ordering path).
METH_DIR="$(mktemp -d)"
trap 'rm -rf "$METH_DIR"' EXIT
python3 - "$METH_DIR" << 'PY'
import os, random, sys

d = sys.argv[1]
random.seed(7)
ref = "".join(random.choice("ACGT") for _ in range(20000))
with open(os.path.join(d, "ref.fa"), "w") as f:
    f.write(">chrM\n" + ref + "\n")

comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
rc = lambda s: "".join(comp[c] for c in reversed(s))
random.seed(11)
q = "I" * 100
with open(os.path.join(d, "r1.fq"), "w") as a, open(os.path.join(d, "r2.fq"), "w") as b:
    for i in range(4000):
        p = random.randint(0, len(ref) - 300)
        a.write("@r%d\n%s\n+\n%s\n" % (i, ref[p:p + 100], q))
        b.write("@r%d\n%s\n+\n%s\n" % (i, rc(ref[p + 150:p + 250]), q))
PY
"$BWA_MEM3" index --meth "$METH_DIR/ref.fa" > /dev/null 2>&1 \
    || {
        echo "FAIL: index --meth nonzero exit" >&2
        exit 1
    }

mrun() { # mrun <out-basename> <bam-threads>
    local err
    # Same fallback assertion as run(): capture stderr instead of discarding it,
    # and fail if a positive --bam-threads silently degraded to serial.
    err="$("$BWA_MEM3" mem --meth --meth-scoring genomic -t 1 --bam=6 --bam-threads="$2" \
        "$METH_DIR/ref.fa" "$METH_DIR/r1.fq" "$METH_DIR/r2.fq" 2>&1 > "$METH_DIR/$1")"
    if [ "$2" -gt 0 ] && printf '%s' "$err" | grep -q "could not be applied"; then
        echo "FAIL: --meth --bam-threads=$2 fell back to serial compression: $err" >&2
        exit 1
    fi
    samtools quickcheck "$METH_DIR/$1"
}
mrun meth_serial.bam 0
mrun meth_threaded_a.bam 4
mrun meth_threaded_b.bam 4

# 4a. Decoded records identical serial vs threaded (view without -h excludes the
#     @PG --bam-threads difference), same as the standard writer above.
meth_serial_md5=$(samtools view "$METH_DIR/meth_serial.bam" | md5sum | cut -d' ' -f1)
meth_threaded_md5=$(samtools view "$METH_DIR/meth_threaded_a.bam" | md5sum | cut -d' ' -f1)
meth_serial_n=$(samtools view -c "$METH_DIR/meth_serial.bam")
meth_threaded_n=$(samtools view -c "$METH_DIR/meth_threaded_a.bam")
if [ "$meth_serial_md5" != "$meth_threaded_md5" ] || [ "$meth_serial_n" != "$meth_threaded_n" ]; then
    echo "FAIL: --meth --bam-threads=4 records differ from serial" \
        "(serial $meth_serial_n/$meth_serial_md5 vs threaded $meth_threaded_n/$meth_threaded_md5)" >&2
    exit 1
fi

# 4b. Threaded --meth output is deterministic across runs.
if ! cmp -s "$METH_DIR/meth_threaded_a.bam" "$METH_DIR/meth_threaded_b.bam"; then
    echo "FAIL: two --meth --bam-threads=4 runs produced different compressed BAM (ordering race)" >&2
    exit 1
fi

echo "PASS: --bam-threads byte-identical to serial (standard $serial_n records," \
    "--meth $meth_serial_n records), deterministic, and range-validated"
