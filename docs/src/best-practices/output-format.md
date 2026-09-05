# Output Format

The choice of output format — SAM, compressed BAM, or uncompressed BAM — has a
measurable effect on end-to-end pipeline wall time. This page explains why
uncompressed BAM is the right default and shows the recommended pipeline.

## Why uncompressed BAM is faster than SAM

When bwa-mem3 writes SAM (the default when `--bam` is not set), every
alignment record must be serialized into ASCII text: integers are formatted as
decimal strings, bases are encoded as characters, and flags are written as
decimal numbers. The receiving process — typically `samtools sort` — then parses
each field back from text into binary integers. Both conversions are pure
overhead: the data is binary inside bwa-mem3 and binary inside samtools; text
is only an interchange format that is immediately discarded.

Uncompressed BAM (`--bam=0`) bypasses this round-trip. bwa-mem3 writes binary
BAM records directly via htslib's `wb0` mode. The write path performs no text
formatting; the read path in `samtools sort` performs no text parsing. The
htslib overhead of the `wb0` write is negligible — it is effectively a
buffered `write(2)` call with a small BAM block header prepended.

Compressed BAM (`--bam=1`) adds BGZF compression on top, which gains nothing
when piping: the pipe is in-process memory or a kernel pipe buffer, and
`samtools sort` will re-compress the output anyway, so compressing into a sort
pipe wastes CPU on both sides — parallelizing the write-side deflate (see
`--bam-threads` below) does not change that, it only makes the wasted work
faster. When writing compressed BAM *directly to final storage* (no downstream
sort), `--bam-threads` (default `-t/8`) keeps the deflate off the throughput
critical path.

## Recommended pipeline

```bash
bwa-mem3 mem --bam=0 -t 16 ref.fa R1.fq.gz R2.fq.gz \
  | samtools sort -@ 8 -o out.bam -
samtools index out.bam
```

The `-@ 8` flag gives `samtools sort` eight compression threads for writing the
final sorted BAM. Tune this number based on available cores; the total core
count should be split so that alignment threads and sort threads do not
contend. A 16:8 split (bwa-mem3:samtools) works well on 24-core machines.

> **Tip — Thread allocation**
>
> Do not give all cores to bwa-mem3.  Downstream `samtools sort` needs threads
> to compress and write the sorted BAM.  Leaving 4–8 threads for `samtools sort`
> keeps the pipeline balanced and prevents a write bottleneck that would stall
> the aligner.

## Methylation output

`--meth` selects bisulfite alignment semantics; it does not select an output
format.  The container is chosen by `--bam` exactly as it is without `--meth`:
SAM text by default, BAM with `--bam`.  The post-processing step (`@SQ` built
from the original reference, Bismark `XR:Z` / `XG:Z` / `XM:Z` tag emission,
opt-in chimera QC) is
performed inline before the record is handed to htslib, so the same pipeline
shape applies:

```bash
bwa-mem3 mem --meth --bam=0 -t 16 ref.fa R1.fq.gz R2.fq.gz \
  | samtools sort -@ 8 -o out.bam -
samtools index out.bam
```

**What the two containers share.** `--meth` and `--meth --bam` run one
`bam1_t` construction path and hand the finished record to htslib, differing
only in the `htsFile` mode string, so the equivalence is structural rather than
sampled: it does not vary with workload, thread count, or SIMD tier. It is over
decoded **records**, not over headers — the `@PG` `CL:` field records the
invoking command line and therefore differs, since one run was given `--bam`
and the other was not. A regression script pins both halves on the committed
phiX fixture in single-end and paired-end layouts (which take different
emission paths): records must compare byte-identical, headers byte-identical
once `CL:` is stripped.

> **Changed in 0.8.0**
>
> Through 0.7.x, `--meth` forced BAM and text SAM was unreachable.  If you have
> a script that relies on `bwa-mem3 mem --meth` producing BAM on stdout, add
> `--bam` to it.  Piping into `samtools` needs no change: samtools autodetects
> SAM text.

## When SAM is appropriate

SAM (the default, equivalent to omitting `--bam`) remains the right choice for:

- **Debugging.** Plain text is readable with `less`, `grep`, and any text
  editor, making it easy to inspect individual records without `samtools view`.
- **Ad-hoc inspection.** When you need to scan a few thousand reads to diagnose
  a mapping problem, piping to SAM and reading the output directly is faster
  than writing a BAM file and then querying it.
- **Compatibility with tools that require SAM input.** Some legacy tools do not
  accept BAM. If the downstream tool does not support BAM, use SAM.

For production alignment jobs that feed `samtools sort`, always use
`--bam=0`.

## Summary table

| Format | `--bam` value | Pipe overhead | Recommended for |
|--------|---------------|---------------|-----------------|
| SAM | (default / omit) | High (text round-trip) | Debugging, ad-hoc inspection |
| Uncompressed BAM | `0` | Negligible | Production pipelines |
| Compressed BAM | `1` | High on write side | Writing directly to a file (no downstream sort) |

---

**See also:**
[Aligning short reads (mem)](../user-guide/aligning.md) ·
[Output: SAM/BAM, headers, tags](../user-guide/output.md) ·
[Threading and resource use](../user-guide/threading.md) ·
[Optimization checklist](optimization-checklist.md) ·
[CLI Reference: mem](../cli/mem.md)
