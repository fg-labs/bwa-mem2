# Output: SAM/BAM, headers, tags

bwa-mem3 writes output in either SAM (default) or BAM (`--bam`) format.
This page covers the header structure and every non-standard SAM tag emitted
by bwa-mem3.

## Output format

By default, `bwa-mem3 mem` writes SAM to stdout. Pass `--bam` (or `--bam=N`
for a specific compression level) to write BAM. Level 0 (uncompressed) is the
default when `--bam` is given without an argument, which is optimal when piping
to a downstream `samtools sort`.

For compressed BAM (`--bam=1..9`), the BGZF deflate is parallelized and
auto-scaled from `-t` (`--bam-threads`, default integer `-t/8`), so compression is
not a throughput bottleneck out of the box at `-t >= 8`. Below `-t 8` the integer
`-t/8` is 0, so the pool is empty and compression stays serial. The compressed record stream is
byte-identical to serial: records decode identically and the compressed record
bytes match, after excluding the `@PG CL:` line (it records the `--bam-threads`
value, so it differs between a serial and an explicit-thread invocation). Override
with `--bam-threads N` (and `--bam-threads 0` to force the old single-threaded
deflate).

```bash
# SAM (default)
bwa-mem3 mem -t 16 ref.fa R1.fq.gz R2.fq.gz > out.sam

# Uncompressed BAM — best for piping
bwa-mem3 mem --bam -t 16 ref.fa R1.fq.gz R2.fq.gz | samtools sort -@ 8 -o out.bam -

# Compressed BAM — useful when the output is the final file
bwa-mem3 mem --bam=6 -t 16 ref.fa R1.fq.gz R2.fq.gz > out.bam
```

## SAM header

### `@HD`

A default `@HD VN:1.5 SO:unsorted GO:query` line is emitted unless the user
supplies one via `-H` (or the index sidecar does). The sort order is `unsorted`
because bwa-mem3 writes records in input read order; downstream sorting is
always a separate step, and `GO:query` records that the records for one query
are adjacent.

The same line is emitted on every output path — SAM text, `--bam` and
`--meth`. Through 0.7.1 the BAM writers emitted `@HD VN:1.6 SO:unsorted`
instead, so the same run produced a different `@HD` depending on `--bam`; they
now agree, on the string upstream bwa uses.

### `@SQ`

One `@SQ` line is written per reference sequence, with the sequence name
(`SN:`) and length (`LN:`) derived from the FM-index. Contigs listed in
`<prefix>.alt` additionally get `AH:*`, marking them as alternate loci.

If a `<prefix>.hdr` or `<baseprefix>.dict` file sits next to the index and
supplies `@SQ` records, those records are used **verbatim** instead of the
auto-generated ones. This is deliberate: the sidecar is authoritative, so
bwa-mem3 neither adds nor removes tags in it.

> **Warning — a Picard/GATK `.dict` drops `AH:*`**
>
> `picard CreateSequenceDictionary` has no notion of a `.alt` file, so the
> `.dict` it writes carries no `AH` tag. Because the sidecar is used verbatim,
> **ALT contigs lose their `AH:*` in the output header** — which matters to
> anything that keys on it, notably `bwa-postalt.js` and ALT-aware duplicate
> marking and QC.
>
> This bites the standard Broad reference layout, because the discovery order
> is `<prefix>.hdr` then `<baseprefix>.dict` — so
> `Homo_sapiens_assembly38.fasta` resolves to `Homo_sapiens_assembly38.dict`,
> exactly the file the GATK resource bundle ships.
>
> Both `index` and `mem` warn when the index has ALT contigs and the sidecar's
> `@SQ` has no `AH`. Two paths do not, because on them there is nothing to lose:
> `mem --meth` (the doubled reference's contigs are `f`/`r`-prefixed and have no
> ALT status of their own — `index --meth` still checks the real reference), and
> `--compat=bwa-mem2`, which ignores the sidecar entirely and generates `@SQ`
> with `AH:*` from the index.
>
> The fix is to regenerate the sidecar with the ALT list — `samtools dict` reads
> a bwa-style `.alt` and emits `AH:*` for you:
>
> ```bash
> samtools dict --alt ref.fasta.alt -o ref.dict ref.fasta
> ```
>
> Alternatively, delete the sidecar and let bwa-mem3 generate `@SQ` itself,
> which always emits `AH:*` — at the cost of the `M5`/`UR`/`AS`/`SP` metadata
> the `.dict` carries.

In methylation mode (`--meth`), the `f`/`r`-prefixed contigs of the doubled
`.meth` seed reference are used for seeding only and never reach the output.
`@SQ` is written directly from the original (un-converted) reference, so the
lines carry the same reference names and lengths a standard non-methylation
alignment does. They are not guaranteed to be byte-identical to it: the
standard path can emit the index's `.hdr`/`.dict` sidecar `@SQ` records
verbatim, whereas methylation mode builds each line from `bns->anns` and then
merges the sidecar's identity tags onto it. See
[Chimera QC and header construction](../methylation/post-processing.md).

### `@PG`

One `@PG` entry is written in standard mode:

| ID | Description |
|----|-------------|
| `bwa-mem3` | The alignment step. `VN:` is the bwa-mem3 version string; `CL:` is the full command line. |

In methylation mode (`--meth`), a second `@PG` entry is appended:

| ID | Description |
|----|-------------|
| `bwa-mem3-meth` | The inline post-processor. `VN:` carries the version with `-meth` suffix; `CL:` is the full command line. |

The `bwa-mem3-meth` entry follows immediately after the `bwa-mem3` entry and
records the post-processing step as a distinct pipeline node, matching the
convention of separate-tool pipelines.

## Tags emitted by bwa-mem3

### Standard tags

bwa-mem3 emits the same standard tags as bwa-mem2 (`NM:i`, `MD:Z`, `AS:i`,
`XS:i`, `SA:Z`, `RG:Z`, `XA:Z`, `MC:Z`, etc.). These are documented in the
SAM specification and are not described further here.

bwa-mem3 additionally emits `MQ:i` on paired-end records — the mate's
mapping quality, set alongside `MC:Z` (the mate's CIGAR) so callers that
key off the mate's MAPQ don't need to look at the mate record. Both SAM
and `--bam` output paths emit it. Backported from `lh3/bwa` PR #330 in
fg-labs PR #35.

The `XA:Z` field set widens from `chr,pos,CIGAR,NM` to
`chr,pos,CIGAR,NM,score,mapq` when `-u` (a.k.a. the upstream "XB" toggle)
is passed; the tag name itself remains `XA:Z` for downstream
compatibility. Tools that parse `XA:Z` need to be aware of the two
possible field widths.

### `HN:i` — total alignment hit count

```text
HN:i:<count>
```

The total number of primary alignments (both reported and suppressed) that
the aligner found for this read, before the `-h` supplementary cap is applied.
Useful for distinguishing "uniquely mapped" from "multi-mapped" reads without
relying solely on MAPQ.

`HN:i` is emitted on the primary alignment record only.

### `pa:f` — ratio to the better ALT hit

```text
pa:f:<ratio>
```

The record's alignment score divided by the score of the better overlapping
hit on an ALT contig. Emitted, as in bwa and bwa-mem2, only on non-secondary
records of a non-ALT hit that an ALT hit beats — so it never appears unless
the index has an ALT sidecar (`<prefix>.alt`) marking contigs as alternate.

The value is rounded to three decimals, matching both upstreams. `--bam` is
defined to store the float32 that rounded decimal parses to — the same value
`samtools view -b` of the SAM text would store — so the container is a format
choice for this tag rather than a content change.

The three writers disagreed on all of this before
[#366](https://github.com/fg-labs/bwa-mem3/pull/366), so a pre-0.8.1 `--bam` or
`--meth --bam` file carries a different `pa:f` value, on a different set of
records, in a different tag position. See
[Equivalence → `pa:f:` renders identically in SAM text and
BAM](../whats-different/equivalence.md#paf-renders-identically-in-sam-text-and-bam-resolved-2026-08-05)
for the divergence record and its verification scope.

### Methylation-only tags

The following Bismark-compatible tags are emitted only when `--meth` is
active. See [SAM tags: XR, XG, XM](../methylation/tags.md) for the full
per-tag reference, including the `XM:Z` character alphabet and the
`XG:Z` strand-pick semantics.

| Tag | Type | Description |
|-----|------|-------------|
| `XR:Z` | string | Read conversion direction: `CT` (R1 / SE) or `GA` (R2) |
| `XG:Z` | string | Genome strand of the alignment: `CT` (OT) or `GA` (OB) |
| `XM:Z` | string | Per-base methylation call string (length = `SEQ`) |

The bwameth-style `YS:Z` / `YC:Z` tags exist only as an internal carrier
on `bseq1_t.comment` for SEQ restoration and `XR:Z` derivation; they
are suppressed at BAM emit and never appear in output. The bwameth
`YD:Z` strand tag has been replaced by Bismark `XG:Z` and is not
emitted.

## MAPQ semantics

MAPQ semantics are inherited from bwa-mem2 and follow the same scoring model.
In methylation mode, alignments identified as chimeras (longest `M`/`=`/`X`
run covering less than 44% of the read length) have their MAPQ capped at 1 and
the `0x200` (QC fail) flag set. See
[Chimera QC and header construction](../methylation/post-processing.md).

---

**See also:**
[Aligning short reads (mem)](aligning.md) ·
[Methylation Reference: SAM tags](../methylation/tags.md) ·
[Methylation Reference: post-processing](../methylation/post-processing.md) ·
[CLI Reference: mem](../cli/mem.md) ·
[Best Practices: output format](../best-practices/output-format.md)
