# fgumi

fgumi (Fulcrum Genomics Unique Molecular Indexing tools) is a high-performance suite
of command-line tools for processing UMI-tagged next-generation sequencing data. Written
in Rust, it provides UMI extraction from FASTQ files, read grouping by UMI with
configurable assignment strategies, UMI-aware deduplication, simplex and duplex consensus
calling, CODEC consensus calling, quality filtering of consensus reads, and overlapping
read-pair clipping. fgumi is the intended successor to the Scala-based
[fgbio](https://github.com/fulcrumgenomics/fgbio) toolkit for UMI processing, targeting
significantly higher throughput on multi-core systems. It is published on Bioconda and
documented at <https://fgumi.readthedocs.io>.

> **Warning — Research preview**
>
> fgumi is currently a research preview. The Fulcrum Genomics team targets June 2026 for
> recommending fgumi over fgbio for production use. Verify fitness for your application
> before deploying in a clinical or production pipeline.

## When you'd use it

Use fgumi when your sequencing library includes unique molecular identifiers and you
need to group reads by UMI, call simplex or duplex consensus sequences, or remove PCR
duplicates in a UMI-aware manner. It handles the standard commercial UMI library
preparations (IDT xGen, KAPA, Twist, QIAseq, and others) and the CODEC protocol for
duplex sequencing. fgumi is designed to be run after alignment with bwa-mem3 (or
bwa-mem2) and before downstream variant calling or methylation analysis.

## How it relates to bwa-mem3

fgumi and bwa-mem3 are sibling projects maintained by Fulcrum Genomics and are designed
to work together in the same alignment-and-consensus pipeline. bwa-mem3 provides the
aligned BAM that fgumi takes as input for grouping and consensus calling. The two projects
share build and documentation conventions (mdbook on Read the Docs, Fulcrum theme,
conventional commits) and are benchmarked together on a shared internal dataset
suite. The intended integration path for in-process alignment within fgumi is
[bwa-mem3-rs](bwa-mem3-rs.md), the Rust bindings for bwa-mem3.

## Links

- GitHub: <https://github.com/fulcrumgenomics/fgumi>
- Docs: <https://fgumi.readthedocs.io>
- License: MIT

---

**See also:**
[bwa-mem3-rs](bwa-mem3-rs.md) ·
[Aligning short reads (mem)](../user-guide/aligning.md) ·
[Best Practices — Multi-sample workflows](../best-practices/multi-sample.md) ·
[bwa-mem3-bench](bwa-mem3-bench.md)
