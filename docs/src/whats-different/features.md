# Features

This page covers user-facing features added to bwa-mem3 on top of upstream
bwa-mem2. None of these features change default behavior: output produced by
`bwa-mem3 mem` without any of these flags is byte-identical to the
corresponding bwa-mem2 output (except for the `@PG` `ID:` and `PN:` fields
which now read `bwa-mem3`).

## `--meth` bisulfite alignment mode (PR #13)

`--meth` adds native bisulfite/EM-seq alignment to `bwa-mem3 index` and
`bwa-mem3 mem` in a single binary — no Python, no separate post-processing step,
no [bwameth.py](https://github.com/brentp/bwa-meth) dependency. In the default
`--meth-scoring collapsed` mode it reproduces bwameth.py's read *placement* (a
placement drop-in, not a byte-for-byte clone); `--meth-scoring genomic` and
`--meth-scoring neutral` (the `--meth=taps` default) opt into variant-aware
scoring bwameth cannot produce.

```bash
bwa-mem3 index --meth ref.fa          # once per reference
bwa-mem3 mem --meth ref.fa R1.fq R2.fq | samtools sort -o out.bam
```

`index --meth` builds a dual index: the normal index over the original reference
plus a converted **seed** index `<ref>.meth.*` (over `<ref>.meth.fa`, a doubled
reference with `f`-prefixed C→T and `r`-prefixed G→A contigs). Reads seed in that
3-letter space but are scored against the original 4-letter reference, so the
index layout is **not** the same as bwameth.py's single `.bwameth.c2t` reference.

`mem --meth` projects each read (R1 C→T, R2 G→A) to find seeds in the `.meth`
index, preserving the original bases on the first-class `bseq1_t.meth_orig_seq`
field (a `YS:Z`/`YC:Z` comment carrier is a fallback only; neither reaches the
BAM). It then extends and **scores against the original 4-letter reference** with
a per-strand asymmetric matrix, writes one `@SQ` per original reference
sequence straight from the original reference
([#174](https://github.com/fg-labs/bwa-mem3/pull/174), which moved seeds onto
original coordinates and removed the `f`/`r` header-rewrite step), emits
Bismark-compatible `XR:Z` (read conversion
direction), `XG:Z` (genome strand), and `XM:Z` (per-base methylation call string)
auxiliary tags, restores the original bases into the BAM SEQ field for CpG-calling
tools, optionally applies a chimera QC heuristic (longest M/=/X run < 44% of read
length → set `0x200`, clear proper-pair `0x2`, cap MAPQ at 1) when `--chimera-qc`
is passed (off by default), and writes a `@PG ID:bwa-mem3-meth` entry.

In the default `collapsed` mode this reproduces bwameth.py's read *placement*
(chrom, pos) for the standard case, while scoring against the original reference
rather than in collapsed space — so it is not byte-for-byte identical to bwameth
output. Stacks on PR #12 (`--bam`). See the
[Methylation Reference](../methylation/overview.md) for full details.

## Vendored mimalloc allocator (PR #19)

bwa-mem3 vendors [mimalloc v3.3.0](https://github.com/microsoft/mimalloc) as a
pinned submodule at `ext/mimalloc` and links it into every binary by default
(`USE_MIMALLOC=1`). On Linux, static linkage uses `--whole-archive`; on macOS,
dyld-interposed shared linkage is used.

Measured on AWS c7g.4xlarge (Graviton3, 16 threads, 29M 150 bp paired-end
exome-capture reads vs hg38, page cache dropped between iterations):
**−24.5% wall-clock time** (528.6 s → 424.7 s) compared to the same build
with `USE_MIMALLOC=0`. No user-visible interface change; no runtime
configuration required.

`USE_MIMALLOC=0` is a supported best-effort opt-out and is CI-gated on Linux
x86. `bwa-mem3 version` prints the mimalloc version string when it is active.

## `--supp-rep-hard-cap` supplementary MAPQ rescoring (PR #56)

Supplementary alignments for a split read inherit MAPQ from the full-read
scoring pipeline. Competing repetitive chains for the supplementary fragment
are filtered out during full-read chain scoring (`mem_chain_flt`) before
Smith-Waterman, so they never contribute to `sub`/`sub_n`. A supp fragment
landing in a CCATCC repeat that would map equally well to 50+ locations
standalone can therefore carry MAPQ=60 from its primary.

`--supp-rep-hard-cap INT` opts into rescoring: if any seed in a supplementary
alignment's chain has `>=INT` genome occurrences (from the SMEM SA count), the
supplementary MAPQ is forced to 0. Primary alignment MAPQ and coordinates are
unaffected. Default output (no flag) is byte-identical to upstream bwa-mem2.

The SMEM SA-occurrence count is preserved on each seed as `mem_seed_t.n_hits`
and propagated to `mem_alnreg_t.chain_n_hits` during chain-to-alignment
conversion. Typical values for `INT` are 5–20; lower is more aggressive. The
upstream [bwa-mem2#260](https://github.com/bwa-mem2/bwa-mem2/issues/260)
reporter case drops from MAPQ=60 to MAPQ=0 at `--supp-rep-hard-cap 18`.
Closes [issue #46](https://github.com/fg-labs/bwa-mem3/issues/46).

## `--proper-pair-from-emitted` proper-pair `FLAG` source (PR #363)

`--proper-pair-from-emitted` derives the proper-pair bit (`FLAG` `0x2`) from the alignment each record actually carries (`a[which]`) instead of the top-scoring region (`a[0]`). Default off, which is bwa's and bwa-mem2's behavior — both derive the bit from `a[0]` unconditionally, even when the record they emit is `a[which]`.

The option is a deliberate deviation from both upstreams, so it is **incompatible with `--compat`**: supplying both is a hard error — overridable with `--compat-allow-divergent` (unlike `--fast`, which stays a hard error because it is a multi-flag bundle rather than a single divergent knob). It is also inert without a `.alt` sidecar — the two regions coincide unless the read has ALT hits — so on a reference without one, the alignment output is byte-identical either way, excluding the `@PG` record, whose `CL:` field records the command line and therefore always reflects the flag.

This is a correctness-argument item rather than a capability, so the reasoning, the measurement, and the [#17](https://github.com/fg-labs/bwa-mem3/pull/17) → [#362](https://github.com/fg-labs/bwa-mem3/issues/362) history live in [Correctness fixes → Proper-pair flag](correctness.md), with the divergence recorded in [Equivalence](equivalence.md).

## Shared-memory index: `bwa-mem3 shm` (PR #65)

`bwa-mem3 mem` reloads the FM-index from disk on every invocation. For hg38
the index is ~18 GB; for short alignment jobs (targeted panels, small sample
batches) this load cost dominates runtime and makes per-invocation IOPS the
bottleneck.

PR #65 ports the `bwa shm` command from bwa-mem v1 to bwa-mem3 with strict v1
CLI parity:

```bash
bwa-mem3 shm <index-prefix>    # load index into shared-memory segment once
bwa-mem3 mem <index-prefix> ...  # subsequent runs attach instead of re-reading
bwa-mem3 shm -d <index-prefix>  # detach and free the segment
```

The index lives in a POSIX shared-memory segment. Multiple `bwa-mem3 mem`
processes on the same host share the same in-memory copy. Closes
[issue #64](https://github.com/fg-labs/bwa-mem3/issues/64).

> **Warning — Stale index**
>
> `bwa-mem3 shm` does not detect when the on-disk index has been rebuilt. Always
> run `bwa-mem3 shm -d <prefix>` before running `bwa-mem3 index` and then
> re-stage with `bwa-mem3 shm <prefix>`. Using a stale shared-memory segment
> produces silently wrong alignments.

## `bwa-mem3 shm --meth` (PR #67)

`bwa-mem3 mem --meth <prefix>` locates the `.meth` seed index built by
`bwa-mem3 index --meth <prefix>` automatically. Before PR #67, staging a
methylation index in shared memory required passing the full suffixed seed-index
path to `shm` while continuing to pass the plain prefix to `mem`. The mismatch was
easy to forget, and the failure mode — a run that silently attached the wrong
segment — was difficult to diagnose.

PR #67 adds `--meth` support to `bwa-mem3 shm` so the same plain-prefix
convention works end-to-end:

```bash
bwa-mem3 shm --meth ref.fa       # stages ref.fa.meth.*
bwa-mem3 mem --meth ref.fa ...   # attaches automatically
bwa-mem3 shm -d --meth ref.fa   # detaches
```

## `HN:i` hit count tag (PR #42)

Every primary SAM/BAM record now carries an `HN:i:<n>` tag reporting the
number of secondary alignment candidates clustered with this primary under
`XA_drop_ratio`. This count is captured before the `-h`/`max_XA_hits` cap
truncates the `XA:Z:` string, so `HN` reports the true number of alternate
loci even when no `XA:Z:` field appears in the record.

This makes it possible to distinguish:

- `HN:i:0` + no `XA:Z:` — genuinely unique mapper.
- `HN:i:N` + `XA:Z:...` (N ≤ `-h`) — multi-mapper with all alternates listed.
- `HN:i:N` + no `XA:Z:` (N > `-h`) — multi-mapper whose alternates were
  suppressed by the cap.

Motivated by [lh3/bwa#438](https://github.com/lh3/bwa/pull/438), which adds
HN to `bwa aln`. HN is emitted in both SAM (`mem_aln2sam`) and BAM
(`mem_aln_to_bam`) paths and is absent when `-a` (`MEM_F_ALL`) is active.

## `--bam=LEVEL` direct BAM output (PR #12)

`bwa-mem3 mem --bam` (or `--bam=0` through `--bam=9`) emits BAM directly via
htslib, bypassing the SAM-text-to-BAM conversion round trip that normally
occurs when the output is piped to `samtools view -bS`.

- `--bam` / `--bam=0`: uncompressed BAM (BGZF framing only) — near-zero CPU
  overhead, smaller than SAM text, fast downstream parsing.
- `--bam=1..9`: BGZF deflate at the specified level.
- No flag: SAM text on stdout (default, unchanged).

The implementation adds `src/bam_writer.{h,cpp}`, a new module that converts
`mem_aln_t` to `bam1_t` via `mem_aln_to_bam`. htslib v1.21 is pulled in as a
submodule at `ext/htslib`. On the bwameth.py example fixture (92,961 records),
`samtools view` of `--bam` output vs SAM text produces a zero-line diff across
all 11 SAM columns and all aux tags. See
[Best Practices → Output format](../best-practices/output-format.md) for the
recommended pipeline.

## `--smem-dedup` SMEM deduplication

`--smem-dedup` opts into removing fully-identical duplicate SMEM seeds before SA
expansion. Off by default → output byte-identical to baseline. When enabled,
duplicate SMEMs (same `rid`, query span `[m,n)`, SA interval `[k,l)` of size `s`) that
appear adjacent in the sorted SMEM array are compacted in O(n) with no
allocation.

Duplicate SMEMs arise because the FM-index is a B-tree and can contain duplicate
keys, particularly in repeat-dense regions. On 50 k WGS reads vs hg38, roughly
10 % fewer SA lookups are performed with `--smem-dedup` active. The wall-clock
impact on an arm64 host is ~1–2 % (seeding is a fraction of total runtime);
on x86 workloads where SA expansion is a larger share the gain is correspondingly
larger.

The accuracy impact is small and bounded: only reads that carried duplicate SMEMs
can be affected, and only if the deduplication changes which chain wins. On the
50 k validation set, 2 reads (0.004 %) differed — one XS tag update on a MAPQ-60
read (primary coordinates unchanged) and one equal-score MAPQ-0 tie-break shift.
Zero uniquely-mapped reads changed. See the [root cause
analysis](https://github.com/fg-labs/bwa-mem3/pull/187) for the full characterization.

> **Not byte-identical**
>
> Do not enable in pipelines that compare output to a bwa-mem2 or bwa-mem3
> baseline. The changes are benign and bounded, but they are real SAM changes.

## `--dedup` extension-DP job deduplication

`--dedup STR` scores each distinct extension Smith-Waterman job once per batch and
copies the result to every identical job, instead of re-running the kernel on
duplicates. Unlike `--smem-dedup`, this preserves **byte-identical alignment
records in every mode**; headers such as `@PG` (which embed argv) are outside this
guarantee. The duplicates would have produced the same score, so collapsing them
changes only how much kernel work runs, never the alignment output. It accepts
three values:

- `off` — always run the kernel on every job (the pre-feature behavior).
- `on` — always dedup identical `(query, target, h0)` jobs within each batch.
- `auto` — **the default**: measure the net benefit at runtime (the bookkeeping
  dedup adds versus the kernel time it removes), latch ON or OFF from the measured
  sign, and periodically re-probe. Because the default is `auto`, dedup can run
  without any flag; alignment records stay byte-identical either way (headers excluded).

Any value other than `off`/`on`/`auto` — including an empty `--dedup=` — is
rejected with a non-zero exit; there is no silent fallback. An explicit `--dedup`
value takes precedence over `BWAMEM3_DEDUP`, and when the flag is set an invalid
`BWAMEM3_DEDUP` mode is ignored rather than fatal.

Three expert env knobs tune the `auto` controller (env-only, no CLI flag). Each is
full-string parsed: a malformed or out-of-range value is fatal, and trailing
non-numeric junk (`2x`, `12M`) is rejected rather than accepted as a numeric prefix.

- `BWAMEM3_DEDUP` — override the mode (`off`/`on`/`auto`); same values as the flag.
  An unrecognized mode is fatal (unless an explicit `--dedup` flag overrides it).
- `BWAMEM3_DEDUP_Z` — the z-score confidence threshold for latching and for
  confirming a re-probe (a number > 0). Reversing an existing latch needs a
  higher bar, `BWAMEM3_DEDUP_Z + 1`, so noise near break-even cannot flap the
  decision.
- `BWAMEM3_DEDUP_REPROBE` — re-probe cadence in jobs (a non-negative integer;
  `0` disables re-probing).

## `--dedup-reads` whole-read-pair memoization

`--dedup-reads STR` aligns each **distinct read-pair** once within a chunk and
reuses the result for byte-identical duplicate pairs, instead of running the full
seed → chain → extend pipeline on every duplicate. Where `--dedup` collapses
identical *extension jobs* (a fine-grained slice), this collapses identical whole
*read-pairs* (a coarse slice) — useful on high-duplication library preps such as
amplicon/UMI panels, and near-inert on WGS/exome where exact-duplicate pairs are
rare. Like `--dedup`, it preserves **byte-identical alignment records in every
mode**: a duplicate replays the per-read SAM stage (pairing, MAPQ, tie-break,
QNAME/QUAL) over the representative's alignment, so nothing position-dependent is
copied. This is a by-construction invariant, verified by an in-repo regression
(`test/regression/dedup_reads_byte_identity.sh`) that asserts `off`==`on`==`auto`
on dup-rich, low-dup, and N+lowercase phix-derived PE reads, single- and
multi-threaded, on one host/binary — not a cross-host or cross-tier measurement.
It accepts three values:

- `off` — always align every pair (the pre-feature behavior).
- `on` — always memoize identical pairs within a chunk.
- `auto` — **the default**: measure the duplicate rate and the net benefit at
  runtime (align work avoided versus fingerprint/copy overhead), latch ON or OFF
  from the measured sign, and periodically re-probe. Alignment records stay
  byte-identical either way.

The dedup window is one align invocation (the `-K` chunk): duplicates are
collapsed within a chunk only, because the alignment result depends on chunk-scoped
insert-size state (`mem_pestat`) that is constant within a chunk but varies across
chunks — so cross-chunk reuse would not be byte-identical.

Any value other than `off`/`on`/`auto` — including an empty `--dedup-reads=` — is
rejected with a non-zero exit; there is no silent fallback. An explicit
`--dedup-reads` value takes precedence over `BWAMEM3_DEDUP_READS`. Three expert env
knobs tune the `auto` controller (env-only, full-string parsed, malformed values
fatal): `BWAMEM3_DEDUP_READS` (override the mode), `BWAMEM3_DEDUP_READS_Z` (z-score
latch threshold, > 0; reversing a latch needs `+ 1`), and
`BWAMEM3_DEDUP_READS_REPROBE` (re-probe cadence in read-pairs; `0` disables).
`BWAMEM3_DEDUP_READS_STATS=1` dumps the duplicate rate and final latch state at
exit. `BWAMEM3_DEDUP_READS_VERIFY=1` is a correctness diagnostic: it aligns
duplicate pairs normally (no work is skipped) and asserts each duplicate's
alignment regions match its representative's field-by-field, aborting on any
divergence — the position-invariance guarantee, checked on real data.

## `--min-ext-len` short-seed extension filter

`--min-ext-len INT` opts into skipping banded Smith-Waterman extension of short
seeds (< `INT` bp) **that sit in a chain with a longer anchor seed** — the
anchor's extension already covers them, so their own extension is redundant. Off
by default (`0`) → output byte-identical to baseline.

Smith-Waterman extension is ~60 % of `bwa-mem3 mem` CPU, and almost all of it is
spent on short seeds: seeds ≤40 bp hold roughly **90 % of all banded-SW cells**
yet are ~99 % wasted, because long seeds already resolve via the ungapped
fast-path at near-zero cost. The filter drops those redundant short seeds before
extension (`mem_chain_drop_short_seeds`, a stable in-place compaction of each
chain's seeds, called from `mem_flt_chained_seeds` in `src/bwamem.cpp`), so their
extension never runs while seeding and chaining are untouched.

**Recall-safe by construction.** A chain whose seeds are *all* short is left
intact — dropping its only evidence would unmap the read — so the filter never
empties a chain. (An earlier version dropped every short seed unconditionally,
which silently unmapped low-mappability reads: a 151 bp low-mappability sample
lost 63 % of its mappings. The anchor guard fixes this; it is a strict recall
improvement that can reduce work but never lose a read.)

Measured single-thread on hg38 (HG002 1M PE WGS, non-emptying filter):

- **~10 % lower `main_mem` CPU at `--min-ext-len 30` (−9.5 % measured), with no
  recall loss** — mapped count is identical to default (99.75 %). ~0.10 % of
  reads change locus (down from ~0.40 % under the old emptying filter), confined
  to the low-confidence tail; ~0.005 % of reads change at MAPQ ≥ 60.
- **Higher thresholds no longer cliff:** mapped count and ~0.10 % divergence hold
  flat across `30`–`50`, because all-short chains are now protected.
- The previously-documented high-error F1 cliff and the cross-architecture speed
  figures were measured under the *emptying* behavior; both need a fresh
  [bwa-mem3-bench](../related-projects/bwa-mem3-bench.md) run under the
  non-emptying filter. Indels and structural variants were never a
  contraindication (an indel leaves two still-long exact segments the fast-path
  handles).

`30` is the recommended opt-in value. Because the speedup thins the extension
stage — not seeding — the wall-clock gain is smaller than the cell-count
reduction suggests (seeding, which the filter does not touch, dominates runtime).
See
[CLI → mem `--min-ext-len`](../cli/mem.md#--min-ext-len-int--skip-smith-waterman-extension-of-short-seeds)
and
[Settings profiles](../best-practices/settings-profiles.md#short-seed-extension---min-ext-len-30)
for the recommended operating point.

---

## `--seed-order` seed reordering before chaining

`--seed-order <mode>` reorders each read's SA-resolved seeds before chaining. The default
`off` preserves byte-identical output. The recommended opt-in mode is `local-longest`,
which sorts seeds by decreasing length so the longest seed anchors its chain first and
absorbs contained shorter seeds — those sub-seeds then never reach banded Smith-Waterman.

```bash
bwa-mem3 mem --seed-order local-longest -t 16 ref.fa R1.fq.gz R2.fq.gz | samtools sort -@ 4 -o out.bam -
```

Measured on 50,000 real WGS reads (1000 Genomes HG00096, hg38), `local-longest` reduces
extended seeds by ~8.9 % (absorbed fraction increases from 38.2 % to 43.7 %). Since
Smith-Waterman extension is typically the dominant per-read cost in `bwa-mem3 mem`, this
translates to a meaningful throughput gain on extension-heavy workloads.

`--seed-order local-longest` is **not byte-identical** to the default — it can shift
secondary alignments, `XA:Z:`, `XS:i`, `HN:i`, and a small number of primaries. Accuracy
is flat on easy simulated data (holodeck, F1 ~94.4 %; no regression vs `off`), but
hard-data F1 validation on divergent/indel-rich reads and GIAB benchmarks is not yet
complete. For that reason, all non-`off` modes are opt-in only and the default stays `off`.

See [Optimization checklist → Reorder seeds longest-first](../best-practices/optimization-checklist.md#6-reorder-seeds-longest-first---seed-order-local-longest) and [Equivalence → Seed ordering](equivalence.md#seed-ordering---seed-order-opt-in) for full details.

## `--huge-pages` 1 GB huge pages for the index

`--huge-pages` (Linux, off by default) backs the FM-index and suffix-array
structures with explicit **1 GB huge pages** to cut data-TLB misses on the random
SA/BWT accesses that dominate seeding. bwa-mem3's allocator is the vendored
mimalloc (above), so the flag reserves the pages through it
(`mi_reserve_huge_os_pages_interleave`) — sizing the reservation from the index
footprint and reserving before the index loads.

It is **safe by default**: when the host has no free 1 GB hugepage pool, has too
few pages, is not Linux, or was built without mimalloc, the flag prints a one-line
`[M::]` note and runs on the default page size. The pages must be pre-reserved on
the host (`nr_hugepages`); transparent huge pages (2 MB) do not reproduce the win.

The **alignment records are byte-identical** — page size does not change
alignments; only the `@PG` record differs, since its `CL:` field records the
`--huge-pages` flag on the command line. Measured effect: ~1.5 % whole-aligner
wall and ~0.9 % user CPU on a 5 M-read WGS slice (HG00096, hg38), AMD Zen3, avx2
tier, 32 threads ([#405](https://github.com/fg-labs/bwa-mem3/pull/405)). This is a
single-host, single-config measurement, not a cross-architecture or cross-thread
claim.

See [Memory allocator → Large pages for the index](../user-guide/allocator.md#large-pages-for-the-index-linux-deployment-lever)
and [Optimization checklist → Reserve 1 GB huge pages](../best-practices/optimization-checklist.md#8-reserve-1-gb-huge-pages-for-the-index-linux-opt-in)
for reservation, verification, and the manual `MIMALLOC_RESERVE_HUGE_OS_PAGES` equivalent.

## Certified adaptive extension band (default on; `--no-band-cert` to disable)

By default, `bwa-mem3 mem` now runs the banded Smith-Waterman **seed extension**
with a *certified adaptive band*: instead of opening every extension at the full
band width `opt->w` and doubling on retry, it first scores each pair at a narrow
probe band and **finalizes only the pairs it can prove are already optimal there**.
Every pair it cannot prove falls through to the exact full-width ceiling ladder
(`[opt->w, 2·opt->w, 4·opt->w, 8·opt->w]`) and is scored identically to the
non-adaptive path.

The narrow probe runs on the scalar and 16-bit extension kernels. The 8-bit kernel
(`BSW_TIER_8`) is deliberately excluded: it serves short reads whose band already
fits its `BSW8_MAX_W` ceiling, so it opens at `opt->w` and runs the standard ladder
— there is no narrow-band win to reclaim there.

**The alignment records are byte-identical to a full-width extension** (the `@PG`
`CL:` command line excepted) — this is a default behavior change to the
*internals* of extension, not to any emitted alignment field. The
proof is per pair: a gapped alignment that reaches diagonal offset `d` costs at
least `o_min + d·e_min` in gap penalty, so its best possible score is bounded by
`h0 + min_len·a − o_min − d·e_min`. That bound can tie or beat the achieved
score only for offsets up to a computable `d_max`; if the probe band already
covers every such offset, no tying-or-better alignment exists beyond it, and the
narrow result equals the full-width result — score, query-end score, and
coordinates alike. A pair is finalized narrow only when that certificate holds
**and** it reaches the query end within the clip penalty of its local maximum, so
the clip-vs-extend decision cannot flip when the band widens. The certificate is
anchored on the achieved score *minus the clip penalty*, which makes the query-end
score, its coordinate, and the clip decision band-invariant too — not just the
local maximum.

The certificate bounds the optimal *score*; it does not bound the extension
kernel's early-termination heuristics (the z-drop and all-zero-row breaks and the
band-edge shrink), which read cells a narrow and a full-width run can compute
differently. Those heuristics stay quiescent across a conservative parameter
envelope — a large enough `-d`/z-drop relative to the certifiable band, clip
penalties below a single gap's cost, and a scoring matrix whose entries do not
exceed the match reward. **Outside that envelope the certified band is disabled
automatically and the exact full-width ladder runs instead**, so output is
byte-identical for any `-d`/`-L`/`-O`/`-E`/`-A`/`-B`. Default parameters are inside
the envelope, so a plain run always takes the fast path.

This differs from **`--adaptive-band`** (below / in the divergence catalog), which
is an *aggressive, opt-in, not byte-identical* narrowing that trades exactness for
more speed. The certified band is the opposite trade: full exactness, a smaller
but free speedup.

```bash
bwa-mem3 mem ref.fa R1.fq R2.fq              # certified adaptive band (default)
bwa-mem3 mem --no-band-cert ref.fa R1.fq R2.fq   # full-width exact ladder, byte-identical, slower
```

On the default preset, pass **`--no-band-cert`** to disable certification and run
the full-width ladder for every pair. The alignment records are byte-identical either
way (the `@PG` `CL:` command line excepted); the flag exists as an escape hatch and
as an A/B handle for the byte-identity regression test. It scopes to the default
preset only: `--no-band-cert` clears `band_cert` but does not reset `band_start`, so
under `--adaptive-band` or `--fast` the aggressive band is already in force (the
certificate off) and `--no-band-cert` is a no-op there — use **`--no-adaptive-band`**
when you need exact full-width extension alongside an aggressive preset. Measured effect: ~3 %
whole-aligner wall on a 5 M-read WGS slice (HG00096, hg38), measured on an
**Apple M2 Max** (MacBook Pro 14″, `Mac14,6`; arm64 / NEON tier), clang, 16
threads — a single-host, single-config measurement, not a cross-architecture or
cross-thread claim. `--fast` and `--adaptive-band` turn the
certificate off (they select the aggressive band instead); `--meth` also runs the
exact full-width ladder (the certificate is a non-meth optimization).

## `--compat-allow-divergent` force a bwa-mem3 lever under `--compat`

`--compat=<target>` is a byte-identity contract, so it rejects any bwa-mem3-only lever that changes
alignments or MAPQ — the upstream has no such knob and cannot reproduce the output at any setting.
`--compat-allow-divergent` downgrades that refusal to a warning (naming the offending flags) and
proceeds: the user keeps the target's output *conventions* (tag/header shaping) with a lever
engaged, but the output is then **not** byte-identical to the target (the `@PG` `CL:` record already
records which flags ran). Default off; no effect without `--compat`.

It deliberately does **not** relax `--fast` or `--meth`: those are category errors, not
divergences — `--fast` is an opaque multi-flag bundle, and no target has a bisulfite mode, so
"byte-identical to X under `--meth`" is undefined rather than merely violated. The full reject list
and the shared-knob rule live in [Equivalence](equivalence.md) and [Alignment modes](modes.md).

## Changes catalog

| Item | bwa-mem3 PR | Upstream PR/issue | Status |
|------|-------------|-------------------|--------|
| `--meth` bisulfite alignment mode | [#13](https://github.com/fg-labs/bwa-mem3/pull/13) | — | fork-only |
| Vendored mimalloc allocator | [#19](https://github.com/fg-labs/bwa-mem3/pull/19) | — | fork-only |
| `--supp-rep-hard-cap` MAPQ rescoring | [#56](https://github.com/fg-labs/bwa-mem3/pull/56) | [bwa-mem2#260](https://github.com/bwa-mem2/bwa-mem2/issues/260) | fork-only (upstream issue open) |
| `--proper-pair-from-emitted` `FLAG` `0x2` source | [#363](https://github.com/fg-labs/bwa-mem3/pull/363) | — | fork-only (opt-in, off by default; default matches both upstreams) |
| `--hic` Hi-C preset | [#372](https://github.com/fg-labs/bwa-mem3/pull/372) | — | fork-only spelling (opt-in, off by default; sets the same flag bits as `-5SP`, which both upstreams also accept, so identical by construction — the regression confirms byte-identity on phix-derived PE reads with `@PG` excluded, not a cross-host/cross-tier claim) |
| `bwa-mem3 shm` shared-memory index | [#65](https://github.com/fg-labs/bwa-mem3/pull/65) | — | fork-only |
| `shm --meth` symmetry | [#67](https://github.com/fg-labs/bwa-mem3/pull/67) | — | fork-only |
| `HN:i` hit count tag | [#42](https://github.com/fg-labs/bwa-mem3/pull/42) | [lh3/bwa#438](https://github.com/lh3/bwa/pull/438) | fork-only (analogous to bwa aln) |
| `--bam=LEVEL` direct BAM output | [#12](https://github.com/fg-labs/bwa-mem3/pull/12) | — | fork-only |
| `--smem-dedup` SMEM deduplication | [#187](https://github.com/fg-labs/bwa-mem3/pull/187) | — | fork-only (opt-in, not byte-identical) |
| `--dedup` extension-DP job deduplication | [#415](https://github.com/fg-labs/bwa-mem3/pull/415) | — | fork-only (on by default via `auto`; alignment records byte-identical in every mode; headers excluded) |
| `--dedup-reads` whole-read-pair memoization | [#433](https://github.com/fg-labs/bwa-mem3/pull/433) | — | fork-only (on by default via `auto`; alignment records byte-identical in every mode; collapses duplicate read-pairs within a `-K` chunk) |
| `--min-ext-len` short-seed extension filter | _pending_ | — | fork-only (opt-in, off by default) |
| `--seed-order` seed reordering | [#186](https://github.com/fg-labs/bwa-mem3/pull/186) | — | fork-only (opt-in, off by default) |
| `--skip-contained-ext` contained-seed extension skip | [#192](https://github.com/fg-labs/bwa-mem3/pull/192) | — | fork-only (opt-in, byte-identical on short/medium non-meth reads, **not** byte-identical on kilobase-scale long reads, no-op under --meth) |
| `--max-extend-chains` chain-extension cap | [#193](https://github.com/fg-labs/bwa-mem3/pull/193) | — | fork-only (opt-in, not byte-identical) |
| `--extend-mate-concordant` mate-concordant chain retention | [#195](https://github.com/fg-labs/bwa-mem3/pull/195) | — | fork-only (opt-in, not byte-identical) |
| `--extend-tie-frac` / `--extend-tie-floor` score-gated chain-extension cap | [#269](https://github.com/fg-labs/bwa-mem3/pull/269) | — | fork-only (**default on under `--fast`** — including `--meth --fast` — at `0.95` / `1`; opt-in and off elsewhere, not byte-identical — the gate can drop candidate chains before banded-SW, moving secondaries / `XS` / `MAPQ`) |
| `--extend-csub` pruned-competitor `MAPQ` seed | [#269](https://github.com/fg-labs/bwa-mem3/pull/269) | — | fork-only (**forced on under `--fast`**, including `--meth --fast`; opt-in elsewhere, not byte-identical — changes emitted `MAPQ` on both the scalar SE and batched paired-end paths when the cap/gate prunes) |
| `--huge-pages` 1 GB huge pages for the index | [#405](https://github.com/fg-labs/bwa-mem3/pull/405) | — | fork-only (Linux, opt-in, off by default; alignment records byte-identical, `@PG` `CL:` excepted; single-host measurement, not a cross-host/cross-tier claim) |
| Certified adaptive extension band (`--no-band-cert` to disable) | [#420](https://github.com/fg-labs/bwa-mem3/pull/420) | — | fork-only (**default on**, alignment records byte-identical by a per-pair certificate — `@PG` `CL:` excepted; throughput measured on a 5M-read WGS slice, Apple M2 Max / arm64 NEON tier — see [the measurement](#certified-adaptive-extension-band-default-on---no-band-cert-to-disable) for full scope) |

---

**See also:**
[Methylation Reference → Overview](../methylation/overview.md) ·
[User Guide → Memory allocator](../user-guide/allocator.md) ·
[User Guide → Output: SAM/BAM, headers, tags](../user-guide/output.md) ·
[Getting Started → Quick start: shared-memory index](../getting-started/quick-shm.md) ·
[Best Practices → Output format](../best-practices/output-format.md)
