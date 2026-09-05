# Alignment modes: plain, `--compat`, and `--fast`

bwa-mem3 can be run three ways, and they differ in **what alignments come out** — not just
in speed. This page is the short, side-by-side statement of that difference; each row links to
the page with the full evidence.

Two things vary between the modes: the **alignment records** (where each read maps, and its
score/`MAPQ`/`FLAG`/`CIGAR`) and the **wrapper** (the additive AUX tags and the header block).
Keeping those two straight is the whole point of this page.

| mode | where reads align (records) | AUX tags | header |
|---|---|---|---|
| **plain** (default) | bwa-mem3's best judgment: bwa-mem2's alignments **plus bonafide correctness fixes** | adds `MQ:i`, `HN:i` | default `@HD`, `@PG:bwa-mem3`, `@SQ` enriched from a `.hdr`/`.dict` sidecar |
| **`--compat=bwa-mem2`** | reproduces **bwa-mem2 v2.2.1**'s alignment records | `MQ`/`HN` suppressed | matches bwa-mem2 v2.2.1 except `@PG` (still `bwa-mem3`) |
| **`--compat=bwa-mem`** | reproduces **bwa 0.7.19**'s alignment records — *including its divergences from bwa-mem2* | `HN` suppressed, `MQ` kept | matches bwa 0.7.19 except `@PG` (still `bwa-mem3`) |
| **`--fast`** | perf-driven — reshuffles the low-confidence tail (details below) | bwa-mem3 tags | bwa-mem3 header |

`--compat` rejects any bwa-mem3-only lever that changes alignments or MAPQ, since the upstream has
no such knob and so cannot reproduce the output at any setting. `--fast` and `--meth` are **always**
refused — category errors, not divergences: `--fast` is an opaque multi-flag bundle, and no target
has a bisulfite mode. The rest — `--smem-dedup`, `--adaptive-band`, `--max-extend-chains`,
`--min-ext-len`, `--extend-tie-frac`, `--rescue-kmer`, `--supp-rep-hard-cap`, `--seed-order`,
`--skip-contained-ext`, `--chunk-cap` and `--proper-pair-from-emitted` — are a hard error too, but
can be forced with **`--compat-allow-divergent`**, which keeps the target's output *conventions*
with a lever engaged (output is then not byte-identical). Shared knobs the upstream also has
(scoring, seeding widths, `-T`, …) are **not** guarded — changing one moves both sides to the same
operating point.

## 1. Plain (default) — a drop-in that fixes real bugs

With no flags, bwa-mem3 emits its own best output: bwa-mem2's alignments plus a set of
**bonafide correctness fixes** (SIMD scoring convergence, deterministic tie resolution, crash and
undefined-behavior fixes), plus two additive tags (`MQ:i`, `HN:i`) and an enriched header. The
differences from bwa/bwa-mem2 are believed to be genuine *improvements*, and they are small. On
the cells re-measured after the release-0.7.1 parity restoration, the plain profile is
**byte-concordant with bwa-mem2 v2.2.1** on the alignment records — differing only by the two
additive tags and the header. Strip those and the records are identical. Scoped to what was
measured: the complete alignment-record stream (`MQ:i`/`HN:i` stripped) is byte-identical on
`wgs-5M`, `wes-5M`, and `hic-1M` — 22.5 M records including 436 k supplementary — measured on
x86 `c6a` (AVX2), with a `c6a`/`c8g` cross-architecture check confirming the Arm `c8g` (NEON)
build is byte-identical to it. Separately — a **primary-alignment-only,
x86-only** spot check that the `c6a`/`c8g` cross-architecture check above does **not** cover —
**zero** primary alignments diverge (same `POS`/`CIGAR`/`FLAG`/`MAPQ`) on a 1,066,668-record
HG00096 WGS slice (hg38); supplementary and secondary records on that slice were not compared.
The genome-wide, multi-sample re-run that would extend this past those named cells is still pending.

This is the mode to run for a new pipeline, and the mode to validate first when migrating. See
[Equivalence with bwa-mem2](equivalence.md) for the field-by-field audit and the exact datasets
each claim is scoped to, and [Coming from bwa or bwa-mem2](../getting-started/migrating.md) for
the migration sequence.

## 2. `--compat=<target>` — reproduce a specific upstream exactly

`--compat` shapes bwa-mem3's output to be byte-for-byte identical to a **specific** upstream
release on the **alignment records**: `--compat=bwa-mem2` for bwa-mem2 v2.2.1, or `--compat=bwa-mem`
for bwa 0.7.19. Use it to validate a bwa-mem3 rollout against an existing bwa/bwa-mem2 golden with a
plain `diff` — excluding the `@PG` line, which still names `bwa-mem3` and is unmatchable by
construction, and passing matching `-t` (or a pinned `-K`) on both sides so batch boundaries line up.

Two things to know:

- **The two targets are not interchangeable.** bwa and bwa-mem2 diverged from each other on the
  very fields `--compat` shapes (bwa gained a default `@HD` and the `MQ:i` tag after bwa-mem2
  forked), so matching one means *not* matching the other. Pick the target for the aligner you
  actually validate against.
- **`--compat` can change an alignment, not only the wrapper — in one narrow case.** It is
  almost entirely output-shaping (tags and header), but `--compat=bwa-mem` also reproduces bwa
  0.7.19's behavior on the "all candidate chains were weight-filtered" path, where bwa leaves the
  read **unmapped** and bwa-mem2 (and plain bwa-mem3) align it from the rejected chain. That path
  is unreachable at default settings — it needs `-W` or an `-x pacbio`/`pbref`/`ont2d` preset —
  but it is why a small fraction of alignments *can* differ between the two compat targets, and
  between a compat target and plain, beyond tags and header.

See [`mem` → `--compat`](../cli/mem.md#--compattarget--byte-identical-output-for-another-aligner)
for the per-target field table and
[Equivalence → Byte-identical output](equivalence.md#byte-identical-output---compat) for the
validation totals.

## 3. `--fast` — faster, at the cost of the low-confidence tail

`--fast` turns on a bundle of characterized speed levers (see
[`mem` → `--fast`](../cli/mem.md#--fast--speed-preset-opt-in-not-byte-identical) and
[Settings profiles](../best-practices/settings-profiles.md)). Unlike the fixes in plain mode,
these levers **change alignments to gain speed**, so `--fast` output is *not* diff-compatible
with the default. It is worth understanding along **two separate axes**:

- **Accuracy (does it map reads correctly?) — neutral.** Against simulated golden truth,
  `--fast` places reads correctly at the same rate as the default: the placement-correct rate
  moves by **≤0.02 pp** across the WGS and methylation sims, with no change in the unmapped rate.
  On the bisulfite sims both placement and per-CpG methylation correlation are unchanged from the
  default (a free ~1.5–2× speedup). It does not map fewer reads, or map them to worse places on
  average.
- **Record concordance (does it produce the same records as the default?) — no.** `--fast`
  re-places a real fraction of records — roughly **3% of positions on WGS, up to ~6% on a
  high-depth amplicon panel** — and moves `MAPQ`, `XS`, and secondary alignments on more.

All the `--fast` figures on this page come from one auditable scope: the
[bwa-mem3-bench](../related-projects/bwa-mem3-bench.md) release-validation matrix — five real
cells (`wgs-5M` for WGS, `panel-twist-5M` for the high-depth amplicon panel, and `wes-5M` at
5 M reads each, plus `hic-1M` and `sbx-1M` at 1 M reads), run across every SIMD tier bwa-mem3
ships (AVX2 `c6a`, AVX-512 `c7a`/`c7i`,
NEON `c7g`/`c8g`; methylation on `m7i`), each a `.4xlarge` (16-vCPU) host at `-t 16` with the
per-batch size pinned to `-K 160000000` (160 M bases, the historical `chunk_size × 16` default)
on both bwa-mem3 and the bwa-mem2 baseline, on hg38, with placement accuracy graded against
simulated golden truth. That page carries the per-cell, per-architecture breakdown.

Those two facts are reconciled by *which* reads move: **`--fast` reshuffles the low-confidence
tail and almost nothing else.** Binning the reads it re-places by the confidence (`MAPQ`) of the
default alignment, on the `wgs-5M` and `panel-twist-5M` cells above:

- **~85%** of all moved reads had `MAPQ 0` in the default run, and **~92%** had `MAPQ < 10`;
- the confident **`MAPQ`-60 core barely moves — ≤0.45%** of those reads on the amplicon panel,
  and just **0.011%** on WGS.

The reads that move are near-ties among repetitive or ambiguous loci, where which copy wins is a
coin-flip that does not affect truth-recall — which is exactly why accuracy stays neutral. The
larger record-divergence on amplicon data is not `--fast` being more aggressive there; it is that
such data is dominated by `MAPQ`-0 reads to begin with, so there is simply more low-confidence
material to shuffle.

**When to use it.** `--fast` is a good opt-in choice for a new high-throughput pipeline where you care
about the confident, uniquely-mapped calls. Evaluate it critically for your application if you
depend on **exact record identity** (e.g. re-validating against a default-mode golden),
**`MAPQ` calibration in repetitive regions**, or **which repeat copy is reported** (some
depth-, CNV-, or repeat-aware analyses).

## Which mode should I use?

- **Migrating an existing bwa/bwa-mem2 pipeline, or validating against one** → **plain**, and
  add `--compat=<your reference aligner>` if you want a byte-clean `diff` of the alignment records
  (exclude `@PG`, and match `-t` or pin `-K` on both sides).
- **A new pipeline** → **plain** (safe drop-in) or **`--fast`** (faster; most confident calls
  are preserved, but a small `MAPQ`-60 fraction can move, and the low-confidence tail is
  reshuffled).
- **Reproducing a specific bwa or bwa-mem2 release exactly** → **`--compat`** with the matching
  target.

---

**See also:**
[Overview](overview.md) ·
[Equivalence with bwa-mem2](equivalence.md) ·
[Coming from bwa or bwa-mem2](../getting-started/migrating.md) ·
[Settings profiles: drop-in vs recommended](../best-practices/settings-profiles.md) ·
[CLI Reference — `mem`](../cli/mem.md)
