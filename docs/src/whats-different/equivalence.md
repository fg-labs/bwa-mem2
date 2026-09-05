# Equivalence with bwa-mem2 (bit-identity)

<div class="warning" style="border:3px solid #2e7d32;border-radius:8px;padding:1rem 1.25rem;margin:1.25rem 0;background:#e8f5e9;color:#1b1b1b;">
<strong style="font-size:1.35em;display:block;margin-bottom:0.4rem;">✅ No alignment record bwa-mem2 emits is changed or removed — and on the cell re-measured for 0.7.1 its primary alignments are reproduced exactly.</strong>
On the drop-in profile (plain <code>bwa-mem3 mem</code>, no flags), release 0.7.1 gives the <strong>same primary alignment as bwa-mem2</strong> for <strong>every</strong> read of the slice re-measured after the parity work landed (1,066,668 records, HG00096 WGS, hg38, x86): identical <code>FLAG</code>, <code>RNAME</code>, <code>POS</code>, <code>MAPQ</code>, <code>CIGAR</code>, <code>AS</code>, and <code>XS</code>. Those record bytes differ by <strong>two additive tags</strong> and nothing else:
<ul style="margin:0.5rem 0 0 0;">
<li><code>MQ:i</code> — mate mapping quality (an extra tag bwa-mem2 never wrote)</li>
<li><code>HN:i</code> — hit count (an extra tag bwa-mem2 never wrote)</li>
</ul>
Strip those two tags and those records are byte-for-byte identical — or pass <strong><code>--compat=bwa-mem2</code></strong> to suppress both at the source (see <a href="#byte-identical-output---compat">Byte-identical output</a>). The <strong>header</strong> is a separate question that these measurements do not cover: on the default path the <code>@PG</code> line names <code>bwa-mem3</code> and bwa-mem3 emits a default <code>@HD</code> line that bwa-mem2 has no code to write at all (<a href="https://github.com/fg-labs/bwa-mem3/issues/288">#288</a>). <code>--compat=bwa-mem2</code> brings the <code>@HD</code> and <code>@SQ</code> blocks to parity as well, leaving only <code>@PG</code> to exclude. Every byte-identity claim measured on this page is about <em>alignment records</em>, not the header block.
<br><br>
<strong>The supplementary gap is now closed.</strong> The residual <strong>+4 supplementary alignments</strong> has been re-measured on <code>e722ed0</code> (a build carrying <a href="https://github.com/fg-labs/bwa-mem3/pull/268">#268</a>) and is <strong>+0</strong>: on <code>wgs-5M</code>, <code>wes-5M</code> and <code>hic-1M</code> the full <em>alignment-record</em> stream is byte-identical to bwa-mem2 v2.2.1 once <code>MQ:i</code>/<code>HN:i</code> are stripped — 22.5 M records including 436 k supplementary alignments. That comparison covers alignment records only, not the header block. What remains open is the genome-wide, multi-sample re-run that would extend this past the cells named on this page. Treat every claim here as scoped to the evidence it cites, not as a guarantee across all inputs and architectures. (The opt-in <code>--fast</code> speed levers are a separate, deliberate deviation — see below.)
</div>

> New here? [Alignment modes: plain, `--compat`, `--fast`](modes.md) is the short, side-by-side
> overview of the three ways to run bwa-mem3; this page is the detailed equivalence audit behind
> the plain and `--compat` claims.

As of **release 0.7.1**, the bwa-mem3 **drop-in profile** (plain `bwa-mem3 mem`, no
extra flags) produces the **same primary alignment** as bwa-mem2 — same `FLAG`,
`RNAME`, `POS`, `MAPQ`, `CIGAR`, `AS`, `XS` — on the data we have re-measured since
the restoration. The **SAM byte stream** is still not a literal byte-for-byte
match: bwa-mem3 emits extra auxiliary tags (`MQ:i`, `HN:i`) that upstream never wrote.
Strip those two tags and the *records* are identical. The header is not covered by that
statement — the `@PG` line names `bwa-mem3`, and the default `@HD` bwa-mem3 writes has no
bwa-mem2 counterpart ([#288](https://github.com/fg-labs/bwa-mem3/issues/288)). This is a change in
posture from earlier releases, which carried three changes that *did* move primary
alignments; all three were reverted for 0.7.1 (see [Restored bwa-mem2 parity
(0.7.1)](#restored-bwa-mem2-parity-071) below).

Two things scope that statement, and both are tracked below rather than assumed away.
The **supplementary** set is now confirmed to match: the extra split alignments were cut
from +239 to +4 by [#257](https://github.com/fg-labs/bwa-mem3/pull/257), and re-measurement on
`e722ed0` (which carries [#268](https://github.com/fg-labs/bwa-mem3/pull/268)) puts it at **+0**
(see [Additional supplementary
alignments](#additional-supplementary-alignments-resolved)).
And the post-restoration evidence base is still narrow, in two different ways: the
**0-divergent-primaries** count comes from a single cell (the 1,066,668-record batch-1 WGS
slice, x86), while the **byte-identity** result spans three cells (`wgs-5M`, `wes-5M`,
`hic-1M`, x86) with a `c6a`/`c8g` cross-architecture check on top. The genome-wide,
multi-sample concordance re-run that would extend either past those cells is still pending.
Read every figure on this page as scoped to the cell it names.

Upstream bwa-mem2 advertises exact equivalence with the original `bwa` — "produces
alignment identical to bwa" and "exact same output as bwa-mem(2)" — and that guarantee
was the right bar for a project whose sole charter was to reproduce `bwa mem` faster.
bwa-mem3 has a broader charter: it adds informative SAM tags, fixes crashes and
undefined behavior, corrects SIMD scoring kernels, and makes tie resolution
deterministic. Where those changes would have moved *which alignment* a read gets, we
have chosen bwa-mem2 output compatibility on the default path; where they are purely
additive (the extra tags), confined to the `@PG` header naming, or strictly opt-in (the
`--fast` speed levers), they remain. This page records exactly where output does and does not match, and gives an auditable
trail back to every merged pull request — both the ones that *introduced* a divergence
and the ones that *restored* parity — so a reader can decide for themselves whether each
difference matters for their workflow.

The short version: on the drop-in profile the **core alignment** — where each read maps
and how, plus its scores — is preserved on essentially every read across every cell we
have measured, and on the cells re-measured after the 0.7.1 restoration it is
preserved exactly. On those records the **SAM byte stream** differs only by the additive
`MQ:i`/`HN:i` tags; the header block is a separate matter (the `@PG` line, and the default
`@HD` — [#288](https://github.com/fg-labs/bwa-mem3/issues/288)). The **supplementary set** is re-confirmed at +0 on `e722ed0`. The **opt-in speed levers** (`--fast` and its constituent flags) are
separately, deliberately *not* byte-identical; they are listed under "What differs" and
described in more depth on the [Correctness fixes](correctness.md), [Performance
improvements](performance.md), and [Features](features.md) pages.

## Restored bwa-mem2 parity (0.7.1)

Three fork-carried changes shipped between 0.5.0 and 0.7.0 each moved **primary**
alignments away from bwa-mem2 on the default path. Each was root-caused to a specific
mechanism and reverted (or repacked) for 0.7.1, pairing every divergence-introducing PR
with the PR that restored parity:

| divergence introduced by | mechanism | restored by |
|---|---|---|
| [#141](https://github.com/fg-labs/bwa-mem3/pull/141) — banded-SW extension opened gaps from the cell max `H` (standard Gotoh) instead of the diagonal `M` | `gscore` inflates upward-only, flipping the `gscore <= score - pen_clip3` local-vs-extend test in `mem_chain2aln` | [#256](https://github.com/fg-labs/bwa-mem3/pull/256) — derive extension gaps from `M` again (bwa-mem2's `bandedSWA` convention) |
| [#123](https://github.com/fg-labs/bwa-mem3/pull/123) — replaced bwa-mem2's partial-order dedup comparator with a strict total order + pdqsort | `mem_sort_dedup_patch`'s merge loop is order-sensitive; a total order picks different survivors among equal-`re` ties, moving `XS`/`MAPQ` and adding split alignments | [#257](https://github.com/fg-labs/bwa-mem3/pull/257) — restore bwa-mem2's `re`-only comparator + introsort outcome on the default path (the pdqsort/total-order sort moves behind `--fast`), plus [#261](https://github.com/fg-labs/bwa-mem3/pull/261) — run pdqsort only when the comparator sees no tie, recovering its speed *with* byte-identity |
| [#174](https://github.com/fg-labs/bwa-mem3/pull/174) — the `--meth` feature appended `int8_t meth_hypothesis` to `mem_chain_t`, growing it 48 → 56 B | klib's kbtree derives node fan-out from `sizeof(key_t)`; the changed fan-out reorders `.pos`-tied chains, regrouping seeds and flipping co-optimal primaries — even on **non-`--meth`** runs | [#268](https://github.com/fg-labs/bwa-mem3/pull/268) — repack `meth_hypothesis` into the existing bitfield word so `sizeof(mem_chain_t)` returns to 48 B (guarded by `static_assert`) |

The reductions were measured per-PR against a real bwa-mem2 v2.2.1 golden (5 M HG00096
WGS pairs, hg38, `-t 16` on both sides so batch boundaries and `mem_pestat` match):
[#256](https://github.com/fg-labs/bwa-mem3/pull/256) removed ~14 divergent primaries
(all `XS`); [#257](https://github.com/fg-labs/bwa-mem3/pull/257) took the residual from
1,501 (0.0150%) to 508 (0.0051%) and the extra supplementary alignments from +239 to +4;
and [#268](https://github.com/fg-labs/bwa-mem3/pull/268), measured on a build that
already carried `#256` and `#257`, took the remaining 54
divergent primaries on a 1,066,668-record batch-1 slice to **0** — every changed record
matching bwa-mem2 exactly, with zero regressions. A combined genome-wide, multi-sample,
cross-architecture concordance re-run to republish the aggregate figures below and
regenerate the [declared divergence catalog](#declared-divergence-catalog) is pending.

> **This is a compatibility choice, not a correctness claim.** Gap-from-`H` is textbook
> Gotoh and what ksw2/minimap2 do; gap-from-`M` is a deliberate bwa idiosyncrasy. `#256`
> chooses bwa-mem2 output compatibility over the convention because bwa-mem3 is a drop-in
> replacement.

## Byte-identical output (`--compat`)

With the primary alignments restored (above), a small set of bwa-mem3-side additions is all
that stands between the drop-in profile and a byte-for-byte match with bwa-mem2 v2.2.1.
`--compat=bwa-mem2` removes exactly those:

```bash
bwa-mem3 mem --compat=bwa-mem2 -t <N> ref.fa R1.fq R2.fq > out.sam
```

| | default | `--compat=bwa-mem2` | bwa-mem2 v2.2.1 | `--compat=bwa-mem` | bwa 0.7.19 |
|---|---|---|---|---|---|
| `MQ:i` | emitted | suppressed | absent | emitted | **present** |
| `HN:i` | emitted | suppressed | absent | suppressed | absent |
| default `@HD` | emitted | suppressed | **none emitted** | emitted | **present** |
| `.hdr`/`.dict` sidecar `@SQ` | honored (`M5`/`AS`/`UR`/`SP`) | ignored → bare `SN`/`LN` (+`AH:*`) | bare `SN`/`LN` (+`AH:*`) | ignored → bare `SN`/`LN` (+`AH:*`) | bare `SN`/`LN` (+`AH:*`) |
| `@PG` | `ID:bwa-mem3` | `ID:bwa-mem3` | `ID:bwa-mem2` | `ID:bwa-mem3` | `ID:bwa` |

**The two upstreams disagree with each other on half of this table, which is why `--compat`
takes a *target* rather than being a boolean.** `MQ:i` is not a bwa-mem3 invention — bwa
emits it too ([lh3/bwa#330](https://github.com/lh3/bwa/pull/330), merged 2022-03-06);
bwa-mem2 lacks it only because it forked at 0.7.17, before that landed. Likewise **bwa emits
a default `@HD`** ([lh3/bwa#336](https://github.com/lh3/bwa/pull/336), merged 2022-03-06,
shipped in 0.7.18 as `6b18630`) and bwa-mem2 does not, for the same reason. Only `HN:i`
and the sidecar are genuinely bwa-mem3-only, and both targets drop them; every other row is
the fork point showing through. `bwa-mem` is pinned at **0.7.19** for exactly this reason — a
0.7.17 target would be the `bwa-mem2` column with a different `@PG`.

**Why a bwa target needs no alignment work.** bwa-mem2 v2.2.1 advertises output identical to
bwa 0.7.17, and auditing `git diff v0.7.17 v0.7.19` in lh3/bwa shows the only
output-affecting changes in that range are the additive `MQ:i` and `@HD` above plus the
opt-in `-u` (`XB:Z`) and `-z` (XA drop ratio) flags — **nothing touches seeding, chaining,
extension, pairing, MAPQ or dedup**. So bwa-mem2's claim carries forward to 0.7.19, and the
bwa-mem2 parity restored in 0.7.1 (see above) is bwa parity too. That is what the measured
three-way agreement below reflects, rather than a coincidence of the cells chosen.

**One structural limit.** `--compat` shapes output and never moves an alignment, so on any
record where bwa and bwa-mem2 disagree on an *alignment* field, at most one of the two targets
can be byte-identical. No such record has been observed across the cells measured here, but
the rare-event tail is not proven empty — see the statistical-power caveat below.

That limit is the reason both targets exist as a table rather than a boolean, and it is not
hypothetical in origin: bwa-mem2's own tracker carries reports of exactly this shape. The
best-documented one,
[bwa-mem2#5](https://github.com/bwa-mem2/bwa-mem2/issues/5), is the only such report with a
complete reproducer — read `ERR001713.5425457` from NA12878 run ERR001713, which bwa 0.7.17
placed at `6:83350583` (`7S29M`) while the then-current bwa-mem2 placed at `13:102092425`
(`8S28M`). **It was re-run in full on GRCh37+decoy and no longer reproduces.** bwa 0.7.19
still emits its original record byte for byte, and bwa-mem2 **v2.2.1 now agrees with it** —
so the divergence was a bwa-mem2 defect fixed upstream before v2.2.1, not a standing
difference. Both bwa-mem3 targets match their respective upstream on that read exactly.

The class is therefore *narrower* than the tracker suggests, but not empty by proof: the
remaining reports lack the data to reproduce, and the one quantified rate
([bwa-mem2#109](https://github.com/bwa-mem2/bwa-mem2/issues/109): 3 reads in 44.5 B pairs,
which that report states as ~88 B primary reads — **~1 in 29 billion** records) sits far
below the detection floor of anything measured here.

**The denominator for that comparison is not the 323 M records below.** This class is a
disagreement *between the two upstreams*, so only a record on which **both** upstreams were
actually run can exhibit it — and the 314.9 M GRCh38 matrix runs bwa against bwa-mem3, never
against bwa-mem2. The cells carrying both are the 12.4 M three-way records (`hic-1M` and
`wes-5M`, recorded further down this page) plus the 8,116,326-record GRCh37+decoy run above:
**20.5 M records, zero alignment-field disagreements**. By the rule of three — no events in
*n* trials puts the 95% upper bound on the rate at 3/*n* — that bounds the class at **~1 in
6.8 million**, roughly **4,000× coarser** than the reported rate, or about three and a half
orders of magnitude.

**A null result at this scale is not evidence of absence for that class**, and the 323 M
total does not improve the bound: a record on which only one upstream ran cannot exhibit an
upstream-vs-upstream divergence at any rate.

`--compat` is **output-shaping, with one deliberate exception**. Every alignment, score,
flag, and tag value is untouched except on the path where the two upstreams themselves
disagree about the alignment: when the chain weight filter drops every chain for a read,
bwa leaves it unmapped and bwa-mem2 aligns it from the chain it just rejected
([#310](https://github.com/fg-labs/bwa-mem3/issues/310)). `--compat=bwa-mem` reproduces
bwa there, `--compat=bwa-mem2` and the default reproduce bwa-mem2, and the drop-in profile
is unchanged. The path is unreachable without `-W` or an `-x pacbio`/`pbref`/`ont2d`
preset, since `min_chain_weight` defaults to 0.

**Verified** against a real bwa-mem2 v2.2.1 run on hg38: with `@PG` excluded on both sides —
it still names `bwa-mem3` and is unmatchable by construction, see the caveats below — the
header is byte-identical on both the SAM-text and `--bam` paths (3,366 lines, md5
`34dabb50dd7a704866e841d3e5a7f68d` on all three), and the records are byte-identical on the
drop-in profile.

```bash
# records
diff <(samtools view out.compat.bam) <(samtools view out.mem2.bam)
# header, @PG excluded on both sides
diff <(samtools view -H --no-PG out.compat.bam | grep -v '^@PG') \
     <(samtools view -H --no-PG out.mem2.bam   | grep -v '^@PG')
```

Caveats:

- **`@PG` is still emitted**, and still names `bwa-mem3`. Suppressing it would only turn a
  changed line into a missing one, since bwa-mem2 writes its own. It is unmatchable by
  construction anyway — `CL:` embeds the invocation and its paths, so even two bwa-mem2 runs
  from different directories differ. Exclude `@PG` on both sides when comparing.
- **`--compat` and `--fast` are mutually exclusive** — passing both is a hard error.
  `--fast` deliberately moves alignments; `--compat` only shapes output, so `--fast
  --compat=bwa-mem2` would produce a diff-clean-looking stream over genuinely different
  alignments. `--compat` is meaningful only on the drop-in profile.
- **`--compat` and `--proper-pair-from-emitted` are mutually exclusive** — passing both is a
  hard error. That flag derives `FLAG` `0x2` from the emitted alignment, which neither target
  does ([#363](https://github.com/fg-labs/bwa-mem3/pull/363)), so the combination asks for
  byte-identity and for a deliberate deviation from it at once.
- **Every other bwa-mem3-only alignment lever is likewise rejected** — a hard error naming the
  offending flags. `--smem-dedup`, `--adaptive-band`, `--max-extend-chains`, `--min-ext-len`,
  `--extend-tie-frac`, `--rescue-kmer`, `--supp-rep-hard-cap`, `--seed-order`, `--skip-contained-ext`
  and `--chunk-cap` each change alignments or MAPQ in a way no upstream can express, so they cannot
  coexist with a byte-identity target. **`--compat-allow-divergent`** downgrades this (and the
  `--proper-pair-from-emitted` case above) to a warning for a user who knowingly wants the target's
  output conventions with a bwa-mem3 lever engaged; the output is then not byte-identical. It does
  not relax `--fast` or `--meth`, which are category errors rather than divergences.
- **The sidecar `@SQ` is skipped, not rewritten.** Outside `--compat` the
  `<prefix>.hdr` / `<baseprefix>.dict` block remains authoritative and is emitted verbatim.
  On those sidecar-honoring paths — the default profile, `--bam` and `--meth` — an index with
  a `.alt` file needs that block to carry `AH:*` itself, because nothing re-derives it from
  the index; see [`@SQ` in Output](../user-guide/output.md#sq) for why, which commands warn,
  and how to regenerate the sidecar. Under **either** `--compat` target the requirement does
  not apply: both ignore the sidecar entirely and generate `AH:*` from the index (the table
  rows above), so a sidecar missing `AH` cannot affect that output.
- **The two targets differ on every mated record — they are not interchangeable.** On a
  4.06 M-pair GRCh37 run, bwa 0.7.19 and bwa-mem2 v2.2.1 differ on **all 8,116,326 records**,
  and on **0** once `MQ:i` is stripped. The alignments are the same; the output surface is not.
  Choosing the wrong target therefore produces a diff on essentially every record that has a
  mapped mate.
- **Evidence base**, stated in full because every claim on this page is scoped to it.
  `--compat=bwa-mem` is validated at **322,978,938 alignment records across two reference
  builds**, every record byte-identical to a real `bwa` 0.7.19 run with the headers matching
  too (`@PG` excluded):

  - **GATK hg38 — 314,862,612 records, 42 cells, 0 differing**, produced by the benchmark
    harness in [fg-labs/bwa-mem3-bench#47](https://github.com/fg-labs/bwa-mem3-bench/pull/47),
    which carries the per-cell breakdown. Seven datasets
    (`sim-wgs-place` 10,724,652 · `sim-wgs-vars` 10,163,754 · `wes-5M` 10,056,288 · `wgs-5M`
    10,030,558 · `panel-twist-5M` 8,100,270 · `hic-1M` 2,381,418 · `sbx-1M` 1,020,162 records
    per cell) × six hosts, `-K 160000000` pinned on both sides. The hosts cover every SIMD
    tier bwa-mem3 ships — `c6a` (AVX2), `c7i` / `c7a` / `m7i` (AVX-512), `c7g` / `c8g` (NEON)
    — with `bwa` built natively for each. Because bwa has had a NEON path since 0.7.18, the
    ARM rows compare ARM-to-ARM directly rather than transitively through an x86 run.
  - **hs37d5 (GRCh37 + decoy) — 8,116,326 records, 0 differing.** The full NA12878
    `ERR001713` FASTQ pair (4.06 M pairs at 36 bp) on one AVX2 host (`r6a`), `-K 10000000
    -t 8`, matching the invocation in [bwa-mem2#5](https://github.com/bwa-mem2/bwa-mem2/issues/5)
    so that the re-run above is faithful to the report it retires.

  `--compat=bwa-mem2` is validated at 22.5 M records (`wgs-5M`, `wes-5M`, `hic-1M`, x86) plus
  full header byte-identity on both the SAM-text and `--bam` paths, and is re-confirmed on the
  hs37d5 cell above. Both targets additionally carry a phiX-scale end-to-end check in
  `test/regression/compat_byte_identical.sh`.

  **Scope limit — the ALT-aware path is untested.** The hg38 build above carries 261 `_alt`
  and 525 HLA contigs but **no `.alt` sidecar**, so `is_alt` is never set and the entire
  ALT-aware branch (ALT primary selection, ALT MAPQ adjustment, `pa:f:`, the `-h INT,INT` cap)
  did not execute in any of those 314.9 M records. That is proven rather than assumed: the
  comparison runs a strict expected-tag allowlist with no `pa` entry and fails by name on an
  unexpected tag, and no `pa:f:` tag — emitted only when the ALT score is positive — appeared
  anywhere in the matrix. That gap is now partly closed in CI rather than only
  documented: `test/regression/alt_pa_parity.sh` builds an ALT sidecar and aligns
  reads sampled from the ALT contig, so `is_alt` → `alt_sc` → `pa:f:` executes on
  every canonical run (see [`pa:f:` renders identically in SAM text and
  BAM](#paf-renders-identically-in-sam-text-and-bam-resolved-2026-08-05)). ALT
  primary selection, ALT MAPQ adjustment and the `-h INT,INT` cap remain
  unmeasured. Two further upstream divergence reports
  ([bwa-mem2#227](https://github.com/bwa-mem2/bwa-mem2/issues/227),
  [bwa-mem2#61](https://github.com/bwa-mem2/bwa-mem2/issues/61)) are specifically about ALT
  handling, and nothing on this page covers them.

  > An earlier revision of this page and of `src/compat_target.cpp` stated that bwa-mem3 and
  > bwa differed on 224 of 63,583 records. **That figure is retracted** — it was measured
  > against a baseline binary built from a worktree on an unmerged branch 48 commits behind
  > `main`, and the corrected re-run of the same cell found the record stream byte-identical
  > (78,547 records including secondary and supplementary). It is recorded here rather than
  > deleted because it was quoted in a shipped CLI error message.

`--compat` is for the standard drop-in path only: neither target has a `--meth` mode to be
identical to, so **`--compat --meth` is a hard error** for both. See
[`mem` → `--compat`](../cli/mem.md#--compattarget--byte-identical-output-for-another-aligner)
for the flag reference.

## What is preserved

The figures in this section were measured **before** the 0.7.1 parity restoration and so
capture the residual drift of the three now-reverted changes ([#141](https://github.com/fg-labs/bwa-mem3/pull/141),
[#123](https://github.com/fg-labs/bwa-mem3/pull/123), [#174](https://github.com/fg-labs/bwa-mem3/pull/174);
see [Restored bwa-mem2 parity (0.7.1)](#restored-bwa-mem2-parity-071)). They are
**pre-restoration measurements**, not a bound on current behavior. The 0.7.1 drop-in profile
measured tighter everywhere it has been re-run: 0 diverging primaries on the batch-1 WGS
slice, and a byte-identical complete alignment-record stream (`MQ:i`/`HN:i` stripped) on `wgs-5M`,
`wes-5M` and `hic-1M` (see
[Additional supplementary alignments](#additional-supplementary-alignments-resolved)) — which
supersedes the `wes-5M` and `wgs-5M` rows below. `panel-twist-5M` has been re-run only for the
cross-architecture check, and `smoke-1M` has not been re-run at all, so those two figures still
stand as written. We ran an empirical concordance check with
[bwa-mem3-bench](https://github.com/fg-labs/bwa-mem3-bench) at commit `a02fcb4`, comparing
bwa-mem3 against upstream `bwa-mem2` v2.2.1 on x86 hosts across whole-genome, whole-exome,
and panel workloads. **Primary-alignment concordance** — reference name, position, CIGAR,
MAPQ, and placement flags compared per read end — was:

| sample | primary concordance (pre-0.7.1) | primary records |
|---|---:|---:|
| wes-5M | 99.9996% | 10,051,170 |
| wgs-5M | 99.9893% | 9,980,872 |
| panel-twist-5M | 99.9414% | 7,913,324 |

Even before restoration, where each read maps was preserved on essentially every read; the
well-under-0.1% of primary records that differed did so in `MAPQ`, `CIGAR`, or position, and
were accounted for by the deterministic tie-break change ([#123](https://github.com/fg-labs/bwa-mem3/pull/123))
and gap-from-`H`/`sizeof` changes now reverted for 0.7.1. (On the 1M-read `smoke-1M` cell the
pre-restoration figure was 99.946%.) Cross-architecture, the NEON (ARM) and x86 builds came
out byte-identical to each other on these cells — 100.0000% concordance over all records,
supplementary alignments included. Read that as scoped to the cells and builds it was measured
on rather than a standing property: a later `panel-twist-5M` run found exactly one record
differing between `c6a` and `c8g` (see [the cross-architecture
exception](#additional-supplementary-alignments-resolved) below).

### Certified adaptive extension band (default, byte-identical)

The default seed-extension path changed *internally* without changing any emitted
alignment-record field (the `@PG` `CL:` command line excepted). `bwa-mem3 mem` now
runs banded Smith-Waterman extension with a **certified
adaptive band**: it scores each pair at a narrow probe band first and finalizes
only the pairs it can *prove* are already optimal there — every other pair falls
through to the exact full-width ceiling ladder and is scored identically to the
non-adaptive path. The proof is per pair (a gapped alignment reaching diagonal
offset `d` costs ≥ `o_min + d·e_min`, bounding its score by
`h0 + min_len·a − o_min − d·e_min`, which can tie-or-beat the achieved score only
for offsets the probe band already covers), anchored on the achieved score minus
the clip penalty so the query-end score, its coordinate, and the clip-vs-extend
decision are band-invariant too — not just the local maximum. The certificate
bounds the optimal *score* but not the extension kernel's early-termination
heuristics (z-drop, all-zero-row break, band-edge shrink), so the certified band
is applied only inside a conservative parameter envelope (a large enough z-drop
relative to the certifiable band, clip penalties below one gap's cost, a matrix
not scoring above the match reward) and **falls back to the exact full-width
ladder outside it** — making the output byte-identical for any
`-d`/`-L`/`-O`/`-E`/`-A`/`-B`. The result is byte-identical to a full-width
extension: measured md5 of the alignment records (`@PG` `CL:` excepted) is unchanged
on a 1M-read WGS slice (HG00096, hg38, Apple Silicon / NEON tier, clang) at
default parameters and across a `-d`/`-L`/`-O`/`-E`/`-A`/`-B` sweep, and the
opt-out **`--no-band-cert`** (which forces the full-width ladder) reproduces the
same md5 exactly. Every comparison uses an identical `-t` on both runs, so the
batch boundaries — and therefore `mem_pestat` and record order — match between
the two streams being compared (`-t`/`-K` change batching and so must be held
equal across a byte-identity comparison). See
[Features → Certified adaptive extension band](features.md#certified-adaptive-extension-band-default-on---no-band-cert-to-disable).
This is distinct from the aggressive, opt-in, *not* byte-identical `--adaptive-band`
narrowing catalogued under [opt-in divergences](#divergences-that-are-latent-opt-in-or-per-architecture).

## What differs

### Additive SAM tags

The most pervasive difference is two **additive** auxiliary tags that bwa-mem3 emits and upstream does not:

- `MQ:i` — mate mapping quality, present on ~100% of bwa-mem3 records and absent from upstream output.
- `HN:i` — total hit count per primary, present on 54,188 of the 64,763 bwa-mem3 records and on 0 upstream records.

A representative record (`SRR34589119.1`) makes the shape of the difference concrete. bwa-mem3 emits:

```text
… MC:Z:53S12M1D6M2D22M58S  MQ:i:19  AS:i:25  XS:i:23  HN:i:9
```

while upstream emits:

```text
… MC:Z:53S12M1D6M2D22M58S  AS:i:25  XS:i:23
```

Same alignment, same scores — two extra tags. Because these tags are inserted into the optional-field area of the record, the line is no longer byte-identical to upstream even though the alignment it describes is. `MQ:i` is one of the lh3/bwa tags ported in [#35](https://github.com/fg-labs/bwa-mem3/pull/35); `HN:i` is added in [#42](https://github.com/fg-labs/bwa-mem3/pull/42). See [Features → `HN:i` hit count tag](features.md) for the full semantics.

Separately, the `@PG` header line reports `ID:bwa-mem3` / `PN:bwa-mem3` rather than `bwa-mem2`, which is also a byte-level header difference by design. It is not the only one: bwa-mem3 emits a default `@HD` line, which bwa-mem2 v2.2.1 has no code to write at all, so the header block differs by more than `@PG` alone ([#288](https://github.com/fg-labs/bwa-mem3/issues/288)). None of the record-level byte-identity results above depend on the header — every one of them compares alignment records only.

### `pa:f:` renders identically in SAM text and BAM (resolved 2026-08-05)

`pa:f:` is not additive — bwa and bwa-mem2 both emit it — but bwa-mem3 has three
writers where they have one, and the three disagreed with each other until
[#366](https://github.com/fg-labs/bwa-mem3/pull/366)
([#365](https://github.com/fg-labs/bwa-mem3/issues/365)):

| writer | value | guard | position |
| --- | --- | --- | --- |
| `mem_aln2sam` (SAM text) | `"%.3f"`, as both upstreams | non-secondary only | after `SA:Z` |
| `mem_aln_to_bam` (`--bam`) | raw unrounded `float` quotient | none | before `SA:Z` |
| `meth_mem_aln_to_bam` (`--meth`) | raw unrounded `float` quotient | none | before `SA:Z` |

So `bwa-mem3 mem --bam` stored `0.806723` where `bwa-mem3 mem \| samtools view -b`
stored `float32("0.807")`, and under `--compat` the BAM path matched neither
upstream. All three now render through one shared definition, so `--bam` is a
container choice for this tag rather than a content change: the stored float32 is
by construction the one the three-decimal SAM token parses to.

**This is an intentional change to `--bam` and `--meth --bam` output.** A consumer
that read `pa` from a pre-0.8.1 `--bam` file sees a different (correctly rounded)
value, no `pa` on secondary records, and `pa` after `SA:Z` instead of before it.
SAM-text output is unchanged.

Verification scope: `test/regression/alt_pa_parity.sh` (canonical CI row — Linux
x86_64, AVX2, mimalloc) generates an ALT-aware fixture whose reads come from the
ALT contig and asserts the decoded float32 is equal record-for-record across the
two containers, at 3,904 records of which 1,904 carry the tag. It fails on the
merge base with 1,356 of those 1,904 differing. Rounding semantics and tag order
are pinned separately in `test/unit/test_bam_pa_tag_parity.cpp`. Nothing here is a
cross-aligner measurement: bwa-mem2 is not run, because the SAM rendering that
both upstreams produce is what the BAM path is being held to.

### Additional supplementary alignments (resolved)

Earlier releases emitted a small number of **additional supplementary (chimeric/split) alignments** that upstream `bwa-mem2` v2.2.1 did not (e.g. on `wes-5M`, 5,123 vs 5,118). This was the previously-unattributed [#127](https://github.com/fg-labs/bwa-mem3/issues/127) finding; it has since been root-caused to the [#123](https://github.com/fg-labs/bwa-mem3/pull/123) total-order dedup sort and **largely reduced by [#257](https://github.com/fg-labs/bwa-mem3/pull/257)**, which cut the extra split alignments from +239 to **+4** on the 5 M WGS cell by restoring bwa-mem2's dedup outcome on the default path. The primary alignment of every affected pair was unchanged throughout; only an extra supplementary record was ever added.

**That +4 is now measured at +0 on a `#268`-carrying build.** Re-measured 2026-07-24 against
bwa-mem2 v2.2.1 on `e722ed0`, comparing the *complete* record stream (all 11 mandatory fields and
every optional tag, byte-for-byte) with only the additive `MQ:i`/`HN:i` removed:

| cell | records | supplementary (bwa-mem2 / bwa-mem3) | result |
|---|---|---|---|
| `wgs-5M` | 10,030,558 | 49,686 / 49,686 | byte-identical |
| `wes-5M` | 10,056,288 | 5,118 / **5,118** | byte-identical |
| `hic-1M` | 2,381,418 | 381,418 / 381,418 | byte-identical |

`hic-1M` is the strongest of the three for this question — it is the cell chosen in
`config/samples.yaml` precisely because chimeric Hi-C ligation products generate heavy
supplementary load, and it shows zero disagreement across 381,418 supplementary records. The
earlier `wes-5M` figure of 5,123 vs 5,118 no longer reproduces; both are 5,118.

Two independent corroborations landed with it. **Cross-architecture:** bwa-mem3 `c6a` (AVX2) and
`c8g` (NEON) are byte-identical on all three cells with *nothing* stripped — `MQ`/`HN` included.
**A third aligner:** `bwa` v0.7.19 was run on `hic-1M` and `wes-5M` (`c6a`, `-t 16`, matching
`mem_flags`) and agrees with both bwa-mem2 and bwa-mem3 over 12.4 M records. That agreement is
byte-identical only *after* the same normalization applied above — alignment records only, since
the `@PG` line names a different program in each of the three outputs, and with the additive
`MQ:i`/`HN:i` stripped from the bwa-mem3 side. The raw SAM streams are **not** byte-identical;
the normalized records are, so the agreement is not an artifact of the shared bwa-mem2 lineage.

`FG-SUPP-ADDITIONS` can be marked resolved for these cells. Scope: three cells on x86, plus the
cross-arch check; it is not a genome-wide guarantee.

**One cross-architecture exception, tracked separately.** On `panel-twist-5M`, `c6a` and `c8g`
differ on exactly **1 of 8,100,270 records** — a phantom `XS` from unzeroed query padding in the
8-bit `kswv` mate-rescue kernels on NEON and AVX-512BW, root-caused in
[#290](https://github.com/fg-labs/bwa-mem3/pull/290), where the fix is still open — so 0.7.1
still carries it. Those two kernels are the entire divergent
set: the AVX2 8-bit kernel and all three 16-bit kernels kept the correct per-cell padding test,
and the SSE tiers have no batched `kswv` kernel at all (mate rescue there runs the scalar
`ksw_align2` path). The split therefore falls between AVX2 and AVX-512BW/NEON, not between x86
and ARM.

(These additions were never a product of the opt-in `--supp-rep-hard-cap` rescoring, which only lowers MAPQ and never adds records.) The `--fast` profile, which keeps the [#123](https://github.com/fg-labs/bwa-mem3/pull/123) sort, still emits them by design.

### Divergences that are latent, opt-in, or per-architecture

The following changes can move alignments, scores, or MAPQ relative to upstream, but did **not** surface as primary-alignment differences on the measured cells because they are gated, latent, or only active on other inputs or architectures:

- **Proper-pair `FLAG` derivation ([#17](https://github.com/fg-labs/bwa-mem3/pull/17), now opt-in via `--proper-pair-from-emitted`).** bwa and bwa-mem2 both derive the `0x2` bit from the top-scoring region `a[0]` even when the record they emit is `a[which]`; `#17` switched bwa-mem3 to `a[which]` so the bit describes the record it rides on. The two differ only when the top region scores below `opt->T` and the first ALT region clears it — which requires the read to have ALT hits, so **without a `.alt` sidecar the branch is structurally unreachable**, not merely rare. This page previously called that latent; it is not. ALT-awareness makes the branch reachable, and on the one cell measured here it did fire: **3,013 of 10,134,006 records** on a 5 M-pair WGS slice (HG00096, GRCh38 with the standard `.alt`), measured on an AWS c6a.4xlarge host (AMD EPYC Milan, x86_64, AVX2 tier), all on decoy (2,377) and ALT/HLA (636) contigs, **none on the primary assembly**. Read that count as this input's, not as a rate: a `.alt` sidecar is what makes the branch reachable, not what guarantees some read reaches it, so another ALT-aware run can move a different number of records or none at all. Because `0x2` is aligner-defined and this made the default path differ from *both* upstreams at once, `#17`'s behavior moved behind `--proper-pair-from-emitted` and the default now matches bwa and bwa-mem2 ([#362](https://github.com/fg-labs/bwa-mem3/issues/362)). The option is a deliberate deviation from both upstreams, so it is **incompatible with `--compat`**: supplying both is a hard error rather than a silently ignored flag — overridable with `--compat-allow-divergent` (unlike `--fast`, which remains a hard error). See [Correctness fixes → Proper-pair flag](correctness.md) and [Features → `--proper-pair-from-emitted`](features.md).
- **SIMD scoring-kernel fixes ([#21](https://github.com/fg-labs/bwa-mem3/pull/21), [#26](https://github.com/fg-labs/bwa-mem3/pull/26), [#28](https://github.com/fg-labs/bwa-mem3/pull/28), [#29](https://github.com/fg-labs/bwa-mem3/pull/29), [#30](https://github.com/fg-labs/bwa-mem3/pull/30), [#31](https://github.com/fg-labs/bwa-mem3/pull/31)).** These correct the batched mate-rescue `kswv` kernels so the suboptimal score (`score2` → `XS:i`/`MAPQ`) converges toward the scalar `ksw_align2` reference. They move `XS`/`MAPQ` on the minority of reads where the SIMD kernel previously diverged, and the affected reads differ by architecture (AVX2 vs NEON vs AVX-512BW). See [Correctness fixes → kswv score2 plateau series](correctness.md).
- **Seeding correctness fixes ([#55](https://github.com/fg-labs/bwa-mem3/pull/55), [#73](https://github.com/fg-labs/bwa-mem3/pull/73), [#100](https://github.com/fg-labs/bwa-mem3/pull/100)).** These fix buffer sizing and a prefetch-mask precedence bug. They change alignments only where the old bug actually triggered (e.g. reads longer than 151 bp for [#55](https://github.com/fg-labs/bwa-mem3/pull/55); [#73](https://github.com/fg-labs/bwa-mem3/pull/73) is a prefetch hint with no semantic change).
- **Opt-in MAPQ rescoring ([#56](https://github.com/fg-labs/bwa-mem3/pull/56), [#101](https://github.com/fg-labs/bwa-mem3/pull/101), [#118](https://github.com/fg-labs/bwa-mem3/pull/118), default-off).** `--supp-rep-hard-cap INT` forces MAPQ=0 on supplementary alignments anchored in repetitive seeds. With no flag the output is unchanged; [#101](https://github.com/fg-labs/bwa-mem3/pull/101) makes the flag actually take effect (it shipped as a silent no-op before), and [#118](https://github.com/fg-labs/bwa-mem3/pull/118) is its regression test. See [Features → `--supp-rep-hard-cap`](features.md).
- **Tie-break determinism ([#123](https://github.com/fg-labs/bwa-mem3/pull/123), reverted on the default path in 0.7.1).** `#123`'s strict total order + pdqsort at the dedup-patch sort sites reordered equal-scoring ties relative to bwa-mem2's partial-order outcome. As of 0.7.1 the default path restores bwa-mem2's comparator + introsort outcome ([#257](https://github.com/fg-labs/bwa-mem3/pull/257)) and recovers pdqsort's speed only on tie-free inputs ([#261](https://github.com/fg-labs/bwa-mem3/pull/261)); `#123`'s behavior now lives behind `--fast`, where it remains **not** byte-identical.
- **Score-gated chain-extension cap and its `MAPQ` seed ([#269](https://github.com/fg-labs/bwa-mem3/pull/269), default on under `--fast`).** Two coupled opt-in levers. `--extend-tie-frac` (with `--extend-tie-floor`) extends a chain to banded-SW only when its weight is within a fraction of the best chain's, so a non-competitive tail chain that the default would have scored is **removed before extension** — which can drop a candidate secondary and move `XS`/secondary/`MAPQ`. `--extend-csub` then feeds the best pruned chain's weight back in as a competitor-score estimate so the pruning does not *inflate* `MAPQ`; this changes emitted `MAPQ` on **both** the scalar single-end path (`mem_approx_mapq_se`) and the batched paired-end pairing path (`mem_capped_pair_subo`). All three options (`--extend-tie-frac`, `--extend-tie-floor`, `--extend-csub`) are off by default, and with them off the gate is a no-op: the alignment records are md5-identical to the baseline build — verified over a 197k-pair panel slice (404,199 records) on both the default path and `--fast` with the gate off, `@PG` and the additive `MQ:i`/`HN:i` tags excluded as elsewhere on this page. `--fast` folds them in — in both non-meth and `--meth` mode — at `--extend-tie-frac 0.95 --extend-tie-floor 1` with `--extend-csub` forced on, where the combination is **not** byte-identical by design. On truth-graded simulations (WGS, exome, panel-targets, and methylation) the gate is placement-neutral — worst-case −0.007 pp correct-placement delta vs the uncapped baseline; the movement is `MAPQ` recalibration (`csub`) and low-confidence secondaries, with the `MAPQ`-error-rate measurements taken on an AWS c6a.4xlarge host (AMD EPYC Milan, x86_64, AVX2 tier). Read those figures as scoped to those workloads and that host/tier, not a guarantee across all inputs and architectures. See [Features → `--extend-tie-frac`](features.md) and [modes → `--fast`](modes.md).

(The resolve→order→chain seeding refactor that backs `--seed-order` is byte-identical in its default `off` mode; it is described in the dedicated section below rather than listed here, since only its non-`off` modes are divergent.)

(The `ksw_global2` global-alignment kernel — banded Gotoh with CIGAR traceback — gained per-architecture anti-diagonal (wavefront) SIMD implementations ([#418](https://github.com/fg-labs/bwa-mem3/pull/418)) for NEON, AVX2, and AVX-512, selected per tier. Each tier carries an int32 kernel plus a narrower int16 kernel with twice the lanes; the int16 kernel runs only where a runtime overflow-safety check proves the whole score range fits int16, and falls back to the int32 kernel or scalar otherwise. All of them are **byte-identical** to the scalar reference: same score, same `n_cigar`, and every CIGAR op — the narrower type is a speed choice, never a semantic one. This is unlike the divergent `kswv` mate-rescue kernel fixes above — the wavefront kernel changes *how* the alignment is computed, never *what* it emits, so it is **not** a divergence and preserves every equivalence claim on this page. It is noted here only so an audit of the SW-kernel files finds the byte-identity trail. The invariant is backed by a written proof (a comment atop `src/ksw_global2_wave.h`), a differential unit test against the scalar path (`test/unit/test_ksw_global2_wave.cpp`, run on every CI tier, exercising both the int16 and int32 kernels), and whole-aligner record-level parity in the CI byte-identity harness. That harness compares `mem` output record-by-record between the scalar and SIMD builds, at a fixed batch size (`-K`, per the *Batch size, `-t`, and why comparisons need `-K`* section below), over its WGS, single-end, and bisulfite read sets at each runtime SIMD tier it exercises (AVX-512BW, AVX2, NEON), and reports zero discordant records on those runs. Those runs were measured on Intel Sapphire Rapids (x86_64) for the AVX-512BW and AVX2 tiers and AWS Graviton4 (arm64) for the NEON tier, built with clang-19. The parity claim is scoped to those measured workloads, tiers, hosts, and batch setting — it is the harness's observation on this input, not an unbounded guarantee.)

### Batch size, `-t`, and why comparisons need `-K`

Any equivalence claim on this page is implicitly a claim **at a fixed batch size**, and the default batch size is `chunk_size × -t`. That is inherited: `bwa`, `bwa-mem2` and bwa-mem3 all compute it the same way. It matters because batch boundaries are not purely a scheduling concern — `mem_pestat` estimates the paired-end insert-size distribution from whatever reads happen to land in a batch, and those percentile bounds feed pairing, mate rescue and MAPQ. Change the partition and a small number of records can change with it.

Two consequences:

- **`bwa-mem3 -t N` must be compared against `bwa-mem2 -t N`**, not against `bwa-mem2 -t M`. Comparing across thread counts measures the batching, not the aligner.
- **Pass `-K` to both sides** if you want the comparison to be independent of `-t` altogether. `-K` pins the batch target for reproducible partitioning, is never capped, and is the reason it exists in all three tools. (It is a target, not a hard delivered-base limit — the reader finishes the record it is on — but all three tools round it the same way, so the partition matches.)

`--chunk-cap` (off by default) bounds the auto-scaled batch and therefore re-partitions the input; it is opt-in for exactly this reason, warns on stderr when it engages, and is implied by `--fast`. A run with `--chunk-cap` in effect is **not** byte-identical to `bwa`/`bwa-mem2` at the same `-t`. See [`mem` → `--chunk-cap`](../cli/mem.md) and [Aligning → `-K`](../user-guide/aligning.md).

### Seed ordering (`--seed-order`, opt-in)

`--seed-order off` (the default) preserves byte-identical output. The internal resolve→order→chain refactor that enables it was verified to be bit-for-bit identical to the previous code path across single-end, paired-end, and threaded runs.

`--seed-order local-longest` (and the unadvertised advanced modes `global-longest`, `absorb-count`, `most-absorb`) are **opt-in** and are **not byte-identical**. They reorder each read's SA-resolved seeds before chaining so that the longest seeds anchor chains first; contained shorter seeds are then absorbed rather than extended. The output shift is non-trivial: secondary alignments, `XA:Z:`, `XS:i`, and `HN:i` can all change, and a small number of primary alignments may shift as well.

Accuracy on an easy simulated profile (holodeck, ~94.4 % F1) is flat relative to `off`. Hard-data F1 validation on divergent/indel-rich reads and GIAB benchmarks is **not yet complete**. Because the accuracy gate is not fully passed, all non-`off` modes remain opt-in only; the default stays `off`. See [Optimization checklist → Reorder seeds longest-first](../best-practices/optimization-checklist.md#6-reorder-seeds-longest-first---seed-order-local-longest) for usage guidance.

## Declared divergence catalog

The divergences described above are tracked as a structured registry in [bwa-mem3-bench](https://github.com/fg-labs/bwa-mem3-bench) (`docs/expected-divergences.yaml`). Each carries a per-sample concordance-drift budget that the benchmark gates against on every run — a new bwa-mem3 build that drifts beyond its budget fails CI rather than silently shipping a regression. The table below is generated from that registry; do not edit it by hand (see [bwa-mem3-bench → Per-release concordance history](../related-projects/bwa-mem3-bench.md) for how it is regenerated).

> **⚠️ This catalog predates the 0.7.1 parity restoration and is pending regeneration.**
> Its `FG-PRIMARY-DRIFT` row still attributes primary drift to [#123](https://github.com/fg-labs/bwa-mem3/pull/123)
> and to per-architecture SIMD `score2`/MAPQ convergence. That second mechanism did not surface
> on the three cells re-measured for 0.7.1 — NEON and x86 came out byte-identical there — but it
> is not hypothetical: `panel-twist-5M` turned up one record where the AVX-512BW and NEON 8-bit
> `kswv` kernels produced a phantom `XS` that AVX2 did not
> ([#290](https://github.com/fg-labs/bwa-mem3/pull/290), fix still open). On the default path the
> `#123` drift is **resolved in 0.7.1**
> ([#256](https://github.com/fg-labs/bwa-mem3/pull/256), [#257](https://github.com/fg-labs/bwa-mem3/pull/257),
> [#261](https://github.com/fg-labs/bwa-mem3/pull/261), [#268](https://github.com/fg-labs/bwa-mem3/pull/268))
> on the slice re-measured so far. `FG-SUPP-ADDITIONS` tracks the [#127](https://github.com/fg-labs/bwa-mem3/issues/127)
> extra supplementaries; its row below has been **hand-updated ahead of regeneration** to carry
> the [+0 result](#additional-supplementary-alignments-resolved) measured on a
> [#268](https://github.com/fg-labs/bwa-mem3/pull/268)-carrying build, which is why it reads
> `RESOLVED` where the registry still says `TBD`. That resolution is scoped to the three
> measured cells (`wgs-5M`, `wes-5M`, `hic-1M`, on x86); the remaining samples have not been
> re-measured. Its `samples` and `budget_%` columns are inherited unchanged from the
> un-regenerated registry, where `samples = all` and `budget_% = 0.0000` mean this class
> contributes nothing to the *primary*-concordance drift budget — they are not a validated
> zero-tolerance gate across every sample.
> The registry in `bwa-mem3-bench` (`docs/expected-divergences.yaml`, which still carries
> `pr: TBD`, `samples: []`, and `expected_drift_pct: 0.0` for this entry) must be updated and
> re-run against a 0.7.1 build and its budgets tightened. Until that lands, only the
> `FG-SUPP-ADDITIONS` row's `pr` and `summary` cells reflect current behavior — they are the
> hand-update. Read the rest of the table as the pre-0.7.1 state: the other two rows in full,
> and that row's `samples` and `budget_%` columns.

<!-- FG-DIVERGENCE-CATALOG:start -->
| id | pr | affected | samples | budget_% | summary |
| --- | --- | --- | --- | --- | --- |
| FG-PRIMARY-DRIFT | fg-labs/bwa-mem3#123 | primary_alignment | wgs-5M, wes-5M, panel-twist-5M, smoke-1M | 0.1000 | Per-architecture SIMD score2/MAPQ convergence (#21, #26, #28-#31) and deterministic tie-break ordering (#123) shift MAPQ, CIGAR, or position on a small fraction of primary alignments relative to bwa-mem2 v2.2.1. Where each read maps is preserved; the affected reads differ in placement detail, and the set varies by SIMD architecture. |
| FG-METH-DIVERGENCE | fg-labs/bwa-mem3#90 | meth_alignment | meth-twist-emseq-5M, smoke-meth | 1.5000 | Bisulfite (--meth) mode against the bwameth.py baseline diverges beyond the ignored YD/XM/XG tag set (Bismark-compatible XR/XG/XM tags and C->T/G->A conversion handling), giving a larger but still-bounded concordance drift on methylation workloads. |
| FG-SUPP-ADDITIONS | RESOLVED | supplementary_alignment | all | 0.0000 | Earlier builds emitted additional supplementary (split/chimeric) alignments vs bwa-mem2 v2.2.1 (e.g. wes-5M: 5123 vs 5118). Re-measured 2026-07-24 on `e722ed0`: **+0** on wgs-5M, wes-5M and hic-1M (x86; the other samples have not been re-measured), with the complete alignment-record stream byte-identical (MQ/HN stripped; header lines not compared) across 22.5 M records incl. 436 k supplementary. Primary alignments were unchanged throughout. |
<!-- FG-DIVERGENCE-CATALOG:end -->

## Per-PR audit trail

The declared divergence catalog above calls out the fork changes that actually
affect output. For the fork changes that were *proposed back upstream* — and
where each stands — see [Fork changes vs. upstream](../reference/pr-catalog.md);
the exhaustive per-PR change list lives in git history and the GitHub PR list, not
in a hand-maintained table.

---

**See also:**
[Overview](overview.md) ·
[Correctness fixes](correctness.md) ·
[Performance improvements](performance.md) ·
[Features](features.md) ·
[Fork changes vs. upstream](../reference/pr-catalog.md)
