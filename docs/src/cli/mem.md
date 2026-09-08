# mem

`bwa-mem3 mem` aligns short DNA reads against an indexed reference genome
using the BWA-MEM algorithm. It accepts one or two FASTQ files (single-end or
paired-end) and writes alignments to stdout in SAM or BAM format. It is the
primary alignment subcommand; nearly all bwa-mem3 usage flows through it.

## Synopsis

```text
{{#include ../../_generated/cli/mem.txt}}
```

## Common usage

Paired-end alignment, 16 threads, SAM to stdout:

```bash
bwa-mem3 mem -t 16 ref.fa R1.fq.gz R2.fq.gz > out.sam
```

Paired-end alignment, emit uncompressed BAM, pipe directly to `samtools sort`:

```bash
bwa-mem3 mem --bam -t 16 ref.fa R1.fq.gz R2.fq.gz \
  | samtools sort -@ 8 -o out.bam -
samtools index out.bam
```

Paired-end methylation alignment with a read group header:

```bash
bwa-mem3 mem --meth -t 16 \
  -R '@RG\tID:lib1\tSM:sample1\tPL:ILLUMINA' \
  ref.fa R1.fq.gz R2.fq.gz \
  | samtools sort -o out.bam -
```

## Flag reference

### Input / output

#### `-o STR` — output file

Write output to `STR` instead of stdout. Honored for both SAM and `--bam`
output; the path is opened lazily so BAM mode can hand it to htslib instead of
truncating it as a SAM-text file. Stdout redirection (`>`) remains an
alternative.

#### `--bam[=N]` — emit BAM

Emit BAM instead of SAM. `N` controls BGZF compression: `0` (default when
`--bam` is used without `=`) writes uncompressed BAM, which costs almost no
CPU and is the recommended mode for piping to `samtools sort`. Values `1`–`9`
select increasing BGZF deflate levels; use `--bam=6` or `--bam=9` only when
writing directly to final storage without a downstream sort step.

Any level above `0` prints a one-line warning to stderr: BGZF deflate runs on
the single writer thread, so for large outputs that serial compression — not
alignment — is usually what caps throughput. The warning reflects the resolved
level, so it is emitted once and not at all when a later `--bam=0` overrides an
earlier compressed level.

> **Tip — Prefer --bam for production pipelines**
>
> Uncompressed BAM (`--bam` or `--bam=0`) eliminates the text-formatting cost on
> the aligner side and the text-parse cost on the `samtools sort` side. For any
> pipeline that immediately sorts or processes the output, this is faster than
> SAM at no quality cost.

#### `--compat=TARGET` — byte-identical output for another aligner

Shapes bwa-mem3's output so it is byte-for-byte identical to another aligner's.
`--compat` takes a required target:

| target | meaning |
|---|---|
| `bwa-mem2` (alias `mem2`) | match bwa-mem2 v2.2.1 |
| `bwa-mem` | match bwa 0.7.19 |
| `off` | bwa-mem3's native output — identical to omitting the flag |

```bash
bwa-mem3 mem --compat=bwa-mem2 -t 16 ref.fa R1.fq R2.fq > out.sam
bwa-mem3 mem --compat bwa-mem2 -t 16 ref.fa R1.fq R2.fq > out.sam   # same
bwa-mem3 mem --compat=bwa-mem  -t 16 ref.fa R1.fq R2.fq > out.sam
```

**`--compat` takes a target rather than being a boolean because the two
upstreams disagree with each other** on half of what it shapes. bwa gained a
default `@HD` ([lh3/bwa#336](https://github.com/lh3/bwa/pull/336), merged 2022-03-06,
shipped in 0.7.18 as `6b18630`) and the `MQ:i` tag
([lh3/bwa#330](https://github.com/lh3/bwa/pull/330), merged 2022-03-06) *after*
bwa-mem2 forked at 0.7.17, so matching one means emitting exactly what matching
the other means suppressing:

| | `off` (default) | `--compat=bwa-mem2` | `--compat=bwa-mem` |
|---|---|---|---|
| `MQ:i` mate MAPQ | emitted | **suppressed** | emitted |
| `HN:i` hit count | emitted | **suppressed** | **suppressed** |
| default `@HD` | emitted | **suppressed** | emitted |
| `.hdr`/`.dict` sidecar `@SQ` | honored | **ignored** | **ignored** |
| `@PG` | `ID:bwa-mem3` | `ID:bwa-mem3` | `ID:bwa-mem3` |
| all chains dropped by the weight filter (`-W`, `-x pacbio`/`pbref`/`ont2d`) | read aligned | read aligned | **read unmapped** |

Only **`HN:i`** and the sidecar are genuinely bwa-mem3 inventions, which is why
both targets drop them. The rest is the fork point.

- **`HN:i`** (hit count) is emitted by neither upstream, so both targets
  suppress it.
- **The `<prefix>.hdr` / `<baseprefix>.dict` sidecar is not read** under either
  target, so `@SQ` is generated from the index as bare `SN`/`LN` (plus `AH:*` on
  ALT contigs, which both upstreams emit). The sidecar is a bwa-mem3-only
  feature — a port of [lh3/bwa#348](https://github.com/lh3/bwa/pull/348), which
  upstream closed unmerged — so its `M5`/`AS`/`UR`/`SP` have no counterpart to
  match.
- **`MQ:i` and the default `@HD`** are suppressed only under
  `--compat=bwa-mem2`. Under `--compat=bwa-mem` both are emitted, and the `@HD`
  string bwa-mem3 writes is byte-identical to bwa's (`bwa.c:426`), so no
  reshaping is needed. A user `@HD` from `-H` is honored under both.

**`bwa-mem` means bwa 0.7.19 specifically**, and the pin is load-bearing: a
0.7.17 target would be this row's opposite on `MQ:i` and `@HD` — and therefore
identical to `bwa-mem2` except for the `@PG` `ID`. If you are on bwa 0.7.17,
`--compat=bwa-mem2` is the target you want.

`--compat` is **output-shaping** — with one exception, it changes no alignment,
no score, no flag, and no tag's value. It exists so pipelines validating
bwa-mem3 against a bwa-mem2 golden can diff raw output without a
post-processing step. The equivalence is a *versioned behavioral target*, not an
unconditional byte-identity guarantee across hosts or SIMD tiers: each `--compat`
value tracks a specific pinned upstream (see the bwa 0.7.19 pin above), and the
raw diff holds for the same invocation on both binaries with `@PG` excluded, as
the verification below shows.

The exception is the last row above, and it exists because the rule and the
purpose collide there. When the chain weight filter drops *every* chain for a
read, bwa reports zero survivors and the read goes out unmapped, while
bwa-mem2 hands back the chain it just rejected and aligns the read
([#310](https://github.com/fg-labs/bwa-mem3/issues/310) for background;
modeled here by [#374](https://github.com/fg-labs/bwa-mem3/pull/374)). Those are different
alignments for the same input, so a target that declined to model the
difference would reproduce neither upstream faithfully — and
`--compat=bwa-mem` returning bwa-mem2's alignments is not a weaker guarantee,
it is a false one. The default path keeps bwa-mem2's behavior, so the drop-in
profile is unchanged.

This is reachable only when `min_chain_weight > 0`, which is never the default:
`-W`, or the `-x pacbio`/`pbref`/`ont2d` presets. In practice `-x pacbio`
almost never reaches it — instrumenting the empty-array case over 500 real HiFi
reads at `-x pacbio` counted zero reads with every chain dropped, because real
long reads build chains far above the threshold. That is a deterministic count
of a code path firing, not a timed measurement, so it does not vary by host,
architecture, or SIMD tier; `pbref` and `ont2d` share the same chain-weight
mechanism but were not separately measured. A `-W` above the read length reaches
it on every read.

Verify it with:

```bash
# records
diff <(samtools view out.compat.bam) <(samtools view out.mem2.bam)
# header, @PG excluded on both sides (see below)
diff <(samtools view -H --no-PG out.compat.bam | grep -v '^@PG') \
     <(samtools view -H --no-PG out.mem2.bam   | grep -v '^@PG')
```

Notes and caveats:

- **`@PG` is still emitted, by design.** bwa-mem2 writes its own `@PG`, so
  suppressing ours would turn a *changed* line into a *missing* one — the diff is
  non-empty either way, and suppression would only cost the record of what
  actually produced the file. `@PG` is unmatchable by construction regardless:
  `CL:` embeds the invocation and its paths, so even two bwa-mem2 runs from
  different directories differ. Exclude `@PG` on both sides when comparing.
- **Mutually exclusive with `--fast`.** Passing both is a hard error
  (`--compat and --fast are mutually exclusive`), because `--fast` deliberately
  *changes alignments* while `--compat` only shapes output — so `--fast
  --compat=bwa-mem2` would produce a diff-clean-*looking* stream over genuinely
  different alignments, defeating the parity-validation purpose. `--compat` is
  for the drop-in profile. (`--compat=off --fast` is fine: `off` selects no
  target.)
- **Mutually exclusive with [`--proper-pair-from-emitted`](#--proper-pair-from-emitted--derive-flag-0x2-from-the-emitted-alignment).**
  Passing both is a hard error (`--compat and --proper-pair-from-emitted are
  mutually exclusive`), for the same reason as `--fast`: that flag derives
  `FLAG` `0x2` from the emitted alignment, which neither target does, so the
  pair asks for byte-identity and for a documented deviation from it in one
  command. (`--compat=off --proper-pair-from-emitted` is fine.)
- **An `@HD` in `-H` warns but is allowed.** bwa-mem3 hoists a *leading* user
  `@HD` above the `@SQ` block, so the header is spec-valid (`@HD` must come
  first). Neither target does that — bwa emits `-H` records after `@SQ` and
  bwa-mem2 has no `@HD` handling at all — so the header differs from the target
  in **line order**. Records are unaffected.

  This is not rejected, unlike `--fast`, `--proper-pair-from-emitted` and
  `--meth`, because it is an explicit and coherent request: *give me a valid SAM
  header, everything else the same*. Those three change what the records say —
  `--fast` silently moves alignments, `--proper-pair-from-emitted` moves `FLAG`
  `0x2`, and `--meth` is a different mode — and a user cannot see any of them in
  their own command line. An `@HD` they typed, they can.

  Only a *leading* `@HD` diverges; a later one is emitted inline after `@SQ`
  exactly as upstream does, and does not warn. Every other `-H` record (`@RG`,
  `@CO`, `@PG`, `@SQ`) and all of `-R` compose with `--compat` normally.
- **Non-`--meth` only; combining them is a hard error**
  (`--compat is not supported with --meth`). Neither target has a bisulfite
  mode, so byte-identity is undefined under `--meth`, which also emits
  methylation-specific tags that no target models.
- **The two targets differ on every mated record, by design.** This is worth
  stating because "both upstreams are the same aligner" invites the opposite
  assumption. Measured on a 4.06 M-pair GRCh37 run, bwa 0.7.19 and bwa-mem2
  v2.2.1 differ on **all 8,116,326 records** — and on **0** once `MQ:i` is
  stripped. So the two targets are not interchangeable today, and picking the
  wrong one produces a diff on essentially every record with a mapped mate.
- **Evidence.** `--compat=bwa-mem` is validated at **322,978,938 alignment
  records across two reference builds** (GATK hg38 and hs37d5), every record
  byte-identical to a real `bwa` 0.7.19 run; `--compat=bwa-mem2` at 22.5 M
  records plus full header byte-identity on both the SAM-text and `--bam` paths.
  Both also carry a phiX-scale end-to-end check in the regression suite. The
  datasets, hosts, architectures and SIMD tiers behind those totals — and the
  one path they leave untested — are itemized in [Equivalence → Byte-identical
  output](../whats-different/equivalence.md#byte-identical-output---compat),
  which is the single scope of record for both targets. Read it before treating
  either number as a general guarantee.
- **Only one target can be byte-identical on a record where the upstreams
  themselves disagree.** `--compat` shapes output and never moves an alignment,
  so if bwa and bwa-mem2 ever emit different alignment fields for the same read,
  bwa-mem3 matches at most one of them. No such record has been observed, and an
  audit of bwa 0.7.17…0.7.19 finds no change to seeding, chaining, extension,
  pairing, MAPQ or dedup — but the tail is not proven empty.

See [Equivalence with bwa-mem2 → Byte-identical output
(`--compat`)](../whats-different/equivalence.md#byte-identical-output---compat).

#### `-R STR` — read group header

Injects a `@RG` header line and tags every alignment with `RG:Z:<ID>`. The
value is a tab-separated `@RG` line with literal `\t` escapes, for example:

```bash
-R '@RG\tID:run1\tSM:HG001\tPL:ILLUMINA\tLB:lib1'
```

bwa-mem3 escapes any literal tab characters inside `-R` values before writing
them to the `@PG CL:` field, preventing header corruption (fix for issue #45).

#### `-H STR/FILE` — extra header lines

If `STR` begins with `@`, it is injected verbatim as a header line. Otherwise
`STR` is treated as a path and every line in the file is injected. Useful for
adding `@CO` comments or custom `@RG` / `@PG` entries.

#### `-p` — smart pairing

Reads interleaved paired-end data from a single FASTQ file (`in1.fq`) rather
than two separate files. The second positional argument (`in2.fq`) is ignored.

#### `-5` — leftmost-coordinate primary

For split alignments, designates the alignment with the smallest genomic
coordinate as primary, rather than the longest alignment. Useful for some
downstream tools that expect the leftmost alignment to be primary.

#### `-q` — preserve supplementary MAPQ

By default, bwa-mem3 may downgrade the MAPQ of supplementary alignments.
`-q` suppresses that adjustment.

#### `-K INT` — fixed batch size

Sets the per-batch **target** to `INT` input bases regardless of the number of
threads. **Recommended whenever reproducibility or bwa/bwa-mem2 comparison
matters.**

`INT` is a target rather than a hard delivered-base limit: the reader stops at
the first whole record on or past `INT`, so a batch delivers slightly more than
`INT` bases (never a partial record, and never a split read pair). This is the
same behaviour as `bwa` and `bwa-mem2`, and it is what makes the batch boundary
reproducible — for a given input and `INT`, the boundary always falls on the same
record.

Without `-K` the batch size is `chunk_size × -t` (10,000,000 × `-t` by default),
so it moves with the thread count. That is not merely a scheduling detail:
`mem_pestat` estimates the paired-end insert-size distribution from whichever
reads land in a batch, and the resulting bounds feed pairing, mate rescue and
MAPQ. Two runs over the same input at different `-t` can therefore disagree on a
small number of records. `bwa` and `bwa-mem2` compute the batch size with the
same formula and have the same property — `-K` exists in all three to pin it.

Fix `-K` to the same value on both sides and the output becomes independent of
`-t`. `-K` always wins: it is never overridden by `--chunk-cap` or `--fast`.

#### `--chunk-cap INT` — upper bound on the auto-scaled batch size

Caps the default `chunk_size × -t` batch **target** at `INT` bases. As with `-K`,
`INT` bounds the target the reader aims for, not the bases it delivers — the
final record is always read whole. `0` (the default) disables the cap, so out of
the box bwa-mem3 batches **exactly** like `bwa` and `bwa-mem2` at any `-t`.

Capping keeps the read/compute/write pipeline overlapped at very high `-t`,
where a single batch would otherwise be enormous (10M × 192 ≈ 1.9 Gbase) and the
input would be only three or four batches — the first batch's read and the last
batch's write then overlap nothing. On a 64-core host that fill/drain cost was
measured at roughly 1.6 s of a 26 s alignment.

> **Note:** that 1.6 s / 26 s figure predates the read-arena lifetime rework and
> has not been re-measured against the current implementation. Treat it as an
> indication of the effect's scale, not as a current number.

It is **opt-in because it re-partitions the input**, and by the `mem_pestat`
mechanism described under `-K` that changes output. A capped run is not
byte-identical to `bwa`/`bwa-mem2` at the same `-t`, and bwa-mem3 prints a
warning to stderr when the cap actually engages. `--fast` implies
`--chunk-cap 256000000` (`--fast` already does not promise byte-identical
output). If you want a bounded batch *and* reproducibility, use `-K` instead.

The value must be a plain non-negative base count — no `K`/`M`/`G` suffix.
`--chunk-cap 100M` is rejected with an error rather than accepted and silently
read as `0` (that is, as no cap at all).

For cap sweeps, `BWA_MEM3_CHUNK_CAP` overrides all of the above, including
`--fast`'s implied cap; setting it to `0` switches an explicit `--chunk-cap`
back off. An empty value is ignored, and a malformed one is reported on stderr
and ignored rather than being read as `0`. `-K` still wins over it — a fixed
batch size is never capped.

Precedence, highest first:

| Setting | Effect |
|---|---|
| `-K INT` | pins the batch target; never capped |
| `BWA_MEM3_CHUNK_CAP` | overrides the cap from either source below (`0` = off) |
| `--chunk-cap INT` | caps the auto-scaled batch (`0` = off) |
| `--fast` | implies `--chunk-cap 256000000` |
| *(default)* | no cap — identical batching to `bwa`/`bwa-mem2` |

#### `-v INT` — verbosity

Controls stderr diagnostic output: `1` = errors only, `2` = warnings,
`3` = informational messages (default), `4+` = debugging.

#### `-a` — all alignments

Output all alignments for single-end or unpaired paired-end reads, including
secondary alignments. Equivalent to enabling secondary-alignment reporting.

#### `-C` — append FASTA/FASTQ comment

Appends the comment field from the FASTA/FASTQ header to the SAM output as
an additional column. Useful when the comment carries barcodes or UMIs.

#### `-V` — reference header in XR tag

Emits the reference FASTA header line for each alignment position as an `XR`
SAM tag.

Under `--meth`, `XR:Z` instead carries the Bismark read-conversion direction
(`CT`/`GA`) and this reference-annotation use of `XR` is suppressed — see
[Methylation Reference → Flags](../methylation/flags.md).

#### `-Y` — soft-clip supplementary alignments

Uses soft clipping instead of hard clipping for supplementary alignments.
Some downstream tools require this.

#### `-M` — mark shorter split hits as secondary

Marks the shorter alignment in a split read as secondary (sets `0x100` flag)
rather than supplementary. Required for compatibility with tools that do not
handle supplementary alignments (e.g. Picard's duplicate-marking before
certain versions).

#### `-j` — treat ALT contigs as primary

Treats ALT contigs as part of the primary assembly by ignoring the
`<idxbase>.alt` file. Use when your workflow does not include ALT-aware
postprocessing.

### Scoring

All scoring flags accept integer values. Changing `-A` (match score) scales
the penalty flags that default to multiples of `-A`; explicit overrides of
individual flags are unaffected.

| Flag | Default | Meaning |
|------|---------|---------|
| `-A INT` | 1 | Score for a sequence match. Scales `-T`, `-d`, `-B`, `-O`, `-E`, `-L`, `-U` unless overridden. |
| `-B INT` | 4 | Mismatch penalty. |
| `-O INT[,INT]` | 6,6 | Gap open penalty for deletions and insertions respectively. |
| `-E INT[,INT]` | 1,1 | Gap extension penalty per base. A gap of length k costs `-O + -E * k`. |
| `-L INT[,INT]` | 5,5 | Clipping penalty for 5' and 3' ends. |
| `-U INT` | 17 | Penalty for an unpaired read pair (affects mate-rescue scoring). |
| `-T INT` | 30 | Minimum alignment score to output. Alignments below this threshold are not reported. |

> **Note — --meth overrides scoring defaults**
>
> When `--meth` is active, bwa-mem3 applies `-L 10 -U 100 -T 40 -M -C` plus a
> mode-dependent mismatch penalty: `-B 2` for `--meth-scoring collapsed` (the
> `--meth`/`--meth=emseq` default, bwameth-compatible) and `-B 4` for
> `--meth-scoring genomic` and `neutral` (`neutral` is the `--meth=taps` default).
> This mirrors bwameth's `bwa mem -T 40 -B 2 -L 10 -CM` (with `-U 100` for
> paired-end). The scoring values (`-B`, `-L`, `-U`, `-T`) can still be overridden
> by passing the flag explicitly, in any position relative to `--meth`; `-M` and
> `-C` cannot, since bwa has no option that unsets them.
>
> Those constants are quoted at bwameth's match score (`-A 1`) and scale with
> `-A` like every other score-derived default above: under `-A 2` the effective
> values are `-L 20 -U 200 -T 80` and `-B 4` (collapsed) / `-B 8` (genomic and
> neutral).

### Paired-end

#### `-I FLOAT[,FLOAT[,INT[,INT]]]` — insert size distribution

Specifies the mean, standard deviation (default: 10% of mean), maximum
(default: 4 sigma above mean), and minimum of the insert size distribution for
FR-orientation paired-end reads. By default bwa-mem3 infers these parameters
from the first batch of reads. Provide them explicitly for speed or when the
reference is short and inference may be inaccurate.

> **`-I` replaces inference; it does not seed it.** Only the FR distribution is
> set, and the estimation pass is skipped, so FF/RF/RR are left without one for
> the whole run — inference would have built a distribution for every
> orientation clearing its thresholds. Pairs in an orientation without one are
> never flagged proper (`0x2`), get no mate rescue, and are skipped when scoring
> the best pair. Nothing is *filtered* by orientation, but the run is not
> otherwise unchanged either: the FR values you supply replace the inferred
> ones, which shifts proper-pair flags, rescue windows and MAPQ on their own.
> The non-FR gap matters most where such an orientation is genuinely populated
> (Hi-C, mate-pair). bwa and bwa-mem2 handle `-I` the same way; `--compat` does
> not affect it.

#### `-m INT` — mate rescue rounds

Maximum number of mate-rescue attempts per read. Reduce to speed up alignment
on data where the default (50) wastes time on unrescuable pairs. See
[Settings profiles](../best-practices/settings-profiles.md) for the benchmarked
`-m 10` recommendation.

#### `--rescue-kmer[=K]` / `--rescue-band INT` — banded mate rescue (opt-in, not byte-identical)

Mate rescue Smith-Watermans a read against a reference window sized by the insert
distribution — on the order of a kilobase for a ~150 bp read — and that unbanded
window is the single largest kernel in the aligner on paired WGS. `--rescue-kmer`
finds the read's dominant K-mer exact-match diagonal within the window and bands the
rescue SW to that diagonal ± `--rescue-band` (default 50 bp), falling back to the full
window when no anchor is found — or when the banded window would come out shorter than
the read, which cannot host a full-length rescue alignment. The `kswv` kernel is
unchanged — it is simply handed a shorter reference window — so the speedup comes from
doing far less DP.

The anchor index keeps one query position per K-mer, so each window K-mer casts a single
diagonal vote. Chaining every occurrence instead would never miss a vote on the true
diagonal, but costs a walk whose length grows as `read_length / alphabet^K` — which is what
makes small K slow. Measured on a 644k-pair simulated bisulfite set (chr20, 150 bp, `K=6`,
`-t 16`) on a 16-vCPU Graviton4 host (arm64, NEON tier), single-position indexing narrows
98.29 % of scans against chaining's 98.84 %, at equal-or-better accuracy (recall at MAPQ ≥ 60
1204271 vs 1204266 of 1288884 primaries; one confident mismap either way) and ~1 % less wall
time — 13.46 s against 13.58 s.

`K` ranges over `1…16` — a K-mer code has to fit the `uint32` the anchor index packs it
into, so larger values are rejected rather than silently treated as `K=16`. Bare
`--rescue-kmer` selects `K=6`. Smaller K makes the anchor scan degenerate and slow (the
diagonal vote's hash chains grow as `read_length / alphabet^K`); larger K lowers the
fraction of rescues that find an anchor. K=6 is the measured wall-time peak (~−22% on the
rescue-heavy WGS tail). `--rescue-band` takes `1…1000000` bp; the default 50 bp band means
the narrowed window is the read length plus 100 bp, against the ~kilobase full window.
Under `--meth` the anchor scan collapses to three letters on each pair's bisulfite strand
(C→T for OT, G→A for OB) so exact K-mers survive the conversion; the SW still scores the
real bases with the methylation matrix.

The value must be attached with `=`: `--rescue-kmer` takes an *optional* argument, so
`--rescue-kmer 6` selects the bare default and leaves `6` as a positional argument. Write
`--rescue-kmer=6`.

**Off by default and NOT byte-identical.** A narrowed window computes a different
suboptimal alignment score, so a rescued read's `csub` — and therefore its MAPQ — can
change; ~0.2% of records move. Truth-based ROC (holodeck simulated hg38 WGS, and
bisulfite chr20) shows accuracy neutral-to-positive and confident (MAPQ ≥ 60) mismaps
unchanged across `K = 1…16`. Enabled by [`--fast`](#--fast--speed-preset-opt-in-not-byte-identical);
use `--rescue-kmer=0` to force it off even under `--fast`.

#### `--rescue-skip` — drop hopeless mate rescues (opt-in, not byte-identical)

`--rescue-kmer` alone always runs the rescue SW: a window with no usable anchor simply
falls back to the full width. `--rescue-skip` instead declines the rescue outright when
the best diagonal clears neither an absolute vote floor nor a fraction of the distinct
K-mers the anchor index actually holds for the query. That index stops inserting at 768
distinct codes, so the denominator is the read's full distinct-K-mer count only up to that
point — which at short-read lengths it always is, since a 150 bp read yields only ~145
6-mers in total. Removing SWs is a larger saving than shortening them.

`--rescue-skip` takes an *optional* `true|false` argument: bare `--rescue-skip` (or
`--rescue-skip=true`) enables it, and `--rescue-skip=false` is the explicit opt-out. As with
`--rescue-kmer`, the value must be attached with `=` — `--rescue-skip false` leaves `false` as a
positional argument (reported with a diagnostic pointing at the `=` form, not silently enabling the
skip). Only `true|false` are accepted.

When enabled, requires `--rescue-kmer` — the decision reuses the same anchor scan, so
`--rescue-skip` (or `--rescue-skip=true`) with `--rescue-kmer=0` is rejected rather than silently
ignored. `--rescue-skip=false` needs no anchor scan, so it carries no such requirement.

**This changes which reads map, not just their MAPQ, and it costs recall.** A read the
full-window SW would have rescued can now stay unmapped. Measured against `--rescue-kmer`
alone on real bisulfite data, on the same 16-vCPU Graviton4 host (arm64, NEON tier): 16 lost
alignments at MAPQ ≥ 30 per 200k primaries on 125 bp WGBS, and 420 per 2M on 75 bp em-seq.
Which reads are lost is tier-dependent in principle, since the rescue SW is one of the
per-ISA kernels — these are the NEON numbers.

The cost scales with read length, because the vote floor does not. A 150 bp read offers
~145 6-mers and a 75 bp read ~70, so the same 10-vote floor is twice as harsh a test on
short reads — skipping 14 % of scans at 150 bp, 42 % at 125 bp and 79 % at 75 bp. Treat it
as a speed/recall trade to evaluate on your own read length, not a free win.

**Not part of [`--fast`](#--fast--speed-preset-opt-in-not-byte-identical)**, which is a
characterized speed preset rather than a recall trade. Combining them explicitly
(`--fast --rescue-skip`) is supported, and the `--fast` audit line then names it.

#### `-S` — skip mate rescue

Disables mate rescue entirely. Faster but may reduce sensitivity for
discordant pairs.

#### `-P` — skip pairing

Skips the pairing step; mate rescue still runs unless `-S` is also given.
Mate rescue is performed *before* the pairing step, so `-P` on its own leaves
the full rescue Smith-Waterman in place. On Hi-C, where the goal is to skip
rescue too, use `--hic` (or spell out `-5SP`).

#### `--hic` — Hi-C preset

Equivalent to `-5SP`, and nothing more: leftmost-coordinate primary, skip
mate rescue, skip pairing. `-5SP` keeps working unchanged; this is an alias
for the invocation Arima, Juicer and Omni-C pipelines already pass, given a
name so it is greppable in a pipeline and self-documenting in `--help`.

Note that minibwa spells its equivalent `--hic` as `-5P`, because it gates
mate rescue inside the pairing step rather than before it: its
[`map-main.c`](https://github.com/lh3/minibwa/blob/master/map-main.c) maps
`--hic` to `-5P` (setting `PRIMARY5 | NO_PAIRING`, no separate no-rescue bit),
and mate rescue runs only inside the pairing routine
([`pe.c`](https://github.com/lh3/minibwa/blob/master/pe.c), reached from the
`!NO_PAIRING` block in
[`map-algo.c`](https://github.com/lh3/minibwa/blob/master/map-algo.c)), so `-P`
skips it wholesale and no `-S` is needed. The letters
differ between the two tools; `--hic` means the same thing in both, which is
the point of having it.

Not to be confused with
[`--rescue-skip`](#--rescue-skip--drop-hopeless-mate-rescues-opt-in-not-byte-identical),
which is not a Hi-C
preset: it leaves the rescue stage, pairing and insert-size estimation all
enabled and only drops the individual rescue attempts whose k-mer anchor fails
a vote floor. Hi-C wants those paths off wholesale, which is what `-5SP` does.

See [Hi-C and other wide-insert data](../user-guide/memory-and-data-types.md)
for why this matters for memory as well as speed.

### Filtering

#### `-c INT` — skip repetitive seeds

Seeds with more than `INT` occurrences in the reference are skipped. Lowering
this (e.g. to 50) speeds up alignment of highly repetitive reads but may
reduce sensitivity. Raising it increases sensitivity in repeat-heavy regions
at a cost in runtime.

#### `-D FLOAT` — chain length fraction

Drops chains shorter than `FLOAT` times the longest overlapping chain. The
default (0.50) discards chains that are less than half the length of the best
chain.

#### `-W INT` — minimum seeded bases

Discards chains with fewer than `INT` seeded bases. Raising this filters out
very short, low-confidence chains.

#### `--min-ext-len INT` — skip Smith-Waterman extension of short seeds

Off by default (`0`) → output byte-identical to baseline. When `INT > 0`, a short
seed (< `INT` bp) is dropped before banded Smith-Waterman **only if its chain
still has a longer anchor seed** — its extension is then redundant (the anchor
already covers it), so skipping it is near output-neutral (~10 % less alignment
CPU at `30`). A chain whose seeds are *all* short is left untouched, so the filter
never empties a chain or drops a read: it is recall-safe by construction. `30` is
the recommended value. For the benchmarks, behavior details, and validation
status, see
[Settings profiles → `--min-ext-len 30`](../best-practices/settings-profiles.md#short-seed-extension---min-ext-len-30).

#### `--max-extend-chains INT` — cap chains extended per read

Off by default (`0`) → output byte-identical to baseline. When `INT > 0`, only the
top-`INT` chains by weight (after chain filtering) reach banded Smith-Waterman
extension; the remaining lower-weight chains are dropped before extension. This is
the only lever that reduces the *number of chains extended* per read, so it is
orthogonal to the seed- and SW-per-chain levers and adds a real marginal speedup on
top of them (~15 % marginal alignment CPU on top of `--fast`, ~23 % standalone, at
`5`). It is **not** byte-identical: dropping candidate chains removes low-weight
secondaries, so `XS`, secondary alignments, and `MAPQ` can shift on multi-mapping
reads. High-confidence (uniquely-placed) reads are unaffected. The cap is a safety
no-op for pathological reads with more than 4096 chains (`MAX_EXTEND_CHAINS_CAP`):
those reads extend all of their chains as usual, so `--max-extend-chains` has no
effect on them. `--fast` sets `20`. For the accuracy/speed curve and validation status,
see
[Settings profiles → `--max-extend-chains`](../best-practices/settings-profiles.md#chain-extension-cap---max-extend-chains).

#### `--adaptive-band` — adaptive banded Smith-Waterman for medium-length reads

> **Kilobase-scale long reads are not currently supported at default settings.**
> `bwa-mem3 mem` on PacBio HiFi (~15–20 kb) or ONT reads exhausts the
> short-read-sized extension buffers and aborts (or, before the guard, OOM'd /
> segfaulted) — the buffer model cannot represent a kilobase-scale extension.
> This lever helps the **medium-length** end of its range (e.g. SBX at ~240 bp,
> where it is validated); it does **not** make HiFi/ONT usable. For a true
> long-read workload use minimap2. Tracking issue: bounding long-read extension
> memory (fg-labs/bwa-mem3#238).

Off by default → output byte-identical to baseline. When set, banded extension
starts at a tight band and expands each pair only to the band its chain's seed
geometry actually needs (the inter-seed indel), rather than the fixed `-w` band
(100) for every extension.

**When to use it: medium-length reads.** The band only constrains the DP matrix
when the extension's reference window exceeds it (`ref_window > 2·w+1`), which
happens once reads are roughly ≥ 200 bp. So this is a lever for the
**medium-length** range — SBX (HG002, ~240 bp), where it cuts alignment CPU by
**~25 %**. It nominally applies to PacBio HiFi and ONT too, but kilobase-scale
reads do not run at default settings (see the caveat above), so in practice its
usable range today is medium reads, not true long reads. On short-read data (WGS ~150 bp, WES ~76 bp) the extension matrix is
already smaller than the band, so there is nothing to trim: those reads run on the
8-bit kernel, which this option deliberately leaves untouched, making it a **no-op
on short reads** (enabling it on a WGS/WES run neither helps nor hurts).

**Accuracy:** placement is unchanged (holodeck `sim-wgs-place`: MAPQ-60+ mismaps
identical to default) and indel representation is preserved — indels up to the
chaining limit still emit a single `D`/`I` CIGAR, matching the `-w 100` default,
so small/mid-size indel callability is unaffected.

**Not byte-identical when on.** Like `--fast`, enabling it shifts a small number of
borderline secondary alignments (starting tight and expanding can change which of
several near-tied placements wins). It is therefore an opt-in flag, not a default.

#### `--no-adaptive-band` — force exact extension (opt out of `--adaptive-band`)

Disables adaptive banded Smith-Waterman, restoring the exact full-width `-w`
extension. This makes the **extension step** byte-identical to a run without
`--adaptive-band` (the adaptive-banded extension introduced in
[fg-labs/bwa-mem3#194](https://github.com/fg-labs/bwa-mem3/pull/194); it is the same full-width
code path, by construction) — it does **not** make a whole `--fast` run
byte-identical, since `--fast`'s other levers
([`--max-extend-chains`](https://github.com/fg-labs/bwa-mem3/pull/193),
[`--smem-dedup`](https://github.com/fg-labs/bwa-mem3/pull/187),
[`--skip-contained-ext`](https://github.com/fg-labs/bwa-mem3/pull/192), …) can
still change output. It is the explicit opt-out for `--adaptive-band` and matters mainly under
`--fast`, which turns `--adaptive-band` on for you: passing `--no-adaptive-band`
alongside `--fast` keeps that one lever off (exact extension) while retaining the rest
of the preset — the same "opt back out" role `--rescue-kmer=0` plays.

The opt-out is order-independent and always wins: `--no-adaptive-band` beats an explicit
`--adaptive-band` given in either order, and beats the `--adaptive-band` that `--fast`
enables. On the default preset (where `--adaptive-band` is already off) it also disables the
on-by-default **certified** adaptive band, forcing the exact full-width code path there too:
output stays byte-identical, but the run forgoes the certified band's speedup — that is precisely
what `--no-band-cert` toggles on its own.
The resolved state is recorded on the `--fast` audit line — `[M::main_mem] --fast: …
--no-adaptive-band …` — so the run record shows exact extension was in force (the off
state would otherwise be invisible, exactly as for `--rescue-kmer=0`).

#### `--extend-mate-concordant` — retain mate-concordant chains under a chain cap

Takes an optional window: `--extend-mate-concordant` (bare) = **auto**, sizing the
window to the estimated proper-pair insert bound (`pes[FR].high`, inferred from the
data during the run); `--extend-mate-concordant=INT` pins a fixed window in bp;
`--extend-mate-concordant=0` disables it. Off by default → no effect. When on (and
`--max-extend-chains` is capping a paired-end read), a chain that would be dropped by
the cap is instead **retained if it is concordant with one of the mate's chains** —
same contig, FR ("innie") orientation, within the window. It only does anything when
a chain cap is in effect, so it is a strict no-op without `--max-extend-chains`.

The window matters: too wide and it retains — and then extends — far/spurious
concordant chains, adding alignment CPU on chain-rich reads; sizing it to the
aligner's own proper-pair insert bound (the auto default) admits only genuine pair
anchors. Before the insert size is estimated (the first chunk), auto falls back to a
built-in default.

**When to use it: `--meth`.** Bisulfite's collapsed 3-letter alphabet flattens chain
weights, so under `--max-extend-chains` the cap often drops a read's true low-weight
chain and starves PE pairing of the anchor that lets the true concordant pair win —
flipping both mates to a wrong concordant locus. This option recovers that anchor.
`--fast` enables it (auto) automatically **under `--meth` only**; on non-meth data the
cap does not regress placement, so `--fast` leaves it off to preserve the speedup.
The recovery is **partial** (it narrows, but does not fully close, the placement gap
to default), and with the auto window the alignment-CPU cost is **~1%** — sizing the
window to the insert bound is what keeps it there (a wide fixed window instead retains
and extends far/spurious concordant chains, costing 15–20% on chain-rich reads). See
the benchmarked per-dataset figures in
[fg-labs/bwa-mem3#195](https://github.com/fg-labs/bwa-mem3/pull/195).

**Not byte-identical when it retains a chain.** Like `--max-extend-chains`, keeping an
extra candidate can move `XS`, secondaries, and `MAPQ` on multi-mapping reads;
high-confidence placement is unaffected. For the placement/mismap validation, see
[Settings profiles → `--extend-mate-concordant`](../best-practices/settings-profiles.md#mate-concordant-chain-retention---extend-mate-concordant).

#### `-h INT[,INT]` — secondary alignment reporting

If there are fewer than `INT` hits with score exceeding `FLOAT` (see `-z`)
times the maximum score, all of them are output in the `XA` auxiliary tag.
The second integer is a hard cap on the number of XA entries. Defaults: 5, 200.

#### `-z FLOAT` — secondary score fraction

Fraction of the maximum alignment score used as the threshold for secondary
hit reporting with `-h`. Default: 0.80.

#### `-u` — emit XB instead of XA

Outputs `XB` in place of `XA`. `XB` is an extension of `XA` that also carries
the alignment score and mapping quality for each secondary hit.

### Speed preset

#### `--fast` — speed preset (opt-in, not byte-identical)

`--fast` is a one-flag shorthand for the characterized speed levers:

```text
bwa-mem3 mem --fast  ≡  -m 10 -y 0 --min-ext-len 30 --smem-dedup --skip-contained-ext --max-extend-chains 20 --adaptive-band --extend-mate-concordant --rescue-kmer=6
```

`--skip-contained-ext` is byte-identical to the default on non-meth short- and
medium-length single- and paired-end reads (validated on WGS/WES/HiC and SBX, and on an
ALT/HLA-enriched set) and no-ops under `--meth` (via its own internal gate). It is **not**
byte-identical on kilobase-scale long reads (PacBio HiFi, ONT): the "longest contained seed
dominates" assumption its skip relies on breaks when extensions terminate early on such
reads, dropping supplementary alignments and occasionally degrading a primary. Those inputs
are in any case not currently usable at default settings (see the long-read caveat under
[`--adaptive-band`](#--adaptive-band--adaptive-banded-smith-waterman-for-medium-length-reads)).

`--adaptive-band` (see above) is included because it is a strict no-op on short reads
(the reads `--fast` primarily targets) and a ~25% alignment-CPU speedup on medium-length
runs (e.g. SBX ~240 bp), so bundling it only helps. Note kilobase-scale HiFi/ONT reads do
not run at default settings regardless of this flag (see
[`--adaptive-band`](#--adaptive-band--adaptive-banded-smith-waterman-for-medium-length-reads)).
Pass [`--no-adaptive-band`](#--no-adaptive-band--force-exact-extension-opt-out-of---adaptive-band)
to opt back out — restoring the exact full-width extension step (byte-identical to a
run without `--adaptive-band`, the adaptive banding introduced in
[fg-labs/bwa-mem3#194](https://github.com/fg-labs/bwa-mem3/pull/194)) while retaining the rest of
`--fast`, whose other levers can still change output.

`--extend-mate-concordant` repairs the chain-cap pairing regression — the true, low-weight but
mate-concordant chain the cap would otherwise drop — and is included for both non-meth and `--meth`
`--fast` (see
[Settings profiles → `--extend-mate-concordant`](../best-practices/settings-profiles.md#mate-concordant-chain-retention---extend-mate-concordant)).

`--rescue-kmer=6` (see
[`--rescue-kmer`](#--rescue-kmerk----rescue-band-int--banded-mate-rescue-opt-in-not-byte-identical))
bands the mate-rescue Smith-Waterman to a 6-mer anchor diagonal. It is the largest single
lever in `--fast` on paired WGS (mate rescue is the biggest kernel there) and is validated
accuracy-neutral by truth-based ROC in both non-meth and `--meth` modes. Pass `--rescue-kmer=0`
to opt back out while keeping the rest of `--fast`.

`--fast` also flips one lever that has **no flag of its own**: the sort that orders alignment
regions by reference end position before the order-sensitive dedup/patch pass. The default
build uses bwa-mem2's comparator — a partial order on the end position alone — and reproduces the
permutation bwa-mem2's unstable introsort leaves, and so its dedup outcome, exactly. `--fast`
switches to a strict total order (end position, then start, score and query start) under pdqsort,
which can resolve equal-end-position ties differently. The two orderings only diverge when the
strict order's deterministic result differs from the permutation that unstable sort leaves;
because bwa-mem2's output is defined by that permutation, a strict total order and exact
bwa-mem2 parity cannot both hold in general. The
effect is confined to reads carrying several regions that end
at the same reference base, and it surfaces as different `XS`/MAPQ and occasionally a different
primary or supplementary record. Runs report the lever as `alnreg-sort=fast` on the audit line
below.

The lever is no longer worth much speed. The default path only needs the unstable introsort when
the comparator actually sees a tie — with no tie the ordering is unique, so pdqsort's result is
what introsort would have produced — and ties occur on under 1% of calls. Detecting them costs a
save-copy and a linear scan, which leaves the bwa-mem2-compatible default within ~0.7% of the
strict-total-order path on 5M HG00096 WGS pairs (arm64/NEON, `-t 16`, 5 interleaved reps), at or
below this benchmark's run-to-run spread.

Under `--meth` it additionally sets `-s 2` (light Pass-2 re-seeding) and lowers the chain cap to `10`.
Earlier releases used `-s 0` (no re-seed), which inflated
MAPQ on bisulfite reads; `-s 2` recovers the MAPQ/placement at nearly the same speed
(see [Settings profiles → Pass-2 re-seeding](../best-practices/settings-profiles.md#pass-2-re-seeding-under---meth--s-2)).

Each lever is applied only if you did not set it explicitly, so explicit flags
win where applicable (`--fast -m 30` keeps `-m 30`; `--fast --max-extend-chains 8`
keeps `8`); `--smem-dedup` and `--skip-contained-ext` are always enabled and
cannot be opted back out of once
`--fast` is set. Output is **not**
byte-identical to the default; the accuracy cost of each lever is characterized in
[Settings profiles](../best-practices/settings-profiles.md) and is confined to
the already-low-confidence tail. `bwa-mem3 mem` prints the resolved preset to
stderr (`[M::main_mem] --fast: ...`) so runs are self-documenting.

### Methylation (`--meth`)

#### `--meth` — enable bisulfite alignment mode

Activates bisulfite alignment: each read is projected (R1 `C→T`, R2 `G→A`) to find
seeds in the converted `.meth` seed index, then extended and scored against the
**original** 4-letter reference, with inline meth post-processing (Bismark
`XR:Z`/`XG:Z`/`XM:Z`, opt-in chimera QC). The reference must have been indexed with
`bwa-mem3 index --meth`.

`--meth` does not choose an output format: records are SAM text by default and BAM
under `--bam`, the same rule as without `--meth`. Both containers serialize the same
`bam1_t` and differ only in the `htsFile` mode string, so the two decode to identical
**records** — headers differ in the `@PG` `CL:` field, which records the invoking
command line. A regression script pins record and (`CL:`-stripped) header equality on
the committed phiX fixture in single-end and paired-end layouts. (Through 0.7.x
`--meth` forced `--bam`; add `--bam` to any script that depended on that.)

Pass the original FASTA prefix as `<idxbase>` (e.g. `ref.fa`); the `ref.fa.meth.*`
seed index alongside it is found automatically. A legacy bwameth `.bwameth.c2t`
index is not used directly — rebuild with `index --meth` (see
[Migrating from bwameth.py c2t](../methylation/external-c2t.md)).

See [Methylation Reference](../methylation/overview.md) for the full treatment.

#### `--meth-scoring {collapsed|genomic|neutral}` — bisulfite/TAPS scoring model

Selects how the 4-letter matrix treats converted bases. `collapsed` (the
`--meth`/`--meth=emseq` default) frees C↔T and G↔A both ways (bwameth-compatible
placement, sets `-B 2`); `genomic` frees only the conversion direction scored as
a full match, keeping real variants as mismatches (variant-aware: variants outside
the conversion direction stay visible in `NM`/`MD`, keeps `-B 4`); `neutral` (the
`--meth=taps` default) frees only the conversion direction but scores it `0` —
tolerated, not rewarded — best for the sparse conversions of TAPS (keeps `-B 4`;
variants stay visible in `NM`/`MD` as in `genomic`). In both, a genuine variant in
the conversion direction itself (a C→T at a reference `C`) shares the freed cell
with a conversion and stays hidden
([why](../methylation/overview.md#which-real-variants-stay-visible)). Only meaningful
with `--meth`.
See [Flags → --meth-scoring](../methylation/flags.md#--meth-scoring-collapsedgenomicneutral).

#### `--meth-tags SPEC` — select which Bismark tags are emitted

`all` (default), `none`, a comma-separated inclusion list (`XR,XG`), or
`^`-prefixed exclusions (`^XM`). Inclusion and exclusion forms cannot be mixed.
A deselected tag is not computed, so `^XM` skips the per-read methylation-call
pass as well as its bytes — `XM:Z` is a read-length string and is ~33% of the
BAM. Keep it for Bismark-family tools; drop it for MethylDackel/biscuit, which
recompute from the reference. Only meaningful with `--meth`.
See [Flags → --meth-tags](../methylation/flags.md#--meth-tags-spec).

#### `--set-as-failed {f|r}` — strand QC-fail flag

Forces the QC-fail bit (`0x200`) on all alignments to the forward (`f`) or
reverse (`r`) bisulfite strand. Used when one strand is known to be
unreliable for a given library preparation.

#### `--chimera-qc` — opt in to bwameth.py-style chimera heuristic

Off by default (matches Bismark, which has no equivalent heuristic).
When set, mapped records whose longest M/=/X CIGAR run is less than 44 % of
the read length get `0x200` set, `0x2` cleared, and MAPQ capped at 1. Useful
for PBAT / scBS-Seq libraries where intra-fragment chimerism is common, or
when reproducing bwameth.py output bit-for-bit.

### Threading

#### `-t INT` — number of threads

Number of worker threads. Defaults to 1, maximum 256 — a representable integer
value above 256 is clamped to 256 (with a warning) rather than rejected, since
the per-thread profiling arrays are sized to that maximum. A value outside the
signed 32-bit `int` range (e.g. `2147483648`) is rejected as a parse error, and
a value of zero or below floors to 1. Set to the number of physical
cores available to this job. Scaling is workload-, hardware-, and storage-dependent.
Measure representative input on the target host to select a thread count. See the threading guide
for measured guidance and per-machine recommendations.

See [User Guide — Threading and resource use](../user-guide/threading.md) for
guidance on thread counts at various machine sizes.

### Memory

#### `--huge-pages` — back the index with 1 GB huge pages (Linux, opt-in)

Reserve the FM-index / suffix-array structures on explicit **1 GB huge pages** to
cut data-TLB misses in seeding. bwa-mem3's allocator is mimalloc, and this flag
sizes the reservation from the index footprint and reserves it before the index
loads. **Safe by default:** when the host has no free 1 GB hugepage pool (or too
few pages), it prints a one-line `[M::]` note and runs on the default page size —
it never fails or degrades a run.

Requires 1 GB hugepages reserved on the host
(`echo N | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages`).
**Linux only**, needs the bundled mimalloc. The **alignment records are
byte-identical** — page size does not affect alignments; only the `@PG` record
differs, since its `CL:` field records the `--huge-pages` flag. Measured ~1.5 %
whole-aligner wall on a 5 M-read WGS slice (HG00096, hg38), AMD Zen3, avx2 tier,
32 threads ([PR #405](https://github.com/fg-labs/bwa-mem3/pull/405)) — a
single-host, single-config measurement, not a cross-architecture or cross-thread
claim. See
[Memory allocator → Large pages for the index](../user-guide/allocator.md#large-pages-for-the-index-linux-deployment-lever)
for reservation, verification, the measurement conditions, and the manual
`MIMALLOC_RESERVE_HUGE_OS_PAGES` equivalent.

### Supplementary MAPQ rescoring

#### `--supp-rep-hard-cap INT` — cap MAPQ for repetitive supplementary alignments

Forces MAPQ=0 for supplementary alignments whose chain contains any seed with
at least `INT` occurrences in the genome. This targets supplementary
alignments anchored in repetitive regions that upstream MAPQ scoring may
overestimate. `0` disables the cap (default). Typical values are 5–20; lower
values are more aggressive. Primary alignment MAPQ is unaffected.

#### `--proper-pair-from-emitted` — derive FLAG `0x2` from the emitted alignment

Derives the proper-pair `FLAG` bit (`0x2`) from the alignment bwa-mem3 actually
emits (`a[which]`) rather than from the top-scoring region (`a[0]`).

**bwa and bwa-mem2 both use `a[0]`**, even when the record they emit is
`a[which]` — so this flag deviates from *both* upstreams, and is a **hard error**
combined with [`--compat`](#--compattarget--byte-identical-output-for-another-aligner).

**It has no effect on alignment output unless the index carries a `.alt`
sidecar** (the `@PG` `CL:` field records the command line either way, so that
one record always reflects the flag). The two
derivations differ only when the top region scores below `-T` and the first ALT
region clears it, which requires the read to have ALT hits; without a sidecar
`is_alt` is never set and the branch is unreachable.

This was bwa-mem3's default until
[#362](https://github.com/fg-labs/bwa-mem3/issues/362), which found it moved
`0x2` on 3,013 of 10,134,006 records on a 5 M-pair WGS slice (HG00096, GRCh38
with the standard `.alt`), measured on an AWS c6a.4xlarge host (AMD EPYC Milan,
x86_64, AVX2 tier) — all on decoy and ALT/HLA contigs, none on the primary
assembly. Because `0x2` is aligner-defined,
the drop-in path now takes upstream's answer and this became opt-in. See
[Correctness fixes → Proper-pair flag](../whats-different/correctness.md).

### Debug

#### `-k INT` — minimum seed length

Minimum exact-match seed length. Shorter seeds increase sensitivity but raise
runtime. The default (19) is calibrated for 100–150 bp Illumina reads.

#### `-w INT` — band width

Band width for the banded Smith-Waterman extension. Wider bands can recover
alignments with long indels at greater CPU cost.

#### `-d INT` — X-dropoff

Off-diagonal X-dropoff for the Z-drop heuristic. Controls how far an alignment
extension continues after a score drop.

#### `-r FLOAT` — re-seeding factor

Seeds longer than `-k * FLOAT` are re-seeded internally to find sub-seeds (bwa-mem's
second seeding round). Lowering produces more seeds / higher sensitivity at greater
cost; raising (e.g. `-r 10`) suppresses the round. Round 2 is genuine
split-read/divergence sensitivity, so only suppress it on known-clean data — see
[Settings profiles → `-y 0`](../best-practices/settings-profiles.md#third-round-seeding--y-0).

#### `-y INT` — third-round seed occurrence threshold

bwa-mem's third seeding round: for each read position, grow an exact match until it
occurs fewer than `INT` times in the genome (default 20), then emit it as a seed — a
repeat-region safety net. **`-y 0` disables the round**, cutting ~11–30 % of alignment
CPU with F1-near-neutral accuracy; it is part of the recommended profile. For the
regime sweep and rationale, see
[Settings profiles → `-y 0`](../best-practices/settings-profiles.md#third-round-seeding--y-0).

#### `--legacy-reader` — use the legacy input reader

Read input with the legacy `gzFile`/`kseq` reader instead of the default
content-detecting fast reader. An escape hatch for A/B baselining or working
around an input the fast reader mishandles; not needed in normal use.

## Notes / Gotchas

> **Warning — --meth requires a --meth index**
>
> Running `bwa-mem3 mem --meth` against a standard (non-c2t) index produces
> incorrect alignments without an error. Confirm that the index was built with
> `bwa-mem3 index --meth` before aligning bisulfite data.
>
> **Note — SIMD variant printed to stderr at startup**
>
> When mem starts it prints a banner (`Executing in AVX512 mode!!` etc.) to
> stderr. This is informational and does not affect stdout output.

---

**See also:**
[User Guide — Aligning short reads](../user-guide/aligning.md) ·
[User Guide — Output: SAM/BAM, headers, tags](../user-guide/output.md) ·
[CLI Reference — index](index-cmd.md) ·
[Methylation Reference — Overview](../methylation/overview.md) ·
[Best Practices — Output format](../best-practices/output-format.md)
