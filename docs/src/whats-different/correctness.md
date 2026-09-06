# Correctness Fixes

This page documents bugs present in upstream bwa-mem2 that bwa-mem3 fixes. Each
fix is isolated to a single PR so it can be reviewed independently and dropped
from `main` once upstream merges the equivalent patch.

## `@PG CL:` tab escaping (PR #54)

When a read-group string is passed via `-R '@RG\tID:x\tSM:y'`, the tab
characters in the argument were copied verbatim into the `@PG CL:` SAM header
field. The SAM specification uses tabs as field delimiters, so the resulting
header line appeared to have extra `ID:` and other tag fields embedded inside
`CL:`. Lenient parsers (samtools, htsjdk) tolerated the output; strict parsers
(noodles, some fgbio configurations) rejected the file as malformed.

The fix replaces each tab character with a space when building the `@PG CL:`
value in `src/main.cpp`. The `@RG` line itself is not modified, so the
read-group metadata is preserved correctly. A regression shell test
(`test/pg_cl_escape_test.sh`) asserts that the `@PG` line contains exactly
five tab-separated fields after the fix. Upstream issue reference:
[bwa-mem2#293](https://github.com/bwa-mem2/bwa-mem2/issues/293).

## SMEM buffer overflow on reads longer than 151 bp (PR #55)

bwa-mem2 hardcoded `READ_LEN 151` in `src/macro.h` to size the per-thread
`matchArray` SMEM buffer at compile time. The FMI walk wrote past this buffer
without bounds checking when reads exceeded 151 bp, causing memory corruption
that manifested as segfaults or silent wrong output on 300 bp MiSeq reads,
error-corrected long reads, and any run with a non-default `-k` that extended
seed length.

A second cap, `MAX_READ_LEN_FOR_LOCKSTEP 512`, guarded the lockstep driver's
per-slot stack arrays with a hard assert that aborted on anything longer.

The fix eliminates both compile-time caps. Every per-thread SMEM buffer is now
heap-allocated on the memory management context (`mmc`) and grown on demand
from each batch's observed `max_readlength`. The pre-walk grow in
`mem_collect_smem` sizes `matchArray[tid]` to `BATCH_MUL * BATCH_SIZE *
max_readlength`, and all array writes are bounds-checked with a structured
`smem_overflow_die` on overflow. Regression tests cover 300 bp, 1 kbp, and
3 kbp phiX reads; all three segfaulted before the fix and produce correct
NM:i:0 alignments after. Upstream references:
[bwa-mem2#210](https://github.com/bwa-mem2/bwa-mem2/issues/210) (issue),
[bwa-mem2#238](https://github.com/bwa-mem2/bwa-mem2/pull/238)
(closed unmerged upstream PR).

## `kswv` nrow==0 guard (PR #51)

When a SIMD batch contained only padding pairs (all `len1 == 0`), the DP loop
never executed and `nrow` was zero. The post-loop `rowMax + (i-1) * SIMD_WIDTH`
store still executed, walking SIMD_WIDTH bytes before the beginning of the
`rowMax` allocation. On glibc this produced a `free(): invalid pointer` abort;
on macOS libc it silently corrupted the heap.

The fix wraps the post-loop store in an `if (i > 0)` guard on all five SIMD
kswv kernels: NEON u8, NEON 16, AVX2 u8, AVX-512BW u8, and AVX-512BW 16. The
upstream patch [bwa-mem2#289](https://github.com/bwa-mem2/bwa-mem2/pull/289)
covered only the two AVX-512BW kernels; bwa-mem3 broadens it to the three
additional kernels carried in this fork. A dedicated regression test
(`test/kswv_nrow_zero_test.cpp`) builds all-padding batches and verifies each
kernel is clean under AddressSanitizer.

## kswv score2 plateau series (PRs #26, #27, #28, #29, #30, #31)

The batched mate-rescue Smith-Waterman path (`kswv`) contains a family of
related bugs across its SIMD kernels that inflated the suboptimal score
(`score2` / `XS`) and consequently deflated MAPQ relative to upstream
bwa-mem2.

**AVX-512BW dispatch guard (PR #26).** GCC with `-mavx512bw` automatically
defines `__AVX2__`, so the `#elif __AVX2__` branch in `src/kswv.h` and
`src/kswv.cpp` matched first on every AVX-512BW build. The 256-bit AVX2 kernel
produced only 32-lane results into 64-lane `score[]`/`te1[]`/`qe[]` arrays
sized for AVX-512BW; the upper 32 lanes held uninitialized values.
`mem_matesw_batch_post` read those bogus `te` values, `bwa_gen_cigar2`
returned NULL, and `mem_reg2aln` triggered an `a.cigar != NULL` assertion on
every AVX-512BW dispatch host (AWS c7a, c7i). The fix qualifies the `#elif
__AVX2__` guard with `!__AVX512BW__`, matching the existing pattern in
`bandedSWA.h`. Closes [issue #25](https://github.com/fg-labs/bwa-mem3/issues/25).

**AVX2 score2 plateau fix (PR #27 closed, PR #28 merged).** The AVX2 256-bit
`kswv` kernel added in PR #20 used a dense SIMD max over every `rowMax` row to
compute the suboptimal score. Scalar `ksw_u8` instead collapses consecutive
rows above `minsc` into a single `b[]` entry anchored at the max-score row,
then finds the best anchor outside the primary region. The dense max pulled in
tail rows from a plateau whose anchor sat inside the primary region, inflating
`XS` by 1–4 on a minority of reads and reducing MAPQ by 2–18 on those reads.
PR #27 (closed) temporarily disabled the AVX2 batched path. PR #28 fixes the
kernel itself by replacing the dense scan with a per-lane scalar emulation of
the `b[]` build-and-scan logic.

**NEON and AVX-512BW 8-bit port (PR #29).** The same dense-rowMax score2 scan
existed in `kswv_neon_u8` and `kswv_512_u8`. Confirmed on ARM: rebuilding
smoke-1M on darwin/arm64 pre-fix produced the identical four MAPQ regressions
as the AVX2 case. PR #29 ports the per-lane scalar `b[]`-emulation fix to both
kernels.

**AVX-512BW 16-bit port (PR #30).** `kswv_512_16` carried four bugs: the same
dense-rowMax plateau pattern, aggregate `maxl`/`minh` bounds instead of
per-lane bounds (a gap from PR #21), no `minsc` filter, and no `qe` mask. The
per-lane scalar emulation from PR #29 fixes all four naturally.

**NEON 16-bit rewrite (PR #31).** `kswv_neon_16` was effectively dead code
before this PR. Five interacting bugs produced 20,435 BAM diffs vs scalar
reference on smoke-1M `-A 2`: the score table reinterpreted `int16` xor
indices as `int8` lookups (inflating match scores by ~256 per cell), the table
was too small for the 16-bit SoA encoding, `rowMax` was never written, the
early-exit fired on row 0 for all pairs without a `KSW_XSTOP` target, and all
the fix-3 class bugs from PRs #28–#30 were missing. The PR rewrites the kernel
from scratch against `kswv_neon_u8`'s structure using 32-byte `int8` tables
indexed via `vqtbl2_s8`, per-lane freeze, `exit0` bitmap, and per-lane scalar
score2.

## `kseq2bseq1` zero-initialization (PR #22)

`bseq_read_orig` grows its sequence buffer with `realloc`, leaving tail entries
uninitialized. `kseq2bseq1` populated only `name`, `comment`, `seq`, `qual`,
and `l_seq` for each entry, leaving `sam`, `bams`, `n_bams`, and `cap_bams` at
whatever values `realloc` happened to return. PR #13 added an unconditional
`free(ret->seqs[i].bams)` in the output loop (`fastmap.cpp:571`), which turned
those garbage values into a crash — a `pointer being freed was not allocated`
abort under system malloc and a SIGSEGV under mimalloc — once input exceeded
the initial 256-sequence allocation. The crash was deterministic and
reproducible with `-t1`.

The fix is a single `memset(s, 0, sizeof(*s))` at the top of `kseq2bseq1`.

## Proper-pair flag from emitted alignment (PR #17)

In the `no_pairing` emission path of `mem_sam_pe` and `mem_sam_pe_batch_post`,
the proper-pair bit (`0x2`) was computed from `a[i].a[0].rb` regardless of
which alignment was actually emitted. When the primary's alignment score fell
below the reporting threshold `opt->T` but a non-primary ALT hit cleared it,
`mem_reg2aln` emitted `a[i].a[n_pri[i]]` while `mem_infer_dir` still read the
below-threshold primary. In that case the SAM flag did not reflect the
coordinates in the record.

`#17` stored the selected alignment index per mate in a `which[2]` array and
passed `a[i].a[which[i]].rb` to `mem_infer_dir`, so the proper-pair flag matched
the emitted record. **That derivation is now opt-in**: since
[#362](https://github.com/fg-labs/bwa-mem3/issues/362) the default reads `a[0]`,
as bwa and bwa-mem2 do, and `a[which]` is selected only under
`--proper-pair-from-emitted` — see the note below for why. The `a[0]` derivation
was present in the bwa-mem2 initial commit from 2019, inherited from bwa, which
still has it at 0.7.19 (`bwamem_pair.c:411`). Upstream reference: pre-existing,
no open upstream PR.

> **Opt-in since [#362](https://github.com/fg-labs/bwa-mem3/issues/362) — the
> default now matches both upstreams.** `0x2` is aligner-defined ("properly
> aligned according to the aligner"), so deriving it from `a[which]` is a
> defensible choice rather than a correction — and as a *default* it made
> bwa-mem3's output differ from bwa **and** bwa-mem2 simultaneously. The
> difference is confined to ALT-aware runs, and within those to the records
> where the emitted alignment is not the top-scoring one (`a[which] != a[0]`,
> which needs the read's top primary region to score below `-T` while its top
> ALT region clears it); a run without a `.alt` sidecar is byte-identical
> either way, excluding the `@PG` record, which records the command line.
> Measured at **3,013 of 10,134,006 records** on a 5 M-pair WGS
> slice (HG00096, GRCh38 with the standard `.alt`), on an AWS c6a.4xlarge host
> (AMD EPYC Milan, x86_64, AVX2 tier), all on decoy and ALT/HLA contigs, none
> on the primary assembly.
>
> The behavior is now reached via `--proper-pair-from-emitted` (a hard error
> with `--compat`, overridable via `--compat-allow-divergent`), and the default derives `0x2` from `a[0]` as bwa and
> bwa-mem2 do. This follows the same call [#256](https://github.com/fg-labs/bwa-mem3/pull/256)
> made on gap-open convention: where a fork-carried improvement moves output,
> the drop-in path takes upstream's answer and the improvement becomes opt-in.
>
> The divergence is unreachable without a `.alt` sidecar — `a[which] != a[0]`
> requires `n_pri < a.n`, i.e. the read has ALT hits, and `is_alt` is never set
> without one.

## 8-bit banded-SW envelope cap at `w = 124` (PR #422)

`bsw8_envelope_ok` — the gate that routes a seed-extension pair to the 8-bit
banded-SW kernel — admitted band widths up to `BSW8_MAX_W = 127`. The 8-bit kernel
encodes its band/position quantities as **signed-int8 diagonal offsets** spanning
`[-(w+1), w+3]` (the band-grow term `myband+1` and the tail-trim term `index+2`,
which reaches `w+3`). At `w >= 125` the positive edge `w+3` exceeds `+127` and wraps
negative: the band collapses on row 0, `head > tail`, and lanes die mid-alignment —
the pair then silently returns "no extension" (`score = h0`, `tle = qle = 0`,
`gscore = -1`) where the scalar and 16-bit kernels return the real extension. The
fix drops `BSW8_MAX_W` to **124**, so `w` in {125, 126, 127} route to the 16-bit
tier (exact, byte-identical to scalar) instead.

The overflow begins at `w >= 125`; `-w 127` is the characterized case — the one
user-supplied width that reaches the wrap on essentially every 8-bit-routed
extension (the wide-band retry doubling to 200/400 is already diverted to 16-bit).
`w = 125/126` also cross the boundary but corrupt only exotic pair shapes, so `-w 127`
is the width the tests reproduce. The default `opt->w = 100` never reaches the 8-bit
cap, so the drop-in path is **byte-identical** — verified by a records-only
(header-excluded) whole-aligner md5 on a 2×150 bp WGS workload, where `-w 100` output
is identical to `main`. A `getScores8` characterization test
(`test/unit/test_bandedswa_longread.cpp`, all six result fields vs the scalar
`scalarBandedSWA` oracle across the SIMD tiers exercised in CI) is byte-identical at
the `w = 124` bound and diverges at `w = 127`, locking the reason for the cap; at
`-w 127` this branch's records-only md5 matches an all-16-bit reference while `main`'s
8-bit tier differs.

## Vector banded-SW z-drop gate at `-d 0` (PR #424)

`scalarBandedSWA` gates its z-drop early-exit on `zdrop > 0`, so with z-drop
disabled (`-d 0`) it never truncates. The vector banded-SW kernels (8-bit and
16-bit, every SIMD tier) applied the z-drop test **unconditionally**: at
`zdrop == 0` they compared the running drop against a zero threshold and killed a
lane as soon as its score fell one point below the row max, truncating alignments
the scalar reference runs to completion. The two then disagreed on the query-end
fields (`gscore`/`gtle`), and once the vector stopped early, on the local fields
(`score`/`tle`/`qle`/`max_off`) too. The fix gates all six vector z-drop mask
updates (the three `ZSCORE16` macros and the three 8-bit wide-`epi32` paths) on
`zdrop > 0`, matching the scalar.

The effect is confined to **`-d 0` (z-drop disabled)**, where essentially all
extension pairs previously diverged between the vector tiers and the scalar
reference; disabled z-drop now preserves the alignments the old kernels wrongly
truncated. On the default `-d 100` the guarded branch is always taken, so the
kernels are **byte-identical** there and the drop-in path is unchanged. This was
verified by a controlled records-only (header-excluded) whole-aligner md5 A/B —
control (`main`) versus this branch on identical inputs, default flags, and
batching (`-K` held fixed for both runs) — on a 2×150 bp WGS workload (HG002,
hg38): the default-`-d 100` output matches the pre-fix baseline on both the ARM64
(NEON) and x86_64 (AVX2) tiers. At `-d 0` the fixed kernels' records-only md5
instead equals that same default-flags output — disabling z-drop preserves
exactly the alignments the default z-drop leaves untruncated on these reads —
while the old kernels diverge. A standalone kernel byte-identity gate also passes
at default parameters (extend and rescue). A regression test
(`test/unit/test_bandedswa_zdrop_gate.cpp`) compares `getScores8`/`getScores16` at
`zdrop == 0` to `scalarBandedSWA` on NEON, AVX2, and AVX-512BW: it fails on the old
kernels and passes on the fixed ones.

## 16-bit banded-SW per-lane band clamp (PR #423)

The three 16-bit banded-SW wrappers (`smithWatermanBatchWrapper16`, at 128/256/512
bits) computed the per-lane band-clamp reach with a **16-bit modular add**, reading
the sum back through a `uint16_t`. Because `qlen[l]` already holds `qlen * max_sc`,
the reach `qlen*max_sc + (end_bonus - o)` can go negative (a short query under a
large gap-open, e.g. `-O16`) or exceed 65535; the 16-bit add then wraps and the
clamp evaluates to a huge band that effectively disappears, so `getScores16` runs
the full band `w` where the scalar reference (`scalarBandedSWA`) clamps it as narrow
as 1. The 8-bit wrappers were already fixed for this; this ports the identical wide
`int` computation (mirroring the scalar "adjust `w` if it is too large" block) to
the three 16-bit wrappers.

The effect is scoped to **non-default scoring** (`-B9 -O16 -E3` and similar) on
short-query pairs, where the mis-sized band diverged from the scalar reference on
the query-end fields (`gscore`/`gtle`) and could change `score`, endpoints, and the
emitted alignment records. On bwa's default `-O6 -E1` at normal read lengths the
intermediate provably never wraps, so the wide-int path computes the identical
band and the drop-in path is **byte-identical by construction** — an
integer-arithmetic invariant that holds independent of host, compiler, and thread
count, so the proof (not any single benchmark) is the load-bearing evidence. A
controlled records-only (header-excluded) whole-aligner md5 A/B corroborates it:
control (`main`) versus this branch on a 2×150 bp WGS workload (HG002, hg38) at
default flags, with inputs, flags, thread count (`-t`), and batching (`-K`) all
held fixed across the two runs — so the batch composition is identical — on both
the ARM64 (NEON) and x86_64 (AVX2 and AVX-512BW) tiers. A `getScores16` band-clamp
regression test compares `score`, `tle`, `qle`, and `max_off` to the
`scalarBandedSWA` oracle unconditionally, and `gscore`/`gtle`
only when either side reports `gscore > 0` (a to-end alignment is observable),
across the wrap scoring sets on NEON, AVX2, and AVX-512BW: it fails on the old
kernel and passes on the fixed one.

## Symmetric ref/query length bounds guards on the SW wrappers (PR #467)

Every SIMD banded-SW wrapper (`getScores8`/`getScores16`, all widths) copies each
pair's reference window (`len1`) and query (`len2`) into the SoA lane buffers and
sizes the per-lane band from those lengths. `SeqPair::len1`/`len2` are signed
`int32_t`, and the wrappers previously did almost no length validation: five of the
six checked neither length, and the sixth (the AVX2 8-bit wrapper) checked only
`len2`'s **upper** bound — a non-fatal `stderr` warning plus a debug `assert` — while
no wrapper ever checked `len1` or a lower bound. So a **negative** length, an
out-of-range `len1` in any wrapper, or an over-limit `len2` in any of the other five
would index the SoA copy/padding buffers out of bounds (negative `k`) or wrap the
per-lane `uint8_t`/`uint16_t` band quantity. The guards are now symmetric: each wrapper
rejects both `len1` and `len2` outside `[0, MAX_SEQ_LEN8)` / `[0, MAX_SEQ_LEN16)`
via `err_fatal` **before any use of those lengths** — the SoA copy and padding
loops and the per-lane band-size calculation (`qlen[j] = len2 * max`, whose signed
multiply an unvalidated length could overflow) — and the fatal diagnostic states
the accepted range rather than only "exceeds".

This is an **invalid-input contract**, not a change to any aligned result: valid
alignment pairs always carry `0 <= len < MAX_SEQ_LEN`, so the guard is never taken
on the valid path and the emitted records are **byte-identical by construction** —
the change only converts previously-undefined behavior on malformed input into a
deterministic fatal error. Empirically, a record-level comparison of all result
fields against the pre-change baseline showed zero discordant pairs for the
extension and mate-rescue kernels on arm64 (Apple Silicon, NEON tier), over
synthetic pairs at query/reference lengths 100/150/250 (including ambiguous bases)
at a fixed batch composition; the x86 tiers (SSE4.1, AVX2, AVX-512BW) are built and
exercised on every CI run. That scopes the measured result to this workload, host,
and tier — the by-construction invariant above is what extends it to all valid
inputs.

---

## Banded-SW `qlen` band-clamp reach slot overflow (PR #468)

The batch banded-SW wrappers (`smithWatermanBatchWrapper{8,16}`, all widths)
pre-scale the per-lane band-clamp reach as `len2 * max_sc` and stash it in an SoA
slot the clamp loop reads back. That slot was the narrow per-lane element type —
`uint8_t` on the 8-bit tiers, `uint16_t` on the 16-bit tiers — so the scaled reach
wraps once `len2 * max_sc` exceeds 255 / 65535. `max_sc` is the largest
scoring-matrix entry, i.e. the match reward `A` (mismatch and ambiguous entries are
stored negative), so the wrap trigger is `len2 * A` crossing the slot ceiling and
does **not** depend on `-B`. A wrapped reach makes the vector band **narrower** than
`scalarBandedSWA` computes (which keeps the reach in native `int`), so the extension
can miss an off-diagonal optimum the scalar reference finds. The fix carries the
scaled reach in a wide `int32_t qlen_scaled[]` slot, matching the scalar band
formula exactly. This is the storage-slot sibling of #423, which fixed the reach
*sum* (a `uint16_t` modular add) but still read `len2 * max_sc` back from the narrow
slot.

The reachable divergence is the **16-bit tier only**. On the 8-bit tier the
production routing envelope (`bsw8_envelope_ok`) admits a pair only when
`h0 + len2 * A < 255 - max_step` with `len1 >= len2`, so `len2 * A < 255` for every
routed pair and the narrow `uint8_t` slot never overflows — the 8-bit widening is
**defensive** (no shipped-output change). The 16-bit slot overflows at
non-default `-A >= 3` with a long query segment (`len2 * A > 65535` — e.g.
`len2 ~ 21846 bp` at `-A3`; `MAX_SEQ_LEN16 = 32768` caps `len2 <= 32767`). Output can
diverge only when an off-diagonal optimum requires a band wider than the wrapped
reach; `-A2` cannot reach it (`len2 * 2 <= 65534`). On bwa's default
`-A1` and normal read lengths the product never wraps, so the wide slot computes the
identical band and the emitted **alignment records** are **byte-identical by
construction** for a fixed batch composition — an integer-arithmetic invariant
independent of host, compiler, and thread count; the SAM header block is outside this
claim.

A `getScores{8,16}` band-clamp regression test compares `score`, `tle`, `qle`, and
`max_off` — plus `gscore`/`gtle` under the kernel's query-end contract — to the
`scalarBandedSWA` oracle across the wrap scoring sets, including a case that
constructs the reachable 16-bit long-read overflow: it fails on the old narrow-slot
kernel and passes on the fixed one (18 subcases across 3 doctest cases, 36 assertions).
Its pairs are generated with deterministic, bounded query/reference lengths at a fixed
batch composition, and it runs on the SIMD tier its binary is compiled for — SSE4.1 and
AVX2 on CI's x86 rows, and NEON on the arm64 row and local Apple Silicon builds.
A full-field kernel byte-identity gate over the extension and mate-rescue result
fields — every field
per pair, 60,009 pairs per cell — additionally showed **zero** discordant pairs at
query/reference lengths 100/150/250 for default `-A1` (and a 10%-ambiguous-base
variant) and non-default `-A3` on the forced 16-bit tier, at a fixed batch
composition on arm64 (Apple Silicon M2 Max, NEON tier); the x86 tiers (SSE4.1, AVX2, AVX-512BW) are built and exercised
on every CI run. That scopes the measured result to this workload, host, and tier —
the by-construction invariant above is what extends it to all valid inputs.
bwa-mem2 inherits the same narrow-slot overflow, so on the affected reads this
diverges from bwa-mem2 while converging bwa-mem3's vector tiers onto its own scalar
reference.

---

## `--meth` SEQ restore clamps unsupported bytes to `N` (PR #463)

In `--meth` mode the original read sequence is restored into the emitted BAM `SEQ`
field along one of four branches, chosen by whether the pre-c2t original sequence
(`meth_orig_seq`, else the `YS:Z` comment) is available and by the read's orientation:

- **Forward, original sequence available** — the bytes are copied directly from the
  original sequence (only lowercase folded to uppercase); they do **not** pass through
  `nst_nt4_table`, so a `-` on this path is emitted verbatim as `-`.
- **Reverse, original sequence available**, plus both **fallback paths** (original
  sequence missing, restoring from `s->seq`, forward and reverse) — each ASCII base is
  mapped through `nst_nt4_table` and used to index a five-character string literal
  (`"ACGTN"` / `"TGCAN"`, indices 0–4).

`nst_nt4_table` maps `A/C/G/T/N` (and their lowercase forms) to `0..4` and every
unrecognized byte to `4` — with one exception: `-` maps to `5`. Index `5` selects the
literal's terminating NUL, so on the three lookup-based paths a `-` emitted a `'\0'`
into `SEQ`: an in-bounds read of the string terminator, but corrupt output rather than an
out-of-bounds access. The fix clamps any index `>= 4` to `4`, so on those paths a `-`
now emits `N`.

This is scoped to **`--meth` mode only**. On the lookup-based paths the clamp changes
output **only for a `-` byte** — the single input that maps to `5` — which now renders
as `N` instead of a corrupt NUL; every other byte already resolved to an index in
`0..4`, so its output is unchanged. That equivalence is **by construction** — the clamp
alters the result solely for the `bi == 5` case — not a measured-fixture generalization;
standard fixtures do not exercise the `-` input. The direct-copy forward path (where a
`-` stays `-`) and the default (non-`--meth`) output are unaffected.

---

## Upstream port divergences: all chains dropped by the weight filter (PR #489) and the SA sentinel offset (PR #469)

bwa-mem2 advertises output identical to bwa. Two records are known where its port is not
faithful, and on both the default path follows bwa; `--compat=bwa-mem2` reproduces bwa-mem2 as
shipped, because parity with that release is its contract (see
[Equivalence → Byte-identical output](equivalence.md#byte-identical-output---compat)).

- **All chains dropped by the weight filter** ([#310](https://github.com/fg-labs/bwa-mem3/issues/310),
  default fixed in [#489](https://github.com/fg-labs/bwa-mem3/pull/489)). With `-W` above zero or
  an `-x pacbio`/`pbref`/`ont2d` preset, when the weight filter drops every chain for a read, bwa
  reports zero survivors and the read goes out unmapped. bwa-mem2's seqid-range bookkeeping
  rebuilds a count of one for the emptied array and extends the chain the filter just rejected —
  whose seeds it has already freed, so what it extends depends on the allocator. Unreachable at
  default settings, since `min_chain_weight` defaults to 0. Pinned by
  `test/regression/chain_flt_empty_compat.sh`.
- **Suffix-array sentinel offset** ([#469](https://github.com/fg-labs/bwa-mem3/pull/469)).
  bwa-mem2's software-pipelined SA lookup drops the accumulated LF-walk offset when the walk
  reaches the sentinel row, placing a hit within the first few bases of the concatenated reference
  up to a few bases too far left; bwa's `bwt_sa` counts the walk. Latent on hg38 (which opens with
  a long `N` run), reachable on small or custom references. Pinned by
  `test/sa_lookup_sentinel_parity_test.cpp` and `test/regression/compat_sa_sentinel.sh`.

## Changes catalog

| Item | bwa-mem3 PR | Upstream PR/issue | Status |
|------|-------------|-------------------|--------|
| `@PG CL:` tab escape | [#54](https://github.com/fg-labs/bwa-mem3/pull/54) | [bwa-mem2#293](https://github.com/bwa-mem2/bwa-mem2/issues/293) | fork-only (open upstream issue) |
| SMEM buffer overflow on >151 bp reads | [#55](https://github.com/fg-labs/bwa-mem3/pull/55) | [bwa-mem2#238](https://github.com/bwa-mem2/bwa-mem2/pull/238), [bwa-mem2#210](https://github.com/bwa-mem2/bwa-mem2/issues/210) | fork-only (upstream PR closed unmerged) |
| kswv nrow==0 guard | [#51](https://github.com/fg-labs/bwa-mem3/pull/51) | [bwa-mem2#289](https://github.com/bwa-mem2/bwa-mem2/pull/289) | fork-only (upstream PR open) |
| AVX-512BW dispatch guard | [#26](https://github.com/fg-labs/bwa-mem3/pull/26) | — | fork-only |
| AVX2 score2 plateau disable (superseded) | [#27](https://github.com/fg-labs/bwa-mem3/issues/27) | — | closed (superseded by #28) |
| AVX2 score2 plateau fix | [#28](https://github.com/fg-labs/bwa-mem3/pull/28) | — | fork-only |
| NEON + AVX-512BW 8-bit score2 fix | [#29](https://github.com/fg-labs/bwa-mem3/pull/29) | — | fork-only |
| AVX-512BW 16-bit score2 fix | [#30](https://github.com/fg-labs/bwa-mem3/pull/30) | — | fork-only |
| NEON 16-bit kernel rewrite | [#31](https://github.com/fg-labs/bwa-mem3/pull/31) | — | fork-only |
| 8-bit banded-SW envelope cap at `w = 124` | [#422](https://github.com/fg-labs/bwa-mem3/pull/422) | — | fork-only (only affects `-w >= 125`; default `-w 100` byte-identical) |
| Vector banded-SW z-drop gate at `-d 0` | [#424](https://github.com/fg-labs/bwa-mem3/pull/424) | — | fork-only (z-drop-disabled only; default `-d 100` byte-identical) |
| 16-bit banded-SW per-lane band clamp (wide arithmetic) | [#423](https://github.com/fg-labs/bwa-mem3/pull/423) | — | fork-only (non-default scoring only; default path unchanged — see the correctness note above) |
| `--meth` SEQ restore clamps unsupported bytes to `N` | [#463](https://github.com/fg-labs/bwa-mem3/pull/463) | — | fork-only (`--meth` only; changes a `-` byte to `N` on the `nst_nt4_table` lookup paths only — the direct-copy forward path, all other bytes, and the default path are unchanged) |
| Symmetric ref/query length bounds guards on the SW wrappers | [#467](https://github.com/fg-labs/bwa-mem3/pull/467) | — | fork-only (invalid-input contract; valid-input records byte-identical by construction — see the PR #467 correctness note above for the measured scope) |
| Banded-SW `qlen` band-clamp reach slot overflow (`int32_t` reach) | [#468](https://github.com/fg-labs/bwa-mem3/pull/468) | — | fork-only (8-bit widening defensive; 16-bit divergence only at non-default `-A >= 3` long-query, `len2 * A > 65535` — default alignment records unchanged for fixed batch composition, see the correctness note above) |
| kseq2bseq1 zero-initialization | [#22](https://github.com/fg-labs/bwa-mem3/pull/22) | — | fork-only |
| Proper-pair flag from emitted alignment | [#17](https://github.com/fg-labs/bwa-mem3/pull/17) | — | fork-only, **opt-in** (`--proper-pair-from-emitted`; default matches both upstreams, [#362](https://github.com/fg-labs/bwa-mem3/issues/362)) |
| All chains dropped by the weight filter: default follows bwa | [#489](https://github.com/fg-labs/bwa-mem3/pull/489) | — ([#310](https://github.com/fg-labs/bwa-mem3/issues/310)) | fork-only; `--compat=bwa-mem2` reproduces bwa-mem2 |
| Suffix-array sentinel offset in the prefetch lookup | [#469](https://github.com/fg-labs/bwa-mem3/pull/469) | — | fork-only; `--compat=bwa-mem2` reproduces bwa-mem2 |

---

**See also:**
[Performance improvements](performance.md) ·
[Architecture support](arch-support.md) ·
[Fork changes vs. upstream](../reference/pr-catalog.md) ·
[Developer Guide → Regression test framework](../developer-guide/regression-tests.md) ·
[Performance → SIMD dispatch matrix](../performance/simd-dispatch.md)
