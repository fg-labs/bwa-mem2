# What's Different from bwa-mem2

This section explains why bwa-mem3 diverges from upstream
[bwa-mem2/bwa-mem2](https://github.com/bwa-mem2/bwa-mem2)'s `master` branch:
curated pages that group the changes by category and give the rationale for each
class. Exhaustive per-change tracking lives in git history and the GitHub PR
list; a separate catalog records which fork changes were proposed upstream and
how each was received.

bwa-mem3 is **not byte-identical** to bwa-mem2. Upstream reproduces the original `bwa` exactly; bwa-mem3 does not — it emits extra SAM tags, fixes crashes and SIMD scoring bugs, and changes tie resolution. On the data tested, the core alignment (position, CIGAR, MAPQ, FLAG) is preserved, but the SAM byte stream is not. See [Equivalence with bwa-mem2](equivalence.md) for the field-by-field comparison.

## How this section is organized

Each page covers one category of change:

- [Alignment modes: plain, `--compat`, `--fast`](modes.md) — the three ways to run bwa-mem3 and how each one changes the output, side by side. Start here if you just want to know which mode to use.
- [Equivalence with bwa-mem2](equivalence.md) — what is and isn't preserved, with the verified concordance check and the declared-divergence catalog.
- [Correctness fixes](correctness.md) — upstream bugs fixed in bwa-mem3 (the kswv `score2` series, the proper-pair regression, the zero-init crash, the SMEM overflow, `@PG` tab-escaping).
- [Performance improvements](performance.md) — lockstep SMEM batching, batched `-H` ingestion, libsais FM-index construction, and the consolidated mapping speedups.
- [Features](features.md) — `--meth`, mimalloc, `--supp-rep-hard-cap`, `bwa-mem3 shm`, the `HN:i` tag, and `--bam=LEVEL`.
- [Architecture support](arch-support.md) — Linux ARM64/aarch64, `arch=avx512bw`, and the NEON / AVX2 kswv mate-rescue kernels.
- [Build & infrastructure](build-infra.md) — the doctest framework, version stamping, PGO targets, flag forwarding, and the CI matrix.

The exhaustive per-PR record is git history and the [GitHub PR list](https://github.com/fg-labs/bwa-mem3/pulls?q=is%3Apr+is%3Amerged+base%3Amain); the pages here explain the *why* behind each class. For the fork changes that were proposed back upstream — and where each stands — see [**Fork changes vs. upstream**](../reference/pr-catalog.md).

## Notable fork-level changes

- **Vendored mimalloc allocator**: `ext/mimalloc` is pinned at `v3.3.0` and
  linked into every binary by default (`USE_MIMALLOC=1`). Linux uses
  `--whole-archive` static linkage; macOS uses dyld-interposed shared linkage.
  `USE_MIMALLOC=1` is the supported and recommended default on all
  platforms; `USE_MIMALLOC=0` is provided as a best-effort opt-out and is
  CI-gated on Linux x86 only. See [Features](features.md) for details.

- **`--supp-rep-hard-cap INT`** (opt-in, default disabled): forces MAPQ=0 on
  supplementary alignments whose chain contains a seed with `>=INT` genome
  occurrences. Addresses the long-standing bwa/bwa-mem2 issue where a supp
  fragment that maps to many places standalone (e.g. a short read in a CCATCC
  repeat) inherits a high MAPQ from its primary because the supp's competing
  repetitive chains get filtered out during the full-read pipeline and
  therefore never contribute to its `sub`/`sub_n`. See
  [upstream #260](https://github.com/bwa-mem2/bwa-mem2/issues/260) for the
  reporter case. Primary MAPQ is unaffected; default output is byte-identical
  to stock bwa-mem2. Typical values are 5–20 (lower = more aggressive); the
  upstream #260 repro drops from MAPQ=60 to MAPQ=0 at `--supp-rep-hard-cap 18`.

- **`--proper-pair-from-emitted`** (opt-in, default disabled): derives the
  proper-pair `FLAG` bit (`0x2`) from the alignment bwa-mem3 actually emits
  rather than the top-scoring region. bwa and bwa-mem2 both use the top-scoring
  region, so this deviates from both and is a hard error with `--compat`
  (overridable via `--compat-allow-divergent`). It has
  no effect without a `.alt` sidecar — the two derivations differ only for reads
  with ALT hits. This was the default until
  [#362](https://github.com/fg-labs/bwa-mem3/issues/362); see
  [Correctness fixes](correctness.md).

## Version stamping

`PACKAGE_VERSION` (the value reported by `bwa-mem3 version` and written to
the `@PG VN:` SAM header field) is generated at build time by the Makefile
from `git describe --tags --dirty`, e.g. `v2.3-30-g61813ef` for a tree 30
commits past upstream tag `v2.3` at commit `61813ef`.

- No manual bumping required: cut a fresh release by tagging the commit
  (`git tag -a vX.Y-fg-labs.N -m ...`) and the next build picks it up.
- Builds where `git describe --tags` fails (source-tarball extractions, or
  shallow clones / checkouts with no tag reachable from `HEAD` — including
  CI's default `actions/checkout` fetch-depth of 1) fall back to the static
  `FG_LABS_VERSION_FALLBACK` in `Makefile`. Bump that when cutting a
  release that will be consumed as a tarball, or in CI artifacts.
- `src/version.h` is generated and `.gitignore`d; `make clean` removes it.

## Branching and update policy

- `master` tracks upstream unchanged.
- `main` is `upstream/master` plus the commits above. Rebased onto upstream roughly quarterly, or sooner when an upstream release we care about lands.
- Contributions go via PR targeting `main`. CI and CodeRabbit gate merges.
- A PR that proposes one of its changes back to an upstream project (bwa-mem2 or Heng Li's `lh3/bwa`), or learns the upstream disposition of one, should record it on the [Fork changes vs. upstream](../reference/pr-catalog.md) page. There is no longer a mandatory per-PR table row — git history and the GitHub PR list are the source of truth for the change list itself.

## Consuming

Clone this repo and check out `main`:

```bash
git clone https://github.com/fg-labs/bwa-mem3.git
cd bwa-mem3
git checkout main
```

Or vendor the branch into a downstream repo by pinning to a specific
commit (not the branch tip) so your build is reproducible.

## Relationship to upstream

We submit the generally-useful fixes and features carried here as PRs against
[bwa-mem2/bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) when the upstream
maintainers are actively merging; while they are not, fixes land here first
and we drop them from `main` once they appear upstream.

---

**See also:**
[Equivalence with bwa-mem2](equivalence.md) ·
[Correctness fixes](correctness.md) ·
[Performance improvements](performance.md) ·
[Features](features.md) ·
[Fork changes vs. upstream](../reference/pr-catalog.md) ·
[Developer Guide → Contributing](../developer-guide/contributing.md)
