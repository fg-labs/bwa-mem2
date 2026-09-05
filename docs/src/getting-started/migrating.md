# Coming from bwa or bwa-mem2

bwa-mem3's **command line is a drop-in for `bwa mem` (the original BWA) and
`bwa-mem2`** — same subcommand, same flags, same FASTQ inputs. The one catch is
the index: a `bwa-mem2` index is reused as-is, but a `bwa` (v1) index must be
rebuilt once (different format) before that unchanged command line applies.
Three things are worth knowing before you switch.

## 1. Rebuild the index (bwa) — or reuse it (bwa-mem2)

bwa-mem3 uses the bwa-mem2 index format (`ref.fa.bwt.2bit.64` + `ref.fa.pac`,
read on demand via *pac-fetch*).

- **Coming from `bwa` (v1):** your `bwa index` files use a **different**
  FM-index format (`.bwt` / `.sa`) and **cannot be used**. Rebuild once — it
  takes a few minutes and leaves your FASTA untouched:

  ```bash
  bwa-mem3 index ref.fa
  ```

- **Coming from `bwa-mem2`:** your existing index works **as-is, no rebuild**.
  bwa-mem3 reads the `.bwt.2bit.64` and `.pac` your `bwa-mem2 index` already
  produced; it ignores the `.0123` file (and no longer builds one — pass
  `index --emit-unpacked-ref` only if some other tool still needs it).

See [Indexing the reference](../user-guide/indexing.md) for details.

## 2. The command line is the same

Every `bwa mem` flag is accepted, so an existing invocation runs unchanged —
swap the binary and go:

```bash
# was: bwa mem -t 16 -R '@RG\tID:s1\tSM:s1' ref.fa R1.fq.gz R2.fq.gz > out.sam
bwa-mem3 mem -t 16 -R '@RG\tID:s1\tSM:s1' ref.fa R1.fq.gz R2.fq.gz > out.sam
```

bwa-mem3 adds a few flags on top (none change default behavior): `--bam[=N]`
(emit BAM directly), `--meth` (native bisulfite/EM-seq), `--supp-rep-hard-cap`
and `--min-ext-len` (opt-in tuning), and `--proper-pair-from-emitted` (opt-in;
moves `FLAG` `0x2` away from what bwa and bwa-mem2 emit, so it is a hard error
with `--compat`, overridable via `--compat-allow-divergent`). See the
[CLI reference](../cli/mem.md).

## 3. Output is equivalent, not byte-identical

Where each read maps — position, CIGAR, MAPQ, FLAG — is preserved on the data
we have tested, but the SAM byte stream is **not** identical to bwa/bwa-mem2:
bwa-mem3 emits a few additional tags, converges per-architecture SIMD
`score2`/MAPQ toward the scalar reference, and breaks ties deterministically.
If you validate against a previous bwa/bwa-mem2 release, expect (and audit)
these differences — see [Equivalence with bwa-mem2](../whats-different/equivalence.md)
for the field-by-field comparison and a per-PR trail. For a byte-clean `diff`
of the **alignment records** against a specific bwa/bwa-mem2 golden, run
`--compat=<target>` and exclude the `@PG` line from both sides — it still names
`bwa-mem3` and is unmatchable by construction; see
[Alignment modes: plain, `--compat`, `--fast`](../whats-different/modes.md) for
which mode fits which situation.

**Use `-K` on both sides when you validate.** The default batch size is
`chunk_size × -t` in all three tools, and each batch's `mem_pestat` insert-size
estimate comes from the reads in that batch — so a run at `-t 16` and a run at
`-t 32` can legitimately differ on a few records, in bwa-mem2 as much as in
bwa-mem3. Pinning `-K` to the same value on both binaries removes that variable
and makes the diff attributable to the aligner:

```bash
bwa-mem2 mem -t 16 -K 100000000 ref.fa R1.fq.gz R2.fq.gz > mem2.sam
bwa-mem3 mem -t 16 -K 100000000 ref.fa R1.fq.gz R2.fq.gz > mem3.sam
```

## Recommended migration sequence

1. **Install** bwa-mem3 — see [Installation](installation.md).
2. **Index** (rebuild from `bwa`, or reuse a `bwa-mem2` index — see above).
3. **Run the drop-in profile first** — no extra flags — and validate against
   your current pipeline so the only changed variable is the binary.
4. **Then opt into the recommended profile** (`-m 10 -y 0`) for bwa-mem3's best
   speed/accuracy trade-off. See
   [Settings profiles: drop-in vs recommended](../best-practices/settings-profiles.md).

---

**See also:**
[Installation](installation.md) ·
[Quick start: align paired-end FASTQs](quick-align.md) ·
[Settings profiles: drop-in vs recommended](../best-practices/settings-profiles.md) ·
[Equivalence with bwa-mem2](../whats-different/equivalence.md) ·
[What's Different from bwa-mem2](../whats-different/overview.md)
