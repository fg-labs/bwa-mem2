# Apple Silicon / NEON port

bwa-mem3 supports ARM64 (Apple Silicon and Linux aarch64) as a first-class build target. The port uses the [sse2neon](https://github.com/DLTcollab/sse2neon) translation shim as a baseline and replaces the most performance-critical SSE path with a native NEON intrinsic.

## Architecture overview

The ARM build compiles a single binary with a single NEON kernel TU. There is only one NEON instruction-set level on all current ARM64 CPUs, so the per-tier dispatch table used by the x86 single-binary build (see [Single-binary SIMD dispatch (x86)](launcher.md)) collapses to a one-entry switch on aarch64 — there is effectively no dispatch overhead. `make arm64` builds and installs the binary at the bare `bwa-mem3` name.

### sse2neon shim

`ext/sse2neon/sse2neon.h` is a header-only library that maps Intel SSE intrinsics to their NEON equivalents. When `APPLE_SILICON=1` is defined (set automatically when `uname -m` is `arm64` or `aarch64`), `src/simd_compat.h` includes sse2neon and defines the SSE feature test macros (`__SSE__` through `__SSE4_2__`) so that code guarded by those macros compiles without changes.

The translation is not zero-cost for all operations. One pattern that sse2neon handles poorly is replaced with native NEON in `src/simd_compat.h`:

- `_mm_movemask_epi16` — used heavily in `bandedSWA.cpp` to extract the sign bit of each 16-bit lane. The native implementation shifts right by 15, narrows to 8-bit with `vmovn_u16`, and reduces with position-weighted `vaddv_u8`.

Because the bulk of the ARM SIMD path is compiler-translated rather than hand-written intrinsics, codegen quality is unusually sensitive to the compiler and its version — a recent `clang` or `gcc` closes most of the gap to a hypothetical full native port. See [Best Practices → Build](../best-practices/build.md#arm--aarch64) for measured numbers and the recommendation.

### Memory alignment

Apple Silicon uses 128-byte cache lines (versus 64 bytes on x86). `simd_compat.h` overrides `_mm_malloc` on ARM to call `posix_memalign` with a minimum alignment of 128 bytes for all SIMD allocations. `CACHE_LINE_BYTES` is set to `128` in `macro.h` when `APPLE_SILICON=1`.

### Accelerate.framework

The Makefile links `-framework Accelerate` on macOS ARM builds. The framework is linked but not used for computation: bwa-mem3's hot paths (Smith-Waterman, FM-index) do not match the large-matrix / large-vector patterns that BLAS and vDSP target. The link is retained to keep the option open and adds no overhead at runtime.

### P-core / E-core detection

`src/fastmap.cpp` calls `HTStatus()` on macOS to detect the Apple Silicon microarchitecture. `HTStatus()` reads the `hw.perflevel0.physicalcpu` and `hw.perflevel1.physicalcpu` sysctl keys to report P-core and E-core counts and the L2 cache size (typically 4 MB on M-series chips). This information is printed at startup for diagnostic purposes. The L2 cache size is used to validate the compile-time `BATCH_SIZE` setting (currently 1024, which was already optimal for a 4 MB L2 cache).

## Benchmark results

All measurements use 100K paired-end reads, 5% error rate, 30% indels, chr17 reference, 8 threads, on an M-series Apple Silicon machine.

| Build | Wall-clock (avg, s) | vs. baseline |
|---|---|---|
| sse2neon baseline (no native NEON) | 15.4 | — |
| + native NEON `kswv.cpp` | 14.4 | ~7% faster |
| + native NEON `bandedSWA.cpp` movemask | 13.8 | ~4% faster |
| PGO on top of native NEON | ~13.4 | ~3% further |

The FM-index (`FMI_search.cpp`) is memory-bound with sequential pointer-chasing dependencies and does not benefit from SIMD. `libsais` benefits from OpenMP-parallel suffix-array construction but not from SIMD widening within a single thread.

## Optimization task summary

| Task | Status | Impact | Notes |
|---|---|---|---|
| Correctness verification | done | — | 200,006 alignments, 0 differences vs. reference |
| Dynamic L2 cache detection | done | ~0% | 4 MB detected; compile-time `BATCH_SIZE=1024` already optimal |
| Native NEON `bandedSWA.cpp` | done | ~4% | `_mm_movemask_epi16` in `simd_compat.h` |
| Per-tier dispatch table | N/A | 0% | Collapses to one entry on ARM (single NEON level) |
| Accelerate.framework | done | ~0% | Linked; no suitable compute patterns |
| M1/M2/M3/M4 detection | done | ~0% | P/E-core counts and L2 cache via sysctl |
| Native NEON `FMI_search.cpp` | N/A | 0% | Memory-bound; SIMD cannot help |
| Profile-Guided Optimization | done | ~3% | `make pgo-generate` / `make pgo-use` |

## Building for Apple Silicon

```bash
# Standard arm64 build
make arch=arm64

# PGO build (recommended for production on Apple Silicon)
make pgo-generate PGO_ARCH=arm64
./bwa-mem3.pgo-instr mem -t 8 ref.fa R1.fq.gz R2.fq.gz > /dev/null
make pgo-use PGO_ARCH=arm64
```

The resulting `bwa-mem3.pgo` binary delivers the full ~10% improvement over the pure sse2neon baseline.

> **Tip — Recommended production build on Apple Silicon**
>
> Use PGO for production deployments. The combined ~10% improvement from native NEON
> kernels plus PGO is consistent and verified on M-series hardware.

## Files modified in the NEON port

- `src/kswv.cpp`, `src/kswv.h` — native NEON batched Smith-Waterman
- `src/bandedSWA.h` — SIMD width definitions for ARM
- `src/simd_compat.h` — sse2neon integration, aligned allocation, `_mm_movemask_epi16`
- `src/fastmap.cpp` — L2 cache detection, `HTStatus()` for non-NUMA (macOS)
- `src/macro.h` — `BATCH_SIZE` and `CACHE_LINE_BYTES` tuning for Apple Silicon
- `Makefile` — `arm64` target, sse2neon flags, Accelerate linkage, PGO targets

---

**See also:**
[SIMD dispatch architecture](simd-dispatch.md) ·
[Building from source](building.md) ·
[Performance → PGO build](../performance/pgo.md) ·
[Performance → SIMD dispatch matrix](../performance/simd-dispatch.md) ·
[What's Different → Architecture support](../whats-different/arch-support.md)
