# version

`bwa-mem3 version` prints the release version, the compiler that built
the binary, the build's compiled-in SIMD floor, the SIMD tier resolved
at runtime, and (when mimalloc is compiled in) the mimalloc version. It
is the canonical way to confirm which build is on `PATH`, what toolchain
produced it, what host class it requires, and what kernel path it will
dispatch to.

`bwa-mem3 version` always exits 0 — even on a host below the build's
SIMD floor — so operators can introspect a binary on a host that cannot
actually run alignment. `bwa-mem3 <subcommand> --help` shares the same
property, as does `-h` for the subcommands where it is a help alias
(`index`, `shm`). Note that `-h` is **not** a help alias for `mem` — there
it is the XA-hits option (it takes an integer), so `bwa-mem3 mem -h N …`
runs alignment and is subject to the SIMD-floor precheck like any other
`mem` invocation.

## Synopsis

```text
{{#include ../../_generated/cli/version.txt}}
```

## Common usage

Confirm the installed version, SIMD floor, and resolved tier:

```bash
./bwa-mem3 version
```

A typical run on an AVX-512BW host with the default `BASELINE_ARCH=avx2`
build prints (every line below goes to stdout; only the host-below-floor
warning described further down goes to stderr):

```text
v0.3.0-12-gabcdef1
Compiler: clang 19.1.7
SIMD floor: avx2 (x86-64-v3, Haswell 2013+); kernels: sse41 sse42 avx avx2 avx512bw
SIMD runtime: avx512bw (BWAMEM3_FORCE_TIER unset)
mimalloc 3.3.0 (active)
```

- **version line** — bwa-mem3's release string, derived from
  `git describe` at build time and stored as `PACKAGE_VERSION` in the
  binary. When building from a tarball without git history, the
  fallback value is set via `FG_LABS_VERSION_FALLBACK` at compile time.
- **`Compiler:`** — the toolchain that compiled the binary, reported
  from the compiler's own preprocessor macros: `clang`, `gcc`, `icpx`
  (Intel oneAPI), or `icpc` (Intel Classic). bwa-mem3 runs ~5–10%
  faster on x86 built with clang than with g++, so this line is how you
  confirm a production binary came from the recommended toolchain — see
  [Best Practices → Build](../best-practices/build.md).
- **`SIMD floor:`** — the compile-time minimum the binary requires.
  Set by `BASELINE_ARCH` (default `avx2`) and listed alongside the
  per-tier kernel set the binary carries.
- **`SIMD runtime:`** — the tier resolved at startup by
  `__builtin_cpu_supports` (or `BWAMEM3_FORCE_TIER` if set in the
  environment). On arm64 this is always `neon`.
- **mimalloc line** — present only when `USE_MIMALLOC=1` (the default).
  The parenthetical reports whether mimalloc is actually intercepting
  `malloc`/`free` — `(active)` when it is, `(linked but NOT overriding
  malloc)` when the binary merely links a libmimalloc that exports the
  `mi_*` API without installing the allocator override.

On a host below the SIMD floor, `version` also writes a
`[W::bwa-mem3]` warning on **stderr** identifying the gap (and the
alignment subcommands will refuse to run with exit code 2 — see
[Host requirements](../getting-started/host-requirements.md) for the
exit-2 message format and rebuild instructions).

## Notes / Gotchas

> **Tip — `version | grep` is safe in CI**
>
> The whole banner — the version, `Compiler:`, `SIMD floor:`,
> `SIMD runtime:`, and mimalloc lines — goes to stdout; only the
> host-below-floor `[W::bwa-mem3]` warning goes to stderr. So
> `bwa-mem3 version | grep '^SIMD'` (or `grep '^Compiler:'` to assert the
> toolchain, `grep '^mimalloc'` to assert the allocator) works in CI
> scripts even on hosts that cannot run alignment. Use `2>/dev/null` to
> suppress the warning line if you want stdout only.
>
> **Tip — No mimalloc line means USE_MIMALLOC=0**
>
> If no mimalloc line appears, the binary was built without the bundled
> allocator (`make USE_MIMALLOC=0`). See
> [User Guide — Memory allocator (mimalloc)](../user-guide/allocator.md)
> for when this is appropriate.

---

**See also:**
[User Guide — Memory allocator (mimalloc)](../user-guide/allocator.md) ·
[Developer Guide — Release process](../developer-guide/release.md) ·
[Getting Started — Installation](../getting-started/installation.md) ·
[What's Different — Build & infrastructure](../whats-different/build-infra.md)
