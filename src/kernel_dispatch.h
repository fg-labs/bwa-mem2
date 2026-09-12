// src/kernel_dispatch.h
//
// Symbol-mangling header for multi-tier kernel builds.
//
// Each kernel TU (`bandedSWA.cpp`, `kswv.cpp`, `ksw.cpp`, `sam_encode.cpp`)
// is compiled multiple times under different `-m...` flags (sse41, sse42,
// avx, avx2, avx512bw on x86_64) with `-DKERNEL_VARIANT=_<suffix>` set on
// the compile line. This header renames every kernel-exported symbol to
// `<name><suffix>` so all per-tier object files can be linked into the same
// binary without symbol collision.
//
// When KERNEL_VARIANT is unset (arm64 build, single-tier x86 fallback),
// names pass through unchanged.
//
// Each kernel TU must include this header *before* any other project
// header that declares the symbols. Kernel headers (bandedSWA.h, kswv.h,
// ksw.h, sam_encode.h) must also include this header at the top so
// declarations and definitions agree.
//
// Non-kernel TUs (bwamem.cpp, fastmap.cpp, etc.) must NEVER define
// KERNEL_VARIANT — they call kernels indirectly through the dispatch
// tables/factories in simd_dispatch.cpp.

#ifndef BWAMEM3_KERNEL_DISPATCH_H
#define BWAMEM3_KERNEL_DISPATCH_H

#ifdef KERNEL_VARIANT
#  define BWAMEM3_PASTE2(a, b) a ## b
#  define BWAMEM3_PASTE(a, b)  BWAMEM3_PASTE2(a, b)

   /* C++ class names — banded SW kernel */
#  define BandedPairWiseSW    BWAMEM3_PASTE(BandedPairWiseSW,    KERNEL_VARIANT)

   /* C++ class names — vectorized KSW batch kernel */
#  define kswv                BWAMEM3_PASTE(kswv,                KERNEL_VARIANT)

   /* C linkage — ksw single-pair entry points (extend, global, align variants).
    * All six are exported from ksw.cpp and called from non-kernel TUs
    * (bwa.cpp, bwamem.cpp, bwamem_pair.cpp). Phase 6 will add C-linkage
    * dispatch wrappers in simd_dispatch.cpp so the unmangled name resolves
    * at link time. */
#  define ksw_extend2         BWAMEM3_PASTE(ksw_extend2,         KERNEL_VARIANT)
#  define ksw_global2         BWAMEM3_PASTE(ksw_global2,         KERNEL_VARIANT)
   /* Test-only: scalar reference behind ksw_global2 (see ksw.cpp), mangled
    * per tier like the rest so it links once per tier and is resolved by a
    * dispatch wrapper in simd_dispatch.cpp. Used only by the wavefront
    * byte-identity unit test. */
#  define ksw_global2_scalar_ref BWAMEM3_PASTE(ksw_global2_scalar_ref, KERNEL_VARIANT)
   /* Internal wavefront kernels (ksw_global2_wave.h, included only by ksw.cpp).
    * Static, so linking never needs these renamed — but each per-tier TU emits
    * a copy at a different source line (the AVX-512/AVX2/NEON blocks are
    * mutually exclusive), and gcov's default strict function-merge rejects one
    * demangled name mapped to two lines when the per-tier .gcda are merged.
    * Mangling per tier gives each copy a distinct name, mirroring the exported
    * kernels above. */
#  define ksw_g2_wave         BWAMEM3_PASTE(ksw_g2_wave,         KERNEL_VARIANT)
#  define ksw_g2_wave16       BWAMEM3_PASTE(ksw_g2_wave16,       KERNEL_VARIANT)
   /* Test-only per-thread wavefront-exec counter getters (ksw.cpp). Defined in
    * every tier TU and dispatched to the active tier by simd_dispatch.cpp, like
    * ksw_global2_scalar_ref. The int16 variant counts only int16-kernel entries. */
#  define ksw_g2_wave_exec_count BWAMEM3_PASTE(ksw_g2_wave_exec_count, KERNEL_VARIANT)
#  define ksw_g2_wave16_exec_count BWAMEM3_PASTE(ksw_g2_wave16_exec_count, KERNEL_VARIANT)
#  define ksw_g2_wave16_wmin  BWAMEM3_PASTE(ksw_g2_wave16_wmin,  KERNEL_VARIANT)
#  define ksw_g2_wave_wmin    BWAMEM3_PASTE(ksw_g2_wave_wmin,    KERNEL_VARIANT)
#  define ksw_g2_wave_zr_capacity BWAMEM3_PASTE(ksw_g2_wave_zr_capacity, KERNEL_VARIANT)
#  define ksw_extend          BWAMEM3_PASTE(ksw_extend,          KERNEL_VARIANT)
#  define ksw_global          BWAMEM3_PASTE(ksw_global,          KERNEL_VARIANT)
#  define ksw_align2          BWAMEM3_PASTE(ksw_align2,          KERNEL_VARIANT)
#  define ksw_align           BWAMEM3_PASTE(ksw_align,           KERNEL_VARIANT)

   /* C linkage — SAM seq/qual encoder */
#  define sam_encode_seq_fwd  BWAMEM3_PASTE(sam_encode_seq_fwd,  KERNEL_VARIANT)
#  define sam_encode_seq_rev  BWAMEM3_PASTE(sam_encode_seq_rev,  KERNEL_VARIANT)
#  define sam_encode_qual_rev BWAMEM3_PASTE(sam_encode_qual_rev, KERNEL_VARIANT)

   /* C linkage — per-tier kernel factories. Defined in each kernel TU
    * (compiled per-tier on x86); called from simd_dispatch.cpp to construct
    * the right per-tier concrete class without needing the class layout
    * visible at the dispatcher's translation unit. */
#  define make_bsw_kernel    BWAMEM3_PASTE(make_bsw_kernel,    KERNEL_VARIANT)
#  define make_kswv_kernel   BWAMEM3_PASTE(make_kswv_kernel,   KERNEL_VARIANT)
   /* Mat-aware 10-arg factory (issue 173): mangles to
    * make_kswv_kernel<VARIANT>_mat (e.g. make_kswv_kernel_avx2_mat) so the
    * tier suffix stays adjacent to the base name, matching the decls in
    * kswv.h and the make_kswv_kernel_<tier>_mat dispatch calls.
    *
    * Glue all three tokens in ONE paste via a dedicated helper. Routing the
    * bare `make_kswv_kernel` token through BWAMEM3_PASTE would argument-
    * prescan it — and `make_kswv_kernel` is itself an object-macro (above),
    * so it expands to make_kswv_kernel<VARIANT> first, doubling the suffix
    * into make_kswv_kernel<VARIANT><VARIANT>_mat (an undefined symbol vs. the
    * dispatcher's make_kswv_kernel<VARIANT>_mat call — an x86-only link error
    * arm64 cannot surface). The `##` operands here are NOT macro-expanded, so
    * the base name stays literal. (The 9-arg make_*_kernel macros escape this
    * only by being self-referential, hence blue-painted.) */
#  define BWAMEM3_KSWV_MAT_GLUE(v) make_kswv_kernel ## v ## _mat
#  define BWAMEM3_KSWV_MAT(v)      BWAMEM3_KSWV_MAT_GLUE(v)
#  define make_kswv_kernel_mat     BWAMEM3_KSWV_MAT(KERNEL_VARIANT)
#endif

#endif /* BWAMEM3_KERNEL_DISPATCH_H */
