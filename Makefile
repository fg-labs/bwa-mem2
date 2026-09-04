##/*************************************************************************************
##                           The MIT License
##
##   BWA-MEM2  (Sequence alignment using Burrows-Wheeler Transform),
##   Copyright (C) 2019  Intel Corporation, Heng Li.
##
##   Permission is hereby granted, free of charge, to any person obtaining
##   a copy of this software and associated documentation files (the
##   "Software"), to deal in the Software without restriction, including
##   without limitation the rights to use, copy, modify, merge, publish,
##   distribute, sublicense, and/or sell copies of the Software, and to
##   permit persons to whom the Software is furnished to do so, subject to
##   the following conditions:
##
##   The above copyright notice and this permission notice shall be
##   included in all copies or substantial portions of the Software.
##
##   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
##   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
##   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
##   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
##   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
##   SOFTWARE.
##
##Contacts: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>;
##                                Heng Li <hli@jimmy.harvard.edu> 
##*****************************************************************************************/

ifneq ($(portable),)
	STATIC_GCC=-static-libgcc -static-libstdc++
endif

EXE=		bwa-mem3
#CXX=		icpc
# Pair CC with CXX for the toolchains we know about so libsais.c (the only
# C TU in this build) doesn't silently fall back to make's default `cc`,
# which can drift from CXX (e.g. CXX=icpx + CC=gcc would mix toolchains).
ifeq ($(CXX), icpc)
	CC= icc
else ifeq ($(CXX), icpx)
	CC= icx
else ifeq ($(CXX), g++)
	CC= gcc
    # bwa-mem3 is consistently faster when built with clang than with g++
    # (~5-10% lower wall + CPU on x86; see
    # docs/src/best-practices/build.md). g++ is GNU make's default CXX, so a
    # bare `make` lands here — but the nudge is deliberately UNguarded by
    # $(origin CXX), so an explicit `CXX=g++` sees it too: the recommendation
    # holds either way. (The else-branch below guards its hint on
    # $(origin CC) because that one flags a real hazard — a defaulted CC
    # mixing toolchains — which an explicit CC cannot hit.) NOTE: space-indented, not
    # tab — a tab-prefixed $(warning) is parsed as a recipe by GNU make 3.81
    # ("commands commence before first target"). Matches the space-indented
    # warning in the else-branch below.
    $(warning bwa-mem3: building with g++. clang builds run ~5-10% faster on x86 — consider `make CXX=clang++ CC=clang`. See docs/src/best-practices/build.md.)
else ifeq ($(CXX), clang++)
	CC= clang
else ifeq ($(CXX), c++)
	# `c++` is GNU make's default and on every system we ship to it is
	# symlinked to the same toolchain as `cc`, so the pairing is safe.
	CC= cc
else
    # Wrappers (`ccache g++`), versioned binaries (`g++-13`), and
    # cross-compilers don't match any case above. Only warn if CC truly
    # is the make-default — operators who pin both CXX and CC explicitly
    # already know what they're doing and shouldn't see a spurious hint.
    ifeq ($(origin CC),default)
        $(warning Unrecognized CXX='$(CXX)'; CC will fall back to default '$(CC)' which may not match CXX. Set CC explicitly to avoid mixing toolchains for libsais.c.)
    endif
endif

# ---- Compiler version floor -------------------------------------------------
# bwa-mem3's shipping and benchmarked builds use clang-19 (bioconda forces
# clang; the performance benchmark suite builds with clang-19). Older toolchains
# produce materially slower binaries: clang-19 is ~5% faster than gcc-15 and
# ~15%+ faster than gcc-11 on ARM (~5-16% on x86), because older compilers emit
# redundant SIMD ops (e.g. extra blend mask-broadcasts) that a modern clang
# already elides. To keep an accidentally-slow build from being mistaken for
# bwa-mem3 being slow, a bare `make` on a below-floor compiler FAILS. Deliberate
# builds on old toolchains (packagers on older distros, compatibility CI, A/B
# tests) opt out with ALLOW_UNSUPPORTED_COMPILER=1.
#
# Thresholds are overridable (?=) so a distro/packager can set its own policy.
# Apple clang has its own, lower floor because its version numbers do not track
# upstream LLVM.
CLANG_MIN      ?= 19
GCC_MIN        ?= 15
APPLECLANG_MIN ?= 15

# The floor gates *builds* — clean-only goals invoke no compiler (they just
# `rm` artifacts) and so must not require a floor-passing toolchain. Without
# this, `make clean` on GNU make's default CXX (`g++`, i.e. gcc-13 on Ubuntu
# 24.04) trips the parse-time $(error) below and aborts — breaking both
# contributor cleanups and any CI step that runs `make clean` before a build
# with a pinned CXX (e.g. the kswv ASan steps in proto-neon-kswv.yml, which
# `make clean` bare, then `make ... CXX="ccache clang++-19"`). Enforce only when
# a real build is requested: a bare `make` (empty MAKECMDGOALS → the default
# build goal) or any goal list containing a non-clean target still enforces it;
# a goal list of clean targets only is exempt.
FLOOR_EXEMPT_GOALS := clean pgo-clean profile-clean lto-clean
ifeq ($(strip $(MAKECMDGOALS)),)
    ENFORCE_COMPILER_FLOOR := 1
else ifeq ($(strip $(filter-out $(FLOOR_EXEMPT_GOALS),$(MAKECMDGOALS))),)
    ENFORCE_COMPILER_FLOOR :=
else
    ENFORCE_COMPILER_FLOOR := 1
endif

ifeq ($(ENFORCE_COMPILER_FLOOR),1)
ifneq ($(ALLOW_UNSUPPORTED_COMPILER),1)
    # Family from the --version banner; major from -dumpversion (gcc/clang/Apple
    # clang all report a usable "<major>[.minor.patch]" there). If $(CXX) can't
    # be probed (missing, or an unrecognized toolchain), CXX_KIND stays empty and
    # the floor is not enforced — fail open, never block an unknown-but-working CXX.
    CXX_VER_1ST  := $(shell $(CXX) --version 2>/dev/null | head -1)
    CXX_VER_FULL := $(shell $(CXX) --version 2>/dev/null)
    CXX_MAJOR    := $(shell $(CXX) -dumpversion 2>/dev/null | cut -d. -f1)
    ifneq (,$(findstring Apple clang,$(CXX_VER_1ST)))
        CXX_KIND  := Apple clang
        CXX_FLOOR := $(APPLECLANG_MIN)
    else ifneq (,$(findstring clang,$(CXX_VER_1ST)))
        CXX_KIND  := clang
        CXX_FLOOR := $(CLANG_MIN)
    else ifneq (,$(findstring Free Software Foundation,$(CXX_VER_FULL)))
        CXX_KIND  := gcc
        CXX_FLOOR := $(GCC_MIN)
    else
        CXX_KIND  :=
    endif
    ifneq ($(CXX_KIND),)
    ifneq (,$(CXX_MAJOR))
    ifeq ($(shell test $(CXX_MAJOR) -lt $(CXX_FLOOR) 2>/dev/null && echo old),old)
        $(info )
        $(info bwa-mem3: unsupported compiler -- $(CXX_KIND) $(CXX_MAJOR) is below the floor ($(CXX_KIND) >= $(CXX_FLOOR)).)
        $(info )
        $(info   Modern compilers are REQUIRED for performance. clang-19 -- what bioconda and the)
        $(info   performance benchmarks ship -- is ~5% faster than gcc-15 and ~15%+ faster than gcc-11)
        $(info   on ARM (~5-16% on x86); older compilers emit redundant SIMD ops a modern clang elides.)
        $(info   An old-compiler build is needlessly slow and must not be benchmarked as representative.)
        $(info )
        $(info   Fix (recommended): install clang >= $(CLANG_MIN) and build with `make CXX=clang++ CC=clang`.)
        $(info   Or gcc >= $(GCC_MIN) (acceptable, ~5% slower on ARM).)
        $(info   Override (packagers / compat CI / deliberate A/B on old toolchains):)
        $(info       make ALLOW_UNSUPPORTED_COMPILER=1 ...)
        $(info )
        $(error unsupported compiler: $(CXX_KIND) $(CXX_MAJOR) < $(CXX_FLOOR); set ALLOW_UNSUPPORTED_COMPILER=1 to override)
    endif
    endif
    endif
endif
endif
# -----------------------------------------------------------------------------

# AddressSanitizer support for catching kswv rowMax / SIMD store overruns
# in regression tests (e.g. kswv_nrow_zero_test). Opt-in with `make ASAN=1 ...`
# Forces USE_MIMALLOC off: mimalloc's malloc override interposes before asan
# and the two can't coexist cleanly. Uses `override` so an explicit
# `USE_MIMALLOC=1` on the command line cannot re-enable the incompatible
# allocator under the sanitizer. CXXFLAGS picks up $(ASAN_FLAGS)
# unconditionally later in the file; it stays empty when ASAN is unset.
ifneq ($(strip $(ASAN)),)
    override USE_MIMALLOC = 0
    ASAN_FLAGS   = -fsanitize=address -fno-omit-frame-pointer -O1
    LDFLAGS     += -fsanitize=address
    CFLAGS      += $(ASAN_FLAGS)
endif

# MemorySanitizer support for catching uninitialized reads, specifically the
# padded-lane input fields the SW kernels read back (see
# kernel_padded_lane_uninit_test). Opt-in with `make MSAN=1 ...`. MSan is a
# clang-only instrumentation; it forces USE_MIMALLOC off via `override` (mimalloc's
# malloc override interposes before the sanitizer runtime, and `override` keeps a
# command-line USE_MIMALLOC=1 from re-enabling it) and reuses the ASAN_FLAGS
# slot so the flags reach CXXFLAGS/CFLAGS/the test link lines unconditionally.
# ASAN and MSAN are mutually exclusive; ASAN wins if both are set.
ifneq ($(strip $(MSAN)),)
ifeq ($(strip $(ASAN)),)
    override USE_MIMALLOC = 0
    ASAN_FLAGS   = -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -O1
    LDFLAGS     += -fsanitize=memory
    CFLAGS      += $(ASAN_FLAGS)
endif
endif

# TESTING_BUILD=1 defines BWAMEM3_TESTING which enables test-only injection
# hooks (BWAMEM3_TESTING_HOST_TIER env var read by simd_dispatch.cpp). Used
# by test/regression/host_floor_enforce.sh. Production builds leave this
# unset; integration tests build with `make TESTING_BUILD=1`.
ifneq ($(TESTING_BUILD),)
    CXXFLAGS += -DBWAMEM3_TESTING
endif

# mimalloc integration. Default on — see FG-MAIN.md.
# Override with USE_MIMALLOC=0 to build a stock bwa-mem3 without mimalloc.
USE_MIMALLOC ?= 1

# Detect architecture
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
# Treat macOS ("arm64") and Linux ("aarch64") as the same ARM build target.
IS_ARM := $(filter $(UNAME_M),arm64 aarch64)

# Where mimalloc lives and where its CMake build writes artifacts.
MIMALLOC_SRC   = ext/mimalloc
MIMALLOC_BUILD = $(MIMALLOC_SRC)/build

# Per-platform library basename. Linux: static archive. macOS: dynamic lib
# (mimalloc's malloc override on macOS requires a dylib + dyld interposing).
ifeq ($(UNAME_S),Darwin)
    MIMALLOC_LIB = $(MIMALLOC_BUILD)/libmimalloc.dylib
    MIMALLOC_CMAKE_FLAGS = -DMI_BUILD_SHARED=ON -DMI_BUILD_STATIC=OFF \
                           -DMI_BUILD_OBJECT=OFF -DMI_BUILD_TESTS=OFF \
                           -DMI_OVERRIDE=ON -DCMAKE_BUILD_TYPE=Release
else
    MIMALLOC_LIB = $(MIMALLOC_BUILD)/libmimalloc.a
    MIMALLOC_CMAKE_FLAGS = -DMI_BUILD_SHARED=OFF -DMI_BUILD_STATIC=ON \
                           -DMI_BUILD_OBJECT=OFF -DMI_BUILD_TESTS=OFF \
                           -DMI_OVERRIDE=ON -DCMAKE_BUILD_TYPE=Release
endif

# Link flags that inject mimalloc's malloc overrides.
# On Linux, --whole-archive forces the linker to keep symbols it would
# otherwise drop (malloc/free would come from libc first). On macOS, we
# just link the dylib; dyld interposes at load time — no --whole-archive
# equivalent is needed (and -force_load does NOT enable malloc interpose).
#
# On macOS we set two rpaths: @executable_path/. is the portable one used
# when the binary ships alongside libmimalloc.dylib; the $(abspath ...)
# rpath is a dev-only fallback that lets the binary run in-tree without
# first copying the dylib. For distribution, the @executable_path rpath
# resolves first and the abspath is harmlessly ignored (or can be removed
# with `install_name_tool -delete_rpath`).
ifeq ($(USE_MIMALLOC),1)
    ifeq ($(UNAME_S),Darwin)
        MIMALLOC_LDFLAGS = -L$(MIMALLOC_BUILD) -lmimalloc \
                           -Wl,-rpath,@executable_path/. \
                           -Wl,-rpath,$(abspath $(MIMALLOC_BUILD))
    else
        MIMALLOC_LDFLAGS = -Wl,--whole-archive $(MIMALLOC_LIB) -Wl,--no-whole-archive
    endif
    CPPFLAGS += -DUSE_MIMALLOC=1
else
    MIMALLOC_LDFLAGS =
endif

# ARM/Apple Silicon support
ifneq ($(IS_ARM),)
    ARCH_FLAGS = -DAPPLE_SILICON=1
    # sse2neon flags - define SSE feature macros for translation
    SSE2NEON_FLAGS = -D__SSE__=1 -D__SSE2__=1 -D__SSE3__=1 -D__SSSE3__=1 -D__SSE4_1__=1 -D__SSE4_2__=1
    SSE2NEON_INCLUDES = -Iext/sse2neon
    CPPFLAGS += $(SSE2NEON_FLAGS)
    INCLUDES += $(SSE2NEON_INCLUDES)
    # CPU tuning. Default is generic/portable scheduling — safe for any ARM core
    # including Apple Silicon, and what portable release binaries should ship.
    # Set ARM_CPU=<name> for a core-tuned build, e.g. ARM_CPU=neoverse-v2
    # (Graviton4) or ARM_CPU=native (match the build host). This goes in CPPFLAGS
    # (not ARCH_FLAGS) because ARCH_FLAGS is reset with `=` in the arch=arm64
    # branch below; CPPFLAGS uses `+=` and reaches every compile line.
    ifneq ($(ARM_CPU),)
        CPPFLAGS += -mcpu=$(ARM_CPU)
    endif
    # Cache-line size used as the minimum SIMD allocation alignment. Apple
    # Silicon uses 128-byte lines; Neoverse/Graviton use 64. Default 128 —
    # over-aligning on a 64-byte core is harmless. A tuned build overrides it
    # with ARM_CACHE_LINE=64.
    ARM_CACHE_LINE ?= 128
    # Reject unsupported alignments before they reach CACHE_LINE_BYTES. An
    # invalid value would make SIMD_ALIGNED_ALLOC call posix_memalign with a bad
    # alignment, which returns EINVAL/NULL and is then dereferenced unchecked in
    # the kswv constructor. CACHE_LINE_BYTES must be exactly one supported value:
    # guard the token count first ($(filter) matches per word, so a multi-token
    # value like "64 128" would slip past a bare membership test; this also
    # rejects empty), then the value itself. The supported values are the
    # $(filter) PATTERN and the user value is the text, not the reverse: a user
    # value used as the pattern could carry a `%` wildcard (e.g. ARM_CACHE_LINE=%)
    # that matches everything and passes the check. Fail at parse time with a
    # clear message instead of building a broken binary.
    ifneq ($(words $(ARM_CACHE_LINE)),1)
        $(error ARM_CACHE_LINE must be a single value, 64 or 128 (got '$(ARM_CACHE_LINE)'))
    endif
    ifeq ($(filter 64 128,$(ARM_CACHE_LINE)),)
        $(error ARM_CACHE_LINE must be 64 or 128 (got '$(ARM_CACHE_LINE)'))
    endif
    CPPFLAGS += -DCACHE_LINE_BYTES=$(ARM_CACHE_LINE)
    # Link Accelerate framework on macOS for potential BLAS/vecLib usage
    ifeq ($(UNAME_S),Darwin)
        LIBS_EXTRA = -framework Accelerate
    endif
else
    ARCH_FLAGS = -msse -msse2 -msse3 -mssse3 -msse4.1
endif

CPPFLAGS+=	-DENABLE_PREFETCH -DV17=1 -DMATE_SORT=0 -DLIBSAIS_OPENMP

# Profiling (--profile stage instrumentation) is a COMPILE-TIME opt-in. The
# default build omits it entirely (no --profile option, zero runtime overhead;
# sp_enabled() folds to a compile-time 0 and the hooks vanish). Build with
# `make STAGE_PROF=1` to compile in the real stage_prof implementation and
# expose --profile. (Distinct from the PGO `profile-build` target below.)
ifeq ($(strip $(STAGE_PROF)),1)
    CPPFLAGS += -DSTAGE_PROF=1
endif

# Version string for `bwa-mem3 version` and the @PG VN: field. The single
# source of truth is `version.txt` (rewritten by release-please on each
# release). The logic of "base version + optional git-describe dev suffix"
# lives in `scripts/version.sh` rather than an inline `$(shell ...)` so
# the multi-line case statement can stay readable and is independently
# testable. Tarball / shallow-clone builds with no .git/ get the bare
# base version; checkouts past or dirty at the tag get an informational
# dev suffix appended.
VERSION_STRING := $(shell scripts/version.sh)
# One path per line so adding or removing a single -I against this list
# is a one-line diff that never overlaps with edits to adjacent lines
# (e.g. the VERSION_STRING block above).
INCLUDES += -Isrc
INCLUDES += -Iext/htslib
INCLUDES += -Iext/libsais/include
ifeq ($(USE_MIMALLOC),1)
    INCLUDES += -Iext/mimalloc/include
endif

# libdeflate: used directly by src/fast_reader.c for BGZF block decode. htslib
# already links -ldeflate transitively; we add the <libdeflate.h> include path
# here. On macOS it lives under the Homebrew prefix, resolved dynamically
# (mirrors the libomp prefix detection below) so the build works on both Apple
# Silicon (/opt/homebrew) and Intel (/usr/local) hosts rather than hardcoding one.
ifeq ($(UNAME_S),Darwin)
    LIBDEFLATE_PREFIX ?= $(shell brew --prefix libdeflate 2>/dev/null)
    INCLUDES += -I$(LIBDEFLATE_PREFIX)/include
endif

# zlib-ng: src/fast_reader.c uses its native (zng_*) streaming inflate for the
# plain-gzip path (SIMD, ~2.3x stock zlib on both x86 and arm64). The native API
# (zng_* symbols, <zlib-ng.h>) => no symbol clash with the system zlib that
# htslib / the legacy gzFile reader still use.
#
# Vendored under ext/zlib-ng (git submodule, pinned to a zlib-ng release) and
# built as a static archive (libz-ng.a with zng_* symbols) via its own CMake
# system -- the same integration pattern as ext/mimalloc -- so the Batch/CI
# fleet does NOT depend on a host zlib-ng package. ZLIB_COMPAT=OFF keeps the
# zng_* names; BUILD_SHARED_LIBS=OFF yields the static archive.
#
# ZLIBNG_USE_VENDORED=1 (default) builds and links the vendored copy. Set it to
# 0 to fall back to a host-installed zlib-ng (Homebrew on macOS, system package
# on Linux) -- handy for dev hosts that already have it, mirroring how libdeflate
# is resolved on macOS.
ZLIBNG_USE_VENDORED ?= 1

ZLIBNG_SRC   = ext/zlib-ng
ZLIBNG_BUILD = $(ZLIBNG_SRC)/build
ZLIBNG_LIB   = $(ZLIBNG_BUILD)/libz-ng.a
# Static, native (non-zlib-compat) build with tests off. Matches mimalloc's
# CMAKE_BUILD_TYPE=Release static integration.
# ZLIB_ENABLE_TESTS=OFF skips the example/infcover/etc. test executables;
# ZLIBNG_ENABLE_TESTS / WITH_GTEST turn off the API + gtest suites. We only
# need libz-ng.a and the generated headers.
ZLIBNG_CMAKE_FLAGS = -DZLIB_COMPAT=OFF -DBUILD_SHARED_LIBS=OFF \
                     -DZLIB_ENABLE_TESTS=OFF -DZLIBNG_ENABLE_TESTS=OFF \
                     -DWITH_GTEST=OFF \
                     -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON

ifeq ($(ZLIBNG_USE_VENDORED),1)
    # zlib-ng.h ships in the source root and #includes zconf-ng.h /
    # zlib_name_mangling-ng.h, which CMake GENERATES into the build dir -- so
    # both paths are needed on the include line.
    INCLUDES += -I$(ZLIBNG_SRC) -I$(ZLIBNG_BUILD)
else ifeq ($(UNAME_S),Darwin)
    ZLIBNG_PREFIX ?= $(shell brew --prefix zlib-ng 2>/dev/null)
    INCLUDES += -I$(ZLIBNG_PREFIX)/include
endif

# libsais (pinned in ext/libsais; see submodule SHA): linear-time suffix
# array / BWT construction via SA-IS. Compiled with OpenMP so
# libsais_gsa_omp can run parallel induced-sorting. libomp is already a
# link dep of bwa-mem3's alignment paths; no new dep.
LIBSAIS_DIR    = ext/libsais
LIBSAIS_OBJS   = $(LIBSAIS_DIR)/src/libsais.o $(LIBSAIS_DIR)/src/libsais64.o
LIBSAIS_CFLAGS = -O3 -std=c99 -DLIBSAIS_OPENMP -I$(LIBSAIS_DIR)/include
ifeq ($(UNAME_S),Darwin)
    # Resolve at parse time but defer the missing-libomp error until the
    # libsais recipe actually runs, so `make clean`, `make print-mimalloc-config`,
    # etc. still work on hosts without libomp installed. The check below in
    # the libsais pattern rule produces the actionable hint when needed.
    LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
    LIBSAIS_OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    LIBSAIS_OPENMP_LIBS   = -L$(LIBOMP_PREFIX)/lib -lomp
else
    LIBSAIS_OPENMP_CFLAGS = -fopenmp
    LIBSAIS_OPENMP_LIBS   = -fopenmp
endif

# Same one-per-line shape as INCLUDES above. -L/-l pairs that name a
# specific library stay grouped (search path + lib are a unit), so
# adding or removing a dep is still a one-line diff.
LIBS  = -lpthread
LIBS += -lm
LIBS += -lz
LIBS += -L. -lbwa
LIBS += -Lext/htslib -lhts
LIBS += $(LIBSAIS_OPENMP_LIBS)
LIBS += $(STATIC_GCC)
LIBS += $(LIBS_EXTRA)
# libdeflate is a direct dependency of src/fast_reader.c (BGZF block decode).
ifeq ($(UNAME_S),Darwin)
    LIBS += -L$(LIBDEFLATE_PREFIX)/lib -ldeflate
else
    LIBS += -ldeflate
endif
# zlib-ng: direct dependency of src/fast_reader.c (streaming plain-gzip inflate).
# Vendored static archive by default; host package as an opt-out fallback.
ifeq ($(ZLIBNG_USE_VENDORED),1)
    LIBS += $(ZLIBNG_LIB)
else ifeq ($(UNAME_S),Darwin)
    LIBS += -L$(ZLIBNG_PREFIX)/lib -lz-ng
else
    LIBS += -lz-ng
endif
# Pull in htslib's transitive deps (-ldeflate when libdeflate is detected,
# bzlib / lzma / curl when those features are enabled) from the generated
# htslib_static.mk -- the same mechanism samtools uses (samtools'
# config.mk.in pulls in both static_LIBS and static_LDFLAGS).
#
# Without this, any htslib configure feature that auto-detects on the
# host -- notably libdeflate on Debian/Ubuntu where libdeflate-dev is
# the default install -- gives libhts.a unresolved symbols that fail
# the bwa-mem3 link step.
#
# Mechanism (subtle): on a clean tree, htslib_static.mk doesn't exist
# at parse time, so `-include` silently skips it. To populate
# HTSLIB_static_LIBS / HTSLIB_static_LDFLAGS in time for the link
# recipe we rely on GNU Make's "remade-makefiles" restart loop, which
# only fires when the include file is itself a target with a non-empty
# recipe in this Makefile. The rule is declared next to the `$(HTS_LIB)`
# recipe further down (where HTS_LIB is in scope and the build-time
# concerns sit together); the `@:` recipe there is required -- a
# prereq-only rule does NOT trigger restart.
-include ext/htslib/htslib_static.mk
LIBS    += $(HTSLIB_static_LIBS)
LDFLAGS += $(HTSLIB_static_LDFLAGS)
# Non-kernel objects: always compiled once at the baseline ISA and linked into
# libbwa.a on every build (arm64 and x86 alike).
OBJS=		src/fastmap.o src/bwtindex.o src/utils.o src/kthread.o \
			src/kstring.o src/bntseq.o src/bwamem.o src/seed_order.o src/profiling.o \
			src/compat_target.o \
			src/FMI_search.o src/fmi_seed_api.o src/read_index_ele.o src/bwamem_pair.o src/bwa.o \
			src/bwamem_extra.o src/kopen.o src/bam_writer.o src/meth_bam.o \
			src/meth_xm.o \
			src/packed_text.o src/fm_index_writer.o src/index_prelude.o \
			src/system.o src/libsais_build.o \
			src/bwa_shm.o src/bwa_hugepages.o src/simd_dispatch.o \
			src/fast_reader.o src/fast_reader_bseq.o src/fr_fastq.o src/stage_prof.o \
			src/smem_dedup.o src/lockstep_width.o src/read_memo.o

# Kernel TUs (bandedSWA, kswv, ksw, sam_encode) are compiled per-tier on x86
# and linked directly via KERNEL_TIER_OBJS_LINK. The dispatch wrappers in
# simd_dispatch.cpp provide the unmangled entry points. On arm64 there is only
# one tier (NEON, unmangled), so the baseline objects ARE the only copies.
# On x86, the baseline (unsuffixed) objects provide the concrete kswv/ksw/etc.
# class bodies needed by test binaries (e.g. kswv_nrow_zero_test) that
# instantiate these classes directly rather than going through the dispatcher.
KERNEL_BASELINE_OBJS = src/bandedSWA.o src/kswv.o src/ksw.o src/sam_encode.o
OBJS += $(KERNEL_BASELINE_OBJS)
BWA_LIB=    libbwa.a
HTS_LIB=    ext/htslib/libhts.a

# Standalone test binaries this Makefile builds and links itself, named once so
# .PHONY, `test:`, `clean` and the object list below all derive from one place.
# (The doctest suites under test/unit and test/integration are built by
# test/Makefile instead.) A name missing from this list would silently lose its
# generated header dependencies, so there must be exactly one copy of it.
STANDALONE_TESTS = kswv_nrow_zero_test kswv_freed_cell_test \
                   bandedswa_padding_test bandedswa_highzdrop_seed_test \
                   bandedswa_high_h0_zdrop_test shm_section_find_test \
                   shm_pack_round_trip_test shm_lock_destroy_test \
                   kt_for_pool_test bns_zero_calloc_test \
                   kernel_padded_lane_uninit_test bam_rec_scratch_test \
                   detect_sa_compx_test nst_nt4_decode_test \
                   read_arena_overflow_test packed_text_neg_nbases_test \
                   packed_text_overflow_nbases_test \
                   mem_gen_alt_zero_calloc_test srtgg_grow_multi_test \
                   kopen_pipe_status_test
STANDALONE_TEST_OBJS = $(STANDALONE_TESTS:%=test/%.o)

# shm_pack_round_trip_test is excluded from `test:` because it runs via
# test/shm_pack_round_trip_test.sh, which has to build a phiX index first.
STANDALONE_TESTS_IN_TEST_TARGET = $(filter-out shm_pack_round_trip_test,$(STANDALONE_TESTS))

# Architecture-specific builds (x86 only, ARM uses default from above)
ifeq ($(IS_ARM),)
ifeq ($(arch),sse41)
	ifeq ($(CXX), icpc)
		ARCH_FLAGS=-msse4.1
	else
		ARCH_FLAGS=-msse -msse2 -msse3 -mssse3 -msse4.1
	endif
else ifeq ($(arch),sse42)
	ifeq ($(CXX), icpc)
		ARCH_FLAGS=-msse4.2
	else
		ARCH_FLAGS=-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2
	endif
else ifeq ($(arch),avx)
	ifeq ($(CXX), icpc)
		ARCH_FLAGS=-mavx ##-xAVX
	else
		ARCH_FLAGS=-mavx
	endif
else ifeq ($(arch),avx2)
	ifeq ($(CXX), icpc)
		ARCH_FLAGS=-march=core-avx2 #-xCORE-AVX2
	else
		ARCH_FLAGS=-mavx2
	endif
else ifeq ($(arch),avx512)
	# Legacy alias for arch=avx512bw (preserved for backward compat with
	# pre-PR #16 invocations). Keep flags identical to the avx512bw branch
	# below — including the -mprefer-vector-width=256 / -qopt-zmm-usage=low
	# autovec cap. See the avx512bw branch for the rationale.
	ifeq ($(CXX), icpc)
		ARCH_FLAGS=-xCORE-AVX512 -qopt-zmm-usage=low
	else
		ARCH_FLAGS=-mavx512f -mavx512bw -mprefer-vector-width=256
	endif
else ifeq ($(arch),avx512bw)
	# Explicit BW target: double the lane width vs AVX2 (64x8-bit / 32x16-bit).
	# AVX-512BW implies AVX-512F; -mavx512bw alone enables BW+F on gcc/clang
	# but we list both flags for clarity.
	#
	# -mprefer-vector-width=256 (gcc/clang) / -qopt-zmm-usage=low (icpc):
	# keep AVX-512BW *capabilities* available (32 zmm registers, mask
	# registers, byte/word lane permutes, gather/scatter) but cap the
	# auto-vectorizer's preferred SIMD width at 256-bit. Two effects:
	#   1. AMD Zen 4 (c7a / Genoa) splits 512-bit AVX-512 ops into
	#      2x 256-bit µops per op. For short-trip-count auto-vec loops
	#      that's a regression — 2x latency without amortizing the
	#      reduced iteration count, plus more I-cache pressure. Capping
	#      at 256-bit keeps the compiler from widening those loops.
	#   2. Intel Sapphire Rapids (c7i / m7i) has native 512-bit
	#      execution but pays a ~3-5% AVX-512 frequency downclock under
	#      sustained heavy use, plus AVX-512↔AVX2 transition penalties
	#      when non-kernel TUs running 512-bit code call into
	#      explicitly-256-bit kernel TUs. Capping at 256-bit avoids
	#      both.
	# Empirical: c7a wgs-5M shm-warmed -4.4% wall vs avx2 baseline
	# (vanilla avx512bw was -2.2%, the cap adds another 2% on top).
	# c7i wgs-5M is a wash either way (-0.7%). The hand-tuned 512-bit
	# kernel TUs are unaffected — they use intrinsics, not auto-vec.
	# Canonical mitigation per FFmpeg, libvpx, ISPC docs.
	ifeq ($(CXX), icpc)
		ARCH_FLAGS=-xCORE-AVX512 -qopt-zmm-usage=low
	else
		ARCH_FLAGS=-mavx512f -mavx512bw -mprefer-vector-width=256
	endif
else ifeq ($(arch),native)
	ARCH_FLAGS=-march=native
else ifneq ($(arch),)
# To provide a different architecture flag like -march=core-avx2.
	ARCH_FLAGS=$(arch)
else
myall:single
DEFAULT_BUILD_GOAL = myall
endif
endif

# ARM64/Apple Silicon single-binary build
ifneq ($(IS_ARM),)
ifeq ($(arch),arm64)
    ARCH_FLAGS = -DAPPLE_SILICON=1
else ifeq ($(arch),)
myall:arm64
DEFAULT_BUILD_GOAL = myall
endif
endif

# State the default goal instead of inheriting "first target of the first rule".
# `myall` gets a rule only in the two no-`arch=` cases above; with `arch=` set on
# x86 — what CI's `make arch=avx2 CXX=g++ USE_MIMALLOC=1` does, and what `single:`
# recurses with — there is no myall rule, so the goal fell through to whatever
# rule came next. That was `all:` by luck of ordering, and adding any rule above
# it silently retargeted every bare `make`: the objects-depend-on-$(FLAGS_STAMP)
# rule below made the default goal `src/fastmap.o`, so the build compiled one
# object, exited 0, and left no binary at all.
DEFAULT_BUILD_GOAL ?= all
.DEFAULT_GOAL := $(DEFAULT_BUILD_GOAL)

CXXFLAGS+=	-g -O3 -std=gnu++14 -fpermissive $(ARCH_FLAGS) $(ASAN_FLAGS) $(LIBSAIS_OPENMP_CFLAGS) $(EXTRA_CXXFLAGS) #-Wall ##-xSSE2

# CXXFLAGS used for per-tier kernel TU compilation. Must NOT contain any
# -m... ISA flag — each per-tier rule appends the right one.
BASE_CXXFLAGS = -g -O3 -std=gnu++14 -fpermissive $(ASAN_FLAGS) $(LIBSAIS_OPENMP_CFLAGS) $(EXTRA_CXXFLAGS)

# Header-dependency generation. Every compile emits a sidecar .d listing the
# headers that TU actually included, which is `-include`d at the bottom of this
# file so a header edit rebuilds exactly the objects that read it.
#
# -MP adds a phony target for each header so a *deleted* header does not wedge
# the build with "No rule to make target". -MMD (not -MD) skips system headers,
# which never change under us and would triple the size of every .d.
#
# This replaces a `makedepend` block that was regenerated by hand and listed
# nothing for test/*.o or for four src TUs, letting an incremental build link
# objects compiled against different layouts of the same struct. test/Makefile
# generates dependencies this way for its framework/unit/integration objects,
# though its legacy per-binary rules still carry a hand-maintained list.
DEPFLAGS = -MMD -MP

# Per-tier ISA flag groups for kernel multi-tier compilation (x86_64 only).
KERNEL_FLAGS_sse41    = -msse4.1
KERNEL_FLAGS_sse42    = -msse4.2 -mpopcnt
KERNEL_FLAGS_avx      = -mavx -mpopcnt
KERNEL_FLAGS_avx2     = -mavx -mavx2 -mpopcnt
KERNEL_FLAGS_avx512bw = -mavx -mavx2 -mavx512f -mavx512bw -mpopcnt

# Source files compiled at every tier on x86_64.
KERNEL_SRCS = src/bandedSWA.cpp src/kswv.cpp src/ksw.cpp src/sam_encode.cpp

# All per-tier kernel objects. On x86_64 these are 5 tiers × 4 kernel TUs = 20 .o files.
ifneq ($(IS_ARM),)
    # arm64: single unsuffixed build (KERNEL_VARIANT unset). Already in $(OBJS).
    KERNEL_TIER_OBJS =
else
    KERNEL_TIER_OBJS = \
        $(patsubst src/%.cpp,src/%.sse41.o,$(KERNEL_SRCS)) \
        $(patsubst src/%.cpp,src/%.sse42.o,$(KERNEL_SRCS)) \
        $(patsubst src/%.cpp,src/%.avx.o,$(KERNEL_SRCS)) \
        $(patsubst src/%.cpp,src/%.avx2.o,$(KERNEL_SRCS)) \
        $(patsubst src/%.cpp,src/%.avx512bw.o,$(KERNEL_SRCS))
endif

# COVERAGE=1 augments CXXFLAGS/BASE_CXXFLAGS/LDFLAGS with --coverage and
# overrides -O3 with -O0 so gcov line numbers correspond 1:1 with source.
# Consumed by the CI `coverage` job; not part of any shipped binary.
#
# BASE_CXXFLAGS has to be augmented too, not just CXXFLAGS: the per-tier kernel
# objects ($(KERNEL_TIER_OBJS) — bandedSWA/kswv/ksw/sam_encode at every tier),
# the native kernel objects and the standalone test objects all compile with
# BASE_CXXFLAGS, so with CXXFLAGS alone the SW kernels report no coverage at
# all. -O0/--coverage are not -m... ISA flags, so the per-tier rules can still
# append their own.
COVERAGE_FLAGS = -O0 --coverage
ifneq ($(COVERAGE),)
    CXXFLAGS      += $(COVERAGE_FLAGS)
    BASE_CXXFLAGS += $(COVERAGE_FLAGS)
    LDFLAGS       += --coverage
endif

# Control build flag for the batched mate-rescue SW port on ARM.
# When set (e.g. `make arm64 DISABLE_BATCHED_MATESW=1`), the source gate for
# the new batched path falls through to the legacy scalar mem_sam_pe. Used by
# the proto-neon-kswv CI to A/B the same commit with the port on vs. off.
# Pass the caller-supplied value through verbatim so `DISABLE_BATCHED_MATESW=0`
# still selects the batched path (ifdef would be true even for =0).
ifneq ($(strip $(DISABLE_BATCHED_MATESW)),)
    CPPFLAGS += -DDISABLE_BATCHED_MATESW=$(DISABLE_BATCHED_MATESW)
endif

# $(STANDALONE_TESTS) is deliberately NOT listed here: those names are real
# linked executables with real prerequisites, and GNU Make treats a .PHONY file
# target as permanently out of date, so listing them relinked all nine on every
# invocation. Their own rules (and the generated .d header deps this Makefile
# now emits) already decide when a relink is needed.
.PHONY:all myall arm64 clean single all-single print-mimalloc-config test test-injection FORCE pgo-generate pgo-use pgo-clean profile-build profile-clean lto-build lto-clean docs docs-serve docs-cli docs-clean docs-install-tools
.SUFFIXES:.cpp .c .o

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# The C TUs in $(OBJS) (src/fast_reader*.c, src/fr_fastq.c). Declared as a
# suffix rule rather than three explicit recipes so a newly added src/*.c cannot
# silently fall through to make's built-in .c.o rule, which carries neither
# $(INCLUDES) (libdeflate, zlib-ng, project headers) nor $(DEPFLAGS).
.c.o:
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Per-tier kernel compile rules. Active only on x86_64 multi-tier builds.
src/%.sse41.o: src/%.cpp
	$(CXX) -c $(BASE_CXXFLAGS) $(KERNEL_FLAGS_sse41) $(CPPFLAGS) -DKERNEL_VARIANT=_sse41 $(INCLUDES) $(DEPFLAGS) $< -o $@

src/%.sse42.o: src/%.cpp
	$(CXX) -c $(BASE_CXXFLAGS) $(KERNEL_FLAGS_sse42) $(CPPFLAGS) -DKERNEL_VARIANT=_sse42 $(INCLUDES) $(DEPFLAGS) $< -o $@

src/%.avx.o: src/%.cpp
	$(CXX) -c $(BASE_CXXFLAGS) $(KERNEL_FLAGS_avx) $(CPPFLAGS) -DKERNEL_VARIANT=_avx $(INCLUDES) $(DEPFLAGS) $< -o $@

src/%.avx2.o: src/%.cpp
	$(CXX) -c $(BASE_CXXFLAGS) $(KERNEL_FLAGS_avx2) $(CPPFLAGS) -DKERNEL_VARIANT=_avx2 $(INCLUDES) $(DEPFLAGS) $< -o $@

src/%.avx512bw.o: src/%.cpp
	$(CXX) -c $(BASE_CXXFLAGS) $(KERNEL_FLAGS_avx512bw) $(CPPFLAGS) -DKERNEL_VARIANT=_avx512bw $(INCLUDES) $(DEPFLAGS) $< -o $@

NATIVE_KERNEL_OBJS = src/kswv.native.o src/bandedSWA.native.o

# Every object also depends on the compile flags themselves, for two reasons.
#
# Flags that change without any source changing must invalidate the objects
# built with the old ones. An object silently kept across an `arch=` or `ASAN=1`
# switch is the same class of bug the .d files fix, from the other side — and
# this build varies flags mostly from the command line (`arch=`, `ASAN=1`,
# `COVERAGE=1`, `EXTRA_CXXFLAGS=`, `CXX=`), which is why several targets below
# still wipe src/*.o by hand.
#
# It is also what bootstraps the .d files. A worktree built before dependency
# generation existed has objects but no .d, and nothing else in the graph would
# ever recompile them, so the tree would stay unprotected indefinitely. A tree
# with no stamp gets one, which forces exactly one full rebuild.
#
# Keyed on the flag *text*, not on this Makefile's mtime: a comment or target
# edit here (the common case — most Makefile commits touch no flag) must not
# cost a full rebuild, while a command-line flag change must. Same
# write-only-if-changed trick as src/version.h below.
FLAGS_STAMP = .build-flags
FLAGS_SIG = CXX=$(CXX)|CC=$(CC)|CXXFLAGS=$(CXXFLAGS)|BASE_CXXFLAGS=$(BASE_CXXFLAGS)|CFLAGS=$(CFLAGS)|CPPFLAGS=$(CPPFLAGS)|INCLUDES=$(INCLUDES)|LIBSAIS_CFLAGS=$(LIBSAIS_CFLAGS)|TIERS=$(KERNEL_FLAGS_sse41),$(KERNEL_FLAGS_sse42),$(KERNEL_FLAGS_avx),$(KERNEL_FLAGS_avx2),$(KERNEL_FLAGS_avx512bw)

# Passed through the ENVIRONMENT and expanded double-quoted in the recipe, not
# interpolated by make into single quotes. `printf '%s\n' '$(FLAGS_SIG)'` puts the
# flag text inside shell single quotes, so a flag that itself contains a `'` ends
# that quote early. Both outcomes are bad, and the quiet one is worse:
#
#   EXTRA_CXXFLAGS=-DTAG='x'    the quotes are stripped, so the stamp records
#                               -DTAG=x -- IDENTICAL to the stamp for a real
#                               -DTAG=x build. Two different flag sets collide on
#                               one stamp and switching between them rebuilds
#                               nothing, which is exactly the drift this stamp
#                               exists to catch.
#   EXTRA_CXXFLAGS=-DTAG=it's   an odd number of quotes leaves the string
#                               unterminated: `make: *** [.build-flags] Error 127`.
#
# With `export` + "$$FLAGS_SIG" the value never passes through shell quoting, so
# any flag text round-trips verbatim.
export FLAGS_SIG

# The $@.tmp scratch file is shared by concurrent makes in the same tree, the
# same way src/version.h's recipe below is; fix both together if that ever bites.
$(FLAGS_STAMP): FORCE
	@printf '%s\n' "$$FLAGS_SIG" > $@.tmp
	@if ! cmp -s $@.tmp $@ 2>/dev/null; then mv $@.tmp $@; else rm -f $@.tmp; fi

$(OBJS) $(KERNEL_TIER_OBJS) $(NATIVE_KERNEL_OBJS) $(STANDALONE_TEST_OBJS) $(LIBSAIS_OBJS): $(FLAGS_STAMP)

all:$(EXE)

# Note: simd_dispatch.cpp is compiled at the regular BASELINE_ARCH like
# every other non-kernel TU. An earlier draft of this PR compiled it at
# -march=x86-64 to keep the precheck path SIGILL-safe on too-old hosts,
# but that broke `g_build_tier` (a `static constexpr` in this TU derived
# from __AVX2__/__SSE4_1__/etc., which only get defined when the matching
# -m flag is in scope). With the override, every binary reported its
# floor as "scalar" and the precheck silently became a no-op.
#
# In practice the precheck path is scalar-only — std::call_once,
# integer comparisons, getenv, snprintf, fputs, exit — with no array
# loops the compiler could autovectorize. main.cpp's preamble before
# the precheck (argc check, rdtsc, sleep, argv strcmp scan) is the
# same shape: scalar branching + libc calls. Even at -mavx2 the
# compiler doesn't emit AVX2 for any of this, so the precheck actually
# fires on EVERY host below the build floor — sse42, sse41, scalar,
# cross-family — not just one tier below. Verified empirically with
# the BWAMEM3_TESTING_HOST_TIER injection test.

# Regenerate src/version.h on every invocation, but only touch the file
# (and thus trigger a main.o rebuild) when the string actually changed.
# Must be declared after `all:$(EXE)` so FORCE is never picked as the
# default goal when the caller supplies `arch=...` (which skips the
# `myall:` dispatch branch).
FORCE:
src/version.h: FORCE
	@printf '#ifndef BWA_MEM3_VERSION_H\n#define BWA_MEM3_VERSION_H\n#define PACKAGE_VERSION "%s"\n#endif\n' '$(VERSION_STRING)' > $@.tmp
	@if ! cmp -s $@.tmp $@ 2>/dev/null; then mv $@.tmp $@; else rm -f $@.tmp; fi

src/main.o: src/version.h
src/fastmap.o: src/version.h

# Baseline ISA tier for non-kernel TUs in the x86 single-binary build.
# Defaults to avx2: every host that runs bwa-mem3 in practice has AVX2
# (Haswell, 2013+; any host with AVX-512 also has AVX2), and dropping the
# baseline below avx2 measurably slows hot non-kernel paths (chain
# extension, FMI BWT walks, mate scoring) because the compiler can no
# longer auto-vectorize them at 256-bit width. Override to sse41 (or
# sse42, avx) for vintage hardware; the per-tier kernel objects are still
# compiled at every tier regardless, so kernel dispatch on lower-tier
# hosts continues to work.
BASELINE_ARCH ?= avx2

# Single-binary multi-tier build. All kernel TUs are compiled at every
# tier; the dispatcher picks the right per-tier subclass at runtime.
# Replaces the `multi` target's 5 sequential clean rebuilds + execv launcher.
.PHONY: single
single: $(if $(filter 1,$(USE_MIMALLOC)),$(MIMALLOC_LIB)) $(if $(filter 1,$(ZLIBNG_USE_VENDORED)),$(ZLIBNG_LIB))
ifneq ($(IS_ARM),)
	@echo "ARM64 detected - building single arm64 binary instead of multi-tier"
	$(MAKE) arm64
else
	$(MAKE) arch=$(BASELINE_ARCH) EXE=bwa-mem3 CXX="$(CXX)" KERNEL_TIER_OBJS_LINK="$(KERNEL_TIER_OBJS)" all-single
endif

# Internal: builds the single binary with the BASELINE_ARCH tier for
# non-kernel TUs and links all KERNEL_TIER_OBJS_LINK on top.
.PHONY: all-single
all-single: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) $(if $(filter 1,$(ZLIBNG_USE_VENDORED)),$(ZLIBNG_LIB)) $(KERNEL_TIER_OBJS_LINK) src/main.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) src/main.o $(KERNEL_TIER_OBJS_LINK) $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) $(MIMALLOC_LDFLAGS) -o $(EXE)

# ARM64/Apple Silicon build target - single binary, no multi-binary launcher needed
arm64:
	rm -f src/*.o $(BWA_LIB)
	$(MAKE) arch=arm64 EXE=bwa-mem3.arm64 CXX="$(CXX)" all
	ln -sf bwa-mem3.arm64 bwa-mem3


$(EXE):$(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) $(if $(filter 1,$(USE_MIMALLOC)),$(MIMALLOC_LIB)) $(if $(filter 1,$(ZLIBNG_USE_VENDORED)),$(ZLIBNG_LIB)) src/main.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) src/main.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) $(MIMALLOC_LDFLAGS) -o $@

# Regression test for issue 38 / upstream PR 289: exercises an all-len1==0
# batch that drives each SIMD kswv kernel through the nrow==0 path. Without
# the post-loop `if (i > 0)` guard, the rowMax store writes SIMD_WIDTH* bytes
# before the allocation and aborts at a later allocator operation; under
# asan the write is reported directly.
#
# On x86 multi-tier builds, libbwa.a's baseline kswv.o is compiled at the
# BASELINE_ARCH tier (avx2 by default; sse41 if overridden, in which case
# the SSE-only stub that calls exit() would fire). Compile a separate
# native-tier copy of kswv.cpp (src/kswv.native.o) and link it ahead of
# libbwa.a so the linker picks the host's native-ISA concrete kswv class
# regardless of BASELINE_ARCH. On arm64 -march=native resolves to the NEON
# path already covered by the baseline objects.
src/kswv.native.o: src/kswv.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

kswv_nrow_zero_test: $(BWA_LIB) $(HTS_LIB) src/kswv.native.o test/kswv_nrow_zero_test.o
	$(CXX) $(BASE_CXXFLAGS) -march=native $(LDFLAGS) test/kswv_nrow_zero_test.o src/kswv.native.o $(BWA_LIB) $(LIBS) -o $@

# Issue 173 / Task 2: mat-aware make_kswv freed-cell detection. Mirrors the
# kswv_nrow_zero_test link line (native-tier kswv ahead of libbwa.a so the
# host's concrete kswv class is linked) plus the doctest include path.
kswv_freed_cell_test: $(BWA_LIB) $(HTS_LIB) src/kswv.native.o test/kswv_freed_cell_test.o
	$(CXX) $(BASE_CXXFLAGS) -march=native $(LDFLAGS) test/kswv_freed_cell_test.o src/kswv.native.o $(BWA_LIB) $(LIBS) -o $@

# Native-tier copy of bandedSWA.cpp, linked ahead of libbwa.a for the same
# reason as src/kswv.native.o: on x86 multi-tier builds libbwa.a's baseline
# bandedSWA.o is compiled at BASELINE_ARCH (avx2), which lacks the 512-wide
# wrappers; the native copy guarantees the host's widest getScores8/16 (and
# their PFD8/PFD16 prefetch sites) are the ones these tests exercise.
# On arm64 -march=native resolves to the NEON baseline already.
src/bandedSWA.native.o: src/bandedSWA.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# getScores8/16 padding-lane / prefetch contract regression. Meaningful under
# ASan (`make ASAN=1 bandedswa_padding_test`): a tight-allocation caller that
# over-reads the SeqPair array aborts; the bounded prefetch runs clean.
bandedswa_padding_test: $(BWA_LIB) $(HTS_LIB) src/bandedSWA.native.o test/bandedswa_padding_test.o
	$(CXX) $(BASE_CXXFLAGS) -march=native $(LDFLAGS) test/bandedswa_padding_test.o src/bandedSWA.native.o $(BWA_LIB) $(LIBS) -o $@

# Uninitialized-padded-lane gate for the banded-SW kernels
# (BandedPairWiseSW::getScores8/16). Meaningful under MemorySanitizer
# (`make MSAN=1 kernel_padded_lane_uninit_test`) or Valgrind memcheck: it hands
# the kernels a RAW-malloc'd (never value-initialized) pair array so the padding
# lanes are genuinely uninitialized, and any tier that reads a padded-lane field
# it did not first write is reported as a use of an uninitialized value. Links
# the native-tier bandedSWA object so the host's widest getScores8/16 wrappers
# are the ones exercised. (The kswv counterpart is a follow-up -- see the test's
# header comment -- so this recipe links bandedSWA only.)
kernel_padded_lane_uninit_test: $(BWA_LIB) $(HTS_LIB) src/bandedSWA.native.o test/kernel_padded_lane_uninit_test.o
	$(CXX) $(BASE_CXXFLAGS) -march=native $(LDFLAGS) test/kernel_padded_lane_uninit_test.o src/bandedSWA.native.o $(BWA_LIB) $(LIBS) -o $@

# Unsigned h0-prefix seed regression: getScores8 vs scalar at zdrop > 126 with
# seed prefix bytes > 127 (the range the old signed seed could not represent).
bandedswa_highzdrop_seed_test: $(BWA_LIB) $(HTS_LIB) src/bandedSWA.native.o test/bandedswa_highzdrop_seed_test.o
	$(CXX) $(BASE_CXXFLAGS) -march=native $(LDFLAGS) test/bandedswa_highzdrop_seed_test.o src/bandedSWA.native.o $(BWA_LIB) $(LIBS) -o $@

# High-h0 / small-zdrop z-drop regression: getScores8 vs scalar with the seed
# score h0 above zdrop+1 at small zdrop (the region a relaxed 8-bit routing
# envelope would newly admit), where the z-drop drift's unset-best sentinel used
# to fire the z-drop one row early.
bandedswa_high_h0_zdrop_test: $(BWA_LIB) $(HTS_LIB) src/bandedSWA.native.o test/bandedswa_high_h0_zdrop_test.o
	$(CXX) $(BASE_CXXFLAGS) -march=native $(LDFLAGS) test/bandedswa_high_h0_zdrop_test.o src/bandedSWA.native.o $(BWA_LIB) $(LIBS) -o $@

# Build the test binaries with the same ARCH_FLAGS as libbwa.a so the
# test binary's kswv.h preprocessor state (SIMD_WIDTH8, BWA_TESTS_HAVE_KSWV)
# matches what libbwa.a was compiled with. Consumed by ci.yml so that e.g.
# the sse41 matrix row builds test/framework with -msse4.1 only (matching
# libbwa.a, which then lacks kswv::getScores8 — the BWA_TESTS_HAVE_KSWV
# macro guards the test away).
.PHONY: test-binaries
# $(HTS_LIB) is a real link-time dep: test/Makefile's bwa_mem3_tests_unit
# recipe references ../ext/htslib/libhts.a directly. Without this prereq,
# callers that skip the bwa-mem3 binary build (which builds it as a
# side-effect of $(EXE) deps) link-fail.
test-binaries: $(BWA_LIB) $(HTS_LIB)
	$(MAKE) -C test framework unit integration \
	    CXX="$(CXX)" \
	    COVERAGE=$(COVERAGE) \
	    ARCH_FLAGS_FROM_PARENT='$(ARCH_FLAGS)' \
	    ARM_CPU='$(ARM_CPU)' \
	    ARM_CACHE_LINE='$(or $(ARM_CACHE_LINE),128)' \
	    HTSLIB_static_LIBS='$(HTSLIB_static_LIBS)' \
	    HTSLIB_static_LDFLAGS='$(HTSLIB_static_LDFLAGS)'

shm_section_find_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/shm_section_find_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/shm_section_find_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@

detect_sa_compx_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/detect_sa_compx_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/detect_sa_compx_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@

shm_pack_round_trip_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/shm_pack_round_trip_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/shm_pack_round_trip_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@

shm_lock_destroy_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/shm_lock_destroy_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/shm_lock_destroy_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@

kt_for_pool_test: $(BWA_LIB) $(HTS_LIB) test/kt_for_pool_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/kt_for_pool_test.o $(BWA_LIB) $(LIBS) -o $@

# mem_chain2aln_across_reads_V2 srtgg growth-loop regression. Meaningful under
# ASan (`make ASAN=1 srtgg_grow_multi_test`): a single chain with a seed count
# past a single doubling of srtgg's initial capacity over-writes the buffer
# under the buggy single-doubling growth; the loop-until-it-fits fix runs
# clean. See the test's header comment.
srtgg_grow_multi_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/srtgg_grow_multi_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/srtgg_grow_multi_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@

# kopen_pipe_status_test links libbwa.a for kopen/kclose (src/kopen.o) and
# exercises the `<cmd` pipe path with synthetic producers; no fixture, no arch.
kopen_pipe_status_test: $(BWA_LIB) $(HTS_LIB) test/kopen_pipe_status_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/kopen_pipe_status_test.o $(BWA_LIB) $(LIBS) -o $@

# Standalone on purpose: it defines its own calloc so a zero-size request can
# return NULL, and that interposition must not reach any other test. See the
# header comment in the source.
bns_zero_calloc_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/bns_zero_calloc_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/bns_zero_calloc_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@

# Compile+link smoke test for the fmi_seed_api.h facade: proves the header's
# extern "C" section and every declared symbol actually resolve against
# libbwa.a, standing in for the external (e.g. minibwa) consumer build. Not
# part of $(STANDALONE_TESTS)/`test:` -- it is a link-surface check, not a
# behavioral regression test.
.PHONY: smoke
smoke: fmi_seed_api_smoke
	./fmi_seed_api_smoke

fmi_seed_api_smoke: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) $(if $(filter 1,$(USE_MIMALLOC)),$(MIMALLOC_LIB)) test/fmi_seed_api_smoke.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/fmi_seed_api_smoke.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) $(MIMALLOC_LDFLAGS) -o $@

test/fmi_seed_api_smoke.o: test/fmi_seed_api_smoke.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Regression test for the fmi_seed_api.h facade's max_occ guard: forwarding
# max_occ <= 0 into FMI_search::get_sa_entries_prefetch divides by max_occ.
# Requires a real index prefix argument (see test/run_unit_tests.sh), unlike
# the link-surface-only fmi_seed_api_smoke target above.
fmi_seed_api_guards_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) $(if $(filter 1,$(USE_MIMALLOC)),$(MIMALLOC_LIB)) test/fmi_seed_api_guards_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/fmi_seed_api_guards_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) $(MIMALLOC_LDFLAGS) -o $@

test/fmi_seed_api_guards_test.o: test/fmi_seed_api_guards_test.cpp $(FLAGS_STAMP)
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Unit test for the nst_nt4_decode helper (nt4 re-encoding, signed-char safety).
nst_nt4_decode_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/nst_nt4_decode_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/nst_nt4_decode_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@
test/nst_nt4_decode_test.o: test/nst_nt4_decode_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Alloc-guard regression tests: memory-safety guard paths that do not fire on
# valid input (see each source header for what old-vs-new behavior it pins).
read_arena_overflow_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/read_arena_overflow_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/read_arena_overflow_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@
test/read_arena_overflow_test.o: test/read_arena_overflow_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

packed_text_neg_nbases_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/packed_text_neg_nbases_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/packed_text_neg_nbases_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@
test/packed_text_neg_nbases_test.o: test/packed_text_neg_nbases_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

packed_text_overflow_nbases_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/packed_text_overflow_nbases_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/packed_text_overflow_nbases_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@
test/packed_text_overflow_nbases_test.o: test/packed_text_overflow_nbases_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

mem_gen_alt_zero_calloc_test: $(BWA_LIB) $(HTS_LIB) $(LIBSAIS_OBJS) test/mem_gen_alt_zero_calloc_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/mem_gen_alt_zero_calloc_test.o $(BWA_LIB) $(LIBSAIS_OBJS) $(LIBS) -o $@
test/mem_gen_alt_zero_calloc_test.o: test/mem_gen_alt_zero_calloc_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# fast_reader is C (not C++); the implicit .c rule omits $(INCLUDES), so give
# these objects explicit rules carrying the project include paths (incl.
# libdeflate) and preprocessor defines.
# fast_reader.c #includes <zlib-ng.h>, which pulls the CMake-generated
# zconf-ng.h / zlib_name_mangling-ng.h from the vendored build dir, so the
# vendored archive (which generates those headers as a side effect) must be
# built first. Gated on ZLIBNG_USE_VENDORED so the host-package fallback build
# doesn't try to build the submodule.
# The header prerequisites these three rules used to list by hand are now
# generated (see $(DEPFLAGS)); the hand lists were already incomplete —
# fast_reader.c includes stage_prof.h and fast_reader_bseq.c includes utils.h
# and stage_prof.h, none of which were listed. Only the extra non-header
# prerequisite remains: fast_reader.c needs the vendored zlib-ng archive built
# first for its CMake-generated headers.
src/fast_reader.o: src/fast_reader.c $(if $(filter 1,$(ZLIBNG_USE_VENDORED)),$(ZLIBNG_LIB))
src/fast_reader_bseq.o: src/fast_reader_bseq.c
src/fr_fastq.o: src/fr_fastq.c

# Standalone stage_prof helper unit test (stats + clocks + NaN init). Links only
# pthread + libm, so it needs no bwa-mem3/htslib build. Built with -DSTAGE_PROF
# so the real implementation (not the no-op inlines) is exercised.
stage_prof_test: src/stage_prof.cpp src/stage_prof.h test/stage_prof_test.cpp
	$(CXX) -O2 -std=gnu++14 -DSTAGE_PROF=1 -Isrc src/stage_prof.cpp test/stage_prof_test.cpp -lpthread -lm -o $@

# Standalone fast_reader correctness self-test (plain/gzip/multi-member/BGZF
# round-trips). Links only zlib + libdeflate, so it needs no bwa-mem3 build.
# Include / link flags shared by the standalone fast_reader test binaries.
# zlib-ng resolution matches the main build: vendored static archive by default
# (ext/zlib-ng build dir for the generated headers, libz-ng.a for the symbols),
# host package as the opt-out fallback.
FAST_READER_TEST_INC = -Isrc
FAST_READER_TEST_LIB =
ifeq ($(ZLIBNG_USE_VENDORED),1)
    FAST_READER_TEST_INC += -I$(ZLIBNG_SRC) -I$(ZLIBNG_BUILD)
    FAST_READER_TEST_ZLIBNG = $(ZLIBNG_LIB)
    FAST_READER_TEST_DEP    = $(ZLIBNG_LIB)
else
    FAST_READER_TEST_ZLIBNG = -lz-ng
endif
ifeq ($(UNAME_S),Darwin)
    FAST_READER_TEST_INC += -I$(LIBDEFLATE_PREFIX)/include
    FAST_READER_TEST_LIB += -L$(LIBDEFLATE_PREFIX)/lib
    ifneq ($(ZLIBNG_USE_VENDORED),1)
        FAST_READER_TEST_INC += -I$(ZLIBNG_PREFIX)/include
        FAST_READER_TEST_LIB += -L$(ZLIBNG_PREFIX)/lib
    endif
endif
fast_reader_selftest: src/fast_reader.c test/fast_reader_selftest.c $(FAST_READER_TEST_DEP)
	$(CC) -O2 -Wall -Wextra $(FAST_READER_TEST_INC) test/fast_reader_selftest.c src/fast_reader.c $(FAST_READER_TEST_LIB) -lz $(FAST_READER_TEST_ZLIBNG) -ldeflate -o $@

# Differential test: fr_fastq vs kseq, byte-identical record parsing. Links only
# zlib + zlib-ng + libdeflate (kseq.h is header-only), so it needs no bwa-mem3 build.
fr_fastq_diff_test: src/fr_fastq.c src/fast_reader.c test/fr_fastq_diff_test.c $(FAST_READER_TEST_DEP)
	$(CC) -O2 -Wall -Wextra $(FAST_READER_TEST_INC) test/fr_fastq_diff_test.c src/fr_fastq.c src/fast_reader.c $(FAST_READER_TEST_LIB) -lz $(FAST_READER_TEST_ZLIBNG) -ldeflate -o $@

# Read+parse microbenchmark: kseq vs fr_fastq on a real FASTQ (no index/align).
# Usage: ./fr_fastq_bench {kseq|frfastq} reads.fq[.gz] [reps]
fr_fastq_bench: src/fr_fastq.c src/fast_reader.c test/fr_fastq_bench.c $(FAST_READER_TEST_DEP)
	$(CC) -O2 -Wall -Wextra $(FAST_READER_TEST_INC) test/fr_fastq_bench.c src/fr_fastq.c src/fast_reader.c $(FAST_READER_TEST_LIB) -lz $(FAST_READER_TEST_ZLIBNG) -ldeflate -o $@

# Regression: bseq_read_fast must not overrun its buffer in paired-end mode when
# the initial capacity estimate is odd (a paired iteration writes two records
# but capacity is checked once). Always built with ASan so the overflowing write
# aborts on regression; links only the reader TUs (no bwa-mem3 build), like
# fr_fastq_diff_test. Built and run per-row by the CI matrix (ci.yml).
bseq_read_pe_oob_test: src/fr_fastq.c src/fast_reader.c src/fast_reader_bseq.c test/bseq_read_pe_oob_test.c $(FAST_READER_TEST_DEP)
	$(CC) -O1 -g -fsanitize=address -fno-omit-frame-pointer -Isrc -Wall -Wextra $(FAST_READER_TEST_INC) test/bseq_read_pe_oob_test.c src/fr_fastq.c src/fast_reader.c src/fast_reader_bseq.c $(FAST_READER_TEST_LIB) -lz $(FAST_READER_TEST_ZLIBNG) -ldeflate -o $@

# Coverage for bseq_read_fast's copy_comment/-C gate (bseq1_t.comment set only
# when the caller asked AND the record had a comment), across SE/PE. The parser
# is covered by fr_fastq_diff_test; this drives the adapter. Reader TUs only.
bseq_read_comment_copy_test: src/fr_fastq.c src/fast_reader.c src/fast_reader_bseq.c test/bseq_read_comment_copy_test.c $(FAST_READER_TEST_DEP)
	$(CC) -O2 -Isrc -Wall -Wextra $(FAST_READER_TEST_INC) test/bseq_read_comment_copy_test.c src/fr_fastq.c src/fast_reader.c src/fast_reader_bseq.c $(FAST_READER_TEST_LIB) -lz $(FAST_READER_TEST_ZLIBNG) -ldeflate -o $@
# kvec_alloc_fail_test -- forces the backing realloc to fail and asserts kvec.h's
# growth macros abort loudly (SIGABRT + an out-of-memory diagnostic) instead of
# leaking the old buffer and writing through the NULL that realloc returns.
# Header-only (kvec.h is standalone), so it links no bwa-mem3 objects, needs no
# index, and runs on every CI row. Forked: the abort is contained so the parent
# asserts *how* the child died -- a regression (a NULL-deref SIGSEGV, or silent
# continuation) fails the test rather than taking the process down with it.
kvec_alloc_fail_test: test/kvec_alloc_fail_test.c src/kvec.h
	$(CC) -O2 -Wall -Wextra -Isrc test/kvec_alloc_fail_test.c -o $@

test/shm_pack_round_trip_test.o: test/shm_pack_round_trip_test.cpp

# Run the in-tree tests via the unit-test harness in test/, plus the
# standalone regressions. shm_pack_round_trip_test runs via
# test/shm_pack_round_trip_test.sh which builds the phiX index first;
# invoked from test/run_unit_tests.sh.
#
# The standalone tests are RUN by looping over the same
# STANDALONE_TESTS_IN_TEST_TARGET that supplies the prerequisites, rather than
# from a second hand-maintained list: a name added to STANDALONE_TESTS would
# otherwise be built here and then silently never executed. The `echo` keeps the
# per-test line the unrolled form got from make's own recipe echoing.
#
# Note: depends on `bwa-mem3` so version_banner.sh has a binary to grep —
# previously `test:` only built the test harness binaries, not the main
# executable.
test: test-binaries $(STANDALONE_TESTS_IN_TEST_TARGET) kvec_alloc_fail_test bwa-mem3
	./test/bwa_mem3_tests_unit
	./test/bwa_mem3_tests_integration
	for t in $(STANDALONE_TESTS_IN_TEST_TARGET); do echo "./$$t"; ./$$t || exit 1; done
	./kvec_alloc_fail_test
	BWA_MEM3=./bwa-mem3 ./test/regression/version_banner.sh
	BWA_MEM3=./bwa-mem3 ./test/regression/meth_rescue_batched_identical.sh
	./test/regression/ndebug_gate_lint_selftest.sh
	./test/regression/ndebug_gate_lint.sh
	./test/regression/debug_macro_flag_lint_selftest.sh
	./test/regression/debug_macro_flag_lint.sh
	./test/regression/shell_lint_selftest.sh
	./test/regression/shell_lint.sh
	./test/regression/regression_coverage_lint_selftest.sh
	./test/regression/regression_coverage_lint.sh
	./test/regression/readme_contract_lint_selftest.sh
	./test/regression/readme_contract_lint.sh

# Shell lint on its own, and its autofix. Separate from `test` because the
# whole point is to run them without a build: both are source-only and finish
# in seconds, so `make shell-fix` is the answer to a red shell-lint job.
# Both no-op with a visible SKIP when shellcheck/shfmt are not installed.
.PHONY: shell-lint shell-fix
shell-lint:
	./test/regression/shell_lint_selftest.sh
	./test/regression/shell_lint.sh

shell-fix:
	./test/regression/shell_lint.sh --fix

# Regression test that requires a binary built with TESTING_BUILD=1
# (enables BWAMEM3_TESTING_HOST_TIER env-var injection). Not invoked by
# the default `test` target because it requires a non-production build.
# CI invokes this target on the canonical row, after rebuilding in-row with
# TESTING_BUILD=1 (see "SIMD floor enforcement" in .github/workflows/ci.yml).
# test/regression/regression_coverage_lint.sh is the backstop: it ignores
# comments, so the step's explanatory prose naming host_floor_enforce.sh no
# longer stands in for the invocation, and deleting the `make test-injection`
# line turns the lint red.
test-injection: bwa-mem3
	BWA_MEM3_TESTING=./bwa-mem3 INJECTED_TIER=sse41 PARITY_FA=/dev/null \
		./test/regression/host_floor_enforce.sh

test/kswv_nrow_zero_test.o: test/kswv_nrow_zero_test.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Carries -Iext so the doctest-based test resolves `#include "doctest/doctest.h"`
# (matching test/unit/*; -Iext/doctest would shadow the C++ <version> header).
# kswv_nrow_zero_test does not use doctest, hence its rule above omits this.
test/kswv_freed_cell_test.o: test/kswv_freed_cell_test.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) -Iext $(DEPFLAGS) $< -o $@

test/bandedswa_padding_test.o: test/bandedswa_padding_test.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/bandedswa_highzdrop_seed_test.o: test/bandedswa_highzdrop_seed_test.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/bandedswa_high_h0_zdrop_test.o: test/bandedswa_high_h0_zdrop_test.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Must be built -march=native like its sibling banded-SW tests: it links against
# src/bandedSWA.native.o (host SIMD widths), and getScores8/16 write padding
# lanes sized by SIMD_WIDTH8/16. The generic .cpp.o rule would compile this TU at
# the baseline (SSE4.1, width 16), so on an AVX2/AVX-512 host the native kernel
# would write padding beyond the test's baseline-sized allocation -- an OOB
# access. Compiling the test at -march=native keeps its widths in lockstep.
test/kernel_padded_lane_uninit_test.o: test/kernel_padded_lane_uninit_test.cpp
	$(CXX) -c $(BASE_CXXFLAGS) -march=native $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/shm_section_find_test.o: test/shm_section_find_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/detect_sa_compx_test.o: test/detect_sa_compx_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/shm_lock_destroy_test.o: test/shm_lock_destroy_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/kt_for_pool_test.o: test/kt_for_pool_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/srtgg_grow_multi_test.o: test/srtgg_grow_multi_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/kopen_pipe_status_test.o: test/kopen_pipe_status_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

test/bns_zero_calloc_test.o: test/bns_zero_calloc_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Header-only: exercises BamRecScratch::grow() geometric growth. Uses no lib
# symbols (POD kstring_t + malloc/free), so it links against libc alone.
bam_rec_scratch_test: test/bam_rec_scratch_test.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) test/bam_rec_scratch_test.o -o $@

test/bam_rec_scratch_test.o: test/bam_rec_scratch_test.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(DEPFLAGS) $< -o $@

# Archive both the baseline (unmangled) kernel objects from $(OBJS) and the
# per-tier (mangled) objects from $(KERNEL_TIER_OBJS) into libbwa.a. The
# duplication with $(KERNEL_TIER_OBJS_LINK) on the all-single recipe is
# intentional: legacy single-arch builds (`make arch=avx2`, `make arch=sse41`,
# etc.) link only libbwa.a on the binary line and rely on libbwa.a to satisfy
# simd_dispatch.o's references to make_kswv_kernel_<tier>, ksw_*_<tier>, and
# sam_encode_*_<tier>. Dropping per-tier objects from libbwa.a breaks those
# matrix rows. On arm64 $(KERNEL_TIER_OBJS) is empty.
$(BWA_LIB):$(OBJS) $(KERNEL_TIER_OBJS)
	ar rcs $(BWA_LIB) $(OBJS) $(KERNEL_TIER_OBJS)

# htslib: minimal configure (no lzma/bz2/curl/S3/GCS/plugins), zlib only.
# Guard on config.mk (only created by ./configure) rather than Makefile, which
# is checked into the htslib tree and would make the guard a no-op.
$(HTS_LIB):
	cd ext/htslib && \
	    ([ -f config.mk ] || (autoreconf -i && \
	        ./configure --disable-lzma --disable-libcurl --disable-gcs \
	                    --disable-s3 --disable-plugins --disable-bz2)) && \
	    $(MAKE) libhts.a htslib_static.mk

# Companion to the `-include ext/htslib/htslib_static.mk` near LIBS:
# declare the include file as a target with a non-empty recipe so GNU
# Make's "remade-makefiles" restart loop fires on the first parse,
# builds $(HTS_LIB) (which generates htslib_static.mk as a side effect
# of the `make libhts.a htslib_static.mk` invocation above), then
# re-parses this Makefile so HTSLIB_static_LIBS / HTSLIB_static_LDFLAGS
# resolve in time for the link recipe. `@:` is required -- without a
# recipe, Make doesn't consider the include file "remade" and skips
# the restart, leaving the include variables empty at link time.
# Must be placed after the default goal (myall / all) so it doesn't
# accidentally become the default goal of `make` with no arguments.
ext/htslib/htslib_static.mk: $(HTS_LIB)
	@:

# libsais: compile the two C sources we use (libsais.c + libsais64.c) as
# plain .o files. OpenMP enabled via LIBSAIS_OPENMP so libsais64_gsa_omp
# can run parallel induced-sorting.
#
# CFLAGS / CPPFLAGS / ASAN_FLAGS are forwarded so that ASAN builds
# instrument libsais and so that package-manager-supplied flags
# (Conda/Homebrew/distro) reach the libsais TU. LIBSAIS_CFLAGS is appended
# last so its -O3/-std=c99 take precedence over any user override.
#
# $(DEPFLAGS) applies here too: libsais64.c includes libsais.h, so a submodule
# bump touching only a shared header would otherwise leave one of the two SA
# objects compiled against the old one. The .d files land inside ext/libsais
# beside the .o files that are already there, and `clean` removes both.
$(LIBSAIS_DIR)/src/%.o: $(LIBSAIS_DIR)/src/%.c
	@if [ ! -f $(LIBSAIS_DIR)/include/libsais64.h ]; then \
	    echo "ERROR: $(LIBSAIS_DIR) is empty. Run: git submodule update --init --recursive"; \
	    exit 1; \
	fi
	@if [ "$(UNAME_S)" = "Darwin" ] && [ -z "$(strip $(LIBOMP_PREFIX))" ]; then \
	    echo "ERROR: libomp not found. Install with 'brew install libomp', or set LIBOMP_PREFIX to its install prefix."; \
	    exit 1; \
	fi
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $(ASAN_FLAGS) $(LIBSAIS_CFLAGS) $(LIBSAIS_OPENMP_CFLAGS) $(DEPFLAGS) $< -o $@

# Build mimalloc via its own CMake system. Shells out to cmake once and
# caches the build tree under ext/mimalloc/build. This rule always builds
# when invoked; USE_MIMALLOC=0 consumers simply don't depend on it.
$(MIMALLOC_LIB):
	@if [ ! -f $(MIMALLOC_SRC)/CMakeLists.txt ]; then \
		echo "ERROR: $(MIMALLOC_SRC) is empty. Run: git submodule update --init --recursive"; \
		exit 1; \
	fi
	mkdir -p $(MIMALLOC_BUILD)
	cd $(MIMALLOC_BUILD) && cmake $(MIMALLOC_CMAKE_FLAGS) .. && $(MAKE)

# Build zlib-ng via its own CMake system, mirroring the mimalloc rule above.
# Shells out to cmake once and caches the build tree under ext/zlib-ng/build;
# produces libz-ng.a (zng_* symbols) plus the generated zconf-ng.h /
# zlib_name_mangling-ng.h headers in the build dir. Only depended on when
# ZLIBNG_USE_VENDORED=1 (the default).
$(ZLIBNG_LIB):
	@if [ ! -f $(ZLIBNG_SRC)/CMakeLists.txt ]; then \
		echo "ERROR: $(ZLIBNG_SRC) is empty. Run: git submodule update --init --recursive"; \
		exit 1; \
	fi
	mkdir -p $(ZLIBNG_BUILD)
	cd $(ZLIBNG_BUILD) && cmake $(ZLIBNG_CMAKE_FLAGS) .. && $(MAKE)

clean: pgo-clean profile-clean lto-clean
	rm -fr src/*.o src/*.d src/version.h test/*.o test/*.d $(FLAGS_STAMP) $(BWA_LIB) $(EXE) $(STANDALONE_TESTS) kvec_alloc_fail_test bwa-mem3.arm64
	rm -f $(LIBSAIS_OBJS) $(LIBSAIS_OBJS:.o=.d)
	rm -f src/*.gcno src/*.gcda
	$(MAKE) -C test clean
	-[ -f ext/htslib/config.mk ] && cd ext/htslib && $(MAKE) distclean
	rm -rf $(MIMALLOC_BUILD)
	rm -rf $(ZLIBNG_BUILD)

# ----------------------------------------------------------------------------
# Documentation (mdbook). See docs/superpowers/specs/ for design.
# ----------------------------------------------------------------------------

# Sub-commands whose --help is captured into docs/_generated/cli/.
#
# The capture below strips or normalizes every HOST-dependent line, so the
# committed snippets are reproducible on any machine and the CI drift check
# (`git diff --exit-code docs/_generated/`) stays meaningful. Dropped outright:
# timing, launcher chatter, and [W:: warnings — none of which are part of the
# documented output. Normalized to a placeholder (kept, because they ARE part
# of the canonical output documented in docs/src/cli/version.md, which includes
# this capture as its synopsis):
#   - the version line (git describe),
#   - the `Compiler:` line (`version` reports the toolchain that built the
#     binary, so it reads `clang X.Y.Z` locally and `gcc X.Y.Z` on a g++ CI
#     runner),
#   - the `SIMD floor:` / `SIMD runtime:` lines (vary by build arch and host
#     CPU: avx2/avx512bw on x86, neon on arm64).
# Any future host-dependent line must be added here too.
DOCS_CLI_SUBCMDS := index mem shm version

docs:
	cd docs && mdbook build

docs-serve:
	cd docs && mdbook serve --open

docs-cli: $(EXE)
	@mkdir -p docs/_generated/cli
	@for sub in $(DOCS_CLI_SUBCMDS); do \
		echo "  CAPTURE  docs/_generated/cli/$$sub.txt"; \
		./$(EXE) $$sub --help 2>&1 \
			| sed -e 's/[[:space:]]*$$//' \
			| grep -v '^Total time taken:' \
			| grep -v '^Looking to launch ' \
			| grep -v '^Launching executable ' \
			| grep -v '^\[W::' \
			| awk '/^v?[0-9]+\.[0-9]+/ {print "v<MAJOR.MINOR>-<N>-g<COMMIT>"; next} \
			       /^Compiler: / {print "Compiler: <TOOLCHAIN> <VERSION>"; next} \
			       /^SIMD floor: / {print "SIMD floor: <FLOOR-TIER> (<HOST-CLASS>); kernels: <KERNEL-LIST>"; next} \
			       /^SIMD runtime: / {print "SIMD runtime: <RUNTIME-TIER> (<FORCE-TIER-STATE>)"; next} {print}' \
			> docs/_generated/cli/$$sub.txt; \
	done

docs-clean:
	rm -rf docs/book

docs-install-tools:
	cargo install mdbook --version 0.5.2 --locked
	cargo install mdbook-mermaid --version 0.17.0 --locked
	# Broken-link checker backend (see docs/book.toml). The mdbook-linkcheck2
	# fork supports mdBook 0.5; the original mdbook-linkcheck does not.
	cargo install mdbook-linkcheck2 --version 0.12.2 --locked

# Profile-Guided Optimization (PGO) targets.
#
# Usage (host-default arch, single shared profile dir — preserves prior
# arm64 behavior on Apple Silicon / aarch64 hosts):
#   make pgo-generate && <run training workload> && make pgo-use
#
# Multi-arch / multi-regime usage (override at command line):
#   make pgo-generate PGO_ARCH=avx2 PGO_PROFILE_DIR=/path/to/regimeA
#   <run training>
#   make pgo-use PGO_ARCH=avx2 PGO_PROFILE_DIR=/path/to/regimeA
#
# PGO_ARCH accepts the same values as the top-level `arch=` knob: arm64,
# sse41, sse42, avx, avx2, avx512, avx512bw, native, or any custom flag
# string. Defaults match the host: arm64 on Apple Silicon / aarch64,
# native otherwise. Output binaries are arch-suffixed when PGO_ARCH is
# non-default, so multiple per-arch builds coexist:
#   PGO_ARCH=arm64  -> bwa-mem3.pgo-instr,    bwa-mem3.pgo
#   PGO_ARCH=avx2   -> bwa-mem3.pgo-instr.avx2, bwa-mem3.pgo.avx2
ifneq ($(IS_ARM),)
    PGO_ARCH ?= arm64
else
    PGO_ARCH ?= native
endif
PGO_PROFILE_DIR ?= pgo_profiles

# Output names: keep the bare names when PGO_ARCH is the default arm64
# (backward-compat); arch-suffix otherwise so per-arch outputs don't collide.
ifeq ($(PGO_ARCH),arm64)
    PGO_INSTR_EXE = bwa-mem3.pgo-instr
    PGO_FINAL_EXE = bwa-mem3.pgo
else
    PGO_INSTR_EXE = bwa-mem3.pgo-instr.$(PGO_ARCH)
    PGO_FINAL_EXE = bwa-mem3.pgo.$(PGO_ARCH)
endif

pgo-generate:
	rm -f src/*.o $(BWA_LIB)
	$(MAKE) arch=$(PGO_ARCH) EXE=$(PGO_INSTR_EXE) EXTRA_CXXFLAGS="-fprofile-generate=$(PGO_PROFILE_DIR)" CXX="$(CXX)" all
	@echo "PGO instrumented binary built: $(PGO_INSTR_EXE) (arch=$(PGO_ARCH), profile dir=$(PGO_PROFILE_DIR))"
	@echo "Run training workload with $(PGO_INSTR_EXE), then: make pgo-use PGO_ARCH=$(PGO_ARCH) PGO_PROFILE_DIR=$(PGO_PROFILE_DIR)"

pgo-use:
	rm -f src/*.o $(BWA_LIB)
	$(MAKE) arch=$(PGO_ARCH) EXE=$(PGO_FINAL_EXE) EXTRA_CXXFLAGS="-fprofile-use=$(PGO_PROFILE_DIR) -fprofile-correction" CXX="$(CXX)" all
	@echo "PGO optimized binary built: $(PGO_FINAL_EXE) (arch=$(PGO_ARCH))"

pgo-clean:
	rm -rf $(PGO_PROFILE_DIR) bwa-mem3.pgo-instr bwa-mem3.pgo bwa-mem3.pgo-instr.* bwa-mem3.pgo.*

# profile-build / lto-build target arch. Mirrors PGO_ARCH: defaults to
# arm64 on Apple Silicon / aarch64 hosts (preserves prior behavior), and
# native otherwise. Override at the command line for cross-builds, e.g.
#   make profile-build PROFILE_ARCH=avx2
#   make lto-build LTO_ARCH=avx512bw
ifneq ($(IS_ARM),)
    PROFILE_ARCH ?= arm64
    LTO_ARCH     ?= arm64
else
    PROFILE_ARCH ?= native
    LTO_ARCH     ?= native
endif

# Compute-only profile build. -DDISABLE_OUTPUT short-circuits BAM/SAM
# per-record writes AND writer open + header emit, so wall-clock measurements
# exclude all output I/O (no -o file open, no @HD/@SQ/@PG emission). All
# upstream alignment work runs unchanged; the per-stage tprof[] counters
# (printed at end of run) are unaffected.
# Usage: make profile-build
#        make profile-build PROFILE_ARCH=avx2     # cross-build
#        ./bwa-mem3.profile mem -t N idx r1.fq.gz r2.fq.gz
profile-build:
	rm -f src/*.o $(BWA_LIB)
	$(MAKE) arch=$(PROFILE_ARCH) EXE=bwa-mem3.profile EXTRA_CXXFLAGS="$(EXTRA_CXXFLAGS) -DDISABLE_OUTPUT" CXX="$(CXX)" all
	@echo "Compute-only profile binary: bwa-mem3.profile (arch=$(PROFILE_ARCH), output I/O skipped)"
	# Drop variant-flagged objects from the shared cache so a subsequent
	# `make all` doesn't relink stale -DDISABLE_OUTPUT objects.
	rm -f src/*.o $(BWA_LIB)

profile-clean:
	rm -f bwa-mem3.profile

# Link-Time Optimization build.
# Usage: make lto-build
#        make lto-build LTO_ARCH=avx2             # cross-build
#        ./bwa-mem3.lto mem -t N idx r1.fq.gz r2.fq.gz
# Compiles all bwa-mem3 sources with LTO and links with LTO. Non-bwa-mem3
# deps (htslib, mimalloc) keep their non-LTO objects; the
# linker still does LTO across bwa-mem3's own .o. On GCC,
# -fno-semantic-interposition additionally allows more aggressive inlining
# across translation units (no effect on clang, silently ignored).

# LTO_FLAG is detected at recipe-time (not Makefile-parse time) so a stale
# or missing $(CXX) doesn't print a "command not found" warning on every
# `make` invocation that doesn't even target lto-build.
lto-build:
	rm -f src/*.o $(BWA_LIB)
	# GCC + Docker BuildKit jobserver workaround (fg-labs/bwa-mem3#121).
	#
	# Symptom: under BuildKit `make lto-build` dies with
	#   make[2]: *** write jobserver: Bad file descriptor.  Stop.
	#   lto-wrapper: fatal error: make returned 2 exit status
	# linux/arm64 happens to dodge it; linux/amd64 reproduces every time.
	#
	# Root cause: GNU make 4.3's jobserver uses a pair of pipe FDs the parent
	# make advertises in MAKEFLAGS. The chain that breaks under BuildKit is
	# `make[1]` → `gcc -flto` → `lto-wrapper` → `make[2]`. GCC's hygienic
	# subprocess infrastructure sets FD_CLOEXEC on inherited file descriptors
	# it doesn't recognize as compiler-relevant, which closes the jobserver
	# FDs before lto-wrapper's `make[2]` can see them. `make[2]` reads
	# MAKEFLAGS, finds the FD numbers, tries to write to them, and dies on
	# EBADF. This is a known GNU make 4.3 limitation;
	# `--jobserver-style=fifo` (GNU make 4.4+) would survive the chain by
	# using a named FIFO instead of FDs, but Debian Bookworm — and therefore
	# the bwa-mem3-bench Dockerfile's base image — ships make 4.3.
	#
	# Tried and rejected:
	#   - `-flto=auto`: still attempts jobserver negotiation on GCC 12.
	#   - `-flto=N` alone: lto-wrapper still spawns a sub-make that inherits
	#     MAKEFLAGS via the GCC env.
	#   - Clearing MAKEFLAGS for the recursive make `+` `-j$N`: the recursive
	#     make freshly opens its own jobserver FDs in its process, but those
	#     FDs still get CLOEXEC'd by GCC during the LTO link step, so
	#     lto-wrapper's make[2] still hits EBADF.
	#
	# Fix that works: drive the recursive make with `-j1` so it does NOT
	# advertise a jobserver in MAKEFLAGS at all. With nothing advertised,
	# lto-wrapper's make[2] doesn't try to inherit one — it uses the
	# parallelism level requested by `-flto=$$LTO_JOBS` directly, via its
	# own freshly-opened FDs scoped to the lto-wrapper process tree. The
	# inner compile phase goes serial (~5 min cost on a bench-fleet build);
	# the LTO link phase, which dominates LTO build time, stays parallel.
	# Acceptable trade-off for a per-SHA one-shot build.
	#
	# When BuildKit is not in the picture (local dev, traditional CI) the
	# -j1 inner compile is the same single-fork-per-recipe-line behavior
	# `make all -j1` would have produced — no behavioral regression vs.
	# running `make lto-build` without -j on the outer level. The fix only
	# changes the parallelism breakdown, not the build product.
	#
	# Clang's `-flto=thin` branch is unaffected — ThinLTO uses its own
	# parallelism, not the GNU make jobserver, and never participated in
	# the FD inheritance dance.
	@CXX_VERSION="$$($(CXX) --version 2>&1 | head -1)"; \
	  LTO_JOBS=$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1); \
	  case "$$CXX_VERSION" in *clang*) LTO_FLAG=-flto=thin ;; *) LTO_FLAG=-flto=$$LTO_JOBS ;; esac; \
	  echo "LTO_FLAG=$$LTO_FLAG (cxx: $$CXX_VERSION, arch: $(LTO_ARCH), lto-jobs: $$LTO_JOBS)"; \
	  MAKEFLAGS= MAKEOVERRIDES= \
	    $(MAKE) -j1 arch=$(LTO_ARCH) EXE=bwa-mem3.lto EXTRA_CXXFLAGS="$(EXTRA_CXXFLAGS) $$LTO_FLAG -fno-semantic-interposition" CXX="$(CXX)" all
	@echo "LTO binary: bwa-mem3.lto (arch=$(LTO_ARCH))"
	# Drop variant-flagged objects from the shared cache so a subsequent
	# `make all` doesn't relink stale -flto / -fno-semantic-interposition
	# objects.
	rm -f src/*.o $(BWA_LIB)

lto-clean:
	rm -f bwa-mem3.lto

# Print the effective mimalloc setting. Used by CI and humans.
print-mimalloc-config:
	@echo "USE_MIMALLOC=$(USE_MIMALLOC)"

# Print any variable's expanded value: `make print-OBJS`. Used by
# test/regression/make_header_deps.sh to enumerate the objects that must carry
# generated header dependencies, so that check covers whatever the lists hold
# today rather than a copy of them.
print-%:
	@echo '$($*)'

# Header dependencies, generated by $(DEPFLAGS) on every compile (see its
# definition above): one .d per object, alongside it, listing the headers that TU
# included. Globbed rather than derived from an object list so per-tier
# (src/%.avx2.o), native and standalone-test objects are all picked up.
# `-include` (not `include`) makes a glob that matches nothing — a clean tree —
# silently fine rather than a hard error.
-include src/*.d test/*.d $(LIBSAIS_OBJS:.o=.d)
