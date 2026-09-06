##*****************************************************************************
##  bwa-mem3 test build
##
##  This Makefile produces:
##    - test/framework/libbwa_test_framework.a    (shared test-helper lib)
##    - test/bwa_mem3_tests_unit                  (doctest unit binary)
##    - test/bwa_mem3_tests_integration           (doctest integration binary)
##    - test/fmi_test, smem2_test, ...            (legacy per-binary tests,
##                                                  migrated one-by-one in
##                                                  follow-up PRs)
##
##  Arch flags (CXX, CXXFLAGS, CPPFLAGS, INCLUDES, LIBS) inherit from the
##  parent Makefile when invoked via `make -C test`. Override on the command
##  line for local builds that don't go through the parent (e.g. macOS ARM
##  direct builds — see the ARM branch below).
##*****************************************************************************

EXE = fmi_test smem2_test bwt_seed_strategy_test sa2ref_test xeonbsw bandedswa_zdrop_eweight_test smem_lockstep_parity_test bwtseed_lockstep_parity_test backwardext_konly_parity_test header_insert_test packed_text_test system_test

# Doctest binaries — rules populated in Task 14 (unit) and Task 15
# (integration). Listed separately from $(EXE) so `all` and
# `run_unit_tests.sh` keep working during the framework-scaffolding PR.
DOCTEST_BINS = bwa_mem3_tests_unit bwa_mem3_tests_integration

UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
IS_ARM  := $(filter $(UNAME_M),arm64 aarch64)

ifneq ($(IS_ARM),)
    CXX ?= clang++
    ifeq ($(UNAME_S),Darwin)
        LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
        ifeq ($(strip $(LIBOMP_PREFIX)),)
            $(error libomp not found: install with `brew install libomp`, or set LIBOMP_PREFIX to its install prefix)
        endif
        CXXFLAGS ?= -std=c++11 -O2 -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
        LIBS     ?= -L.. -L$(LIBOMP_PREFIX)/lib -lomp -lbwa ../ext/libsais/src/libsais.o ../ext/libsais/src/libsais64.o -lz -framework Accelerate ../ext/htslib/libhts.a -lpthread
    else
        CXXFLAGS ?= -std=c++11 -O2 -fopenmp
        LIBS     ?= -L.. -fopenmp -lbwa ../ext/libsais/src/libsais.o ../ext/libsais/src/libsais64.o -lz ../ext/htslib/libhts.a -lpthread
    endif
else
    CXX      ?= g++
    CXXFLAGS ?= -std=c++11 -fopenmp -mtune=native -march=native
    LIBS     ?= -L.. -fopenmp -lbwa ../ext/libsais/src/libsais.o ../ext/libsais/src/libsais64.o -lz ../ext/htslib/libhts.a -lpthread
endif

# Inherit htslib's transitive deps from the parent Makefile so libhts.a's
# libdeflate (and any other configure-autodetected feature) symbols
# resolve at link time. The parent's `test-binaries` recipe propagates
# HTSLIB_static_LIBS / HTSLIB_static_LDFLAGS as command-line overrides.
# Both default to empty when this Makefile is invoked standalone, in
# which case the link will fail the same way the parent's would without
# the parent's htslib_static.mk include block -- see the parent Makefile
# for the mechanism.
LIBS    += $(HTSLIB_static_LIBS)
LDFLAGS += $(HTSLIB_static_LDFLAGS)

ifneq ($(IS_ARM),)
    # Keep the ARM cache-line alignment in lock-step with the top-level Makefile:
    # CACHE_LINE_BYTES feeds _mm_malloc, so a build overriding
    # ARM_CACHE_LINE=64 (Neoverse/Graviton) must exercise the same allocation
    # configuration here as in the shipped libbwa.a objects. Command-line and
    # environment overrides win over this default; the top-level test-binaries
    # recipe also forwards it explicitly.
    ARM_CACHE_LINE ?= 128
    # Keep the ARM_CACHE_LINE validation in lock-step with the top-level Makefile:
    # CACHE_LINE_BYTES must be exactly one supported value, or _mm_malloc's
    # posix_memalign fails. Guard the token count first ($(filter) matches per
    # word, so "64 128" would slip past a bare membership test; this also rejects
    # empty), then the value itself with the supported values as the $(filter)
    # pattern (a user value used as the pattern could carry a `%` wildcard that
    # matches everything).
    ifneq ($(words $(ARM_CACHE_LINE)),1)
        $(error ARM_CACHE_LINE must be a single value, 64 or 128 (got '$(ARM_CACHE_LINE)'))
    endif
    ifeq ($(filter 64 128,$(ARM_CACHE_LINE)),)
        $(error ARM_CACHE_LINE must be 64 or 128 (got '$(ARM_CACHE_LINE)'))
    endif
    CPPFLAGS = -DENABLE_PREFETCH -D__SSE__=1 -D__SSE2__=1 -D__SSE3__=1 -D__SSSE3__=1 -D__SSE4_1__=1 -D__SSE4_2__=1 -DAPPLE_SILICON=1 -DCACHE_LINE_BYTES=$(ARM_CACHE_LINE)
    # Match the top-level Makefile's ARM_CPU tuning so the test binaries are built
    # with the same -mcpu as the shipped objects (forwarded by the test-binaries
    # recipe). Unset -> generic scheduling, identical to before.
    ifneq ($(ARM_CPU),)
        CPPFLAGS += -mcpu=$(ARM_CPU)
    endif
    INCLUDES = -I../src -I../ext/sse2neon -I../ext -I../ext/htslib
else
    CPPFLAGS = -DENABLE_PREFETCH
    INCLUDES = -I../src -I../ext -I../ext/htslib
endif

FRAMEWORK_DIR   := framework
INTEGRATION_DIR := integration
UNIT_DIR        := unit

# ARCH_FLAGS_FROM_PARENT: propagated from the parent Makefile's
# `test-binaries` target so the test build sees the same __AVX2__ /
# __AVX512BW__ / APPLE_SILICON macros as libbwa.a. Required for the
# BWA_TESTS_HAVE_KSWV macro in kswv_runner.h to evaluate consistently
# between libbwa.a and the test binary.
#
# IMPORTANT: we also strip -march=native / -mtune=native from the
# default CXXFLAGS. Those flags enable every feature the runner CPU
# supports (so e.g. an AVX2-capable runner silently defines __AVX2__
# even on the `sse41` matrix row) and GCC only has -mno-<feat> to
# disable, not -m<arch-only>. Stripping + re-adding the parent's
# arch flags gives the test build the same ISA surface libbwa.a saw.
ifneq ($(ARCH_FLAGS_FROM_PARENT),)
    CXXFLAGS := $(filter-out -march=native -mtune=native,$(CXXFLAGS)) $(ARCH_FLAGS_FROM_PARENT)
endif

# COVERAGE=1 augments CXXFLAGS/LDFLAGS with --coverage and overrides -O2 with
# -O0 so gcov line numbers correspond 1:1 with source. Consumed by the CI
# `coverage` job. Matches the parent Makefile's coverage plumbing.
ifneq ($(COVERAGE),)
    CXXFLAGS += -O0 --coverage
    LDFLAGS  += --coverage
endif
FRAMEWORK_SRCS := $(FRAMEWORK_DIR)/scoring.cpp \
                  $(FRAMEWORK_DIR)/seqpair_gen.cpp \
                  $(FRAMEWORK_DIR)/seqpair_batch.cpp \
                  $(FRAMEWORK_DIR)/ksw_runner.cpp \
                  $(FRAMEWORK_DIR)/kswv_runner.cpp \
                  $(FRAMEWORK_DIR)/kswr_cmp.cpp \
                  $(FRAMEWORK_DIR)/junit_reporter.cpp \
                  $(FRAMEWORK_DIR)/test_main.cpp
FRAMEWORK_OBJS := $(FRAMEWORK_SRCS:.cpp=.o)
FRAMEWORK_LIB  := $(FRAMEWORK_DIR)/libbwa_test_framework.a

FRAMEWORK_INCLUDES := $(INCLUDES) -I$(FRAMEWORK_DIR)

# Legacy integration-binary sources live under integration/ after the
# Task 11 migration. We still emit their binaries at test/<name> so that
# run_unit_tests.sh and the proto-neon-kswv.yml workflow keep working.
# No `vpath` is needed: every legacy target spells out its source as
# $(INTEGRATION_DIR)/<name>.o, satisfied by the explicit pattern rule
# further down.

.PHONY: all framework unit integration clean libsais-objs
.SUFFIXES: .cpp .o

# Default: build everything.
all: framework unit integration $(EXE)

# -- Framework static lib ----------------------------------------------------

framework: $(FRAMEWORK_LIB)

$(FRAMEWORK_LIB): $(FRAMEWORK_OBJS)
	ar rcs $@ $^

$(FRAMEWORK_DIR)/%.o: $(FRAMEWORK_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(FRAMEWORK_INCLUDES) -MMD -MP $< -o $@

# -- Doctest unit/integration binaries (populated in Task 12 + 13) ---------

UNIT_SRCS := $(wildcard $(UNIT_DIR)/test_*.cpp)
UNIT_OBJS := $(UNIT_SRCS:.cpp=.o)
UNIT_BIN  := bwa_mem3_tests_unit

# Fail fast if no test_*.cpp files are present in unit/. The link line
# would otherwise pull only test_main.o out of the framework lib and
# silently produce a binary with zero TEST_CASEs that reports "no
# assertions" as a green CI run.
#
# Gate the check on the active goals so `make clean` (and other
# non-build targets) still works on a checkout with no unit tests yet.
# `$(error ...)` is evaluated at parse time, regardless of the requested
# goal, so without the MAKECMDGOALS filter every invocation would abort.
ifeq ($(strip $(UNIT_SRCS)),)
  ifneq ($(filter unit all,$(or $(MAKECMDGOALS),all)),)
    $(error No test_*.cpp sources found in $(UNIT_DIR)/. Add a unit test or pick a different target (e.g. `make -C test clean`).)
  endif
endif

unit: $(UNIT_BIN)

$(UNIT_BIN): $(UNIT_OBJS) $(FRAMEWORK_LIB) ../libbwa.a
	$(CXX) $(LDFLAGS) -o $@ $(UNIT_OBJS) $(FRAMEWORK_LIB) $(LIBS)

$(UNIT_DIR)/%.o: $(UNIT_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(FRAMEWORK_INCLUDES) -MMD -MP $< -o $@

INTEGRATION_TEST_SRCS := $(wildcard $(INTEGRATION_DIR)/test_*.cpp) \
                         $(wildcard $(INTEGRATION_DIR)/placeholder.cpp)
INTEGRATION_TEST_OBJS := $(INTEGRATION_TEST_SRCS:.cpp=.o)
INTEGRATION_BIN       := bwa_mem3_tests_integration

integration: $(INTEGRATION_BIN)

$(INTEGRATION_BIN): $(INTEGRATION_TEST_OBJS) $(FRAMEWORK_LIB) ../libbwa.a
	$(CXX) $(LDFLAGS) -o $@ $(INTEGRATION_TEST_OBJS) $(FRAMEWORK_LIB) $(LIBS)

# Integration tests use the framework headers; override the generic
# integration/%.o rule added in Task 12 for test_* / placeholder files.
$(INTEGRATION_DIR)/test_%.o: $(INTEGRATION_DIR)/test_%.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(FRAMEWORK_INCLUDES) -MMD -MP $< -o $@

$(INTEGRATION_DIR)/placeholder.o: $(INTEGRATION_DIR)/placeholder.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(FRAMEWORK_INCLUDES) -MMD -MP $< -o $@

# -- Legacy per-binary tests (sources under integration/; binaries emit in test/) -

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

# libsais object files live in the parent build tree and link into every
# test binary via $(LIBS) — they appear as bare paths inside the link
# command, NOT as make prerequisites, so make doesn't know to build them
# on demand. Delegate to the parent Makefile's LIBSAIS_OBJS rule
# (-DLIBSAIS_OPENMP / -O3 flag set stays in one place) and gate every test
# binary behind it via an order-only prerequisite — fires once per
# invocation without forcing the test binaries to relink. The parent's
# target paths are relative to its own dir, so we drop the leading `../`
# when invoking it.
#
# Cover both the legacy $(EXE) binaries AND the doctest unit/integration
# binaries: when libbwa.a is restored from a CI cache without a fresh
# parent `make` (e.g. proto-neon-kswv.yml's both-caches-hit path), $(BWA_LIB)
# is already up-to-date so the parent never triggers libsais.o on its own.
libsais-objs:
	$(MAKE) -C .. ext/libsais/src/libsais.o ext/libsais/src/libsais64.o

$(EXE) $(UNIT_BIN) $(INTEGRATION_BIN): | libsais-objs

# NB: $(LDFLAGS) is threaded through every link rule below so COVERAGE=1
# (which appends --coverage to LDFLAGS at the top of the file) propagates
# uniformly. Without it, `make -C test all COVERAGE=1` would leave the
# legacy binaries with unresolved __gcov_* symbols even though the .o
# files were compiled with --coverage. CI doesn't rebuild these targets
# under coverage, but local-dev `make all` did.
fmi_test: $(INTEGRATION_DIR)/fmi_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

smem2_test: $(INTEGRATION_DIR)/smem2_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

sa2ref_test: $(INTEGRATION_DIR)/sa2ref_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

bwt_seed_strategy_test: $(INTEGRATION_DIR)/bwt_seed_strategy_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

xeonbsw: $(INTEGRATION_DIR)/main_banded.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

bandedswa_zdrop_eweight_test: $(INTEGRATION_DIR)/bandedswa_zdrop_eweight_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

smem_lockstep_parity_test: smem_lockstep_parity_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

bwtseed_lockstep_parity_test: bwtseed_lockstep_parity_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

backwardext_konly_parity_test: backwardext_konly_parity_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

header_insert_test: header_insert_test.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Build test-local copies of packed_text.cpp / system.cpp into test/
# rather than ../src/. The parent build's CXXFLAGS (`-O3 -fpermissive`,
# `-DSAIS=1`, `-DLIBSAIS_OPENMP`) differ from this Makefile's
# (`-O2 -fopenmp -mtune=native -march=native`, `-DENABLE_PREFETCH`),
# and writing back into ../src/*.o would silently overwrite parent build
# artifacts with a different ABI/flag set — `ar rcs libbwa.a ...` would
# then archive the test-flagged objects on the next parent rebuild.
packed_text_test: packed_text_test.o packed_text_lib.o
	$(CXX) -o $@ $^ $(LIBS)

packed_text_test.o: ../src/packed_text.h
packed_text_lib.o: ../src/packed_text.cpp ../src/packed_text.h
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

system_test: system_test.o system_lib.o
	$(CXX) -o $@ $^ $(LIBS)

system_test.o: ../src/system.h
system_lib.o: ../src/system.cpp ../src/system.h
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

$(INTEGRATION_DIR)/%.o: $(INTEGRATION_DIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@


clean:
	rm -f *.o *.d $(INTEGRATION_DIR)/*.o $(INTEGRATION_DIR)/*.d \
	      $(UNIT_DIR)/*.o $(UNIT_DIR)/*.d \
	      $(FRAMEWORK_OBJS) $(FRAMEWORK_DIR)/*.d $(FRAMEWORK_LIB) $(EXE) $(DOCTEST_BINS)
	rm -f *.gcno *.gcda $(INTEGRATION_DIR)/*.gcno $(INTEGRATION_DIR)/*.gcda \
	      $(UNIT_DIR)/*.gcno $(UNIT_DIR)/*.gcda \
	      $(FRAMEWORK_DIR)/*.gcno $(FRAMEWORK_DIR)/*.gcda

# Pull in auto-generated header deps so editing a header triggers a
# rebuild of dependent .o files. Generated by -MMD -MP on each compile.
-include $(FRAMEWORK_OBJS:.o=.d) $(UNIT_OBJS:.o=.d) $(INTEGRATION_TEST_OBJS:.o=.d)

# Common header prerequisites shared by the FM-index / seeding integration and
# parity tests. Extracted to a single variable so each target keeps one
# dependency line (checkmake flags a target whose prerequisites are split across
# multiple rules) and the header set has one source of truth.
FMI_TEST_HEADERS := ../src/FMI_search.h ../src/bntseq.h ../src/read_index_ele.h \
                    ../src/bwa.h ../src/bwt.h ../src/utils.h ../src/macro.h

# DO NOT DELETE
integration/bwt_seed_strategy_test.o: $(FMI_TEST_HEADERS)
integration/fmi_test.o: $(FMI_TEST_HEADERS)
integration/main_banded.o: ../src/bandedSWA.h ../src/macro.h
integration/bandedswa_zdrop_eweight_test.o: ../src/bandedSWA.h ../src/macro.h
integration/sa2ref_test.o: $(FMI_TEST_HEADERS)
integration/smem2_test.o: $(FMI_TEST_HEADERS)
smem_lockstep_parity_test.o: $(FMI_TEST_HEADERS)
bwtseed_lockstep_parity_test.o: $(FMI_TEST_HEADERS)
backwardext_konly_parity_test.o: $(FMI_TEST_HEADERS)
header_insert_test.o: ../src/bwa.h
