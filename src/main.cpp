/*************************************************************************************
                           The MIT License

   BWA-MEM2  (Sequence alignment using Burrows-Wheeler Transform),
   Copyright (C) 2019 Intel Corporation, Heng Li.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

Contacts: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>;
                                Heng Li <hli@jimmy.harvard.edu> 
*****************************************************************************************/

// ----------------------------------
#include "main.h"
#include "simd_dispatch.h"
#include "version.h"
#include "bwa_shm.h"
#include <time.h>   /* clock_gettime, nanosleep: proc_freq calibration */

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif


// ----------------------------------
// Profiling globals are now defined in profiling.cpp so they live in
// libbwa.a and are visible to library consumers that don't link main.o.
// ----------------------------------

int usage()
{
    fprintf(stderr, "Usage: bwa-mem3 <command> <arguments>\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  index         create index (add --meth to build a bwameth-style doubled c2t reference)\n");
    fprintf(stderr, "  mem           alignment (add --meth for bisulfite-seq: inline c2t + BAM output)\n");
    fprintf(stderr, "  shm           load/list/drop the index in POSIX shared memory\n");
    fprintf(stderr, "  version       print version number\n");
    fprintf(stderr, "Run `bwa-mem3 <command> --help` for command-specific options.\n");
    return 1;
}

// kstring.h's kputc grows *s by storing realloc's result straight into s->s
// and then writing through it on the next line, with no NULL check -- an OOM
// there is a null-pointer write, not a clean failure. kstring.h is a shared,
// widely-included primitive (bam_writer.cpp, meth_bam.cpp, bwamem_extra.cpp
// all call kputc on their own hot paths too), so changing its OOM contract
// here would reach far outside this file. Instead, pre-grow the buffer with
// our own checked realloc before every kputc call this file makes: once
// pg->l + 1 < pg->m holds, kputc's internal growth check is false and it
// only ever executes the plain assignment, never the unchecked realloc.
static void kputc_checked(int c, kstring_t *pg)
{
    if (pg->l + 1 >= pg->m) {
        size_t new_m = pg->l + 2;
        kroundup32(new_m);
        char *tmp = (char*) realloc(pg->s, new_m);
        xassert(tmp != NULL, "out of memory: @PG CL: buffer");
        pg->s = tmp;
        pg->m = new_m;
    }
    kputc(c, pg);
}

// Append `arg` to the @PG CL: value, replacing any tab/newline/CR with a space
// so it cannot bleed into the @PG line as a tag-separator or record terminator
// (issue #45 / upstream bwa-mem2#293: unescaped tabs/newlines from e.g.
// `-R $'@RG\tID:x\tSM:y'` would otherwise corrupt the @PG record). Does not
// mutate argv itself.
static void append_pg_cl_sanitized(kstring_t *pg, const char *arg)
{
    for (const char *c = arg; *c != '\0'; ++c) {
        kputc_checked((*c == '\t' || *c == '\n' || *c == '\r') ? ' ' : *c, pg);
    }
}

// A single space-separated argv token (argv[1..]). argv[0] is appended by
// append_pg_cl_sanitized directly (no leading space, right after "CL:").
static void append_pg_cl_arg(kstring_t *pg, const char *arg)
{
    kputc_checked(' ', pg);
    append_pg_cl_sanitized(pg, arg);
}

// Measure the __rdtsc() tick rate (ticks per second) that every timing
// report divides by: display_stats(), the `index` "Total time taken" line,
// and the `mem` profiling trailer.
//
// This used to be `tim = __rdtsc(); sleep(1); proc_freq = __rdtsc() - tim;`,
// which spent a full second of wall time on EVERY invocation -- including
// `version`, `shm -l`, and plain usage errors, none of which ever read
// proc_freq. Invisible next to a real alignment run, but it dominates the
// test suite: run_unit_tests.sh spawns ~60 short bwa-mem3 processes, so it
// spent ~60 s of its 74 s asleep against under 5 s of real CPU work.
//
// Measure over a short window instead and divide by the wall time
// CLOCK_MONOTONIC reports for that same window. Both clocks track wall
// time -- the TSC is invariant on every x86-64 this targets, and __rdtsc()
// reads the fixed-rate CNTVCT_EL0 on arm64 -- so a couple of milliseconds
// is plenty. Residual error is the nanosecond-scale cost of the counter
// reads spread over the window (well under 0.1%), far below the precision
// the timing reports claim. It is also no less representative than the old
// sleep: a sleeping core sits at its lowest P-state, so on the parts whose
// TSC is *not* invariant, sleep(1) measured precisely the frequency the
// aligner never runs at.
static uint64_t calibrate_proc_freq(void)
{
    // 5 ms, doubling per retry. nanosleep() returns early when a signal
    // arrives; we divide by the *measured* elapsed time rather than the
    // requested one, so a short window is still usable -- but if it came
    // back too short to divide by with any precision, try again on a
    // longer one.
    const long base_window_ns = 5L * 1000 * 1000;
    const int max_attempts = 4;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        struct timespec window = {0, base_window_ns << attempt};
        // Zero-initialized so that a failing clock_gettime() -- which
        // CLOCK_MONOTONIC does not do in practice, but which would
        // otherwise leave these indeterminate -- yields elapsed == 0 and
        // falls into the retry below rather than reading uninitialized
        // memory and dividing by garbage.
        struct timespec t0 = {0, 0}, t1 = {0, 0};

        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t c0 = __rdtsc();
        nanosleep(&window, NULL);
        uint64_t c1 = __rdtsc();
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                       + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (elapsed >= 1e-4 && c1 > c0)
            return (uint64_t)((double)(c1 - c0) / elapsed + 0.5);
    }
    // Not reachable in practice (it would take max_attempts consecutive
    // windows cut short to under 100 us). Say so on stderr rather than
    // silently returning a bogus rate: without this, a pathological
    // environment -- nanosleep() blocked by a seccomp filter, a process
    // under relentless signal pressure -- would print implausible timings
    // with nothing explaining why. stderr only, so the SAM on stdout is
    // unaffected.
    fprintf(stderr, "[W::%s] could not calibrate the processor tick rate in "
                    "%d attempts; timing reports will be meaningless.\n",
            __func__, max_attempts);
    // Return 1 rather than 0 so the reports print implausible seconds
    // instead of dividing by zero.
    return 1;
}

int main(int argc, char* argv[])
{
    bwamem3_simd_init();

    // ---------------------------------
    proc_freq = calibrate_proc_freq();

    int ret = -1;
    if (argc < 2) return usage();

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }

    /* SIMD host-floor precheck — refuse with exit(2) if the host CPU
     * cannot execute the build's compiled-in instructions, before any
     * subcommand body (and the AVX2-compiled banner / @PG ksprintf in
     * the `mem` branch below) gets a chance to SIGILL on a too-old
     * host. Diagnostic invocations opt out so operators can still
     * introspect the binary on a host below floor:
     *   - `version`               always prints + warns (never refuses)
     *   - `<subcommand> --help`   prints help (never refuses)
     *   - `index`/`shm` `-h`      same (there `-h` is a help alias)
     * `-h` is NOT a help alias for `mem`: there it is the XA-hits option
     * (`h:` in main_mem's optstring, fastmap.cpp), taking an INT value. So
     * `mem -h 5 ref r1 r2` must NOT be treated as help -- doing so skipped the
     * precheck and ran the full alignment on a below-floor host, the SIGILL the
     * precheck exists to turn into a clean exit(2).
     * Only argv[2] is checked: matching --help / -h anywhere in argv
     * would false-positive on pathological invocations like
     * `mem -R --help ref r1 r2` (where --help is the VALUE of -R) and
     * skip the precheck for an actual alignment run. Realistic help
     * invocations put --help right after the subcommand. */
    bool wants_help = (argc >= 3) &&
                      (strcmp(argv[2], "--help") == 0 ||
                       (strcmp(argv[2], "-h") == 0 && strcmp(argv[1], "mem") != 0));
    if (strcmp(argv[1], "version") != 0 && !wants_help) {
        bwamem3_enforce_host_floor();
    }

    if (strcmp(argv[1], "index") == 0)
    {
         uint64_t tim = __rdtsc();
         ret = bwa_index(argc-1, argv+1);
         fprintf(stderr, "Total time taken: %0.4lf\n", (__rdtsc() - tim)*1.0/proc_freq);
         return ret;
    }
    else if (strcmp(argv[1], "mem") == 0)
    {
        // Short-circuit `mem --help` so we skip the AVX/SA banner and the
        // post-run profiling trailer (printed below when ret==0). Skip
        // tokens that are the value of an option that takes an argument
        // -- otherwise `mem -R --help ...`, `mem -o --help ...`, or
        // `mem --set-as-failed --help ...` would treat the option's
        // value as a help request and suppress the banner. Keep the
        // short-option string and long-option list in sync with the
        // optstring and long_opts table in fastmap.cpp::main_mem.
        static const char *const MEM_SHORT_OPTS_WITH_ARG =
            "kcvsrtRABOEUwLdTQDmINWxGhyKXHofz";
        // Every required_argument entry in main_mem's long_opts (fastmap.cpp).
        // optional_argument options (--bam, --meth, --rescue-kmer,
        // --extend-mate-concordant) are omitted: getopt_long only binds those
        // with `=`, so a following --help token is not consumed as their value.
        static const char *const MEM_LONG_OPTS_WITH_ARG[] = {
            "--min-ext-len", "--max-extend-chains", "--dedup", "--dedup-reads",
            "--chunk-cap", "--cohort-slices", "--rescue-band",
            "--cohort-ramp-ratio", "--cohort-ramp-first", "--meth-scoring",
            "--meth-tags", "--set-as-failed", "--supp-rep-hard-cap",
            "--seed-order", "--compat",
#ifdef STAGE_PROF
            "--profile",
#endif
            NULL,
        };
        for (int i = 2; i < argc; ++i) {
            const char *t = argv[i];
            if (strcmp(t, "--") == 0) break;
            if (strcmp(t, "--help") == 0) return main_mem(argc-1, argv+1);
            // Long option (no `=`): next argv token is its value.
            int matched_long = 0;
            for (int j = 0; MEM_LONG_OPTS_WITH_ARG[j]; ++j) {
                if (strcmp(t, MEM_LONG_OPTS_WITH_ARG[j]) == 0) {
                    matched_long = 1; break;
                }
            }
            if (matched_long) { ++i; continue; }
            // Short-option bundle `-abc[value]`: walk chars; the first
            // arg-taking option consumes the rest of the bundle (if any)
            // or the next argv token (if the bundle ends at it).
            if (t[0] != '-' || t[1] == '\0' || t[1] == '-') continue;
            for (const char *p = t + 1; *p; ++p) {
                if (strchr(MEM_SHORT_OPTS_WITH_ARG, *p)) {
                    if (*(p + 1) == '\0') ++i;
                    break;
                }
            }
        }

        tprof[MEM][0] = __rdtsc();
        kstring_t pg = {0,0,0};
        extern char *bwa_pg;

        fprintf(stderr, "-----------------------------\n");
        // Print the runtime-dispatched kernel tier rather than the compile-time
        // baseline. Non-kernel TUs (including this one) build at BASELINE_ARCH
        // (avx2 by default, per the Makefile), so a compile-time
        // __AVX2__/__AVX512BW__ banner here would misreport the tier that
        // actually ran.
        fprintf(stderr, "Executing in %s mode!!\n",
                bwamem3_simd_tier_name(bwamem3_simd_tier()));
        fprintf(stderr, "-----------------------------\n");

        // The "SA compression enabled" banner used to print here, but that's
        // before the index is loaded (main_mem hasn't run yet), so it could
        // only report the compile-time SA_COMPX default rather than the rate
        // a `-u`-built index actually loaded at. It now prints from
        // fastmap.cpp::main_mem right after aux.fmi->load_index().

        // argv[0] is the first CL: token; sanitize it too (a tab/newline in the
        // executable path would corrupt the @PG line exactly as a -R value did,
        // issue #45) but without append_pg_cl_arg's leading space so the value
        // stays "CL:<argv0> <argv1> ..." rather than "CL: <argv0> ...".
        ksprintf(&pg, "@PG\tID:bwa-mem3\tPN:bwa-mem3\tVN:%s\tCL:", PACKAGE_VERSION);
        append_pg_cl_sanitized(&pg, argv[0]);  // first token, no leading space

        for (int i = 1; i < argc; ++i) append_pg_cl_arg(&pg, argv[i]);
        ksprintf(&pg, "\n");
        bwa_pg = pg.s;
        ret = main_mem(argc-1, argv+1);
        free(bwa_pg);
    }
    else if (strcmp(argv[1], "version") == 0)
    {
        puts(PACKAGE_VERSION);
        /* Report the compiler that built this binary. bwa-mem3 is markedly
         * faster when built with clang than with g++ (see the Makefile's
         * build-time note and docs/src/best-practices/build.md), so surfacing
         * the compiler here makes it trivial to confirm which toolchain a
         * given binary came from. Purely factual — one token, easy to parse. */
        /* Order matters: both Intel front-ends impersonate another toolchain.
         * icpx (oneAPI, LLVM-based) also defines __clang__, and icpc (Classic)
         * also defines __GNUC__ to match the host GCC's ABI — so the Intel
         * checks must come FIRST or an Intel build reports as clang/gcc. The
         * Makefile has real `ifeq ($(CXX), icpc)` / `icpx` flag branches, so
         * these are builds we actually ship flags for. */
#if defined(__INTEL_LLVM_COMPILER)
        /* Two encodings, split on 1000000 the way CMake's
         * Modules/Compiler/IntelLLVM-DetermineCompiler.cmake does:
         *   >= 1000000: 8-digit VVVVRRPP (2021.2.0 and later),
         *               e.g. 20250100 -> 2025.1.0
         *   <  1000000: 6-digit VVVVRP  (pre-2021.2.0),
         *               e.g. 202110   -> 2021.1.0 */
#  if __INTEL_LLVM_COMPILER >= 1000000
        fprintf(stdout, "Compiler: icpx %d.%d.%d\n",
                __INTEL_LLVM_COMPILER / 10000,
                (__INTEL_LLVM_COMPILER / 100) % 100,
                __INTEL_LLVM_COMPILER % 100);
#  else
        fprintf(stdout, "Compiler: icpx %d.%d.%d\n",
                __INTEL_LLVM_COMPILER / 100,
                (__INTEL_LLVM_COMPILER / 10) % 10,
                __INTEL_LLVM_COMPILER % 10);
#  endif
#elif defined(__INTEL_COMPILER)
        /* Year-based version (e.g. 2021) plus the update number. Every icpc
         * the Makefile targets defines __INTEL_COMPILER_UPDATE, but guard it
         * anyway: on a pre-15.0 icc it is undefined, and an undefined macro in
         * an argument list is an undeclared identifier, not a zero. */
#  if defined(__INTEL_COMPILER_UPDATE)
        fprintf(stdout, "Compiler: icpc %d.%d\n",
                __INTEL_COMPILER, __INTEL_COMPILER_UPDATE);
#  else
        fprintf(stdout, "Compiler: icpc %d\n", __INTEL_COMPILER);
#  endif
#elif defined(__clang__)
        fprintf(stdout, "Compiler: clang %d.%d.%d\n",
                __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
        fprintf(stdout, "Compiler: gcc %d.%d.%d\n",
                __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(__VERSION__)
        fprintf(stdout, "Compiler: %s\n", __VERSION__);
#endif
        bwamem3_print_version_simd(stdout);
#ifdef USE_MIMALLOC
        {
            int mv = mi_version();
            // Report whether mimalloc is actually intercepting the standard
            // allocator, not merely linked. mi_version() resolves as long as
            // libmimalloc is on the link line, so the version alone is a
            // false-positive signal: a build that links a libmimalloc which
            // exports only the mi_* API (e.g. some distro/conda libmimalloc.so
            // built without the malloc override) prints a version here while
            // every real malloc/free still goes to the system allocator.
            // Probe by allocating through the standard malloc and asking
            // mimalloc whether the pointer lives in one of its heap regions —
            // true only when malloc was routed to mimalloc.
            void *probe = malloc(64);
            int active = (probe != NULL) && mi_is_in_heap_region(probe);
            free(probe);
            // Emit on stdout so the whole `version` block stays on one stream
            // (PACKAGE_VERSION and the SIMD lines above also go to stdout);
            // downstream scripts that capture stdout can then parse it.
            fprintf(stdout, "mimalloc %d.%d.%d (%s)\n",
                    mv / 10000, (mv / 100) % 100, mv % 100,
                    active ? "active" : "linked but NOT overriding malloc");
        }
#endif
        return 0;
    }
    else if (strcmp(argv[1], "shm") == 0)
    {
        return main_shm(argc - 1, argv + 1);
    }
    else {
        fprintf(stderr, "ERROR: unknown command '%s'\n", argv[1]);
        return 1;
    }

    if (ret == 0) {
        fprintf(stderr, "\nImportant parameter settings: \n");
        fprintf(stderr, "\tBATCH_SIZE: %d\n", BATCH_SIZE);
        fprintf(stderr, "\tMAX_SEQ_LEN_REF: %d\n", MAX_SEQ_LEN_REF);
        fprintf(stderr, "\tMAX_SEQ_LEN_QER: %d\n", MAX_SEQ_LEN_QER);
        fprintf(stderr, "\tMAX_SEQ_LEN8: %d\n", MAX_SEQ_LEN8);
        fprintf(stderr, "\tSEEDS_PER_READ: %d\n", SEEDS_PER_READ);
        // SIMD_WIDTH8/16 are omitted here: this TU is compiled at BASELINE_ARCH,
        // so its compile-time values misreport the runtime-dispatched kernel
        // tier (already printed as "Executing in <tier> mode" above).
        fprintf(stderr, "\tAVG_SEEDS_PER_READ: %d\n", AVG_SEEDS_PER_READ);
    }
    
    return ret;
}
