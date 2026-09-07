// test/unit/test_index_load_threads.cpp
//
// Regression test for the fail-closed BWA3_LOAD_THREADS parse in
// index_load_threads (src/FMI_search.cpp).
//
// The override used to go through atoi, which silently accepts garbage
// ("abc" -> 0, "0x8" -> 0) and any magnitude ("100000" -> ~1280 pthread_creates
// on a bandwidth-bound load). The hardened parser strtol's the value and, on
// anything non-numeric / with trailing garbage / non-positive, warns and keeps
// the computed default instead of running with a bad worker count. A valid
// override is clamped to a 64-thread ceiling. These are the "malformed
// BWA3_LOAD_THREADS" fail-closed cases; they need no index on disk, so they are
// a pure unit test of the parse contract.

#include "doctest/doctest.h"
#include "../../src/FMI_search.h"

#include <cstdlib>
#include <string>

namespace {

// RAII guard so each case sets BWA3_LOAD_THREADS in isolation and the process
// environment is restored to its prior state on scope exit, even if a CHECK
// throws. It captures whether the variable was originally set (and its value)
// before mutating it, so a BWA3_LOAD_THREADS already present in the test
// runner's environment survives the test instead of being unconditionally
// cleared -- keeping later cases (and any other test in the binary) isolated.
struct LoadThreadsEnv {
    explicit LoadThreadsEnv(const char *value)
    {
        if (const char *prev = getenv("BWA3_LOAD_THREADS")) {
            had_prev_ = true;
            prev_     = prev;  // copy: the setenv/unsetenv below may invalidate `prev`
        }
        if (value != nullptr)
            setenv("BWA3_LOAD_THREADS", value, /*overwrite=*/1);
        else
            unsetenv("BWA3_LOAD_THREADS");
    }
    ~LoadThreadsEnv()
    {
        if (had_prev_)
            setenv("BWA3_LOAD_THREADS", prev_.c_str(), /*overwrite=*/1);
        else
            unsetenv("BWA3_LOAD_THREADS");
    }

  private:
    bool had_prev_ = false;
    std::string prev_;
};

}  // namespace

TEST_CASE("index_load_threads: no override clamps the caller's request to [1,8]"
          * doctest::test_suite("unit/index_load_threads")) {
    LoadThreadsEnv env(nullptr);  // ensure a stray value from the shell can't leak in

    // A non-positive request clamps up to 1, never to 0 (callers spawn `t`
    // pread workers and would otherwise create none / divide by zero).
    CHECK(index_load_threads(0) == 1);
    CHECK(index_load_threads(-4) == 1);

    // In-range requests pass through; over the cap saturates at 8.
    CHECK(index_load_threads(1) == 1);
    CHECK(index_load_threads(4) == 4);
    CHECK(index_load_threads(8) == 8);
    CHECK(index_load_threads(9) == 8);
    CHECK(index_load_threads(64) == 8);
}

TEST_CASE("index_load_threads: a malformed override is ignored, keeping the default"
          * doctest::test_suite("unit/index_load_threads")) {
    // Every malformed form must fall back to the computed default (here 4),
    // proving the parse fails closed rather than running with a bad count. Each
    // form is its own SUBCASE so a failure names the offending input directly
    // instead of hiding behind a shared loop iteration.
    const char *value = nullptr;
    SUBCASE("abc (non-numeric)")                      { value = "abc"; }
    SUBCASE("empty (treated as unset: default stands)") { value = ""; }
    SUBCASE("12x (trailing garbage)")                { value = "12x"; }
    SUBCASE("0x8 (hex-looking; base-10 strtol stops at 'x')") { value = "0x8"; }
    SUBCASE("whitespace only (no digits consumed)")  { value = "  "; }
    SUBCASE("0 (not positive)")                      { value = "0"; }
    SUBCASE("-1 (not positive)")                     { value = "-1"; }
    SUBCASE("3.5 (trailing '.5')")                   { value = "3.5"; }
    SUBCASE("overflows long (errno == ERANGE)")      { value = "9999999999999999999999"; }

    LoadThreadsEnv env(value);
    CHECK(index_load_threads(4) == 4);
}

TEST_CASE("index_load_threads: a valid override wins and is clamped to the 64 ceiling"
          * doctest::test_suite("unit/index_load_threads")) {
    {
        LoadThreadsEnv env("16");  // above the [1,8] default cap, below the ceiling
        CHECK(index_load_threads(4) == 16);
    }
    {
        LoadThreadsEnv env("1");
        CHECK(index_load_threads(8) == 1);  // override can also lower the count
    }
    {
        LoadThreadsEnv env("64");
        CHECK(index_load_threads(4) == 64);
    }
    {
        LoadThreadsEnv env("100000");  // the runaway atoi used to accept
        CHECK(index_load_threads(4) == 64);
    }
    {
        LoadThreadsEnv env("+5");  // strtol accepts a leading '+'
        CHECK(index_load_threads(4) == 5);
    }
}
