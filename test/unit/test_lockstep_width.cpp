// test/unit/test_lockstep_width.cpp
//
// Unit tests for the phase-2 SMEM lockstep width resolution: the pure policy
// that turns a raw memory-level-parallelism estimate (or a BWA3_SMEM_LOCKSTEP_N
// override) into the width the lockstep SMEM driver runs at. The measurement
// itself (the startup pointer-chase) and the one-shot installer
// (bwa3_init_smem_lockstep_width, idempotent + global) are validated by the
// whole-run byte-identity and throughput gates, not here; this file pins the
// clamp/override policy those two consult -- including the initializer's
// probe-vs-pin decision, which lives entirely in bwa3_lockstep_width_parse_env.
//
// Every value fed to the functions under test is derived from the configured
// [SMEM_LOCKSTEP_N, SMEM_LOCKSTEP_N_MAX] range, never the 16/64 shipped default,
// so the assertions follow the contract under any valid -D override of either
// width (the compile-time guard in lockstep_width.h bounds that range). Cases
// that only exist for a non-degenerate range (a strict interior value) or that
// would overflow at the INT32_MAX extreme are compile-guarded accordingly.

#include "doctest/doctest.h"
#include "../../src/lockstep_width.h"

#include <cstdint>  // INT32_MAX
#include <cstdio>   // snprintf
#include <unistd.h> // rmdir, unlink
#include <cstdlib>  // mkdtemp, getenv
#include <cstring>  // strlen
#include <string>
#include <vector>
#include <sys/stat.h> // mkdir

// Stringify a macro's *value* (two-level expansion) so the configured widths
// can be fed to parse_env as decimal string literals.
#define BWA3_STR2(x) #x
#define BWA3_STR(x) BWA3_STR2(x)

// --- bwa3_lockstep_width_from_probe: clamp raw MLP to [floor, ceiling] ---

TEST_CASE("lockstep width: from_probe clamps a raw MLP estimate to [floor, ceiling]"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("a non-positive probe (measurement failure) falls back to the floor") {
        CHECK(bwa3_lockstep_width_from_probe(0)  == SMEM_LOCKSTEP_N);
        CHECK(bwa3_lockstep_width_from_probe(-3) == SMEM_LOCKSTEP_N);
    }
    SUBCASE("the floor and ceiling map to themselves") {
        CHECK(bwa3_lockstep_width_from_probe(SMEM_LOCKSTEP_N)     == SMEM_LOCKSTEP_N);
        CHECK(bwa3_lockstep_width_from_probe(SMEM_LOCKSTEP_N_MAX) == SMEM_LOCKSTEP_N_MAX);
    }
    SUBCASE("a probe above the ceiling clamps to the max") {
#if SMEM_LOCKSTEP_N_MAX < INT32_MAX
        // Guarded so SMEM_LOCKSTEP_N_MAX + 1 cannot overflow int32_t.
        CHECK(bwa3_lockstep_width_from_probe(SMEM_LOCKSTEP_N_MAX + 1) == SMEM_LOCKSTEP_N_MAX);
#endif
        CHECK(bwa3_lockstep_width_from_probe(INT32_MAX) == SMEM_LOCKSTEP_N_MAX);
    }
#if SMEM_LOCKSTEP_N > 1
    SUBCASE("a probe below the floor clamps up to the floor") {
        CHECK(bwa3_lockstep_width_from_probe(SMEM_LOCKSTEP_N - 1) == SMEM_LOCKSTEP_N);
        CHECK(bwa3_lockstep_width_from_probe(1)                   == SMEM_LOCKSTEP_N);
    }
#endif
#if SMEM_LOCKSTEP_N_MAX > SMEM_LOCKSTEP_N
    SUBCASE("a probe inside the range is used verbatim") {
        constexpr int32_t mid = SMEM_LOCKSTEP_N + (SMEM_LOCKSTEP_N_MAX - SMEM_LOCKSTEP_N) / 2;
        CHECK(bwa3_lockstep_width_from_probe(mid) == mid);
    }
#endif
}

// --- bwa3_lockstep_width_parse_env: classify the BWA3_SMEM_LOCKSTEP_N override ---
//
// Tri-state contract the initializer keys off: > 0 pins the width (skip probe),
// 0 means unset (no override -- keep the compile-time default, probe only if
// opted in), -1 means set-but-invalid (report, then resolve as if unset).
// The regression this guards: an invalid override must NOT resolve to a usable
// width -- previously a nonempty-but-garbage value silently disabled auto-tuning
// by flooring, and an overflowed value silently selected the max.

TEST_CASE("lockstep width: parse_env classifies the override into pin/unset/invalid"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("unset or empty is 0 (no override; caller keeps default, probes only if opted in)") {
        CHECK(bwa3_lockstep_width_parse_env(nullptr) == 0);
        CHECK(bwa3_lockstep_width_parse_env("")      == 0);
    }
    SUBCASE("a valid integer at the floor or ceiling pins that width") {
        CHECK(bwa3_lockstep_width_parse_env(BWA3_STR(SMEM_LOCKSTEP_N))     == SMEM_LOCKSTEP_N);
        CHECK(bwa3_lockstep_width_parse_env(BWA3_STR(SMEM_LOCKSTEP_N_MAX)) == SMEM_LOCKSTEP_N_MAX);
        CHECK(bwa3_lockstep_width_parse_env("1") == 1);  // escape hatch: env may go below the floor
    }
#if SMEM_LOCKSTEP_N_MAX > SMEM_LOCKSTEP_N
    SUBCASE("a valid integer inside the range pins that width") {
        constexpr int32_t mid = SMEM_LOCKSTEP_N + (SMEM_LOCKSTEP_N_MAX - SMEM_LOCKSTEP_N) / 2;
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", mid);
        CHECK(bwa3_lockstep_width_parse_env(buf) == mid);
    }
#endif
#if SMEM_LOCKSTEP_N_MAX < INT32_MAX
    SUBCASE("a value above the ceiling clamps to the max") {
        char buf[16];
        snprintf(buf, sizeof(buf), "%lld", (long long)SMEM_LOCKSTEP_N_MAX + 1);
        CHECK(bwa3_lockstep_width_parse_env(buf) == SMEM_LOCKSTEP_N_MAX);
    }
#endif
    SUBCASE("a malformed value is -1 (invalid: reported, not a silent floor)") {
        CHECK(bwa3_lockstep_width_parse_env("garbage") == -1);
        CHECK(bwa3_lockstep_width_parse_env("12x")     == -1);
    }
    SUBCASE("a non-positive value is -1 (invalid: reported, not a silent floor)") {
        CHECK(bwa3_lockstep_width_parse_env("0")  == -1);
        CHECK(bwa3_lockstep_width_parse_env("-5") == -1);
    }
    SUBCASE("an overflowed value is -1 (invalid: reported, not a silent clamp to max)") {
        CHECK(bwa3_lockstep_width_parse_env("99999999999999999999999999") == -1);
    }
}

// --- bwa3_lockstep_probe_enabled: gate the startup MLP probe opt-in ---
//
// BWA3_SMEM_LOCKSTEP_PROBE is a truthy opt-in: the startup sweep runs only for
// a value that is present, non-empty, and not "0". Unset, empty, and "0" all
// leave the probe off (the shipped default keeps the compile-time width). This
// pins the gating the initializer keys off so an empty or "0" value can never
// silently pay the probe's startup cost.

TEST_CASE("lockstep width: probe_enabled is a truthy opt-in (unset/empty/\"0\" disable)"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("unset, empty, or \"0\" leaves the probe disabled") {
        CHECK(bwa3_lockstep_probe_enabled(nullptr) == 0);
        CHECK(bwa3_lockstep_probe_enabled("")      == 0);
        CHECK(bwa3_lockstep_probe_enabled("0")     == 0);
    }
    SUBCASE("any other value enables the probe") {
        CHECK(bwa3_lockstep_probe_enabled("1")   != 0);
        CHECK(bwa3_lockstep_probe_enabled("2")   != 0);
        CHECK(bwa3_lockstep_probe_enabled("yes") != 0);
    }
}

// --- third-pass bwtseed lockstep: on/off policy ---
//
// The rule is "one worker thread per physical core": the lockstep overlaps
// cp_occ misses that an idle SMT sibling would otherwise leave exposed, and
// stops paying once both siblings are busy. Unknown topology keeps the platform
// default. The env pin is a strict 0/1 so a typo can never silently flip the
// driver. The one-shot installer and the arm64 always-on branch are exercised
// by the whole-run byte-identity gates, not here.

TEST_CASE("bwtseed lockstep: rule enables only when threads <= physical cores"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("threads at or below the core count enable it") {
        CHECK(bwa3_bwtseed_lockstep_rule(1,  16, 0) == 1);
        CHECK(bwa3_bwtseed_lockstep_rule(16, 16, 0) == 1);
    }
    SUBCASE("more threads than cores (SMT siblings busy) disable it") {
        CHECK(bwa3_bwtseed_lockstep_rule(17, 16, 1) == 0);
        CHECK(bwa3_bwtseed_lockstep_rule(32, 16, 1) == 0);
    }
    SUBCASE("unknown topology keeps the platform default, whichever it is") {
        CHECK(bwa3_bwtseed_lockstep_rule(8,  0, 0) == 0);
        CHECK(bwa3_bwtseed_lockstep_rule(8,  0, 1) == 1);
        CHECK(bwa3_bwtseed_lockstep_rule(8, -1, 1) == 1);
    }
}

TEST_CASE("bwtseed lockstep: parse_env is a strict 0/1 pin with unset and invalid distinct"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("unset or empty is -1 (apply the rule)") {
        CHECK(bwa3_bwtseed_lockstep_parse_env(nullptr) == -1);
        CHECK(bwa3_bwtseed_lockstep_parse_env("")      == -1);
    }
    SUBCASE("\"0\" and \"1\" pin off and on") {
        CHECK(bwa3_bwtseed_lockstep_parse_env("0") == 0);
        CHECK(bwa3_bwtseed_lockstep_parse_env("1") == 1);
    }
    SUBCASE("anything else is -2 (invalid: reported, then the rule applies)") {
        CHECK(bwa3_bwtseed_lockstep_parse_env("2")    == -2);
        CHECK(bwa3_bwtseed_lockstep_parse_env("yes")  == -2);
        CHECK(bwa3_bwtseed_lockstep_parse_env("01")   == -2);
        CHECK(bwa3_bwtseed_lockstep_parse_env(" 1")   == -2);
    }
}

TEST_CASE("bwtseed lockstep: resolve composes pin > arm64 default > rule"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("a 0/1 pin wins over everything, on either platform") {
        CHECK(bwa3_bwtseed_lockstep_resolve(1, /*arm64*/0, 32, 16, 0) == 1);
        CHECK(bwa3_bwtseed_lockstep_resolve(0, /*arm64*/1, 8,  16, 1) == 0);
    }
    SUBCASE("arm64 keeps its compiled default without consulting the rule") {
        CHECK(bwa3_bwtseed_lockstep_resolve(-1, 1, 64, 16, 1) == 1);   // would be off by the rule
        CHECK(bwa3_bwtseed_lockstep_resolve(-1, 1, 64,  0, 1) == 1);   // unknown topology too
    }
    SUBCASE("unset and invalid both fall through to the rule elsewhere") {
        CHECK(bwa3_bwtseed_lockstep_resolve(-1, 0, 16, 16, 0) == 1);
        CHECK(bwa3_bwtseed_lockstep_resolve(-2, 0, 16, 16, 0) == 1);
        CHECK(bwa3_bwtseed_lockstep_resolve(-1, 0, 32, 16, 0) == 0);
        CHECK(bwa3_bwtseed_lockstep_resolve(-2, 0, 32, 16, 0) == 0);
        CHECK(bwa3_bwtseed_lockstep_resolve(-1, 0, 32,  0, 0) == 0);   // unknown: default
    }
}

#if defined(__aarch64__)
TEST_CASE("bwtseed lockstep: init resolves default from the compile-time platform default, "
          "not a leaked prior value"
          * doctest::test_suite("unit/smem")) {
    // Regression: bwa3_init_bwtseed_lockstep must feed the compile-time platform
    // default (arm64 -> on) as `default_on`, never the mutable g_bwtseed_lockstep.
    // Otherwise an explicit pin (or an unknown-topology resolution) leaks into a
    // later run whose environment is unset -- which on arm64 must revert to the
    // compiled default. Deterministic only on arm64, where resolve returns
    // default_on directly without consulting the host-dependent core count.
    const int32_t saved_global = g_bwtseed_lockstep;
    const char *saved_env = getenv("BWA3_BWTSEED_LOCKSTEP");
    const bool had_env = saved_env != NULL;
    const std::string saved_env_val = had_env ? saved_env : "";

    setenv("BWA3_BWTSEED_LOCKSTEP", "0", 1);
    bwa3_init_bwtseed_lockstep(8);
    CHECK(g_bwtseed_lockstep == 0);   // explicit pin honored

    unsetenv("BWA3_BWTSEED_LOCKSTEP");
    bwa3_init_bwtseed_lockstep(8);
    CHECK(g_bwtseed_lockstep == 1);   // reverts to the arm64 compiled default, not the leaked pin

    if (had_env) setenv("BWA3_BWTSEED_LOCKSTEP", saved_env_val.c_str(), 1);
    else unsetenv("BWA3_BWTSEED_LOCKSTEP");
    g_bwtseed_lockstep = saved_global;
}
#endif

namespace {
// A synthetic sysfs cpu tree: `online` plus one thread_siblings_list per CPU.
// Files are written verbatim so the test controls every spelling the parser
// must accept. Returns the root; caller removes it with remove_tree.
struct SysfsTree {
    std::string root;
    std::vector<std::string> files;  // for cleanup, deepest first
    std::vector<std::string> dirs;
    bool ok = true;
    explicit SysfsTree(const char *online) {
        const char *tmp = getenv("TMPDIR");
        std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/bwa3_sysfs_XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end()); buf.push_back('\0');
        char *made = mkdtemp(buf.data());
        if (!made) { ok = false; return; }
        root = made;
        write("online", online);
    }
    void write(const std::string &rel, const char *content) {
        const std::string path = root + "/" + rel;
        FILE *fp = fopen(path.c_str(), "w");
        if (!fp) { ok = false; return; }
        fputs(content, fp); fclose(fp); files.push_back(path);
    }
    void cpu(int n, const char *siblings) {
        const std::string d1 = root + "/cpu" + std::to_string(n);
        const std::string d2 = d1 + "/topology";
        if (mkdir(d1.c_str(), 0700) != 0 || mkdir(d2.c_str(), 0700) != 0) { ok = false; return; }
        dirs.push_back(d2); dirs.push_back(d1);
        write("cpu" + std::to_string(n) + "/topology/thread_siblings_list", siblings);
    }
    ~SysfsTree() {
        for (const auto &f : files) unlink(f.c_str());
        for (const auto &d : dirs) rmdir(d.c_str());
        if (!root.empty()) rmdir(root.c_str());
    }
};
}  // namespace

TEST_CASE("bwtseed lockstep: the sysfs parser counts one core per sibling-list leader"
          * doctest::test_suite("unit/smem")) {
    SUBCASE("4 cores x 2 SMT threads, comma-spelled sibling lists (x86 style: 0,4)") {
        SysfsTree t("0-7\n");
        const char *sib[8] = {"0,4\n", "1,5\n", "2,6\n", "3,7\n", "0,4\n", "1,5\n", "2,6\n", "3,7\n"};
        for (int i = 0; i < 8; i++) t.cpu(i, sib[i]);
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 4);   // not 8: physical, not logical
    }
    SUBCASE("range-spelled sibling lists (0-1) and a fragmented online list") {
        SysfsTree t("0-3,8-11\n");
        const char *sib[4] = {"0-1\n", "0-1\n", "2-3\n", "2-3\n"};
        for (int i = 0; i < 4; i++) t.cpu(i, sib[i]);
        const char *sib2[4] = {"8-9\n", "8-9\n", "10-11\n", "10-11\n"};
        for (int i = 0; i < 4; i++) t.cpu(8 + i, sib2[i]);
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 4);
    }
    SUBCASE("no SMT: every CPU leads its own list") {
        SysfsTree t("0-2\n");
        for (int i = 0; i < 3; i++) t.cpu(i, (std::to_string(i) + "\n").c_str());
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 3);
    }
    SUBCASE("a missing sibling file is unknown (0), never a partial count") {
        SysfsTree t("0-1\n");
        t.cpu(0, "0\n");
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 0);
    }
    SUBCASE("a missing or malformed online list is unknown (0)") {
        SysfsTree t("garbage\n");
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 0);
        CHECK(bwa3_physical_core_count_from("/nonexistent/bwa3/cpu") == 0);
    }
    SUBCASE("trailing non-whitespace after a single token is unknown (0), not a miscount") {
        // "0x": the numeric prefix parses to 0, but the 'x' suffix means the
        // token is malformed. Present cpu0 so the count would be 1 if the suffix
        // were ignored -- the parser must reject it as unknown instead.
        SysfsTree t("0x\n");
        t.cpu(0, "0\n");
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 0);
    }
    SUBCASE("trailing non-whitespace after a range endpoint is unknown (0), not a miscount") {
        // "0-1x": lo=0, hi=1 parse, but the trailing 'x' on the endpoint is junk.
        // Present cpu0/cpu1 so the count would be 2 if the suffix were ignored.
        SysfsTree t("0-1x\n");
        t.cpu(0, "0\n");
        t.cpu(1, "1\n");
        REQUIRE(t.ok);
        CHECK(bwa3_physical_core_count_from(t.root.c_str()) == 0);
    }
    SUBCASE("an affinity mask counts only the physical cores the process may run on") {
        // 4 cores x 2 SMT threads, x86-style sibling spelling (leader, leader+4).
        SysfsTree t("0-7\n");
        const char *sib[8] = {"0,4\n", "1,5\n", "2,6\n", "3,7\n", "0,4\n", "1,5\n", "2,6\n", "3,7\n"};
        for (int i = 0; i < 8; i++) t.cpu(i, sib[i]);
        REQUIRE(t.ok);
        const unsigned char full[8]      = {1, 1, 1, 1, 1, 1, 1, 1};  // all CPUs
        const unsigned char one_core[8]  = {1, 0, 0, 0, 1, 0, 0, 0};  // both SMT threads of core 0
        const unsigned char non_leader[8] = {0, 0, 0, 0, 0, 1, 0, 0}; // only cpu5 (leader is cpu1)
        const unsigned char two_cores[8] = {1, 1, 0, 0, 0, 0, 0, 0};  // cpu0 (core0) + cpu1 (core1)
        const unsigned char none[8]      = {0, 0, 0, 0, 0, 0, 0, 0};  // no CPU allowed
        // NULL mask and a full mask both reproduce the host-wide count.
        CHECK(bwa3_physical_core_count_masked_from(t.root.c_str(), nullptr, 0) == 4);
        CHECK(bwa3_physical_core_count_masked_from(t.root.c_str(), full, 8) == 4);
        // Restricted to one core's two SMT threads: one physical core, not two.
        CHECK(bwa3_physical_core_count_masked_from(t.root.c_str(), one_core, 8) == 1);
        // A single non-leader sibling still makes its physical core available.
        CHECK(bwa3_physical_core_count_masked_from(t.root.c_str(), non_leader, 8) == 1);
        // Two CPUs on distinct cores: two physical cores.
        CHECK(bwa3_physical_core_count_masked_from(t.root.c_str(), two_cores, 8) == 2);
        // No allowed CPU: zero cores available (the sysfs read itself is fine).
        CHECK(bwa3_physical_core_count_masked_from(t.root.c_str(), none, 8) == 0);
        // A malformed sibling list is still unknown (0) under a mask.
        SysfsTree bad("0-1\n");
        bad.cpu(0, "0x\n");
        bad.cpu(1, "1\n");
        REQUIRE(bad.ok);
        CHECK(bwa3_physical_core_count_masked_from(bad.root.c_str(), full, 8) == 0);
    }
}

// The production entry point bwa3_physical_core_count() reads live, mutable host
// topology (sysfs / sysctl), so it cannot be pinned by a deterministic unit test
// -- CPU hotplug or affinity changes between calls would make it flaky, and the
// path instructions require unit tests to use only programmatically generated
// synthetic inputs. Its parsing logic is fully covered by bwa3_physical_core_count_from
// against the synthetic sysfs trees above; the live host smoke check belongs in an
// integration test, not here.
