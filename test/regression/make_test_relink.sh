#!/usr/bin/env bash
# test/regression/make_test_relink.sh
#
# Regression: every standalone binary test/Makefile builds is relinked when
# ../libbwa.a changes.
#
# Was: the standalone rules depended only on their own object, so `make <test>`
# after a src/ change said "up to date" and the test ran against the OLD
# library. A parity test then passes or fails against code that is no longer
# in the tree, and nothing says so.
#
# The check is a dry run: it first refreshes the binaries' mtimes so none is
# older than ../libbwa.a (earlier CI probe steps can re-archive the library
# after the binaries were linked, which is not this check's concern), then
# `make -n <bin>` must print no link for any binary; after ../libbwa.a is
# touched, it must print one for every binary. Nothing is compiled or linked —
# only mtimes are moved, and the library's and every binary's mtime are put
# back afterwards so the next real build sees no spurious change.
#
# Inputs:
#   MAKE — make binary (default: make). No MAKE_ARGS: test/Makefile is driven
#          directly and only under -n, so the flags the tree was built with do
#          not matter here.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MAKE="${MAKE:-make}"
LIB=libbwa.a

if [[ ! -f "$LIB" ]]; then
    echo "FAIL: $LIB is not built — run this against a built tree"
    exit 1
fi

# Whole-second mtime, for the same reason as make_header_deps.sh: GNU Make 3.81
# (Apple's stock make) compares whole seconds, so the touched library has to
# land a full second past the binaries or make sees nothing to do.
if stat -c %Y . > /dev/null 2>&1; then
    mtime_sec() { stat -c %Y "$1"; } # GNU coreutils
else
    mtime_sec() { stat -f %m "$1"; } # BSD / macOS
fi

# Enumerated from the Makefile, not from a copy of its list, so a binary added
# to EXE without the ../libbwa.a prerequisite fails here.
read -r -a all_bins <<< "$("$MAKE" -s -C test print-EXE)"
if ((${#all_bins[@]} == 0)); then
    echo "FAIL: test/Makefile EXE is empty — is print-% missing?"
    exit 1
fi

# The relink assertion only means something when it covers the COMPLETE EXE set.
# Filtering absent entries out and asserting over the survivors would let a
# missing binary slip through while the remaining ones report PASS, so require
# every enumerated binary to be built and executable and fail loudly on any that
# is not, rather than proving the dependency only for whatever happens to exist.
missing=()
for bin in "${all_bins[@]}"; do
    [[ -x "test/$bin" ]] || missing+=("test/$bin")
done
if ((${#missing[@]} != 0)); then
    echo "FAIL: standalone test binaries not built or not executable: ${missing[*]} — run 'make -C test' first"
    exit 1
fi
# Validated above: every EXE entry exists, so every assertion loop below covers
# the full set rather than a subset that silently dropped the missing ones.
bins=("${all_bins[@]}")

# would_link <bin>: true when a dry run of test/Makefile would link <bin>.
would_link() {
    "$MAKE" -n -C test "$1" 2> /dev/null | grep -qE -- "(^| )-o $1( |\$)"
}

# Snapshot the library's and every binary's mtime up front and restore them all
# on exit, so this check leaves the tree exactly as it found it regardless of
# where it fails.
MTIME_REF="$(mktemp)"
touch -r "$LIB" "$MTIME_REF"
BIN_REFDIR="$(mktemp -d)"
for bin in "${bins[@]}"; do
    touch -r "test/$bin" "$BIN_REFDIR/$bin"
done
restore_mtimes() {
    local b
    touch -r "$MTIME_REF" "$LIB" 2> /dev/null || true
    for b in "${bins[@]}"; do
        touch -r "$BIN_REFDIR/$b" "test/$b" 2> /dev/null || true
    done
    rm -rf "$MTIME_REF" "$BIN_REFDIR"
}
trap restore_mtimes EXIT

# --- Baseline: make every built binary newer than the library, so the
# --- precondition below is clean no matter what order CI ran the earlier
# --- build/probe steps in. Some of those recompile a src object and re-archive
# --- libbwa.a after the binaries were linked, leaving the library newer than
# --- them; a real build would relink the binaries, so that is not this check's
# --- concern and it must not mask the dependency being verified. Whole-second
# --- bump, for the GNU Make 3.81 reason noted above.
lib_m="$(mtime_sec "$LIB")"
for _ in 1 2 3; do
    for bin in "${bins[@]}"; do touch "test/$bin"; done
    oldest_bin=0
    for bin in "${bins[@]}"; do
        m="$(mtime_sec "test/$bin")"
        ((oldest_bin == 0 || m < oldest_bin)) && oldest_bin=$m
    done
    ((oldest_bin > lib_m)) && break
    sleep 1
done
if (($(mtime_sec "$LIB") >= oldest_bin)); then
    echo "FAIL: could not make the test binaries newer than $LIB (timestamp granularity?)"
    exit 1
fi

fail=0

# --- Precondition: with every binary now newer than the library and its own
# --- object, a dry run must want to link nothing. A link line here means a
# --- prerequisite the refresh above did not cover — a genuinely out-of-date
# --- tree — so fail loudly rather than proving nothing below.
for bin in "${bins[@]}"; do
    if would_link "$bin"; then
        echo "FAIL: test/$bin is out of date even after refreshing the binaries — build test/ first"
        fail=1
    fi
done
((fail == 0)) || exit 1

newest=0
for bin in "${bins[@]}"; do
    m="$(mtime_sec "test/$bin")"
    ((m > newest)) && newest=$m
done
for _ in 1 2 3; do
    touch "$LIB"
    (($(mtime_sec "$LIB") > newest)) && break
    sleep 1
done
if (($(mtime_sec "$LIB") <= newest)); then
    echo "FAIL: could not make $LIB newer than the test binaries (timestamp granularity?)"
    exit 1
fi

# --- The check: every standalone binary relinks after the library changed.
relinked=0
for bin in "${bins[@]}"; do
    if would_link "$bin"; then
        relinked=$((relinked + 1))
    else
        echo "FAIL: test/$bin would NOT relink after $LIB changed — its rule lacks the ../libbwa.a prerequisite"
        fail=1
    fi
done

if ((fail != 0)); then
    echo "FAIL: test/Makefile standalone binaries do not all track $LIB"
    exit 1
fi
echo "PASS: all $relinked built standalone test binaries relink after $LIB changes (${#all_bins[@]} listed in EXE)"
