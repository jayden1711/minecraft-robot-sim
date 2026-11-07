#!/usr/bin/env bash
# Full test matrix: for EVERY supported Minecraft version, run (1) the Python tests, (2) the mod build + Java tests including the
# mod <-> Python server integration test (InteropTest must run, not skip), and (3) with --smoke a real Fabric server boot with the
# mod jar and a console `robot ping` / `robot export` against real blocks.
# Usage: scripts/test_matrix.sh [--smoke] [version ...]     (default: all versions)
# Env: MCROBOT_PY (python with numpy/mujoco deps, default .venv/bin/python), MCROBOT_JDK_17/21/25 (default: ~/opt/jdk-<n>.*),
#      MCROBOT_EULA_FILE (owner's accepted eula.txt; required for --smoke).
# Runs at nice 10, Gradle with 2 workers; waits while NUC load is >= 20. Summary goes to results/matrix/summary.tsv (git-ignored).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/versions.sh"

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi
VERSIONS=("$@"); [ ${#VERSIONS[@]} -gt 0 ] || VERSIONS=("${MC_VERSIONS[@]}")

export MCROBOT_PY="${MCROBOT_PY:-$ROOT/.venv/bin/python}"
export MCROBOT_PYPATH="$ROOT/server"
for n in 17 21 25; do
  if d=$(jdk_home "$n" 2>/dev/null); then export "MCROBOT_JDK_$n=$d"; fi
done
GRADLE_JAVA="${MCROBOT_JDK_25:?need JDK 25 to run Gradle (Loom 1.19.4)}"
mkdir -p "$ROOT/results/matrix"
SUMMARY="$ROOT/results/matrix/summary.tsv"
printf 'version\tpython\tmod_build_and_java_tests\tinterop_ran\tsmoke\n' > "$SUMMARY"
FAIL=0

for v in "${VERSIONS[@]}"; do
  wait_for_load || exit 1
  echo "=== Minecraft $v ==="
  py=FAIL; build=FAIL; interop=NO; smoke=skipped

  if MCROBOT_MC_VERSION="$v" PYTHONPATH="$ROOT/server" nice -n 10 "$MCROBOT_PY" -m pytest "$ROOT/server/tests" -q -p no:cacheprovider; then py=pass; fi

  if JAVA_HOME="$GRADLE_JAVA" nice -n 10 "$ROOT/mod/gradlew" -p "$ROOT/mod" --max-workers=2 --console=plain ":$v:build"; then build=pass; fi
  xml="$ROOT/mod/versions/$v/build/test-results/test/TEST-dev.mcrobot.InteropTest.xml"
  if [ -f "$xml" ] && grep -q 'tests="1"' "$xml" && grep -q 'skipped="0"' "$xml" && grep -q 'failures="0"' "$xml"; then interop=yes; fi

  if [ "$SMOKE" = 1 ]; then
    if "$ROOT/scripts/smoke_server.sh" "$v"; then smoke=pass; else smoke=FAIL; fi
    # versions with pinned Via jars (the NUC server's version) also get the Via protocol-translation check
    if [ "$smoke" = pass ] && grep -q "^via@$v	" "$ROOT/scripts/pins.tsv"; then
      if "$ROOT/scripts/smoke_server.sh" "$v" --via; then smoke=pass+via; else smoke=FAIL; fi
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$v" "$py" "$build" "$interop" "$smoke" >> "$SUMMARY"
  if [ "$py" != pass ] || [ "$build" != pass ] || [ "$interop" != yes ] || [ "$smoke" = FAIL ]; then FAIL=1; fi
done

echo
column -t -s $'\t' "$SUMMARY"
if [ "$FAIL" = 0 ]; then echo "MATRIX PASS"; else echo "MATRIX FAIL"; exit 1; fi
