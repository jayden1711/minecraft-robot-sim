#!/usr/bin/env bash
# End-to-end smoke test for one Minecraft version: a real Fabric server (flat throwaway world, offline) loads the mod jar
# built for that version, the real Python server runs, and the console runs `robot ping` and `robot export` against real blocks.
# Usage: smoke_server.sh <mc_version> [--via]   (--via: also install ViaFabric/ViaVersion/ViaBackwards and check protocol translation)
# Needs MCROBOT_EULA_FILE (the accepted eula.txt), mod/dist/*.jar, .venv.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/versions.sh"
MC="${1:?mc version}"; VIA=0; SFX=""; [ "${2:-}" = "--via" ] && { VIA=1; SFX="-via"; }
: "${MCROBOT_EULA_FILE:?set MCROBOT_EULA_FILE to the accepted eula.txt}"
PY="${MCROBOT_PY:-$ROOT/.venv/bin/python}"
IDX=$(printf '%s\n' "${MC_VERSIONS[@]}" | grep -nx "$MC" | cut -d: -f1)
PORT=$((25800 + IDX)); BPORT=$((25700 + IDX))
DIR="$ROOT/server-mc-test/$MC$SFX"
OUT="$ROOT/results/smoke/$MC$SFX"
mkdir -p "$OUT"
wait_for_load

"$ROOT/scripts/setup_server.sh" "$MC" "$DIR" $([ "$VIA" = 1 ] && echo --via) --smoke --eula-from "$MCROBOT_EULA_FILE" --port "$PORT" --bridge-port "$BPORT"
# a stale world from an earlier run is kept on purpose (flat, deterministic); logs are per run
LOG="$OUT/server.log"; : > "$LOG"
PYLOG="$OUT/python.log"

PYTHONPATH="$ROOT/server" nice -n 10 "$PY" -m mcrobot.server --port "$BPORT" --out "$OUT/exports" > "$PYLOG" 2>&1 &
PYPID=$!
if [ "$MC" != 1.20.1 ]; then mkdir -p "$DIR/world/datapacks"; cp -r "$ROOT/datapack/mcrobot_demo" "$DIR/world/datapacks/"; fi
FIFO="$DIR/console.fifo"; [ -p "$FIFO" ] || mkfifo "$FIFO"
exec 3<>"$FIFO"
JAVA_HOME="$(jdk_home "$(java_major "$MC")")"
( exec env -C "$DIR" nice -n 10 "$JAVA_HOME/bin/java" -Xms256M -Xmx1G -XX:ActiveProcessorCount=2 -jar fabric-server-launch.jar nogui <&3 > "$LOG" 2>&1 ) &
MCPID=$!
cleanup() { echo stop >&3 2>/dev/null || true; sleep 8; kill "$MCPID" "$PYPID" 2>/dev/null || true; exec 3>&-; }
trap cleanup EXIT

waitfor() { # pattern timeout_s file
  local i; for i in $(seq 1 "$2"); do
    grep -q -- "$1" "$3" 2>/dev/null && return 0
    kill -0 "$MCPID" 2>/dev/null || { echo "FAIL($MC): server exited early"; tail -20 "$LOG"; return 1; }
    sleep 1
  done
  echo "FAIL($MC): timed out waiting for '$1'"; tail -20 "$LOG"; return 1
}

waitfor 'Done (' 420 "$LOG"
grep -q "mcrobot $MOD_VERSION loaded" "$LOG" || { echo "FAIL($MC): mod did not report loading"; exit 1; }
echo "robot ping" >&3
waitfor 'pong' 30 "$LOG"
echo "execute in minecraft:overworld run robot export 0 -64 0 3 -60 3" >&3
waitfor 'export ack' 30 "$LOG"
ACK=$(grep 'export ack' "$LOG" | tail -1)
case "$ACK" in
  *'"solid_cells":64'*'"collider_boxes":1'*|*'"collider_boxes":1'*'"solid_cells":64'*) echo "PASS($MC): $ACK" ;;
  *) echo "FAIL($MC): unexpected ack: $ACK"; exit 1 ;;
esac

# demo data pack (functions/ became function/ in 1.21, so the pack only targets >= 1.21): build the maze, export it, expect its known box and marker counts
if [ "$MC" != 1.20.1 ]; then
  echo "execute in minecraft:overworld run forceload add 8 8 24 24" >&3  # no player online: chunks are not loaded, fills would fail
  sleep 3
  echo "execute in minecraft:overworld positioned 10 -60 10 run function mcrobot_demo:maze" >&3
  sleep 3
  N0=$(grep -c "export ack" "$LOG")
  echo "execute in minecraft:overworld run robot export 10 -60 10 18 -57 18" >&3
  for i in $(seq 1 30); do [ "$(grep -c "export ack" "$LOG")" -gt "$N0" ] && break; sleep 1; done
  ACK2=$(grep 'export ack' "$LOG" | tail -1)
  case "$ACK2" in
    *'"collider_boxes":16'*'"markers":5'*) echo "PASS($MC demo data pack): 16 boxes, 5 markers" ;;
    *) echo "FAIL($MC): demo maze export unexpected: $ACK2"; exit 1 ;;
  esac
fi

if [ "$VIA" = 1 ]; then
  grep -q 'ViaVersion detected server version' "$LOG" && grep -qi 'viabackwards' "$LOG" || { echo "FAIL($MC): Via did not load"; exit 1; }
  # client protocol numbers -> names; the server answers with the client's number only if Via can translate that client
  FAILV=0; CLAIM=()
  for pair in 107:1.9 340:1.12.2 754:1.16.5 758:1.18.2 762:1.19.4 763:1.20.1 765:1.20.4 767:1.21.1 769:1.21.4 774:1.21.11 775:26.1 776:26.2 777:26.3; do
    p=${pair%%:*}; n=${pair##*:}
    r=$("$PY" "$ROOT/scripts/ping_status.py" 127.0.0.1 "$PORT" "$p" 2>&1 || true)
    if echo "$r" | grep -q "\"answered\": $p"; then echo "via-ok   $n ($p)"; else echo "via-FAIL $n ($p): $r"; FAILV=1; fi
  done
  [ "$FAILV" = 0 ] || exit 1
  echo "PASS($MC via)"
fi

if [ "$VIA" = 0 ]; then
  # negative control for the Via check: a plain server must NOT answer a 1.9 client (107) with 107
  r=$("$PY" "$ROOT/scripts/ping_status.py" 127.0.0.1 "$PORT" 107)
  if echo "$r" | grep -q '"answered": 107'; then echo "FAIL($MC): plain server claims to translate 1.9: $r"; exit 1; fi
  echo "control-ok: plain server answers its own protocol: $r"
fi
