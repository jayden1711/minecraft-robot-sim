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

  # ghosts: a stale ghost from an "earlier session" must be removed on spawn, the fresh ghost must read back as non-air,
  # and /robot stop must leave no ghost entity in the world
  count_ghosts() { # prints the entity count of the last `execute if entity` answer, 0 on "Test failed"
    local n0; n0=$(grep -c -E 'Test (passed|failed)' "$LOG")
    echo "execute in minecraft:overworld run execute if entity @e[tag=mcrobot_ghost]" >&3
    for i in $(seq 1 15); do [ "$(grep -c -E 'Test (passed|failed)' "$LOG")" -gt "$n0" ] && break; sleep 1; done
    local l; l=$(grep -E 'Test (passed|failed)' "$LOG" | tail -1)
    case "$l" in *[Cc]ount:*) echo "${l##*[Cc]ount: }" | tr -dc 0-9 ;; *) echo 0 ;; esac
  }
  echo 'execute in minecraft:overworld run summon minecraft:block_display 12 -58 12 {Tags:["mcrobot_ghost"]}' >&3
  sleep 2
  [ "$(count_ghosts)" = 0 ] || { echo "FAIL($MC): a ghost outside any session was not discarded on load"; exit 1; }
  echo "PASS($MC): stale ghost discarded on load"
  # control for the air read-back: a display with the default (air) block_state must be matched by one of the two NBT forms
  echo 'execute in minecraft:overworld run summon minecraft:block_display 12 -58 12 {Tags:["smoke_air"]}' >&3
  sleep 2
  NA=$(grep -c -E 'Test (passed|failed)' "$LOG")
  echo 'execute in minecraft:overworld run execute if entity @e[tag=smoke_air,nbt={block_state:"minecraft:air"}]' >&3
  echo 'execute in minecraft:overworld run execute if entity @e[tag=smoke_air,nbt={block_state:{Name:"minecraft:air"}}]' >&3
  sleep 3
  grep -E 'Test (passed|failed)' "$LOG" | tail -n +"$((NA + 1))" | grep -q 'Test passed' || { echo "FAIL($MC): air read-back query does not match an air display"; exit 1; }
  echo "PASS($MC): air read-back query matches an air display"
  echo 'execute in minecraft:overworld run kill @e[tag=smoke_air]' >&3
  echo "robot demo" >&3
  waitfor 'ghost spawned: [0-9]* parts at' 120 "$LOG"
  NP=$(grep -o 'ghost spawned: [0-9]* parts' "$LOG" | tail -1 | tr -dc 0-9)
  if grep -q 'ghost check failed\|ERROR: ghost' "$LOG"; then echo "FAIL($MC): ghost read-back reported a problem"; grep 'ghost check failed\|ERROR: ghost' "$LOG"; exit 1; fi
  [ "$(count_ghosts)" = "$NP" ] || { echo "FAIL($MC): expected exactly $NP ghost entities (stale one removed)"; exit 1; }
  echo "robot stop" >&3
  waitfor 'ghost streaming stopped: stopped by command' 30 "$LOG"
  sleep 1
  [ "$(count_ghosts)" = 0 ] || { echo "FAIL($MC): ghosts remain after robot stop"; exit 1; }
  echo "PASS($MC ghosts): $NP parts spawned, stale ghost removed, read-back non-air, none left after stop"

  # ---- sky arena (data pack function), scale, run timer, gamerules, mob clearing, cam command, demo on the arena with both ghost styles
  test_cmd() { # command; prints yes if the answer was "Test passed", else no
    local n0 i; n0=$(grep -c -E 'Test (passed|failed)' "$LOG")
    echo "$1" >&3
    for i in $(seq 1 40); do [ "$(grep -c -E 'Test (passed|failed)' "$LOG")" -gt "$n0" ] && break; sleep 0.5; done
    if grep -E 'Test (passed|failed)' "$LOG" | tail -n +"$((n0 + 1))" | head -1 | grep -q passed; then echo yes; else echo no; fi
  }
  count_of() { # selector -> number of matching entities
    local n0 i l; n0=$(grep -c -E 'Test (passed|failed)' "$LOG")
    echo "execute in minecraft:overworld run execute if entity $1" >&3
    for i in $(seq 1 30); do [ "$(grep -c -E 'Test (passed|failed)' "$LOG")" -gt "$n0" ] && break; sleep 0.5; done
    l=$(grep -E 'Test (passed|failed)' "$LOG" | tail -1)
    case "$l" in *[Cc]ount:*) echo "${l##*[Cc]ount: }" | tr -dc 0-9 ;; *) echo 0 ;; esac
  }
  expect() { # description, command, expected yes|no
    [ "$(test_cmd "$2")" = "$3" ] || { echo "FAIL($MC arena): $1"; tail -15 "$LOG"; exit 1; }
  }
  PYTHONPATH="$ROOT/server" "$PY" "$ROOT/scripts/export_arena.py" > "$OUT/arena_export.txt"  # also (re)writes results/m1/arena/, which `robot demo arena` reads
  SYN_SHA=$(grep -o 'sha256=[0-9a-f]*' "$OUT/arena_export.txt" | head -1 | cut -d= -f2)
  EXPORT_ARGS=$(grep '^export_args' "$OUT/arena_export.txt" | cut -d' ' -f2-)
  echo "scoreboard players reset * mcr_arena" >&3  # the smoke world is kept between runs: start the counters from zero
  echo "execute in minecraft:overworld run function mcrobot_demo:arena" >&3
  for i in $(seq 1 60); do [ "$(test_cmd 'execute if score #arena mcr_arena matches 1')" = yes ] && break; sleep 1; done
  expect "arena was not built (score #arena)" 'execute if score #arena mcr_arena matches 1' yes
  N0=$(grep -c "export ack" "$LOG")
  echo "execute in minecraft:overworld run robot export $EXPORT_ARGS" >&3
  for i in $(seq 1 60); do [ "$(grep -c "export ack" "$LOG")" -gt "$N0" ] && break; sleep 1; done
  ACK3=$(grep 'export ack' "$LOG" | tail -1)
  case "$ACK3" in *"\"sha256\":\"$SYN_SHA\""*) echo "PASS($MC arena export): the real exporter's MJCF of the built arena is byte-identical to the synthetic arena (sha256 ${SYN_SHA:0:16}...)" ;;
    *) echo "FAIL($MC arena): exported arena differs from mcrobot.arena: $ACK3 (expected sha256 $SYN_SHA)"; exit 1 ;; esac
  grep -q '"meters_per_block": 0.5' "$OUT"/exports/region_*.json || { echo "FAIL($MC): export header does not record meters_per_block 0.5"; exit 1; }
  echo "PASS($MC scale): export header records meters_per_block 0.5"
  GR=$(for r in advance_time advance_weather; do n=$(grep -c "$r" "$LOG"); echo "gamerule $r" >&3; sleep 1; grep "$r" "$LOG" | tail -1; done)
  echo "$GR" | grep -c 'false' | grep -qx 2 || { echo "FAIL($MC arena): gamerules not locked: $GR"; exit 1; }
  echo "PASS($MC arena): advance_time and advance_weather are false"
  # run timer: start twice (nothing-changed errors must not break it), progress, finish, fall
  echo "function mcrobot_demo:run/start" >&3; sleep 1; echo "function mcrobot_demo:run/start" >&3; sleep 2
  expect "timer not running" 'execute if score #running mcr_arena matches 1' yes
  expect "timer ticks do not count" 'execute if score #ticks mcr_arena matches 20..' yes
  echo "function mcrobot_demo:run/progress {pct:42}" >&3; sleep 1
  expect "progress not stored" 'execute if score #progress mcr_arena matches 42' yes
  echo "function mcrobot_demo:run/finish {cs:1234}" >&3; sleep 1
  expect "finish did not stop the timer" 'execute if score #running mcr_arena matches 0' yes
  expect "best time" 'execute if score Best_0.1s mcr_arena matches 123' yes
  expect "runs counter" 'execute if score Runs mcr_arena matches 1' yes
  echo "function mcrobot_demo:run/fall" >&3; sleep 1
  expect "falls counter" 'execute if score Falls mcr_arena matches 1' yes
  echo "PASS($MC arena): boss bar timer and scoreboard (start twice, progress, finish, fall)"
  # mob-free: a pig inside the arena box is removed, a control pig outside is not
  echo "execute in minecraft:overworld run forceload add 0 0 40 40" >&3; sleep 3  # the ghost stop above removed the earlier forceload there
  echo "execute in minecraft:overworld run summon minecraft:pig 110 205 -145" >&3
  echo "execute in minecraft:overworld run summon minecraft:pig 20 -60 20" >&3
  sleep 4
  [ "$(count_of '@e[type=minecraft:pig,x=100,y=190,z=-170,dx=60,dy=60,dz=60]')" = 0 ] && echo "PASS($MC arena): pig inside the arena removed" || { echo "FAIL($MC arena): mob not cleared"; exit 1; }
  if [ "$(test_cmd 'execute if entity @e[type=minecraft:pig,x=0,y=-70,z=0,dx=60,dy=20,dz=60]')" != yes ]; then
    echo 'execute as @e[type=minecraft:pig] run data get entity @s Pos' >&3; sleep 2; tail -4 "$LOG"
    echo "FAIL($MC arena): control pig outside the arena must survive"; exit 1
  fi
  echo 'execute in minecraft:overworld run kill @e[type=minecraft:pig]' >&3
  # /robot cam from the console (no player entity): the function must exist and run; tp @s needs a player, so only look for unknown-function errors
  echo "robot cam 1" >&3; sleep 2
  grep -q 'camera spot 1' "$LOG" && ! grep -q 'Unknown or invalid function\|Unknown function' "$LOG" || { echo "FAIL($MC): robot cam"; tail -5 "$LOG"; exit 1; }
  echo "PASS($MC arena): /robot cam 1 ran the data pack function"
  # demo on the arena, both ghost styles: item_display parts with the item_model component, then block_display parts
  for style in models blocks; do
    echo "robot style $style" >&3; sleep 1
    echo "robot demo arena" >&3
    NS0=$(grep -c 'ghost spawned: [0-9]* parts at' "$LOG")
    for i in $(seq 1 120); do [ "$(grep -c 'ghost spawned: [0-9]* parts at' "$LOG")" -gt "$NS0" ] && break; sleep 1; done
    [ "$(grep -c 'ghost spawned: [0-9]* parts at' "$LOG")" -gt "$NS0" ] || { echo "FAIL($MC arena): no ghost with style $style"; tail -10 "$LOG"; exit 1; }
    NA=$(grep -o 'ghost spawned: [0-9]* parts' "$LOG" | tail -1 | tr -dc 0-9)
    if [ "$style" = models ]; then
      NI=$(count_of '@e[type=minecraft:item_display,tag=mcrobot_ghost,nbt={item:{id:"minecraft:stick"}}]')
      NC=$(count_of '@e[type=minecraft:item_display,tag=mcrobot_ghost,nbt={item:{components:{"minecraft:item_model":"mcrobot:g1/pelvis"}}}]')
      [ "$NI" = "$NA" ] && [ "$NC" = 1 ] || { echo "FAIL($MC arena): models style: $NI of $NA item displays, $NC with the pelvis item_model"; exit 1; }
      echo "PASS($MC arena): models style: $NA item_display parts with item_model components on the server"
    else
      NB=$(count_of '@e[type=minecraft:block_display,tag=mcrobot_ghost]')
      [ "$NB" = "$NA" ] || { echo "FAIL($MC arena): blocks style: $NB of $NA block displays"; exit 1; }
      echo "PASS($MC arena): blocks style: $NA block_display parts"
    fi
    sleep 3
    expect "run timer not started by the demo" 'execute if score #running mcr_arena matches 1' yes
    echo "robot stop" >&3; sleep 3
    [ "$(count_of '@e[tag=mcrobot_ghost]')" = 0 ] || { echo "FAIL($MC arena): ghosts left after stop"; exit 1; }
  done
  echo "PASS($MC arena): demo on the arena with both styles; run events reached the data pack"

  # ---- sky arena v2: build from the data pack, export byte-identical to the synthetic export (snow, trapdoor, piston, beacon and banner collision shapes as the
  # game reports them), labels and piston heads survive the mob clearing, cues light a checkpoint and move a piston, no redstone block anywhere in the arena
  PYTHONPATH="$ROOT/server" "$PY" "$ROOT/scripts/export_arena_v2.py" > "$OUT/arena_v2_export.txt"  # also (re)writes results/m1/arena_v2/, which `robot demo arena_v2` reads
  SYN2=$(grep -o 'sha256=[0-9a-f]*' "$OUT/arena_v2_export.txt" | head -1 | cut -d= -f2)
  EXPORT_ARGS2=$(grep '^export_args' "$OUT/arena_v2_export.txt" | cut -d' ' -f2-)
  read -r STRIP3 <<< "$(grep '^probe_strip3' "$OUT/arena_v2_export.txt" | cut -d' ' -f2-)"
  read -r LAMP3 <<< "$(grep '^probe_lamp3' "$OUT/arena_v2_export.txt" | cut -d' ' -f2-)"
  read -r NLABELS NRAMS <<< "$(grep '^probe_labels' "$OUT/arena_v2_export.txt" | cut -d' ' -f2-)"
  echo "execute in minecraft:overworld run function mcrobot_demo:arena_v2" >&3
  for i in $(seq 1 90); do [ "$(test_cmd 'execute if score #v2 mcr_arena matches 1')" = yes ] && break; sleep 1; done
  expect "arena v2 was not built (score #v2)" 'execute if score #v2 mcr_arena matches 1' yes
  N0=$(grep -c "export ack" "$LOG")
  echo "execute in minecraft:overworld run robot export $EXPORT_ARGS2" >&3
  for i in $(seq 1 90); do [ "$(grep -c "export ack" "$LOG")" -gt "$N0" ] && break; sleep 1; done
  ACK4=$(grep 'export ack' "$LOG" | tail -1)
  case "$ACK4" in *"\"sha256\":\"$SYN2\""*) echo "PASS($MC arena v2 export): the real exporter's MJCF of the built arena v2 is byte-identical to the synthetic one (sha256 ${SYN2:0:16}...)" ;;
    *) echo "FAIL($MC arena v2): exported arena differs from mcrobot.arena_v2: $ACK4 (expected sha256 $SYN2)"; exit 1 ;; esac
  LASTJ=$(ls -1 "$OUT"/exports/region_*.json | tail -1)  # the header of that export lists every block type the game placed
  if grep -qiE 'redstone|repeater|comparator|observer' "$LASTJ"; then echo "FAIL($MC arena v2): the built arena contains a redstone block: $(grep -iE 'redstone|repeater|comparator|observer' "$LASTJ" | head -3)"; exit 1; fi
  echo "PASS($MC arena v2): no redstone, repeater, comparator or observer among the block types the game placed"
  sleep 3  # at least one pass of the once-a-second mob clearing
  [ "$(count_of '@e[type=minecraft:text_display,tag=mcr_label]')" = "$NLABELS" ] || { echo "FAIL($MC arena v2): expected $NLABELS labels"; exit 1; }
  [ "$(count_of '@e[type=minecraft:block_display,tag=mcr_ram]')" = "$NRAMS" ] || { echo "FAIL($MC arena v2): expected $NRAMS piston heads"; exit 1; }
  echo 'data get entity @e[type=minecraft:block_display,tag=mcr_ram,limit=1] block_state' >&3  # diagnostics for the log
  echo 'data get entity @e[type=minecraft:text_display,tag=mcr_label,limit=1] text' >&3; sleep 1
  expect "STAIRS label text" 'execute if entity @e[type=minecraft:text_display,tag=mcr_label,nbt={text:{text:"STAIRS"}}]' yes
  expect "a piston head reads back as air" 'execute if entity @e[tag=mcr_ram,nbt={block_state:"minecraft:air"}]' no
  expect "a piston head reads back as air (compound form)" 'execute if entity @e[tag=mcr_ram,nbt={block_state:{Name:"minecraft:air"}}]' no
  PH1=$(test_cmd 'execute if entity @e[tag=mcr_ram,nbt={block_state:{id:"minecraft:piston_head"}}]'); PH2=$(test_cmd 'execute if entity @e[tag=mcr_ram,nbt={block_state:{Name:"minecraft:piston_head"}}]')
  [ "$PH1" = yes ] || [ "$PH2" = yes ] || { echo "FAIL($MC arena v2): the piston heads do not read back as piston_head"; exit 1; }
  echo "PASS($MC arena v2): $NLABELS labels and $NRAMS piston heads exist, survive the mob clearing, none is air, STAIRS text reads back"
  expect "checkpoint strip starts off" "execute if block $STRIP3 minecraft:gray_concrete" yes
  echo "function mcrobot_demo:run/cue {n:3}" >&3; sleep 2
  expect "cue 3 did not light the strip" "execute if block $STRIP3 minecraft:emerald_block" yes
  expect "cue 3 did not light the gate lamp" "execute if block $LAMP3 minecraft:sea_lantern" yes
  expect "cue 3 did not set the checkpoint score" 'execute if score #cp mcr_arena matches 3' yes
  echo "function mcrobot_demo:run/cue {n:12}" >&3; sleep 0.6
  expect "cue 12 did not extend the piston" 'execute if entity @e[tag=mcr_ram_2,nbt={transformation:{translation:[0.0f,0.0f,0.0f]}}]' yes
  sleep 1.5
  expect "piston 2 did not retract after the hold" 'execute if entity @e[tag=mcr_ram_2,nbt={transformation:{translation:[0.0f,0.0f,1.0f]}}]' yes
  echo "function mcrobot_demo:run/start" >&3; sleep 2
  expect "run/start did not switch the strip off" "execute if block $STRIP3 minecraft:gray_concrete" yes
  expect "run/start did not reset the checkpoint score" 'execute if score #cp mcr_arena matches 0' yes
  echo "function mcrobot_demo:run/fall" >&3
  echo "PASS($MC arena v2): cue 3 lit the strip, the gate lamp and #cp; cue 12 slid piston 2 out and back; run/start reset it all"
  echo "robot cam 5" >&3; sleep 2
  grep -q 'camera spot 5' "$LOG" || { echo "FAIL($MC arena v2): robot cam 5"; tail -5 "$LOG"; exit 1; }
  echo "robot style blocks" >&3; sleep 1
  echo "robot demo arena_v2" >&3
  NS0=$(grep -c 'ghost spawned: [0-9]* parts at' "$LOG")
  for i in $(seq 1 120); do [ "$(grep -c 'ghost spawned: [0-9]* parts at' "$LOG")" -gt "$NS0" ] && break; sleep 1; done
  [ "$(grep -c 'ghost spawned: [0-9]* parts at' "$LOG")" -gt "$NS0" ] || { echo "FAIL($MC arena v2): no ghost on arena_v2"; tail -10 "$LOG"; exit 1; }
  sleep 3
  expect "run timer not started by the demo on arena_v2" 'execute if score #running mcr_arena matches 1' yes
  echo "robot stop" >&3; sleep 3
  [ "$(count_of '@e[tag=mcrobot_ghost]')" = 0 ] || { echo "FAIL($MC arena v2): ghosts left after stop"; exit 1; }
  echo "PASS($MC arena v2): robot cam 5 and robot demo arena_v2 ran, run events reached the data pack"
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
