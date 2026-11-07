#!/usr/bin/env bash
# Prepare a headless Fabric server directory (git-ignored; nothing downloaded here is ever committed or redistributed).
# Usage: setup_server.sh <mc_version> <dir> [--via] [--eula-from FILE] [--port N] [--bridge-port N] [--smoke]
#   --via          also install ViaFabric + ViaVersion + ViaBackwards (pinned in scripts/pins.tsv; only pinned for 26.3)
#   --eula-from F  copy the owner's accepted eula.txt (must say eula=true); without it the server stops at the EULA
#   --smoke        throwaway test world: flat, offline, no whitelist, small view distance
# Needs the mod jar from `mod/gradlew buildAll` in mod/dist/.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/versions.sh"

MC="${1:?mc version}"; DIR="${2:?server dir}"; shift 2
VIA=0; EULA=""; PORT=25566; BPORT=25599; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --via) VIA=1 ;;
    --eula-from) EULA="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --bridge-port) BPORT="$2"; shift ;;
    --smoke) SMOKE=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

java_major "$MC" >/dev/null
mkdir -p "$DIR/mods" "$DIR/config"
DIR="$(readlink -f "$DIR")"
echo "$MC" > "$DIR/.mc_version"

# Fabric server launcher (downloads Mojang's server jar on first start). Its bytes vary per request, so it is not hash-pinned [verify].
if [ ! -s "$DIR/fabric-server-launch.jar" ]; then
  curl -fsSL -m 120 -o "$DIR/fabric-server-launch.jar" \
    "https://meta.fabricmc.net/v2/versions/loader/$MC/$FABRIC_LOADER/$FABRIC_INSTALLER/server/jar"
fi

fetch_pin() { # key
  local key="$1" sha name url
  while IFS=$'\t' read -r k sha name url; do
    [ "$k" = "$key" ] || continue
    if [ ! -s "$DIR/mods/$name" ]; then
      curl -fsSL -m 180 -o "$DIR/mods/$name" "$url"
    fi
    echo "$sha  $DIR/mods/$name" | sha256sum -c --quiet - || { echo "SHA-256 mismatch for $name" >&2; exit 1; }
  done < <(grep -v '^#' "$ROOT/scripts/pins.tsv")
}
fetch_pin "fabric-api@$MC"
if [ "$VIA" = 1 ]; then
  grep -q "^via@$MC" "$ROOT/scripts/pins.tsv" || { echo "no Via jars pinned for $MC" >&2; exit 1; }
  fetch_pin "via@$MC"
fi

JAR="$ROOT/mod/dist/mcrobot-$MOD_VERSION+$MC.jar"
[ -s "$JAR" ] || { echo "missing $JAR: run mod/gradlew buildAll first" >&2; exit 1; }
rm_old=$(ls "$DIR"/mods/mcrobot-*.jar 2>/dev/null || true)
if [ -n "$rm_old" ]; then mkdir -p "$ROOT/results/_trash"; mv $rm_old "$ROOT/results/_trash/"; fi
cp "$JAR" "$DIR/mods/"

if [ -n "$EULA" ]; then
  grep -q '^eula=true' "$EULA" || { echo "$EULA does not say eula=true; the EULA is the owner's to accept" >&2; exit 1; }
  cp "$EULA" "$DIR/eula.txt"
fi

printf 'host=127.0.0.1\nport=%s\n' "$BPORT" > "$DIR/config/mcrobot.properties"

if [ ! -f "$DIR/server.properties" ]; then
  if [ "$SMOKE" = 1 ]; then
    cat > "$DIR/server.properties" <<EOF
server-port=$PORT
online-mode=false
level-type=minecraft\:flat
generate-structures=false
view-distance=4
simulation-distance=4
spawn-protection=0
pause-when-empty-seconds=-1
motd=mcrobot smoke test $MC
EOF
  else
    cat > "$DIR/server.properties" <<EOF
server-port=$PORT
online-mode=true
white-list=true
enforce-whitelist=true
view-distance=8
simulation-distance=6
spawn-protection=16
motd=minecraft-robot-sim (simulation)
EOF
  fi
fi
echo "server dir ready: $DIR ($MC, via=$VIA)"
