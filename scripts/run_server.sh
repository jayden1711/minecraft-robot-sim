#!/usr/bin/env bash
# Start a headless Fabric Minecraft server (nice 10, 2 CPU threads, 2 GB heap). Needs eula.txt accepted by the owner.
# Usage: run_server.sh [server-dir]   (default server-mc/; the JDK is chosen from <dir>/.mc_version written by setup_server.sh)
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/versions.sh"
DIR="$(readlink -f "${1:-$ROOT/server-mc}")"
MC="$(cat "$DIR/.mc_version")"
JAVA_HOME="${JAVA_HOME_OVERRIDE:-$(jdk_home "$(java_major "$MC")")}"
cd "$DIR"
exec nice -n 10 "$JAVA_HOME/bin/java" -Xms512M -Xmx2G -XX:ActiveProcessorCount=2 -jar fabric-server-launch.jar nogui
