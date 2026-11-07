#!/usr/bin/env bash
# Start the headless Fabric Minecraft server in server-mc/ (git-ignored). Needs eula.txt accepted by the owner.
set -euo pipefail
ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
JAVA_HOME="${JAVA_HOME:-$HOME/opt/jdk-21.0.13+9}"
cd "$ROOT/server-mc"
exec nice -n 10 "$JAVA_HOME/bin/java" -Xms512M -Xmx2G -XX:ActiveProcessorCount=2 -jar fabric-server-launch.jar nogui
