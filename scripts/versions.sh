#!/usr/bin/env bash
# Sourced by the other scripts: supported Minecraft versions, JDKs, and shared helpers.
# Supported versions are the Stonecutter versions in mod/settings.gradle; the Fabric loader is pinned in mod/gradle.properties.
ROOT="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
MC_VERSIONS=(1.20.1 1.21.11 26.3)
FABRIC_LOADER=0.20.1
FABRIC_INSTALLER=1.1.2
MOD_VERSION=0.0.6

# Java major version each Minecraft version needs (from Mojang's version json javaVersion).
java_major() {
  case "$1" in
    1.20.1) echo 17 ;;
    1.21.11) echo 21 ;;
    26.*) echo 25 ;;
    *) echo "unknown Minecraft version $1" >&2; return 1 ;;
  esac
}

# Directory of the JDK for a Java major: MCROBOT_JDK_<n> if set, else the Temurin install in ~/opt.
jdk_home() {
  local n="$1" var="MCROBOT_JDK_$1" d
  if [ -n "${!var:-}" ]; then echo "${!var}"; return; fi
  d=$(ls -d "$HOME"/opt/jdk-"$n".* 2>/dev/null | sort -V | tail -1)
  [ -n "$d" ] || { echo "no JDK $n: set MCROBOT_JDK_$n or install Temurin $n in ~/opt" >&2; return 1; }
  echo "$d"
}

load_now() { cut -d' ' -f1 /proc/loadavg; }

# Block until the 1-minute load average is under 20 (NUC rule), polling every 30 s for up to 30 min.
wait_for_load() {
  local i
  for i in $(seq 1 60); do
    if awk -v l="$(load_now)" 'BEGIN{exit !(l<20)}'; then return 0; fi
    echo "load $(load_now) >= 20, waiting" >&2
    sleep 30
  done
  echo "load stayed above 20" >&2
  return 1
}
