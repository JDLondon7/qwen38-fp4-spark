#!/usr/bin/env bash
# memguard.sh NAME [FLOOR_GIB]
# Stops container NAME when host MemAvailable falls below FLOOR_GIB (default 4).
# On the GB10's unified memory the kernel OOM killer (or earlyoom) picks whatever
# is largest or newest, which can be a co-tenant or your desktop. Stopping the
# model ourselves, early, keeps that choice in our hands. Exits with the container.
NAME=$1; FLOOR_KB=$(( ${2:-4} * 1024 * 1024 ))
while [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; do
  AV=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$AV" -lt "$FLOOR_KB" ]; then
    echo "$(date '+%F %T') MemAvailable ${AV} kB < ${2:-4} GiB floor: stopping $NAME"
    docker kill "$NAME"; exit 1
  fi
  sleep 0.5
done
