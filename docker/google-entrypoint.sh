#!/usr/bin/env bash
set -e

if [ "${ENABLE_NOVNC:-0}" = "1" ]; then
    export DISPLAY="${DISPLAY:-:99}"
    Xvfb "${DISPLAY}" -screen 0 1440x900x24 >/tmp/xvfb.log 2>&1 &
    x11vnc -display "${DISPLAY}" -nopw -forever -shared -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
    websockify --web=/usr/share/novnc 7900 localhost:5900 >/tmp/novnc.log 2>&1 &
fi

exec "$@"
