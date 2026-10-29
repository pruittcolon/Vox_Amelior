#!/usr/bin/env bash
# Starts Agent Hub inside Ubuntu: the web server in the foreground (and Chrome
# pointed at it). It shows and types into your tmux sessions - the ones your
# `claude` command uses (run ~/agent-hub/allow-app once so tmux lets it in).
# Run from Termux with `all` (see ~/.bashrc on the Termux side).

PORT="${HUB_PORT:-8787}"
URL="http://127.0.0.1:$PORT/"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export LANG="${LANG:-C.UTF-8}"

open_chrome() {
  am start -a android.intent.action.VIEW -d "$URL" -p com.android.chrome >/dev/null 2>&1 \
    || termux-open-url "$URL" 2>/dev/null
}

# Already running (e.g. you typed `all` twice)? Just reopen the page.
if curl -fs -o /dev/null "$URL" 2>/dev/null; then
  echo "Agent Hub is already running - opening $URL"
  # HUB_NO_BROWSER=1: started by the Even Suite app, which is its own window.
  [ "${HUB_NO_BROWSER:-}" = 1 ] || open_chrome
  exit 0
fi

termux-wake-lock 2>/dev/null

printf '\n  \033[38;2;232;132;107m●\033[38;2;122;167;255m●\033[0m  \033[1mAgent Hub\033[0m\n\n'
echo "  Claude Code   -> your 'claude' sessions in Termux (the app lists and opens them)"
echo "  Web page      -> $URL"
echo
echo "  Leave this screen open. Press Ctrl+C here to stop the web page."
echo

exec python3 "$HERE/server.py"
