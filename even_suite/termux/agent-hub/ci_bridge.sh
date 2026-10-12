#!/usr/bin/env bash
# CI only: start Agent Hub on :8799 with a fake `claude`, in a private tmux,
# and export BRIDGE_TEST_URL so the Flutter tests can talk to it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
H="$(mktemp -d)"
mkdir -p "$H/.local/bin"
cat > "$H/.local/bin/claude" <<'PY'
#!/usr/bin/env python3
import json, sys
prompt = sys.stdin.read().strip().splitlines()
mark = " [glasses]" if "--append-system-prompt" in sys.argv else ""
print(json.dumps({"result": "echo: " + (prompt[-1] if prompt else "") + mark, "is_error": False}))
PY
chmod +x "$H/.local/bin/claude"
cp "$H/.local/bin/claude" "$H/.local/bin/agy"
env -u TMUX HOME="$H" HUB_PORT=8799 HUB_NO_BROWSER=1 TMUX_TMPDIR="$H" HUB_WORKDIR="$H" \
  nohup python3 "$HERE/server.py" > "$H/server.log" 2>&1 &
for _ in $(seq 50); do
  if curl -fs http://127.0.0.1:8799/api/ping > /dev/null; then
    echo "BRIDGE_TEST_URL=http://127.0.0.1:8799" >> "${GITHUB_ENV:-/dev/null}"
    echo "Test bridge is up on :8799"
    exit 0
  fi
  sleep 0.2
done
cat "$H/server.log"
exit 1
