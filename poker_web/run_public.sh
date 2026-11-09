#!/usr/bin/env bash
# Dedicated temporary HTTPS deployment; leaves the SSH-only server untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="$PWD/.runtime/web"
mkdir -p "$runtime"
exec 9>"$runtime/public.lock"
flock -n 9 || { echo "The public launcher is already running." >&2; exit 1; }
command -v cloudflared >/dev/null
python_bin="$PWD/.runtime/envs/poker-hu-validation/bin/python"
"$python_bin" - <<'PY'
import socket
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 8789))
PY
server_pid=''
tunnel_pid=''
cleanup() {
  trap - EXIT INT TERM
  for pid in "$tunnel_pid" "$server_pid"; do
    if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  wait || true
  rm -f "$runtime/public-url.txt"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PYTHONDONTWRITEBYTECODE=1 "$python_bin" -u -m poker_web.server \
  --port 8789 --secure-cookie --idle-timeout 1800 \
  >>"$runtime/public-server.log" 2>&1 9>&- &
server_pid=$!
ready=0
for ((i=0; i<40; i++)); do
  kill -0 "$server_pid" 2>/dev/null || { echo "Server exited; inspect public-server.log." >&2; exit 1; }
  if curl --silent --fail http://127.0.0.1:8789/health >/dev/null; then ready=1; break; fi
  sleep .25
done
[[ "$ready" == 1 ]] || { echo "Server did not become ready." >&2; exit 1; }

# Isolate this from every named tunnel and any default cloudflared config.
printf 'no-autoupdate: true\n' >"$runtime/quick-tunnel.yml"
touch "$runtime/public-tunnel.log"
offset=$(wc -c <"$runtime/public-tunnel.log")
cloudflared --config "$runtime/quick-tunnel.yml" tunnel \
  --url http://127.0.0.1:8789 --metrics 127.0.0.1:20243 \
  >>"$runtime/public-tunnel.log" 2>&1 9>&- &
tunnel_pid=$!
for ((i=0; i<60; i++)); do
  kill -0 "$tunnel_pid" 2>/dev/null || { echo "Tunnel exited; inspect public-tunnel.log." >&2; exit 1; }
  if "$python_bin" - "$runtime" "$offset" <<'PY'
from pathlib import Path
import re
import sys
runtime = Path(sys.argv[1])
with (runtime / "public-tunnel.log").open() as log:
    log.seek(int(sys.argv[2]))
    urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log.read())
if not urls:
    sys.exit(1)
(runtime / "public-url.txt").write_text(urls[-1] + "\n")
print("Tunnel URL (verify HTTPS before sharing): " + urls[-1], flush=True)
PY
  then break; fi
  sleep 1
done
[[ -s "$runtime/public-url.txt" ]] || { echo "No tunnel URL received." >&2; exit 1; }
wait -n "$server_pid" "$tunnel_pid"
