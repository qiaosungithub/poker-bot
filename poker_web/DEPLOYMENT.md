# Public practice deployment

The user authorized public sharing on November 9, 2026 and chose a temporary
URL after confirming they could not manage the domain's Cloudflare settings.

Retired on November 13, 2026: both owned poker tmux sessions and the dedicated
Quick Tunnel were stopped for project cleanup. The old temporary URL is inactive.
The components and commands below are historical deployment/recovery guidance.
See [the archive](../archive/2026-11-13/README.md) to restore artifacts first.

## Historical components

| Component | Location |
| --- | --- |
| Public launcher | `bash poker_web/run_public.sh`, tmux `poker-web-public` |
| HTTP origin | `127.0.0.1:8789`, `--secure-cookie --idle-timeout 1800` |
| HTTPS ingress | Dedicated Cloudflare Quick Tunnel; no account credentials |
| Tunnel metrics/readiness | `127.0.0.1:20243` |
| Current URL | `.runtime/web/public-url.txt` |
| Launcher / server / tunnel logs | `.runtime/web/public-{launch,server,tunnel}.log` |
| Per-table process logs | `.runtime/web/logs/` |
| Original SSH-only instance | `127.0.0.1:8788`, tmux `poker-web` |

All paths are relative to `/kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot`.
The public instance uses the existing pinned Python environments, source and
models. The launcher holds an exclusive lock, rejects an occupied origin port,
checks the server, starts an isolated tunnel and stops its own children if
either exits. The HTTP server closes table workers on graceful shutdown.

This is a temporary team service. Each browser gets its own human-vs-bots table;
it is not a multiplayer room. Up to eight tables are retained, and inactivity
for 30 minutes frees a slot. The server checks expiry even without new visitors.
No account is required, and anyone with the URL can play. State lives only in
memory, and browser cookies expire after three hours. Refresh can resume a
table while both state and cookie remain valid.

## Inspect and start

```bash
tmux list-sessions
cat .runtime/web/public-url.txt
curl --fail http://127.0.0.1:8789/health
curl --fail http://127.0.0.1:20243/ready
```

Only if `poker-web-public` is absent and port 8789 is unused:

```bash
tmux new-session -d -s poker-web-public \
  -c /kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot \
  'bash poker_web/run_public.sh >> .runtime/web/public-launch.log 2>&1'
```

The URL file records what Cloudflare assigned; its existence alone does not
prove the public endpoint works. Request `<URL>/health` over HTTPS and play a
hand through the browser. Verify both classic and Bay Area before handing over
a deployment that changes the server or game code.

## Stop, restart and recover

Attach with `tmux attach -t poker-web-public` and press Ctrl-C to stop its
launcher, server and tunnel. Do not kill other tmux sessions or processes by
pattern. A restart loses public in-memory tables and changes the tunnel URL;
consider current players before doing it. Then use the start command above,
verify the new link, and update this document and README. Static assets are
read per request; editing them only needs browser refresh, not a server restart.

The launcher removes the current URL file when it exits normally. After a VM
failure the file may be stale: check the processes and external HTTPS response.
tmux survives an SSH disconnect, but does not automatically recover a VM reboot.
Cloudflare may reconnect transient transport failures without recreating the
tunnel. If its process exits, relaunch and distribute the new URL.

Use [Cloudflare's Quick Tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
for service limits: temporary random hostname, no uptime guarantee, 200 in-flight
requests and no SSE. The current application uses ordinary HTTP requests.

## Future fixed domain

The existing `chat.kaiming.me` and `family-chat.kaiming.me` sites use a different
named tunnel from the `agent-web` checkout. Their credentials permit running
that tunnel, but this server has no discovered DNS/account-management credential.
Neither those sites nor their tunnel configuration was changed for this task.

For `poker.kaiming.me`, ask the zone administrator to create a dedicated named
tunnel and hostname route. Keep its credential in ignored private storage and
point ingress to `http://127.0.0.1:8789`, preserving the request Host. HTTPS is
terminated by Cloudflare; keep `--secure-cookie` enabled. Do not assume a tunnel
credential grants DNS management, or mix new routes into the chat tunnel.

## Verification evidence

`public-validation.json` records the tested public URL, timestamp, source hashes
and browser checks. Original local engine checks remain in `validation.json`.
Screenshots and raw browser output are under ignored `.runtime/web/qa/`.
Run the existing browser script with `POKER_WEB_URL` set to the live HTTPS URL;
it verifies real gameplay in both modes, mobile layout, hidden cards, cookie
flags and cross-origin rejection, using disposable browser sessions.
