# Practice website

Read README.md and DEPLOYMENT.md here. The user authorized public sharing on
November 9, 2026 and selected a temporary URL because DNS management is unavailable.
Preserve the pinned strategy and benchmark source. Both deployments were retired on November 13 during project archival.
Historically the public server and its independent tunnel used tmux
`poker-web-public` on 127.0.0.1:8789; the SSH-only server used `poker-web`
on 127.0.0.1:8788. Restore dependencies and artifacts before starting either.
Do not restart a live service without considering the user's
in-memory table; static edits load on refresh, engine edits affect new tables.

Each mode uses its existing Python environment in a separate worker process.
Do not merge incompatible PokerKit versions into the HTTP process. Never send
engine organizer logs, bot cards, deck seeds or bot memories to the browser.
Only the human's own cards and genuinely public reveals belong in snapshots.

Respect hand boundaries, SDK event-delivery bookkeeping, simultaneous selection
and exact legal raise totals. Invalid human requests must leave the game
unchanged. Retain focused tests for these boundaries when changing an adapter.
Runtime files, sessions, browser artifacts and logs belong in `.runtime/web/`.
Public cookies must be Secure; the tunnel must preserve the browser Host header.
Do not change or restart the unrelated `chat.kaiming.me` tunnel or services.
Restarting the quick tunnel changes the public URL; verify HTTPS gameplay and
update deployment notes before sharing the new address.
