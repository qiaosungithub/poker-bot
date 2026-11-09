# Practice website

Read README.md here. Keep this local/SSH-hosted and preserve the pinned strategy
and benchmark source. The server runs in the `poker-web` tmux session on
127.0.0.1:8788. Do not restart a live service without considering the user's
in-memory table; static edits load on refresh, engine edits affect new tables.

Each mode uses its existing Python environment in a separate worker process.
Do not merge incompatible PokerKit versions into the HTTP process. Never send
engine organizer logs, bot cards, deck seeds or bot memories to the browser.
Only the human's own cards and genuinely public reveals belong in snapshots.

Respect hand boundaries, SDK event-delivery bookkeeping, simultaneous selection
and exact legal raise totals. Invalid human requests must leave the game
unchanged. Retain focused tests for these boundaries when changing an adapter.
Runtime files, sessions, browser artifacts and logs belong in `.runtime/web/`.
