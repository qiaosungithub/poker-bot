# Infrastructure and terminal workflow

Checked 2026-11-08. The user runs the agent in a **server terminal**, not a local
ChatGPT desktop app. There are three different environments:

| Environment | Purpose | Status |
|---|---|---|
| This research server and `poker-bot/` | Existing training, benchmarks and documentation | Working; Fullhouse training complete |
| Organizer EC2 build box | Team development/testing near the production runtime | Offered by portal; not configured or accessed by this agent |
| Organizer bot microVMs | Execute submitted bots in scored tables | No access or accepted submission verified |

The build box is described as similar to tournament machines, not proof of an
identical CPU model. Exact instance/CPU specifications have not been established.

## Own equipment and GPU training: evidence boundary

The official setup guide's section 2 offers either the team's provided remote
build machine or setup on the participant's own computer; section 6 documents
the local route. Local development/testing is therefore explicitly supported.

The portal's **no GPU / no network** restriction describes submitted bots in
the gameplay sandbox. It does not, by itself, prohibit GPU training elsewhere.
However, the public rules, signed-in Getting started / Format, setup guide and
recorded organizer email do **not explicitly settle whether participants may
use their own remote GPU/TPU training servers during the event**, or any compute
quota. Own GPU training is plausible, but has not been confirmed as an event
permission. Do not report it as confirmed or assume the provided build box has
a GPU. No organizer clarification has been requested or received by this agent.

The separate ban on pre-written bots/code still applies. The existing Fullhouse
checkpoint's eligibility is unresolved; local hardware support is not permission
to submit a pre-event model. See [event coding rules](rules.md#november-8-organizer-email).

## Portal identity and current access

- Portal: <https://pokerbot.sequoiacap.com/tournament>.
- Boston competition ID: `abb5539f-60ce-48b6-8925-c0c56a3102e3`.
- Observed team ID: `f570222b-b5b0-4dce-8499-a32476665215`.
- Login identity: `sqa24@mit.edu`.
- [Team page](https://pokerbot.sequoiacap.com/tournament/teams/f570222b-b5b0-4dce-8499-a32476665215).
- [API-key settings](https://pokerbot.sequoiacap.com/tournament#settings/api-keys)
  and [SSH-key settings](https://pokerbot.sequoiacap.com/tournament#settings/ssh-keys).

On November 8 we authenticated, verified `/api/me`, and rendered Getting started
and Format. We did not create API keys, register SSH keys, claim credits, connect
to the EC2 box, upload a bot, or change the active bot. Do not infer that another
team member has or has not done those things.

## Logging in from a server

The observed flow checks Cloudflare Turnstile before sending an email code.
Submitting a valid email code then succeeded from this server without another
Turnstile token. This is an observation of this site, not a universal guarantee.

The user can request a fresh code in any ordinary browser. The Outlook connector
can find the email but may redact an active one-time code. Do not retry expired
codes or attempt to defeat redaction. The prepared local helper accepts a code
with hidden terminal input:

```bash
cd /kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot
python .runtime/private-web/login_code.py
```

It verifies the authenticated identity endpoint and saves a private cookie jar.
The start page is client-rendered, so the first raw HTML/text fetch may contain
only a title. That does not mean the documentation is absent.

Private operational files live in `.runtime/private-web/` (directory mode 700,
credential files mode 600). `cookies.txt` and `browser-state.json` are secret;
never print their contents or put them in docs/submission archives. Saved
`start-rendered.txt` and `format-rendered.txt` are the successfully rendered
source captures used for these notes. Session validity must be rechecked.

The ad-hoc browser helpers reuse an existing Playwright installation under
`/tmp/21h-browser-qa/` and Chromium under `/home/sqa/.cache/ms-playwright/`.
Those are external prerequisites, not canonical poker project dependencies.
If absent, install replacement tooling under `.runtime/`, not root storage.
The successful launch needed `LD_PRELOAD` unset for the child process, the
existing browser dependency library directory, and a short work-local `TMPDIR`
such as `.runtime/tmp` to stay within Unix socket path limits. No desktop app
was needed. These details describe the helper, not tournament requirements.

## Official setup guide

The organizer's [setup skill](https://pokerbot.sequoiacap.com/tournament/no-limit-2026-setup/SKILL.md)
was read as documentation, **not invoked for setup**. Observed version:
`2026-11-05.2`; re-download before applying it. It can register an SSH public key,
prepare the SSH alias and later install the starter kit. The portal recommends a
smaller model for this routine setup. Every competitor is offered $50 of OpenAI
credits; claiming them is a separate account action.

For a future authorized setup, inspect existing configuration before changing
it. Keep private keys and the tournament API key on the connecting machine; only
the public `.pub` key goes to the organizer. Do not place token values in this
documentation or shell examples. The guide's normal API base is
`https://pokerbot.sequoiacap.com/tournament/api`:

| API endpoint | Documented purpose |
|---|---|
| `GET /auth/api-key` | Validate bearer key and obtain associated teams |
| `GET /teams/{team_id}/machine` | Obtain machine host, registered key fingerprints and kit download link |
| `POST /teams/{team_id}/machine/keys` | Register the chosen public SSH key during setup |

Use the download URL returned by the API; do not guess a starter-kit URL. A null
machine/host means provisioning is not ready. Kit download can return 404 before
the variant reveal. Existing portal cookie access is not an API key.

## SSH build box without a desktop app

1. Register an ed25519/RSA/ECDSA public key through account settings; maximum five
   per account, normally active within thirty seconds. DSA and `-sk` keys are not
   supported. Each teammate uses their own key.
2. Read the real SSH address from the team's **Build box** tab. Outbound TCP 22
   must be available. The user on the box is `ubuntu`.
3. Use a dedicated SSH host alias, `IdentitiesOnly yes`, no agent forwarding and
   `StrictHostKeyChecking accept-new`. A changed known host key requires checking
   with the organizer, not disabling host-key checks.
4. Before the configured opening time the server may print the UTC opening time
   and disconnect. That can mean the key worked and access is time-gated.
5. Once open, work in `~/work`. Wait for `cloud-init status --wait` if provisioning
   has not finished. The starter kit is expected at `~/work/starter-kit`.

After the alias has been configured with the real host and private-key path:

```bash
ssh no-limit-2026-build
cd ~/work
tmux new-session -A -s no-limit-2026
codex login status
# Only if not already signed in:
codex login --device-auth
# Complete its browser sign-in, then verify status and run:
codex
```

The last Codex sign-in on a shared box replaces the previous one; preserve a
working team sign-in. The optional desktop-app connection is not needed for this
terminal workflow. These organizer-documented commands have not yet been tested
on this team's box.

## Starter kit and local verification

At reveal, inspect the actual kit's `AGENTS.md`, `CLAUDE.md`, `README.md`,
`RULES.md` and `requirements.txt`. It contains the protocol, local engine, SDK,
example bots, ICM utilities and sample hand-history reader.

Inside the extracted kit on the build box:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ./sdk/agent-protocol -e ./sdk/engine -e ./sdk/agent-sdk
.venv/bin/poker-agent check agent.py
.venv/bin/poker-agent play . examples/check_or_fold examples/min_raiser --tables 2 --no-history
```

This example is **three-handed**. Local play supports 2–10 folder arguments;
repeat folders to obtain eight seats. It reports score, finishing places and
failed decisions per seat. Use `--seed` for replay and increase samples after the
smoke test. Changes to the local SDK/engine do not change production.

Package a separate bot directory with `poker-agent pack DIR OUT.zip`, run checks
on the whole archive, and inspect acceptance logs and the active-bot indicator
after upload. [Runtime notes](runtime.md) explain why a local pass with a time
bank does not prove acceptance under the bank-free 1 ms upload test.

## This server's research tooling

All paths below are relative to `/kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot`:

| Path | Use |
|---|---|
| `.runtime/envs/poker-fullhouse-train312/bin/python` | Python 3.12 training/export environment, CPU PyTorch plus JAX/TPU backend |
| `.runtime/envs/poker-hu-validation/bin/python` | Frozen Python 3.11 benchmark environment; not the official Python 3.12 runtime |
| `fullhouse-repro/runtime_paths.py` | Redirects cache, temporary files, TPU logs and model caches under `.runtime` |
| `fullhouse-repro/status.json` | Completed run result; safe to inspect without launching training |

Use [Fullhouse commands](../research/fullhouse.md) for evaluation and timing.
Do not rerun `supervise.py` for a status check. Do not restart training into the
completed run directory. For new TPU work, first use the workspace's infra
operations instructions and verify device ownership/live allocation; the old
run's device choice is not a reservation.
