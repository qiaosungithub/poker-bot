# November 13 project archive

This closeout preserves the November 9–10 work. The local `poker-bot` checkout
was deleted after all recovery assets passed verification; see the
[cleanup receipt](cleanup-receipt.json). No strategy, frozen benchmark definition, opponent, seed
or historical measurement is changed by the archive operation.

## Where to start

- [Auction Flop guide](../../auction_flop/README.md) and
  [macro comparison](../../auction_flop/research/macro_program/README.md): the
  strongest retained local v1 mean is 19.80713 for prior + Hboth. Its transfer
  result against different opponent styles remains inconclusive.
- [Patient-policy cycle](../../auction_flop/research/patient_loop/RESULTS.md):
  previously local simple-policy experiments, including rejected candidates,
  complete raw benchmark tables, fresh-seed confirmation and packaging evidence.
- [Build-box timing](../../fullhouse-repro/latency/2026-11-09-buildbox-comparison/REPORT.md):
  the previously unpublished November 9 Fullhouse host comparison.
- [Model inventory](model-manifest.json): five small public model exports,
  including selected iteration 210, the final comparison export and the
  released ONNX reference. Hashes match the original artifacts.

The last retained active-submission receipt is separate from the local leader;
see [submission confirmation](../../auction_flop/evidence/submission-confirmation.json).
Final official standings were not queried as part of this closeout.

## Event history

`event-history.bundle` contains the complete nested event repository, all 22
development branches and its tags. `event-documents/` preserves its final main
branch handoff documents as historical records, without relabeling old status
as current. Restore independently of the parent checkout:

```bash
git clone --mirror archive/2026-11-13/event-history.bundle event-history.git
git clone event-history.git event-source
```

Uncommitted files in the event worktrees are preserved in the encrypted
workspace archive below. The bundle alone contains only Git objects and refs.
The Fullhouse training modification remains in
[`training_changes.patch`](../../fullhouse-repro/training_changes.patch).

## Encrypted workspace recovery

The [workspace archive release](https://github.com/qiaosungithub/poker-bot/releases/tag/workspace-archive-2026-11-13)
holds a compressed, encrypted snapshot of the local checkout, including
training/resume checkpoints, all nested Git/worktree metadata, unpublished
files, downloaded competition data and private session records. These contents
are not plaintext public Git files. Regenerable Python environments, package
caches, downloaded Python installation and temporary/cache directories are
excluded; the exact exclusions and encrypted-part hashes are in
`release-manifest.json`. Environment dependency lists are in `environments/`.

Encryption uses GnuPG AES-256 with integrity protection and a randomly generated
256-bit passphrase. The passphrase is **not** in Git or the release. Its retained
location on the original server is:

`/home/sqa/.config/poker-bot-archive/2026-11-13.passphrase` (mode 0600).

Keep that file separately when moving accounts or machines. The encrypted
archive cannot be recovered from GitHub alone. Normal benchmark use and the
five public model exports do not need this key.

Download all release assets into an empty directory, then verify and decrypt:

```bash
sha256sum -c SHA256SUMS
cat poker-bot-workspace.tar.zst.gpg.part-* > workspace.tar.zst.gpg
gpg --batch --pinentry-mode loopback \
  --passphrase-file /home/sqa/.config/poker-bot-archive/2026-11-13.passphrase \
  --output workspace.tar.zst --decrypt workspace.tar.zst.gpg
# Extract only after gpg exits successfully, into a new empty directory.
mkdir restored
tar --zstd -xf workspace.tar.zst -C restored
```

The archive retains historical absolute worktree paths and symlinks. Restore
to the original location or repair worktree paths with `git worktree repair`
before using linked checkouts. Recreate environments rather than expecting
absolute-path virtual environments to be portable. Fetch current `main` from
GitHub for archive receipts added after the snapshot was made.

## Cleanup boundary

The cleanup covers this server's `poker-bot/` directory and its two local
practice servers/tunnel. It does not remove the organizer's remote build
machine, unrelated services, SSH configuration, the official API key stored
outside the project, or the archive recovery key. A compact local receipt is
retained outside the removed checkout. See `release-manifest.json` for the
verified snapshot and cleanup scope.
