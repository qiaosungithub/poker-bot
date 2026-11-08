# Bay Area No Limit: public resources

Research date: **2026-11-08**. The November 6 event is officially listed as
**Menlo Park / Bay Area**, rather than a Stanford-hosted tournament. The
[official schedule](https://pokerbot.sequoiacap.com/rules.html) distinguishes it
from Boston on November 9 and the later finals.

## Strongest public source found

[nnicholas-c/poker_sequoia](https://github.com/nnicholas-c/poker_sequoia)
identifies itself as a Bay Area entry. It publishes a starter-kit copy, simulator,
SDK, sample bots, many participant bot versions, and local evaluation reports.
The inspected snapshot is `e6e68a1aa7c71f78fec17bd7ead813687689fc61`.

This is a participant's published copy, not an organizer-owned GitHub release.
The kit's rules agree with its variant engine. That is useful implementation
evidence, but not proof that every file matches the exact final production build.

| Resource | Pinned source |
|---|---|
| Variant rules | [kit/RULES.md](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/kit/RULES.md) |
| Variant implementation | [flop_discard_reveal.py](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/kit/sdk/engine/src/poker_engine/variants/flop_discard_reveal.py) |
| Setup / local simulator | [kit/README.md](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/kit/README.md) |
| Participant strategies | [bots/](https://github.com/nnicholas-c/poker_sequoia/tree/e6e68a1aa7c71f78fec17bd7ead813687689fc61/bots) |
| Example strategy | [gto_supplied/agent.py](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/bots/gto_supplied/agent.py) |
| Local comparison, not contest standings | [gto-first-place report](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/arena/benchmarks/gto-first-place/README.md) |

## Game formulation

From the published kit rules:

1. Standard 52-card deck; deal **three private cards** and play preflop betting.
2. After the three-card flop, **before flop betting**, each remaining player
   privately selects one discard and one of the other two cards to expose.
   There are `3 * 2 = 6` choices. Reveal everyone's exposed cards together.
3. The discard is dead and stays private. Each player now retains one public
   card and one hidden card. A later fold does not hide the exposed card.
4. Continue ordinary no-limit flop, turn and river betting. Showdown takes the
   best five of the board and two retained cards; using both hole cards is not
   required. All-in players must also select.
5. Failed selection discards the third card and exposes the first; it does not
   forfeit the pot. Ordinary invalid betting uses the usual fallback.

The variant implementation identifies itself as `flop_discard_reveal` with
public ID `bay_area`. In `state.phase == "select"`, return
`Action("select", cards=(discard_index, reveal_index))`, with distinct original
indices 0–2. `state.seats[].shown_cards` exposes retained public cards;
`state.variant["discarded"]` carries only the acting player's discard. Choices
are collected before any of the new cards become visible.

The kit also describes three side-prize statistics: low-card revealed bluffs,
all-in bets/raises, and calling frequency. They are not extra chip awards.
Consult `RULES.md` for their exact eligibility and aggregation rules.

**Implication for our research (inference):** Fullhouse's existing two-card
inputs and standard Hold'em training distribution cannot be transplanted
unchanged. A new selection decision, opponents' exposed-card information, and
the known dead discard must enter the policy/equity calculations. This source
does not establish that Boston will use the same variant; Boston's live variant
remains null in the observed competition response.

## Champion and checkpoint status

No verifiable official Bay Area final standings, champion identity, or champion
checkpoint was found in this search. The logged-in competition list exposed
only Boston, so it did not provide a Bay Area leaderboard. Public search results
were largely announcements. This is a search limitation, not proof that no
champion code exists.

The participant repository had no GitHub Releases and no files ending in
`.pt`, `.pth`, `.onnx`, `.npz`, `.ckpt` or `.safetensors` in the inspected tree.
It does contain executable strategies with embedded tables. For example,
`gto_supplied/agent.py` uses preflop equity tables and strategy logic; neither
the folder name nor its local results proves a Nash-equilibrium policy or a
regional victory. The `gto-first-place` report explicitly compares local
candidates and targets first-place frequency. It is not a winner announcement.

Other resources found:

- [JamesGongYC/nolimit-pokerbot](https://github.com/JamesGongYC/nolimit-pokerbot):
  a rehearsal/optimization project with public tree models and a small NPZ
  model. Its README labels the event settings provisional. Not verified as a
  Bay Area winner or as trained on this revealed variant.
- [EthanXu07/poker-bot-overview](https://github.com/EthanXu07/poker-bot-overview):
  model overview only; explicitly says code, thresholds and equity tables are
  private. Its reported results are local simulations, not official placements.

Public availability does not override the Boston email's requirement to write
the competing bot during the event. These links are research references.

## Provenance

The source tree and files were fetched directly from GitHub. Selected raw
captures are retained locally under ignored `.runtime/private-web/`, while
[bay-area-sources.json](bay-area-sources.json) records URLs, commits and SHA-256
hashes without carrying any session credentials. The JSON also records the
search scope used for the negative winner/checkpoint finding.
