# Classic txnujk rule continuation

This is a fresh event-time implementation of the decision rules surveyed before
the event, not a copy of a pre-event bot. It isolates a classic betting strategy
while retaining the exact tournament auction from the submitted hybrid. No neural
network, learned weights, or training is used.

The research reference is `txnujk-bot/bots/floorfix/bot.py`, upstream commit
`4fe154318949aa60430682416f148733c6f0d564`. The earlier six-player research candidate
`txnujk_32` changed only its postflop sampling cap to 32. That earlier score is
not evidence of performance in Auction Flop. The source-to-rule audit imports
the original only in an offline verifier; the packaged bot never imports it.

| Original rule | Fresh implementation | Boston change |
| --- | --- | --- |
| 169 canonical hand classes, 900-sample heads-up random-hand equity; non-pair seed 12345 and pair seed 67890 | `build_preflop.py` regenerates `bot/preflop_table.py` during the event; all 169 values match the original exactly | Store the generated immutable table so startup does not repeat simulations |
| Opening equity thresholds .585 strong, .52 late, .47 late suited/pair; late means at most two players after hero | `preflop`, `players_after` | Use official button and non-busted seat order; actual current big blind replaces original hard-coded 100 |
| Open 3 BB early or 2.5 BB late, plus 1 BB per limper | `preflop` | Clamp raise-to to SDK legal bounds; unavailable raise becomes legal call/check |
| QQ+ can commit deep; TT+, AQs and AK can commit when at most 25 BB; costly call is at least 30% of stack | `preflop` | Clamp actual call cost to remaining stack; public eligible-pot helper replaces full-pot odds |
| One raise versus multiple raises selects 3-bet, flat or fold; .33/.40 price thresholds and maniac +.06/nit -.02 adjustments | `preflop` | Current-hand history is bounded to 128 events; long pathological histories are a documented approximation |
| Opponent classes require eight observed actions; aggression .55, foldiness .55, station calls .55 and aggression at most .12 | `update_profiles` | Bounded four-counter rows are passed only in SDK memory; incremental public events replace the unavailable full match log. At most 512 events are processed per decision |
| Multiway showdown share versus random hands; ties split by number of tied hands | `equity`, fixed 32 samples | Remove both unselected auction cards, known board and private cards; honor public shown opponent cards. Use a decision-local deterministic RNG instead of cross-call mutable globals |
| Distinguish made-hand value category from stack commitment; top pair and board-assisted two-pair normally are not stack-off hands | `hand_features` | Numeric eval7 categories avoid label ambiguity; frozen shared-board correctness helpers cover public quads and provable shared nuts |
| Stack-worthy made hands bet 75% pot; top pair or better gets one 50% flop value bet; marginal pairs check; turn/river pot control | `postflop` | Same thresholds and street-specific branching, with legal raise-to output |
| Flop/turn draws may semi-bluff 55% pot when equity is at least .34 and at most two opponents; late flop air stabs half the time | `postflop` | Same decision rules; RNG stream is keyed to public decision identity rather than mutable table-global state |
| Draw continuation uses 9/8/15 outs and the 2/4 rule; large calls need draw equity at least price +.03, small draw calls risk at most 20% of stack | `postflop` | Subtract visibly dead auction-card outs. This retains the original coarse draw detector, including its approximate open-ended/wheel treatment |
| Equity discount .06 normally or .13 for big bets, zero versus maniacs; normal call margin .06/.02 | `postflop` | Price and bet-size classification use reachable normal-chip pot. Auction chips never enter the normal pot |
| Large calls at 28% of stack, 70% of pot; stack-off at 55% of stack | `postflop` | Same thresholds; call cost is clipped to available chips |
| No auction exists in classic Hold'em | `tournament_agent.py`, `tournament_strategy.py` | Exact frozen tournament auction from hybrid `7e7aa88`; 458 observed auction decisions matched byte-for-byte actions in the audit |

## Shared event dependencies

- Public observer: `7e7aa88:bot/public_features.py`.
- Auction: `7e7aa88:bot/tournament_agent.py` and `bot/tournament_strategy.py`.
- Eligible call pot: `93405ba:bot/pot_accounting.py`, SHA256
  `f1d788be7be367125d13f04cd92701b2d2c7307b307c75f84726eb15d84cfaee`.
- Shared-board correction: `3c797f5:fixes/hand_contribution.py`.

The old classic implementation could treat a shared royal flush as worthless
and public quads with a poor kicker as stack-off worthy. The new implementation
calls/checks provable shared nuts and uses the public-quads reference share.
The quads formula is exact heads-up against uniform hidden cards; exponentiating
it for multiple opponents is only a conservative heuristic, not an exact
multiway posterior.

## Continuity evidence and limits

`verify_classic.py` exercises the released engine, then compares both policies
on its observed ordinary states. It forces equal equity inputs and equal mixed
action streams to isolate decision-rule continuity from sampling differences.
It excludes deliberately corrected side-pot/partial-all-in/shared-board cases
and removes auction dead cards for the classic-game comparison. It does not
claim identical tournament trajectories or original random streams.

- 169/169 regenerated preflop class values equal the reference.
- 32,656 decision-rule comparisons across four opponent classes and four
  postflop equity levels: zero differences.
- 1,206 made-hand/draw classifications equal the reference.
- Eight eight-seat native tables, 5,037 decisions: zero illegal actions/failures.
- Maximum observed memory: 1,042 bytes.

These are correctness and continuity checks, not strength results. Strength is
delegated to the designated fixed `auction-flop-v1` benchmark. SDK qualification
has a separate report. CPU measurements include the full `Agent.act` but not
IPC; the observed preflop maximum was 32.99 ms and is retained as a limitation.
No strict every-call 1 ms bound is claimed. Original policy weaknesses not
explicitly repaired here remain, including coarse draw classification and
unconditional commitment with some strong-looking made hands.

## Reproduce

From the poker-bot root, use the official `.runtime/envs/boston-event312/bin/python`
to execute this worktree's `build_preflop.py` and `verify_classic.py`. The former
regenerates the static table; the latter writes `evidence/continuity.json`.
Only `bot/` enters the upload ZIP; the reference source and verifier do not.
