# Descriptive diagnostics and separate transfer validation

These tools are separate from the frozen `auction-flop-v1` runner, opponents,
rules and seeds. They neither modify the submitted policies nor send diagnostic
features back into a bot. All decision features use the hero's actual SDK view;
opponent private cards and future outcomes never enter candidate actions.

## Scoring audit

The released engine's `scoring.table_points` invokes `equity` on final stacks,
bust order, ties and payouts. Busted players receive placement payouts; at the
250-hand cap the remaining prizes are allocated among survivors using ICM.
`table._current_equity` also records settled-hand ICM, rounded to four decimals,
for history. Those intermediate changes are not extra reward. Luck correction
and resampling are disabled in v1. Only one of the 4,096 retained Hboth tables
ended at the cap with multiple survivors, so this is primarily a placement
optimization problem rather than a hand-cap allocation problem.

The deployed Hboth agent calls its tournament donor only during auctions.
That donor's betting risk premium is unreachable. Its actual preflop has a
fixed bubble adjustment, and postflop uses category gates and sampled showdown
equity; neither calls the prize ICM evaluator for action selection.

## Instrumented matched replay

`observe.py` wraps the released table with read-only hooks and invokes the
original `arena.run_table` transition loop. Candidate decisions, memories,
seating and RNG streams are unchanged. The 32-table smoke and the subsequent
256-table balanced replay both match every retained v1 table record, excluding
wall-clock time. Full action hashes also match an independent minimal action
capture with all feature computation disabled. Both have zero failures.

[Retained 256-table evidence](../evidence/hboth-diagnostic-256/manifest.json)
contains 40,155 hero decisions and 41,734 table-hands, including hands after
hero elimination. This subset is descriptive, not a new ranking score.

Logs include street, position, remaining players, effective BB, pre-action
liquid stack and stack-plus-current-street-bet ranks/shares, hand-start stack
state, call price and actual normal-chip cost. Mid-hand chips already in the
pot are not assigned to their eventual winner. Settled checkpoints are hands
20/40/80/120, first transitions to at most 4/3/2 survivors, and heads-up entry.
Every fixed-hand checkpoint includes absorbed final states for tables ending
earlier; alive, absorbed and total-table denominators are explicit. Intermediate
hand ICM uses the engine's four-decimal log; checkpoints use exact settled ICM.

## Highest-frequency investigation targets

| Decision population | Observed action | Interpretation |
| --- | --- | --- |
| 8,691 unopened late-position/SB/button opportunities with legal raises | 5,109 folds, 3,425 raises, 157 calls | Opening policy has broad exposure; folds are not individually proved wrong. SB alone folds 1,715/2,961. |
| 2,328 free options with sampled equity >=70%, made category >=1 and a legal raise | 746 checks: 69/1,040 on flop, 338/686 on turn, 339/602 on river | Investigate turn/river value realization against continuing ranges. Uniform 32-sample equity is not an action-value oracle. |
| 2,495 decisions facing a call >=25% of available stack plus current bet | 361 calls, 1,867 folds, 267 raises | Investigate expensive continuation modeling; calls span 323 hands and their mean settled-hand ICM change is -1.199. This is not causal regret. |

The third group includes 144 preflop, 40 flop, 83 turn and 94 river calls.
Expensive raises occur 2,050 times and have positive mean settled-hand change
in this sample; the data does not support a blanket reduction in aggression.
Cheap closed-all-in folds occur only six times and board-only flush-draw calls
three times. They deserve correctness review but are not the frequent macro
opportunities in these data.

Of 44 tables entering heads-up with at least half the chips, Hboth wins 32:
22/33 with 50–75% share and 10/11 with >=75%. This small local subset does not
show a systematic lead-conversion deficit relative to entry ICM. Older official
rounds do show a deficit, but they used different submissions and opponents;
see [the separately attributed official diagnosis](../research/official_checkpoints/README.md).

Every outcome sign is included in the summaries. Flags overlap, so their hand
changes cannot be summed into total loss or potential gain. The three targets
are hypotheses requiring intervention tests, not three established policy
failures. [Aggregates](../evidence/hboth-diagnostic-256/descriptive-summary.json),
[opportunity denominators](../evidence/hboth-diagnostic-256/opportunity-denominators.json),
[native examples](../evidence/hboth-diagnostic-256/native-examples.json).

## Withheld transfer checks

[Finalist 1 protocol](../evidence/macro-transfer-finalist1/protocol.json) was
frozen at 01:17:01 UTC, before any of its games. SHA256 is
`0363d25ed333b373aefe4d535d489642ef84bd494686e3ad3b41a770a9a5eaec`.
It pins exact candidate and control source bytes, rules, fresh seeds, four
equally weighted panels per field and 4,096 tables per candidate per field.
One field preserves v1's opponents with new seeds. The other equally exposes
exploit, equity, classic txnujk and TournamentSpecialist styles excluded from
v1 evaluation, without checkfold/minraise fillers. Different-field absolute
scores are not comparable. Style representativeness for the live event is not
guaranteed.

The only tested finalist is the already frozen `prior_hybrid_both_v1` versus
Hboth. Its fresh-v1-field gain is +1.61377, two-field-adjusted 95% interval
[+0.84115,+2.38639]; style-field gain is +0.20508, interval [-0.61194,+1.02210].
This supports a same-field fresh-seed gain, not a robust two-point gain across
styles. These results will not be used to tune the tested candidate. A possible
second finalist requires a separately frozen fresh seed set.

```bash
PYTHONHASHSEED=0 python -m auction_flop.diagnostics.run \
  --out .runtime/new-hboth-diagnostics --blocks-per-panel 8 --workers 16 --validate-actions
python -m auction_flop.diagnostics.analyze .runtime/new-hboth-diagnostics
python -m auction_flop.diagnostics.audit_transfer auction_flop/evidence/macro-transfer-finalist1
```
