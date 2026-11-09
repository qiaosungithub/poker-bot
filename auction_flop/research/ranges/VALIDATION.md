# Ranges v1 validation

Fresh event implementation after live reveal. Native adapter follows released
Boston RULES.md, game.py, and flop_auction_holdem.py. Source is bot/agent.py plus
bot/range_core.py; package both. No neural nets or cross-table mutable state.

Three core test groups and native SDK dataclass tests passed. They cover bid
budget/integer invariants, zero budget, last-hand spend, duplicate-card rejection,
raise clamping/reopening, trips/quads, board-only shared strength, corrupted
memory, and native Action objects. The bot returns empty memory.

Official SDK static check passed. Official local three-seat smoke with seed
923110, 2 tables, against check_or_fold and min_raiser completed with zero failed
decisions for every seat. Ranges averaged 35 points in this tiny smoke; this is
not evidence of comparative strength. Eight-seat and remote validation remain
with the common benchmark manager.

Provisional Python 3.12 environment: poker-bay-area312 with released official
protocol/SDK/engine source on PYTHONPATH, no environment modifications. Exact
competition eval7 and pokerkit pins are available there. timing-v1.json is a
600-call per-phase synthetic full-act timing on the research server, excluding
SDK wire encoding, IPC, and engine. It does not certify the production deadline.

The auction uses the same eight blocker-compatible, action-tightened opponent
holdings to compare ten possible flops. Current made-hand percentile and draw
adjustments are heuristic scores, not calibrated showdown equity. It combines
best-board interactions with marginal card effects and a stack/horizon budget.
Betting uses position/range thresholds, conservative multiway adjustments, and
small selective semi-bluffs. All unselected auction cards are blocked from
opponent samples in later betting. No interpretation of unseen individual bids.

Further evidence needed: official eight-seat benchmark, paid-bid efficiency,
preflop participation and postflop fold frequencies, and production-like timing.
