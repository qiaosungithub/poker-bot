# Tournament kernel snapshot

Fresh source written after the user-provided Boston auction-flop reveal on
2026-11-09. No source copied from pre-event bots. Official SDK adapter is pending
starter-kit availability, so this is not yet a complete runnable submission.

Implemented in bot/strategy.py:

- Exact standard-high 5–7-card hand-rank kernel, direct integer categories.
- Private-card-aware made-hand and draw quality; this heuristic is not equity.
- Deterministic preflop hand quality and bounded payout-risk premium.
- Auction evaluates all ten three-card boards. It chooses a target triple and
  allocates budget using losses under all one-card replacements, preserving
  interactions such as straight-flush cards that work together.
- Auction spend responds to future auction opportunities, stack depth, current
  pot exposure and all-in risk. Last-hand budget is not capped below 100%.
- Unknown deck return rules, precise tie ordering and protocol field names are
  not guessed. Opponent count is accepted for future calibration but not used
  in the current board-quality proxy.

Validation: three unittest cases cover categories/private contribution, budget
invariants across random deals, auction synergy, late spending and risk payouts.
An independent eval7 comparison matched ordering and ties across 15,000 random
5/6/7-card hands. One local 2,000-call auction-only timing run using the existing
Python 3.11 research interpreter measured P99 0.308 ms, max 0.571 ms, 0 over 1 ms.
That excludes SDK, IPC and official Python 3.12; it is only a preliminary check.

Next: inspect the actual kit, bind card/auction/legal interfaces, finish betting
policy, run official validation, then implement the shared benchmark harness.
Benchmark standard-cycle target increases to >=128 independent blocks / 1,024
rotated tables if measured throughput permits; 128-table screen is exploratory.
