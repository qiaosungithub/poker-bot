# ICM survival candidate

Created during the Boston event on November 9, 2026. Uses the newly written
`value_heuristic` candidate's card-strength helpers and original payout-aware
betting and auction rules.

- Calls gain a risk premium near the payout bubble, especially for a covered
  medium stack with a much shorter survivor.
- Large heads-up calls compare fold, win and lose stack distributions through
  an exact top-three own-player ICM calculation. This matches the default
  three-paid-place structure. Additional paid places use an approximation.
- Preflop opening ranges depend on surviving players, short-stack pressure,
  position and chip coverage. A covering stack can apply more pressure late.
- Auction spending scales with improvement over the free default flop and
  pot commitment. All-in and short-stack opportunities receive more budget.
- Card quality is a fast heuristic. The ICM call calculation treats a large
  call as terminal and omits future betting and opponent range inference.

The agent carries no table state and uses only standard-library, SDK and eval7
imports. Syntax validation passed. The own-player top-three ICM function matched
the official SDK calculator on 360 randomized stack vectors across 2–10 players,
checking every seat, with maximum absolute error 1.43e-14 points. Dependency
installation was still in progress at implementation time, so official matches
and latency measurements are pending the parent agent's shared evaluation.
