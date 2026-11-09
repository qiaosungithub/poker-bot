# Joint three-stage rule candidate

Hypothesis: small cross-stage coordination terms may improve the independently
selected variant-betting/tournament-auction composition. This is an experiment,
not a claim that joint design is better. The coupling-off control retains exact
frozen component behavior with identical available observations.

The three entry points are choose_preflop, choose_bid, choose_postflop. Every
function accepts the official state and shared immutable features, has no private
persistent state, and returns a native Action or five integer bid amounts.

## Bounded coordination terms

- Preflop: a small relative-auction-budget realization adjustment only for cheap,
  sufficiently deep speculative entries (small/medium pairs or suited connected
  hands). Premium hands and large commitments stay unchanged. Equal budgets
  produce no adjustment. This is a hypothesis rather than calibrated equity.
- Auction: use our recorded discretionary preflop participation to identify a
  speculative option. Spend somewhat more only when actual candidates offer a
  meaningful good outcome, and somewhat less when the option has failed. Never
  treat normal chips or auction spend as interchangeable.
- Postflop: use previous initiative and actual board support to choose bounded
  continuation aggression. A missed preferred auction outcome plus public
  competing spending can reduce marginal aggression, not promote a hand.
  Own bids supply a record of earlier decisions, not new equity evidence.

The first candidate leaves existing made-hand and legal-action machinery intact.
No model training, solver, hidden cards, raw opponent bids or cross-table state.
All auction probabilities/strength values remain explicit heuristic proxies.

## Comparison and evidence

Freeze ON and OFF code. Root's common benchmark compares them using matched
seeds/seat rotations and later fresh confirmation. Tests must establish exact
OFF baseline parity, neutral ON parity, legal budgets/actions, observation-loss
fallbacks, currency separation, realized-board sensitivity and deterministic
interleaved-table behavior. Final timing includes shared observer and SDK/IPC;
pure function timings are only preliminary.

Source components are event-time commits variant-v1 6af47e4 and tournament-v1
25ef579, vendored with unique names and recorded hashes. No old research bot
source is included. New policy source and every meaningful test remain here.
