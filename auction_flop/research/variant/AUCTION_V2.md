# Auction v2: pivotal spending and free riding

Candidate only. `agent.py`, betting behavior, `auction.py` board utilities and
frozen `variant-v1` tag are unchanged. No v2 tournament-strength claim and no
wall-time qualification were made while the root's parallel screen was active.

## Replaceable module contract

`auction_v2.choose_with_scenarios(utilities, budget, scenarios, shadow_price,
tie_order=(0,1,2,3,4), downside_weight=.15) -> five integer bids`.

- `utilities`: ten continuation values in `auction.TRIPLES` order. The shared
  valuation module may supply relative equity; raw hand rank is not equity.
- `budget`: actual legal remaining flop-chip balance, separate from normal chips.
- `scenarios`: small tuple of `(aggregate_opponent_bids[5], probability)`. Each
  aggregate uses nonnegative integers; probabilities should sum to one. The
  shared feature/model layer should supply physically feasible profiles from
  only public information. This selector never inspects hidden opponents.
- `shadow_price`: continuation-value units per consumed flop chip.
- Output sum is constrained by budget across **all five** bids, even though the
  expected-payment objective charges only selected cards.

`choose_bids(...)` retains a v1-compatible convenience wrapper with its heuristic
three-profile public-budget prior. That wrapper is not the new shared-history
model; its pessimistic high-pressure profile may exceed physically feasible
spending for very short modeled horizons. Prefer the explicit scenario API for
the modular system, where total-opponent-budget feasibility can be enforced.

## Mechanism

For each scenario, compute the zero-bid flop and search all ten target flops.
Construct the componentwise minimum forcing vector for that scenario. Rank
positive net improvements, retaining at most two proposals per scenario. This
can select an affordable partial improvement instead of the two best absolute
boards selected by v1.

Evaluate each retained vector on every scenario by:

`expected utility gain versus zero - shadow_price * expected paid chips
 - .15 * expected negative utility change`.

Then reduce the winning vector while preserving the full selected flop in
every supplied scenario. A selected card only needs enough own votes to beat
the highest excluded total after opponents' votes are counted. Repeat at most
three times to propagate reductions. This cleanup preserves modeled utility
and weakly lowers payment, but can change outcomes against bids outside the
modeled support; it is not a dominance theorem for arbitrary opponents.

The zero vector remains a candidate, so no model-positive influence means no
purchase. Cases where the board changes but value does not improve also do not
justify spending. Compared with v1's zero-opposition special case, the new
selector also applies the supplied currency shadow price when opponents cannot
bid; scarce remaining currency is not automatically spent for any tiny gain.

## Public-information boundary

Visible per-seat balance decreases reveal total paid only, not the allocation
across winning cards. A zero decrease is censored: the player may have bid on
losing candidates. It does not prove abstention or no interest. The selector
accepts a prior zero-bid scenario without claiming that public zero spending
establishes that scenario. Historical spending-pressure inference belongs in
the shared feature/scenario provider, not the betting or selector modules.

## Validation

Sixteen unit checks pass across the existing agent/kernel and new selector.
New tests cover free riding, a single pivotal complementary vote, cheap partial
improvement over an unaffordable best board, losing bids costing nothing, bad
cross-scenario effects, the full-vector budget constraint, and 250 randomized
cases where pruning must preserve every scenario's selected flop and never
increase modeled payment. Records are executable in `tests/test_auction_v2.py`.

Next comparison must hold the betting and valuation modules fixed, compare v1,
v2 and zero bidding on paired rotated seed blocks, then confirm on new seeds.
Report auction spend/remaining balance and decision failures with payout; do
not attribute a whole-policy improvement to bidding without this ablation.
