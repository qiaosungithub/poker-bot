# Modular auction / betting contract v1

Published 2026-11-09 during the Boston event. Policies are independent, stateless
functions. A single shared observer updates public per-table memory once per
`Agent.act` and supplies the same immutable feature object to either policy.
All persistent state is returned as bytes; policies must not mutate features.

```python
# auction_policy.py
# Return FIVE nonnegative integer bid amounts, sum <= current legal bid budget.
def choose_bid(state, features): ...

# betting_policy.py
# Return an official poker_agent_protocol.Action from current legal actions.
def choose_bet(state, features): ...

# public_features.py
# Updated memory is returned even when a policy deliberately uses no adaptation.
def observe(state, memory: bytes) -> tuple[Features, bytes]: ...
```

`state` is the official native GameState; no converted private simulator state.
Both policies can read cards, pot, ordinary stacks, legal actions and config.
Auction currency never enters normal-chip pot odds or ICM stack vectors.

## Features (immutable attributes)

All seat-indexed tuples have `state.config.players` entries.

- `balances: tuple[int, ...]`: currently public remaining flop-chips.
- `paid_this_hand: tuple[int | None, ...]`: actual selected-card payments this
  hand, only when a before-auction balance snapshot is available. Unknown is
  None, not zero. During an uncompleted auction every entry is None.
- `balance_before_hand: tuple[int | None, ...]`: known pre-auction balances.
- `auction_cards: tuple[str, ...]`: five public candidates, or empty.
- `selected_cards: tuple[str, ...]`: selected flop (board[:3]), or empty.
- `dead_cards: tuple[str, ...]`: candidates not selected, after settlement only.
- `auction_settled: bool`: actual selected flop exists.
- `history: tuple[OpponentStats, ...]`: seat histories from public observations.
- `hand_number: int`; `memory_reset: bool`.

OpponentStats attributes:
- `auctions: int`: observed completed-auction participation count.
- `paid_total: int`: total paid since trustworthy observation began.
- `paid_mean: float`: paid_total / auctions (zero if no observations).
- `positive_auctions: int`: individually observed auctions with positive paid.
- `positive_observations: int`: auctions whose per-auction paid amount is known.
- `positive_rate: float`: shrunk (positive_auctions + 1)/(positive_observations + 2).
- `fraction_mean: float`: mean paid/pre-auction-balance over individually known
  auctions with positive available budget; zero if none.
- `fraction_observations: int`: support for fraction_mean.
- `last_paid: int | None`: last individually identified paid amount.
- `hands_seen: int`, `vpip_hands: int`, `preflop_raises: int`: optional conservative
  public betting summaries; these fields exist even if no observation is available.

`auctions` includes observed participations, but per-auction positive/fraction
statistics update only when the public balance change is attributable to one
auction. If several auctions pass between balance snapshots, total payment is
known while individual payments are censored. Never manufacture their split.

A zero payment does not reveal zero bids: bids on losing cards are unobserved.
A positive payment reveals only the sum on selected cards, not per-card bids,
card scores, or hidden hole cards. At auction, balances contain no current bids.

## Integration and testing

Common `agent.py` calls observe once, dispatches on phase, and validates/clamps
its own returned action against native descriptors. No silent policy replacement
in the benchmark: failures must be counted. The observer defensively resets on
empty/truncated/wrong-table/wrong-version memory. Replaying the same state from
the same accepted memory is deterministic. No module globals or self hold state.

New policy managers may import `Features` only for typing, or use duck typing.
Their submitted module should export `choose_bid` or `choose_bet`; configuration
can be immutable module constants or an optional factory outside this v1 API.
Root composes source-only artifacts with explicit module hashes. Initial fallback
adapters use the frozen variant-v1 Agent without changing its strategy source.
