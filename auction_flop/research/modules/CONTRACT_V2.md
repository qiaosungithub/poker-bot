# Three-stage policy contract v2

This extends CONTRACT.md without changing choose_bid(state, features) or
choose_bet(state, features). The v1 observer/baseline remains frozen separately.

## Dispatch

```python
features, observed_memory = observe(state, memory)
if state.phase == 'auction':
    action = Action('bid', cards=choose_bid(state, features))
elif not state.board:
    action = choose_preflop(state, features)
else:
    action = choose_postflop(state, features)
# Validate/clamp the final legal returned action, then append our own decision.
updated_memory = record_own_decision(state, action, observed_memory)
return action, updated_memory
```

- `preflop_policy.py` exports choose_preflop(state, features) -> Action.
- `auction_policy.py` keeps choose_bid(state, features) -> five integer amounts.
- `postflop_policy.py` exports choose_postflop(state, features) -> Action.
- Existing betting modules remain usable through wrappers that call choose_bet.
- Forced blind payments have no separate decision stage.

## Added features

All are per-hand, own-player information, reset on a new hand. Existing fields
retain their exact v1 meanings. Public opponent actions remain in state.history.

- `own_preflop_actions`: tuple of OwnDecision records, bounded to the most recent
  16 entries. Each has kind, amount (official raise-to amount, otherwise zero),
  added_chips (call/raise chips beyond the existing street bet), and history_length.
- `own_preflop_added`: total normal chips added by our recorded discretionary
  preflop calls/raises in this hand; excludes forced blinds and antes.
- `own_bid`: own last submitted five-item vector, or None when unavailable.
- `own_bid_reconciled`: None before settlement; True if the selected-card cost
  implied by our stored vector equals official flop_chips_paid; False if it
  disagrees (then own_bid is hidden from downstream policies as unknown).

The trace is appended only after final legal action validation, and exactly once
per decision key. A timeout/exception discards the returned memory in the official
runtime, so the previous trace remains authoritative. Own actions are checked
against subsequently delivered public history; unexpected fallback execution
must not masquerade as an accepted intended action. Auction reconciliation uses
only our own known vector, selected public cards and official own paid total.
It never reconstructs an opponent's hidden bids. Matching paid totals are a
consistency check, not proof of a uniquely identified vector.

Empty/truncated memory loses the own trace; unknown stays None. No trace is
reconstructed from an assumed 1000 starting balance. All memory stays explicit
per table; no self/global/cross-table state.
