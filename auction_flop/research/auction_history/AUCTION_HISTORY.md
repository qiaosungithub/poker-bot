# Auction-history module v1

Pure entrypoint `auction_policy.choose_bid(state, features)` follows the shared
modules/CONTRACT.md. No betting policy or persistent memory is implemented here.
The shared observer supplies public, per-table features. Auction balances never
enter normal-chip stacks, pots, or ICM calculations.

Package these files: auction_policy.py, auction_value.py, auction_selector.py.
For the OFF ablation also package auction_no_history.py and make the adapter
import `choose_bid` from that module. ON and OFF keep the same value function,
optimizer, opportunity cost and legal bounds; only historical pressure inference
is switched. `choose_bid_no_history` is also directly callable for a harness.

## Model

For each still-in opponent, estimate spending pressure from current public
balance, remaining useful hands and the opponent's stack/all-in status. Blend
historical paid/balance fractions with eight prior observations. Blend the
frequency of positive observed payments likewise. No current-auction payment
or private bid is read. Sparse observations stay close to the same public prior.

Payment is censored by selection: bids on dead cards are invisible. The 1.35
payment-to-total-pressure multiplier is an uncertain modeling assumption, not
an unbiased estimate. A nonzero pressure floor and high-pressure case remain
even after repeated zero payments. Positive payment identifies neither exact
card allocation nor a strong made hand.

Build four scenarios: quiet, low, typical, high. Each opponent's scenario vector
is individually capped by their public balance before aggregation, so scenarios
are physically feasible. Rotated card profiles express unobserved preference,
not inferred hidden cards. Busted/folded seats and zero-budget opponents cannot
supply current votes. The quiet probability is bounded to avoid overconfidence.

Use all ten flop combination utilities from fresh event-time variant source.
This is a structural strength/draw proxy, NOT a calibrated equity estimator.
The pivotal selector compares outcomes against zero bids and charges our own
bid only on selected cards, with a future-currency cost and downside penalty.
It can free ride, select cheaper partial improvement, or choose zero. All-in
accessible-pot scaling uses public preflop contributions to exclude upper side
pots; unknown synthetic histories fall back to the displayed pot as a coarse
importance scale. Actual normal-chip balances and payoffs are never changed.

## Provenance and tests

SOURCE_PROVENANCE.json pins the fresh variant source and local changes. The
standard-index tie helpers have equivalent fast paths tested against retained
generic implementations on 1,000 random profiles. No pre-event bot was copied.

Eight tests pass: 400 randomized legal bids and feasible scenarios, exact OFF
ablation, pressure response to history, censored-zero floor, scarce/zero budgets,
stateless interleaving, precise no-opposition tie behavior, and all-in side-pot
eligibility. Synthetic timing excludes shared observation, SDK, IPC and microVM;
root may have concurrent strength jobs, so it is not runtime qualification.

Next experiment: fixed preflop/postflop policies and identical shared observer,
ON versus OFF auction history, paired deal blocks/seat rotations, then independent
confirmation. Compare against the stronger tournament-auction baseline too.
No tournament-strength improvement is claimed for this module yet.
