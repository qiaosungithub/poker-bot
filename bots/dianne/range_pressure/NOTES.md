# Range pressure candidate

Written during the Boston event on November 9, 2026. Standalone original code.

- Preflop selection uses position, stack depth, prior raises and deterministic mixed late-position aggression.
- Postflop uses 28 bounded samples. Opponent holdings are weighted toward stronger starting hands after preflop raises and toward made hands after current-street raises. Heads-up sampled equity is adjusted for the number of opponents.
- All five auction cards are excluded from future runouts, accounting for the two dead cards.
- Auction compares ten possible flops with ten common opponent/runout samples. Range-weighted gains over the free default flop guide budget use. Marginal inclusion values allocate chips among the three preferred cards.
- Lifetime budget, pot leverage and all-in urgency set spending. Sparse pressure bets target checked dry boards in late position. Calls carry an additional approximate bubble risk premium.
- Per-state seeded sampling makes decisions reproducible. Returned memory is empty.

## Validation

Official `poker-agent check` and Python compilation passed. Four eight-seat official-engine self-play tables capped at 35 hands produced 1,023 decisions with zero invalid actions or exceptions. Betting direct-call P99 was 0.100 ms. Eight auction calls had one 2.849 ms wall-time outlier during the initial smoke run.

A separate randomized auction legality/timing run checked 2,200 legal seven-card states and discarded the first 200 timings as warm-up. All bid amounts were nonnegative integers and total bids respected the budget. Among 2,000 measured auction calls, median was 0.114 ms, P99 0.125 ms and maximum 0.304 ms, with zero calls exceeding 1 ms.

These measurements cover `Agent.act` only and exclude SDK transport. Parent-run mixed-field and SDK validation determine selection. Source was frozen before the parent comparison.

## Limitations

Small-sample variance remains material. Range weights are heuristic. Multiway equity uses an independence approximation, and opponent auction bids are unobserved. The policy does not reconstruct betting ranges across earlier streets or learn opponent behavior between hands.
