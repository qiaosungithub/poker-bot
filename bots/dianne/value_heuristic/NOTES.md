# Value and texture heuristic

Original Boston policy written after the November 9 reveal. It uses a fixed,
bounded amount of work per action and requires only the SDK and eval7.

Preflop selection uses pair rank, high cards, suitedness, connectivity, table
size and short-stack pressure. Postflop play combines exact made-hand class
with private-card contribution, board texture, draws, pot odds and multiway
discounts. It raises for value and checks weak hands when checking is legal.

The auction evaluates all ten possible flops and bids toward the best coherent
three-card set when it improves substantially over cards 0, 1 and 2. Spending
scales with remaining balance, remaining hands, pot size and made-hand value.
All-in hands spend more because the current board can determine survival.

The strength numbers are hand-built heuristics. They are not calibrated equity
estimates. The policy uses no table memory. Timing and arena results are recorded
after the environment is available.
