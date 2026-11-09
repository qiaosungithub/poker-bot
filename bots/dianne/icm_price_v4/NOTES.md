# ICM price v4

Derived from frozen selected ICM v2, with the v3 nearly-free blind continuation
preserved. Adds cheap continuations after hero has raised preflop and the sole
opponent moves all-in. The contestable pot price must be at most 9%, and the
existing ICM threshold must fit a modest hand-class equity estimate of 18–26%.
This estimate is a heuristic rather than a solved opponent range.

Replays four actual cheap folds from the 128-table diagnostic. The hands J5o,
A5o, K8o and J9s had contestable prices of 4.17%, 6.06%, 8.47% and 3.07%.
All now call. Large prices, large ICM risk, and absent prior raises retain
selective behavior. The regression fixtures preserve their observed public
states, without using their opponents' hidden cards.

Three focused regression tests and SDK static validation passed. Tournament
strength comparison is pending.
