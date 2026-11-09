# Heads-up big-blind defense v6

Based on texture-price v5. Only heads-up preflop big-blind defense changes.
The new call requires an active opponent, a cost no larger than two big blinds
and 10% of the remaining stack, pot odds no higher than 34%, and a hand-quality
guard. Existing raises and strong-hand decisions take priority.

Actual Boston-engine seed 0 deals Jd8c to the big blind. V5 folds to a raise to
44 with 10/20 blinds and 2000 stacks. V6 calls, while continuing to fold against
raises to 1000 or an all-in to 2000. Non-heads-up states, auctions and postflop
decisions are unchanged in the small official-state equivalence check.

Rebuild with `tools/build_modular.py`, selecting `modular_v4/common_texture.py`,
`modular_v4/preflop_hu_v6.py`, `modular_v4/auction_texture.py` and
`modular_v4/postflop.py`. SDK static validation passed. Strength comparison is pending.
