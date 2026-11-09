# Push/fold v2

This is an independent candidate based on the frozen `push_fold` agent. The original candidate remains unchanged.

The call-price denominator subtracts every current-street contribution above the hero's maximum total contribution. The kit computes `pot` as all chips committed and `to_call` as the uncapped difference to the largest street bet, so an all-in caller must remove unreachable side-pot chips. With stack 980, own bet 20, opponent bet 10,000 and pot 10,020, the price is now 980 / 2,000 = 49%. Multiple opponents' excess contributions are each removed. Prior-street pot remains available because a player still able to act matched those streets.

Stack-to-pot ratio uses the same contestable pot. On boards with four or more cards of one suit, flush strength discounts missing higher suit ranks so a low one-card flush receives less credit than the best available private flush.

Official submission check passed. Targeted regressions covered the 49% call, two oversized opponents, an ordinary matched pot, and low versus high flushes. Another 6,000 randomized states produced legal actions and valid auction budgets. Direct-call P99 was 0.153334 ms, maximum 0.809417 ms and no call exceeded 1 ms in that run, excluding transport and with other work running on the laptop.

Official SDK smoke completed four three-player tables capped at 100 hands, seed `push-fold-v2-smoke`. This candidate scored 49.158 average points, with zero failed decisions. The two supplied opponents also had zero failures. The small smoke run establishes execution, while the shared comparison supplies competitive evidence.
