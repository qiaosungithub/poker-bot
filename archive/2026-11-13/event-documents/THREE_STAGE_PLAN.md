# Three-stage independent and joint strategy experiments

Latest user steering: separate preflop betting, auction, and postflop betting;
also test a jointly designed rule policy. The user's phrase 'blind stage' is
interpreted as preflop decisions; mandatory blind posting has no bot choice.
Read shared contract in branches/modules/CONTRACT_V2.md.

## Information flow

1. Preflop: own holes, public betting, position, ordinary stacks, blind level,
   remaining auction budgets, public per-table history.
2. Auction: all preceding public observations and own preflop decisions, five
   candidates, historical aggregate auction spending, current public budgets.
3. Postflop: preceding own/public actions, OUR full private five-element bid,
   chosen flop and dead candidates, observed per-seat selected-card payments,
   public history, current turn/river and ordinary legal betting state.

Never expose opponents' raw bids or hidden holes. Auction currency is separate
from normal chips and never enters pot odds/ICM chip vectors. Own historical
spending is a sunk cost and cannot justify chasing a losing poker hand. An own
bid is not independent evidence of strength beyond information already used.

The observer updates once per call; an adapter records the final legal returned
action, resets context per hand, and handles timeout/old-memory and missing
observations conservatively. Own bids can be checked against observed own cost
on selected cards; mismatches invalidate knowledge rather than fabricating it.

## Experiments

- Preserve all five original complete bots and the fixed three-part composition
  with variant preflop, tournament auction, and variant postflop.
- Independently vary one stage at a time, then cross the promising modules.
- Add a new joint-policy branch with explicit cross-stage coordination terms
  and a coupling-off control. Both receive the same public/own information.
- Compare historical-feature off/on separately from coordination off/on.
- Freeze sources before strength runs. New confirmation seeds are disjoint
  from screens. Keep ordinary payout, illegal/error rates, auction expenditure,
  budget left at elimination, and each stage's complete SDK timing.

## Existing evidence

Fresh hybrid-confirm01 used512 blocks/4096 tables. Within the same eight-seat
field, tournament-auction + variant-betting scored21.73096 versus18.93799 for
variant-v1; paired improvement2.79297, simultaneous95%CI[1.24343,4.34251].
All scores independently audited and zero decision failures without clock
penalties. This supports that particular composition in that field; it does
not establish a globally optimal stage combination or production timing.

All five original v1 candidates passed the remote qualification script with
zero failed decisions and exact source hashes. It used official SDK ordinary
processes on the supplied build box, 1ms decisions plus2ms delivery grace,
six short zero-bank heads-up tables and three eight-seat tables per candidate.
It is not the production microVM. Sparse phase samples and occasional >1ms
transport observations remain in the raw reports; no strict wall-time guarantee.
