# Boston variant: user-provided live reveal slides

Observed 2026-11-09T23:09Z. Sources are three JPEG photos in the workspace root:
微信图片_20261109180603_37_44.jpg, 微信图片_20261109180654_38_44.jpg,
微信图片_20261109180734_39_44.jpg. The portal still has null variant; the
photos establish that the live event has revealed the auction-flop mechanic.
Exact build deadline remains to be verified from official event state.

## Explicit rules

- Standard two private cards and preflop betting.
- Five community-card candidates dealt face up.
- Every player still in the hand, including all-ins, secretly bids flop-chips
  on each of the five cards. Zero bids allowed.
- The sum of all five bids cannot exceed the player's remaining flop-chip budget.
- Sum bids across players per card. Three cards with the highest totals form flop.
- Equal totals are resolved by "flop ordering" (precise index precedence unverified).
- Each player pays their own bids only on selected cards. Other bids cost nothing.
- Bids never revealed, but each player's remaining flop-chip balance is public.
- Turn and river follow standard NLHE. Best five of seven at showdown.
- Table ends at 250 hands or one player owning all normal chips.
- Unspent flop-chips have no terminal value and do not carry to next table.
- Placement payouts 50/30/20 for first/second/third; places 4–8 zero.
- Round score is average table score.

## Unresolved from these photos

1. Initial flop-chip budget, replenishment, cross-hand accounting details.
2. Whether ordinary flop betting occurs between auction and turn.
3. Whether two unselected candidates are permanently dead or returned to deck.
4. Hand-cap survivor scoring: old portal says ICM; slide only says placement.
5. Simultaneous settlement/visibility: later bidders must not assume they see
   earlier bids or updated balance unless actual public observations provide it.
6. Exact tie index, integer bid representation, legal descriptor and invalid-bid fallback.

## Strategy directions for all branch managers

The auction is an interaction over ten possible three-card boards, not five
independent card utilities. Evaluate combinations and marginal influence.
Flop-chip currency is separate from normal chips; spending does not directly
reduce pot odds/ICM stack. Its cost is future auction opportunity, falling with
remaining useful hands and elimination risk. Initial bidding should be bounded,
variant-specific, and independently designed per branch. Public balance deltas
can imply total paid, not each card's bid. No hidden opponent information.

Root continues 60-second authenticated portal/kit polling. Read official kit
rules/engine before fixing unresolved semantics. Fresh event-time implementation
may now start from revealed concepts; do not copy pre-event bot source.
