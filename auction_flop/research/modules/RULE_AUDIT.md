# Released-rule audit

Official reveal: 2026-11-09T23:14:42.543801Z. Download observed 12 seconds later.
Kit SHA256: 4a461905d7f7ecf3f7f45e7aacf6460ba148bb77335bb271d7a86c864c09412d.
User confirms the event has started. Official tournament list was still empty
at 23:16Z; nominal first/final snapshots are 00:14:42Z / 01:54:42Z on Nov 10
if the published +60/+160-minute schedule applies. Reserve validation margin;
verify actual scheduled rounds instead of treating these estimates as confirmed.

Sources: released starter-kit/RULES.md, README.md, AGENTS.md, protocol game.py,
engine variants/flop_auction_holdem.py, table.py and holdem.py. SDK engine stays
unmodified. Local copy is ../.runtime/competition-20261109/starter-kit/.

| Slide uncertainty | Released semantics |
| --- | --- |
| Auction budget | 1000 per seat per table, carries across hands, never replenished, never enters normal pots or points |
| Losing candidates | Permanently dead for that hand; remove both from equity decks |
| Ties | Higher aggregate bid first; lower original candidate index wins ties; zero bids select 0,1,2 |
| Flop betting | Auction resolves then ordinary Hold'em street transitions/betting continue |
| Public balances | Updated only after every bid is submitted; individual amounts and aggregate card scores stay hidden |
| Hand-cap scoring | Remaining top payouts split by ICM; actual state.config.payouts controls each table |
| Auction action | Action('bid', cards=(five nonnegative integer amounts)); amount=0; total <= can('bid').max_amount |
| Failed auction | Five zeros; does not fold the player |
| All-in visibility | Auction completes before preflop all-in hands are shown; no opponent hole-card information while bidding |
| Table defaults | 8 seats, 2000 normal chips, blinds advance every20 hands, cap250; variant supports2–10 seats |

Public bid events identify only the bidder. Current auction balances must not
be used to infer earlier per-card bids. Differences between post-settlement
balances reveal each seat's total paid on the selected cards only.

The live competition weighting remains fibonacci. This has no effect on the
within-table strategy objective; it increases the importance of maintaining
validated improvements through later snapshots. Other side-prize events in
the engine do not change the official main table payout objective.
