# Simple policy parameter tuning

Keep the simple v1 policy structure and all fixed benchmark files unchanged.
Parameterize existing numeric thresholds and one preflop chart shift. No opponent
models, neural networks or new strategic subsystem. All sources and raw games
are retained separately from v1. User authorized tuning on 2026-11-10.

There is no existing duration parameter. Test two meanings: a nominal auction
budget horizon, and the number of postflop Monte Carlo samples per decision.
Never shorten the 250-hand cap or alter blinds/payouts to obtain a faster score.

## Search declared before results

Use four frozen v1 panels, 16 independent blocks each, eight seat rotations:
512 tables per arm. All arms receive the same new seed prefix
`simple-tuning-v2-screen-20261110`, disjoint from permanent v1 seeds. Screen
intervals are descriptive and not multiplicity-adjusted. Only zero-failure arms
are eligible. Baseline parameterization must reproduce the old actions exactly.

| Arm | Change from v1 |
| --- | --- |
| control | Original frozen simple_classic_v1 |
| bid_zero | BASE_BID 0 instead of 20 |
| bid_60 | BASE_BID 60 |
| bid_120 | BASE_BID 120 |
| duration_10 | Replace constant base by remaining auction balance / 10 |
| duration_25 | Replace constant base by remaining auction balance / 25 |
| loose_open | Early uses middle chart, middle uses late chart; late unchanged |
| call_wider | Multiply both preflop call-price caps by 1.7 |
| shove_later | Short-stack opening shove threshold 8 BB instead of 12 |
| bet_larger | Postflop pot-based bet/raise sizes multiplied by 1.3 |
| no_air_bluff | Flop air-stab frequency 0 instead of 0.5 |
| samples_16 | 16 instead of 32 postflop equity deals |

The duration parameter is a spending scale, not a guarantee of surviving or
exhausting chips in that many auctions; actual spending still depends on hand
improvement, live player count and which cards win. BASE_BID is unused when
duration mode is enabled.

One optional combination round: combine the best bid/duration setting and each
other parameter change whose screen mean exceeds control by at least 0.5
points. Evaluate that combination on the same screen schedule. This threshold
is a search heuristic, not statistical evidence. Select the highest screen mean
among control, individual arms and this combination; exact ties favor control,
then lexicographic name. Do not change parameters after seeing confirmation.

Confirm the selected candidate on the complete unchanged 4,096-table v1
benchmark, compare against retained simple v1 with paired block intervals.
If control wins selection, retain it and its already audited full result.
Report all screening arms, selection uncertainty and total wall time. Record
separate decision timing only after parallel strength work finishes; it is not
production IPC or a hard 1 ms guarantee. Use 32 currently quiet distinct
physical cores, with the same core set for all arms.
