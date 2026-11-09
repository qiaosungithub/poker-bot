# Equity sampling candidate

Original Boston implementation written during the November 9 event. The source
is standalone and imports only the SDK, standard library and the permitted
`eval7` evaluator.

Preflop uses a compact rank, pair, suitedness and connectivity score with
position-dependent opening thresholds. Postflop estimates showdown equity
with 32 bounded multiway samples. Samples exclude all five auction cards, so
the rejected two cards cannot reappear on the turn, river or in an opponent's
hand. Calls require equity above pot odds plus an aggression and bubble margin.
Raises use value thresholds and legal `raise_to` bounds.

The auction evaluates all ten possible flops using the same twelve sampled
opponent hands and runouts. Common samples reduce noise in differences between
flops. Bids target the best flop and are allocated by each card's marginal
contribution. Spending scales with the equity spread, pot size and remaining
lifetime budget. A preflop all-in permits more urgent spending. The auction
estimate uses one opponent to keep work bounded, so its ordering is an
approximation in multiway pots.

Random generators are local to each decision, seeded from the visible state.
There is no persistent mutable strategy state or memory dependency. This also
makes repeated evaluations of a fixed state deterministic.

## Verification

Python 3.12 syntax compilation passed. Runtime tests and latency measurements
were pending installation of `eval7` and the official SDK when this candidate
was handed to the coordinating agent. The fixed sample counts bound work but
do not establish compliance with the production deadline. No strength claim
has been measured yet.

The uniformly sampled opponent ranges can overestimate equity against tight
betting ranges. The explicit call margin mitigates this approximately. The
small sample budget introduces sampling noise, particularly around a close
calling threshold.
