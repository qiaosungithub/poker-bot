# Practice 4 public-action preflop prior

The top teams' public betting actions support useful contextual priors even
without their private cards. A static projection of those priors improved the
corrected loose baseline on the unchanged `auction-flop-v1` benchmark.

## Result

| Candidate | Mean points/table | Paired gain over corrected loose | 95% paired interval |
| --- | ---: | ---: | --- |
| `corrected_preflop_loose_v2` | 16.62598 | — | — |
| `practice_prior_half_v1` | 17.24365 | +0.61768 | +0.08524 to +1.15011 |
| `practice_prior_full_v1` | 17.65625 | +1.03027 | +0.38054 to +1.68001 |

Each candidate has 4,096 tables and 512 independent seed blocks, with the same
four frozen panels, eight seat rotations, rules, opponents and seeds. Points
are tournament payouts (50/30/20, or ICM allocation at the hand cap), not win
percentage or chips. Each table distributes 100 points; symmetric eight-seat
performance is 12.5. These scores are not comparable to live Practice scores
against different opponents.

Both blend strengths were declared before full evaluation. The full version has
the larger observed mean; its advantage over half is uncertain (+0.41260,
95% interval -0.17642 to +1.00162). Both candidate-versus-control gains remain
positive under the [two-comparison Bonferroni intervals](evidence/two-arm-comparison.json).
This is evidence within v1's fixed field, not proof of improvement against the
live tournament population. No additional parameter search or fresh-seed strength
holdout was run. No live submission was changed.

Raw tables, source snapshots, manifests and audited scores are retained in
[half](../../evidence/candidate-practice_prior_half_v1/) and
[full](../../evidence/candidate-practice_prior_full_v1/). Pairwise comparisons are
[half/control](evidence/half-vs-control.json),
[full/control](evidence/full-vs-control.json) and
[full/half](evidence/full-vs-half.json).

## Data and teacher inspection

Source: official Practice 4, tournament
`dc16f7e0-94a4-4edd-8cc6-e932965dbcbc`, downloaded 2026-11-10 through the
team-authorized hands API. The 154,770,890-byte gzip has SHA256
`c3387f9f05235d531cef9c6504edc84efd53e22ff777393e2c9aa8db9696591a`.
The private local archive and download metadata are under the original checkout's
`.runtime/private-web/preflop-prior-20261110/`. Credentials and private hole-card
records are not included here or in either candidate package.

The archive has 10,254 tables. Excluding our 254 filler tables leaves 10,000
scored tables and 1,436,464 hands. We extract the first voluntary preflop decision
per seat per hand: 6,944,093 field decisions, including 348,911 from the top ten
teams. Opponent filler decisions and our marked automatic decision are excluded.
Opponent automatic-action flags are not available. All sampled opponent tables
are tables shared with our entry; they are not those opponents' complete histories.

Teachers are the ten highest nonforfeited teams by the official round results.
The following opening rates are raw training frequencies for 7–10 players,
over 60 effective BB, no earlier call or raise. Small-blind cells have only
95–125 observations per teacher, motivating smoothing.

| Teacher | Live round points | Early-position raise | Button raise | Small-blind raise |
| --- | ---: | ---: | ---: | ---: |
| Du-o | 26.275 | 2.6% | 6.4% | 35.4% |
| BHTG | 23.747 | 5.8% | 59.1% | 95.7% |
| orz | 22.005 | 18.5% | 83.3% | 100.0% |
| artisans | 21.874 | 5.8% | 16.6% | 66.3% |
| Nyx | 21.770 | 10.0% | 39.1% | 96.8% |
| Group Chat | 20.753 | 10.3% | 57.7% | 61.6% |
| Preesh | 20.712 | 15.6% | 68.0% | 94.0% |
| Team Rob Watts | 20.460 | 16.7% | 46.6% | 53.8% |
| big little sharks | 20.342 | 11.5% | 60.3% | 68.6% |
| Golem All in | 20.011 | 7.9% | 51.8% | 44.0% |

The leader is particularly tight: high ranking does not imply uniformly wide
preflop ranges. Auction and postflop behavior may explain other parts of its
success; this experiment does not identify those mechanisms.

## Estimation and implementation

`extract.py` reconstructs state only from stacks before the hand, blinds, button
and earlier public actions. Raises are street totals; calls are increments.
Folded players cannot supply effective covering stacks. Contexts use position,
players at the start of the hand, effective BB bucket, and prior raise/limp state.
Future board cards, showdown cards, private cards and final stacks are never
inputs. The same partition contains every table and hand from an iteration:
`iteration % 5 == 4` is held out (2,000 tables); the remaining 8,000 train priors.

`fit.py` estimates fold/check/call/raise frequencies. Each teacher has equal
weight, so surviving longer does not automatically give it more weight in a
cell. Each teacher-cell shrinks toward the pooled same-position/player-count/
action-context parent with 50 pseudo-observations; parents have smoothing mass
100. The full-round ranking selects teachers, so the ranking itself is not held
out. This split measures action prediction conditional on that selected cohort.

On 33,826 held-out deep, unopened decisions, cross entropy is 0.57081 nats versus
0.64839 for the field's contextual prior and 0.77853 for a global teacher prior.
This measures imitation, not expected poker profit. [Counts](evidence/practice4-counts.json)
and [estimated priors](evidence/practice4-priors.json) preserve all denominators.

The runnable policy only uses the raise probability in 36 supported contexts
with at least 300 training decisions. It changes the first, unopened decision
above 10 effective BB. Original short-stack shoves, responses to limps/raises,
call rules, raise sizing, auction policy and all later streets stay byte-identical
apart from the three-line preflop hook. `build.py` enumerates all 1,326 starting
card combinations using the baseline's existing quality function. It maps the
observed raise rate to a quality quantile; half blends observed and baseline
opening probabilities 50/50, full uses the observed probability. Tied qualities
cause small quantization differences. This assumes a monotone opening range;
it does not recover which hidden cards a teacher actually played. No neural
network, runtime learning or cross-table memory is used.

For the deep full-table example, the smoothed cohort raises 10.5% early, 16.2%
middle, 32.4% cutoff, 50.3% button and 72.9% small blind. The main policy change
is a wider small-blind opening range (baseline 21.0% of all starting combinations),
with more selective early openings. Uniform-card range percentages are a
projection, not an assertion about actual conditional card distributions.

## Validation and artifacts

- Four tests cover future-card invariance, an intentionally changed public price,
  raise/call accounting, first-decision filtering, folded cover, heads-up blinds
  and exclusion of held-out labels from fitting.
- Official-engine replay: 8 tables per arm, 9,264 and 9,762 legal decisions;
  runtime context parity and unchanged guards verified.
- 64-table interface smoke per arm; zero failures. These are not ranking scores.
- Official SDK/IPC play: 8 eight-seat tables, 1 ms decisions plus the SDK's
  100 ms/table time bank; zero failed decisions for all seats. This is a local
  integration check, not remote production qualification.
- Full strength: 631,161 half-arm and 639,605 full-arm decisions; zero failures.
- Prior lookup p99 approximately 8.4 microseconds locally. This excludes SDK/IPC
  and the rest of the bot, on a shared host.
- Packed ZIP contents hash-match candidate source; benchmark comparison audits
  raw payouts, schedules and source snapshots. Baseline and v1 bytes are unchanged.

See [catalog](../../candidate_catalog_practice_prior_v1.json),
[full candidate](../../candidates/practice_prior_full_v1/),
[full package](evidence/practice-prior-full-v1.zip) and
[half package](evidence/practice-prior-half-v1.zip). These candidates start from
repository revision `64752ff`'s corrected loose snapshot. Integrating the prior
into later postflop fixes requires a new candidate and evaluation.

## Reproduce

Use the repository's pinned Python 3.12 dependencies, from the repository root.
Private history extraction requires the authorized archive; all later steps
and candidate evaluation work with the committed aggregates, without credentials.

```bash
python -m unittest auction_flop.research.practice_preflop_prior.test_extract -v
python -m auction_flop.research.practice_preflop_prior.fit \
  --counts auction_flop/research/practice_preflop_prior/evidence/practice4-counts.json \
  --out .runtime/reproduced-priors.json
python -m auction_flop.research.practice_preflop_prior.build \
  --model .runtime/reproduced-priors.json --alpha 1 \
  --target .runtime/reproduced-prior-bot
python -m auction_flop.research.practice_preflop_prior.validate \
  --candidate .runtime/reproduced-prior-bot --out .runtime/prior-validation.json
python -m auction_flop.benchmark.run evaluate \
  --candidate .runtime/reproduced-prior-bot --workers 16 --out .runtime/new-prior-run
python -m auction_flop.benchmark.run compare .runtime/new-prior-run \
  auction_flop/evidence/candidate-corrected_preflop_loose_v2
```
