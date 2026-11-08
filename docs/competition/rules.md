# Competition rules and discrepancies

Verified **2026-11-08**, before the Boston variant announcement. Sources:
[Getting started](https://pokerbot.sequoiacap.com/tournament/competitions/abb5539f-60ce-48b6-8925-c0c56a3102e3/start),
[Format](https://pokerbot.sequoiacap.com/tournament/competitions/abb5539f-60ce-48b6-8925-c0c56a3102e3/format),
[public rules](https://pokerbot.sequoiacap.com/rules.html), and the November 8
organizer email recorded in [source provenance](../sources.json).

## What is established

- Build phase: 60 minutes after the variant reveal. Editing and submitting
  continue during the scored phase.
- Six scored rounds start at build end and then +20, +40, +60, +80, +100 minutes.
  Twenty minutes is the interval between starts, not a round's execution limit.
- Each round snapshots the team's active bot at its start. Later accepted
  changes affect later rounds. No active bot means zero for that round.
- Round score is average points per table. Final score is
  `T1 + 2*T2 + 3*T3 + 4*T4 + 5*T5 + 6*T6`.
- Normal tables have eight bots. Initial stacks are 2,000 and initial blinds
  10/20. Opponents are other entrants' bots; upload validation uses an organizer
  check/fold bot instead.
- Seat identities are anonymized per table. Per-table history is usable;
  carrying information or learning across tables is prohibited.
- The variant was still unannounced in the recorded live competition response.
  Do not assume standard Hold'em, unchanged action kinds, or unchanged phases.

## Exact development timeline

Public rules section 4.2 explicitly permits editing and submitting after the
build phase; the signed-in Format page independently says to keep building
between scored rounds. The one-hour wording is the initial build period, not
a freeze for all six rounds.

| Time since reveal | Snapshot |
|---|---|
| 60 minutes | T1 |
| 80 minutes | T2 |
| 100 minutes | T3 |
| 120 minutes | T4 |
| 140 minutes | T5 |
| 160 minutes | T6 |

Under the announced schedule, improvements can affect scored play until the T6
snapshot, approximately **2 hours 40 minutes after reveal**. Allow time for
validation and activation before each deadline. The last round's computation
does not create a seventh opportunity to improve the score. Organizer delays
or an explicit competition freeze can change this operational window.

## Scoring

The Format page publishes **default** payouts `[50, 30, 20, 0, 0, 0, 0, 0]`.
Read each table's `state.config.payouts`: a round can override settings and may
draw different payouts for different tables. These are not the research
benchmark's `[100, 60, 40, 25, 10, 0]` payouts.

Eliminated bots receive their finishing-place payout. At the hand cap, survivors
receive their ICM expected share of the remaining top-place payouts:

1. First-place probability is own chips divided by all surviving chips.
2. Remove a hypothetical winner and repeat proportional allocation for the next
   place, continuing through the survivors.
3. Sum `P(place=k) * payout[k]` over possible places.

This is not simply payout proportional to chips, except in special cases such
as heads-up winner-take-all. See also [the short ICM note](../../scoring.md).

For simultaneous busts, Format ranks the larger start-of-hand stack higher.
The page does not fully specify equal-stack ties; inspect the released engine.
Equal final competition scores use weighted first-place frequencies, then
second-place frequencies and so on according to Format.

## Differences that must not be silently resolved

| Subject | Public rules | Signed-in portal | Operational interpretation |
|---|---|---|---|
| Hand cap | 200 | Format default 250 | Read `state.config.hand_cap`; reconcile at reveal |
| Blind increases | Every 25 hands | Format default every 20 hands | Read schedule and `hands_per_level` |
| Scored iterations | 10,000 | Format at least 5,000, one table per bot per iteration | Use actual round configuration/results |
| Submission cooldown | Five minutes | Getting started: 30 seconds; live competition response also 30 | Portal configuration observed, public text still differs |
| Package size | 100 MB | Portal: 50 MB uploaded, 200 MB unpacked | Prepare to the stricter portal limits |
| Activation | Most recent accepted bot active | Getting started says passing validation auto-activates; Format says upload alone does not | Inspect the team's actual active bot before each round |
| Tiebreak | Organizer-defined challenge | Format describes finishing-place countback | Preserve both descriptions until clarified |
| Fillers | Random copies of submissions | Format uses bots not already at that table; fewer than eight entrants use short tables | Do not assume duplicate-seat self-play |

The organizer can change defaults. `state.config` controls how a particular
table actually runs; it does not itself resolve eligibility or conduct rules.
The starter kit and event announcements are needed to reconcile remaining
documentary differences.

## Additional portal details

- Build-phase practice rounds: approximately 1,000 iterations every ten minutes,
  excluded from the final score.
- Round results normally arrive in the next twenty-minute window, including a
  compute interval after the final round.
- A competition-wide freeze rejects uploads and active-bot changes.
- Each round's downloadable hand histories are gzip JSONL, available in full or
  smaller subsets. They include own cards, public opponent cards, executed
  fallback actions and opponent names. Live seat identities remain anonymized.
- Five teams per regional advance; Format describes five wildcard teams across
  regionals. Public rules say up to five wildcards. Finals use another variant.

## November 8 organizer email

The Boston pre-event email requires all competing code to be written during the
event and prohibits pre-written bots/code. It allows all AI tools and says Codex
is optional. Existing Fullhouse source cannot simply be uploaded as the entry.
The email does not separately state whether previously trained model weights
may be used; do not assume either permission or a definitive separate ban.

The email asks entrants to register their team in the portal, claim credits and
prepare their remote computer before arrival. Help desk: November 9 at 4:30 PM
ET; check-in/dinner: 5–6 PM ET; kickoff/reveal: 6 PM ET. Location: David Rubenstein
Treehouse, Harvard University, 20 Western Ave, Boston, MA 02134. Bring a charged
laptop/power bank and government ID or a photo of it; charging is limited.
These are emailed arrangements, not a verified future schedule after this date.
