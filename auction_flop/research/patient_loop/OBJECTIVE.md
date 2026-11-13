# Patient policy improvement loop

User authorized repeated diagnose -> improve -> compare cycles on 2026-11-10.
Start from patient_value_v1 (12.02392578125 on frozen auction-flop-v1).
Retain readable rules, mostly conservative full-table play and active strong-hand
value betting. No opponent identity leakage, neural training, uploads or edits
to other agents' candidates/frozen benchmark files.

Inspect concrete public decisions before proposing fixes. Preserve each
candidate source and test its intended behavior. Use fresh per-cycle diagnostic
and screening seeds, identical opponent panels and seat rotations. Screen
single changes against the current champion, optionally screen one declared
combination, then run full v1 confirmation before promotion. No candidate is
promoted solely on a small screen. Add a separate fresh holdout when cumulative
public-benchmark selection could plausibly flatter the reported gain.

Record each cycle's hypotheses, source hashes, paths, elapsed times, failures,
screen results, full comparisons and next diagnosis in this directory. Failed
hypotheses remain recorded. Do not replace known failures with retuned scores
under the same name. Stop new launches promptly if the user pauses the loop.

Current status: cycle 1 completed and wrapped up following the parent threads'
relay of the user's final competition deadline. No new directions pending.
Local champion: patient_defend_v2, 13.134765625 on full v1; paired +1.11083984375
(95% CI +0.90215 to +1.31953), separately confirmed on fresh 1,024-table-per-arm
holdout (+0.9765625, 95% CI +0.64593 to +1.30720). Below the parent submission
threshold; not submitted. See RESULTS.md for retained failed hypotheses,
remaining weaknesses, exact artifacts and resumption guidance.
