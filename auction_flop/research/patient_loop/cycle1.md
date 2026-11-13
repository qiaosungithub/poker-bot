# Cycle 1: prices and short-handed defense

Baseline: patient_value_v1, full v1 score 12.02392578125. Fresh diagnostic
64-table sample retained under diagnosis-v1: 589 folds in 776 short-handed
preflop decisions; 20 postflop folds with a genuine made hand or top/overpair.
Not every fold is wrong. A concrete AhAd/Qc4d2h decision folded its last 502
chips against a 2,179-chip visible pot because single-pair hands cannot bypass
the generic large-bet veto. Test narrow exceptions rather than general calling.

Four isolated proposals plus control, before screening results:

- `patient_price_v2`: after existing strong-hand decisions, allow top pair or
  overpair to call at low stack/pot ratio (<=2), heads-up, price <=.35 and
  uniform equity >= max(.65, price+.20). No new bluff/raise logic.
- `patient_shortcall_v2`: only change an existing preflop fold when <=3 players
  remain and own stack <=12 BB. Compare public eligible pot odds with the
  retained 169-class uniform heads-up equity, exponentiated for live opponents,
  minus .12 (.18 after multiple raises); cap price at .40. This is an approximate
  price safeguard, not a solved push/fold chart or an opponent-range posterior.
- `patient_loose_v2`: loosen full-table opening chart one tier and cheap-call
  scale to 1.7, as in earlier simple tuning; all patient postflop rules retained.
- `patient_huopen_v2`: only heads-up, add any king, suited queens/jacks, Q8+,
  J9+, T8+ and 98/97 to the opening/short-stack opening range. Responses to
  existing raises stay unchanged.

Screen 512 tables/arm using fresh `patient-loop-c1-screen-20261110` seeds.
Optionally combine isolated changes with mean gain >=.50 on this screen; if
more than one qualifies, screen exactly that combination on the same seeds.
Full-confirm the highest eligible screen mean on unchanged 4,096-table v1.
Promote only if full paired 95% interval versus baseline excludes zero on the
positive side. Preserve a non-improving full result and keep the old champion.
No tuning after confirmation. Subsequent cycles get new screen seeds.
