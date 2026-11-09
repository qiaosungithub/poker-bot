# ICM v3 committed-call candidate

Derived from selected ICM v2 during the event. A narrowly bounded preflop
exception handles nearly completed heads-up calls. The call costs at most
4% of the contestable pot and must pass the existing ICM threshold at 10%
or below. Otherwise the original policy applies.

The official Boston engine reproduced a concrete overfold in ICM v2. With
stacks 21 and 3979, hero posted the 20 big blind and held Jd8c. An opponent
raise to 100 left hero needing only one chip for a contestable pot of 41.
The ICM threshold was 8.38%, yet the fixed preflop range threshold folded.
V3 calls. A normal expensive call with the same cards still folds.

Two focused official-engine regressions and static SDK validation passed.
Tournament-strength comparison is pending.
