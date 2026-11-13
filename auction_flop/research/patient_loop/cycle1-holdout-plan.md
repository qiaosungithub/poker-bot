# Independent final confirmation

Before the full v1 result is read, freeze a fresh 1,024-table per arm comparison
(32 blocks x four frozen panels x eight rotations) of patient_defend_v2 against
patient_value_v1. Seed prefix: patient-loop-c1-holdout-20261110. This is external
validation of the selected combination, not further parameter selection.
Reuse the screen runner for scheduling/source locks, but label the resulting
score as a separate holdout, never as the permanent auction-flop-v1 score.
No further changes or arms are allowed in this holdout. Also report block-paired
95% interval. This closes the current cycle following the parent thread's relay
of the user's final competition wrap-up; do not launch additional directions.
