# ICM v2 candidate

Built from the frozen `icm_survival` candidate during the Boston event.
Preserves its payout-aware play and auction budgeting.

Current-street contributions above hero's affordable total are excluded from
pot odds and betting sizes. For a heads-up terminal call, the opponent receives
the uncallable excess even when hero wins the contested pot. Fold, win and lose
outcomes conserve chips, including oversized all-ins.

On four-flush boards, private flush strength now depends on the available
higher suited ranks. On board-trip full houses, the pair rank affects strength.

The original ICM and hand-strength approximations still apply. Match comparison
is pending. Targeted regressions and the SDK static checker passed.
