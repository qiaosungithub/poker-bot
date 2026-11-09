# ICM auction opportunity budget

This isolated auction candidate starts from frozen selected source `73ed48b9128862704c8123b0e9c582ab366c4fa3b84206998874154fc4b17f56`. An AST comparison confirmed that every definition except `auction_bid` remains identical.

The 128-table diagnostic observed 1,573 auctions and 738 positive auction payments for the focus bot. That is approximately 12.3 auctions and 5.8 paid auctions per table. Among 105 eliminations, 55 retained more than 500 flop-chips. These observations motivate budgeting against future valuable auctions rather than treating 45 remaining poker hands as 45 future payments. The estimates describe the tested field and may shift against tournament opponents.

The proposed horizon is six to fifteen future valuable auction opportunities. It depends on stack depth in big blinds, remaining players, and the hand cap. Shorter stacks and fewer remaining hands accelerate spending. Candidate flop selection and the existing minimum improvement threshold remain unchanged.

Ordinary spending is capped between 20 and 30 percent of the remaining auction currency according to the improvement. The cap rises to 45 percent only when pot commitment exceeds 75 percent and the quality gain exceeds 0.25. All-in decisions retain the baseline urgency calculation.

The summed public balances of opponents still in the hand bound their aggregate bid on any single card. Three equal bids above that sum guarantee the preferred flop, so spending never exceeds that sufficient amount. When all those balances are zero, spending is at most three chips. An own balance below three still respects its legal maximum.

Validation passed the official SDK checker and 1,200 randomized auction states spanning two through ten players, balances zero through 1,000, short stacks, all-ins, and exhausted opponents. Direct auction timing measured 0.265792 ms P99 and 0.709708 ms maximum in this sample, excluding SDK transport. Selection requires the parent's held-out comparison.

Frozen candidate SHA256 is `91a193c533f2eebae6ef59de50e037aee0af9a51512f9e2abe500b2b6a9429eb`.
