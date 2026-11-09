"""V2 auction-only refinement: price pivotal outcomes and remove redundant votes.

The board utility and opponent scenarios match v1. Beliefs about hidden bids are
heuristic scenarios, never observations. In-model dominance is not a guarantee
against unmodeled opponent bids.
"""
from auction_value import TRIPLES, TRIPLE_INDEX, chosen_indices, forcing_bid

ZERO = (0, 0, 0, 0, 0)


def bid_metrics(utilities, bid, scenarios, tie_order=(0, 1, 2, 3, 4)):
    """(expected gain vs zero, paid currency, change probability, downside)."""
    gain = paid = changed = downside = 0.
    for opposition, weight in scenarios:
        before = chosen_indices(opposition, tie_order)
        after = chosen_indices(tuple(opposition[i] + bid[i] for i in range(5)), tie_order)
        delta = utilities[TRIPLE_INDEX[after]] - utilities[TRIPLE_INDEX[before]]
        gain += weight * delta
        paid += weight * sum(bid[i] for i in after)
        changed += weight * (after != before)
        downside += weight * max(0., -delta)
    return gain, paid, changed, downside


def prune_nonpivotal(bid, scenarios, tie_order=(0, 1, 2, 3, 4)):
    """Reduce a vector while preserving its complete flop in every scenario.

    Each selected card need only beat the highest excluded total, with the exact
    tie rule. Votes already supplied by opponents are subtracted. Lowering a bid
    on an excluded card cannot hurt the current selected set. At most three
    passes propagate these reductions, keeping runtime independent of balances.
    """
    priority = {card: place for place, card in enumerate(tie_order)}
    current = bid
    for _ in range(3):
        necessary = [0] * 5
        for opposition, _ in scenarios:
            selected = chosen_indices(tuple(opposition[i] + current[i] for i in range(5)), tie_order)
            excluded = tuple(i for i in range(5) if i not in selected)
            for i in selected:
                needed = max(opposition[j] + current[j] + (priority[i] > priority[j]) - opposition[i]
                             for j in excluded)
                necessary[i] = max(necessary[i], needed)
        reduced = tuple(min(current[i], max(0, necessary[i])) for i in range(5))
        if reduced == current:
            break
        current = reduced
    return current


def choose_with_scenarios(utilities, budget, scenarios, shadow_price,
                          tie_order=(0, 1, 2, 3, 4), downside_weight=.15):
    """At most two priced target proposals per scenario, then pivotal pruning.

    Unlike v1's two largest absolute utilities, a candidate enters this set by
    improvement over the zero-bid outcome minus the price of causing it. This
    admits cheap, useful partial improvements and preserves complementary cards.
    """
    budget = max(0, int(budget))
    if not budget or max(utilities) - min(utilities) < .005:
        return ZERO
    proposals = {ZERO}
    for opposition, _ in scenarios:
        default = chosen_indices(opposition, tie_order)
        baseline = utilities[TRIPLE_INDEX[default]]
        ranked = []
        for index, target in enumerate(TRIPLES):
            gain = utilities[index] - baseline
            if gain <= .005:
                continue
            bid = forcing_bid(target, opposition, tie_order)
            spend = sum(bid)  # All nonzero components are in this forced target.
            if spend <= budget:
                net = gain - shadow_price * spend
                if net > 0:
                    ranked.append((net, -spend, bid))
        ranked.sort(reverse=True)
        for _, _, bid in ranked[:2]:
            proposals.add(bid)
    best, best_net, best_paid = ZERO, 0., 0.
    for bid in sorted(proposals):
        gain, paid, changed, downside = bid_metrics(utilities, bid, scenarios, tie_order)
        net = gain - shadow_price * paid - downside_weight * downside
        if net > best_net + 1e-12 or abs(net - best_net) <= 1e-12 and paid < best_paid:
            best, best_net, best_paid = bid, net, paid
    # Dominance cleanup preserves every modeled result, so it cannot worsen
    # modeled gain/downside and weakly reduces selected-card payment.
    return prune_nonpivotal(best, scenarios, tie_order) if any(best) else ZERO


def choose_bids(utilities, budget, opponent_balances, hands_left=80,
                stack_bb=30., pot_bb=5., all_in=False, tie_order=(0, 1, 2, 3, 4)):
    """Drop-in v1 auction selector; same valuation/scenario/currency premises."""
    budget = max(0, int(budget))
    if not budget:
        return ZERO
    horizon = min(30., max(1., hands_left * .25), max(2., stack_bb * .7))
    if all_in:
        horizon = min(horizon, 3.)
    spend = sum(min(b, max(0, int(b / max(2., horizon)))) for b in opponent_balances)
    if spend <= 0:
        scenarios = ((ZERO, 1.),)
    else:
        scale = max(1, spend // 5)
        shift = int(sum((i + 1) * u for i, u in enumerate(utilities)) * 31) % 5
        pattern = (0, 1, 1, 2, 4)
        typical = tuple(pattern[(i + shift) % 5] * scale for i in range(5))
        high = tuple(pattern[(i + shift + 2) % 5] * scale * 2 for i in range(5))
        scenarios = ((ZERO, .3), (typical, .45), (high, .25))
    cost = .025 * horizon / max(1, budget) / max(1., (pot_bb / 5.) ** .5) if hands_left > 1 else 0.
    return choose_with_scenarios(utilities, budget, scenarios, cost, tie_order)
