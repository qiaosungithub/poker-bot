"""Two independently switchable private-contribution refinements.

Neither flag changes auction policy or preflop. The flush rule only vetoes
automatic value re-raises while facing a wager; it does not reduce equity,
fold the hand, or prevent an ordinary unopened value bet.
"""
FLUSH_RERAISE_GUARD = False
PRIVATE_PAIR_COMMITMENT_GUARD = True
RANKS = '23456789TJQKA'
RANK = {r: i + 2 for i, r in enumerate(RANKS)}


def low_one_card_flush(cards, board, category, dead_cards=()):
    if category != 5 or len(board) < 4:
        return False
    suit = next((s for s in 'cdhs' if sum(c[1] == s for c in board) == 4), None)
    if suit is None:
        return False
    private = [c for c in cards if c[1] == suit]
    if len(private) != 1:
        return False
    known = set(cards) | set(board) | set(dead_cards)
    rank = RANK[private[0][0]]
    higher = sum(RANK[r] > rank and r + suit not in known for r in RANKS)
    # Nut and second-nut private suit cards retain the frozen policy.
    return higher >= 2


def pocket_pair_with_public_pair(cards, board, category):
    """Identify a private underpair plus a shared pair, without changing equity.

    The caller can apply the already-existing one-pair commitment rule. This
    does not say the hand is weak against random cards or should fold cheaply.
    """
    if category != 2 or cards[0][0] != cards[1][0]:
        return False
    ranks = [RANK[c[0]] for c in board]
    counts = {r: ranks.count(r) for r in ranks}
    if sum(n == 2 for n in counts.values()) != 1 or max(counts.values()) != 2:
        return False
    singletons = sorted((r for r, n in counts.items() if n == 1), reverse=True)
    pair = RANK[cards[0][0]]
    return bool(singletons) and pair < singletons[0]
