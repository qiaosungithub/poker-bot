"""Bounded hand-contribution corrections for standard Boston Hold'em.

No opponent hand enumeration runs in decisions. Public-quads equity is computed
from thirteen rank counts under a uniform-opponent reference distribution. This
is an exact heads-up reference, not an action-conditioned multiway equity.
"""
from collections import Counter
import eval7
from poker_agent_sdk import Action

RANKS = '23456789TJQKA'
DECK = tuple(r + s for r in RANKS for s in 'cdhs')
CARD = {c: eval7.Card(c) for c in DECK}
RANK = {r: i + 2 for i, r in enumerate(RANKS)}


def _available_rank_counts(cards, board, dead_cards=()):
    known = set(cards) | set(board) | set(dead_cards)
    counts = [0] * 15
    for card in DECK:
        if card not in known:
            counts[RANK[card[0]]] += 1
    return counts


def _pairs(n):
    return n * (n - 1) / 2


def _quad_share(counts, kicker, board_kicker=0):
    denominator = _pairs(sum(counts))
    if denominator <= 0:
        return .5
    below = sum(counts[:kicker])
    at_or_below = below + counts[kicker]
    if board_kicker == kicker:
        # We play the shared kicker: no outright wins, only ties or losses.
        return .5 * _pairs(at_or_below) / denominator
    return (_pairs(below) + .5 * (_pairs(at_or_below) - _pairs(below))) / denominator


def public_quads_share(cards, board, dead_cards=()):
    """Uniform heads-up pot share, including a random river if board has 4 cards."""
    if len(board) not in (4, 5):
        return None
    board_ranks = [RANK[c[0]] for c in board]
    quad_rank = next((r for r in set(board_ranks) if board_ranks.count(r) == 4), None)
    if quad_rank is None:
        return None
    hole_kicker = max(RANK[c[0]] for c in cards)
    counts = _available_rank_counts(cards, board, dead_cards)
    if len(board) == 5:
        public_kicker = next(r for r in board_ranks if r != quad_rank)
        return _quad_share(counts, max(hole_kicker, public_kicker), public_kicker)
    # A river is exchangeable with hidden opponent cards. Integrate 13 ranks,
    # then use the exact rank-count formula for the remaining two-card hands.
    total = sum(counts)
    if not total:
        return .5
    share = 0.
    for river_rank in range(2, 15):
        count = counts[river_rank]
        if count:
            counts[river_rank] -= 1
            share += count / total * _quad_share(counts, max(hole_kicker, river_rank), river_rank)
            counts[river_rank] += 1
    return share


def guaranteed_shared_nuts(cards, board, dead_cards=()):
    """Prove selected shared-board nut cases; false means unproven, not beatable."""
    if len(cards) != 2 or len(board) != 5:
        return False
    board_value = eval7.evaluate([CARD[c] for c in board])
    if eval7.evaluate([CARD[c] for c in cards + board]) != board_value:
        return False
    category = board_value >> 24
    ranks = [RANK[c[0]] for c in board]
    known = set(cards) | set(board) | set(dead_cards)
    if category == 8:
        # Only a higher straight flush can beat a straight-flush board. A
        # possible higher window may require at most two available hole cards.
        suit = board[0][1]
        board_set = set(ranks)
        if 14 in board_set:
            board_set.add(1)
        high = max(h for h in range(5, 15) if all(r in board_set for r in range(h - 4, h + 1)))
        for candidate_high in range(high + 1, 15):
            missing = [r for r in range(candidate_high - 4, candidate_high + 1) if r not in board_set]
            if len(missing) <= 2 and all(RANKS[(14 if r == 1 else r) - 2] + suit not in known for r in missing):
                return False
        return True
    if category == 7:
        counts = Counter(ranks)
        quad_rank = next(r for r, n in counts.items() if n == 4)
        kicker = next(r for r, n in counts.items() if n == 1)
        available = _available_rank_counts(cards, board, dead_cards)
        return not any(available[r] for r in range(kicker + 1, 15))
    if category == 6:
        counts = Counter(ranks)
        trips = next(r for r, n in counts.items() if n == 3)
        pair = next(r for r, n in counts.items() if n == 2)
        available = _available_rank_counts(cards, board, dead_cards)
        if available[trips] or available[pair] >= 2:
            return False  # An opponent can make quads.
        if pair > trips and available[pair]:
            return False  # The board pair can become a higher triplet.
        return not any(r != trips and available[r] >= 2 for r in range(pair + 1, 15))
    if category == 4 and set(ranks) == {10, 11, 12, 13, 14}:
        # Broadway cannot be beaten by another straight. With at most two
        # cards of any suit on board, two hole cards cannot complete a flush.
        return max(Counter(c[1] for c in board).values()) <= 2
    return False


def adjust_hand_quality(cards, board, legacy_result, dead_cards=()):
    quads = public_quads_share(cards, board, dead_cards)
    if quads is not None:
        return quads, 7, False
    if guaranteed_shared_nuts(cards, board, dead_cards):
        return .5, legacy_result[1], False
    return legacy_result


def shared_nut_action(state):
    if (state.phase != 'betting' or len(state.board) != 5
            or not guaranteed_shared_nuts(state.cards, state.board,
                                          tuple(state.variant.get('auction_cards', ())))):
        return None
    # With zero rake and every surviving hand tied, a legal call cannot lose
    # incremental chips in any matched side-pot layer. Avoid needless raises.
    if state.can('check'):
        return Action.check()
    if state.can('call'):
        return Action.call()
    return None
