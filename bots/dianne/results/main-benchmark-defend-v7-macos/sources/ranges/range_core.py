"""Fresh event-time range/board-advantage heuristic; no SDK assumptions here."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import ceil
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARDS = {r+s: eval7.Card(r+s) for r in RANKS for s in SUITS}
DECK = tuple(CARDS)
TRIPLES = tuple(combinations(range(5), 3))
CARD_INFO = {c: (RANKS.index(c[0])+2, SUITS.index(c[1])) for c in DECK}
STRAIGHT_MASKS = tuple(31 << low for low in range(1,11))

@dataclass(frozen=True)
class RangeConfig:
    samples: int = 8
    call_margin: float = 0.035
    bluff_frequency: float = 0.18
    auction_rate: float = 2.0

@dataclass(frozen=True)
class BettingView:
    hole: tuple[str, str]
    board: tuple[str, ...]
    pot: int
    to_call: int
    stack: int
    own_bet: int
    big_blind: int
    opponents: int
    position: float = 0.5  # 0 early, 1 last to act among remaining players
    raises: int = 0
    hand_number: int = 0
    dead: tuple[str, ...] = ()


def starting_strength(hole):
    a, b = sorted((RANKS.index(c[0])+2 for c in hole), reverse=True)
    if a == b:
        return min(1.0, 0.46 + a * 0.038)
    gap = a-b-1
    return max(0.0, min(0.98, (a*0.044+b*0.021-0.24)
        + (0.065 if hole[0][1] == hole[1][1] else 0)
        + (0.055 if gap == 0 else 0.025 if gap == 1 else 0)
        - 0.018*max(0, gap-2)))


def _seed(cards):
    value = 2166136261
    for card in cards:
        value = ((value ^ (RANKS.index(card[0])*4+SUITS.index(card[1])+1))*16777619)&0xffffffff
    return value


def _opponents(known, raises, count):
    """Bounded deterministic sample of a broad or action-tightened range."""
    blocked = set(known)
    deck = [c for c in DECK if c not in blocked]
    n = len(deck)
    threshold = min(0.68, 0.28+0.12*max(0, raises))
    x = _seed(known)
    result = []
    for _ in range(count*10):
        x = (1664525*x+1013904223)&0xffffffff
        i = (x >> 8) % n
        x = (1664525*x+1013904223)&0xffffffff
        j = (x >> 8) % (n-1)
        if j >= i:
            j += 1
        pair = (deck[i], deck[j])
        if starting_strength(pair) >= threshold:
            result.append(pair)
            if len(result) == count:
                break
    # Very tight ranges can reject all samples; retain finite, legal fallback.
    if not result:
        result.append((deck[0], deck[-1]))
    return result


def _draw(hole, board):
    if len(board) >= 5:
        return 0.0
    counts = [0,0,0,0]
    own_high = [0,0,0,0]
    mask = board_mask = 0
    for c in board:
        rank, suit = CARD_INFO[c]
        counts[suit] += 1
        board_mask |= 1 << rank
    if board_mask & (1 << 14):
        board_mask |= 2
    mask = board_mask
    for c in hole:
        rank, suit = CARD_INFO[c]
        counts[suit] += 1
        own_high[suit] = max(own_high[suit],rank)
        mask |= 1 << rank
    if mask & (1 << 14):
        mask |= 2
    bonus = 0.0
    for n, own in zip(counts,own_high):
        if own and n == 4:
            bonus += 0.095 + (0.035 if own == 14 else 0)
        elif own and n == 3 and len(board) == 3:
            bonus += 0.012
    for window in STRAIGHT_MASKS:
        if (mask & window).bit_count() == 4 and (board_mask & window).bit_count() < 4:
            bonus += 0.065
            break
    return min(0.19,bonus)


def _relative(hole, board, opponents):
    """Current made-hand percentile plus symmetric draw adjustment, not equity."""
    board_cards = [CARDS[c] for c in board]
    own = eval7.evaluate([CARDS[c] for c in hole]+board_cards)
    wins = 0.0
    draw_other = 0.0
    for pair in opponents:
        rank = eval7.evaluate([CARDS[c] for c in pair]+board_cards)
        wins += 1.0 if own > rank else 0.5 if own == rank else 0.0
        draw_other += _draw(pair, board)
    n = len(opponents)
    score = wins/n + _draw(hole, board) - draw_other/n
    return max(0.02, min(0.98, score))


def board_advantages(hole, candidates, raises=0, config=RangeConfig()):
    """Score all ten flops against the SAME blocker-compatible opponent range.

    No runout is dealt: this does not assume whether rejected candidates return
    to the deck. All five visible candidates cannot be opponent private cards.
    """
    hole, candidates = tuple(hole), tuple(candidates)
    if len(hole) != 2 or len(candidates) != 5 or len(set(hole+candidates)) != 7:
        raise ValueError("expected two distinct hole and five distinct candidate cards")
    opp = _opponents(hole+candidates, raises, config.samples)
    return tuple(_relative(hole, tuple(candidates[i] for i in ix), opp) for ix in TRIPLES)


def auction_bids(hole, candidates, budget, remaining_hands, stack_bb,
                 raises=0, pot_commitment=0.0, config=RangeConfig()):
    budget = max(0, int(budget))
    if budget == 0:
        return (0,)*5
    scores = board_advantages(hole, candidates, raises, config)
    best_i = max(range(10), key=scores.__getitem__)
    mean = sum(scores)/10
    gain = scores[best_i]-mean
    spread = max(scores)-min(scores)
    if gain < 0.018 or spread < 0.045:
        return (0,)*5
    desired = TRIPLES[best_i]
    marginals = []
    for i in range(5):
        yes = sum(s for ix,s in zip(TRIPLES,scores) if i in ix)/6
        no = sum(s for ix,s in zip(TRIPLES,scores) if i not in ix)/4
        marginals.append(yes-no)
    # Survival limits how many future auctions this stack can use. Actual hand
    # cap is supplied by the adapter; no initial/replenishment budget assumed.
    horizon = max(1.0, min(float(remaining_hands), 8.0+1.5*max(0.0, stack_bb)))
    multiplier = config.auction_rate*(0.65+2.0*gain)*(1.0+min(2.0,max(0.0,pot_commitment)))
    spend = min(budget, max(1, ceil(budget/horizon*multiplier)))
    if remaining_hands <= 1:
        spend = budget
    weights = [max(0.0,marginals[i])+0.25*gain if i in desired else 0.0 for i in range(5)]
    total = sum(weights)
    raw = [spend*w/total for w in weights]
    bids = [int(x) for x in raw]
    order = sorted(desired,key=lambda i:raw[i]-bids[i],reverse=True)
    for i in order[:spend-sum(bids)]:
        bids[i] += 1
    return tuple(bids)


def betting_intent(v: BettingView, config=RangeConfig()):
    """Return desired kind and raise-to total; legal adapter still validates."""
    bb=max(1,v.big_blind); n=max(1,v.opponents)
    stack_bb=v.stack/bb
    mix=(_seed(v.hole+v.board) ^ ((v.hand_number+1)*2654435761))&0xffff
    mix/=65536.0
    if not v.board:
        strength=starting_strength(v.hole)
        open_cut=0.63-0.16*v.position+0.024*max(0,n-3)
        if n == 1:
            open_cut -= 0.11
        facing=max(0,v.raises)
        call_cut=open_cut+0.065*facing
        if v.to_call <= bb and facing == 0:
            if strength >= open_cut:
                return 'raise',max(v.own_bet+v.to_call+bb,int(2.4*bb))
            return ('check',0) if v.to_call == 0 else ('fold',0)
        pot_odds=v.to_call/max(1,v.pot+v.to_call)
        call_cut += 0.22*max(0.0,pot_odds-0.23)
        if strength > min(0.94,call_cut+0.19):
            return 'raise',v.own_bet+v.to_call+int(0.75*(v.pot+v.to_call))
        if strength >= call_cut or (stack_bb<5 and strength>0.62 and pot_odds<0.34):
            return 'call',0
        return ('check',0) if v.to_call == 0 else ('fold',0)
    opp=_opponents(v.hole+v.board+v.dead,v.raises,config.samples)
    estimate=_relative(v.hole,v.board,opp)
    # Percentiles are only a proxy; draw uncertainty and multiway realization
    # reduce the optimistic current showdown fraction before betting decisions.
    estimate=max(0.01,min(0.99,estimate**(1+0.52*(n-1))))
    if len(v.board)<5:
        estimate=0.10+0.80*estimate
    odds=v.to_call/max(1,v.pot+v.to_call)
    margin=config.call_margin+0.025*max(0,n-1)+0.025*max(0,v.raises-1)
    if v.to_call:
        if estimate >= 0.88 and (n<=2 or estimate>=0.94):
            return 'raise',v.own_bet+v.to_call+int(0.68*(v.pot+v.to_call))
        return ('call',0) if estimate>=odds+margin else ('fold',0)
    if estimate>=0.74:
        return 'raise',v.own_bet+max(bb,int(v.pot*(0.65 if estimate>=0.86 else 0.38)))
    draw=_draw(v.hole,v.board)
    if n==1 and draw>=0.065 and mix<config.bluff_frequency:
        return 'raise',v.own_bet+max(bb,int(v.pot*0.45))
    return 'check',0


def legalize(intent, legal):
    """Pure normalization for tests/adapters: legal maps kind to (min,max)."""
    kind, amount=intent
    if kind in legal:
        if kind=='raise':
            lower,upper=legal[kind]
            return kind,max(int(lower),min(int(upper),int(amount)))
        return kind,0
    if kind=='raise' and 'call' in legal:
        return 'call',0
    for fallback in ('check','fold','call'):
        if fallback in legal:
            return fallback,0
    if not legal:
        raise ValueError('no legal action')
    kind=next(iter(legal));return kind,int(legal[kind][0])
