"""Event-time regeneration of the documented txnujk floorfix equity classes."""
from pathlib import Path
import random
import eval7

RANKS = '23456789TJQKA'
DECK = tuple(eval7.Card(r + s) for r in RANKS for s in 'shdc')

def estimate(first, second, stream):
    hole = [eval7.Card(first), eval7.Card(second)]
    deck = [card for card in DECK if card not in hole]
    total = 0.0
    for _ in range(900):
        sample = stream.sample(deck, 7)
        hero = eval7.evaluate(hole + sample[2:])
        other = eval7.evaluate(sample)
        total += (hero > other) + 0.5 * (hero == other)
    return total / 900

def main():
    result = {}
    stream = random.Random(12345)
    for low, r in enumerate(RANKS):
        for high in RANKS[low + 1:]:
            for suited in (True, False):
                key = high + r + ('s' if suited else 'o')
                result[key] = estimate(r + 's', high + ('s' if suited else 'h'), stream)
    pairs = random.Random(67890)
    for rank in RANKS:
        result[rank + rank] = estimate(rank + 's', rank + 'h', pairs)
    assert len(result) == 169
    target = Path(__file__).parent / 'bot/preflop_table.py'
    target.write_text('"""169 deterministic 900-sample equity classes, regenerated during event."""\nEQUITY = ' + repr(result) + '\n')

if __name__ == '__main__':
    main()
