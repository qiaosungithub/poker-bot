import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bot'))
from range_core import *

def test_legality():
    assert legalize(('raise',1),{'raise':(40,200),'call':(0,0)})==('raise',40)
    assert legalize(('raise',999),{'raise':(40,200)})==('raise',200)
    assert legalize(('raise',999),{'call':(0,0),'fold':(0,0)})==('call',0)
    assert legalize(('fold',0),{'check':(0,0)})==('check',0)

def test_ranges():
    assert starting_strength(('As','Ah'))>starting_strength(('2c','7d'))
    assert starting_strength(('As','Ks'))>starting_strength(('Ac','Kd'))
    assert _relative(('9s','9h'),('9c','9d','2s'),[("As","Ah")])>0.9
    assert abs(_relative(('2s','3h'),('Ah','Ad','Ks','Kd','Qc'),[("2c","3c")])-0.5)<1e-9

def test_auction():
    hole=('As','Ks');cards=('Qs','Js','Ts','2d','3c')
    scores=board_advantages(hole,cards)
    assert len(scores)==10
    assert scores[TRIPLES.index((0,1,2))]==max(scores)
    for budget in (0,1,2,10,100,10000):
        bids=auction_bids(hole,cards,budget,120,25)
        assert len(bids)==5 and min(bids)>=0 and sum(bids)<=budget
        assert all(type(b)==int for b in bids)
        assert bids==auction_bids(hole,cards,budget,120,25)
    assert sum(auction_bids(hole,cards,100,1,25))==100
    try:
        board_advantages(hole,('As','Js','Ts','2d','3c'))
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate card negative control did not fail')

if __name__=='__main__':
    from range_core import _relative
    test_legality();test_ranges();test_auction()
    print('3 core test groups passed')
