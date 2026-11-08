import itertools
import unittest
import eval7
import multiway_benchmark as M


class MultiwayTests(unittest.TestCase):
    def test_simultaneous_eliminations(self):
        names=['a','b','c','d','e','f']
        self.assertEqual(M.elimination_scores(names,[100,200,300,400,500,600],[0,0,500,600,500,500]),{'a':0.,'b':10.})
        self.assertEqual(M.elimination_scores(names,[100,100,300,400,500,600],[0,0,400,500,600,500]),{'a':5.,'b':5.})
        self.assertEqual(M.elimination_scores(names[:3],[100,200,300],[0,0,600]),{'a':40.,'b':60.})

    def test_icm_against_permutations(self):
        stacks=[1,2,3,7,11,23];prizes=M.PAYOUTS
        out=[0.]*6
        for order in itertools.permutations(range(6)):
            probability=1.;total=sum(stacks)
            for i in order:probability*=stacks[i]/total;total-=stacks[i]
            for rank,i in enumerate(order):out[i]+=probability*prizes[rank]
        for a,b in zip(out,M.icm(stacks,prizes)):self.assertAlmostEqual(a,b,10)

    def test_symmetric_icm(self):
        for n in range(1,7):
            for v in M.icm([100]*n,M.PAYOUTS[:n]):self.assertAlmostEqual(v,sum(M.PAYOUTS[:n])/n)

    def test_initial_actor_and_hidden_cards(self):
        for n in range(2,7):
            for dealer in range(n):
                h=M.Hand(list(map(str,range(n))),[10000]*n,dealer,123+n,'test')
                s=h.state();self.assertEqual(s['seat_to_act'],dealer if n==2 else (dealer+3)%n)
                self.assertTrue(all(p['hole_cards'] is None for p in s['players']))
                self.assertEqual(s['your_cards'],h.holes[s['seat_to_act']])

    def test_short_raise_does_not_reopen(self):
        h=M.Hand(['a','b','c'],[1000,250,1000],0,321,'short')
        h.act({'action':'raise','amount':200});h.act({'action':'all_in'});h.act({'action':'call'})
        s=h.state();self.assertEqual(s['seat_to_act'],0);self.assertEqual(s['amount_owed'],50)
        self.assertFalse(s['can_raise']);self.assertEqual(s['min_raise_to'],0)
        h.act({'action':'raise','amount':800})
        self.assertEqual(h.normalizations['a_raise_not_reopened_to_call'],1)
        self.assertEqual(h.state()['pot'],750)

    def test_sidepots_against_independent_eval7(self):
        for seed in range(200):
            n=2+seed%5;stacks=[100,250,700,1300,2700,5000][:n]
            h=M.Hand(list(map(str,range(n))),stacks,seed%n,seed,'allin')
            steps=0
            while h.state() is not None:
                h.act({'action':'all_in'});steps+=1;self.assertLess(steps,100)
            ranks=[eval7.evaluate([eval7.Card(c) for c in cards+h.board]) for cards in h.holes]
            expected=[0]*n;previous=0
            for level in sorted(set(stacks)):
                contenders=[i for i in range(n) if stacks[i]>=level]
                pot=(level-previous)*len(contenders);previous=level
                best=max(ranks[i] for i in contenders)
                winners=sorted([i for i in contenders if ranks[i]==best],key=h.inverse.get)
                share,odd=divmod(pot,len(winners))
                for j,i in enumerate(winners):expected[i]+=share+(j<odd)
            self.assertEqual(h.final_stacks(),expected,(seed,stacks,h.holes,h.board))

    def test_six_seat_mirror(self):
        M.C.R.initialize()
        roster=['a','b','c','d','e','f']
        legs=[M.match(roster[r:]+roster[:r],9000222,200,mirror=True) for r in range(6)]
        for name in roster:self.assertAlmostEqual(sum(g['scores'][name] for g in legs)/6,sum(M.PAYOUTS)/6,9)


if __name__=='__main__':unittest.main()
