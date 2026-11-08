"""Differential validation against independent PokerKit rules and hand evaluator.

Run with /tmp/poker-hu-validation/bin/python validate_against_pokerkit.py.
Requires Python >=3.11, pokerkit==0.7.7, eval7==0.1.11.
No policy or benchmark configuration is modified by this validation.
"""
import json
import random
import time
import run_benchmark  # installs local engine path, does not initialize policy
from engine_adapter import PokerEngine
from pokerkit import Automation, NoLimitTexasHoldem


def check_hand(seed, stacks, dealer):
    rng=random.Random(seed+73419)
    e=PokerEngine(str(seed),['p0','p1'],dealer,dict(zip(('p0','p1'),stacks)),seed)
    state=e.start_hand()
    # PokerKit heads-up index 0=BB, 1=button/SB.
    order=[1-dealer,dealer]
    automations=tuple(a for a in Automation if a.name not in
                      ('HOLE_DEALING','BOARD_DEALING','RUNOUT_COUNT_SELECTION','CARD_BURNING'))
    k=NoLimitTexasHoldem.create_state(automations,True,0,(50,100),100,
                                    [stacks[i] for i in order],2)
    for j,seat in enumerate(order):
        k.deal_hole(''.join(str(c) for c in e.players[seat].hole_cards),j)
    board=[str(c) for c in e._deck_cards[4:9]]
    dealt=0
    def advance():
        nonlocal dealt
        while k.can_burn_card() or k.can_deal_board():
            if k.can_burn_card():k.burn_card('??')
            else:
                count=k.board_dealing_count
                k.deal_board(''.join(board[dealt:dealt+count]));dealt+=count
    advance()
    actions=0
    while state['type']=='action_request':
        seat=state['seat_to_act']; ki=order.index(seat)
        assert k.actor_index==ki, ('actor',seed,stacks,dealer,state,k.actor_index)
        assert k.checking_or_calling_amount==min(state['amount_owed'],state['your_stack']), ('owed',seed,state)
        assert k.stacks[ki]==state['your_stack'],('stack',seed,state,k.stacks)
        r=rng.random()
        if r<.15 and not state['can_check']: action={'action':'fold'}
        elif r<.60:action={'action':'check' if state['can_check'] else 'call'}
        elif r<.80:action={'action':'all_in'}
        else:action={'action':'raise','amount':max(state['min_raise_to'],state['current_bet']+rng.choice((1,50,100,200,700)))}
        valid=e._validate(seat,action)
        if valid.action=='fold':k.fold()
        elif valid.action in ('call','check'):k.check_or_call()
        else:
            amount=valid.amount
            if amount<=max(k.bets) or k.stacks[1-ki]==0:k.check_or_call()
            else:k.complete_bet_or_raise_to(amount)
        state=e.apply_action(seat,action)
        advance();actions+=1
        assert actions<300
    assert not k.status, ('not_done',seed,state,k)
    expected=[state['final_stacks'][f'p{i}'] for i in order]
    assert k.stacks==expected,('settlement',seed,stacks,dealer,k.stacks,expected,state)
    return actions


def main():
    start=time.perf_counter();rng=random.Random(6814);actions=0
    # Include blind-sized and short-raise stacks as well as arbitrary stacks.
    sizes=[1,20,49,50,51,99,100,101,149,150,199,200,249,250,251,499,500,1000,10000]
    for i in range(2000):
        size=sizes[i%len(sizes)] if i<200 else rng.randrange(1,20000)
        actions+=check_hand(8000000+i,(size,20000-size),i%2)
    result={'reference':'PokerKit 0.7.7','hands':2000,'actions':actions,'mismatches':0,
            'seconds':time.perf_counter()-start,'checks':['actor order','amount to call','stack before actions','final stacks','independent hand evaluator']}
    open('engine_differential_validation.json','w').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
