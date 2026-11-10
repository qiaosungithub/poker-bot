"""Offline continuity audit and real-engine legality, never shipped in ZIP."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import process_time_ns

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(ROOT / 'bot'))
import classic_policy as policy
from agent import Agent
from preflop_table import EQUITY
from poker_agent_protocol.game import is_legal
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

def oracle():
    path = REPO / 'txnujk-bot/bots/floorfix/bot.py'
    spec = importlib.util.spec_from_file_location('original_classic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def dictionary(state):
    me = state.seats[state.seat]
    return {'your_cards': state.cards, 'your_stack': me.stack,
            'your_bet_this_street': me.bet, 'pot': state.pot,
            'amount_owed': state.to_call, 'current_bet': me.bet + state.to_call,
            'can_check': bool(state.can('check')),
            'min_raise_to': state.can('raise').min_amount if state.can('raise') else me.bet + me.stack,
            'action_log': [{'action': e.kind, 'seat': e.seat, 'amount': e.amount} for e in state.history]}

def normalized(result, state):
    kind = result['action']
    if kind in ('raise', 'all_in'):
        me = state.seats[state.seat]
        return policy.raise_to(state, me.stack + me.bet if kind == 'all_in' else result['amount'])
    if kind == 'call':
        return policy.call(state)
    return policy.passive(state)

def main():
    original = oracle()
    assert EQUITY == original.PREFLOP_EQ, '169-class values changed'
    states = []
    failures = []
    memory_max = 0
    timings = {}
    auction_control = __import__('tournament_agent').Agent()
    auction_equal = 0
    for table_index in range(8):
        rules = TableRules(variant='flop_auction_holdem', players=8, hand_cap=60)
        table = SitAndGo(rules=rules, variant=load_variant(rules.variant), table_id=f'classic-audit-{table_index}', seed=f'classic-audit-{table_index}')
        memories = [b''] * 8
        while (state := table.decision()) is not None:
            start = process_time_ns()
            action, memory = Agent().act(state, memories[state.seat])
            elapsed = (process_time_ns() - start) / 1e6
            phase = state.phase if state.phase == 'auction' else str(len(state.board))
            timings.setdefault(phase, []).append(elapsed)
            memory_max = max(memory_max, len(memory))
            memories[state.seat] = memory
            if state.phase == 'auction':
                expected, _ = auction_control.act(state, b'')
                assert action == expected
                auction_equal += 1
            elif len(states) < 6000:
                states.append(state)
            if not is_legal(action, state.legal_actions) or not table.apply(action):
                failures.append((table_index, state.hand_number, str(action)))
    assert not failures, failures[:3]
    mismatches = []
    count = 0
    feature_cases = 0
    saved_equity = policy.equity
    for source in states:
        state = replace(source, variant={})
        price = policy.eligible_call(state)
        if not price.exact or state.to_call > state.seats[state.seat].stack or price.eligible_pot_before_call != state.pot:
            continue  # Deliberate side-pot and partial-all-in correction.
        original.BIG_BLIND = state.blinds.big_blind
        d = dictionary(state)
        live = sum(s.status in ('active','all_in') for s in state.seats)
        ctx = {'players_after': policy.players_after(state), 'n_live': live, 'n_opp': live - 1,
               'hole': [policy.CARDS[c] for c in state.cards], 'board': [policy.CARDS[c] for c in state.board], 'deadline': 1e99}
        if state.board:
            if policy.shared_nut_action(state) is not None or policy.public_quads_share(state.cards,state.board) is not None:
                continue
            f = original.hand_features(ctx['hole'],ctx['board'])
            assert policy.hand_features(state.cards,state.board) == (f['cat'],f['commit'],f['fd'],f['oesd'])
            feature_cases += 1
        for archetype in ('unknown','maniac','nit','station'):
            ctx['facing_archetype'] = archetype
            for forced_equity in ((.1,.35,.6,.85) if state.board else (.5,)):
                original._RNG = policy.random_stream(state)
                original.mc_equity = lambda *args, q=forced_equity: q
                policy.equity = lambda *args, q=forced_equity: q
                old = original._postflop(d,ctx) if state.board else original._preflop(d,ctx)
                new = policy.postflop(state,archetype) if state.board else policy.preflop(state,archetype)
                count += 1
                if normalized(old,state) != new:
                    mismatches.append({'hand':state.hand_number,'board':state.board,'cards':state.cards,'eq':forced_equity,'archetype':archetype,'old':old,'new':str(new)})
    policy.equity = saved_equity
    assert not mismatches, mismatches[:5]
    # Correctness regressions independent of upstream mistakes.
    royal = replace(next(s for s in states if len(s.board)==5), cards=('2c','3d'), board=('As','Ks','Qs','Js','Ts'))
    assert policy.postflop(royal).kind in ('check','call')
    assert policy.public_quads_share(('3c','4d'),('As','Ah','Ad','Ac','2s'),('8c','9d')) < .03
    # Fresh/malformed memory and opponent-history isolation.
    s = states[0]
    assert Agent().act(s,b'bad json') == Agent().act(s,b'')
    result = {'source_rule_decisions':count,'rule_mismatches':len(mismatches),'feature_parity_states':feature_cases,
              'preflop_classes_equal':169,'native_tables':8,'native_decisions':sum(len(v) for v in timings.values()),
              'illegal_or_failed':len(failures),'auction_identical_decisions':auction_equal,'max_memory_bytes':memory_max,
              'timing_scope':'process CPU, full Agent.act including observation and memory, excludes IPC; concurrent benchmark load may affect cache',
              'timings':{k:{'n':len(v),'p99_ms':sorted(v)[int(.99*(len(v)-1))],'max_ms':max(v)} for k,v in timings.items()}}
    (ROOT/'evidence/continuity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
