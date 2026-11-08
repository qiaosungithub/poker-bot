"""Paired heads-up tournament evaluation of the frozen Fullhouse policy.

100 BB each, 0.5/1 BB fixed blinds, up to 200 hands; payout [100, 0].
Unfinished games receive exact two-player ICM = final chip share * 100.
Independent sampling units are two games with the same decks and swapped bots.
No time limit; no policy training. Pilot and confirmation use disjoint seeds.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import statistics
import sys
import time
import types

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent / 'latency-survey'
REPO = SURVEY / 'fullhouse-bot'
sys.path[:0] = [str(SURVEY), str(REPO), str(REPO / 'engine_vendored')]
PATHS = {
    'strong': REPO / 'bots/strong/bot.py',
    **{name: REPO / f'bots/adversarial/{name}.py'
       for name in ('nit', 'station', 'maniac', 'solver_like', 'random_bot')},
    **{name: REPO / f'bots/competitors/{name}/bot.py'
       for name in ('cfr_lite', 'solver_hybrid', 'equity_table', 'bayesian_exploiter', 'adaptive_hydra', 'cfr_equity_v28')},
    **{name: REPO / f'bots/extra/{name}/bot.py'
       for name in ('mc_equity', 'cfr_opp_profiler', 'cfr_variance_guard')},
    'purujit': Path('/tmp/poker-candidates-existing/fullhouse-purujit/bots/mybot/bot_impl.py'),
}
HERO = None
CODE = {}


def initialize():
    global HERO
    from bot import bot
    from fullhouse_onnx import ONNXLookup
    bot._DEEP_CFR = ONNXLookup(SURVEY / 'fullhouse_fp32.onnx')
    def forbidden_fallback(state):
        raise RuntimeError('Fullhouse emergency fallback used')
    bot._safe_default_action = forbidden_fallback
    HERO = bot


class FailOnDecideException(ast.NodeTransformer):
    """Expose swallowed runtime exceptions without changing successful decisions."""
    def visit_FunctionDef(self, node):
        if node.name == 'decide':
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    child.body.insert(0, ast.Raise())
        return node


def opponent(name):
    if name.startswith('lbr_'):
        from lbr_baseline import LocalBestResponse
        return LocalBestResponse({'lbr_fullhouse':1.,'lbr_cautious':.5,
                                  'lbr_cautious25':.25,'lbr_cautious10':.1}[name])
    if name == 'mirror':
        return HERO.decide
    if name == 'always_fold':
        return lambda s: {'action': 'check' if s['can_check'] else 'fold'}
    if name == 'always_call':
        return lambda s: {'action': 'check' if s['can_check'] else 'call'}
    path = PATHS[name]
    if name == 'purujit':
        os.environ['MYBOT_EAGER_LOAD'] = '1'
        sys.path.insert(0, str(path.parent))
        import opponent_model, range_tracker
        opponent_model.reset_profiles()
        range_tracker.reset_all_ranges()
    if name not in CODE:
        tree = FailOnDecideException().visit(ast.parse(path.read_text()))
        CODE[name] = compile(ast.fix_missing_locations(tree), str(path), 'exec')
    module = types.ModuleType('opponent_' + name)
    module.__file__ = str(path)
    exec(CODE[name], module.__dict__)
    return module.decide


def play_game(name, seed, hero_seat, max_hands):
    from engine_adapter import PokerEngine
    random.seed(seed * 13 + hero_seat)
    opp = opponent(name)  # fresh opponent memory for every independent game
    rngs = [random.Random(seed * 101 + i) for i in (0, 1)]
    stacks = [10000, 10000]
    log = []
    actions = Counter()
    corrections = Counter()
    hands = 0
    for h in range(max_hands):
        if min(stacks) == 0:
            break
        e = PokerEngine(f'{seed}_{hero_seat}_{h}', ['p0', 'p1'], h % 2,
                        dict(zip(('p0', 'p1'), stacks)), seed * 1000003 + h)
        s = e.start_hand()
        if hasattr(opp,'new_hand'):
            opp.new_hand(1-hero_seat, [str(c) for c in e.players[1-hero_seat].hole_cards])
        steps = 0
        while s['type'] == 'action_request':
            seat = s['seat_to_act']
            assert e.players[seat].is_active
            assert all(p['hole_cards'] is None for p in s['players'])
            s['match_action_log'] = log[-200:]
            HERO._RNG = rngs[seat]
            action = HERO.decide(s) if seat == hero_seat else opp(s)
            if not isinstance(action, dict) or action.get('action') not in ('fold','check','call','raise','all_in'):
                raise ValueError(f'Invalid action from {name}: {action}')
            role = 'hero' if seat == hero_seat else 'opponent'
            if action['action'] == 'check' and not s['can_check']:
                corrections[role + '_illegal_check'] += 1
            if action['action'] == 'raise':
                if action['amount'] < min(s['min_raise_to'], s['your_stack'] + s['your_bet_this_street']):
                    corrections[role + '_undersize_raise'] += 1
            actions[role + '_' + s['street'] + '_' + action['action']] += 1
            if seat == hero_seat and hasattr(opp,'observe'):
                public=dict(s);public.pop('your_cards')
                opp.observe(public,action)
            s = e.apply_action(seat, action)
            actual = e.action_log[-1]
            log.append(dict(actual, hand_num=h, bot_id=f'p{seat}'))
            steps += 1
            if steps > 300:
                raise RuntimeError('Too many actions')
        stacks = [p.stack for p in e.players]
        assert min(stacks) >= 0 and sum(stacks) == 20000, stacks
        hands += 1
    chips = stacks[hero_seat]
    return dict(score=chips / 200, chip_delta=chips - 10000, hands=hands,
                eliminated=chips == 0, opponent_eliminated=chips == 20000,
                ahead=chips > 10000, tied=chips == 10000,
                corrections=dict(corrections), actions=dict(actions), opponent_diagnostics=getattr(opp,'stats',{}))


def play_pair(job):
    name, seed, max_hands = job
    start = time.perf_counter()
    try:
        legs = [play_game(name, seed, seat, max_hands) for seat in (0, 1)]
        return dict(opponent=name, seed=seed, legs=legs,
                    score=sum(g['score'] for g in legs)/2,
                    seconds=time.perf_counter()-start)
    except Exception:
        import traceback
        return dict(opponent=name, seed=seed, error=traceback.format_exc(),
                    seconds=time.perf_counter()-start)


def mean_ci(values, z=1.96):
    n = len(values)
    if not n:
        return None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return dict(mean=mean, se=se, ci95=[mean-z*se, mean+z*se] if se is not None else None)


def summarize(rows):
    result = {}
    for name in sorted({r['opponent'] for r in rows}):
        all_rows = [r for r in rows if r['opponent'] == name]
        good = [r for r in all_rows if 'error' not in r]
        legs = [g for r in good for g in r['legs']]
        corrections, actions = Counter(), Counter()
        for leg in legs:
            corrections.update(leg['corrections']); actions.update(leg['actions'])
        result[name] = dict(pairs=len(good), games=len(legs), errors=len(all_rows)-len(good),
                            hands=sum(g['hands'] for g in legs), score=mean_ci([r['score'] for r in good]),
                            corrections=dict(corrections), actions=dict(actions))
        for key in ('eliminated', 'opponent_eliminated', 'ahead', 'tied'):
            result[name][key] = mean_ci([100 * sum(g[key] for g in r['legs'])/2 for r in good])
        # Ratio describes this tournament protocol, NOT a reset-stack cash-game win rate.
        if legs:
            result[name]['tournament_bb100'] = sum(g['chip_delta'] for g in legs) / sum(g['hands'] for g in legs)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opponents', default='strong,cfr_lite,solver_hybrid,equity_table,nit,station,maniac,solver_like')
    ap.add_argument('--pairs', type=int, default=32)
    ap.add_argument('--hands', type=int, default=200)
    ap.add_argument('--seed', type=int, default=100000)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = HERE / args.output
    raw_path = output.with_suffix('.jsonl')
    if output.exists() or raw_path.exists():
        raise FileExistsError('Refusing to overwrite experiment results')
    names = args.opponents.split(',')
    jobs = [(n, args.seed+i, args.hands) for i in range(args.pairs) for n in names]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [REPO/'bot/bot.py', REPO/'bot/features.py', REPO/'data/deep_cfr_model.npz',
                        REPO/'engine_vendored/engine/game.py'] + [PATHS[n] for n in names if n in PATHS]}
    metadata = dict(config=vars(args), protocol=__doc__, hashes=hashes,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    engine_adapter_sha256=hashlib.sha256((HERE/'engine_adapter.py').read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256((SURVEY/'fullhouse_fp32.onnx').read_bytes()).hexdigest())
    metadata['local_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in HERE.glob('*.py')}
    start = time.perf_counter()
    rows = []
    with raw_path.open('w') as raw, ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        pending = [pool.submit(play_pair, job) for job in jobs]
        last_report = start
        for fut in as_completed(pending):
            row = fut.result(); rows.append(row)
            raw.write(json.dumps(row)+'\n'); raw.flush()
            if 'error' in row:
                print(json.dumps(row), flush=True)
            now = time.perf_counter()
            if now-last_report > 20:
                print(json.dumps(dict(completed=len(rows), total=len(jobs), seconds=now-start,
                                      summary={k:{'pairs':v['pairs'],'score':v['score'],'errors':v['errors']}
                                               for k,v in summarize(rows).items()})), flush=True)
                last_report = now
    result = dict(metadata=metadata, elapsed_seconds=time.perf_counter()-start, results=summarize(rows))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'], results={
        k:{f:v[f] for f in ('pairs','hands','errors','score','corrections')}
        for k,v in result['results'].items()}), indent=2), flush=True)


if __name__ == '__main__':
    main()
