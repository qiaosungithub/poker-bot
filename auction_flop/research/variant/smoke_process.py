"""Reusable qualification of a frozen bot snapshot through official SDK processes.

This runs ordinary local processes, not the production microVM. Defaults include
six short zero-bank heads-up tables and three normal-bank eight-seat tables.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import poker_agent_protocol
from poker_agent_protocol.process import BotPool, CLEAN_ENV
from poker_agent_protocol.runner import BotError
from poker_engine.scoring import table_points
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant


def hashes(folder):
    result = {}
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        if '__pycache__' in relative.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError(f'Frozen input must not contain symlinks: {relative}')
        if path.is_file():
            result[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


async def run(args):
    bot_dir, kit = args.bot_dir.resolve(), args.kit_dir.resolve()
    if not (bot_dir / 'agent.py').is_file():
        raise ValueError('The bot snapshot requires agent.py.')
    before = hashes(bot_dir)
    env = {**CLEAN_ENV, 'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
           'PYTHONDONTWRITEBYTECODE': '1'}
    folders = [bot_dir, kit / 'examples/check_or_fold', kit / 'examples/min_raiser']
    pools = [BotPool([sys.executable, '-m', 'poker_agent_sdk.guest', str(p)],
                     copies=2, cwd=str(p), env=env, grace_ms=2) for p in folders]
    times = defaultdict(list)
    records, logs, fatal = [], [], None
    try:
        for pool in pools:
            await pool.start()
        for players, cap, bank, count in [(2, 20, 0, args.acceptance_tables),
                                         (8, 100, 100, args.eight_seat_tables)]:
            for number in range(count):
                seed = f'{args.seed}-{players}-{number}'
                rules = TableRules(variant='flop_auction_holdem', players=players,
                                   hand_cap=cap, time_bank_ms=bank, decision_ms=1)
                table = SitAndGo(rules=rules, variant=load_variant(rules.variant),
                                 table_id=f'qualification-{players}-{number}', seed=seed)
                memory = [b''] * players
                failures, decisions, errors = [0] * players, [0] * players, []
                while (state := table.decision()) is not None:
                    seat = state.seat
                    decisions[seat] += 1
                    pool = pools[0] if seat == 0 else pools[1 if players == 2 or seat % 2 else 2]
                    try:
                        reply = await pool.act(state, memory[seat])
                        memory[seat] = reply.memory
                        if seat == 0 and reply.elapsed_ms is not None:
                            phase = state.phase if state.phase != 'betting' else f'betting_{len(state.board)}'
                            times[phase].append(reply.elapsed_ms)
                        good = table.apply(reply.action, bank_ms=reply.bank_ms)
                        if not good:
                            errors.append({'seat': seat, 'hand': state.hand_number,
                                           'type': 'IllegalAction', 'action': repr(reply.action)})
                    except BotError as error:
                        errors.append({'seat': seat, 'hand': state.hand_number,
                                       'type': type(error).__name__, 'message': str(error)})
                        good = table.apply(None, answered=False, bank_ms=state.time_bank_ms)
                    if not good:
                        failures[seat] += 1
                result = table.result(tuple(failures), tuple(decisions))
                records.append({'players': players, 'hand_cap': cap, 'time_bank_ms': bank,
                                'decision_ms': 1, 'delivery_grace_ms': 2, 'seed': seed,
                                'hands_played': result.hands_played, 'decisions': decisions,
                                'failures': failures, 'errors': errors,
                                'points': table_points(result, table.payouts)})
    except Exception as error:
        fatal = {'type': type(error).__name__, 'message': str(error)}
    finally:
        logs = [pool.take_errors() for pool in pools]
        for pool in pools:
            await pool.stop()
    after = hashes(bot_dir)
    summary = {phase: {'n': len(v), 'p50_ms': statistics.median(v),
                      'p99_ms': sorted(v)[max(0, int(len(v) * .99) - 1)],
                      'max_ms': max(v), 'above_1ms': sum(t > 1 for t in v)}
               for phase, v in times.items()}
    hero_failures = sum(r['failures'][0] for r in records)
    all_failures = sum(sum(r['failures']) for r in records)
    passed = fatal is None and hero_failures == 0 and before == after and len(records) == args.acceptance_tables + args.eight_seat_tables
    report = {'passed': passed,
              'scope': 'Official BotPool + guest, includes serialization and IPC; ordinary local processes, not production microVM',
              'protocol': {'decision_ms': 1, 'delivery_grace_ms': 2, 'acceptance_time_bank_ms': 0,
                           'acceptance_hand_cap': 20, 'eight_seat_time_bank_ms': 100,
                           'eight_seat_hand_cap': 100, 'worker_copies_per_bot': 2},
              'bot_dir': str(bot_dir), 'kit_dir': str(kit), 'python': sys.version,
              'eval7_version': importlib.metadata.version('eval7'),
              'protocol_module': poker_agent_protocol.__file__,
              'source_hashes_before': before, 'source_hashes_after': after,
              'source_unchanged': before == after, 'summary': summary, 'tables': records,
              'timings_ms': dict(times), 'pool_errors': logs, 'fatal_error': fatal,
              'hero_failures': hero_failures, 'all_failures': all_failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Output already exists; choose a fresh evidence filename.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'summary', 'hero_failures', 'all_failures', 'fatal_error')}, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bot-dir', type=Path, required=True)
    parser.add_argument('--kit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acceptance-tables', type=int, default=6)
    parser.add_argument('--eight-seat-tables', type=int, default=3)
    parser.add_argument('--seed', default='variant-process-v1')
    args = parser.parse_args()
    if args.acceptance_tables < 1 or args.eight_seat_tables < 0:
        parser.error('Use at least one acceptance table and a nonnegative eight-seat count.')
    if args.output.exists():
        parser.error('Output already exists; use a new evidence path.')
    raise SystemExit(asyncio.run(run(args)))


if __name__ == '__main__':
    main()
