"""Decision-only timings on the same retained public states, outside strength runs."""
import gzip
import json
import os
from pathlib import Path
import platform
import statistics
import time

from auction_flop.benchmark import run
from poker_agent_protocol.game import state_from_dict


def main():
    here = Path(__file__).resolve().parent
    cpus = json.loads(Path('.runtime/simple-tuning-cpus.json').read_text())
    os.sched_setaffinity(0, {cpus[0]})
    paths = {
        'original_32': run.ROOT/'candidates/simple_classic_v1',
        'original_16': here/'candidates/samples_16',
        'selected_32': run.ROOT/'candidates/simple_tuned_v2',
    }
    agents = {name: run.arena.import_agent(path, name)() for name, path in paths.items()}
    with gzip.open(here/'states.jsonl.gz', 'rt') as stream:
        corpus = [(r['phase'], state_from_dict(r['state']))
                  for r in map(json.loads, stream)]
    for agent in agents.values():
        for _, state in corpus:
            agent.act(state, b'')
    samples = {name: {} for name in agents}
    names = list(agents)
    with gzip.open(here/'latency-samples.jsonl.gz', 'wt') as stream:
        for repeat in range(5):
            for index, (phase, state) in enumerate(corpus):
                offset = (index + repeat) % len(names)
                for name in names[offset:] + names[:offset]:
                    start = time.perf_counter_ns()
                    agents[name].act(state, b'')
                    ns = time.perf_counter_ns() - start
                    samples[name].setdefault(phase, []).append(ns/1e6)
                    stream.write(json.dumps([name, phase, repeat, index, ns])+'\n')
    result = {'scope': 'Warm Agent.act only; no SDK IPC, no production deadline qualification.',
              'python': platform.python_version(), 'cpu': cpus[0],
              'states_per_arm': len(corpus), 'repeats': 5, 'timings_ms': {}}
    for name, phases in samples.items():
        result['timings_ms'][name] = {}
        for phase, values in phases.items():
            ordered = sorted(values)
            result['timings_ms'][name][phase] = {
                'calls': len(values), 'p50': statistics.median(values),
                'p99': ordered[int(.99*(len(ordered)-1))],
                'max': max(values), 'over_1_ms': sum(x > 1 for x in values)}
    run.write_json(here/'latency.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
