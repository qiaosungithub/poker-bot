"""Independently audit frozen confirmation runs and write the final report."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(filename, expected, first_seed, pairs, plan):
    path = HERE / filename
    result = json.loads(path.read_text())
    raw = path.with_suffix('.jsonl')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows) == pairs * len(expected)
    assert set(r['opponent'] for r in rows) == set(expected)
    assert not any('error' in r for r in rows)
    meta = result['metadata']
    assert meta['engine_adapter_sha256'] == plan['hashes']['engine_adapter.py']
    assert meta['script_sha256'] == plan['hashes']['run_benchmark.py']
    assert meta['local_sources']['lbr_baseline.py'] == plan['hashes']['lbr_baseline.py']
    assert meta['config']['hands'] == 200
    assert meta['config']['seed'] == first_seed
    output = {}
    for name in expected:
        selected = [r for r in rows if r['opponent'] == name]
        assert len(selected) == pairs
        assert sorted(r['seed'] for r in selected) == list(range(first_seed, first_seed + pairs))
        scores, legs = [], []
        for row in selected:
            assert len(row['legs']) == 2
            scores.append(sum(leg['score'] for leg in row['legs']) / 2)
            assert math.isclose(row['score'], scores[-1], abs_tol=1e-12)
            for leg in row['legs']:
                assert 0 <= leg['score'] <= 100
                assert math.isclose(leg['score'], (leg['chip_delta'] + 10000) / 200, abs_tol=1e-12)
                assert leg['eliminated'] == (leg['score'] == 0)
                assert leg['opponent_eliminated'] == (leg['score'] == 100)
                assert 1 <= leg['hands'] <= 200
                assert not any(leg['corrections'].values())
                assert leg['opponent_diagnostics'].get('zero_posterior', 0) == 0
                legs.append(leg)
        mean = statistics.mean(scores)
        se = statistics.stdev(scores) / math.sqrt(pairs)
        ci = [mean - 1.96 * se, mean + 1.96 * se]
        reported = result['results'][name]
        assert math.isclose(mean, reported['score']['mean'], abs_tol=1e-12)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(ci, reported['score']['ci95']))
        hands = sum(leg['hands'] for leg in legs)
        assert hands == reported['hands']
        output[name] = dict(
            mean_score=mean, ci95=ci, ci95_half_width=1.96 * se,
            independent_pairs=pairs, matches=len(legs), hands=hands,
            hero_busted=sum(leg['eliminated'] for leg in legs),
            opponent_busted=sum(leg['opponent_eliminated'] for leg in legs),
            reached_cutoff_without_bust=sum(not (leg['eliminated'] or leg['opponent_eliminated']) for leg in legs),
            ahead_at_end=sum(leg['ahead'] for leg in legs),
            mean_hands_per_match=hands / len(legs),
            mean_pair_worker_seconds=statistics.mean(r['seconds'] for r in selected),
            runtime_errors=0, action_corrections=0, zero_posterior_errors=0,
            result_file=filename, raw_file=raw.name,
        )
    return output, dict(file=filename, raw_sha256=sha(raw), result_sha256=sha(path),
                        elapsed_seconds=result['elapsed_seconds'], workers=meta['config']['workers'])


def main():
    plan = json.loads((HERE / 'experiment_plan_v4.json').read_text())
    for name, expected_hash in plan['hashes'].items():
        assert sha(HERE / name) == expected_hash, f'Frozen source changed: {name}'
    medium, run1 = audit('final_medium_easy_v4.json', ['cfr_lite', 'station'], 6000000, 4096, plan)
    hard, run2 = audit('final_hard_v4.json', ['lbr_cautious25'], 7000000, 1024, plan)
    results = {**hard, **medium}
    report = dict(
        target='Released Fullhouse checkpoint, original decision policy with FP32 ONNX CPU inference',
        target_repository='https://github.com/advitrocks9/fullhouse-bot',
        target_commit='e504793d480b1b975f25258d25939b45c6dbd5a4',
        protocol=plan['rules'], score=plan['score'], sampling=plan['sampling'],
        public_match_history='Last 200 public actions, matching the upstream sandbox contract.',
        environment=json.loads((HERE / 'benchmark_environment.json').read_text()),
        results=results, total_matches=sum(r['matches'] for r in results.values()),
        total_hands=sum(r['hands'] for r in results.values()), runs=[run1, run2],
        opponent_descriptions={
            'lbr_cautious25': 'Locally implemented targeted white-box stress test inspired by Local Best Response; Bayesian hand-range updates, known target policy, 16 sampled runouts per combination, utility stack**0.25. No access to actual opponent cards or future deck. Not a released champion or an exact best response.',
            'cfr_lite': 'Unmodified existing rule/equity hybrid from the Fullhouse repository; name does not establish equilibrium play.',
            'station': 'Unmodified existing loose-calling baseline from the Fullhouse repository.',
        },
        sources={
            'lbr_method': 'https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf',
            'cfr_lite': 'https://github.com/advitrocks9/fullhouse-bot/blob/e504793d480b1b975f25258d25939b45c6dbd5a4/bots/competitors/cfr_lite/bot.py',
            'station': 'https://github.com/advitrocks9/fullhouse-bot/blob/e504793d480b1b975f25258d25939b45c6dbd5a4/bots/adversarial/station.py',
        },
        validation=plan['independent_validation'], unit_tests_passed=plan['tests_passed'],
        limitations=[
            'This is a custom heads-up ICM-like scoring protocol, not the unpublished official payout table.',
            'No wall-clock action penalties; this experiment measures strength, not compliance with a 1 ms deadline.',
            'The hard opponent knows the frozen target policy and was selected using separate pilot matches; it is a targeted exploitability stress test.',
            'Medium and easy baselines share the target repository provenance; this is not an independent champion-level ranking.',
            'Confidence intervals quantify sampling uncertainty for this fixed protocol and these opponents; they do not prove eight-player tournament performance.',
            'Original vendor engine all-in handling was corrected in engine_adapter.py; the upstream checkout and target weights remain unmodified.',
        ],
        frozen_plan='experiment_plan_v4.json', frozen_sources='frozen_sources_v4.zip',
        reproduction_commands=[
            'python run_benchmark.py --opponents cfr_lite,station --pairs 4096 --workers 16 --seed 6000000 --output reproduction_medium_easy.json',
            'python run_benchmark.py --opponents lbr_cautious25 --pairs 1024 --workers 64 --seed 7000000 --output reproduction_hard.json',
        ],
    )
    with (HERE / 'benchmark_report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
