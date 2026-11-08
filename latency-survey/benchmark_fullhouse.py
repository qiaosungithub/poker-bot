"""Measure an existing, unmodified Fullhouse strategy, not a synthetic network.

Run with /tmp/poker-fullhouse-venv/bin/python benchmark_fullhouse.py.
No model training or policy changes. Model import/preloading is outside timing.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import collections
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "fullhouse-bot"
sys.path[:0] = [str(REPO), str(REPO / "engine_vendored")]

def stats(times):
    import numpy as np
    x = np.asarray(times, dtype=float) / 1e6
    return {"n": len(x), "mean_ms": float(x.mean()),
            "p50_ms": float(np.quantile(x, .5)), "p95_ms": float(np.quantile(x, .95)),
            "p99_ms": float(np.quantile(x, .99)), "p999_ms": float(np.quantile(x, .999)),
            "max_ms": float(x.max()), "over_1ms": int((x > 1).sum())}

def generate_states(count_per_street, seed):
    """Generate legal game trajectories, stratified by street and stack depths."""
    from engine.game import PokerEngine
    rng = random.Random(seed)
    buckets = {s: [] for s in ("preflop", "flop", "turn", "river")}
    hand = 0
    while min(map(len, buckets.values())) < count_per_street:
        n = 6
        ids = [f"p{i}" for i in range(n)]
        stacks = {p: rng.choice((500, 1000, 2000, 5000, 10000, 20000)) for p in ids}
        engine = PokerEngine(f"bench_{seed}_{hand}", ids, hand % n, stacks, seed + hand)
        state = engine.start_hand()
        steps = 0
        # Passive trajectories cover later streets; other trajectories include
        # raises, all-ins and folds. These policies generate inputs only.
        passive = hand % 4 == 0
        while state.get("type") == "action_request":
            bucket = buckets[state["street"]]
            if len(bucket) < count_per_street:
                bucket.append(state)
            seat = state["seat_to_act"]
            x = rng.random()
            if passive or x < .62:
                action = {"action": "check" if state["can_check"] else "call"}
            elif x < .76:
                action = {"action": "fold"}
            elif x < .96:
                total = state["your_bet_this_street"] + state["amount_owed"]
                amount = max(state["min_raise_to"], total + int(rng.choice((.33, .5, 1, 2)) * (state["pot"] + state["amount_owed"])))
                amount = min(amount, state["your_bet_this_street"] + state["your_stack"])
                action = {"action": "raise", "amount": amount}
            else:
                action = {"action": "all_in"}
            state = engine.apply_action(seat, action)
            steps += 1
            if steps > 300:
                raise RuntimeError("Generator exceeded 300 actions")
        hand += 1
    states = [s for bucket in buckets.values() for s in bucket]
    rng.shuffle(states)
    return states, hand

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-street", type=int, default=2500)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--seed", type=int, default=19073)
    p.add_argument("--output", default="fullhouse_results.json")
    p.add_argument("--disable-gc", action="store_true")
    p.add_argument("--onnx", action="store_true")
    args = p.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    import numpy as np
    import eval7
    from bot import bot

    # Fail loudly if upstream silently substitutes its emergency action.
    fallback_calls = []
    def unexpected_fallback(state):
        fallback_calls.append(state)
        raise RuntimeError("Original bot entered emergency fallback")
    bot._safe_default_action = unexpected_fallback

    states, hands = generate_states(args.per_street, args.seed)
    parity = None
    if args.onnx:
        from fullhouse_onnx import install
        parity = install(bot, states)
    for state in states[:1000]:
        bot.decide(state)
    gc.collect()
    if args.disable_gc:
        gc.disable()
    wall, cpu, stages, actions = [], [], collections.defaultdict(list), collections.Counter()
    legal_failures = []
    slow_calls = []
    # Serialisation cost is measured separately; primary timing includes all
    # preprocessing, encoding, inference, legality masking, sampling and dict return.
    runs = []
    for repeat in range(args.repeats):
        run_wall = []
        for index, state in enumerate(states):
            c0 = time.thread_time_ns()
            t0 = time.perf_counter_ns()
            action = bot.decide(state)
            elapsed = time.perf_counter_ns() - t0
            used = time.thread_time_ns() - c0
            wall.append(elapsed); cpu.append(used); run_wall.append(elapsed)
            stages[state["street"]].append(elapsed)
            if elapsed > 1_000_000:
                slow_calls.append({"repeat": repeat, "index": index, "wall_ms": elapsed / 1e6, "cpu_ms": used / 1e6, "state": state})
            actions[action["action"]] += 1
            if action["action"] == "check" and not state["can_check"]:
                legal_failures.append({"state": state, "action": action})
            if action["action"] == "raise":
                amount = action["amount"]
                if not state["min_raise_to"] <= amount <= state["your_stack"] + state["your_bet_this_street"]:
                    legal_failures.append({"state": state, "action": action})
        runs.append(stats(run_wall))

    serialized = [json.dumps(s) for s in states]
    json_times = []
    for raw in serialized:
        t0 = time.perf_counter_ns()
        action = bot.decide(json.loads(raw))
        json.dumps(action)
        json_times.append(time.perf_counter_ns() - t0)

    models = {}
    for path in (REPO / "data").glob("*.npz"):
        data = np.load(path, allow_pickle=False)
        models[path.name] = {"bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "arrays": {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)} for k in data.files}}
    result = {"source": "https://github.com/advitrocks9/fullhouse-bot",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "cpu_model": next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "numpy": np.__version__,
              "python": sys.version, "threads": 1, "seed": args.seed,
              "cyclic_gc_enabled": gc.isenabled(),
              "onnx_parity": parity,
              "players": 6, "generated_hands": hands, "unique_states": len(states),
              "stack_bb_choices": [5, 10, 20, 50, 100, 200],
              "max_action_history": max(len(s["action_log"]) for s in states),
              "measurement": "Warm state-dict -> original decide -> action-dict; model loaded before timing; original fallback replaced by raising assertion; policy unchanged",
              "wall": stats(wall), "thread_cpu": stats(cpu), "runs": runs,
              "by_street_wall": {s: stats(t) for s,t in stages.items()},
              "json_decode_decide_encode_wall": stats(json_times),
              "fallback_count": len(fallback_calls), "legal_failure_count": len(legal_failures),
              "slow_calls": slow_calls,
              "action_counts": dict(actions), "models": models}
    out = ROOT / args.output
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "models"}, indent=2))
    print("Results:", out)

if __name__ == "__main__":
    main()
