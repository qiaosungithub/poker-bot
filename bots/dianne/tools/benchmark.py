"""Boston comparisons on the official engine, with cyclic seat rotations.

Direct mode measures Agent.act and checks the SDK reply contract. SDK mode
also exercises the guest processes and applies the SDK's wall-clock deadline.
"""

import argparse
import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bay_area.environment import BayAreaEnv
from poker_agent_protocol.process import BotPool
from poker_engine.play import play_table
from poker_engine.scoring import table_points
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

VARIANT = "flop_auction_holdem"


def load_agent(folder):
    spec = importlib.util.spec_from_file_location("candidate_" + folder.name, folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Agent


def direct_block(task):
    folders, block, seed, hand_cap = task
    constructors = {folder.name: load_agent(folder) for folder in folders}
    order = [folder.name for folder in folders]
    random.Random(f"{seed}:seating:{block}").shuffle(order)
    rows, timings = [], defaultdict(list)
    for rotation in range(len(order)):
        names = order[rotation:] + order[:rotation]
        agents = [constructors[name]() for name in names]
        env = BayAreaEnv(f"{seed}:{block}", players=len(names), hand_cap=hand_cap, variant=VARIANT)
        errors = []
        while env.state is not None:
            state = env.state
            started = time.perf_counter_ns()
            try:
                answer = agents[state.seat].act(state, env.memory[state.seat])
            except Exception as error:
                errors.append({"name": names[state.seat], "phase": state.phase, "error": repr(error)})
                env.step(crashed=True)
                continue
            elapsed = time.perf_counter_ns() - started
            phase = "auction" if state.phase == "auction" else f"betting_{len(state.board)}"
            timings[(names[state.seat], phase)].append(elapsed)
            _, reason = env.step(answer)
            if reason:
                errors.append({"name": names[state.seat], "phase": state.phase, "error": reason})
        result, points = env.result()
        rows.append(table_row(block, rotation, names, result, points, errors, env.table.payouts))
    return rows, {"|".join(key): values for key, values in timings.items()}


def table_row(block, rotation, names, result, points, errors, payouts):
    expected_points = sum(payouts[:len(names)])
    if sum(result.final_stacks) != 2000 * len(names) or abs(sum(points) - expected_points) > 1e-7:
        raise AssertionError("Chip or point conservation failed")
    return {"block": block, "rotation": rotation, "hands": result.hands_played, "payouts": payouts,
            "errors": errors, "entries": [
                {"name": name, "seat": seat, "points": points[seat],
                 "stack": result.final_stacks[seat], "decisions": result.decisions[seat],
                 "failures": result.failures[seat]}
                for seat, name in enumerate(names)]}


async def sdk_tables(folders, blocks, seed, hand_cap, time_bank):
    pools = {folder.name: BotPool(
        [sys.executable, "-m", "poker_agent_sdk.guest", str(folder)], copies=1, cwd=str(folder))
        for folder in folders}
    rules = TableRules(variant=VARIANT, players=len(folders), hand_cap=hand_cap, time_bank_ms=time_bank)
    rows = []
    try:
        for block in range(blocks):
            order = list(pools)
            random.Random(f"{seed}:seating:{block}").shuffle(order)
            for rotation in range(len(order)):
                names = order[rotation:] + order[:rotation]
                table = SitAndGo(rules=rules, variant=load_variant(VARIANT),
                                 table_id=f"{seed}:{block}", seed=f"{seed}:{block}")
                result = await play_table(table, [pools[name] for name in names])
                errors = {name: pools[name].take_errors() for name in names}
                rows.append(table_row(block, rotation, names, result,
                                      table_points(result, table.payouts), errors, table.payouts))
            print(json.dumps({"completed_blocks": block + 1, "total_blocks": blocks}), flush=True)
    finally:
        await asyncio.gather(*(pool.stop() for pool in pools.values()))
    return rows


def summarize(rows, timings):
    scores, blocks, failures, decisions = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(int), defaultdict(int)
    for row in rows:
        for entry in row["entries"]:
            name = entry["name"]
            scores[name].append(entry["points"])
            blocks[name][row["block"]].append(entry["points"])
            failures[name] += entry["failures"]
            decisions[name] += entry["decisions"]
    result = {}
    for name, values in scores.items():
        block_means = [statistics.mean(values) for values in blocks[name].values()]
        se = statistics.stdev(block_means) / math.sqrt(len(block_means)) if len(block_means) > 1 else None
        result[name] = {"mean_points": statistics.mean(values), "tables": len(values),
                        "blocks": len(block_means), "block_standard_error": se,
                        "failures": failures[name], "decisions": decisions[name], "timing": {}}
        for key, samples in timings.items():
            candidate, phase = key.split("|")
            if candidate != name:
                continue
            samples = sorted(samples)
            result[name]["timing"][phase] = {
                "calls": len(samples), "p50_ms": samples[len(samples) // 2] / 1e6,
                "p99_ms": samples[min(len(samples)-1, math.ceil(len(samples)*0.99)-1)] / 1e6,
                "max_ms": samples[-1] / 1e6, "over_1ms": sum(value > 1_000_000 for value in samples)}
    return dict(sorted(result.items(), key=lambda item: -item[1]["mean_points"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", default="boston-screen-20261109")
    parser.add_argument("--hand-cap", type=int, default=250)
    parser.add_argument("--sdk", action="store_true")
    parser.add_argument("--time-bank", type=int, default=100)
    args = parser.parse_args()
    folders = [folder.resolve() for folder in args.bots]
    if not 2 <= len(folders) <= 10 or len({folder.name for folder in folders}) != len(folders):
        parser.error("Supply 2–10 uniquely named bot folders")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "variant": VARIANT,
                "seed": args.seed, "blocks": args.blocks, "hand_cap": args.hand_cap,
                "mode": "sdk" if args.sdk else "direct", "python": sys.version,
                "decision_ms": 1, "delivery_grace_ms": 2 if args.sdk else None,
                "time_bank_ms": args.time_bank if args.sdk else None,
                "sources": {folder.name: hashlib.sha256((folder / "agent.py").read_bytes()).hexdigest()
                            for folder in folders},
                "timing_scope": "SDK host clock" if args.sdk else "Agent.act only, no deadlines imposed; includes scheduler noise",
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, timings = [], defaultdict(list)
    with (args.out / "tables.jsonl").open("x") as output:
        if args.sdk:
            rows = asyncio.run(sdk_tables(folders, args.blocks, args.seed, args.hand_cap, args.time_bank))
            for row in rows:
                output.write(json.dumps(row) + "\n")
        else:
            tasks = [(folders, block, args.seed, args.hand_cap) for block in range(args.blocks)]
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(direct_block, task) for task in tasks]
                for index, future in enumerate(as_completed(futures), 1):
                    block_rows, block_timings = future.result()
                    rows.extend(block_rows)
                    for row in block_rows:
                        output.write(json.dumps(row) + "\n")
                    output.flush()
                    for key, samples in block_timings.items():
                        timings[key].extend(samples)
                    print(json.dumps({"completed_blocks": index, "total_blocks": args.blocks,
                                      "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    summary = summarize(rows, timings)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["tables_sha256"] = hashlib.sha256((args.out / "tables.jsonl").read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
