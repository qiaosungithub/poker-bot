"""Eight-player benchmarks using the actual SDK guest/host protocol and deadlines."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from poker_agent_protocol.process import BotPool, CLEAN_ENV
from poker_engine.play import play_table
from poker_engine.scoring import table_places, table_points
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "flop_discard_reveal"
_worker = None


class Worker:
    def __init__(self, registry, names, hand_cap, clock, cpus):
        identity = multiprocessing.current_process()._identity
        ordinal = (identity[0] - 1) if identity else 0
        if cpus:
            # Reserve two logical CPUs for host plus sequential guest work.
            selected = {cpus[(2 * ordinal + i) % len(cpus)] for i in range(2)}
            os.sched_setaffinity(0, selected)
        self.loop = asyncio.new_event_loop()
        self.rules = TableRules(variant=VARIANT, hand_cap=hand_cap,
                               decision_ms=1 if clock else 60_000,
                               time_bank_ms=100 if clock else 0)
        self.variant = load_variant(VARIANT)
        env = dict(CLEAN_ENV)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMBA_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1",
                   TMPDIR=str(ROOT / ".runtime/tmp"))
        self.pools = {
            name: BotPool([sys.executable, str(ROOT / "bay_area/seeded_guest.py"),
                           str(ROOT / registry[name]["path"])], copies=1,
                          cwd=str(ROOT / registry[name]["path"]), env=env)
            for name in names
        }
        self.registry = registry

    async def block(self, task):
        block, seed, groups = task
        output = []
        for group_index, group in enumerate(groups):
            for rotation in range(8):
                names = group[rotation:] + group[:rotation]
                table_seed = f"{seed}:block:{block}:group:{group_index}"
                table = SitAndGo(rules=self.rules, variant=self.variant,
                                 table_id=table_seed, seed=table_seed)
                started = time.perf_counter()
                result = await play_table(table, [self.pools[n] for n in names])
                points = table_points(result, table.payouts)
                if sum(result.final_stacks) != 16000 or abs(sum(points) - 100) > 1e-7:
                    raise AssertionError("Chip/point conservation failed.")
                errors = {n: self.pools[n].take_errors() for n in names}
                row = {"block": block, "group": group_index, "rotation": rotation,
                       "seed": table_seed, "hands": result.hands_played,
                       "elapsed_seconds": time.perf_counter() - started,
                       "payouts": table.payouts, "bust_order": result.bust_order,
                       "ties": result.ties, "entries": []}
                places = table_places(result)
                for seat, name in enumerate(names):
                    row["entries"].append({"name": name, "seat": seat,
                        "points": points[seat], "place": places[seat],
                        "final_stack": result.final_stacks[seat],
                        "decisions": result.decisions[seat], "failures": result.failures[seat],
                        "outright_win": result.final_stacks[seat] == 16000,
                        "errors": errors[name]})
                output.append(row)
        return output

    async def stop(self):
        await asyncio.gather(*(p.stop() for p in self.pools.values()))


def initialize(registry, names, hand_cap, clock, cpus):
    global _worker
    _worker = Worker(registry, names, hand_cap, clock, cpus)
    from multiprocessing.util import Finalize
    Finalize(None, close_worker, exitpriority=10)


def close_worker():
    if _worker is not None:
        _worker.loop.run_until_complete(_worker.stop())
        _worker.loop.close()


def execute(task):
    return _worker.loop.run_until_complete(_worker.block(task))


def provenance(registry, names):
    result = {}
    for name in names:
        folder = ROOT / registry[name]["path"]
        # Some projects put development utilities in their bot folder. Capture all
        # root-level runtime sources/assets; do not inspect private runtime files.
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".npz")]
        result[name] = {**registry[name], "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    registry = json.loads((ROOT / "bay_area/candidates.json").read_text())["candidates"]
    names, blocks, seed = config["candidates"], config["blocks"], config["seed"]
    if len(names) % 8 or len(set(names)) != len(names):
        raise ValueError("Use a multiple of eight distinct candidate names.")
    if blocks < 1 or args.workers < 1:
        raise ValueError("Positive blocks and workers required.")
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = []
    for block in range(blocks):
        order = names.copy()
        random.Random(f"{seed}:seating:{block}").shuffle(order)
        tasks.append((block, seed, [order[i:i+8] for i in range(0, len(order), 8)]))
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "candidates": provenance(registry, names),
                "rules": asdict(TableRules(variant=VARIANT, hand_cap=config.get("hand_cap", 250))),
                "runner_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "bay_area").glob("*.py"))},
                "workers": args.workers, "python": sys.version, "platform": platform.platform(),
                "packages": {n: importlib.metadata.version(n) for n in ("pokerkit", "eval7", "numpy")},
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "clock": "SDK wall clock; 1 ms decision + 2 ms delivery grace + 100 ms bank"
                         if config.get("clock", True) else "60 s per decision; effectively untimed",
                "seed_note": "Deck and seating seeds plus per-request PRNG seeding; existing wall-time guards remain."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    count = 0
    with (args.out / "tables.jsonl").open("x") as file:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(registry, names, config.get("hand_cap", 250),
                config.get("clock", True), sorted(os.sched_getaffinity(0)))) as executor:
            futures = [executor.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                for row in future.result():
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                file.flush()
                count += 1
                print(json.dumps({"completed_blocks": count, "total_blocks": blocks,
                                  "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["completed_blocks"] = count
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["raw_sha256"] = hashlib.sha256((args.out / "tables.jsonl").read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
