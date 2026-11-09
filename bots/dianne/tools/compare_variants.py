"""Compare each candidate against the same seven opponents and paired deck seeds."""

import argparse
from collections import defaultdict
import json
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", type=Path)
    parser.add_argument("--opponents", nargs=7, required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--baseline-results", type=Path,
                        help="Reuse an identical frozen baseline run instead of rerunning it")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    names = [p.name for p in args.variants]
    if len(set(names)) != len(names):
        raise ValueError("Variant names must be unique")
    means, layouts, result = {}, {}, {}
    for candidate in args.variants:
        destination = args.out / candidate.name
        command = [sys.executable, str(Path(__file__).with_name("benchmark.py")),
                   str(candidate), *map(str, args.opponents), "--out", str(destination),
                   "--blocks", str(args.blocks), "--workers", str(args.workers), "--seed", args.seed]
        if candidate.name == names[0] and args.baseline_results:
            manifest = json.loads((args.baseline_results / "manifest.json").read_text())
            expected = {p.name: hashlib.sha256((p / "agent.py").read_bytes()).hexdigest()
                        for p in [candidate, *args.opponents]}
            assert manifest["sources"] == expected
            assert manifest["seed"] == args.seed and manifest["blocks"] == args.blocks
            assert manifest["mode"] == "direct" and manifest["hand_cap"] == 250
            shutil.copytree(args.baseline_results, destination)
        else:
            with (args.out / (candidate.name + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (destination / "tables.jsonl").read_text().splitlines()]
        groups, seating = defaultdict(list), {}
        for row in rows:
            seating[(row["block"], row["rotation"])] = [
                "hero" if x["name"] == candidate.name else x["name"] for x in row["entries"]]
            hero = next(x for x in row["entries"] if x["name"] == candidate.name)
            groups[row["block"]].append(hero["points"])
        means[candidate.name] = {block: statistics.mean(scores) for block, scores in groups.items()}
        layouts[candidate.name] = seating
        assert seating == layouts[names[0]], "Opponent seating differs between variants"
        summary = json.loads((destination / "summary.json").read_text())[candidate.name]
        differences = [means[candidate.name][block] - means[names[0]][block] for block in sorted(groups)]
        delta = statistics.mean(differences)
        se = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0
        result[candidate.name] = {
            "mean_points": summary["mean_points"], "standard_error": summary["block_standard_error"],
            "paired_difference": delta, "normal_95_interval": [delta - 1.96 * se, delta + 1.96 * se],
            "failures": summary["failures"], "decisions": summary["decisions"]}
        report = {"baseline": names[0], "seed": args.seed, "blocks": args.blocks,
                  "tables_per_variant": args.blocks * 8,
                  "opponents": [p.name for p in args.opponents],
                  "note": "Whole-block paired normal intervals; no multiple-comparison correction. Direct engine calls do not enforce deadlines.",
                  "results": result}
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(candidate.name, json.dumps(result[candidate.name]), flush=True)


if __name__ == "__main__":
    main()
